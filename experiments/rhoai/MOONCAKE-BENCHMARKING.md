# Mooncake KV Cache Offloading Benchmark on Diadochos

## Architecture

Two-cluster setup:
- **Hub cluster** (aperdomo-lab): Runs Tekton pipelines via benchflow
- **Remote cluster** (diadochos): Hosts mooncake vLLM deployment on node gjfjh

The hub cluster's kubeconfig secret `psap-h100-diadochos` bridges the two clusters.

### Component Roles

| Component | Purpose |
|-----------|---------|
| `mooncake-master` | Metadata server — tracks which blocks exist and where. Workers discover each other via P2PHANDSHAKE, then communicate directly over RDMA. Runs eviction when memory watermark is reached. |
| `mooncake-dist-nvme-vllm` | vLLM pod 1 (TP=2, hostNetwork, port 30080). In embedded mode, each TP worker runs its own mooncake store segment. |
| `mooncake-dist-nvme-vllm-2` | vLLM pod 2 (TP=2, hostNetwork, port 30081). Same config as pod 1. |
| `mooncake-nginx-lb` | Nginx round-robin LB at `mooncake-vllm-lb.benchflow.svc:8080` across both vLLM pods. |

Model: `google/gemma-4-31B-it`, gpu-memory-utilization=0.85, max-num-seqs=256.

### How Mooncake Prefix Cache Works

1. Request arrives → scheduler calls `start_load(block_hashes)` → connector does `lookup_exists` against the master
2. If blocks found → `load_get` fetches data via RDMA from the owning segment → **external prefix cache hit**
3. If blocks in local GPU KV cache → **local prefix cache hit**
4. Otherwise → blocks computed from scratch by the engine
5. After processing → `save_exists` (dedup check) then `save_put` stores blocks in the mooncake store via master

**PoolKey composition**: `{cache_prefix}@{model_name}@tp_rank:{tp_rank}@pcp{pcp_rank}@dcp{dcp_rank}@pp_rank:{pp_rank}@group:{group_id}@{chunk_hash}`

Key constraint: **TP=2 ALL-ranks lookup** — the lookup requires ALL TP ranks to have a block for it to be a hit. If the master evicts either rank's metadata, the block is considered missing.

### Mooncake Config

Located at `/mnt/nvme-kv-cache/mooncake-config/mooncake-rdma-nvme-64g.json` on the NVMe hostPath:

```json
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "mooncake-master.benchflow.svc.cluster.local:50051",
  "global_segment_size": "64GB",
  "local_buffer_size": "64GB",
  "protocol": "rdma",
  "device_name": "mlx5_7",
  "enable_offload": true,
  "enable_ssd_offload": true,
  "ssd_offload_path": "/mnt/nvme-kv-cache"
}
```

Key settings:
- `local_buffer_size`: RDMA-pinned memory per TP worker. 64GB works (4 workers × 64GB = 256GB pinned). 256GB OOM-kills (1TB total).
- `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES` env var (set on vLLM pods): NVMe I/O staging buffer. Default is ~32MB which causes BUFFER_OVERFLOW under high concurrency. Set to 4294967296 (4GB).
- Total master capacity = `local_buffer_size` × number of TP workers across all pods = 64GB × 4 = 256GB.

### NVMe RAID-0

7x NVMe drives (nvme1n1-nvme7n1) in md0 RAID-0, mounted at `/var/mnt/mooncake-nvme`. nvme0n1 is reserved for benchflow and must not be touched.

### Nginx Load Balancer

The nginx LB uses HTTP/1.1 keepalive connections to the upstream pool. With streaming responses and keepalive, connections tend to stick to one backend. In practice during benchmarks, most traffic goes to pod 1 (port 30080). This actually helps prefix cache hit rate since all blocks are saved/looked up on the same engine.

ConfigMap `mooncake-nginx-lb-config` contains the nginx config. The custom config is mounted at `/etc/nginx-custom/nginx.conf` and nginx is started with `-c /etc/nginx-custom/nginx.conf`.

## Running the Benchmark

### Step 1: Extract diadochos kubeconfig

```bash
export DIADOCHOS_KUBECONFIG=/tmp/diadochos-kubeconfig.tmp

KUBECONFIG=/Users/rdoddaia/work/aperdomo/Aperdomo-kubeconfig \
  kubectl get secret psap-h100-diadochos -n benchflow \
  -o jsonpath='{.data.kubeconfig}' | base64 -d > $DIADOCHOS_KUBECONFIG
```

### Step 2: Verify the endpoint is healthy

```bash
# Check pods
KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl get pods -n benchflow -o wide | grep -E "mooncake|vllm"

# Health check via LB — confirm max_model_len matches your target
KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl exec -n benchflow deploy/mooncake-nginx-lb -- \
  curl -s http://localhost:8080/v1/models | python3 -m json.tool
```

### Step 3: Submit the benchmark

```bash
KUBECONFIG=/Users/rdoddaia/work/aperdomo/Aperdomo-kubeconfig bflow experiment run \
  experiments/rhoai/mooncake-offloading.yaml \
  --target-url http://mooncake-vllm-lb.benchflow.svc:8080 \
  --benchflow-image ghcr.io/albertoperdomo2/benchflow:manual-a53adf3 \
  --cluster-name psap-h100-diadochos
```

Key flags:
- `--target-url`: Skips deploy/cleanup/download steps — benchmarks against the existing mooncake vLLM endpoint
- `--benchflow-image`: Container image for the Tekton pipeline steps (registry is `ghcr.io`, not `quay.io`)
- `--cluster-name psap-h100-diadochos`: Routes to the correct Kueue LocalQueue

This runs aiperf with weka agentic traces (`semianalysisai/cc-traces-weka-with-subagents-060826`), 32 concurrency, 1800s duration, scenario `inferencex-agentx-mvp`. The dataset has 680 conversations.

### Step 4: Monitor progress

```bash
# Watch pipeline
KUBECONFIG=/Users/rdoddaia/work/aperdomo/Aperdomo-kubeconfig bflow watch <pipelinerun-name>

# Live aiperf metrics (on diadochos — find job name first)
KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl get jobs -n benchflow | grep benchmark-mooncake
KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl logs -f -n benchflow job/<job-name> | grep -E "profiling|ext_cache"

# Live vLLM KV transfer metrics
KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl logs -f -n benchflow deployment/mooncake-dist-nvme-vllm | grep -E "KV Transfer|prefix cache"

# Live master state (eviction, Gets, capacity)
KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl logs -f -n benchflow deployment/mooncake-master | grep -oE '(Mem Storage[^|]+|Eviction[^|]+|Get:\([^)]+\))'
```

### Step 5: Collect I/O metrics

```bash
HUB_KUBECONFIG=/Users/rdoddaia/work/aperdomo/Aperdomo-kubeconfig \
  KUBECONFIG=$DIADOCHOS_KUBECONFIG \
  ./scripts/benchmark-io-metrics.sh <pipelinerun-name>
```

### Step 6: Get benchmark results

```bash
KUBECONFIG=/Users/rdoddaia/work/aperdomo/Aperdomo-kubeconfig bflow logs <pipelinerun-name> --step benchmark
```

## Changing max-model-len

`max-model-len` is the single most impactful parameter for prefix cache hit rates. It controls the maximum sequence length vLLM accepts, which determines how many turns of a multi-turn conversation fit before context overflow.

```bash
# Patch both vLLM pods (replace NEW_VALUE)
for deploy in mooncake-dist-nvme-vllm mooncake-dist-nvme-vllm-2; do
  KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl get deployment $deploy -n benchflow -o json \
    | python3 -c "
import sys, json, re
d = json.load(sys.stdin)
cmd = d['spec']['template']['spec']['containers'][0]['command']
cmd = [re.sub(r'--max-model-len=\d+', '--max-model-len=NEW_VALUE', c) for c in cmd]
d['spec']['template']['spec']['containers'][0]['command'] = cmd
json.dump(d, sys.stdout)
" | KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl apply -f -
done

# Always restart master together to clear stale segment registrations
KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl rollout restart -n benchflow deployment/mooncake-master

# Wait for all pods to be ready
KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl wait --for=condition=available -n benchflow \
  deployment/mooncake-master deployment/mooncake-dist-nvme-vllm deployment/mooncake-dist-nvme-vllm-2 --timeout=600s
```

## Experiment File

`experiments/rhoai/mooncake-offloading.yaml` — uses:
- Deployment profile: `mooncake-embedded-cpu` (profiles/deployment/rhoai/)
- Benchmark profile: `aiperf-agentx-inference` (profiles/benchmark/)
- Node affinity: gjfjh only
- TP=2, replicas=1, gpu-memory-utilization=0.55

Note: `aiperf-agentx-inference` profile has `max_context_length: 131072` and `min_max_model_len: 131072`. When using `--target-url`, the `min_max_model_len` check is bypassed and aiperf truncates/terminates trajectories that exceed the endpoint's actual max-model-len.

## Updating mooncake_config.json

The config lives on the NVMe hostPath, not a ConfigMap. To change it:

```bash
KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl exec -n benchflow deploy/mooncake-dist-nvme-vllm -- \
  bash -c 'cat > /mnt/nvme-kv-cache/mooncake-config/mooncake-rdma-nvme-64g.json << EOF
{
  ... updated config ...
}
EOF'

# Restart all pods to pick up:
KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl rollout restart -n benchflow \
  deployment/mooncake-master \
  deployment/mooncake-dist-nvme-vllm \
  deployment/mooncake-dist-nvme-vllm-2
```

Restart mooncake-master first (or together) to clear stale segment registrations.

## Benchmark Runs

### Run 1: mooncake-offloading-208d51 (local_buffer_size=4GB, no MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES)
- MLflow run: `f9687164d65f4d1eb51697d9edcbc317`
- max-model-len: 131072
- Result: 2,088/2,199 requests errored (500s)
- Root cause: `BUFFER_OVERFLOW` in NVMe staging buffer (32MB default) under 32 concurrent lanes with 56K+ token contexts
- vLLM pods restarted 4 times during benchmark
- Disk I/O: avg 1.67 GB/s write, peak 5.34 GB/s; RDMA: avg 1.78 GB/s, peak 5.92 GB/s; total 4.44 TB written to NVMe

### Run 2: mooncake-offloading-3f858b (local_buffer_size=64GB, MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=4GB)
- MLflow run: `a7134ec2d4ae43e5aa59c5e00d887d40`
- max-model-len: 131072, eviction_high_watermark: 0.95 (default)
- Result: Passed (assert-benchmark-status succeeded)
- vLLM pods: 0 restarts, 0 failed KV transfer keys throughout
- Benchmark window: 2026-07-29T19:47:22Z to 2026-07-29T20:26:53Z (~40 min)
- External prefix cache hit rate: 0.8-0.9% (near zero)
- Master: 69 eviction cycles in 40 min, 93% capacity, 234M keys looked up but only 4,726 loaded (0.002%)
- I/O: RDMA avg 3.58 GB/s (peak 12.86 GB/s), 9.08 TB total; NVMe avg 483 MB/s, 1.20 TB total

### Run 3: mooncake-offloading-69295b (max-model-len=8192)
- max-model-len: 8192, eviction_high_watermark: 0.99
- Result: **Benchmark stalled** — only 76 requests completed, then 0.0 RPS
- Root cause: 8192 too small for weka agentic traces — all 32 trajectories terminated early due to context-overflow (most overflow by turn 7-20, some at turn 1)
- Master: 56.56 GB / 256 GB (22%), 46K keys, zero evictions, **zero Get requests**
- External prefix cache: 0.0% (theoretical also 0.0%)
- Lesson: gemma-4-31B-it tokenizer is less efficient than Qwen3.6-35B-A3B — same traces that work at 8192 with Qwen overflow with gemma

### Run 4: mooncake-offloading-7be0ad (max-model-len=32768)
- max-model-len: 32768, eviction_high_watermark: 0.99
- Result: 77 requests completed, then stalled at 0.0 RPS (trajectories overflow at turns 25-49)
- `theoretical_prefix_cache_hit=24.8%`, `prefix_cache_hit=21.5%` (local GPU — working!)
- `ext_cache_hit=0.0%` (mooncake external — still zero despite 8 master Get requests)
- Master: 118 GB / 256 GB (46.3%), 97K keys, zero evictions
- Key insight: local GPU prefix cache achieves 87% of theoretical (21.5/24.8), proving prefix sharing exists. GPU KV cache usage too low (2.8%) — blocks stay in GPU and never need the external store.

### Run 5: mooncake-offloading-2bb41e (max-model-len=65536)
- max-model-len: 65536, eviction_high_watermark: 0.99
- Result: **Mooncake external prefix cache working** — first run with meaningful ext_cache_hit
- At 14:39 profiling (of 30 min), 323 requests completed, 0 errors, 0.3 avg RPS
- `theoretical_prefix_cache_hit=92.8%`, `ext_cache_hit=22.2%`, `prefix_cache_hit=3.3%`
- GPU KV cache usage: 3-70% (fluctuates as large requests come and go)
- Master: 251 GB / 256 GB (97.9%), 475 eviction cycles, 5.4M keys evicted (6.45 TB), 540 Get requests (380K items)
- Context overflows: 27 trajectories terminated, avg turn depth 21 (range 5-49), total trajectory turns 19-97
- Combined actual cache hit: ~25% of 93% theoretical — gap driven by master eviction

### Run 6: mooncake-offloading-a5e2db (scheduler.py patch, gpu-memory-utilization=0.85) ← best result
- max-model-len: 65536, eviction_high_watermark: 0.99, gpu-memory-utilization: 0.85
- **vLLM scheduler.py patched** to fix `ValueError: too many values to unpack (expected 1)` crash (see Troubleshooting)
- MLflow run: `ccbdc65157e644c7a3ed534a47eedc44`
- Result: **76% ext_cache_hit, 0 errors, 0 pod restarts**
- 396 requests completed, 0 errors, 0.2 avg RPS
- `theoretical_prefix_cache_hit=92.9%`, `ext_cache_hit=76.0%`, `prefix_cache_hit=4.8%`
- Combined actual cache hit: ~81% of 93% theoretical
- TTFT p50: 2,925 ms (36% faster than Run 5's 4,566 ms)
- Output throughput: 122 tokens/s
- **NVMe disk tier statistics** (from vLLM tier summary logs):
  - Total keys loaded: 634,952
  - From NVMe SSD: 504,862 (79.5%) — 661 GB
  - From DRAM: 130,090 (20.5%) — 170.5 GB
  - Total bytes loaded from mooncake store: 831.5 GB
  - Failed keys: 553 (0.09% — handled gracefully by the patch)
- **Why 76% vs 22%**: The jump from Run 5 was driven by `gpu-memory-utilization=0.85` (was 0.55 in Run 5). At 0.85, GPU KV cache is 35 GB per GPU — blocks are evicted to the mooncake store earlier and more consistently, while still being large enough to hold the active working set. At 0.55, the GPU KV cache was only ~20 GB, causing thrashing between GPU and store. The scheduler patch also prevented the crash-loop that was killing pods in intermediate runs.

## Key Learnings

### 1. max-model-len is the primary tuning lever

The relationship between max-model-len and prefix cache hit rate is non-linear:

| max-model-len | gpu-mem-util | Theoretical prefix cache | External cache hit | Local GPU cache | Requests | Status |
|---------------|-------------|--------------------------|--------------------|-----------------|--------------------|--------|
| 8192 | 0.55 | 0.0% | 0.0% | 0.0% | 76 (stalled) | Too small — traces overflow immediately |
| 32768 | 0.55 | 24.8% | 0.0% | 21.5% | 77 (stalled) | Some prefix sharing, but GPU cache holds everything |
| 65536 | 0.55 | 92.8% | 22.2% | 3.3% | 323+ | Working but limited by eviction |
| **65536** | **0.85** | **92.9%** | **76.0%** | **4.8%** | **396** | **Best — scheduler patch + higher gpu-mem-util** |
| 131072 | 0.55 | (Run 2) | 0.8% | 0.0% | full run | Too large — aggressive eviction destroys metadata |

**Why 65536 works best**: It's large enough for multi-turn traces (avg 21 turns before overflow) generating 93% theoretical prefix sharing, AND it's large enough to pressure the GPU KV cache (52-70% usage), forcing blocks to be evicted to the mooncake store where they can be found by subsequent requests. At 32768, the GPU KV cache is only 2.8% full — blocks stay in GPU and the external store is never needed.

### 1b. gpu-memory-utilization is the secondary tuning lever

`gpu-memory-utilization` controls how much GPU memory is allocated for KV cache. Combined with max-model-len, it determines how quickly blocks are evicted from GPU to the mooncake store.

| gpu-memory-utilization | KV cache per GPU | ext_cache_hit | Why |
|------------------------|------------------|---------------|-----|
| 0.55 | ~20 GB | 22% | Small KV cache causes thrashing — blocks evicted too aggressively |
| **0.85** | **~35 GB** | **76%** | Larger KV cache holds active working set while still pushing older blocks to store |
| 0.90 | ~37 GB | (not tested) | May hold too many blocks in GPU, reducing store utilization |

At 0.85, the GPU KV cache is large enough to hold the active batch's blocks, but not so large that blocks never reach the mooncake store. The 3.5x improvement (22% → 76%) came from this change combined with the scheduler crash fix.

### 2. Master eviction is the main limiter for external cache hits

Even at the best setting (65536), the master evicts 5.4M of 5.6M keys put (96%). The 256 GB combined DRAM pool fills within minutes under 32 concurrent lanes with 65K-token contexts. Each request at 65536 max-model-len can produce ~1,600 KV blocks × 1.3 MB each ≈ 2 GB of KV data.

Eviction watermark 0.99 helps vs 0.95, but the fundamental issue is capacity: 256 GB DRAM can hold only ~130 requests worth of KV data, while the benchmark runs 32 concurrent lanes generating new blocks continuously.

### 3. External vs local prefix cache — different mechanisms

- **Local GPU prefix cache** (`prefix_cache_hit`): Blocks reused from GPU KV cache. Works when GPU cache is large enough to hold reusable blocks. Dominant at lower max-model-len (32768 → 21.5%).
- **External prefix cache** (`ext_cache_hit`): Blocks loaded from mooncake store (DRAM/NVMe via RDMA). Works when blocks have been evicted from GPU, saved to the store, and not yet evicted from the master. Dominant at higher max-model-len (65536 → 22.2%).
- **Non-mooncake OffloadingConnector**: The non-mooncake baseline (`multi-tier-offloading-nvme`) uses a completely different connector (`OffloadingConnector` with `TieringOffloadingSpec`) that offloads within a single engine (GPU → CPU DRAM → NVMe filesystem). Its "external prefix cache hit" measures blocks loaded from local DRAM/NVMe tiers, NOT cross-pod sharing. Direct comparison of hit rates between mooncake and non-mooncake is misleading.

### 4. Weka agentic traces and tokenizer efficiency

The weka traces (`inferencex-agentx-mvp` scenario) replay multi-turn agent conversations. Each turn includes the full prior conversation as context, creating natural prefix sharing. With 680 total conversations and 32 concurrency:
- Turn 1 of trajectory A sends [system + user1]
- Turn 2 sends [system + user1 + assistant1 + user2] — prefix from Turn 1 is shared
- Context grows each turn until max-model-len is exceeded

gemma-4-31B-it's tokenizer is less efficient than Qwen3.6-35B-A3B's. The same text produces more tokens with gemma, causing earlier context overflow. What fits in 8192 tokens with Qwen may need 32K+ with gemma.

### 5. Nginx LB sticky connections

With `proxy_http_version 1.1` and `proxy_set_header Connection ""`, nginx uses HTTP/1.1 keepalive. With streaming responses, connections stick to one backend. Most benchmark traffic goes to pod 1. This actually HELPS prefix cache because all blocks are saved/looked up on the same engine's store segments.

### 6. Master SSD storage not configured

The master shows `SSD Storage: 0 B / 0 B` despite `--enable_disk_eviction=1` being set. The master binary appears to silently ignore this flag in the current version. If disk-backed metadata worked, evicted block metadata could be recovered from disk on lookup, dramatically improving ext_cache_hit rate.

### 7. PYTHONHASHSEED=0 is required

All vLLM pods MUST set `PYTHONHASHSEED=0` so that all TP ranks generate identical block hashes for the same token sequences. Without this, block lookups from one engine can't match blocks saved by another.

## Reproducing the Best Result (Run 6 — 76% ext_cache_hit)

### Prerequisites

1. mooncake-master running with `--eviction_high_watermark_ratio=0.99`
2. vLLM pod(s) running with `--max-model-len=65536`, `--gpu-memory-utilization=0.85`, and `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=4294967296`
3. **Scheduler.py patch applied** (see Troubleshooting → "ValueError crash on SSD read failures")
4. **worker.py SSD offload patch applied** (enables `enable_ssd_offload=True` in `store.setup()`)
5. Mooncake config: embedded mode, 64GB local_buffer_size, RDMA on mlx5_7
6. Nginx LB routing to vLLM pod(s)

### Master args

```
mooncake_master --port=50051 --admin_port=9003
  --eviction_high_watermark_ratio=0.99
  --enable_disk_eviction=1 --enable_offload=1
```

### vLLM args

```
--model=/models/models/google-gemma-4-31B-it
--tensor-parallel-size=2
--max-model-len=65536
--gpu-memory-utilization=0.85
--max-num-seqs=256
--enable-prefix-caching
--kv-cache-metrics
--kv-cache-metrics-sample=0.01
--kv-transfer-config={"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}
--trust-remote-code
--no-enable-log-requests
```

### Required environment variables on vLLM pods

```yaml
env:
  - name: PYTHONHASHSEED
    value: "0"
  - name: USER
    value: benchflow
  - name: MOONCAKE_CONFIG_PATH
    value: /mnt/nvme-kv-cache/mooncake-config/mooncake-rdma-nvme-64g.json
  - name: VLLM_MOONCAKE_STORE_TIER_LOG
    value: "1"
  - name: MOONCAKE_OFFLOAD_FILE_STORAGE_PATH
    value: /mnt/nvme-kv-cache
  - name: MOONCAKE_OFFLOAD_ENABLED
    value: "true"
  - name: MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES
    value: "4294967296"                    # 4GB NVMe staging buffer
  - name: MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES
    value: "8796093022208"                 # 8TB — must exceed accumulated bucket files
```

### Startup patches (in container command, before `exec vllm serve`)

Two patches are needed in vLLM v0.26.0 for gemma-4 models with NVMe SSD offloading:

```bash
# Patch 1: Enable SSD offload in worker.py's store.setup() call
WORKER=/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py
sed -i '/store_config\.master_server_address,$/{ n; s|^        )|            enable_ssd_offload=True,\n            ssd_offload_path=os.getenv("MOONCAKE_OFFLOAD_FILE_STORAGE_PATH", ""),\n        )| }' "$WORKER"

# Patch 2: Fix scheduler.py for hybrid attention models (gemma-4)
SCHED=/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py
if grep -q 'TODO (davidb): add support for hybrid memory allocator' "$SCHED" 2>/dev/null; then
  sed -i 's|            # TODO (davidb): add support for hybrid memory allocator|            # Fixed: flatten block IDs from all KV cache groups|' "$SCHED"
  sed -i 's|            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)|            all_group_block_ids = self.kv_cache_manager.get_block_ids(req_id)\n            req_block_ids = [bid for group in all_group_block_ids for bid in group]|' "$SCHED"
fi
```

### Mooncake config JSON

```json
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "mooncake-master.benchflow.svc.cluster.local:50051",
  "global_segment_size": "64GB",
  "local_buffer_size": "64GB",
  "protocol": "rdma",
  "device_name": "mlx5_7",
  "enable_offload": true,
  "enable_ssd_offload": true,
  "ssd_offload_path": "/mnt/nvme-kv-cache"
}
```

### NVMe RAID-0 setup (node prerequisite)

7x NVMe drives (nvme1n1-nvme7n1) in md0 RAID-0, mounted at `/var/mnt/mooncake-nvme` on gjfjh. nvme0n1 is reserved for benchflow and must not be touched. ~49TB raw, ~47TB usable.

```bash
# Create RAID-0 (one-time, from a privileged pod on gjfjh)
mdadm --create /dev/md0 --level=0 --raid-devices=7 \
  /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1 /dev/nvme4n1 \
  /dev/nvme5n1 /dev/nvme6n1 /dev/nvme7n1
mkfs.xfs /dev/md0
mkdir -p /var/mnt/mooncake-nvme
mount -o noatime,nodiratime /dev/md0 /var/mnt/mooncake-nvme
```

### Submit benchmark

```bash
KUBECONFIG=/Users/rdoddaia/work/aperdomo/Aperdomo-kubeconfig bflow experiment run \
  experiments/rhoai/mooncake-offloading.yaml \
  --target-url http://mooncake-vllm-lb.benchflow.svc:8080 \
  --benchflow-image ghcr.io/albertoperdomo2/benchflow:manual-a53adf3 \
  --cluster-name psap-h100-diadochos
```

### Expected results

- `theoretical_prefix_cache_hit`: ~93%
- `ext_cache_hit`: **70-77%** (stable after cache warmup)
- `prefix_cache_hit`: 3-5% (GPU cache under pressure)
- Combined actual cache hit: ~75-81%
- RPS: 0.2-0.3
- TTFT p50: ~2,900 ms
- Output throughput: ~120 tokens/s
- Error rate: 0%
- Pod restarts: 0
- NVMe tier: ~80% of loaded blocks from NVMe SSD, ~20% from DRAM
- Total bytes loaded from store: ~800 GB per 30-min benchmark

## Future Improvements

1. **`kv_load_failure_policy='recompute'`**: Instead of aborting requests with failed SSD reads, recompute the missing blocks. Set via `--kv-transfer-config='{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute"}'`. The scheduler reads this at `scheduler.py:145-146` and sets `self.recompute_kv_load_failures`. Should eliminate even the 0.09% failed-key impact on request quality.
2. **Larger NVMe staging buffer**: Bump `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES` from 4GB to 16-32GB to reduce `BUFFER_OVERFLOW` frequency during burst reads. gjfjh has 2TB RAM.
3. **Master disk-backed metadata**: If `--enable_disk_eviction` can be made to work, evicted metadata could be recovered from NVMe, potentially pushing ext_cache_hit from 76% toward the 93% theoretical.
4. **Larger DRAM segments**: Increasing `local_buffer_size` beyond 64GB per worker would increase master capacity, reducing eviction pressure. Needs >256GB available host RAM per pod.
5. **East-west (cross-pod) sharing**: Enable pod 2 and fix nginx LB stickiness to distribute traffic. Both pods share the same RDMA NIC on gjfjh, so cross-pod RDMA transfers should work. Requires fixing HTTP/1.1 keepalive stickiness in the LB config.
6. **Upstream the scheduler.py fix**: Report the `(req_block_ids,)` bug to the vLLM project. The fix is straightforward (flatten across groups) and affects all hybrid attention models when SSD reads fail.
7. **Model choice**: Qwen3.6-35B-A3B has a more token-efficient tokenizer, allowing more turns per trajectory at any given max-model-len.

## Troubleshooting

### ValueError crash on SSD read failures with hybrid attention models (CRITICAL)

Symptom: `ValueError: too many values to unpack (expected 1)` in EngineCore → `EngineDeadError` → pod crash-loop. Triggered by `FILE_READ_FAIL` or `BUFFER_OVERFLOW` from NVMe reads.

Root cause: A bug in vLLM v0.26.0's `scheduler.py:2694`:
```python
(req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
```
This single-element tuple unpacking assumes the model has exactly 1 KV cache group. **gemma-4-31B-it has 2 groups** (50 sliding-window-attention layers + 10 full-attention layers), so `get_block_ids()` returns a 2-element tuple. The unpacking fails with `ValueError`.

This code path only executes when `invalid_block_ids` is non-empty — i.e., when mooncake SSD reads fail. Normal operation never hits this bug.

**Crash chain**:
1. C++ `real_client.cpp` → `FILE_READ_FAIL` or `BUFFER_OVERFLOW` from NVMe reads
2. `worker.py:_add_load_error_block_ids()` records failed block IDs
3. `scheduler.py:update_from_output()` → `_handle_invalid_blocks()` → `_update_requests_with_invalid_blocks()`
4. Line 2694: `(req_block_ids,) = ...` tries to unpack 2 KV cache groups into 1 → **ValueError**
5. Exception propagates to `core.py:1332` → `EngineDeadError` → pod crash
6. **Cascading failure**: Pod 1 crashes → master retains stale metadata → pod 2 tries RPC to pod 1 → `RPC_FAIL` → pod 2 crashes too

**Fix** — add this sed patch to the vLLM container startup command (before `exec vllm serve`):
```bash
SCHED=/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py
if grep -q 'TODO (davidb): add support for hybrid memory allocator' "$SCHED" 2>/dev/null; then
  sed -i 's|            # TODO (davidb): add support for hybrid memory allocator|            # Fixed: flatten block IDs from all KV cache groups|' "$SCHED"
  sed -i 's|            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)|            all_group_block_ids = self.kv_cache_manager.get_block_ids(req_id)\n            req_block_ids = [bid for group in all_group_block_ids for bid in group]|' "$SCHED"
  echo "=== Scheduler patch applied: hybrid KV cache group fix ==="
fi
```

This flattens block IDs from all KV cache groups into a single list. Block IDs are globally unique, so membership checks against the flat `invalid_block_ids` set remain correct regardless of group count.

**Impact**: Without this patch, any SSD read failure crashes the entire pod. With the patch, failed reads are handled gracefully — affected requests are either aborted (with `kv_load_failure_policy='fail'`, the default) or recomputed (with `kv_load_failure_policy='recompute'`).

**Affected models**: Any model with hybrid/mixed attention (SWA + full attention layers). Includes gemma-4-31B-it, gemma-4-27B-it, and other models with `layer_types` containing both `sliding_attention` and `full_attention`. Models with a single attention type (e.g., Llama, Qwen) are not affected.

### BUFFER_OVERFLOW crashes
Symptom: `batch_get_offload_object failed with error: BUFFER_OVERFLOW` in vLLM logs. With the scheduler patch applied, this logs a warning but doesn't crash the pod. Without the patch, it triggers the ValueError crash above.
Cause: The NVMe I/O staging buffer (`AlignedClientBufferAllocator`) is too small for concurrent disk reads. Happens during burst activity — 32 concurrent requests each loading 800+ blocks from disk can exceed the buffer capacity.
Fix: Set env var `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=4294967296` (4GB) on vLLM deployments. Consider bumping to 16-32GB on nodes with sufficient RAM (gjfjh has 2TB):
```bash
KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl set env -n benchflow \
  deployment/mooncake-dist-nvme-vllm \
  deployment/mooncake-dist-nvme-vllm-2 \
  MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=4294967296
```

### FILE_READ_FAIL crashes
Symptom: `batch_get_offload_object failed with error: FILE_READ_FAIL`. With the scheduler patch, handled gracefully. Without, triggers ValueError crash.
Cause: Stale bucket files on the NVMe mount. When a pod crashes, bucket files from the previous session remain on the hostPath mount. The master retains metadata pointing to those blocks. When a new pod (or the other pod) tries to read them, the bucket file may be incomplete, corrupted, or owned by a different store instance.
**Cascading failure pattern**: Pod 1 crashes → master still has pod 1's block metadata → pod 2 receives a request, looks up blocks, finds them on pod 1 → tries RPC to pod 1's store → `RPC_FAIL` → pod 2 also crashes.
Fix: Apply the scheduler patch (prevents crash). Optionally clean stale bucket files before deploy:
```bash
kubectl exec deploy/mooncake-dist-nvme-vllm -- \
  find /mnt/nvme-kv-cache -maxdepth 1 -name '*.bucket' -delete
```

### SEGMENT_NOT_FOUND heartbeat errors
Symptom: Continuous `OffloadObjectHeartbeat failed, error code is SEGMENT_NOT_FOUND` in vLLM logs.
Cause: Stale segment registrations after master or vLLM pod restarts.
Fix: Restart mooncake-master and all vLLM pods together.

### Benchmark stalls at 0.0 RPS
Symptom: aiperf reports 0.0 RPS after initial burst of requests, `done` count stops increasing.
Cause: All trajectories terminated due to context-overflow. max-model-len is too small for the traces.
Fix: Increase max-model-len. For gemma-4-31B-it with weka agentic traces, 65536 is the sweet spot.

### External prefix cache 0% but local prefix cache working
Symptom: `prefix_cache_hit` is non-zero but `ext_cache_hit` is 0%.
Cause: GPU KV cache has enough room to hold reusable blocks — they never need to be loaded from the external store.
Fix: Increase max-model-len to put more pressure on GPU KV cache, forcing blocks to be evicted and served from the mooncake store.

### SSD read failures under high concurrency
Symptom: `FILE_READ_FAIL` or `BUFFER_OVERFLOW` errors in vLLM logs. With the scheduler patch, these are handled gracefully (affected requests aborted or recomputed). Without the patch, they crash the pod.
Cause: Both types are primarily triggered during burst read activity — 32 concurrent requests each loading 800+ KV cache blocks from NVMe disk. `BUFFER_OVERFLOW` means the staging buffer is full. `FILE_READ_FAIL` means a bucket file on disk is corrupted, missing, or from a stale session.
Mitigations (stack them):
1. **Apply scheduler.py patch** (prevents crash — see above)
2. **Increase staging buffer**: `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=17179869184` (16GB)
3. **Set `kv_load_failure_policy='recompute'`**: Failed blocks are recomputed instead of aborting the request
4. **Clean stale bucket files** before deploy
5. **Restart master** when restarting pods to clear stale metadata (prevents cascading RPC_FAIL)

### Image pull timeout
Symptom: vLLM image (~8-10GB) exceeds rollout deadline.
Fix: Pre-pull with a DaemonSet targeting GPU nodes before deploying.
