# Mooncake North-South KV Cache Offloading: Getting External Prefix Cache Hits

This document covers the exact changes made to go from 0% to **76% external prefix cache hit rate** with mooncake's MooncakeStoreConnector on the diadochos cluster. Use this as a step-by-step guide to reproduce the result.

> **Latest result (Run 6)**: 76% ext_cache_hit, 0 errors, 0 pod restarts, 831 GB loaded from mooncake store (79.5% from NVMe SSD). See [MOONCAKE-BENCHMARKING.md](MOONCAKE-BENCHMARKING.md) for full benchmark history.

## What is North-South KV Cache Traffic?

North-south refers to **vertical** movement of KV cache blocks within a single pod:

```
┌─────────────────────────────────────────┐
│                  Pod 1                  │
│                                         │
│   GPU KV Cache  ◄──── compute ────►     │
│        │                                │
│        │ evict (save_put)               │
│        ▼                                │
│   Mooncake DRAM Segment (64GB×2 ranks)  │
│        │                                │
│        │ offload                        │
│        ▼                                │
│   NVMe RAID-0 (SSD tier)               │
│        │                                │
│        │ load_get (on prefix match)     │
│        ▼                                │
│   GPU KV Cache  ◄── reuse ──►           │
│                                         │
└─────────────────────────────────────────┘
```

A request's KV blocks are computed on GPU, saved to the mooncake DRAM store, and when a later request shares the same prefix, those blocks are loaded back from the store instead of recomputing — that's an **external prefix cache hit**.

## The Problem: 0% External Prefix Cache Hits

Starting config (Run 2):
- `max-model-len=131072`
- `eviction_high_watermark_ratio=0.95` (master default)
- `local_buffer_size=64GB` per TP worker (4 workers = 256GB total master DRAM)

Result: 0.8% ext_cache_hit. Master evicted 69 times in 40 min, destroying block metadata faster than it could be reused. 234M keys looked up, only 4,726 loaded (0.002%).

## What We Changed

### Change 1: Master eviction watermark 0.95 → 0.99

The master evicts block metadata when memory usage exceeds the high watermark. At 0.95, eviction starts at 243 GB / 256 GB. At 0.99, it starts at 253 GB / 256 GB — keeping blocks around longer.

```bash
# Master deployment — patch the command args
KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl get deployment mooncake-master -n benchflow -o json \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
# Find the container and update/add the eviction flag in the command
container = d['spec']['template']['spec']['containers'][0]
# Add --eviction_high_watermark_ratio=0.99 to the command args
# (exact method depends on how your master command is structured)
json.dump(d, sys.stdout)
" | KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl apply -f -
```

Master startup args that worked:
```
mooncake_master --port=50051 --admin_port=9003 \
  --eviction_high_watermark_ratio=0.99 \
  --enable_disk_eviction=1 \
  --enable_offload=1
```

**Impact**: Reduced eviction frequency. But alone this was not enough — the real bottleneck was max-model-len.

### Change 2: max-model-len tuning (the critical change)

This was the single most impactful parameter. We iterated through 4 values:

| max-model-len | gpu-mem-util | What happened | Why |
|---------------|-------------|---------------|-----|
| 8192 | 0.55 | 0% cache hit, benchmark stalled at 76 requests | Traces overflow at turn 1-7. Not enough context for multi-turn conversations. `theoretical_prefix_cache_hit=0%` — no shared prefixes exist at this length. |
| 32768 | 0.55 | 21.5% LOCAL cache hit, 0% EXTERNAL | Traces fit ~25 turns. Prefix sharing exists (24.8% theoretical). But GPU KV cache usage only 2.8% — blocks stay in GPU, never need the external store. |
| 65536 | 0.55 | 22-24% EXTERNAL cache hit | Working — traces fit ~21 turns avg (93% theoretical prefix sharing). GPU pressured enough to push blocks to store. |
| **65536** | **0.85** | **76% EXTERNAL cache hit** | **Best result.** Larger GPU KV cache (35 GB vs 20 GB) holds active working set while steadily pushing older blocks to store. Combined with scheduler patch. |
| 131072 | 0.55 | 0.8% external cache hit | Store fills too fast. 131K tokens per request = ~2 GB KV data. Master evicts 96%+ of keys before they can be reused. |

The key insight: **max-model-len controls TWO things simultaneously**:
1. How many turns of a multi-turn conversation fit (determines prefix sharing opportunity)
2. How much GPU KV cache pressure exists (determines whether blocks get pushed to the external store)

At 65536, both conditions are met: high prefix sharing AND enough GPU pressure to use the store.

### Change 2b: gpu-memory-utilization tuning

`gpu-memory-utilization` was the second major lever. Going from 0.55 to 0.85 produced a 3.5x improvement in ext_cache_hit (22% → 76%). At 0.85, each GPU allocates ~35 GB for KV cache instead of ~20 GB. The larger cache holds the active batch's blocks without thrashing, while older blocks are steadily evicted to the mooncake store where they accumulate and produce cache hits.

```bash
# Patch both vLLM pods
for deploy in mooncake-dist-nvme-vllm mooncake-dist-nvme-vllm-2; do
  KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl get deployment $deploy -n benchflow -o json \
    | python3 -c "
import sys, json, re
d = json.load(sys.stdin)
cmd = d['spec']['template']['spec']['containers'][0]['command']
cmd = [re.sub(r'--max-model-len=\d+', '--max-model-len=65536', c) for c in cmd]
d['spec']['template']['spec']['containers'][0]['command'] = cmd
json.dump(d, sys.stdout)
" | KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl apply -f -
done
```

### Change 3: Scheduler.py patch for hybrid attention models (CRITICAL for gemma-4)

vLLM v0.26.0 has a bug in `scheduler.py:2694` that crashes the pod when NVMe SSD reads fail:

```python
# BUGGY — assumes exactly 1 KV cache group:
(req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
```

gemma-4-31B-it has **2 KV cache groups** (50 sliding-window-attention + 10 full-attention layers). When `get_block_ids()` returns a 2-element tuple, the single-element unpacking raises `ValueError: too many values to unpack (expected 1)` → `EngineDeadError` → pod crash.

This only triggers on the error path — when mooncake SSD reads fail and produce `invalid_block_ids`. Without SSD read failures, the bug is dormant.

**Fix** — add to container startup command (before `exec vllm serve`):

```bash
SCHED=/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py
if grep -q 'TODO (davidb): add support for hybrid memory allocator' "$SCHED" 2>/dev/null; then
  sed -i 's|            # TODO (davidb): add support for hybrid memory allocator|            # Fixed: flatten block IDs from all KV cache groups|' "$SCHED"
  sed -i 's|            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)|            all_group_block_ids = self.kv_cache_manager.get_block_ids(req_id)\n            req_block_ids = [bid for group in all_group_block_ids for bid in group]|' "$SCHED"
  echo "=== Scheduler patch applied ==="
fi
```

**Impact**: Without this patch, any SSD read failure (FILE_READ_FAIL, BUFFER_OVERFLOW, RPC_FAIL) crashes the entire pod. With it, failed reads are handled gracefully — the 0.09% failure rate in Run 6 produced zero crashes.

### Change 4: Always restart master with vLLM pods

After any vLLM config change, always restart the master too. Stale segment registrations cause `SEGMENT_NOT_FOUND` errors and prevent block discovery.

```bash
KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl rollout restart -n benchflow \
  deployment/mooncake-master \
  deployment/mooncake-dist-nvme-vllm \
  deployment/mooncake-dist-nvme-vllm-2
```

## Complete Working Configuration

### Model

- **google/gemma-4-31B-it**
- TP=2 (tensor parallelism across 2 GPUs per pod)
- `gpu-memory-utilization=0.55`

Note: gemma-4-31B-it has a less token-efficient tokenizer than Qwen3.6-35B-A3B. The same text produces more tokens with gemma, so max-model-len needs to be higher to fit the same number of conversation turns.

### Master Deployment

```yaml
# Key args in the container command:
mooncake_master
  --port=50051
  --admin_port=9003
  --eviction_high_watermark_ratio=0.99
  --enable_disk_eviction=1
  --enable_offload=1
```

Note: `--enable_disk_eviction=1` is set but silently ignored by the current binary version (SSD Storage remains 0 B). If a future version supports it, this would allow evicted metadata to be recovered from disk, potentially increasing ext_cache_hit significantly.

### vLLM Deployment (both pods)

Container command args:
```
vllm serve /models/models/google-gemma-4-31B-it \
  --port 30080 \
  --host 0.0.0.0 \
  --served-model-name google/gemma-4-31B-it \
  --tensor-parallel-size 2 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 256 \
  --enable-prefix-caching \
  --kv-cache-metrics \
  --kv-cache-metrics-sample 0.01 \
  --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}' \
  --trust-remote-code \
  --no-enable-log-requests
```

Required environment variables:
```yaml
env:
  - name: PYTHONHASHSEED
    value: "0"                          # All TP ranks must produce identical block hashes
  - name: MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES
    value: "4294967296"                  # 4GB NVMe staging buffer (default 32MB causes BUFFER_OVERFLOW)
  - name: MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES
    value: "8796093022208"               # 8TB — default 2TB disables offloading when bucket files accumulate
  - name: MOONCAKE_OFFLOAD_FILE_STORAGE_PATH
    value: /mnt/nvme-kv-cache            # SSD tier storage directory
  - name: MOONCAKE_OFFLOAD_ENABLED
    value: "true"
  - name: VLLM_MOONCAKE_STORE_TIER_LOG
    value: "1"                           # Emits disk_keys vs memory_keys in tier summaries
  - name: USER
    value: benchflow
  - name: MOONCAKE_CONFIG_PATH
    value: /mnt/nvme-kv-cache/mooncake-config/mooncake-rdma-nvme-64g.json
```

Required startup patches (in bash -c command, before `exec vllm serve`):
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

### Mooncake Config File

Path on node: `/mnt/nvme-kv-cache/mooncake-config/mooncake-rdma-nvme-64g.json`
(hostPath mount, not a ConfigMap)

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

Key parameter relationships:
- `local_buffer_size=64GB` × 4 TP workers (2 pods × 2 TP ranks) = **256 GB total master DRAM capacity**
- `global_segment_size=64GB` — each TP worker's RDMA-pinned memory segment
- `mode=embedded` — each TP worker runs its own mooncake store (no sidecar)
- `device_name=mlx5_7` — the Mellanox NIC on gjfjh for RDMA
- `ssd_offload_path=/mnt/nvme-kv-cache` — NVMe RAID-0 mount for disk tier

### Nginx Load Balancer

ConfigMap `mooncake-nginx-lb-config`:
```nginx
worker_processes auto;
events { worker_connections 4096; }
http {
  upstream vllm_pool {
    server 10.243.65.15:30080;   # pod 1
    server 10.243.65.15:30081;   # pod 2
  }
  server {
    listen 8080;
    location / {
      proxy_pass http://vllm_pool;
      proxy_http_version 1.1;
      proxy_set_header Connection "";
      proxy_set_header Host $host;
      proxy_read_timeout 600s;
      proxy_send_timeout 600s;
    }
  }
}
```

**Known issue**: HTTP/1.1 keepalive with streaming responses causes connection stickiness. All traffic goes to pod 1. This means all ext_cache_hits are north-south only (within pod 1). Pod 2 never saves or serves blocks. This will need to be fixed for east-west testing.

### Benchmark Traces

- Tool: aiperf
- Scenario: `inferencex-agentx-mvp`
- Dataset: `semianalysisai/cc-traces-weka-with-subagents-060826` (680 conversations)
- Concurrency: 32
- Duration: 1800s
- Streaming: true

The traces are multi-turn agent conversations. Each turn includes the full prior conversation as context, creating natural prefix sharing:
- Turn 1: [system + user1] → ~2,400 tokens
- Turn 2: [system + user1 + assistant1 + user2] → ~5,000 tokens
- Turn N: accumulates until context overflow at max-model-len

With max-model-len=65536, trajectories reach an average of 21 turns (range 5-49) before overflowing.

### NVMe RAID-0

7× NVMe drives (nvme1n1-nvme7n1) in md0 RAID-0, mounted at `/var/mnt/mooncake-nvme` on gjfjh. nvme0n1 is reserved for benchflow.

### Hardware (gjfjh node)

- 8× H100 GPUs
- 8× Mellanox ConnectX NICs (mlx5_0 through mlx5_7, all ACTIVE)
- RDMA protocol for KV cache transfers
- hostNetwork on vLLM pods (ports 30080, 30081)

## Submitting the Benchmark

```bash
# 1. Extract diadochos kubeconfig
export DIADOCHOS_KUBECONFIG=/tmp/diadochos-kubeconfig.tmp
KUBECONFIG=/Users/rdoddaia/work/aperdomo/Aperdomo-kubeconfig \
  kubectl get secret psap-h100-diadochos -n benchflow \
  -o jsonpath='{.data.kubeconfig}' | base64 -d > $DIADOCHOS_KUBECONFIG

# 2. Verify endpoint
KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl exec -n benchflow deploy/mooncake-nginx-lb -- \
  curl -s http://localhost:8080/v1/models | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(f'model={d[\"data\"][0][\"id\"]}, max_model_len={d[\"data\"][0][\"max_model_len\"]}')"
# Expected: model=google/gemma-4-31B-it, max_model_len=65536

# 3. Submit
KUBECONFIG=/Users/rdoddaia/work/aperdomo/Aperdomo-kubeconfig bflow experiment run \
  experiments/rhoai/mooncake-offloading.yaml \
  --target-url http://mooncake-vllm-lb.benchflow.svc:8080 \
  --benchflow-image ghcr.io/albertoperdomo2/benchflow:manual-a53adf3 \
  --cluster-name psap-h100-diadochos
```

## How to Verify It's Working

### 1. Check aiperf metrics (primary indicator)

```bash
# Find the benchmark job on diadochos
KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl get jobs -n benchflow | grep benchmark-mooncake

# Watch live metrics
KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl logs -f -n benchflow job/<job-name> \
  | grep -E "profiling|ext_cache"
```

What to look for:
```
trace theoretical_prefix_cache_hit=93.4%     ← should be >80%
srv  prefix_cache_hit=8.1% ... ext_cache_hit=24.3%  ← ext_cache_hit should be >15%
```

If `theoretical_prefix_cache_hit=0%`: max-model-len is too small, traces are overflowing.
If `ext_cache_hit=0%` but `prefix_cache_hit` is high: GPU KV cache is holding everything, increase max-model-len.

### 2. Check vLLM KV transfer metrics

```bash
KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl logs -f -n benchflow deployment/mooncake-dist-nvme-vllm \
  | grep -E "KV Transfer|External prefix"
```

What to look for:
```
External prefix cache hit rate: 22.3%        ← should be >15%
KV Transfer metrics: lookup_exists_total_bytes=XXX  ← should be non-zero
```

If `lookup_exists_total_bytes=0`: blocks are being saved but not found. Check master eviction.

### 3. Check master state

```bash
KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl logs -f -n benchflow deployment/mooncake-master \
  | grep -oE '(Mem Storage[^|]+|Eviction[^|]+|Get:\([^)]+\))'
```

What to look for:
```
Mem Storage: 250 GB / 256 GB (97.9%)         ← expect near capacity
Get:(Req=668/0/668, Item=455712/455712)      ← Get count should be non-zero and growing
Eviction: Success/Attempts=485/485, keys=5531631  ← eviction happening but blocks still found
```

If `Get:(Req=0/0/0, Item=0/0)`: no blocks being loaded from store. Check if lookup_exists is finding anything.

## Results Achieved (Run 6 — Best)

| Metric | Run 6 (best) | Run 5 (prev best) |
|--------|-------------|-------------------|
| PipelineRun | `mooncake-offloading-a5e2db` | `mooncake-offloading-2bb41e` |
| MLflow run | `ccbdc65157e644c7a3ed534a47eedc44` | `52ee6440cf15480f998604d4d9a02b61` |
| gpu-memory-utilization | 0.85 | 0.55 |
| Scheduler patch | Yes | No |
| Requests (profiling) | 396 | 348 |
| Errors | **0** | 0 |
| Pod restarts | **0** | 0 |
| Theoretical prefix cache | 92.9% | 93.4% |
| **External prefix cache** | **76.0%** | 22-24% |
| Local GPU prefix cache | 4.8% | 8-21% |
| **Combined actual** | **~81%** | ~30-44% |
| TTFT p50 | **2,925 ms** | 4,566 ms |
| Output throughput | 122 tokens/s | 112.3 tokens/s |
| RPS | 0.2 | 0.19 |
| **NVMe disk reads** | **504,862 keys (661 GB)** | N/A |
| **DRAM reads** | **130,090 keys (170 GB)** | N/A |
| **Total store reads** | **831.5 GB** | N/A |
| Failed keys | 553 (0.09%) | 0 |

### What changed between Run 5 and Run 6

1. **gpu-memory-utilization 0.55 → 0.85**: Larger GPU KV cache (35 GB vs 20 GB per GPU) holds active working set without thrashing, while steadily evicting older blocks to the store. This was the primary driver of the 22% → 76% improvement.
2. **Scheduler.py patch applied**: Fixed the `ValueError: too many values to unpack (expected 1)` crash that killed pods on SSD read failures. Without this patch, intermediate runs (between Run 5 and Run 6) had 80-84% error rates from pod crash-loops.
3. **NVMe SSD tier actively serving**: 79.5% of all loaded blocks came from NVMe disk (661 GB), 20.5% from DRAM (170 GB). The SSD tier is the backbone of the cache — DRAM fills to capacity quickly, but SSD-backed blocks persist and serve hits long after DRAM eviction.

## Current Limitations (to Address in East-West Phase)

1. **All traffic goes to pod 1** — nginx keepalive stickiness means pod 2 never receives inference requests, so zero east-west (cross-pod) sharing occurs. All ext_cache_hits are north-south within pod 1.

2. **Master eviction still active** — with gpu-memory-utilization=0.85, eviction pressure is much lower than at 0.55, but the master still evicts blocks. The 76% vs 93% gap is partly due to eviction and partly due to timing (blocks not yet saved when the next request arrives).

3. **SSD read failures under burst load** — 0.09% of keys fail to load (553 out of 634,952). The scheduler patch prevents crashes, but `kv_load_failure_policy='recompute'` would further improve reliability by recomputing failed blocks instead of aborting requests.

4. **NVMe staging buffer limits** — `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=4GB` can still cause `BUFFER_OVERFLOW` during high burst activity. Bumping to 16-32GB would help (node has 2TB RAM).

5. **Startup patches required** — two sed patches must be applied on every pod startup (worker.py SSD offload + scheduler.py hybrid attention fix). These should be upstreamed to vLLM.

## Next: East-West Traffic

To enable cross-pod KV cache sharing (east-west), the following changes are needed:

1. **Fix nginx LB distribution** — disable keepalive or use `least_conn` so both pods receive inference traffic
2. **Verify PoolKey compatibility** — ensure blocks saved by pod 1 can be found and loaded by pod 2 (PoolKeys don't include engine_id, so this should work)
3. **RDMA cross-pod transfer** — verify pod 2 can RDMA-read from pod 1's DRAM segments (both on gjfjh via hostNetwork, so same-node RDMA should work)
4. **Monitor master Get patterns** — look for Gets where the requesting segment differs from the owning segment (cross-pod load)
