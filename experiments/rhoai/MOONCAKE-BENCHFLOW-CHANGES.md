# Mooncake Integration: Changes Needed in Benchflow

This document covers all the changes, configs, and parameters needed in benchflow to make mooncake work for both north-south (single-pod DRAM+NVMe offloading) and east-west (cross-pod RDMA sharing) KV cache sharing with high external prefix cache hits.

## Current State vs. What's Needed

### What benchflow supports today

| Feature | Profile | Status |
|---------|---------|--------|
| Embedded DRAM-only | `mooncake-embedded-cpu` | Works but 0% ext_cache_hit with default params |
| Standalone-store NVMe | `mooncake-nvme` | Has sidecar `mooncake_client`, not embedded SSD |
| Predeployed (skip vLLM deploy) | `mooncake-embedded-cpu-predeployed` | Works with manual deployment |

### What's missing for high ext_cache_hits

| Gap | Impact | Fix |
|-----|--------|-----|
| `max-model-len=8192` default | 0% prefix cache hit — traces overflow | Profile or experiment override to 65536 |
| No `--eviction_high_watermark_ratio` | Master evicts too aggressively at 0.95 default | Add `--eviction_high_watermark_ratio=0.99` to master |
| No `--admin_port` on master | Can't scrape master metrics (Mem/SSD usage, eviction stats) | Add `--admin_port=9003` to master |
| No NVMe SSD tier in embedded mode | Blocks never go to SSD, only DRAM | New profile or extend embedded profile |
| No multi-replica vLLM | Only 1 vLLM pod = no east-west traffic | Support `replicas: 2` with different ports |
| No load balancer | Need nginx/envoy LB for multi-pod | Deploy LB with `hash $request_id consistent` |
| Missing `gpu-memory-utilization` tuning | Default 0.9 means GPU holds everything, store unused | Override to 0.55-0.85 depending on model |
| `worker.py` sed patch for embedded SSD | `store.setup()` doesn't pass `enable_ssd_offload` | Upstream vLLM fix or startup script patch |
| Missing `MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES` | Default 2TB limit disables offloading when buckets accumulate | Set to match NVMe capacity |

---

## Change 1: Deployment Profile for Embedded NVMe SSD Tier

Create a new profile `mooncake-embedded-nvme.yaml`:

```yaml
apiVersion: benchflow.io/v1alpha1
kind: DeploymentProfile
metadata:
  name: mooncake-embedded-nvme
spec:
  platform: rhoai
  mode: distributed-default
  platform_version: RHOAI-3.5.0-ea.2
  platform_channel: beta
  endpoint_path: /v1/models
  model_storage:
    pvc_name: models-storage
    cache_dir: /models
  runtime:
    image: vllm/vllm-openai:v0.26.0
    replicas: 1
    tensor_parallelism: 2
    env:
      USER: benchflow
      PYTHONHASHSEED: "0"
      VLLM_MOONCAKE_STORE_TIER_LOG: "1"
      MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES: "4294967296"    # 4GB NVMe staging buffer
      MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES: "8796093022208"  # 8TB
    service_account_name: benchflow-hostpath-runtime
    host_paths:
      - name: mooncake-nvme
        host_path: /var/mnt/mooncake-nvme          # RAID-0 of 7x NVMe drives
        mount_path: /mnt/nvme-kv-cache
        type: Directory
        read_only: false
    vllm_args:
      - --max-model-len=65536
      - --gpu-memory-utilization=0.85
      - --max-num-seqs=256
      - --kv-transfer-config={"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}
      - --trust-remote-code
      - --no-enable-log-requests
      - --enable-prefix-caching
      - --kv-cache-metrics
      - --kv-cache-metrics-sample=0.01
  options:
    enable_auth: false
    startup_probe:
      failureThreshold: 180
    mooncake_store:
      global_segment_size: 64GB
      local_buffer_size: 64GB
      protocol: rdma
      device_name: mlx5_7
      mode: embedded
      # NEW fields needed for embedded SSD tier:
      enable_ssd_offload: true
      ssd_offload_path: /mnt/nvme-kv-cache
```

### Code changes needed in `rhoai_mooncake.py`

#### 1. Extend `RhoaiMooncakeSpec` for embedded SSD

```python
@dataclass(frozen=True, slots=True)
class RhoaiMooncakeSpec:
    mode: str
    global_segment_size: str
    local_buffer_size: str
    protocol: str
    device_name: str
    store_global_segment_size: str = ""
    host_path_name: str = ""
    offload_path: str = ""
    offload_size_limit_bytes: str = ""
    # NEW:
    enable_ssd_offload: bool = False
    ssd_offload_path: str = ""

    @property
    def is_nvme(self) -> bool:
        return self.mode == "standalone-store"

    @property
    def has_ssd_offload(self) -> bool:
        return self.enable_ssd_offload or self.is_nvme
```

#### 2. Include SSD fields in mooncake config JSON

In `render_rhoai_mooncake_manifests()`:

```python
config = {
    "mode": spec.mode,
    "metadata_server": "P2PHANDSHAKE",
    "master_server_address": _master_address(plan),
    "global_segment_size": spec.global_segment_size,
    "local_buffer_size": spec.local_buffer_size,
    "protocol": spec.protocol,
    "device_name": spec.device_name,
    "enable_offload": spec.has_ssd_offload,
    # NEW:
    "enable_ssd_offload": spec.enable_ssd_offload,
    "ssd_offload_path": spec.ssd_offload_path,
}
```

#### 3. Add master offload flags for embedded SSD

```python
if spec.has_ssd_offload:
    master_flags += [
        "--enable_offload=true",
        "--enable_disk_eviction=true",
        "--eviction_high_watermark_ratio=0.99",
        "--admin_port=9003",
    ]
```

#### 4. Add env vars for embedded SSD

In `rhoai_mooncake_model_env()`:

```python
if spec.enable_ssd_offload:
    env.extend([
        {"name": "MOONCAKE_OFFLOAD_FILE_STORAGE_PATH", "value": spec.ssd_offload_path},
        {"name": "MOONCAKE_OFFLOAD_ENABLED", "value": "true"},
    ])
```

#### 5. vLLM startup patches

Two patches are needed for vLLM v0.26.0 with gemma-4 and NVMe SSD offloading:

```python
if spec.enable_ssd_offload:
    startup_patches = """
# Patch 1: Enable SSD offload in worker.py's store.setup() call
WORKER=/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py
sed -i '/store_config\\.master_server_address,$/{ n; s|^        )|            enable_ssd_offload=True,\\n            ssd_offload_path=os.getenv("MOONCAKE_OFFLOAD_FILE_STORAGE_PATH", ""),\\n        )| }' "$WORKER"

# Patch 2: Fix scheduler.py for hybrid attention models (gemma-4)
# Bug: (req_block_ids,) unpacking fails when model has >1 KV cache group
# (gemma-4 has 2 groups: 50 SWA layers + 10 full attention layers)
SCHED=/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py
if grep -q 'TODO (davidb): add support for hybrid memory allocator' "$SCHED" 2>/dev/null; then
  sed -i 's|            # TODO (davidb): add support for hybrid memory allocator|            # Fixed: flatten block IDs from all KV cache groups|' "$SCHED"
  sed -i 's|            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)|            all_group_block_ids = self.kv_cache_manager.get_block_ids(req_id)\\n            req_block_ids = [bid for group in all_group_block_ids for bid in group]|' "$SCHED"
fi
"""
```

**Patch 1** adds `enable_ssd_offload=True` and `ssd_offload_path` kwargs to `worker.py`'s `self.store.setup()` call.

**Patch 2** fixes a crash in `scheduler.py:2694` where `(req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)` assumes exactly 1 KV cache group. Models with hybrid attention (like gemma-4) have 2+ groups, causing `ValueError: too many values to unpack (expected 1)` whenever NVMe SSD reads fail. The fix flattens block IDs from all groups into a single list — block IDs are globally unique so this is safe.

---

## Change 2: Master Configuration

### Current benchflow master flags (NVMe mode only)

```
--rpc_port=50051 --rpc_address=0.0.0.0
--enable_offload=true --offload_on_evict=true --enable_disk_eviction=true
```

### Required master flags for high ext_cache_hits

```
mooncake_master
  --port=50051
  --admin_port=9003                         # Exposes metrics endpoint
  --eviction_high_watermark_ratio=0.99      # Keep blocks longer before evicting
  --enable_offload=true                     # Enable SSD offload tier
  --enable_disk_eviction=true               # Evict to SSD instead of deleting
```

### Why each flag matters

| Flag | Default | Recommended | Impact |
|------|---------|-------------|--------|
| `eviction_high_watermark_ratio` | 0.95 | 0.99 | At 0.95, eviction starts at 243/256 GB. At 0.99, at 253/256 GB. Higher = more blocks survive for reuse |
| `enable_offload` | false | true | Required for SSD tier to accept data |
| `enable_disk_eviction` | false | true | Evicted blocks go to SSD instead of being deleted (note: silently ignored by master binary in v0.3.10.post2) |
| `admin_port` | none | 9003 | Exposes Mem/SSD storage, eviction, and request metrics every 10s |
| `offload_on_evict` | false | false (omit) | false = immediate offload at PutEnd (what we want). true = defer to eviction time |

---

## Change 3: Environment Variables

### Required on vLLM pods

| Variable | Value | Why |
|----------|-------|-----|
| `PYTHONHASHSEED` | `0` | All TP ranks must produce identical block hashes |
| `USER` | `benchflow` | vLLM needs a USER env var |
| `MOONCAKE_CONFIG_PATH` | `/etc/benchflow/mooncake/mooncake_config.json` | Points to mooncake config |
| `VLLM_MOONCAKE_STORE_TIER_LOG` | `1` | Emits `disk_keys` vs `memory_keys` in load tier summaries |
| `MOONCAKE_OFFLOAD_FILE_STORAGE_PATH` | `/mnt/nvme-kv-cache` | SSD tier storage directory (the NVMe mount) |
| `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES` | `4294967296` | 4GB NVMe staging buffer. Default 32MB causes `BUFFER_OVERFLOW` |
| `MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES` | `8796093022208` | 8TB total SSD limit. **Default 2TB is too small** — causes `IsEnableOffloading()` to return false when bucket files accumulate |
| `MOONCAKE_OFFLOAD_ENABLED` | `true` | Master flag for client offloading |

### The `MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES` problem

The `BucketStorageBackend::IsEnableOffloading()` function checks:
```
total_size + bucket_size_limit (256MB) <= total_size_limit (default 2TB)
```

When bucket files accumulate across pod restarts on the shared hostPath mount, they can exceed 2TB. At that point `IsEnableOffloading()` returns `false` and the master reports `enable offloading is: 0`. Setting this to 8TB (or matching actual NVMe capacity) fixes the issue.

---

## Change 4: Mooncake Config JSON

### Embedded DRAM-only (current)

```json
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "<release>-mooncake-master.<ns>.svc:50051",
  "global_segment_size": "64GB",
  "local_buffer_size": "64GB",
  "protocol": "rdma",
  "device_name": "mlx5_7",
  "enable_offload": false
}
```

### Embedded with NVMe SSD tier (needed)

```json
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "<release>-mooncake-master.<ns>.svc:50051",
  "global_segment_size": "64GB",
  "local_buffer_size": "64GB",
  "protocol": "rdma",
  "device_name": "mlx5_7",
  "enable_offload": true,
  "enable_ssd_offload": true,
  "ssd_offload_path": "/mnt/nvme-kv-cache"
}
```

---

## Change 5: vLLM Parameters for High Cache Hits

### Critical: max-model-len

This is the single most impactful parameter. Benchflow should expose it as an experiment-level override and set the right default per workload.

| max-model-len | ext_cache_hit | Why |
|---------------|---------------|-----|
| 8192 (default) | 0% | Traces overflow, no shared prefixes |
| 32768 | 0% external, 21% local | GPU holds everything, store unused |
| **65536** | **22-24%** | Sweet spot: GPU pressured, store active |
| 131072 | 0.8% | Master eviction destroys 96%+ of keys |

For gemma-4-31B-it with weka agentic traces: **max-model-len=65536**.

### gpu-memory-utilization

Default 0.9 means GPU KV cache is large enough to hold most blocks without needing the external store. Reducing to 0.55-0.85 forces blocks to the mooncake store, where they get ext_cache_hits.

### Recommended vLLM args

```
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

---

## Change 6: Multi-Pod / East-West Support

### Current limitation

Benchflow deploys 1 vLLM replica. All ext_cache_hits are north-south (within the single pod). For east-west (cross-pod) sharing, need:

### Required changes

1. **Multiple vLLM replicas**: Deploy 2+ vLLM pods with different hostNetwork ports

```yaml
# In experiment or deployment profile:
spec:
  runtime:
    replicas: 2
```

Each replica needs a unique hostNetwork port (30080, 30081, etc.) for the mooncake store's RDMA endpoint.

2. **Load balancer**: Deploy nginx/envoy that distributes requests across pods

```nginx
upstream vllm_pool {
  hash $request_id consistent;    # Same conversation goes to same pod
  server <node-ip>:30080;
  server <node-ip>:30081;
}
```

**Known issue**: HTTP/1.1 keepalive with streaming responses causes connection stickiness. All traffic goes to one pod, preventing east-west sharing. Fix: use `hash $request_id consistent` to break stickiness while keeping same-conversation affinity.

3. **RDMA cross-pod transfer**: Both pods must use the same RDMA NIC (`device_name: mlx5_7`) and be on the same node (hostNetwork). The mooncake store handles RDMA transfers automatically.

### Benchflow changes for multi-pod

Add LB deployment/service to `render_rhoai_mooncake_manifests()`:

```python
def _render_nginx_lb(plan, replicas, node_ip, base_port=30080):
    upstream_servers = "\n".join(
        f"    server {node_ip}:{base_port + i};"
        for i in range(replicas)
    )
    nginx_config = f"""
worker_processes auto;
events {{ worker_connections 4096; }}
http {{
  upstream vllm_pool {{
    hash $request_id consistent;
    {upstream_servers}
  }}
  server {{
    listen 8080;
    location / {{
      proxy_pass http://vllm_pool;
      proxy_http_version 1.1;
      proxy_set_header Connection "";
      proxy_set_header Host $host;
      proxy_read_timeout 600s;
      proxy_send_timeout 600s;
    }}
  }}
}}"""
    # Return ConfigMap + Deployment + Service for nginx LB
```

---

## Change 7: Benchmark Profile

### aiperf-agentx-inference.yaml needs adjustment

The profile has `min_max_model_len: 131072` but mooncake works best at 65536. Either:
- Remove the `min_max_model_len` requirement (it's a validation check, not a parameter)
- Or create a separate profile for mooncake workloads

### Recommended benchmark settings

```yaml
aiperf:
  scenario: inferencex-agentx-mvp
  public_dataset: weka_hf
  hf_weka_repo: semianalysisai/cc-traces-weka-with-subagents-060826
  endpoint_type: chat
  endpoint_path: /v1/chat/completions
  streaming: true
  use_server_token_count: true
  tokenizer_trust_remote_code: true
  max_context_length: 65536      # Match max-model-len
  concurrency: 32
  benchmark_duration: 1800
  max_seconds: 7200
  random_seed: 42
```

---

## Change 8: NVMe RAID-0 Setup (Node Prerequisite)

The NVMe RAID-0 must exist on the node before deploying. This is a node-level prerequisite, not a benchflow deployment concern.

On diadochos gjfjh:
- 7x NVMe drives (nvme1n1-nvme7n1) in md0 RAID-0
- Mounted at `/var/mnt/mooncake-nvme` (hostPath)
- nvme0n1 reserved for benchflow
- ~49TB raw, ~47TB usable

For benchflow to support this generically, document the node prerequisite and validate the hostPath mount exists before deployment.

---

## Change 9: Metrics Collection

### Master metrics (via admin_port=9003)

Add Prometheus scraping for master metrics endpoint:

```yaml
# Metrics profile addition for mooncake master
- name: mooncake_master_mem_storage_bytes
  query: mooncake_master_mem_storage_bytes
- name: mooncake_master_ssd_storage_bytes
  query: mooncake_master_ssd_storage_bytes
- name: mooncake_master_eviction_count
  query: mooncake_master_eviction_count
- name: mooncake_master_get_count
  query: mooncake_master_get_count
```

### vLLM tier metrics

When `VLLM_MOONCAKE_STORE_TIER_LOG=1` is set, vLLM logs include:
```
Mooncake load tier summary: ... disk_keys=N memory_keys=M ...
```

Parse these to extract:
- `disk_keys` — blocks loaded from NVMe SSD
- `memory_keys` — blocks loaded from DRAM
- `unknown_keys` — blocks from unknown tier (error)
- `failed_keys` — blocks that failed to load

### Key metrics for validating mooncake works

| Metric | Source | Target | Meaning |
|--------|--------|--------|---------|
| `ext_cache_hit` | aiperf profiling output | >15% | External prefix cache hit rate |
| `theoretical_prefix_cache_hit` | aiperf profiling output | >80% | Maximum possible cache hit from trace structure |
| `disk_keys` | vLLM tier log | >0 | Blocks being read from NVMe SSD |
| `memory_keys` | vLLM tier log | >0 | Blocks being read from DRAM |
| Master Mem Storage | master admin_port | >90% | DRAM pool utilization |
| Master SSD Storage | master admin_port | >0 | SSD tier utilization |
| Master Eviction count | master admin_port | steady | Eviction happening but not thrashing |

---

## Complete Example: Working Experiment

```yaml
apiVersion: benchflow.io/v1alpha1
kind: Experiment
metadata:
  name: mooncake-nvme-offloading
spec:
  model:
    name:
      - google/gemma-4-31B-it
  deployment_profile:
    - mooncake-embedded-nvme      # NEW profile with SSD tier
  benchmark_profile:
    - aiperf-agentx-inference
  metrics_profile: detailed
  model_overrides:
    google/gemma-4-31B-it:
      scale:
        tensor_parallelism: 2
        replicas: 1              # 2 for east-west
      runtime:
        vllm_extra_args:
          - --gpu-memory-utilization=0.85
          - --max-model-len=65536
  target:
    endpoint_scope: internal
  mlflow:
    tags:
      offload_type: mooncake-nvme
  overrides:
    runtime:
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: kubernetes.io/hostname
                    operator: In
                    values:
                      - diadochos-hqxzk-gpu-h100-gjfjh
  namespace: benchflow
  execution:
    timeout: 8h
```

---

## Summary of All Required Changes

### In benchflow code (`rhoai_mooncake.py`)

1. Add `enable_ssd_offload` and `ssd_offload_path` to `RhoaiMooncakeSpec`
2. Add `has_ssd_offload` property
3. Include SSD fields in mooncake config JSON
4. Add master `--eviction_high_watermark_ratio=0.99` and `--admin_port=9003`
5. Add SSD-related env vars (`MOONCAKE_OFFLOAD_FILE_STORAGE_PATH`, `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES`, `MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES`)
6. Add `worker.py` sed patch for `enable_ssd_offload=True` in startup script
7. Add `scheduler.py` sed patch for hybrid attention models (fixes ValueError crash on SSD read failures)
8. Add stale bucket cleanup in startup script (`find $SSD_PATH -maxdepth 1 -name '*.bucket' -delete`)

### New deployment profile

- `profiles/deployment/rhoai/mooncake-embedded-nvme.yaml` — embedded mode with NVMe SSD tier

### Experiment overrides

- `max-model-len=65536` (was 8192)
- `gpu-memory-utilization=0.85` (was 0.55)
- `max-num-seqs=256`

### For east-west (future)

- Multi-replica vLLM with different hostNetwork ports
- Nginx LB with `hash $request_id consistent`
- LB deployment/service in mooncake manifests

---

## Verification Commands

```bash
# 1. Check enable offloading is: 1 in master logs
kubectl logs deploy/mooncake-master | grep "enable offloading"

# 2. Check SSD Storage is non-zero in master metrics
kubectl logs deploy/mooncake-master --tail=1 | grep -oE 'SSD Storage:[^|]+'

# 3. Check disk_keys > 0 in vLLM logs
kubectl logs deploy/mooncake-dist-nvme-vllm | grep disk_keys | tail -5

# 4. Check bucket files on NVMe
kubectl exec deploy/mooncake-dist-nvme-vllm -- \
  find /mnt/nvme-kv-cache -maxdepth 1 -name '*.bucket' | wc -l

# 5. Check aiperf ext_cache_hit
kubectl logs job/benchmark-... | grep ext_cache
```
