# Mooncake KV Cache Sharing: North-South and East-West Reproduction Guide

Complete guide to reproducing mooncake KV cache sharing (both north-south intra-pod and east-west cross-pod) on the diadochos cluster. Covers all deployment configs, vLLM parameters, workload parameters, and the specific changes that made each traffic pattern work.

**Cluster**: diadochos (psap-h100-diadochos)
**Node**: gjfjh (diadochos-hqxzk-gpu-h100-gjfjh)
**Namespace**: benchflow
**Date validated**: 2026-07-17

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Hardware and Infrastructure](#hardware-and-infrastructure)
3. [Deployment Components](#deployment-components)
4. [Mooncake Master Configuration](#mooncake-master-configuration)
5. [vLLM Pod Configuration](#vllm-pod-configuration)
6. [Mooncake Config File](#mooncake-config-file)
7. [Nginx Load Balancer Configuration](#nginx-load-balancer-configuration)
8. [Workload and Benchmark Parameters](#workload-and-benchmark-parameters)
9. [Key Changes: North-South Traffic](#key-changes-north-south-traffic)
10. [Key Changes: East-West Traffic](#key-changes-east-west-traffic)
11. [Submitting a Benchmark](#submitting-a-benchmark)
12. [Verification and Monitoring](#verification-and-monitoring)
13. [Results](#results)
14. [Known Limitations](#known-limitations)

---

## Architecture Overview

```
                              ┌──────────────────────────────┐
                              │       Nginx LB (8080)        │
                              │  hash $request_id consistent │
                              └──────┬──────────────┬────────┘
                                     │              │
                    ┌────────────────▼──┐     ┌────▼────────────────┐
                    │   vLLM Pod 1      │     │   vLLM Pod 2        │
                    │   port 30080      │     │   port 30081        │
                    │   TP=2 (2 GPUs)   │     │   TP=2 (2 GPUs)     │
                    │                   │     │                     │
                    │  GPU KV Cache     │     │  GPU KV Cache       │
                    │       │           │     │       │             │
                    │  save_put/load_get│     │  save_put/load_get  │
                    │       │           │     │       │             │
                    │  Mooncake DRAM    │     │  Mooncake DRAM      │
                    │  Segment (64GB×2) │     │  Segment (64GB×2)   │
                    └───────┬───────────┘     └───────┬─────────────┘
                            │                         │
                    ┌───────▼─────────────────────────▼───────┐
                    │         Mooncake Master (:50051)        │
                    │         Metadata + Eviction Control     │
                    │         256 GB total DRAM pool           │
                    └─────────────────────────────────────────┘
```

**North-south traffic**: Vertical movement within a single pod. A request's KV blocks are computed on GPU, evicted to the pod's mooncake DRAM segment via `save_put`, and later loaded back via `load_get` when another request shares the same prefix. This is an **external prefix cache hit**.

**East-west traffic**: Horizontal movement across pods. Pod 2 loads KV blocks that were saved by Pod 1's DRAM segment (or vice versa) via RDMA. Both pods share the same mooncake master metadata, so blocks saved by either pod are discoverable by both.

---

## Hardware and Infrastructure

### Node: gjfjh

| Resource | Details |
|----------|---------|
| GPUs | 8× NVIDIA H100 80GB |
| NICs | 8× Mellanox ConnectX (mlx5_0 through mlx5_7, all ACTIVE) |
| RDMA device used | mlx5_7 |
| NVMe drives | 8× 7.68 TB NVMe SSDs |
| NVMe layout | nvme0n1 reserved for benchflow; nvme1n1-nvme7n1 in RAID-0 (md0) |
| NVMe RAID-0 mount | `/var/mnt/mooncake-nvme` (~53 TB) |
| Memory | Sufficient for 300Gi per vLLM pod + 128Gi shared memory |

### NVMe RAID-0 Setup

7× NVMe drives (nvme1n1 through nvme7n1) are assembled into an md0 RAID-0 array, formatted XFS, and mounted at `/var/mnt/mooncake-nvme`. This was created via a privileged debug pod:

```bash
mdadm --create /dev/md0 --level=0 --raid-devices=7 \
  /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1 /dev/nvme4n1 \
  /dev/nvme5n1 /dev/nvme6n1 /dev/nvme7n1
mkfs.xfs /dev/md0
mkdir -p /var/mnt/mooncake-nvme
mount -o noatime,nodiratime /dev/md0 /var/mnt/mooncake-nvme
```

**Warning**: This is NOT persistent across reboots. A systemd unit would be needed for persistence. nvme0n1 is reserved for benchflow — do not touch it.

---

## Deployment Components

All components run in the `benchflow` namespace on diadochos.

| Component | Deployment Name | Image | Port | Network |
|-----------|----------------|-------|------|---------|
| Mooncake Master | `mooncake-master` | `vllm/vllm-openai:latest` | 50051 (gRPC), 9003 (metrics) | ClusterIP |
| vLLM Pod 1 | `mooncake-dist-nvme-vllm` | `vllm/vllm-openai:latest` | 30080 | hostNetwork |
| vLLM Pod 2 | `mooncake-dist-nvme-vllm-2` | `vllm/vllm-openai:latest` | 30081 | hostNetwork |
| Nginx LB | `mooncake-nginx-lb` | `nginx:1.27-alpine` | 8080 | ClusterIP |

### Kubernetes Services

| Service | Type | Port | Selector |
|---------|------|------|----------|
| `mooncake-master` | ClusterIP | 50051 (gRPC), 9003 (metrics) | `app: mooncake-master` |
| `mooncake-vllm-lb` | ClusterIP | 8080 | `app: mooncake-nginx-lb` |

---

## Mooncake Master Configuration

The master binary is extracted from the vLLM image at runtime. It manages block metadata, coordinates segment registration, and controls eviction.

### Container Command

```bash
MOONCAKE_DIR=/usr/local/lib/python3.12/dist-packages/mooncake
cp "$MOONCAKE_DIR/mooncake_master" /tmp/mooncake_master
chmod +x /tmp/mooncake_master
export LD_LIBRARY_PATH="$MOONCAKE_DIR:/usr/local/lib/python3.12/dist-packages/mooncake_transfer_engine.libs:${LD_LIBRARY_PATH:-}"
exec /tmp/mooncake_master \
  --rpc_address 0.0.0.0 \
  --rpc_port 50051 \
  --metrics_port 9003 \
  --enable_offload=1 \
  --offload_on_evict=1 \
  --enable_disk_eviction=1 \
  --eviction_high_watermark_ratio=0.99 \
  --eviction_ratio=0.05
```

### Master Args Explained

| Arg | Value | Purpose |
|-----|-------|---------|
| `--rpc_address` | `0.0.0.0` | Listen on all interfaces |
| `--rpc_port` | `50051` | gRPC port for vLLM workers |
| `--metrics_port` | `9003` | Admin/metrics endpoint |
| `--enable_offload` | `1` | Enable KV cache offloading |
| `--offload_on_evict` | `1` | Offload blocks on eviction |
| `--enable_disk_eviction` | `1` | Enable SSD-backed eviction (silently ignored by current binary) |
| `--eviction_high_watermark_ratio` | `0.99` | Start evicting at 99% capacity (253/256 GB). Default 0.95 evicts too aggressively. |
| `--eviction_ratio` | `0.05` | Evict 5% of capacity per cycle |

### Master Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `USER` | `benchflow` | Required for process identity |
| `HOME` | `/tmp` | Writable home directory |

### Master Node Selector

```yaml
nodeSelector:
  kubernetes.io/hostname: diadochos-hqxzk-gpu-h100-gjfjh
```

---

## vLLM Pod Configuration

Both pods are identical except for port number (30080 vs 30081) and resource claim template name. They run on hostNetwork for RDMA access.

### Container Startup Script

The startup script patches the mooncake store worker to enable SSD offload parameters, then launches vLLM:

```bash
#!/bin/bash
# Fix passwd for non-root user
if ! whoami &>/dev/null; then
  echo "vllm:x:$(id -u):0:vllm user:/tmp:/bin/bash" >> /etc/passwd 2>/dev/null || true
fi

# Patch worker.py to pass SSD offload params to mooncake store
WORKER=/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py
sed -i '/store_config\.master_server_address,$/{n;s/^        )/            enable_ssd_offload=True,
            ssd_offload_path=os.getenv("MOONCAKE_OFFLOAD_FILE_STORAGE_PATH", ""),
        )/}' "$WORKER"
echo "=== Patch applied to worker.py ==="
grep -A4 'master_server_address' "$WORKER" | head -6

# Launch vLLM
exec vllm serve /models/models/google-gemma-4-31B-it \
  --port=30080 \                              # 30081 for pod 2
  --host=0.0.0.0 \
  --served-model-name=google/gemma-4-31B-it \
  --tensor-parallel-size=2 \
  --max-model-len=65536 \
  --kv-transfer-config='{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}' \
  --trust-remote-code \
  --no-enable-log-requests \
  --enable-prefix-caching \
  --kv-cache-metrics \
  --kv-cache-metrics-sample=0.01 \
  --gpu-memory-utilization=0.85 \
  --max-num-seqs=256
```

### vLLM Args Explained

| Arg | Value | Purpose |
|-----|-------|---------|
| `--model` | `/models/models/google-gemma-4-31B-it` | Model path on PVC |
| `--served-model-name` | `google/gemma-4-31B-it` | Name returned in `/v1/models` |
| `--tensor-parallel-size` | `2` | 2 GPUs per pod, each GPU runs one TP worker with its own mooncake segment |
| `--max-model-len` | `65536` | **Critical tuning parameter.** Controls max context length AND GPU KV cache pressure. See [tuning section](#key-changes-north-south-traffic). |
| `--kv-transfer-config` | `{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}` | Enables mooncake store connector in both save and load mode |
| `--enable-prefix-caching` | (flag) | Required for prefix cache hit tracking |
| `--kv-cache-metrics` | (flag) | Enables KV cache transfer metrics logging |
| `--kv-cache-metrics-sample` | `0.01` | Sample 1% of requests for detailed KV metrics |
| `--gpu-memory-utilization` | `0.85` | Fraction of GPU memory for KV cache. Higher = more cache space = fewer evictions to mooncake store. |
| `--max-num-seqs` | `256` | Max concurrent sequences |
| `--trust-remote-code` | (flag) | Required for gemma model |
| `--no-enable-log-requests` | (flag) | Reduces log noise |

### vLLM Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `PYTHONHASHSEED` | `0` | **Critical.** All TP ranks must produce identical block hashes. Without this, blocks saved by rank 0 can't be found by rank 1. |
| `MOONCAKE_CONFIG_PATH` | `/mnt/nvme-kv-cache/mooncake-config/mooncake-rdma-nvme-64g.json` | Path to mooncake config file (hostPath mount) |
| `MC_STORE_MEMCPY` | `1` | Use memcpy for store operations |
| `MOONCAKE_OFFLOAD_FILE_STORAGE_PATH` | `/mnt/nvme-kv-cache` | NVMe path for SSD offload tier |
| `MOONCAKE_OFFLOAD_ENABLED` | `true` | Enable offloading |
| `MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES` | `4294967296` | **4 GB NVMe staging buffer.** Default 32 MB causes `BUFFER_OVERFLOW` errors. |
| `VLLM_MOONCAKE_STORE_TIER_LOG` | `1` | Enables per-request tier logging (`memory_keys` vs `disk_keys`) |
| `USER` | `benchflow` | Required for process identity |
| `HOME` | `/tmp` | Writable home directory |
| `HF_HOME` | `/tmp` | Hugging Face cache directory |

### vLLM Pod Spec

```yaml
hostNetwork: true
dnsPolicy: ClusterFirstWithHostNet
securityContext:
  privileged: true
resources:
  claims:
    - name: gpu
  limits:
    cpu: "32"
    memory: 300Gi
  requests:
    cpu: "16"
    memory: 200Gi
```

### Volume Mounts

| Mount | hostPath / Source | Purpose |
|-------|-------------------|---------|
| `/dev/shm` | `emptyDir: { medium: Memory, sizeLimit: 128Gi }` | Shared memory for TP workers |
| `/mnt/nvme-kv-cache` | `hostPath: /var/mnt/mooncake-nvme` | NVMe RAID-0 for mooncake config + SSD tier |
| `/dev/infiniband` | `hostPath: /dev/infiniband` | RDMA device access |
| `/models` | `PVC: models-storage` | Model weights |

### GPU Resource Claim

Each pod uses DRA (Dynamic Resource Allocation) with the nvidia-dra-driver:

```yaml
resourceClaims:
  - name: gpu
    resourceClaimTemplateName: mooncake-dist-nvme-gpu-claim   # -2 for pod 2
spec:
  devices:
    requests:
      - name: gpus
        exactly:
          allocationMode: ExactCount
          count: 2
          deviceClassName: gpu.nvidia.com
```

### Readiness Probe

```yaml
readinessProbe:
  httpGet:
    path: /health
    port: 30080  # 30081 for pod 2
  initialDelaySeconds: 300
  periodSeconds: 10
  failureThreshold: 3
  timeoutSeconds: 5
```

---

## Mooncake Config File

Path on node: `/var/mnt/mooncake-nvme/mooncake-config/mooncake-rdma-nvme-64g.json`
(mounted into pods at `/mnt/nvme-kv-cache/mooncake-config/mooncake-rdma-nvme-64g.json`)

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

### Config Parameters Explained

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `mode` | `embedded` | Each TP worker runs its own mooncake store (no sidecar process) |
| `metadata_server` | `P2PHANDSHAKE` | Peer-to-peer handshake for segment discovery |
| `master_server_address` | `mooncake-master.benchflow.svc.cluster.local:50051` | Master gRPC endpoint via K8s service DNS |
| `global_segment_size` | `64GB` | RDMA-pinned memory segment per TP worker |
| `local_buffer_size` | `64GB` | DRAM contributed per TP worker to master pool. 4 workers × 64GB = **256 GB total** |
| `protocol` | `rdma` | Use RDMA for KV cache transfers |
| `device_name` | `mlx5_7` | Specific Mellanox NIC for RDMA |
| `enable_offload` | `true` | Enable CPU DRAM offloading |
| `enable_ssd_offload` | `true` | Enable NVMe SSD tier (configured but inactive — see [Known Limitations](#known-limitations)) |
| `ssd_offload_path` | `/mnt/nvme-kv-cache` | NVMe mount for SSD tier |

### Capacity Math

- `local_buffer_size=64GB` × 2 TP workers per pod × 2 pods = **256 GB total master DRAM capacity**
- `global_segment_size=64GB` = each worker's RDMA-pinned segment for block transfer
- At `eviction_high_watermark_ratio=0.99`, eviction starts at **253 GB / 256 GB**

---

## Nginx Load Balancer Configuration

ConfigMap: `mooncake-nginx-lb-config`

```nginx
worker_processes auto;
events { worker_connections 4096; }
http {
  log_format upstream_log '$remote_addr [$time_local] "$request" '
                          '$status $body_bytes_sent '
                          'upstream=$upstream_addr rt=$upstream_response_time';

  upstream vllm_pool {
    hash $request_id consistent;
    server 10.243.65.15:30080;
    server 10.243.65.15:30081;
  }
  server {
    listen 8080;
    access_log /dev/stdout upstream_log;

    location / {
      proxy_pass http://vllm_pool;
      proxy_http_version 1.1;
      proxy_set_header Connection "";
      proxy_set_header Host $host;
      proxy_read_timeout 600s;
      proxy_send_timeout 600s;
    }
    location /health {
      proxy_pass http://vllm_pool;
      proxy_next_upstream error timeout http_502 http_503;
    }
  }
}
```

### Why `hash $request_id consistent`

The default nginx round-robin with HTTP/1.1 keepalive causes **connection stickiness**: streaming SSE responses keep the TCP connection open, so subsequent requests reuse the same connection and land on the same backend. This caused 100% of traffic to go to pod 1, making east-west sharing impossible.

`hash $request_id consistent` hashes each request's unique `$request_id` (auto-generated by nginx) for per-request distribution. This achieves near-perfect 50/50 split (measured: 220:221 across 441 requests).

**Note**: The upstream servers use the node IP (`10.243.65.15`) with hostNetwork ports, not pod IPs. Both vLLM pods run on the same node (gjfjh) via hostNetwork.

### Nginx Deployment

- Image: `nginx:1.27-alpine`
- Custom config mounted via `-c /etc/nginx-custom/nginx.conf` command override
- NodeSelector: gjfjh
- Service: `mooncake-vllm-lb` (ClusterIP, port 8080)

---

## Workload and Benchmark Parameters

### Experiment File

File: `experiments/rhoai/mooncake-offloading.yaml`

```yaml
apiVersion: benchflow.io/v1alpha1
kind: Experiment
metadata:
  name: mooncake-offloading
spec:
  model:
    name:
      - google/gemma-4-31B-it
  deployment_profile:
    - mooncake-embedded-cpu
  benchmark_profile:
    - aiperf-agentx-inference
  metrics_profile: detailed
  model_overrides:
    google/gemma-4-31B-it:
      scale:
        tensor_parallelism: 2
        replicas: 1
      runtime:
        vllm_extra_args:
          - --gpu-memory-utilization=0.55
  target:
    endpoint_scope: internal
  mlflow:
    tags:
      offload_type: mooncake
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

**Note**: The experiment file references `mooncake-embedded-cpu` deployment profile and `--gpu-memory-utilization=0.55`, but the actual manually-deployed pods use `0.85`. The benchmark runs against the pre-deployed pods via `--target-url`, so the experiment file's deployment profile is not used for deployment — only the benchmark profile matters.

### Benchmark Profile: aiperf-agentx-inference

File: `profiles/benchmark/aiperf-agentx-inference.yaml`

```yaml
apiVersion: benchflow.io/v1alpha1
kind: BenchmarkProfile
metadata:
  name: aiperf-agentx-inference
spec:
  tool: aiperf
  env:
    AIPERF_HTTP_SSL_VERIFY: "false"
  requirements:
    min_max_model_len: 131072
  aiperf:
    scenario: inferencex-agentx-mvp
    public_dataset: weka_hf
    hf_weka_repo: semianalysisai/cc-traces-weka-with-subagents-060826
    endpoint_type: chat
    endpoint_path: /v1/chat/completions
    streaming: true
    use_server_token_count: true
    tokenizer_trust_remote_code: true
    max_context_length: 131072
    concurrency: 32
    benchmark_duration: 1800
    max_seconds: 7200
    random_seed: 42
```

### AIPerf / Weka Agentic Workload Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `scenario` | `inferencex-agentx-mvp` | Multi-turn agentic conversation replay |
| `public_dataset` | `weka_hf` | Use Hugging Face-hosted Weka traces |
| `hf_weka_repo` | `semianalysisai/cc-traces-weka-with-subagents-060826` | Exact dataset (680 conversations) |
| `endpoint_type` | `chat` | OpenAI-compatible chat completions API |
| `endpoint_path` | `/v1/chat/completions` | API endpoint |
| `streaming` | `true` | SSE streaming responses |
| `use_server_token_count` | `true` | Use server-reported token counts |
| `tokenizer_trust_remote_code` | `true` | Required for gemma tokenizer |
| `max_context_length` | `131072` | Maximum context length for trace replay |
| `concurrency` | `32` | 32 concurrent requests |
| `benchmark_duration` | `1800` | 30-minute profiling window |
| `max_seconds` | `7200` | 2-hour hard timeout |
| `random_seed` | `42` | Reproducible request ordering |

### Weka Trace Characteristics

The traces are multi-turn agent conversations from the Weka codebase. Each turn includes the full prior conversation as context:

- Turn 1: `[system + user1]` → ~2,400 tokens
- Turn 2: `[system + user1 + assistant1 + user2]` → ~5,000 tokens
- Turn N: accumulates until context overflow at max-model-len

With max-model-len=65536, trajectories reach an average of **21 turns** (range 5-49) before overflowing. The theoretical prefix cache hit rate at this length is **93%** — meaning 93% of input tokens in later turns overlap with earlier turns of the same conversation.

### Model Parameters

| Parameter | Value |
|-----------|-------|
| Model | google/gemma-4-31B-it |
| Parameter count | 31 billion |
| Architecture | Gemma 4 (decoder-only transformer) |
| Tensor parallelism | 2 (2 GPUs per pod) |
| Total GPUs | 4 (2 pods × 2 GPUs) |
| Precision | Default (bfloat16) |
| Tokenizer | SentencePiece (less token-efficient than Qwen — same text produces ~4x more tokens) |

### Metrics Profile

File: `profiles/metrics/detailed.yaml` — collects Prometheus metrics for GPU utilization, KV cache usage, TTFT/ITL/TPOT histograms, NVMe I/O, prefix cache hit rates, and Ceph storage metrics.

---

## Key Changes: North-South Traffic

These changes took external prefix cache hit from 0% to 22-24%.

### Change 1: max-model-len = 65536 (the critical change)

`max-model-len` controls two things simultaneously:
1. **How many conversation turns fit** — determines prefix sharing opportunity
2. **How much GPU KV cache pressure exists** — determines whether blocks get pushed to the mooncake store

| max-model-len | ext_cache_hit | What happened |
|---------------|---------------|---------------|
| 8192 | 0% | Traces overflow at turn 1-7. No prefix sharing (theoretical=0%). |
| 32768 | 0% | Traces fit ~25 turns. GPU KV cache usage only 2.8% — blocks stay in GPU, never reach mooncake. |
| **65536** | **22-24%** | **Sweet spot.** 93% theoretical sharing. GPU usage 52-70% — blocks ARE evicted to mooncake and found by later requests. |
| 131072 | 0.8% | Store fills too fast. 131K tokens/request ≈ 2 GB KV data. Master evicts 96%+ before reuse. |

### Change 2: eviction_high_watermark_ratio = 0.99

Default 0.95 starts evicting at 243/256 GB. At 0.99, eviction starts at 253/256 GB — keeping blocks around longer for prefix reuse.

### Change 3: Always restart master with vLLM pods

After any vLLM config change, always restart the master too. Stale segment registrations cause `SEGMENT_NOT_FOUND` errors and prevent block discovery.

```bash
KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl rollout restart -n benchflow \
  deployment/mooncake-master \
  deployment/mooncake-dist-nvme-vllm \
  deployment/mooncake-dist-nvme-vllm-2
```

---

## Key Changes: East-West Traffic

These changes went from 0% east-west sharing (all traffic to pod 1) to functional cross-pod KV cache sharing.

### Change: Nginx LB hash distribution

**Problem**: Default nginx round-robin with HTTP/1.1 keepalive causes connection stickiness. Streaming SSE responses keep the TCP connection alive, so all requests from one benchmark worker stick to the same backend. Result: 100% of traffic goes to pod 1, pod 2 never receives requests, zero east-west sharing.

**Attempted fix**: `least_conn` — failed because all 32 concurrent requests arrive in a burst and nginx sees all backends at 0 active connections, picks the first server every time.

**Working fix**: `hash $request_id consistent` — hashes nginx's auto-generated unique request ID per request. Each request is independently routed regardless of TCP connection state. Achieves near-perfect 50/50 distribution.

```nginx
upstream vllm_pool {
  hash $request_id consistent;    # <-- THIS IS THE KEY CHANGE
  server 10.243.65.15:30080;
  server 10.243.65.15:30081;
}
```

### Why east-west sharing works without code changes

PoolKeys (the lookup key for KV cache blocks in the mooncake store) are composed of:

```
{cache_prefix}@{model_name}@tp_rank:{tp_rank}@pcp{pcp_rank}@dcp{dcp_rank}@pp_rank:{pp_rank}@group:{group_id}@{chunk_hash}
```

Critically, **PoolKeys do not include engine_id or pod identity**. Both pods use:
- Same `model_name` (last path component of model path)
- Same `tp_rank` (0 or 1)
- Same `group_id` (defaults to 0)
- Same `cache_prefix` (from extra_config, defaults to empty)
- Same `chunk_hash` (deterministic because `PYTHONHASHSEED=0`)

So blocks saved by Pod 1 produce the exact same PoolKey that Pod 2 uses for lookups, enabling cross-pod block discovery through the shared master.

### Applying the nginx change

```bash
KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl edit configmap mooncake-nginx-lb-config -n benchflow
# Change the upstream block to use hash $request_id consistent
# Then restart nginx:
KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl rollout restart -n benchflow deployment/mooncake-nginx-lb
```

---

## Submitting a Benchmark

### Prerequisites

```bash
# 1. Extract diadochos kubeconfig
export DIADOCHOS_KUBECONFIG=/tmp/diadochos-kubeconfig.tmp
KUBECONFIG=/Users/rdoddaia/work/aperdomo/Aperdomo-kubeconfig \
  kubectl get secret psap-h100-diadochos -n benchflow \
  -o jsonpath='{.data.kubeconfig}' | base64 -d > $DIADOCHOS_KUBECONFIG

# 2. Verify all pods are running
KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl get pods -n benchflow | grep mooncake

# 3. Verify endpoint responds
KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl exec -n benchflow deploy/mooncake-nginx-lb -- \
  curl -s http://localhost:8080/v1/models | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(f'model={d[\"data\"][0][\"id\"]}, max_model_len={d[\"data\"][0][\"max_model_len\"]}')"
# Expected: model=google/gemma-4-31B-it, max_model_len=65536
```

### Submit Command

```bash
KUBECONFIG=/Users/rdoddaia/work/aperdomo/Aperdomo-kubeconfig bflow experiment run \
  experiments/rhoai/mooncake-offloading.yaml \
  --target-url http://mooncake-vllm-lb.benchflow.svc:8080 \
  --benchflow-image ghcr.io/albertoperdomo2/benchflow:manual-a53adf3 \
  --cluster-name psap-h100-diadochos
```

The `--target-url` flag skips deployment and sends traffic directly to the pre-deployed pods via the nginx LB service.

### MLflow Credentials

| Field | Value |
|-------|-------|
| Username | benchflow |
| Password | amyl1Cc-cSBf_VKE0ZpBUg4Y |
| Workspace | benchflow |
| Tracking URI | https://mlflow.apps.psap-automation.ibm.rhperfscale.org |

---

## Verification and Monitoring

### 1. Check aiperf metrics (primary indicator)

```bash
KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl get jobs -n benchflow | grep benchmark-mooncake
KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl logs -f -n benchflow job/<job-name> \
  | grep -E "profiling|ext_cache"
```

What to look for:
```
trace theoretical_prefix_cache_hit=93.4%     ← should be >80%
srv  prefix_cache_hit=8.1% ... ext_cache_hit=24.3%  ← ext_cache_hit should be >15%
```

**Diagnostic table:**

| Symptom | Cause | Fix |
|---------|-------|-----|
| `theoretical_prefix_cache_hit=0%` | max-model-len too small, traces overflow | Increase max-model-len |
| `ext_cache_hit=0%` but `prefix_cache_hit` high | GPU holds everything, no pressure | Increase max-model-len to add GPU pressure |
| `ext_cache_hit=0%` and `prefix_cache_hit=0%` | Multiple issues possible | Check master logs, worker connection |
| `ext_cache_hit` low despite high theoretical | Master evicting too fast | Raise `eviction_high_watermark_ratio` |

### 2. Check vLLM KV transfer metrics

```bash
KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl logs -f -n benchflow deployment/mooncake-dist-nvme-vllm \
  | grep -E "KV Transfer|External prefix"
```

Expected:
```
External prefix cache hit rate: 22.3%
KV Transfer metrics: lookup_exists_total_bytes=XXX   ← non-zero
```

### 3. Check traffic distribution (east-west)

```bash
KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl logs -n benchflow deploy/mooncake-nginx-lb --tail=500 \
  | grep -oP 'upstream=\K[^ ]+' | sort | uniq -c
```

Expected: roughly equal counts for both backends.

### 4. Check master state

```bash
KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl logs -f -n benchflow deployment/mooncake-master \
  | grep -oE '(Mem Storage[^|]+|Eviction[^|]+|Get:\([^)]+\))'
```

Expected:
```
Mem Storage: 250 GB / 256 GB (97.9%)
Get:(Req=668/0/668, Item=455712/455712)
Eviction: Success/Attempts=485/485, keys=5531631
```

### 5. Check tier logs (DRAM vs SSD)

```bash
KUBECONFIG=$DIADOCHOS_KUBECONFIG kubectl logs -n benchflow deploy/mooncake-dist-nvme-vllm \
  | grep -i "tier\|disk_keys\|memory_keys"
```

Currently shows `memory_keys=N, disk_keys=0` for every load — SSD tier is inactive.

---

## Results

### North-South Only (Run 5: `mooncake-offloading-2bb41e`)

All traffic to pod 1 (nginx keepalive stickiness). Pod 2 idle.

| Metric | Value |
|--------|-------|
| PipelineRun | `mooncake-offloading-2bb41e` |
| MLflow run | `52ee6440cf15480f998604d4d9a02b61` |
| Duration | 1795.72s |
| Requests (profiling) | 348 (0 errors) |
| Theoretical prefix cache | 93.4% |
| **External prefix cache** | **22-24%** |
| Local GPU prefix cache | 8-21% |
| Combined actual | ~30-44% |
| TTFT p50 | 4,566 ms |
| ITL p50 | 30 ms |
| Output throughput | 112.3 tokens/s |
| Input throughput | 8,997 tokens/s |
| Master capacity at end | 97.9% (251/256 GB) |
| Master eviction cycles | 485 |
| Context overflows | 27 trajectories, avg turn 21 |

### East-West (Run 6: `mooncake-offloading-425773`)

Traffic split 50/50 across both pods. Both pods save and load blocks.

| Metric | Value |
|--------|-------|
| PipelineRun | `mooncake-offloading-425773` |
| MLflow run | `d2f212db62b74716860a2d72e5b8e526` |
| Duration | ~30:19 profiling |
| Requests (profiling) | 399 (0 errors, 41 context-overflow terminations) |
| Traffic distribution | **220:221 (50/50)** |
| Theoretical prefix cache | 92.9% |
| **External prefix cache** (combined) | **21.9-26.2%** |
| Pod 1 ext_cache_hit | 23.4% |
| Pod 2 ext_cache_hit | 18.7% |
| TTFT p50 | 3,797 ms |
| ITL p50 | 23 ms |
| Output throughput | 123.4 tokens/s |
| Master Get requests | 736 (461K items) |
| Master eviction cycles | 409 |
| Disk tier | **Completely inactive — all loads from DRAM** |

### Comparison

| Metric | North-South Only | East-West |
|--------|-----------------|-----------|
| ext_cache_hit | 22-24% | 21.9-26.2% |
| TTFT p50 | 4,566 ms | 3,797 ms |
| ITL p50 | 30 ms | 23 ms |
| Output throughput | 112.3 tok/s | 123.4 tok/s |
| Pods active | 1 | 2 |
| Disk reads | 0 | 0 |

East-west provides better latency and throughput because both pods share the inference load. External cache hit rate is similar because the mooncake store capacity and eviction dynamics are the same.

---

## Known Limitations

1. **NVMe SSD tier is completely inactive**. Despite `enable_ssd_offload: true` in mooncake config and `--enable_disk_eviction=1` on master, `SSD Storage` stays at `0 B / 0 B`. The current mooncake binary version silently ignores disk offloading. All ext_cache_hits are from DRAM segments via RDMA only — zero disk reads. If fixed, this could significantly improve ext_cache_hit by preserving metadata that currently gets evicted.

2. **Master eviction destroys ~96% of saved keys**. 256 GB DRAM fills within minutes at 32 concurrency. At max-model-len=65536, blocks are saved but evicted before the next request with the same prefix arrives. This is the primary reason for the gap between 93% theoretical and ~22% actual ext_cache_hit.

3. **RAID-0 not persistent across reboots**. The md0 array on gjfjh is not backed by a systemd unit. Rebooting the node will lose the array and the mount.

4. **Pod 2 gpu-memory-utilization discrepancy noted**. Both pods currently use 0.85, matching each other. The experiment yaml specifies 0.55 but that is only used for benchflow-managed deployments, not these manually deployed pods.

5. **Non-mooncake baseline comparison caveat**. The `OffloadingConnector` (non-mooncake) uses a completely different connector measuring intra-engine tier movement. Its "external prefix cache hit" metric is not comparable to mooncake's. It also uses Qwen3.6-35B-A3B (more token-efficient tokenizer), making direct rate comparison misleading.

---

## Appendix: Tuning Iteration History

| Run | max-model-len | eviction_ratio | ext_cache_hit | Key learning |
|-----|---------------|----------------|---------------|--------------|
| 1 | 8192 | 0.95 | 0% | Too short — traces overflow immediately |
| 2 | 131072 | 0.95 | 0.8% | Too long — store overwhelmed by eviction |
| 3 | 32768 | 0.99 | 0% (21.5% local) | GPU holds everything — no store pressure |
| 4 | 65536 | 0.99 | 22.3% | Sweet spot found |
| 5 | 65536 | 0.99 | 22-24% | Confirmed (north-south only) |
| 6 | 65536 | 0.99 | 21.9-26.2% | East-west confirmed with nginx hash fix |
