# Kimi-K3 Deployment Characterization

**Namespace:** `kimi-k3`
**Date observed:** 2026-08-20
**Owner:** mimehta@redhat.com (`llm-d.ai/owner` annotation)

## Executive Summary

Kimi-K3 is deployed as a **4-node Expert-Parallel (EP) cluster** serving `moonshotai/Kimi-K3` with speculative decoding. There is **no Prefill/Decode (P/D) disaggregation** — all nodes perform both prefill and decode phases. The parallelism strategy is **TP=8 within each node, DP=4 across nodes with EP enabled**, meaning the 896 MoE experts are distributed and load-balanced across data-parallel ranks via all-to-all communication over RDMA/InfiniBand.

## Model

| Property | Value |
|---|---|
| Model | `moonshotai/Kimi-K3` |
| Architecture | `KimiK3ForConditionalGeneration` / `KimiLinearForCausalLM` |
| Type | Mixture of Experts (MoE) with hybrid linear + full attention |
| Hidden layers | 93 |
| Hidden size | 7168 |
| Attention heads | 96 |
| Experts (total) | 896 |
| Experts per token | 16 |
| Shared experts | 2 |
| MoE intermediate size | 3072 (routed: 3584) |
| MLA KV LoRA rank | 512 |
| Max context length | 1,048,576 tokens (1M) |
| Vocab size | 163,840 |
| Quantization | MXFP4 (4-bit float, group_size=32) on routed expert weights only |
| Quantization exclusions | Attention, shared experts, dense MLP, lm_head, vision tower |
| On-disk size | ~1.5 TB (96 safetensor shards) |
| Vision tower | Present in weights but disabled (`--language-model-only`) |

### Attention Architecture

Kimi-K3 uses a hybrid attention scheme:
- **Full attention** on 24 layers: 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 93
- **KDA (linear) attention** on the remaining 69 layers
- Multi-head Latent Attention (MLA) with `q_lora_rank=1536`, `qk_nope_head_dim=128`, `qk_rope_head_dim=64`, `v_head_dim=128`

## Infrastructure

### Nodes

| Pod | Node | Node Rank | Role | GPUs | IP |
|---|---|---|---|---|---|
| `vllm-recipe-0` | `g13c364` | 0 | **Master (API)** | 8x H200 | 10.202.208.69 |
| `vllm-recipe-1` | `gc37d78` | 1 | Headless worker | 8x H200 | 10.202.206.47 |
| `vllm-recipe-2` | `gf2a612` | 2 | Headless worker | 8x H200 | 10.202.208.117 |
| `vllm-recipe-3` | `g1191e4` | 3 | Headless worker | 8x H200 | 10.202.210.23 |

### Per-Node Hardware

| Component | Spec |
|---|---|
| GPUs | 8x NVIDIA H200 (143 GiB HBM3e each, 1150 GiB total per node) |
| GPU interconnect | NVLink NV18 (full mesh, all-pairs NV18) |
| CPU | 128 cores |
| System memory | ~2 TiB |
| RDMA/IB devices | 64 per node |
| Network | hostNetwork + hostIPC (for RDMA passthrough) |

**Total cluster:** 32x H200 GPUs, ~4600 GiB HBM3e, 512 CPU cores, ~8 TiB system memory.

### Storage

- **Model storage:** hostPath `/mnt/local/kimi-k3/models` (local SSD/NVMe per node)
- **No PVCs** — each node has a local copy of the model weights

### Networking

- `hostNetwork: true` — pods use the host network stack directly
- `hostIPC: true` — required for RDMA/shared memory communication
- RDMA/IB resource (`rdma/ib: 1`) requested per pod
- Headless Service `vllm-recipe` (ClusterIP: None) on port 8000

## Parallelism Strategy

### Tensor Parallelism (TP=8)

Each node shards the model across all 8 local GPUs using tensor parallelism. Intra-node communication uses NVLink NV18 (full mesh). Custom all-reduce is disabled (`--disable-custom-all-reduce`).

### Data Parallelism (DP=4) with Expert Parallelism (EP)

The 4 nodes form a DP=4 group with `--enable-expert-parallel`. In this configuration:

- Each DP rank (node) holds a **full copy of attention layers and shared experts** (these are replicated)
- The **896 routed experts are partitioned across the 4 DP ranks** (~224 experts per node)
- At inference time, tokens are routed to the correct expert via **all-to-all communication over RDMA/IB** across nodes
- This reduces per-node memory usage for expert weights while keeping attention fully replicated

The topology is: `[TP=8] x [DP=4 with EP]` = 32 GPUs total.

```
Node 0 (g13c364)          Node 1 (gc37d78)          Node 2 (gf2a612)          Node 3 (g1191e4)
rank=0, MASTER            rank=1, headless          rank=2, headless          rank=3, headless
+-------------------+     +-------------------+     +-------------------+     +-------------------+
| 8x H200 (TP=8)   |     | 8x H200 (TP=8)   |     | 8x H200 (TP=8)   |     | 8x H200 (TP=8)   |
| Attn: FULL copy   |     | Attn: FULL copy   |     | Attn: FULL copy   |     | Attn: FULL copy   |
| Experts: ~224/896 |     | Experts: ~224/896 |     | Experts: ~224/896 |     | Experts: ~224/896 |
+-------------------+     +-------------------+     +-------------------+     +-------------------+
         |                          |                          |                          |
         +------------- RDMA/IB all-to-all (expert routing) --+--------------------------+
```

### Prefill/Decode Disaggregation: NOT PRESENT

There is no P/D disaggregation. All 4 nodes participate in both prefill and decode for every request. There are no `--prefill-only` or `--decode-only` flags in the vLLM command line. The deployment is a unified serving topology.

## vLLM Configuration

| Parameter | Value | Notes |
|---|---|---|
| vLLM version | `0.1.dev19262+gb6bbf29dd` (dev build, 2026-07-27) | Custom build with Kimi-K3 support |
| Image | `vllm/vllm-openai:kimi-k3` | |
| `--tensor-parallel-size` | 8 | Full intra-node TP |
| `--data-parallel-size` | 4 | One DP rank per node |
| `--enable-expert-parallel` | yes | EP across DP ranks |
| `--nnodes` | 4 | Multi-node distributed |
| `--gpu-memory-utilization` | 0.95 | 95% HBM allocation |
| `--max-num-seqs` | 8 | Max concurrent sequences |
| `--max-num-batched-tokens` | 4096 | Prefill budget per step |
| `--attention-backend` | FLASHMLA | Specialized MLA kernel |
| `--moe-backend` | marlin | Quantized MoE kernel |
| `--enable-prefix-caching` | yes | Prompt/prefix KV reuse |
| `--enforce-eager` | yes | No CUDA graphs |
| `--load-format` | fastsafetensors | Optimized weight loading |
| `--language-model-only` | yes | Vision tower disabled |
| `--no-async-scheduling` | yes | Synchronous scheduling |
| `--disable-custom-all-reduce` | yes | Use NCCL all-reduce |
| `--served-model-name` | `moonshotai/Kimi-K3` | |

### Speculative Decoding

| Parameter | Value |
|---|---|
| Method | DSpark |
| Draft model | `Inferact/Kimi-K3-DSpark` |
| Speculative tokens | 2 |
| Draft sample method | probabilistic |
| Rejection sample method | synthetic |
| Synthetic acceptance length | 2.51 |

DSpark (Draft Speculation with Parallel Kernel) uses a lightweight draft model to predict 2 tokens ahead, with synthetic rejection sampling for verification. The draft model is cached locally at `/models/hf/hub/models--Inferact--Kimi-K3-DSpark`.

### Tool Calling

- `--enable-auto-tool-choice` with `--tool-call-parser kimi_k3`
- `--reasoning-parser kimi_k3`

## Resource Allocation

### Per Pod

| Resource | Request | Limit |
|---|---|---|
| CPU | 16 cores | - |
| Memory | 64 GiB | - |
| Ephemeral storage | 50 GiB | - |
| GPUs | 8 | 8 |
| RDMA/IB | 1 | 1 |

### GPU Memory Usage (observed)

All 8 GPUs on node 0 show ~139.5 GiB used out of 143.7 GiB (~97% utilization), consistent with `--gpu-memory-utilization 0.95` plus runtime overhead.

## Companion Pods

| Pod | Node | Purpose |
|---|---|---|
| `vllm-bench` | `gf2a612` | Benchmarking client (co-located with vllm-recipe-2) |
| `aiperf-agentx` | `g1191e4` | AIPerf AgentX agent (co-located with vllm-recipe-3) |

## Summary Table

| Dimension | Value |
|---|---|
| P/D disaggregation | **No** — unified serving |
| Expert Parallelism (EP) | **Yes** — 896 experts distributed across 4 DP ranks |
| Tensor Parallelism (TP) | 8 (intra-node) |
| Data Parallelism (DP) | 4 (inter-node, carries EP) |
| Pipeline Parallelism (PP) | None |
| Speculative Decoding | Yes (DSpark, 2 tokens) |
| Total GPUs | 32x H200 |
| Total HBM | ~4600 GiB |
| Inter-node fabric | RDMA/InfiniBand |
| Intra-node fabric | NVLink NV18 (full mesh) |
