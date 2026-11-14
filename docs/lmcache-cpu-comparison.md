# LMCache CPU-only AgentX comparison on Diadochos

These experiments collect the LMCache counterparts of the native CPU-offload
observations recorded in Arche. They run **only LMCache** and reuse the existing
native and HBM reference runs. Nothing here launches another native campaign.
No performance runs have been launched for these definitions.

| Entrypoint | Model | TP | CPU cache | GPU utilization | AgentX concurrency | Shared memory |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `experiments/rhoai/offloading/lmcache-cpu-qwen36.yaml` | Qwen/Qwen3.6-35B-A3B | 2 | 64 GiB | 0.55 | 32 | 200 GiB |
| `experiments/rhoai/offloading/lmcache-cpu-256g.yaml`, child 1 | google/gemma-4-31B-it | 2 | 256 GiB | 0.92 | 8 | 300 GiB |
| `experiments/rhoai/offloading/lmcache-cpu-256g.yaml`, child 2 | nvidia/Llama-3_1-Nemotron-Ultra-253B-v1-FP8 | 8 | 256 GiB | 0.80 | 32 | 300 GiB |

The 256 GiB entrypoint is a sequential two-model matrix. Qwen is separate because
it needs a different CPU cache budget, shared memory size, and recurrent-cache
settings. This keeps the existing Cartesian matrix semantics and avoids running
each model with the other models' deployment profiles. Run Qwen to completion
before submitting the 256 GiB matrix; separate submissions are not serialized
by `execution.mode`.

Every configuration uses one replica, PP1, a 131,072-token context limit,
prefix caching, and a fresh LMCache CPU pool. The shared AgentX profile uses
the standard BenchFlow image's **AIPerf 0.12.0**. Its dated public-dataset alias,
`semianalysis_cc_traces_weka_with_subagents_060826`, selects the historical
`semianalysisai/cc-traces-weka-with-subagents-060826` corpus. Seed 42,
1,800 seconds, trajectory ratios 0.25–0.75, and streaming remain explicit.
It does not reuse today's rolling `aiperf-agentx-inference` profile, whose
corpus and seed have changed.

AIPerf 0.12 requires the scenario's **10-second system idle cap** and forbids
the old per-trace idle cap. The archived native runs used AIPerf 0.8.0 with a
60-second per-trace cap, so pacing and prompt reconstruction can differ even
with the same corpus and seed. The two caps have different semantics.
See the [0.12 scenario](https://github.com/ai-dynamo/aiperf/blob/v0.12.0/src/aiperf/common/scenario/inferencex_agentx_mvp.py)
and its [validation rules](https://github.com/ai-dynamo/aiperf/blob/v0.12.0/src/aiperf/common/scenario/validator.py).
No legacy runtime, alternate control-image build, or unsafe scenario override
is required.

## Run

Use the standard BenchFlow control image `v0.10.0` or a compatible newer image
containing commit `f4a2cd5`; `v0.9.0` rejects these models. Reuse the supplied
compatible vLLM+LMCache serving image;
the serving image is not built by this experiment or in CI. The previously
qualified small-model image uses vLLM 0.32.0 and LMCache 0.5.7. The new model
layouts and TP2/TP8 still need live qualification; a successful small-model smoke
does not qualify them.

Run from the BenchFlow management-cluster context. Set the image variables to
your published image references, preferably digests:

```bash
export BENCHFLOW_IMAGE=ghcr.io/albertoperdomo2/benchflow:v0.10.0
export VLLM_LMCACHE_IMAGE=quay.io/rh-ee-aperdomo/vllm:vllm-0.32.0-lmcache-0.5.7

bflow experiment validate experiments/rhoai/offloading/lmcache-cpu-qwen36.yaml \
  --runtime-image "$VLLM_LMCACHE_IMAGE"
bflow experiment validate experiments/rhoai/offloading/lmcache-cpu-256g.yaml \
  --runtime-image "$VLLM_LMCACHE_IMAGE"

bflow experiment run experiments/rhoai/offloading/lmcache-cpu-qwen36.yaml \
  --runtime-image "$VLLM_LMCACHE_IMAGE" \
  --benchflow-image "$BENCHFLOW_IMAGE" \
  --target-kubeconfig /Users/aperdomo/workspace/redhat/clusters/psap-diadochos-h100/auth/kubeconfig \
  --follow

# After Qwen and its cleanup complete:
bflow experiment run experiments/rhoai/offloading/lmcache-cpu-256g.yaml \
  --runtime-image "$VLLM_LMCACHE_IMAGE" \
  --benchflow-image "$BENCHFLOW_IMAGE" \
  --target-kubeconfig /Users/aperdomo/workspace/redhat/clusters/psap-diadochos-h100/auth/kubeconfig \
  --follow
```

Submit from an updated checkout so the resolved RunPlan contains the 0.12
dataset alias and pacing flags. A retry of an old execution retains its old
RunPlan, including the 0.8 requirement; submit the YAML again instead.
The standard `v0.10.0` image can execute these updated profiles without rebuilding
the control image or updating the capacity controller.

The deployment profiles select the existing eight-H100 node
`diadochos-hqxzk-gpu-h100-mt46x`. This also limits the DaemonSet to one cache
server, so idle nodes do not allocate additional CPU pools. The eight-GPU model
needs that node's GPUs available through normal scheduling; setup does not evict
other workloads. Both profiles target the currently installed RHOAI 3.5.0
`stable-3.5` platform (read-only target verification on 2026-11-12).
No platform teardown is requested; final cleanup removes the model and the
execution-owned cache while retaining base RHOAI and model storage.

## Qualification and evidence

Qwen3.6 requires the hybrid KV cache manager and `--mamba-cache-mode=align`.
BenchFlow starts its cache server with `--separate-object-groups`, as required
by [LMCache's Qwen hybrid recipe](https://docs.lmcache.ai/recipes/qwen3_5.html).
The server must provide that flag. Gemma uses the MP connector's hybrid
attention handling ([upstream recipe](https://docs.lmcache.ai/recipes/gemma4.html)).
The configured 256-token LMCache chunk is a minimum; record the effective
negotiated chunk size from startup logs, especially for recurrent models.
No sleep mode, weight offload, secondary storage, speculative decoding, or
experimental native block-member fallback is enabled.

Every run requires new CPU cache writes and successful loads with stable cache
servers. Failure marks the benchmark failed; artifacts and final cleanup still
run. `results/lmcache-readback.json`, raw `metrics/lmcache/*.prom`, model/cache
logs, manifests, and MLflow benchmark artifacts are preserved. The `lmcache-cpu`
metrics profile collects native 15-second samples for prompt-token sources,
generation, running/waiting requests, GPU KV occupancy, model/cache CPU and RSS,
CPU-pool occupancy, lookup hits, chunk writes/reads, completed transfers, and
dropped telemetry events. Unsupported transfer-byte counters are not fabricated.
LMCache MP queries retain the final-scrape collection window separately from
the actual benchmark end time.

Accept performance numbers only after inspecting readback, errors, restarts,
memory pressure, and completed sessions/branches. Store/load evidence alone
does not establish output correctness or a performance improvement. AgentX
branches as it makes progress, so retain request mix and session completion
alongside TTFT, E2E, ITL, and request/output throughput.

## Historical references and comparison limits

These definitions were checked against the archived `metadata/run-plan.json`
for the native runs below, and the deployed Gemma/Nemotron pod manifests. The
Arche reports are the dated July 29 Qwen U0.55 matrix, August 3 Gemma comparison,
and July 31 FP8 Nemotron Revision 2.

| Model | Historical HBM run | Historical native CPU run |
| --- | --- | --- |
| Qwen3.6-35B-A3B | [6b1884118aec42fbbaf9ed9accc1c05e](https://mlflow.apps.psap-automation.ibm.rhperfscale.org/#/experiments/256/runs/6b1884118aec42fbbaf9ed9accc1c05e) | [326fb9ac213b4a7facaf7a087f31e623](https://mlflow.apps.psap-automation.ibm.rhperfscale.org/#/experiments/256/runs/326fb9ac213b4a7facaf7a087f31e623) |
| Gemma 4 31B | [3fca95c95ebf4607a3de6347c18eb0b2](https://mlflow.apps.psap-automation.ibm.rhperfscale.org/#/experiments/318/runs/3fca95c95ebf4607a3de6347c18eb0b2) | [0c12c93eb595463a922c9d492a8c4285](https://mlflow.apps.psap-automation.ibm.rhperfscale.org/#/experiments/318/runs/0c12c93eb595463a922c9d492a8c4285) |
| Nemotron 253B FP8 | [5f9a464d28e84284943c8454f0bd5db3](https://mlflow.apps.psap-automation.ibm.rhperfscale.org/#/experiments/328/runs/5f9a464d28e84284943c8454f0bd5db3) | [17704f73746a417eba8cc6677570b5fc](https://mlflow.apps.psap-automation.ibm.rhperfscale.org/#/experiments/328/runs/17704f73746a417eba8cc6677570b5fc) |

This is a **historical comparison**, not a claim that only the connector changed:

- References used vLLM 0.23.0 and RHOAI 3.5 EA2. These definitions use the supplied
  LMCache-compatible image and the target's current RHOAI 3.5 GA. Preserve image
  digests and actual GPU KV capacity; a newer serving engine remains a confounder.
- These runs use AIPerf 0.12.0 while the references used 0.8.0. The corpus,
  seed, duration, and initial trajectory window are preserved, but the idle-cap
  mechanism and reconstruction code have changed. The archived runs do not pin
  their dataset revision. Inspect the exported input config and source metadata;
  do not treat these as identical client workloads or attribute every delta to
  the KV connector.
- The references have one seed per cell and lack pinned model snapshot revisions.
  Preserve those limitations; do not claim a confidence interval.
- The reports describe Gemma U0.92 and FP8 Nemotron KV. The archived Gemma pod
  does not explicitly set GPU utilization, and neither archived pod sets
  `--kv-cache-dtype`. These recipes use the report's Gemma U0.92 target and retain
  automatic KV dtype rather than silently forcing FP8 KV from the FP8 model name.
- Host-memory resource envelopes are now explicit. The LMCache server reserves
  CPU-pool capacity plus 8 GiB overhead; model processes have a separate envelope.
  Include both processes' actual CPU/RAM and the server's GPU contexts in costs.
  Identical GPU-utilization flags do not guarantee identical GPU KV token capacity.
- LMCache chunk geometry, hybrid/recurrent-cache handling, node placement, driver
  versions, and available GPU memory can differ from the historical observations.
  Report those differences instead of attributing the whole delta to the connector.

The experiments tag historical run IDs and corpus identity into MLflow. No
native rerun is required to collect these LMCache observations. If software
differences prevent a strict conclusion, label it as such rather than silently
launching additional controls.
