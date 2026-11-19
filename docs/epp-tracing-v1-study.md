# EPP tracing overhead on llm-d 1.0.0

These matrices rerun the study behind
[website PR #543](https://github.com/llm-d/llm-d.github.io/pull/543) and
[router PR #3068](https://github.com/llm-d/llm-d-router/pull/3068).
The two backends are live vLLM and the inference simulator. Both use synthetic
GuideLLM prompts, as in the study; no recorded AgentX replay is included.
The definitions and released Helm/Kustomize output are validated offline.
No workload or cluster change has been launched for this preparation.

## What was measured

The retained study has eight primary performance observations: simulator and
live vLLM, each at tracing off, 1%, 10% and 100%. Each retained primary cell
has one repetition. Two additional 100% runs collect CPU and heap profiles;
their request performance is excluded because profiling adds overhead.

The fixed workload is nonstreaming Poisson **400 requests/s**, **200 input /
100 output tokens**, **300 seconds**, seed **20260919**, with a maximum
concurrency of 512. There is one EPP replica and eight backend replicas.
The live model is `Qwen/Qwen3-0.6B`, TP1 per replica, consuming eight GPUs.
The simulator requires zero GPUs and uses constant 20 ms TTFT / 1 ms ITL.

The simulator sensitivity study also covers streaming and nonstreaming:

| Family | Input / output tokens | Load stages per invocation |
| --- | --- | --- |
| Rate | 200 / 100 | 25, 100, 400, 800 requests/s |
| Short output | 200 / 1 | 100 requests/s |
| Long input | 8000 / 100 | 20 requests/s |
| Long output | 200 / 1000 | 10 requests/s |
| Closed loop | 200 / 100 | Concurrency 1, 32, 128 |

Each stage lasts 300 seconds. Each family is exercised at off, enabled 0%,
1%, 10%, 50% and 100% sampling. This is 60 stages per response mode, not a
Cartesian product of every token shape and every load. The live backend's
historical sweep was limited to the fixed workload.

Historical exclusions matter: live 1% had two HTTP 503 responses; the first
fixed simulator cohort lacked complete metric coverage; streaming 800 RPS and
concurrency 128 failed at every sampling setting because of load-generator
connection errors. Clean nonstreaming replacements exist. The supporting
AIPerf cohort also had errors and is excluded. These are not accepted evidence
of tracing-specific regressions. The source study did not independently verify
effective sampling or establish that absent throttling series meant zero.

Canonical files in `llm-d-router/test/perf/config/studyanalysis/` are
`epp_tracing_overhead_analysis.md`, `epp_tracing_overhead/README.md`,
`epp_tracing_overhead/guidellm/run.sh` and the study worktree's
`epp_tracing_overhead/data/run_registry.csv` (MLflow experiment 461).

## Version and instrumentation controls

| Component | Historical study | Rerun |
| --- | --- | --- |
| llm-d guide | v0.9.0 | **v1.0.0** |
| Router chart / app / EPP image | v0.10.0 | **v1.0.0 / v1.0.0 / v1.0.0** |
| EPP configuration API | Historical recipe | `llm-d.ai/v1`, from released recipe |
| GuideLLM | 0.7.3 | 0.7.3 |
| Simulator | v0.11.2 | v0.11.2 |
| Live vLLM | v0.27.0 | v0.27.0 |

The latest released llm-d and router were verified as v1.0.0. Simulator and
vLLM versions remain fixed to isolate the router upgrade. The simulator has
newer releases, but this matrix deliberately preserves the measured backend.
Record actual image digests and the cloned guide commit with each run.
The released router chart reports both `version` and `appVersion` as v1.0.0.

Study metrics profiles explicitly use:

```yaml
tracing:
  scope: epp
  mode: standard
  sample_ratio: 0.1
```

Only the EPP receives tracing flags/environment. vLLM and the P/D proxy remain
uninstrumented. `mode: off` sets the router's tracing flag false; enabled 0%
initializes the SDK/exporter with ratio zero. Other tracing profiles retain the
default `scope: deployment` behavior. EPP scope is supported only for
BenchFlow-managed llm-d deployments.

BenchFlow keeps the shared collector/Jaeger plane available even for the off
control. Collector metrics on port 8888 are scraped through a 10-second
PodMonitor. The metrics profiles retain accepted/refused/exported/failed span
counters, raw EPP request/CPU counters and latency buckets, alongside the
existing CPU, memory, throttling and histogram-quantile views. This provides
evidence for sampling validation without counting retained Jaeger traces as
the total number exported. Do not run independent tracing studies concurrently
in the same namespace: the collector counters are namespace-wide.

## Matrices

| Experiment file under `experiments/llm-d/epp-tracing/` | Backend | Scope | Child runs | Measured stages / time |
| --- | --- | --- | ---: | --- |
| `epp-tracing-v1-primary.yaml` | Simulator and live | Off / 1% / 10% / 100%, three repetitions | 24 | 24 stages / 2 hours |
| `epp-tracing-v1-pprof.yaml` | Simulator and live | 100%, CPU + heap diagnostic | 2 | 2 stages / 10 minutes |
| `epp-tracing-v1-sensitivity.yaml` | Simulator | Five families x two response modes x six settings | 60 | 120 stages / 10 hours |

Deployment, readiness, trace export and artifact collection add time to these
measurement durations. All children run sequentially with a fresh deployment
and cleanup. Separate parent submissions are not globally serialized.

The primary profile list contains three explicit off/1%/10%/100% rounds for
each backend. Duplicate profile references are intentional; BenchFlow retains
them. Within each backend, child order identifies repetitions 1, 2 and 3;
retain the matrix child index and use the matching round when comparing ratios.
The fixed order is not counterbalanced, so preserve that limitation in analysis.
CPU profiling starts 60 seconds after benchmark invocation, captures 30 seconds
of CPU plus heap from every ready EPP, and is kept out of primary performance.

## Release and launch

The affinity is profile configuration. EPP-only instrumentation is a code change
and requires a new BenchFlow release for both the controller and matrix tasks.
The first attempt, `epp-tracing-v1-primary-matrix-444d4a` on 2026-11-19, used
v0.10.0 and failed before submitting children with:

```text
error metrics.tracing contains unsupported fields: scope
```

Push the study changes to main and cut a release through the existing release
workflow. The serving images require no build. Use the resulting published tag
below; v0.10.0 cannot execute these matrices.

From the management-cluster context, reconcile the controller and execution
resources with the released image:

```bash
export KUBECONFIG="$HOME/workspace/redhat/clusters/aperdomo-lab/auth/kubeconfig"
export TARGET_CLUSTER=psap-h100-diadochos
export BENCHFLOW_IMAGE=ghcr.io/albertoperdomo2/benchflow:vX.Y.Z # New study-capable release
bflow bootstrap --benchflow-image "$BENCHFLOW_IMAGE"
```

Run all three matrices sequentially. `--follow` keeps each invocation attached
until it finishes; without it, the CLI prints the submitted parent name and
returns immediately. The loop stops on a failed matrix so its cause can be
checked before proceeding:

```bash
for phase in primary pprof sensitivity; do
  bflow experiment run \
    "experiments/llm-d/epp-tracing/epp-tracing-v1-${phase}.yaml" \
    --cluster-name "$TARGET_CLUSTER" \
    --benchflow-image "$BENCHFLOW_IMAGE" --follow || break
done
```

The three matrices total **86 child runs / 146 measured stages**, including
primary and diagnostics. They must not run concurrently in the same namespace
because collector counters are shared.

Both deployment profiles use required hostname `NotIn` node affinity excluding
`diadochos-hqxzk-gpu-h100-mt46x` from the model pods. This uses the existing
`runtime.affinity` field; no node labels, taints or GPU allocator changes are
needed. The affinity does not constrain gateway, EPP or operational helper pods.

When RHOAI is already recorded as the installed platform, these llm-d deployments
without LMCache validate existing Gateway API/InferencePool CRDs and Istio with
inference-extension support. They preserve RHOAI's operators and platform state.
If prerequisites are absent or unsuitable, setup fails without resetting RHOAI.
Teardown also preserves an installed platform different from the requested one.
Kueue continues serializing incompatible setup-key waves.

The target must already be registered with BenchFlow. Live inference requires
eight schedulable GPUs and sufficient host memory for the declared resource
envelope; the simulator reserves zero GPUs and skips model download. BenchFlow
sets up llm-d prerequisites and tracing services, deploys the selected backend,
executes GuideLLM, collects metrics/traces/pprof, uploads to MLflow and cleans up
each execution. Cluster readiness and an actual benchmark are not established
by the offline checks described here.

## Acceptance and reporting

Accept each GuideLLM stage separately: require zero request errors, preserve
stage boundaries and record restarts/readiness/connection failures. Reject
the affected streaming 800-RPS/C128 stages again if they reproduce the client
failure; clean earlier stages remain usable. Do not silently count requested
load as achieved throughput. Retain all attempted cells and their disposition.

Verify actual EPP image/app version, one ready EPP, eight backend replicas,
the configured sampler and ratio, and usable 10-second Prometheus samples.
Compare collector span-count deltas per EPP request against the corresponding
100% baseline to detect parent-sampling overrides or span losses. Keep refused
and failed exports visible. Missing collector/throttling series mean the
measurement is unavailable, not zero; a configured ratio alone is not proof
of effective sampling. Jaeger retention can limit stored traces independently
of accepted/exported span counters.

Report throughput, mean/p50/p95/p99 client latency, EPP CPU/memory, scheduler
and request/response-processing latency, errors, and diagnostic pprof stacks.
EPP histogram-quantile time samples are not pooled request percentiles. Keep
all native samples and report repetition-level results before summarizing.
Separate whole-request outcomes from EPP-internal cost, and keep diagnostic
profiles outside the primary latency comparison. The simulator's configured
response delay can hide small router changes; flat end-to-end results do not
establish zero tracing overhead.
