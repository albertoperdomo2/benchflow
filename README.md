# BenchFlow

[![Image build status](https://github.com/albertoperdomo2/benchflow/actions/workflows/build-images.yaml/badge.svg)](https://github.com/albertoperdomo2/benchflow/actions/workflows/build-images.yaml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

*Repeatable LLM inference benchmarks for OpenShift.*

> [!NOTE]
> This project is experimental and for learning purposes mainly, but the implemented execution paths today are `llm-d` and `RHOAI`. Expect some parts to still be highly coupled.

BenchFlow is a packaged control plane for running benchmark scenarios, not a loose collection of scripts. It resolves an experiment into one immutable `RunPlan`, executes it through Tekton `PipelineRun`s, captures metrics and artifacts, and pushes the result to MLflow. The current benchmark backends are [vllm-project/guidellm](https://github.com/vllm-project/guidellm) and one narrow [AIPerf](https://github.com/ai-dynamo/aiperf) Mooncake trace replay path.

> [!WARNING]
> BenchFlow now locks shared `llm-d` and `RHOAI` platform mutations per target cluster by setup key. Same-key runs can share a wave and use spare GPUs in parallel; different-key runs wait until the current admitted wave finishes. Shared platform prerequisites stay installed until a different setup key is requested or you explicitly tear them down.

## Quickstart

Install the CLI from the repository root:

```bash
pip install -e .
```

Before bootstrapping, create the real secret manifests next to the examples under `config/cluster/secrets/`. BenchFlow applies every `*.yaml` file there except `*.example.yaml`, so `cp` the examples secrets and remove the `.example` suffix, and populate them with your credentials.

BenchFlow supports two cluster topologies.

### Single cluster

```bash
bflow bootstrap --single-cluster
bflow experiment run experiments/smoke/qwen3-06b-llm-d-smoke.yaml
```

BenchFlow installs Tekton, Kueue, the BenchFlow remote-capacity controller, Grafana, RBAC, GPU prerequisites, and the required PVCs in the same cluster. Kueue admits runs locally against the discovered GPU capacity, and Tekton runs the full workflow there.

### Management cluster + remote target cluster(s)

Management cluster:

```bash
bflow bootstrap
```

Target cluster(s):

```bash
bflow bootstrap --target-kubeconfig ~/.kube/target-cluster --cluster-name target-cluster
```

Run from the management cluster:

```bash
bflow experiment run experiments/smoke/qwen3-06b-llm-d-smoke.yaml --cluster-name target-cluster
```

In this mode, the management cluster runs Tekton, Kueue, and the BenchFlow remote-capacity controller. Kueue queues executions by target cluster and admits them only when the target has enough GPU capacity. The target cluster does not need Tekton; BenchFlow launches the runtime work there through plain Kubernetes `Job`s using the stored kubeconfig Secret.

The narrow path is the shipped smoke experiment:

```bash
bflow experiment run experiments/smoke/qwen3-06b-llm-d-smoke.yaml
```

Then follow the execution:

```bash
bflow watch <execution-name> --namespace benchflow
```

BenchFlow also supports matrix experiments by turning one or more profile fields into lists; the cluster then runs the cartesian product through child executions. `rhoai` and `llm-d` child executions can be admitted in parallel when target-cluster GPU capacity allows it. Once the parent has submitted child executions, canceling the parent is only best-effort; already queued or running children may need individual cancellation.

For the full command surface, RunPlan PipelineRun flow, matrix execution, and lower-level runtime commands, see [docs/ADVANCED.md](docs/ADVANCED.md).

## Known Limitations

- The current AIPerf support is intentionally narrow: one `aiperf-mooncake-trace` benchmark profile that downloads the Mooncake conversation trace, optionally caps it, and benchmarks an existing chat-style endpoint. It is not generic AIPerf support.
- Comparison reports do not mix benchmark tools. `bflow benchmark plot comparison` can compare multiple GuideLLM runs or multiple AIPerf runs, but not both in one report.
- Legacy target clusters without BenchFlow platform state are adopted heuristically; the first mutating run after upgrading BenchFlow may reset and reinstall shared `llm-d` or `RHOAI` prerequisites.
- BenchFlow coordinates only the shared platform state it manages itself. Manual or out-of-band cluster mutations are not version-reconciled.
- `llm-d` matrix children depend on release-scoped chart values; use the current BenchFlow image when testing parallel llm-d runs.
- Matrix parent cancellation is best-effort after child executions have already been submitted; queued or running children may need to be cancelled individually.
- BenchFlow manages GuideLLM benchmark output paths itself. `GUIDELLM_OUTPUT_DIR` is set automatically during benchmark execution, and `GUIDELLM_OUTPUT_PATH` is not supported in benchmark environment overrides.
