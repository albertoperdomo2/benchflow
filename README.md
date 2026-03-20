# BenchFlow

[![Image build status](https://github.com/albertoperdomo2/benchflow/actions/workflows/build-images.yaml/badge.svg)](https://github.com/albertoperdomo2/benchflow/actions/workflows/build-images.yaml)

*Repeatable LLM inference benchmarks for OpenShift.*

BenchFlow is a packaged control plane for running benchmark scenarios, not a loose collection of scripts. It resolves an experiment into one immutable `RunPlan`, executes it through Tekton `PipelineRun`s, captures metrics and artifacts, and pushes the result to MLflow. It is powered by [vllm-project/guidellm](https://github.com/vllm-project/guidellm).

> [!NOTE]
> This project is experimental and for learning purposes mainly, but the implemented execution paths today are `llm-d` and `RHOAI`. Expect some parts to still be highly coupled.

> [!WARNING]
> BenchFlow does not yet implement a cluster-level lock for shared platform setup. Until that exists, let each benchmark job run end to end before launching another one that mutates the cluster, or use one matrix experiment so BenchFlow can queue and run multiple combinations through Kueue with controlled parallelism.

## Quickstart

Install the CLI from the repository root:

```bash
pip install -e .
```

Before bootstrapping, create the real secret manifests next to the examples under `config/cluster/secrets/`. BenchFlow applies every `*.yaml` file there except `*.example.yaml`, so `cp` the examples secrets and remove the `.example` suffix, and populate them with your credentials.

Then bootstrap the cluster:

```bash
bflow bootstrap --single-cluster
```

Use `bflow bootstrap --single-cluster` when BenchFlow will orchestrate and run workloads in the same cluster. Plain `bflow bootstrap` is the management-cluster path: it installs Tekton, Kueue, the BenchFlow remote-capacity controller, Grafana, RBAC, the `benchmark-results` PVC, and the repo-root Tekton tasks and pipelines, but it does not install NFD, the GPU Operator, or the `models-storage` PVC. For a remote target cluster, use `bflow bootstrap --target-kubeconfig ... --cluster-name ...`; BenchFlow bootstraps the target runtime pieces there and automatically registers that cluster in the management-cluster Kueue queues.

The narrow path is the shipped smoke experiment:

```bash
bflow experiment run experiments/smoke/qwen3-06b-llm-d-smoke.yaml
```

Then follow the execution:

```bash
bflow watch <execution-name> --namespace benchflow
```

BenchFlow also supports matrix experiments by turning one or more profile fields into lists; the cluster then runs the cartesian product through child executions that Kueue can admit in parallel when target-cluster GPU capacity allows it. Once the parent has submitted child executions, canceling the parent is only best-effort; already queued or running children may need individual cancellation.

For the full command surface, RunPlan PipelineRun flow, matrix execution, and lower-level runtime commands, see [docs/ADVANCED.md](docs/ADVANCED.md).
