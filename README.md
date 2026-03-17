# BenchFlow

[![Image build status](https://github.com/albertoperdomo2/benchflow/actions/workflows/build-images.yaml/badge.svg)](https://github.com/albertoperdomo2/benchflow/actions/workflows/build-images.yaml)

*Repeatable LLM inference benchmarks for OpenShift.*

BenchFlow is a packaged control plane for running benchmark scenarios, not a loose collection of scripts. It resolves an experiment into one immutable `RunPlan`, executes it through Argo Workflows, captures metrics and artifacts, and pushes the result to MLflow. It is powered by [vllm-project/guidellm](https://github.com/vllm-project/guidellm).

> [!NOTE]
> This project is experimental and for learning purposes mainly, but the implemented execution paths today are `llm-d` and `RHOAI`. Expect some parts to still be highly coupled.

## Quickstart

Install the CLI from the repository root:

```bash
pip install -e .
```

Before bootstrapping, create the real secret manifests next to the examples under `config/cluster/secrets/`. BenchFlow applies every `*.yaml` file there except `*.example.yaml`, so `cp` the examples secrets and remove the `.example` suffix, and populate them with your credentials.

Then bootstrap the cluster:

```bash
bflow bootstrap
```

`bflow bootstrap` installs the shared baseline BenchFlow owns: NFD, the NVIDIA GPU Operator, Argo Workflows, Grafana, RBAC, PVCs, and the repo-root Argo reusable templates and workflow templates. BenchFlow assumes an OpenShift cluster with cluster monitoring enabled, a reachable MLflow deployment backed by S3, and a usable storage class.

The narrow path is the shipped smoke experiment:

```bash
bflow experiment run experiments/smoke/qwen3-06b-llm-d-smoke.yaml
```

Then follow the execution:

```bash
bflow watch <execution-name> --namespace benchflow
```

BenchFlow also supports matrix experiments by turning one or more profile fields into lists; the cluster then runs the cartesian product sequentially in the cluster.

For the full command surface, RunPlan workflow, matrix execution, and lower-level runtime commands, see [docs/ADVANCED.md](docs/ADVANCED.md).
