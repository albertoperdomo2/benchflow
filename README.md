# BenchFlow

[![Image build status](https://github.com/albertoperdomo2/benchflow/actions/workflows/build-images.yaml/badge.svg)](https://github.com/albertoperdomo2/benchflow/actions/workflows/build-images.yaml)

BenchFlow is a control plane for repeatable LLM inference benchmarks on OpenShift. It turns an experiment file, or an equivalent set of CLI flags, into one resolved run plan, deploys the scenario, runs the benchmark, captures metrics and artifacts, and pushes the result to MLflow. The current implemented execution paths are `llm-d` and `rhoai`. Experiments may also define cartesian products of deployment, benchmark, and metrics profiles; in that case BenchFlow submits one supervisor PipelineRun that executes the child combinations sequentially in the cluster.

The default runtime image is `ghcr.io/albertoperdomo2/benchflow/benchflow:latest`. The default namespace is `benchflow`.

For the full command reference and advanced workflows, see [docs/ADVANCED.md](docs/ADVANCED.md).

BenchFlow bootstraps the cluster resources it owns. That includes NFD, the NVIDIA GPU Operator, Tekton, Grafana, RBAC, PVCs, and the packaged Tekton tasks and pipelines. It currently supports `llm-d` and `rhoai` execution, leaves `rhaiis` as future work, and assumes an OpenShift cluster with cluster monitoring enabled and a reachable MLflow deployment backed by S3.

## Bootstrap

Install the CLI from the repository root:

```bash
pip install -e .
```

Before bootstrapping the cluster, create the real secret manifests. BenchFlow reads `config/cluster/secrets/` and applies every `*.yaml` file there except `*.example.yaml`, so the intended flow is to copy the example files, fill in the real values, and keep the copied files alongside them:

```bash
cp config/cluster/secrets/huggingface-token.example.yaml config/cluster/secrets/huggingface-token.yaml
cp config/cluster/secrets/mlflow-auth.example.yaml config/cluster/secrets/mlflow-auth.yaml
cp config/cluster/secrets/mlflow-s3-creds.example.yaml config/cluster/secrets/mlflow-s3-creds.yaml
```

Then bootstrap the cluster:

```bash
bflow bootstrap
```

## Run

The narrow path is the shipped smoke experiment:

```bash
bflow experiment validate experiments/smoke/qwen3-06b-llm-d-smoke.yaml
bflow experiment run experiments/smoke/qwen3-06b-llm-d-smoke.yaml
```

Then follow the PipelineRun:

```bash
bflow watch <pipelinerun-name> --namespace benchflow
```

The same run can be launched directly from flags:

```bash
bflow experiment run \
  --name qwen3-06b \
  --model Qwen/Qwen3-0.6B \
  --model-revision main \
  --deployment-profile llm-d-inference-scheduling \
  --benchmark-profile guidellm-concurrent-1k-1k \
  --metrics-profile detailed \
  --namespace benchflow \
  --mlflow-experiment benchflow-qwen
```

A matrix experiment can keep the same shape and turn one or more profile fields into lists:

```yaml
spec:
  deployment_profile: [llm-d-inference-scheduling, llm-d-precise-prefix-cache]
  benchmark_profile: [guidellm-smoke, guidellm-concurrent-1k-1k]
  metrics_profile: detailed
```

`bflow experiment run` will expand that cartesian product, submit one supervisor PipelineRun, and let the cluster run the child combinations without keeping the local CLI process alive.

If you want to inspect the packaged profiles first:

```bash
bflow profiles list
bflow profiles show llm-d-inference-scheduling --kind deployment
```
