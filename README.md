# BenchFlow

BenchFlow runs end-to-end LLM inference benchmarks on OpenShift. It takes either an experiment file or direct CLI flags, resolves them into a single run plan, deploys the model, runs the benchmark, captures metrics and artifacts, and uploads the results to MLflow. The current implemented execution path is `llm-d`, using the `bflow` CLI locally and inside the Tekton tasks.

## Install

Install the CLI locally from the repository root:

```bash
pip install -e .
```

Then bootstrap the cluster resources:

```bash
bflow install
```

That installs or verifies Tekton, installs Grafana, creates the BenchFlow namespace, applies RBAC, provisions the PVCs, creates the Grafana route and dashboard, and installs the Tekton tasks and pipelines. After that, create the required secrets with your real values:

```bash
oc apply -n benchflow -f config/cluster/secrets/huggingface-token.example.yaml
oc apply -n benchflow -f config/cluster/secrets/mlflow-auth.example.yaml
oc apply -n benchflow -f config/cluster/secrets/mlflow-s3-creds.example.yaml
```

## Run With An Experiment File

The simplest path is to use the shipped smoke experiment:

```bash
bflow experiment validate experiments/examples/qwen3-06b-llm-d-smoke.yaml
bflow experiment run experiments/examples/qwen3-06b-llm-d-smoke.yaml
```

Then watch the PipelineRun:

```bash
bflow watch <pipelinerun-name> --namespace benchflow
```

## Run With CLI Flags

The same run can be launched without an experiment file:

```bash
bflow experiment run \
  --name qwen3-06b \
  --model Qwen/Qwen3-0.6B \
  --model-revision main \
  --deployment-profile llm-d-inference-scheduling \
  --benchmark-profile guidellm-concurrent-1k-1k \
  --metrics-profile default \
  --namespace benchflow \
  --mlflow-experiment benchflow-qwen
```

If you want to inspect what is available before running, use:

```bash
bflow profiles list
bflow profiles show llm-d-inference-scheduling --kind deployment
```