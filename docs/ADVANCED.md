# BenchFlow Advanced Guide

This document is the full operational guide for BenchFlow as it exists today.
The implemented execution paths are `llm-d` and `rhoai`. `rhaiis` remains future
work and should be treated as an unsupported placeholder.

## Mental Model

BenchFlow has two public configuration layers.

`Experiment`
- user-facing input
- references packaged profiles by name
- can be written as a single scenario or as a cartesian product of profiles

`RunPlan`
- fully resolved, immutable execution document
- contains the exact deployment, benchmark, metrics, and stage configuration
- is what Tekton and the internal `bflow task ...` entrypoints actually run

The normal path is:

```bash
bflow experiment validate my-experiment.yaml
bflow experiment run my-experiment.yaml
```

The advanced path is:

```bash
bflow experiment resolve my-experiment.yaml --format json > runplan.json
# edit runplan.json
bflow run-plan run runplan.json
```

## Bootstrap

Install BenchFlow locally from the repository root:

```bash
pip install -e .
```

Before bootstrapping the cluster, create real secret manifests next to the
examples under `config/cluster/secrets/`. BenchFlow applies every `*.yaml` file
there except `*.example.yaml`.

Typical flow:

```bash
cp config/cluster/secrets/huggingface-token.example.yaml config/cluster/secrets/huggingface-token.yaml
cp config/cluster/secrets/mlflow-auth.example.yaml config/cluster/secrets/mlflow-auth.yaml
cp config/cluster/secrets/mlflow-s3-creds.example.yaml config/cluster/secrets/mlflow-s3-creds.yaml
```

Then bootstrap the cluster:

```bash
bflow bootstrap
```

Today `bflow bootstrap` installs or configures:

- NFD operator and `NodeFeatureDiscovery` instance
- NVIDIA GPU Operator and `ClusterPolicy`
- OpenShift Pipelines
- Grafana
- BenchFlow RBAC
- BenchFlow PVCs
- packaged Tekton tasks and pipelines

## Profiles

Profiles are packaged with the tool and are resolved by name.

List them:

```bash
bflow profiles list
bflow profiles list --kind deployment
bflow profiles list --kind benchmark
bflow profiles list --kind metrics
```

Inspect one:

```bash
bflow profiles show llm-d-inference-scheduling --kind deployment
bflow profiles show guidellm-smoke --kind benchmark
bflow profiles show detailed --kind metrics
```

## Experiments

An `Experiment` is the normal user-facing document:

```yaml
apiVersion: benchflow.io/v1alpha1
kind: Experiment
metadata:
  name: qwen3-06b
spec:
  model:
    name: Qwen/Qwen3-0.6B
  deployment_profile: llm-d-inference-scheduling
  benchmark_profile: guidellm-smoke
  metrics_profile: detailed
  namespace: benchflow
```

Validate it:

```bash
bflow experiment validate experiments/smoke/qwen3-06b-llm-d-smoke.yaml
```

Resolve it into a concrete `RunPlan`:

```bash
bflow experiment resolve experiments/smoke/qwen3-06b-llm-d-smoke.yaml --format json
```

Render the PipelineRun without submitting it:

```bash
bflow experiment render-pipelinerun experiments/smoke/qwen3-06b-llm-d-smoke.yaml
```

Submit it:

```bash
bflow experiment run experiments/smoke/qwen3-06b-llm-d-smoke.yaml
```

Follow it later:

```bash
bflow watch <pipelinerun-name> --namespace benchflow
```

List running and finished experiments:

```bash
bflow experiment list
bflow experiment list --all
```

Cancel one:

```bash
bflow experiment cancel <pipelinerun-name>
```

Submit a cleanup-only run:

```bash
bflow experiment cleanup experiments/smoke/qwen3-06b-llm-d-smoke.yaml
```

## Direct CLI Experiments

Every single-scenario experiment has a CLI equivalent:

```bash
bflow experiment run \
  --name qwen3-06b \
  --model Qwen/Qwen3-0.6B \
  --deployment-profile llm-d-inference-scheduling \
  --benchmark-profile guidellm-smoke \
  --metrics-profile detailed \
  --namespace benchflow
```

Direct CLI flags are best for one-off runs. Files are better for repeatability.

## RunPlan Workflow

`RunPlan` is the advanced interface when you want to inspect or edit the fully
resolved configuration before submitting it.

Generate one:

```bash
bflow experiment resolve experiments/smoke/qwen3-06b-llm-d-smoke.yaml --format json > runplan.json
```

Validate it:

```bash
bflow run-plan validate runplan.json
```

Render the PipelineRun from it:

```bash
bflow run-plan render-pipelinerun runplan.json
```

Submit it:

```bash
bflow run-plan run runplan.json
```

Submit a cleanup-only run from it:

```bash
bflow run-plan cleanup runplan.json
```

This is the closest BenchFlow has to a `helm template` style workflow:

1. resolve an experiment
2. save the `RunPlan`
3. edit the JSON
4. run the edited plan

`run-plan` commands expect exactly one resolved `RunPlan`. If you resolve a
matrix experiment, the output is a JSON array of `RunPlan` objects, not a
single file suitable for `bflow run-plan run`.

## Matrix Experiments

An experiment can specify one or more values for each profile axis:

```yaml
apiVersion: benchflow.io/v1alpha1
kind: Experiment
metadata:
  name: qwen3-06b
spec:
  model:
    name: Qwen/Qwen3-0.6B
  deployment_profile:
    - llm-d-inference-scheduling
    - llm-d-precise-prefix-cache
  benchmark_profile:
    - guidellm-smoke
    - guidellm-concurrent-1k-1k
  metrics_profile: detailed
```

BenchFlow expands the cartesian product of those profile lists.

Current behavior:

- each combination becomes one normal child `RunPlan`
- each child `RunPlan` becomes one normal child PipelineRun
- `bflow experiment run` submits one supervisor PipelineRun
- the supervisor runs the child combinations sequentially in the cluster
- each child benchmark still creates its own MLflow run

So this is safe to submit and walk away from:

```bash
bflow experiment run experiments/smoke/qwen3-06b-llm-d-matrix-smoke.yaml
```

The shipped matrix smoke example intentionally produces two child runs.

## Dynamic MLflow Defaults

If you do not set `spec.mlflow.experiment`, BenchFlow derives one automatically:

```text
{sanitized-model-name}-{benchmark-profile}
```

Example:

```text
qwen-qwen3-06b-guidellm-smoke
```

BenchFlow also sets default MLflow tags from the resolved run:

- `deployment_type`
- `deployment_profile`
- `benchmark_profile`
- `metrics_profile`

The benchmark runtime adds:

- `vllm_version`
- `guidellm_version`

User-provided tags still override the defaults if you need to force a value.

## Runtime Commands

BenchFlow also exposes the lower-level runtime commands that Tekton uses inside
the control image. These are useful for debugging or local step-by-step work.

Download the model:

```bash
bflow model download --run-plan-file runplan.json --models-storage-path /path/to/models
```

Deploy:

```bash
bflow deploy llm-d --run-plan-file runplan.json
```

Wait for readiness:

```bash
bflow wait endpoint --run-plan-file runplan.json
```

Run the benchmark:

```bash
bflow benchmark run --run-plan-file runplan.json --output-dir ./results
```

Collect artifacts:

```bash
bflow artifacts collect --run-plan-file runplan.json --pipeline-run-name <name> --artifacts-dir ./artifacts
```

Collect metrics:

```bash
bflow metrics collect \
  --run-plan-file runplan.json \
  --benchmark-start-time <iso8601> \
  --benchmark-end-time <iso8601> \
  --artifacts-dir ./artifacts
```

Upload to MLflow:

```bash
bflow mlflow upload \
  --run-plan-file runplan.json \
  --mlflow-run-id <run-id> \
  --benchmark-start-time <iso8601> \
  --benchmark-end-time <iso8601> \
  --artifacts-dir ./artifacts
```

Cleanup:

```bash
bflow undeploy llm-d --run-plan-file runplan.json
```

## Monitoring and Results

BenchFlow installs Grafana for live dashboards and uploads results to MLflow.

Today the live path is:

- benchmark outputs and reports go to MLflow
- collected metrics go to MLflow
- collected logs and manifests go to MLflow
- MLflow runs get a `grafana_url` tag for the live dashboard window

The archive dashboard and Infinity datasource were intentionally removed. The
current supported Grafana path is the live Prometheus-backed dashboard only.

## Current Assumptions

BenchFlow currently assumes:

- OpenShift
- cluster monitoring is available
- MLflow is reachable
- MLflow artifacts are backed by S3
- a suitable storage class exists for the BenchFlow PVCs
- `llm-d` and `rhoai` are the implemented execution platforms

It does not currently implement:

- `rhaiis` execution
- public cluster-stored custom profiles
- public RunPlan matrix submission from a JSON array

## Troubleshooting

Check the resolved plan first:

```bash
bflow experiment resolve my-experiment.yaml --format json
```

Render the PipelineRun before submitting:

```bash
bflow experiment render-pipelinerun my-experiment.yaml
```

Or, for a resolved plan:

```bash
bflow run-plan render-pipelinerun runplan.json
```

If a run is already in the cluster:

```bash
bflow experiment list --all
bflow watch <pipelinerun-name> --namespace benchflow
```

If you need to stop it:

```bash
bflow experiment cancel <pipelinerun-name>
```
