# BenchFlow Advanced Guide

This document is the full operational guide for BenchFlow as it exists today.
The implemented execution paths are `llm-d` and `rhoai`. `rhaiis` remains future
work and should be treated as an unsupported placeholder.

## Table of Contents

- [Mental Model](#mental-model)
- [Bootstrap](#bootstrap)
- [Cluster Topologies](#cluster-topologies)
- [Profiles](#profiles)
- [Experiments](#experiments)
- [Direct CLI Experiments](#direct-cli-experiments)
- [RunPlan Workflow](#runplan-workflow)
- [Matrix Experiments](#matrix-experiments)
- [Dynamic MLflow Defaults](#dynamic-mlflow-defaults)
- [Runtime Commands](#runtime-commands)
- [Monitoring and Results](#monitoring-and-results)
- [Comparison Reports](#comparison-reports)
- [Current Assumptions](#current-assumptions)
- [Troubleshooting](#troubleshooting)

## Mental Model

BenchFlow has two public configuration layers.

`Experiment`
- user-facing input
- references packaged profiles by name
- can be written as a single scenario or as a cartesian product of profiles

`RunPlan`
- fully resolved, immutable execution document
- contains the exact deployment, benchmark, metrics, and stage configuration
- is what the execution backend and the internal `bflow task ...` entrypoints actually run

Internally, BenchFlow is split into two implementation layers:

`contracts`
- shared `RunPlan`, execution context, and execution summary types
- the explicit boundary between orchestration and toolbox

`orchestration`
- PipelineRun rendering, submission, watch, and cancellation
- Tekton-specific sequencing and cluster execution behavior

`toolbox`
- reusable operational actions driven by a `RunPlan`
- setup, deploy, benchmark, artifact collection, metrics collection, and MLflow upload
- callable from the CLI and from orchestration without duplicating business logic

The normal path is:

```bash
bflow experiment validate my-experiment.yaml
bflow experiment run my-experiment.yaml
```

For `llm-d` and `rhoai`, the normal path now includes a reversible platform
setup step before deployment. BenchFlow records what it changed and tears
those changes down during cleanup.

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
bflow bootstrap --single-cluster
```

`bflow bootstrap` supports three modes.

### Management cluster only

```bash
bflow bootstrap
```

This prepares the control plane only:

- OpenShift Pipelines
- Kueue
- BenchFlow remote-capacity controller
- Grafana
- BenchFlow RBAC
- `benchmark-results` PVC
- repo-root Tekton tasks and pipelines

This mode does not install:

- NFD
- NVIDIA GPU Operator
- `models-storage` PVC

### Single cluster

```bash
bflow bootstrap --single-cluster
```

This installs everything in one cluster so BenchFlow can both orchestrate and
run workloads there:

- NFD operator and `NodeFeatureDiscovery` instance
- NVIDIA GPU Operator and `ClusterPolicy`
- OpenShift Pipelines
- Kueue
- BenchFlow remote-capacity controller
- Grafana
- BenchFlow RBAC
- `models-storage` and `benchmark-results` PVCs
- repo-root Tekton tasks and pipelines
- a `local` Kueue queue sized from the cluster's discovered GPU capacity

### Remote target cluster

To bootstrap a target cluster from the management cluster:

```bash
bflow bootstrap \
  --target-kubeconfig ~/.kube/target-cluster \
  --cluster-name target-cluster
```

When `--target-kubeconfig` is set, BenchFlow defaults to a runtime-only target
bootstrap:

- NFD and the GPU Operator are installed
- `models-storage` and `benchmark-results` PVCs are installed
- Tekton is not installed unless `--install-tekton` is passed
- Grafana is not installed unless `--install-grafana` is passed
- when `--cluster-name` is also set, BenchFlow creates a management-cluster
  kubeconfig Secret with the same name
- when `--cluster-name` is also set, BenchFlow also registers a Kueue queue for
  that target cluster in the management cluster, sized from the target
  cluster's discovered GPU capacity
- if the target cluster API needs a static host entry, pass one or more
  `--host-alias hostname=ip-address` values; BenchFlow stores them on the named
  target-cluster Secret and injects them into management-cluster controller and
  Tekton pods automatically

If you are developing BenchFlow itself, pass `--benchflow-image ghcr.io/...`
to `bflow bootstrap` so the management-cluster remote-capacity controller uses
the same image tag as your Tekton runs.

## Cluster Topologies

BenchFlow supports two operating modes.

### Same-cluster

Tekton, BenchFlow, and the actual benchmarked workloads all live in the same
cluster.

```bash
bflow bootstrap --single-cluster
bflow experiment run my-experiment.yaml
```

This is the default and simplest path.

### Management cluster targeting a remote cluster

Tekton runs in the management cluster, but setup, deploy, benchmark, metrics,
and cleanup affect a different target cluster. The target cluster does not need
Tekton.

1. Bootstrap the management cluster normally:

```bash
bflow bootstrap
```

2. Bootstrap the target cluster:

```bash
bflow bootstrap \
  --target-kubeconfig ~/.kube/target-cluster \
  --cluster-name target-cluster
```

If the target cluster API hostname is not resolvable from management-cluster
pods, register a static host entry once during bootstrap:

```bash
bflow bootstrap \
  --target-kubeconfig ~/.kube/target-cluster \
  --cluster-name target-cluster \
  --host-alias api.target.example.com=192.0.2.10
```

3. Launch the experiment from the management cluster:

```bash
bflow experiment run my-experiment.yaml \
  --cluster-name target-cluster
```

This is equivalent to resolving `--target-kubeconfig-secret target-cluster`.
The explicit Secret command is still available if you want to manage the Secret
yourself:

```bash
bflow target kubeconfig-secret create \
  --name target-cluster \
  --kubeconfig ~/.kube/target-cluster \
  --host-alias api.target.example.com=192.0.2.10 \
  --namespace benchflow
```

Or embed the Secret reference in the Experiment itself:

```yaml
spec:
  target_cluster:
    kubeconfig_secret: target-cluster
```

Use `--target-kubeconfig` only for direct local BenchFlow commands such as
target-cluster bootstrap. Tekton `PipelineRun`s cannot mount your local
filesystem, so in-cluster executions must use `kubeconfig_secret`.

BenchFlow uses Kueue only in the management cluster. When you submit an
execution, BenchFlow creates a Kueue reservation `Workload` plus a stored
pending `PipelineRun` manifest in the management cluster and then exits. The
BenchFlow remote-capacity controller watches that `Workload`; once the
AdmissionCheck admits it, the controller creates the Tekton `PipelineRun`
itself. For remote target clusters, the queue name is the `--cluster-name`
value; for same-cluster runs, the queue name is `local`.

This means `bflow experiment run ...` is set-and-forget again: your laptop does
not need to stay alive while the execution waits in queue.

Important limitation:
- this Kueue admission path only gates GPU capacity and start order
- it does not replace a cluster-level lock for shared platform mutations
- concurrent runs can still race on setup and teardown of shared platform state
- matrix parent cancellation is best-effort once child executions have already been submitted; queued or running children may need to be cancelled individually
- until BenchFlow has an explicit cluster lock, treat Kueue as capacity admission, not full mutation safety

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

Full schema:

```yaml
apiVersion: benchflow.io/v1alpha1
kind: Experiment
metadata:
  name: qwen3-06b # --name
  labels: # --label KEY=VALUE
    team: perf
spec:
  model:
    name: Qwen/Qwen3-0.6B # --model, string or list for matrix
  deployment_profile: llm-d-inference-scheduling # --deployment-profile
  benchmark_profile: guidellm-smoke # --benchmark-profile
  metrics_profile: detailed # --metrics-profile
  namespace: benchflow # --namespace
  service_account: benchflow-runner # --service-account
  target_cluster:
    kubeconfig_secret: target-cluster-kubeconfig # --target-kubeconfig-secret
    kubeconfig: /absolute/path/to/kubeconfig # --target-kubeconfig, local CLI only
  ttl_seconds_after_finished: 3600 # --ttl-seconds-after-finished
  stages:
    download: true # --download / --no-download
    deploy: true # --deploy / --no-deploy
    benchmark: true # --benchmark / --no-benchmark
    collect: true # --collect / --no-collect
    cleanup: true # --cleanup / --no-cleanup
  mlflow:
    experiment: qwen-qwen3-06b-smoke # --mlflow-experiment
    tags:
      owner: perf # --mlflow-tag owner=perf
  execution:
    timeout: 3h # --timeout
  overrides:
    images:
      runtime: ghcr.io/acme/vllm:dev # --runtime-image, string or list for matrix
      scheduler: ghcr.io/acme/router:dev # --scheduler-image, string or list for matrix
    scale:
      replicas: 2 # --replicas, integer or list for matrix
      tensor_parallelism: 4 # --tp, integer or list for matrix
    runtime:
      vllm_args: # --vllm-arg, repeat to append more arguments
        - --max-num-seqs=256
      env: # --env KEY=VALUE, repeat to set multiple variables
        LOG_LEVEL: DEBUG
    llm_d:
      repo_ref: v0.4.1 # --llmd-repo-ref, string or list for matrix
    rhoai:
      enable_auth: false # --rhoai-auth / --no-rhoai-auth
```

Override semantics:

- profile values remain the base
- `images.runtime`, `images.scheduler`, `scale.replicas`, `scale.tensor_parallelism`, and `llm_d.repo_ref` replace the profile value
- `runtime.vllm_args` appends to the profile vLLM args
- `runtime.env` merges by key and override values win on collisions
- benchmark `requirements` can raise the effective deployment runtime settings for a given child `RunPlan`
- today `requirements.min_max_model_len` raises the effective `--max-model-len` for that resolved run when the benchmark needs a larger context window than the deployment default
- list-valued `model.name`, profile refs, and override axes produce a cartesian-product matrix
- matrix children are submitted as independent child executions; Kueue can admit them in parallel when target-cluster GPU capacity allows it

Target-cluster semantics:

- omit `spec.target_cluster` for the normal same-cluster path
- use `spec.target_cluster.kubeconfig_secret` for Tekton executions that must act on a remote target cluster
- use `--target-kubeconfig` only for direct local BenchFlow commands; Tekton `PipelineRun`s cannot see your local filesystem
- create the management-cluster Secret with `bflow target kubeconfig-secret create`
- the control cluster runs Tekton, but target clusters do not need Tekton
- setup, deploy, teardown, and cleanup run from the control cluster against the target kubeconfig
- download, wait-for-endpoint, benchmark, artifact collection, and metrics collection run as plain Kubernetes `Job`s in the target cluster and copy results back when needed

Full `DeploymentProfile` schema:

```yaml
apiVersion: benchflow.io/v1alpha1
kind: DeploymentProfile
metadata:
  name: llm-d-inference-scheduling # no direct CLI override
spec:
  platform: llm-d # llm-d | rhoai | rhaiis
  mode: inference-scheduling # platform-specific mode, no direct CLI override
  runtime:
    image: ghcr.io/llm-d/llm-d-cuda:v0.4.0 # overridden by spec.overrides.images.runtime or --runtime-image
    replicas: 1 # overridden by spec.overrides.scale.replicas or --replicas
    tensor_parallelism: 1 # overridden by spec.overrides.scale.tensor_parallelism or --tp
    vllm_args:
      - --max-model-len=8192 # appended to by spec.overrides.runtime.vllm_args or --vllm-arg
    env:
      VLLM_LOGGING_LEVEL: INFO # merged with spec.overrides.runtime.env or --env
  model_storage:
    pvc_name: models-storage # no CLI override
    cache_dir: /models # no CLI override
    mount_path: /model-cache # no CLI override
  namespace: benchflow # overridden by Experiment spec.namespace or --namespace
  repo_url: https://github.com/llm-d/llm-d.git # no CLI override
  repo_ref: v0.4.0 # overridden by spec.overrides.llm_d.repo_ref or --llmd-repo-ref
  gateway: istio # llm-d only, no CLI override
  endpoint_path: /v1/models # no CLI override
  scheduler_profile: "" # no CLI override
  scheduler_image: "" # overridden by spec.overrides.images.scheduler or --scheduler-image
  options:
    enable_auth: false # rhoai only, overridden by spec.overrides.rhoai.enable_auth or --rhoai-auth
```

Full `BenchmarkProfile` schema:

```yaml
apiVersion: benchflow.io/v1alpha1
kind: BenchmarkProfile
metadata:
  name: smoke # no direct CLI override
spec:
  tool: guidellm # implemented value today
  backend_type: openai_http # no CLI override
  rate_type: concurrent # no CLI override
  rates:
    - 1 # no CLI override today
  data: prompt_tokens=1000,output_tokens=1000 # no CLI override today
  max_seconds: 600 # no CLI override today
  max_requests: null # no CLI override today
  requirements:
    min_max_model_len: 8192 # no CLI override; raises the resolved deployment max-model-len when needed
  env:
    LOG_LEVEL: INFO # no CLI override today
```

Full `MetricsProfile` schema:

```yaml
apiVersion: benchflow.io/v1alpha1
kind: MetricsProfile
metadata:
  name: detailed # no direct CLI override
spec:
  prometheus_url: https://thanos-querier.openshift-monitoring.svc:9091 # no CLI override
  query_step: 15s # no CLI override
  query_timeout: 30s # no CLI override
  verify_tls: false # no CLI override
  queries:
    request_success_total: sum(rate(vllm:request_success_total[5m])) # no CLI override
```

`spec.execution.timeout` defaults to `3h`. BenchFlow uses Tekton implicitly.

Validate it:

```bash
bflow experiment validate experiments/smoke/qwen3-06b-llm-d-smoke.yaml
```

Resolve it into a concrete `RunPlan`:

```bash
bflow experiment resolve experiments/smoke/qwen3-06b-llm-d-smoke.yaml --format json
```

Render the execution manifest without submitting it:

```bash
bflow experiment render-pipelinerun experiments/smoke/qwen3-06b-llm-d-smoke.yaml
```

Submit it:

```bash
bflow experiment run experiments/smoke/qwen3-06b-llm-d-smoke.yaml
```

Follow it later:

```bash
bflow watch <execution-name> --namespace benchflow
```

Inspect logs:

```bash
bflow logs <execution-name>
bflow logs <execution-name> --step benchmark
bflow logs <execution-name> --step benchmark --all-containers
bflow logs <execution-name> --all
```

List running and finished experiments:

```bash
bflow experiment list
```

Cancel one:

```bash
bflow experiment cancel <execution-name>
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

With overrides:

```bash
bflow experiment run \
  --name qwen3-06b \
  --model Qwen/Qwen3-0.6B \
  --deployment-profile llm-d-inference-scheduling \
  --benchmark-profile guidellm-smoke \
  --metrics-profile detailed \
  --namespace benchflow \
  --runtime-image ghcr.io/acme/vllm:dev \
  --scheduler-image ghcr.io/acme/router:dev \
  --replicas 2 \
  --tp 4 \
  --vllm-arg --max-num-seqs=256 \
  --env LOG_LEVEL=DEBUG \
  --llmd-repo-ref v0.4.1
```

Model matrix:

```bash
bflow experiment run \
  --name model-matrix \
  --model Qwen/Qwen3-0.6B \
  --model meta-llama/Llama-3.1-8B \
  --deployment-profile llm-d-inference-scheduling \
  --benchmark-profile guidellm-smoke \
  --metrics-profile detailed
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

Render the execution manifest from it:

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
- each child `RunPlan` becomes one normal child execution
- `bflow experiment run` submits one supervisor execution
- the supervisor runs the child combinations sequentially in the cluster
- each child benchmark still creates its own MLflow run
- if every child combination uses `llm-d` and keeps cleanup enabled, the
  supervisor sets up llm-d once and tears it down once at the end

So this is safe to submit and walk away from:

```bash
bflow experiment run experiments/smoke/qwen3-06b-matrix-smoke.yaml
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

BenchFlow also exposes the lower-level runtime commands that the execution backends use inside
the control image. These are useful for debugging or local step-by-step work.

Download the model:

```bash
bflow model download --run-plan-file runplan.json --models-storage-path /path/to/models
```

Deploy:

```bash
bflow deploy llm-d --run-plan-file runplan.json
```

Set up llm-d explicitly:

```bash
bflow setup llm-d --run-plan-file runplan.json --state-path setup-state.json
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
bflow artifacts collect --run-plan-file runplan.json --execution-name <name> --artifacts-dir ./artifacts
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

Tear down llm-d setup explicitly:

```bash
bflow teardown llm-d --run-plan-file runplan.json --state-path setup-state.json
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

## Comparison Reports

Use `bflow benchmark report` to generate a comparison report from existing
MLflow runs.

You can either pass `--mlflow-tracking-uri` explicitly or let BenchFlow and the
MLflow client read the standard environment variables:

```bash
export MLFLOW_TRACKING_URI=https://mlflow.example.com
export MLFLOW_TRACKING_USERNAME=my-user
export MLFLOW_TRACKING_PASSWORD=my-password
export MLFLOW_TRACKING_INSECURE_TLS=true
```

With those environment variables set, you can omit `--mlflow-tracking-uri`:

```bash
bflow benchmark report \
  --mlflow-run-ids 3f0c1f...,91ab22...,c72de9...
```

Minimal MLflow comparison:

```bash
bflow benchmark report \
  --mlflow-run-ids 3f0c1f...,91ab22...,c72de9... \
  --mlflow-tracking-uri https://mlflow.example.com
```

This path:

- fetches the referenced MLflow runs
- downloads their benchmark artifacts
- validates that the runs can be compared together
- generates one HTML comparison report
- prints the final report path to stdout

Filter to a subset of versions:

```bash
bflow benchmark report \
  --mlflow-run-ids 3f0c1f...,91ab22...,c72de9... \
  --mlflow-tracking-uri https://mlflow.example.com \
  --versions llm-d-v0.4.0,RHOAI-3.3
```

`--versions` filters the compared runs by base version labels. This is useful
when the same MLflow experiment contains multiple product versions and you only
want a specific subset in the final report.

Rename versions in the report:

```bash
bflow benchmark report \
  --mlflow-run-ids 3f0c1f...,91ab22...,c72de9... \
  --mlflow-tracking-uri https://mlflow.example.com \
  --version-override llm-d-v0.4.0=llm-d-0.4 \
  --version-override RHOAI-3.3=rhoai-33
```

`--version-override` is applied after the runs are fetched and filtered. Repeat
it to rename multiple version labels in the generated charts.

Include extra CSV data in the same report:

```bash
bflow benchmark report \
  --mlflow-run-ids 3f0c1f...,91ab22... \
  --mlflow-tracking-uri https://mlflow.example.com \
  --additional-csv ./local-baseline.csv \
  --additional-csv ./historical.csv
```

BenchFlow merges the MLflow runs with the additional CSV inputs and generates a
single combined comparison report.

Write the report under a specific directory:

```bash
bflow benchmark report \
  --mlflow-run-ids 3f0c1f...,91ab22... \
  --mlflow-tracking-uri https://mlflow.example.com \
  --output-dir ./reports
```

Useful notes:

- `--mlflow-run-ids` is the key input for MLflow-backed comparison reports
- `--mlflow-tracking-uri` should point to the MLflow server that owns those run IDs
- if `--mlflow-tracking-uri` is omitted, BenchFlow falls back to `MLFLOW_TRACKING_URI`
- MLflow authentication can come from `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`
- set `MLFLOW_TRACKING_INSECURE_TLS=true` when you need MLflow access without TLS verification
- `--versions` narrows the compared version set
- `--version-override` renames version labels in the final report
- `--additional-csv` lets you mix MLflow runs with local CSV inputs
- the command prints the resulting HTML report path when it succeeds

Typical workflow:

```bash
bflow benchmark report \
  --mlflow-run-ids RUN_ID_1,RUN_ID_2,RUN_ID_3 \
  --mlflow-tracking-uri https://mlflow.example.com \
  --versions llm-d-v0.4.0,RHOAI-3.3 \
  --version-override llm-d-v0.4.0=llm-d-0.4 \
  --output-dir ./reports
```

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

Render the execution manifest before submitting:

```bash
bflow experiment render-pipelinerun my-experiment.yaml
```

Or, for a resolved plan:

```bash
bflow run-plan render-pipelinerun runplan.json
```

If a run is already in the cluster:

```bash
bflow experiment list
bflow watch <execution-name> --namespace benchflow
```

If you need to stop it:

```bash
bflow experiment cancel <execution-name>
```
