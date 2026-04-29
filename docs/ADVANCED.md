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
- [Local Metrics Viewer](#local-metrics-viewer)
- [Comparison Reports](#comparison-reports)
- [RHOAI Profiling](#rhoai-profiling)
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
- accelerator prerequisites are installed unless `--no-install-accelerator-prerequisites` is passed
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

If the target cluster is already provisioned and you do not want BenchFlow to
touch operator versions there, bootstrap it like this:

```bash
bflow bootstrap \
  --target-kubeconfig ~/.kube/target-cluster \
  --cluster-name target-cluster \
  --no-install-accelerator-prerequisites
```

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

Important behavior:
- for `llm-d` and `RHOAI`, BenchFlow adds a setup key to each queued execution
- once a target cluster has an admitted setup key, BenchFlow admits only matching-key workloads until that admitted wave finishes
- same-key workloads can keep using spare GPUs in parallel; different-key workloads wait in queue
- shared platform switching happens only in `prepare`, and the installed platform stays in place until another setup key is requested or you run explicit teardown
- matrix parent cancellation is best-effort once child executions have already been submitted; queued or running children may need to be cancelled individually

Remaining limitations:
- legacy target clusters without BenchFlow platform state are adopted heuristically; the first mutating run after upgrading BenchFlow may reset and reinstall shared platform prerequisites
- BenchFlow does not reconcile manual or out-of-band platform changes made outside its own setup state

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
  target:
    base_url: https://my-existing-endpoint.example.com # --target-url
    path: /v1/models # --target-path; defaults to /v1/models
    metrics_release_name: my-existing-release # --target-metrics-release-name; enables metrics collection for an existing endpoint
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
    verify_completions: true # --verify-completions / --no-verify-completions
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
      resources:
        requests:
          cpu: "16" # --runtime-cpu-request
        limits:
          cpu: "32" # --runtime-cpu-limit
      node_selector:
        kubernetes.io/hostname: worker-0
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: kubernetes.io/hostname
                    operator: In
                    values: [worker-0]
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
    llm_d:
      repo_ref: v0.4.1 # --llmd-repo-ref, string or list for matrix
    rhoai:
      enable_auth: false # --rhoai-auth / --no-rhoai-auth
    benchmark:
      env:
        GUIDELLM__LOGGING__CONSOLE_LOG_LEVEL: DEBUG # profile-owned benchmark env can be overridden per experiment
```

Override semantics:

- profile values remain the base
- `images.runtime`, `images.scheduler`, `scale.replicas`, `scale.tensor_parallelism`, and `llm_d.repo_ref` replace the profile value
- `runtime.vllm_args` appends to the profile vLLM args
- `runtime.env` merges by key and override values win on collisions
- `runtime.resources.requests` and `runtime.resources.limits` merge by resource name and override values win; CPU request and limit can also be set with `--runtime-cpu-request` and `--runtime-cpu-limit`
- `runtime.node_selector`, `runtime.affinity`, and `runtime.tolerations` replace the profile value when set in `spec.overrides.runtime`
- `runtime.image_pull_secrets` is profile-owned and currently rendered for RHOAI runtime pods
- `benchmark.env` merges by key and override values win on collisions
- benchmark `requirements` can raise the effective deployment runtime settings for a given child `RunPlan`
- today `requirements.min_max_model_len` raises the effective `--max-model-len` for that resolved run when the benchmark needs a larger context window than the deployment default
- list-valued `model.name`, profile refs, and override axes produce a cartesian-product matrix
- matrix children are submitted as independent child executions
- `rhoai` and `llm-d` child executions can be admitted in parallel when target-cluster GPU capacity allows it

Target-cluster semantics:

- omit `spec.target_cluster` for the normal same-cluster path
- use `spec.target_cluster.kubeconfig_secret` for Tekton executions that must act on a remote target cluster
- use `--target-kubeconfig` only for direct local BenchFlow commands; Tekton `PipelineRun`s cannot see your local filesystem
- create the management-cluster Secret with `bflow target kubeconfig-secret create`
- the control cluster runs Tekton, but target clusters do not need Tekton

Existing endpoint path:

- set `spec.target.base_url` to benchmark an already deployed endpoint
- BenchFlow automatically disables `download`, `deploy`, and `cleanup`
- BenchFlow automatically disables `collect` unless `spec.target.metrics_release_name` is set
- when `spec.target.metrics_release_name` is set, BenchFlow treats the collect phase as metrics-focused for the existing endpoint path
- in that mode, BenchFlow still preserves benchmark outputs and execution logs, but it does not try to sweep workload logs or manifests from an arbitrary existing deployment
- BenchFlow resolves the target as a static URL and skips deployment discovery entirely

Example:

```yaml
apiVersion: benchflow.io/v1alpha1
kind: Experiment
metadata:
  name: existing-endpoint-benchmark
spec:
  model:
    name: Qwen/Qwen3-0.6B
  deployment_profile: rhoai-distributed-default
  benchmark_profile: guidellm-smoke
  metrics_profile: detailed
  target:
    base_url: https://my-existing-endpoint.example.com
    metrics_release_name: qwen-existing-release
```
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
    resources:
      requests:
        cpu: "16" # overridden by spec.overrides.runtime.resources.requests.cpu or --runtime-cpu-request
      limits:
        cpu: "32" # overridden by spec.overrides.runtime.resources.limits.cpu or --runtime-cpu-limit
    node_selector:
      nvidia.com/gpu.product: NVIDIA-H200 # profile-owned node selector for runtime pods
    affinity:
      nodeAffinity:
        requiredDuringSchedulingIgnoredDuringExecution:
          nodeSelectorTerms:
            - matchExpressions:
                - key: kubernetes.io/hostname
                  operator: In
                  values: [worker-0.example]
    tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
    image_pull_secrets:
      - name: rh-ee-aperdomo-pull-secret # rhoai only, rendered as runtime pod imagePullSecrets
  model_storage:
    pvc_name: models-storage # no CLI override
    cache_dir: /models # no CLI override
    mount_path: /model-cache # no CLI override
  namespace: benchflow # overridden by Experiment spec.namespace or --namespace
  repo_url: https://github.com/llm-d/llm-d.git # no CLI override
  repo_ref: v0.4.0 # overridden by spec.overrides.llm_d.repo_ref or --llmd-repo-ref
  platform_version: RHOAI-3.4.0-ea.2 # rhoai only, used for setup key and requested operator CSV
  platform_channel: beta # rhoai only, defaults to fast-3.x when omitted
  gateway: istio # llm-d only, no CLI override
  endpoint_path: /v1/models # no CLI override
  scheduler_profile: "" # no CLI override
  scheduler_image: "" # overridden by spec.overrides.images.scheduler or --scheduler-image
  options:
    enable_auth: false # rhoai only, overridden by spec.overrides.rhoai.enable_auth or --rhoai-auth
    epp_config: "" # rhoai/llm-d only, optional EndpointPickerConfig rendered with Jinja
```

RHOAI deployment profiles can provide a custom EPP configuration with
`spec.options.epp_config`. BenchFlow renders this YAML with the same Jinja context
used for the `LLMInferenceService` manifest, validates that it renders an
`EndpointPickerConfig`, enables the custom scheduler path, and places it directly
in the scheduler `--config-text`.

Upstream `llm-d` deployment profiles can use the same `spec.options.epp_config`
field. BenchFlow renders and validates the `EndpointPickerConfig`, then patches
the guide scheduler values by setting `inferenceExtension.pluginsConfigFile` to
`benchflow-epp-config.yaml` and placing the rendered YAML in
`inferenceExtension.pluginsCustomConfig`. The upstream guide still creates the
EPP ConfigMap through Helm.

This is intended for deployment-profile variants, not experiment overrides:

```yaml
apiVersion: benchflow.io/v1alpha1
kind: DeploymentProfile
metadata:
  name: rhoai-approximate-prefix-cache-queue-heavy
spec:
  platform: rhoai
  mode: approximate-prefix-cache
  platform_version: RHOAI-3.3
  runtime:
    replicas: 4
    tensor_parallelism: 2
    vllm_args:
      - --max-model-len=8192
      - --gpu-memory-utilization=0.92
      - --trust-remote-code
      - --disable-log-requests
      - --enable-prefix-caching
  options:
    enable_auth: false
    epp_config: |
      apiVersion: inference.networking.x-k8s.io/v1alpha1
      kind: EndpointPickerConfig
      plugins:
      - type: queue-scorer
      - type: kv-cache-utilization-scorer
      - type: prefix-cache-scorer
      schedulingProfiles:
      - name: default
        plugins:
        - pluginRef: queue-scorer
          weight: 5.0
        - pluginRef: kv-cache-utilization-scorer
          weight: 2.0
        - pluginRef: prefix-cache-scorer
          weight: 3.0
```

For precise prefix-cache variants, the custom config can reference the derived
tokenizer path:

```yaml
modelTokenizerMap:
  base: "{{ precise_prefix_cache_tokenizer_model_path }}"
```

Full `BenchmarkProfile` schema:

Only the section that matches `spec.tool` is required for a profile. The other
tool section is shown here to document the supported contract.

```yaml
apiVersion: benchflow.io/v1alpha1
kind: BenchmarkProfile
metadata:
  name: smoke # no direct CLI override
spec:
  tool: guidellm # supported values: guidellm, aiperf
  guidellm:
    backend_type: openai_http # no CLI override today
    rate_type: concurrent # no CLI override
    rates:
      - 1 # overridden by spec.overrides.benchmark.rates
    request_type: "" # optional; if empty BenchFlow defers to GuideLLM's internal default
    data: prompt_tokens=1000,output_tokens=1000 # no CLI override today
    max_seconds: 600 # overridden by spec.overrides.benchmark.max_seconds
    max_requests: null # overridden by spec.overrides.benchmark.max_requests
  aiperf:
    dataset_url: "" # required when tool: aiperf
    dataset_name: "" # optional local/cache filename; defaults to the URL basename
    dataset_type: mooncake_trace # required when tool: aiperf
    endpoint_type: chat # required when tool: aiperf
    endpoint_path: /v1/chat/completions
    tokenizer: "" # defaults to spec.model.name when empty
    streaming: true
    fixed_schedule: true
    fixed_schedule_auto_offset: true
    synthesis_max_isl: 131072
    fixed_schedule_end_offset: null
    dataset_cap: null # optional; trim JSONL input to the first N non-empty records
    export_level: "" # optional; for example records
    export_http_trace: false
    max_seconds: 7200 # remote benchmark job timeout hint
  requirements:
    min_max_model_len: 8192 # no CLI override; raises the resolved deployment max-model-len when needed
  env:
    LOG_LEVEL: INFO # no CLI override today
```

Safe GuideLLM benchmark overrides can be applied from the `Experiment` without
changing the benchmark profile identity:

```yaml
spec:
  benchmark_profile: guidellm-multi-turn
  overrides:
    benchmark:
      rates: [128]
      max_seconds: 180
      max_requests: 500
      request_type: text_completions
```

`data` is intentionally not overrideable. It is treated as part of what defines
the benchmark profile itself.

For AIPerf profiles, `dataset_name` is optional. When omitted, BenchFlow derives
the cached file name from `dataset_url`, for example `toolagent_trace.jsonl`.
Use `dataset_name` only when the URL does not have a useful basename or when a
stable local/report label is needed.

Use `aiperf.dataset_cap` to trim large JSONL datasets. BenchFlow downloads the
full dataset once, writes a cached `-cap<N>` JSONL containing the first `N`
non-empty records, and passes that trimmed file to `aiperf profile`.

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
  --runtime-cpu-request 16 \
  --runtime-cpu-limit 32 \
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
- `rhoai` and `llm-d` child executions are submitted together and Kueue can admit them in parallel
- each child benchmark still creates its own MLflow run
- if every child combination uses `llm-d` and keeps cleanup enabled, the
  supervisor sets up llm-d once and tears it down once at the end

Name-length note for `llm-d` with Istio:

- the llm-d guide derives the Gateway name as `infra-<release>-inference-gateway`
- Istio appends `-istio` when creating backing resources and uses that value as a Kubernetes label
- Kubernetes label values are limited to 63 characters
- keep llm-d experiment names short enough that `infra-<release>-inference-gateway-istio` is no more than 63 characters
- if the Gateway is present but the backing Istio Deployment or Service is missing, check for label-length validation errors in the Istio controller events

So this is safe to submit and walk away from:

```bash
bflow experiment run experiments/smoke/qwen3-06b-matrix-smoke.yaml
```

The shipped matrix smoke example intentionally produces two child runs.

### Rerunning failed or successful executions

BenchFlow can rerun a previous execution directly from its recorded `RunPlan`
without going back to the original experiment YAML.

Single execution:

```bash
bflow experiment run <execution-name> --status failed
```

Supported status selectors:

- `failed`
- `succeeded`
- `all`

Examples:

```bash
bflow experiment run qwen3-06b-abc123 --status failed
bflow experiment run qwen3-06b-abc123 --status all
```

For matrix executions, BenchFlow uses the matrix supervisor execution name and
reruns only the child executions that match the requested status:

```bash
bflow experiment run qwen3-06b-matrix-abc123 --status failed
```

Current limitation:

- matrix rerun-by-status only works for matrix runs created after BenchFlow
  started labeling child executions with their parent matrix execution
  identity
- older matrix runs do not have that linkage, so BenchFlow will fail
  explicitly instead of guessing which child executions belonged to the
  matrix
- `--status` only applies when the positional argument is an execution name;
  it must not be combined with an experiment file path

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

If you do not set `spec.mlflow.version`, BenchFlow also derives a default
version label:

- `llm-d` uses `llm-d-<repo_ref>`
- `rhoai` tries the live `rhods-operator` subscription version and normalizes it
  to labels like `RHOAI-3.3` or `RHOAI-3.4-EA1`
- if that live RHOAI lookup fails, BenchFlow falls back to its pinned RHOAI
  series label
- `rhaiis` uses `RHAIIS-<runtime image tag>` when the deployment runtime image
  has an explicit tag

The benchmark runtime adds:

- `vllm_version`
- `guidellm_version`

If `accelerator` is not already present in MLflow tags, BenchFlow also tries to
derive it from the cluster by inspecting the nodes backing the serving pods and
normalizing labels like `nvidia.com/gpu.product` to values such as `H200`.

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

## Local Metrics Viewer

Use `bflow metrics serve` to inspect the stored Prometheus metrics for one
MLflow run locally in a Plotly-based dashboard that mirrors the live Grafana
layout closely enough for interactive analysis.

BenchFlow:

- downloads the run's `metrics/` artifact tree from MLflow
- caches downloaded run metrics under `/tmp/benchflow-metrics-viewer`
- reuses the stored Prometheus query results instead of querying Prometheus again
- renders a local single-run dashboard with interactive legends and zoom
- serves it on `http://127.0.0.1:8765/`

Minimal usage:

```bash
export MLFLOW_TRACKING_URI=https://mlflow.example.com
export MLFLOW_TRACKING_USERNAME=my-user
export MLFLOW_TRACKING_PASSWORD=my-password
export MLFLOW_TRACKING_INSECURE_TLS=true

bflow metrics serve --mlflow-run-id 3f0c1f...
```

Compare multiple runs:

```bash
bflow metrics serve \
  --mlflow-run-id 3f0c1f... \
  --mlflow-run-id 91ab22... \
  --mlflow-run-id c72de9...
```

In compare mode, BenchFlow aligns the traces on relative benchmark time instead
of wall-clock time so separate executions can be overlaid meaningfully.

Notes:

- the port is intentionally fixed to `8765`
- the command does not open a browser automatically
- press `Ctrl-C` to stop the local server
- repeat `--mlflow-run-id` to compare multiple runs in one viewer
- cached runs are reused from `/tmp/benchflow-metrics-viewer`
- the run must already contain BenchFlow `metrics/` artifacts in MLflow

## Benchmark Reports

BenchFlow has two report commands:

- `bflow benchmark plot run`
- `bflow benchmark plot comparison`

`bflow benchmark report` still exists as a compatibility alias for
`bflow benchmark plot comparison`.

### Post-Run Reports

Use `bflow benchmark plot run` when you already have a collected BenchFlow
artifact directory and want the richer single-run HTML report.

```bash
bflow benchmark plot run \
  --artifacts-dir ./artifacts
```

Write the report under a specific directory:

```bash
bflow benchmark plot run \
  --artifacts-dir ./artifacts \
  --output-dir ./reports
```

Write the report to an exact file path:

```bash
bflow benchmark plot run \
  --artifacts-dir ./artifacts \
  --output-file ./reports/full-run.html
```

Control the diagnostics grid density:

```bash
bflow benchmark plot run \
  --artifacts-dir ./artifacts \
  --columns 2
```

This command expects a collected artifact tree with benchmark outputs and
metrics. The report derives total accelerator count from `tp * replicas` in the
collected `metadata.json`. In the normal BenchFlow workflow the same post-run
report is generated automatically after collection, before artifacts are
uploaded to MLflow.

### Comparison Reports

Use `bflow benchmark plot comparison` to generate the comparison report from
existing MLflow runs or benchmark JSON inputs.

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
bflow benchmark plot comparison \
  --mlflow-run-ids 3f0c1f...,91ab22...,c72de9...
```

Minimal MLflow comparison:

```bash
bflow benchmark plot comparison \
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
bflow benchmark plot comparison \
  --mlflow-run-ids 3f0c1f...,91ab22...,c72de9... \
  --mlflow-tracking-uri https://mlflow.example.com \
  --versions llm-d-v0.4.0,RHOAI-3.3
```

`--versions` filters the compared runs by base version labels. This is useful
when the same MLflow experiment contains multiple product versions and you only
want a specific subset in the final report.

Rename versions in the report:

```bash
bflow benchmark plot comparison \
  --mlflow-run-ids 3f0c1f...,91ab22...,c72de9... \
  --mlflow-tracking-uri https://mlflow.example.com \
  --version-override llm-d-v0.4.0=llm-d-0.4 \
  --version-override RHOAI-3.3=rhoai-33
```

`--version-override` is applied after the runs are fetched and filtered. Repeat
it to rename multiple version labels in the generated charts.

Include extra CSV data in the same report:

```bash
bflow benchmark plot comparison \
  --mlflow-run-ids 3f0c1f...,91ab22... \
  --mlflow-tracking-uri https://mlflow.example.com \
  --additional-csv ./local-baseline.csv \
  --additional-csv ./historical.csv
```

BenchFlow merges the MLflow runs with the additional CSV inputs and generates a
single combined comparison report.

Write the report under a specific directory:

```bash
bflow benchmark plot comparison \
  --mlflow-run-ids 3f0c1f...,91ab22... \
  --mlflow-tracking-uri https://mlflow.example.com \
  --output-dir ./reports
```

Write the report to an exact file path:

```bash
bflow benchmark plot comparison \
  --mlflow-run-ids 3f0c1f...,91ab22... \
  --mlflow-tracking-uri https://mlflow.example.com \
  --output-file ./reports/rhoai-vs-llmd.html
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
- `--output-file` overrides `--output-dir` when both are set
- the command prints the resulting HTML report path when it succeeds

Typical workflow:

```bash
bflow benchmark plot comparison \
  --mlflow-run-ids RUN_ID_1,RUN_ID_2,RUN_ID_3 \
  --mlflow-tracking-uri https://mlflow.example.com \
  --versions llm-d-v0.4.0,RHOAI-3.3 \
  --version-override llm-d-v0.4.0=llm-d-0.4 \
  --output-dir ./reports
```

## RHOAI Profiling

BenchFlow includes a narrow RHOAI-only worker profiling path inspired by:

- `mnmehta/vllm-profiler`: https://github.com/mnmehta/vllm-profiler

BenchFlow does not install that repository directly. Instead, it injects a
small BenchFlow-owned `sitecustomize.py` payload into the RHOAI runtime worker
pod and hooks `vllm.v1.worker.gpu_worker.Worker.execute_model`.

Enable it in the `Experiment`:

```yaml
spec:
  execution:
    profiling:
      enabled: true
      call_ranges: "100-150"
```

`call_ranges` are per-process `Worker.execute_model` invocation windows, not
wall-clock windows and not GuideLLM concurrency phases. BenchFlow starts
profiling when a worker process reaches the beginning of a configured range and
exports results after that process completes the final call in the range.

BenchFlow fixes the rest of the profiler contract intentionally:

- activities are always `CPU,CUDA`
- Chrome trace export is always enabled
- a text summary table is also exported for each captured range
- artifacts are written under `/tmp/benchflow-profiler` inside the worker pod
- collected profiler artifacts are uploaded with the rest of the execution
  artifacts under `profiling/`
- every RHOAI worker pod receives the same profiler injection when profiling is
  enabled

This path works best when you run a benchmark with a single intended
concurrency. If you want to inspect a specific concurrency, create or reuse a
benchmark profile that runs only that concurrency and then use `call_ranges` to
capture a steady-state window within that run.

Current limitations:

- RHOAI only
- runtime worker profiling only; scheduler/EPP pods are not profiled
- all worker pods are profiled with the same call windows; BenchFlow does not
  currently let you target only one replica
- call windows apply independently in each worker process, including multi-rank
  worker pods
- range selection is based on local `execute_model` call counts, not time,
  benchmark stage, or global request ordering
- no compare or merge workflow for multiple trace captures yet

If profiling is enabled on a non-RHOAI deployment, BenchFlow fails fast during
RunPlan resolution.

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
