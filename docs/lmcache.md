# LMCache CPU offloading

LMCache is a platform service selected by a deployment profile. Platform setup
installs one MP server on each matching NVIDIA GPU node; platform teardown
removes it. Model cleanup leaves the service installed. Changing from LMCache
to a native-vLLM profile removes the servers without reinstalling RHOAI or
llm-d. Changing the underlying platform still uses the normal platform reset.

This is the initial implementation, verified locally with tests and manifest
rendering. GPU cache readback, image compatibility, OpenShift admission, and
performance have **not** been qualified on a live cluster. Do not treat a Ready
DaemonSet or the smoke workload as evidence that offloading improved performance.

## Initial scope and prerequisites

- RHOAI `distributed-default` and llm-d `inference-scheduling`.
- Qwen3-0.6B for functional bring-up; Qwen3-32B for subsequent comparison.
- One model replica, single-node tensor parallelism, pipeline parallelism 1.
- CPU tier only. Disk, CephFS, hybrid attention, P/D, and cache-aware routing
  integration are outside this slice.
- Whole NVIDIA GPUs without MIG or time slicing. Setup detects NVIDIA
  RuntimeClass access or GPU Operator CDI/NRI. With CDI/NRI it adds the
  BenchFlow namespace to the toolkit's management-device allowlist, preserving
  other entries, and waits for the toolkit rollout. This path requires
  NVIDIA Container Toolkit >=1.20.0. An older toolkit is reported as an
  unsupported GPU prerequisite before a namespace grant is changed.
- GPU nodes must have `nvidia.com/gpu.present=true`. The server inherits the
  runtime's node selector, affinity, tolerations, and image pull secrets. Use
  a deployment profile's node selector to limit the platform's GPU pool.
- `bflow bootstrap` enables OpenShift user workload monitoring. The service has a
  ServiceMonitor for `/metrics` on port 8080, scraped every 15 seconds.

The server uses CUDA IPC and needs access to the same physical GPUs as vLLM.
Its service uses `internalTrafficPolicy: Local` so the client always reaches
the server on its own node. Both ends enable isolated IPC; no host network,
host IPC, privileged container, or host `/dev/shm` mount is used. The server
does not reserve a second `nvidia.com/gpu` allocation. Its CUDA contexts still
consume GPU memory, which must be included when comparing resource costs.

The transport and HTTP frontend are ClusterIP-only. A NetworkPolicy permits
clients from the BenchFlow namespace and HTTP scraping from the OpenShift
monitoring namespaces. Treat that namespace as the trust boundary for this
unauthenticated MP transport; use a cluster network that enforces NetworkPolicy.

## Images and processes

Supply a compatible vLLM+LMCache image through the deployment profile's
`runtime.image` or the existing `--runtime-image` option. Use an existing image
or prepare one with the explicit local utility below. There is no default
serving image and no LMCache image build or validation workflow in CI.

| Role | Image | Process |
| --- | --- | --- |
| Orchestration and benchmarking | BenchFlow control image | `bflow` |
| Model serving | Supplied vLLM+LMCache image | vLLM with the LMCache MP connector |
| Node-local cache service | The same supplied image | `lmcache server` |

These are separate pods. The cache server shares its Python/Torch/CUDA/LMCache
versions with the model runtime because both use the same image. LMCache is
not added to the BenchFlow control image. The Python library is needed in the
model runtime because the connector executes inside vLLM; the cache service
is deployed separately and offloads KV into its own CPU memory.

The supplied image must include the external
`lmcache.integration.vllm.lmcache_mp_connector.LMCacheMPConnector`, CUDA Python
bindings for isolated IPC, and the `lmcache server` CLI with the MP flags used
by the platform manifests. Platform setup checks connector import and actual
CUDA access on the GPU nodes. The server's startup and health checks verify
the CLI and HTTP API. Versions are logged for provenance, without enforcing a
specific vLLM or LMCache version pair.

BenchFlow owns the platform service manifests, connector configuration,
readiness, benchmarking, artifacts, and teardown. Platform setup applies the
DaemonSet, Service, NetworkPolicy, and ServiceMonitor directly; it installs
neither Helm charts nor LMCache-specific CRDs. It uses the cluster's existing
GPU Operator and monitoring APIs. Image builds never happen during setup or a
benchmark. Import and readiness checks are prerequisites, not GPU readback or
performance qualification.

## Prepare an image locally

`bflow image build` builds, checks, and publishes one vLLM+LMCache image using
Podman. The recipe and Python helpers live in
`src/benchflow/assets/images/vllm-lmcache/` and ship with the CLI wheel.

Create `lmcache-image.yaml` with the three required values. These version pins
are an example to qualify, not a claim of validated GPU compatibility. Replace
the Quay organization with your own repository:

```yaml
base_image: vllm/vllm-openai:v0.32.0
lmcache_version: "0.5.7"
destination: quay.io/your-org/vllm-lmcache:vllm-0.32.0-lmcache-0.5.7
```

An explicit base digest is preferable for reproducibility; version tags are
also accepted. Floating `latest` tags are rejected. The LMCache version selects
the upstream `v<version>` Git release tag; its resolved commit is recorded.
The builder currently targets Linux amd64 with NVIDIA CUDA 12.1–13.x, using an
Ubuntu vLLM base with `uv` and configured NVIDIA CUDA apt repositories. It
requires vLLM >=0.20.0 to load the external MP connector. Arbitrary base images
and version combinations are not guaranteed to work.

Use a running amd64 Linux Podman builder with ample disk and memory for the
base image and CUDA compilation. An ARM Podman machine, including the default
on Apple Silicon, is rejected before login or build. The utility does not
create or reconfigure Podman machines. A physical GPU is not required to build.
The compiled GPU targets cover Ampere, Ada, and Hopper, plus Blackwell when
the base uses CUDA >=12.8. Older GPU architectures are outside this recipe.

Quay authentication uses existing Podman credentials, or `QUAY_USERNAME` and
`QUAY_PASSWORD` environment variables (including robot accounts). With the
variables set, BenchFlow logs in automatically using password stdin. Do not
put credentials in the configuration file. Stored credentials are checked
before building; destination push permissions are verified by the push itself.
Private base images must also be accessible through Podman's registry auth.

```bash
bflow image build lmcache-image.yaml --dry-run
bflow image build lmcache-image.yaml
```

The dry run validates configuration and prints the recipe and commands without
calling Podman, authenticating, building, or pushing. The real operation:

1. Installs the matching CUDA compiler and headers in the derived image.
2. Checks out the selected LMCache release, installs its build requirements,
   and compiles it against the existing PyTorch with build isolation disabled.
3. Preserves vLLM, PyTorch, torchvision/torchaudio, Triton, and installed NVIDIA
   Python package versions, including base-image `UV_OVERRIDE` constraints.
   Other dependencies can change to satisfy LMCache. New dependency conflicts
   fail the build; inherited conflicts are recorded for provenance.
4. Checks native imports, CUDA Python bindings, the external connector, and
   required server flags without GPU access. A failed build or check prevents push.
5. Pushes the exact built image ID to Quay, then writes `lmcache-image.image.json`
   alongside the configuration with the published digest, versions, CUDA build
   targets, and resolved LMCache commit. Existing records are replaced after a
   successful push. A failed push is reported as an error.

Use the record's `image` value (`quay.io/...@sha256:...`) as `runtime.image` or
`--runtime-image`. Both serving and cache pods then use that same image. Registry
credentials for a private image must be available through the deployment
profile's image pull secrets. Build-time checks cannot prove GPU KV readback;
run the cluster smoke experiment below before using the image for comparisons.

## Automated smoke tests

After normal BenchFlow bootstrap, set `VLLM_LMCACHE_IMAGE` to the image you
provide and run either experiment:

```bash
bflow experiment run experiments/smoke/qwen3-06b-rhoai-lmcache-smoke.yaml \
  --runtime-image "$VLLM_LMCACHE_IMAGE"
bflow experiment run experiments/smoke/qwen3-06b-llm-d-lmcache-smoke.yaml \
  --runtime-image "$VLLM_LMCACHE_IMAGE"
```

Each uses Qwen3-0.6B, TP1, one replica, a 128 MiB GPU KV cache, a 1 GiB CPU
cache, and 24 seeded requests drawing from two 768-token prefixes. This pool
exceeds the GPU KV capacity while each request fits the 1024-token model
limit. BenchFlow handles platform setup, GPU management access, deployment,
the workload, metrics/log collection, MLflow upload, and model cleanup.

The smoke profile enables `options.lmcache.verify_readback`. BenchFlow samples
server counters before and after the workload, rejects pod/container restarts,
and requires **new stored chunks and new successfully loaded chunks**. A
completed transfer counter alone cannot pass. The benchmark fails if the gate
fails and preserves `lmcache-readback.json` with both snapshots and deltas.
This is a functional test; it does not establish a performance improvement.

## Profiles and workflow

The six deployment profiles are `rhoai-kv-comparison-{hbm,native-cpu,lmcache-cpu}`
and `llm-d-kv-comparison-{hbm,native-cpu,lmcache-cpu}`. Their scale, GPU utilization,
model length, and model-pod memory settings match. The native CPU arm uses the
stock `OffloadingConnector` with 64 GiB total CPU cache. The LMCache arm uses
64 GiB per node-local server, with a requested minimum chunk size of 256 tokens
(the connector may negotiate a larger size). All arms request 16-token GPU
blocks, and native offload explicitly uses 256-token chunks. Model-pod memory limits include
the native CPU tier; LMCache additionally reserves 72 GiB and 2 CPU cores on
**every matching GPU node**, including nodes not serving this model. Record
both the active node's costs and the total platform footprint.

The LMCache profile contains:

```yaml
options:
  lmcache:
    cpu_gib: 64
    chunk_size: 256
```

Use profiles to change these settings. BenchFlow owns `--kv-transfer-config`;
conflicting connectors, weight offloading, sleep mode, and expandable-segment
CUDA allocation are rejected.

Resolve a functional run without changing the cluster:

```bash
bflow experiment resolve experiments/smoke/qwen3-06b-rhoai-lmcache-smoke.yaml \
  --runtime-image "$VLLM_LMCACHE_IMAGE" --format json > lmcache-runplan.json
```

Normal Tekton execution owns setup, deployment, benchmarking, collection,
MLflow upload, and model cleanup:

```bash
bflow run-plan run lmcache-runplan.json
```

For explicit platform administration, use the existing lifecycle commands:

```bash
bflow setup rhoai --run-plan-file lmcache-runplan.json --state-path lmcache-setup.json
bflow teardown rhoai --run-plan-file lmcache-runplan.json --state-path lmcache-setup.json
```

Use `llm-d` in those commands and the corresponding deployment profile for llm-d.
Do not issue manual setup/teardown while executions are active. Explicit teardown
removes the whole recorded platform, including LMCache; it is broader than
model cleanup. A deployed LMCache model requires matching recorded platform
state, including when normal setup is skipped.

After the functional GPU gate, submit the three arms with the same image and
a benchmark profile designed to exceed GPU KV capacity:

```bash
bflow experiment run --model Qwen/Qwen3-0.6B \
  --deployment-profile rhoai-kv-comparison-hbm \
  --deployment-profile rhoai-kv-comparison-native-cpu \
  --deployment-profile rhoai-kv-comparison-lmcache-cpu \
  --benchmark-profile guidellm-smoke --metrics-profile kv-cache-comparison \
  --runtime-image "$VLLM_LMCACHE_IMAGE"
```

This command is a workflow smoke test. Replace `guidellm-smoke` with a calibrated
reuse workload before measuring performance. Qwen3-32B also needs appropriately
sized GPU memory and tensor parallelism in its deployment profile.

## Lifecycle, isolation, and artifacts

The platform compatibility key includes the resolved LMCache configuration,
serving image, model/runtime settings, and scoped execution release. Kueue's
existing setup-key waves serialize LMCache executions against one another
and against native runs. Matrix setup is not hoisted for LMCache. A new release
deletes the old servers and waits for their pods to exit before starting fresh
CPU cache state. Repeating setup for the same release/configuration reuses its
warm cache. Rerun the execution to get a new release and cold servers.

The platform state records an unready transition before changing cache
services. Failed setup can be retried or torn down; it cannot advertise the
previous native key as ready while cache servers remain running. Per-run
cleanup removes the model and preserves shared services. Native setup removes
any remaining BenchFlow LMCache service so its resource overhead does not
silently contaminate the baseline.

`render-deployment` writes `platform-lmcache.yaml` alongside the serving assets
for review. Collection includes server logs, DaemonSet/Service/NetworkPolicy/
ServiceMonitor/pod manifests, and a raw post-run `/metrics` snapshot under
`metrics/lmcache/`. The `kv-cache-comparison` metrics profile captures native
offload metrics, LMCache L1 reads/writes/evictions, successful chunk loads,
transfer counts and bytes, prompt-token sources, queueing,
latency, and separate server/model CPU and memory series at 15-second resolution.
Missing scrape series remain missing; an empty MP endpoint before traffic is
not proof that the service failed.

Completed transfer counts include misses and empty stores; use successful
chunk loads and transferred bytes to establish readback. The raw endpoint
snapshot preserves the full metric inventory for subsequent investigation.

## GPU acceptance gate

Before ranking performance, validate a pinned image on the target cluster:

1. Setup succeeds on the intended node pool; server pods are admitted without
   reserving GPUs, expose CUDA, and report the expected software versions.
2. The model starts using the external MP connector and reaches the same-node
   server. Record actual GPU KV capacity and negotiated LMCache chunk size.
3. A controlled repeated-prefix workload exceeds GPU KV capacity, stores KV
   externally, then retrieves it into the GPU. Verify nonzero store **and**
   readback telemetry and correct outputs. Include a no-reuse control.
4. Repeat from cold servers and verify setup reuse versus new-execution reset.
   Switch back to native and verify all cache pods have exited. Exercise final
   platform teardown and retry after an intentionally failed image preflight.
5. Compare HBM-only, native CPU, and LMCache CPU with matched effective GPU KV
   budgets, 64 GiB CPU tiers, frozen requests/arrivals, drained warmup, multiple
   repetitions, and counterbalanced order. Include server CUDA context memory,
   CPU, total memory, request errors, and latency tails in the cost comparison.

References: [MP deployment](https://docs.lmcache.ai/mp/deployment.html),
[configuration](https://docs.lmcache.ai/mp/configuration.html), and
[MP metrics](https://docs.lmcache.ai/mp/observability/metrics.html).
