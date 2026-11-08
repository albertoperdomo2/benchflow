# LMCache CPU offloading

LMCache is a platform service selected by the deployment profile. The single
shipped smoke uses RHOAI: platform setup installs the cache, the workload proves
KV readback, collection saves metrics and logs, and final cleanup removes the
model and this execution's cache service. The shared RHOAI installation remains.

This is the initial implementation, verified locally with tests and manifest
rendering. GPU cache readback, image compatibility, OpenShift admission, and
performance have **not** been qualified on a live cluster. Do not treat a Ready
DaemonSet or the smoke workload as evidence that offloading improved performance.

## Initial scope and prerequisites

- The smoke targets RHOAI `3.6.0-ea.2` (`beta` channel), `distributed-default`.
  The implementation also handles llm-d `inference-scheduling`.
- The smoke uses Qwen/Qwen3-0.6B for functional validation.
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

The checked-in build configuration is
[`config/images/lmcache-image.yaml`](../config/images/lmcache-image.yaml).
Edit its Quay destination and, when needed, its base image and LMCache version.
The checked-in version pins are a starting point to qualify, not a claim of
validated GPU compatibility. Credentials stay outside the YAML.

An explicit base digest is preferable for reproducibility; version tags are
also accepted. Floating `latest` tags are rejected. The LMCache version selects
the upstream `v<version>` Git release tag; its resolved commit is recorded.
The builder currently targets Linux amd64 with NVIDIA CUDA 12.1–13.x, using an
Ubuntu vLLM base with `uv` and configured NVIDIA CUDA apt repositories. It
requires vLLM >=0.20.0 to load the external MP connector. Arbitrary base images
and version combinations are not guaranteed to work.

Use a running Linux Podman server with ample disk and memory for the base image
and CUDA compilation. Native amd64 runs directly. On ARM, including Apple
Silicon, BenchFlow checks that the configured amd64 base image can execute
Python and import PyTorch through Rosetta/QEMU before starting the build. Both
compilation and image validation explicitly target `linux/amd64`. Emulated
compilation can take longer. The utility does not create or reconfigure Podman
machines. A physical GPU is not required to build.
The compiled GPU targets cover Ampere, Ada, and Hopper, plus Blackwell when
the base uses CUDA >=12.8. Older GPU architectures are outside this recipe.

Quay authentication uses existing Podman credentials, or `QUAY_USERNAME` and
`QUAY_PASSWORD` environment variables (including robot accounts). With the
variables set, BenchFlow logs in automatically using password stdin. Do not
put credentials in the configuration file. Stored credentials are checked
before building; destination push permissions are verified by the push itself.
Private base images must also be accessible through Podman's registry auth.

```bash
bflow image build config/images/lmcache-image.yaml --dry-run
bflow image build config/images/lmcache-image.yaml
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
5. Pushes the exact built image ID to Quay, then writes `config/images/lmcache-image.image.json`
   alongside the configuration with the published digest, versions, CUDA build
   targets, and resolved LMCache commit. Existing records are replaced after a
   successful push. A failed push is reported as an error.

Use the record's `image` value (`quay.io/...@sha256:...`) as `runtime.image` or
`--runtime-image`. Both serving and cache pods then use that same image. Registry
credentials for a private image must be available through the deployment
profile's image pull secrets. Build-time checks cannot prove GPU KV readback;
run the cluster smoke experiment below to qualify GPU cache transfers.

## Run the smoke

After normal `bflow bootstrap`, use a compatible prepared image or the digest
from the local build's JSON record:

```bash
bflow experiment run experiments/smoke/qwen3-06b-rhoai-lmcache-smoke.yaml \
  --runtime-image "$VLLM_LMCACHE_IMAGE"
```

This is the only LMCache smoke experiment. It uses three profiles:

- `rhoai-lmcache-smoke`: one Qwen3-0.6B replica, TP1, a 128 MiB GPU KV cache,
  a 1 GiB LMCache CPU tier, and the external MP connector.
- `guidellm-lmcache-smoke`: 24 seeded requests using two 768-token prefixes,
  sized to exceed GPU KV capacity while each request fits the model limit.
- `lmcache-smoke`: cache stores, successful loads, reads, evictions, transfer
  bytes/counts, prompt-token sources, GPU KV occupancy, and model/server CPU
  and memory series at 15-second resolution.

BenchFlow handles setup, deployment, the workload, collection, MLflow upload,
and final cleanup. The readback gate requires **new stored chunks and new
successfully loaded chunks** during the workload, rejects server restarts, and
preserves `lmcache-readback.json` with both snapshots and deltas. Completed
transfer counters alone cannot pass. This is a functional smoke, not evidence
of a performance improvement.

For LMCache MP metrics, collection includes two scrape intervals after the
benchmark ends (30 seconds with the smoke profile), rounded up to the next
query step. BenchFlow waits for that window before querying, so counters created
during a short workload reach Prometheus before cleanup. The metrics summary
and archive index record `collection_end_time` and each metric's `query_end_time`.
Benchmark timestamps and other metrics retain the original benchmark window.

## Cleanup and artifacts

Tekton's final cleanup runs after normal collection or failure finalization,
including when setup or the benchmark fails. It removes the model, then the
LMCache DaemonSet, Service, NetworkPolicy, ServiceMonitor, and any GPU management
namespace grant created by BenchFlow. Cache deletion waits for pods to exit.
Disabling the cleanup stage explicitly preserves resources.

Cache cleanup uses the platform lock and verifies the recorded setup key still
belongs to this execution. It cannot delete a later execution's cache. Failed
cache deletion or GPU reconciliation leaves an unready state for retry. Once
cleanup succeeds, the recorded key returns to the base platform; RHOAI is not
uninstalled. Explicit platform teardown remains available through the normal
administration commands and removes the full recorded platform.

Setup keys include the image, model/runtime settings, cache configuration, and
execution release. Existing Kueue waves serialize incompatible configurations.
Setup is not hoisted across LMCache matrix runs. Repeated setup for the same
execution can reuse its cache until final cleanup.

Collection preserves server logs and resource manifests plus a full raw
`/metrics` snapshot under `metrics/lmcache/`, including custom metrics beyond
those selected in the smoke profile. Collection completes before cache removal.
Missing Prometheus series remain missing. A Ready server alone does not prove
KV offloading: qualify the image and OpenShift admission on the target GPU
cluster, require store/readback evidence, and verify final resource removal.

References: [MP deployment](https://docs.lmcache.ai/mp/deployment.html),
[configuration](https://docs.lmcache.ai/mp/configuration.html), and
[MP metrics](https://docs.lmcache.ai/mp/observability/metrics.html).
