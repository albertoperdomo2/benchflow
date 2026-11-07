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

## Serving image

CI builds one default serving image for **all three** comparison arms. The same image runs
the LMCache server and vLLM, avoiding independently selected Torch/CUDA/Python
ABIs. The initial development contract is vLLM `0.32.0`, LMCache `0.5.7`, and
`cuda-python`, with the external `lmcache.integration.vllm.lmcache_mp_connector`
module available. Server startup checks package versions, connector import,
and CUDA access, and logs the software versions and visible GPU count.

`build/Containerfile.lmcache` builds LMCache from its immutable v0.5.7 source
commit against `vllm/vllm-openai:v0.32.0`, preserving the existing Torch/CUDA ABI
and core serving dependencies. The image build imports the external connector
and CUDA extension, checks package dependencies, and verifies the server CLI.
Dependency checks honor the base image's explicit NCCL override and run before
compilation as well as after LMCache installation.
The `Build LMCache Serving Image` workflow publishes
`ghcr.io/albertoperdomo2/benchflow:lmcache-vllm0.32.0-lmcache0.5.7` on main.
Every shipped profile uses that image by default; no local image build,
package installation, server deployment, or connector configuration is required.

PR builds publish `lmcache-pr-<number>-<sha>` tags and their digest in the workflow
summary. Use `--runtime-image` to test that PR revision with its BenchFlow
control image. The version/import checks are prerequisites, not GPU readback
or performance qualification. Floating `latest`/`nightly` images are rejected.

## Automated smoke tests

After normal BenchFlow bootstrap, run either experiment:

```bash
bflow experiment run experiments/smoke/qwen3-06b-rhoai-lmcache-smoke.yaml
bflow experiment run experiments/smoke/qwen3-06b-llm-d-lmcache-smoke.yaml
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
  --format json > lmcache-runplan.json
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
  --benchmark-profile guidellm-smoke --metrics-profile kv-cache-comparison
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
