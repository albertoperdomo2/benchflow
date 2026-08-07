# RHOAI LLMInferenceService Gateway Isolation

## Status

Implemented for BenchFlow RHOAI `LLMInferenceService` deployments. This is the
only BenchFlow path; there is no matrix-only or precise-profile-only switch.

## Problem

On affected RHOAI and Istio versions, attaching multiple HTTPRoutes to the same
wildcard Gateway listener causes Istio to aggregate them into one generated
VirtualService. The per-route `ExtProcPerRoute` override is then absent, so the
EndpointPicker is bypassed and traffic falls back to round-robin routing.

This affects precise-prefix-cache deployments and can be triggered by any
additional route, including a concurrent BenchFlow release, token endpoint,
echo service, or test route. A single experiment does not make this safe:
another user can submit an independent experiment at the same time.

## Design

Every BenchFlow RHOAI `LLMInferenceService` release creates and owns one
listener section on the bootstrap-managed `openshift-ai-inference` Gateway in
`openshift-ingress`:

- The listener name is a deterministic hash of the deployment namespace and
  release name, avoiding collisions between concurrent releases.
- Its HTTPS and TLS configuration is copied from the bootstrap listener, but
  its hostname is omitted. The external hostname belongs to the bootstrap
  Gateway and resolves to its single load balancer.
- The `LLMInferenceService` explicitly uses
  `spec.router.gateway.refs[0]`, including that listener's `sectionName`. It
  never relies on an empty `router.gateway` default.

The isolation boundary is the release-scoped listener section, not a separate
Gateway object or the external hostname. Creating multiple Gateway objects
with the bootstrap hostname is invalid on clusters where that hostname has one
DNS target: only one Gateway's load balancer can receive external traffic,
leaving the other accepted HTTPRoutes externally unreachable.

This applies to all BenchFlow RHOAI `LLMInferenceService` modes, including
default, approximate-prefix-cache, and precise-prefix-cache. It does not apply
to the raw `InferenceService` deployment path, which does not use the
LLMInferenceService router/EPP path.

## Lifecycle

1. Bootstrap creates or reconciles the shared `openshift-ai-inference` Gateway.
2. Deployment reads its trusted HTTPS listener source, atomically appends a
   release listener section, and waits for that listener's `Accepted=True` and
   `Programmed=True` conditions.
3. Deployment applies the LLMInferenceService with an explicit Gateway ref.
   The listener patch and LLMInferenceService are captured as artifacts.
4. Cleanup deletes the LLMInferenceService first, then removes only its
   listener section. It also removes that listener if the service is already
   absent, covering partial failed runs.

If a release already exists but points at the shared Gateway, BenchFlow fails
instead of silently reusing it. Clean up and redeploy that release with the
current image.

## Preconditions

- The bootstrap-managed `openshift-ai-inference` Gateway must have an HTTPS
  listener with a hostname and a local TLS Secret reference.
- The BenchFlow runner must be allowed to get and patch Gateways in
  `openshift-ingress`.

BenchFlow fails clearly when any precondition is missing. It does not fall back
to the shared Gateway because that would make precise-prefix-cache results
silently invalid.

## Verification

Before accepting this path on a RHOAI/Istio combination, launch two concurrent
precise-prefix-cache releases and verify:

- each generated HTTPRoute has only its release listener `sectionName` parent
  reference;
- the two HTTPRoutes attach to distinct listener sections on the bootstrap
  Gateway;
- Istio retains an `ExtProcPerRoute` override for each route;
- EndpointPicker logs show per-request activity and prefix-cache scoring; and
- cleanup removes only the corresponding release listener section.

The RHOAI documentation supports explicit `spec.router.gateway.refs`.
INFERENG-6962 documents the failure caused by multiple HTTPRoutes sharing one
wildcard listener and the resulting EndpointPicker bypass.
