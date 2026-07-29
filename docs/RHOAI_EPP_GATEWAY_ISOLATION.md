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
Gateway in `openshift-ingress`:

- The Gateway name is a compact deterministic hash of the deployment namespace
  and release name, avoiding collisions and keeping Istio-derived Deployment
  labels below Kubernetes' 63-character label limit.
- Its GatewayClass, Istio revision label, HTTPS listener, hostname, TLS
  reference, and allowed-routes policy are copied from the bootstrap-managed
  `openshift-ai-inference` Gateway.
- The `LLMInferenceService` explicitly uses
  `spec.router.gateway.refs[0]` for this Gateway. It never relies on an empty
  `router.gateway` default.

The isolation boundary is the release-scoped Gateway object and its listener,
not the external hostname. This follows the documented RHOAI workaround for
INFERENG-6962 while preserving the existing endpoint, DNS, and TLS path.

Diadochos initially made this look unsupported because the first generated
Gateway names were too long. Istio derives a Deployment label from
`<gateway>-<class>`; the derived label exceeded 63 characters, preventing the
controller from creating its Deployment and Service. The resulting Gateway
status was `Programmed=False` with `ServiceNotFound` and `AddressNotUsable`.
Compact release Gateway names correct that controller failure.

This applies to all BenchFlow RHOAI `LLMInferenceService` modes, including
default, approximate-prefix-cache, and precise-prefix-cache. It does not apply
to the raw `InferenceService` deployment path, which does not use the
LLMInferenceService router/EPP path.

## Lifecycle

1. Bootstrap creates or reconciles the shared `openshift-ai-inference` Gateway.
2. Deployment reads that Gateway as the trusted TLS and listener source,
   applies its release Gateway, and waits for `Accepted=True` and
   `Programmed=True`.
3. Deployment applies the LLMInferenceService with an explicit Gateway ref.
   The rendered Gateway and LLMInferenceService are captured as artifacts.
4. Cleanup deletes the LLMInferenceService first, then its release Gateway. It
   also deletes the Gateway if the service is already absent, covering partial
   failed runs.

If a release already exists but points at the shared Gateway, BenchFlow fails
instead of silently reusing it. Clean up and redeploy that release with the
current image.

## Preconditions

- The bootstrap-managed `openshift-ai-inference` Gateway must have an HTTPS
  listener with a hostname and a local TLS Secret reference.
- The BenchFlow runner must be allowed to get, create, and delete Gateways in
  `openshift-ingress`.

BenchFlow fails clearly when any precondition is missing. It does not fall back
to the shared Gateway because that would make precise-prefix-cache results
silently invalid.

## Verification

Before accepting this path on a RHOAI/Istio combination, launch two concurrent
precise-prefix-cache releases and verify:

- each generated HTTPRoute has only its release Gateway parent reference;
- the two HTTPRoutes attach to distinct Gateway objects and listeners;
- Istio retains an `ExtProcPerRoute` override for each route;
- EndpointPicker logs show per-request activity and prefix-cache scoring; and
- cleanup removes only the corresponding release Gateway.

The RHOAI documentation supports explicit `spec.router.gateway.refs`.
INFERENG-6962 documents the failure caused by multiple HTTPRoutes sharing one
wildcard listener and the resulting EndpointPicker bypass.
