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

- The Gateway name is deterministic from the deployment namespace and release
  name, with a hash to avoid collisions between namespaces.
- It has one independently identified HTTPS listener. The listener preserves
  the bootstrap Gateway hostname and TLS reference because clusters such as
  Diadochos use a certificate valid only for the data-science Gateway's
  internal service names, not a public wildcard domain.
- Its GatewayClass, Istio revision label, listener options, and TLS reference
  are copied from the bootstrap-managed `openshift-ai-inference` Gateway.
- BenchFlow validates that the referenced TLS certificate covers the generated
  hostname before applying the Gateway.
- The `LLMInferenceService` explicitly uses
  `spec.router.gateway.refs[0]` for this Gateway. It never relies on an empty
  `router.gateway` default.

The isolation boundary is the release-scoped `Gateway` object and its listener,
not the external hostname. This follows the documented RHOAI workaround for
INFERENG-6962: move each route to a separate Gateway. Reusing the existing
hostname and certificate avoids introducing new cluster DNS or TLS
infrastructure.

This applies to all BenchFlow RHOAI `LLMInferenceService` modes, including
default, approximate-prefix-cache, and precise-prefix-cache. It does not apply
to the raw `InferenceService` deployment path, which does not use the
LLMInferenceService router/EPP path.

## Lifecycle

1. Bootstrap creates or reconciles the shared `openshift-ai-inference` Gateway.
   BenchFlow keeps it for manual or pre-existing workloads, but its own
   LLMInferenceServices do not attach to it.
2. Deployment reads that Gateway as the trusted source for the class, listener,
   Istio revision label, and TLS configuration.
3. Deployment applies the release Gateway and waits for `Accepted=True` and
   `Programmed=True`.
4. Deployment applies the LLMInferenceService with its explicit Gateway ref.
   The rendered Gateway and LLMInferenceService are both captured as artifacts.
5. Cleanup deletes the LLMInferenceService first, then its release Gateway.
   It also deletes the Gateway if the service is already absent, which covers
   partial failed runs. Cross-namespace owner references are invalid, so
   deterministic naming and explicit cleanup are required.

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
- the two HTTPRoutes attach to different Gateway objects and listeners;
- Istio retains an `ExtProcPerRoute` override for each route;
- EndpointPicker logs show per-request activity and prefix-cache scoring; and
- cleanup removes only the corresponding release Gateway.

The RHOAI documentation supports explicit `spec.router.gateway.refs`.
INFERENG-6962 documents the failure caused by multiple HTTPRoutes sharing one
wildcard listener and the resulting EndpointPicker bypass.
