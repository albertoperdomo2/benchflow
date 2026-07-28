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

- The listener name is deterministic from the deployment namespace and release
  name, with a hash to avoid collisions between namespaces.
- It copies the working HTTPS listener's TLS and route policy but omits its
  hostname. Gateway API rejects two listeners with the same hostname, port,
  and protocol, while a hostname-less listener is a valid distinct section.
- The `LLMInferenceService` explicitly sets
  `spec.router.gateway.refs[0].sectionName` to that listener. It never relies
  on an empty `router.gateway` default.
- Listener creation uses an atomic JSON Patch append. Cleanup performs a JSON
  Patch name test before removing the indexed listener, so concurrent releases
  cannot replace or remove each other's sections.
- Bootstrap reuses an existing Gateway rather than applying its single-listener
  manifest again, which preserves active BenchFlow listener sections.

Diadochos cannot use a separate Gateway object for this purpose. Its OpenShift
GatewayClass accepts the object but does not provision the required load
balancer Service, leaving it `Programmed=False` with `ServiceNotFound` and
`AddressNotUsable`. A listener section on the existing programmed Gateway was
accepted and programmed immediately, while continuing to use its working load
balancer, DNS, and TLS configuration.

The isolation boundary is the release-scoped listener section, not the external
hostname. It introduces no new DNS or TLS infrastructure.

This applies to all BenchFlow RHOAI `LLMInferenceService` modes, including
default, approximate-prefix-cache, and precise-prefix-cache. It does not apply
to the raw `InferenceService` deployment path, which does not use the
LLMInferenceService router/EPP path.

## Lifecycle

1. Bootstrap creates the shared `openshift-ai-inference` Gateway when absent;
   otherwise it validates and reuses it without replacing active listeners.
2. Deployment reads its working HTTPS listener, appends the release listener,
   and waits for that listener to report `Accepted=True` and `Programmed=True`.
3. Deployment applies the LLMInferenceService with an explicit Gateway ref and
   `sectionName`. The listener patch and LLMInferenceService are captured as
   artifacts.
4. Cleanup deletes the LLMInferenceService first, then removes only its named
   listener. It also removes that listener if the service is already absent,
   covering partial failed runs.

If a release already exists but points at the shared Gateway, BenchFlow fails
instead of silently reusing it. Clean up and redeploy that release with the
current image.

## Preconditions

- The bootstrap-managed `openshift-ai-inference` Gateway must have an HTTPS
  listener with a hostname and a local TLS Secret reference.
- The BenchFlow runner must be allowed to get and patch the shared Gateway in
  `openshift-ingress`.

BenchFlow fails clearly when any precondition is missing. It does not fall back
to the shared `https` listener because that would make precise-prefix-cache
results silently invalid.

## Verification

Before accepting this path on a RHOAI/Istio combination, launch two concurrent
precise-prefix-cache releases and verify:

- each generated HTTPRoute has only its release listener parent reference;
- the two HTTPRoutes attach to distinct listener sections;
- Istio retains an `ExtProcPerRoute` override for each route;
- EndpointPicker logs show per-request activity and prefix-cache scoring; and
- cleanup removes only the corresponding release listener.

The RHOAI documentation supports explicit `spec.router.gateway.refs`.
INFERENG-6962 documents the failure caused by multiple HTTPRoutes sharing one
wildcard listener and the resulting EndpointPicker bypass.
