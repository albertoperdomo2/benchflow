from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import dataclass
from typing import Any

from .cluster import CommandError, run_command, run_json_command
from .models import ResolvedRunPlan


RHOAI_GATEWAY_CLASS_NAME = "openshift-ai-inference"
RHOAI_GATEWAY_NAME = "openshift-ai-inference"
RHOAI_GATEWAY_NAMESPACE = "openshift-ingress"


@dataclass(frozen=True, slots=True)
class RhoaiGatewayConfiguration:
    namespace: str
    gateway_class_name: str
    labels: dict[str, str]
    listener: dict[str, Any]


def rhoai_release_gateway_name(plan: ResolvedRunPlan) -> str:
    """Return a compact, cluster-unique Gateway name for a BenchFlow release.

    Istio derives its gateway Deployment label from ``<gateway>-<class>``.
    Keep the Gateway name short enough for Kubernetes' 63-character label
    limit with the RHOAI GatewayClass used by BenchFlow.
    """
    identity = f"{plan.deployment.namespace}/{plan.deployment.release_name}"
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return f"benchflow-{suffix}"


def _release_gateway_hostname(plan: ResolvedRunPlan, shared_hostname: str) -> str:
    """Derive a wildcard-compatible external hostname for one release Gateway."""
    _, separator, domain = shared_hostname.partition(".")
    if not separator or not domain:
        raise CommandError(
            "RHOAI shared Gateway listener hostname must include a DNS domain: "
            f"{shared_hostname}"
        )
    return f"{rhoai_release_gateway_name(plan)}.{domain}"


def _https_listener(payload: dict[str, Any]) -> dict[str, Any]:
    listeners = (payload.get("spec") or {}).get("listeners") or []
    for listener in listeners:
        if not isinstance(listener, dict):
            continue
        if str(listener.get("protocol") or "").upper() != "HTTPS":
            continue
        hostname = str(listener.get("hostname") or "").strip()
        tls = listener.get("tls")
        if hostname and isinstance(tls, dict):
            return listener
    raise CommandError(
        "RHOAI shared Gateway must expose an HTTPS listener with a hostname and TLS "
        f"configuration: {RHOAI_GATEWAY_NAMESPACE}/{RHOAI_GATEWAY_NAME}"
    )


def load_rhoai_gateway_configuration(kubectl_cmd: str) -> RhoaiGatewayConfiguration:
    """Read the bootstrap-managed Gateway used as the TLS and domain source."""
    payload = run_json_command(
        [
            kubectl_cmd,
            "get",
            "gateway",
            RHOAI_GATEWAY_NAME,
            "-n",
            RHOAI_GATEWAY_NAMESPACE,
            "-o",
            "json",
        ]
    )
    spec = payload.get("spec") or {}
    metadata = payload.get("metadata") or {}
    labels = metadata.get("labels") or {}
    gateway_class_name = str(
        spec.get("gatewayClassName") or RHOAI_GATEWAY_CLASS_NAME
    ).strip()
    return RhoaiGatewayConfiguration(
        namespace=RHOAI_GATEWAY_NAMESPACE,
        gateway_class_name=gateway_class_name,
        labels={
            key: value
            for key, value in labels.items()
            if isinstance(key, str) and isinstance(value, str) and key == "istio.io/rev"
        },
        listener=copy.deepcopy(_https_listener(payload)),
    )


def render_rhoai_release_gateway(
    plan: ResolvedRunPlan,
    config: RhoaiGatewayConfiguration,
) -> dict[str, Any]:
    """Render a release-scoped Gateway with an isolated HTTPS listener."""
    listener = copy.deepcopy(config.listener)
    listener["name"] = "benchflow"
    listener["hostname"] = _release_gateway_hostname(plan, str(listener["hostname"]))
    return {
        "apiVersion": "gateway.networking.k8s.io/v1",
        "kind": "Gateway",
        "metadata": {
            "name": rhoai_release_gateway_name(plan),
            "namespace": config.namespace,
            "labels": {
                "app.kubernetes.io/name": "benchflow",
                "app.kubernetes.io/managed-by": "benchflow",
                "benchflow.io/release": plan.deployment.release_name,
                "benchflow.io/source-namespace": plan.deployment.namespace,
                **config.labels,
            },
        },
        "spec": {
            "gatewayClassName": config.gateway_class_name,
            "listeners": [listener],
        },
    }


def rhoai_release_gateway_reference(plan: ResolvedRunPlan) -> dict[str, str]:
    return {
        "name": rhoai_release_gateway_name(plan),
        "namespace": RHOAI_GATEWAY_NAMESPACE,
    }


def wait_for_rhoai_gateway_ready(
    kubectl_cmd: str,
    *,
    namespace: str,
    name: str,
    timeout_seconds: int,
) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        result = run_command(
            [kubectl_cmd, "get", "gateway", name, "-n", namespace, "-o", "json"],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            payload = run_json_command(
                [kubectl_cmd, "get", "gateway", name, "-n", namespace, "-o", "json"]
            )
            conditions = (payload.get("status") or {}).get("conditions") or []
            accepted = any(
                item.get("type") == "Accepted" and item.get("status") == "True"
                for item in conditions
                if isinstance(item, dict)
            )
            programmed = any(
                item.get("type") == "Programmed" and item.get("status") == "True"
                for item in conditions
                if isinstance(item, dict)
            )
            if accepted and programmed:
                return
        time.sleep(5)
    raise CommandError(
        f"timed out waiting for Gateway {namespace}/{name} to become ready"
    )
