from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from .cluster import CommandError, run_command, run_json_command
from .models import ResolvedRunPlan


RHOAI_GATEWAY_NAME = "openshift-ai-inference"
RHOAI_GATEWAY_NAMESPACE = "openshift-ingress"
_LISTENER_PATCH_RETRIES = 5


@dataclass(frozen=True, slots=True)
class RhoaiGatewayConfiguration:
    namespace: str
    listener: dict[str, Any]


def rhoai_release_gateway_listener_name(plan: ResolvedRunPlan) -> str:
    """Return a stable listener name that is unique across BenchFlow namespaces."""
    identity = f"{plan.deployment.namespace}/{plan.deployment.release_name}"
    return f"benchflow-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]}"


def _gateway_payload(kubectl_cmd: str) -> dict[str, Any]:
    return run_json_command(
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


def _https_listener(payload: dict[str, Any]) -> dict[str, Any]:
    listeners = (payload.get("spec") or {}).get("listeners") or []
    for listener in listeners:
        if not isinstance(listener, dict):
            continue
        if str(listener.get("protocol") or "").upper() != "HTTPS":
            continue
        if str(listener.get("hostname") or "").strip() and isinstance(
            listener.get("tls"), dict
        ):
            return listener
    raise CommandError(
        "RHOAI shared Gateway must expose an HTTPS listener with a hostname and TLS "
        f"configuration: {RHOAI_GATEWAY_NAMESPACE}/{RHOAI_GATEWAY_NAME}"
    )


def load_rhoai_gateway_configuration(kubectl_cmd: str) -> RhoaiGatewayConfiguration:
    """Read the bootstrap-managed Gateway's working HTTPS listener."""
    return RhoaiGatewayConfiguration(
        namespace=RHOAI_GATEWAY_NAMESPACE,
        listener=copy.deepcopy(_https_listener(_gateway_payload(kubectl_cmd))),
    )


def render_rhoai_release_gateway_listener(
    plan: ResolvedRunPlan,
    config: RhoaiGatewayConfiguration,
) -> dict[str, Any]:
    """Create an isolated listener on the existing, programmed Gateway.

    OpenShift's GatewayClass creates a LoadBalancer only for the bootstrap
    Gateway. A listener on that Gateway shares the working LoadBalancer while
    sectionName keeps the generated HTTPRoute out of other listener sections.
    """
    listener = copy.deepcopy(config.listener)
    listener["name"] = rhoai_release_gateway_listener_name(plan)
    # A listener with the bootstrap hostname would conflict on port 443. The
    # hostname-less listener is a distinct section on the existing Gateway.
    listener.pop("hostname", None)
    return listener


def render_rhoai_release_gateway_listener_patch(
    plan: ResolvedRunPlan,
    config: RhoaiGatewayConfiguration,
) -> list[dict[str, Any]]:
    return [
        {
            "op": "add",
            "path": "/spec/listeners/-",
            "value": render_rhoai_release_gateway_listener(plan, config),
        }
    ]


def rhoai_release_gateway_reference(plan: ResolvedRunPlan) -> dict[str, str]:
    return {
        "name": RHOAI_GATEWAY_NAME,
        "namespace": RHOAI_GATEWAY_NAMESPACE,
        "sectionName": rhoai_release_gateway_listener_name(plan),
    }


def _listener_index(payload: dict[str, Any], listener_name: str) -> int | None:
    listeners = (payload.get("spec") or {}).get("listeners") or []
    for index, listener in enumerate(listeners):
        if isinstance(listener, dict) and listener.get("name") == listener_name:
            return index
    return None


def _listener_ready(payload: dict[str, Any], listener_name: str) -> bool:
    listeners = (payload.get("status") or {}).get("listeners") or []
    for listener in listeners:
        if not isinstance(listener, dict) or listener.get("name") != listener_name:
            continue
        conditions = listener.get("conditions") or []
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
        return accepted and programmed
    return False


def wait_for_rhoai_release_gateway_listener(
    kubectl_cmd: str,
    *,
    plan: ResolvedRunPlan,
    timeout_seconds: int,
) -> None:
    listener_name = rhoai_release_gateway_listener_name(plan)
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _listener_ready(_gateway_payload(kubectl_cmd), listener_name):
            return
        time.sleep(5)
    raise CommandError(
        f"timed out waiting for RHOAI Gateway listener {listener_name} on "
        f"{RHOAI_GATEWAY_NAMESPACE}/{RHOAI_GATEWAY_NAME} to become ready"
    )


def ensure_rhoai_release_gateway_listener(
    plan: ResolvedRunPlan,
    *,
    kubectl_cmd: str,
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    """Atomically append a release listener without replacing concurrent listeners."""
    config = load_rhoai_gateway_configuration(kubectl_cmd)
    patch = render_rhoai_release_gateway_listener_patch(plan, config)
    listener_name = rhoai_release_gateway_listener_name(plan)
    for attempt in range(_LISTENER_PATCH_RETRIES):
        if _listener_index(_gateway_payload(kubectl_cmd), listener_name) is not None:
            wait_for_rhoai_release_gateway_listener(
                kubectl_cmd, plan=plan, timeout_seconds=timeout_seconds
            )
            return patch
        result = run_command(
            [
                kubectl_cmd,
                "patch",
                "gateway",
                RHOAI_GATEWAY_NAME,
                "-n",
                RHOAI_GATEWAY_NAMESPACE,
                "--type=json",
                "-p",
                json.dumps(patch, separators=(",", ":")),
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            wait_for_rhoai_release_gateway_listener(
                kubectl_cmd, plan=plan, timeout_seconds=timeout_seconds
            )
            return patch
        if attempt + 1 == _LISTENER_PATCH_RETRIES:
            raise CommandError(
                "failed to add RHOAI release Gateway listener "
                f"{listener_name}: {(result.stderr or result.stdout).strip()}"
            )
        time.sleep(1)
    raise AssertionError("unreachable")


def remove_rhoai_release_gateway_listener(
    plan: ResolvedRunPlan, *, kubectl_cmd: str
) -> None:
    """Remove only this release's listener, retrying concurrent list updates."""
    listener_name = rhoai_release_gateway_listener_name(plan)
    for attempt in range(_LISTENER_PATCH_RETRIES):
        index = _listener_index(_gateway_payload(kubectl_cmd), listener_name)
        if index is None:
            return
        result = run_command(
            [
                kubectl_cmd,
                "patch",
                "gateway",
                RHOAI_GATEWAY_NAME,
                "-n",
                RHOAI_GATEWAY_NAMESPACE,
                "--type=json",
                "-p",
                json.dumps(
                    [
                        {
                            "op": "test",
                            "path": f"/spec/listeners/{index}/name",
                            "value": listener_name,
                        },
                        {"op": "remove", "path": f"/spec/listeners/{index}"},
                    ],
                    separators=(",", ":"),
                ),
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return
        if attempt + 1 == _LISTENER_PATCH_RETRIES:
            raise CommandError(
                "failed to remove RHOAI release Gateway listener "
                f"{listener_name}: {(result.stderr or result.stdout).strip()}"
            )
        time.sleep(1)
