from __future__ import annotations

import json
import socket
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from .cluster import CommandError, run_command
from .contracts import ResolvedRunPlan

SETUP_KEY_ANNOTATION = "benchflow.io/setup-key"
PLATFORM_STATE_CONFIGMAP_NAME = "benchflow-platform-state"
PLATFORM_STATE_DATA_KEY = "state.json"
PLATFORM_PREPARE_LEASE_NAME = "benchflow-platform-prepare"
DEFAULT_LOCK_TIMEOUT_SECONDS = 1800
DEFAULT_LOCK_STALE_SECONDS = 1800


def _now_rfc3339() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _parse_rfc3339(value: str) -> datetime | None:
    cleaned = str(value).strip()
    if not cleaned:
        return None
    try:
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def base_setup_key_for_plan(plan: ResolvedRunPlan) -> str:
    platform = str(plan.deployment.platform or "").strip()
    if platform == "llm-d":
        return f"llm-d:{plan.deployment.repo_ref}:{plan.deployment.gateway}"
    if platform == "rhoai":
        version = str(plan.deployment.platform_version or "").strip() or "unknown"
        return f"rhoai:{version}"
    return f"{platform}:{plan.deployment.mode}"


def setup_key_for_plan(plan: ResolvedRunPlan) -> str:
    from .lmcache import lmcache_identity

    base = base_setup_key_for_plan(plan)
    cache = lmcache_identity(plan)
    return f"{base}|lmcache:{cache}" if cache else base


def setup_key_from_annotations(annotations: dict[str, str] | None) -> str:
    if not annotations:
        return ""
    return str(annotations.get(SETUP_KEY_ANNOTATION) or "").strip()


def _empty_cluster_platform_state() -> dict[str, Any]:
    return {
        "apiVersion": "benchflow.io/v1alpha1",
        "kind": "ClusterPlatformState",
        "installed_key": "",
        "setup_state": {},
    }


def load_cluster_platform_state(kubectl_cmd: str, namespace: str) -> dict[str, Any]:
    result = run_command(
        [
            kubectl_cmd,
            "get",
            "configmap",
            PLATFORM_STATE_CONFIGMAP_NAME,
            "-n",
            namespace,
            "-o",
            "json",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return _empty_cluster_platform_state()

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return _empty_cluster_platform_state()

    raw_state = str(
        (payload.get("data", {}) or {}).get(PLATFORM_STATE_DATA_KEY) or ""
    ).strip()
    if not raw_state:
        return _empty_cluster_platform_state()

    try:
        state = json.loads(raw_state)
    except json.JSONDecodeError:
        return _empty_cluster_platform_state()
    if not isinstance(state, dict):
        return _empty_cluster_platform_state()
    return {
        **_empty_cluster_platform_state(),
        **state,
        "setup_state": dict(state.get("setup_state") or {}),
    }


def persist_cluster_platform_state(
    kubectl_cmd: str,
    namespace: str,
    state: dict[str, Any],
) -> None:
    payload = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": PLATFORM_STATE_CONFIGMAP_NAME,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/name": "benchflow",
                "benchflow.io/managed-by": "benchflow",
            },
        },
        "data": {
            PLATFORM_STATE_DATA_KEY: json.dumps(
                {
                    **_empty_cluster_platform_state(),
                    **state,
                    "setup_state": dict(state.get("setup_state") or {}),
                },
                indent=2,
                sort_keys=True,
            )
        },
    }
    run_command(
        [kubectl_cmd, "apply", "-f", "-"],
        input_text=json.dumps(payload, separators=(",", ":"), sort_keys=True),
    )


def clear_cluster_platform_state(kubectl_cmd: str, namespace: str) -> None:
    run_command(
        [
            kubectl_cmd,
            "delete",
            "configmap",
            PLATFORM_STATE_CONFIGMAP_NAME,
            "-n",
            namespace,
            "--ignore-not-found=true",
        ],
        check=False,
    )


def _lease_document(
    *,
    namespace: str,
    holder_identity: str,
    lease_duration_seconds: int,
    resource_version: str = "",
) -> dict[str, Any]:
    now = _now_rfc3339()
    metadata: dict[str, Any] = {
        "name": PLATFORM_PREPARE_LEASE_NAME,
        "namespace": namespace,
        "labels": {
            "app.kubernetes.io/name": "benchflow",
            "benchflow.io/managed-by": "benchflow",
        },
    }
    if resource_version:
        metadata["resourceVersion"] = resource_version
    return {
        "apiVersion": "coordination.k8s.io/v1",
        "kind": "Lease",
        "metadata": metadata,
        "spec": {
            "holderIdentity": holder_identity,
            "leaseDurationSeconds": int(lease_duration_seconds),
            "acquireTime": now,
            "renewTime": now,
        },
    }


def _lease_expired(payload: dict[str, Any], default_stale_seconds: int) -> bool:
    spec = payload.get("spec", {}) or {}
    renew_time = _parse_rfc3339(
        str(spec.get("renewTime") or spec.get("acquireTime") or "")
    )
    if renew_time is None:
        return True
    try:
        lease_duration = int(spec.get("leaseDurationSeconds") or default_stale_seconds)
    except (TypeError, ValueError):
        lease_duration = default_stale_seconds
    elapsed = (
        datetime.now(timezone.utc) - renew_time.astimezone(timezone.utc)
    ).total_seconds()
    return elapsed >= max(1, lease_duration)


def _command_failure_details(result: Any) -> str:
    stderr = str(getattr(result, "stderr", "") or "").strip()
    stdout = str(getattr(result, "stdout", "") or "").strip()
    return stderr or stdout or "command failed"


@contextmanager
def platform_prepare_lock(
    kubectl_cmd: str,
    namespace: str,
    *,
    holder_identity: str = "",
    timeout_seconds: int = DEFAULT_LOCK_TIMEOUT_SECONDS,
    stale_seconds: int = DEFAULT_LOCK_STALE_SECONDS,
):
    holder = holder_identity.strip() or (
        f"{socket.gethostname()}-{int(time.time() * 1000)}"
    )
    deadline = time.time() + max(1, int(timeout_seconds))

    while True:
        result = run_command(
            [
                kubectl_cmd,
                "get",
                "lease",
                PLATFORM_PREPARE_LEASE_NAME,
                "-n",
                namespace,
                "-o",
                "json",
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            created = run_command(
                [kubectl_cmd, "create", "-f", "-"],
                input_text=json.dumps(
                    _lease_document(
                        namespace=namespace,
                        holder_identity=holder,
                        lease_duration_seconds=stale_seconds,
                    ),
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                capture_output=True,
                check=False,
            )
            if created.returncode == 0:
                break
            details = _command_failure_details(created)
            if "AlreadyExists" not in details:
                raise CommandError(
                    "failed to create BenchFlow platform prepare lease "
                    f"in namespace {namespace}: {details}"
                )
        else:
            payload = json.loads(result.stdout or "{}")
            spec = payload.get("spec", {}) or {}
            current_holder = str(spec.get("holderIdentity") or "").strip()
            if current_holder in {"", holder} or _lease_expired(payload, stale_seconds):
                replaced = run_command(
                    [kubectl_cmd, "replace", "-f", "-"],
                    input_text=json.dumps(
                        _lease_document(
                            namespace=namespace,
                            holder_identity=holder,
                            lease_duration_seconds=stale_seconds,
                            resource_version=str(
                                (payload.get("metadata", {}) or {}).get(
                                    "resourceVersion"
                                )
                                or ""
                            ),
                        ),
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    capture_output=True,
                    check=False,
                )
                if replaced.returncode == 0:
                    break
                details = _command_failure_details(replaced)
                if not any(
                    marker in details
                    for marker in (
                        "NotFound",
                        "Conflict",
                        "the object has been modified",
                    )
                ):
                    raise CommandError(
                        "failed to replace BenchFlow platform prepare lease "
                        f"in namespace {namespace}: {details}"
                    )

        if time.time() >= deadline:
            raise CommandError(
                f"timed out waiting for BenchFlow platform prepare lock in namespace {namespace}"
            )
        time.sleep(2)

    try:
        yield
    finally:
        current = run_command(
            [
                kubectl_cmd,
                "get",
                "lease",
                PLATFORM_PREPARE_LEASE_NAME,
                "-n",
                namespace,
                "-o",
                "json",
            ],
            capture_output=True,
            check=False,
        )
        if current.returncode != 0:
            return
        payload = json.loads(current.stdout or "{}")
        current_holder = str(
            (payload.get("spec", {}) or {}).get("holderIdentity") or ""
        ).strip()
        if current_holder != holder:
            return
        run_command(
            [
                kubectl_cmd,
                "delete",
                "lease",
                PLATFORM_PREPARE_LEASE_NAME,
                "-n",
                namespace,
                "--ignore-not-found=true",
                "--wait=false",
            ],
            check=False,
        )
