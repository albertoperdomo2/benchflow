"""GPU management access for LMCache without reserving serving GPUs."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from .cluster import CommandError, run_command

_ENV = "NRI_MANAGEMENT_CDI_DEVICE_NAMESPACES"
_OWNERS = "benchflow.io/lmcache-nri-namespaces"
_TOOLKIT = "app=nvidia-container-toolkit-daemonset"


def _policy(kubectl: str) -> dict[str, Any] | None:
    result = run_command(
        [kubectl, "get", "clusterpolicies.nvidia.com", "-o", "json"],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        if any(
            word in (result.stderr or "").lower()
            for word in (
                "notfound",
                "doesn't have a resource type",
                "could not find the requested resource",
            )
        ):
            return None
        raise CommandError(f"Cannot inspect NVIDIA ClusterPolicy: {result.stderr}")
    items = json.loads(result.stdout or "{}").get("items", [])
    if len(items) > 1:
        raise CommandError("LMCache requires one NVIDIA ClusterPolicy")
    return items[0] if items else None


def _toolkits(kubectl: str) -> list[dict[str, Any]]:
    result = run_command(
        [kubectl, "get", "daemonsets", "-A", "-l", _TOOLKIT, "-o", "json"],
        capture_output=True,
    )
    return json.loads(result.stdout).get("items", [])


def _namespace_patch(
    kubectl: str,
    policy: dict[str, Any],
    namespace: str,
    *,
    add: bool,
    retain_owner: bool = False,
) -> bool:
    metadata = policy["metadata"]
    annotations = dict(metadata.get("annotations") or {})
    owners = set(json.loads(annotations.get(_OWNERS, "[]")))
    toolkit = dict(policy.get("spec", {}).get("toolkit") or {})
    env = [dict(item) for item in (toolkit.get("env") or [])]
    entry = next((item for item in env if item.get("name") == _ENV), None)
    if entry and "valueFrom" in entry:
        raise CommandError(
            "Cannot reconcile LMCache namespace access from a valueFrom NVIDIA toolkit setting"
        )
    namespaces = {
        n.strip() for n in (entry or {}).get("value", "").split(",") if n.strip()
    }
    if add:
        if namespace in namespaces:
            return False  # Never claim or later revoke somebody else's grant.
        namespaces.add(namespace)
        owners.add(namespace)
    else:
        if namespace not in owners:
            return False
        if not retain_owner:
            owners.remove(namespace)
        namespaces.discard(namespace)
    env = [item for item in env if item.get("name") != _ENV]
    if namespaces:
        env.append({"name": _ENV, "value": ",".join(sorted(namespaces))})
    toolkit["env"] = env
    annotations[_OWNERS] = json.dumps(sorted(owners))
    patch = [
        {
            "op": "test",
            "path": "/metadata/resourceVersion",
            "value": metadata["resourceVersion"],
        },
        {"op": "add", "path": "/metadata/annotations", "value": annotations},
        {"op": "add", "path": "/spec/toolkit", "value": toolkit},
    ]
    run_command(
        [
            kubectl,
            "patch",
            "clusterpolicy",
            metadata["name"],
            "--type=json",
            "-p",
            json.dumps(patch),
        ]
    )
    return True


def _wait_toolkit_namespace(kubectl: str, namespace: str, *, present: bool) -> None:
    deadline = time.monotonic() + 300
    while True:
        toolkits = _toolkits(kubectl)
        ready = bool(toolkits)
        for daemonset in toolkits:
            containers = daemonset["spec"]["template"]["spec"]["containers"]
            values = [
                item.get("value", "")
                for container in containers
                for item in container.get("env", [])
                if item.get("name") == _ENV
            ]
            included = any(
                namespace in {n.strip() for n in value.split(",")} for value in values
            )
            ready = ready and included == present
        if ready:
            for daemonset in toolkits:
                meta = daemonset["metadata"]
                run_command(
                    [
                        kubectl,
                        "rollout",
                        "status",
                        f"daemonset/{meta['name']}",
                        "-n",
                        meta["namespace"],
                        "--timeout=300s",
                    ]
                )
            return
        if time.monotonic() >= deadline:
            raise CommandError(
                "GPU Operator did not reconcile LMCache management access within 300 seconds"
            )
        time.sleep(2)


def ensure_lmcache_gpu_access(kubectl: str, namespace: str) -> str:
    runtime = run_command(
        [kubectl, "get", "runtimeclass", "nvidia", "--ignore-not-found", "-o", "json"],
        capture_output=True,
    )
    if runtime.stdout.strip():
        return "runtime-class"
    policy = _policy(kubectl)
    cdi = (policy or {}).get("spec", {}).get("cdi", {})
    if not policy or not cdi.get("enabled") or not cdi.get("nriPluginEnabled"):
        raise CommandError(
            "LMCache requires the BenchFlow GPU prerequisites: NVIDIA RuntimeClass or enabled GPU Operator CDI/NRI"
        )
    toolkits = _toolkits(kubectl)
    if not toolkits:
        raise CommandError("NVIDIA Container Toolkit DaemonSet was not found")
    for daemonset in toolkits:
        images = [
            c.get("image", "")
            for c in daemonset["spec"]["template"]["spec"]["containers"]
        ]
        versions = [
            tuple(map(int, match.groups()))
            for image in images
            if (match := re.search(r":v?(\d+)\.(\d+)\.(\d+)", image))
        ]
        if not versions or max(versions) < (1, 20, 0):
            raise CommandError(
                "LMCache CDI/NRI needs NVIDIA Container Toolkit >=1.20.0 in the BenchFlow GPU prerequisites"
            )
    _namespace_patch(kubectl, policy, namespace, add=True)
    # Also wait on retries after a successful patch but failed reconciliation.
    _wait_toolkit_namespace(kubectl, namespace, present=True)
    return "nri"


def release_lmcache_gpu_access(kubectl: str, namespace: str) -> None:
    policy = _policy(kubectl)
    if policy and _namespace_patch(
        kubectl, policy, namespace, add=False, retain_owner=True
    ):
        # Retain ownership until reconciliation succeeds so a failed wait can
        # be retried instead of silently leaving toolkit access enabled.
        _wait_toolkit_namespace(kubectl, namespace, present=False)
        policy = _policy(kubectl)
        if policy:
            _namespace_patch(kubectl, policy, namespace, add=False)
