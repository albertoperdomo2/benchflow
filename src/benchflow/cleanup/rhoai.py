from __future__ import annotations

import shlex
import time

from ..cluster import CommandError, require_any_command, run_command, run_json_command
from ..models import ResolvedRunPlan
from ..renderers.deployment import rhoai_profiler_configmap_name
from ..rhoai_mooncake import (
    mooncake_configmap_name,
    mooncake_master_name,
    mooncake_master_service_name,
    mooncake_nvme_release_directory,
    rhoai_mooncake_spec,
)
from ..rhoai_gateway import remove_rhoai_release_gateway_listener
from ..ui import detail


def _deployment_kind(plan: ResolvedRunPlan) -> str:
    return str(plan.deployment.target.resource_kind or "LLMInferenceService").strip()


def _deployment_resource(plan: ResolvedRunPlan) -> str:
    return _deployment_kind(plan).lower()


def _delete_profiler_configmap(
    plan: ResolvedRunPlan, *, kubectl_cmd: str, namespace: str
) -> None:
    if not plan.execution.profiling.enabled:
        return
    run_command(
        [
            kubectl_cmd,
            "delete",
            "configmap",
            rhoai_profiler_configmap_name(plan),
            "-n",
            namespace,
            "--ignore-not-found",
        ],
        check=False,
    )


def _delete_runtime_pvcs(
    plan: ResolvedRunPlan, *, kubectl_cmd: str, namespace: str
) -> None:
    for pvc_mount in plan.deployment.runtime.pvc_mounts:
        if not pvc_mount.create:
            continue
        run_command(
            [
                kubectl_cmd,
                "delete",
                "pvc",
                pvc_mount.claim_name,
                "-n",
                namespace,
                "--ignore-not-found",
            ],
            check=False,
        )


def _remove_release_gateway_listener(
    plan: ResolvedRunPlan, *, kubectl_cmd: str
) -> None:
    if _deployment_kind(plan) != "LLMInferenceService":
        return
    remove_rhoai_release_gateway_listener(plan, kubectl_cmd=kubectl_cmd)


def _delete_mooncake_resources(
    plan: ResolvedRunPlan, *, kubectl_cmd: str, namespace: str
) -> None:
    if rhoai_mooncake_spec(plan) is None:
        return
    for resource, name in (
        ("deployment", mooncake_master_name(plan)),
        ("service", mooncake_master_service_name(plan)),
        ("configmap", mooncake_configmap_name(plan)),
    ):
        run_command(
            [
                kubectl_cmd,
                "delete",
                resource,
                name,
                "-n",
                namespace,
                "--ignore-not-found",
            ],
            check=False,
        )


def _clean_mooncake_nvme_store(
    plan: ResolvedRunPlan, *, kubectl_cmd: str, namespace: str
) -> None:
    release_dir = mooncake_nvme_release_directory(plan)
    if release_dir is None:
        return
    payload = run_json_command(
        [kubectl_cmd, "get", "pods", "-n", namespace, "-o", "json"]
    )
    for item in payload.get("items", []):
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") or {}
        pod_name = str(metadata.get("name") or "")
        if not pod_name.startswith(plan.deployment.release_name):
            continue
        spec = item.get("spec") or {}
        containers = spec.get("containers") or []
        container_names = {
            str(container.get("name") or "")
            for container in containers
            if isinstance(container, dict)
        }
        if "mooncake-store" not in container_names:
            continue
        target = shlex.quote(f"{release_dir}/{pod_name}")
        result = run_command(
            [
                kubectl_cmd,
                "exec",
                pod_name,
                "-n",
                namespace,
                "-c",
                "mooncake-store",
                "--",
                "/bin/sh",
                "-ec",
                f"rm -rf -- {target}",
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            detail(
                "Failed to delete Mooncake NVMe cache from "
                f"{pod_name}: {str(result.stderr or result.stdout or '').strip()}"
            )
            continue
        detail(f"Deleted Mooncake NVMe cache from {pod_name}")


def _runtime_container_with_host_path(pod: dict, host_path_name: str) -> str:
    containers = pod.get("spec", {}).get("containers", [])
    if not isinstance(containers, list):
        return ""
    for container in containers:
        if not isinstance(container, dict):
            continue
        mounts = container.get("volumeMounts", [])
        if not isinstance(mounts, list):
            continue
        if any(
            isinstance(mount, dict) and mount.get("name") == host_path_name
            for mount in mounts
        ):
            return str(container.get("name") or "").strip()
    return ""


def _clean_runtime_host_path_contents(
    plan: ResolvedRunPlan, *, kubectl_cmd: str, namespace: str
) -> None:
    host_paths = [
        host_path
        for host_path in plan.deployment.runtime.host_paths
        if host_path.cleanup and not host_path.read_only
    ]
    if not host_paths:
        return

    payload = run_json_command(
        [kubectl_cmd, "get", "pods", "-n", namespace, "-o", "json"]
    )
    pods_by_node: dict[tuple[str, str], tuple[str, str]] = {}
    for item in payload.get("items", []):
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") or {}
        pod_name = str(metadata.get("name") or "")
        if not pod_name.startswith(plan.deployment.release_name):
            continue
        if str((item.get("status") or {}).get("phase") or "") != "Running":
            continue
        node_name = str((item.get("spec") or {}).get("nodeName") or "")
        if not node_name:
            continue
        for host_path in host_paths:
            container = _runtime_container_with_host_path(item, host_path.name)
            if container:
                pods_by_node.setdefault(
                    (node_name, host_path.name), (pod_name, container)
                )

    if not pods_by_node:
        raise CommandError(
            "cannot clean rhoai runtime hostPath contents: no running model pod "
            f"was found for release {plan.deployment.release_name}"
        )

    failures: list[str] = []
    cleaned_nodes: set[str] = set()
    host_paths_by_name = {host_path.name: host_path for host_path in host_paths}
    for (node_name, host_path_name), (pod_name, container) in sorted(
        pods_by_node.items()
    ):
        host_path = host_paths_by_name[host_path_name]
        target = shlex.quote(host_path.mount_path.rstrip("/"))
        script = (
            "set -eu; "
            f"if [ -d {target} ]; then "
            f"find {target} -mindepth 1 -maxdepth 1 -exec rm -rf -- {{}} +; "
            "fi"
        )
        result = run_command(
            [
                kubectl_cmd,
                "exec",
                pod_name,
                "-n",
                namespace,
                "-c",
                container,
                "--",
                "sh",
                "-lc",
                script,
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            failures.append(
                f"{pod_name} ({host_path.mount_path}): "
                f"{str(result.stderr or result.stdout or '').strip()}"
            )
            continue
        cleaned_nodes.add(node_name)

    if failures:
        raise CommandError(
            "failed to clean rhoai runtime hostPath contents: " + "; ".join(failures)
        )
    detail(
        "Deleted rhoai runtime hostPath contents on node(s): "
        + ", ".join(sorted(cleaned_nodes))
    )


def cleanup_rhoai(
    plan: ResolvedRunPlan,
    *,
    wait_for_deletion: bool = True,
    timeout_seconds: int = 300,
    skip_if_not_exists: bool = True,
) -> None:
    kubectl_cmd = require_any_command("oc", "kubectl")
    namespace = plan.deployment.namespace
    release_name = plan.deployment.release_name
    resource = _deployment_resource(plan)
    resource_kind = _deployment_kind(plan)

    exists = run_command(
        [
            kubectl_cmd,
            "get",
            resource,
            release_name,
            "-n",
            namespace,
            "-o",
            "name",
        ],
        capture_output=True,
        check=False,
    )
    if exists.returncode != 0:
        _delete_profiler_configmap(plan, kubectl_cmd=kubectl_cmd, namespace=namespace)
        _delete_runtime_pvcs(plan, kubectl_cmd=kubectl_cmd, namespace=namespace)
        _remove_release_gateway_listener(plan, kubectl_cmd=kubectl_cmd)
        _delete_mooncake_resources(plan, kubectl_cmd=kubectl_cmd, namespace=namespace)
        if skip_if_not_exists:
            return
        raise CommandError(
            f"{resource_kind} {release_name} not found in namespace {namespace}"
        )

    _clean_mooncake_nvme_store(plan, kubectl_cmd=kubectl_cmd, namespace=namespace)
    _clean_runtime_host_path_contents(
        plan, kubectl_cmd=kubectl_cmd, namespace=namespace
    )
    run_command(
        [
            kubectl_cmd,
            "delete",
            resource,
            release_name,
            "-n",
            namespace,
        ]
    )

    if not wait_for_deletion:
        _delete_profiler_configmap(plan, kubectl_cmd=kubectl_cmd, namespace=namespace)
        _delete_runtime_pvcs(plan, kubectl_cmd=kubectl_cmd, namespace=namespace)
        _remove_release_gateway_listener(plan, kubectl_cmd=kubectl_cmd)
        _delete_mooncake_resources(plan, kubectl_cmd=kubectl_cmd, namespace=namespace)
        return

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        current = run_command(
            [
                kubectl_cmd,
                "get",
                resource,
                release_name,
                "-n",
                namespace,
                "-o",
                "name",
            ],
            capture_output=True,
            check=False,
        )
        if current.returncode != 0:
            _delete_profiler_configmap(
                plan, kubectl_cmd=kubectl_cmd, namespace=namespace
            )
            _delete_runtime_pvcs(plan, kubectl_cmd=kubectl_cmd, namespace=namespace)
            _remove_release_gateway_listener(plan, kubectl_cmd=kubectl_cmd)
            _delete_mooncake_resources(
                plan, kubectl_cmd=kubectl_cmd, namespace=namespace
            )
            return
        time.sleep(5)

    raise CommandError(
        f"timed out waiting for {resource_kind} deletion: {release_name}"
    )
