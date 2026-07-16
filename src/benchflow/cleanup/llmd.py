from __future__ import annotations

import hashlib
import json
import shlex
import time

from ..cluster import (
    CommandError,
    require_any_command,
    require_command,
    run_command,
    run_json_command,
)
from ..llmd_layout import uses_recipe_layout
from ..models import ResolvedRunPlan
from ..storage_offloading import (
    STORAGE_OFFLOADING_TYPE_HOST_PATH,
    STORAGE_OFFLOADING_TYPE_PVC,
    storage_offloading_config,
)
from ..ui import detail

_SHARED_GATEWAY_NAME = "llm-d-inference-gateway"
_SHARED_GATEWAY_INFRA_NAME = "llm-d-inference-gateway-istio"


def _release_names(plan: ResolvedRunPlan) -> list[str]:
    release = plan.deployment.release_name
    return [f"ms-{release}", f"gaie-{release}", f"infra-{release}"]


def _gaie_rbac_name(release_name: str) -> str:
    suffix = hashlib.sha1(release_name.encode("utf-8")).hexdigest()[:10]
    return f"benchflow-gaie-epp-rbac-{suffix}"


def _modelserver_deployment_name(release_name: str) -> str:
    return f"ms-{release_name}-decode"


def _modelserver_service_account_name(release_name: str) -> str:
    return f"ms-{release_name}-sa"


def _llmd_recipe_layout_from_repo_ref(repo_ref: str) -> bool:
    return uses_recipe_layout(repo_ref)


def _pvc_exists(kubectl_cmd: str, namespace: str, pvc_name: str) -> bool:
    probe = run_command(
        [
            kubectl_cmd,
            "get",
            "pvc",
            pvc_name,
            "-n",
            namespace,
            "-o",
            "name",
        ],
        capture_output=True,
        check=False,
    )
    return probe.returncode == 0 and bool(str(probe.stdout or "").strip())


def _delete_storage_offloading_pvc(
    plan: ResolvedRunPlan,
    kubectl_cmd: str,
    *,
    wait_for_deletion: bool,
    timeout_seconds: int,
) -> bool:
    config = storage_offloading_config(plan)
    if not config or config["type"] != STORAGE_OFFLOADING_TYPE_PVC:
        return False
    pvc_name = config["pvc_name"]

    namespace = plan.deployment.namespace
    existed = _pvc_exists(kubectl_cmd, namespace, pvc_name)
    run_command(
        [
            kubectl_cmd,
            "delete",
            "pvc",
            pvc_name,
            "-n",
            namespace,
            "--ignore-not-found=true",
        ],
    )
    if not wait_for_deletion or not existed:
        return existed

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if not _pvc_exists(kubectl_cmd, namespace, pvc_name):
            return True
        time.sleep(5)
    raise CommandError(
        f"timed out waiting for storage offloading PVC deletion: {pvc_name}"
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


def _stale_helm_release_names(
    kubectl_cmd: str, namespace: str, release_names: set[str]
) -> set[str]:
    result = run_command(
        [
            kubectl_cmd,
            "get",
            "secret",
            "-n",
            namespace,
            "-l",
            "owner=helm",
            "-o",
            "json",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return set()

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return set()

    stale: set[str] = set()
    for item in payload.get("items", []):
        if not isinstance(item, dict):
            continue
        labels = item.get("metadata", {}).get("labels", {})
        if not isinstance(labels, dict):
            continue
        name = str(labels.get("name") or "")
        status = str(labels.get("status") or "")
        if name in release_names and status == "uninstalling":
            stale.add(name)
    return stale


def _delete_stale_helm_release_secrets(
    kubectl_cmd: str, namespace: str, release_names: set[str]
) -> None:
    for release_name in sorted(
        _stale_helm_release_names(kubectl_cmd, namespace, release_names)
    ):
        result = run_command(
            [
                kubectl_cmd,
                "delete",
                "secret",
                "-n",
                namespace,
                "-l",
                f"owner=helm,name={release_name}",
                "--ignore-not-found=true",
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise CommandError(
                "failed to delete stale Helm release metadata for "
                f"{release_name}: "
                f"{str(result.stderr or result.stdout or '').strip()}"
            )
        detail(f"Deleted stale Helm release metadata: {release_name}")


def _pod_container_name(pod: dict) -> str:
    containers = pod.get("spec", {}).get("containers", [])
    if not isinstance(containers, list):
        return ""
    for container in containers:
        if not isinstance(container, dict):
            continue
        name = str(container.get("name") or "").strip()
        if name:
            return name
    return ""


def _release_pods(kubectl_cmd: str, namespace: str, release_label: str) -> list[dict]:
    result = run_command(
        [
            kubectl_cmd,
            "get",
            "pods",
            "-n",
            namespace,
            "-l",
            release_label,
            "-o",
            "json",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    import json

    payload = json.loads(result.stdout or "{}")
    items = payload.get("items", [])
    return [item for item in items if isinstance(item, dict)]


def _delete_storage_offloading_host_path_contents(
    plan: ResolvedRunPlan,
    kubectl_cmd: str,
    release_label: str,
) -> bool:
    config = storage_offloading_config(plan)
    if not config or config["type"] != STORAGE_OFFLOADING_TYPE_HOST_PATH:
        return False

    namespace = plan.deployment.namespace
    directory_path = str(config["directory_path"])
    target = shlex.quote(directory_path)
    script = (
        "set -eu; "
        f"if [ -d {target} ]; then "
        f"du -sh {target} 2>/dev/null || true; "
        f"find {target} -mindepth 1 -maxdepth 1 -exec rm -rf -- {{}} +; "
        "fi"
    )
    deleted = False
    for pod in _release_pods(kubectl_cmd, namespace, release_label):
        metadata = pod.get("metadata", {})
        pod_name = str(metadata.get("name") or "").strip()
        if not pod_name or not pod_name.startswith(
            f"ms-{plan.deployment.release_name}"
        ):
            continue
        container = _pod_container_name(pod)
        if not container:
            continue
        result = run_command(
            [
                kubectl_cmd,
                "exec",
                pod_name,
                "-c",
                container,
                "-n",
                namespace,
                "--",
                "sh",
                "-lc",
                script,
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            detail(
                "Failed to delete storage offloading hostPath contents from "
                f"{pod_name}: {str(result.stderr or result.stdout or '').strip()}"
            )
            continue
        deleted = True
    if deleted:
        detail(
            f"Deleted storage offloading hostPath contents mounted at {directory_path}"
        )
    return deleted


def _delete_runtime_host_path_contents(
    plan: ResolvedRunPlan,
    kubectl_cmd: str,
    release_label: str,
) -> None:
    host_paths = [
        host_path
        for host_path in plan.deployment.runtime.host_paths
        if host_path.cleanup and not host_path.read_only
    ]
    if not host_paths:
        return

    namespace = plan.deployment.namespace
    pods_by_node: dict[str, tuple[str, str]] = {}
    for pod in _release_pods(kubectl_cmd, namespace, release_label):
        metadata = pod.get("metadata", {})
        pod_name = str(metadata.get("name") or "").strip()
        if not pod_name.startswith(f"ms-{plan.deployment.release_name}"):
            continue
        if str(pod.get("status", {}).get("phase") or "") != "Running":
            continue
        node_name = str(pod.get("spec", {}).get("nodeName") or "").strip()
        container = _pod_container_name(pod)
        if node_name and container and node_name not in pods_by_node:
            pods_by_node[node_name] = (pod_name, container)

    if not pods_by_node:
        raise CommandError(
            "cannot clean llm-d runtime hostPath contents: no running model pod "
            f"was found for release {plan.deployment.release_name}"
        )

    failures: list[str] = []
    cleaned_nodes: set[str] = set()
    for node_name, (pod_name, container) in sorted(pods_by_node.items()):
        for host_path in host_paths:
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
                    "-c",
                    container,
                    "-n",
                    namespace,
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
            "failed to clean llm-d runtime hostPath contents: " + "; ".join(failures)
        )
    if cleaned_nodes:
        detail(
            "Deleted llm-d runtime hostPath contents on node(s): "
            + ", ".join(sorted(cleaned_nodes))
        )


def _shared_gateway_has_routes(kubectl_cmd: str, namespace: str) -> bool:
    result = run_command(
        [kubectl_cmd, "get", "httproute", "-n", namespace, "-o", "json"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    payload = json.loads(result.stdout or "{}")
    for item in payload.get("items", []):
        if not isinstance(item, dict):
            continue
        parent_refs = item.get("spec", {}).get("parentRefs", [])
        if not isinstance(parent_refs, list):
            continue
        for parent_ref in parent_refs:
            if not isinstance(parent_ref, dict):
                continue
            if parent_ref.get("name") == _SHARED_GATEWAY_NAME:
                return True
    return False


def _delete_shared_gateway_if_unused(
    kubectl_cmd: str,
    namespace: str,
    *,
    wait_for_deletion: bool,
    timeout_seconds: int,
) -> None:
    if _shared_gateway_has_routes(kubectl_cmd, namespace):
        return

    for kind, name in (
        ("gateway", _SHARED_GATEWAY_NAME),
        ("configmap", _SHARED_GATEWAY_NAME),
        ("deployment", _SHARED_GATEWAY_INFRA_NAME),
        ("service", _SHARED_GATEWAY_INFRA_NAME),
        ("serviceaccount", _SHARED_GATEWAY_INFRA_NAME),
    ):
        run_command(
            [
                kubectl_cmd,
                "delete",
                kind,
                name,
                "-n",
                namespace,
                "--ignore-not-found=true",
            ],
        )

    if not wait_for_deletion:
        return

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        result = run_command(
            [
                kubectl_cmd,
                "get",
                "deployment",
                _SHARED_GATEWAY_INFRA_NAME,
                "-n",
                namespace,
                "-o",
                "name",
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0 or not str(result.stdout or "").strip():
            return
        time.sleep(5)
    raise CommandError(
        f"timed out waiting for shared llm-d gateway deletion: {_SHARED_GATEWAY_INFRA_NAME}"
    )


def cleanup_llmd(
    plan: ResolvedRunPlan,
    *,
    wait_for_deletion: bool = True,
    timeout_seconds: int = 600,
    skip_if_not_exists: bool = True,
) -> None:
    require_command("helm")
    kubectl_cmd = require_any_command("oc", "kubectl")

    namespace = plan.deployment.namespace
    releases = _release_names(plan)
    helm_releases = run_json_command(["helm", "list", "-n", namespace, "-o", "json"])
    existing = {entry.get("name") for entry in helm_releases}
    recipe_layout = _llmd_recipe_layout_from_repo_ref(plan.deployment.repo_ref)
    release_label = f"benchflow.io/release={plan.deployment.release_name}"
    # `helm list` hides releases left in the uninstalling state. Include them
    # so cleanup can finish deleting resources and release metadata.
    existing.update(_stale_helm_release_names(kubectl_cmd, namespace, set(releases)))

    if not existing.intersection(releases):
        if recipe_layout:
            probe = run_command(
                [
                    kubectl_cmd,
                    "get",
                    "gateway,configmap,deployment,service,serviceaccount,podmonitor,httproute",
                    "-n",
                    namespace,
                    "-l",
                    release_label,
                    "-o",
                    "name",
                ],
                capture_output=True,
                check=False,
            )
            if probe.returncode == 0 and str(probe.stdout or "").strip():
                existing = set()
            elif skip_if_not_exists:
                _delete_storage_offloading_pvc(
                    plan,
                    kubectl_cmd,
                    wait_for_deletion=wait_for_deletion,
                    timeout_seconds=timeout_seconds,
                )
                _delete_runtime_pvcs(plan, kubectl_cmd=kubectl_cmd, namespace=namespace)
                _delete_shared_gateway_if_unused(
                    kubectl_cmd,
                    namespace,
                    wait_for_deletion=wait_for_deletion,
                    timeout_seconds=timeout_seconds,
                )
                return
            else:
                pvc_deleted = _delete_storage_offloading_pvc(
                    plan,
                    kubectl_cmd,
                    wait_for_deletion=wait_for_deletion,
                    timeout_seconds=timeout_seconds,
                )
                _delete_runtime_pvcs(plan, kubectl_cmd=kubectl_cmd, namespace=namespace)
                if pvc_deleted:
                    _delete_shared_gateway_if_unused(
                        kubectl_cmd,
                        namespace,
                        wait_for_deletion=wait_for_deletion,
                        timeout_seconds=timeout_seconds,
                    )
                    return
                raise CommandError(
                    f"no llm-d releases found for {plan.deployment.release_name}"
                )
        elif skip_if_not_exists:
            _delete_storage_offloading_pvc(
                plan,
                kubectl_cmd,
                wait_for_deletion=wait_for_deletion,
                timeout_seconds=timeout_seconds,
            )
            _delete_runtime_pvcs(plan, kubectl_cmd=kubectl_cmd, namespace=namespace)
            return
        else:
            pvc_deleted = _delete_storage_offloading_pvc(
                plan,
                kubectl_cmd,
                wait_for_deletion=wait_for_deletion,
                timeout_seconds=timeout_seconds,
            )
            _delete_runtime_pvcs(plan, kubectl_cmd=kubectl_cmd, namespace=namespace)
            if pvc_deleted:
                return
            raise CommandError(
                f"no llm-d releases found for {plan.deployment.release_name}"
            )

    run_command(
        [
            kubectl_cmd,
            "delete",
            "httproute",
            f"llm-d-{plan.deployment.release_name}",
            "-n",
            namespace,
            "--ignore-not-found=true",
        ],
    )
    resource_name = _gaie_rbac_name(plan.deployment.release_name)
    run_command(
        [
            kubectl_cmd,
            "delete",
            "rolebinding",
            resource_name,
            "-n",
            namespace,
            "--ignore-not-found=true",
        ],
    )
    run_command(
        [
            kubectl_cmd,
            "delete",
            "role",
            resource_name,
            "-n",
            namespace,
            "--ignore-not-found=true",
        ],
    )

    _delete_runtime_host_path_contents(plan, kubectl_cmd, release_label)
    _delete_storage_offloading_host_path_contents(plan, kubectl_cmd, release_label)

    if recipe_layout:
        for kind, name in (
            ("deployment", _modelserver_deployment_name(plan.deployment.release_name)),
            (
                "serviceaccount",
                _modelserver_service_account_name(plan.deployment.release_name),
            ),
        ):
            run_command(
                [
                    kubectl_cmd,
                    "delete",
                    kind,
                    name,
                    "-n",
                    namespace,
                    "--ignore-not-found=true",
                ],
            )
        for kind in (
            "gateway",
            "configmap",
            "deployment",
            "service",
            "serviceaccount",
            "podmonitor",
            "httproute",
        ):
            run_command(
                [
                    kubectl_cmd,
                    "delete",
                    kind,
                    "-n",
                    namespace,
                    "-l",
                    release_label,
                    "--ignore-not-found=true",
                ],
            )

    deadline = time.time() + timeout_seconds
    for release_name in releases:
        if release_name not in existing:
            continue
        result = run_command(
            ["helm", "uninstall", release_name, "-n", namespace],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            detail(
                f"Helm uninstall did not complete for {release_name}; "
                "continuing resource cleanup: "
                f"{str(result.stderr or result.stdout or '').strip()}"
            )
            continue
        if not wait_for_deletion:
            continue
        while time.time() < deadline:
            current = run_json_command(["helm", "list", "-n", namespace, "-o", "json"])
            if all(entry.get("name") != release_name for entry in current):
                break
            time.sleep(5)
        else:
            raise CommandError(
                f"timed out waiting for Helm release deletion: {release_name}"
            )

    _delete_storage_offloading_pvc(
        plan,
        kubectl_cmd,
        wait_for_deletion=wait_for_deletion,
        timeout_seconds=timeout_seconds,
    )
    _delete_runtime_pvcs(plan, kubectl_cmd=kubectl_cmd, namespace=namespace)
    _delete_stale_helm_release_secrets(kubectl_cmd, namespace, set(releases))
    if recipe_layout:
        _delete_shared_gateway_if_unused(
            kubectl_cmd,
            namespace,
            wait_for_deletion=wait_for_deletion,
            timeout_seconds=timeout_seconds,
        )
