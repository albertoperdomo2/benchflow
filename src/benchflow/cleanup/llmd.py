from __future__ import annotations

import hashlib
import re
import shlex
import time

from ..cluster import (
    CommandError,
    require_any_command,
    require_command,
    run_command,
    run_json_command,
)
from ..models import ResolvedRunPlan
from ..storage_offloading import (
    STORAGE_OFFLOADING_TYPE_HOST_PATH,
    STORAGE_OFFLOADING_TYPE_PVC,
    storage_offloading_config,
)
from ..ui import detail


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
    normalized = repo_ref.strip().lower()
    if normalized == "main":
        return True
    match = re.search(r"v?(\d+)\.(\d+)\.(\d+)(?:[-+][a-z0-9_.-]+)?", normalized)
    if match is None:
        return False
    version = tuple(int(part) for part in match.groups())
    return version >= (0, 6, 0)


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
                return
            else:
                pvc_deleted = _delete_storage_offloading_pvc(
                    plan,
                    kubectl_cmd,
                    wait_for_deletion=wait_for_deletion,
                    timeout_seconds=timeout_seconds,
                )
                if pvc_deleted:
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
            return
        else:
            pvc_deleted = _delete_storage_offloading_pvc(
                plan,
                kubectl_cmd,
                wait_for_deletion=wait_for_deletion,
                timeout_seconds=timeout_seconds,
            )
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
        run_command(["helm", "uninstall", release_name, "-n", namespace])
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
