from __future__ import annotations

import hashlib
import re
import time

from ..cluster import (
    CommandError,
    require_any_command,
    require_command,
    run_command,
    run_json_command,
)
from ..models import ResolvedRunPlan


def _release_names(plan: ResolvedRunPlan) -> list[str]:
    release = plan.deployment.release_name
    return [f"ms-{release}", f"gaie-{release}", f"infra-{release}"]


def _gaie_rbac_name(release_name: str) -> str:
    suffix = hashlib.sha1(release_name.encode("utf-8")).hexdigest()[:10]
    return f"benchflow-gaie-epp-rbac-{suffix}"


def _llmd_recipe_layout_from_repo_ref(repo_ref: str) -> bool:
    normalized = repo_ref.strip().lower()
    if normalized == "main":
        return True
    match = re.search(r"v?(\d+)\.(\d+)\.(\d+)(?:[-+][a-z0-9_.-]+)?", normalized)
    if match is None:
        return False
    version = tuple(int(part) for part in match.groups())
    return version >= (0, 6, 0)


def _storage_offloading_pvc_name(plan: ResolvedRunPlan) -> str:
    raw_config = plan.deployment.options.get("storage_offloading")
    if raw_config in (None, "", False):
        return ""
    if not isinstance(raw_config, dict):
        raise CommandError(
            "deployment profile options.storage_offloading must be a mapping"
        )
    if raw_config.get("enabled") is False:
        return ""
    return str(raw_config.get("pvc_name") or "llm-d-storage-offloading-cache").strip()


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
    pvc_name = _storage_offloading_pvc_name(plan)
    if not pvc_name:
        return False

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

    if recipe_layout:
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
