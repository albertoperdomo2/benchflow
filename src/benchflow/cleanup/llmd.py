from __future__ import annotations

import hashlib
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

    if not existing.intersection(releases):
        if skip_if_not_exists:
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
