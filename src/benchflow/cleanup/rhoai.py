from __future__ import annotations

import time

from ..cluster import CommandError, require_any_command, run_command
from ..models import ResolvedRunPlan


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

    exists = run_command(
        [
            kubectl_cmd,
            "get",
            "llminferenceservice",
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
        if skip_if_not_exists:
            return
        raise CommandError(
            f"LLMInferenceService {release_name} not found in namespace {namespace}"
        )

    run_command(
        [
            kubectl_cmd,
            "delete",
            "llminferenceservice",
            release_name,
            "-n",
            namespace,
        ]
    )

    if not wait_for_deletion:
        return

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        current = run_command(
            [
                kubectl_cmd,
                "get",
                "llminferenceservice",
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
            return
        time.sleep(5)

    raise CommandError(
        f"timed out waiting for LLMInferenceService deletion: {release_name}"
    )
