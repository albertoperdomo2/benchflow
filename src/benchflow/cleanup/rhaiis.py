from __future__ import annotations

import time

from ..cluster import CommandError, require_any_command, run_command
from ..models import ResolvedRunPlan, ValidationError
from ..renderers.deployment import (
    rhaiis_raw_vllm_deployment_name,
    rhaiis_raw_vllm_service_name,
)


def _ensure_supported_mode(plan: ResolvedRunPlan) -> None:
    if plan.deployment.mode != "raw-vllm":
        raise ValidationError(
            f"unsupported RHAIIS deployment mode: {plan.deployment.mode}"
        )


def cleanup_rhaiis(
    plan: ResolvedRunPlan,
    *,
    wait_for_deletion: bool = True,
    timeout_seconds: int = 300,
    skip_if_not_exists: bool = True,
) -> None:
    _ensure_supported_mode(plan)

    kubectl_cmd = require_any_command("oc", "kubectl")
    namespace = plan.deployment.namespace
    deployment_name = rhaiis_raw_vllm_deployment_name(plan)
    service_name = rhaiis_raw_vllm_service_name(plan)

    exists = run_command(
        [
            kubectl_cmd,
            "get",
            "deployment",
            deployment_name,
            "-n",
            namespace,
            "-o",
            "name",
        ],
        capture_output=True,
        check=False,
    )
    if exists.returncode != 0:
        run_command(
            [
                kubectl_cmd,
                "delete",
                "service",
                service_name,
                "-n",
                namespace,
                "--ignore-not-found",
            ],
            check=False,
        )
        if skip_if_not_exists:
            return
        raise CommandError(
            f"Deployment {deployment_name} not found in namespace {namespace}"
        )

    run_command(
        [
            kubectl_cmd,
            "delete",
            "service",
            service_name,
            "-n",
            namespace,
            "--ignore-not-found",
        ],
        check=False,
    )
    run_command(
        [
            kubectl_cmd,
            "delete",
            "deployment",
            deployment_name,
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
                "deployment",
                deployment_name,
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

    raise CommandError(f"timed out waiting for Deployment deletion: {deployment_name}")
