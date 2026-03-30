from __future__ import annotations

import time

from ..cluster import CommandError, require_any_command, run_command
from ..models import ResolvedRunPlan
from ..renderers.deployment import rhoai_profiler_configmap_name


def _deployment_resource(plan: ResolvedRunPlan) -> str:
    return "llminferenceservice"


def _deployment_kind(plan: ResolvedRunPlan) -> str:
    return "LLMInferenceService"


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
        if skip_if_not_exists:
            return
        raise CommandError(
            f"{resource_kind} {release_name} not found in namespace {namespace}"
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
            return
        time.sleep(5)

    raise CommandError(
        f"timed out waiting for {resource_kind} deletion: {release_name}"
    )
