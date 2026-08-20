from __future__ import annotations

import time

from ..cluster import CommandError, require_any_command, run_command
from ..models import ResolvedRunPlan, ValidationError
from ..renderers.deployment import (
    rhaiis_raw_vllm_deployment_name,
    rhaiis_raw_vllm_headless_service_name,
    rhaiis_raw_vllm_is_distributed,
    rhaiis_raw_vllm_service_name,
    rhaiis_raw_vllm_servicemonitor_name,
    rhaiis_raw_vllm_workload_kind,
)


def _ensure_supported_mode(plan: ResolvedRunPlan) -> None:
    if plan.deployment.mode != "raw-vllm":
        raise ValidationError(
            f"unsupported RHAIIS deployment mode: {plan.deployment.mode}"
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
    workload_name = rhaiis_raw_vllm_deployment_name(plan)
    workload_kind = rhaiis_raw_vllm_workload_kind(plan)
    service_name = rhaiis_raw_vllm_service_name(plan)
    service_names = [service_name]
    if rhaiis_raw_vllm_is_distributed(plan):
        service_names.append(rhaiis_raw_vllm_headless_service_name(plan))
    servicemonitor_name = rhaiis_raw_vllm_servicemonitor_name(plan)

    exists = run_command(
        [
            kubectl_cmd,
            "get",
            workload_kind,
            workload_name,
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
                "servicemonitor",
                servicemonitor_name,
                "-n",
                namespace,
                "--ignore-not-found",
            ],
            check=False,
        )
        for name in service_names:
            run_command(
                [
                    kubectl_cmd,
                    "delete",
                    "service",
                    name,
                    "-n",
                    namespace,
                    "--ignore-not-found",
                ],
                check=False,
            )
        _delete_runtime_pvcs(plan, kubectl_cmd=kubectl_cmd, namespace=namespace)
        if skip_if_not_exists:
            return
        raise CommandError(
            f"{workload_kind} {workload_name} not found in namespace {namespace}"
        )

    run_command(
        [
            kubectl_cmd,
            "delete",
            "servicemonitor",
            servicemonitor_name,
            "-n",
            namespace,
            "--ignore-not-found",
        ],
        check=False,
    )
    for name in service_names:
        run_command(
            [
                kubectl_cmd,
                "delete",
                "service",
                name,
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
            workload_kind,
            workload_name,
            "-n",
            namespace,
        ]
    )

    if not wait_for_deletion:
        _delete_runtime_pvcs(plan, kubectl_cmd=kubectl_cmd, namespace=namespace)
        return

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        current = run_command(
            [
                kubectl_cmd,
                "get",
                workload_kind,
                workload_name,
                "-n",
                namespace,
                "-o",
                "name",
            ],
            capture_output=True,
            check=False,
        )
        if current.returncode != 0:
            _delete_runtime_pvcs(plan, kubectl_cmd=kubectl_cmd, namespace=namespace)
            return
        time.sleep(5)

    raise CommandError(
        f"timed out waiting for {workload_kind} deletion: {workload_name}"
    )
