from __future__ import annotations

from pathlib import Path

import yaml

from ..cluster import CommandError, require_any_command, run_command
from ..models import ResolvedRunPlan, ValidationError
from ..renderers.deployment import (
    render_runtime_pvc_manifests,
    render_rhaiis_raw_vllm_manifests,
    rhaiis_raw_vllm_deployment_name,
    rhaiis_raw_vllm_workload_kind,
)
from ..ui import detail, step, success


def _ensure_supported_mode(plan: ResolvedRunPlan) -> None:
    if plan.deployment.mode != "raw-vllm":
        raise ValidationError(
            f"unsupported RHAIIS deployment mode: {plan.deployment.mode}"
        )


def _workload_exists(
    namespace: str, workload_kind: str, workload_name: str, kubectl_cmd: str
) -> bool:
    result = run_command(
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
    return result.returncode == 0


def _verify_workload(
    namespace: str,
    workload_kind: str,
    workload_name: str,
    kubectl_cmd: str,
    timeout_seconds: int,
) -> None:
    step(
        f"Waiting for RHAIIS {workload_kind} {workload_name} in namespace {namespace} to become ready"
    )
    run_command(
        [
            kubectl_cmd,
            "rollout",
            "status",
            f"{workload_kind}/{workload_name}",
            "-n",
            namespace,
            f"--timeout={timeout_seconds}s",
        ]
    )
    success(f"RHAIIS {workload_kind} {workload_name} is ready")


def _manifest_filename(manifest: dict, plan: ResolvedRunPlan) -> str:
    kind = str(manifest.get("kind") or "manifest").lower()
    name = str((manifest.get("metadata") or {}).get("name") or "")
    if kind == "service" and name != plan.deployment.release_name:
        return "headless-service.yaml"
    return f"{kind}.yaml"


def _apply_runtime_pvc_manifests(plan: ResolvedRunPlan, kubectl_cmd: str) -> None:
    for manifest in render_runtime_pvc_manifests(plan):
        name = str(manifest.get("metadata", {}).get("name") or "").strip()
        step(f"Ensuring runtime PVC {name} in namespace {plan.deployment.namespace}")
        run_command(
            [kubectl_cmd, "apply", "-f", "-"],
            input_text=yaml.safe_dump(manifest, sort_keys=False),
        )


def deploy_rhaiis(
    plan: ResolvedRunPlan,
    *,
    manifests_dir: Path | None = None,
    skip_if_exists: bool = True,
    verify: bool = True,
    verify_timeout_seconds: int = 1800,
) -> Path:
    _ensure_supported_mode(plan)

    kubectl_cmd = require_any_command("oc", "kubectl")
    namespace = plan.deployment.namespace
    workload_name = rhaiis_raw_vllm_deployment_name(plan)
    workload_kind = rhaiis_raw_vllm_workload_kind(plan)
    manifests = render_rhaiis_raw_vllm_manifests(plan)

    if skip_if_exists and _workload_exists(
        namespace, workload_kind, workload_name, kubectl_cmd
    ):
        success(f"Skipping deploy; {workload_kind} {workload_name} already exists")
        return manifests_dir.resolve() if manifests_dir else Path.cwd()

    if manifests_dir is not None:
        manifests_dir.mkdir(parents=True, exist_ok=True)
        for pvc_manifest in render_runtime_pvc_manifests(plan):
            pvc_name = str(pvc_manifest.get("metadata", {}).get("name") or "runtime")
            pvc_target = manifests_dir / f"pvc-{pvc_name}.yaml"
            pvc_target.write_text(
                yaml.safe_dump(pvc_manifest, sort_keys=False), encoding="utf-8"
            )
            detail(f"Rendered runtime PVC manifest written to {pvc_target}")
        for manifest in manifests:
            name = _manifest_filename(manifest, plan)
            target = manifests_dir / name
            target.write_text(
                yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
            )
            detail(f"Rendered RHAIIS manifest written to {target}")

    step(
        f"Applying RHAIIS {plan.deployment.mode} deployment {plan.deployment.release_name} "
        f"in namespace {namespace}"
    )
    _apply_runtime_pvc_manifests(plan, kubectl_cmd)
    for manifest in manifests:
        run_command(
            [kubectl_cmd, "apply", "-f", "-"],
            input_text=yaml.safe_dump(manifest, sort_keys=False),
        )
    success(
        f"Applied RHAIIS raw-vLLM {workload_kind} {workload_name} and supporting services in namespace {namespace}"
    )

    if verify:
        try:
            _verify_workload(
                namespace,
                workload_kind,
                workload_name,
                kubectl_cmd,
                verify_timeout_seconds,
            )
        except CommandError as exc:
            raise CommandError(
                f"failed to verify RHAIIS {workload_kind} {workload_name}: {exc}"
            ) from exc

    return manifests_dir.resolve() if manifests_dir else Path.cwd()
