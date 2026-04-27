from __future__ import annotations

from pathlib import Path

import yaml

from ..cluster import CommandError, require_any_command, run_command
from ..models import ResolvedRunPlan, ValidationError
from ..renderers.deployment import (
    render_rhaiis_raw_vllm_manifests,
    rhaiis_raw_vllm_deployment_name,
)
from ..ui import detail, step, success


def _ensure_supported_mode(plan: ResolvedRunPlan) -> None:
    if plan.deployment.mode != "raw-vllm":
        raise ValidationError(
            f"unsupported RHAIIS deployment mode: {plan.deployment.mode}"
        )


def _deployment_exists(namespace: str, deployment_name: str, kubectl_cmd: str) -> bool:
    result = run_command(
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
    return result.returncode == 0


def _verify_deployment(
    namespace: str,
    deployment_name: str,
    kubectl_cmd: str,
    timeout_seconds: int,
) -> None:
    step(
        f"Waiting for RHAIIS deployment {deployment_name} in namespace {namespace} to become ready"
    )
    run_command(
        [
            kubectl_cmd,
            "rollout",
            "status",
            f"deployment/{deployment_name}",
            "-n",
            namespace,
            f"--timeout={timeout_seconds}s",
        ]
    )
    success(f"RHAIIS deployment {deployment_name} is ready")


def deploy_rhaiis(
    plan: ResolvedRunPlan,
    *,
    manifests_dir: Path | None = None,
    skip_if_exists: bool = True,
    verify: bool = True,
    verify_timeout_seconds: int = 900,
) -> Path:
    _ensure_supported_mode(plan)

    kubectl_cmd = require_any_command("oc", "kubectl")
    namespace = plan.deployment.namespace
    deployment_name = rhaiis_raw_vllm_deployment_name(plan)
    manifests = render_rhaiis_raw_vllm_manifests(plan)

    if skip_if_exists and _deployment_exists(namespace, deployment_name, kubectl_cmd):
        success(f"Skipping deploy; Deployment {deployment_name} already exists")
        return manifests_dir.resolve() if manifests_dir else Path.cwd()

    if manifests_dir is not None:
        manifests_dir.mkdir(parents=True, exist_ok=True)
        for manifest, name in zip(
            manifests,
            ["deployment.yaml", "service.yaml", "servicemonitor.yaml"],
            strict=True,
        ):
            target = manifests_dir / name
            target.write_text(
                yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
            )
            detail(f"Rendered RHAIIS manifest written to {target}")

    step(
        f"Applying RHAIIS {plan.deployment.mode} deployment {plan.deployment.release_name} "
        f"in namespace {namespace}"
    )
    for manifest in manifests:
        run_command(
            [kubectl_cmd, "apply", "-f", "-"],
            input_text=yaml.safe_dump(manifest, sort_keys=False),
        )
    success(
        f"Applied Deployment {deployment_name}, Service {plan.deployment.release_name}, and ServiceMonitor in namespace {namespace}"
    )

    if verify:
        try:
            _verify_deployment(
                namespace, deployment_name, kubectl_cmd, verify_timeout_seconds
            )
        except CommandError as exc:
            raise CommandError(
                f"failed to verify RHAIIS deployment {deployment_name}: {exc}"
            ) from exc

    return manifests_dir.resolve() if manifests_dir else Path.cwd()
