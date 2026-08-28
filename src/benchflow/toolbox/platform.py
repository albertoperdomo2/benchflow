from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from ..cleanup import cleanup_llmd, cleanup_rhaiis, cleanup_rhoai
from ..cluster import require_any_command, resolve_target_base_url, use_kubeconfig
from ..contracts import ExecutionContext, ResolvedRunPlan, ValidationError
from ..deploy import deploy_llmd, deploy_rhaiis, deploy_rhoai
from ..platform_state import (
    clear_cluster_platform_state,
    load_cluster_platform_state,
    persist_cluster_platform_state,
    platform_prepare_lock,
    setup_key_for_plan,
)
from ..setup import (
    discover_rhoai_mlflow_version,
    llmd_platform_present,
    load_setup_state,
    normalize_rhoai_platform_version,
    reset_llmd_platform,
    reset_rhoai_platform,
    resolve_llmd_repo_head,
    rhoai_platform_present,
    setup_llmd,
    setup_rhoai,
)
from ..ui import detail, step


def _write_state_path(state_path: Path | None, state: dict[str, Any]) -> None:
    if state_path is None:
        return
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _legacy_cluster_platform_state(
    plan: ResolvedRunPlan, kubectl_cmd: str
) -> dict[str, Any]:
    has_llmd = llmd_platform_present(kubectl_cmd)
    has_rhoai = rhoai_platform_present(kubectl_cmd)
    if has_llmd and has_rhoai:
        return {
            "installed_key": "",
            "setup_state": {
                "platform": "mixed",
                "repo_url": plan.deployment.repo_url,
                "repo_ref": plan.deployment.repo_ref,
                "gateway": plan.deployment.gateway,
            },
        }
    if has_rhoai:
        version = normalize_rhoai_platform_version(discover_rhoai_mlflow_version())
        return {
            "installed_key": f"rhoai:{version}",
            "setup_state": {
                "platform": "rhoai",
                "platform_version": version,
            },
        }
    if has_llmd:
        return {
            "installed_key": "",
            "setup_state": {
                "platform": "llm-d",
                "repo_url": plan.deployment.repo_url,
                "repo_ref": plan.deployment.repo_ref,
                "gateway": plan.deployment.gateway,
            },
        }
    return {"installed_key": "", "setup_state": {}}


def _reset_platform_for_state(
    plan: ResolvedRunPlan,
    setup_state: dict[str, Any],
    *,
    workspace_dir: Path | None = None,
) -> None:
    platform = str(setup_state.get("platform") or "").strip()
    if platform in {"", "unknown"}:
        return
    if platform in {"mixed", "llm-d"}:
        reset_llmd_platform(
            repo_url=str(
                setup_state.get("repo_url") or plan.deployment.repo_url
            ).strip(),
            repo_ref=str(
                setup_state.get("repo_ref") or plan.deployment.repo_ref
            ).strip(),
            gateway=str(
                setup_state.get("gateway") or plan.deployment.gateway or "istio"
            ).strip(),
            workspace_dir=workspace_dir,
        )
    if platform in {"mixed", "rhoai"}:
        reset_rhoai_platform()


def _ensure_llmd_main_repo_head(
    plan: ResolvedRunPlan,
    kubectl_cmd: str,
    installed_key: str,
    setup_state: dict[str, Any],
    *,
    workspace_dir: Path | None = None,
) -> dict[str, Any]:
    if plan.deployment.platform != "llm-d":
        return setup_state
    if str(plan.deployment.repo_ref or "").strip() != "main":
        return setup_state
    if str(setup_state.get("repo_head") or "").strip():
        return setup_state

    repo_head = resolve_llmd_repo_head(
        repo_url=plan.deployment.repo_url,
        repo_ref=plan.deployment.repo_ref,
        workspace_dir=workspace_dir,
    )
    setup_state = {
        **setup_state,
        "platform": "llm-d",
        "repo_url": plan.deployment.repo_url,
        "repo_ref": plan.deployment.repo_ref,
        "repo_head": repo_head,
        "gateway": plan.deployment.gateway,
    }
    persist_cluster_platform_state(
        kubectl_cmd,
        plan.deployment.namespace,
        {
            "installed_key": installed_key,
            "setup_state": setup_state,
        },
    )
    return setup_state


def resolve_target_url(
    plan: ResolvedRunPlan,
    *,
    target_url: str | None = None,
    endpoint_path: str | None = None,
) -> tuple[str, str]:
    with use_kubeconfig(plan.target_cluster.kubeconfig):
        base_url = target_url or resolve_target_base_url(
            plan.deployment.target, plan.deployment.namespace
        )
    resolved_path = endpoint_path or plan.deployment.target.path
    return base_url, resolved_path


def setup_platform(
    plan: ResolvedRunPlan,
    *,
    context: ExecutionContext | None = None,
) -> dict[str, Any]:
    workspace_dir = context.workspace_dir if context is not None else None
    state_path = context.state_path if context is not None else None

    with use_kubeconfig(plan.target_cluster.kubeconfig):
        if plan.deployment.platform in {"llm-d", "rhoai"}:
            kubectl_cmd = require_any_command("oc", "kubectl")
            requested_key = setup_key_for_plan(plan)
            holder_identity = (
                context.execution_name if context is not None else ""
            ) or f"{plan.metadata.name}-{plan.deployment.release_name}"
            with platform_prepare_lock(
                kubectl_cmd,
                plan.deployment.namespace,
                holder_identity=holder_identity,
            ):
                cluster_state = load_cluster_platform_state(
                    kubectl_cmd, plan.deployment.namespace
                )
                installed_key = str(cluster_state.get("installed_key") or "").strip()
                installed_setup_state = dict(cluster_state.get("setup_state") or {})

                if not installed_key and not installed_setup_state:
                    legacy_state = _legacy_cluster_platform_state(plan, kubectl_cmd)
                    legacy_setup_state = dict(legacy_state.get("setup_state") or {})
                    if legacy_setup_state:
                        cluster_state = legacy_state
                        installed_key = str(
                            legacy_state.get("installed_key") or ""
                        ).strip()
                        installed_setup_state = legacy_setup_state
                        if installed_key:
                            persist_cluster_platform_state(
                                kubectl_cmd,
                                plan.deployment.namespace,
                                {
                                    "installed_key": installed_key,
                                    "setup_state": installed_setup_state,
                                },
                            )

                if installed_key == requested_key and installed_setup_state:
                    installed_setup_state = _ensure_llmd_main_repo_head(
                        plan,
                        kubectl_cmd,
                        installed_key,
                        installed_setup_state,
                        workspace_dir=workspace_dir,
                    )
                    detail(
                        f"Platform prerequisites already match setup key {requested_key}"
                    )
                    _write_state_path(state_path, installed_setup_state)
                    return installed_setup_state

                if installed_setup_state:
                    current_platform = str(
                        installed_setup_state.get("platform") or "unknown"
                    )
                    step(
                        f"Resetting current platform prerequisites ({current_platform}) before switching to {requested_key}"
                    )
                    _reset_platform_for_state(
                        plan,
                        installed_setup_state,
                        workspace_dir=workspace_dir,
                    )
                    clear_cluster_platform_state(kubectl_cmd, plan.deployment.namespace)

                if plan.deployment.platform == "llm-d":
                    state = setup_llmd(
                        plan,
                        workspace_dir=workspace_dir,
                        state_path=state_path,
                    )
                else:
                    state = setup_rhoai(plan, state_path=state_path)

                persist_cluster_platform_state(
                    kubectl_cmd,
                    plan.deployment.namespace,
                    {"installed_key": requested_key, "setup_state": state},
                )
                return state
        detail(
            f"No platform setup implemented for {plan.deployment.platform}; continuing without changes"
        )
        return {}


def teardown_platform(
    plan: ResolvedRunPlan,
    state: dict[str, Any],
    *,
    context: ExecutionContext | None = None,
) -> None:
    workspace_dir = context.workspace_dir if context is not None else None
    if not state and context is not None and context.state_path is not None:
        state = load_setup_state(context.state_path)

    with use_kubeconfig(plan.target_cluster.kubeconfig):
        if plan.deployment.platform in {"llm-d", "rhoai"}:
            kubectl_cmd = require_any_command("oc", "kubectl")
            holder_identity = (
                context.execution_name if context is not None else ""
            ) or f"{plan.metadata.name}-{plan.deployment.release_name}"
            with platform_prepare_lock(
                kubectl_cmd,
                plan.deployment.namespace,
                holder_identity=holder_identity,
            ):
                cluster_state = load_cluster_platform_state(
                    kubectl_cmd, plan.deployment.namespace
                )
                current_state = dict(cluster_state.get("setup_state") or {}) or dict(
                    state or {}
                )
                if not current_state:
                    legacy_state = _legacy_cluster_platform_state(plan, kubectl_cmd)
                    current_state = dict(legacy_state.get("setup_state") or {})
                platform = str(current_state.get("platform") or "").strip()
                if platform in {"llm-d", "rhoai", "mixed"}:
                    _reset_platform_for_state(
                        plan,
                        current_state,
                        workspace_dir=workspace_dir,
                    )
                    clear_cluster_platform_state(kubectl_cmd, plan.deployment.namespace)
                    return
                detail("No shared platform state found; skipping platform teardown")
                return
        detail(
            f"No platform teardown implemented for {plan.deployment.platform}; cleanup removed only scenario resources"
        )


def deploy_platform(
    plan: ResolvedRunPlan,
    *,
    context: ExecutionContext | None = None,
    skip_if_exists: bool = True,
    verify: bool = True,
    verify_timeout_seconds: int = 1800,
) -> Path:
    workspace_dir = context.workspace_dir if context is not None else None
    manifests_dir = context.manifests_dir if context is not None else None
    execution_name = context.execution_name if context is not None else ""

    with use_kubeconfig(plan.target_cluster.kubeconfig):
        if plan.deployment.platform == "llm-d":
            return deploy_llmd(
                plan,
                workspace_dir=workspace_dir,
                manifests_dir=manifests_dir,
                execution_name=execution_name,
                skip_if_exists=skip_if_exists,
                verify=verify,
                verify_timeout_seconds=verify_timeout_seconds,
            )
        if plan.deployment.platform == "rhoai":
            return deploy_rhoai(
                plan,
                manifests_dir=manifests_dir,
                skip_if_exists=skip_if_exists,
                verify=verify,
                verify_timeout_seconds=verify_timeout_seconds,
            )
        if plan.deployment.platform == "rhaiis":
            return deploy_rhaiis(
                plan,
                manifests_dir=manifests_dir,
                skip_if_exists=skip_if_exists,
                verify=verify,
                verify_timeout_seconds=verify_timeout_seconds,
            )
        raise ValidationError(
            f"unsupported deployment platform: {plan.deployment.platform}"
        )


def cleanup_deployment(
    plan: ResolvedRunPlan,
    *,
    wait_for_deletion: bool,
    timeout_seconds: int,
    skip_if_not_exists: bool,
) -> None:
    with use_kubeconfig(plan.target_cluster.kubeconfig):
        if plan.deployment.platform == "llm-d":
            cleanup_llmd(
                plan,
                wait_for_deletion=wait_for_deletion,
                timeout_seconds=timeout_seconds,
                skip_if_not_exists=skip_if_not_exists,
            )
            return
        if plan.deployment.platform == "rhoai":
            cleanup_rhoai(
                plan,
                wait_for_deletion=wait_for_deletion,
                timeout_seconds=timeout_seconds,
                skip_if_not_exists=skip_if_not_exists,
            )
            return
        if plan.deployment.platform == "rhaiis":
            cleanup_rhaiis(
                plan,
                wait_for_deletion=wait_for_deletion,
                timeout_seconds=timeout_seconds,
                skip_if_not_exists=skip_if_not_exists,
            )
            return
        raise ValidationError(
            f"unsupported deployment platform: {plan.deployment.platform}"
        )
