from __future__ import annotations

from pathlib import Path
from typing import Any

from ..cleanup import cleanup_llmd, cleanup_rhoai
from ..cluster import resolve_target_base_url
from ..contracts import ExecutionContext, ResolvedRunPlan, ValidationError
from ..deploy import deploy_llmd, deploy_rhoai
from ..setup import (
    load_setup_state,
    setup_llmd,
    setup_rhoai,
    teardown_llmd,
    teardown_rhoai,
)
from ..ui import detail


def resolve_target_url(
    plan: ResolvedRunPlan,
    *,
    target_url: str | None = None,
    endpoint_path: str | None = None,
) -> tuple[str, str]:
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

    if plan.deployment.platform == "llm-d":
        return setup_llmd(plan, workspace_dir=workspace_dir, state_path=state_path)
    if plan.deployment.platform == "rhoai":
        return setup_rhoai(plan, state_path=state_path)
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

    if plan.deployment.platform == "llm-d":
        teardown_llmd(plan, state, workspace_dir=workspace_dir)
        return
    if plan.deployment.platform == "rhoai":
        teardown_rhoai(plan, state)
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
    verify_timeout_seconds: int = 900,
) -> Path:
    workspace_dir = context.workspace_dir if context is not None else None
    manifests_dir = context.manifests_dir if context is not None else None
    execution_name = context.execution_name if context is not None else ""

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
    raise ValidationError(
        f"unsupported deployment platform: {plan.deployment.platform}"
    )
