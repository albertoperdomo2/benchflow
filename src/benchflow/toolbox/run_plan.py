from __future__ import annotations

import json

from ..contracts import ExecutionContext, ResolvedRunPlan, ValidationError
from ..setup import load_setup_state
from ..tasking import write_stage_results
from ..ui import detail, emit, step, success
from .platform import cleanup_deployment, teardown_platform


def resolve_run_plan_stages(
    plan: ResolvedRunPlan,
    *,
    stage_download_path,
    stage_deploy_path,
    stage_benchmark_path,
    stage_collect_path,
    stage_cleanup_path,
) -> None:
    if plan.benchmark.tool != "guidellm":
        raise ValidationError(
            f"unsupported benchmark tool: {plan.benchmark.tool}; only guidellm is implemented"
        )
    step(
        f"Resolved RunPlan for {plan.metadata.name} "
        f"({plan.deployment.platform}/{plan.deployment.mode})"
    )
    detail(
        "Stages: "
        f"download={plan.stages.download}, deploy={plan.stages.deploy}, "
        f"benchmark={plan.stages.benchmark}, collect={plan.stages.collect}, "
        f"cleanup={plan.stages.cleanup}"
    )
    emit(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    write_stage_results(
        plan,
        stage_download_path=stage_download_path,
        stage_deploy_path=stage_deploy_path,
        stage_benchmark_path=stage_benchmark_path,
        stage_collect_path=stage_collect_path,
        stage_cleanup_path=stage_cleanup_path,
    )
    success("RunPlan resolved and stage outputs written")


def cleanup_run_plan(
    plan: ResolvedRunPlan,
    *,
    context: ExecutionContext | None = None,
    teardown: bool = True,
    wait_for_deletion: bool = True,
    timeout_seconds: int = 600,
    skip_if_not_exists: bool = True,
) -> None:
    setup_state_path = context.state_path if context is not None else None
    setup_state = load_setup_state(setup_state_path)
    cleanup_error: Exception | None = None

    try:
        cleanup_deployment(
            plan,
            wait_for_deletion=wait_for_deletion,
            timeout_seconds=timeout_seconds,
            skip_if_not_exists=skip_if_not_exists,
        )
    except Exception as exc:  # pragma: no cover - cleanup fallback path
        cleanup_error = exc

    if teardown:
        teardown_platform(plan, setup_state, context=context)

    if cleanup_error is not None:
        raise cleanup_error

    if setup_state_path is not None and setup_state_path.exists():
        setup_state_path.unlink()
        parent = setup_state_path.parent
        if parent.exists():
            try:
                next(parent.iterdir())
            except StopIteration:
                try:
                    parent.rmdir()
                except OSError:
                    pass
