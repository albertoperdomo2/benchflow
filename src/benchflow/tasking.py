from __future__ import annotations

from pathlib import Path

from .models import ResolvedRunPlan, ValidationError


def write_stage_results(
    plan: ResolvedRunPlan,
    *,
    stage_download_path: Path,
    stage_deploy_path: Path,
    stage_benchmark_path: Path,
    stage_collect_path: Path,
    stage_cleanup_path: Path,
) -> None:
    stage_download_path.write_text(str(plan.stages.download).lower(), encoding="utf-8")
    stage_deploy_path.write_text(str(plan.stages.deploy).lower(), encoding="utf-8")
    stage_benchmark_path.write_text(
        str(plan.stages.benchmark).lower(), encoding="utf-8"
    )
    stage_collect_path.write_text(str(plan.stages.collect).lower(), encoding="utf-8")
    stage_cleanup_path.write_text(str(plan.stages.cleanup).lower(), encoding="utf-8")


def assert_task_status(
    task_name: str, task_status: str, allowed_statuses: list[str]
) -> None:
    if task_status in allowed_statuses:
        return
    raise ValidationError(
        f"task {task_name} finished with disallowed status: {task_status}"
    )
