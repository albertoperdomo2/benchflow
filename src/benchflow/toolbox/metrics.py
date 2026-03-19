from __future__ import annotations

from pathlib import Path

from ..contracts import ExecutionContext, ResolvedRunPlan, ValidationError
from ..metrics import collect_metrics
from ..remote_jobs import (
    REMOTE_ARTIFACTS_DIR,
    copy_remote_directory,
    remote_run_plan_json,
    run_remote_job,
)


def collect_plan_metrics(
    plan: ResolvedRunPlan,
    *,
    benchmark_start_time: str,
    benchmark_end_time: str,
    context: ExecutionContext,
) -> Path:
    if context.artifacts_dir is None:
        raise ValidationError("metrics collection requires an artifacts directory")
    if plan.target_cluster.enabled():
        remote = run_remote_job(
            plan,
            job_kind="metrics",
            args=[
                "metrics",
                "collect",
                "--run-plan-json",
                remote_run_plan_json(plan),
                "--benchmark-start-time",
                benchmark_start_time,
                "--benchmark-end-time",
                benchmark_end_time,
                "--artifacts-dir",
                REMOTE_ARTIFACTS_DIR,
            ],
        )
        metrics_dir = context.artifacts_dir / "metrics"
        copy_remote_directory(
            plan,
            pod_name=remote.pod_name,
            remote_path=f"{REMOTE_ARTIFACTS_DIR}/metrics",
            local_dir=metrics_dir,
        )
        return metrics_dir
    return collect_metrics(
        plan,
        benchmark_start_time=benchmark_start_time,
        benchmark_end_time=benchmark_end_time,
        artifacts_dir=context.artifacts_dir,
    )
