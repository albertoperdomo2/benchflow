from __future__ import annotations

from pathlib import Path

from ..contracts import ExecutionContext, ResolvedRunPlan, ValidationError
from ..metrics import collect_metrics


def collect_plan_metrics(
    plan: ResolvedRunPlan,
    *,
    benchmark_start_time: str,
    benchmark_end_time: str,
    context: ExecutionContext,
) -> Path:
    if context.artifacts_dir is None:
        raise ValidationError("metrics collection requires an artifacts directory")
    return collect_metrics(
        plan,
        benchmark_start_time=benchmark_start_time,
        benchmark_end_time=benchmark_end_time,
        artifacts_dir=context.artifacts_dir,
    )
