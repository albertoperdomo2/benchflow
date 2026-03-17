from __future__ import annotations

from pathlib import Path

from ..artifacts import collect_artifacts
from ..contracts import ExecutionContext, ResolvedRunPlan, ValidationError


def collect_plan_artifacts(
    plan: ResolvedRunPlan,
    *,
    context: ExecutionContext,
) -> Path:
    if context.artifacts_dir is None:
        raise ValidationError("artifacts collection requires an artifacts directory")
    return collect_artifacts(
        plan,
        artifacts_dir=context.artifacts_dir,
        execution_name=context.execution_name,
    )
