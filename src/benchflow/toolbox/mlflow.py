from __future__ import annotations

from ..contracts import ExecutionContext, ResolvedRunPlan, ValidationError
from ..mlflow_upload import upload_to_mlflow


def upload_plan_results(
    plan: ResolvedRunPlan,
    *,
    mlflow_run_id: str,
    benchmark_start_time: str,
    benchmark_end_time: str,
    context: ExecutionContext,
    grafana_url: str = "",
) -> None:
    if context.artifacts_dir is None:
        raise ValidationError("MLflow upload requires an artifacts directory")
    upload_to_mlflow(
        plan,
        mlflow_run_id=mlflow_run_id,
        benchmark_start_time=benchmark_start_time,
        benchmark_end_time=benchmark_end_time,
        artifacts_dir=context.artifacts_dir,
        grafana_url=grafana_url,
    )
