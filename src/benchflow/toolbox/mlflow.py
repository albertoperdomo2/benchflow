from __future__ import annotations

from pathlib import Path

from ..contracts import ExecutionContext, ResolvedRunPlan, ValidationError
from ..mlflow_upload import (
    initialize_mlflow_run,
    mark_mlflow_run,
    upload_artifact_directory_to_mlflow,
    upload_to_mlflow,
)


def init_plan_mlflow_run(
    plan: ResolvedRunPlan,
    *,
    execution_name: str = "",
) -> tuple[str, str]:
    return initialize_mlflow_run(plan, execution_name=execution_name)


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


def upload_artifact_directory(
    *,
    mlflow_run_id: str,
    artifacts_dir: Path,
    artifact_path_prefix: str = "",
    cleanup_after_upload: bool = False,
    preserve_names: set[str] | None = None,
    exclude_names: set[str] | None = None,
) -> Path:
    return upload_artifact_directory_to_mlflow(
        mlflow_run_id=mlflow_run_id,
        artifacts_dir=artifacts_dir,
        artifact_path_prefix=artifact_path_prefix,
        cleanup_after_upload=cleanup_after_upload,
        preserve_names=preserve_names,
        exclude_names=exclude_names,
    )


def finish_mlflow_run(
    *,
    mlflow_run_id: str,
    status: str,
    reason: str = "",
) -> None:
    mark_mlflow_run(
        mlflow_run_id=mlflow_run_id,
        status=status,
        reason=reason,
    )
