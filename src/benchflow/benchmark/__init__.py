from __future__ import annotations

import os
from pathlib import Path

import mlflow
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

from ..models import ResolvedRunPlan, ValidationError
from . import aiperf as aiperf_backend
from . import guidellm as guidellm_backend
from .common import BenchmarkRunFailed, benchmark_version_from_plan
from .run_report import generate_run_report as generate_guidellm_run_report


def _list_run_artifact_paths(artifact_uri: str) -> set[str]:
    repo = get_artifact_repository(artifact_uri)
    pending = [""]
    discovered: set[str] = set()
    while pending:
        current = pending.pop()
        for entry in repo.list_artifacts(current):
            if entry.is_dir:
                pending.append(entry.path)
            else:
                discovered.add(entry.path)
    return discovered


def _detect_mlflow_report_tool(
    mlflow_run_ids: list[str], mlflow_tracking_uri: str | None
) -> str:
    tracking_uri = mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return "guidellm"
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    detected: set[str] = set()
    for run_id in mlflow_run_ids:
        run = client.get_run(run_id)
        artifact_paths = _list_run_artifact_paths(run.info.artifact_uri)
        if any(path.endswith("profile_export_aiperf.json") for path in artifact_paths):
            detected.add("aiperf")
        if any(
            path.endswith("benchmark_output.json") or "benchmark_output_rate_" in path
            for path in artifact_paths
        ):
            detected.add("guidellm")
    if len(detected) > 1:
        raise ValidationError(
            "mixed benchmark tools are not supported in one comparison report"
        )
    return next(iter(detected), "guidellm")


def run_benchmark(
    *,
    plan: ResolvedRunPlan,
    target: str | None = None,
    output_dir: Path | None = None,
    mlflow_tracking_uri: str | None = None,
    enable_mlflow: bool = True,
    extra_tags: dict[str, str] | None = None,
) -> tuple[str, str, str]:
    if plan.benchmark.tool == "guidellm":
        return guidellm_backend.run_benchmark(
            plan=plan,
            target=target,
            output_dir=output_dir,
            mlflow_tracking_uri=mlflow_tracking_uri,
            enable_mlflow=enable_mlflow,
            extra_tags=extra_tags,
        )
    if plan.benchmark.tool == "aiperf":
        return aiperf_backend.run_benchmark(
            plan=plan,
            target=target,
            output_dir=output_dir,
            mlflow_tracking_uri=mlflow_tracking_uri,
            enable_mlflow=enable_mlflow,
            extra_tags=extra_tags,
        )
    raise ValidationError(f"unsupported benchmark tool: {plan.benchmark.tool}")


def generate_report(
    *,
    plan: ResolvedRunPlan | None = None,
    json_path: Path | None = None,
    model: str | None = None,
    accelerator: str | None = None,
    version: str | None = None,
    tp_size: int = 1,
    runtime_args: str = "",
    output_dir: Path | None = None,
    output_file: Path | None = None,
    replicas: int = 1,
    mlflow_run_ids: list[str] | None = None,
    mlflow_tracking_uri: str | None = None,
    versions: list[str] | None = None,
    version_overrides: dict[str, str] | None = None,
    additional_csv_files: list[str] | None = None,
    notes: list[str] | None = None,
    repeat_section_legends: bool = False,
) -> Path:
    tool = (
        plan.benchmark.tool
        if plan is not None
        else (
            _detect_mlflow_report_tool(mlflow_run_ids, mlflow_tracking_uri)
            if mlflow_run_ids
            else "guidellm"
        )
    )
    if tool == "aiperf":
        if json_path is not None:
            raise ValidationError(
                "AIPerf comparison reports do not support --json-path; use --mlflow-run-ids"
            )
        if additional_csv_files:
            raise ValidationError(
                "AIPerf comparison reports do not support --additional-csv"
            )
        return aiperf_backend.generate_report(
            mlflow_run_ids=mlflow_run_ids or [],
            mlflow_tracking_uri=mlflow_tracking_uri,
            output_dir=output_dir,
            output_file=output_file,
            version_overrides=version_overrides,
            notes=notes,
        )
    return guidellm_backend.generate_report(
        json_path=json_path,
        model=model,
        accelerator=accelerator,
        version=version,
        tp_size=tp_size,
        runtime_args=runtime_args,
        output_dir=output_dir,
        output_file=output_file,
        replicas=replicas,
        mlflow_run_ids=mlflow_run_ids,
        mlflow_tracking_uri=mlflow_tracking_uri,
        versions=versions,
        version_overrides=version_overrides,
        additional_csv_files=additional_csv_files,
        notes=notes,
        repeat_section_legends=repeat_section_legends,
    )


def generate_run_report(
    *,
    artifacts_dir: Path,
    output_dir: Path | None = None,
    output_file: Path | None = None,
    columns: int = 3,
) -> Path:
    if aiperf_backend.is_aiperf_artifacts_dir(artifacts_dir):
        return aiperf_backend.generate_run_report(
            artifacts_dir=artifacts_dir,
            output_dir=output_dir,
            output_file=output_file,
        )
    return generate_guidellm_run_report(
        artifacts_dir=artifacts_dir,
        output_dir=output_dir,
        output_file=output_file,
        columns=columns,
    )


__all__ = [
    "BenchmarkRunFailed",
    "benchmark_version_from_plan",
    "generate_report",
    "generate_run_report",
    "run_benchmark",
]
