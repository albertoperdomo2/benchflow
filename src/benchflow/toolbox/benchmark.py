from __future__ import annotations

import os
from pathlib import Path

from ..benchmark import benchmark_version_from_plan, generate_report, run_benchmark
from ..contracts import BenchmarkOutcome, ResolvedRunPlan
from ..ui import detail, step, success


def run_plan_benchmark(
    plan: ResolvedRunPlan,
    *,
    target_url: str,
    output_dir: Path | None = None,
    mlflow_tracking_uri: str | None = None,
    enable_mlflow: bool = True,
    extra_tags: dict[str, str] | None = None,
    execution_name: str = "",
) -> BenchmarkOutcome:
    step(f"Running {plan.benchmark.tool} benchmark against {target_url}")
    detail(
        "Rates: "
        + ",".join(str(rate) for rate in plan.benchmark.rates)
        + f", rate type: {plan.benchmark.rate_type}, backend: {plan.benchmark.backend_type}"
    )
    detail(f"Benchmark data: {plan.benchmark.data}")
    detail(
        f"MLflow: {'enabled' if enable_mlflow else 'disabled'}, "
        f"output dir: {str(output_dir) if output_dir is not None else 'not requested'}"
    )

    previous_execution_name = os.environ.get("EXECUTION_NAME")
    try:
        if execution_name:
            os.environ["EXECUTION_NAME"] = execution_name
        run_id, start_time, end_time = run_benchmark(
            plan=plan,
            target=target_url,
            output_dir=output_dir,
            mlflow_tracking_uri=mlflow_tracking_uri,
            enable_mlflow=enable_mlflow,
            extra_tags=extra_tags or {},
        )
    finally:
        if execution_name:
            if previous_execution_name is None:
                os.environ.pop("EXECUTION_NAME", None)
            else:
                os.environ["EXECUTION_NAME"] = previous_execution_name

    success(
        f"Benchmark finished. Start: {start_time}, end: {end_time}, "
        f"MLflow run: {run_id or 'not created'}"
    )
    return BenchmarkOutcome(run_id=run_id, start_time=start_time, end_time=end_time)


def generate_plan_report(
    *,
    plan: ResolvedRunPlan | None,
    json_path: Path | None,
    model_name: str | None,
    accelerator: str | None,
    version: str | None,
    tp: int | None,
    runtime_args: str | None,
    replicas: int | None,
    output_dir: Path | None,
    mlflow_run_ids: list[str] | None,
    mlflow_tracking_uri: str | None,
    versions: list[str] | None,
    version_overrides: dict[str, str],
    additional_csv_files: list[str] | tuple[str, ...] | None,
) -> Path:
    model = model_name or (plan.model.name if plan is not None else None)
    resolved_version = version or (
        benchmark_version_from_plan(plan) if plan is not None else None
    )
    tp_size = (
        tp
        if tp is not None
        else (plan.deployment.runtime.tensor_parallelism if plan is not None else 1)
    )
    resolved_runtime_args = runtime_args or (
        " ".join(plan.deployment.runtime.vllm_args) if plan is not None else ""
    )
    resolved_replicas = (
        replicas
        if replicas is not None
        else (plan.deployment.runtime.replicas if plan is not None else 1)
    )

    return generate_report(
        json_path=json_path,
        model=model,
        accelerator=accelerator,
        version=resolved_version,
        tp_size=tp_size,
        runtime_args=resolved_runtime_args,
        output_dir=output_dir,
        replicas=resolved_replicas,
        mlflow_run_ids=mlflow_run_ids,
        mlflow_tracking_uri=mlflow_tracking_uri,
        versions=versions,
        version_overrides=version_overrides,
        additional_csv_files=additional_csv_files,
    )
