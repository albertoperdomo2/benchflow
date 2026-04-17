from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from ..benchmark import (
    BenchmarkRunFailed,
    benchmark_version_from_plan,
    generate_report,
    generate_run_report,
    run_benchmark,
)
from ..contracts import BenchmarkOutcome, ResolvedRunPlan
from ..remote_jobs import (
    RemoteJobFailed,
    copy_remote_results_directory,
    remote_job_benchmark_dir,
    remote_run_plan_json,
    run_remote_job,
)
from ..ui import detail, step, success


def _read_optional_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


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
    if plan.target_cluster.enabled():
        step(
            f"Running {plan.benchmark.tool} benchmark against {target_url} "
            "via a remote target-cluster Job"
        )
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

        staging_dir = (
            Path(output_dir).resolve()
            if output_dir is not None
            else Path(tempfile.mkdtemp(prefix="benchflow-remote-benchmark-"))
        )
        try:
            try:
                remote = run_remote_job(
                    plan,
                    job_kind="benchmark",
                    args_builder=lambda job_name: [
                        "benchmark",
                        "run",
                        "--run-plan-json",
                        remote_run_plan_json(plan),
                        "--output-dir",
                        remote_job_benchmark_dir(job_name),
                        "--mlflow-run-id-output",
                        f"{remote_job_benchmark_dir(job_name)}/.mlflow-run-id",
                        "--benchmark-start-time-output",
                        f"{remote_job_benchmark_dir(job_name)}/.benchmark-start-time",
                        "--benchmark-end-time-output",
                        f"{remote_job_benchmark_dir(job_name)}/.benchmark-end-time",
                        *(["--target-url", target_url] if target_url else []),
                        *(
                            ["--mlflow-tracking-uri", mlflow_tracking_uri]
                            if mlflow_tracking_uri
                            else []
                        ),
                        *(["--no-mlflow"] if not enable_mlflow else []),
                        *(
                            ["--execution-name", execution_name]
                            if execution_name
                            else []
                        ),
                        *[
                            item
                            for key, value in sorted((extra_tags or {}).items())
                            for item in ("--tag", f"{key}={value}")
                        ],
                    ],
                    timeout_seconds=max(plan.benchmark.max_seconds + 900, 7200),
                    mount_results_pvc=True,
                )
            except RemoteJobFailed as exc:
                try:
                    copy_remote_results_directory(
                        plan,
                        remote_path=remote_job_benchmark_dir(exc.job_name),
                        local_dir=staging_dir,
                    )
                except Exception as copy_exc:  # noqa: BLE001
                    detail(
                        "Failed to copy remote benchmark outputs after failure: "
                        f"{copy_exc}"
                    )
                raise BenchmarkRunFailed(
                    str(exc),
                    run_id=_read_optional_text(staging_dir / ".mlflow-run-id"),
                    start_time=_read_optional_text(
                        staging_dir / ".benchmark-start-time"
                    ),
                    end_time=_read_optional_text(staging_dir / ".benchmark-end-time"),
                ) from exc

            copy_remote_results_directory(
                plan,
                remote_path=remote_job_benchmark_dir(remote.job_name),
                local_dir=staging_dir,
            )
            outcome = BenchmarkOutcome(
                run_id=_read_optional_text(staging_dir / ".mlflow-run-id"),
                start_time=_read_optional_text(staging_dir / ".benchmark-start-time"),
                end_time=_read_optional_text(staging_dir / ".benchmark-end-time"),
            )
            success(
                f"Benchmark finished. Start: {outcome.start_time}, end: {outcome.end_time}, "
                f"MLflow run: {outcome.run_id or 'not created'}"
            )
            return outcome
        finally:
            if output_dir is None:
                shutil.rmtree(staging_dir, ignore_errors=True)

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
    output_file: Path | None,
    mlflow_run_ids: list[str] | None,
    mlflow_tracking_uri: str | None,
    versions: list[str] | None,
    version_overrides: dict[str, str],
    additional_csv_files: list[str] | tuple[str, ...] | None,
    notes: list[str] | tuple[str, ...] | None,
    repeat_section_legends: bool = False,
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
        plan=plan,
        json_path=json_path,
        model=model,
        accelerator=accelerator,
        version=resolved_version,
        tp_size=tp_size,
        runtime_args=resolved_runtime_args,
        output_dir=output_dir,
        output_file=output_file,
        replicas=resolved_replicas,
        mlflow_run_ids=mlflow_run_ids,
        mlflow_tracking_uri=mlflow_tracking_uri,
        versions=versions,
        version_overrides=version_overrides,
        additional_csv_files=additional_csv_files,
        notes=list(notes or []),
        repeat_section_legends=repeat_section_legends,
    )


def generate_artifacts_run_report(
    *,
    artifacts_dir: Path,
    output_dir: Path | None,
    output_file: Path | None,
    columns: int = 3,
) -> Path:
    return generate_run_report(
        artifacts_dir=artifacts_dir,
        output_dir=output_dir,
        output_file=output_file,
        columns=columns,
    )
