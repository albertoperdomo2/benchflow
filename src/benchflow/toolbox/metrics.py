from __future__ import annotations

import json
from pathlib import Path

from ..contracts import ExecutionContext, ResolvedRunPlan, ValidationError
from ..metrics import (
    collect_metrics,
    generate_mlflow_metrics_dashboard_report,
    serve_mlflow_metrics_dashboard,
)
from ..remote_jobs import (
    generate_remote_job_name,
    remote_job_artifacts_dir,
    remote_run_plan_json,
    run_remote_job,
)


def collect_plan_metrics(
    plan: ResolvedRunPlan,
    *,
    benchmark_start_time: str,
    benchmark_end_time: str,
    context: ExecutionContext,
    mlflow_run_id: str = "",
) -> Path:
    if context.artifacts_dir is None:
        raise ValidationError("metrics collection requires an artifacts directory")
    if plan.target_cluster.enabled():
        metrics_dir = context.artifacts_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_reference = metrics_dir / "remote-target-metrics.json"
        reference = {}
        if metrics_reference.exists():
            reference = json.loads(
                metrics_reference.read_text(encoding="utf-8") or "{}"
            )
        if str(reference.get("status") or "").strip() in {
            "materialized",
            "uploaded",
        }:
            return metrics_dir

        remote_root_override = ""
        artifacts_reference = context.artifacts_dir / "remote-target-artifacts.json"
        if artifacts_reference.exists():
            payload = json.loads(
                artifacts_reference.read_text(encoding="utf-8") or "{}"
            )
            remote_root_override = str(payload.get("remote_path") or "").strip()

        def remote_artifacts_root(job_name: str) -> str:
            return remote_root_override or remote_job_artifacts_dir(job_name)

        job_name = str(reference.get("remote_job_name") or "").strip()
        if not job_name:
            job_name = generate_remote_job_name(plan, "metrics")
        remote_path = str(reference.get("remote_path") or "").strip()
        if not remote_path:
            remote_path = f"{remote_artifacts_root(job_name)}/metrics"
        metrics_reference.write_text(
            json.dumps(
                {
                    "remote_job_name": job_name,
                    "remote_path": remote_path,
                    "uploaded_to_mlflow": False,
                    "status": "collecting",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        remote = run_remote_job(
            plan,
            job_kind="metrics",
            job_name=job_name,
            args_builder=lambda _job_name: [
                "metrics",
                "collect",
                "--run-plan-json",
                remote_run_plan_json(plan),
                "--benchmark-start-time",
                benchmark_start_time,
                "--benchmark-end-time",
                benchmark_end_time,
                "--artifacts-dir",
                str(Path(remote_path).parent),
            ],
            mount_results_pvc=True,
        )
        metrics_reference.write_text(
            json.dumps(
                {
                    "remote_job_name": remote.job_name,
                    "remote_path": remote_path,
                    "uploaded_to_mlflow": False,
                    "status": "collected",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return metrics_dir
    return collect_metrics(
        plan,
        benchmark_start_time=benchmark_start_time,
        benchmark_end_time=benchmark_end_time,
        artifacts_dir=context.artifacts_dir,
    )


def serve_metrics_dashboard(
    *,
    mlflow_run_ids: list[str],
    mlflow_tracking_uri: str = "",
) -> str:
    return serve_mlflow_metrics_dashboard(
        mlflow_run_ids=mlflow_run_ids,
        mlflow_tracking_uri=mlflow_tracking_uri or None,
    )


def generate_metrics_dashboard_report(
    *,
    mlflow_run_ids: list[str],
    mlflow_tracking_uri: str = "",
    output_file: str | Path,
) -> str:
    return generate_mlflow_metrics_dashboard_report(
        mlflow_run_ids=mlflow_run_ids,
        mlflow_tracking_uri=mlflow_tracking_uri or None,
        output_file=output_file,
    )
