from __future__ import annotations

import json
from pathlib import Path

from ..artifacts import collect_artifacts, collect_execution_logs
from ..contracts import ExecutionContext, ResolvedRunPlan, ValidationError
from ..remote_jobs import (
    generate_remote_job_name,
    remote_job_artifacts_dir,
    remote_run_plan_json,
    run_remote_job,
)


def _write_remote_reference(
    path: Path,
    *,
    job_name: str,
    remote_path: str,
    uploaded_to_mlflow: bool,
    status: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "remote_job_name": job_name,
                "remote_path": remote_path,
                "uploaded_to_mlflow": uploaded_to_mlflow,
                "status": status,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def collect_plan_artifacts(
    plan: ResolvedRunPlan,
    *,
    context: ExecutionContext,
    mlflow_run_id: str = "",
    benchmark_start_time: str = "",
    benchmark_end_time: str = "",
) -> Path:
    if context.artifacts_dir is None:
        raise ValidationError("artifacts collection requires an artifacts directory")
    if plan.target_cluster.enabled():
        reference_path = context.artifacts_dir / "remote-target-artifacts.json"
        reference = {}
        if reference_path.exists():
            reference = json.loads(reference_path.read_text(encoding="utf-8") or "{}")
        reference_status = str(reference.get("status") or "").strip()
        if reference_status in {"materialized", "uploaded"}:
            return context.artifacts_dir

        execution_pod_count = 0
        if context.execution_name:
            execution_pod_count = collect_execution_logs(
                plan,
                artifacts_dir=context.artifacts_dir,
                execution_name=context.execution_name,
            )
        job_name = str(reference.get("remote_job_name") or "").strip()
        if not job_name:
            job_name = generate_remote_job_name(plan, "artifacts")
        remote_path = str(reference.get("remote_path") or "").strip()
        if not remote_path:
            remote_path = remote_job_artifacts_dir(job_name)
        _write_remote_reference(
            reference_path,
            job_name=job_name,
            remote_path=remote_path,
            uploaded_to_mlflow=False,
            status="collecting",
        )

        remote = run_remote_job(
            plan,
            job_kind="artifacts",
            job_name=job_name,
            args_builder=lambda _job_name: [
                "artifacts",
                "collect",
                "--run-plan-json",
                remote_run_plan_json(plan),
                *(
                    ["--benchmark-start-time", benchmark_start_time]
                    if benchmark_start_time
                    else []
                ),
                *(
                    ["--benchmark-end-time", benchmark_end_time]
                    if benchmark_end_time
                    else []
                ),
                "--artifacts-dir",
                remote_path,
            ],
            mount_results_pvc=True,
        )
        metadata_path = context.artifacts_dir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8") or "{}")
        metadata["execution_name"] = context.execution_name
        metadata["execution_pods"] = execution_pod_count
        metadata["target_artifacts_job"] = remote.job_name
        metadata["target_artifacts_uploaded_to_mlflow"] = False
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        _write_remote_reference(
            reference_path,
            job_name=remote.job_name,
            remote_path=remote_path,
            uploaded_to_mlflow=False,
            status="collected",
        )
        return context.artifacts_dir
    if plan.deployment.target.discovery == "static" and not plan.stages.deploy:
        return collect_artifacts(
            plan,
            artifacts_dir=context.artifacts_dir,
            execution_name=context.execution_name or "",
            include_execution_logs=True,
            include_workload=False,
            include_manifests=False,
            benchmark_start_time=benchmark_start_time,
            benchmark_end_time=benchmark_end_time,
        )
    return collect_artifacts(
        plan,
        artifacts_dir=context.artifacts_dir,
        execution_name=context.execution_name,
        benchmark_start_time=benchmark_start_time,
        benchmark_end_time=benchmark_end_time,
    )
