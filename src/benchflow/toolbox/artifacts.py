from __future__ import annotations

import json
from pathlib import Path

from ..artifacts import collect_artifacts, collect_execution_logs
from ..contracts import ExecutionContext, ResolvedRunPlan, ValidationError
from ..remote_jobs import (
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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "remote_job_name": job_name,
                "remote_path": remote_path,
                "uploaded_to_mlflow": uploaded_to_mlflow,
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
) -> Path:
    if context.artifacts_dir is None:
        raise ValidationError("artifacts collection requires an artifacts directory")
    if plan.deployment.target.discovery == "static":
        return collect_artifacts(
            plan,
            artifacts_dir=context.artifacts_dir,
            execution_name=context.execution_name or "",
            include_execution_logs=True,
            include_workload=False,
            include_manifests=False,
        )
    if plan.target_cluster.enabled():
        execution_pod_count = 0
        if context.execution_name:
            execution_pod_count = collect_execution_logs(
                plan,
                artifacts_dir=context.artifacts_dir,
                execution_name=context.execution_name,
            )
        remote = run_remote_job(
            plan,
            job_kind="artifacts",
            args_builder=lambda job_name: [
                "artifacts",
                "collect",
                "--run-plan-json",
                remote_run_plan_json(plan),
                "--artifacts-dir",
                remote_job_artifacts_dir(job_name),
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
            context.artifacts_dir / "remote-target-artifacts.json",
            job_name=remote.job_name,
            remote_path=remote_job_artifacts_dir(remote.job_name),
            uploaded_to_mlflow=False,
        )
        return context.artifacts_dir
    return collect_artifacts(
        plan,
        artifacts_dir=context.artifacts_dir,
        execution_name=context.execution_name,
    )
