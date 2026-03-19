from __future__ import annotations

import json
from pathlib import Path

from ..artifacts import collect_artifacts, collect_execution_logs
from ..contracts import ExecutionContext, ResolvedRunPlan, ValidationError
from ..remote_jobs import (
    REMOTE_ARTIFACTS_DIR,
    copy_remote_directory,
    remote_run_plan_json,
    run_remote_job,
)


def collect_plan_artifacts(
    plan: ResolvedRunPlan,
    *,
    context: ExecutionContext,
) -> Path:
    if context.artifacts_dir is None:
        raise ValidationError("artifacts collection requires an artifacts directory")
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
            args=[
                "artifacts",
                "collect",
                "--run-plan-json",
                remote_run_plan_json(plan),
                "--artifacts-dir",
                REMOTE_ARTIFACTS_DIR,
            ],
        )
        copy_remote_directory(
            plan,
            pod_name=remote.pod_name,
            remote_path=REMOTE_ARTIFACTS_DIR,
            local_dir=context.artifacts_dir,
        )
        metadata_path = context.artifacts_dir / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8") or "{}")
            metadata["execution_name"] = context.execution_name
            metadata["execution_pods"] = execution_pod_count
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return context.artifacts_dir
    return collect_artifacts(
        plan,
        artifacts_dir=context.artifacts_dir,
        execution_name=context.execution_name,
    )
