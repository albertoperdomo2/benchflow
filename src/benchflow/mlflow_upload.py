from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

from .cluster import require_any_command, run_command
from .models import ResolvedRunPlan
from .ui import detail, step, success, warning


def _discover_grafana_base_url(namespace: str) -> str:
    kubectl_cmd = require_any_command("oc", "kubectl")
    for candidate_namespace in (
        f"{namespace}-grafana",
        namespace,
        "benchflow",
        "grafana",
        "openshift-monitoring",
    ):
        if not candidate_namespace:
            continue
        result = run_command(
            [
                kubectl_cmd,
                "get",
                "route",
                "-n",
                candidate_namespace,
                "-o",
                'jsonpath={range .items[*]}{.metadata.name}{"\\t"}{.spec.host}{"\\n"}{end}',
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            continue
        for line in result.stdout.splitlines():
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            name, host = parts
            if "grafana" in name and host:
                return f"https://{host}"
    return ""


def _build_grafana_url(
    plan: ResolvedRunPlan,
    run_id: str,
    benchmark_start_time: str,
    benchmark_end_time: str,
    grafana_base_url: str,
) -> str:
    if not grafana_base_url or not benchmark_start_time or not benchmark_end_time:
        return ""
    start_dt = datetime.fromisoformat(benchmark_start_time.replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(benchmark_end_time.replace("Z", "+00:00"))
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    return (
        f"{grafana_base_url}/d/benchflow"
        f"?var-run_id={run_id}"
        f"&var-namespace={plan.deployment.namespace}"
        f"&var-release={plan.deployment.release_name}"
        f"&from={start_ms}"
        f"&to={end_ms}"
    )


def _cleanup_dir_contents(
    directory: Path, *, preserve_names: set[str] | None = None
) -> None:
    if not directory.exists():
        return
    preserved = preserve_names or set()
    for item in directory.iterdir():
        if item.name in preserved:
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def _count_files(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for path in directory.rglob("*") if path.is_file())


def upload_to_mlflow(
    plan: ResolvedRunPlan,
    *,
    mlflow_run_id: str,
    benchmark_start_time: str,
    benchmark_end_time: str,
    artifacts_dir: Path,
    grafana_url: str = "",
) -> Path:
    import mlflow

    explicit_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if explicit_tracking_uri:
        mlflow.set_tracking_uri(explicit_tracking_uri)
    step(
        f"Uploading artifacts to MLflow run {mlflow_run_id or '<missing>'} "
        f"for release {plan.deployment.release_name}"
    )
    if not mlflow_run_id or not explicit_tracking_uri:
        warning(
            "Skipping MLflow upload because "
            + (
                "the run ID is missing"
                if not mlflow_run_id
                else "MLFLOW_TRACKING_URI is not configured"
            )
        )
        return artifacts_dir

    client = mlflow.tracking.MlflowClient()
    detail(f"MLflow tracking URI: {explicit_tracking_uri}")
    run = client.get_run(mlflow_run_id)
    experiment_id = str(run.info.experiment_id)
    detail(f"MLflow experiment ID: {experiment_id}")
    client.set_tag(mlflow_run_id, "mlflow_experiment_id", experiment_id)
    grafana_base_url = grafana_url or _discover_grafana_base_url(
        plan.deployment.namespace
    )
    full_grafana_url = _build_grafana_url(
        plan,
        mlflow_run_id,
        benchmark_start_time,
        benchmark_end_time,
        grafana_base_url,
    )
    if full_grafana_url:
        detail(f"Setting Grafana URL tag: {full_grafana_url}")
        client.set_tag(mlflow_run_id, "grafana_url", full_grafana_url)

    artifact_count = 0
    if artifacts_dir.exists():
        for child in sorted(artifacts_dir.iterdir()):
            if child.name == "benchmark":
                detail(
                    "Skipping benchmark workspace bundle because GuideLLM already "
                    "logs its own results, reports, and console output"
                )
                continue
            if child.is_dir():
                child_count = _count_files(child)
                if child_count == 0:
                    continue
                detail(
                    f"Uploading directory {child.name} ({child_count} file(s)) to MLflow"
                )
                client.log_artifacts(
                    mlflow_run_id, str(child), artifact_path=child.name
                )
                artifact_count += child_count
            elif child.is_file():
                detail(f"Uploading file {child.name} to MLflow")
                client.log_artifact(mlflow_run_id, str(child))
                artifact_count += 1
    detail(
        f"Uploaded {artifact_count} additional artifact file(s) from {artifacts_dir}"
    )
    success(f"MLflow upload complete for run {mlflow_run_id}")
    if artifacts_dir.exists():
        _cleanup_dir_contents(artifacts_dir, preserve_names={"platform-state"})
    return artifacts_dir
