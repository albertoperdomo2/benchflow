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
    experiment_id: str,
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
        f"{grafana_base_url}/d/benchflow-archive"
        f"?var-experiment_id={experiment_id}"
        f"&var-run_id={run_id}"
        f"&var-namespace={plan.deployment.namespace}"
        f"&var-release={plan.deployment.release_name}"
        f"&from={start_ms}"
        f"&to={end_ms}"
    )


def _build_metrics_archive_url(
    *, experiment_id: str, run_id: str, public_base_url: str
) -> str:
    if not public_base_url:
        return ""
    return (
        f"{public_base_url.rstrip('/')}/{experiment_id}/{run_id}/artifacts/metrics/raw"
    )


def _cleanup_dir_contents(directory: Path) -> None:
    if not directory.exists():
        return
    for item in directory.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


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
        _cleanup_dir_contents(artifacts_dir)
        return artifacts_dir

    try:
        client = mlflow.tracking.MlflowClient()
        detail(f"MLflow tracking URI: {explicit_tracking_uri}")
        run = client.get_run(mlflow_run_id)
        experiment_id = str(run.info.experiment_id)
        detail(f"MLflow experiment ID: {experiment_id}")
        client.set_tag(mlflow_run_id, "mlflow_experiment_id", experiment_id)
        public_base_url = os.environ.get("BENCHFLOW_ARCHIVE_BASE_URL", "").strip()
        archive_url = _build_metrics_archive_url(
            experiment_id=experiment_id,
            run_id=mlflow_run_id,
            public_base_url=public_base_url,
        )
        if not archive_url:
            warning(
                "BENCHFLOW_ARCHIVE_BASE_URL is not configured; "
                "the archive dashboard URL will not be recorded"
            )
        grafana_base_url = grafana_url or _discover_grafana_base_url(
            plan.deployment.namespace
        )
        full_grafana_url = _build_grafana_url(
            plan,
            experiment_id,
            mlflow_run_id,
            benchmark_start_time,
            benchmark_end_time,
            grafana_base_url,
        )
        if archive_url:
            detail(f"Setting metrics archive URL tag: {archive_url}")
            client.set_tag(mlflow_run_id, "metrics_archive_url", archive_url)
        if full_grafana_url:
            detail(f"Setting Grafana URL tag: {full_grafana_url}")
            client.set_tag(mlflow_run_id, "grafana_url", full_grafana_url)

        artifact_count = 0
        if artifacts_dir.exists():
            for file_path in sorted(
                path for path in artifacts_dir.rglob("*") if path.is_file()
            ):
                relative_path = file_path.relative_to(artifacts_dir)
                artifact_path = (
                    str(relative_path.parent)
                    if str(relative_path.parent) != "."
                    else None
                )
                client.log_artifact(
                    mlflow_run_id, str(file_path), artifact_path=artifact_path
                )
                artifact_count += 1
        detail(f"Uploaded {artifact_count} artifact file(s) from {artifacts_dir}")
        success(f"MLflow upload complete for run {mlflow_run_id}")
        return artifacts_dir
    finally:
        _cleanup_dir_contents(artifacts_dir)
