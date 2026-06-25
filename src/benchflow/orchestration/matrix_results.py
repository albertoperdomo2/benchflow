from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

from ..contracts import ResolvedRunPlan
from ..mlflow_compat import create_mlflow_client
from ..ui import detail, step, warning
from .matrix_payloads import patch_matrix_result, read_matrix_results_configmap


def _mlflow_run_url(run_id: str) -> str:
    tracking_uri = str(os.environ.get("MLFLOW_TRACKING_URI") or "").strip()
    if not run_id or not tracking_uri.startswith(("http://", "https://")):
        return ""

    experiment_id = ""
    try:
        run = create_mlflow_client(tracking_uri).get_run(run_id)
        experiment_id = str(run.info.experiment_id or "").strip()
    except Exception as exc:  # noqa: BLE001 - summary publication must not hide run ID.
        warning(
            f"Unable to resolve MLflow experiment for matrix result {run_id}: {exc}"
        )

    if not experiment_id:
        return ""

    url = f"{tracking_uri.rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}"
    workspace = str(os.environ.get("MLFLOW_WORKSPACE") or "").strip()
    if workspace:
        url = f"{url}?workspace={quote(workspace)}"
    return url


def publish_matrix_result(
    plan: ResolvedRunPlan,
    *,
    configmap_name: str,
    child_execution_name: str,
    mlflow_run_id: str,
    benchmark_start_time: str,
    benchmark_end_time: str,
) -> dict[str, Any]:
    record = {
        "pipeline_run": child_execution_name,
        "experiment": plan.metadata.name,
        "deployment_profile": plan.profiles.deployment,
        "benchmark_profile": plan.profiles.benchmark,
        "metrics_profile": plan.profiles.metrics,
        "mlflow_run_id": mlflow_run_id,
        "mlflow_url": _mlflow_run_url(mlflow_run_id),
        "benchmark_start_time": benchmark_start_time,
        "benchmark_end_time": benchmark_end_time,
    }
    patch_matrix_result(
        namespace=plan.deployment.namespace,
        configmap_name=configmap_name,
        child_execution_name=child_execution_name,
        record=record,
    )
    return record


def print_matrix_results_summary(
    *,
    namespace: str,
    configmap_name: str,
    child_names: list[str],
) -> None:
    if not configmap_name:
        return

    records = read_matrix_results_configmap(
        namespace=namespace,
        configmap_name=configmap_name,
    )
    by_child = {
        str(record.get("pipeline_run") or "").strip(): record for record in records
    }

    step("Matrix MLflow runs")
    headers = ("child", "deployment", "benchmark", "mlflow")
    rows: list[tuple[str, str, str, str]] = []
    for child_name in child_names:
        record = by_child.get(child_name)
        if record is None:
            rows.append((child_name, "-", "-", "not published"))
            continue
        mlflow_value = str(
            record.get("mlflow_url") or record.get("mlflow_run_id") or ""
        )
        rows.append(
            (
                child_name,
                str(record.get("deployment_profile") or ""),
                str(record.get("benchmark_profile") or ""),
                mlflow_value or "not available",
            )
        )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    detail(
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    )
    for row in rows:
        detail("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
