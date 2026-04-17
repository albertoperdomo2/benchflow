from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import mlflow
import plotly.graph_objects as go
import requests
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

from ..cluster import CommandError, require_command
from ..models import ResolvedRunPlan, ValidationError
from ..ui import detail, step, success
from .common import (
    BenchmarkRunFailed,
    benchmark_version_from_plan,
    resolved_accelerator,
)

_AIPERF_SUMMARY_CANDIDATES = (
    "results/profile_export_aiperf.json",
    "benchmark/profile_export_aiperf.json",
    "profile_export_aiperf.json",
)
_AIPERF_ARTIFACT_ROOT = "benchmark"
_PLOTLY_CONFIG = {"displaylogo": False, "responsive": True}


def _iso8601_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _configure_aiperf_runtime() -> dict[str, str]:
    runtime_root = Path("/tmp/benchflow-aiperf")
    home_dir = runtime_root / "home"
    hf_home = runtime_root / "huggingface"
    xdg_cache_home = runtime_root / "xdg-cache"
    for path in (home_dir, hf_home, xdg_cache_home):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "HOME": str(home_dir),
        "HF_HOME": str(hf_home),
        "XDG_CACHE_HOME": str(xdg_cache_home),
        "HF_HUB_CACHE": str(hf_home / "hub"),
        "TRANSFORMERS_CACHE": str(hf_home / "transformers"),
    }


def _aiperf_options(plan: ResolvedRunPlan) -> dict[str, Any]:
    options = dict(plan.benchmark.options or {})
    if not options:
        raise ValidationError(
            "aiperf benchmark profile requires spec.options to be configured"
        )
    return options


def _required_option(options: dict[str, Any], key: str) -> str:
    value = str(options.get(key, "") or "").strip()
    if not value:
        raise ValidationError(f"aiperf benchmark profile is missing option {key!r}")
    return value


def _parse_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_positive_int(value: Any, *, field_name: str) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = int(str(value).strip())
    except ValueError as exc:
        raise ValidationError(f"{field_name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValidationError(f"{field_name} must be a positive integer")
    return parsed


def _dataset_cache_root() -> Path:
    root = Path("/tmp/benchflow-aiperf/datasets")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _dataset_filename(dataset_url: str, dataset_name: str) -> str:
    explicit = str(dataset_name or "").strip()
    if explicit:
        return explicit
    parsed = urlparse(dataset_url)
    name = Path(parsed.path).name
    return name or "dataset.jsonl"


def _download_dataset(*, dataset_url: str, dataset_name: str) -> Path:
    target_path = _dataset_cache_root() / _dataset_filename(dataset_url, dataset_name)
    if target_path.exists():
        detail(f"Using cached AIPerf dataset {target_path.name}")
        return target_path
    step(f"Downloading AIPerf dataset from {dataset_url}")
    with requests.get(dataset_url, stream=True, timeout=120) as response:
        response.raise_for_status()
        temp_path = target_path.with_suffix(target_path.suffix + ".download")
        with temp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        temp_path.replace(target_path)
    detail(f"Stored dataset at {target_path}")
    return target_path


def _cap_dataset_entries(dataset_path: Path, *, dataset_cap: int | None) -> Path:
    if dataset_cap is None:
        return dataset_path
    capped_path = dataset_path.with_name(
        f"{dataset_path.stem}-cap{dataset_cap}{dataset_path.suffix}"
    )
    step(f"Capping AIPerf dataset to {dataset_cap} entries")
    written = 0
    with (
        dataset_path.open("r", encoding="utf-8") as src,
        capped_path.open("w", encoding="utf-8") as dst,
    ):
        for line in src:
            if not line.strip():
                continue
            dst.write(line)
            written += 1
            if written >= dataset_cap:
                break
    detail(f"Wrote capped dataset file with {written} entries")
    return capped_path


def _artifact_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    path = Path(tempfile.mkdtemp(prefix="benchflow-aiperf-"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_subprocess(argv: list[str], *, env: dict[str, str]) -> None:
    completed = subprocess.run(argv, env=env, text=True, check=False)
    if completed.returncode != 0:
        raise CommandError(
            f"{' '.join(argv)} exited with status {completed.returncode}"
        )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _nested_metric_value(
    summary: dict[str, Any], key: str, field: str = "avg"
) -> float | None:
    value = summary.get(key)
    if isinstance(value, dict):
        metric = value.get(field)
        if metric is None:
            return None
        try:
            return float(metric)
        except (TypeError, ValueError):
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _log_summary_metrics(summary: dict[str, Any]) -> None:
    metric_map = {
        "throughput/request_throughput": ("request_throughput", "avg"),
        "tokens/output_token_throughput": ("output_token_throughput", "avg"),
        "tokens/total_token_throughput": ("total_token_throughput", "avg"),
        "latency/request_latency_ms": ("request_latency", "avg"),
        "latency/request_latency_p95_ms": ("request_latency", "p95"),
        "ttft/time_to_first_token_ms": ("time_to_first_token", "avg"),
        "ttft/time_to_first_token_p95_ms": ("time_to_first_token", "p95"),
        "itl/inter_token_latency_ms": ("inter_token_latency", "avg"),
        "itl/inter_token_latency_p95_ms": ("inter_token_latency", "p95"),
    }
    for mlflow_name, (summary_key, field_name) in metric_map.items():
        value = _nested_metric_value(summary, summary_key, field_name)
        if value is not None:
            mlflow.log_metric(mlflow_name, value)

    for counter_key in (
        "request_count",
        "error_request_count",
        "total_output_tokens",
        "total_token_throughput",
    ):
        value = summary.get(counter_key)
        try:
            mlflow.log_metric(counter_key, float(value))
        except (TypeError, ValueError):
            continue


def _log_artifacts(artifact_dir: Path) -> None:
    for child in sorted(artifact_dir.iterdir()):
        if child.is_dir():
            mlflow.log_artifacts(
                str(child), artifact_path=f"{_AIPERF_ARTIFACT_ROOT}/{child.name}"
            )
        else:
            mlflow.log_artifact(str(child), artifact_path=_AIPERF_ARTIFACT_ROOT)


def _summary_path(artifact_dir: Path) -> Path:
    path = artifact_dir / "profile_export_aiperf.json"
    if not path.exists():
        raise BenchmarkRunFailed(
            f"AIPerf benchmark did not produce {path.name} in {artifact_dir}"
        )
    return path


def _build_command(
    *,
    plan: ResolvedRunPlan,
    target: str,
    artifact_dir: Path,
    dataset_path: Path,
    options: dict[str, Any],
) -> list[str]:
    endpoint_type = _required_option(options, "endpoint_type")
    dataset_type = _required_option(options, "dataset_type")
    endpoint_path = str(
        options.get("endpoint_path")
        or plan.deployment.target.path
        or "/v1/chat/completions"
    ).strip()
    tokenizer = str(options.get("tokenizer") or plan.model.name).strip()
    command = [
        "aiperf",
        "profile",
        "--model",
        plan.model.name,
        "--url",
        target,
        "--endpoint-type",
        endpoint_type,
        "--endpoint",
        endpoint_path,
        "--input-file",
        str(dataset_path),
        "--custom-dataset-type",
        dataset_type,
        "--tokenizer",
        tokenizer,
        "--artifact-dir",
        str(artifact_dir),
        "--ui",
        "none",
    ]
    if _parse_bool(options.get("streaming"), default=True):
        command.append("--streaming")
    if _parse_bool(options.get("fixed_schedule"), default=True):
        command.append("--fixed-schedule")
    if _parse_bool(options.get("fixed_schedule_auto_offset"), default=True):
        command.append("--fixed-schedule-auto-offset")
    synthesis_max_isl = _parse_positive_int(
        options.get("synthesis_max_isl"),
        field_name="benchmark.options.synthesis_max_isl",
    )
    if synthesis_max_isl is not None:
        command.extend(["--synthesis-max-isl", str(synthesis_max_isl)])
    fixed_schedule_end_offset = _parse_positive_int(
        options.get("fixed_schedule_end_offset"),
        field_name="benchmark.options.fixed_schedule_end_offset",
    )
    if fixed_schedule_end_offset is not None:
        command.extend(["--fixed-schedule-end-offset", str(fixed_schedule_end_offset)])
    return command


def run_benchmark(
    *,
    plan: ResolvedRunPlan,
    target: str | None = None,
    output_dir: Path | None = None,
    mlflow_tracking_uri: str | None = None,
    enable_mlflow: bool = True,
    extra_tags: dict[str, str] | None = None,
) -> tuple[str, str, str]:
    require_command("aiperf")
    options = _aiperf_options(plan)
    benchmark_target = target or plan.deployment.target.base_url
    dataset_url = _required_option(options, "dataset_url")
    dataset_name = str(options.get("dataset_name") or "").strip()
    dataset_cap = _parse_positive_int(
        options.get("dataset_cap"),
        field_name="benchmark.options.dataset_cap",
    )
    artifact_dir = _artifact_dir(output_dir)
    remove_artifact_dir = output_dir is None
    start_time = _iso8601_now()
    run_id = ""
    benchmark_env = _configure_aiperf_runtime()
    benchmark_env.update(os.environ)
    benchmark_env.update(plan.benchmark.env)
    if output_dir is not None:
        benchmark_env["AIPERF_ARTIFACT_DIR"] = str(output_dir)
    dataset_path = _cap_dataset_entries(
        _download_dataset(dataset_url=dataset_url, dataset_name=dataset_name),
        dataset_cap=dataset_cap,
    )
    command = _build_command(
        plan=plan,
        target=benchmark_target,
        artifact_dir=artifact_dir,
        dataset_path=dataset_path,
        options=options,
    )

    tags = dict(plan.mlflow.tags)
    if extra_tags:
        tags.update(extra_tags)
    tags.setdefault("accelerator", resolved_accelerator(plan))
    tags.setdefault("version", benchmark_version_from_plan(plan))
    tags.setdefault("benchmark_tool", "aiperf")

    step(f"Preparing AIPerf benchmark run for {plan.model.name}")
    detail(f"Target: {benchmark_target}")
    detail(f"Dataset: {dataset_path}")
    detail(f"Artifact directory: {artifact_dir}")
    detail(f"MLflow: {'enabled' if enable_mlflow else 'disabled'}")
    try:
        if enable_mlflow:
            tracking_uri = mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
            if not tracking_uri:
                raise BenchmarkRunFailed(
                    "MLFLOW_TRACKING_URI is required when MLflow is enabled"
                )
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(plan.mlflow.experiment)
            with mlflow.start_run(tags=tags) as run:
                run_id = run.info.run_id
                mlflow.log_param("benchmark_tool", "aiperf")
                mlflow.log_param("backend_type", plan.benchmark.backend_type)
                mlflow.log_param("dataset_url", dataset_url)
                if dataset_cap is not None:
                    mlflow.log_param("dataset_cap", dataset_cap)
                mlflow.log_param(
                    "dataset_type", _required_option(options, "dataset_type")
                )
                mlflow.log_param(
                    "endpoint_type", _required_option(options, "endpoint_type")
                )
                mlflow.log_param("target", benchmark_target)
                mlflow.log_param("model", plan.model.name)
                mlflow.log_param("tp", plan.deployment.runtime.tensor_parallelism)
                mlflow.log_param("replicas", plan.deployment.runtime.replicas)
                mlflow.log_param("version", benchmark_version_from_plan(plan))
                _run_subprocess(command, env=benchmark_env)
                summary = _load_json(_summary_path(artifact_dir))
                _log_summary_metrics(summary)
                _log_artifacts(artifact_dir)
        else:
            _run_subprocess(command, env=benchmark_env)
            _summary_path(artifact_dir)
    except Exception as exc:  # noqa: BLE001
        end_time = _iso8601_now()
        raise BenchmarkRunFailed(
            str(exc),
            run_id=run_id,
            start_time=start_time,
            end_time=end_time,
        ) from exc
    finally:
        if remove_artifact_dir:
            shutil.rmtree(artifact_dir, ignore_errors=True)

    end_time = _iso8601_now()
    success(
        f"AIPerf benchmark completed for {plan.model.name} "
        f"({'MLflow run ' + run_id if run_id else 'local output'})"
    )
    return run_id, start_time, end_time


def is_aiperf_artifacts_dir(path: Path) -> bool:
    candidates = (
        path / "profile_export_aiperf.json",
        path / "benchmark" / "profile_export_aiperf.json",
        path / "results" / "profile_export_aiperf.json",
    )
    return any(candidate.exists() for candidate in candidates)


def _resolve_artifact_file(
    artifact_uri: str, candidates: tuple[str, ...]
) -> str | None:
    repo = get_artifact_repository(artifact_uri)
    for candidate in candidates:
        root = str(Path(candidate).parent).replace("\\", "/")
        filename = Path(candidate).name
        for entry in repo.list_artifacts("" if root == "." else root):
            if not entry.is_dir and entry.path.endswith(filename):
                return entry.path
    return None


def _download_run_file(artifact_uri: str, artifact_path: str, cache_dir: Path) -> Path:
    repo = get_artifact_repository(artifact_uri)
    downloaded = repo.download_artifacts(artifact_path, dst_path=str(cache_dir))
    return Path(downloaded)


def _load_jsonl_metrics(path: Path) -> dict[str, list[float]]:
    buckets: dict[str, list[float]] = {
        "request_latency": [],
        "time_to_first_token": [],
        "inter_token_latency": [],
    }
    if not path.exists():
        return buckets
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            metrics = payload.get("metrics") or {}
            for name in list(buckets):
                value = (metrics.get(name) or {}).get("value")
                if isinstance(value, (int, float)):
                    buckets[name].append(float(value))
    return buckets


def _resolve_output_path(
    *,
    default_filename: str,
    output_dir: Path | None,
    output_file: Path | None,
    default_dir: Path | None = None,
) -> Path:
    if output_file is not None:
        path = output_file.resolve()
    elif output_dir is not None:
        path = (output_dir / default_filename).resolve()
    elif default_dir is not None:
        path = (default_dir / default_filename).resolve()
    else:
        path = Path(default_filename).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _label_for_run(run_payload: dict[str, Any]) -> str:
    version = str(run_payload.get("version") or "unknown").strip()
    accelerator = str(run_payload.get("accelerator") or "unknown").strip()
    tp = run_payload.get("tp")
    replicas = run_payload.get("replicas")
    return f"{version} | {accelerator} | tp={tp} | r={replicas}"


def _comparison_bar_figure(
    *,
    title: str,
    x_labels: list[str],
    metric_label: str,
    values: list[float],
) -> go.Figure:
    figure = go.Figure(
        data=[
            go.Bar(
                x=x_labels,
                y=values,
                text=[f"{value:.2f}" for value in values],
                textposition="outside",
                marker_color="#3366cc",
            )
        ]
    )
    figure.update_layout(
        title=title,
        height=520,
        margin=dict(l=40, r=20, t=60, b=140),
        xaxis=dict(title="", tickangle=-20),
        yaxis=dict(title=metric_label),
    )
    return figure


def _render_html(
    *,
    title: str,
    subtitle_lines: list[str],
    figures: list[go.Figure],
    output_path: Path,
) -> None:
    from plotly.offline import get_plotlyjs

    parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        f"<title>{title}</title>",
        f"<script type='text/javascript'>{get_plotlyjs()}</script>",
        "</head>",
        "<body style='background: white; margin: 12px; font-family: sans-serif;'>",
        f"<h1 style='margin-bottom: 4px;'>{title}</h1>",
    ]
    for line in subtitle_lines:
        parts.append(f"<div style='color: #555; margin-bottom: 4px;'>{line}</div>")
    for figure in figures:
        parts.append(
            figure.to_html(
                include_plotlyjs=False,
                full_html=False,
                config=_PLOTLY_CONFIG,
            )
        )
    parts.extend(["</body>", "</html>"])
    output_path.write_text("\n".join(parts), encoding="utf-8")


def generate_report(
    *,
    mlflow_run_ids: list[str],
    mlflow_tracking_uri: str | None,
    output_dir: Path | None,
    output_file: Path | None,
    notes: list[str] | None = None,
) -> Path:
    if not mlflow_run_ids:
        raise ValidationError("AIPerf comparison reports require --mlflow-run-ids")
    tracking_uri = mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValidationError(
            "MLFLOW_TRACKING_URI is required for AIPerf comparison reports"
        )
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    cache_dir = Path(tempfile.mkdtemp(prefix="benchflow-aiperf-report-"))
    runs_data: list[dict[str, Any]] = []
    try:
        for run_id in mlflow_run_ids:
            run = client.get_run(run_id)
            summary_artifact = _resolve_artifact_file(
                run.info.artifact_uri,
                _AIPERF_SUMMARY_CANDIDATES,
            )
            if not summary_artifact:
                raise ValidationError(
                    f"MLflow run {run_id} does not contain AIPerf summary artifacts"
                )
            summary = _load_json(
                _download_run_file(run.info.artifact_uri, summary_artifact, cache_dir)
            )
            runs_data.append(
                {
                    "run_id": run_id,
                    "summary": summary,
                    "version": str(
                        run.data.params.get("version")
                        or run.data.tags.get("version")
                        or "unknown"
                    ),
                    "accelerator": str(
                        run.data.params.get("accelerator")
                        or run.data.tags.get("accelerator")
                        or "unknown"
                    ),
                    "tp": str(run.data.params.get("tp") or "1"),
                    "replicas": str(run.data.params.get("replicas") or "1"),
                }
            )
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    labels = [_label_for_run(item) for item in runs_data]
    figures = [
        _comparison_bar_figure(
            title="Request Throughput",
            x_labels=labels,
            metric_label="requests/sec",
            values=[
                _nested_metric_value(item["summary"], "request_throughput") or 0.0
                for item in runs_data
            ],
        ),
        _comparison_bar_figure(
            title="Output Token Throughput",
            x_labels=labels,
            metric_label="tokens/sec",
            values=[
                _nested_metric_value(item["summary"], "output_token_throughput") or 0.0
                for item in runs_data
            ],
        ),
        _comparison_bar_figure(
            title="TTFT P95",
            x_labels=labels,
            metric_label="ms",
            values=[
                _nested_metric_value(item["summary"], "time_to_first_token", "p95")
                or 0.0
                for item in runs_data
            ],
        ),
        _comparison_bar_figure(
            title="ITL P95",
            x_labels=labels,
            metric_label="ms",
            values=[
                _nested_metric_value(item["summary"], "inter_token_latency", "p95")
                or 0.0
                for item in runs_data
            ],
        ),
        _comparison_bar_figure(
            title="Request Latency P95",
            x_labels=labels,
            metric_label="ms",
            values=[
                _nested_metric_value(item["summary"], "request_latency", "p95") or 0.0
                for item in runs_data
            ],
        ),
    ]
    output_path = _resolve_output_path(
        default_filename="benchmark-comparison-aiperf.html",
        output_dir=output_dir,
        output_file=output_file,
    )
    subtitle = [f"MLflow runs: {', '.join(mlflow_run_ids)}"]
    if notes:
        subtitle.extend([f"Notes: {notes[0]}", *notes[1:]])
    _render_html(
        title="BenchFlow AIPerf Comparison Report",
        subtitle_lines=subtitle,
        figures=figures,
        output_path=output_path,
    )
    return output_path


def generate_run_report(
    *,
    artifacts_dir: Path,
    output_dir: Path | None,
    output_file: Path | None,
) -> Path:
    summary_path = next(
        (
            candidate
            for candidate in (
                artifacts_dir / "profile_export_aiperf.json",
                artifacts_dir / "benchmark" / "profile_export_aiperf.json",
                artifacts_dir / "results" / "profile_export_aiperf.json",
            )
            if candidate.exists()
        ),
        None,
    )
    if summary_path is None:
        raise ValidationError(f"could not find AIPerf summary under {artifacts_dir}")
    jsonl_path = next(
        (
            candidate
            for candidate in (
                artifacts_dir / "profile_export.jsonl",
                artifacts_dir / "benchmark" / "profile_export.jsonl",
            )
            if candidate.exists()
        ),
        None,
    )
    summary = _load_json(summary_path)
    distributions = _load_jsonl_metrics(jsonl_path) if jsonl_path else {}

    figures = [
        _comparison_bar_figure(
            title="Run Summary",
            x_labels=[
                "request throughput",
                "output token throughput",
                "total token throughput",
            ],
            metric_label="value",
            values=[
                _nested_metric_value(summary, "request_throughput") or 0.0,
                _nested_metric_value(summary, "output_token_throughput") or 0.0,
                _nested_metric_value(summary, "total_token_throughput") or 0.0,
            ],
        )
    ]
    for metric_name, title in (
        ("time_to_first_token", "TTFT Distribution"),
        ("inter_token_latency", "ITL Distribution"),
        ("request_latency", "Request Latency Distribution"),
    ):
        values = list(distributions.get(metric_name) or [])
        if not values:
            continue
        figure = go.Figure(data=[go.Histogram(x=values, marker_color="#3366cc")])
        figure.update_layout(
            title=title,
            height=480,
            margin=dict(l=40, r=20, t=60, b=60),
            xaxis=dict(title="ms"),
            yaxis=dict(title="count"),
        )
        figures.append(figure)

    output_path = _resolve_output_path(
        default_filename="full_run_artifacts_report.html",
        output_dir=output_dir,
        output_file=output_file,
        default_dir=artifacts_dir,
    )
    subtitle = [
        f"Model: {summary.get('input_config', {}).get('model') or 'unknown'}",
        f"Requests: {summary.get('request_count', 'unknown')}",
    ]
    _render_html(
        title="BenchFlow AIPerf Run Report",
        subtitle_lines=subtitle,
        figures=figures,
        output_path=output_path,
    )
    return output_path
