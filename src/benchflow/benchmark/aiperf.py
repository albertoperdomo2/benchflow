from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import mlflow
import plotly.graph_objects as go
import requests
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from ..cluster import CommandError, require_command
from ..mlflow_compat import create_mlflow_client, configure_mlflow_tracking
from ..models import AiperfBenchmarkSpec, ResolvedRunPlan, ValidationError
from ..plotting import REPORT_COLOR_PALETTE
from ..ui import detail, step, success
from .common import (
    BenchmarkRunFailed,
    benchmark_version_from_plan,
    resolved_accelerator,
)
from .cli_args import render_cli_args
from .comparison_metrics import build_comparison_metric_panels

_AIPERF_SUMMARY_CANDIDATES = (
    "results/profile_export_aiperf.json",
    "benchmark/profile_export_aiperf.json",
    "profile_export_aiperf.json",
)
_AIPERF_ARTIFACT_ROOT = "benchmark"
_PLOTLY_CONFIG = {"displaylogo": False, "responsive": True}
_PUBLIC_DATASET_LABELS = {
    "semianalysis_cc_traces_weka_with_subagents": (
        "semianalysisai/cc-traces-weka-with-subagents-052726"
    ),
    "weka_hf": "semianalysisai/cc-traces-weka-061526",
}
_AIPERF_BASELINE_METRICS = (
    ("Successful requests", "request_count", "avg", "requests", True, 0),
    ("Request throughput", "request_throughput", "avg", "req/s", True, 3),
    ("Output token throughput", "output_token_throughput", "avg", "tok/s", True, 2),
    ("Total token throughput", "total_token_throughput", "avg", "tok/s", True, 2),
    ("TTFT p50", "time_to_first_token", "p50", "ms", False, 2),
    ("TTFT p95", "time_to_first_token", "p95", "ms", False, 2),
    ("ITL p50", "inter_token_latency", "p50", "ms", False, 2),
    ("ITL p95", "inter_token_latency", "p95", "ms", False, 2),
    ("Latency p50", "request_latency", "p50", "ms", False, 2),
    ("Latency p95", "request_latency", "p95", "ms", False, 2),
    ("Error requests", "error_request_count", "avg", "requests", False, 0),
)
_COLORS = {
    "black": "#222222",
    "gray": "#6f6f6f",
    "grid": "#e8e8e8",
    "paper": "white",
    "blue": REPORT_COLOR_PALETTE[0],
    "orange": REPORT_COLOR_PALETTE[1],
    "green": REPORT_COLOR_PALETTE[2],
    "red": REPORT_COLOR_PALETTE[3],
    "purple": REPORT_COLOR_PALETTE[4],
}
_AIPERF_COMPARISON_VERSION_PALETTE = [
    *REPORT_COLOR_PALETTE,
    "#8c564b",
    "#17becf",
    "#bcbd22",
    "#7f7f7f",
    "#e377c2",
]
_HEADER_WIDTH = 1440
_REPORT_FONT = "Arial, Helvetica, sans-serif"
_TITLE_FONT = "Times New Roman, Georgia, serif"


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


def _aiperf_spec(plan: ResolvedRunPlan) -> AiperfBenchmarkSpec:
    if plan.benchmark.tool != "aiperf":
        raise ValidationError("AIPerf benchmark runner requires tool: aiperf")
    return plan.benchmark.aiperf


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
    if capped_path.exists():
        detail(f"Using cached trimmed AIPerf dataset {capped_path.name}")
        return capped_path
    step(f"Trimming AIPerf dataset to {dataset_cap} entries")
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
    aiperf: AiperfBenchmarkSpec,
) -> list[str]:
    args = dict(aiperf.args)
    args.setdefault(
        "endpoint_path", plan.deployment.target.path or "/v1/chat/completions"
    )
    args.setdefault("tokenizer", plan.model.name)
    command = [
        "aiperf",
        "profile",
        "--model",
        plan.model.name,
        "--url",
        target,
        "--artifact-dir",
        str(artifact_dir),
        "--ui",
        "none",
    ]
    if aiperf.dataset_url:
        command.extend(
            [
                "--input-file",
                str(dataset_path),
            ]
        )
    command.extend(
        render_cli_args(
            args,
            aliases={
                "endpoint_path": "endpoint",
                "dataset_type": "custom_dataset_type",
            },
        )
    )
    return command


def run_benchmark(
    *,
    plan: ResolvedRunPlan,
    target: str | None = None,
    output_dir: Path | None = None,
    mlflow_tracking_uri: str | None = None,
    enable_mlflow: bool = True,
    mlflow_run_id: str = "",
    extra_tags: dict[str, str] | None = None,
) -> tuple[str, str, str]:
    require_command("aiperf")
    aiperf = _aiperf_spec(plan)
    benchmark_target = target or plan.deployment.target.base_url
    artifact_dir = _artifact_dir(output_dir)
    remove_artifact_dir = output_dir is None
    start_time = _iso8601_now()
    run_id = ""
    benchmark_env = _configure_aiperf_runtime()
    benchmark_env.update(os.environ)
    benchmark_env.update(plan.benchmark.env)
    if output_dir is not None:
        benchmark_env["AIPERF_ARTIFACT_DIR"] = str(output_dir)
    dataset_path = Path("/tmp/benchflow-aiperf/public-dataset-placeholder.jsonl")
    public_dataset = str(aiperf.args.get("public_dataset", "") or "").strip()
    if not public_dataset:
        dataset_path = _cap_dataset_entries(
            _download_dataset(
                dataset_url=aiperf.dataset_url,
                dataset_name=aiperf.dataset_name,
            ),
            dataset_cap=aiperf.dataset_cap,
        )
    command = _build_command(
        plan=plan,
        target=benchmark_target,
        artifact_dir=artifact_dir,
        dataset_path=dataset_path,
        aiperf=aiperf,
    )

    tags = dict(plan.mlflow.tags)
    if extra_tags:
        tags.update(extra_tags)
    tags.setdefault("accelerator", resolved_accelerator(plan))
    tags.setdefault("version", benchmark_version_from_plan(plan))
    tags.setdefault("benchmark_tool", "aiperf")

    step(f"Preparing AIPerf benchmark run for {plan.model.name}")
    detail(f"Target: {benchmark_target}")
    if public_dataset:
        detail(f"AIPerf public dataset: {public_dataset}")
    else:
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
            configure_mlflow_tracking(tracking_uri)
            mlflow.set_experiment(plan.mlflow.experiment or "Default")
            start_run_kwargs = (
                {"run_id": mlflow_run_id.strip()}
                if str(mlflow_run_id or "").strip()
                else {"tags": tags}
            )
            with mlflow.start_run(**start_run_kwargs) as run:
                run_id = run.info.run_id
                if mlflow_run_id:
                    mlflow.set_tags(tags)
                mlflow.log_param("benchmark_tool", "aiperf")
                mlflow.log_param("backend_type", "openai_http")
                if public_dataset:
                    mlflow.log_param("public_dataset", public_dataset)
                else:
                    mlflow.log_param("dataset_url", aiperf.dataset_url)
                    if aiperf.dataset_cap is not None:
                        mlflow.log_param("dataset_cap", aiperf.dataset_cap)
                mlflow.log_param("max_seconds", aiperf.max_seconds)
                for key, value in aiperf.args.items():
                    mlflow.log_param(
                        key,
                        json.dumps(value) if isinstance(value, (dict, list)) else value,
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


def _full_label_for_run(run_payload: dict[str, Any]) -> str:
    version = str(run_payload.get("version") or "unknown").strip()
    accelerator = str(run_payload.get("accelerator") or "unknown").strip()
    tp = run_payload.get("tp")
    replicas = run_payload.get("replicas")
    return f"{version} | {accelerator} | tp={tp} | r={replicas}"


def _label_for_run(run_payload: dict[str, Any]) -> str:
    version = str(run_payload.get("version") or "unknown").strip()
    tp = run_payload.get("tp")
    replicas = run_payload.get("replicas")
    return f"{version}<br>tp={tp} | r={replicas}"


def _composed_version_from_mlflow_run(run: mlflow.entities.Run) -> str:
    base_version = str(
        run.data.params.get("version") or run.data.tags.get("version") or "unknown"
    ).strip()
    suffix = str(
        run.data.tags.get("epp")
        or run.data.tags.get("deployment_profile")
        or run.data.tags.get("deployment_type")
        or ""
    ).strip()
    if suffix:
        return f"{base_version}-{suffix}"
    return base_version


def _comparison_model_name(runs_data: list[dict[str, Any]]) -> str:
    if not runs_data:
        return "unknown"
    input_config = runs_data[0].get("summary", {}).get("input_config", {}) or {}
    endpoint = input_config.get("endpoint", {}) or {}
    model_names = endpoint.get("model_names") or []
    if isinstance(model_names, list) and model_names:
        return str(model_names[0]).strip() or "unknown"
    return "unknown"


def _comparison_dataset_label(runs_data: list[dict[str, Any]]) -> str:
    if not runs_data:
        return "unknown"
    input_config = runs_data[0].get("summary", {}).get("input_config", {}) or {}
    input_section = input_config.get("input", {}) or {}
    public_dataset = str(input_section.get("public_dataset") or "").strip()
    detected_loader = str(input_section.get("detected_loader") or "").strip()
    for dataset_id in (public_dataset, detected_loader):
        if not dataset_id:
            continue
        return _PUBLIC_DATASET_LABELS.get(dataset_id, dataset_id)
    dataset_file = str(input_section.get("file") or "").strip()
    dataset_type = str(input_section.get("custom_dataset_type") or "").strip()
    dataset_name = Path(dataset_file).name if dataset_file else ""
    if dataset_name and dataset_type:
        return f"{dataset_name} ({dataset_type})"
    if dataset_name:
        return dataset_name
    if dataset_type:
        return dataset_type
    return "unknown"


def _apply_axis_style(figure: go.Figure) -> None:
    figure.update_xaxes(
        showgrid=True,
        gridcolor=_COLORS["grid"],
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor=_COLORS["black"],
        mirror=True,
        title_font={"size": 14},
        tickfont={"size": 12},
    )
    figure.update_yaxes(
        showgrid=True,
        gridcolor=_COLORS["grid"],
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor=_COLORS["black"],
        mirror=True,
        title_font={"size": 14},
        tickfont={"size": 12},
    )


def _comparison_bar_figure(
    *,
    title: str,
    x_labels: list[str],
    metric_label: str,
    values: list[float],
    color: str,
) -> go.Figure:
    figure = go.Figure(
        data=[
            go.Bar(
                x=x_labels,
                y=values,
                text=[f"{value:.2f}" for value in values],
                textposition="outside",
                marker_color=color,
                marker_line={"color": color, "width": 1},
            )
        ]
    )
    figure.update_layout(
        title=title,
        width=_HEADER_WIDTH,
        height=500,
        paper_bgcolor=_COLORS["paper"],
        plot_bgcolor=_COLORS["paper"],
        font={"family": _REPORT_FONT, "size": 12, "color": _COLORS["black"]},
        margin=dict(l=75, r=35, t=80, b=140),
        xaxis=dict(title="", tickangle=-20),
        yaxis=dict(title=metric_label),
        showlegend=False,
    )
    _apply_axis_style(figure)
    return figure


def _metric_stat(
    summary: dict[str, Any], metric_name: str, stat_name: str = "avg"
) -> float | None:
    metric = summary.get(metric_name)
    if not isinstance(metric, dict):
        return None
    value = metric.get(stat_name)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_number(value: float | None, *, precision: int = 0) -> str:
    if value is None:
        return "—"
    if precision <= 0:
        return f"{value:,.0f}"
    return f"{value:,.{precision}f}"


def _format_compact(value: float | None) -> str:
    if value is None:
        return "—"
    absolute = abs(value)
    if absolute >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if absolute >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:.0f}"


def _baseline_candidates(runs_data: list[dict[str, Any]]) -> list[str]:
    candidates: list[str] = []
    for run_data in runs_data:
        for key in ("composed_version", "version"):
            value = str(run_data.get(key) or "").strip()
            if value and value not in candidates:
                candidates.append(value)
    return candidates


def _resolve_baseline_run(
    runs_data: list[dict[str, Any]], baseline_version: str | None
) -> dict[str, Any] | None:
    cleaned = str(baseline_version or "").strip()
    if not cleaned:
        return None
    matches = [
        run_data
        for run_data in runs_data
        if cleaned
        in {
            str(run_data.get("composed_version") or "").strip(),
            str(run_data.get("version") or "").strip(),
        }
    ]
    if not matches:
        available = ", ".join(_baseline_candidates(runs_data)) or "none"
        raise ValidationError(
            f"baseline version {cleaned!r} is not present in the AIPerf comparison "
            f"data; available values: {available}"
        )
    if len(matches) > 1:
        raise ValidationError(
            f"baseline version {cleaned!r} matched multiple AIPerf runs; "
            "use a unique composed version or display label"
        )
    return matches[0]


def _format_baseline_value(value: float | None, precision: int) -> str:
    if value is None:
        return "—"
    if precision <= 0:
        return f"{value:,.0f}"
    return f"{value:,.{precision}f}"


def _baseline_delta_html(
    *,
    value: float | None,
    baseline: float | None,
    higher_is_better: bool,
    is_baseline_row: bool,
) -> str:
    if value is None or baseline is None:
        return ""
    if is_baseline_row:
        return (
            "<span class='benchflow-report-delta "
            "benchflow-report-delta-baseline'>Δ baseline</span>"
        )
    if baseline == 0:
        return (
            "<span class='benchflow-report-delta "
            "benchflow-report-delta-neutral'>Δ n/a</span>"
        )
    delta_pct = ((value - baseline) / baseline) * 100.0
    if delta_pct == 0:
        delta_class = "neutral"
    elif higher_is_better:
        delta_class = "positive" if delta_pct > 0 else "negative"
    else:
        delta_class = "positive" if delta_pct < 0 else "negative"
    return (
        f"<span class='benchflow-report-delta "
        f"benchflow-report-delta-{delta_class}'>{delta_pct:+.1f}%</span>"
    )


def _render_baseline_comparison_table(
    runs_data: list[dict[str, Any]], baseline_run: dict[str, Any] | None
) -> str:
    if baseline_run is None:
        return ""
    headers = ["Run"] + [
        f"{label}<br><span class='benchflow-report-unit'>{unit}</span>"
        for label, _, _, unit, _, _ in _AIPERF_BASELINE_METRICS
    ]
    metric_column_width = (100 - 26) / (len(headers) - 1)
    colgroup = (
        "<colgroup>"
        "<col style='width: 26%;'>"
        + "".join(
            f"<col style='width: {metric_column_width:.3f}%;'>" for _ in headers[1:]
        )
        + "</colgroup>"
    )
    header_cells = "".join(f"<th>{header}</th>" for header in headers)
    baseline_summary = baseline_run["summary"]
    baseline_values = {
        (key, field): _nested_metric_value(baseline_summary, key, field)
        for _, key, field, _, _, _ in _AIPERF_BASELINE_METRICS
    }
    rows: list[str] = []
    for run_data in runs_data:
        is_baseline_row = run_data is baseline_run
        cells = [
            f"<td>{html.escape(_full_label_for_run(run_data))}</td>",
        ]
        for _, key, field, _, higher_is_better, precision in _AIPERF_BASELINE_METRICS:
            value = _nested_metric_value(run_data["summary"], key, field)
            baseline_value = baseline_values.get((key, field))
            value_text = html.escape(_format_baseline_value(value, precision))
            delta_html = _baseline_delta_html(
                value=value,
                baseline=baseline_value,
                higher_is_better=higher_is_better,
                is_baseline_row=is_baseline_row,
            )
            cells.append(
                "<td>"
                f"<span class='benchflow-report-value'>{value_text}</span>"
                f"{delta_html}"
                "</td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")

    baseline_label = html.escape(_full_label_for_run(baseline_run))
    return f"""
<section class="benchflow-report-table-section">
  <details class="benchflow-report-table-details" open>
    <summary>AIPerf Baseline Comparison</summary>
    <p>Inline Δ values compare each run to baseline {baseline_label}. Higher throughput and successful request count are better; lower latency and error count are better.</p>
    <div class="benchflow-report-table-shell">
      <table class="benchflow-report-table">
        {colgroup}
        <thead>
          <tr>{header_cells}</tr>
        </thead>
        <tbody>
          {"".join(rows)}
        </tbody>
      </table>
    </div>
  </details>
</section>
"""


def _render_mooncake_stats_table(runs_data: list[dict[str, Any]]) -> str:
    if not runs_data:
        return ""
    headers = [
        "Run",
        "Requests",
        "ISL avg",
        "ISL stddev",
        "ISL p50",
        "ISL p95",
        "ISL max",
        "OSL avg",
        "OSL stddev",
        "OSL p50",
        "OSL p95",
        "OSL max",
        "Total ISL",
        "Total OSL",
    ]
    metric_column_width = (100 - 30) / (len(headers) - 1)
    colgroup = (
        "<colgroup>"
        "<col style='width: 30%;'>"
        + "".join(
            f"<col style='width: {metric_column_width:.3f}%;'>" for _ in headers[1:]
        )
        + "</colgroup>"
    )
    header_cells = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    table_rows: list[str] = []
    for run_data in runs_data:
        summary = run_data["summary"]
        isl = "input_sequence_length"
        osl = "output_sequence_length"
        values = [
            _label_for_run(run_data),
            _format_number(_nested_metric_value(summary, "request_count")),
            _format_number(_metric_stat(summary, isl, "avg")),
            _format_number(_metric_stat(summary, isl, "std")),
            _format_number(_metric_stat(summary, isl, "p50")),
            _format_number(_metric_stat(summary, isl, "p95")),
            _format_number(_metric_stat(summary, isl, "max")),
            _format_number(_metric_stat(summary, osl, "avg")),
            _format_number(_metric_stat(summary, osl, "std")),
            _format_number(_metric_stat(summary, osl, "p50")),
            _format_number(_metric_stat(summary, osl, "p95")),
            _format_number(_metric_stat(summary, osl, "max")),
            _format_compact(_nested_metric_value(summary, "total_isl")),
            _format_compact(_nested_metric_value(summary, "total_osl")),
        ]
        value_cells = "".join(f"<td>{html.escape(value)}</td>" for value in values[1:])
        table_rows.append(f"<tr><td>{html.escape(values[0])}</td>{value_cells}</tr>")

    return f"""
<section class="benchflow-report-table-section">
  <details class="benchflow-report-table-details">
    <summary>Mooncake Trace Data Profile</summary>
    <p>Raw input and output sequence length statistics from the AIPerf Mooncake trace artifacts.</p>
    <div class="benchflow-report-table-shell">
      <table class="benchflow-report-table">
        {colgroup}
        <thead>
          <tr>{header_cells}</tr>
        </thead>
        <tbody>
          {"".join(table_rows)}
        </tbody>
      </table>
    </div>
  </details>
</section>
"""


def _report_table_css() -> str:
    return """
    .benchflow-report-table-section {
      width: 100%;
      margin: 24px 0 48px;
    }
    .benchflow-report-table-details {
      background: white;
    }
    .benchflow-report-table-details summary {
      padding: 10px 12px;
      font-size: 20px;
      font-weight: 700;
      cursor: pointer;
      list-style-position: inside;
    }
    .benchflow-report-table-details[open] summary {
      border-bottom: none;
    }
    .benchflow-report-table-section p {
      margin: 12px 0 14px;
      font-size: 12px;
      text-align: center;
    }
    .benchflow-report-table-shell {
      overflow-x: auto;
      padding: 0 10px 10px;
    }
    .benchflow-report-table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 11px;
      background: white;
    }
    .benchflow-report-table th,
    .benchflow-report-table td {
      border: 1px solid #1f2a44;
      padding: 6px 7px;
      vertical-align: top;
    }
    .benchflow-report-table thead th {
      background: #f4f6f8;
      font-weight: 700;
      text-align: left;
    }
    .benchflow-report-table th:first-child,
    .benchflow-report-table td:first-child {
      word-break: break-word;
      white-space: normal;
    }
    .benchflow-report-table tbody td {
      text-align: right;
    }
    .benchflow-report-table tbody td:first-child,
    .benchflow-report-table tbody th {
      text-align: left;
    }
    .benchflow-report-table tbody tr:nth-child(even) td {
      background: #fafbfc;
    }
    .benchflow-report-unit {
      color: #6f6f6f;
      font-size: 10px;
      font-weight: 500;
    }
    .benchflow-report-value {
      display: block;
      font-weight: 600;
    }
    .benchflow-report-delta {
      display: block;
      margin-top: 2px;
      font-size: 10px;
      font-weight: 700;
    }
    .benchflow-report-delta-positive {
      color: #0f7b43;
    }
    .benchflow-report-delta-negative {
      color: #b42318;
    }
    .benchflow-report-delta-neutral,
    .benchflow-report-delta-baseline {
      color: #6f6f6f;
    }
    .metric-panel {
      width: 1440px;
      margin: 28px 0 42px;
      background: white;
    }
    .metric-panel h2 {
      margin: 0 0 4px;
      color: #222222;
      font-family: Arial, Helvetica, sans-serif;
      font-size: 24px;
      font-weight: 700;
      line-height: 1.2;
    }
    .metric-description {
      margin: 0 0 12px;
      color: #222222;
      font-family: Arial, Helvetica, sans-serif;
      font-size: 15px;
      line-height: 1.35;
    }
    .metric-legend {
      display: flex;
      flex-wrap: wrap;
      align-items: flex-start;
      gap: 8px 18px;
      margin: 8px 0 14px;
      padding: 0;
      color: #222222;
      font-family: Arial, Helvetica, sans-serif;
      font-size: 12px;
      line-height: 1.25;
    }
    .metric-legend-item {
      display: inline-flex;
      align-items: center;
      min-width: 0;
      max-width: 460px;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .metric-legend-line {
      display: inline-block;
      flex: 0 0 auto;
      width: 42px;
      height: 3px;
      margin-right: 8px;
      border-radius: 999px;
    }
"""


def _subtitle_text(lines: list[str]) -> str:
    return "<br>".join(
        f"<span style='font-size:13px;color:{_COLORS['gray']}'>{line}</span>"
        for line in lines
    )


def _build_header_figure(*, title: str, subtitle_lines: list[str]) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        width=_HEADER_WIDTH,
        height=120,
        paper_bgcolor=_COLORS["paper"],
        plot_bgcolor=_COLORS["paper"],
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        xaxis={"visible": False},
        yaxis={"visible": False},
        showlegend=False,
        annotations=[
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0.0,
                "y": 0.78,
                "xanchor": "left",
                "yanchor": "middle",
                "showarrow": False,
                "align": "left",
                "text": title,
                "font": {
                    "family": _TITLE_FONT,
                    "size": 28,
                    "color": _COLORS["black"],
                },
            },
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0.0,
                "y": 0.28,
                "xanchor": "left",
                "yanchor": "middle",
                "showarrow": False,
                "align": "left",
                "text": _subtitle_text(subtitle_lines),
                "font": {
                    "family": _REPORT_FONT,
                    "size": 13,
                    "color": _COLORS["gray"],
                },
            },
        ],
    )
    return figure


def _render_report_html(
    *,
    title: str,
    subtitle_lines: list[str],
    figures: list[go.Figure],
    raw_sections: list[str] | None = None,
    output_path: Path,
) -> None:
    parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        f"<title>{title}</title>",
        f"<script type='text/javascript'>{get_plotlyjs()}</script>",
        "<style>",
        _report_table_css(),
        "</style>",
        "</head>",
        "<body style='background: white; margin: 12px;'>",
        "<div style='overflow-x: auto;'>",
        "<table cellspacing='12' cellpadding='0' style='border-collapse: separate;'>",
    ]
    header_html = _build_header_figure(
        title=title,
        subtitle_lines=subtitle_lines,
    ).to_html(
        include_plotlyjs=False,
        full_html=False,
        config=_PLOTLY_CONFIG,
    )
    parts.append(f"<tr><td style='vertical-align: top;'>{header_html}</td></tr>")
    for figure in figures:
        parts.append("<tr>")
        parts.append(
            "<td style='vertical-align: top;'>"
            + figure.to_html(
                include_plotlyjs=False,
                full_html=False,
                config=_PLOTLY_CONFIG,
            )
            + "</td>"
        )
        parts.append("</tr>")
    for section in raw_sections or []:
        if not section:
            continue
        parts.append("<tr>")
        parts.append(
            f"<td style='vertical-align: top; width: {_HEADER_WIDTH}px;'>{section}</td>"
        )
        parts.append("</tr>")
    parts.extend(["</table>", "</div>", "</body>", "</html>"])
    output_path.write_text("\n".join(parts), encoding="utf-8")


def _render_comparison_figure(
    *,
    labels: list[str],
    hover_labels: list[str],
    series_labels: list[str],
    metrics: list[tuple[str, str, list[float]]],
) -> go.Figure:
    rows = max(1, (len(metrics) + 1) // 2)
    figure = make_subplots(
        rows=rows,
        cols=2,
        subplot_titles=[item[0] for item in metrics],
        vertical_spacing=0.06,
        horizontal_spacing=0.08,
    )
    version_colors: dict[str, str] = {}
    for version in series_labels:
        if version not in version_colors:
            version_colors[version] = _AIPERF_COMPARISON_VERSION_PALETTE[
                len(version_colors) % len(_AIPERF_COMPARISON_VERSION_PALETTE)
            ]

    legend_versions: set[str] = set()
    for index, (_, y_axis_title, values) in enumerate(metrics, start=1):
        row = ((index - 1) // 2) + 1
        col = ((index - 1) % 2) + 1
        for label, hover_label, series_label, value in zip(
            labels, hover_labels, series_labels, values, strict=True
        ):
            showlegend = index == 1 and series_label not in legend_versions
            if showlegend:
                legend_versions.add(series_label)
            color = version_colors[series_label]
            figure.add_trace(
                go.Bar(
                    x=[label],
                    y=[value],
                    text=[f"{value:.2f}"],
                    textposition="outside",
                    customdata=[hover_label],
                    hovertemplate="%{customdata}<br>%{y:.2f}<extra></extra>",
                    marker_color=color,
                    marker_line={"color": color, "width": 1},
                    cliponaxis=False,
                    showlegend=showlegend,
                    name=series_label,
                    legendgroup=series_label,
                ),
                row=row,
                col=col,
            )
        figure.update_yaxes(title_text=y_axis_title, row=row, col=col)
        figure.update_xaxes(tickangle=-18, row=row, col=col)
        max_value = max(values) if values else 0.0
        upper_bound = 1.0 if max_value <= 0 else max_value * 1.18
        figure.update_yaxes(range=[0, upper_bound], row=row, col=col)

    figure.update_layout(
        width=_HEADER_WIDTH,
        height=430 * rows + 140,
        paper_bgcolor=_COLORS["paper"],
        plot_bgcolor=_COLORS["paper"],
        font={"family": _REPORT_FONT, "size": 12, "color": _COLORS["black"]},
        margin=dict(l=75, r=35, t=70, b=40),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            font={"size": 12, "color": _COLORS["black"]},
        ),
    )
    _apply_axis_style(figure)
    for annotation in figure.layout.annotations:
        annotation.font = {"size": 14, "color": _COLORS["black"]}
    return figure


def _comparison_metric_panel_trace_specs(panel: Any) -> list[tuple[Any, str]]:
    return [
        (trace, REPORT_COLOR_PALETTE[index % len(REPORT_COLOR_PALETTE)])
        for index, trace in enumerate(list(getattr(panel, "traces", []) or []))
    ]


def _render_comparison_metric_panel_section(panel: Any) -> str:
    figure = go.Figure()
    trace_specs = _comparison_metric_panel_trace_specs(panel)
    if not trace_specs:
        figure.add_annotation(
            text="No archived series found for this metric",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 13, "color": _COLORS["gray"]},
        )
    for trace, color in trace_specs:
        figure.add_trace(
            go.Scatter(
                x=list(getattr(trace, "x", []) or []),
                y=list(getattr(trace, "y", []) or []),
                mode="lines",
                name=str(getattr(trace, "name", "series")),
                line={"color": color, "width": 2},
                showlegend=False,
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Run progress: %{x:.2f} min<br>"
                    "%{y:.4g}<extra></extra>"
                ),
            )
        )
    for reference_line in list(getattr(panel, "reference_lines", []) or []):
        line_label = str(getattr(reference_line, "label", "") or "")
        hline_kwargs: dict[str, Any] = {}
        if line_label:
            hline_kwargs = {
                "annotation_text": line_label,
                "annotation_position": "top left",
                "annotation_font_size": 10,
                "annotation_font_color": str(
                    getattr(reference_line, "color", "#64748b") or "#64748b"
                ),
            }
        figure.add_hline(
            y=float(getattr(reference_line, "y")),
            line_dash=str(getattr(reference_line, "dash", "dash") or "dash"),
            line_color=str(getattr(reference_line, "color", "#64748b") or "#64748b"),
            **hline_kwargs,
        )

    title = str(getattr(panel, "title", "") or "Metric")
    description = str(getattr(panel, "description", "") or "")
    figure.update_layout(
        width=_HEADER_WIDTH,
        height=430,
        paper_bgcolor=_COLORS["paper"],
        plot_bgcolor=_COLORS["paper"],
        font={"family": _REPORT_FONT, "size": 12, "color": _COLORS["black"]},
        margin=dict(l=75, r=35, t=20, b=65),
        xaxis=dict(title="Run progress (min)"),
        yaxis=dict(title=str(getattr(panel, "yaxis_title", "") or "Value")),
    )
    _apply_axis_style(figure)
    legend_items = []
    for trace, color in trace_specs:
        name = html.escape(str(getattr(trace, "name", "series")))
        legend_items.append(
            "<span class='metric-legend-item'>"
            f"<span class='metric-legend-line' style='background:{color}'></span>"
            f"<span>{name}</span>"
            "</span>"
        )
    legend_html = (
        "<div class='metric-legend'>" + "\n".join(legend_items) + "</div>"
        if legend_items
        else ""
    )
    description_html = (
        f"<div class='metric-description'>{html.escape(description)}</div>"
        if description
        else ""
    )
    plot_html = figure.to_html(
        include_plotlyjs=False,
        full_html=False,
        config=_PLOTLY_CONFIG,
    )
    return (
        "<section class='metric-panel'>"
        f"<h2>{html.escape(title)}</h2>"
        f"{description_html}"
        f"{legend_html}"
        f"{plot_html}"
        "</section>"
    )


def _summary_table_figure(summary: dict[str, Any]) -> go.Figure:
    rows = [
        ("Request throughput", _nested_metric_value(summary, "request_throughput")),
        (
            "Output token throughput",
            _nested_metric_value(summary, "output_token_throughput"),
        ),
        (
            "Total token throughput",
            _nested_metric_value(summary, "total_token_throughput"),
        ),
        ("TTFT avg", _nested_metric_value(summary, "time_to_first_token")),
        ("TTFT p95", _nested_metric_value(summary, "time_to_first_token", "p95")),
        ("ITL avg", _nested_metric_value(summary, "inter_token_latency")),
        ("ITL p95", _nested_metric_value(summary, "inter_token_latency", "p95")),
        ("Latency p95", _nested_metric_value(summary, "request_latency", "p95")),
    ]
    labels = [item[0] for item in rows]
    values = ["—" if item[1] is None else f"{item[1]:.2f}" for item in rows]
    figure = go.Figure(
        data=[
            go.Table(
                header={
                    "values": ["Metric", "Value"],
                    "fill_color": _COLORS["blue"],
                    "font": {
                        "color": "white",
                        "size": 13,
                        "family": _REPORT_FONT,
                    },
                    "align": "left",
                    "height": 34,
                },
                cells={
                    "values": [labels, values],
                    "fill_color": [["#f7f7f7", "white"] * 4, ["#f7f7f7", "white"] * 4],
                    "font": {
                        "color": _COLORS["black"],
                        "size": 12,
                        "family": _REPORT_FONT,
                    },
                    "align": "left",
                    "height": 30,
                },
            )
        ]
    )
    figure.update_layout(
        width=_HEADER_WIDTH,
        height=360,
        paper_bgcolor=_COLORS["paper"],
        margin=dict(l=20, r=20, t=25, b=10),
    )
    return figure


def generate_report(
    *,
    mlflow_run_ids: list[str],
    mlflow_tracking_uri: str | None,
    output_dir: Path | None,
    output_file: Path | None,
    version_overrides: dict[str, str] | None = None,
    notes: list[str] | None = None,
    metrics_yaml_path: Path | None = None,
    baseline_version: str | None = None,
) -> Path:
    if not mlflow_run_ids:
        raise ValidationError("AIPerf comparison reports require --mlflow-run-ids")
    tracking_uri = mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValidationError(
            "MLFLOW_TRACKING_URI is required for AIPerf comparison reports"
        )
    configure_mlflow_tracking(tracking_uri)
    client = create_mlflow_client(tracking_uri)
    cache_dir = Path(tempfile.mkdtemp(prefix="benchflow-aiperf-report-"))
    runs_data: list[dict[str, Any]] = []
    overrides = dict(version_overrides or {})
    try:
        for run_id in mlflow_run_ids:
            run = client.get_run(run_id)
            composed_version = _composed_version_from_mlflow_run(run)
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
                    "artifact_uri": run.info.artifact_uri,
                    "summary": summary,
                    "composed_version": composed_version,
                    "version": overrides.get(composed_version, composed_version),
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

    baseline_run = _resolve_baseline_run(runs_data, baseline_version)
    comparison_metric_panels = build_comparison_metric_panels(
        metrics_yaml_path=metrics_yaml_path,
        runs_data=runs_data,
        version_overrides=overrides,
    )
    labels = [_label_for_run(item) for item in runs_data]
    hover_labels = [_full_label_for_run(item) for item in runs_data]
    series_labels = [item["version"] for item in runs_data]
    figures = [
        _render_comparison_figure(
            labels=labels,
            hover_labels=hover_labels,
            series_labels=series_labels,
            metrics=[
                (
                    "Request Throughput",
                    "requests/sec",
                    [
                        _nested_metric_value(item["summary"], "request_throughput")
                        or 0.0
                        for item in runs_data
                    ],
                ),
                (
                    "Output Token Throughput",
                    "tokens/sec",
                    [
                        _nested_metric_value(item["summary"], "output_token_throughput")
                        or 0.0
                        for item in runs_data
                    ],
                ),
                (
                    "Total Token Throughput",
                    "tokens/sec",
                    [
                        _nested_metric_value(item["summary"], "total_token_throughput")
                        or 0.0
                        for item in runs_data
                    ],
                ),
                (
                    "TTFT P50",
                    "ms",
                    [
                        _nested_metric_value(
                            item["summary"], "time_to_first_token", "p50"
                        )
                        or 0.0
                        for item in runs_data
                    ],
                ),
                (
                    "TTFT P95",
                    "ms",
                    [
                        _nested_metric_value(
                            item["summary"], "time_to_first_token", "p95"
                        )
                        or 0.0
                        for item in runs_data
                    ],
                ),
                (
                    "ITL P50",
                    "ms",
                    [
                        _nested_metric_value(
                            item["summary"], "inter_token_latency", "p50"
                        )
                        or 0.0
                        for item in runs_data
                    ],
                ),
                (
                    "ITL P95",
                    "ms",
                    [
                        _nested_metric_value(
                            item["summary"], "inter_token_latency", "p95"
                        )
                        or 0.0
                        for item in runs_data
                    ],
                ),
                (
                    "Request Latency P50",
                    "ms",
                    [
                        _nested_metric_value(item["summary"], "request_latency", "p50")
                        or 0.0
                        for item in runs_data
                    ],
                ),
                (
                    "Request Latency P95",
                    "ms",
                    [
                        _nested_metric_value(item["summary"], "request_latency", "p95")
                        or 0.0
                        for item in runs_data
                    ],
                ),
                (
                    "Input Sequence Length Avg",
                    "tokens",
                    [
                        _nested_metric_value(item["summary"], "input_sequence_length")
                        or 0.0
                        for item in runs_data
                    ],
                ),
                (
                    "Output Sequence Length Avg",
                    "tokens",
                    [
                        _nested_metric_value(item["summary"], "output_sequence_length")
                        or 0.0
                        for item in runs_data
                    ],
                ),
                (
                    "Prefill Throughput Per User Avg",
                    "tokens/sec/user",
                    [
                        _nested_metric_value(
                            item["summary"], "prefill_throughput_per_user"
                        )
                        or 0.0
                        for item in runs_data
                    ],
                ),
                (
                    "Output Token Throughput Per User Avg",
                    "tokens/sec/user",
                    [
                        _nested_metric_value(
                            item["summary"], "output_token_throughput_per_user"
                        )
                        or 0.0
                        for item in runs_data
                    ],
                ),
                (
                    "Time to Second Token P95",
                    "ms",
                    [
                        _nested_metric_value(
                            item["summary"], "time_to_second_token", "p95"
                        )
                        or 0.0
                        for item in runs_data
                    ],
                ),
                (
                    "Error Request Count",
                    "requests",
                    [
                        _nested_metric_value(item["summary"], "error_request_count")
                        or 0.0
                        for item in runs_data
                    ],
                ),
            ],
        ),
    ]
    raw_sections = [
        _render_baseline_comparison_table(runs_data, baseline_run),
        _render_mooncake_stats_table(runs_data),
    ]
    raw_sections.extend(
        _render_comparison_metric_panel_section(panel)
        for panel in comparison_metric_panels
    )
    output_path = _resolve_output_path(
        default_filename="benchmark-comparison-aiperf.html",
        output_dir=output_dir,
        output_file=output_file,
    )
    subtitle = [
        f"Model: {_comparison_model_name(runs_data)}",
        f"Dataset: {_comparison_dataset_label(runs_data)}",
        f"MLflow runs: {', '.join(mlflow_run_ids)}",
    ]
    if notes:
        subtitle.extend([f"Notes: {notes[0]}", *notes[1:]])
    _render_report_html(
        title="AIPerf Mooncake Comparison Report",
        subtitle_lines=subtitle,
        figures=figures,
        raw_sections=raw_sections,
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

    figures = [_summary_table_figure(summary)]
    for metric_name, title in (
        ("time_to_first_token", "TTFT Distribution"),
        ("inter_token_latency", "ITL Distribution"),
        ("request_latency", "Request Latency Distribution"),
    ):
        values = list(distributions.get(metric_name) or [])
        if not values:
            continue
        figure = go.Figure(
            data=[
                go.Histogram(
                    x=values,
                    marker_color=_COLORS["blue"],
                    marker_line={"color": _COLORS["blue"], "width": 1},
                )
            ]
        )
        figure.update_layout(
            title=title,
            width=_HEADER_WIDTH,
            height=460,
            paper_bgcolor=_COLORS["paper"],
            plot_bgcolor=_COLORS["paper"],
            font={"family": _REPORT_FONT, "size": 12, "color": _COLORS["black"]},
            margin=dict(l=75, r=35, t=80, b=60),
            xaxis=dict(title="ms"),
            yaxis=dict(title="count"),
            bargap=0.08,
            showlegend=False,
        )
        _apply_axis_style(figure)
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
    _render_report_html(
        title="BenchFlow AIPerf Run Report",
        subtitle_lines=subtitle,
        figures=figures,
        output_path=output_path,
    )
    return output_path
