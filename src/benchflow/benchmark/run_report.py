from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from textwrap import fill

import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from ..models import ValidationError
from ..plotting import REPORT_COLOR_PALETTE
from .processor.processor import _extract_intended_concurrency
from .run_report_insights import (
    load_benchmarks,
    parse_thresholds,
    summarize_benchmarks,
)
from .run_report_insights_plotly import (
    build_figures as build_plotly_benchmark_figures,
)

COLORS = {
    "blue": "#356f9d",
    "orange": "#e67e22",
    "green": "#6abf69",
    "red": "#d84b4b",
    "purple": "#9b59b6",
    "gray": "#7a7a7a",
    "black": "#222222",
    "gold": "#c7a252",
    "teal": "#3b8ea5",
}

PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": False,
    "modeBarButtonsToRemove": [
        "lasso2d",
        "select2d",
        "autoScale2d",
    ],
}

FIGURE_WIDTH = 640
FIGURE_HEIGHT = 520
SUBTITLE_WRAP = 64


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    root: Path
    benchmark_json: Path
    metrics_root: Path
    metadata_json: Path | None
    manifest_roots: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class RunReportMetadata:
    model: str
    version: str
    accelerator: str
    tp: int
    replicas: int
    runtime_args: str
    execution_name: str
    platform: str
    mode: str
    data_spec: str
    profile: str
    backend: str
    load_points: str
    duration: str
    step: str
    total_gpus: float


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_artifact_paths(artifacts_dir: Path) -> ArtifactPaths:
    root = artifacts_dir.resolve()
    benchmark_candidates = (
        root / "results" / "benchmark_output.json",
        root / "benchmark" / "benchmark_output.json",
    )
    benchmark_json = next(
        (path for path in benchmark_candidates if path.exists()), None
    )
    if benchmark_json is None:
        raise ValidationError(
            f"could not find benchmark_output.json under {root}; expected "
            "results/benchmark_output.json or benchmark/benchmark_output.json"
        )

    metrics_root = root / "metrics"
    if not metrics_root.exists():
        raise ValidationError(f"metrics directory not found under {root}")

    metadata_json = root / "metadata.json"
    manifest_roots = tuple(
        path
        for path in (root / "rendered-manifests", root / "manifests")
        if path.exists()
    )

    return ArtifactPaths(
        root=root,
        benchmark_json=benchmark_json,
        metrics_root=metrics_root,
        metadata_json=metadata_json if metadata_json.exists() else None,
        manifest_roots=manifest_roots,
    )


def _metric_rows(paths: ArtifactPaths, metric_name: str) -> list[dict]:
    metric_path = paths.metrics_root / "raw" / f"{metric_name}.json"
    if not metric_path.exists():
        return []
    payload = _load_json(metric_path)
    return payload if isinstance(payload, list) else []


def _rows_to_series(rows: list[dict]) -> dict[int, float]:
    series: dict[int, float] = {}
    for row in rows:
        try:
            series[int(row["timestamp"])] = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
    return dict(sorted(series.items()))


def _rows_to_grouped_series(rows: list[dict], group_fn) -> dict[str, dict[int, float]]:
    grouped: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        try:
            grouped[group_fn(row)][int(row["timestamp"])] = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
    return {key: dict(sorted(value.items())) for key, value in grouped.items()}


def _align_series(timestamps: list[int], series: dict[int, float]) -> np.ndarray:
    return np.array([series.get(ts, np.nan) for ts in timestamps], dtype=float)


def _aggregate_grouped(
    grouped: dict[str, dict[int, float]],
    timestamps: list[int],
    reducer,
) -> np.ndarray:
    values: list[float] = []
    for ts in timestamps:
        samples = [
            float(series[ts])
            for series in grouped.values()
            if ts in series and not np.isnan(series[ts])
        ]
        values.append(
            float(reducer(np.array(samples, dtype=float))) if samples else np.nan
        )
    return np.array(values, dtype=float)


def _coefficient_of_variation(values: np.ndarray) -> float:
    if values.size == 0:
        return np.nan
    mean = float(np.mean(values))
    if abs(mean) < 1e-12:
        return 0.0
    return float(np.std(values) / mean)


def _grouped_share_matrix(
    grouped: dict[str, dict[int, float]],
    timestamps: list[int],
) -> tuple[list[str], np.ndarray]:
    labels = _sort_labels(list(grouped.keys()))
    matrix = np.array(
        [
            [float(grouped[label].get(ts, np.nan)) for ts in timestamps]
            for label in labels
        ],
        dtype=float,
    )
    totals = np.nansum(matrix, axis=0)
    shares = np.divide(matrix, np.where(totals > 0, totals, np.nan))
    return labels, shares


def _phase_reducer(
    timestamps: list[int],
    values: np.ndarray,
    segments: list[dict[str, float | int | str]],
    reducer=np.nanmedian,
) -> list[dict[str, float | int | str]]:
    ts_array = np.array(timestamps, dtype=float)
    value_array = np.array(values, dtype=float)
    summary: list[dict[str, float | int | str]] = []
    for segment in segments:
        mask = (ts_array >= float(segment["start_ts"])) & (
            ts_array <= float(segment["end_ts"])
        )
        segment_values = value_array[mask]
        segment_values = segment_values[np.isfinite(segment_values)]
        reduced = float(reducer(segment_values)) if segment_values.size else np.nan
        summary.append(
            {
                "concurrency": segment["concurrency"],
                "label": segment["label"],
                "value": reduced,
            }
        )
    return summary


def _relative_minutes(timestamps: list[int]) -> list[float]:
    if not timestamps:
        return []
    start = timestamps[0]
    return [(ts - start) / 60.0 for ts in timestamps]


def _short_pod_name(name: str) -> str:
    if not name:
        return "unknown"
    rank_match = re.search(r"-([^-]+)-rank-(\d+)$", name)
    if rank_match:
        return f"{rank_match.group(1)}/r{rank_match.group(2)}"
    suffix = name.split("-")[-1]
    return suffix or name


def _gpu_row_label(row: dict) -> str:
    labels = row.get("labels", {})
    pod = _short_pod_name(labels.get("exported_pod", row.get("series", "unknown")))
    gpu = labels.get("gpu", "?")
    return f"{pod}/g{gpu}"


def _workload_pod_label(row: dict) -> str:
    labels = row.get("labels", {})
    pod = labels.get("pod") or row.get("series", "unknown")
    return _short_pod_name(pod)


def _sort_labels(labels: list[str]) -> list[str]:
    def key(label: str):
        match = re.match(r"(.+)/g(\d+)$", label)
        if match:
            return (match.group(1), int(match.group(2)))
        rank_match = re.match(r"(.+)/r(\d+)$", label)
        if rank_match:
            return (rank_match.group(1), int(rank_match.group(2)))
        return (label, -1)

    return sorted(labels, key=key)


def _title_text(title: str, subtitle: str) -> str:
    wrapped = "<br>".join(fill(subtitle, width=SUBTITLE_WRAP).splitlines())
    return (
        f"{title}"
        f"<br><span style='font-size:11px;color:{COLORS['gray']}'>{wrapped}</span>"
    )


def _base_layout(height: int = FIGURE_HEIGHT) -> dict:
    return {
        "width": FIGURE_WIDTH,
        "height": height,
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "font": {
            "family": "Arial, Helvetica, sans-serif",
            "size": 12,
            "color": COLORS["black"],
        },
        "margin": {"l": 70, "r": 70, "t": 110, "b": 60},
        "legend": {
            "bgcolor": "rgba(255,255,255,0.8)",
            "borderwidth": 0,
            "font": {"size": 11},
        },
        "hovermode": "x unified",
    }


def _apply_common_axes(fig: go.Figure) -> None:
    fig.update_xaxes(showgrid=True, gridcolor="#e8e8e8", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e8e8e8", zeroline=False)


def _apply_phase_bands(
    fig: go.Figure,
    segments: list[dict[str, float | int | str]],
    *,
    label_y: float = 1.06,
) -> None:
    for index, segment in enumerate(segments):
        if index % 2 == 0:
            fig.add_vrect(
                x0=float(segment["start_min"]),
                x1=float(segment["end_min"]),
                fillcolor="rgba(0, 0, 0, 0.03)",
                line_width=0,
                layer="below",
            )
        fig.add_annotation(
            x=float(segment["mid_min"]),
            y=label_y,
            xref="x",
            yref="paper",
            text=str(segment["label"]),
            showarrow=False,
            font={"size": 10, "color": COLORS["gray"]},
            align="center",
        )


def _required_int(metadata: dict[str, object], key: str) -> int:
    value = metadata.get(key)
    if value in (None, "", "unknown"):
        raise ValidationError(
            f"metadata.json must contain integer {key!r} for post-run report generation"
        )
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValidationError(
            f"metadata.json field {key!r} must be an integer, got {value!r}"
        ) from None


def _load_report_metadata(paths: ArtifactPaths) -> RunReportMetadata:
    metadata = {}
    if paths.metadata_json is not None:
        payload = _load_json(paths.metadata_json)
        if isinstance(payload, dict):
            metadata = payload

    benchmark = _load_json(paths.benchmark_json)
    if not isinstance(benchmark, dict):
        raise ValidationError("benchmark_output.json must contain a JSON object")

    metrics_summary = _load_json(paths.metrics_root / "metrics_summary.json")
    if not isinstance(metrics_summary, dict):
        raise ValidationError("metrics_summary.json must contain a JSON object")

    args = benchmark.get("args", {}) or {}
    backend_kwargs = args.get("backend_kwargs", {}) or {}

    replicas = _required_int(metadata, "replicas")
    tp = _required_int(metadata, "tp")
    model = (
        str(metadata.get("model_name") or "").strip()
        or str(backend_kwargs.get("model") or "").strip()
        or "unknown"
    )
    data_spec = (
        str(metadata.get("data_spec") or "").strip()
        or ", ".join(str(item) for item in (args.get("data") or []))
        or "unknown"
    )
    profile = str(metadata.get("profile") or args.get("profile") or "unknown")
    backend = str(metadata.get("backend") or args.get("backend") or "unknown")
    runtime_args = str(metadata.get("runtime_args") or "").strip()
    execution_name = str(metadata.get("execution_name") or "unknown")
    platform = str(metadata.get("platform") or "unknown")
    mode = str(metadata.get("mode") or "unknown")
    version = str(metadata.get("version") or "unknown")
    accelerator = str(metadata.get("accelerator") or "unknown")

    rates = args.get("rate") or []
    try:
        load_points = ", ".join(
            str(int(value)) for value in sorted(float(v) for v in rates)
        )
    except (TypeError, ValueError):
        load_points = ", ".join(str(v) for v in rates)
    load_points = load_points or "unknown"

    start = metrics_summary.get("benchmark_start_time")
    end = metrics_summary.get("benchmark_end_time")
    duration_minutes = None
    if start and end:
        try:
            start_dt = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
            duration_minutes = (end_dt - start_dt).total_seconds() / 60.0
        except ValueError:
            duration_minutes = None
    duration_text = (
        f"{duration_minutes:.1f} min" if duration_minutes is not None else "unknown"
    )

    return RunReportMetadata(
        model=model,
        version=version,
        accelerator=accelerator,
        tp=tp,
        replicas=replicas,
        runtime_args=runtime_args,
        execution_name=execution_name,
        platform=platform,
        mode=mode,
        data_spec=data_spec,
        profile=profile,
        backend=backend,
        load_points=load_points,
        duration=duration_text,
        step=str(metrics_summary.get("query_step", "unknown")),
        total_gpus=float(tp * replicas),
    )


def _load_benchmark_phase_segments(
    paths: ArtifactPaths,
) -> list[dict[str, float | int | str]]:
    payload = _load_json(paths.benchmark_json)
    if not isinstance(payload, dict):
        return []
    benchmarks = payload.get("benchmarks", [])
    if not isinstance(benchmarks, list) or not benchmarks:
        return []

    numeric_times = [
        float(item["start_time"])
        for item in benchmarks
        if isinstance(item, dict) and "start_time" in item
    ]
    if not numeric_times:
        return []
    global_start = min(numeric_times)

    segments: list[dict[str, float | int | str]] = []
    for index, item in enumerate(benchmarks):
        if not isinstance(item, dict):
            continue
        try:
            start = float(item["start_time"])
            end = float(item["end_time"])
        except (KeyError, TypeError, ValueError):
            continue
        concurrency = _extract_intended_concurrency(item, index)
        try:
            concurrency_int = int(concurrency)
        except (TypeError, ValueError):
            continue
        segments.append(
            {
                "concurrency": concurrency_int,
                "label": f"C={concurrency_int}",
                "start_ts": start,
                "end_ts": end,
                "start_min": (start - global_start) / 60.0,
                "end_min": (end - global_start) / 60.0,
                "mid_min": ((start + end) / 2.0 - global_start) / 60.0,
            }
        )
    return segments


def _common_timestamps(*series_dicts: dict[int, float]) -> list[int]:
    non_empty = [set(item.keys()) for item in series_dicts if item]
    if not non_empty:
        return []
    timestamps = sorted(set.intersection(*non_empty))
    return timestamps


def _heatmap_matrix(
    grouped: dict[str, dict[int, float]],
    timestamps: list[int],
    labels: list[str] | None = None,
) -> tuple[list[str], list[list[float]]]:
    ordered_labels = labels or _sort_labels(list(grouped.keys()))
    matrix = [
        [float(grouped[label].get(ts, np.nan)) for ts in timestamps]
        for label in ordered_labels
    ]
    return ordered_labels, matrix


def _build_replica_heatmap_figure(
    minutes: list[float],
    timestamps: list[int],
    token_rate_by_pod: dict[str, dict[int, float]],
    success_rate_by_pod: dict[str, dict[int, float]],
    segments: list[dict[str, float | int | str]],
) -> go.Figure:
    layout = _base_layout(height=590)
    layout["margin"] = {"l": 70, "r": 70, "t": 150, "b": 60}
    pod_labels = _sort_labels(list(token_rate_by_pod.keys()))
    _, token_matrix = _heatmap_matrix(token_rate_by_pod, timestamps, pod_labels)
    _, success_matrix = _heatmap_matrix(success_rate_by_pod, timestamps, pod_labels)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(
            "Output token rate by replica",
            "Successful request rate by replica",
        ),
    )
    fig.add_trace(
        go.Heatmap(
            x=minutes,
            y=pod_labels,
            z=token_matrix,
            colorscale="Blues",
            zmin=0,
            colorbar={
                "title": "Output token rate (tok/s)",
                "x": 1.02,
                "y": 0.79,
                "len": 0.36,
            },
            hovertemplate="Replica %{y}<br>Run minute %{x:.1f}<br>Output tok/s %{z:.1f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=minutes,
            y=pod_labels,
            z=success_matrix,
            colorscale="Viridis",
            zmin=0,
            colorbar={
                "title": "Successful request rate (req/s)",
                "x": 1.02,
                "y": 0.21,
                "len": 0.36,
            },
            hovertemplate="Replica %{y}<br>Run minute %{x:.1f}<br>Successful req/s %{z:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title={
            "text": _title_text(
                "(w) Replica imbalance heatmaps",
                "Uniform bands are better. Persistent hot or cold replicas mean throughput delivery or completion quality is uneven across the serving pool.",
            ),
            "x": 0.5,
            "xanchor": "center",
            "y": 0.97,
        },
        **layout,
    )
    fig.update_xaxes(title_text="Run progress (min)", row=2, col=1)
    fig.update_yaxes(title_text="Replica", automargin=True)
    _apply_phase_bands(fig, segments, label_y=1.14)
    return fig


def _build_replica_share_figure(
    minutes: list[float],
    timestamps: list[int],
    token_rate_by_pod: dict[str, dict[int, float]],
    segments: list[dict[str, float | int | str]],
) -> go.Figure:
    labels, share_matrix = _grouped_share_matrix(token_rate_by_pod, timestamps)
    palette = list(REPORT_COLOR_PALETTE)

    fig = go.Figure()
    for index, label in enumerate(labels):
        fig.add_trace(
            go.Scatter(
                x=minutes,
                y=share_matrix[index],
                mode="lines",
                name=label,
                line={"color": palette[index % len(palette)], "width": 1.8},
                hovertemplate=f"Replica {label}<br>Run minute %{{x:.1f}}<br>Share of cluster output %{{y:.1%}}<extra></extra>",
            )
        )

    fig.add_hline(
        y=1.0 / max(len(labels), 1),
        line_dash="dash",
        line_color=COLORS["gray"],
        line_width=1,
    )
    fig.update_layout(
        title={
            "text": _title_text(
                "(x) Replica throughput share over time",
                "Computed as replica output tok/s divided by cluster output tok/s at each timestamp. Closer to the equal-share guide is better.",
            ),
            "x": 0.5,
            "xanchor": "center",
            "y": 0.97,
        },
        **_base_layout(),
    )
    fig.update_xaxes(title_text="Run progress (min)")
    fig.update_yaxes(
        title_text="Replica share of cluster output (%)",
        range=[0, 1],
        tickformat=".0%",
    )
    _apply_common_axes(fig)
    _apply_phase_bands(fig, segments)
    return fig


def _build_gpu_heatmap_figure(
    minutes: list[float],
    timestamps: list[int],
    gpu_util_by_rank: dict[str, dict[int, float]],
    segments: list[dict[str, float | int | str]],
) -> go.Figure:
    layout = _base_layout(height=570)
    layout["margin"] = {"l": 70, "r": 70, "t": 145, "b": 60}
    rank_labels = _sort_labels(list(gpu_util_by_rank.keys()))
    _, util_matrix = _heatmap_matrix(gpu_util_by_rank, timestamps, rank_labels)
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=minutes,
            y=rank_labels,
            z=util_matrix,
            colorscale="YlOrRd",
            zmin=0,
            zmax=100,
            colorbar={"title": "GPU utilization (%)"},
            hovertemplate="GPU %{y}<br>Run minute %{x:.1f}<br>Utilization %{z:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": _title_text(
                "(z) GPU heatmap by replica/rank",
                "Balanced utilization bands are better. Persistent cold ranks imply underfilled work, while isolated hot ranks suggest uneven TP placement or scheduler skew.",
            ),
            "x": 0.5,
            "xanchor": "center",
            "y": 0.97,
        },
        **layout,
    )
    fig.update_xaxes(title_text="Run progress (min)")
    fig.update_yaxes(title_text="Replica / GPU rank", automargin=True)
    _apply_phase_bands(fig, segments, label_y=1.15)
    return fig


def _build_gpu_frontier_figure(
    timestamps: list[int],
    avg_gpu_util: np.ndarray,
    total_output_tok_rate: np.ndarray,
    ttft_p99: np.ndarray,
    queue_p99: np.ndarray,
    total_gpus: float,
    segments: list[dict[str, float | int | str]],
) -> go.Figure:
    layout = _base_layout()
    layout["margin"] = {"l": 70, "r": 130, "t": 110, "b": 60}
    layout["legend"] = {
        "bgcolor": "rgba(255,255,255,0.88)",
        "borderwidth": 0,
        "font": {"size": 11},
        "x": 0.02,
        "y": 0.98,
    }
    tok_per_gpu = total_output_tok_rate / max(total_gpus, 1.0)
    phase_gpu = _phase_reducer(timestamps, avg_gpu_util, segments)
    phase_tok = _phase_reducer(timestamps, tok_per_gpu, segments)
    phase_ttft = _phase_reducer(timestamps, ttft_p99, segments)
    phase_queue = _phase_reducer(timestamps, queue_p99, segments)
    frontier_points = sorted(
        zip(phase_gpu, phase_tok),
        key=lambda pair: float(pair[0]["concurrency"]),
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[float(gpu_entry["value"]) for gpu_entry, _ in frontier_points],
            y=[float(tok_entry["value"]) for _, tok_entry in frontier_points],
            mode="lines",
            name="Load trajectory",
            line={"color": COLORS["gray"], "width": 1.6, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[entry["value"] for entry in phase_gpu],
            y=[entry["value"] for entry in phase_tok],
            mode="markers+text",
            text=[str(entry["concurrency"]) for entry in phase_gpu],
            textposition="top center",
            name="Load point",
            marker={
                "size": 16,
                "color": [entry["value"] for entry in phase_ttft],
                "colorscale": "YlGnBu",
                "showscale": True,
                "colorbar": {"title": "TTFT p99 (s)", "x": 1.12},
                "line": {"color": "white", "width": 0.7},
            },
            customdata=np.array(
                [
                    [entry["value"] for entry in phase_queue],
                    [entry["label"] for entry in phase_gpu],
                ],
                dtype=object,
            ).T,
            hovertemplate=(
                "Concurrency %{text}<br>"
                "Phase %{customdata[1]}<br>"
                "Avg GPU util %{x:.1f}%<br>"
                "Output tok/s/GPU %{y:.1f}<br>"
                "TTFT p99 %{marker.color:.2f}s<br>"
                "Queue p99 %{customdata[0]:.2f}s<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title={
            "text": _title_text(
                "(ab) Per-GPU productivity frontier by load point",
                "Each marker uses phase-median average GPU utilization and output tok/s divided by GPU count. Higher and further right is better only if TTFT stays controlled.",
            ),
            "x": 0.5,
            "xanchor": "center",
            "y": 0.97,
        },
        **layout,
    )
    fig.update_xaxes(title_text="Average GPU utilization (%)", range=[0, 100])
    fig.update_yaxes(title_text="Output token throughput (tok/s/GPU)")
    _apply_common_axes(fig)
    return fig


def _build_system_figures(
    paths: ArtifactPaths,
    metadata: RunReportMetadata,
    segments: list[dict[str, float | int | str]],
) -> list[go.Figure]:
    figures: list[go.Figure] = []

    queue_p99 = _rows_to_series(_metric_rows(paths, "queue_time_p99_seconds"))
    ttft_p99 = _rows_to_series(_metric_rows(paths, "ttft_p99_seconds"))
    avg_gpu_util = _rows_to_series(_metric_rows(paths, "avg_gpu_utilization"))
    total_output_tok_rate = _rows_to_series(
        _metric_rows(paths, "generation_token_rate_sum_per_second")
    )

    token_rate_by_pod = _rows_to_grouped_series(
        _metric_rows(paths, "generation_token_rate_per_second"),
        _workload_pod_label,
    )
    success_rate_by_pod = _rows_to_grouped_series(
        _metric_rows(paths, "request_success_rate_by_pod"),
        _workload_pod_label,
    )
    gpu_util_by_rank = _rows_to_grouped_series(
        _metric_rows(paths, "gpu_utilization_by_pod"),
        _gpu_row_label,
    )

    if token_rate_by_pod and success_rate_by_pod:
        timestamps = sorted(
            set.intersection(
                *(
                    [set(series.keys()) for series in token_rate_by_pod.values()]
                    + [set(series.keys()) for series in success_rate_by_pod.values()]
                )
            )
        )
        if timestamps:
            minutes = _relative_minutes(timestamps)
            figures.append(
                _build_replica_heatmap_figure(
                    minutes,
                    timestamps,
                    token_rate_by_pod,
                    success_rate_by_pod,
                    segments,
                )
            )
            figures.append(
                _build_replica_share_figure(
                    minutes,
                    timestamps,
                    token_rate_by_pod,
                    segments,
                )
            )

    if gpu_util_by_rank:
        timestamps = sorted(
            set.intersection(
                *[set(series.keys()) for series in gpu_util_by_rank.values()]
            )
        )
        if timestamps:
            figures.append(
                _build_gpu_heatmap_figure(
                    _relative_minutes(timestamps),
                    timestamps,
                    gpu_util_by_rank,
                    segments,
                )
            )

    frontier_timestamps = _common_timestamps(
        avg_gpu_util, total_output_tok_rate, ttft_p99, queue_p99
    )
    if frontier_timestamps:
        figures.append(
            _build_gpu_frontier_figure(
                frontier_timestamps,
                _align_series(frontier_timestamps, avg_gpu_util),
                _align_series(frontier_timestamps, total_output_tok_rate),
                _align_series(frontier_timestamps, ttft_p99),
                _align_series(frontier_timestamps, queue_p99),
                metadata.total_gpus,
                segments,
            )
        )

    return figures


def _build_benchmark_figures(
    paths: ArtifactPaths,
    metadata: RunReportMetadata,
) -> list[go.Figure]:
    benchmark_rows = summarize_benchmarks(
        load_benchmarks(paths.benchmark_json),
        strict_slo=(200.0, 25.0),
        relaxed_slo=(500.0, 40.0),
        gpu_count=float(metadata.total_gpus),
    )
    ttft_thresholds = parse_thresholds("100,150,200,300,500,750,1000,2000,4000,8000")
    itl_thresholds = parse_thresholds("15,20,25,30,40,50,60,80,100")
    benchmark_figures = build_plotly_benchmark_figures(
        benchmark_rows,
        ttft_thresholds,
        itl_thresholds,
        10,
    )
    return [
        figure for slug, figure in benchmark_figures if slug != "best_concurrency_sweep"
    ]


def _resolve_output_path(
    artifacts_dir: Path,
    *,
    output_dir: Path | None,
    output_file: Path | None,
) -> Path:
    if output_file is not None:
        return output_file.resolve()
    if output_dir is not None:
        return output_dir.resolve() / "full_run_artifacts_report.html"
    return (artifacts_dir / "reports" / "full_run_artifacts_report.html").resolve()


def _build_header_figure(metadata: RunReportMetadata, columns: int) -> go.Figure:
    width = columns * FIGURE_WIDTH + max(columns - 1, 0) * 24
    subtitle = (
        f"{metadata.data_spec} | "
        f"concurrency levels {metadata.load_points} | "
        f"TP {metadata.tp} | replicas {metadata.replicas}"
    )

    fig = go.Figure()
    fig.update_layout(
        width=width,
        height=110,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": 6, "r": 6, "t": 8, "b": 8},
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
                "text": metadata.model,
                "font": {
                    "family": "Times New Roman, Georgia, serif",
                    "size": 28,
                    "color": COLORS["black"],
                },
            },
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0.0,
                "y": 0.30,
                "xanchor": "left",
                "yanchor": "middle",
                "showarrow": False,
                "align": "left",
                "text": subtitle,
                "font": {
                    "family": "Arial, Helvetica, sans-serif",
                    "size": 13,
                    "color": COLORS["gray"],
                },
            },
        ],
    )
    return fig


def _render_run_report_html(
    *,
    figures: list[go.Figure],
    metadata: RunReportMetadata,
    output_path: Path,
    columns: int,
) -> None:
    plotly_js = get_plotlyjs()
    parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        f"<title>{escape(metadata.model)}</title>",
        f"<script type='text/javascript'>{plotly_js}</script>",
        "</head>",
        "<body style='background: white; margin: 12px;'>",
        "<div style='overflow-x: auto;'>",
        "<table cellspacing='12' cellpadding='0' style='border-collapse: separate;'>",
    ]

    header_html = _build_header_figure(metadata, columns).to_html(
        include_plotlyjs=False,
        full_html=False,
        config=PLOTLY_CONFIG,
    )
    parts.append(
        f"<tr><td colspan='{columns}' style='vertical-align: top;'>{header_html}</td></tr>"
    )

    for index, figure in enumerate(figures):
        if index % columns == 0:
            parts.append("<tr>")
        parts.append(
            "<td style='vertical-align: top;'>"
            + figure.to_html(
                include_plotlyjs=False,
                full_html=False,
                config=PLOTLY_CONFIG,
            )
            + "</td>"
        )
        if index % columns == columns - 1:
            parts.append("</tr>")

    if figures and len(figures) % columns != 0:
        for _ in range(columns - (len(figures) % columns)):
            parts.append("<td></td>")
        parts.append("</tr>")

    parts.extend(["</table>", "</div>", "</body>", "</html>"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def generate_run_report(
    *,
    artifacts_dir: Path,
    output_dir: Path | None = None,
    output_file: Path | None = None,
    columns: int = 3,
) -> Path:
    paths = _resolve_artifact_paths(artifacts_dir)
    metadata = _load_report_metadata(paths)
    segments = _load_benchmark_phase_segments(paths)
    benchmark_figures = _build_benchmark_figures(paths, metadata)
    system_figures = _build_system_figures(paths, metadata, segments)
    output_path = _resolve_output_path(
        paths.root,
        output_dir=output_dir,
        output_file=output_file,
    )
    _render_run_report_html(
        figures=[*benchmark_figures, *system_figures],
        metadata=metadata,
        output_path=output_path,
        columns=max(columns, 1),
    )
    return output_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate the BenchFlow post-run report from a collected artifact directory."
    )
    parser.add_argument(
        "--artifacts-dir",
        required=True,
        help="Path to the BenchFlow artifact directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory where the HTML report should be written.",
    )
    parser.add_argument(
        "--output-file",
        default="",
        help="Exact output HTML path. Overrides --output-dir.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=3,
        help="Number of columns for the diagnostics section.",
    )
    args = parser.parse_args()
    path = generate_run_report(
        artifacts_dir=Path(args.artifacts_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        output_file=Path(args.output_file) if args.output_file else None,
        columns=args.columns,
    )
    print(path)


if __name__ == "__main__":
    main()
