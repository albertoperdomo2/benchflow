from __future__ import annotations

import gzip
import html
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from ..plotting import REPORT_COLOR_PALETTE


_REPORT_FILENAME = "trace_distribution_report.html"
_REPORT_WIDTH = 1440
_METRIC_PANEL_WIDTH = 696
_REPORT_FONT = "Arial, Helvetica, sans-serif"
_TITLE_FONT = "Times New Roman, Georgia, serif"
_PLOTLY_CONFIG = {"displaylogo": False, "responsive": True}
_PAPER_COLOR = "#ffffff"
_TEXT_COLOR = "#222222"
_MUTED_COLOR = "#666666"
_GROUP_ORDER = (
    "Model latency",
    "Span duration",
    "Token usage",
    "EPP scheduling",
    "Request attributes",
    "Other trace attributes",
)
_TAG_TITLES = {
    "gen_ai.latency.e2e": "End-to-end latency",
    "gen_ai.latency.time_in_model_decode": "Model decode time",
    "gen_ai.latency.time_in_model_inference": "Model inference time",
    "gen_ai.latency.time_in_model_prefill": "Model prefill time",
    "gen_ai.latency.time_in_queue": "Queue wait time",
    "gen_ai.latency.time_to_first_token": "Time to first token",
    "gen_ai.request.max_tokens": "Requested maximum output tokens",
    "gen_ai.request.n": "Requested completions",
    "gen_ai.request.temperature": "Sampling temperature",
    "gen_ai.request.top_p": "Top-p sampling threshold",
    "gen_ai.usage.completion_tokens": "Completion tokens",
    "gen_ai.usage.prompt_tokens": "Prompt tokens",
    "llm_d.epp.filter.candidate_endpoints": "EPP filter candidate endpoints",
    "llm_d.epp.filter.filtered_endpoints": "EPP filtered endpoints",
    "llm_d.epp.picker.candidate_endpoints": "EPP picker candidate endpoints",
    "llm_d.epp.picker.top_scores": "EPP selected endpoint score",
    "request_prio": "Request priority",
}


@dataclass
class TraceMetric:
    key: str
    title: str
    group: str
    unit: str
    source: str
    values: list[float] = field(default_factory=list)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _iter_spans(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                yield value


def _humanize(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("_", " ").replace(".", " ")).strip()


def _operation_title(span: dict[str, Any]) -> str:
    service_name = str(span.get("service_name") or "service")
    role = "EPP" if service_name.endswith("-epp") else "vLLM"
    operation = _humanize(str(span.get("operation_name") or "span"))
    return f"{role} {operation} duration"


def _tag_group(key: str) -> str:
    if key.startswith("gen_ai.latency."):
        return "Model latency"
    if key.startswith("gen_ai.usage."):
        return "Token usage"
    if key.startswith("llm_d.epp."):
        return "EPP scheduling"
    if key.startswith("gen_ai.request.") or key == "request_prio":
        return "Request attributes"
    return "Other trace attributes"


def _tag_unit(key: str) -> str:
    if key.startswith("gen_ai.latency."):
        return "ms"
    if key.endswith("_tokens") or ".tokens" in key:
        return "tokens"
    if key.endswith("endpoints"):
        return "endpoints"
    if key.endswith("scores"):
        return "score"
    if key == "request_prio":
        return "priority"
    return "value"


def _numeric_tag_values(tag: dict[str, Any]) -> list[float]:
    value = tag.get("value")
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return [float(value)]
    if not isinstance(value, str):
        return []
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(decoded, list):
        return []
    return [
        float(item)
        for item in decoded
        if not isinstance(item, bool)
        and isinstance(item, (int, float))
        and math.isfinite(float(item))
    ]


def collect_trace_metrics(trace_path: Path) -> list[TraceMetric]:
    metrics: dict[str, TraceMetric] = {}
    for span in _iter_spans(trace_path):
        operation = str(span.get("operation_name") or "span")
        service = str(span.get("service_name") or "service")
        duration_key = f"span.duration.{service}.{operation}"
        duration = span.get("duration_microseconds")
        if (
            not isinstance(duration, bool)
            and isinstance(duration, (int, float))
            and math.isfinite(float(duration))
        ):
            metric = metrics.setdefault(
                duration_key,
                TraceMetric(
                    key=duration_key,
                    title=_operation_title(span),
                    group="Span duration",
                    unit="ms",
                    source=f"span duration · {operation}",
                ),
            )
            metric.values.append(float(duration) / 1000.0)

        for tag in span.get("tags") or []:
            if not isinstance(tag, dict):
                continue
            key = str(tag.get("key") or "").strip()
            if not key:
                continue
            values = _numeric_tag_values(tag)
            if not values:
                continue
            unit = _tag_unit(key)
            if key.startswith("gen_ai.latency."):
                values = [value * 1000.0 for value in values]
            metric = metrics.setdefault(
                key,
                TraceMetric(
                    key=key,
                    title=_TAG_TITLES.get(key, _humanize(key).title()),
                    group=_tag_group(key),
                    unit=unit,
                    source=f"span attribute · {key}",
                ),
            )
            metric.values.extend(values)

    group_order = {name: index for index, name in enumerate(_GROUP_ORDER)}
    return sorted(
        metrics.values(),
        key=lambda metric: (
            group_order.get(metric.group, len(group_order)),
            metric.title.lower(),
        ),
    )


def _statistics(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    stddev = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    return {
        "count": float(len(array)),
        "min": float(np.min(array)),
        "p25": float(np.percentile(array, 25)),
        "p50": float(np.percentile(array, 50)),
        "mean": mean,
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
        "stddev": stddev,
        "cv": abs(stddev / mean) if mean else 0.0,
    }


def _format_number(value: float) -> str:
    absolute = abs(value)
    if absolute == 0:
        return "0"
    if absolute >= 1000:
        return f"{value:,.1f}"
    if absolute >= 100:
        return f"{value:.1f}"
    if absolute >= 10:
        return f"{value:.2f}"
    if absolute >= 1:
        return f"{value:.3f}"
    if absolute >= 0.01:
        return f"{value:.4f}"
    return f"{value:.3e}"


def _panel_label(index: int) -> str:
    value = index + 1
    label = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        label = chr(97 + remainder) + label
    return label


def _kernel_density(values: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(values) < 2:
        return None
    stddev = float(np.std(values, ddof=1))
    value_range = float(np.ptp(values))
    if not math.isfinite(stddev) or stddev <= 0 or value_range <= 0:
        return None
    bandwidth = 1.06 * stddev * len(values) ** (-1 / 5)
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        return None
    padding = max(3 * bandwidth, value_range * 0.08)
    grid = np.linspace(
        float(np.min(values)) - padding,
        float(np.max(values)) + padding,
        240,
    )
    scaled = (grid[:, np.newaxis] - values[np.newaxis, :]) / bandwidth
    density = np.exp(-0.5 * scaled**2).sum(axis=1)
    density /= len(values) * bandwidth * math.sqrt(2 * math.pi)
    return grid, density


def _metric_figure(metric: TraceMetric, index: int) -> go.Figure:
    stats = _statistics(metric.values)
    values = np.asarray(metric.values, dtype=float)
    unit = html.escape(metric.unit)
    sorted_values = np.sort(values)
    cumulative = np.arange(1, len(sorted_values) + 1, dtype=float) / len(sorted_values)
    bins = min(16, max(4, int(math.ceil(math.sqrt(len(values))))))
    color = REPORT_COLOR_PALETTE[index % len(REPORT_COLOR_PALETTE)]
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Observed distribution",
            "Empirical cumulative distribution",
        ),
        horizontal_spacing=0.1,
    )
    figure.add_trace(
        go.Histogram(
            x=values,
            nbinsx=bins,
            histnorm="probability density",
            opacity=0.48,
            marker={"color": color, "line": {"color": color, "width": 0.8}},
            hovertemplate=(
                f"{metric.unit}: %{{x}}<br>probability density: %{{y:.4g}}"
                "<extra></extra>"
            ),
            name="Observed histogram",
        ),
        row=1,
        col=1,
    )
    density = _kernel_density(values)
    if density is not None:
        density_x, density_y = density
        figure.add_trace(
            go.Scatter(
                x=density_x,
                y=density_y,
                mode="lines",
                line={"color": color, "width": 3},
                hovertemplate=(
                    f"{metric.unit}: %{{x}}<br>kernel density: %{{y:.4g}}"
                    "<extra></extra>"
                ),
                name="Gaussian kernel density",
            ),
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Scatter(
            x=sorted_values,
            y=cumulative,
            mode="lines+markers",
            line={"color": color, "width": 2},
            marker={"color": color, "size": 6},
            hovertemplate=f"{metric.unit}: %{{x}}<br>ECDF: %{{y:.1%}}<extra></extra>",
            name="ECDF",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    reference_lines = (
        ("mean", "Mean", "#222222", "solid"),
        ("p50", "P50", "#777777", "dash"),
        ("p95", "P95", "#b33a3a", "dot"),
    )
    for key, _label, line_color, dash in reference_lines:
        figure.add_vline(
            x=stats[key],
            line={"color": line_color, "width": 1.5, "dash": dash},
            row=1,
            col=1,
        )
    figure.update_layout(
        width=_METRIC_PANEL_WIDTH,
        height=410,
        paper_bgcolor=_PAPER_COLOR,
        plot_bgcolor=_PAPER_COLOR,
        font={"family": _REPORT_FONT, "size": 12, "color": _TEXT_COLOR},
        margin={"l": 72, "r": 24, "t": 52, "b": 82},
        bargap=0.08,
        showlegend=False,
    )
    for column in (1, 2):
        figure.update_xaxes(
            title={
                "text": f"{html.escape(metric.title)} ({unit})",
                "font": {"size": 10},
                "standoff": 8,
            },
            showgrid=True,
            gridcolor="#e8e8e8",
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor="#222222",
            mirror=True,
            row=1,
            col=column,
        )
        figure.update_yaxes(
            title={
                "text": (
                    f"Density (1/{unit})"
                    if column == 1
                    else "Cumulative probability (%)"
                ),
                "font": {"size": 10},
                "standoff": 4,
            },
            showgrid=True,
            gridcolor="#e8e8e8",
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor="#222222",
            mirror=True,
            row=1,
            col=column,
        )
    figure.update_yaxes(range=[0, 1.02], tickformat=".0%", row=1, col=2)
    return figure


def _metric_panel(metric: TraceMetric, index: int) -> str:
    stats = _statistics(metric.values)
    color = REPORT_COLOR_PALETTE[index % len(REPORT_COLOR_PALETTE)]
    density_available = (
        _kernel_density(np.asarray(metric.values, dtype=float)) is not None
    )
    statistics_line = (
        f"n={int(stats['count'])} · min {_format_number(stats['min'])} · "
        f"P25 {_format_number(stats['p25'])} · P50 {_format_number(stats['p50'])} · "
        f"P75 {_format_number(stats['p75'])} · P90 {_format_number(stats['p90'])} · "
        f"P95 {_format_number(stats['p95'])} · P99 {_format_number(stats['p99'])} · "
        f"max {_format_number(stats['max'])}"
    )
    details_line = (
        f"mean {_format_number(stats['mean'])} · SD {_format_number(stats['stddev'])} · "
        f"CV {stats['cv']:.3f} · unit {html.escape(metric.unit)} · "
        f"source {html.escape(metric.source)}"
    )
    density_legend = (
        "<span class='metric-legend-item'><span class='metric-legend-line density-line' "
        f"style='border-top-color:{color}'></span>"
        "Gaussian kernel density</span>"
        if density_available
        else "<span class='metric-legend-note'>Kernel density omitted: constant or singleton series</span>"
    )
    figure_html = _metric_figure(metric, index).to_html(
        include_plotlyjs=False,
        full_html=False,
        config=_PLOTLY_CONFIG,
    )
    return (
        "<section class='metric-panel'>"
        f"<h2>({_panel_label(index)}) {html.escape(metric.title)}</h2>"
        f"<p class='metric-statistics'>{statistics_line}<br>{details_line}</p>"
        "<div class='metric-legend'>"
        "<span class='metric-legend-item'><span class='metric-legend-box' "
        f"style='border-color:{color};background:{color}7a'></span>Observed histogram</span>"
        f"{density_legend}"
        "<span class='metric-legend-item'><span class='metric-legend-line mean-line'></span>Mean</span>"
        "<span class='metric-legend-item'><span class='metric-legend-line p50-line'></span>P50</span>"
        "<span class='metric-legend-item'><span class='metric-legend-line p95-line'></span>P95</span>"
        "</div>"
        f"{figure_html}"
        "</section>"
    )


def _header_figure(*, title: str, subtitle_lines: list[str]) -> go.Figure:
    subtitle = "<br>".join(
        f"<span style='font-size:13px;color:{_MUTED_COLOR}'>{html.escape(line)}</span>"
        for line in subtitle_lines
    )
    figure = go.Figure()
    figure.update_layout(
        width=_REPORT_WIDTH,
        height=max(120, 80 + len(subtitle_lines) * 18),
        paper_bgcolor=_PAPER_COLOR,
        plot_bgcolor=_PAPER_COLOR,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        xaxis={"visible": False},
        yaxis={"visible": False},
        showlegend=False,
        annotations=[
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0.0,
                "y": 0.8,
                "xanchor": "left",
                "yanchor": "middle",
                "showarrow": False,
                "align": "left",
                "text": title,
                "font": {"family": _TITLE_FONT, "size": 28, "color": _TEXT_COLOR},
            },
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0.0,
                "y": 0.32,
                "xanchor": "left",
                "yanchor": "middle",
                "showarrow": False,
                "align": "left",
                "text": subtitle,
                "font": {"family": _REPORT_FONT, "size": 13, "color": _MUTED_COLOR},
            },
        ],
    )
    return figure


def _metadata_lines(metadata: dict[str, Any], summary: dict[str, Any]) -> list[str]:
    services = summary.get("services") or []
    return [
        f"Model: {metadata.get('model_name') or 'unknown'}",
        (
            f"Platform: {metadata.get('version') or metadata.get('platform') or 'unknown'}"
            f" · mode {metadata.get('mode') or 'unknown'} · TP {metadata.get('tp', 'unknown')}"
            f" · replicas {metadata.get('replicas', 'unknown')}"
        ),
        (
            f"Trace window: {summary.get('benchmark_start_time') or 'unknown'} to "
            f"{summary.get('benchmark_end_time') or 'unknown'}"
        ),
        (
            f"Traces: {summary.get('trace_count', 0)} · spans: {summary.get('span_count', 0)}"
            f" · complete multi-service traces: {summary.get('complete_multi_service_traces', 0)}"
            f" · sampling: {summary.get('sample_ratio', 'unknown')}"
        ),
        f"Services: {', '.join(str(service) for service in services) or 'unknown'}",
    ]


def _summary_strip(summary: dict[str, Any], metric_count: int) -> str:
    items = (
        ("Traces", summary.get("trace_count", 0)),
        ("Spans", summary.get("span_count", 0)),
        ("Services", len(summary.get("services") or [])),
        ("Complete trees", summary.get("complete_multi_service_traces", 0)),
        ("Metric series", metric_count),
    )
    return (
        "<div class='summary-strip'>"
        + "".join(
            "<div class='summary-item'>"
            f"<span class='summary-value'>{html.escape(str(value))}</span>"
            f"<span class='summary-label'>{html.escape(label)}</span>"
            "</div>"
            for label, value in items
        )
        + "</div>"
    )


def _methodology() -> str:
    return """
<footer class="methodology">
  <p><sup>*</sup> <strong>Methodology.</strong> Each panel uses every finite numeric
  observation found in the normalized trace archive. Span durations are converted
  from microseconds to milliseconds; <code>gen_ai.latency.*</code> attributes are
  converted from seconds to milliseconds; and numeric JSON arrays, including EPP
  top scores, are flattened. Percentiles use linear interpolation. SD is the sample
  standard deviation and CV is SD divided by the absolute mean. Histograms are
  normalized to probability density and overlaid with a Gaussian-kernel density
  estimate, omitted for singleton and constant-valued series. ECDF is the fraction
  of observations at or below each value.</p>
</footer>
"""


def generate_trace_distribution_report(
    *,
    artifacts_dir: Path,
    output_file: Path | None = None,
) -> Path | None:
    trace_dir = artifacts_dir / "traces"
    summary_path = trace_dir / "trace-summary.json"
    traces_path = trace_dir / "traces.jsonl.gz"
    if not summary_path.exists() or not traces_path.exists():
        return None
    summary = _load_json(summary_path)
    if not summary.get("enabled") or summary.get("status") != "collected":
        return None
    metrics = collect_trace_metrics(traces_path)
    if not metrics:
        return None
    metadata_path = artifacts_dir / "metadata.json"
    metadata = _load_json(metadata_path) if metadata_path.exists() else {}
    resolved_output = output_file or (artifacts_dir / "reports" / _REPORT_FILENAME)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Bench Flow Tracing Distribution Report</title>",
        f"<script type='text/javascript'>{get_plotlyjs()}</script>",
        """
<style>
  body { margin: 12px; background: #fff; color: #222; font-family: Arial, Helvetica, sans-serif; }
  .report-shell { overflow-x: auto; }
  .report-table { border-collapse: separate; border-spacing: 12px; }
  .section-title { margin: 28px 0 6px; font-size: 20px; font-weight: 700; }
  .summary-strip { display: flex; align-items: baseline; margin: 2px 0 14px; padding: 5px 0 7px; border-bottom: 1px solid #d8d8d8; }
  .summary-item { display: inline-flex; align-items: baseline; gap: 6px; padding: 0 22px; border-left: 1px solid #d8d8d8; }
  .summary-item:first-child { padding-left: 0; border-left: 0; }
  .summary-value { font: 21px/1 "Times New Roman", Georgia, serif; }
  .summary-label { color: #666; font-size: 10px; text-transform: uppercase; letter-spacing: .05em; }
  .metric-cell { width: 696px; vertical-align: top; }
  .metric-panel { width: 696px; margin: 20px 0 28px; background: white; }
  .metric-panel h2 { margin: 0 0 6px; color: #222; font-size: 21px; font-weight: 700; line-height: 1.2; }
  .metric-statistics { margin: 0 0 10px; color: #555; font-size: 13px; line-height: 1.45; }
  .metric-legend { display: flex; flex-wrap: wrap; align-items: center; gap: 7px 14px; margin: 8px 0 2px; color: #222; font-size: 11px; }
  .metric-legend-item { display: inline-flex; align-items: center; }
  .metric-legend-note { color: #777; font-style: italic; }
  .metric-legend-box { display: inline-block; width: 26px; height: 11px; margin-right: 7px; border: 1px solid #1f77b4; }
  .metric-legend-line { display: inline-block; width: 30px; height: 0; margin-right: 7px; border-top: 3px solid #1f77b4; }
  .density-line { border-top-width: 3px; }
  .mean-line { border-top-color: #222; border-top-width: 2px; }
  .p50-line { border-top-color: #777; border-top-style: dashed; border-top-width: 2px; }
  .p95-line { border-top-color: #b33a3a; border-top-style: dotted; border-top-width: 2px; }
  .methodology { width: 1396px; margin: 18px 0 8px; padding-top: 10px; border-top: 1px solid #d8d8d8; color: #666; font-size: 11px; line-height: 1.45; }
  .methodology p { margin: 0; }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .92em; }
  @media (max-width: 900px) { .summary-strip { flex-wrap: wrap; gap: 8px 0; } }
</style>
""",
        "</head>",
        "<body>",
        "<div class='report-shell'>",
        "<table class='report-table' cellspacing='12' cellpadding='0'>",
    ]
    header_html = _header_figure(
        title="Bench Flow Tracing Distribution Report",
        subtitle_lines=_metadata_lines(metadata, summary),
    ).to_html(include_plotlyjs=False, full_html=False, config=_PLOTLY_CONFIG)
    parts.append(f"<tr><td colspan='2'>{header_html}</td></tr>")
    parts.append(
        f"<tr><td colspan='2'>{_summary_strip(summary, len(metrics))}</td></tr>"
    )
    grouped_metrics: list[tuple[str, list[tuple[int, TraceMetric]]]] = []
    for index, metric in enumerate(metrics):
        if not grouped_metrics or grouped_metrics[-1][0] != metric.group:
            grouped_metrics.append((metric.group, []))
        grouped_metrics[-1][1].append((index, metric))
    for group, entries in grouped_metrics:
        parts.append(
            "<tr><td colspan='2'>"
            f"<div class='section-title'>{html.escape(group)}</div>"
            "</td></tr>"
        )
        for offset in range(0, len(entries), 2):
            parts.append("<tr>")
            for index, metric in entries[offset : offset + 2]:
                parts.append(
                    f"<td class='metric-cell'>{_metric_panel(metric, index)}</td>"
                )
            if len(entries[offset : offset + 2]) == 1:
                parts.append("<td class='metric-cell'></td>")
            parts.append("</tr>")
    parts.extend(
        [
            f"<tr><td colspan='2'>{_methodology()}</td></tr>",
            "</table>",
            "</div>",
            "</body>",
            "</html>",
        ]
    )
    resolved_output.write_text("\n".join(parts), encoding="utf-8")
    return resolved_output
