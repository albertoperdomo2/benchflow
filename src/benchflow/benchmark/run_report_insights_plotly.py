"""
Interactive Plotly benchmark insight plots for benchmark_output.json.

This script mirrors the same benchmark diagnostics as benchmark_insight_plots.py,
but renders them as interactive Plotly figures in a single HTML report arranged
as a 3-by-N grid.

Usage:
    python benchmark_insight_plots_plotly.py
    python benchmark_insight_plots_plotly.py --input benchmark_output.json
    python benchmark_insight_plots_plotly.py --output-prefix benchmark_insights_plotly

Output:
    - <prefix>_report.html
"""

from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from textwrap import fill

import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from .run_report_insights import (
    COLORS,
    DEFAULT_GPU_COUNT,
    actual_concurrency_percentiles,
    ccdf,
    compute_slo_sweep,
    format_load_value,
    load_benchmarks,
    pareto_frontier,
    parse_thresholds,
    select_ccdf_levels,
    summarize_benchmarks,
    temporal_bins,
)


PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": [
        "lasso2d",
        "select2d",
        "autoScale2d",
    ],
}

FIGURE_HEIGHT = 580
FIGURE_WIDTH = 640
SUBTITLE_WRAP = 58


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create interactive Plotly benchmark diagnostics from benchmark_output.json."
    )
    parser.add_argument(
        "--input",
        default="benchmark_output.json",
        help="Path to benchmark_output.json",
    )
    parser.add_argument(
        "--output-prefix",
        default="benchmark_insights_plotly",
        help="Output prefix for generated HTML",
    )
    parser.add_argument(
        "--strict-ttft-ms",
        type=float,
        default=200.0,
        help="Example strict TTFT threshold used for shared metric computation.",
    )
    parser.add_argument(
        "--strict-itl-ms",
        type=float,
        default=25.0,
        help="Example strict ITL threshold used for shared metric computation.",
    )
    parser.add_argument(
        "--relaxed-ttft-ms",
        type=float,
        default=500.0,
        help="Example relaxed TTFT threshold used for shared metric computation.",
    )
    parser.add_argument(
        "--relaxed-itl-ms",
        type=float,
        default=40.0,
        help="Example relaxed ITL threshold used for shared metric computation.",
    )
    parser.add_argument(
        "--ttft-thresholds",
        default="100,150,200,300,500,750,1000,2000,4000,8000",
        help="Comma-separated TTFT thresholds in milliseconds for SLO sweep plots.",
    )
    parser.add_argument(
        "--itl-thresholds",
        default="15,20,25,30,40,50,60,80,100",
        help="Comma-separated ITL thresholds in milliseconds for SLO sweep plots.",
    )
    parser.add_argument(
        "--time-bins",
        type=int,
        default=10,
        help="Number of equal-duration bins for temporal stability plots.",
    )
    parser.add_argument(
        "--gpu-count",
        type=float,
        default=DEFAULT_GPU_COUNT,
        help="Accelerator count used for per-GPU normalization.",
    )
    return parser.parse_args()


def title_text(title: str, subtitle: str) -> str:
    wrapped = "<br>".join(fill(subtitle, width=SUBTITLE_WRAP).splitlines())
    return (
        f"{title}"
        f"<br><span style='font-size:11px;color:{COLORS['gray']}'>{wrapped}</span>"
    )


def row_load_label(row: dict) -> str:
    return str(row.get("load_label") or format_load_value(row["concurrency"]))


def load_axis_label(rows: list[dict]) -> str:
    labels = {
        str(row.get("load_axis") or "Concurrency").strip() or "Concurrency"
        for row in rows
    }
    return labels.pop() if len(labels) == 1 else "Load"


def target_load_axis_title(rows: list[dict], *, per_gpu: bool = False) -> str:
    axis_label = load_axis_label(rows)
    if axis_label == "RPS":
        return "Target RPS per GPU" if per_gpu else "Target RPS"
    if axis_label == "Concurrency":
        return "Target concurrency per GPU" if per_gpu else "Target concurrency"
    return f"Target {axis_label.lower()} per GPU" if per_gpu else f"Target {axis_label}"


def load_hover_label(rows: list[dict]) -> str:
    axis_label = load_axis_label(rows)
    return "Load point" if axis_label == "Load" else axis_label


def load_report_metadata(
    input_path: Path, rows: list[dict], gpu_count: float
) -> dict[str, str]:
    payload = json.loads(input_path.read_text())
    args = payload.get("args", {})
    first_benchmark = payload.get("benchmarks", [{}])[0]

    backend_kwargs = args.get("backend_kwargs", {}) or {}
    backend_config = first_benchmark.get("config", {}).get("backend", {})
    model = (
        backend_kwargs.get("model") or backend_config.get("model") or "Unknown model"
    )

    data_items = args.get("data") or []
    data_spec = (
        ", ".join(str(item) for item in data_items)
        if data_items
        else "Unknown workload"
    )

    profile = (
        args.get("profile")
        or first_benchmark.get("config", {}).get("profile", {}).get("type_")
        or "unknown"
    )
    backend = args.get("backend") or "unknown"
    max_seconds = args.get("max_seconds")
    config_backend = first_benchmark.get("config", {}).get("backend", {}) or {}
    config_environment = first_benchmark.get("config", {}).get("environment", {}) or {}

    load_points = ", ".join(row_load_label(row) for row in rows)

    duration_text = (
        f"{int(max_seconds)}s cap"
        if isinstance(max_seconds, (int, float))
        else "unknown cap"
    )

    def choose_value(*candidates):
        for candidate in candidates:
            if candidate is None:
                continue
            if isinstance(candidate, str) and not candidate.strip():
                continue
            return candidate
        return None

    tp_value = choose_value(
        args.get("tp"),
        args.get("tensor_parallel_size"),
        backend_kwargs.get("tp"),
        backend_kwargs.get("tensor_parallel_size"),
        config_backend.get("tp"),
        config_backend.get("tensor_parallel_size"),
        (config_environment.get("attributes") or {}).get("tp"),
        (config_environment.get("attributes") or {}).get("tensor_parallel_size"),
    )
    replica_value = choose_value(
        args.get("replicas"),
        args.get("num_replicas"),
        backend_kwargs.get("replicas"),
        backend_kwargs.get("num_replicas"),
        config_backend.get("replicas"),
        config_backend.get("num_replicas"),
        (config_environment.get("attributes") or {}).get("replicas"),
        (config_environment.get("attributes") or {}).get("num_replicas"),
    )

    if tp_value is None and replica_value is None:
        tp_value = int(gpu_count) if float(gpu_count).is_integer() else gpu_count
        replica_value = 1
    elif tp_value is None and replica_value is not None and gpu_count:
        try:
            tp_value = gpu_count / float(replica_value)
        except (TypeError, ValueError, ZeroDivisionError):
            tp_value = gpu_count
    elif replica_value is None and tp_value is not None and gpu_count:
        try:
            replica_value = gpu_count / float(tp_value)
        except (TypeError, ValueError, ZeroDivisionError):
            replica_value = 1

    def format_parallel_value(value):
        if value is None:
            return "unknown"
        try:
            numeric = float(value)
            if numeric.is_integer():
                return str(int(numeric))
        except (TypeError, ValueError):
            pass
        return str(value)

    return {
        "model": str(model),
        "data_spec": str(data_spec),
        "profile": str(profile),
        "backend": str(backend),
        "load_points": load_points,
        "duration_text": duration_text,
        "tp": format_parallel_value(tp_value),
        "replicas": format_parallel_value(replica_value),
    }


def render_report_header(metadata: dict[str, str]) -> str:
    return f"""
  <div style="margin: 10px 0 18px 0; color: #111111; font-family: 'Times New Roman', Georgia, serif;">
    <div style="font-size: 12px; letter-spacing: 0.05em; color: #666666; line-height: 1.2;">
      llm serving performance analysis
    </div>
    <div style="margin-top: 4px; font-size: 28px; font-weight: 700; line-height: 1.12;">
      Post run benchmark report
    </div>
    <div style="margin-top: 4px; font-size: 17px; font-style: italic; color: #222222; line-height: 1.25;">
      {escape(metadata["model"])}
    </div>
    <div style="margin-top: 10px; font-size: 13px; color: #333333; line-height: 1.5;">
      <span style="font-weight: 700;">workload</span> {escape(metadata["data_spec"])}
      <span style="padding: 0 10px; color: #888888;">|</span>
      <span style="font-weight: 700;">profile</span> {escape(metadata["profile"])}
      <span style="padding: 0 10px; color: #888888;">|</span>
      <span style="font-weight: 700;">load points</span> {escape(metadata["load_points"])}<br>
      <span style="font-weight: 700;">backend</span> {escape(metadata["backend"])}
      <span style="padding: 0 10px; color: #888888;">|</span>
      <span style="font-weight: 700;">TP</span> {escape(metadata["tp"])}
      <span style="padding: 0 10px; color: #888888;">|</span>
      <span style="font-weight: 700;">replicas</span> {escape(metadata["replicas"])}
      <span style="padding: 0 10px; color: #888888;">|</span>
      <span style="font-weight: 700;">duration</span> {escape(metadata["duration_text"])}
    </div>
  </div>
"""


def apply_layout(
    fig: go.Figure,
    *,
    title: str,
    subtitle: str,
    height: int = FIGURE_HEIGHT,
    legend_x: float = 0.02,
    legend_y: float = 0.98,
) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Arial, DejaVu Sans, sans-serif", "size": 12},
        title={
            "text": title_text(title, subtitle),
            "x": 0.5,
            "xanchor": "center",
            "y": 0.97,
            "yanchor": "top",
            "font": {"size": 18},
            "automargin": True,
        },
        height=height,
        width=FIGURE_WIDTH,
        margin={"l": 70, "r": 40, "t": 145, "b": 70},
        legend={
            "bgcolor": "rgba(255,255,255,0.85)",
            "bordercolor": "black",
            "borderwidth": 1,
            "font": {"size": 10},
            "x": legend_x,
            "y": legend_y,
        },
        hovermode="closest",
    )
    return fig


def apply_log_concurrency_axis(
    fig: go.Figure, axis: str, concurrency: np.ndarray
) -> None:
    tickvals = list(concurrency)
    ticktext = [
        str(int(value)) if abs(value - round(value)) < 1e-9 else f"{value:g}"
        for value in concurrency
    ]
    kwargs = {
        "type": "log",
        "tickmode": "array",
        "tickvals": tickvals,
        "ticktext": ticktext,
        "showgrid": True,
        "gridcolor": "rgba(0,0,0,0.08)",
        "zeroline": False,
    }
    if axis == "x":
        fig.update_xaxes(**kwargs)
    else:
        fig.update_yaxes(**kwargs)


def apply_linear_axis(
    fig: go.Figure,
    axis: str,
    *,
    showgrid: bool = True,
    gridcolor: str = "rgba(0,0,0,0.08)",
    zeroline: bool = False,
) -> None:
    kwargs = {"showgrid": showgrid, "gridcolor": gridcolor, "zeroline": zeroline}
    if axis == "x":
        fig.update_xaxes(**kwargs)
    else:
        fig.update_yaxes(**kwargs)


def discrete_colorscale(colors: list[str]) -> list[list[float | str]]:
    if len(colors) == 1:
        return [[0.0, colors[0]], [1.0, colors[0]]]
    scale = []
    steps = len(colors)
    for index, color in enumerate(colors):
        start = index / steps
        end = (index + 1) / steps
        scale.append([start, color])
        scale.append([end, color])
    return scale


def build_completion_breakdown(rows: list[dict]) -> go.Figure:
    concurrency = [row_load_label(row) for row in rows]
    hover_label = load_hover_label(rows)
    success_pct = [row["success_rate"] * 100.0 for row in rows]
    incomplete_pct = [row["incomplete_rate"] * 100.0 for row in rows]
    errored_pct = [row["errored_rate"] * 100.0 for row in rows]

    fig = go.Figure()
    fig.add_bar(
        x=concurrency,
        y=success_pct,
        name="Successful",
        marker_color=COLORS["blue"],
        customdata=[[row["successful"], row["total"]] for row in rows],
        hovertemplate=f"{hover_label}=%{{x}}<br>Successful=%{{y:.2f}}%<br>Count=%{{customdata[0]}}/%{{customdata[1]}}<extra></extra>",
    )
    fig.add_bar(
        x=concurrency,
        y=incomplete_pct,
        name="Incomplete",
        marker_color=COLORS["orange"],
        customdata=[[row["incomplete"], row["total"]] for row in rows],
        hovertemplate=f"{hover_label}=%{{x}}<br>Incomplete=%{{y:.2f}}%<br>Count=%{{customdata[0]}}/%{{customdata[1]}}<extra></extra>",
    )
    if any(value > 0 for value in errored_pct):
        fig.add_bar(
            x=concurrency,
            y=errored_pct,
            name="Errored",
            marker_color=COLORS["red"],
            customdata=[[row["errored"], row["total"]] for row in rows],
            hovertemplate=f"{hover_label}=%{{x}}<br>Errored=%{{y:.2f}}%<br>Count=%{{customdata[0]}}/%{{customdata[1]}}<extra></extra>",
        )

    fig.update_layout(barmode="stack")
    fig.update_yaxes(
        title_text="Request share (%)",
        range=[0, 105],
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
    )
    fig.update_xaxes(title_text=target_load_axis_title(rows))
    return apply_layout(
        fig,
        title="(a) Completion Breakdown",
        subtitle="Higher successful share is better; rising incomplete share indicates the system is accepting work it cannot retire in time.",
        legend_x=0.01,
        legend_y=0.02,
    )


def build_tail_amplification(rows: list[dict]) -> go.Figure:
    concurrency = np.asarray([row["concurrency"] for row in rows], dtype=float)
    hover_label = load_hover_label(rows)
    fig = go.Figure()
    series = [
        ("TTFT p95/p50", "ttft_tail_p95_p50", COLORS["blue"], "solid", "circle"),
        ("TTFT p99/p50", "ttft_tail_p99_p50", COLORS["orange"], "dash", "circle"),
        ("ITL p95/p50", "itl_tail_p95_p50", COLORS["green"], "solid", "square"),
        ("ITL p99/p50", "itl_tail_p99_p50", COLORS["red"], "dash", "square"),
    ]
    for name, key, color, dash, symbol in series:
        fig.add_scatter(
            x=concurrency,
            y=[row[key] for row in rows],
            mode="lines+markers",
            name=name,
            line={"color": color, "width": 2, "dash": dash},
            marker={"symbol": symbol, "size": 8},
            hovertemplate=f"{hover_label}=%{{x}}<br>{name}=%{{y:.2f}}<extra></extra>",
        )
    apply_log_concurrency_axis(fig, "x", concurrency)
    fig.update_xaxes(title_text=target_load_axis_title(rows))
    fig.update_yaxes(
        title_text="Tail amplification ratio (pXX/p50)",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
    )
    return apply_layout(
        fig,
        title="(b) Tail Amplification",
        subtitle="Computed as p95/p50 and p99/p50 for TTFT and ITL. Ratios closer to 1 are better; rising TTFT ratios indicate startup or prefill instability under load.",
    )


def build_ttft_tpot_coupling(rows: list[dict]) -> go.Figure:
    concurrency = np.asarray([row["concurrency"] for row in rows], dtype=float)
    hover_label = load_hover_label(rows)
    fig = go.Figure()
    fig.add_scatter(
        x=concurrency,
        y=[row["ttft_tpot_corr"] for row in rows],
        mode="lines+markers",
        line={"color": COLORS["purple"], "width": 2},
        marker={"size": 8},
        hovertemplate=f"{hover_label}=%{{x}}<br>Correlation=%{{y:.3f}}<extra></extra>",
        name="Correlation",
        showlegend=False,
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color=COLORS["gray"], line_width=1)
    apply_log_concurrency_axis(fig, "x", concurrency)
    fig.update_xaxes(title_text=target_load_axis_title(rows))
    fig.update_yaxes(
        title_text="Correlation: TTFT vs TPOT",
        range=[-0.1, 1.0],
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
    )
    return apply_layout(
        fig,
        title="(c) TTFT/TPOT Coupling",
        subtitle="Lower is better; higher coupling means requests that start badly also stream badly, a stronger sign of full-system saturation.",
    )


def build_ttft_ccdf(rows: list[dict]) -> go.Figure:
    levels = select_ccdf_levels([row["concurrency"] for row in rows])
    row_by_concurrency = {row["concurrency"]: row for row in rows}
    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["red"]]
    symbols = ["circle", "square", "triangle-up", "triangle-down"]
    level_prefix = "RPS" if load_axis_label(rows) == "RPS" else "C"

    fig = go.Figure()
    for level, color, symbol in zip(levels, colors, symbols):
        ttft_values = [
            request["time_to_first_token_ms"]
            for request in row_by_concurrency[level]["requests"]
        ]
        x_values, y_values = ccdf(ttft_values)
        fig.add_scatter(
            x=x_values,
            y=y_values,
            mode="lines+markers",
            line={"color": color, "width": 2, "shape": "hv"},
            marker={"symbol": symbol, "size": 6},
            name=f"{level_prefix}={format_load_value(level)}",
            hovertemplate="TTFT=%{x:.1f} ms<br>P(TTFT > x)=%{y:.4f}<extra></extra>",
        )

    fig.update_xaxes(
        type="log", title_text="TTFT (ms)", showgrid=True, gridcolor="rgba(0,0,0,0.08)"
    )
    fig.update_yaxes(
        type="log",
        title_text="P(TTFT > x)",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
    )
    return apply_layout(
        fig,
        title="(d) TTFT Tail Distribution",
        subtitle="Curves farther left and dropping faster are better; this exposes heavy TTFT tails that boxplots tend to hide.",
    )


def build_ttft_box_distribution(rows: list[dict]) -> go.Figure:
    categories = []
    ttft_values = []
    for row in rows:
        label = row_load_label(row)
        values = [request["time_to_first_token_ms"] for request in row["requests"]]
        categories.extend([label] * len(values))
        ttft_values.extend(values)
    category_order = [row_load_label(row) for row in rows]
    hover_label = load_hover_label(rows)

    fig = go.Figure()
    fig.add_box(
        x=categories,
        y=ttft_values,
        name="Distribution",
        boxpoints=False,
        line={"color": COLORS["blue"]},
        fillcolor="rgba(50,116,161,0.45)",
        hovertemplate=f"{hover_label}=%{{x}}<br>TTFT=%{{y:.1f}} ms<extra></extra>",
    )
    fig.add_scatter(
        x=category_order,
        y=[row["ttft_p50_ms"] for row in rows],
        mode="lines+markers",
        line={"color": COLORS["green"], "width": 2},
        marker={"size": 8},
        name="p50",
        hovertemplate=f"{hover_label}=%{{x}}<br>TTFT p50=%{{y:.1f}} ms<extra></extra>",
    )
    fig.add_scatter(
        x=category_order,
        y=[row["ttft_p95_ms"] for row in rows],
        mode="lines+markers",
        line={"color": COLORS["orange"], "width": 2, "dash": "dash"},
        marker={"size": 8},
        name="p95",
        hovertemplate=f"{hover_label}=%{{x}}<br>TTFT p95=%{{y:.1f}} ms<extra></extra>",
    )
    fig.update_xaxes(
        title_text=load_axis_label(rows),
        categoryorder="array",
        categoryarray=category_order,
    )
    fig.update_yaxes(
        type="log", title_text="TTFT (ms)", showgrid=True, gridcolor="rgba(0,0,0,0.08)"
    )
    return apply_layout(
        fig,
        title="(e) TTFT Box Distribution",
        subtitle="Lower and tighter is better; the boxes summarize spread while the p50 and p95 overlays show center and tail movement.",
    )


def build_throughput_latency_frontier(rows: list[dict]) -> go.Figure:
    frontier = pareto_frontier(rows)
    hover_label = load_hover_label(rows)
    fig = go.Figure()
    fig.add_scatter(
        x=[row["ttft_p95_ms"] for row in rows],
        y=[row["raw_success_rps"] for row in rows],
        mode="markers+text",
        text=[row_load_label(row) for row in rows],
        textposition="top center",
        marker={
            "size": 10,
            "color": COLORS["blue"],
            "line": {"color": "black", "width": 0.5},
            "opacity": 0.8,
        },
        name="Tested points",
        hovertemplate=f"{hover_label}=%{{text}}<br>TTFT p95=%{{x:.1f}} ms<br>Successful req/s=%{{y:.3f}}<extra></extra>",
    )
    fig.add_scatter(
        x=[row["ttft_p95_ms"] for row in frontier],
        y=[row["raw_success_rps"] for row in frontier],
        mode="lines",
        line={"color": COLORS["orange"], "width": 2, "dash": "dash"},
        name="Pareto frontier",
        hoverinfo="skip",
    )
    fig.update_xaxes(
        type="log",
        title_text="TTFT p95 (ms)",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
    )
    fig.update_yaxes(
        title_text="Successful request throughput (req/s)",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
    )
    return apply_layout(
        fig,
        title="(f) Throughput/Latency Frontier",
        subtitle="Up-left is better; frontier points are not dominated by another tested load point on both throughput and tail latency.",
    )


def build_temporal_stability(rows: list[dict], bin_count: int) -> go.Figure:
    highest_row = max(rows, key=lambda row: row["concurrency"])
    temporal = temporal_bins(highest_row["benchmark"], bin_count)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_scatter(
        x=temporal["progress_percent"],
        y=temporal["ttft_p95_ms"],
        mode="lines+markers",
        line={"color": COLORS["blue"], "width": 2},
        marker={"size": 8},
        name="TTFT p95",
        hovertemplate="Progress=%{x:.1f}%<br>TTFT p95=%{y:.1f} ms<extra></extra>",
        secondary_y=False,
    )
    fig.add_scatter(
        x=temporal["progress_percent"],
        y=temporal["tpot_p95_ms"],
        mode="lines+markers",
        line={"color": COLORS["orange"], "width": 2},
        marker={"size": 8, "symbol": "square"},
        name="TPOT p95",
        hovertemplate="Progress=%{x:.1f}%<br>TPOT p95=%{y:.1f} ms<extra></extra>",
        secondary_y=True,
    )
    fig.update_xaxes(
        title_text="Run progress (%)", showgrid=True, gridcolor="rgba(0,0,0,0.08)"
    )
    fig.update_yaxes(
        title_text="TTFT p95 (ms)",
        type="log",
        secondary_y=False,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        color=COLORS["blue"],
    )
    fig.update_yaxes(
        title_text="TPOT p95 (ms)",
        secondary_y=True,
        showgrid=False,
        color=COLORS["orange"],
    )
    return apply_layout(
        fig,
        title="(g) Temporal Stability",
        subtitle="Faster settling and lower tails are better; early spikes imply startup transients, persistent elevation implies steady-state stress.",
    )


def build_request_throughput(rows: list[dict]) -> go.Figure:
    concurrency = np.asarray([row["concurrency"] for row in rows], dtype=float)
    hover_label = load_hover_label(rows)
    fig = go.Figure()
    fig.add_scatter(
        x=concurrency,
        y=[row["raw_success_rps"] for row in rows],
        mode="lines+markers",
        line={"color": COLORS["blue"], "width": 2},
        marker={"size": 8},
        name="Successful req/s",
        hovertemplate=f"{hover_label}=%{{x}}<br>Successful req/s=%{{y:.3f}}<extra></extra>",
        showlegend=False,
    )
    apply_log_concurrency_axis(fig, "x", concurrency)
    fig.update_xaxes(title_text=target_load_axis_title(rows))
    fig.update_yaxes(
        title_text="Successful request throughput (req/s)",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
    )
    return apply_layout(
        fig,
        title="(h) Request Throughput",
        subtitle="Higher is better for service capacity, but it must be read together with latency and completion integrity.",
    )


def build_delivered_token_throughput(rows: list[dict]) -> go.Figure:
    concurrency = np.asarray([row["concurrency"] for row in rows], dtype=float)
    hover_label = load_hover_label(rows)
    fig = go.Figure()
    series = [
        (
            "Prompt",
            [row["successful_prompt_toksps"] / 1000.0 for row in rows],
            COLORS["green"],
            "circle",
        ),
        (
            "Output",
            [row["successful_output_toksps"] / 1000.0 for row in rows],
            COLORS["orange"],
            "square",
        ),
        (
            "Total",
            [row["successful_total_toksps"] / 1000.0 for row in rows],
            COLORS["blue"],
            "triangle-up",
        ),
    ]
    for name, values, color, symbol in series:
        fig.add_scatter(
            x=concurrency,
            y=values,
            mode="lines+markers",
            line={"color": color, "width": 2},
            marker={"size": 8, "symbol": symbol},
            name=name,
            hovertemplate=f"{hover_label}=%{{x}}<br>{name}=%{{y:.3f}} k tok/s<extra></extra>",
        )
    apply_log_concurrency_axis(fig, "x", concurrency)
    fig.update_xaxes(title_text=target_load_axis_title(rows))
    fig.update_yaxes(
        title_text="Delivered token throughput (k tok/s)",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
    )
    return apply_layout(
        fig,
        title="(i) Delivered Token Throughput",
        subtitle="Higher is better; comparing prompt and output tok/s helps separate prefill-side and decode-side capacity.",
    )


def build_scaling_efficiency(rows: list[dict]) -> go.Figure:
    concurrency = np.asarray([row["concurrency"] for row in rows], dtype=float)
    baseline = min(rows, key=lambda row: row["concurrency"])
    hover_label = load_hover_label(rows)
    baseline_load = baseline["concurrency"] or 1.0
    req_eff = [
        row["raw_success_rps"]
        / (baseline["raw_success_rps"] * (row["concurrency"] / baseline_load))
        for row in rows
    ]
    tok_eff = [
        row["successful_output_toksps"]
        / (baseline["successful_output_toksps"] * (row["concurrency"] / baseline_load))
        for row in rows
    ]

    fig = go.Figure()
    fig.add_scatter(
        x=concurrency,
        y=req_eff,
        mode="lines+markers",
        line={"color": COLORS["gray"], "width": 2},
        marker={"size": 8},
        name="Req/s",
        hovertemplate=f"{hover_label}=%{{x}}<br>Req/s efficiency=%{{y:.3f}}<extra></extra>",
    )
    fig.add_scatter(
        x=concurrency,
        y=tok_eff,
        mode="lines+markers",
        line={"color": COLORS["orange"], "width": 2},
        marker={"size": 8, "symbol": "square"},
        name="Output tok/s",
        hovertemplate=f"{hover_label}=%{{x}}<br>Output tok/s efficiency=%{{y:.3f}}<extra></extra>",
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color=COLORS["gray"], line_width=1)
    apply_log_concurrency_axis(fig, "x", concurrency)
    fig.update_xaxes(title_text=target_load_axis_title(rows))
    fig.update_yaxes(
        title_text="Scaling efficiency = observed / linear baseline",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
    )
    return apply_layout(
        fig,
        title="(j) Scaling Efficiency",
        subtitle=f"Computed as throughput / (baseline throughput x target {load_axis_label(rows).lower()} ratio), shown for req/s and output tok/s. Values closer to 1 are better.",
    )


def build_per_request_decode_rate(rows: list[dict]) -> go.Figure:
    concurrency = np.asarray([row["concurrency"] for row in rows], dtype=float)
    hover_label = load_hover_label(rows)
    fig = go.Figure()
    fig.add_scatter(
        x=concurrency,
        y=[row["output_toks_per_second_p50"] for row in rows],
        mode="lines+markers",
        line={"color": COLORS["blue"], "width": 2},
        marker={"size": 8},
        name="p50",
        hovertemplate=f"{hover_label}=%{{x}}<br>p50=%{{y:.2f}} output tok/s<extra></extra>",
    )
    fig.add_scatter(
        x=concurrency,
        y=[row["output_toks_per_second_p05"] for row in rows],
        mode="lines+markers",
        line={"color": COLORS["red"], "width": 2},
        marker={"size": 8, "symbol": "square"},
        name="p05",
        hovertemplate=f"{hover_label}=%{{x}}<br>p05=%{{y:.2f}} output tok/s<extra></extra>",
    )
    apply_log_concurrency_axis(fig, "x", concurrency)
    fig.update_xaxes(title_text=target_load_axis_title(rows))
    fig.update_yaxes(
        title_text="Per-request decode rate (output tok/s)",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
    )
    return apply_layout(
        fig,
        title="(k) Per-Request Decode Rate",
        subtitle="Higher is better; the lower tail highlights user-visible slow streaming even when aggregate throughput still looks healthy.",
    )


def build_max_goodput_sweep(
    rows: list[dict], ttft_thresholds: list[float], itl_thresholds: list[float]
) -> go.Figure:
    max_goodput, _ = compute_slo_sweep(rows, ttft_thresholds, itl_thresholds)
    fig = go.Figure(
        data=go.Heatmap(
            z=max_goodput,
            x=itl_thresholds,
            y=ttft_thresholds,
            colorscale="Blues",
            colorbar={"title": "Max goodput (req/s)"},
            hovertemplate="TTFT threshold=%{y:.0f} ms<br>ITL threshold=%{x:.0f} ms<br>Max goodput=%{z:.3f} req/s<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="ITL threshold (ms)", tickangle=45, showgrid=False)
    fig.update_yaxes(title_text="TTFT threshold (ms)", showgrid=False)
    return apply_layout(
        fig,
        title="(l) Max Goodput by Candidate SLO",
        subtitle="Each cell is max SLO-goodput in req/s over all tested loads for that TTFT and ITL threshold pair. Brighter is better.",
    )


def build_best_concurrency_sweep(
    rows: list[dict], ttft_thresholds: list[float], itl_thresholds: list[float]
) -> go.Figure:
    _, best_concurrency = compute_slo_sweep(rows, ttft_thresholds, itl_thresholds)
    unique_concurrency = sorted({float(row["concurrency"]) for row in rows})
    axis_label = load_axis_label(rows)
    best_axis_title = "Best RPS" if axis_label == "RPS" else "Best concurrency"
    palette = [
        COLORS["blue"],
        COLORS["orange"],
        COLORS["green"],
        COLORS["red"],
        COLORS["purple"],
        COLORS["brown"],
        COLORS["pink"],
    ][: len(unique_concurrency)]
    colorscale = discrete_colorscale(palette)

    fig = go.Figure(
        data=go.Heatmap(
            z=best_concurrency,
            x=itl_thresholds,
            y=ttft_thresholds,
            zmin=min(unique_concurrency) - 0.5,
            zmax=max(unique_concurrency) + 0.5,
            colorscale=colorscale,
            colorbar={
                "title": axis_label,
                "tickmode": "array",
                "tickvals": unique_concurrency,
                "ticktext": [format_load_value(value) for value in unique_concurrency],
            },
            hovertemplate=f"TTFT threshold=%{{y:.0f}} ms<br>ITL threshold=%{{x:.0f}} ms<br>{best_axis_title}=%{{z:g}}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="ITL threshold (ms)", tickangle=45, showgrid=False)
    fig.update_yaxes(title_text="TTFT threshold (ms)", showgrid=False)
    return apply_layout(
        fig,
        title=f"(m) {best_axis_title} by SLO",
        subtitle=f"Each cell shows which tested {axis_label.lower()} maximizes goodput for that candidate SLO; it turns a latency target into an operating point.",
    )


def build_scheduler_queue_wait(rows: list[dict]) -> go.Figure:
    concurrency = np.asarray([row["concurrency"] for row in rows], dtype=float)
    hover_label = load_hover_label(rows)
    fig = go.Figure()
    fig.add_scatter(
        x=concurrency,
        y=[row["queue_wait_p50_s"] for row in rows],
        mode="lines+markers",
        line={"color": COLORS["blue"], "width": 2},
        marker={"size": 8},
        name="p50",
        hovertemplate=f"{hover_label}=%{{x}}<br>Queue wait p50=%{{y:.4f}} s<extra></extra>",
    )
    fig.add_scatter(
        x=concurrency,
        y=[row["queue_wait_p95_s"] for row in rows],
        mode="lines+markers",
        line={"color": COLORS["red"], "width": 2},
        marker={"size": 8, "symbol": "square"},
        name="p95",
        hovertemplate=f"{hover_label}=%{{x}}<br>Queue wait p95=%{{y:.4f}} s<extra></extra>",
    )
    apply_log_concurrency_axis(fig, "x", concurrency)
    fig.update_xaxes(title_text=target_load_axis_title(rows))
    fig.update_yaxes(
        type="log",
        title_text="Scheduler queue wait (s)",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
    )
    return apply_layout(
        fig,
        title="(n) Scheduler Queue Wait",
        subtitle="Lower is better; this isolates waiting before request execution and can reveal admission or scheduler pressure before model execution degrades.",
    )


def build_effective_prefill_decode_rate(rows: list[dict]) -> go.Figure:
    concurrency = np.asarray([row["concurrency"] for row in rows], dtype=float)
    hover_label = load_hover_label(rows)
    fig = go.Figure()
    fig.add_scatter(
        x=concurrency,
        y=[row["effective_prefill_toksps_p50"] / 1000.0 for row in rows],
        mode="lines+markers",
        line={"color": COLORS["green"], "width": 2},
        marker={"size": 8},
        name="Prefill tok/s p50",
        hovertemplate=f"{hover_label}=%{{x}}<br>Prefill p50=%{{y:.3f}} k tok/s<extra></extra>",
    )
    fig.add_scatter(
        x=concurrency,
        y=[row["effective_decode_toksps_p50"] / 1000.0 for row in rows],
        mode="lines+markers",
        line={"color": COLORS["orange"], "width": 2},
        marker={"size": 8, "symbol": "square"},
        name="Decode tok/s p50",
        hovertemplate=f"{hover_label}=%{{x}}<br>Decode p50=%{{y:.3f}} k tok/s<extra></extra>",
    )
    apply_log_concurrency_axis(fig, "x", concurrency)
    fig.update_xaxes(title_text=target_load_axis_title(rows))
    fig.update_yaxes(
        title_text="Effective stage throughput (k tok/s)",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
    )
    return apply_layout(
        fig,
        title="(o) Effective Prefill/Decode Rate",
        subtitle="Computed as prompt_tokens / TTFT for prefill and output_tokens / decode_duration for decode. Higher is better; divergence shows which stage loses efficiency first.",
    )


def build_useful_vs_wasted_output(rows: list[dict]) -> go.Figure:
    concurrency = [row_load_label(row) for row in rows]
    hover_label = load_hover_label(rows)
    useful = [row["successful_output_toksps"] / 1000.0 for row in rows]
    wasted = [row["incomplete_output_toksps"] / 1000.0 for row in rows]
    fig = go.Figure()
    fig.add_bar(
        x=concurrency,
        y=useful,
        name="Useful output tok/s",
        marker_color=COLORS["blue"],
        hovertemplate=f"{hover_label}=%{{x}}<br>Useful output=%{{y:.3f}} k tok/s<extra></extra>",
    )
    fig.add_bar(
        x=concurrency,
        y=wasted,
        name="Wasted partial output tok/s",
        marker_color=COLORS["orange"],
        hovertemplate=f"{hover_label}=%{{x}}<br>Wasted output=%{{y:.3f}} k tok/s<extra></extra>",
    )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title_text=target_load_axis_title(rows))
    fig.update_yaxes(
        title_text="Output token throughput (k tok/s)",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
    )
    return apply_layout(
        fig,
        title="(p) Useful vs Wasted Output Work",
        subtitle="Blue is useful completed generation and orange is work spent on cancelled requests; higher blue share and lower orange waste are better.",
    )


def build_cancelled_request_progress(rows: list[dict]) -> go.Figure:
    concurrency = np.asarray([row["concurrency"] for row in rows], dtype=float)
    hover_label = load_hover_label(rows)
    fig = go.Figure()
    fig.add_scatter(
        x=concurrency,
        y=[row["incomplete_progress_mean"] * 100.0 for row in rows],
        mode="lines+markers",
        line={"color": COLORS["blue"], "width": 2},
        marker={"size": 8},
        name="Mean",
        hovertemplate=f"{hover_label}=%{{x}}<br>Mean progress=%{{y:.1f}}%<extra></extra>",
    )
    fig.add_scatter(
        x=concurrency,
        y=[row["incomplete_progress_p50"] * 100.0 for row in rows],
        mode="lines+markers",
        line={"color": COLORS["orange"], "width": 2},
        marker={"size": 8, "symbol": "square"},
        name="p50",
        hovertemplate=f"{hover_label}=%{{x}}<br>p50 progress=%{{y:.1f}}%<extra></extra>",
    )
    apply_log_concurrency_axis(fig, "x", concurrency)
    fig.update_xaxes(title_text=target_load_axis_title(rows))
    fig.update_yaxes(
        title_text="Cancelled progress (% of target output)",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
    )
    return apply_layout(
        fig,
        title="(q) Cancelled Request Progress",
        subtitle="Lower is better from a waste perspective; high values mean cancellations happen late after substantial decode work has already been spent.",
    )


def build_token_throughput_per_gpu(rows: list[dict]) -> go.Figure:
    concurrency_per_gpu = np.asarray(
        [row["concurrency_per_gpu"] for row in rows], dtype=float
    )
    hover_label = load_hover_label(rows)
    fig = go.Figure()
    series = [
        (
            "Prompt",
            [row["successful_prompt_toksps_per_gpu"] / 1000.0 for row in rows],
            COLORS["green"],
            "circle",
        ),
        (
            "Output",
            [row["successful_output_toksps_per_gpu"] / 1000.0 for row in rows],
            COLORS["orange"],
            "square",
        ),
        (
            "Total",
            [row["successful_total_toksps_per_gpu"] / 1000.0 for row in rows],
            COLORS["blue"],
            "triangle-up",
        ),
    ]
    for name, values, color, symbol in series:
        fig.add_scatter(
            x=concurrency_per_gpu,
            y=values,
            mode="lines+markers",
            line={"color": color, "width": 2},
            marker={"size": 8, "symbol": symbol},
            name=name,
            hovertemplate=f"{hover_label}/GPU=%{{x:g}}<br>{name}=%{{y:.3f}} k tok/s/GPU<extra></extra>",
        )
    apply_log_concurrency_axis(fig, "x", concurrency_per_gpu)
    fig.update_xaxes(title_text=target_load_axis_title(rows, per_gpu=True))
    fig.update_yaxes(
        title_text="Delivered token throughput (k tok/s/GPU)",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
    )
    return apply_layout(
        fig,
        title="(r) Token Throughput per GPU",
        subtitle=f"Computed as delivered prompt, output, and total tok/s divided by GPU count; x is {target_load_axis_title(rows, per_gpu=True).lower()}. Higher is better.",
    )


def build_throughput_efficiency_per_gpu(rows: list[dict]) -> go.Figure:
    axis_label = load_axis_label(rows)
    axis_label_lower = axis_label.lower()
    hover_label = load_hover_label(rows)
    interactivity = [
        row["successful_output_toksps"] / row["concurrency"] for row in rows
    ]
    throughput_per_gpu = [row["successful_output_toksps_per_gpu"] for row in rows]
    fig = go.Figure()
    fig.add_scatter(
        x=interactivity,
        y=throughput_per_gpu,
        mode="lines+markers+text",
        line={"color": COLORS["gray"], "width": 1.5, "dash": "dash"},
        marker={
            "size": 9,
            "color": COLORS["orange"],
            "line": {"color": "black", "width": 0.6},
        },
        text=[row_load_label(row) for row in rows],
        textposition="top center",
        name="Load point",
        hovertemplate=(
            f"{hover_label}=%{{text}}<br>"
            f"Interactivity=%{{x:.2f}} output tok/s/{axis_label_lower}<br>"
            "Output throughput=%{y:.2f} tok/s/GPU<extra></extra>"
        ),
    )
    fig.update_xaxes(
        title_text=f"Interactivity (output tok/s per target {axis_label_lower})",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
    )
    fig.update_yaxes(
        title_text="Output token throughput (tok/s/GPU)",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
    )
    return apply_layout(
        fig,
        title="(s) Throughput Efficiency per GPU",
        subtitle=f"Computed as x = successful output tok/s / target {axis_label_lower} and y = successful output tok/s / GPU count. Up-right is better.",
    )


def build_ttft_vs_actual_concurrency(rows: list[dict]) -> go.Figure:
    summary = actual_concurrency_percentiles(rows)
    x_values = [entry["x"] for entry in summary]
    tick_text = [entry["label"] for entry in summary]
    fig = go.Figure()
    fig.add_scatter(
        x=x_values,
        y=[entry["ttft_p50_ms"] for entry in summary],
        mode="lines+markers",
        line={"color": COLORS["green"], "width": 2},
        marker={"size": 8},
        name="p50",
        hovertemplate="Observed concurrency=%{x:g}<br>TTFT p50=%{y:.1f} ms<extra></extra>",
    )
    fig.add_scatter(
        x=x_values,
        y=[entry["ttft_p95_ms"] for entry in summary],
        mode="lines+markers",
        line={"color": COLORS["orange"], "width": 2},
        marker={"size": 8, "symbol": "square"},
        name="p95",
        hovertemplate="Observed concurrency=%{x:g}<br>TTFT p95=%{y:.1f} ms<extra></extra>",
    )
    fig.add_scatter(
        x=x_values,
        y=[entry["ttft_p99_ms"] for entry in summary],
        mode="lines+markers",
        line={"color": COLORS["red"], "width": 2},
        marker={"size": 8, "symbol": "triangle-up"},
        name="p99",
        hovertemplate="Observed concurrency=%{x:g}<br>TTFT p99=%{y:.1f} ms<extra></extra>",
    )
    fig.update_xaxes(
        type="log",
        title_text="Observed in-flight requests",
        tickmode="array",
        tickvals=x_values,
        ticktext=tick_text,
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=False,
    )
    fig.update_yaxes(
        type="log", title_text="TTFT (ms)", showgrid=True, gridcolor="rgba(0,0,0,0.08)"
    )
    return apply_layout(
        fig,
        title="(t) TTFT vs Actual Concurrency",
        subtitle="Lower and flatter are better; actual concurrency is derived from overlapping request lifetimes, which exposes saturation more directly than target load.",
    )


def build_delay_decomposition(rows: list[dict]) -> go.Figure:
    concurrency = np.asarray([row["concurrency"] for row in rows], dtype=float)
    hover_label = load_hover_label(rows)
    fig = go.Figure()
    series = [
        (
            "Queue wait p50",
            [row["queue_wait_p50_s"] for row in rows],
            COLORS["blue"],
            "solid",
            "circle",
        ),
        (
            "TTFT p50",
            [row["ttft_p50_ms"] / 1000.0 for row in rows],
            COLORS["green"],
            "solid",
            "circle",
        ),
        (
            "Decode p50",
            [row["decode_duration_p50_s"] for row in rows],
            COLORS["orange"],
            "solid",
            "circle",
        ),
        (
            "Queue wait p95",
            [row["queue_wait_p95_s"] for row in rows],
            COLORS["blue"],
            "dash",
            "square",
        ),
        (
            "TTFT p95",
            [row["ttft_p95_ms"] / 1000.0 for row in rows],
            COLORS["green"],
            "dash",
            "square",
        ),
        (
            "Decode p95",
            [row["decode_duration_p95_s"] for row in rows],
            COLORS["orange"],
            "dash",
            "square",
        ),
    ]
    for name, values, color, dash, symbol in series:
        fig.add_scatter(
            x=concurrency,
            y=values,
            mode="lines+markers",
            line={"color": color, "width": 2, "dash": dash},
            marker={"symbol": symbol, "size": 8},
            name=name,
            hovertemplate=f"{hover_label}=%{{x}}<br>{name}=%{{y:.4f}} s<extra></extra>",
        )
    apply_log_concurrency_axis(fig, "x", concurrency)
    fig.update_xaxes(title_text=target_load_axis_title(rows))
    fig.update_yaxes(
        type="log",
        title_text="Latency component (s)",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
    )
    return apply_layout(
        fig,
        title="(u) Delay Decomposition",
        subtitle="Lower is better; the first component to rise identifies whether scheduler wait, first-token latency, or decode duration is the limiting stage.",
    )


def build_figures(
    rows: list[dict],
    ttft_thresholds: list[float],
    itl_thresholds: list[float],
    bin_count: int,
) -> list[tuple[str, go.Figure]]:
    return [
        ("completion_breakdown", build_completion_breakdown(rows)),
        ("tail_amplification", build_tail_amplification(rows)),
        ("ttft_tpot_coupling", build_ttft_tpot_coupling(rows)),
        ("ttft_ccdf", build_ttft_ccdf(rows)),
        ("ttft_boxplot", build_ttft_box_distribution(rows)),
        ("throughput_latency_frontier", build_throughput_latency_frontier(rows)),
        ("temporal_stability", build_temporal_stability(rows, bin_count)),
        ("request_throughput", build_request_throughput(rows)),
        ("delivered_token_throughput", build_delivered_token_throughput(rows)),
        ("scaling_efficiency", build_scaling_efficiency(rows)),
        ("per_request_decode_rate", build_per_request_decode_rate(rows)),
        (
            "max_goodput_sweep",
            build_max_goodput_sweep(rows, ttft_thresholds, itl_thresholds),
        ),
        (
            "best_concurrency_sweep",
            build_best_concurrency_sweep(rows, ttft_thresholds, itl_thresholds),
        ),
        ("scheduler_queue_wait", build_scheduler_queue_wait(rows)),
        ("effective_prefill_decode_rate", build_effective_prefill_decode_rate(rows)),
        ("useful_vs_wasted_output", build_useful_vs_wasted_output(rows)),
        ("cancelled_request_progress", build_cancelled_request_progress(rows)),
        ("token_throughput_per_gpu", build_token_throughput_per_gpu(rows)),
        ("throughput_efficiency_per_gpu", build_throughput_efficiency_per_gpu(rows)),
        ("ttft_vs_actual_concurrency", build_ttft_vs_actual_concurrency(rows)),
        ("delay_decomposition", build_delay_decomposition(rows)),
    ]


def save_html_report(
    output_prefix: str,
    figures: list[tuple[str, go.Figure]],
    metadata: dict[str, str],
) -> Path:
    html_path = Path(f"{output_prefix}_report.html")

    cells = []
    for _, figure in figures:
        cells.append(
            figure.to_html(
                full_html=False,
                include_plotlyjs=False,
                config=PLOTLY_CONFIG,
                default_width=f"{FIGURE_WIDTH}px",
                default_height=f"{FIGURE_HEIGHT}px",
            )
        )

    rows = []
    for start in range(0, len(cells), 3):
        row_cells = []
        for cell in cells[start : start + 3]:
            row_cells.append(
                f"""
        <td valign="top">
          {cell}
        </td>"""
            )
        while len(row_cells) < 3:
            row_cells.append('<td valign="top"></td>')
        rows.append(f"""
      <tr>
{"".join(row_cells)}
      </tr>""")

    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Post Run Benchmark Report</title>
  <script type="text/javascript">{get_plotlyjs()}</script>
</head>
<body bgcolor="white" style="margin: 16px 20px 24px 20px;">
{render_report_header(metadata)}
  <table border="0" cellspacing="12" cellpadding="0">
{"".join(rows)}
  </table>
</body>
</html>
"""
    html_path.write_text(content)
    print(f"✓ Saved: {html_path.name}")
    return html_path


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    strict_slo = (args.strict_ttft_ms, args.strict_itl_ms)
    relaxed_slo = (args.relaxed_ttft_ms, args.relaxed_itl_ms)
    ttft_thresholds = parse_thresholds(args.ttft_thresholds)
    itl_thresholds = parse_thresholds(args.itl_thresholds)

    benchmarks = load_benchmarks(input_path)
    rows = summarize_benchmarks(
        benchmarks, strict_slo, relaxed_slo, gpu_count=args.gpu_count
    )
    figures = build_figures(rows, ttft_thresholds, itl_thresholds, args.time_bins)
    metadata = load_report_metadata(input_path, rows, args.gpu_count)
    save_html_report(args.output_prefix, figures, metadata)

    print("\n✓ Interactive Plotly report created successfully!")
    print("\nGenerated plots:")
    for index, (slug, _) in enumerate(figures, start=1):
        print(f"  {index}. {slug}")


if __name__ == "__main__":
    main()
