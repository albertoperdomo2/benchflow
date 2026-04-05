"""
Academic-style benchmark insight plots for GuideLLM benchmark_output.json.

This script follows the same publication-oriented visual style as
academic_plots.py while focusing on LLM-serving plots that are useful for
performance engineering:
    - Completion breakdown vs concurrency
    - Tail amplification vs concurrency
    - TTFT/TPOT coupling vs concurrency
    - TTFT tail distributions (CCDF)
    - Throughput/latency Pareto frontier
    - Temporal stability within the highest-concurrency run
    - Candidate-SLO threshold sweep heatmaps

Usage:
    python benchmark_insight_plots.py
    python benchmark_insight_plots.py --input benchmark_output.json
    python benchmark_insight_plots.py --output-prefix benchmark_insights

Outputs:
    - <prefix>_overview.png
    - <prefix>_overview.pdf
    - <prefix>_overview.svg
    - <prefix>_throughput.png
    - <prefix>_throughput.pdf
    - <prefix>_throughput.svg
    - <prefix>_slo_sweep.png
    - <prefix>_slo_sweep.pdf
    - <prefix>_slo_sweep.svg
    - <prefix>_report.pdf
    - <prefix>_all_plots.pdf
    - <prefix>_report.html
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import tempfile
from pathlib import Path
from textwrap import fill

_CACHE_ROOT = Path(tempfile.gettempdir()) / "benchflow-plot-cache"
_MPL_CACHE = _CACHE_ROOT / "matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT.resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE.resolve()))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import rcParams  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

# ============================================================================
# CONFIGURATION: Match academic_plots.py exactly
# ============================================================================
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
rcParams["font.size"] = 10
rcParams["axes.labelsize"] = 11
rcParams["axes.titlesize"] = 12
rcParams["xtick.labelsize"] = 10
rcParams["ytick.labelsize"] = 10
rcParams["legend.fontsize"] = 10
rcParams["figure.titlesize"] = 14
rcParams["figure.dpi"] = 300
rcParams["savefig.dpi"] = 300
rcParams["savefig.bbox"] = "tight"

COLORS = {
    "blue": "#3274A1",
    "orange": "#E1812C",
    "green": "#7CB57C",
    "red": "#D95F5F",
    "purple": "#9E67AB",
    "brown": "#B8704F",
    "pink": "#E89CAE",
    "gray": "#7C7C7C",
}

PANEL_TITLE_PAD = 72
CELL_TITLE_PAD = 28
DEFAULT_GPU_COUNT = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create academic-style SLO and tail-diagnostic plots from benchmark_output.json."
    )
    parser.add_argument(
        "--input",
        default="benchmark_output.json",
        help="Path to benchmark_output.json",
    )
    parser.add_argument(
        "--output-prefix",
        default="benchmark_insights",
        help="Output prefix for generated figures",
    )
    parser.add_argument(
        "--strict-ttft-ms",
        type=float,
        default=200.0,
        help="Example strict TTFT threshold to overlay on the SLO sweep figure",
    )
    parser.add_argument(
        "--strict-itl-ms",
        type=float,
        default=25.0,
        help="Example strict ITL threshold to overlay on the SLO sweep figure",
    )
    parser.add_argument(
        "--relaxed-ttft-ms",
        type=float,
        default=500.0,
        help="Example relaxed TTFT threshold to overlay on the SLO sweep figure",
    )
    parser.add_argument(
        "--relaxed-itl-ms",
        type=float,
        default=40.0,
        help="Example relaxed ITL threshold to overlay on the SLO sweep figure",
    )
    parser.add_argument(
        "--ttft-thresholds",
        default="100,150,200,300,500,750,1000,2000,4000,8000",
        help="Comma-separated TTFT thresholds in milliseconds for the SLO sweep figure",
    )
    parser.add_argument(
        "--itl-thresholds",
        default="15,20,25,30,40,50,60,80,100",
        help="Comma-separated ITL thresholds in milliseconds for the SLO sweep figure",
    )
    parser.add_argument(
        "--time-bins",
        type=int,
        default=10,
        help="Number of equal-duration bins for temporal stability plots",
    )
    parser.add_argument(
        "--gpu-count",
        type=float,
        default=DEFAULT_GPU_COUNT,
        help="Accelerator count used for per-GPU normalization.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving it",
    )
    return parser.parse_args()


def load_benchmarks(path: Path) -> list[dict]:
    with path.open() as handle:
        payload = json.load(handle)

    benchmarks = payload["benchmarks"]
    benchmarks.sort(key=lambda item: item["config"]["strategy"]["max_concurrency"])
    return benchmarks


def get_percentile(metric_blob: dict, percentile_name: str) -> float:
    return float(metric_blob["successful"]["percentiles"][percentile_name])


def get_mean(metric_blob: dict) -> float:
    return float(metric_blob["successful"]["mean"])


def parse_thresholds(raw: str) -> list[float]:
    values = [float(token.strip()) for token in raw.split(",") if token.strip()]
    return sorted(values)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), pct))


def ccdf(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(np.asarray(values, dtype=float))
    exceedance = np.arange(len(ordered), 0, -1) / len(ordered)
    return ordered, exceedance


def style_axes(ax, axis: str = "both") -> None:
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5, axis=axis)


def add_panel_subtitle(ax, text: str, width: int = 72) -> None:
    wrapped = fill(text, width=width)
    ax.annotate(
        wrapped,
        xy=(0.5, 1.0),
        xycoords="axes fraction",
        xytext=(0, 2),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9.2,
        color=COLORS["gray"],
        linespacing=1.15,
        clip_on=False,
    )


def request_timing_arrays(requests: list[dict]) -> dict[str, list[float]]:
    queue_wait_s = []
    ttft_s = []
    decode_s = []
    effective_prefill_toksps = []
    effective_decode_toksps = []

    for request in requests:
        timings = request["info"]["timings"]
        queued = timings.get("queued")
        dequeued = timings.get("dequeued")
        request_start = timings.get("request_start")
        first_token = timings.get("first_token_iteration")
        last_token = timings.get("last_token_iteration")
        prompt_tokens = request.get("prompt_tokens")
        output_tokens = request.get("output_tokens")

        if queued is not None and dequeued is not None and dequeued >= queued:
            queue_wait_s.append(float(dequeued - queued))

        if (
            request_start is not None
            and first_token is not None
            and first_token >= request_start
        ):
            ttft_duration = float(first_token - request_start)
            ttft_s.append(ttft_duration)
            if prompt_tokens is not None and ttft_duration > 0:
                effective_prefill_toksps.append(float(prompt_tokens) / ttft_duration)

        if (
            first_token is not None
            and last_token is not None
            and last_token >= first_token
        ):
            decode_duration = float(last_token - first_token)
            decode_s.append(decode_duration)
            decode_tokens = None
            if output_tokens is not None:
                decode_tokens = max(float(output_tokens) - 1.0, 1.0)
            if decode_tokens is not None and decode_duration > 0:
                effective_decode_toksps.append(decode_tokens / decode_duration)

    return {
        "queue_wait_s": queue_wait_s,
        "ttft_s": ttft_s,
        "decode_s": decode_s,
        "effective_prefill_toksps": effective_prefill_toksps,
        "effective_decode_toksps": effective_decode_toksps,
    }


def pearson_correlation(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) < 2 or len(y_values) < 2:
        return float("nan")

    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    x_std = x.std()
    y_std = y.std()
    if x_std == 0 or y_std == 0:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def goodput_rps(
    requests: list[dict],
    duration_seconds: float,
    ttft_threshold_ms: float,
    itl_threshold_ms: float,
) -> float:
    if duration_seconds <= 0:
        return float("nan")

    good_requests = 0
    for request in requests:
        ttft = request.get("time_to_first_token_ms")
        itl = request.get("inter_token_latency_ms")
        if ttft is None or itl is None:
            continue
        if ttft <= ttft_threshold_ms and itl <= itl_threshold_ms:
            good_requests += 1

    return good_requests / duration_seconds


def goodput_output_toksps(
    requests: list[dict],
    duration_seconds: float,
    ttft_threshold_ms: float,
    itl_threshold_ms: float,
) -> float:
    if duration_seconds <= 0:
        return float("nan")

    good_output_tokens = 0.0
    for request in requests:
        ttft = request.get("time_to_first_token_ms")
        itl = request.get("inter_token_latency_ms")
        output_tokens = request.get("output_tokens")
        if ttft is None or itl is None or output_tokens is None:
            continue
        if ttft <= ttft_threshold_ms and itl <= itl_threshold_ms:
            good_output_tokens += float(output_tokens)

    return good_output_tokens / duration_seconds


def summarize_benchmarks(
    benchmarks: list[dict],
    strict_slo: tuple[float, float],
    relaxed_slo: tuple[float, float],
    gpu_count: float = DEFAULT_GPU_COUNT,
) -> list[dict]:
    rows = []

    for benchmark in benchmarks:
        metrics = benchmark["metrics"]
        totals = metrics["request_totals"]
        requests = benchmark["requests"]["successful"]
        incomplete_requests = benchmark["requests"]["incomplete"]
        concurrency = benchmark["config"]["strategy"]["max_concurrency"]
        duration = float(benchmark["duration"])
        total_requests = totals["total"]
        concurrency_per_gpu = concurrency / gpu_count if gpu_count else np.nan

        ttft = metrics["time_to_first_token_ms"]
        itl = metrics["inter_token_latency_ms"]
        successful_timings = request_timing_arrays(requests)
        strict_goodput_rps_value = goodput_rps(requests, duration, *strict_slo)
        relaxed_goodput_rps_value = goodput_rps(requests, duration, *relaxed_slo)
        strict_goodput_output_toksps_value = goodput_output_toksps(
            requests, duration, *strict_slo
        )
        relaxed_goodput_output_toksps_value = goodput_output_toksps(
            requests, duration, *relaxed_slo
        )

        successful_mean_output_tokens = (
            sum(
                request["output_tokens"]
                for request in requests
                if request["output_tokens"] is not None
            )
            / len(requests)
            if requests
            else 0.0
        )
        incomplete_progress = [
            request["output_tokens"] / successful_mean_output_tokens
            for request in incomplete_requests
            if request["output_tokens"] is not None
            and successful_mean_output_tokens > 0
        ]

        rows.append(
            {
                "benchmark": benchmark,
                "concurrency": concurrency,
                "duration": duration,
                "requests": requests,
                "incomplete_requests": incomplete_requests,
                "successful": totals["successful"],
                "incomplete": totals["incomplete"],
                "errored": totals["errored"],
                "total": total_requests,
                "success_rate": totals["successful"] / total_requests
                if total_requests
                else 0.0,
                "incomplete_rate": totals["incomplete"] / total_requests
                if total_requests
                else 0.0,
                "errored_rate": totals["errored"] / total_requests
                if total_requests
                else 0.0,
                "raw_success_rps": totals["successful"] / duration if duration else 0.0,
                "concurrency_per_gpu": concurrency_per_gpu,
                "raw_success_rps_per_gpu": (
                    totals["successful"] / duration / gpu_count
                    if duration and gpu_count
                    else np.nan
                ),
                "successful_prompt_toksps": (
                    sum(
                        request["prompt_tokens"]
                        for request in requests
                        if request["prompt_tokens"] is not None
                    )
                    / duration
                    if duration
                    else 0.0
                ),
                "successful_prompt_toksps_per_gpu": (
                    (
                        sum(
                            request["prompt_tokens"]
                            for request in requests
                            if request["prompt_tokens"] is not None
                        )
                        / duration
                        / gpu_count
                    )
                    if duration and gpu_count
                    else np.nan
                ),
                "successful_output_toksps": (
                    sum(
                        request["output_tokens"]
                        for request in requests
                        if request["output_tokens"] is not None
                    )
                    / duration
                    if duration
                    else 0.0
                ),
                "successful_output_toksps_per_gpu": (
                    (
                        sum(
                            request["output_tokens"]
                            for request in requests
                            if request["output_tokens"] is not None
                        )
                        / duration
                        / gpu_count
                    )
                    if duration and gpu_count
                    else np.nan
                ),
                "successful_total_toksps": (
                    sum(
                        request["total_tokens"]
                        for request in requests
                        if request["total_tokens"] is not None
                    )
                    / duration
                    if duration
                    else 0.0
                ),
                "successful_total_toksps_per_gpu": (
                    (
                        sum(
                            request["total_tokens"]
                            for request in requests
                            if request["total_tokens"] is not None
                        )
                        / duration
                        / gpu_count
                    )
                    if duration and gpu_count
                    else np.nan
                ),
                "incomplete_output_toksps": (
                    sum(
                        request["output_tokens"]
                        for request in incomplete_requests
                        if request["output_tokens"] is not None
                    )
                    / duration
                    if duration
                    else 0.0
                ),
                "incomplete_total_toksps": (
                    sum(
                        request["total_tokens"]
                        for request in incomplete_requests
                        if request["total_tokens"] is not None
                    )
                    / duration
                    if duration
                    else 0.0
                ),
                "strict_goodput_rps": strict_goodput_rps_value,
                "relaxed_goodput_rps": relaxed_goodput_rps_value,
                "strict_goodput_rps_per_gpu": (
                    strict_goodput_rps_value / gpu_count if gpu_count else np.nan
                ),
                "relaxed_goodput_rps_per_gpu": (
                    relaxed_goodput_rps_value / gpu_count if gpu_count else np.nan
                ),
                "strict_goodput_output_toksps": strict_goodput_output_toksps_value,
                "relaxed_goodput_output_toksps": relaxed_goodput_output_toksps_value,
                "strict_goodput_output_toksps_per_gpu": (
                    strict_goodput_output_toksps_value / gpu_count
                    if gpu_count
                    else np.nan
                ),
                "relaxed_goodput_output_toksps_per_gpu": (
                    relaxed_goodput_output_toksps_value / gpu_count
                    if gpu_count
                    else np.nan
                ),
                "ttft_p50_ms": get_percentile(ttft, "p50"),
                "ttft_p95_ms": get_percentile(ttft, "p95"),
                "ttft_p99_ms": get_percentile(ttft, "p99"),
                "itl_p95_ms": get_percentile(itl, "p95"),
                "itl_p99_ms": get_percentile(itl, "p99"),
                "output_toks_per_second_p50": get_percentile(
                    metrics["output_tokens_per_second"], "p50"
                ),
                "output_toks_per_second_p05": get_percentile(
                    metrics["output_tokens_per_second"], "p05"
                ),
                "ttft_tail_p95_p50": get_percentile(ttft, "p95")
                / get_percentile(ttft, "p50"),
                "ttft_tail_p99_p50": get_percentile(ttft, "p99")
                / get_percentile(ttft, "p50"),
                "itl_tail_p95_p50": get_percentile(itl, "p95")
                / get_percentile(itl, "p50"),
                "itl_tail_p99_p50": get_percentile(itl, "p99")
                / get_percentile(itl, "p50"),
                "ttft_tpot_corr": pearson_correlation(
                    [request["time_to_first_token_ms"] for request in requests],
                    [request["time_per_output_token_ms"] for request in requests],
                ),
                "queue_wait_p50_s": percentile(successful_timings["queue_wait_s"], 50),
                "queue_wait_p95_s": percentile(successful_timings["queue_wait_s"], 95),
                "decode_duration_p50_s": percentile(successful_timings["decode_s"], 50),
                "decode_duration_p95_s": percentile(successful_timings["decode_s"], 95),
                "effective_prefill_toksps_p50": percentile(
                    successful_timings["effective_prefill_toksps"], 50
                ),
                "effective_decode_toksps_p50": percentile(
                    successful_timings["effective_decode_toksps"], 50
                ),
                "useful_output_fraction": (
                    (
                        sum(
                            request["output_tokens"]
                            for request in requests
                            if request["output_tokens"] is not None
                        )
                    )
                    / (
                        sum(
                            request["output_tokens"]
                            for request in requests
                            if request["output_tokens"] is not None
                        )
                        + sum(
                            request["output_tokens"]
                            for request in incomplete_requests
                            if request["output_tokens"] is not None
                        )
                    )
                    if (
                        sum(
                            request["output_tokens"]
                            for request in requests
                            if request["output_tokens"] is not None
                        )
                        + sum(
                            request["output_tokens"]
                            for request in incomplete_requests
                            if request["output_tokens"] is not None
                        )
                    )
                    > 0
                    else np.nan
                ),
                "incomplete_progress_mean": (
                    float(np.mean(incomplete_progress))
                    if incomplete_progress
                    else np.nan
                ),
                "incomplete_progress_p50": percentile(incomplete_progress, 50),
                "queued_time_avg_s": float(
                    benchmark["scheduler_metrics"]["queued_time_avg"]
                ),
                "request_targeted_start_delay_avg_s": float(
                    benchmark["scheduler_metrics"]["request_targeted_start_delay_avg"]
                ),
            }
        )

    return rows


def actual_concurrency_percentiles(
    rows: list[dict],
    min_samples: int = 20,
) -> list[dict]:
    samples: list[tuple[int, float]] = []

    for row in rows:
        timed_requests = []
        for request in row["requests"]:
            start_time = request.get("request_start_time")
            end_time = request.get("request_end_time")
            ttft_ms = request.get("time_to_first_token_ms")
            if start_time is None or end_time is None or ttft_ms is None:
                continue
            if end_time < start_time:
                continue
            timed_requests.append((float(start_time), float(end_time), float(ttft_ms)))

        if not timed_requests:
            continue

        start_times = np.sort(
            np.asarray([start for start, _, _ in timed_requests], dtype=float)
        )
        end_times = np.sort(
            np.asarray([end for _, end, _ in timed_requests], dtype=float)
        )

        for start_time, _, ttft_ms in timed_requests:
            actual_concurrency = int(
                np.searchsorted(start_times, start_time, side="right")
                - np.searchsorted(end_times, start_time, side="right")
            )
            samples.append((actual_concurrency, ttft_ms))

    if not samples:
        return []

    max_actual = max(actual for actual, _ in samples)
    bins: list[tuple[int, int]] = []
    lower = 1
    upper = 1
    while lower <= max_actual:
        bins.append((lower, min(upper, max_actual)))
        if lower == 1:
            lower, upper = 2, 3
        else:
            lower *= 2
            upper = lower * 2 - 1

    summaries = []
    for lower, upper in bins:
        ttft_values = [
            ttft_ms for actual, ttft_ms in samples if lower <= actual <= upper
        ]
        if len(ttft_values) < min_samples:
            continue
        x_value = float(lower) if lower == upper else float(np.sqrt(lower * upper))
        label = str(lower) if lower == upper else f"{lower}-{upper}"
        summaries.append(
            {
                "x": x_value,
                "label": label,
                "count": len(ttft_values),
                "ttft_p50_ms": percentile(ttft_values, 50),
                "ttft_p95_ms": percentile(ttft_values, 95),
                "ttft_p99_ms": percentile(ttft_values, 99),
            }
        )

    return summaries


def select_ccdf_levels(concurrency_levels: list[int]) -> list[int]:
    preferred = [concurrency_levels[0], 100, 300, concurrency_levels[-1]]
    selected = []
    seen = set()

    for target in preferred:
        chosen = min(concurrency_levels, key=lambda value: (abs(value - target), value))
        if chosen not in seen:
            selected.append(chosen)
            seen.add(chosen)

    return selected


def temporal_bins(benchmark: dict, bin_count: int) -> dict[str, np.ndarray]:
    requests = benchmark["requests"]["successful"]
    start_time = float(benchmark["start_time"])
    end_time = float(benchmark["end_time"])

    boundaries = np.linspace(start_time, end_time, bin_count + 1)
    centers = (boundaries[:-1] + boundaries[1:]) / 2.0

    ttft_p95 = np.full(bin_count, np.nan)
    tpot_p95 = np.full(bin_count, np.nan)
    completion_rps = np.zeros(bin_count)

    for index in range(bin_count):
        begin = boundaries[index]
        finish = boundaries[index + 1]
        window = [
            request
            for request in requests
            if begin <= request["request_end_time"] < finish
        ]
        if not window:
            continue

        ttft_values = [request["time_to_first_token_ms"] for request in window]
        tpot_values = [request["time_per_output_token_ms"] for request in window]

        ttft_p95[index] = percentile(ttft_values, 95)
        tpot_p95[index] = percentile(tpot_values, 95)
        completion_rps[index] = len(window) / max(finish - begin, 1e-9)

    progress_percent = (centers - start_time) / max(end_time - start_time, 1e-9) * 100.0
    return {
        "progress_percent": progress_percent,
        "ttft_p95_ms": ttft_p95,
        "tpot_p95_ms": tpot_p95,
        "completion_rps": completion_rps,
    }


def pareto_frontier(rows: list[dict]) -> list[dict]:
    ordered = sorted(
        rows, key=lambda row: (row["ttft_p95_ms"], -row["raw_success_rps"])
    )
    frontier = []
    best_goodput = float("-inf")

    for row in ordered:
        if row["raw_success_rps"] > best_goodput:
            frontier.append(row)
            best_goodput = row["raw_success_rps"]

    return frontier


def compute_slo_sweep(
    rows: list[dict],
    ttft_thresholds: list[float],
    itl_thresholds: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    max_goodput = np.zeros((len(ttft_thresholds), len(itl_thresholds)))
    best_concurrency = np.zeros((len(ttft_thresholds), len(itl_thresholds)))

    for ttft_index, ttft_threshold in enumerate(ttft_thresholds):
        for itl_index, itl_threshold in enumerate(itl_thresholds):
            best_row = None
            best_value = float("-inf")
            for row in rows:
                value = goodput_rps(
                    row["requests"],
                    row["duration"],
                    ttft_threshold,
                    itl_threshold,
                )
                if value > best_value + 1e-12:
                    best_value = value
                    best_row = row
                elif abs(value - best_value) <= 1e-12 and best_row is not None:
                    if row["concurrency"] < best_row["concurrency"]:
                        best_row = row

            max_goodput[ttft_index, itl_index] = best_value
            best_concurrency[ttft_index, itl_index] = (
                best_row["concurrency"] if best_row else np.nan
            )

    return max_goodput, best_concurrency


def save_figure(fig: plt.Figure, stem: str) -> None:
    png_path = f"{stem}.png"
    pdf_path = f"{stem}.pdf"
    svg_path = f"{stem}.svg"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {png_path}")

    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"✓ Saved: {pdf_path}")

    fig.savefig(svg_path, bbox_inches="tight")
    print(f"✓ Saved: {svg_path}")


def save_pdf_report(output_prefix: str, figures: list[tuple[str, plt.Figure]]) -> None:
    report_path = f"{output_prefix}_report.pdf"
    with PdfPages(report_path) as pdf:
        title_fig = plt.figure(figsize=(8.27, 11.69))
        title_ax = title_fig.add_subplot(111)
        title_ax.axis("off")
        title_ax.text(
            0.5,
            0.92,
            "Post Run Benchmark Report",
            ha="center",
            va="top",
            fontsize=18,
            fontweight="bold",
        )
        title_ax.text(
            0.5,
            0.86,
            "Exploratory diagnostics, throughput characterization, and candidate-SLO sweep",
            ha="center",
            va="top",
            fontsize=11,
            color=COLORS["gray"],
        )
        title_ax.text(
            0.12,
            0.72,
            "Included figures:",
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
        )
        for index, (label, _) in enumerate(figures, start=1):
            title_ax.text(
                0.14,
                0.72 - index * 0.055,
                f"{index}. {label}",
                ha="left",
                va="top",
                fontsize=11,
            )
        pdf.savefig(title_fig, bbox_inches="tight")
        plt.close(title_fig)

        for _, figure in figures:
            pdf.savefig(figure, bbox_inches="tight")

    print(f"✓ Saved: {report_path}")


def save_pdf_only(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, bbox_inches="tight")
    print(f"✓ Saved: {path}")


def save_plot_cells(
    rows: list[dict],
    ttft_thresholds: list[float],
    itl_thresholds: list[float],
    strict_slo: tuple[float, float],
    relaxed_slo: tuple[float, float],
    bin_count: int,
    output_prefix: str,
) -> list[dict]:
    concurrency = np.asarray([row["concurrency"] for row in rows], dtype=float)
    x_pos = np.arange(len(rows))
    success_pct = np.asarray([row["success_rate"] * 100.0 for row in rows])
    incomplete_pct = np.asarray([row["incomplete_rate"] * 100.0 for row in rows])
    error_pct = np.asarray([row["errored_rate"] * 100.0 for row in rows])
    frontier = pareto_frontier(rows)
    row_by_concurrency = {row["concurrency"]: row for row in rows}
    ccdf_levels = select_ccdf_levels([row["concurrency"] for row in rows])
    ccdf_colors = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["red"]]
    markers = ["o", "s", "^", "v"]
    ttft_samples = [
        [request["time_to_first_token_ms"] for request in row["requests"]]
        for row in rows
    ]
    ttft_box_positions = np.arange(1, len(rows) + 1)
    highest_row = max(rows, key=lambda row: row["concurrency"])
    temporal = temporal_bins(highest_row["benchmark"], bin_count)
    baseline = min(rows, key=lambda row: row["concurrency"])
    max_goodput, best_concurrency = compute_slo_sweep(
        rows, ttft_thresholds, itl_thresholds
    )
    actual_concurrency_summary = actual_concurrency_percentiles(rows)
    unique_concurrency = sorted({int(row["concurrency"]) for row in rows})
    discrete_colors = [
        COLORS["blue"],
        COLORS["orange"],
        COLORS["green"],
        COLORS["red"],
        COLORS["purple"],
        COLORS["brown"],
        COLORS["pink"],
    ][: len(unique_concurrency)]
    cmap = ListedColormap(discrete_colors)
    boundaries = [unique_concurrency[0] - 0.5]
    boundaries.extend(value + 0.5 for value in unique_concurrency)
    norm = BoundaryNorm(boundaries, cmap.N)

    def new_cell_figure(
        *,
        left: float = 0.14,
        right: float = 0.96,
        bottom: float = 0.15,
        top: float = 0.80,
    ) -> tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots(figsize=(6.8, 5.2))
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
        return fig, ax

    def add_cell_title(ax: plt.Axes, title: str, subtitle: str) -> None:
        ax.set_title(title, fontweight="bold", pad=CELL_TITLE_PAD)
        add_panel_subtitle(ax, subtitle, width=72)

    def save_cell(fig: plt.Figure, index: int, slug: str, title: str) -> dict:
        for axis in fig.axes:
            axis.tick_params(labelsize=8)
        path = Path(f"{output_prefix}_cell_{index:02d}_{slug}.png")
        with plt.rc_context({"savefig.bbox": None}):
            fig.savefig(path, dpi=300, facecolor="white")
        plt.close(fig)
        return {"title": title, "path": path}

    plot_cells = []

    # 1. Completion breakdown
    fig, ax = new_cell_figure()
    ax.bar(x_pos, success_pct, color=COLORS["blue"], label="Successful")
    ax.bar(
        x_pos,
        incomplete_pct,
        bottom=success_pct,
        color=COLORS["orange"],
        label="Incomplete",
    )
    if np.any(error_pct > 0):
        ax.bar(
            x_pos,
            error_pct,
            bottom=success_pct + incomplete_pct,
            color=COLORS["red"],
            label="Errored",
        )
    add_cell_title(
        ax,
        "(a) Completion Breakdown",
        "Higher successful share is better; rising incomplete share indicates the system is accepting work it cannot retire in time.",
    )
    ax.set_ylabel("Requests (%)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(int(value)) for value in concurrency])
    ax.set_ylim(0, 105)
    style_axes(ax, axis="y")
    ax.legend(
        frameon=True, fancybox=False, edgecolor="black", fontsize=8, loc="lower left"
    )
    plot_cells.append(save_cell(fig, 1, "completion_breakdown", "Completion Breakdown"))

    # 2. Tail amplification
    fig, ax = new_cell_figure()
    ax.plot(
        concurrency,
        [row["ttft_tail_p95_p50"] for row in rows],
        "o-",
        color=COLORS["blue"],
        linewidth=2,
        markersize=5,
        label="TTFT p95/p50",
    )
    ax.plot(
        concurrency,
        [row["ttft_tail_p99_p50"] for row in rows],
        "o--",
        color=COLORS["orange"],
        linewidth=2,
        markersize=5,
        label="TTFT p99/p50",
    )
    ax.plot(
        concurrency,
        [row["itl_tail_p95_p50"] for row in rows],
        "s-",
        color=COLORS["green"],
        linewidth=2,
        markersize=5,
        label="ITL p95/p50",
    )
    ax.plot(
        concurrency,
        [row["itl_tail_p99_p50"] for row in rows],
        "s--",
        color=COLORS["red"],
        linewidth=2,
        markersize=5,
        label="ITL p99/p50",
    )
    ax.set_xscale("log")
    add_cell_title(
        ax,
        "(b) Tail Amplification",
        "Ratios closer to 1 are better; rising TTFT ratios indicate startup/prefill instability under load.",
    )
    ax.set_ylabel("Tail / Median")
    ax.set_xticks(concurrency)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    style_axes(ax)
    ax.legend(
        frameon=True, fancybox=False, edgecolor="black", fontsize=8, loc="upper left"
    )
    plot_cells.append(save_cell(fig, 2, "tail_amplification", "Tail Amplification"))

    # 3. TTFT/TPOT coupling
    fig, ax = new_cell_figure()
    ax.plot(
        concurrency,
        [row["ttft_tpot_corr"] for row in rows],
        "o-",
        color=COLORS["purple"],
        linewidth=2,
        markersize=5,
    )
    ax.axhline(0.0, color=COLORS["gray"], linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    add_cell_title(
        ax,
        "(c) TTFT/TPOT Coupling",
        "Lower is better; higher coupling means requests that start badly also stream badly, a stronger sign of full-system saturation.",
    )
    ax.set_ylabel("Correlation")
    ax.set_xticks(concurrency)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(-0.1, 1.0)
    style_axes(ax)
    plot_cells.append(save_cell(fig, 3, "ttft_tpot_coupling", "TTFT/TPOT Coupling"))

    # 4. TTFT tail distribution
    fig, ax = new_cell_figure()
    for level, color, marker in zip(ccdf_levels, ccdf_colors, markers):
        ttft_values = [
            request["time_to_first_token_ms"]
            for request in row_by_concurrency[level]["requests"]
        ]
        x_values, y_values = ccdf(ttft_values)
        ax.step(
            x_values,
            y_values,
            where="post",
            color=color,
            linewidth=2,
            label=f"C={level}",
        )
        marker_idx = np.linspace(
            0, len(x_values) - 1, num=min(6, len(x_values)), dtype=int
        )
        ax.plot(
            x_values[marker_idx],
            y_values[marker_idx],
            linestyle="None",
            marker=marker,
            color=color,
            markersize=4,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    add_cell_title(
        ax,
        "(d) TTFT Tail Distribution",
        "Curves farther left and dropping faster are better; this exposes heavy TTFT tails that boxplots tend to hide.",
    )
    ax.set_ylabel("P(TTFT > x)")
    style_axes(ax)
    ax.legend(
        frameon=True, fancybox=False, edgecolor="black", fontsize=8, loc="upper right"
    )
    plot_cells.append(save_cell(fig, 4, "ttft_ccdf", "TTFT Tail Distribution"))

    # 5. TTFT box distribution
    fig, ax = new_cell_figure()
    box = ax.boxplot(
        ttft_samples,
        positions=ttft_box_positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.4},
        boxprops={"edgecolor": COLORS["blue"], "linewidth": 1.1},
        whiskerprops={"color": COLORS["gray"], "linewidth": 1.0},
        capprops={"color": COLORS["gray"], "linewidth": 1.0},
    )
    for patch in box["boxes"]:
        patch.set_facecolor(COLORS["blue"])
        patch.set_alpha(0.45)
    ax.plot(
        ttft_box_positions,
        [row["ttft_p50_ms"] for row in rows],
        "o-",
        color=COLORS["green"],
        linewidth=1.6,
        markersize=4,
        label="p50",
    )
    ax.plot(
        ttft_box_positions,
        [row["ttft_p95_ms"] for row in rows],
        "o--",
        color=COLORS["orange"],
        linewidth=1.6,
        markersize=4,
        label="p95",
    )
    ax.set_yscale("log")
    add_cell_title(
        ax,
        "(e) TTFT Box Distribution",
        "Lower and tighter is better; the boxes summarize spread while the p50 and p95 overlays show center and tail movement.",
    )
    ax.set_ylabel("TTFT (ms)")
    ax.set_xticks(ttft_box_positions)
    ax.set_xticklabels([str(int(value)) for value in concurrency])
    style_axes(ax, axis="y")
    ax.legend(
        frameon=True, fancybox=False, edgecolor="black", fontsize=8, loc="upper left"
    )
    plot_cells.append(save_cell(fig, 5, "ttft_boxplot", "TTFT Box Distribution"))

    # 6. Throughput/latency frontier
    fig, ax = new_cell_figure()
    ax.scatter(
        [row["ttft_p95_ms"] for row in rows],
        [row["raw_success_rps"] for row in rows],
        s=50,
        color=COLORS["blue"],
        alpha=0.75,
        edgecolors="black",
        linewidth=0.5,
    )
    ax.plot(
        [row["ttft_p95_ms"] for row in frontier],
        [row["raw_success_rps"] for row in frontier],
        "--",
        color=COLORS["orange"],
        linewidth=2,
    )
    for row in rows:
        ax.annotate(
            str(row["concurrency"]),
            (row["ttft_p95_ms"], row["raw_success_rps"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xscale("log")
    add_cell_title(
        ax,
        "(f) Throughput/Latency Frontier",
        "Up-left is better; frontier points are not dominated by another tested load point on both throughput and tail latency.",
    )
    ax.set_ylabel("Successful req/s")
    style_axes(ax)
    plot_cells.append(
        save_cell(fig, 6, "throughput_latency_frontier", "Throughput/Latency Frontier")
    )

    # 7. Temporal stability
    fig, ax = new_cell_figure(right=0.88)
    ax.plot(
        temporal["progress_percent"],
        temporal["ttft_p95_ms"],
        "o-",
        color=COLORS["blue"],
        linewidth=2,
        markersize=5,
    )
    ax.set_yscale("log")
    add_cell_title(
        ax,
        "(g) Temporal Stability",
        "Faster settling and lower tails are better; early spikes imply startup transients, persistent elevation implies steady-state stress.",
    )
    ax.set_ylabel("TTFT p95 (ms)", color=COLORS["blue"])
    ax.tick_params(axis="y", labelcolor=COLORS["blue"])
    ax2 = ax.twinx()
    ax2.plot(
        temporal["progress_percent"],
        temporal["tpot_p95_ms"],
        "s-",
        color=COLORS["orange"],
        linewidth=2,
        markersize=5,
    )
    ax2.set_ylabel("TPOT p95 (ms)", color=COLORS["orange"])
    ax2.tick_params(axis="y", labelcolor=COLORS["orange"])
    style_axes(ax)
    plot_cells.append(save_cell(fig, 7, "temporal_stability", "Temporal Stability"))

    # 8. Request throughput
    fig, ax = new_cell_figure()
    ax.plot(
        concurrency,
        [row["raw_success_rps"] for row in rows],
        "o-",
        color=COLORS["blue"],
        linewidth=2,
        markersize=5,
    )
    ax.set_xscale("log")
    add_cell_title(
        ax,
        "(h) Request Throughput",
        "Higher is better for service capacity, but it must be read together with latency and completion integrity.",
    )
    ax.set_ylabel("Requests / sec")
    ax.set_xticks(concurrency)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    style_axes(ax)
    plot_cells.append(save_cell(fig, 8, "request_throughput", "Request Throughput"))

    # 9. Delivered token throughput
    fig, ax = new_cell_figure()
    ax.plot(
        concurrency,
        [row["successful_prompt_toksps"] / 1000.0 for row in rows],
        "o-",
        color=COLORS["green"],
        linewidth=2,
        markersize=5,
        label="Prompt",
    )
    ax.plot(
        concurrency,
        [row["successful_output_toksps"] / 1000.0 for row in rows],
        "s-",
        color=COLORS["orange"],
        linewidth=2,
        markersize=5,
        label="Output",
    )
    ax.plot(
        concurrency,
        [row["successful_total_toksps"] / 1000.0 for row in rows],
        "^-",
        color=COLORS["blue"],
        linewidth=2,
        markersize=5,
        label="Total",
    )
    ax.set_xscale("log")
    add_cell_title(
        ax,
        "(i) Delivered Token Throughput",
        "Higher is better; comparing prompt and output tok/s helps separate prefill-side and decode-side capacity.",
    )
    ax.set_ylabel("k tok/s")
    ax.set_xticks(concurrency)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    style_axes(ax)
    ax.legend(
        frameon=True, fancybox=False, edgecolor="black", fontsize=8, loc="upper left"
    )
    plot_cells.append(
        save_cell(fig, 9, "delivered_token_throughput", "Delivered Token Throughput")
    )

    # 10. Scaling efficiency
    fig, ax = new_cell_figure()
    ax.plot(
        concurrency,
        [
            row["raw_success_rps"] / (baseline["raw_success_rps"] * row["concurrency"])
            for row in rows
        ],
        "o-",
        color=COLORS["gray"],
        linewidth=2,
        markersize=5,
        label="Req/s",
    )
    ax.plot(
        concurrency,
        [
            row["successful_output_toksps"]
            / (baseline["successful_output_toksps"] * row["concurrency"])
            for row in rows
        ],
        "s-",
        color=COLORS["orange"],
        linewidth=2,
        markersize=5,
        label="Output tok/s",
    )
    ax.axhline(1.0, color=COLORS["gray"], linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    add_cell_title(
        ax,
        "(j) Scaling Efficiency",
        "Values closer to 1 are better; falling efficiency means extra concurrency is no longer converting cleanly into useful work.",
    )
    ax.set_ylabel("Relative to Linear")
    ax.set_xticks(concurrency)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    style_axes(ax)
    ax.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=8, loc="best")
    plot_cells.append(save_cell(fig, 10, "scaling_efficiency", "Scaling Efficiency"))

    # 11. Per-request decode rate
    fig, ax = new_cell_figure()
    ax.plot(
        concurrency,
        [row["output_toks_per_second_p50"] for row in rows],
        "o-",
        color=COLORS["blue"],
        linewidth=2,
        markersize=5,
        label="p50",
    )
    ax.plot(
        concurrency,
        [row["output_toks_per_second_p05"] for row in rows],
        "s-",
        color=COLORS["red"],
        linewidth=2,
        markersize=5,
        label="p05",
    )
    ax.set_xscale("log")
    add_cell_title(
        ax,
        "(k) Per-Request Decode Rate",
        "Higher is better; the lower tail highlights user-visible slow streaming even when aggregate throughput still looks healthy.",
    )
    ax.set_ylabel("Output tok/s")
    ax.set_xticks(concurrency)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    style_axes(ax)
    ax.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=8, loc="best")
    plot_cells.append(
        save_cell(fig, 11, "per_request_decode_rate", "Per-Request Decode Rate")
    )

    # 12. Max goodput by candidate SLO
    fig, ax = new_cell_figure(bottom=0.22, right=0.88)
    im = ax.imshow(max_goodput, origin="lower", aspect="auto", cmap="Blues")
    add_cell_title(
        ax,
        "(l) Max Goodput by Candidate SLO",
        "Brighter is better; this shows the best goodput achievable for each candidate TTFT and ITL latency contract.",
    )
    ax.set_xticks(np.arange(len(itl_thresholds)))
    ax.set_xticklabels(
        [f"{value:.0f}" for value in itl_thresholds], rotation=45, ha="right"
    )
    ax.set_yticks(np.arange(len(ttft_thresholds)))
    ax.set_yticklabels([f"{value:.0f}" for value in ttft_thresholds])
    ax.set_xlabel("ITL threshold (ms)")
    ax.set_ylabel("TTFT threshold (ms)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plot_cells.append(
        save_cell(fig, 12, "max_goodput_sweep", "Max Goodput by Candidate SLO")
    )

    # 13. Best concurrency by candidate SLO
    fig, ax = new_cell_figure(bottom=0.22, right=0.88)
    im = ax.imshow(
        best_concurrency, origin="lower", aspect="auto", cmap=cmap, norm=norm
    )
    add_cell_title(
        ax,
        "(m) Best Concurrency by SLO",
        "Each cell shows which tested concurrency maximizes goodput for that candidate SLO; it turns a latency target into an operating point.",
    )
    ax.set_xticks(np.arange(len(itl_thresholds)))
    ax.set_xticklabels(
        [f"{value:.0f}" for value in itl_thresholds], rotation=45, ha="right"
    )
    ax.set_yticks(np.arange(len(ttft_thresholds)))
    ax.set_yticklabels([f"{value:.0f}" for value in ttft_thresholds])
    ax.set_xlabel("ITL threshold (ms)")
    ax.set_ylabel("TTFT threshold (ms)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=unique_concurrency)
    plot_cells.append(
        save_cell(fig, 13, "best_concurrency_sweep", "Best Concurrency by SLO")
    )

    # 14. Scheduler queue wait
    fig, ax = new_cell_figure()
    ax.plot(
        concurrency,
        [row["queue_wait_p50_s"] for row in rows],
        "o-",
        color=COLORS["blue"],
        linewidth=2,
        markersize=5,
        label="p50",
    )
    ax.plot(
        concurrency,
        [row["queue_wait_p95_s"] for row in rows],
        "s-",
        color=COLORS["red"],
        linewidth=2,
        markersize=5,
        label="p95",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    add_cell_title(
        ax,
        "(n) Scheduler Queue Wait",
        "Lower is better; this isolates waiting before request execution and can reveal admission or scheduler pressure before model execution degrades.",
    )
    ax.set_ylabel("Seconds")
    ax.set_xticks(concurrency)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    style_axes(ax)
    ax.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=8, loc="best")
    plot_cells.append(
        save_cell(fig, 14, "scheduler_queue_wait", "Scheduler Queue Wait")
    )

    # 15. Effective prefill/decode rate
    fig, ax = new_cell_figure()
    ax.plot(
        concurrency,
        [row["effective_prefill_toksps_p50"] / 1000.0 for row in rows],
        "o-",
        color=COLORS["green"],
        linewidth=2,
        markersize=5,
        label="Prefill tok/s p50",
    )
    ax.plot(
        concurrency,
        [row["effective_decode_toksps_p50"] / 1000.0 for row in rows],
        "s-",
        color=COLORS["orange"],
        linewidth=2,
        markersize=5,
        label="Decode tok/s p50",
    )
    ax.set_xscale("log")
    add_cell_title(
        ax,
        "(o) Effective Prefill/Decode Rate",
        "Higher is better; comparing these curves shows whether prefill or decode is the first stage to lose efficiency under load.",
    )
    ax.set_ylabel("k tok/s")
    ax.set_xticks(concurrency)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    style_axes(ax)
    ax.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=8, loc="best")
    plot_cells.append(
        save_cell(
            fig, 15, "effective_prefill_decode_rate", "Effective Prefill/Decode Rate"
        )
    )

    # 16. Useful vs wasted output work
    fig, ax = new_cell_figure()
    useful = np.asarray([row["successful_output_toksps"] / 1000.0 for row in rows])
    wasted = np.asarray([row["incomplete_output_toksps"] / 1000.0 for row in rows])
    ax.bar(x_pos, useful, color=COLORS["blue"], label="Useful output tok/s")
    ax.bar(
        x_pos,
        wasted,
        bottom=useful,
        color=COLORS["orange"],
        label="Wasted partial output tok/s",
    )
    add_cell_title(
        ax,
        "(p) Useful vs Wasted Output Work",
        "Blue is useful completed generation and orange is work spent on cancelled requests; higher blue share and lower orange waste are better.",
    )
    ax.set_ylabel("k output tok/s")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(int(value)) for value in concurrency])
    style_axes(ax, axis="y")
    ax.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=8, loc="best")
    plot_cells.append(
        save_cell(fig, 16, "useful_vs_wasted_output", "Useful vs Wasted Output Work")
    )

    # 17. Cancelled request progress
    fig, ax = new_cell_figure()
    ax.plot(
        concurrency,
        [row["incomplete_progress_mean"] * 100.0 for row in rows],
        "o-",
        color=COLORS["blue"],
        linewidth=2,
        markersize=5,
        label="Mean",
    )
    ax.plot(
        concurrency,
        [row["incomplete_progress_p50"] * 100.0 for row in rows],
        "s-",
        color=COLORS["orange"],
        linewidth=2,
        markersize=5,
        label="p50",
    )
    ax.set_xscale("log")
    add_cell_title(
        ax,
        "(q) Cancelled Request Progress",
        "Lower is better from a waste perspective; high values mean cancellations happen late after substantial decode work has already been spent.",
    )
    ax.set_ylabel("% of Target Output")
    ax.set_xticks(concurrency)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    style_axes(ax)
    ax.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=8, loc="best")
    plot_cells.append(
        save_cell(fig, 17, "cancelled_request_progress", "Cancelled Request Progress")
    )

    # 18. Token throughput per GPU
    fig, ax = new_cell_figure(bottom=0.20)
    per_gpu_x = [row["concurrency_per_gpu"] for row in rows]
    per_gpu_labels = [f"{value:g}" for value in per_gpu_x]
    ax.plot(
        per_gpu_x,
        [row["successful_prompt_toksps_per_gpu"] / 1000.0 for row in rows],
        "o-",
        color=COLORS["green"],
        linewidth=2,
        markersize=5,
        label="Prompt",
    )
    ax.plot(
        per_gpu_x,
        [row["successful_output_toksps_per_gpu"] / 1000.0 for row in rows],
        "s-",
        color=COLORS["orange"],
        linewidth=2,
        markersize=5,
        label="Output",
    )
    ax.plot(
        per_gpu_x,
        [row["successful_total_toksps_per_gpu"] / 1000.0 for row in rows],
        "^-",
        color=COLORS["blue"],
        linewidth=2,
        markersize=5,
        label="Total",
    )
    ax.set_xscale("log")
    add_cell_title(
        ax,
        "(r) Token Throughput per GPU",
        "Higher is better; the x-axis is concurrency per GPU, which makes accelerator-normalized capacity easier to compare across cluster shapes.",
    )
    ax.set_xlabel("Concurrency / GPU")
    ax.set_ylabel("k tok/s / GPU")
    ax.set_xticks(per_gpu_x)
    ax.set_xticklabels(per_gpu_labels, rotation=35, ha="right")
    style_axes(ax)
    ax.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=8, loc="best")
    plot_cells.append(
        save_cell(fig, 18, "token_throughput_per_gpu", "Token Throughput per GPU")
    )

    # 19. Throughput efficiency per GPU
    fig, ax = new_cell_figure(bottom=0.20)
    interactivity = [
        row["successful_output_toksps"] / row["concurrency"] for row in rows
    ]
    throughput_per_gpu = [row["successful_output_toksps_per_gpu"] for row in rows]
    ax.plot(
        interactivity,
        throughput_per_gpu,
        "--",
        color=COLORS["gray"],
        linewidth=1.4,
        alpha=0.8,
    )
    ax.scatter(
        interactivity,
        throughput_per_gpu,
        s=42,
        color=COLORS["orange"],
        edgecolors="black",
        linewidth=0.5,
        zorder=3,
    )
    for row, x_value, y_value in zip(rows, interactivity, throughput_per_gpu):
        ax.annotate(
            str(int(row["concurrency"])),
            (x_value, y_value),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )
    add_cell_title(
        ax,
        "(s) Throughput Efficiency per GPU",
        "Up-right is better; x captures interactivity as output tok/s per concurrency while y captures delivered output tok/s per GPU.",
    )
    ax.set_xlabel("Interactivity (output tok/s/concurrency)")
    ax.set_ylabel("Output throughput (tok/s/GPU)")
    style_axes(ax)
    plot_cells.append(
        save_cell(
            fig, 19, "throughput_efficiency_per_gpu", "Throughput Efficiency per GPU"
        )
    )

    # 20. TTFT vs actual concurrency
    fig, ax = new_cell_figure(bottom=0.25)
    actual_x = [entry["x"] for entry in actual_concurrency_summary]
    actual_labels = [entry["label"] for entry in actual_concurrency_summary]
    ax.plot(
        actual_x,
        [entry["ttft_p50_ms"] for entry in actual_concurrency_summary],
        "o-",
        color=COLORS["green"],
        linewidth=2,
        markersize=5,
        label="p50",
    )
    ax.plot(
        actual_x,
        [entry["ttft_p95_ms"] for entry in actual_concurrency_summary],
        "s-",
        color=COLORS["orange"],
        linewidth=2,
        markersize=5,
        label="p95",
    )
    ax.plot(
        actual_x,
        [entry["ttft_p99_ms"] for entry in actual_concurrency_summary],
        "^-",
        color=COLORS["red"],
        linewidth=2,
        markersize=5,
        label="p99",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    add_cell_title(
        ax,
        "(t) TTFT vs Actual Concurrency",
        "Lower and flatter are better; actual concurrency is derived from overlap of request lifetimes, which is a better saturation lens than target load alone.",
    )
    ax.set_xlabel("Observed in-flight requests")
    ax.set_ylabel("TTFT (ms)")
    ax.set_xticks(actual_x)
    ax.set_xticklabels(actual_labels, rotation=35, ha="right")
    style_axes(ax)
    ax.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=8, loc="best")
    plot_cells.append(
        save_cell(fig, 20, "ttft_vs_actual_concurrency", "TTFT vs Actual Concurrency")
    )

    # 21. Delay decomposition
    fig, ax = new_cell_figure()
    ax.plot(
        concurrency,
        [row["queue_wait_p50_s"] for row in rows],
        "o-",
        color=COLORS["blue"],
        linewidth=2,
        markersize=5,
        label="Queue wait p50",
    )
    ax.plot(
        concurrency,
        [row["ttft_p50_ms"] / 1000.0 for row in rows],
        "o-",
        color=COLORS["green"],
        linewidth=2,
        markersize=5,
        label="TTFT p50",
    )
    ax.plot(
        concurrency,
        [row["decode_duration_p50_s"] for row in rows],
        "o-",
        color=COLORS["orange"],
        linewidth=2,
        markersize=5,
        label="Decode p50",
    )
    ax.plot(
        concurrency,
        [row["queue_wait_p95_s"] for row in rows],
        "s--",
        color=COLORS["blue"],
        linewidth=2,
        markersize=5,
        label="Queue wait p95",
    )
    ax.plot(
        concurrency,
        [row["ttft_p95_ms"] / 1000.0 for row in rows],
        "s--",
        color=COLORS["green"],
        linewidth=2,
        markersize=5,
        label="TTFT p95",
    )
    ax.plot(
        concurrency,
        [row["decode_duration_p95_s"] for row in rows],
        "s--",
        color=COLORS["orange"],
        linewidth=2,
        markersize=5,
        label="Decode p95",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    add_cell_title(
        ax,
        "(u) Delay Decomposition",
        "Lower is better; the first component to rise identifies whether scheduler wait, first-token latency, or decode duration is becoming the limiting stage.",
    )
    ax.set_ylabel("Seconds")
    ax.set_xticks(concurrency)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    style_axes(ax)
    ax.legend(
        frameon=True,
        fancybox=False,
        edgecolor="black",
        fontsize=7.3,
        loc="best",
        ncol=2,
    )
    plot_cells.append(save_cell(fig, 21, "delay_decomposition", "Delay Decomposition"))

    print(f"✓ Saved: {len(plot_cells)} HTML plot cells")
    return plot_cells


def save_html_report(output_prefix: str, plot_cells: list[dict]) -> None:
    html_path = Path(f"{output_prefix}_report.html")

    def data_uri(path: Path) -> str:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    rows = []
    for start in range(0, len(plot_cells), 3):
        row_cells = []
        for cell in plot_cells[start : start + 3]:
            row_cells.append(
                f"""
        <td valign="top">
          <img src="{data_uri(cell["path"])}" alt="{cell["title"]}" width="100%">
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
</head>
<body bgcolor="white">
  <table width="100%" border="0" cellspacing="12" cellpadding="0">
{"".join(rows)}
  </table>
</body>
</html>
"""
    html_path.write_text(content)
    print(f"✓ Saved: {html_path.name}")


def create_overview_figure(
    rows: list[dict],
    strict_slo: tuple[float, float],
    relaxed_slo: tuple[float, float],
    bin_count: int,
) -> plt.Figure:
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        "LLM Serving Exploratory Diagnostics",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    concurrency = np.asarray([row["concurrency"] for row in rows], dtype=float)
    success_pct = np.asarray([row["success_rate"] * 100.0 for row in rows])
    incomplete_pct = np.asarray([row["incomplete_rate"] * 100.0 for row in rows])
    error_pct = np.asarray([row["errored_rate"] * 100.0 for row in rows])

    # (a) Completion breakdown
    ax1 = plt.subplot(2, 3, 1)
    x_pos = np.arange(len(rows))
    ax1.bar(x_pos, success_pct, color=COLORS["blue"], label="Successful")
    ax1.bar(
        x_pos,
        incomplete_pct,
        bottom=success_pct,
        color=COLORS["orange"],
        label="Incomplete",
    )
    if np.any(error_pct > 0):
        ax1.bar(
            x_pos,
            error_pct,
            bottom=success_pct + incomplete_pct,
            color=COLORS["red"],
            label="Errored",
        )
    ax1.set_xlabel("Target Concurrency")
    ax1.set_ylabel("Requests (%)")
    ax1.set_title("(a) Completion Breakdown", fontweight="bold", pad=PANEL_TITLE_PAD)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([str(int(value)) for value in concurrency])
    ax1.set_ylim(0, 105)
    style_axes(ax1, axis="y")
    ax1.legend(frameon=True, fancybox=False, edgecolor="black")
    add_panel_subtitle(
        ax1,
        "Shows what fraction of offered work finishes at each load point; higher successful share and lower incomplete share are better.",
    )

    # (b) Tail amplification
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(
        concurrency,
        [row["ttft_tail_p95_p50"] for row in rows],
        "o-",
        color=COLORS["blue"],
        linewidth=2,
        markersize=6,
        label="TTFT p95 / p50",
        markeredgewidth=0.5,
        markeredgecolor="white",
    )
    ax2.plot(
        concurrency,
        [row["ttft_tail_p99_p50"] for row in rows],
        "o--",
        color=COLORS["orange"],
        linewidth=2,
        markersize=6,
        label="TTFT p99 / p50",
        markeredgewidth=0.5,
        markeredgecolor="white",
    )
    ax2.plot(
        concurrency,
        [row["itl_tail_p95_p50"] for row in rows],
        "s-",
        color=COLORS["green"],
        linewidth=2,
        markersize=6,
        label="ITL p95 / p50",
        markeredgewidth=0.5,
        markeredgecolor="white",
    )
    ax2.plot(
        concurrency,
        [row["itl_tail_p99_p50"] for row in rows],
        "s--",
        color=COLORS["red"],
        linewidth=2,
        markersize=6,
        label="ITL p99 / p50",
        markeredgewidth=0.5,
        markeredgecolor="white",
    )
    ax2.set_xscale("log")
    ax2.set_xlabel("Target Concurrency")
    ax2.set_ylabel("Tail / Median Ratio")
    ax2.set_title("(b) Tail Amplification", fontweight="bold", pad=PANEL_TITLE_PAD)
    ax2.set_xticks(concurrency)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    style_axes(ax2)
    ax2.legend(frameon=True, fancybox=False, edgecolor="black", loc="best")
    add_panel_subtitle(
        ax2,
        "Shows how much the tail stretches relative to the median; ratios closer to 1 mean a tighter, more stable latency distribution.",
    )

    # (c) TTFT/TPOT coupling
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(
        concurrency,
        [row["ttft_tpot_corr"] for row in rows],
        "o-",
        color=COLORS["purple"],
        linewidth=2,
        markersize=6,
        label="Pearson corr(TTFT, TPOT)",
        markeredgewidth=0.5,
        markeredgecolor="white",
    )
    ax3.axhline(0.0, color=COLORS["gray"], linestyle="--", linewidth=1.0)
    ax3.set_xscale("log")
    ax3.set_xlabel("Target Concurrency")
    ax3.set_ylabel("Correlation Coefficient")
    ax3.set_title("(c) TTFT/TPOT Coupling", fontweight="bold", pad=PANEL_TITLE_PAD)
    ax3.set_xticks(concurrency)
    ax3.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax3.set_ylim(-0.1, 1.0)
    style_axes(ax3)
    ax3.legend(frameon=True, fancybox=False, edgecolor="black", loc="best")
    add_panel_subtitle(
        ax3,
        "Shows whether requests with bad TTFT also decode badly; lower coupling is better because startup and decode failures remain less entangled.",
    )

    # (d) TTFT CCDF
    ax4 = plt.subplot(2, 3, 4)
    ccdf_levels = select_ccdf_levels([row["concurrency"] for row in rows])
    ccdf_colors = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["red"]]
    markers = ["o", "s", "^", "v"]
    row_by_concurrency = {row["concurrency"]: row for row in rows}

    for level, color, marker in zip(ccdf_levels, ccdf_colors, markers):
        ttft_values = [
            request["time_to_first_token_ms"]
            for request in row_by_concurrency[level]["requests"]
        ]
        x_values, y_values = ccdf(ttft_values)
        ax4.step(
            x_values,
            y_values,
            where="post",
            color=color,
            linewidth=2,
            label=f"Concurrency {level}",
        )
        # Add a sparse marker set so the plot still matches the academic template.
        marker_idx = np.linspace(
            0, len(x_values) - 1, num=min(8, len(x_values)), dtype=int
        )
        ax4.plot(
            x_values[marker_idx],
            y_values[marker_idx],
            linestyle="None",
            marker=marker,
            color=color,
            markersize=5,
            markeredgewidth=0.5,
            markeredgecolor="white",
        )

    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_xlabel("TTFT (ms)")
    ax4.set_ylabel("CCDF: P(TTFT > x)")
    ax4.set_title("(d) TTFT Tail Distribution", fontweight="bold", pad=PANEL_TITLE_PAD)
    style_axes(ax4)
    ax4.legend(frameon=True, fancybox=False, edgecolor="black", loc="best")
    add_panel_subtitle(
        ax4,
        "Shows the fraction of requests exceeding any TTFT threshold; curves farther left and dropping faster indicate better tail behavior.",
    )

    # (e) Pareto frontier
    ax5 = plt.subplot(2, 3, 5)
    raw_success_rps = np.asarray([row["raw_success_rps"] for row in rows])
    ttft_p95 = np.asarray([row["ttft_p95_ms"] for row in rows])
    frontier = pareto_frontier(rows)
    ax5.scatter(
        ttft_p95,
        raw_success_rps,
        s=70,
        color=COLORS["blue"],
        alpha=0.75,
        edgecolors="black",
        linewidth=0.5,
        label="Load point",
    )
    ax5.plot(
        [row["ttft_p95_ms"] for row in frontier],
        [row["raw_success_rps"] for row in frontier],
        "--",
        color=COLORS["orange"],
        linewidth=2,
        label="Pareto frontier",
    )
    for row in rows:
        ax5.annotate(
            str(row["concurrency"]),
            (row["ttft_p95_ms"], row["raw_success_rps"]),
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax5.set_xscale("log")
    ax5.set_xlabel("TTFT p95 (ms)")
    ax5.set_ylabel("Successful Throughput (req/s)")
    ax5.set_title(
        "(e) Throughput/Latency Pareto Frontier", fontweight="bold", pad=PANEL_TITLE_PAD
    )
    style_axes(ax5)
    ax5.legend(frameon=True, fancybox=False, edgecolor="black", loc="best")
    add_panel_subtitle(
        ax5,
        "Shows the throughput versus TTFT tail tradeoff across load points; points up and left are better, frontier points are not dominated.",
    )

    # (f) Temporal stability
    ax6 = plt.subplot(2, 3, 6)
    highest_concurrency_row = max(rows, key=lambda row: row["concurrency"])
    temporal = temporal_bins(highest_concurrency_row["benchmark"], bin_count)

    ax6.plot(
        temporal["progress_percent"],
        temporal["ttft_p95_ms"],
        "o-",
        color=COLORS["blue"],
        linewidth=2,
        markersize=6,
        label="TTFT p95",
        markeredgewidth=0.5,
        markeredgecolor="white",
    )
    ax6.set_xlabel("Run Progress (%)")
    ax6.set_ylabel("TTFT p95 (ms)", color=COLORS["blue"])
    ax6.tick_params(axis="y", labelcolor=COLORS["blue"])
    ax6.set_title(
        f"(f) Temporal Stability at Concurrency {highest_concurrency_row['concurrency']}",
        fontweight="bold",
        pad=PANEL_TITLE_PAD,
    )
    ax6.set_yscale("log")
    style_axes(ax6)

    ax6b = ax6.twinx()
    ax6b.plot(
        temporal["progress_percent"],
        temporal["tpot_p95_ms"],
        "s-",
        color=COLORS["orange"],
        linewidth=2,
        markersize=6,
        label="TPOT p95",
        markeredgewidth=0.5,
        markeredgecolor="white",
    )
    ax6b.set_ylabel("TPOT p95 (ms)", color=COLORS["orange"])
    ax6b.tick_params(axis="y", labelcolor=COLORS["orange"])

    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6b.get_legend_handles_labels()
    ax6.legend(
        lines1 + lines2,
        labels1 + labels2,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        loc="best",
    )
    add_panel_subtitle(
        ax6,
        "Shows whether worst-case latency is transient or persistent during the run; faster settling and lower tails indicate a healthier operating regime.",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def create_throughput_figure(rows: list[dict]) -> plt.Figure:
    fig = plt.figure(figsize=(15, 9.5))
    fig.suptitle(
        "LLM Serving Throughput Diagnostics",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    concurrency = np.asarray([row["concurrency"] for row in rows], dtype=float)
    baseline = min(rows, key=lambda row: row["concurrency"])
    baseline_output_toksps = baseline["successful_output_toksps"]
    baseline_total_toksps = baseline["successful_total_toksps"]
    baseline_reqps = baseline["raw_success_rps"]

    # (a) Request throughput
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(
        concurrency,
        [row["raw_success_rps"] for row in rows],
        "o-",
        color=COLORS["blue"],
        linewidth=2,
        markersize=6,
        label="Successful req/s",
        markeredgewidth=0.5,
        markeredgecolor="white",
    )
    ax1.set_xscale("log")
    ax1.set_xlabel("Target Concurrency")
    ax1.set_ylabel("Requests / sec")
    ax1.set_title("(a) Request Throughput", fontweight="bold", pad=PANEL_TITLE_PAD)
    ax1.set_xticks(concurrency)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    style_axes(ax1)
    ax1.legend(frameon=True, fancybox=False, edgecolor="black", loc="best")
    add_panel_subtitle(
        ax1,
        "Counts completed user requests per second; higher is better, but it should be read together with completion rate and tail latency.",
    )

    # (b) Delivered token throughput decomposition
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(
        concurrency,
        [row["successful_prompt_toksps"] / 1000.0 for row in rows],
        "o-",
        color=COLORS["green"],
        linewidth=2,
        markersize=6,
        label="Prompt tok/s",
        markeredgewidth=0.5,
        markeredgecolor="white",
    )
    ax2.plot(
        concurrency,
        [row["successful_output_toksps"] / 1000.0 for row in rows],
        "s-",
        color=COLORS["orange"],
        linewidth=2,
        markersize=6,
        label="Output tok/s",
        markeredgewidth=0.5,
        markeredgecolor="white",
    )
    ax2.plot(
        concurrency,
        [row["successful_total_toksps"] / 1000.0 for row in rows],
        "^-",
        color=COLORS["blue"],
        linewidth=2,
        markersize=6,
        label="Total tok/s",
        markeredgewidth=0.5,
        markeredgecolor="white",
    )
    ax2.set_xscale("log")
    ax2.set_xlabel("Target Concurrency")
    ax2.set_ylabel("Delivered Throughput (k tok/s)")
    ax2.set_title(
        "(b) Delivered Token Throughput", fontweight="bold", pad=PANEL_TITLE_PAD
    )
    ax2.set_xticks(concurrency)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    style_axes(ax2)
    ax2.legend(frameon=True, fancybox=False, edgecolor="black", loc="best")
    add_panel_subtitle(
        ax2,
        "Separates prompt, output, and total delivered token work; higher is better and helps distinguish prefill-side versus decode-side capacity.",
    )

    # (c) Scaling efficiency
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(
        concurrency,
        [
            row["raw_success_rps"] / (baseline_reqps * row["concurrency"])
            if baseline_reqps > 0
            else np.nan
            for row in rows
        ],
        "o-",
        color=COLORS["gray"],
        linewidth=2,
        markersize=6,
        label="Req/s scaling efficiency",
        markeredgewidth=0.5,
        markeredgecolor="white",
    )
    ax3.plot(
        concurrency,
        [
            row["successful_output_toksps"]
            / (baseline_output_toksps * row["concurrency"])
            if baseline_output_toksps > 0
            else np.nan
            for row in rows
        ],
        "s-",
        color=COLORS["orange"],
        linewidth=2,
        markersize=6,
        label="Output tok/s scaling efficiency",
        markeredgewidth=0.5,
        markeredgecolor="white",
    )
    ax3.plot(
        concurrency,
        [
            row["successful_total_toksps"]
            / (baseline_total_toksps * row["concurrency"])
            if baseline_total_toksps > 0
            else np.nan
            for row in rows
        ],
        "^-",
        color=COLORS["blue"],
        linewidth=2,
        markersize=6,
        label="Total tok/s scaling efficiency",
        markeredgewidth=0.5,
        markeredgecolor="white",
    )
    ax3.axhline(1.0, color=COLORS["gray"], linestyle="--", linewidth=1.0)
    ax3.set_xscale("log")
    ax3.set_xlabel("Target Concurrency")
    ax3.set_ylabel("Relative to Linear Scaling")
    ax3.set_title(
        "(c) Throughput Scaling Efficiency", fontweight="bold", pad=PANEL_TITLE_PAD
    )
    ax3.set_xticks(concurrency)
    ax3.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    style_axes(ax3)
    ax3.legend(frameon=True, fancybox=False, edgecolor="black", loc="best")
    add_panel_subtitle(
        ax3,
        "Measures how much extra concurrency translates into delivered work relative to the single-stream baseline; values closer to 1 scale more efficiently.",
    )

    # (d) Per-request decode throughput
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(
        concurrency,
        [row["output_toks_per_second_p50"] for row in rows],
        "o-",
        color=COLORS["blue"],
        linewidth=2,
        markersize=6,
        label="Per-request output tok/s p50",
        markeredgewidth=0.5,
        markeredgecolor="white",
    )
    ax4.plot(
        concurrency,
        [row["output_toks_per_second_p05"] for row in rows],
        "s-",
        color=COLORS["red"],
        linewidth=2,
        markersize=6,
        label="Per-request output tok/s p05",
        markeredgewidth=0.5,
        markeredgecolor="white",
    )
    ax4.set_xscale("log")
    ax4.set_xlabel("Target Concurrency")
    ax4.set_ylabel("Output Tokens / sec")
    ax4.set_title("(d) Per-Request Decode Rate", fontweight="bold", pad=PANEL_TITLE_PAD)
    ax4.set_xticks(concurrency)
    ax4.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    style_axes(ax4)
    ax4.legend(frameon=True, fancybox=False, edgecolor="black", loc="best")
    add_panel_subtitle(
        ax4,
        "Shows the generation speed seen by individual successful requests; higher is better and the lower tail indicates user-visible slow streaming.",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


def create_slo_sweep_figure(
    rows: list[dict],
    ttft_thresholds: list[float],
    itl_thresholds: list[float],
    strict_slo: tuple[float, float],
    relaxed_slo: tuple[float, float],
) -> plt.Figure:
    max_goodput, best_concurrency = compute_slo_sweep(
        rows, ttft_thresholds, itl_thresholds
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.8))
    fig.suptitle(
        "Candidate-SLO Threshold Sweep",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    goodput_im = ax1.imshow(max_goodput, origin="lower", aspect="auto", cmap="Blues")
    ax1.set_title("(a) Max Achievable Goodput", fontweight="bold", pad=PANEL_TITLE_PAD)
    ax1.set_xlabel("ITL Threshold (ms)")
    ax1.set_ylabel("TTFT Threshold (ms)")
    ax1.set_xticks(np.arange(len(itl_thresholds)))
    ax1.set_xticklabels([f"{value:.0f}" for value in itl_thresholds])
    ax1.set_yticks(np.arange(len(ttft_thresholds)))
    ax1.set_yticklabels([f"{value:.0f}" for value in ttft_thresholds])
    ax1.grid(False)
    cbar1 = fig.colorbar(goodput_im, ax=ax1)
    cbar1.set_label("Goodput (requests/sec)")
    add_panel_subtitle(
        ax1,
        "For each candidate TTFT and ITL threshold pair, this shows the best goodput achievable across tested load points; brighter is better.",
    )

    unique_concurrency = sorted({int(row["concurrency"]) for row in rows})
    discrete_colors = [
        COLORS["blue"],
        COLORS["orange"],
        COLORS["green"],
        COLORS["red"],
        COLORS["purple"],
        COLORS["brown"],
        COLORS["pink"],
    ][: len(unique_concurrency)]
    cmap = ListedColormap(discrete_colors)
    boundaries = [unique_concurrency[0] - 0.5]
    boundaries.extend(value + 0.5 for value in unique_concurrency)
    norm = BoundaryNorm(boundaries, cmap.N)

    concurrency_im = ax2.imshow(
        best_concurrency,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=norm,
    )
    ax2.set_title(
        "(b) Best Concurrency by Candidate SLO", fontweight="bold", pad=PANEL_TITLE_PAD
    )
    ax2.set_xlabel("ITL Threshold (ms)")
    ax2.set_ylabel("TTFT Threshold (ms)")
    ax2.set_xticks(np.arange(len(itl_thresholds)))
    ax2.set_xticklabels([f"{value:.0f}" for value in itl_thresholds])
    ax2.set_yticks(np.arange(len(ttft_thresholds)))
    ax2.set_yticklabels([f"{value:.0f}" for value in ttft_thresholds])
    ax2.grid(False)
    cbar2 = fig.colorbar(concurrency_im, ax=ax2, ticks=unique_concurrency)
    cbar2.set_label("Best concurrency")

    for y_index in range(len(ttft_thresholds)):
        for x_index in range(len(itl_thresholds)):
            ax2.text(
                x_index,
                y_index,
                f"{int(best_concurrency[y_index, x_index])}",
                ha="center",
                va="center",
                fontsize=7,
                color="white"
                if best_concurrency[y_index, x_index] in unique_concurrency[-3:]
                else "black",
            )

    example_points = [
        ("Strict example", strict_slo, COLORS["red"], "o"),
        ("Relaxed example", relaxed_slo, COLORS["purple"], "s"),
    ]
    for label, (ttft_threshold, itl_threshold), color, marker in example_points:
        ttft_idx = int(np.argmin(np.abs(np.asarray(ttft_thresholds) - ttft_threshold)))
        itl_idx = int(np.argmin(np.abs(np.asarray(itl_thresholds) - itl_threshold)))
        for axis in (ax1, ax2):
            axis.scatter(
                itl_idx,
                ttft_idx,
                s=90,
                color=color,
                marker=marker,
                edgecolors="black",
                linewidth=0.6,
                label=label,
            )

    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        ax1.legend(frameon=True, fancybox=False, edgecolor="black", loc="best")
    add_panel_subtitle(
        ax2,
        "For each candidate TTFT and ITL threshold pair, this shows which concurrency maximizes goodput; it helps convert a target SLO into an operating point.",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    return fig


def create_all_plots_figure(
    rows: list[dict],
    ttft_thresholds: list[float],
    itl_thresholds: list[float],
    strict_slo: tuple[float, float],
    relaxed_slo: tuple[float, float],
    bin_count: int,
) -> tuple[plt.Figure, list[dict]]:
    max_goodput, best_concurrency = compute_slo_sweep(
        rows, ttft_thresholds, itl_thresholds
    )

    fig = plt.figure(figsize=(24, 36))
    plot_specs = []

    concurrency = np.asarray([row["concurrency"] for row in rows], dtype=float)
    success_pct = np.asarray([row["success_rate"] * 100.0 for row in rows])
    incomplete_pct = np.asarray([row["incomplete_rate"] * 100.0 for row in rows])
    error_pct = np.asarray([row["errored_rate"] * 100.0 for row in rows])
    frontier = pareto_frontier(rows)
    row_by_concurrency = {row["concurrency"]: row for row in rows}
    ccdf_levels = select_ccdf_levels([row["concurrency"] for row in rows])
    ccdf_colors = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["red"]]
    markers = ["o", "s", "^", "v"]
    ttft_samples = [
        [request["time_to_first_token_ms"] for request in row["requests"]]
        for row in rows
    ]
    ttft_box_positions = np.arange(1, len(rows) + 1)
    actual_concurrency_summary = actual_concurrency_percentiles(rows)

    # (a) Completion breakdown
    ax1 = plt.subplot(6, 4, 1)
    x_pos = np.arange(len(rows))
    ax1.bar(x_pos, success_pct, color=COLORS["blue"], label="Successful")
    ax1.bar(
        x_pos,
        incomplete_pct,
        bottom=success_pct,
        color=COLORS["orange"],
        label="Incomplete",
    )
    if np.any(error_pct > 0):
        ax1.bar(
            x_pos,
            error_pct,
            bottom=success_pct + incomplete_pct,
            color=COLORS["red"],
            label="Errored",
        )
    ax1.set_title("(a) Completion Breakdown", fontweight="bold", pad=PANEL_TITLE_PAD)
    ax1.set_ylabel("Requests (%)")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([str(int(value)) for value in concurrency])
    ax1.set_ylim(0, 105)
    style_axes(ax1, axis="y")
    ax1.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=8, loc="best")
    add_panel_subtitle(
        ax1,
        "Higher successful share is better; rising incomplete share indicates the system is accepting work it cannot retire in time.",
        width=64,
    )
    plot_specs.append(
        {"slug": "completion_breakdown", "title": "Completion Breakdown", "axes": [ax1]}
    )

    # (b) Tail amplification
    ax2 = plt.subplot(6, 4, 2)
    ax2.plot(
        concurrency,
        [row["ttft_tail_p95_p50"] for row in rows],
        "o-",
        color=COLORS["blue"],
        linewidth=2,
        markersize=5,
        label="TTFT p95/p50",
    )
    ax2.plot(
        concurrency,
        [row["ttft_tail_p99_p50"] for row in rows],
        "o--",
        color=COLORS["orange"],
        linewidth=2,
        markersize=5,
        label="TTFT p99/p50",
    )
    ax2.plot(
        concurrency,
        [row["itl_tail_p95_p50"] for row in rows],
        "s-",
        color=COLORS["green"],
        linewidth=2,
        markersize=5,
        label="ITL p95/p50",
    )
    ax2.plot(
        concurrency,
        [row["itl_tail_p99_p50"] for row in rows],
        "s--",
        color=COLORS["red"],
        linewidth=2,
        markersize=5,
        label="ITL p99/p50",
    )
    ax2.set_xscale("log")
    ax2.set_title("(b) Tail Amplification", fontweight="bold", pad=PANEL_TITLE_PAD)
    ax2.set_ylabel("Tail / Median")
    ax2.set_xticks(concurrency)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    style_axes(ax2)
    ax2.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=8, loc="best")
    add_panel_subtitle(
        ax2,
        "Ratios closer to 1 are better; rising TTFT ratios indicate startup/prefill instability under load.",
        width=64,
    )
    plot_specs.append(
        {"slug": "tail_amplification", "title": "Tail Amplification", "axes": [ax2]}
    )

    # (c) TTFT/TPOT coupling
    ax3 = plt.subplot(6, 4, 3)
    ax3.plot(
        concurrency,
        [row["ttft_tpot_corr"] for row in rows],
        "o-",
        color=COLORS["purple"],
        linewidth=2,
        markersize=5,
    )
    ax3.axhline(0.0, color=COLORS["gray"], linestyle="--", linewidth=1.0)
    ax3.set_xscale("log")
    ax3.set_title("(c) TTFT/TPOT Coupling", fontweight="bold", pad=PANEL_TITLE_PAD)
    ax3.set_ylabel("Correlation")
    ax3.set_xticks(concurrency)
    ax3.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax3.set_ylim(-0.1, 1.0)
    style_axes(ax3)
    add_panel_subtitle(
        ax3,
        "Lower is better; higher coupling means requests that start badly also stream badly, a stronger sign of full-system saturation.",
        width=64,
    )
    plot_specs.append(
        {"slug": "ttft_tpot_coupling", "title": "TTFT/TPOT Coupling", "axes": [ax3]}
    )

    # (d) TTFT CCDF
    ax4 = plt.subplot(6, 4, 4)
    for level, color, marker in zip(ccdf_levels, ccdf_colors, markers):
        ttft_values = [
            request["time_to_first_token_ms"]
            for request in row_by_concurrency[level]["requests"]
        ]
        x_values, y_values = ccdf(ttft_values)
        ax4.step(
            x_values,
            y_values,
            where="post",
            color=color,
            linewidth=2,
            label=f"C={level}",
        )
        marker_idx = np.linspace(
            0, len(x_values) - 1, num=min(6, len(x_values)), dtype=int
        )
        ax4.plot(
            x_values[marker_idx],
            y_values[marker_idx],
            linestyle="None",
            marker=marker,
            color=color,
            markersize=4,
        )
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.set_title("(d) TTFT Tail Distribution", fontweight="bold", pad=PANEL_TITLE_PAD)
    ax4.set_ylabel("P(TTFT > x)")
    style_axes(ax4)
    ax4.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=8, loc="best")
    add_panel_subtitle(
        ax4,
        "Curves farther left and dropping faster are better; this exposes heavy TTFT tails that boxplots tend to hide.",
        width=64,
    )
    plot_specs.append(
        {"slug": "ttft_ccdf", "title": "TTFT Tail Distribution", "axes": [ax4]}
    )

    # (e) TTFT box distribution
    ax5box = plt.subplot(6, 4, 5)
    box = ax5box.boxplot(
        ttft_samples,
        positions=ttft_box_positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.4},
        boxprops={"edgecolor": COLORS["blue"], "linewidth": 1.1},
        whiskerprops={"color": COLORS["gray"], "linewidth": 1.0},
        capprops={"color": COLORS["gray"], "linewidth": 1.0},
    )
    for patch in box["boxes"]:
        patch.set_facecolor(COLORS["blue"])
        patch.set_alpha(0.45)
    ax5box.plot(
        ttft_box_positions,
        [row["ttft_p50_ms"] for row in rows],
        "o-",
        color=COLORS["green"],
        linewidth=1.6,
        markersize=4,
        label="p50",
    )
    ax5box.plot(
        ttft_box_positions,
        [row["ttft_p95_ms"] for row in rows],
        "o--",
        color=COLORS["orange"],
        linewidth=1.6,
        markersize=4,
        label="p95",
    )
    ax5box.set_yscale("log")
    ax5box.set_title(
        "(e) TTFT Box Distribution", fontweight="bold", pad=PANEL_TITLE_PAD
    )
    ax5box.set_ylabel("TTFT (ms)")
    ax5box.set_xticks(ttft_box_positions)
    ax5box.set_xticklabels([str(int(value)) for value in concurrency])
    style_axes(ax5box, axis="y")
    ax5box.legend(
        frameon=True, fancybox=False, edgecolor="black", fontsize=8, loc="upper left"
    )
    add_panel_subtitle(
        ax5box,
        "Lower and tighter is better; the boxes summarize spread while the p50 and p95 overlays show center and tail movement.",
        width=64,
    )
    plot_specs.append(
        {"slug": "ttft_boxplot", "title": "TTFT Box Distribution", "axes": [ax5box]}
    )

    # (f) Throughput/latency Pareto
    ax5 = plt.subplot(6, 4, 6)
    ax5.scatter(
        [row["ttft_p95_ms"] for row in rows],
        [row["raw_success_rps"] for row in rows],
        s=50,
        color=COLORS["blue"],
        alpha=0.75,
        edgecolors="black",
        linewidth=0.5,
    )
    ax5.plot(
        [row["ttft_p95_ms"] for row in frontier],
        [row["raw_success_rps"] for row in frontier],
        "--",
        color=COLORS["orange"],
        linewidth=2,
    )
    for row in rows:
        ax5.annotate(
            str(row["concurrency"]),
            (row["ttft_p95_ms"], row["raw_success_rps"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )
    ax5.set_xscale("log")
    ax5.set_title(
        "(f) Throughput/Latency Frontier", fontweight="bold", pad=PANEL_TITLE_PAD
    )
    ax5.set_ylabel("Successful req/s")
    style_axes(ax5)
    add_panel_subtitle(
        ax5,
        "Up-left is better; frontier points are not dominated by another tested load point on both throughput and tail latency.",
        width=64,
    )
    plot_specs.append(
        {
            "slug": "throughput_latency_frontier",
            "title": "Throughput/Latency Frontier",
            "axes": [ax5],
        }
    )

    # (g) Temporal stability
    ax6 = plt.subplot(6, 4, 7)
    highest_row = max(rows, key=lambda row: row["concurrency"])
    temporal = temporal_bins(highest_row["benchmark"], bin_count)
    ax6.plot(
        temporal["progress_percent"],
        temporal["ttft_p95_ms"],
        "o-",
        color=COLORS["blue"],
        linewidth=2,
        markersize=5,
        label="TTFT p95",
    )
    ax6.set_yscale("log")
    ax6.set_title("(g) Temporal Stability", fontweight="bold", pad=PANEL_TITLE_PAD)
    ax6.set_ylabel("TTFT p95 (ms)", color=COLORS["blue"])
    ax6.tick_params(axis="y", labelcolor=COLORS["blue"])
    ax6b = ax6.twinx()
    ax6b.plot(
        temporal["progress_percent"],
        temporal["tpot_p95_ms"],
        "s-",
        color=COLORS["orange"],
        linewidth=2,
        markersize=5,
        label="TPOT p95",
    )
    ax6b.set_ylabel("TPOT p95 (ms)", color=COLORS["orange"])
    ax6b.tick_params(axis="y", labelcolor=COLORS["orange"])
    style_axes(ax6)
    add_panel_subtitle(
        ax6,
        "Faster settling and lower tails are better; early spikes imply startup transients, persistent elevation implies steady-state stress.",
        width=64,
    )
    plot_specs.append(
        {
            "slug": "temporal_stability",
            "title": "Temporal Stability",
            "axes": [ax6, ax6b],
        }
    )

    # (h) Request throughput
    ax7 = plt.subplot(6, 4, 8)
    ax7.plot(
        concurrency,
        [row["raw_success_rps"] for row in rows],
        "o-",
        color=COLORS["blue"],
        linewidth=2,
        markersize=5,
    )
    ax7.set_xscale("log")
    ax7.set_title("(h) Request Throughput", fontweight="bold", pad=PANEL_TITLE_PAD)
    ax7.set_ylabel("Requests / sec")
    ax7.set_xticks(concurrency)
    ax7.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    style_axes(ax7)
    add_panel_subtitle(
        ax7,
        "Higher is better for service capacity, but it must be read together with latency and completion integrity.",
        width=64,
    )
    plot_specs.append(
        {"slug": "request_throughput", "title": "Request Throughput", "axes": [ax7]}
    )

    # (i) Delivered token throughput
    ax8 = plt.subplot(6, 4, 9)
    ax8.plot(
        concurrency,
        [row["successful_prompt_toksps"] / 1000.0 for row in rows],
        "o-",
        color=COLORS["green"],
        linewidth=2,
        markersize=5,
        label="Prompt",
    )
    ax8.plot(
        concurrency,
        [row["successful_output_toksps"] / 1000.0 for row in rows],
        "s-",
        color=COLORS["orange"],
        linewidth=2,
        markersize=5,
        label="Output",
    )
    ax8.plot(
        concurrency,
        [row["successful_total_toksps"] / 1000.0 for row in rows],
        "^-",
        color=COLORS["blue"],
        linewidth=2,
        markersize=5,
        label="Total",
    )
    ax8.set_xscale("log")
    ax8.set_title(
        "(i) Delivered Token Throughput", fontweight="bold", pad=PANEL_TITLE_PAD
    )
    ax8.set_ylabel("k tok/s")
    ax8.set_xticks(concurrency)
    ax8.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    style_axes(ax8)
    ax8.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=8, loc="best")
    add_panel_subtitle(
        ax8,
        "Higher is better; comparing prompt and output tok/s helps separate prefill-side and decode-side capacity.",
        width=64,
    )
    plot_specs.append(
        {
            "slug": "delivered_token_throughput",
            "title": "Delivered Token Throughput",
            "axes": [ax8],
        }
    )

    # (j) Scaling efficiency
    ax9 = plt.subplot(6, 4, 10)
    baseline = min(rows, key=lambda row: row["concurrency"])
    ax9.plot(
        concurrency,
        [
            row["raw_success_rps"] / (baseline["raw_success_rps"] * row["concurrency"])
            for row in rows
        ],
        "o-",
        color=COLORS["gray"],
        linewidth=2,
        markersize=5,
        label="Req/s",
    )
    ax9.plot(
        concurrency,
        [
            row["successful_output_toksps"]
            / (baseline["successful_output_toksps"] * row["concurrency"])
            for row in rows
        ],
        "s-",
        color=COLORS["orange"],
        linewidth=2,
        markersize=5,
        label="Output tok/s",
    )
    ax9.axhline(1.0, color=COLORS["gray"], linestyle="--", linewidth=1.0)
    ax9.set_xscale("log")
    ax9.set_title("(j) Scaling Efficiency", fontweight="bold", pad=PANEL_TITLE_PAD)
    ax9.set_ylabel("Relative to Linear")
    ax9.set_xticks(concurrency)
    ax9.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    style_axes(ax9)
    ax9.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=8, loc="best")
    add_panel_subtitle(
        ax9,
        "Values closer to 1 are better; falling efficiency means extra concurrency is no longer converting cleanly into useful work.",
        width=64,
    )
    plot_specs.append(
        {"slug": "scaling_efficiency", "title": "Scaling Efficiency", "axes": [ax9]}
    )

    # (k) Per-request decode rate
    ax10 = plt.subplot(6, 4, 11)
    ax10.plot(
        concurrency,
        [row["output_toks_per_second_p50"] for row in rows],
        "o-",
        color=COLORS["blue"],
        linewidth=2,
        markersize=5,
        label="p50",
    )
    ax10.plot(
        concurrency,
        [row["output_toks_per_second_p05"] for row in rows],
        "s-",
        color=COLORS["red"],
        linewidth=2,
        markersize=5,
        label="p05",
    )
    ax10.set_xscale("log")
    ax10.set_title(
        "(k) Per-Request Decode Rate", fontweight="bold", pad=PANEL_TITLE_PAD
    )
    ax10.set_ylabel("Output tok/s")
    ax10.set_xticks(concurrency)
    ax10.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    style_axes(ax10)
    ax10.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=8, loc="best")
    add_panel_subtitle(
        ax10,
        "Higher is better; the lower tail highlights user-visible slow streaming even when aggregate throughput still looks healthy.",
        width=64,
    )
    plot_specs.append(
        {
            "slug": "per_request_decode_rate",
            "title": "Per-Request Decode Rate",
            "axes": [ax10],
        }
    )

    # (l) Max goodput sweep
    ax11 = plt.subplot(6, 4, 12)
    goodput_im = ax11.imshow(max_goodput, origin="lower", aspect="auto", cmap="Blues")
    ax11.set_title(
        "(l) Max Goodput by Candidate SLO", fontweight="bold", pad=PANEL_TITLE_PAD
    )
    ax11.set_xticks(np.arange(len(itl_thresholds)))
    ax11.set_xticklabels(
        [f"{value:.0f}" for value in itl_thresholds], rotation=45, ha="right"
    )
    ax11.set_yticks(np.arange(len(ttft_thresholds)))
    ax11.set_yticklabels([f"{value:.0f}" for value in ttft_thresholds])
    ax11.set_xlabel("ITL threshold (ms)")
    ax11.set_ylabel("TTFT threshold (ms)")
    ax11.grid(False)
    cbar11 = fig.colorbar(goodput_im, ax=ax11, fraction=0.046, pad=0.04)
    add_panel_subtitle(
        ax11,
        "Brighter is better; this shows the best goodput achievable for each candidate TTFT and ITL latency contract.",
        width=64,
    )
    plot_specs.append(
        {
            "slug": "max_goodput_sweep",
            "title": "Max Goodput by Candidate SLO",
            "axes": [ax11, cbar11.ax],
        }
    )

    # (m) Best concurrency by candidate SLO
    ax12 = plt.subplot(6, 4, 13)
    unique_concurrency = sorted({int(row["concurrency"]) for row in rows})
    discrete_colors = [
        COLORS["blue"],
        COLORS["orange"],
        COLORS["green"],
        COLORS["red"],
        COLORS["purple"],
        COLORS["brown"],
        COLORS["pink"],
    ][: len(unique_concurrency)]
    cmap = ListedColormap(discrete_colors)
    boundaries = [unique_concurrency[0] - 0.5]
    boundaries.extend(value + 0.5 for value in unique_concurrency)
    norm = BoundaryNorm(boundaries, cmap.N)
    concurrency_im = ax12.imshow(
        best_concurrency, origin="lower", aspect="auto", cmap=cmap, norm=norm
    )
    ax12.set_title(
        "(m) Best Concurrency by SLO", fontweight="bold", pad=PANEL_TITLE_PAD
    )
    ax12.set_xticks(np.arange(len(itl_thresholds)))
    ax12.set_xticklabels(
        [f"{value:.0f}" for value in itl_thresholds], rotation=45, ha="right"
    )
    ax12.set_yticks(np.arange(len(ttft_thresholds)))
    ax12.set_yticklabels([f"{value:.0f}" for value in ttft_thresholds])
    ax12.set_xlabel("ITL threshold (ms)")
    ax12.set_ylabel("TTFT threshold (ms)")
    ax12.grid(False)
    cbar12 = fig.colorbar(
        concurrency_im, ax=ax12, fraction=0.046, pad=0.04, ticks=unique_concurrency
    )
    add_panel_subtitle(
        ax12,
        "Each cell shows which tested concurrency maximizes goodput for that candidate SLO; it turns a latency target into an operating point.",
        width=64,
    )
    plot_specs.append(
        {
            "slug": "best_concurrency_sweep",
            "title": "Best Concurrency by SLO",
            "axes": [ax12, cbar12.ax],
        }
    )

    # (n) Scheduler queue wait
    ax13 = plt.subplot(6, 4, 14)
    ax13.plot(
        concurrency,
        [row["queue_wait_p50_s"] for row in rows],
        "o-",
        color=COLORS["blue"],
        linewidth=2,
        markersize=5,
        label="p50",
    )
    ax13.plot(
        concurrency,
        [row["queue_wait_p95_s"] for row in rows],
        "s-",
        color=COLORS["red"],
        linewidth=2,
        markersize=5,
        label="p95",
    )
    ax13.set_xscale("log")
    ax13.set_yscale("log")
    ax13.set_title("(n) Scheduler Queue Wait", fontweight="bold", pad=PANEL_TITLE_PAD)
    ax13.set_ylabel("Seconds")
    ax13.set_xticks(concurrency)
    ax13.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    style_axes(ax13)
    ax13.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=8, loc="best")
    add_panel_subtitle(
        ax13,
        "Lower is better; this isolates waiting before request execution and can reveal admission or scheduler pressure before model execution degrades.",
        width=64,
    )
    plot_specs.append(
        {
            "slug": "scheduler_queue_wait",
            "title": "Scheduler Queue Wait",
            "axes": [ax13],
        }
    )

    # (o) Effective prefill/decode throughput
    ax14 = plt.subplot(6, 4, 15)
    ax14.plot(
        concurrency,
        [row["effective_prefill_toksps_p50"] / 1000.0 for row in rows],
        "o-",
        color=COLORS["green"],
        linewidth=2,
        markersize=5,
        label="Prefill tok/s p50",
    )
    ax14.plot(
        concurrency,
        [row["effective_decode_toksps_p50"] / 1000.0 for row in rows],
        "s-",
        color=COLORS["orange"],
        linewidth=2,
        markersize=5,
        label="Decode tok/s p50",
    )
    ax14.set_xscale("log")
    ax14.set_title(
        "(o) Effective Prefill/Decode Rate", fontweight="bold", pad=PANEL_TITLE_PAD
    )
    ax14.set_ylabel("k tok/s")
    ax14.set_xticks(concurrency)
    ax14.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    style_axes(ax14)
    ax14.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=8, loc="best")
    add_panel_subtitle(
        ax14,
        "Higher is better; comparing these curves shows whether prefill or decode is the first stage to lose efficiency under load.",
        width=64,
    )
    plot_specs.append(
        {
            "slug": "effective_prefill_decode_rate",
            "title": "Effective Prefill/Decode Rate",
            "axes": [ax14],
        }
    )

    # (p) Useful vs wasted output throughput
    ax15 = plt.subplot(6, 4, 16)
    useful = np.asarray([row["successful_output_toksps"] / 1000.0 for row in rows])
    wasted = np.asarray([row["incomplete_output_toksps"] / 1000.0 for row in rows])
    ax15.bar(x_pos, useful, color=COLORS["blue"], label="Useful output tok/s")
    ax15.bar(
        x_pos,
        wasted,
        bottom=useful,
        color=COLORS["orange"],
        label="Wasted partial output tok/s",
    )
    ax15.set_title(
        "(p) Useful vs Wasted Output Work", fontweight="bold", pad=PANEL_TITLE_PAD
    )
    ax15.set_ylabel("k output tok/s")
    ax15.set_xticks(x_pos)
    ax15.set_xticklabels([str(int(value)) for value in concurrency])
    style_axes(ax15, axis="y")
    ax15.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=8, loc="best")
    add_panel_subtitle(
        ax15,
        "Blue is useful completed generation and orange is work spent on cancelled requests; higher blue share and lower orange waste are better.",
        width=64,
    )
    plot_specs.append(
        {
            "slug": "useful_vs_wasted_output",
            "title": "Useful vs Wasted Output Work",
            "axes": [ax15],
        }
    )

    # (q) Incomplete request progress
    ax16 = plt.subplot(6, 4, 17)
    ax16.plot(
        concurrency,
        [row["incomplete_progress_mean"] * 100.0 for row in rows],
        "o-",
        color=COLORS["blue"],
        linewidth=2,
        markersize=5,
        label="Mean",
    )
    ax16.plot(
        concurrency,
        [row["incomplete_progress_p50"] * 100.0 for row in rows],
        "s-",
        color=COLORS["orange"],
        linewidth=2,
        markersize=5,
        label="p50",
    )
    ax16.set_xscale("log")
    ax16.set_title(
        "(q) Cancelled Request Progress", fontweight="bold", pad=PANEL_TITLE_PAD
    )
    ax16.set_ylabel("% of Target Output")
    ax16.set_xticks(concurrency)
    ax16.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    style_axes(ax16)
    ax16.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=8, loc="best")
    add_panel_subtitle(
        ax16,
        "Lower is better from a waste perspective; high values mean cancellations happen late after substantial decode work has already been spent.",
        width=64,
    )
    plot_specs.append(
        {
            "slug": "cancelled_request_progress",
            "title": "Cancelled Request Progress",
            "axes": [ax16],
        }
    )

    # (r) Token throughput per GPU
    ax17 = plt.subplot(6, 4, 18)
    per_gpu_x = [row["concurrency_per_gpu"] for row in rows]
    per_gpu_labels = [f"{value:g}" for value in per_gpu_x]
    ax17.plot(
        per_gpu_x,
        [row["successful_prompt_toksps_per_gpu"] / 1000.0 for row in rows],
        "o-",
        color=COLORS["green"],
        linewidth=2,
        markersize=5,
        label="Prompt",
    )
    ax17.plot(
        per_gpu_x,
        [row["successful_output_toksps_per_gpu"] / 1000.0 for row in rows],
        "s-",
        color=COLORS["orange"],
        linewidth=2,
        markersize=5,
        label="Output",
    )
    ax17.plot(
        per_gpu_x,
        [row["successful_total_toksps_per_gpu"] / 1000.0 for row in rows],
        "^-",
        color=COLORS["blue"],
        linewidth=2,
        markersize=5,
        label="Total",
    )
    ax17.set_xscale("log")
    ax17.set_title(
        "(r) Token Throughput per GPU", fontweight="bold", pad=PANEL_TITLE_PAD
    )
    ax17.set_xlabel("Concurrency / GPU")
    ax17.set_ylabel("k tok/s / GPU")
    ax17.set_xticks(per_gpu_x)
    ax17.set_xticklabels(per_gpu_labels, rotation=35, ha="right")
    style_axes(ax17)
    ax17.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=8, loc="best")
    add_panel_subtitle(
        ax17,
        "Higher is better; the x-axis normalizes offered load by accelerator count so throughput becomes comparable across cluster shapes.",
        width=64,
    )
    plot_specs.append(
        {
            "slug": "token_throughput_per_gpu",
            "title": "Token Throughput per GPU",
            "axes": [ax17],
        }
    )

    # (s) Throughput efficiency per GPU
    ax18 = plt.subplot(6, 4, 19)
    interactivity = [
        row["successful_output_toksps"] / row["concurrency"] for row in rows
    ]
    throughput_per_gpu = [row["successful_output_toksps_per_gpu"] for row in rows]
    ax18.plot(
        interactivity,
        throughput_per_gpu,
        "--",
        color=COLORS["gray"],
        linewidth=1.4,
        alpha=0.8,
    )
    ax18.scatter(
        interactivity,
        throughput_per_gpu,
        s=42,
        color=COLORS["orange"],
        edgecolors="black",
        linewidth=0.5,
        zorder=3,
    )
    for row, x_value, y_value in zip(rows, interactivity, throughput_per_gpu):
        ax18.annotate(
            str(int(row["concurrency"])),
            (x_value, y_value),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
        )
    ax18.set_title(
        "(s) Throughput Efficiency per GPU", fontweight="bold", pad=PANEL_TITLE_PAD
    )
    ax18.set_xlabel("Interactivity (output tok/s/concurrency)")
    ax18.set_ylabel("Output throughput (tok/s/GPU)")
    style_axes(ax18)
    add_panel_subtitle(
        ax18,
        "Up-right is better; x captures interactivity as output tok/s per concurrency while y captures delivered output tok/s per GPU.",
        width=64,
    )
    plot_specs.append(
        {
            "slug": "throughput_efficiency_per_gpu",
            "title": "Throughput Efficiency per GPU",
            "axes": [ax18],
        }
    )

    # (t) TTFT vs actual concurrency
    ax19 = plt.subplot(6, 4, 20)
    actual_x = [entry["x"] for entry in actual_concurrency_summary]
    actual_labels = [entry["label"] for entry in actual_concurrency_summary]
    ax19.plot(
        actual_x,
        [entry["ttft_p50_ms"] for entry in actual_concurrency_summary],
        "o-",
        color=COLORS["green"],
        linewidth=2,
        markersize=5,
        label="p50",
    )
    ax19.plot(
        actual_x,
        [entry["ttft_p95_ms"] for entry in actual_concurrency_summary],
        "s-",
        color=COLORS["orange"],
        linewidth=2,
        markersize=5,
        label="p95",
    )
    ax19.plot(
        actual_x,
        [entry["ttft_p99_ms"] for entry in actual_concurrency_summary],
        "^-",
        color=COLORS["red"],
        linewidth=2,
        markersize=5,
        label="p99",
    )
    ax19.set_xscale("log")
    ax19.set_yscale("log")
    ax19.set_title(
        "(t) TTFT vs Actual Concurrency", fontweight="bold", pad=PANEL_TITLE_PAD
    )
    ax19.set_xlabel("Observed in-flight requests")
    ax19.set_ylabel("TTFT (ms)")
    ax19.set_xticks(actual_x)
    ax19.set_xticklabels(actual_labels, rotation=35, ha="right")
    style_axes(ax19)
    ax19.legend(frameon=True, fancybox=False, edgecolor="black", fontsize=8, loc="best")
    add_panel_subtitle(
        ax19,
        "Lower and flatter are better; actual concurrency is derived from overlapping request lifetimes, which exposes saturation more directly than target load.",
        width=64,
    )
    plot_specs.append(
        {
            "slug": "ttft_vs_actual_concurrency",
            "title": "TTFT vs Actual Concurrency",
            "axes": [ax19],
        }
    )

    # (u) Delay decomposition
    ax20 = plt.subplot(6, 4, 21)
    ax20.plot(
        concurrency,
        [row["queue_wait_p50_s"] for row in rows],
        "o-",
        color=COLORS["blue"],
        linewidth=2,
        markersize=5,
        label="Queue wait p50",
    )
    ax20.plot(
        concurrency,
        [row["ttft_p50_ms"] / 1000.0 for row in rows],
        "o-",
        color=COLORS["green"],
        linewidth=2,
        markersize=5,
        label="TTFT p50",
    )
    ax20.plot(
        concurrency,
        [row["decode_duration_p50_s"] for row in rows],
        "o-",
        color=COLORS["orange"],
        linewidth=2,
        markersize=5,
        label="Decode p50",
    )
    ax20.plot(
        concurrency,
        [row["queue_wait_p95_s"] for row in rows],
        "s--",
        color=COLORS["blue"],
        linewidth=2,
        markersize=5,
        label="Queue wait p95",
    )
    ax20.plot(
        concurrency,
        [row["ttft_p95_ms"] / 1000.0 for row in rows],
        "s--",
        color=COLORS["green"],
        linewidth=2,
        markersize=5,
        label="TTFT p95",
    )
    ax20.plot(
        concurrency,
        [row["decode_duration_p95_s"] for row in rows],
        "s--",
        color=COLORS["orange"],
        linewidth=2,
        markersize=5,
        label="Decode p95",
    )
    ax20.set_xscale("log")
    ax20.set_yscale("log")
    ax20.set_title("(u) Delay Decomposition", fontweight="bold", pad=PANEL_TITLE_PAD)
    ax20.set_ylabel("Seconds")
    ax20.set_xticks(concurrency)
    ax20.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    style_axes(ax20)
    ax20.legend(
        frameon=True,
        fancybox=False,
        edgecolor="black",
        fontsize=7.3,
        loc="best",
        ncol=2,
    )
    add_panel_subtitle(
        ax20,
        "Lower is better; the first component to rise identifies whether scheduler wait, first-token latency, or decode duration is the limiting stage.",
        width=64,
    )
    plot_specs.append(
        {"slug": "delay_decomposition", "title": "Delay Decomposition", "axes": [ax20]}
    )

    for axis in fig.axes:
        axis.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0.02, 1, 0.99])
    return fig, plot_specs


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
    overview = create_overview_figure(rows, strict_slo, relaxed_slo, args.time_bins)
    throughput = create_throughput_figure(rows)
    slo_sweep = create_slo_sweep_figure(
        rows,
        ttft_thresholds,
        itl_thresholds,
        strict_slo,
        relaxed_slo,
    )
    all_plots, plot_specs = create_all_plots_figure(
        rows,
        ttft_thresholds,
        itl_thresholds,
        strict_slo,
        relaxed_slo,
        args.time_bins,
    )
    plot_cells = save_plot_cells(
        rows,
        ttft_thresholds,
        itl_thresholds,
        strict_slo,
        relaxed_slo,
        args.time_bins,
        args.output_prefix,
    )

    save_figure(overview, f"{args.output_prefix}_overview")
    save_figure(throughput, f"{args.output_prefix}_throughput")
    save_figure(slo_sweep, f"{args.output_prefix}_slo_sweep")
    save_pdf_only(all_plots, f"{args.output_prefix}_all_plots.pdf")
    save_html_report(args.output_prefix, plot_cells)
    save_pdf_report(
        args.output_prefix,
        [
            ("All-plots compendium", all_plots),
            ("Exploratory overview", overview),
            ("Throughput diagnostics", throughput),
            ("Candidate-SLO threshold sweep", slo_sweep),
        ],
    )
    if args.show:
        plt.show()
    else:
        plt.close(all_plots)
        plt.close(overview)
        plt.close(throughput)
        plt.close(slo_sweep)

    print("\n✓ Figure created successfully!")
    print("\nGenerated plots:")
    print("  1. Completion breakdown vs concurrency")
    print("  2. Tail amplification vs concurrency")
    print("  3. TTFT/TPOT coupling vs concurrency")
    print("  4. TTFT tail distribution (CCDF)")
    print("  5. TTFT box distribution")
    print("  6. Throughput/latency Pareto frontier")
    print("  7. Temporal stability for the highest-concurrency run")
    print("  8. Request throughput vs concurrency")
    print("  9. Delivered token throughput vs concurrency")
    print("  10. Throughput scaling efficiency")
    print("  11. Per-request decode rate")
    print("  12. Max-goodput candidate-SLO threshold sweep")
    print("  13. Best-concurrency candidate-SLO threshold sweep")
    print("  14. Token throughput per GPU")
    print("  15. Throughput efficiency per GPU")
    print("  16. TTFT vs actual concurrency")
    print("  17. Delay decomposition")
    print("  18. Single-page all-plots compendium PDF")


if __name__ == "__main__":
    main()
