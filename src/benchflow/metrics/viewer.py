from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import mlflow
import plotly.graph_objects as go
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from plotly.offline import get_plotlyjs

from ..contracts import ValidationError
from ..plotting import REPORT_COLOR_PALETTE
from ..ui import detail, step, success

DEFAULT_METRICS_VIEWER_PORT = 8765
_QUERY_RATE_INTERVAL = "5m"
_CACHE_ROOT = Path("/tmp/benchflow-metrics-viewer")
_LEGEND_TEMPLATE_RE = re.compile(r"\{\{([^{}]+)\}\}")
_DASHBOARD_ASSET = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "bootstrap"
    / "operators"
    / "grafana"
    / "benchflow-live-dashboard.json"
)
_PALETTE = list(REPORT_COLOR_PALETTE)
_METRICS_CACHE_FILES = (
    "archive_index.json",
    "metrics_summary.json",
    "resolved_queries.json",
)


@dataclass(slots=True)
class ViewerRunData:
    run: Any
    metrics_dir: Path
    archive_index: dict[str, Any]
    metrics_summary: dict[str, Any]
    query_map: dict[str, str]
    params: dict[str, str]
    tags: dict[str, str]
    namespace: str
    release_name: str
    benchmark_start_time: str
    benchmark_end_time: str
    start_timestamp: int
    label: str
    color: str


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_iso8601(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _normalize_query(query: str) -> str:
    normalized = str(query).replace("$__rate_interval", _QUERY_RATE_INTERVAL)
    normalized = normalized.replace("\n", "")
    return re.sub(r"\s+", "", normalized)


def _hex_to_rgba(color: str, alpha: float) -> str:
    raw = str(color).lstrip("#")
    if len(raw) != 6:
        return f"rgba(53, 80, 112, {alpha})"
    red = int(raw[0:2], 16)
    green = int(raw[2:4], 16)
    blue = int(raw[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def _metrics_cache_ready(metrics_dir: Path) -> bool:
    return metrics_dir.exists() and all(
        (metrics_dir / name).exists() for name in _METRICS_CACHE_FILES
    )


def _cache_dir_for_run(mlflow_run_id: str) -> Path:
    return _CACHE_ROOT / "runs" / mlflow_run_id


def _download_metrics_artifacts(
    *,
    mlflow_run_id: str,
    mlflow_tracking_uri: str | None = None,
) -> tuple[Any, Path]:
    tracking_uri = str(mlflow_tracking_uri or "").strip()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(mlflow_run_id)
    cache_dir = _cache_dir_for_run(mlflow_run_id)
    metrics_dir = cache_dir / "metrics"

    if _metrics_cache_ready(metrics_dir):
        detail(f"Using cached metrics artifacts for MLflow run {mlflow_run_id}")
        return run, metrics_dir

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    step(f"Downloading MLflow metrics artifacts for run {mlflow_run_id}")
    repo = get_artifact_repository(run.info.artifact_uri)
    try:
        downloaded_path = repo.download_artifacts("metrics", dst_path=str(cache_dir))
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(
            f"failed to download metrics artifacts for MLflow run {mlflow_run_id}: {exc}"
        ) from exc

    downloaded_dir = Path(downloaded_path)
    if not _metrics_cache_ready(downloaded_dir):
        raise ValidationError(
            f"MLflow run {mlflow_run_id} does not contain a complete BenchFlow metrics artifact tree"
        )
    return run, downloaded_dir


def _load_dashboard_spec() -> dict[str, Any]:
    return _load_json(_DASHBOARD_ASSET)


def _group_dashboard_sections(dashboard: dict[str, Any]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for panel in dashboard.get("panels", []):
        panel_type = str(panel.get("type") or "").strip()
        if panel_type == "row":
            if current and current["panels"]:
                sections.append(current)
            current = {"title": panel.get("title") or "Metrics", "panels": []}
            continue
        if panel_type == "text":
            continue
        if current is None:
            current = {"title": "Metrics", "panels": []}
        current["panels"].append(panel)

    if current and current["panels"]:
        sections.append(current)
    return sections


def _build_query_map(metrics_summary: dict[str, Any]) -> dict[str, str]:
    query_map: dict[str, str] = {}
    for metric_name, metadata in (metrics_summary.get("queries") or {}).items():
        query = str((metadata or {}).get("query") or "").strip()
        if query:
            query_map[_normalize_query(query)] = metric_name
    return query_map


def _resolve_target_metric(
    target: dict[str, Any],
    *,
    namespace: str,
    release_name: str,
    query_map: dict[str, str],
) -> str | None:
    expr = str(target.get("expr") or "").strip()
    if not expr:
        return None
    expr = expr.replace("$namespace", namespace).replace("$release", release_name)
    return query_map.get(_normalize_query(expr))


def _load_metric_series(metrics_dir: Path, metric_name: str) -> list[dict[str, Any]]:
    metric_path = metrics_dir / "raw" / f"{metric_name}.json"
    if not metric_path.exists():
        return []
    payload = _load_json(metric_path)
    return payload if isinstance(payload, list) else []


def _panel_unit(panel: dict[str, Any]) -> str:
    field_defaults = (panel.get("fieldConfig") or {}).get("defaults") or {}
    return str(field_defaults.get("unit") or "").strip()


def _panel_span(panel: dict[str, Any]) -> int:
    width = int(((panel.get("gridPos") or {}).get("w") or 12))
    return max(3, min(12, max(1, width // 2)))


def _clean_legend_text(value: str) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    text = re.sub(r"\s*-\s*$", "", text)
    text = re.sub(r"^\s*-\s*", "", text)
    text = re.sub(r"\s+-\s+-\s+", " - ", text)
    return text or "series"


def _render_legend_template(template: str, labels: dict[str, Any]) -> str:
    raw = str(template or "").strip()
    if not raw:
        return ""

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        value = labels.get(key)
        return "" if value in {None, ""} else str(value)

    return _clean_legend_text(_LEGEND_TEMPLATE_RE.sub(_replace, raw))


def _group_points_by_series(points: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for point in points:
        series_name = str(point.get("series") or "series")
        bucket = grouped.setdefault(
            series_name,
            {
                "labels": dict(point.get("labels") or {}),
                "points": [],
            },
        )
        bucket["points"].append(point)
        if not bucket["labels"]:
            bucket["labels"] = dict(point.get("labels") or {})
    for bucket in grouped.values():
        bucket["points"].sort(key=lambda item: item.get("timestamp", 0))
    return grouped


def _format_bytes(value: float) -> str:
    suffixes = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    magnitude = float(value)
    for suffix in suffixes:
        if abs(magnitude) < 1024.0 or suffix == suffixes[-1]:
            return f"{magnitude:.1f} {suffix}"
        magnitude /= 1024.0
    return f"{magnitude:.1f} PiB"


def _format_bytes_per_second(value: float) -> str:
    return f"{_format_bytes(value)}/s"


def _format_seconds(value: float) -> str:
    if value < 1.0:
        return f"{value * 1000.0:.1f} ms"
    if value < 60.0:
        return f"{value:.2f} s"
    minutes = int(value // 60)
    seconds = value % 60
    return f"{minutes}m {seconds:.0f}s"


def _format_value(value: float | None, unit: str | None) -> str:
    if value is None:
        return "N/A"
    normalized_unit = str(unit or "").strip()
    if normalized_unit in {"percent", "percentunit"}:
        scale = 100.0 if normalized_unit == "percentunit" else 1.0
        return f"{value * scale:.1f}%"
    if normalized_unit == "bytes":
        return _format_bytes(value)
    if normalized_unit == "Bps":
        return _format_bytes_per_second(value)
    if normalized_unit in {"ops", "tps", "reqps"}:
        return f"{value:,.2f}"
    if normalized_unit == "s":
        return _format_seconds(value)
    return f"{value:,.3f}"


def _metric_summary_value(
    panel: dict[str, Any], metric_summary: dict[str, Any]
) -> float | None:
    reducer = (
        (((panel.get("options") or {}).get("reduceOptions") or {}).get("calcs") or [])
        or ["lastNotNull"]
    )[0]
    return (
        metric_summary.get("avg")
        if reducer == "mean"
        else metric_summary.get("latest_avg")
    )


def _compose_run_label(
    *,
    run_id: str,
    params: dict[str, str],
    tags: dict[str, str],
    release_name: str,
) -> str:
    version = str(params.get("version") or "").strip()
    deployment = str(tags.get("deployment_profile") or "").strip()
    benchmark = str(tags.get("benchmark_profile") or "").strip()
    model = str(params.get("model") or "").strip()
    model_short = model.split("/")[-1] if model else ""

    for candidate in (
        " · ".join(part for part in (version, deployment) if part),
        version,
        " · ".join(part for part in (deployment, benchmark) if part),
        release_name,
        model_short,
    ):
        label = str(candidate).strip()
        if label and label.lower() != "unknown":
            return label
    return run_id[:8]


def _ensure_unique_labels(run_data: list[ViewerRunData]) -> list[ViewerRunData]:
    counts: dict[str, int] = {}
    for item in run_data:
        counts[item.label] = counts.get(item.label, 0) + 1

    result: list[ViewerRunData] = []
    for item in run_data:
        label = item.label
        if counts[label] > 1:
            label = f"{label} · {item.run.info.run_id[:8]}"
        result.append(
            ViewerRunData(
                run=item.run,
                metrics_dir=item.metrics_dir,
                archive_index=item.archive_index,
                metrics_summary=item.metrics_summary,
                query_map=item.query_map,
                params=item.params,
                tags=item.tags,
                namespace=item.namespace,
                release_name=item.release_name,
                benchmark_start_time=item.benchmark_start_time,
                benchmark_end_time=item.benchmark_end_time,
                start_timestamp=item.start_timestamp,
                label=label,
                color=item.color,
            )
        )
    return result


def _load_viewer_run_data(
    *,
    mlflow_run_id: str,
    mlflow_tracking_uri: str | None,
    color: str,
) -> ViewerRunData:
    run, metrics_dir = _download_metrics_artifacts(
        mlflow_run_id=mlflow_run_id,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )
    archive_index = _load_json(metrics_dir / "archive_index.json")
    metrics_summary = _load_json(metrics_dir / "metrics_summary.json")
    params = dict(run.data.params or {})
    tags = dict(run.data.tags or {})
    namespace = str(
        archive_index.get("namespace") or params.get("namespace") or "unknown"
    )
    release_name = str(
        archive_index.get("release_name")
        or params.get("release")
        or params.get("target")
        or "unknown"
    )
    start_time = str(archive_index.get("benchmark_start_time") or "")
    end_time = str(archive_index.get("benchmark_end_time") or "")
    start_timestamp = int(_parse_iso8601(start_time).timestamp()) if start_time else 0
    return ViewerRunData(
        run=run,
        metrics_dir=metrics_dir,
        archive_index=archive_index,
        metrics_summary=metrics_summary,
        query_map=_build_query_map(metrics_summary),
        params=params,
        tags=tags,
        namespace=namespace,
        release_name=release_name,
        benchmark_start_time=start_time,
        benchmark_end_time=end_time,
        start_timestamp=start_timestamp,
        label=_compose_run_label(
            run_id=run.info.run_id,
            params=params,
            tags=tags,
            release_name=release_name,
        ),
        color=color,
    )


def _trace_x_values(
    points: list[dict[str, Any]],
    *,
    start_timestamp: int,
    compare_mode: bool,
) -> tuple[list[Any], str]:
    if compare_mode:
        return [
            max(0.0, (point["timestamp"] - start_timestamp) / 60.0) for point in points
        ], "Elapsed benchmark time (minutes)"
    return [point["time"] for point in points], "Wall clock time (UTC)"


def _build_trace_name(
    *,
    run_label: str,
    metric_name: str,
    series_name: str,
    labels: dict[str, Any],
    legend_format: str,
    compare_mode: bool,
) -> str:
    rendered = _render_legend_template(legend_format, labels)
    base = rendered or (
        series_name if series_name and series_name != metric_name else metric_name
    )
    if compare_mode:
        return _clean_legend_text(f"{run_label} · {base}")
    return base


def _build_stat_panel_html(
    panel: dict[str, Any], *, run_data: list[ViewerRunData]
) -> str:
    unit = _panel_unit(panel)
    title = str(panel.get("title") or "Metric")
    if len(run_data) == 1:
        item = run_data[0]
        target = ((panel.get("targets") or [{}])[:1] or [{}])[0]
        metric_name = _resolve_target_metric(
            target,
            namespace=item.namespace,
            release_name=item.release_name,
            query_map=item.query_map,
        )
        metric_summary = (
            ((item.metrics_summary.get("queries") or {}).get(metric_name or "") or {})
            if metric_name
            else {}
        )
        value = _metric_summary_value(panel, metric_summary)
        subtitle = (
            " · ".join(
                part
                for part in (
                    f"{metric_summary.get('series_count')} series"
                    if isinstance(metric_summary.get("series_count"), int)
                    else "",
                    f"{metric_summary.get('sample_count')} samples"
                    if isinstance(metric_summary.get("sample_count"), int)
                    else "",
                )
                if part
            )
            or "No samples available"
        )
        return f"""
        <section class="panel stat-panel span-{_panel_span(panel)}">
          <header class="panel-header">
            <h3>{html.escape(title)}</h3>
          </header>
          <div class="stat-value">{html.escape(_format_value(value, unit))}</div>
          <div class="stat-subtitle">{html.escape(subtitle)}</div>
        </section>
        """

    rows_html: list[str] = []
    for item in run_data:
        target = ((panel.get("targets") or [{}])[:1] or [{}])[0]
        metric_name = _resolve_target_metric(
            target,
            namespace=item.namespace,
            release_name=item.release_name,
            query_map=item.query_map,
        )
        metric_summary = (
            ((item.metrics_summary.get("queries") or {}).get(metric_name or "") or {})
            if metric_name
            else {}
        )
        value = _metric_summary_value(panel, metric_summary)
        rows_html.append(
            f"""
            <div class="stat-compare-row">
              <div class="stat-compare-label">
                <span class="run-swatch" style="background:{html.escape(item.color)}"></span>
                <span>{html.escape(item.label)}</span>
              </div>
              <div class="stat-compare-value">{html.escape(_format_value(value, unit))}</div>
            </div>
            """
        )

    return f"""
    <section class="panel stat-panel stat-panel-compare span-{_panel_span(panel)}">
      <header class="panel-header">
        <h3>{html.escape(title)}</h3>
      </header>
      <div class="stat-compare-grid">
        {"".join(rows_html)}
      </div>
    </section>
    """


def _build_timeseries_figure(
    panel: dict[str, Any], *, run_data: list[ViewerRunData]
) -> go.Figure:
    compare_mode = len(run_data) > 1
    figure = go.Figure()
    x_axis_title = "Wall clock time (UTC)"
    trace_index = 0
    normalized_title = str(panel.get("title") or "").strip().lower()
    unit = _panel_unit(panel)
    is_memory_panel = unit == "bytes" or "memory" in normalized_title

    for item in run_data:
        for target in panel.get("targets", []):
            metric_name = _resolve_target_metric(
                target,
                namespace=item.namespace,
                release_name=item.release_name,
                query_map=item.query_map,
            )
            if not metric_name:
                continue
            grouped_points = _group_points_by_series(
                _load_metric_series(item.metrics_dir, metric_name)
            )
            for series_name, bucket in grouped_points.items():
                points = bucket["points"]
                if not points:
                    continue
                x_values, x_axis_title = _trace_x_values(
                    points,
                    start_timestamp=item.start_timestamp,
                    compare_mode=compare_mode,
                )
                line_color = _PALETTE[trace_index % len(_PALETTE)]
                fill_alpha = 0.12 if not compare_mode else 0.05
                figure.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=[point["value"] for point in points],
                        mode="lines",
                        name=_build_trace_name(
                            run_label=item.label,
                            metric_name=metric_name,
                            series_name=series_name,
                            labels=dict(bucket.get("labels") or {}),
                            legend_format=str(target.get("legendFormat") or ""),
                            compare_mode=compare_mode,
                        ),
                        line={"width": 2.35, "color": line_color},
                        fill="tozeroy",
                        fillcolor=_hex_to_rgba(line_color, fill_alpha),
                    )
                )
                trace_index += 1

    legend_right_margin = 310
    fill_alpha = 0.008 if is_memory_panel else 0.022
    if compare_mode:
        fill_alpha /= 2

    for trace in figure.data:
        trace.fillcolor = _hex_to_rgba(str(trace.line.color), fill_alpha)

    figure.update_layout(
        template="simple_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": 68, "r": legend_right_margin, "t": 20, "b": 72},
        hovermode="x unified",
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 1,
            "xanchor": "left",
            "x": 1.02,
            "bgcolor": "white",
            "bordercolor": "black",
            "borderwidth": 1,
            "font": {"size": 10, "family": "Arial, sans-serif"},
            "title": {"text": "Configuration"},
        },
        hoverlabel={
            "bgcolor": "white",
            "bordercolor": "black",
            "font": {"family": "Arial, sans-serif", "size": 11},
        },
        font={"family": "Arial, sans-serif", "size": 11, "color": "#222222"},
        xaxis={
            "title": x_axis_title,
            "gridcolor": "lightgray",
            "gridwidth": 0.5,
            "linecolor": "black",
            "linewidth": 1,
            "mirror": True,
            "tickfont": {"size": 10, "color": "#333333"},
            "ticks": "outside",
            "tickcolor": "black",
            "title_standoff": 16,
            "zeroline": False,
        },
        yaxis={
            "title": unit,
            "gridcolor": "lightgray",
            "gridwidth": 0.5,
            "linecolor": "black",
            "linewidth": 1,
            "mirror": True,
            "tickfont": {"size": 10, "color": "#333333"},
            "ticks": "outside",
            "tickcolor": "black",
            "separatethousands": True,
            "zeroline": False,
        },
    )
    if compare_mode:
        figure.update_xaxes(ticksuffix=" min")
    if not figure.data:
        figure.add_annotation(
            text="No stored data for this panel",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 14, "color": "#6b7280"},
        )
        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False)
    return figure


def _build_timeseries_panel_html(
    panel: dict[str, Any], *, run_data: list[ViewerRunData]
) -> str:
    figure = _build_timeseries_figure(panel, run_data=run_data)
    chart_html = figure.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": False,
            "doubleClick": "reset",
        },
    )
    return f"""
    <section class="panel chart-panel span-{_panel_span(panel)}">
      <header class="panel-header">
        <h3>{html.escape(str(panel.get("title") or "Metric"))}</h3>
      </header>
      <div class="chart-body">{chart_html}</div>
    </section>
    """


def _run_card_html(item: ViewerRunData) -> str:
    params = item.params
    tags = item.tags
    rows = [
        ("Run ID", item.run.info.run_id),
        ("Version", str(params.get("version") or "unknown")),
        ("TP", str(params.get("tp") or "unknown")),
        ("Replicas", str(params.get("replicas") or "unknown")),
        (
            "Accelerator",
            str(params.get("accelerator") or tags.get("accelerator") or "unknown"),
        ),
        ("Release", item.release_name),
    ]
    rows_html = "".join(
        f'<div class="run-card-row"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
        for label, value in rows
    )
    return f"""
    <article class="run-card">
      <div class="run-card-header">
        <span class="run-swatch" style="background:{html.escape(item.color)}"></span>
        <h3>{html.escape(item.label)}</h3>
      </div>
      <div class="run-card-body">{rows_html}</div>
    </article>
    """


def _render_dashboard_html(
    *, run_data: list[ViewerRunData], dashboard: dict[str, Any]
) -> str:
    section_html: list[str] = []
    for section in _group_dashboard_sections(dashboard):
        if str(section["title"]).strip().lower() == "overview":
            continue
        panel_html: list[str] = []
        for panel in section["panels"]:
            panel_type = str(panel.get("type") or "").strip()
            if panel_type == "stat":
                panel_html.append(_build_stat_panel_html(panel, run_data=run_data))
            elif panel_type == "timeseries":
                panel_html.append(
                    _build_timeseries_panel_html(panel, run_data=run_data)
                )
        if panel_html:
            section_html.append(
                f"""
                <section class="dashboard-section">
                  <div class="section-heading">{html.escape(str(section["title"]))}</div>
                  <div class="panel-grid">
                    {"".join(panel_html)}
                  </div>
                </section>
                """
            )

    run_cards_html = "".join(_run_card_html(item) for item in run_data)
    headline = "BenchFlow interactive Prometheus archive"

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>BenchFlow Metrics Viewer</title>
    <style>
      :root {{
        --bg: #ffffff;
        --panel: #ffffff;
        --panel-border: #000000;
        --ink: #222222;
        --muted: #666666;
        --accent: #666666;
        --shadow: none;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        min-height: 100vh;
        color: var(--ink);
        background: var(--bg);
        font-family: Arial, sans-serif;
      }}
      .page {{
        max-width: 1640px;
        margin: 0 auto;
        padding: 1.1rem 1.1rem 2.5rem;
      }}
      .hero {{
        padding: 0 0 0.7rem;
      }}
      h1 {{
        margin: 0;
        font-size: 1.35rem;
        line-height: 1.1;
        font-weight: 700;
        text-align: center;
      }}
      .run-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 0.6rem;
        margin-top: 0.8rem;
      }}
      .run-card {{
        background: var(--panel);
        border: 1px solid var(--panel-border);
        border-radius: 0;
        box-shadow: var(--shadow);
        padding: 0.7rem 0.75rem;
      }}
      .run-card-header {{
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin-bottom: 0.55rem;
      }}
      .run-card-header h3 {{
        margin: 0;
        font-size: 0.9rem;
        line-height: 1.25;
      }}
      .run-card-body {{
        display: grid;
        gap: 0.32rem;
      }}
      .run-card-row {{
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        font-size: 0.84rem;
      }}
      .run-card-row span {{
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.68rem;
      }}
      .run-card-row strong {{
        font-weight: 600;
        text-align: right;
      }}
      .dashboard-section {{
        margin-top: 1rem;
      }}
      .section-heading {{
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 0.55rem;
        color: var(--accent);
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }}
      .panel-grid {{
        display: grid;
        grid-template-columns: 1fr;
        gap: 0.7rem;
      }}
      .panel {{
        background: var(--panel);
        border: 1px solid var(--panel-border);
        border-radius: 0;
        box-shadow: var(--shadow);
        overflow: hidden;
      }}
      .panel-header {{
        padding: 0.55rem 0.7rem 0;
      }}
      .panel-header h3 {{
        margin: 0;
        font-size: 0.85rem;
        line-height: 1.25;
        text-align: center;
      }}
      .chart-body {{
        padding: 0.08rem 0.16rem 0.5rem;
      }}
      .stat-panel {{
        padding: 0.9rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-height: 156px;
      }}
      .stat-panel-compare {{
        justify-content: flex-start;
      }}
      .stat-value {{
        font-size: 1.9rem;
        margin-top: 0.8rem;
        line-height: 1.1;
      }}
      .stat-subtitle {{
        margin-top: 0.5rem;
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.45;
      }}
      .stat-compare-grid {{
        display: grid;
        gap: 0.55rem;
        margin-top: 0.9rem;
      }}
      .stat-compare-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding-bottom: 0.45rem;
        border-bottom: 1px solid rgba(17, 24, 39, 0.06);
      }}
      .stat-compare-row:last-child {{
        border-bottom: none;
        padding-bottom: 0;
      }}
      .stat-compare-label {{
        display: flex;
        align-items: center;
        gap: 0.55rem;
        font-size: 0.86rem;
      }}
      .stat-compare-value {{
        font-size: 0.92rem;
        font-weight: 600;
      }}
      .run-swatch {{
        display: inline-block;
        width: 0.78rem;
        height: 0.78rem;
        border-radius: 0;
        box-shadow: none;
        flex: none;
      }}
      .span-3 {{ grid-column: span 1; }}
      .span-4 {{ grid-column: span 1; }}
      .span-5 {{ grid-column: span 1; }}
      .span-6 {{ grid-column: span 1; }}
      .span-7 {{ grid-column: span 1; }}
      .span-8 {{ grid-column: span 1; }}
      .span-9 {{ grid-column: span 1; }}
      .span-10 {{ grid-column: span 1; }}
      .span-11 {{ grid-column: span 1; }}
      .span-12 {{ grid-column: span 1; }}
      @media (max-width: 720px) {{
        .page {{ padding: 1rem 0.9rem 2rem; }}
        h1 {{ font-size: 1.18rem; }}
        .panel-grid {{ grid-template-columns: 1fr; }}
        .span-3, .span-4, .span-5, .span-6, .span-7, .span-8, .span-9, .span-10, .span-11, .span-12 {{
          grid-column: span 1;
        }}
      }}
    </style>
    <script>{get_plotlyjs()}</script>
  </head>
  <body>
    <main class="page">
      <section class="hero">
        <h1>{html.escape(headline)}</h1>
        <div class="run-grid">{run_cards_html}</div>
      </section>
      {"".join(section_html)}
    </main>
  </body>
</html>
"""


def _viewer_output_dir(mlflow_run_ids: list[str]) -> Path:
    key = hashlib.sha1(",".join(mlflow_run_ids).encode("utf-8")).hexdigest()[:12]
    output_dir = _CACHE_ROOT / "views" / key
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _write_dashboard_page(
    *,
    run_data: list[ViewerRunData],
    mlflow_run_ids: list[str],
    output_file: str | Path | None = None,
) -> Path:
    dashboard = _load_dashboard_spec()
    html_text = _render_dashboard_html(run_data=run_data, dashboard=dashboard)
    if output_file:
        output_path = Path(output_file).expanduser().resolve()
        if output_path.exists() and output_path.is_dir():
            raise ValidationError(
                f"metrics report output file is a directory: {output_path}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = _viewer_output_dir(mlflow_run_ids)
        output_path = output_dir / "index.html"
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def _load_dashboard_run_data(
    *,
    mlflow_run_ids: list[str],
    mlflow_tracking_uri: str | None = None,
) -> list[ViewerRunData]:
    if not mlflow_run_ids:
        raise ValidationError("at least one MLflow run ID is required")
    loaded = [
        _load_viewer_run_data(
            mlflow_run_id=run_id,
            mlflow_tracking_uri=mlflow_tracking_uri,
            color=_PALETTE[index % len(_PALETTE)],
        )
        for index, run_id in enumerate(mlflow_run_ids)
    ]
    return _ensure_unique_labels(loaded)


def generate_mlflow_metrics_dashboard_report(
    *,
    mlflow_run_ids: list[str],
    mlflow_tracking_uri: str | None = None,
    output_file: str | Path,
) -> str:
    run_data = _load_dashboard_run_data(
        mlflow_run_ids=mlflow_run_ids,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )
    html_path = _write_dashboard_page(
        run_data=run_data,
        mlflow_run_ids=mlflow_run_ids,
        output_file=output_file,
    )
    success(
        f"Wrote metrics HTML report for {len(mlflow_run_ids)} MLflow run(s) to {html_path}"
    )
    return str(html_path)


def serve_mlflow_metrics_dashboard(
    *,
    mlflow_run_ids: list[str],
    mlflow_tracking_uri: str | None = None,
) -> str:
    run_data = _load_dashboard_run_data(
        mlflow_run_ids=mlflow_run_ids,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )
    html_path = _write_dashboard_page(run_data=run_data, mlflow_run_ids=mlflow_run_ids)

    handler = partial(SimpleHTTPRequestHandler, directory=str(html_path.parent))
    try:
        server = ThreadingHTTPServer(
            ("127.0.0.1", DEFAULT_METRICS_VIEWER_PORT), handler
        )
    except OSError as exc:
        raise ValidationError(
            f"unable to bind metrics viewer to http://127.0.0.1:{DEFAULT_METRICS_VIEWER_PORT}: {exc}"
        ) from exc

    url = f"http://127.0.0.1:{DEFAULT_METRICS_VIEWER_PORT}/"
    mode = "comparison" if len(mlflow_run_ids) > 1 else "viewer"
    success(f"Serving metrics {mode} for {len(mlflow_run_ids)} MLflow run(s) at {url}")
    try:
        if webbrowser.open(url):
            detail("Opened metrics viewer in the default browser")
        else:
            detail("Unable to auto-open the browser; open the viewer URL manually")
    except Exception:
        detail("Unable to auto-open the browser; open the viewer URL manually")
    detail("Press Ctrl-C to stop the local server")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        detail("Stopping metrics viewer")
    finally:
        server.server_close()
    return url
