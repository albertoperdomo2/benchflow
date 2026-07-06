from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

from ..contracts import ValidationError

logger = logging.getLogger(__name__)

_QUERY_RATE_INTERVAL = "5m"
_LEGEND_TEMPLATE_RE = re.compile(r"\{\{([^{}]+)\}\}")
_POD_HASH_RE = re.compile(r"-[a-z0-9]{8,10}-[a-z0-9]{4,6}$")


@dataclass(slots=True)
class ReportMetricSpec:
    name: str
    metric: str
    fallback_metric: str
    title: str
    description: str
    unit: str
    yaxis: str
    scale: float | None
    query: str
    series_template: str
    fallback_series_template: str
    reference_lines: list["ReportMetricReferenceLine"]


@dataclass(slots=True)
class ReportMetricReferenceLine:
    y: float
    label: str
    color: str
    dash: str


@dataclass(slots=True)
class ReportMetricsSpec:
    title: str
    description: str
    metrics: list[ReportMetricSpec]


@dataclass(slots=True)
class ComparisonMetricTrace:
    name: str
    x: list[float]
    y: list[float]
    yaxis_title: str


@dataclass(slots=True)
class ComparisonMetricPanel:
    title: str
    description: str
    yaxis_title: str
    traces: list[ComparisonMetricTrace]
    missing_runs: list[str]
    reference_lines: list[ReportMetricReferenceLine]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_query(query: str) -> str:
    normalized = str(query).replace("$__rate_interval", _QUERY_RATE_INTERVAL)
    normalized = normalized.replace("\n", "")
    return re.sub(r"\s+", "", normalized)


def _clean_label(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or "series"


def _humanize_metric_name(value: str) -> str:
    text = re.sub(r"[_\-.]+", " ", str(value or "").strip())
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "Prometheus metric"
    replacements = {
        "kv": "KV",
        "rss": "RSS",
        "cpu": "CPU",
        "gpu": "GPU",
        "nfs": "NFS",
        "pcie": "PCIe",
        "gib": "GiB",
    }
    return " ".join(replacements.get(part.lower(), part) for part in text.split())


def _inferred_scale(metric_name: str, spec: ReportMetricSpec) -> float:
    if spec.scale is not None:
        return spec.scale
    lowered = metric_name.lower()
    if "bytes_rate" in lowered or "bytes_per_second" in lowered:
        return 1 / 1024 / 1024 / 1024
    if "bytes" in lowered:
        return 1 / 1024 / 1024 / 1024
    if (
        "percent" in lowered
        or "usage_perc" in lowered
        or "ratio" in lowered
        or "hit_rate" in lowered
    ):
        return 100.0
    return 1.0


def _inferred_yaxis(metric_name: str, spec: ReportMetricSpec) -> str:
    if spec.yaxis:
        return spec.yaxis
    lowered = metric_name.lower()
    if "bytes_rate" in lowered or "bytes_per_second" in lowered:
        return "GiB/s"
    if "bytes" in lowered:
        return "GiB"
    if "gpu_utilization" in lowered:
        return "%"
    if "gpu_memory" in lowered:
        return "MiB"
    if (
        "percent" in lowered
        or "usage_perc" in lowered
        or "ratio" in lowered
        or "hit_rate" in lowered
    ):
        return "%"
    if "seconds_per_gib" in lowered:
        return "seconds/GiB"
    if "seconds" in lowered:
        return "seconds"
    return "Value"


def _gpu_id(labels: dict[str, Any]) -> str:
    gpu = labels.get("gpu")
    if gpu not in {None, ""}:
        return str(gpu)
    device = str(labels.get("device") or "").strip()
    match = re.fullmatch(r"nvidia(\d+)", device)
    if match:
        return match.group(1)
    return ""


def _pod_suffix(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parts = [part for part in text.split("-") if part]
    return parts[-1] if parts else text


def _series_pod_name(labels: dict[str, Any]) -> str:
    exported_pod = str(labels.get("exported_pod") or "").strip()
    if exported_pod:
        return exported_pod
    pod = str(labels.get("pod") or "").strip()
    if pod and not pod.startswith("nvidia-dcgm"):
        return pod
    return ""


def _pod_role(labels: dict[str, Any]) -> str:
    pod = _series_pod_name(labels).lower()
    container = str(labels.get("container") or "").strip().lower()
    image = str(labels.get("image") or "").strip().lower()
    joined = " ".join(value for value in [pod, container, image] if value)
    if any(
        marker in joined
        for marker in [
            "router-scheduler",
            "inference-scheduler",
            "scheduler",
            "gaie-",
            "-epp",
            "gateway",
        ]
    ):
        return "scheduler"
    if any(
        marker in joined
        for marker in [
            "-kserve-",
            "-predictor-",
            "vllm",
            "modelmesh",
            "model-server",
        ]
    ) or pod.startswith("ms-"):
        return "model"
    return ""


def _default_series_suffix(labels: dict[str, Any], series: str) -> str:
    gpu = _gpu_id(labels)
    if gpu:
        return f"gpu-{gpu}"

    pod_name = _series_pod_name(labels)
    suffix = _pod_suffix(pod_name)
    role = _pod_role(labels)
    role_prefix = f"{role}-" if role else ""

    transfer_type = labels.get("transfer_type")
    transfer_suffix = (
        f"-{_clean_label(str(transfer_type)).replace(' ', '-')}"
        if transfer_type not in {None, ""}
        else ""
    )
    container = labels.get("container")
    container_suffix = (
        f"-container-{_clean_label(str(container)).replace(' ', '-')}"
        if container not in {None, "", "POD"}
        else ""
    )
    if suffix:
        return f"{role_prefix}pod-{suffix}{container_suffix}{transfer_suffix}"

    instance = labels.get("instance")
    if instance not in {None, ""}:
        return f"series-{_short_resource_name(str(instance)) or _clean_label(str(instance))}"
    return f"series-{_short_resource_name(series) or _clean_label(series)}"


def _short_resource_name(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    # KServe/RHOAI pod names usually duplicate the service name and add hashes.
    if "-kserve-" in text:
        text = text.split("-kserve-", 1)[0]
    text = _POD_HASH_RE.sub("", text)
    return text


def _render_template(template: str, *, labels: dict[str, Any], series: str) -> str:
    raw = str(template or "").strip()
    if not raw:
        return _clean_label(series)

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        if key == "series":
            value = series
        else:
            value = labels.get(key)
        return "" if value in {None, ""} else str(value)

    return _clean_label(_LEGEND_TEMPLATE_RE.sub(_replace, raw))


def _shorten_templated_series_label(value: str, labels: dict[str, Any]) -> str:
    """Preserve templated dimensions while shortening verbose pod identifiers."""
    label = _clean_label(value)
    pod_name = _series_pod_name(labels)
    if pod_name:
        suffix = _pod_suffix(pod_name)
        role = _pod_role(labels)
        role_prefix = f"{role}-" if role else ""
        replacement = f"{role_prefix}pod-{suffix}" if suffix else pod_name
        label = label.replace(pod_name, replacement)
    return _clean_label(label).replace(" ", "-")


def _metric_spec_from_dict(raw: dict[str, Any], index: int) -> ReportMetricSpec:
    if not isinstance(raw, dict):
        raise ValidationError(f"metrics[{index}] must be a mapping")
    metric = str(raw.get("metric") or raw.get("name") or "").strip()
    query = str(raw.get("query") or "").strip()
    if not metric and not query:
        raise ValidationError(f"metrics[{index}] must define either metric or query")
    title = str(raw.get("title") or "").strip()
    scale = raw.get("scale")
    reference_lines = _reference_lines_from_raw(
        raw.get("reference_lines"), f"metrics[{index}].reference_lines"
    )
    return ReportMetricSpec(
        name=str(raw.get("name") or metric or f"metric_{index + 1}").strip(),
        metric=metric,
        fallback_metric=str(raw.get("fallback_metric") or "").strip(),
        title=title,
        description=str(raw.get("description") or "").strip(),
        unit=str(raw.get("unit") or "").strip(),
        yaxis=str(raw.get("yaxis") or raw.get("unit") or "").strip(),
        scale=float(scale) if scale is not None else None,
        query=query,
        series_template=str(
            raw.get("series") or raw.get("series_template") or ""
        ).strip(),
        fallback_series_template=str(
            raw.get("fallback_series") or raw.get("fallback_series_template") or ""
        ).strip(),
        reference_lines=reference_lines,
    )


def _reference_lines_from_raw(
    raw: Any, field_name: str
) -> list[ReportMetricReferenceLine]:
    """Parse optional report-only y-axis guide lines from report-metrics YAML.

    These lines are intentionally not used by any packaged profile yet. They are
    available for future PCIe/link-capacity annotations where the reference
    value is known outside Prometheus.
    """
    if raw is None or raw == "":
        return []
    if not isinstance(raw, list):
        raise ValidationError(f"{field_name} must be a list")

    reference_lines: list[ReportMetricReferenceLine] = []
    for line_index, item in enumerate(raw):
        item_name = f"{field_name}[{line_index}]"
        if not isinstance(item, dict):
            raise ValidationError(f"{item_name} must be a mapping")
        if "y" not in item:
            raise ValidationError(f"{item_name}.y is required")
        try:
            y_value = float(item["y"])
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{item_name}.y must be numeric") from exc
        reference_lines.append(
            ReportMetricReferenceLine(
                y=y_value,
                label=str(item.get("label") or "").strip(),
                color=str(item.get("color") or "#64748b").strip(),
                dash=str(item.get("dash") or item.get("style") or "dash").strip(),
            )
        )
    return reference_lines


def load_report_metrics_spec(path: Path) -> ReportMetricsSpec:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValidationError(f"report metrics YAML must be a mapping: {path}")
    metrics_raw = payload.get("metrics")
    if not isinstance(metrics_raw, list) or not metrics_raw:
        raise ValidationError(
            "report metrics YAML must contain a non-empty metrics list"
        )
    return ReportMetricsSpec(
        title=str(payload.get("title") or "Prometheus Metrics").strip(),
        description=str(payload.get("description") or "").strip(),
        metrics=[
            _metric_spec_from_dict(item, index)
            for index, item in enumerate(metrics_raw)
        ],
    )


def _metrics_cache_ready(metrics_dir: Path) -> bool:
    return (
        (metrics_dir / "metrics_summary.json").exists()
        and (metrics_dir / "resolved_queries.json").exists()
        and (metrics_dir / "raw").is_dir()
    )


def _download_metrics_artifacts(run_data: dict[str, Any]) -> Path | None:
    run_id = str(run_data["run_id"])
    metrics_dir = Path("/tmp/mlflow") / run_id / "metrics"
    if _metrics_cache_ready(metrics_dir):
        return metrics_dir

    artifact_uri = str(run_data.get("artifact_uri") or "").strip()
    if not artifact_uri:
        logger.warning("Run %s has no artifact URI; skipping metrics", run_id)
        return None

    cache_root = metrics_dir.parent
    cache_root.mkdir(parents=True, exist_ok=True)
    if metrics_dir.exists():
        shutil.rmtree(metrics_dir)
    repo = get_artifact_repository(artifact_uri)
    try:
        downloaded_path = Path(
            repo.download_artifacts("metrics", dst_path=str(cache_root))
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Run %s has no downloadable metrics artifacts: %s", run_id, exc)
        return None
    if downloaded_path != metrics_dir and downloaded_path.exists():
        if metrics_dir.exists():
            shutil.rmtree(metrics_dir)
        shutil.copytree(downloaded_path, metrics_dir)
    if not _metrics_cache_ready(metrics_dir):
        logger.warning("Run %s metrics artifact tree is incomplete", run_id)
        return None
    return metrics_dir


def _metric_name_for_spec(metrics_dir: Path, spec: ReportMetricSpec) -> str | None:
    if spec.metric and (metrics_dir / "raw" / f"{spec.metric}.json").exists():
        return spec.metric
    if not spec.query:
        return None
    resolved_queries_path = metrics_dir / "resolved_queries.json"
    if not resolved_queries_path.exists():
        return None
    try:
        resolved_queries = _load_json(resolved_queries_path)
    except Exception:  # noqa: BLE001
        return None
    query = spec.query
    summary_path = metrics_dir / "metrics_summary.json"
    if summary_path.exists():
        try:
            context = _load_json(summary_path).get("query_context") or {}
            for key, value in sorted(
                context.items(), key=lambda item: len(str(item[0])), reverse=True
            ):
                query = query.replace(f"${key}", str(value))
        except Exception:  # noqa: BLE001
            pass
    wanted = _normalize_query(query)
    for metric_name, query in (resolved_queries or {}).items():
        if _normalize_query(str(query)) == wanted:
            return str(metric_name)
    return None


def _load_points(metrics_dir: Path, metric_name: str) -> list[dict[str, Any]]:
    path = metrics_dir / "raw" / f"{metric_name}.json"
    if not path.exists():
        return []
    payload = _load_json(path)
    return payload if isinstance(payload, list) else []


def _metric_name_and_points_for_spec(
    metrics_dir: Path,
    spec: ReportMetricSpec,
) -> tuple[str | None, list[dict[str, Any]], bool]:
    metric_name = _metric_name_for_spec(metrics_dir, spec)
    if metric_name is not None:
        points = _load_points(metrics_dir, metric_name)
        if points:
            return metric_name, points, False

    if spec.fallback_metric:
        fallback_path = metrics_dir / "raw" / f"{spec.fallback_metric}.json"
        if fallback_path.exists():
            points = _load_points(metrics_dir, spec.fallback_metric)
            if points:
                return spec.fallback_metric, points, True

    return metric_name, [], False


def _run_start_timestamp(metrics_dir: Path, points: list[dict[str, Any]]) -> float:
    summary_path = metrics_dir / "metrics_summary.json"
    if summary_path.exists():
        try:
            summary = _load_json(summary_path)
            start = str(summary.get("benchmark_start_time") or "")
            if start:
                from datetime import datetime, timezone

                return (
                    datetime.fromisoformat(start.replace("Z", "+00:00"))
                    .astimezone(timezone.utc)
                    .timestamp()
                )
        except Exception:  # noqa: BLE001
            pass
    timestamps = [
        float(point["timestamp"])
        for point in points
        if isinstance(point, dict) and point.get("timestamp") is not None
    ]
    return min(timestamps) if timestamps else 0.0


def _series_group_key(point: dict[str, Any]) -> str:
    labels = point.get("labels") or {}
    if isinstance(labels, dict) and labels:
        pairs = [
            (str(key), str(value))
            for key, value in labels.items()
            if key != "__name__" and value not in {None, ""}
        ]
        if pairs:
            return json.dumps(sorted(pairs), separators=(",", ":"))
    return str(point.get("series") or "series")


def _group_points(points: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for point in points:
        series = _series_group_key(point)
        bucket = grouped.setdefault(
            series,
            {"labels": dict(point.get("labels") or {}), "points": []},
        )
        bucket["points"].append(point)
    for bucket in grouped.values():
        bucket["points"].sort(key=lambda item: float(item.get("timestamp") or 0.0))
    return grouped


def _trace_name(
    *,
    spec: ReportMetricSpec,
    labels: dict[str, Any],
    series: str,
    version_label: str,
    series_template: str | None = None,
) -> str:
    template = spec.series_template if series_template is None else series_template
    if template:
        short_series = _shorten_templated_series_label(
            _render_template(template, labels=labels, series=series),
            labels,
        )
    else:
        short_series = _short_resource_name(_default_series_suffix(labels, series))
    if short_series and short_series != "series":
        if version_label:
            return f"{version_label}-{short_series}"
        return short_series
    return version_label or "series"


def _append_averaged_point(
    grouped: dict[str, dict[float, list[float]]],
    *,
    trace_name: str,
    timestamp: float,
    value: float,
) -> None:
    grouped.setdefault(trace_name, {}).setdefault(timestamp, []).append(value)


def _build_metric_panel(
    *,
    spec: ReportMetricSpec,
    runs_data: list[dict[str, Any]],
    version_overrides: dict[str, str],
) -> ComparisonMetricPanel:
    missing_runs: list[str] = []
    first_metric_name: str | None = None
    grouped_values: dict[str, dict[float, list[float]]] = {}
    trace_yaxis: dict[str, str] = {}

    for run_data in runs_data:
        metrics_dir = _download_metrics_artifacts(run_data)
        run_label = str(run_data.get("composed_version") or run_data["run_id"])
        if metrics_dir is None:
            missing_runs.append(run_label)
            continue
        metric_name, points, used_fallback = _metric_name_and_points_for_spec(
            metrics_dir, spec
        )
        if metric_name is None:
            missing_runs.append(run_label)
            continue
        if first_metric_name is None:
            first_metric_name = metric_name
        if not points:
            missing_runs.append(run_label)
            continue

        start_timestamp = _run_start_timestamp(metrics_dir, points)
        composed_version = str(run_data.get("composed_version") or "unknown")
        version_label = version_overrides.get(composed_version, composed_version)
        yaxis_title = _inferred_yaxis(metric_name, spec)
        value_scale = _inferred_scale(metric_name, spec)
        for series, bucket in _group_points(points).items():
            labels = dict(bucket.get("labels") or {})
            trace_name = _trace_name(
                spec=spec,
                labels=labels,
                series=series,
                version_label=version_label,
                series_template=(
                    spec.fallback_series_template
                    if used_fallback and spec.fallback_series_template
                    else None
                ),
            )
            trace_yaxis[trace_name] = yaxis_title
            for point in bucket["points"]:
                try:
                    timestamp = (float(point["timestamp"]) - start_timestamp) / 60.0
                    value = float(point["value"]) * value_scale
                except (KeyError, TypeError, ValueError):
                    continue
                _append_averaged_point(
                    grouped_values,
                    trace_name=trace_name,
                    timestamp=timestamp,
                    value=value,
                )

    traces: list[ComparisonMetricTrace] = []
    for trace_name, points_by_timestamp in sorted(grouped_values.items()):
        timestamps = sorted(points_by_timestamp)
        traces.append(
            ComparisonMetricTrace(
                name=trace_name,
                x=timestamps,
                y=[
                    sum(points_by_timestamp[timestamp])
                    / len(points_by_timestamp[timestamp])
                    for timestamp in timestamps
                ],
                yaxis_title=trace_yaxis.get(
                    trace_name,
                    _inferred_yaxis(
                        first_metric_name or spec.metric or spec.name, spec
                    ),
                ),
            )
        )

    title = spec.title or _humanize_metric_name(
        first_metric_name or spec.metric or spec.name
    )
    description = spec.description
    if missing_runs:
        missing = ", ".join(missing_runs[:4])
        suffix = f"Missing archived series for: {missing}"
        if len(missing_runs) > 4:
            suffix += f", +{len(missing_runs) - 4} more"
        description = f"{description} {suffix}".strip()

    return ComparisonMetricPanel(
        title=title,
        description=description,
        yaxis_title=_inferred_yaxis(
            first_metric_name or spec.metric or spec.name, spec
        ),
        traces=traces,
        missing_runs=missing_runs,
        reference_lines=list(spec.reference_lines),
    )


def _build_run_metric_panel(
    *,
    spec: ReportMetricSpec,
    metrics_dir: Path,
) -> ComparisonMetricPanel:
    metric_name, points, used_fallback = _metric_name_and_points_for_spec(
        metrics_dir, spec
    )
    trace_yaxis: dict[str, str] = {}
    grouped_values: dict[str, dict[float, list[float]]] = {}

    if metric_name and points:
        start_timestamp = _run_start_timestamp(metrics_dir, points)
        yaxis_title = _inferred_yaxis(metric_name, spec)
        value_scale = _inferred_scale(metric_name, spec)
        for series, bucket in _group_points(points).items():
            labels = dict(bucket.get("labels") or {})
            trace_name = _trace_name(
                spec=spec,
                labels=labels,
                series=series,
                version_label="",
                series_template=(
                    spec.fallback_series_template
                    if used_fallback and spec.fallback_series_template
                    else None
                ),
            )
            trace_yaxis[trace_name] = yaxis_title
            for point in bucket["points"]:
                try:
                    timestamp = (float(point["timestamp"]) - start_timestamp) / 60.0
                    value = float(point["value"]) * value_scale
                except (KeyError, TypeError, ValueError):
                    continue
                _append_averaged_point(
                    grouped_values,
                    trace_name=trace_name,
                    timestamp=timestamp,
                    value=value,
                )

    traces: list[ComparisonMetricTrace] = []
    for trace_name, points_by_timestamp in sorted(grouped_values.items()):
        timestamps = sorted(points_by_timestamp)
        traces.append(
            ComparisonMetricTrace(
                name=trace_name,
                x=timestamps,
                y=[
                    sum(points_by_timestamp[timestamp])
                    / len(points_by_timestamp[timestamp])
                    for timestamp in timestamps
                ],
                yaxis_title=trace_yaxis.get(
                    trace_name,
                    _inferred_yaxis(metric_name or spec.metric or spec.name, spec),
                ),
            )
        )

    return ComparisonMetricPanel(
        title=spec.title
        or _humanize_metric_name(metric_name or spec.metric or spec.name),
        description=spec.description,
        yaxis_title=_inferred_yaxis(metric_name or spec.metric or spec.name, spec),
        traces=traces,
        missing_runs=[] if traces else ["local artifacts"],
        reference_lines=list(spec.reference_lines),
    )


def build_comparison_metric_panels(
    *,
    metrics_yaml_path: Path | None,
    runs_data: list[dict[str, Any]],
    version_overrides: dict[str, str] | None = None,
) -> list[ComparisonMetricPanel]:
    if metrics_yaml_path is None:
        return []
    spec = load_report_metrics_spec(metrics_yaml_path)
    return [
        _build_metric_panel(
            spec=metric,
            runs_data=runs_data,
            version_overrides=version_overrides or {},
        )
        for metric in spec.metrics
    ]


def find_metrics_artifacts_dir(root: Path) -> Path | None:
    for candidate in (
        root,
        root / "metrics",
        root / "benchmark" / "metrics",
        root / "results" / "metrics",
    ):
        if _metrics_cache_ready(candidate):
            return candidate
    return None


def build_run_metric_panels(
    *,
    metrics_yaml_path: Path | None,
    metrics_dir: Path | None,
) -> list[ComparisonMetricPanel]:
    if metrics_yaml_path is None or metrics_dir is None:
        return []
    spec = load_report_metrics_spec(metrics_yaml_path)
    return [
        _build_run_metric_panel(spec=metric, metrics_dir=metrics_dir)
        for metric in spec.metrics
    ]
