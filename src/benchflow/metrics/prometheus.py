from __future__ import annotations

import json
import math
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from ..cluster import CommandError
from ..llmd_layout import uses_recipe_layout as _llmd_uses_recipe_layout
from ..models import ResolvedRunPlan
from ..ui import detail, step, success, warning

_NO_POD_MATCH = "a^"
_HOST_PATH_FILESYSTEM_TYPES = "tmpfs|overlay|squashfs|proc|sysfs|devtmpfs|cgroup2"


def _parse_iso8601(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _parse_duration_seconds(value: str) -> int:
    value = value.strip().lower()
    if value.endswith("ms"):
        return max(1, math.ceil(float(value[:-2]) / 1000.0))
    if value.endswith("s"):
        return max(1, math.ceil(float(value[:-1])))
    if value.endswith("m"):
        return max(1, math.ceil(float(value[:-1]) * 60.0))
    if value.endswith("h"):
        return max(1, math.ceil(float(value[:-1]) * 3600.0))
    return max(1, math.ceil(float(value)))


def _summarize_series(result: list[dict[str, object]]) -> dict[str, object]:
    samples: list[float] = []
    latest_values: list[float] = []

    for series in result:
        values = series.get("values") or []
        if not isinstance(values, list):
            continue
        parsed_values: list[float] = []
        for pair in values:
            try:
                parsed_values.append(float(pair[1]))
            except (IndexError, TypeError, ValueError):
                continue
        if parsed_values:
            samples.extend(parsed_values)
            latest_values.append(parsed_values[-1])

    if not samples:
        return {
            "series_count": len(result),
            "sample_count": 0,
            "min": None,
            "max": None,
            "avg": None,
            "latest_avg": None,
        }

    return {
        "series_count": len(result),
        "sample_count": len(samples),
        "min": min(samples),
        "max": max(samples),
        "avg": sum(samples) / len(samples),
        "latest_avg": (sum(latest_values) / len(latest_values))
        if latest_values
        else None,
    }


def _series_name(metric_name: str, metric_labels: dict[str, object]) -> str:
    exported_pod = metric_labels.get("exported_pod")
    if exported_pod:
        parts = [str(exported_pod)]
        gpu = metric_labels.get("gpu")
        device = metric_labels.get("device")
        if gpu not in {None, ""}:
            parts.append(f"gpu={gpu}")
        elif device not in {None, ""}:
            parts.append(str(device))
        return " ".join(parts)

    preferred = (
        "pod",
        "exported_pod",
        "container",
        "instance",
        "device",
        "gpu",
        "model_name",
    )
    for key in preferred:
        value = metric_labels.get(key)
        if value:
            return str(value)
    if not metric_labels:
        return metric_name
    pairs = [
        f"{key}={value}"
        for key, value in sorted(metric_labels.items())
        if key != "__name__"
    ]
    return ", ".join(pairs) if pairs else metric_name


def _normalize_series(
    metric_name: str, result: list[dict[str, object]]
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for series in result:
        metric_labels = series.get("metric") or {}
        if not isinstance(metric_labels, dict):
            metric_labels = {}
        series_name = _series_name(metric_name, metric_labels)
        values = series.get("values") or []
        if not isinstance(values, list):
            continue
        for pair in values:
            try:
                timestamp = float(pair[0])
                value = float(pair[1])
            except (IndexError, TypeError, ValueError):
                continue
            normalized.append(
                {
                    "time": datetime.fromtimestamp(timestamp, tz=timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "timestamp": int(timestamp),
                    "value": value,
                    "series": series_name,
                    "labels": metric_labels,
                }
            )
    return normalized


def _query_template_values(
    plan: ResolvedRunPlan, metrics_release_name: str
) -> dict[str, str]:
    platform = str(plan.deployment.platform or "").strip().lower()
    mode = str(plan.deployment.mode or "").strip().lower()
    modelserver_pod_regex = f".*{metrics_release_name}.*"
    scheduler_pod_regex = _NO_POD_MATCH
    scheduler_endpoint_regex = _NO_POD_MATCH

    if platform == "llm-d":
        if _llmd_uses_recipe_layout(plan.deployment.repo_ref):
            modelserver_pod_regex = f"ms-{metrics_release_name}-.*"
            scheduler_pod_regex = f"gaie-{metrics_release_name}-epp.*"
        else:
            scheduler_pod_regex = f".*{metrics_release_name}.*"
        scheduler_endpoint_regex = "(metrics|http-metrics)"
    elif platform == "rhoai":
        if mode != "isvc":
            scheduler_pod_regex = f".*{metrics_release_name}.*router-scheduler.*"
            scheduler_endpoint_regex = "metrics"

    return {
        "$namespace": plan.deployment.namespace,
        "$release": metrics_release_name,
        "$modelserver_pod_regex": modelserver_pod_regex,
        "$scheduler_pod_regex": scheduler_pod_regex,
        "$scheduler_endpoint_regex": scheduler_endpoint_regex,
        "$hostpath_mount_regex": _hostpath_mount_regex(plan),
        "$hostpath_filesystem_exclude_regex": _HOST_PATH_FILESYSTEM_TYPES,
        "$runtime_pvc_regex": _runtime_pvc_regex(plan),
        "$ceph_pvc_regex": _ceph_pvc_regex(plan),
        "$ceph_pool_regex": _ceph_pool_regex(plan),
    }


def _runtime_pvc_regex(plan: ResolvedRunPlan) -> str:
    claim_names = {
        str(pvc_mount.claim_name or "").strip()
        for pvc_mount in plan.deployment.runtime.pvc_mounts
    }
    claim_names.discard("")
    if not claim_names:
        return _NO_POD_MATCH
    escaped = [re.escape(item) for item in sorted(claim_names)]
    return f"^({'|'.join(escaped)})$"


def _is_ceph_pvc_mount(pvc_mount: object) -> bool:
    storage_class = str(getattr(pvc_mount, "storage_class_name", "") or "").lower()
    return not storage_class or "ceph" in storage_class or "rook" in storage_class


def _ceph_pvc_regex(plan: ResolvedRunPlan) -> str:
    claim_names = {
        str(pvc_mount.claim_name or "").strip()
        for pvc_mount in plan.deployment.runtime.pvc_mounts
        if _is_ceph_pvc_mount(pvc_mount)
    }
    claim_names.discard("")
    if not claim_names:
        return _NO_POD_MATCH
    escaped = [re.escape(item) for item in sorted(claim_names)]
    return f"^({'|'.join(escaped)})$"


def _ceph_pool_regex(plan: ResolvedRunPlan) -> str:
    pvc_mounts = [
        pvc_mount
        for pvc_mount in plan.deployment.runtime.pvc_mounts
        if _is_ceph_pvc_mount(pvc_mount)
    ]
    if not pvc_mounts:
        return _NO_POD_MATCH

    return ".*(cephfs|cephfilesystem|filesystem).*|.*[-_.]data[0-9]*.*"


def _hostpath_mount_regex(plan: ResolvedRunPlan) -> str:
    mountpoints: set[str] = set()
    for host_path in plan.deployment.runtime.host_paths:
        raw_path = str(host_path.host_path or "").strip()
        if not raw_path.startswith("/"):
            continue
        path = Path(raw_path)
        candidates = [path]
        if path.parent != path and str(path.parent) != "/":
            candidates.append(path.parent)
        for candidate in candidates:
            value = str(candidate).rstrip("/") or "/"
            if value != "/":
                mountpoints.add(value)
    if not mountpoints:
        return _NO_POD_MATCH
    escaped = [re.escape(item) for item in sorted(mountpoints, key=len, reverse=True)]
    return f"^({'|'.join(escaped)})$"


def _resolve_query_template(query_template: str, values: dict[str, str]) -> str:
    resolved = query_template
    for key, value in sorted(
        values.items(), key=lambda item: len(item[0]), reverse=True
    ):
        resolved = resolved.replace(key, value)
    return resolved


def collect_metrics(
    plan: ResolvedRunPlan,
    *,
    benchmark_start_time: str,
    benchmark_end_time: str,
    artifacts_dir: Path,
) -> Path:
    start_time = _parse_iso8601(benchmark_start_time)
    end_time = _parse_iso8601(benchmark_end_time)
    metrics_dir = artifacts_dir / "metrics"
    raw_dir = metrics_dir / "raw"
    prometheus_dir = metrics_dir / "prometheus"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    prometheus_dir.mkdir(parents=True, exist_ok=True)
    step(
        f"Collecting Prometheus metrics for "
        f"{plan.deployment.target.scoped_release_name(plan.deployment.release_name)} "
        f"from {benchmark_start_time} to {benchmark_end_time}"
    )
    detail(f"Prometheus URL: {plan.metrics.prometheus_url}")
    detail(
        f"Query step: {plan.metrics.query_step}, timeout: {plan.metrics.query_timeout}, "
        f"TLS verification: {'enabled' if plan.metrics.verify_tls else 'disabled'}"
    )

    metrics_release_name = plan.deployment.target.scoped_release_name(
        plan.deployment.release_name
    )
    template_values = _query_template_values(plan, metrics_release_name)

    queries = plan.metrics.queries or {}
    if not queries:
        (metrics_dir / "metrics_summary.json").write_text(
            json.dumps(
                {"status": "disabled", "reason": "no metrics queries configured"},
                indent=2,
            ),
            encoding="utf-8",
        )
        warning("No metrics queries configured; wrote disabled metrics summary")
        return metrics_dir

    token_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
    if not token_path.exists():
        raise CommandError(
            "service account token not found; metrics collection requires in-cluster execution"
        )
    token = token_path.read_text(encoding="utf-8").strip()

    if plan.metrics.verify_tls:
        ca_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt")
        if not ca_path.exists():
            raise CommandError(
                "verify_tls=true but service account CA bundle is missing"
            )
        ssl_context = ssl.create_default_context(cafile=str(ca_path))
    else:
        ssl_context = ssl._create_unverified_context()

    request_timeout = _parse_duration_seconds(plan.metrics.query_timeout) + 15
    query_metadata: dict[str, str] = {}
    archive_index: dict[str, object] = {
        "namespace": plan.deployment.namespace,
        "release_name": metrics_release_name,
        "benchmark_start_time": start_time.isoformat().replace("+00:00", "Z"),
        "benchmark_end_time": end_time.isoformat().replace("+00:00", "Z"),
        "metrics": {},
    }
    summary: dict[str, object] = {
        "namespace": plan.deployment.namespace,
        "release_name": metrics_release_name,
        "prometheus_url": plan.metrics.prometheus_url,
        "verify_tls": plan.metrics.verify_tls,
        "benchmark_start_time": start_time.isoformat().replace("+00:00", "Z"),
        "benchmark_end_time": end_time.isoformat().replace("+00:00", "Z"),
        "query_step": plan.metrics.query_step,
        "query_timeout": plan.metrics.query_timeout,
        "query_context": {
            key.lstrip("$"): value for key, value in template_values.items()
        },
        "queries": {},
        "failures": {},
    }
    detail(
        f"Collecting {len(queries)} metrics quer{'y' if len(queries) == 1 else 'ies'}"
    )

    for metric_name, query_template in sorted(queries.items()):
        resolved_query = _resolve_query_template(query_template, template_values)
        query_metadata[metric_name] = resolved_query
        detail(f"Querying metric {metric_name}")
        params = urllib.parse.urlencode(
            {
                "query": resolved_query,
                "start": int(start_time.timestamp()),
                "end": int(end_time.timestamp()),
                "step": plan.metrics.query_step,
                "timeout": plan.metrics.query_timeout,
            }
        )
        url = f"{plan.metrics.prometheus_url.rstrip('/')}/api/v1/query_range?{params}"
        request = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(
                request, timeout=request_timeout, context=ssl_context
            ) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            summary["failures"][metric_name] = f"HTTP {exc.code}: {exc.reason}"
            warning(
                f"Metric query failed for {metric_name}: HTTP {exc.code}: {exc.reason}"
            )
            continue
        except Exception as exc:  # noqa: BLE001
            summary["failures"][metric_name] = str(exc)
            warning(f"Metric query failed for {metric_name}: {exc}")
            continue

        (raw_dir / f"{metric_name}.json").write_text(
            json.dumps(
                _normalize_series(
                    metric_name, payload.get("data", {}).get("result", [])
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        (prometheus_dir / f"{metric_name}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        if payload.get("status") != "success":
            summary["failures"][metric_name] = payload.get("error", "query failed")
            warning(
                f"Metric query returned non-success status for {metric_name}: "
                f"{summary['failures'][metric_name]}"
            )
            continue
        result = payload.get("data", {}).get("result", [])
        metric_summary = _summarize_series(result)
        metric_summary["query"] = resolved_query
        summary["queries"][metric_name] = metric_summary
        archive_index["metrics"][metric_name] = {
            "query": resolved_query,
            "series_count": metric_summary["series_count"],
            "sample_count": metric_summary["sample_count"],
            "path": f"metrics/raw/{metric_name}.json",
            "prometheus_path": f"metrics/prometheus/{metric_name}.json",
        }
        detail(
            f"{metric_name}: {metric_summary['series_count']} series, "
            f"{metric_summary['sample_count']} samples"
        )

    (metrics_dir / "resolved_queries.json").write_text(
        json.dumps(query_metadata, indent=2), encoding="utf-8"
    )
    (metrics_dir / "archive_index.json").write_text(
        json.dumps(archive_index, indent=2), encoding="utf-8"
    )
    (metrics_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if summary["failures"]:
        raise CommandError(json.dumps(summary["failures"], indent=2))
    success(
        f"Collected metrics into {metrics_dir} "
        f"({len(summary['queries'])} successful quer{'y' if len(summary['queries']) == 1 else 'ies'})"
    )
    return metrics_dir
