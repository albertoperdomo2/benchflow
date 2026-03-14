from __future__ import annotations

import json
import math
import ssl
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from ..cluster import CommandError
from ..models import ResolvedRunPlan
from ..ui import detail, step, success, warning


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
    metrics_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    step(
        f"Collecting Prometheus metrics for {plan.deployment.release_name} "
        f"from {benchmark_start_time} to {benchmark_end_time}"
    )
    detail(f"Prometheus URL: {plan.metrics.prometheus_url}")
    detail(
        f"Query step: {plan.metrics.query_step}, timeout: {plan.metrics.query_timeout}, "
        f"TLS verification: {'enabled' if plan.metrics.verify_tls else 'disabled'}"
    )

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
    summary: dict[str, object] = {
        "namespace": plan.deployment.namespace,
        "release_name": plan.deployment.release_name,
        "prometheus_url": plan.metrics.prometheus_url,
        "verify_tls": plan.metrics.verify_tls,
        "benchmark_start_time": start_time.isoformat().replace("+00:00", "Z"),
        "benchmark_end_time": end_time.isoformat().replace("+00:00", "Z"),
        "query_step": plan.metrics.query_step,
        "query_timeout": plan.metrics.query_timeout,
        "queries": {},
        "failures": {},
    }
    detail(
        f"Collecting {len(queries)} metrics quer{'y' if len(queries) == 1 else 'ies'}"
    )

    for metric_name, query_template in sorted(queries.items()):
        resolved_query = query_template.replace(
            "$namespace", plan.deployment.namespace
        ).replace("$release", plan.deployment.release_name)
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
        detail(
            f"{metric_name}: {metric_summary['series_count']} series, "
            f"{metric_summary['sample_count']} samples"
        )

    (metrics_dir / "resolved_queries.json").write_text(
        json.dumps(query_metadata, indent=2), encoding="utf-8"
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
