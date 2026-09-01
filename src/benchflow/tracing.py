from __future__ import annotations

import gzip
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import ProxyHandler, build_opener

import yaml

from .assets import render_yaml_documents
from .cluster import run_command
from .models import ResolvedRunPlan, TracingSpec
from .ui import detail, step, warning

_COLLECTOR_SERVICE = "benchflow-otel-collector"
_JAEGER_SERVICE = "benchflow-jaeger"
_TRACING_LABEL = "app.kubernetes.io/component=tracing"
_TRACE_FLUSH_DELAY_SECONDS = 5


def tracing_spec(plan: ResolvedRunPlan) -> TracingSpec:
    return plan.metrics.tracing


def tracing_enabled(plan: ResolvedRunPlan) -> bool:
    return tracing_spec(plan).enabled()


def _without_cli_flag(args: list[str], flag: str) -> list[str]:
    output: list[str] = []
    index = 0
    while index < len(args):
        item = str(args[index])
        if item == flag:
            index += 1
            if index < len(args) and not str(args[index]).startswith("--"):
                index += 1
            continue
        if item.startswith(f"{flag}="):
            index += 1
            continue
        output.append(item)
        index += 1
    return output


def vllm_tracing_args(plan: ResolvedRunPlan, args: list[str]) -> list[str]:
    if not tracing_enabled(plan):
        return args
    output = _without_cli_flag(args, "--otlp-traces-endpoint")
    output = _without_cli_flag(output, "--collect-detailed-traces")
    output.append(f"--otlp-traces-endpoint={otlp_endpoint(plan)}")
    if tracing_spec(plan).detailed():
        output.append("--collect-detailed-traces=all")
    return output


def routing_proxy_tracing_args(args: list[str]) -> list[str]:
    output = _without_cli_flag(args, "--tracing")
    output.append("--tracing=true")
    return output


def otlp_endpoint(plan: ResolvedRunPlan) -> str:
    namespace = plan.deployment.namespace
    return f"http://{_COLLECTOR_SERVICE}.{namespace}.svc.cluster.local:4317"


def tracing_service_name(plan: ResolvedRunPlan, role: str) -> str:
    return f"{plan.deployment.release_name}-{role}"


def tracing_environment(plan: ResolvedRunPlan, role: str) -> dict[str, str]:
    return {
        "OTEL_SERVICE_NAME": tracing_service_name(plan, role),
        "OTEL_EXPORTER_OTLP_ENDPOINT": otlp_endpoint(plan),
        "OTEL_TRACES_EXPORTER": "otlp",
        "OTEL_TRACES_SAMPLER": "parentbased_traceidratio",
        "OTEL_TRACES_SAMPLER_ARG": str(tracing_spec(plan).sample_ratio),
        "OTEL_RESOURCE_ATTRIBUTES": (
            f"benchflow.release={plan.deployment.release_name},"
            f"benchflow.experiment={plan.metadata.name},"
            f"benchflow.model={plan.model.resource_name},"
            f"benchflow.component.role={role}"
        ),
    }


def ensure_tracing_plane(plan: ResolvedRunPlan, kubectl_cmd: str) -> None:
    if not tracing_enabled(plan):
        return
    step(
        "Ensuring shared OpenTelemetry Collector and Jaeger tracing plane in "
        f"namespace {plan.deployment.namespace}"
    )
    documents = render_yaml_documents("setup/llmd/tracing.yaml", {})
    run_command(
        [kubectl_cmd, "apply", "-n", plan.deployment.namespace, "-f", "-"],
        input_text=yaml.safe_dump_all(documents, sort_keys=False),
    )
    for deployment in ("benchflow-otel-collector", "benchflow-jaeger"):
        run_command(
            [
                kubectl_cmd,
                "rollout",
                "status",
                f"deployment/{deployment}",
                "-n",
                plan.deployment.namespace,
                "--timeout=300s",
            ]
        )


def remove_tracing_plane(kubectl_cmd: str, namespace: str) -> None:
    run_command(
        [
            kubectl_cmd,
            "delete",
            "deployment,service,configmap",
            "-n",
            namespace,
            "-l",
            _TRACING_LABEL,
            "--ignore-not-found=true",
        ],
        check=False,
    )


def _parse_time_microseconds(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1_000_000)


def _jaeger_request(url: str, *, attempts: int = 5) -> dict[str, Any]:
    last_error: Exception | None = None
    opener = build_opener(ProxyHandler({}))
    for attempt in range(attempts):
        try:
            with opener.open(url, timeout=30) as response:  # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Jaeger returned a non-object response")
            return payload
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 8))
    assert last_error is not None
    raise last_error


def _normalized_spans(trace: dict[str, Any]) -> list[dict[str, Any]]:
    trace_id = str(trace.get("traceID") or "")
    processes = trace.get("processes") or {}
    output: list[dict[str, Any]] = []
    for span in trace.get("spans") or []:
        if not isinstance(span, dict):
            continue
        process = processes.get(span.get("processID"), {}) or {}
        references = span.get("references") or []
        parent_span_id = ""
        for reference in references:
            if isinstance(reference, dict) and reference.get("refType") == "CHILD_OF":
                parent_span_id = str(reference.get("spanID") or "")
                break
        start = int(span.get("startTime") or 0)
        duration = int(span.get("duration") or 0)
        tags = span.get("tags") or []
        tag_values = {
            str(tag.get("key") or ""): tag.get("value")
            for tag in tags
            if isinstance(tag, dict)
        }
        status = str(tag_values.get("otel.status_code") or "UNSET")
        if status == "UNSET" and tag_values.get("error") is True:
            status = "ERROR"
        output.append(
            {
                "trace_id": trace_id,
                "span_id": str(span.get("spanID") or ""),
                "parent_span_id": parent_span_id,
                "service_name": str(process.get("serviceName") or ""),
                "operation_name": str(span.get("operationName") or ""),
                "start_time_unix_microseconds": start,
                "end_time_unix_microseconds": start + duration,
                "duration_microseconds": duration,
                "status": status,
                "tags": tags,
                "logs": span.get("logs") or [],
                "process_tags": process.get("tags") or [],
            }
        )
    return output


def _query_service_traces(
    base_url: str,
    service: str,
    start: int,
    end: int,
    *,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    query = urlencode({"service": service, "start": start, "end": end, "limit": limit})
    payload = _jaeger_request(f"{base_url}/api/traces?{query}")
    traces = [item for item in (payload.get("data") or []) if isinstance(item, dict)]
    if len(traces) < limit or end - start <= 1_000_000:
        return traces
    midpoint = start + ((end - start) // 2)
    return [
        *_query_service_traces(base_url, service, start, midpoint, limit=limit),
        *_query_service_traces(base_url, service, midpoint + 1, end, limit=limit),
    ]


def collect_traces(
    plan: ResolvedRunPlan,
    *,
    artifacts_dir: Path,
    benchmark_start_time: str,
    benchmark_end_time: str,
) -> dict[str, Any]:
    if not tracing_enabled(plan):
        return {"enabled": False}

    trace_dir = artifacts_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "enabled": True,
        "status": "unavailable",
        "backend": "jaeger",
        "mode": tracing_spec(plan).mode,
        "sample_ratio": tracing_spec(plan).sample_ratio,
        "benchmark_start_time": benchmark_start_time,
        "benchmark_end_time": benchmark_end_time,
        "trace_count": 0,
        "span_count": 0,
        "services": [],
        "complete_multi_service_traces": 0,
    }
    try:
        if not benchmark_start_time or not benchmark_end_time:
            raise ValueError("benchmark start and end timestamps are required")
        base_url = f"http://{_JAEGER_SERVICE}.{plan.deployment.namespace}.svc.cluster.local:16686"
        # Components use batch span processors. Give their final benchmark
        # spans time to reach Jaeger before querying the completed window.
        time.sleep(_TRACE_FLUSH_DELAY_SECONDS)
        services_payload = _jaeger_request(f"{base_url}/api/services")
        all_services = [str(item) for item in services_payload.get("data") or []]
        release_prefix = f"{plan.deployment.release_name}-"
        services = sorted(
            service for service in all_services if service.startswith(release_prefix)
        )
        (trace_dir / "services.json").write_text(
            json.dumps({"services": services}, indent=2), encoding="utf-8"
        )
        start = _parse_time_microseconds(benchmark_start_time)
        end = _parse_time_microseconds(benchmark_end_time)
        traces_by_id: dict[str, dict[str, Any]] = {}
        for service in services:
            for trace in _query_service_traces(base_url, service, start, end):
                trace_id = str(trace.get("traceID") or "")
                if trace_id:
                    traces_by_id[trace_id] = trace

        spans = [
            span for trace in traces_by_id.values() for span in _normalized_spans(trace)
        ]
        with gzip.open(trace_dir / "traces.jsonl.gz", "wt", encoding="utf-8") as out:
            for span in spans:
                out.write(json.dumps(span, separators=(",", ":")) + "\n")
        complete = 0
        for trace in traces_by_id.values():
            trace_services = {
                str(process.get("serviceName") or "")
                for process in (trace.get("processes") or {}).values()
                if isinstance(process, dict)
            }
            if len(trace_services) > 1:
                complete += 1
        summary.update(
            {
                "status": "collected" if traces_by_id else "empty",
                "trace_count": len(traces_by_id),
                "span_count": len(spans),
                "services": services,
                "complete_multi_service_traces": complete,
            }
        )
        detail(
            f"Collected {len(traces_by_id)} distributed trace(s) and {len(spans)} span(s)"
        )
    except Exception as exc:  # noqa: BLE001
        summary["error"] = str(exc)
        warning(f"Distributed trace collection was unavailable: {exc}")
        if not (trace_dir / "services.json").exists():
            (trace_dir / "services.json").write_text(
                json.dumps({"services": []}, indent=2), encoding="utf-8"
            )
    (trace_dir / "trace-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary
