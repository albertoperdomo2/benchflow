from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from benchflow.reports.tracing import (
    _kernel_density,
    collect_trace_metrics,
    generate_trace_distribution_report,
)


def _write_trace_artifacts(artifacts_dir: Path) -> Path:
    trace_dir = artifacts_dir / "traces"
    trace_dir.mkdir(parents=True)
    (trace_dir / "trace-summary.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "status": "collected",
                "trace_count": 2,
                "span_count": 4,
                "complete_multi_service_traces": 2,
                "sample_ratio": 1.0,
                "benchmark_start_time": "2026-09-01T10:00:00Z",
                "benchmark_end_time": "2026-09-01T10:01:00Z",
                "services": ["release-epp", "release-vllm-modelserver"],
            }
        ),
        encoding="utf-8",
    )
    (artifacts_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model_name": "Qwen/Qwen3-0.6B",
                "platform": "rhoai",
                "version": "RHOAI-3.5",
                "mode": "distributed-default",
                "tp": 1,
                "replicas": 1,
            }
        ),
        encoding="utf-8",
    )
    spans = [
        {
            "service_name": "release-vllm-modelserver",
            "operation_name": "llm_request",
            "duration_microseconds": 40_000,
            "tags": [
                {"key": "gen_ai.latency.e2e", "type": "float64", "value": 0.04},
                {"key": "gen_ai.usage.prompt_tokens", "type": "int64", "value": 64},
            ],
        },
        {
            "service_name": "release-vllm-modelserver",
            "operation_name": "llm_request",
            "duration_microseconds": 60_000,
            "tags": [
                {"key": "gen_ai.latency.e2e", "type": "float64", "value": 0.06},
                {"key": "gen_ai.usage.prompt_tokens", "type": "int64", "value": 72},
            ],
        },
        {
            "service_name": "release-epp",
            "operation_name": "pick_endpoints",
            "duration_microseconds": 8,
            "tags": [
                {
                    "key": "llm_d.epp.picker.top_scores",
                    "type": "string",
                    "value": "[5.9]",
                }
            ],
        },
        {
            "service_name": "release-epp",
            "operation_name": "pick_endpoints",
            "duration_microseconds": 12,
            "tags": [
                {
                    "key": "llm_d.epp.picker.top_scores",
                    "type": "string",
                    "value": "[6.1]",
                }
            ],
        },
    ]
    trace_path = trace_dir / "traces.jsonl.gz"
    with gzip.open(trace_path, "wt", encoding="utf-8") as stream:
        for span in spans:
            stream.write(json.dumps(span) + "\n")
    return trace_path


def test_collect_trace_metrics_discovers_tags_and_operation_durations(
    tmp_path: Path,
) -> None:
    trace_path = _write_trace_artifacts(tmp_path)

    metrics = {metric.key: metric for metric in collect_trace_metrics(trace_path)}

    assert metrics["gen_ai.latency.e2e"].values == [40.0, 60.0]
    assert metrics["gen_ai.latency.e2e"].unit == "ms"
    assert metrics["gen_ai.usage.prompt_tokens"].values == [64.0, 72.0]
    assert metrics["llm_d.epp.picker.top_scores"].values == [5.9, 6.1]
    assert metrics["span.duration.release-epp.pick_endpoints"].values == [
        0.008,
        0.012,
    ]


def test_generate_trace_distribution_report_is_standalone_html(
    tmp_path: Path,
) -> None:
    _write_trace_artifacts(tmp_path)

    report_path = generate_trace_distribution_report(artifacts_dir=tmp_path)

    assert report_path == tmp_path / "reports" / "trace_distribution_report.html"
    report = report_path.read_text(encoding="utf-8")
    assert "Bench Flow Tracing Distribution Report" in report
    assert "End-to-end latency" in report
    assert "EPP selected endpoint score" in report
    assert "Observed distribution" in report
    assert "Gaussian kernel density" in report
    assert "Empirical cumulative distribution" in report
    assert "P50" in report
    assert "P95" in report
    assert "P99" in report
    assert "plotly.js" in report


def test_kernel_density_skips_constant_series() -> None:
    assert _kernel_density(np.asarray([4.0, 4.0, 4.0])) is None


def test_kernel_density_returns_curve_for_variable_series() -> None:
    density = _kernel_density(np.asarray([1.0, 2.0, 3.0, 4.0]))

    assert density is not None
    x_values, y_values = density
    assert len(x_values) == len(y_values) == 240
    assert np.all(y_values >= 0)


def test_generate_trace_distribution_report_skips_missing_traces(
    tmp_path: Path,
) -> None:
    assert generate_trace_distribution_report(artifacts_dir=tmp_path) is None
