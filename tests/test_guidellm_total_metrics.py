from __future__ import annotations

from typing import Any

from benchflow.benchmark.processor.processor import BenchmarkProcessor
from benchflow.benchmark.run_report_insights import summarize_benchmarks
from benchflow.benchmark.runtime import extract_metrics_from_benchmark


def _metric_group(total: float, successful: float) -> dict[str, dict[str, Any]]:
    def values(value: float) -> dict[str, Any]:
        return {
            "count": 1,
            "mean": value,
            "median": value,
            "min": value,
            "max": value,
            "total_sum": value,
            "percentiles": {
                "p01": value,
                "p05": value,
                "p50": value,
                "p90": value,
                "p95": value,
                "p99": value,
                "p999": value,
            },
        }

    return {"total": values(total), "successful": values(successful)}


def _benchmark() -> dict[str, Any]:
    metrics = {
        name: _metric_group(200.0 + index, 100.0 + index)
        for index, name in enumerate(
            (
                "requests_per_second",
                "tokens_per_second",
                "output_tokens_per_second",
                "request_concurrency",
                "request_latency",
                "time_to_first_token_ms",
                "inter_token_latency_ms",
                "time_per_output_token_ms",
                "prompt_token_count",
                "output_token_count",
            )
        )
    }
    metrics["request_totals"] = {
        "total": 10,
        "successful": 8,
        "incomplete": 1,
        "errored": 1,
    }
    return {
        "duration": 60,
        "config": {
            "run_id": "total-metrics-test",
            "strategy": {"streams": [4]},
        },
        "scheduler_metrics": {
            "queued_time_avg": 0.0,
            "request_targeted_start_delay_avg": 0.0,
            "requests_made": {
                "total": 10,
                "successful": 8,
                "incomplete": 1,
                "errored": 1,
            },
        },
        "metrics": metrics,
        "requests": {"successful": [], "incomplete": [], "errored": []},
    }


def test_mlflow_metric_extraction_uses_only_total_aggregates() -> None:
    extracted = extract_metrics_from_benchmark(_benchmark())

    assert extracted["throughput_requests_per_sec"] == 200.0
    assert extracted["total_tokens_per_second"] == 201.0
    assert extracted["throughput_output_tokens_per_sec"] == 202.0
    assert extracted["request_concurrency_mean"] == 203.0
    assert extracted["latency_p99_sec"] == 204.0
    assert extracted["ttft_p95_ms"] == 205.0
    assert extracted["itl_mean_ms"] == 206.0
    assert extracted["tpot_p99_ms"] == 207.0
    assert extracted["total_input_tokens"] == 208.0
    assert extracted["total_output_tokens"] == 209.0


def test_comparison_report_processor_uses_only_total_aggregates() -> None:
    processor = BenchmarkProcessor.__new__(BenchmarkProcessor)
    processor.accelerator = "H100"
    processor.model_name = "test-model"
    processor.version = "test-version"
    processor.tp_size = 1
    processor.runtime_args = ""
    processor.replicas = 1
    processor.data_profile = {"prompt_tokens": 128, "output_tokens": 64}
    processor.prompt_tokens = 128
    processor.output_tokens = 64

    row = processor.process_benchmark_section(_benchmark(), 0)

    assert row["measured rps"] == 200.0
    assert row["output_tok/sec"] == 202.0
    assert row["total_tok/sec"] == 201.0
    assert row["ttft_p95"] == 205.0
    assert row["itl_mean"] == 206.0
    assert row["tpot_p99"] == 207.0
    assert row["prompt_token_count_mean"] == 208.0
    assert row["output_token_count_mean"] == 209.0


def test_run_report_insights_use_only_total_aggregates() -> None:
    row = summarize_benchmarks(
        [_benchmark()],
        strict_slo=(500.0, 50.0),
        relaxed_slo=(1000.0, 100.0),
        gpu_count=1,
    )[0]

    assert row["raw_success_rps"] == 200.0
    assert row["successful_output_toksps"] == 202.0
    assert row["successful_total_toksps"] == 201.0
    assert row["successful_prompt_toksps"] == 200.0 * 208.0
    assert row["ttft_p95_ms"] == 205.0
    assert row["itl_p99_ms"] == 206.0
