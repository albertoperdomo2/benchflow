from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
import yaml

from benchflow.cluster import CommandError
from benchflow.deploy.llmd import (
    _apply_pd_proxy_tracing,
    _apply_vllm_tracing,
    _patch_scheduler_values,
)
from benchflow.loaders import (
    ProfileCatalog,
    load_experiment,
    load_metrics_profile,
    load_run_plan_data,
)
from benchflow.matrix import resolve_experiment_matrix
from benchflow.models import TracingSpec, ValidationError
from benchflow import tracing


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tracing_plan(tmp_path: Path):
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        """apiVersion: benchflow.io/v1alpha1
kind: Experiment
metadata:
  name: tracing-test
spec:
  model:
    name: Qwen/Qwen3.6-35B-A3B
  deployment_profile: llm-d-optimized-baseline-scalability
  benchmark_profile: aiperf-smoke
  metrics_profile: detailed-tracing
  namespace: benchflow
""",
        encoding="utf-8",
    )
    catalog = ProfileCatalog.load(REPO_ROOT / "profiles")
    return resolve_experiment_matrix(load_experiment(experiment), catalog)[0]


@pytest.mark.parametrize("mode", ["off", "standard", "detailed"])
def test_metrics_profile_loads_supported_tracing_modes(
    tmp_path: Path, mode: str
) -> None:
    path = tmp_path / "metrics.yaml"
    path.write_text(
        f"""apiVersion: benchflow.io/v1alpha1
kind: MetricsProfile
metadata:
  name: tracing-test
spec:
  prometheus_url: https://prometheus.example
  tracing:
    mode: {mode}
    sample_ratio: 0.25
""",
        encoding="utf-8",
    )

    profile = load_metrics_profile(path)

    assert profile.spec.tracing == TracingSpec(mode=mode, sample_ratio=0.25)


@pytest.mark.parametrize("ratio", [-0.1, 1.1, "nope", True])
def test_metrics_profile_rejects_invalid_sample_ratio(
    tmp_path: Path, ratio: object
) -> None:
    path = tmp_path / "metrics.yaml"
    path.write_text(
        "\n".join(
            [
                "apiVersion: benchflow.io/v1alpha1",
                "kind: MetricsProfile",
                "metadata:",
                "  name: tracing-test",
                "spec:",
                "  prometheus_url: https://prometheus.example",
                "  tracing:",
                "    mode: standard",
                f"    sample_ratio: {json.dumps(ratio)}",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="sample_ratio"):
        load_metrics_profile(path)


def test_vllm_detailed_tracing_is_release_scoped(tracing_plan) -> None:
    container = {
        "args": ["model", "--otlp-traces-endpoint=old", "--max-num-seqs=256"],
        "env": [{"name": "OTEL_SERVICE_NAME", "value": "old"}],
    }

    _apply_vllm_tracing(container, tracing_plan, role="vllm-decode")

    assert (
        "--otlp-traces-endpoint=http://benchflow-otel-collector."
        "benchflow.svc.cluster.local:4317" in container["args"]
    )
    assert "--collect-detailed-traces=all" in container["args"]
    env = {item["name"]: item["value"] for item in container["env"]}
    assert env["OTEL_SERVICE_NAME"] == (
        f"{tracing_plan.deployment.release_name}-vllm-decode"
    )
    assert env["OTEL_TRACES_SAMPLER"] == "parentbased_traceidratio"
    assert env["OTEL_TRACES_SAMPLER_ARG"] == "1.0"


def test_tracing_survives_run_plan_round_trip(tracing_plan) -> None:
    restored = load_run_plan_data(tracing_plan.to_dict())

    assert restored.metrics.tracing == TracingSpec(mode="detailed", sample_ratio=1.0)


def test_standard_tracing_omits_detailed_vllm_flag(tracing_plan) -> None:
    tracing_plan.metrics.tracing.mode = "standard"
    container = {"args": ["model"], "env": []}

    _apply_vllm_tracing(container, tracing_plan, role="vllm-decode")

    assert not any(
        item.startswith("--collect-detailed-traces") for item in container["args"]
    )


def test_router_tracing_values_enable_epp(tracing_plan, tmp_path: Path) -> None:
    values_path = tmp_path / "values.yaml"
    values_path.write_text("router: {}\n", encoding="utf-8")

    _patch_scheduler_values(
        tracing_plan, values_path, recipe_layout=True, router_chart=True
    )

    values = yaml.safe_load(values_path.read_text(encoding="utf-8"))
    router = values["router"]
    assert router["tracing"] == {
        "enabled": True,
        "otelExporterEndpoint": (
            "http://benchflow-otel-collector.benchflow.svc.cluster.local:4317"
        ),
        "sampling": {
            "sampler": "parentbased_traceidratio",
            "samplerArg": "1.0",
        },
    }
    env = {item["name"]: item["value"] for item in router["epp"]["env"]}
    assert env["OTEL_SERVICE_NAME"] == (f"{tracing_plan.deployment.release_name}-epp")


def test_router_values_allow_experimental_plugins(tracing_plan, tmp_path: Path) -> None:
    tracing_plan.deployment.options["epp_allow_experimental_plugins"] = True
    values_path = tmp_path / "values.yaml"
    values_path.write_text("router: {}\n", encoding="utf-8")

    _patch_scheduler_values(
        tracing_plan, values_path, recipe_layout=True, router_chart=True
    )

    values = yaml.safe_load(values_path.read_text(encoding="utf-8"))
    assert values["router"]["epp"]["flags"]["allow-experimental-plugins"] is True


def test_router_rejects_non_boolean_experimental_plugins_option(
    tracing_plan, tmp_path: Path
) -> None:
    tracing_plan.deployment.options["epp_allow_experimental_plugins"] = "true"
    values_path = tmp_path / "values.yaml"
    values_path.write_text("router: {}\n", encoding="utf-8")

    with pytest.raises(CommandError, match="must be a boolean"):
        _patch_scheduler_values(
            tracing_plan, values_path, recipe_layout=True, router_chart=True
        )


def test_pd_proxy_requires_flag_and_otlp_exporter(tracing_plan) -> None:
    container = {"args": ["--port=8000", "--tracing", "false"], "env": []}

    _apply_pd_proxy_tracing(container, tracing_plan)

    assert "--tracing=true" in container["args"]
    assert "--tracing" not in container["args"]
    assert "false" not in container["args"]
    env = {item["name"]: item["value"] for item in container["env"]}
    assert env["OTEL_TRACES_EXPORTER"] == "otlp"
    assert env["OTEL_SERVICE_NAME"].endswith("-routing-proxy")


def test_tracing_plane_is_applied_once_and_waited_ready(
    monkeypatch, tracing_plan
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run_command(args, **kwargs):
        calls.append((args, kwargs))
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(tracing, "run_command", fake_run_command)

    tracing.ensure_tracing_plane(tracing_plan, "oc")

    assert calls[0][0] == ["oc", "apply", "-n", "benchflow", "-f", "-"]
    assert "benchflow-otel-collector" in str(calls[0][1]["input_text"])
    assert "benchflow-jaeger" in str(calls[0][1]["input_text"])
    assert [call[0][2] for call in calls[1:]] == ["status", "status"]


def test_collect_traces_exports_deduplicated_spans(
    monkeypatch, tmp_path: Path, tracing_plan
) -> None:
    trace = {
        "traceID": "trace-1",
        "processes": {
            "p1": {"serviceName": f"{tracing_plan.deployment.release_name}-epp"},
            "p2": {
                "serviceName": f"{tracing_plan.deployment.release_name}-vllm-decode"
            },
        },
        "spans": [
            {
                "traceID": "trace-1",
                "spanID": "span-1",
                "processID": "p1",
                "operationName": "gateway.request",
                "startTime": 100,
                "duration": 20,
                "references": [],
            },
            {
                "traceID": "trace-1",
                "spanID": "span-2",
                "processID": "p2",
                "operationName": "llm_request",
                "startTime": 105,
                "duration": 10,
                "references": [{"refType": "CHILD_OF", "spanID": "span-1"}],
            },
        ],
    }

    def fake_request(url: str, **_kwargs):
        if url.endswith("/api/services"):
            return {
                "data": [
                    f"{tracing_plan.deployment.release_name}-epp",
                    f"{tracing_plan.deployment.release_name}-vllm-decode",
                    "another-release-epp",
                ]
            }
        return {"data": [trace]}

    monkeypatch.setattr(tracing, "_jaeger_request", fake_request)
    monkeypatch.setattr(tracing, "_TRACE_FLUSH_DELAY_SECONDS", 0)

    summary = tracing.collect_traces(
        tracing_plan,
        artifacts_dir=tmp_path,
        benchmark_start_time="2026-08-31T10:00:00Z",
        benchmark_end_time="2026-08-31T10:01:00Z",
    )

    assert summary["trace_count"] == 1
    assert summary["span_count"] == 2
    assert summary["complete_multi_service_traces"] == 1
    with gzip.open(tmp_path / "traces" / "traces.jsonl.gz", "rt") as stream:
        spans = [json.loads(line) for line in stream]
    assert {span["operation_name"] for span in spans} == {
        "gateway.request",
        "llm_request",
    }
