from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from benchflow.loaders import ProfileCatalog, load_experiment
from benchflow.matrix import resolve_experiment_matrix
from benchflow.models import ValidationError
from benchflow.renderers.deployment import render_rhoai_manifest
from benchflow.toolbox import platform


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACING_SMOKE = (
    REPO_ROOT
    / "experiments/smoke/qwen3-06b-rhoai-distributed-default-tracing-smoke.yaml"
)


@pytest.fixture
def catalog() -> ProfileCatalog:
    return ProfileCatalog.load(REPO_ROOT / "profiles")


@pytest.fixture
def tracing_plan(catalog: ProfileCatalog):
    return resolve_experiment_matrix(load_experiment(TRACING_SMOKE), catalog)[0]


def _epp_config() -> dict:
    return {
        "apiVersion": "llm-d.ai/v1alpha1",
        "kind": "EndpointPickerConfig",
        "plugins": [
            {"type": "single-profile-handler"},
            {"type": "queue-scorer"},
            {"type": "kv-cache-utilization-scorer"},
            {"type": "prefix-cache-scorer"},
            {"type": "no-hit-lru-scorer"},
            {"type": "max-score-picker"},
            {
                "parameters": {"scheme": "https"},
                "type": "metrics-data-source",
            },
        ],
        "schedulingProfiles": [
            {
                "name": "default",
                "plugins": [
                    {"pluginRef": "queue-scorer", "weight": 2},
                    {"pluginRef": "kv-cache-utilization-scorer", "weight": 2},
                    {"pluginRef": "prefix-cache-scorer", "weight": 3},
                    {"pluginRef": "no-hit-lru-scorer", "weight": 2},
                    {"pluginRef": "max-score-picker"},
                ],
            }
        ],
    }


def test_rhoai_tracing_smoke_uses_blessed_profile(tracing_plan) -> None:
    assert tracing_plan.deployment.platform == "rhoai"
    assert tracing_plan.deployment.mode == "distributed-default"
    assert tracing_plan.deployment.platform_version == "RHOAI-3.5.0"
    assert tracing_plan.deployment.options["tracing_provider"] == "explicit-epp"
    assert tracing_plan.deployment.runtime.replicas == 1
    assert tracing_plan.deployment.runtime.tensor_parallelism == 1
    assert tracing_plan.metrics.tracing.mode == "detailed"
    assert tracing_plan.metrics.tracing.sample_ratio == 1.0


def test_rhoai_tracing_renders_exact_default_epp_config(tracing_plan) -> None:
    manifest = render_rhoai_manifest(tracing_plan)
    epp = manifest["spec"]["router"]["scheduler"]["template"]["containers"][0]
    args = epp["args"]

    assert "--tracing=true" in args
    config_index = args.index("--config-text")
    assert yaml.safe_load(args[config_index + 1]) == _epp_config()

    env = {entry["name"]: entry["value"] for entry in epp["env"]}
    assert env["OTEL_SERVICE_NAME"] == (f"{tracing_plan.deployment.release_name}-epp")
    assert env["OTEL_TRACES_EXPORTER"] == "otlp"
    assert env["OTEL_TRACES_SAMPLER"] == "parentbased_traceidratio"
    assert env["OTEL_TRACES_SAMPLER_ARG"] == "1.0"
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == (
        "http://benchflow-otel-collector.benchflow.svc.cluster.local:4317"
    )


def test_rhoai_tracing_instruments_vllm_model_server(tracing_plan) -> None:
    manifest = render_rhoai_manifest(tracing_plan)
    model_server = manifest["spec"]["template"]["containers"][0]

    assert (
        "--otlp-traces-endpoint=http://benchflow-otel-collector."
        "benchflow.svc.cluster.local:4317" in model_server["args"]
    )
    assert "--collect-detailed-traces=all" in model_server["args"]
    env = {entry["name"]: entry["value"] for entry in model_server["env"]}
    assert env["OTEL_SERVICE_NAME"] == (
        f"{tracing_plan.deployment.release_name}-vllm-modelserver"
    )
    assert env["OTEL_TRACES_SAMPLER_ARG"] == "1.0"


@pytest.mark.parametrize(
    "deployment_profile",
    ["rhoai-distributed-default", "rhoai-isvc"],
)
def test_rhoai_tracing_rejects_every_ordinary_profile(
    tmp_path: Path,
    catalog: ProfileCatalog,
    deployment_profile: str,
) -> None:
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        TRACING_SMOKE.read_text(encoding="utf-8").replace(
            "rhoai-distributed-default-tracing", deployment_profile
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValidationError,
        match="only by deployment profile rhoai-distributed-default-tracing",
    ):
        resolve_experiment_matrix(load_experiment(experiment), catalog)


def test_ordinary_rhoai_profile_remains_uninstrumented(catalog: ProfileCatalog) -> None:
    experiment = load_experiment(
        REPO_ROOT / "experiments/smoke/qwen3-06b-rhoai-distributed-default-smoke.yaml"
    )
    plan = resolve_experiment_matrix(experiment, catalog)[0]
    manifest = render_rhoai_manifest(plan)

    assert manifest["spec"]["router"]["scheduler"] == {}
    model_server = manifest["spec"]["template"]["containers"][0]
    assert not any("traces" in arg for arg in model_server["args"])
    assert not any(entry["name"].startswith("OTEL_") for entry in model_server["env"])


def test_rhoai_platform_reset_removes_shared_tracing_plane(
    monkeypatch, tracing_plan
) -> None:
    resets: list[str] = []
    removals: list[tuple[str, str]] = []

    monkeypatch.setattr(
        platform, "reset_rhoai_platform", lambda: resets.append("rhoai")
    )
    monkeypatch.setattr(platform, "require_any_command", lambda *_commands: "oc")
    monkeypatch.setattr(
        platform,
        "remove_tracing_plane",
        lambda command, namespace: removals.append((command, namespace)),
    )

    platform._reset_platform_for_state(tracing_plan, {"platform": "rhoai"})

    assert resets == ["rhoai"]
    assert removals == [("oc", "benchflow")]
