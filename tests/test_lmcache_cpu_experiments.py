from dataclasses import replace
import json
from pathlib import Path
import subprocess

import pytest

from benchflow.benchmark.aiperf import _build_command
from benchflow.lmcache import configure_lmcache, lmcache_documents, lmcache_spec
from benchflow.loaders import ProfileCatalog, load_experiment, load_run_plan_data
from benchflow.matrix import resolve_experiment_matrix
from benchflow.models import ValidationError
from benchflow.orchestration.tekton import render_pipelinerun
from benchflow.renderers.deployment import render_rhoai_manifest
from benchflow.toolbox import run_plan as run_plan_toolbox


ROOT = Path(__file__).resolve().parents[1]
IMAGE = "quay.io/benchflow-test/vllm-lmcache:provided"
QWEN = "Qwen/Qwen3.6-35B-A3B"
GEMMA = "google/gemma-4-31B-it"
NEMOTRON = "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1-FP8"
EXPERIMENTS = ROOT / "experiments/rhoai/offloading"


@pytest.fixture(scope="module")
def plans():
    catalog = ProfileCatalog.load(ROOT / "profiles")
    result = []
    for filename in ("lmcache-cpu-qwen36.yaml", "lmcache-cpu-256g.yaml"):
        experiment = load_experiment(EXPERIMENTS / filename)
        experiment.spec.overrides.images.runtime = IMAGE
        result.extend(resolve_experiment_matrix(experiment, catalog))
    return result


def test_only_the_three_historical_model_and_cpu_budget_pairs_are_generated(plans):
    expected = {
        QWEN: (2, 64, 0.55, 32, "200Gi"),
        GEMMA: (2, 256, 0.92, 8, "300Gi"),
        NEMOTRON: (8, 256, 0.8, 32, "300Gi"),
    }
    assert len(plans) == 3
    assert {plan.model.name for plan in plans} == set(expected)
    for plan in plans:
        tp, cpu_gib, utilization, concurrency, shared_memory = expected[plan.model.name]
        runtime = plan.deployment.runtime
        assert plan.deployment.platform_version == "RHOAI-3.5.0"
        assert plan.deployment.platform_channel == "stable-3.5"
        assert runtime.tensor_parallelism == tp
        assert runtime.replicas == runtime.pipeline_parallelism == 1
        assert runtime.shared_memory_size == shared_memory
        assert f"--gpu-memory-utilization={utilization}" in runtime.vllm_args
        assert runtime.image == IMAGE
        assert lmcache_spec(plan.deployment).cpu_gib == cpu_gib
        assert lmcache_spec(plan.deployment).verify_readback
        assert plan.benchmark.aiperf.args["concurrency"] == concurrency
        assert plan.execution.mode == "sequential"
        assert plan.stages.download and plan.stages.deploy and plan.stages.benchmark
        assert plan.stages.collect and plan.stages.cleanup
        assert not runtime.host_paths and not runtime.pvc_mounts
        assert not any("OffloadingConnector" in arg for arg in runtime.vllm_args)
        assert plan.profiles.metrics == "lmcache-cpu"


def test_agentx_command_pins_historical_corpus_seed_context_and_duration(
    plans, tmp_path
):
    for plan in plans:
        spec = plan.benchmark.aiperf
        command = _build_command(
            plan=plan,
            target="https://model.test",
            artifact_dir=tmp_path,
            dataset_path=tmp_path / "unused.jsonl",
            aiperf=spec,
        )
        expected = {
            "--public-dataset": "weka_hf",
            "--hf-weka-repo": "semianalysisai/cc-traces-weka-with-subagents-060826",
            "--random-seed": "42",
            "--benchmark-duration": "1800",
            "--max-context-length": "131072",
            "--scenario": "inferencex-agentx-mvp",
            "--trajectory-start-min-ratio": "0.25",
            "--trajectory-start-max-ratio": "0.75",
            "--trace-idle-gap-cap-seconds": "60",
        }
        for flag, value in expected.items():
            assert command[command.index(flag) + 1] == value
        assert "--streaming" in command
        assert "--use-server-token-count" in command
        assert "--hf-weka-dataset" not in command
        assert "--system-idle-gap-cap-seconds" not in command
        assert "--unsafe-override" not in command
        assert spec.dataset_cap is None
        assert not plan.benchmark.guidellm.pre_warmup.enabled


def test_model_cache_and_pipeline_rendering_survive_remote_run_plan_roundtrip(plans):
    for plan in plans:
        restored = load_run_plan_data(plan.to_dict())
        runtime = restored.deployment.runtime
        model = render_rhoai_manifest(restored)
        container = model["spec"]["template"]["containers"][0]
        assert container["image"] == IMAGE
        assert container["resources"]["limits"]["nvidia.com/gpu"] == str(
            runtime.tensor_parallelism
        )
        assert sum("LMCacheMPConnector" in arg for arg in container["args"]) == 1
        documents = {doc["kind"]: doc for doc in lmcache_documents(restored)}
        pod = documents["DaemonSet"]["spec"]["template"]["spec"]
        server = pod["containers"][0]
        assert pod["nodeSelector"] == {
            "kubernetes.io/hostname": "diadochos-hqxzk-gpu-h100-mt46x",
            "nvidia.com/gpu.present": "true",
        }
        assert server["image"] == IMAGE
        assert "nvidia.com/gpu" not in json.dumps(server["resources"])
        assert server["resources"]["requests"]["memory"] == (
            f"{lmcache_spec(restored.deployment).cpu_gib + 8}Gi"
        )
        assert ("--separate-object-groups" in server["args"]) == (
            restored.model.name == QWEN
        )
        if restored.model.name == QWEN:
            assert "--mamba-cache-mode=align" in container["args"]
        pipeline = render_pipelinerun(
            restored, pipeline_name="benchflow-e2e", setup_mode="auto"
        )
        assert json.loads(
            next(
                p["value"]
                for p in pipeline["spec"]["params"]
                if p["name"] == "RUN_PLAN"
            )
        )["stages"]["cleanup"]


def test_historical_reference_ids_are_preserved_in_mlflow_tags(plans):
    expected = {
        QWEN: "326fb9ac213b4a7facaf7a087f31e623",
        GEMMA: "0c12c93eb595463a922c9d492a8c4285",
        NEMOTRON: "17704f73746a417eba8cc6677570b5fc",
    }
    for plan in plans:
        tags = plan.mlflow.tags
        assert tags["comparison_scope"] == "historical-native-reference"
        assert tags["historical_vllm_version"] == "0.23.0"
        actual = (
            tags["historical_native_run"]
            if plan.model.name == QWEN
            else json.loads(tags["historical_native_runs"])[plan.model.name]
        )
        assert actual == expected[plan.model.name]


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--mamba-cache-mode=all"],
        ["--mamba-cache-mode=align"] * 2,
        ["--mamba-cache-mode=align", "--disable-hybrid-kv-cache-manager"],
    ],
)
def test_qwen_recurrent_cache_cannot_silently_use_an_unsupported_mode(plans, args):
    plan = next(plan for plan in plans if plan.model.name == QWEN)
    base_args = [
        arg
        for arg in plan.deployment.runtime.vllm_args
        if not arg.startswith("--mamba-cache-mode")
    ]
    deployment = replace(
        plan.deployment,
        runtime=replace(plan.deployment.runtime, vllm_args=base_args + args),
    )
    with pytest.raises(ValidationError, match="mamba-cache-mode=align"):
        configure_lmcache(deployment, model_name=QWEN)


@pytest.mark.parametrize(
    "filename", ["lmcache-cpu-qwen36.yaml", "lmcache-cpu-256g.yaml"]
)
def test_experiment_requires_a_supplied_serving_image(filename):
    with pytest.raises(ValidationError, match="supplied runtime.image"):
        resolve_experiment_matrix(
            load_experiment(EXPERIMENTS / filename),
            ProfileCatalog.load(ROOT / "profiles"),
        )


def test_wrong_agentx_runtime_is_rejected_before_stage_results_are_written(
    plans, monkeypatch, tmp_path
):
    from benchflow.benchmark import aiperf

    monkeypatch.setattr(aiperf, "require_command", lambda *_: None)
    monkeypatch.setattr(
        aiperf.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, "0.12.0\n", ""),
    )
    outputs = {
        name: tmp_path / name
        for name in (
            "stage_download_path",
            "stage_deploy_path",
            "stage_benchmark_path",
            "stage_collect_path",
            "stage_cleanup_path",
            "verify_completions_path",
        )
    }
    with pytest.raises(ValidationError, match="requires AIPerf 0.8.0"):
        run_plan_toolbox.resolve_run_plan_stages(plans[0], **outputs)
    assert not any(path.exists() for path in outputs.values())


def test_matching_agentx_runtime_passes_and_ordinary_profiles_skip_version_probe(
    plans, monkeypatch
):
    from benchflow.benchmark import aiperf

    calls = []
    monkeypatch.setattr(aiperf, "require_command", lambda *_: None)

    def version(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "0.8.0\n", "")

    monkeypatch.setattr(aiperf.subprocess, "run", version)
    aiperf.require_profile_aiperf_version(plans[0])
    assert calls == [["aiperf", "--version"]]
    ordinary = replace(plans[0], benchmark=replace(plans[0].benchmark, env={}))
    aiperf.require_profile_aiperf_version(ordinary)
    assert len(calls) == 1
