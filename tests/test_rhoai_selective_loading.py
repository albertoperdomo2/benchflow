from pathlib import Path

import yaml

from benchflow.loaders import ProfileCatalog, load_experiment
from benchflow.matrix import resolve_experiment_matrix
from benchflow.renderers.deployment import render_rhoai_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_rhoai_selective_loading_renders_distributed_default_epp(
    tmp_path: Path,
) -> None:
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        """apiVersion: benchflow.io/v1alpha1
kind: Experiment
metadata:
  name: rhoai-selective-loading-test
spec:
  model:
    name: nvidia/Llama-3_1-Nemotron-Ultra-253B-v1-FP8
  deployment_profile: rhoai-distributed-default-selective-loading
  benchmark_profile: aiperf-smoke
  metrics_profile: detailed
  namespace: benchflow
  overrides:
    images:
      runtime: quay.io/example/vllm:selective-loading
      scheduler: quay.io/example/epp:selective-loading
""",
        encoding="utf-8",
    )
    plan = resolve_experiment_matrix(
        load_experiment(experiment_path),
        ProfileCatalog.load(REPO_ROOT / "profiles"),
    )[0]

    assert plan.deployment.platform == "rhoai"
    assert plan.deployment.mode == "distributed-default"
    assert plan.deployment.platform_version == "RHOAI-3.5.0"
    assert plan.deployment.runtime.replicas == 1
    assert plan.deployment.runtime.tensor_parallelism == 8
    assert plan.deployment.runtime.shared_memory_size == "300Gi"
    assert plan.deployment.runtime.resources.requests == {"memory": "300Gi"}
    assert plan.deployment.runtime.resources.limits == {"memory": "400Gi"}

    manifest = render_rhoai_manifest(plan)
    scheduler = manifest["spec"]["router"]["scheduler"]["template"]["containers"][0]
    assert scheduler["image"] == "quay.io/example/epp:selective-loading"
    assert "--allow-experimental-plugins" in scheduler["args"]
    config_index = scheduler["args"].index("--config-text")
    config = yaml.safe_load(scheduler["args"][config_index + 1])
    assert config["kind"] == "EndpointPickerConfig"
    selective_policy = next(
        plugin
        for plugin in config["plugins"]
        if plugin["type"] == "selective-kv-policy"
    )
    assert selective_policy["parameters"] == {
        "engineCapability": "binary-opt-out-v1",
        "loadPolicy": "disable",
        "offloadPolicy": "preserve",
    }

    model_server = manifest["spec"]["template"]["containers"][0]
    assert model_server["image"] == "quay.io/example/vllm:selective-loading"
    assert model_server["resources"]["requests"]["nvidia.com/gpu"] == "8"
    assert model_server["resources"]["limits"]["nvidia.com/gpu"] == "8"
    assert any(arg.startswith("--kv-transfer-config=") for arg in model_server["args"])
