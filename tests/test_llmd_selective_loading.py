from pathlib import Path

import yaml

from benchflow.deploy.llmd import _patch_scheduler_values
from benchflow.loaders import ProfileCatalog, load_experiment
from benchflow.matrix import resolve_experiment_matrix


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_selective_loading_renders_optimized_baseline_epp_config(
    tmp_path: Path,
) -> None:
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        """apiVersion: benchflow.io/v1alpha1
kind: Experiment
metadata:
  name: selective-loading-test
spec:
  model:
    name: Qwen/Qwen3.6-35B-A3B
  deployment_profile: llm-d-optimized-baseline-selective-loading
  benchmark_profile: aiperf-smoke
  metrics_profile: detailed
  namespace: benchflow
""",
        encoding="utf-8",
    )
    plan = resolve_experiment_matrix(
        load_experiment(experiment_path),
        ProfileCatalog.load(REPO_ROOT / "profiles"),
    )[0]
    runtime = plan.deployment.runtime
    assert runtime.replicas == 1
    assert runtime.tensor_parallelism == 8
    assert runtime.shared_memory_size == "300Gi"
    assert runtime.resources.requests == {"memory": "300Gi"}
    assert runtime.resources.limits == {"memory": "400Gi"}
    assert "--gpu-memory-utilization=0.8" in runtime.vllm_args
    assert "--trust-remote-code" in runtime.vllm_args
    assert any(arg.startswith("--kv-transfer-config=") for arg in runtime.vllm_args)
    values_path = tmp_path / "values.yaml"
    values_path.write_text("router: {}\n", encoding="utf-8")

    _patch_scheduler_values(
        plan,
        values_path,
        recipe_layout=True,
        router_chart=True,
    )

    values = yaml.safe_load(values_path.read_text(encoding="utf-8"))
    epp = values["router"]["epp"]
    assert epp["flags"]["allow-experimental-plugins"] is True
    assert epp["pluginsConfigFile"] == "benchflow-epp-config.yaml"
    assert yaml.safe_load(epp["pluginsCustomConfig"]["benchflow-epp-config.yaml"]) == {
        "apiVersion": "llm-d.ai/v1alpha1",
        "kind": "EndpointPickerConfig",
        "plugins": [
            {"type": "queue-scorer"},
            {"type": "kv-cache-utilization-scorer"},
            {"type": "prefix-cache-scorer"},
            {"type": "no-hit-lru-scorer"},
            {
                "type": "selective-kv-policy",
                "name": "selective-kv-policy",
                "parameters": {
                    "engineCapability": "binary-opt-out-v1",
                    "loadPolicy": "disable",
                    "offloadPolicy": "preserve",
                },
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
                ],
            }
        ],
    }
