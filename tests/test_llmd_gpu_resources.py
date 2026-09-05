from pathlib import Path

import yaml

from benchflow.deploy.llmd import _patch_recipe_modelserver_overlay
from benchflow.loaders import ProfileCatalog, load_experiment
from benchflow.matrix import resolve_experiment_matrix


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_recipe_gpu_allocation_follows_tensor_parallelism(tmp_path: Path) -> None:
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        """apiVersion: benchflow.io/v1alpha1
kind: Experiment
metadata:
  name: llmd-gpu-resources-test
spec:
  model:
    name: Qwen/Qwen3.6-35B-A3B
  deployment_profile: llm-d-optimized-baseline-scalability
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
    plan.deployment.runtime.tensor_parallelism = 8
    plan.deployment.runtime.pipeline_parallelism = 1

    overlay_dir = tmp_path / "modelserver"
    overlay_dir.mkdir()
    (overlay_dir / "kustomization.yaml").write_text(
        "resources: []\nlabels: []\n",
        encoding="utf-8",
    )
    (overlay_dir / "patch-vllm.yaml").write_text(
        """apiVersion: apps/v1
kind: Deployment
metadata:
  name: decode
spec:
  template:
    spec:
      containers:
        - name: modelserver
          resources:
            requests:
              cpu: "8"
              memory: 96Gi
              nvidia.com/gpu: 2
            limits:
              cpu: "16"
              memory: 128Gi
              nvidia.com/gpu: 2
""",
        encoding="utf-8",
    )

    _patch_recipe_modelserver_overlay(plan, overlay_dir, router_chart=True)

    rendered_patch = yaml.safe_load(
        (overlay_dir / "patch-vllm.yaml").read_text(encoding="utf-8")
    )
    resources = rendered_patch["spec"]["template"]["spec"]["containers"][0]["resources"]
    assert resources["requests"] == {
        "cpu": "8",
        "memory": "96Gi",
        "nvidia.com/gpu": "8",
    }
    assert resources["limits"] == {
        "cpu": "16",
        "memory": "128Gi",
        "nvidia.com/gpu": "8",
    }
