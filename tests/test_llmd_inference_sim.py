from pathlib import Path

import pytest
import yaml

from benchflow.deploy.llmd import _patch_recipe_modelserver_overlay

# Import the orchestration package before kueue to follow the package's existing
# import order and avoid its service/kueue initialization cycle in isolation.
from benchflow.orchestration.tekton import render_pipelinerun  # noqa: F401
from benchflow.kueue import requested_gpus
from benchflow.loaders import ProfileCatalog, load_experiment
from benchflow.matrix import resolve_experiment_matrix
from benchflow.models import ValidationError
from benchflow.runtime_images import image_repository_basename, is_inference_sim_image


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_EXPERIMENT = REPO_ROOT / "experiments/smoke/llm-d-inference-sim-smoke.yaml"


def _simulator_plan():
    return resolve_experiment_matrix(
        load_experiment(SMOKE_EXPERIMENT),
        ProfileCatalog.load(REPO_ROOT / "profiles"),
    )[0]


@pytest.mark.parametrize(
    ("image", "basename", "expected"),
    [
        (
            "ghcr.io/llm-d/llm-d-inference-sim:v0.11.2",
            "llm-d-inference-sim",
            True,
        ),
        (
            "registry.example/team/llm-d-inference-sim@sha256:abc",
            "llm-d-inference-sim",
            True,
        ),
        ("vllm/vllm-openai:v0.27.0", "vllm-openai", False),
    ],
)
def test_inference_sim_image_detection(
    image: str, basename: str, expected: bool
) -> None:
    assert image_repository_basename(image) == basename
    assert is_inference_sim_image(image) is expected


def test_inference_sim_plan_skips_download_and_gpu_reservation() -> None:
    plan = _simulator_plan()

    assert plan.stages.download is False
    assert plan.stages.deploy is True
    assert plan.mlflow.tags["runtime_kind"] == "inference-sim"
    assert plan.mlflow.tags["accelerator"] == "SIMULATED"
    assert requested_gpus(plan) == 0


def test_inference_sim_rejects_benchflow_managed_arguments(tmp_path: Path) -> None:
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        SMOKE_EXPERIMENT.read_text(encoding="utf-8")
        + """
  overrides:
    runtime:
      vllm_extra_args:
        - --port=9000
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="BenchFlow manages.*--port"):
        resolve_experiment_matrix(
            load_experiment(experiment_path),
            ProfileCatalog.load(REPO_ROOT / "profiles"),
        )


def test_recipe_overlay_replaces_vllm_with_inference_sim(tmp_path: Path) -> None:
    plan = _simulator_plan()
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
          command: [vllm, serve]
          env:
            - name: HF_TOKEN
              value: unused
            - name: VLLM_CPU_KVCACHE_SPACE
              value: "32"
          resources:
            limits:
              cpu: "64"
              memory: 64Gi
          volumeMounts:
            - name: dshm
              mountPath: /dev/shm
      volumes:
        - name: dshm
          emptyDir: {}
""",
        encoding="utf-8",
    )

    _patch_recipe_modelserver_overlay(plan, overlay_dir, router_chart=True)

    patch = yaml.safe_load(
        (overlay_dir / "patch-vllm.yaml").read_text(encoding="utf-8")
    )
    pod_spec = patch["spec"]["template"]["spec"]
    container = pod_spec["containers"][0]
    assert container["command"] == []
    assert container["image"] == plan.deployment.runtime.image
    assert container["args"][:6] == [
        "--model",
        "Qwen/Qwen3-0.6B",
        "--served-model-name",
        "Qwen/Qwen3-0.6B",
        "--port",
        "8000",
    ]
    assert "--time-to-first-token=20ms" in container["args"]
    assert {entry["name"] for entry in container["env"]} == {
        "POD_NAME",
        "POD_NAMESPACE",
        "POD_IP",
    }
    assert container["resources"] == {
        "requests": {"cpu": "500m", "memory": "256Mi"},
        "limits": {"cpu": "2", "memory": "1Gi"},
    }
    assert container["volumeMounts"] == []
    assert pod_spec["volumes"] == []
    assert container["startupProbe"]["httpGet"]["path"] == "/health/ready"
    assert container["readinessProbe"]["httpGet"]["path"] == "/health/ready"
    assert container["livenessProbe"]["httpGet"]["path"] == "/health"
