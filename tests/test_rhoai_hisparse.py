import json
from pathlib import Path

from benchflow.loaders import ProfileCatalog, load_experiment
from benchflow.matrix import resolve_experiment_matrix
from benchflow.renderers.deployment import render_rhoai_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_hisparse_matrix_renders_bounded_startup_and_shared_memory() -> None:
    plans = resolve_experiment_matrix(
        load_experiment(
            REPO_ROOT
            / "experiments/rhoai/hisparse/glm-53-hisparse-offload-matrix.yaml"
        ),
        ProfileCatalog.load(REPO_ROOT / "profiles"),
    )

    assert len(plans) == 6
    cpu_offload_cells = 0
    for plan in plans:
        runtime = plan.deployment.runtime
        assert runtime.replicas == 1
        assert runtime.tensor_parallelism == 8
        assert runtime.shared_memory_size == "80Gi"

        manifest = render_rhoai_manifest(plan)
        model_server = manifest["spec"]["template"]["containers"][0]
        assert model_server["startupProbe"] == {
            "failureThreshold": 540,
            "httpGet": {"path": "/health", "port": 8000, "scheme": "HTTPS"},
            "periodSeconds": 10,
            "timeoutSeconds": 10,
        }
        shared_memory = next(
            volume["emptyDir"]
            for volume in manifest["spec"]["template"]["volumes"]
            if volume["name"] == "dshm"
        )
        assert shared_memory == {"medium": "Memory", "sizeLimit": "80Gi"}

        transfer_arg = next(
            (
                arg
                for arg in model_server["args"]
                if arg.startswith("--kv-transfer-config=")
            ),
            None,
        )
        if transfer_arg and "cpu_bytes_to_use" in transfer_arg:
            cpu_offload_cells += 1
            transfer_config = json.loads(transfer_arg.split("=", 1)[1])
            assert "68719476736" in json.dumps(transfer_config)

    assert cpu_offload_cells == 4


def test_deploy_task_outlives_hisparse_startup_probe() -> None:
    import yaml

    pipeline = yaml.safe_load(
        (REPO_ROOT / "tekton/pipelines/e2e.yaml").read_text(encoding="utf-8")
    )
    tasks = {task["name"]: task for task in pipeline["spec"]["tasks"]}

    assert tasks["deploy"]["timeout"] == "2h"
