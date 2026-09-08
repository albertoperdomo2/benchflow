from pathlib import Path

from benchflow.loaders import ProfileCatalog, load_experiment
from benchflow.matrix import resolve_experiment_matrix
from benchflow.renderers.deployment import render_rhoai_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_rhoai_model_uri_mounts_model_for_server_and_tokenizer(
    tmp_path: Path,
) -> None:
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        """apiVersion: benchflow.io/v1alpha1
kind: Experiment
metadata:
  name: rhoai-tokenizer-path-test
spec:
  model:
    name: Qwen/Qwen3-32B
  deployment_profile: rhoai-distributed-default-selective-loading
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

    manifest = render_rhoai_manifest(plan)
    model_server = manifest["spec"]["template"]["containers"][0]

    assert manifest["spec"]["model"]["uri"] == (
        "pvc://models-storage/models/Qwen-Qwen3-32B"
    )
    assert "--model=/mnt/models" in model_server["args"]
    assert "tokenizer" not in manifest["spec"]["router"]["scheduler"]
