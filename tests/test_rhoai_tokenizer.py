from pathlib import Path

from benchflow.loaders import ProfileCatalog, load_experiment
from benchflow.matrix import resolve_experiment_matrix
from benchflow.renderers.deployment import render_rhoai_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_custom_rhoai_scheduler_uses_model_specific_tokenizer_path(
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
    tokenizer = manifest["spec"]["router"]["scheduler"]["tokenizer"]["template"]
    container = tokenizer["containers"][0]

    assert container["name"] == "main"
    assert "/mnt/models/base/models/Qwen-Qwen3-32B" in container["command"][2]
