from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from benchflow.loaders import (
    ProfileCatalog,
    _runtime_placement_from_dict,
    load_experiment,
)
from benchflow.deploy.llmd import _patch_values
from benchflow.matrix import resolve_experiment_matrix
from benchflow.models import RuntimePlacementSpec, ValidationError
from benchflow.placement import (
    PLACEMENT_POOL_NODE_LABEL,
    runtime_node_selector,
)
from benchflow.renderers.deployment import (
    render_llmd_values,
    render_rhaiis_raw_vllm_manifests,
    render_rhoai_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG = ProfileCatalog.load(REPO_ROOT / "profiles")


def _plan(experiment_path: str, *, pool: str = "two-h200-nodes"):
    plan = resolve_experiment_matrix(
        load_experiment(REPO_ROOT / experiment_path), CATALOG
    )[0]
    runtime = replace(
        plan.deployment.runtime,
        replicas=4,
        placement=RuntimePlacementSpec(
            mode="topology-spread",
            spread_pool=pool,
        ),
    )
    return replace(plan, deployment=replace(plan.deployment, runtime=runtime))


def test_loader_requires_pool_for_topology_spread() -> None:
    with pytest.raises(ValidationError, match="spread_pool"):
        _runtime_placement_from_dict({"mode": "topology-spread"}, "runtime.placement")


def test_node_selector_rejects_conflicting_pool() -> None:
    plan = _plan("experiments/smoke/qwen3-06b-llm-d-smoke.yaml")
    runtime = replace(
        plan.deployment.runtime,
        node_selector={PLACEMENT_POOL_NODE_LABEL: "another-pool"},
    )
    plan = replace(plan, deployment=replace(plan.deployment, runtime=runtime))

    with pytest.raises(ValidationError, match="conflicts"):
        runtime_node_selector(plan)


def test_llmd_values_render_strict_release_scoped_spread() -> None:
    plan = _plan("experiments/smoke/qwen3-06b-llm-d-smoke.yaml")

    runtime = render_llmd_values(plan)["runtime"]

    assert runtime["nodeSelector"] == {PLACEMENT_POOL_NODE_LABEL: "two-h200-nodes"}
    constraint = runtime["topologySpreadConstraints"][0]
    assert constraint["maxSkew"] == 1
    assert constraint["minDomains"] == 2
    assert constraint["topologyKey"] == "kubernetes.io/hostname"
    assert constraint["whenUnsatisfiable"] == "DoNotSchedule"
    assert constraint["labelSelector"]["matchLabels"] == {
        "llm-d.ai/inferenceServing": "true",
        "llm-d.ai/model": plan.deployment.release_name,
    }


def test_llmd_recipe_values_receive_topology_spread(tmp_path: Path) -> None:
    plan = _plan("experiments/smoke/qwen3-06b-llm-d-smoke.yaml")
    values_path = tmp_path / "values.yaml"
    values_path.write_text("decode: {}\n", encoding="utf-8")

    decode = _patch_values(plan, values_path)["decode"]

    assert decode["nodeSelector"] == {PLACEMENT_POOL_NODE_LABEL: "two-h200-nodes"}
    assert (
        decode["topologySpreadConstraints"][0]["labelSelector"]["matchLabels"][
            "llm-d.ai/model"
        ]
        == plan.deployment.release_name
    )


def test_rhoai_manifest_uses_generated_workload_identity() -> None:
    plan = _plan("experiments/smoke/qwen3-06b-rhoai-distributed-default-smoke.yaml")

    pod_spec = render_rhoai_manifest(plan)["spec"]["template"]

    assert pod_spec["nodeSelector"] == {PLACEMENT_POOL_NODE_LABEL: "two-h200-nodes"}
    constraint = pod_spec["topologySpreadConstraints"][0]
    assert constraint["matchLabelKeys"] == ["app.kubernetes.io/name"]
    assert constraint["labelSelector"]["matchLabels"] == {
        "app.kubernetes.io/part-of": "llminferenceservice",
        "kserve.io/component": "workload",
    }


def test_rhoai_isvc_manifest_uses_kserve_workload_identity() -> None:
    plan = _plan("experiments/smoke/qwen3-06b-rhoai-isvc-smoke.yaml")

    pod_spec = render_rhoai_manifest(plan)["spec"]["predictor"]

    assert pod_spec["nodeSelector"] == {PLACEMENT_POOL_NODE_LABEL: "two-h200-nodes"}
    constraint = pod_spec["topologySpreadConstraints"][0]
    assert constraint["matchLabelKeys"] == ["serving.kserve.io/inferenceservice"]
    assert constraint["labelSelector"] == {"matchLabels": {}}


def test_rhaiis_manifest_uses_release_scoped_spread() -> None:
    plan = _plan("experiments/smoke/qwen3-06b-rhaiis-raw-vllm-smoke.yaml")

    manifests = render_rhaiis_raw_vllm_manifests(plan)
    workload = next(
        manifest
        for manifest in manifests
        if manifest["kind"] in {"Deployment", "StatefulSet"}
    )
    pod_spec = workload["spec"]["template"]["spec"]

    assert pod_spec["nodeSelector"] == {PLACEMENT_POOL_NODE_LABEL: "two-h200-nodes"}
    constraint = pod_spec["topologySpreadConstraints"][0]
    assert constraint["labelSelector"]["matchLabels"] == {
        "app.kubernetes.io/component": "raw-vllm",
        "benchflow.io/release": plan.deployment.release_name,
    }
