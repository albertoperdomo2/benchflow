from __future__ import annotations

from pathlib import Path

import pytest

from benchflow.loaders import (
    ProfileCatalog,
    _runtime_pvc_mounts_from_dict,
    load_experiment,
)
from benchflow.matrix import resolve_experiment_matrix
from benchflow.models import ValidationError
from benchflow.renderers.autoscaling import render_scaled_object
from benchflow.renderers.deployment import (
    _runtime_pvc_volume_mounts,
    render_rhoai_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _queue_plan():
    experiment = load_experiment(
        REPO_ROOT / "experiments/smoke/qwen3-06b-rhoai-matrix-smoke.yaml"
    )
    experiment.spec.deployment_profile = [
        "rhoai-distributed-default-keda-queue"
    ]
    return resolve_experiment_matrix(
        experiment, ProfileCatalog.load(REPO_ROOT / "profiles")
    )[0]


def test_queue_profile_renders_scoped_scaled_object() -> None:
    plan = _queue_plan()
    manifest = render_scaled_object(plan)

    assert manifest is not None
    assert manifest["apiVersion"] == "keda.sh/v1alpha1"
    assert manifest["kind"] == "ScaledObject"
    assert manifest["metadata"]["namespace"] == plan.deployment.namespace
    assert (
        manifest["spec"]["scaleTargetRef"]["name"]
        == f"{plan.deployment.release_name}-kserve"
    )
    query = manifest["spec"]["triggers"][0]["metadata"]["query"]
    assert f'namespace="{plan.deployment.namespace}"' in query


def test_queue_profile_renders_pvc_sub_path() -> None:
    plan = _queue_plan()
    mount = plan.deployment.runtime.pvc_mounts[0]

    assert mount.sub_path == "torch_compile_cache"
    assert _runtime_pvc_volume_mounts(plan) == [
        {
            "name": "models-storage",
            "mountPath": "/tmp/vllm/torch_compile_cache",
            "subPath": "torch_compile_cache",
            "readOnly": False,
        }
    ]


def test_queue_profile_omits_llminferenceservice_replicas_when_unspecified() -> None:
    plan = _queue_plan()
    assert plan.deployment.runtime.replicas is None

    manifest = render_rhoai_manifest(plan)

    assert "replicas" not in manifest["spec"]


@pytest.mark.parametrize(
    "raw, message",
    [
        ("not-a-mapping", "must be a mapping"),
        ({"kind": "ScaledObject"}, "apiVersion"),
        ({"apiVersion": "keda.sh/v1alpha1"}, "kind"),
    ],
)
def test_scaled_object_validation(raw, message: str) -> None:
    plan = _queue_plan()
    plan.deployment.options["scaled_object"] = raw

    with pytest.raises(ValidationError, match=message):
        render_scaled_object(plan)


@pytest.mark.parametrize("sub_path", ["/cache", "cache/../other", "cache//other"])
def test_pvc_sub_path_must_be_relative(sub_path: str) -> None:
    with pytest.raises(ValidationError, match="sub_path must be a relative path"):
        _runtime_pvc_mounts_from_dict(
            [
                {
                    "name": "cache",
                    "claim_name": "cache-pvc",
                    "mount_path": "/tmp/cache",
                    "sub_path": sub_path,
                }
            ],
            "runtime.pvc_mounts",
        )
