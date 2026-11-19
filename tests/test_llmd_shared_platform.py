from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path

import pytest

from benchflow.cluster import CommandError
from benchflow.contracts import ExecutionContext
from benchflow.loaders import ProfileCatalog, load_experiment
from benchflow.matrix import resolve_experiment_matrix
from benchflow.setup import llmd
from benchflow.toolbox import platform


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def study_plan():
    return resolve_experiment_matrix(
        load_experiment(
            ROOT / "experiments/llm-d/epp-tracing/epp-tracing-v1-primary.yaml"
        ),
        ProfileCatalog.load(ROOT / "profiles"),
    )[0]


@pytest.fixture
def shared_cluster(monkeypatch):
    installed = {
        "installed_key": "rhoai:RHOAI-3.5.0",
        "setup_state": {"platform": "rhoai", "datasciencecluster_managed": True},
    }
    monkeypatch.setattr(platform, "require_any_command", lambda *_: "oc")
    monkeypatch.setattr(llmd, "require_any_command", lambda *_: "oc")
    monkeypatch.setattr(
        platform, "platform_prepare_lock", lambda *a, **kw: nullcontext()
    )
    monkeypatch.setattr(platform, "load_cluster_platform_state", lambda *_: installed)
    for name in (
        "_reset_platform_for_state",
        "persist_cluster_platform_state",
        "clear_cluster_platform_state",
        "setup_llmd",
        "setup_rhoai",
        "ensure_lmcache",
    ):
        monkeypatch.setattr(
            platform, name, lambda *a, **kw: pytest.fail("shared platform changed")
        )
    return installed


@pytest.mark.parametrize(
    "crds_ready,istio_ready", [(True, True), (False, True), (True, False)]
)
def test_setup_validates_shared_prerequisites_before_changing_services(
    study_plan, shared_cluster, monkeypatch, tmp_path, crds_ready, istio_ready
):
    original = deepcopy(shared_cluster)
    calls = []
    monkeypatch.setattr(llmd, "_gateway_dependencies_present", lambda *_: crds_ready)
    monkeypatch.setattr(
        llmd, "_istio_gateway_api_extension_enabled", lambda *_: istio_ready
    )
    monkeypatch.setattr(llmd, "run_command", lambda cmd, **kw: calls.append(cmd))
    monkeypatch.setattr(
        platform, "ensure_tracing_plane", lambda *_: calls.append("tracing")
    )
    state_path = tmp_path / "state.json"
    if crds_ready and istio_ready:
        state = platform.setup_platform(
            study_plan, context=ExecutionContext(state_path=state_path)
        )
        assert state["platform"] == "llm-d"
        assert not state["gateway_dependencies_managed"]
        assert not state["istio_releases_managed"]
        assert json.loads(state_path.read_text()) == state
        assert calls[-1] == "tracing"
        assert calls[0][:4] == [
            "oc",
            "wait",
            "--for=condition=available",
            "deployment/istiod",
        ]
    else:
        with pytest.raises(CommandError):
            platform.setup_platform(
                study_plan, context=ExecutionContext(state_path=state_path)
            )
        assert calls == []
        assert not state_path.exists()
    assert shared_cluster == original


def test_teardown_does_not_reset_the_other_installed_platform(
    study_plan, shared_cluster
):
    original = deepcopy(shared_cluster)
    platform.teardown_platform(study_plan, {})
    assert shared_cluster == original


@pytest.mark.parametrize(
    "phase,size", [("primary", 24), ("pprof", 2), ("sensitivity", 60)]
)
def test_study_matrices_preserve_protected_node_affinity(phase, size):
    plans = resolve_experiment_matrix(
        load_experiment(
            ROOT / f"experiments/llm-d/epp-tracing/epp-tracing-v1-{phase}.yaml"
        ),
        ProfileCatalog.load(ROOT / "profiles"),
    )
    assert len(plans) == size
    for plan in plans:
        assert plan.metrics.tracing.scope == "epp"
        terms = plan.deployment.runtime.affinity["nodeAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"
        ]["nodeSelectorTerms"]
        assert terms
        assert all(
            {
                "key": "kubernetes.io/hostname",
                "operator": "NotIn",
                "values": ["diadochos-hqxzk-gpu-h100-mt46x"],
            }
            in term["matchExpressions"]
            for term in terms
        )
