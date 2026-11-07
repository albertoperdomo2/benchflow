from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from benchflow import lmcache_gpu as gpu
from benchflow.cluster import CommandError


def result(payload):
    return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")


def test_classic_gpu_access_does_not_change_operator(monkeypatch):
    calls = []

    def command(args, **_):
        calls.append(args)
        return result({"kind": "RuntimeClass", "metadata": {"name": "nvidia"}})

    monkeypatch.setattr(gpu, "run_command", command)
    assert gpu.ensure_lmcache_gpu_access("oc", "benchflow") == "runtime-class"
    assert len(calls) == 1


@pytest.fixture
def policy():
    return {
        "metadata": {
            "name": "gpu-policy",
            "resourceVersion": "1",
            "annotations": {"other": "keep"},
        },
        "spec": {
            "cdi": {"enabled": True, "nriPluginEnabled": True},
            "toolkit": {
                "version": "v1.20.0",
                "env": [
                    {"name": "OTHER", "value": "keep"},
                    {"name": gpu._ENV, "value": "existing"},
                ],
            },
        },
    }


def test_nri_namespace_reconciliation_preserves_other_configuration(
    monkeypatch, policy
):
    patches = []
    monkeypatch.setattr(
        gpu, "run_command", lambda args, **_: patches.append(json.loads(args[-1]))
    )
    original = deepcopy(policy)
    assert gpu._namespace_patch("oc", policy, "benchflow", add=True)
    patch = patches[0]
    assert patch[0] == {"op": "test", "path": "/metadata/resourceVersion", "value": "1"}
    toolkit = patch[-1]["value"]
    assert toolkit["version"] == "v1.20.0"
    assert {"name": "OTHER", "value": "keep"} in toolkit["env"]
    assert {"name": gpu._ENV, "value": "benchflow,existing"} in toolkit["env"]
    assert policy == original
    installed = deepcopy(policy)
    installed["metadata"]["annotations"] = patch[1]["value"]
    installed["spec"]["toolkit"] = toolkit
    assert not gpu._namespace_patch("oc", installed, "existing", add=False)
    assert gpu._namespace_patch("oc", installed, "benchflow", add=False)
    assert patches[-1][-1]["value"]["env"] == policy["spec"]["toolkit"]["env"]


def test_nri_setup_and_teardown_manage_only_owned_namespace(monkeypatch, policy):
    calls = []
    toolkit = {
        "metadata": {"name": "toolkit", "namespace": "gpu-operator"},
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {"image": "nvcr.io/nvidia/k8s/container-toolkit:v1.20.0-ubi9"}
                    ]
                }
            }
        },
    }
    monkeypatch.setattr(gpu, "_policy", lambda *_: policy)
    monkeypatch.setattr(gpu, "_toolkits", lambda *_: [toolkit])
    monkeypatch.setattr(
        gpu,
        "_wait_toolkit_namespace",
        lambda *a, **kw: calls.append((a[-1], kw["present"])),
    )

    def command(args, **_):
        if "runtimeclass" in args:
            return SimpleNamespace(stdout="", returncode=0, stderr="")
        patch = json.loads(args[-1])
        policy["metadata"]["annotations"] = patch[1]["value"]
        policy["spec"]["toolkit"] = patch[-1]["value"]
        return result({})

    monkeypatch.setattr(gpu, "run_command", command)
    assert gpu.ensure_lmcache_gpu_access("oc", "benchflow") == "nri"
    gpu.release_lmcache_gpu_access("oc", "benchflow")
    assert calls == [("benchflow", True), ("benchflow", False)]
    assert {"name": gpu._ENV, "value": "existing"} in policy["spec"]["toolkit"]["env"]


def test_old_toolkit_is_rejected_before_mutating_policy(monkeypatch, policy):
    monkeypatch.setattr(
        gpu, "run_command", lambda *_a, **_kw: SimpleNamespace(stdout="")
    )
    monkeypatch.setattr(gpu, "_policy", lambda *_: policy)
    monkeypatch.setattr(
        gpu,
        "_toolkits",
        lambda *_: [
            {
                "spec": {
                    "template": {"spec": {"containers": [{"image": "toolkit:v1.19.0"}]}}
                }
            }
        ],
    )
    with pytest.raises(CommandError, match=">=1.20.0"):
        gpu.ensure_lmcache_gpu_access("oc", "benchflow")
