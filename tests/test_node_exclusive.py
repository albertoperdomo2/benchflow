from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchflow.loaders import ProfileCatalog, _runtime_placement_from_dict, load_experiment
from benchflow.matrix import resolve_experiment_matrix
from benchflow.models import RuntimePlacementSpec, ValidationError
from benchflow.node_exclusive import allocate_nodes, release_nodes


REPO_ROOT = Path(__file__).resolve().parents[1]


def _plan():
    experiment = load_experiment(
        REPO_ROOT / "experiments/smoke/qwen3-06b-rhoai-distributed-default-smoke.yaml"
    )
    plans = resolve_experiment_matrix(experiment, ProfileCatalog.load(REPO_ROOT / "profiles"))
    return plans[0]


def _nodes():
    return {
        "items": [
            {
                "metadata": {"name": "gpu-a", "labels": {"benchflow.io/placement-pool": "h100"}},
                "status": {"conditions": [{"type": "Ready", "status": "True"}], "allocatable": {"nvidia.com/gpu": "8"}},
            },
            {
                "metadata": {"name": "gpu-b", "labels": {"benchflow.io/placement-pool": "h100"}},
                "status": {"conditions": [{"type": "Ready", "status": "True"}], "allocatable": {"nvidia.com/gpu": "8"}},
            },
        ]
    }


def _dra_nodes():
    payload = _nodes()
    payload["items"][0]["metadata"]["labels"]["nvidia.com/gpu.count"] = "8"
    payload["items"][0]["status"]["allocatable"]["nvidia.com/gpu"] = "0"
    payload["items"] = payload["items"][:1]
    return payload


class NodeExclusiveTest(unittest.TestCase):
    def test_loader_requires_pool(self) -> None:
        with self.assertRaisesRegex(ValidationError, "spread_pool"):
            _runtime_placement_from_dict({"mode": "node-exclusive"}, "runtime.placement")

    @patch("benchflow.node_exclusive.run_command")
    @patch("benchflow.node_exclusive.run_json_command", return_value=_nodes())
    @patch("benchflow.node_exclusive.require_any_command", return_value="oc")
    @patch("benchflow.node_exclusive.use_kubeconfig")
    def test_allocate_injects_hostname_affinity(
        self, use_kubeconfig, _, __, run_command
    ) -> None:
        plan = _plan()
        runtime = replace(
            plan.deployment.runtime,
            placement=RuntimePlacementSpec(mode="node-exclusive", spread_pool="h100"),
            replicas=4,
            tensor_parallelism=2,
        )
        plan = replace(plan, deployment=replace(plan.deployment, runtime=runtime))
        run_command.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")

        allocated = allocate_nodes(
            plan,
            timeout_seconds=1,
            kubeconfig="/workspace/target-kubeconfig/config",
        )

        terms = allocated.deployment.runtime.affinity["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"]
        self.assertEqual(terms[-1]["matchExpressions"][0]["values"], ["gpu-a"])
        self.assertEqual(run_command.call_count, 1)
        use_kubeconfig.assert_called_once_with("/workspace/target-kubeconfig/config")

    @patch("benchflow.node_exclusive.run_command")
    @patch("benchflow.node_exclusive.run_json_command", return_value=_dra_nodes())
    @patch("benchflow.node_exclusive.require_any_command", return_value="oc")
    def test_allocate_uses_dra_gpu_count_label(self, _, __, run_command) -> None:
        plan = _plan()
        runtime = replace(
            plan.deployment.runtime,
            placement=RuntimePlacementSpec(mode="node-exclusive", spread_pool="h100"),
            replicas=4,
            tensor_parallelism=2,
        )
        plan = replace(plan, deployment=replace(plan.deployment, runtime=runtime))
        run_command.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")

        allocated = allocate_nodes(plan, timeout_seconds=1)

        values = allocated.deployment.runtime.affinity["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][-1]["matchExpressions"][0]["values"]
        self.assertEqual(values, ["gpu-a"])

    @patch("benchflow.node_exclusive.run_command")
    @patch("benchflow.node_exclusive.require_any_command", return_value="oc")
    def test_release_deletes_release_leases(self, _, run_command) -> None:
        plan = _plan()
        runtime = replace(plan.deployment.runtime, placement=RuntimePlacementSpec(mode="node-exclusive", spread_pool="h100"))
        plan = replace(plan, deployment=replace(plan.deployment, runtime=runtime))

        release_nodes(plan)

        command = run_command.call_args.args[0]
        self.assertIn("lease", command)
        self.assertIn(f"benchflow.io/node-exclusive-release={plan.deployment.release_name}", command)


if __name__ == "__main__":
    unittest.main()
