from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from benchflow.loaders import (
    ProfileCatalog,
    _runtime_placement_from_dict,
    load_experiment,
)
from benchflow.matrix import resolve_experiment_matrix
from benchflow.models import RuntimePlacementSpec, ValidationError
from benchflow.node_exclusive import (
    NODE_EXCLUSIVE_RELEASE_LABEL,
    release_nodes,
    reserve_nodes,
)
from benchflow.orchestration.service import _materialize_execution_name
from benchflow.toolbox.platform import cleanup_deployment


REPO_ROOT = Path(__file__).resolve().parents[1]


def _plan(*, replicas: int = 4):
    experiment = load_experiment(
        REPO_ROOT / "experiments/smoke/qwen3-06b-rhoai-distributed-default-smoke.yaml"
    )
    plan = resolve_experiment_matrix(
        experiment, ProfileCatalog.load(REPO_ROOT / "profiles")
    )[0]
    runtime = replace(
        plan.deployment.runtime,
        placement=RuntimePlacementSpec(mode="node-exclusive", spread_pool="h100"),
        replicas=replicas,
        tensor_parallelism=2,
    )
    return replace(
        plan,
        deployment=replace(plan.deployment, runtime=runtime),
    )


def _nodes(*, dra: bool = False):
    items = []
    for name in ("gpu-a", "gpu-b"):
        gpu_count = "0" if dra else "8"
        labels = {"benchflow.io/placement-pool": "h100"}
        if dra:
            labels["nvidia.com/gpu.count"] = "8"
        items.append(
            {
                "metadata": {"name": name, "labels": labels},
                "spec": {},
                "status": {
                    "conditions": [{"type": "Ready", "status": "True"}],
                    "allocatable": {"nvidia.com/gpu": gpu_count},
                },
            }
        )
    return {"items": items}


def _cluster_payload(command, *, nodes=None, leases=None, pods=None):
    if "nodes" in command:
        return nodes or _nodes()
    if "lease" in command:
        return {"items": leases or []}
    if "pods" in command:
        return {"items": pods or []}
    raise AssertionError(command)


class NodeExclusiveTest(unittest.TestCase):
    def test_loader_requires_pool(self) -> None:
        with self.assertRaisesRegex(ValidationError, "spread_pool"):
            _runtime_placement_from_dict(
                {"mode": "node-exclusive"}, "runtime.placement"
            )

    @patch("benchflow.node_exclusive.run_command")
    @patch("benchflow.node_exclusive.run_json_command")
    @patch("benchflow.node_exclusive.require_any_command", return_value="oc")
    @patch("benchflow.node_exclusive.use_kubeconfig")
    def test_reserve_injects_hostname_affinity(
        self, use_kubeconfig, _, run_json, run_command
    ) -> None:
        plan = _plan()
        plan = replace(
            plan,
            target_cluster=replace(
                plan.target_cluster,
                kubeconfig="/workspace/target/kubeconfig",
            ),
        )
        run_json.side_effect = lambda command: _cluster_payload(command)
        run_command.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")

        allocated = reserve_nodes(plan)

        self.assertIsNotNone(allocated)
        terms = allocated.deployment.runtime.affinity["nodeAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"
        ]["nodeSelectorTerms"]
        self.assertEqual(terms[-1]["matchExpressions"][0]["values"], ["gpu-a"])
        self.assertEqual(run_command.call_count, 1)
        use_kubeconfig.assert_called_once_with("/workspace/target/kubeconfig")

    @patch("benchflow.node_exclusive.run_command")
    @patch("benchflow.node_exclusive.run_json_command")
    @patch("benchflow.node_exclusive.require_any_command", return_value="oc")
    def test_reservation_constrains_every_existing_affinity_term(
        self, _, run_json, run_command
    ) -> None:
        plan = _plan()
        runtime = replace(
            plan.deployment.runtime,
            affinity={
                "nodeAffinity": {
                    "requiredDuringSchedulingIgnoredDuringExecution": {
                        "nodeSelectorTerms": [
                            {
                                "matchExpressions": [
                                    {"key": "zone", "operator": "In", "values": ["a"]}
                                ]
                            },
                            {
                                "matchExpressions": [
                                    {"key": "zone", "operator": "In", "values": ["b"]}
                                ]
                            },
                        ]
                    }
                }
            },
        )
        plan = replace(plan, deployment=replace(plan.deployment, runtime=runtime))
        run_json.side_effect = lambda command: _cluster_payload(command)
        run_command.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")

        allocated = reserve_nodes(plan)

        self.assertIsNotNone(allocated)
        terms = allocated.deployment.runtime.affinity["nodeAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"
        ]["nodeSelectorTerms"]
        self.assertEqual(len(terms), 2)
        for term in terms:
            hostname = next(
                expression
                for expression in term["matchExpressions"]
                if expression["key"] == "kubernetes.io/hostname"
            )
            self.assertEqual(hostname["values"], ["gpu-a"])

    @patch("benchflow.node_exclusive.run_command")
    @patch("benchflow.node_exclusive.run_json_command")
    @patch("benchflow.node_exclusive.require_any_command", return_value="oc")
    def test_reserve_uses_dra_gpu_count_label(self, _, run_json, run_command) -> None:
        run_json.side_effect = lambda command: _cluster_payload(
            command, nodes=_nodes(dra=True)
        )
        run_command.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")

        allocated = reserve_nodes(_plan())

        self.assertIsNotNone(allocated)
        values = allocated.deployment.runtime.affinity["nodeAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"
        ]["nodeSelectorTerms"][-1]["matchExpressions"][0]["values"]
        self.assertEqual(values, ["gpu-a"])

    @patch("benchflow.node_exclusive.run_command")
    @patch("benchflow.node_exclusive.run_json_command")
    @patch("benchflow.node_exclusive.require_any_command", return_value="oc")
    def test_multi_node_reservation(self, _, run_json, run_command) -> None:
        run_json.side_effect = lambda command: _cluster_payload(command)
        run_command.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")

        allocated = reserve_nodes(_plan(replicas=8))

        self.assertIsNotNone(allocated)
        values = allocated.deployment.runtime.affinity["nodeAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"
        ]["nodeSelectorTerms"][-1]["matchExpressions"][0]["values"]
        self.assertEqual(values, ["gpu-a", "gpu-b"])
        self.assertEqual(run_command.call_count, 2)

    @patch("benchflow.node_exclusive.run_command")
    @patch("benchflow.node_exclusive.run_json_command")
    @patch("benchflow.node_exclusive.require_any_command", return_value="oc")
    def test_partial_reservation_rolls_back(self, _, run_json, run_command) -> None:
        nodes = _nodes()
        nodes["items"] = nodes["items"][:1]
        run_json.side_effect = lambda command: _cluster_payload(command, nodes=nodes)
        run_command.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")

        allocated = reserve_nodes(_plan(replicas=8))

        self.assertIsNone(allocated)
        delete = run_command.call_args_list[-1].args[0]
        self.assertEqual(delete[1:3], ["delete", "lease"])

    @patch("benchflow.node_exclusive.run_command")
    @patch("benchflow.node_exclusive.run_json_command")
    @patch("benchflow.node_exclusive.require_any_command", return_value="oc")
    def test_gpu_occupied_node_is_skipped(self, _, run_json, run_command) -> None:
        pods = [
            {
                "metadata": {},
                "status": {"phase": "Running"},
                "spec": {
                    "nodeName": "gpu-a",
                    "containers": [
                        {"resources": {"requests": {"nvidia.com/gpu": "1"}}}
                    ],
                },
            }
        ]
        run_json.side_effect = lambda command: _cluster_payload(command, pods=pods)
        run_command.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")

        allocated = reserve_nodes(_plan())

        self.assertIsNotNone(allocated)
        values = allocated.deployment.runtime.affinity["nodeAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"
        ]["nodeSelectorTerms"][-1]["matchExpressions"][0]["values"]
        self.assertEqual(values, ["gpu-b"])

    @patch("benchflow.node_exclusive.run_command")
    @patch("benchflow.node_exclusive.require_any_command", return_value="oc")
    def test_release_deletes_release_leases(self, _, run_command) -> None:
        plan = _plan()

        release_nodes(plan)

        command = run_command.call_args.args[0]
        self.assertIn("lease", command)
        self.assertIn(
            f"benchflow.io/node-exclusive-release={plan.deployment.release_name}",
            command,
        )

    def test_materialized_child_labels_its_scoped_release(self) -> None:
        plan = _plan()
        manifest = {
            "metadata": {"generateName": "exclusive-child-", "labels": {}},
            "spec": {
                "params": [{"name": "RUN_PLAN", "value": json.dumps(plan.to_dict())}]
            },
        }

        with patch(
            "benchflow.orchestration.service.secrets.token_hex",
            return_value="a1b2c3",
        ):
            materialized, execution_name = _materialize_execution_name(manifest)

        run_plan = json.loads(materialized["spec"]["params"][0]["value"])
        release = run_plan["deployment"]["release_name"]
        self.assertEqual(execution_name, "exclusive-child-a1b2c3")
        self.assertEqual(
            materialized["metadata"]["labels"][NODE_EXCLUSIVE_RELEASE_LABEL],
            release,
        )
        self.assertNotEqual(release, plan.deployment.release_name)

    @patch("benchflow.toolbox.platform.release_nodes")
    @patch("benchflow.toolbox.platform.cleanup_rhoai", side_effect=RuntimeError("boom"))
    @patch("benchflow.toolbox.platform.use_kubeconfig")
    def test_cleanup_releases_nodes_when_platform_cleanup_fails(
        self, _use_kubeconfig, _cleanup_rhoai, release
    ) -> None:
        plan = _plan()

        with self.assertRaisesRegex(RuntimeError, "boom"):
            cleanup_deployment(
                plan,
                wait_for_deletion=True,
                timeout_seconds=30,
                skip_if_not_exists=False,
            )

        release.assert_called_once_with(plan)


if __name__ == "__main__":
    unittest.main()
