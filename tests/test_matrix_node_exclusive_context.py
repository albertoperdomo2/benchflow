from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from benchflow.loaders import ProfileCatalog, load_experiment
from benchflow.matrix import resolve_experiment_matrix
from benchflow.orchestration.service import run_matrix_supervisor


REPO_ROOT = Path(__file__).resolve().parents[1]


def _plan():
    experiment = load_experiment(
        REPO_ROOT / "experiments/smoke/qwen3-06b-rhoai-distributed-default-smoke.yaml"
    )
    plan = resolve_experiment_matrix(
        experiment, ProfileCatalog.load(REPO_ROOT / "profiles")
    )[0]
    return replace(plan, stages=replace(plan.stages, cleanup=False))


def _render(plan, **_kwargs):
    return {
        "apiVersion": "tekton.dev/v1",
        "kind": "PipelineRun",
        "metadata": {"generateName": f"{plan.metadata.name}-", "labels": {}},
        "spec": {
            "params": [
                {
                    "name": "RUN_PLAN",
                    "value": json.dumps(plan.to_dict()),
                }
            ]
        },
    }


class MatrixNodeExclusiveContextTest(unittest.TestCase):
    @patch("benchflow.orchestration.service.summarize_execution")
    @patch("benchflow.orchestration.service.submit_execution_manifest")
    @patch(
        "benchflow.orchestration.service.render_execution_manifest", side_effect=_render
    )
    @patch("benchflow.orchestration.service.allocate_nodes")
    @patch(
        "benchflow.orchestration.service.secrets.token_hex",
        side_effect=["matrixscope", "aaaaaa", "bbbbbb"],
    )
    def test_children_are_scoped_before_target_allocation(
        self,
        _token_hex,
        allocate_nodes,
        _render_manifest,
        submit_execution,
        summarize_execution,
    ) -> None:
        plans = [_plan(), _plan()]
        allocate_nodes.side_effect = lambda plan, **_kwargs: plan
        submit_execution.side_effect = ["child-aaaaaa", "child-bbbbbb"]
        summarize_execution.return_value = {
            "status": "Succeeded",
            "finished": True,
            "succeeded": True,
            "message": "",
        }

        run_matrix_supervisor(
            plans,
            child_execution_name="benchflow-e2e",
            target_kubeconfig="/workspace/target-kubeconfig/config",
        )

        allocated = [call.args[0] for call in allocate_nodes.call_args_list]
        self.assertNotEqual(
            allocated[0].deployment.release_name,
            allocated[1].deployment.release_name,
        )
        for call in allocate_nodes.call_args_list:
            self.assertEqual(
                call.kwargs["kubeconfig"],
                "/workspace/target-kubeconfig/config",
            )

        submitted_names = [
            call.args[0]["metadata"]["name"] for call in submit_execution.call_args_list
        ]
        self.assertEqual(
            submitted_names,
            [
                f"{plans[0].metadata.name}-aaaaaa",
                f"{plans[1].metadata.name}-bbbbbb",
            ],
        )
        submitted_releases = []
        for call in submit_execution.call_args_list:
            params = call.args[0]["spec"]["params"]
            run_plan = next(item for item in params if item["name"] == "RUN_PLAN")
            submitted_releases.append(
                json.loads(run_plan["value"])["deployment"]["release_name"]
            )
        self.assertEqual(
            submitted_releases,
            [plan.deployment.release_name for plan in allocated],
        )


if __name__ == "__main__":
    unittest.main()
