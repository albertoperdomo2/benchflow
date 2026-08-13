from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

from benchflow.loaders import ProfileCatalog, load_experiment
from benchflow.matrix import resolve_experiment_matrix
from benchflow.models import RuntimePVCMountSpec
from benchflow.orchestration.service import _materialize_execution_name
from benchflow.orchestration.tekton import render_pipelinerun
from benchflow.plans import scope_execution_release, scope_matrix_child_release


REPO_ROOT = Path(__file__).resolve().parents[1]


def _smoke_plan():
    experiment = load_experiment(
        REPO_ROOT / "experiments/smoke/qwen3-06b-rhoai-distributed-default-smoke.yaml"
    )
    catalog = ProfileCatalog.load(REPO_ROOT / "profiles")
    plans = resolve_experiment_matrix(experiment, catalog)
    assert len(plans) == 1
    return plans[0]


class ExecutionReleaseScopingTest(unittest.TestCase):
    def test_distinct_executions_get_distinct_releases(self) -> None:
        plan = _smoke_plan()

        first = scope_execution_release(plan, execution_name="qwen3-06b-a1b2c3")
        second = scope_execution_release(plan, execution_name="qwen3-06b-d4e5f6")

        self.assertNotEqual(first.deployment.release_name, plan.deployment.release_name)
        self.assertNotEqual(
            first.deployment.release_name, second.deployment.release_name
        )
        self.assertLessEqual(len(first.deployment.release_name), 42)
        self.assertEqual(
            first.deployment.target.resource_name, first.deployment.release_name
        )
        self.assertEqual(
            first.deployment.target.metrics_release_name,
            plan.deployment.target.metrics_release_name,
        )

    def test_rerun_replaces_the_execution_scope(self) -> None:
        plan = _smoke_plan()
        first = scope_execution_release(plan, execution_name="qwen3-06b-a1b2c3")

        rerun = scope_execution_release(first, execution_name="qwen3-06b-d4e5f6")

        self.assertNotEqual(
            rerun.deployment.release_name, first.deployment.release_name
        )
        self.assertTrue(
            rerun.deployment.release_name.startswith(plan.deployment.release_name)
        )
        self.assertNotIn(first.deployment.release_name, rerun.deployment.release_name)

    def test_created_runtime_pvcs_are_scoped_per_execution(self) -> None:
        plan = _smoke_plan()
        runtime = replace(
            plan.deployment.runtime,
            pvc_mounts=[
                RuntimePVCMountSpec(
                    name="kv-cache",
                    claim_name="vllm-kv-cache",
                    mount_path="/var/lib/vllm-kv-cache",
                    create=True,
                    storage_class_name="ocs-storagecluster-cephfs",
                    size="500Gi",
                    access_modes=["ReadWriteMany"],
                ),
                RuntimePVCMountSpec(
                    name="shared-models",
                    claim_name="shared-models",
                    mount_path="/models",
                    create=False,
                ),
            ],
        )
        plan = replace(
            plan,
            deployment=replace(plan.deployment, runtime=runtime),
        )

        first = scope_execution_release(plan, execution_name="qwen3-06b-a1b2c3")
        second = scope_execution_release(plan, execution_name="qwen3-06b-d4e5f6")
        rerun = scope_execution_release(first, execution_name="qwen3-06b-d4e5f6")

        first_claims = first.deployment.runtime.pvc_mounts
        second_claims = second.deployment.runtime.pvc_mounts
        rerun_claims = rerun.deployment.runtime.pvc_mounts
        self.assertNotEqual(first_claims[0].claim_name, second_claims[0].claim_name)
        self.assertEqual(second_claims[0].claim_name, rerun_claims[0].claim_name)
        self.assertTrue(first_claims[0].claim_name.startswith("vllm-kv-cache-"))
        self.assertEqual(first_claims[1].claim_name, "shared-models")
        self.assertEqual(second_claims[1].claim_name, "shared-models")

    def test_submission_materialization_scopes_the_serialized_run_plan(self) -> None:
        plan = _smoke_plan()
        manifest = render_pipelinerun(
            plan,
            pipeline_name="benchflow-e2e",
            setup_mode="auto",
        )
        manifest["metadata"]["name"] = "qwen3-06b-a1b2c3"
        manifest["metadata"].pop("generateName", None)

        rendered, execution_name = _materialize_execution_name(manifest)
        run_plan_json = next(
            param["value"]
            for param in rendered["spec"]["params"]
            if param["name"] == "RUN_PLAN"
        )
        scoped_release = json.loads(run_plan_json)["deployment"]["release_name"]

        self.assertEqual(execution_name, "qwen3-06b-a1b2c3")
        self.assertNotEqual(scoped_release, plan.deployment.release_name)
        self.assertEqual(
            rendered["metadata"]["labels"]["benchflow.io/execution-name"],
            execution_name,
        )

    def test_execution_scope_preserves_matrix_release_isolation(self) -> None:
        plan = _smoke_plan()
        matrix_plan = scope_matrix_child_release(
            plan,
            matrix_execution_name="qwen3-06b-matrix-a1b2c3",
        )

        scoped = scope_execution_release(
            matrix_plan,
            execution_name="qwen3-06b-child-d4e5f6",
        )

        self.assertNotEqual(
            scoped.deployment.release_name, plan.deployment.release_name
        )
        self.assertNotEqual(
            scoped.deployment.release_name,
            matrix_plan.deployment.release_name,
        )
        self.assertLessEqual(len(scoped.deployment.release_name), 42)


if __name__ == "__main__":
    unittest.main()
