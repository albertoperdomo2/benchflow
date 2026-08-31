from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

from benchflow.loaders import ProfileCatalog, load_experiment
from benchflow.llmd_layout import recipe_gateway_name
from benchflow.matrix import resolve_experiment_matrix
from benchflow.models import RuntimePVCMountSpec
from benchflow.orchestration.service import _materialize_execution_name
from benchflow.orchestration.tekton import render_pipelinerun
from benchflow.plans import (
    _target_for,
    reset_matrix_child_release_for_rerun,
    scope_execution_release,
    scope_matrix_child_release,
)


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

    def test_scoping_updates_release_embedded_target_resource_name(self) -> None:
        plan = _smoke_plan()
        target = replace(
            plan.deployment.target,
            resource_name=(f"infra-{plan.deployment.release_name}-inference-gateway"),
        )
        plan = replace(plan, deployment=replace(plan.deployment, target=target))

        matrix_plan = scope_matrix_child_release(
            plan,
            matrix_execution_name="qwen3-06b-matrix-a1b2c3",
        )
        scoped = scope_execution_release(
            matrix_plan,
            execution_name="qwen3-06b-child-d4e5f6",
        )

        self.assertEqual(
            scoped.deployment.target.resource_name,
            f"infra-{scoped.deployment.release_name}-inference-gateway",
        )

    def test_llmd_gateway_target_uses_recipe_name_for_long_releases(self) -> None:
        release_name = "qwen36-35b-offloading-scalability-1cd6491028"

        target = _target_for(
            "llm-d",
            "optimized-baseline",
            release_name,
            "benchflow",
            "istio",
            "/v1/models",
            "v0.9.0",
            "external",
        )

        self.assertEqual(target.resource_name, recipe_gateway_name(release_name))
        self.assertEqual(
            target.resource_name,
            "infra-qwen36-35b-offloading-1cd6491028-inference-gateway",
        )

    def test_execution_scoping_rebuilds_truncated_llmd_gateway_target(self) -> None:
        release_name = "qwen36-35b-offloading-scalability-m1"
        target = _target_for(
            "llm-d",
            "optimized-baseline",
            release_name,
            "benchflow",
            "istio",
            "/v1/models",
            "v0.9.0",
            "external",
        )
        base_plan = _smoke_plan()
        plan = replace(
            base_plan,
            deployment=replace(
                base_plan.deployment,
                platform="llm-d",
                release_name=release_name,
                target=target,
            ),
        )

        scoped = scope_execution_release(
            plan,
            execution_name="qwen36-35b-offloading-scalability-m1-a1b2c3",
        )

        self.assertEqual(
            scoped.deployment.target.resource_name,
            recipe_gateway_name(scoped.deployment.release_name),
        )

    def test_matrix_rerun_resets_scoped_llmd_release(self) -> None:
        release_name = "qwen36-35b-offloading-scalability-m1"
        target = _target_for(
            "llm-d",
            "optimized-baseline",
            release_name,
            "benchflow",
            "istio",
            "/v1/models",
            "v0.9.0",
            "external",
        )
        base_plan = _smoke_plan()
        plan = replace(
            base_plan,
            deployment=replace(
                base_plan.deployment,
                platform="llm-d",
                release_name=release_name,
                target=target,
            ),
        )
        original_matrix_plan = scope_matrix_child_release(
            plan,
            matrix_execution_name="qwen36-35b-matrix-original-a1b2c3",
        )
        recorded_child = scope_execution_release(
            original_matrix_plan,
            execution_name="qwen36-35b-child-original-d4e5f6",
        )

        reset_plan = reset_matrix_child_release_for_rerun(recorded_child)
        rerun_plan = scope_matrix_child_release(
            reset_plan,
            matrix_execution_name="qwen36-35b-matrix-rerun-0f1e2d",
        )

        self.assertEqual(reset_plan.deployment.release_name, release_name)
        self.assertEqual(
            reset_plan.deployment.target.resource_name,
            recipe_gateway_name(release_name),
        )
        self.assertNotEqual(
            rerun_plan.deployment.release_name,
            original_matrix_plan.deployment.release_name,
        )
        self.assertEqual(
            rerun_plan.deployment.target.resource_name,
            recipe_gateway_name(rerun_plan.deployment.release_name),
        )


if __name__ == "__main__":
    unittest.main()
