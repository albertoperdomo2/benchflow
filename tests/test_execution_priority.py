from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchflow.orchestration.tekton import render_pipelinerun
from benchflow.kueue import (
    PRIORITY_LABEL,
    _workload_json,
    execution_labels_for_plan,
    priority_from_labels,
)
from benchflow.loaders import ProfileCatalog, load_experiment, load_run_plan_data
from benchflow.matrix import resolve_experiment_matrix
from benchflow.models import ValidationError


REPO_ROOT = Path(__file__).resolve().parents[1]


def _experiment_yaml(priority: object = 0) -> str:
    return f"""apiVersion: benchflow.io/v1alpha1
kind: Experiment
metadata:
  name: priority-smoke
spec:
  model:
    name: Qwen/Qwen3-0.6B
  deployment_profile: rhoai-distributed-default
  benchmark_profile: guidellm-smoke
  metrics_profile: detailed
  execution:
    priority: {priority}
  overrides:
    scale:
      replicas: [1, 2]
      tensor_parallelism: 1
"""


class ExecutionPriorityTest(unittest.TestCase):
    def test_priority_defaults_to_zero(self) -> None:
        experiment = load_experiment(
            REPO_ROOT
            / "experiments/smoke/qwen3-06b-rhoai-distributed-default-smoke.yaml"
        )

        self.assertEqual(experiment.spec.execution.priority, 0)

    def test_priority_survives_matrix_and_run_plan_serialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "experiment.yaml"
            path.write_text(_experiment_yaml(100), encoding="utf-8")
            plans = resolve_experiment_matrix(
                load_experiment(path), ProfileCatalog.load(REPO_ROOT / "profiles")
            )

        self.assertEqual(len(plans), 2)
        self.assertEqual({plan.execution.priority for plan in plans}, {100})
        restored = load_run_plan_data(plans[0].to_dict())
        self.assertEqual(restored.execution.priority, 100)

    def test_priority_reaches_pipelinerun_label_and_kueue_workload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "experiment.yaml"
            path.write_text(_experiment_yaml(500), encoding="utf-8")
            plan = resolve_experiment_matrix(
                load_experiment(path), ProfileCatalog.load(REPO_ROOT / "profiles")
            )[0]

        manifest = render_pipelinerun(
            plan,
            pipeline_name="benchflow-e2e",
            setup_mode="auto",
        )
        labels = manifest["metadata"]["labels"]
        self.assertEqual(labels[PRIORITY_LABEL], "500")
        self.assertEqual(execution_labels_for_plan(plan)[PRIORITY_LABEL], "500")
        self.assertEqual(priority_from_labels(labels), 500)

        workload = _workload_json(
            namespace="benchflow",
            cluster_name="target-cluster",
            execution_prefix="priority-smoke",
            execution_name="priority-smoke-a1b2c3",
            submission_configmap_name="priority-smoke-a1b2c3-submission",
            requested_gpu_count=2,
            priority=priority_from_labels(labels),
            max_execution_seconds=3600,
        )
        self.assertEqual(workload["spec"]["priority"], 500)

    def test_invalid_priorities_are_rejected(self) -> None:
        for priority in (-1, 2_147_483_648, 1.5, True, "fast"):
            with self.subTest(priority=priority), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "experiment.yaml"
                path.write_text(_experiment_yaml(priority), encoding="utf-8")
                with self.assertRaisesRegex(ValidationError, "execution.priority"):
                    load_experiment(path)


if __name__ == "__main__":
    unittest.main()
