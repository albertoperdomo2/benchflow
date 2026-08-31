from __future__ import annotations

from dataclasses import replace
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from benchflow.contracts import ValidationError
from benchflow.loaders import ProfileCatalog, load_experiment
from benchflow.matrix import resolve_experiment_matrix
from benchflow.orchestration.matrix_payloads import (
    RUN_PLANS_CHECKSUM_ANNOTATION,
    RUN_PLANS_COMPRESSED_CONFIGMAP_KEY,
    RUN_PLANS_CONFIGMAP_KEY,
    RUN_PLANS_ENCODING_ANNOTATION,
    _MAX_COMPRESSED_RUN_PLANS_BYTES,
    create_matrix_run_plans_configmap,
    decode_matrix_run_plans_configmap,
    write_matrix_run_plans_file,
)
from benchflow.orchestration.service import submit_execution_manifest
from benchflow.orchestration.tekton import render_matrix_pipelinerun


REPO_ROOT = Path(__file__).resolve().parents[1]


class MatrixPayloadConfigMapTest(unittest.TestCase):
    def test_create_compresses_and_round_trips_run_plans(self) -> None:
        run_plans_json = json.dumps(
            [{"kind": "RunPlan", "payload": "repeated-value" * 10_000}]
        )

        with patch(
            "benchflow.orchestration.matrix_payloads.create_manifest"
        ) as create_manifest:
            name = create_matrix_run_plans_configmap(
                namespace="benchflow",
                execution_name="large-matrix-a1b2c3",
                run_plans_json=run_plans_json,
            )

        payload = json.loads(create_manifest.call_args.args[0])
        self.assertEqual(name, "large-matrix-a1b2c3-run-plans")
        self.assertNotIn("data", payload)
        self.assertIn(
            RUN_PLANS_COMPRESSED_CONFIGMAP_KEY,
            payload["binaryData"],
        )
        self.assertEqual(
            payload["metadata"]["annotations"][RUN_PLANS_ENCODING_ANNOTATION],
            "gzip",
        )
        self.assertEqual(decode_matrix_run_plans_configmap(payload), run_plans_json)

    def test_decode_supports_legacy_uncompressed_configmaps(self) -> None:
        run_plans_json = '[{"kind":"RunPlan"}]'
        payload = {"data": {RUN_PLANS_CONFIGMAP_KEY: run_plans_json}}

        self.assertEqual(decode_matrix_run_plans_configmap(payload), run_plans_json)

    def test_decode_rejects_checksum_mismatch(self) -> None:
        run_plans_json = '[{"kind":"RunPlan"}]'
        with patch(
            "benchflow.orchestration.matrix_payloads.create_manifest"
        ) as create_manifest:
            create_matrix_run_plans_configmap(
                namespace="benchflow",
                execution_name="matrix-a1b2c3",
                run_plans_json=run_plans_json,
            )
        payload = json.loads(create_manifest.call_args.args[0])
        payload["metadata"]["annotations"][RUN_PLANS_CHECKSUM_ANNOTATION] = "0" * 64

        with self.assertRaisesRegex(ValidationError, "checksum does not match"):
            decode_matrix_run_plans_configmap(payload)

    def test_create_rejects_compressed_payload_over_safe_limit(self) -> None:
        with (
            patch(
                "benchflow.orchestration.matrix_payloads.gzip.compress",
                return_value=b"x" * (_MAX_COMPRESSED_RUN_PLANS_BYTES + 1),
            ),
            self.assertRaisesRegex(ValidationError, "safe ConfigMap payload limit"),
        ):
            create_matrix_run_plans_configmap(
                namespace="benchflow",
                execution_name="oversized-matrix",
                run_plans_json='[{"kind":"RunPlan"}]',
            )

    def test_write_fetches_verifies_and_materializes_payload(self) -> None:
        run_plans_json = '[{"kind":"RunPlan"}]'
        with patch(
            "benchflow.orchestration.matrix_payloads.create_manifest"
        ) as create_manifest:
            create_matrix_run_plans_configmap(
                namespace="benchflow",
                execution_name="matrix-a1b2c3",
                run_plans_json=run_plans_json,
            )
        payload = json.loads(create_manifest.call_args.args[0])

        with tempfile.TemporaryDirectory() as tmp:
            output_file = Path(tmp) / "run-plans.json"
            with (
                patch(
                    "benchflow.orchestration.matrix_payloads.require_any_command",
                    return_value="oc",
                ),
                patch(
                    "benchflow.orchestration.matrix_payloads.run_json_command",
                    return_value=payload,
                ) as run_json,
            ):
                write_matrix_run_plans_file(
                    namespace="benchflow",
                    configmap_name="matrix-a1b2c3-run-plans",
                    output_file=output_file,
                )

            self.assertEqual(output_file.read_text(), run_plans_json)
            run_json.assert_called_once_with(
                [
                    "oc",
                    "get",
                    "configmap",
                    "matrix-a1b2c3-run-plans",
                    "-n",
                    "benchflow",
                    "-o",
                    "json",
                ]
            )

    def test_matrix_task_uses_bflow_materializer(self) -> None:
        task_path = REPO_ROOT / "tekton/tasks/common/run-experiment-matrix.yaml"
        task = yaml.safe_load(task_path.read_text())
        script = task["spec"]["steps"][0]["args"][1]
        env = {entry["name"]: entry for entry in task["spec"]["steps"][0]["env"]}

        self.assertIn("bflow task materialize-matrix-run-plans", script)
        self.assertIn('--namespace "${BENCHFLOW_NAMESPACE}"', script)
        self.assertNotIn("--target-kubeconfig", script)
        self.assertNotIn("BENCHFLOW_TARGET_KUBECONFIG_WORKSPACE", script)
        self.assertNotIn("export KUBECONFIG", script)
        self.assertNotIn("context.pipelineRun.namespace", script)
        self.assertNotIn("jsonpath='{.data.run-plans", script)
        self.assertEqual(
            env["BENCHFLOW_NAMESPACE"]["valueFrom"]["fieldRef"]["fieldPath"],
            "metadata.namespace",
        )

    def test_remote_matrix_keeps_compatible_kubeconfig_workspace(self) -> None:
        experiment = load_experiment(
            REPO_ROOT
            / "experiments/llm-d/qwen36-35b-cephfs-offloading-scalability.yaml"
        )
        plan = resolve_experiment_matrix(
            experiment, ProfileCatalog.load(REPO_ROOT / "profiles")
        )[0]
        plan = replace(
            plan,
            target_cluster=replace(
                plan.target_cluster,
                kubeconfig_secret="psap-h100-diadochos",
            ),
        )

        manifest = render_matrix_pipelinerun(
            [plan],
            pipeline_name="benchflow-matrix",
            child_pipeline_name="benchflow-e2e",
        )
        self.assertEqual(
            manifest["spec"]["workspaces"],
            [
                {
                    "name": "target-kubeconfig",
                    "secret": {"secretName": "psap-h100-diadochos"},
                }
            ],
        )

        pipeline = yaml.safe_load(
            (REPO_ROOT / "tekton/pipelines/matrix.yaml").read_text()
        )
        self.assertIn(
            "target-kubeconfig",
            {workspace["name"] for workspace in pipeline["spec"]["workspaces"]},
        )

    def test_kueue_workload_receives_only_materialized_metadata(self) -> None:
        manifest = {
            "apiVersion": "tekton.dev/v1",
            "kind": "PipelineRun",
            "metadata": {
                "name": "large-matrix-a1b2c3",
                "labels": {
                    "benchflow.io/platform": "matrix",
                    "benchflow.io/mode": "matrix",
                    "benchflow.io/cluster-name": "psap-h100-diadochos",
                    "benchflow.io/requested-gpus": "16",
                    "benchflow.io/priority": "0",
                    "benchflow.io/kueue-skip-reservation": "false",
                },
                "annotations": {
                    "benchflow.io/run-plans-json": '[{"kind":"RunPlan"}]',
                    "benchflow.io/keep": "yes",
                },
            },
            "spec": {
                "pipelineRef": {"name": "benchflow-matrix"},
                "timeouts": {"pipeline": "8h"},
                "params": [{"name": "RUN_PLANS_CONFIGMAP", "value": ""}],
            },
        }

        with (
            patch(
                "benchflow.orchestration.matrix_payloads.create_matrix_run_plans_configmap",
                return_value="large-matrix-a1b2c3-run-plans",
            ),
            patch(
                "benchflow.orchestration.service.create_submission_configmap",
                return_value="large-matrix-a1b2c3-submission",
            ),
            patch(
                "benchflow.orchestration.service.create_reservation_workload",
                return_value="large-matrix-a1b2c3-reservation",
            ) as create_workload,
        ):
            execution_name = submit_execution_manifest(manifest, "benchflow")

        self.assertEqual(execution_name, "large-matrix-a1b2c3")
        workload_args = create_workload.call_args.kwargs
        self.assertEqual(
            workload_args["execution_annotations"],
            {"benchflow.io/keep": "yes"},
        )
        self.assertEqual(
            workload_args["execution_labels"]["benchflow.io/run-plans-configmap"],
            "large-matrix-a1b2c3-run-plans",
        )


if __name__ == "__main__":
    unittest.main()
