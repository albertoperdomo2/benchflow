from __future__ import annotations

import unittest
from unittest.mock import patch

from benchflow.assets import render_yaml_documents

# Initialize the orchestration package before importing kueue, which is also
# imported by orchestration.service.
from benchflow.orchestration import tekton as _tekton  # noqa: F401
from benchflow.kueue import (
    SETUP_KEY_ANNOTATION,
    _cluster_active_setup_key,
    _workload_has_quota_reservation,
    _workload_is_admitted,
    _workload_needs_capacity_check,
    _workload_status_summary,
    run_remote_capacity_controller,
)


def _workload(*, admitted: bool, setup_key: str = "rhoai:raw-vllm") -> dict:
    conditions = []
    if admitted:
        conditions.append({"type": "Admitted", "status": "True"})
    return {
        "metadata": {
            "name": "queued-run-reservation",
            "creationTimestamp": "2026-08-27T10:00:00Z",
            "annotations": {SETUP_KEY_ANNOTATION: setup_key},
            "labels": {
                "benchflow.io/cluster-name": "local",
                "benchflow.io/execution-name": "queued-run",
                "benchflow.io/requested-gpus": "4",
                "benchflow.io/submission-configmap": "queued-run-submission",
            },
        },
        "status": {
            "admission": {"clusterQueue": "local"},
            "conditions": conditions,
        },
    }


class StopController(RuntimeError):
    pass


class KueueRemoteCapacityTest(unittest.TestCase):
    def test_quota_reservation_is_not_admission(self) -> None:
        reserved = _workload(admitted=False)

        self.assertTrue(_workload_has_quota_reservation(reserved))
        self.assertFalse(_workload_is_admitted(reserved))
        self.assertTrue(_workload_needs_capacity_check(reserved))
        self.assertEqual(_workload_status_summary(reserved)[0], "Queued")

    def test_admitted_workload_no_longer_needs_capacity_check(self) -> None:
        admitted = _workload(admitted=True)

        self.assertTrue(_workload_has_quota_reservation(admitted))
        self.assertTrue(_workload_is_admitted(admitted))
        self.assertFalse(_workload_needs_capacity_check(admitted))
        self.assertEqual(_workload_status_summary(admitted)[0], "Starting")

    def test_only_admitted_workloads_lock_the_setup_key(self) -> None:
        reserved = _workload(admitted=False, setup_key="rhoai:raw-vllm")
        admitted = _workload(admitted=True, setup_key="llm-d:default")

        self.assertEqual(_cluster_active_setup_key([reserved]), (None, ""))
        self.assertEqual(
            _cluster_active_setup_key([reserved, admitted]),
            ("llm-d:default", ""),
        )

    def test_reserved_workload_is_checked_without_creating_pipelinerun(self) -> None:
        reserved = _workload(admitted=False)

        with (
            patch("benchflow.kueue._patch_admission_check_active"),
            patch(
                "benchflow.kueue.list_reservation_workloads", return_value=[reserved]
            ),
            patch("benchflow.kueue._pipeline_run_payload", return_value=None),
            patch(
                "benchflow.kueue._submission_configmap_payload",
                return_value={"data": {"manifest.json": "{}"}},
            ),
            patch("benchflow.kueue.discover_cluster_gpu_capacity", return_value=8),
            patch("benchflow.kueue.discover_live_gpu_usage", return_value=0),
            patch("benchflow.kueue._patch_workload_check") as patch_check,
            patch("benchflow.kueue._create_execution_from_workload") as create,
            patch("benchflow.kueue.time.sleep", side_effect=StopController),
            self.assertRaises(StopController),
        ):
            run_remote_capacity_controller(namespace="benchflow")

        create.assert_not_called()
        self.assertEqual(patch_check.call_args.kwargs["state"], "Ready")

    def test_admitted_workload_creates_pipelinerun(self) -> None:
        admitted = _workload(admitted=True)

        with (
            patch("benchflow.kueue._patch_admission_check_active"),
            patch(
                "benchflow.kueue.list_reservation_workloads", return_value=[admitted]
            ),
            patch("benchflow.kueue._pipeline_run_payload", return_value=None),
            patch(
                "benchflow.kueue._submission_configmap_payload",
                return_value={"data": {"manifest.json": "{}"}},
            ),
            patch("benchflow.kueue._create_execution_from_workload") as create,
            patch("benchflow.kueue.time.sleep", side_effect=StopController),
            self.assertRaises(StopController),
        ):
            run_remote_capacity_controller(namespace="benchflow")

        create.assert_called_once_with("benchflow", admitted)

    def test_controller_rbac_covers_capacity_discovery(self) -> None:
        documents = render_yaml_documents(
            "bootstrap/operators/kueue/controller.yaml",
            {
                "BENCHFLOW_NAMESPACE": "benchflow",
                "BENCHFLOW_IMAGE": "example.invalid/benchflow:test",
                "BENCHFLOW_CONTROLLER_HOST_ALIASES": [],
            },
        )
        role = next(
            document for document in documents if document["kind"] == "ClusterRole"
        )
        rules = role["rules"]

        def verbs_for(api_group: str, resource: str) -> set[str]:
            return {
                verb
                for rule in rules
                if api_group in rule["apiGroups"] and resource in rule["resources"]
                for verb in rule["verbs"]
            }

        expected = {"get", "list", "watch"}
        self.assertGreaterEqual(verbs_for("", "nodes"), expected)
        self.assertGreaterEqual(verbs_for("", "pods"), expected)
        self.assertGreaterEqual(verbs_for("resource.k8s.io", "deviceclasses"), expected)
        self.assertGreaterEqual(
            verbs_for("resource.k8s.io", "resourceclaims"), expected
        )
        self.assertGreaterEqual(
            verbs_for("resource.k8s.io", "resourceslices"), expected
        )


if __name__ == "__main__":
    unittest.main()
