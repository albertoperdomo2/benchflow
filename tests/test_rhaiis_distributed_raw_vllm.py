from __future__ import annotations

import unittest
from pathlib import Path

from benchflow.loaders import ProfileCatalog, load_experiment, load_run_plan_data
from benchflow.matrix import resolve_experiment_matrix
from benchflow.renderers.deployment import render_rhaiis_raw_vllm_manifests


REPO_ROOT = Path(__file__).resolve().parents[1]


def _kimi_plan():
    experiment = load_experiment(
        REPO_ROOT / "experiments/rhaiis/kimi-k3-tp8-dp4-ep32.yaml"
    )
    catalog = ProfileCatalog.load(REPO_ROOT / "profiles")
    plans = resolve_experiment_matrix(experiment, catalog)
    assert len(plans) == 1
    return plans[0]


class RhaiisDistributedRawVllmTest(unittest.TestCase):
    def test_raw_vllm_renders_tp4_pp2_on_eight_gpus(self) -> None:
        experiment = load_experiment(
            REPO_ROOT / "experiments/rhaiis/llama-33-70b-release.yaml"
        )
        plan = resolve_experiment_matrix(
            experiment, ProfileCatalog.load(REPO_ROOT / "profiles")
        )[0]
        plan.deployment.runtime.tensor_parallelism = 4
        plan.deployment.runtime.pipeline_parallelism = 2

        self.assertEqual(plan.deployment.runtime.tensor_parallelism, 4)
        self.assertEqual(plan.deployment.runtime.pipeline_parallelism, 2)
        restored = load_run_plan_data(plan.to_dict())
        self.assertEqual(restored.deployment.runtime.pipeline_parallelism, 2)

        manifests = render_rhaiis_raw_vllm_manifests(plan)
        deployment = next(
            manifest for manifest in manifests if manifest["kind"] == "Deployment"
        )
        self.assertEqual(deployment["spec"]["progressDeadlineSeconds"], 3600)
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        self.assertIn("--tensor-parallel-size=4", container["args"])
        self.assertIn("--pipeline-parallel-size=2", container["args"])
        self.assertEqual(container["resources"]["limits"]["nvidia.com/gpu"], "8")
        self.assertEqual(container["resources"]["requests"]["nvidia.com/gpu"], "8")

    def test_existing_raw_vllm_profile_keeps_deployment_shape(self) -> None:
        experiment = load_experiment(
            REPO_ROOT / "experiments/rhaiis/llama-33-70b-release.yaml"
        )
        plan = resolve_experiment_matrix(
            experiment, ProfileCatalog.load(REPO_ROOT / "profiles")
        )[0]

        manifests = render_rhaiis_raw_vllm_manifests(plan)

        self.assertEqual(
            [manifest["kind"] for manifest in manifests],
            ["Deployment", "Service", "ServiceMonitor"],
        )

    def test_kimi_profile_resolves_characterized_topology(self) -> None:
        plan = _kimi_plan()

        self.assertEqual(plan.deployment.platform, "rhaiis")
        self.assertEqual(plan.deployment.mode, "raw-vllm")
        self.assertEqual(plan.deployment.namespace, "benchflow")
        self.assertEqual(plan.deployment.runtime.replicas, 4)
        self.assertEqual(plan.deployment.runtime.tensor_parallelism, 8)
        self.assertEqual(
            plan.deployment.runtime.service_account_name,
            "benchflow-hostpath-runtime",
        )
        self.assertEqual(plan.deployment.options["model_path"], "/models")
        self.assertTrue(plan.deployment.options["distributed"]["enabled"])
        self.assertIn("--data-parallel-size=4", plan.deployment.runtime.vllm_args)
        self.assertIn("--enable-expert-parallel", plan.deployment.runtime.vllm_args)
        self.assertIn("--max-model-len=1048576", plan.deployment.runtime.vllm_args)
        self.assertFalse(plan.stages.download)

    def test_renderer_creates_ranked_statefulset_and_leader_service(self) -> None:
        plan = _kimi_plan()
        manifests = render_rhaiis_raw_vllm_manifests(plan)
        by_kind_name = {
            (manifest["kind"], manifest["metadata"]["name"]): manifest
            for manifest in manifests
        }
        workload_name = f"{plan.deployment.release_name}-vllm"
        headless_name = f"{workload_name}-headless"

        statefulset = by_kind_name[("StatefulSet", workload_name)]
        self.assertEqual(statefulset["spec"]["replicas"], 4)
        self.assertEqual(statefulset["spec"]["podManagementPolicy"], "Parallel")
        self.assertEqual(statefulset["spec"]["serviceName"], headless_name)

        pod_spec = statefulset["spec"]["template"]["spec"]
        self.assertEqual(
            pod_spec["serviceAccountName"],
            "benchflow-hostpath-runtime",
        )
        self.assertTrue(pod_spec["hostNetwork"])
        self.assertTrue(pod_spec["hostIPC"])
        self.assertEqual(pod_spec["dnsPolicy"], "ClusterFirstWithHostNet")
        anti_affinity = pod_spec["affinity"]["podAntiAffinity"]
        self.assertEqual(
            anti_affinity["requiredDuringSchedulingIgnoredDuringExecution"][0][
                "topologyKey"
            ],
            "kubernetes.io/hostname",
        )

        container = pod_spec["containers"][0]
        self.assertEqual(container["command"], ["/bin/sh", "-c"])
        self.assertIn("${HOSTNAME##*-}", container["args"][0])
        self.assertIn("--headless", container["args"][0])
        self.assertIn("--nnodes=4", container["args"])
        self.assertIn("--master-port=29500", container["args"])
        self.assertIn("--model=/models", container["args"])
        self.assertNotIn(
            "model-storage", {volume["name"] for volume in pod_spec["volumes"]}
        )
        self.assertEqual(container["resources"]["limits"]["nvidia.com/gpu"], "8")
        self.assertEqual(container["resources"]["limits"]["rdma/ib"], "1")

        headless = by_kind_name[("Service", headless_name)]
        self.assertEqual(headless["spec"]["clusterIP"], "None")
        self.assertTrue(headless["spec"]["publishNotReadyAddresses"])

        api_service = by_kind_name[("Service", plan.deployment.release_name)]
        self.assertEqual(api_service["spec"]["type"], "ExternalName")
        self.assertEqual(
            api_service["spec"]["externalName"],
            f"{workload_name}-0.{headless_name}.{plan.deployment.namespace}.svc.cluster.local",
        )

        monitor = by_kind_name[("ServiceMonitor", workload_name)]
        relabeling = monitor["spec"]["endpoints"][0]["relabelings"][0]
        self.assertEqual(relabeling["action"], "keep")
        self.assertEqual(relabeling["regex"], f"{workload_name}-0")


if __name__ == "__main__":
    unittest.main()
