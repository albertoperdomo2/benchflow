from __future__ import annotations

import json
import unittest
from pathlib import Path

from benchflow.loaders import ProfileCatalog, load_experiment
from benchflow.matrix import resolve_experiment_matrix


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments" / "llm-d"
REPLICAS = {1, 2, 4, 8}
CONCURRENCIES = {4, 8, 16, 32, 64, 128, 256, 512, 1024}


class LlmdOffloadingScalabilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = ProfileCatalog.load(REPO_ROOT / "profiles")

    def _plans(self, filename: str):
        experiment = load_experiment(EXPERIMENTS / filename)
        return resolve_experiment_matrix(experiment, self.catalog)

    def test_baseline_and_cpu_matrix(self) -> None:
        plans = self._plans("qwen36-35b-offloading-scalability.yaml")

        self.assertEqual(len(plans), 2 * len(REPLICAS) * len(CONCURRENCIES))
        self.assertEqual(
            {plan.profiles.deployment for plan in plans},
            {
                "llm-d-optimized-baseline-scalability",
                "llm-d-optimized-baseline-scalability-cpu-offload",
            },
        )
        self._assert_common_matrix(plans, placement_mode="")
        for plan in plans:
            kv_args = [
                arg
                for arg in plan.deployment.runtime.vllm_args
                if arg.startswith("--kv-transfer-config=")
            ]
            if plan.profiles.deployment.endswith("cpu-offload"):
                self.assertEqual(plan.deployment.runtime.shared_memory_size, "300Gi")
                self.assertEqual(len(kv_args), 1)
                self.assertIn('"cpu_bytes_to_use":274877906944', kv_args[0])
                self.assertNotIn("secondary_tiers", kv_args[0])
            else:
                self.assertEqual(kv_args, [])

    def test_cephfs_matrix(self) -> None:
        plans = self._plans("qwen36-35b-cephfs-offloading-scalability.yaml")

        self.assertEqual(len(plans), len(REPLICAS) * len(CONCURRENCIES))
        self._assert_common_matrix(plans, placement_mode="sequential")
        for plan in plans:
            self.assertEqual(
                plan.profiles.deployment,
                "llm-d-optimized-baseline-scalability-cephfs-offload",
            )
            mount = plan.deployment.runtime.pvc_mounts[0]
            self.assertTrue(mount.create)
            self.assertEqual(mount.storage_class_name, "rook-cephfs-fast")
            self.assertEqual(mount.access_modes, ["ReadWriteMany"])
            self.assertEqual(mount.mount_path, "/mnt/files-storage")
            kv_arg = self._kv_transfer_arg(plan)
            self.assertIn('"cpu_bytes_to_use":274877906944', kv_arg)
            self.assertIn('"type":"fs"', kv_arg)
            self._assert_native_chunk_size(kv_arg)
            self.assertEqual(plan.deployment.runtime.shared_memory_size, "300Gi")

    def test_nvme_matrix(self) -> None:
        plans = self._plans("qwen36-35b-nvme-offloading-scalability.yaml")

        self.assertEqual(len(plans), len(REPLICAS) * len(CONCURRENCIES))
        self._assert_common_matrix(plans, placement_mode="node-exclusive")
        self.assertEqual(
            {plan.deployment.runtime.placement.spread_pool for plan in plans},
            {"h100-benchflow"},
        )
        for plan in plans:
            self.assertEqual(
                plan.profiles.deployment,
                "llm-d-optimized-baseline-scalability-nvme-offload",
            )
            mount = plan.deployment.runtime.host_paths[0]
            self.assertEqual(
                mount.host_path,
                "/var/mnt/benchflow-nvme/benchflow-kv-cache",
            )
            self.assertEqual(mount.mount_path, "/mnt/nvme-kv-cache")
            self.assertTrue(mount.cleanup)
            kv_arg = self._kv_transfer_arg(plan)
            self.assertIn('"cpu_bytes_to_use":274877906944', kv_arg)
            self.assertIn('"type":"fs"', kv_arg)
            self._assert_native_chunk_size(kv_arg)
            self.assertEqual(plan.deployment.runtime.shared_memory_size, "300Gi")

    def _assert_common_matrix(self, plans, *, placement_mode: str) -> None:
        self.assertEqual({plan.deployment.runtime.replicas for plan in plans}, REPLICAS)
        self.assertEqual(
            {int(plan.benchmark.aiperf.args["concurrency"]) for plan in plans},
            CONCURRENCIES,
        )
        for plan in plans:
            self.assertEqual(plan.model.name, "Qwen/Qwen3.6-35B-A3B")
            self.assertEqual(plan.deployment.platform, "llm-d")
            self.assertEqual(plan.deployment.mode, "inference-scheduling")
            self.assertEqual(plan.deployment.repo_ref, "v0.9.0")
            self.assertEqual(plan.deployment.gateway, "istio")
            self.assertEqual(plan.deployment.runtime.tensor_parallelism, 2)
            self.assertEqual(plan.deployment.runtime.image, "vllm/vllm-openai:v0.27.0")
            self.assertEqual(plan.deployment.runtime.placement.mode, placement_mode)
            self.assertNotIn("epp_config", plan.deployment.options)
            self.assertIn(
                "--gpu-memory-utilization=0.55",
                plan.deployment.runtime.vllm_args,
            )
            self.assertIn("--max-num-seqs=256", plan.deployment.runtime.vllm_args)

    @staticmethod
    def _kv_transfer_arg(plan) -> str:
        return next(
            arg
            for arg in plan.deployment.runtime.vllm_args
            if arg.startswith("--kv-transfer-config=")
        )

    def _assert_native_chunk_size(self, kv_arg: str) -> None:
        config = json.loads(kv_arg.split("=", 1)[1])
        extra = config["kv_connector_extra_config"]
        self.assertNotIn("block_size", extra)
        self.assertNotIn("blocks_per_chunk", extra)


if __name__ == "__main__":
    unittest.main()
