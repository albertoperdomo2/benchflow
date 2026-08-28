from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from benchflow.deploy.llmd import _create_httproute


class LlmdGatewayRouteIsolationTest(unittest.TestCase):
    @patch("benchflow.deploy.llmd.run_command")
    def test_route_targets_release_scoped_gateway(self, run_command) -> None:
        plan = SimpleNamespace(
            deployment=SimpleNamespace(
                repo_ref="v0.9.0",
                release_name="qwen36-35b-offloading-abc123",
                namespace="benchflow",
            )
        )

        _create_httproute(plan, "oc")

        manifest = yaml.safe_load(run_command.call_args.kwargs["input_text"])
        self.assertEqual(
            manifest["spec"]["parentRefs"][0]["name"],
            "infra-qwen36-35b-offloading-abc123-inference-gateway",
        )
        rule = manifest["spec"]["rules"][0]
        self.assertEqual(
            rule["matches"][0]["path"]["value"],
            "/",
        )
        self.assertNotIn("filters", rule)


if __name__ == "__main__":
    unittest.main()
