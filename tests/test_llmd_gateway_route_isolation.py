from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from benchflow.deploy.llmd import _create_httproute
from benchflow.toolbox.platform import _llmd_recipe_gateway_base_url


class LlmdGatewayRouteIsolationTest(unittest.TestCase):
    def test_appends_release_prefix_to_gateway_url(self) -> None:
        self.assertEqual(
            _llmd_recipe_gateway_base_url(
                "http://llm-d-inference-gateway-istio.benchflow.svc.cluster.local/",
                "qwen36-35b-offloading-abc123",
            ),
            "http://llm-d-inference-gateway-istio.benchflow.svc.cluster.local/"
            "benchflow/qwen36-35b-offloading-abc123",
        )

    @patch("benchflow.deploy.llmd.run_command")
    def test_shared_gateway_route_selects_and_rewrites_release_prefix(
        self, run_command
    ) -> None:
        plan = SimpleNamespace(
            deployment=SimpleNamespace(
                repo_ref="v0.9.0",
                release_name="qwen36-35b-offloading-abc123",
                namespace="benchflow",
            )
        )

        _create_httproute(plan, "oc", shared_recipe_gateway=True)

        manifest = yaml.safe_load(run_command.call_args.kwargs["input_text"])
        self.assertEqual(
            manifest["spec"]["parentRefs"][0]["name"],
            "llm-d-inference-gateway",
        )
        rule = manifest["spec"]["rules"][0]
        self.assertEqual(
            rule["matches"][0]["path"]["value"],
            "/benchflow/qwen36-35b-offloading-abc123",
        )
        self.assertEqual(
            rule["filters"][0]["urlRewrite"]["path"]["replacePrefixMatch"],
            "/",
        )


if __name__ == "__main__":
    unittest.main()
