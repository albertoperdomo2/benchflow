from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchflow.cluster import CommandError
from benchflow.deploy.llmd import (
    _llmd_router_chart_settings,
    _llmd_router_epp_selectors_for_release,
    _llmd_recipe_gateway_route_prefix,
)


class LlmdRouterChartSettingsTest(unittest.TestCase):
    def _checkout(self, env_contents: str) -> tempfile.TemporaryDirectory[str]:
        temporary_directory = tempfile.TemporaryDirectory()
        guides = Path(temporary_directory.name) / "guides"
        guides.mkdir()
        (guides / "env.sh").write_text(env_contents, encoding="utf-8")
        return temporary_directory

    def test_resolves_v090_shell_defaults(self) -> None:
        checkout = self._checkout(
            """
export ROUTER_CHART_VERSION=${ROUTER_CHART_VERSION:-v0.10.0}
export ROUTER_STANDALONE_CHART=${ROUTER_STANDALONE_CHART:-oci://ghcr.io/llm-d/charts/llm-d-router-standalone}
export ROUTER_GATEWAY_CHART=${ROUTER_GATEWAY_CHART:-oci://ghcr.io/llm-d/charts/llm-d-router-gateway}
"""
        )
        self.addCleanup(checkout.cleanup)

        with patch.dict(
            "os.environ",
            {},
            clear=True,
        ):
            gateway = _llmd_router_chart_settings(
                Path(checkout.name), gateway_mode="istio"
            )
            standalone = _llmd_router_chart_settings(
                Path(checkout.name), gateway_mode="standalone"
            )

        self.assertEqual(
            gateway,
            ("oci://ghcr.io/llm-d/charts/llm-d-router-gateway", "v0.10.0"),
        )
        self.assertEqual(
            standalone,
            ("oci://ghcr.io/llm-d/charts/llm-d-router-standalone", "v0.10.0"),
        )

    def test_honors_environment_overrides(self) -> None:
        checkout = self._checkout(
            """
export ROUTER_CHART_VERSION=${ROUTER_CHART_VERSION:-v0.10.0}
export ROUTER_GATEWAY_CHART=${ROUTER_GATEWAY_CHART:-oci://ghcr.io/llm-d/charts/llm-d-router-gateway}
"""
        )
        self.addCleanup(checkout.cleanup)

        with patch.dict(
            "os.environ",
            {
                "ROUTER_CHART_VERSION": "v0.10.1",
                "ROUTER_GATEWAY_CHART": "oci://example.invalid/router-gateway",
            },
            clear=True,
        ):
            settings = _llmd_router_chart_settings(
                Path(checkout.name), gateway_mode="istio"
            )

        self.assertEqual(
            settings,
            ("oci://example.invalid/router-gateway", "v0.10.1"),
        )

    def test_preserves_legacy_literal_assignments(self) -> None:
        checkout = self._checkout(
            """
export ROUTER_CHART_VERSION=v0.9.0
export ROUTER_GATEWAY_CHART=oci://example.invalid/router-gateway
"""
        )
        self.addCleanup(checkout.cleanup)

        self.assertEqual(
            _llmd_router_chart_settings(Path(checkout.name), gateway_mode="istio"),
            ("oci://example.invalid/router-gateway", "v0.9.0"),
        )

    def test_rejects_unknown_shell_expressions(self) -> None:
        checkout = self._checkout(
            """
export ROUTER_CHART_VERSION=$(router-version)
export ROUTER_GATEWAY_CHART=oci://example.invalid/router-gateway
"""
        )
        self.addCleanup(checkout.cleanup)

        with self.assertRaisesRegex(CommandError, "unsupported shell expressions"):
            _llmd_router_chart_settings(Path(checkout.name), gateway_mode="istio")

    @patch("benchflow.deploy.llmd.run_json_command")
    def test_epp_selector_uses_helm_deployment_selector(self, run_json_command) -> None:
        run_json_command.return_value = {
            "items": [
                {
                    "metadata": {
                        "annotations": {
                            "meta.helm.sh/release-name": "gaie-very-long-release-name",
                            "meta.helm.sh/release-namespace": "benchflow",
                        }
                    },
                    "spec": {
                        "selector": {
                            "matchLabels": {
                                "llm-d-router-gateway": "gaie-very-long-release-na-epp"
                            }
                        }
                    },
                }
            ]
        }

        selectors = _llmd_router_epp_selectors_for_release(
            "benchflow", "very-long-release-name", "istio", "oc"
        )

        self.assertEqual(
            selectors[0],
            "llm-d-router-gateway=gaie-very-long-release-na-epp",
        )
        self.assertIn("llm-d-router-gateway=gaie-very-long-release-name-epp", selectors)

    def test_shared_gateway_route_prefix_is_release_specific(self) -> None:
        self.assertEqual(
            _llmd_recipe_gateway_route_prefix("qwen36-35b-offloading-abc123"),
            "/benchflow/qwen36-35b-offloading-abc123",
        )


if __name__ == "__main__":
    unittest.main()
