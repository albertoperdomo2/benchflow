from __future__ import annotations

import unittest

from benchflow.benchmark.forge import _compose_version


class ForgeVersionCompositionTest(unittest.TestCase):
    def test_appends_benchflow_tag_after_deployment_profile(self) -> None:
        version = _compose_version(
            {"version": "v0.5.0"},
            {"benchflow_tag": "rc2"},
            "Qwen/Qwen3-32B",
            "precise-prefix-cache",
        )

        self.assertEqual(version, "v0.5.0-precise-prefix-cache-rc2")

    def test_ignores_empty_benchflow_tag(self) -> None:
        version = _compose_version(
            {"version": "v0.5.0"},
            {"benchflow_tag": "  "},
            "Qwen/Qwen3-32B",
            "precise-prefix-cache",
        )

        self.assertEqual(version, "v0.5.0-precise-prefix-cache")

    def test_appends_tag_without_deployment_profile(self) -> None:
        version = _compose_version(
            {},
            {"version": "v0.5.0", "benchflow_tag": "rc2"},
            "Qwen/Qwen3-32B",
            "",
        )

        self.assertEqual(version, "v0.5.0-rc2")


if __name__ == "__main__":
    unittest.main()
