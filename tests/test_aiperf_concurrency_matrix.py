from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchflow.loaders import ProfileCatalog, load_benchmark_profile, load_experiment
from benchflow.matrix import (
    expand_experiment_matrix,
    experiment_matrix_size,
    is_matrix_experiment,
    resolve_experiment_matrix,
)
from benchflow.models import (
    Experiment,
    ExperimentSpec,
    Metadata,
    ModelSpec,
    OverrideBenchmarkSpec,
    OverrideScaleSpec,
    OverrideSpec,
    ValidationError,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _experiment(
    *,
    benchmark_profile: str = "aiperf-agentx-inference",
    concurrency: int | list[int] = 4,
) -> Experiment:
    return Experiment(
        api_version="benchflow.io/v1alpha1",
        kind="Experiment",
        metadata=Metadata(name="aiperf-concurrency-matrix"),
        spec=ExperimentSpec(
            model=ModelSpec(name="Qwen/Qwen3.6-35B-A3B"),
            deployment_profile=["rhoai-distributed-default"],
            benchmark_profile=[benchmark_profile],
            overrides=OverrideSpec(
                scale=OverrideScaleSpec(replicas=[4, 8]),
                benchmark=OverrideBenchmarkSpec(concurrency=concurrency),
            ),
        ),
    )


class AiperfConcurrencyMatrixTest(unittest.TestCase):
    def test_concurrency_and_replicas_expand_as_cartesian_axes(self) -> None:
        experiment = _experiment(concurrency=[4, 16])

        self.assertTrue(is_matrix_experiment(experiment))
        self.assertEqual(experiment_matrix_size(experiment), 4)
        children = expand_experiment_matrix(experiment)
        self.assertTrue(
            all(
                isinstance(child.spec.overrides.benchmark.concurrency, int)
                for child in children
            )
        )

        plans = resolve_experiment_matrix(
            experiment, ProfileCatalog.load(REPO_ROOT / "profiles")
        )
        self.assertEqual(
            {
                (
                    plan.deployment.runtime.replicas,
                    plan.benchmark.aiperf.args["concurrency"],
                )
                for plan in plans
            },
            {(4, 4), (4, 16), (8, 4), (8, 16)},
        )

    def test_concurrency_override_rejects_non_aiperf_profiles(self) -> None:
        with self.assertRaisesRegex(
            ValidationError,
            "benchmark.concurrency is currently supported only for aiperf",
        ):
            resolve_experiment_matrix(
                _experiment(benchmark_profile="guidellm-smoke"),
                ProfileCatalog.load(REPO_ROOT / "profiles"),
            )

    def test_loader_rejects_duplicate_or_nonpositive_concurrency(self) -> None:
        for value, message in (
            ("[4, 4]", "must not contain duplicate values"),
            ("[0, 4]", "must contain only positive integers"),
        ):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "experiment.yaml"
                path.write_text(
                    f"""apiVersion: benchflow.io/v1alpha1
kind: Experiment
metadata:
  name: invalid-concurrency
spec:
  model:
    name: Qwen/Qwen3.6-35B-A3B
  deployment_profile: rhoai-distributed-default
  benchmark_profile: aiperf-agentx-inference
  overrides:
    benchmark:
      concurrency: {value}
""",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValidationError, message):
                    load_experiment(path)

    def test_benchmark_profile_concurrency_remains_scalar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "benchmark.yaml"
            path.write_text(
                """apiVersion: benchflow.io/v1alpha1
kind: BenchmarkProfile
metadata:
  name: invalid-aiperf-concurrency
spec:
  tool: aiperf
  aiperf:
    endpoint_type: chat
    public_dataset: weka_hf
    concurrency: [4, 8]
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValidationError,
                "spec.aiperf.concurrency must be a positive integer",
            ):
                load_benchmark_profile(path)


if __name__ == "__main__":
    unittest.main()
