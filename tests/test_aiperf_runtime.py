from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from benchflow.benchmark import aiperf
from benchflow.loaders import load_benchmark_profile


def test_aiperf_profile_supports_synthetic_dataset(tmp_path: Path) -> None:
    profile_path = tmp_path / "synthetic.yaml"
    profile_path.write_text(
        """apiVersion: benchflow.io/v1alpha1
kind: BenchmarkProfile
metadata:
  name: synthetic
spec:
  tool: aiperf
  aiperf:
    endpoint_type: chat
    synthetic_input_tokens_mean: 64
    output_tokens_mean: 16
    request_count: 10
""",
        encoding="utf-8",
    )

    profile = load_benchmark_profile(profile_path)

    assert profile.spec.aiperf.dataset_url == ""
    assert profile.spec.aiperf.args["synthetic_input_tokens_mean"] == 64
    assert "fixed_schedule" not in profile.spec.aiperf.args


@pytest.mark.parametrize(
    ("dataset_source", "missing_field"),
    (
        ("dataset_url: https://example.test/data.jsonl", "dataset_type"),
        ("dataset_type: single_turn", "dataset_url"),
    ),
)
def test_aiperf_file_dataset_requires_url_and_type(
    tmp_path: Path, dataset_source: str, missing_field: str
) -> None:
    profile_path = tmp_path / "incomplete-file-dataset.yaml"
    profile_path.write_text(
        f"""apiVersion: benchflow.io/v1alpha1
kind: BenchmarkProfile
metadata:
  name: incomplete-file-dataset
spec:
  tool: aiperf
  aiperf:
    endpoint_type: chat
    {dataset_source}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=missing_field):
        load_benchmark_profile(profile_path)


def test_aiperf_artifact_directory_is_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    artifact_dir = aiperf._artifact_dir(Path("results"))

    assert artifact_dir == tmp_path / "results"
    assert artifact_dir.is_dir()


def test_aiperf_subprocess_uses_private_writable_working_directory() -> None:
    observed_work_dir: Path | None = None

    def fake_run(
        argv: list[str],
        *,
        env: dict[str, str],
        text: bool,
        check: bool,
        cwd: str,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal observed_work_dir
        observed_work_dir = Path(cwd)
        assert observed_work_dir.is_dir()
        (observed_work_dir / ".cache").mkdir()
        return subprocess.CompletedProcess(argv, 0)

    with patch.object(aiperf.subprocess, "run", side_effect=fake_run):
        aiperf._run_subprocess(["aiperf", "profile"], env={})

    assert observed_work_dir is not None
    assert not observed_work_dir.exists()
