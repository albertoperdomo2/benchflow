from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from benchflow.benchmark import aiperf


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
