from __future__ import annotations

import subprocess
from unittest.mock import patch

from benchflow.cleanup.retry import run_cleanup_command_with_retry


def _result(returncode: int, *, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["oc"],
        returncode=returncode,
        stdout="",
        stderr=stderr,
    )


@patch("benchflow.cleanup.retry.time.sleep")
@patch("benchflow.cleanup.retry.run_command")
def test_cleanup_retries_transient_kubernetes_api_errors(run_command, sleep) -> None:
    run_command.side_effect = [
        _result(1, stderr="TLS handshake timeout\nerror: unexpected EOF"),
        _result(0),
    ]

    result = run_cleanup_command_with_retry(
        ["oc", "exec", "model-pod"],
        description="cleaning runtime hostPath",
    )

    assert result.returncode == 0
    assert run_command.call_count == 2
    sleep.assert_called_once_with(1)


@patch("benchflow.cleanup.retry.time.sleep")
@patch("benchflow.cleanup.retry.run_command")
def test_cleanup_does_not_retry_non_transient_failures(run_command, sleep) -> None:
    run_command.return_value = _result(1, stderr="find: permission denied")

    result = run_cleanup_command_with_retry(
        ["oc", "exec", "model-pod"],
        description="cleaning runtime hostPath",
    )

    assert result.returncode == 1
    run_command.assert_called_once()
    sleep.assert_not_called()
