from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest

from benchflow.cluster import CommandError

# Initialize the orchestration package before importing kueue, which is also
# imported by orchestration.service.
from benchflow.orchestration import tekton
from benchflow.kueue import list_reservation_workloads


def _result(
    returncode: int, *, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["oc"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


@patch("benchflow.cluster.time.sleep")
@patch("benchflow.orchestration.tekton.require_command")
@patch("benchflow.orchestration.tekton.run_command")
def test_pipelinerun_read_retries_transient_api_failure(
    run_command, _require_command, sleep
) -> None:
    payload = {"metadata": {"name": "child"}}
    run_command.side_effect = [
        _result(
            1,
            stderr=(
                "rpc error: code = Unavailable desc = keepalive ping failed "
                "to receive ACK within timeout"
            ),
        ),
        _result(0, stdout=json.dumps(payload)),
    ]

    assert tekton._get_pipelinerun("benchflow", "child") == payload
    assert run_command.call_count == 2
    sleep.assert_called_once_with(1)


@patch("benchflow.orchestration.tekton.require_command")
@patch("benchflow.orchestration.tekton.run_command")
def test_pipelinerun_read_does_not_hide_non_transient_failure(
    run_command, _require_command
) -> None:
    run_command.return_value = _result(1, stderr="forbidden")

    with pytest.raises(CommandError, match="forbidden"):
        tekton._get_pipelinerun("benchflow", "child")


@patch("benchflow.cluster.time.sleep")
@patch("benchflow.kueue.require_any_command", return_value="oc")
@patch("benchflow.kueue.run_json_command")
def test_reservation_workload_list_retries_transient_api_failure(
    run_json_command, _require_any_command, sleep
) -> None:
    run_json_command.side_effect = [
        CommandError(
            "oc get workloads: rpc error: code = Unavailable desc = "
            "keepalive ping failed to receive ACK within timeout"
        ),
        {"items": [{"metadata": {"name": "reservation"}}]},
    ]

    assert list_reservation_workloads("benchflow") == [
        {"metadata": {"name": "reservation"}}
    ]
    assert run_json_command.call_count == 2
    sleep.assert_called_once_with(1)
