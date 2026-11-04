from __future__ import annotations

from unittest.mock import patch

import pytest

from benchflow.cluster import CommandError
from benchflow.orchestration.service import _wait_for_matrix_children


@patch("benchflow.orchestration.service.time.sleep")
@patch("benchflow.orchestration.service.summarize_execution")
def test_matrix_watcher_survives_transient_api_failure(summarize, sleep) -> None:
    summarize.side_effect = [
        CommandError(
            "oc get pipelinerun child: The connection to the server was refused"
        ),
        {
            "status": "Succeeded",
            "finished": True,
            "succeeded": True,
            "message": "",
        },
    ]
    failures: list[str] = []

    _wait_for_matrix_children(["child"], namespace="benchflow", failures=failures)

    assert failures == []
    assert summarize.call_count == 2
    sleep.assert_called_once_with(5)


@patch("benchflow.orchestration.service.summarize_execution")
def test_matrix_watcher_preserves_non_transient_failure(summarize) -> None:
    summarize.side_effect = CommandError("oc get pipelinerun child: forbidden")

    with pytest.raises(CommandError, match="forbidden"):
        _wait_for_matrix_children(["child"], namespace="benchflow", failures=[])
