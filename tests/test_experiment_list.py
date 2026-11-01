from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from benchflow.cli import cli
from benchflow.commands.experiment import _list_status_matches


def _entry(
    name: str,
    status: str,
    *,
    finished: bool = False,
    succeeded: bool = False,
) -> dict[str, object]:
    return {
        "name": name,
        "namespace": "benchflow",
        "experiment": "smoke",
        "platform": "llm-d",
        "mode": "inference-scheduling",
        "backend": "tekton",
        "status": status,
        "finished": finished,
        "succeeded": succeeded,
        "start_time": "2026-11-01T10:00:00Z",
        "completion_time": "",
        "message": "",
    }


@pytest.mark.parametrize(
    ("selector", "entry", "expected"),
    [
        ("all", _entry("queued", "Queued"), True),
        ("active", _entry("running", "Running"), True),
        ("active", _entry("done", "Succeeded", finished=True, succeeded=True), False),
        ("running", _entry("running", "Running"), True),
        ("running", _entry("queued", "Queued"), False),
        ("succeeded", _entry("done", "Completed", finished=True, succeeded=True), True),
        ("failed", _entry("failed", "Failed", finished=True), True),
        ("failed", _entry("rejected", "Rejected", finished=True), True),
    ],
)
def test_list_status_matches(
    selector: str, entry: dict[str, object], expected: bool
) -> None:
    assert _list_status_matches(entry, selector) is expected


def test_experiment_list_filters_running_executions() -> None:
    entries = [
        _entry("queued-run", "Queued"),
        _entry("running-run", "Running"),
        _entry("finished-run", "Succeeded", finished=True, succeeded=True),
    ]

    with patch(
        "benchflow.commands.experiment.list_benchflow_executions",
        return_value=entries,
    ):
        result = CliRunner().invoke(
            cli,
            [
                "experiment",
                "list",
                "--namespace",
                "benchflow",
                "--status",
                "running",
                "--format",
                "json",
            ],
        )

    assert result.exit_code == 0, result.output
    assert [entry["name"] for entry in json.loads(result.output)] == ["running-run"]
