from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from benchflow.cli import cli
from benchflow.models import ValidationError
from benchflow.toolbox.skill import SKILL_MARKER, install_benchflow_skill


def _source(tmp_path: Path, *, body: str = "instructions") -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "SKILL.md").write_text(
        "---\nname: benchflow\ndescription: Use BenchFlow.\n---\n\n" + body + "\n",
        encoding="utf-8",
    )
    references = source / "references"
    references.mkdir()
    (references / "operations.md").write_text("# Operations\n", encoding="utf-8")
    return source


def test_installs_same_skill_for_codex_and_claude(tmp_path: Path) -> None:
    home = tmp_path / "home"
    codex_home = tmp_path / "custom-codex"

    results = install_benchflow_skill(
        home=home,
        codex_home=codex_home,
        source_dir=_source(tmp_path),
        version="1.2.3",
    )

    assert [(result.agent, result.status) for result in results] == [
        ("Codex", "installed"),
        ("Claude", "installed"),
    ]
    destinations = [
        codex_home / "skills" / "benchflow",
        home / ".claude" / "skills" / "benchflow",
    ]
    for destination in destinations:
        assert (destination / "SKILL.md").is_file()
        marker = json.loads((destination / SKILL_MARKER).read_text(encoding="utf-8"))
        assert marker["managed_by"] == "benchflow"
        assert marker["benchflow_version"] == "1.2.3"
    assert (destinations[0] / "SKILL.md").read_bytes() == (
        destinations[1] / "SKILL.md"
    ).read_bytes()


def test_install_is_idempotent_and_updates_managed_copies(tmp_path: Path) -> None:
    home = tmp_path / "home"
    codex_home = tmp_path / "codex"
    source = _source(tmp_path)
    install_benchflow_skill(
        home=home, codex_home=codex_home, source_dir=source, version="1.0"
    )

    unchanged = install_benchflow_skill(
        home=home, codex_home=codex_home, source_dir=source, version="1.0"
    )
    assert {result.status for result in unchanged} == {"unchanged"}

    (source / "SKILL.md").write_text("updated\n", encoding="utf-8")
    updated = install_benchflow_skill(
        home=home, codex_home=codex_home, source_dir=source, version="1.1"
    )
    assert {result.status for result in updated} == {"updated"}
    assert (codex_home / "skills" / "benchflow" / "SKILL.md").read_text(
        encoding="utf-8"
    ) == "updated\n"


def test_unmanaged_skill_requires_force(tmp_path: Path) -> None:
    home = tmp_path / "home"
    codex_home = tmp_path / "codex"
    destination = codex_home / "skills" / "benchflow"
    destination.mkdir(parents=True)
    (destination / "SKILL.md").write_text("user skill\n", encoding="utf-8")
    source = _source(tmp_path)

    with pytest.raises(ValidationError, match="--force"):
        install_benchflow_skill(
            home=home,
            codex_home=codex_home,
            source_dir=source,
            version="1.0",
        )

    results = install_benchflow_skill(
        force=True,
        home=home,
        codex_home=codex_home,
        source_dir=source,
        version="1.0",
    )
    assert {result.status for result in results} == {"installed", "updated"}
    assert destination.joinpath(SKILL_MARKER).is_file()


def test_failed_second_update_restores_both_managed_skills(tmp_path: Path) -> None:
    home = tmp_path / "home"
    codex_home = tmp_path / "codex"
    source = _source(tmp_path, body="original")
    install_benchflow_skill(
        home=home, codex_home=codex_home, source_dir=source, version="1.0"
    )
    (source / "SKILL.md").write_text("replacement\n", encoding="utf-8")
    real_replace = os.replace
    call_count = 0

    def fail_second_install(source_path, destination_path):
        nonlocal call_count
        call_count += 1
        if call_count == 4:
            raise OSError("simulated Claude install failure")
        return real_replace(source_path, destination_path)

    with (
        patch("benchflow.toolbox.skill.os.replace", side_effect=fail_second_install),
        pytest.raises(OSError, match="simulated"),
    ):
        install_benchflow_skill(
            home=home,
            codex_home=codex_home,
            source_dir=source,
            version="1.1",
        )

    for destination in (
        codex_home / "skills" / "benchflow",
        home / ".claude" / "skills" / "benchflow",
    ):
        assert "original" in (destination / "SKILL.md").read_text(encoding="utf-8")
        marker = json.loads((destination / SKILL_MARKER).read_text(encoding="utf-8"))
        assert marker["benchflow_version"] == "1.0"


def test_cli_installs_both_agent_skills(tmp_path: Path) -> None:
    runner = CliRunner()
    home = tmp_path / "home"
    codex_home = tmp_path / "codex"

    result = runner.invoke(
        cli,
        ["skill", "install"],
        env={"HOME": str(home), "CODEX_HOME": str(codex_home)},
    )

    assert result.exit_code == 0, result.output
    assert "Codex:" in result.output
    assert "Claude:" in result.output
    assert (codex_home / "skills" / "benchflow" / "SKILL.md").is_file()
    assert (home / ".claude" / "skills" / "benchflow" / "SKILL.md").is_file()
