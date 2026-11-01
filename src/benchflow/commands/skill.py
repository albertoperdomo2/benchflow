from __future__ import annotations

import argparse

import click

from ..toolbox.skill import install_benchflow_skill
from ..ui import detail, success
from .shared import invoke_handler


def cmd_skill_install(args: argparse.Namespace) -> int:
    results = install_benchflow_skill(force=bool(args.force))
    changed = [result for result in results if result.status != "unchanged"]
    if changed:
        success("Installed BenchFlow agent skill")
    else:
        success("BenchFlow agent skill is already current")
    for result in results:
        detail(f"{result.agent}: {result.path} ({result.status})")
    return 0


@click.group(
    "skill",
    help="Install BenchFlow guidance for supported coding agents.",
    short_help="Manage the BenchFlow agent skill",
)
def skill_group() -> None:
    pass


@skill_group.command(
    "install",
    help="Install the packaged BenchFlow skill for Codex and Claude Code.",
    short_help="Install for Codex and Claude",
)
@click.option(
    "--force",
    is_flag=True,
    help="Replace an existing skill that was not installed by BenchFlow.",
)
def skill_install(**kwargs: object) -> int:
    return invoke_handler(cmd_skill_install, **kwargs)
