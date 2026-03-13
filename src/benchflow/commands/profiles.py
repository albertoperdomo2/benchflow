from __future__ import annotations

import argparse

from ..loaders import list_profile_entries
from .shared import (
    add_parser,
    add_profile_source_arguments,
    dump,
    format_profile_list,
    load_profile_document,
    profiles_dir_from,
)


PROFILE_SUBCOMMANDS = ("list", "show")


def cmd_profiles_list(args: argparse.Namespace) -> int:
    profiles_dir = profiles_dir_from(args)
    entries = [
        entry
        for entry in list_profile_entries(profiles_dir)
        if args.kind in (None, entry.kind)
    ]
    payload = [entry.to_dict() for entry in entries]

    if args.format == "table":
        output = format_profile_list(payload)
        if output:
            print(output)
        return 0

    print(dump(payload, args.format))
    return 0


def cmd_profiles_show(args: argparse.Namespace) -> int:
    profiles_dir = profiles_dir_from(args)
    document = load_profile_document(profiles_dir, args.name, args.kind)
    print(dump(document, args.format))
    return 0


def register_profiles_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    profiles_list = add_parser(
        subparsers,
        "list",
        help_text="List available profiles",
        description="List the profiles available in the current repository.",
    )
    add_profile_source_arguments(profiles_list)
    profiles_list.add_argument(
        "--kind",
        choices=("deployment", "benchmark", "metrics"),
        help="Restrict the list to one profile kind.",
    )
    profiles_list.add_argument(
        "--format",
        choices=("table", "yaml", "json"),
        default="table",
        help="Output format.",
    )
    profiles_list.set_defaults(func=cmd_profiles_list)

    profiles_show = add_parser(
        subparsers,
        "show",
        help_text="Show a single profile",
        description="Print a single profile document in YAML or JSON.",
    )
    add_profile_source_arguments(profiles_show)
    profiles_show.add_argument("name", help="Profile name.")
    profiles_show.add_argument(
        "--kind",
        choices=("deployment", "benchmark", "metrics"),
        help="Profile kind when the name is ambiguous.",
    )
    profiles_show.add_argument(
        "--format",
        choices=("yaml", "json"),
        default="yaml",
        help="Output format.",
    )
    profiles_show.set_defaults(func=cmd_profiles_show)


def profile_command_options(subcommand: str | None) -> tuple[str, ...]:
    if subcommand == "list":
        return ("--repo-root", "--profiles-dir", "--kind", "--format")
    if subcommand == "show":
        return ("--repo-root", "--profiles-dir", "--kind", "--format")
    return ()
