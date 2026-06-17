from __future__ import annotations

import json
from typing import Any


def _flag_name(key: str, aliases: dict[str, str] | None = None) -> str:
    flag_key = (aliases or {}).get(key, key)
    return f"--{flag_key.replace('_', '-')}"


def _stringify(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def render_cli_args(
    args: dict[str, Any],
    *,
    aliases: dict[str, str] | None = None,
    join_lists: set[str] | None = None,
    omit_keys: set[str] | None = None,
) -> list[str]:
    argv: list[str] = []
    omitted = omit_keys or set()
    list_join_keys = join_lists or set()
    for key, value in args.items():
        if key in omitted or value is None or value == "":
            continue
        flag = _flag_name(key, aliases)
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        if isinstance(value, list):
            if not value:
                continue
            if key in list_join_keys:
                argv.extend([flag, ",".join(str(item) for item in value)])
                continue
            for item in value:
                if item is None or item == "":
                    continue
                argv.extend([flag, _stringify(item)])
            continue
        argv.extend([flag, _stringify(value)])
    return argv
