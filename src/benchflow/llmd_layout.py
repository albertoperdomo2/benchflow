from __future__ import annotations

import re


def uses_recipe_layout(repo_ref: str) -> bool:
    normalized = str(repo_ref or "").strip().lower()
    if not normalized:
        return True
    if normalized == "main":
        return True

    match = re.search(r"v?(\d+)\.(\d+)\.(\d+)(?:[-+][a-z0-9_.-]+)?", normalized)
    if match is None:
        # Named branches are assumed to track the current upstream layout.
        return True

    version = tuple(int(part) for part in match.groups())
    return version >= (0, 6, 0)
