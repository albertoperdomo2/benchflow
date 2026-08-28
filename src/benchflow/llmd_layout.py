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


def recipe_gateway_name(release_name: str) -> str:
    """Return the release-scoped Gateway name used by the llm-d recipe."""
    # Istio appends ``-istio`` to generated infrastructure names. Keep the
    # Gateway name below the Kubernetes label limit while retaining the
    # release suffix that makes concurrent matrix children distinct.
    max_release_length = 33
    if len(release_name) > max_release_length:
        suffix = release_name[-10:]
        prefix_length = max_release_length - len(suffix) - 1
        release_name = f"{release_name[:prefix_length].rstrip('-')}-{suffix}"
    return f"infra-{release_name}-inference-gateway"
