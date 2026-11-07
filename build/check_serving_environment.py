"""Check installed dependencies, respecting vLLM's explicit NCCL override.

The upstream image sets UV_OVERRIDE because DeepEP needs a newer NCCL than
Torch's wheel metadata requests. Every other dependency must satisfy its
declared requirement; this is not a blanket exemption for inherited conflicts.
"""

from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def dependency_errors(distributions, overrides):
    # A system interpreter can expose both /usr/local and Ubuntu dist-packages.
    # Match importlib.metadata.version(): the first search-path match wins.
    installed = {}
    for dist in distributions:
        installed.setdefault(canonicalize_name(dist.metadata["Name"]), dist)
    rules = {}
    for text in overrides:
        requirement = Requirement(text)
        name = canonicalize_name(requirement.name)
        if not name.startswith("nvidia-nccl-cu"):
            raise ValueError(f"Unexpected serving-image dependency override: {text}")
        rules[name] = requirement
    errors = []
    for name, dist in sorted(installed.items()):
        for text in dist.requires or []:
            declared = Requirement(text)
            if declared.marker and not declared.marker.evaluate({"extra": ""}):
                continue
            dependency = canonicalize_name(declared.name)
            effective = rules.get(dependency, declared)
            candidate = installed.get(dependency)
            if candidate is None or candidate.version not in effective.specifier:
                actual = candidate.version if candidate is not None else "missing"
                errors.append(
                    f"{name} {dist.version}: requires {effective}, installed {actual}"
                )
    return errors


if __name__ == "__main__":
    override_file = Path("/etc/uv-overrides.txt")
    overrides = (
        [
            line.strip()
            for line in override_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if override_file.exists()
        else []
    )
    errors = dependency_errors(metadata.distributions(), overrides)
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"All serving dependencies satisfied; upstream NCCL overrides: {overrides}")
