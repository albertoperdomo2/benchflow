from __future__ import annotations

import shutil
from pathlib import Path

from .cluster import require_command, run_command


def clone_repo(
    *,
    url: str,
    revision: str,
    output_dir: Path,
    delete_existing: bool = True,
) -> str:
    require_command("git")
    if delete_existing and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    run_command(["git", "clone", url, str(output_dir)])
    run_command(["git", "checkout", revision], cwd=output_dir)
    result = run_command(
        ["git", "rev-parse", "HEAD"], cwd=output_dir, capture_output=True
    )
    return result.stdout.strip()
