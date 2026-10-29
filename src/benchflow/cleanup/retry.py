from __future__ import annotations

import subprocess
import time

from ..cluster import is_transient_kubernetes_error, run_command
from ..ui import detail, warning


def run_cleanup_command_with_retry(
    argv: list[str],
    *,
    description: str,
    attempts: int = 5,
) -> subprocess.CompletedProcess[str]:
    """Run a cleanup command, retrying transient Kubernetes API failures."""
    total_attempts = max(1, int(attempts))
    for attempt in range(1, total_attempts + 1):
        result = run_command(
            argv,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return result

        output = str(result.stderr or result.stdout or "").strip()
        if attempt >= total_attempts or not is_transient_kubernetes_error(output):
            return result

        delay_seconds = min(2 ** (attempt - 1), 15)
        warning(
            f"Transient Kubernetes API error while {description}; "
            f"retrying in {delay_seconds}s ({attempt}/{total_attempts})"
        )
        if output:
            detail(output)
        time.sleep(delay_seconds)

    raise AssertionError("cleanup retry loop exhausted without returning")
