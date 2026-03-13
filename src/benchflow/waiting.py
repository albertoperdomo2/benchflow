from __future__ import annotations

import time
import urllib.error
import urllib.request

from .cluster import CommandError


def wait_for_endpoint(
    *,
    target_url: str,
    endpoint_path: str = "/v1/models",
    timeout_seconds: int = 3600,
    retry_interval_seconds: int = 10,
    verify_tls: bool = False,
) -> None:
    target = f"{target_url.rstrip('/')}{endpoint_path}"
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        try:
            request = urllib.request.Request(
                target, headers={"Accept": "application/json"}
            )
            context = None
            if not verify_tls and target.startswith("https://"):
                import ssl

                context = ssl._create_unverified_context()
            with urllib.request.urlopen(
                request, timeout=30, context=context
            ) as response:
                if 200 <= response.status < 400:
                    return
        except urllib.error.HTTPError as exc:
            if 500 <= exc.code < 600:
                pass
        except Exception:
            pass
        time.sleep(retry_interval_seconds)

    raise CommandError(f"timed out waiting for endpoint: {target}")
