from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from .cluster import CommandError
from .ui import detail, step, success


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
    attempt = 0
    last_status = ""

    step(f"Waiting for endpoint {target}")
    detail(
        f"Timeout: {timeout_seconds}s, retry interval: {retry_interval_seconds}s, "
        f"TLS verification: {'enabled' if verify_tls else 'disabled'}"
    )

    while time.time() < deadline:
        attempt += 1
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
                    success(
                        f"Endpoint ready after {attempt} attempt"
                        f"{'' if attempt == 1 else 's'}: {target}"
                    )
                    return
                status = f"HTTP {response.status}"
                if status != last_status:
                    detail(f"Endpoint not ready yet: {status}")
                    last_status = status
        except urllib.error.HTTPError as exc:
            status = f"HTTP {exc.code}"
            if status != last_status:
                detail(f"Endpoint not ready yet: {status}")
                last_status = status
            if 500 <= exc.code < 600:
                pass
        except Exception as exc:  # noqa: BLE001
            status = exc.__class__.__name__
            if status != last_status:
                detail(f"Endpoint not ready yet: {status}")
                last_status = status
        time.sleep(retry_interval_seconds)

    raise CommandError(f"timed out waiting for endpoint: {target}")


def wait_for_completions(
    *,
    target_url: str,
    model_name: str,
    endpoint_path: str = "/v1/completions",
    timeout_seconds: int = 600,
    retry_interval_seconds: int = 10,
    verify_tls: bool = False,
) -> None:
    target = f"{target_url.rstrip('/')}{endpoint_path}"
    deadline = time.time() + timeout_seconds
    attempt = 0
    last_status = ""
    payload = json.dumps(
        {
            "model": model_name,
            "prompt": "Say hello in one sentence.",
            "max_tokens": 8,
            "stream": False,
        }
    ).encode("utf-8")

    step(f"Waiting for completions endpoint {target}")
    detail(
        f"Model: {model_name}, timeout: {timeout_seconds}s, "
        f"retry interval: {retry_interval_seconds}s, "
        f"TLS verification: {'enabled' if verify_tls else 'disabled'}"
    )

    while time.time() < deadline:
        attempt += 1
        try:
            request = urllib.request.Request(
                target,
                data=payload,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            context = None
            if not verify_tls and target.startswith("https://"):
                import ssl

                context = ssl._create_unverified_context()
            with urllib.request.urlopen(
                request, timeout=45, context=context
            ) as response:
                response_body = response.read().decode("utf-8", errors="replace")
                if 200 <= response.status < 400:
                    try:
                        payload_json = json.loads(response_body or "{}")
                    except json.JSONDecodeError:
                        status = "invalid JSON response"
                    else:
                        if isinstance(payload_json, dict) and isinstance(
                            payload_json.get("choices"), list
                        ):
                            success(
                                f"Completions endpoint ready after {attempt} attempt"
                                f"{'' if attempt == 1 else 's'}: {target}"
                            )
                            return
                        status = "response missing choices"
                else:
                    status = f"HTTP {response.status}"

                if status != last_status:
                    detail(f"Completions endpoint not ready yet: {status}")
                    last_status = status
        except urllib.error.HTTPError as exc:
            status = f"HTTP {exc.code}"
            if status != last_status:
                detail(f"Completions endpoint not ready yet: {status}")
                last_status = status
            if 500 <= exc.code < 600:
                pass
        except Exception as exc:  # noqa: BLE001
            status = exc.__class__.__name__
            if status != last_status:
                detail(f"Completions endpoint not ready yet: {status}")
                last_status = status
        time.sleep(retry_interval_seconds)

    raise CommandError(f"timed out waiting for completions endpoint: {target}")
