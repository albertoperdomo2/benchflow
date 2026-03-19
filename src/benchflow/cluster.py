from __future__ import annotations

import os
import json
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class CommandError(RuntimeError):
    """Raised when a required external command fails."""


def discover_repo_root(start: Path | None = None) -> Path:
    candidates: list[Path] = []

    if start is not None:
        start = start.resolve()
        candidates.extend([start, *start.parents])

    package_path = Path(__file__).resolve()
    candidates.extend([package_path.parent, *package_path.parents])

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (
            (candidate / "pyproject.toml").exists()
            and (candidate / "profiles").exists()
            and (candidate / "tekton").exists()
        ):
            return candidate

    raise CommandError(
        "could not discover repository root; pass --profiles-dir explicitly"
    )


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise CommandError(f"required command not found: {name}")


def require_any_command(*names: str) -> str:
    for name in names:
        path = shutil.which(name)
        if path is not None:
            return name
    joined = ", ".join(names)
    raise CommandError(f"none of the required commands are available: {joined}")


@contextmanager
def use_kubeconfig(kubeconfig: str | Path | None):
    if not kubeconfig:
        yield
        return

    kubeconfig_path = Path(kubeconfig).expanduser().resolve()
    previous = os.environ.get("KUBECONFIG")
    os.environ["KUBECONFIG"] = str(kubeconfig_path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("KUBECONFIG", None)
        else:
            os.environ["KUBECONFIG"] = previous


def run_command(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
    capture_output: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            env=env,
            input=input_text,
            text=True,
            capture_output=capture_output,
            check=check,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        stdout = exc.stdout.strip() if exc.stdout else ""
        details = stderr or stdout or "command failed"
        raise CommandError(f"{' '.join(argv)}: {details}") from exc


def run_json_command(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
) -> dict[str, Any]:
    result = run_command(
        argv,
        cwd=cwd,
        env=env,
        input_text=input_text,
        capture_output=True,
    )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise CommandError(
            f"{' '.join(argv)}: command did not return valid JSON"
        ) from exc


def resolve_target_base_url(target: Any, namespace: str) -> str:
    if target.discovery == "static":
        if not target.base_url:
            raise CommandError(
                "target discovery is static but target.base_url is empty"
            )
        return str(target.base_url).rstrip("/")

    if target.discovery == "llminferenceservice-status-url":
        kubectl_cmd = require_any_command("oc", "kubectl")
        resource_name = str(target.resource_name or "").strip()
        if not resource_name:
            raise CommandError(
                "target discovery is llminferenceservice-status-url but "
                "target.resource_name is empty"
            )
        payload = run_json_command(
            [
                kubectl_cmd,
                "get",
                "llminferenceservice",
                resource_name,
                "-n",
                namespace,
                "-o",
                "json",
            ]
        )
        url = str(payload.get("status", {}).get("url") or "").strip()
        if not url:
            raise CommandError(
                f"LLMInferenceService {resource_name} in namespace {namespace} "
                "does not have status.url yet"
            )
        return url.rstrip("/")

    raise CommandError(f"unsupported target discovery strategy: {target.discovery}")


def create_manifest(manifest_yaml: str, namespace: str) -> dict[str, Any]:
    require_command("oc")
    result = run_command(
        ["oc", "create", "-n", namespace, "-f", "-", "-o", "json"],
        input_text=manifest_yaml,
        capture_output=True,
    )
    return json.loads(result.stdout)


def get_current_namespace() -> str:
    require_command("oc")
    result = run_command(["oc", "project", "-q"], capture_output=True)
    return result.stdout.strip()
