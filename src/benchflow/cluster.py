from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .ui import detail, step, success, warning


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


def get_pipelinerun(namespace: str, name: str) -> dict[str, Any]:
    require_command("oc")
    result = run_command(
        ["oc", "get", "pipelinerun", name, "-n", namespace, "-o", "json"],
        capture_output=True,
    )
    return json.loads(result.stdout)


def pipelinerun_state(pipelinerun: dict[str, Any]) -> tuple[str, bool, bool, str]:
    conditions = pipelinerun.get("status", {}).get("conditions", [])
    if not conditions:
        return ("Pending", False, False, "")

    condition = conditions[0]
    status = condition.get("status", "Unknown")
    reason = condition.get("reason", "Unknown")
    message = condition.get("message", "")

    if status == "True":
        return (reason or "Succeeded", True, True, message)
    if status == "False":
        return (reason or "Failed", True, False, message)
    return (reason or "Running", False, False, message)


def list_pipelineruns(
    namespace: str, *, label_selector: str = ""
) -> list[dict[str, Any]]:
    require_command("oc")
    argv = ["oc", "get", "pipelinerun", "-n", namespace]
    if label_selector:
        argv.extend(["-l", label_selector])
    argv.extend(["-o", "json"])
    payload = run_json_command(argv)
    items = payload.get("items", [])
    if not isinstance(items, list):
        return []
    return items


def summarize_pipelinerun(pipelinerun: dict[str, Any]) -> dict[str, Any]:
    metadata = pipelinerun.get("metadata", {})
    labels = metadata.get("labels", {}) or {}
    status, finished, succeeded, message = pipelinerun_state(pipelinerun)
    status_payload = pipelinerun.get("status", {}) or {}
    return {
        "name": metadata.get("name", ""),
        "namespace": metadata.get("namespace", ""),
        "experiment": labels.get("benchflow.io/experiment", ""),
        "platform": labels.get("benchflow.io/platform", ""),
        "mode": labels.get("benchflow.io/mode", ""),
        "pipeline": labels.get("tekton.dev/pipeline", ""),
        "status": status,
        "finished": finished,
        "succeeded": succeeded,
        "start_time": status_payload.get("startTime")
        or metadata.get("creationTimestamp", ""),
        "completion_time": status_payload.get("completionTime", ""),
        "message": message,
    }


def list_benchflow_pipelineruns(
    namespace: str, *, include_completed: bool = False
) -> list[dict[str, Any]]:
    items = list_pipelineruns(
        namespace, label_selector="app.kubernetes.io/name=benchflow"
    )
    summaries = [summarize_pipelinerun(item) for item in items]
    summaries.sort(key=lambda item: item.get("start_time") or "", reverse=True)
    if include_completed:
        return summaries
    return [item for item in summaries if not item.get("finished")]


def cancel_pipelinerun(namespace: str, name: str) -> None:
    require_command("oc")
    if shutil.which("tkn") is not None:
        result = subprocess.run(
            ["tkn", "pipelinerun", "cancel", "-n", namespace, name],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return
        warning(
            "tkn pipelinerun cancel failed; falling back to oc patch: "
            + (result.stderr.strip() or result.stdout.strip() or "unknown error")
        )
    run_command(
        [
            "oc",
            "patch",
            "pipelinerun",
            name,
            "-n",
            namespace,
            "--type",
            "merge",
            "-p",
            '{"spec":{"status":"Cancelled"}}',
        ]
    )


def follow_pipelinerun(namespace: str, name: str, *, poll_interval: int = 5) -> bool:
    require_command("oc")

    if shutil.which("tkn") is not None:
        step(f"Following PipelineRun {name} in namespace {namespace}")
        subprocess.run(
            ["tkn", "pipelinerun", "logs", "-f", "-n", namespace, name],
            check=False,
        )
        state, finished, succeeded, message = pipelinerun_state(
            get_pipelinerun(namespace, name)
        )
        if succeeded:
            success(f"{name}: {state}")
        else:
            warning(f"{name}: {state}")
        if message:
            detail(message)
        return finished and succeeded

    last_state: tuple[str, bool, bool, str] | None = None
    step(f"Watching PipelineRun {name} in namespace {namespace}")
    while True:
        state = pipelinerun_state(get_pipelinerun(namespace, name))
        if state != last_state:
            label, _, _, message = state
            detail(f"{name}: {label}")
            if message:
                detail(message)
            last_state = state
        label, finished, succeeded, _ = state
        if finished:
            if succeeded:
                success(f"{name}: {label}")
            else:
                warning(f"{name}: {label}")
            return succeeded
        time.sleep(poll_interval)
