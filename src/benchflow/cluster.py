from __future__ import annotations

import os
import json
import shutil
import subprocess
from contextlib import contextmanager
from urllib.parse import urlsplit, urlunsplit
from pathlib import Path
from typing import Any


TARGET_KUBECONFIG_HOST_ALIASES_ANNOTATION = "benchflow.io/host-aliases"


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

    kubeconfig_path = Path(kubeconfig).expanduser()
    if not kubeconfig_path.exists():
        raise CommandError(f"target kubeconfig not found: {kubeconfig_path}")
    kubeconfig_path = kubeconfig_path.resolve()
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


def _with_default_port(url: str, default_port: int) -> str:
    parsed = urlsplit(url)
    if parsed.port is not None or not parsed.netloc:
        return url
    netloc = f"{parsed.hostname}:{default_port}"
    if parsed.username:
        auth = parsed.username
        if parsed.password:
            auth += f":{parsed.password}"
        netloc = f"{auth}@{netloc}"
    return urlunsplit(
        (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
    )


def _target_scope(target: Any) -> str:
    scope = str(getattr(target, "endpoint_scope", "") or "external").strip()
    if scope not in {"external", "internal"}:
        raise CommandError(f"unsupported target endpoint scope: {scope}")
    return scope


def _service_payload(
    kubectl_cmd: str, namespace: str, name: str
) -> dict[str, Any] | None:
    result = run_command(
        [kubectl_cmd, "get", "service", name, "-n", namespace, "-o", "json"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise CommandError(
            f"{kubectl_cmd} get service {name} -n {namespace}: "
            "command did not return valid JSON"
        ) from exc
    return payload if isinstance(payload, dict) else None


def _service_port(payload: dict[str, Any]) -> int:
    ports = payload.get("spec", {}).get("ports") or []
    if not isinstance(ports, list) or not ports:
        return 80
    preferred_names = {"http", "http2", "http-80", "web"}
    for port in ports:
        if not isinstance(port, dict):
            continue
        if str(port.get("name") or "") in preferred_names:
            return int(port.get("port") or 80)
    first = ports[0] if isinstance(ports[0], dict) else {}
    return int(first.get("port") or 80)


def _gateway_internal_service_url(
    kubectl_cmd: str, namespace: str, gateway_name: str
) -> str:
    candidates = [
        gateway_name,
        f"{gateway_name}-istio",
    ]
    for service_name in candidates:
        payload = _service_payload(kubectl_cmd, namespace, service_name)
        if payload:
            port = _service_port(payload)
            suffix = "" if port == 80 else f":{port}"
            return f"http://{service_name}.{namespace}.svc.cluster.local{suffix}"
    raise CommandError(
        f"Gateway {gateway_name} in namespace {namespace} does not have a "
        "known internal Service"
    )


def _llminferenceservice_internal_url(resource_name: str, namespace: str) -> str:
    return (
        f"https://{resource_name}-kserve-workload-svc."
        f"{namespace}.svc.cluster.local:8000"
    )


def _inferenceservice_internal_url(resource_name: str, namespace: str) -> str:
    return f"http://{resource_name}-predictor.{namespace}.svc.cluster.local:8080"


def resolve_target_base_url(target: Any, namespace: str) -> str:
    endpoint_scope = _target_scope(target)
    if target.discovery == "static":
        if not target.base_url:
            raise CommandError(
                "target discovery is static but target.base_url is empty"
            )
        return str(target.base_url).rstrip("/")

    if target.discovery == "gateway-status-url":
        kubectl_cmd = require_any_command("oc", "kubectl")
        resource_name = str(target.resource_name or "").strip()
        if not resource_name:
            raise CommandError(
                "target discovery is gateway-status-url but target.resource_name is empty"
            )
        if endpoint_scope == "internal":
            return _gateway_internal_service_url(kubectl_cmd, namespace, resource_name)
        payload = run_json_command(
            [
                kubectl_cmd,
                "get",
                "gateway",
                resource_name,
                "-n",
                namespace,
                "-o",
                "json",
            ]
        )
        addresses = payload.get("status", {}).get("addresses") or []
        if not isinstance(addresses, list) or not addresses:
            raise CommandError(
                f"Gateway {resource_name} in namespace {namespace} does not have status.addresses yet"
            )
        address = addresses[0] if isinstance(addresses[0], dict) else {}
        value = str(address.get("value") or address.get("hostname") or "").strip()
        if not value:
            raise CommandError(
                f"Gateway {resource_name} in namespace {namespace} does not have a usable status address yet"
            )
        if value.startswith(("http://", "https://")):
            return value.rstrip("/")
        return f"http://{value}".rstrip("/")

    if target.discovery == "llminferenceservice-status-url":
        kubectl_cmd = require_any_command("oc", "kubectl")
        resource_name = str(target.resource_name or "").strip()
        if not resource_name:
            raise CommandError(
                "target discovery is llminferenceservice-status-url but "
                "target.resource_name is empty"
            )
        if endpoint_scope == "internal":
            return _llminferenceservice_internal_url(resource_name, namespace)
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

    if target.discovery == "inferenceservice-status-url":
        kubectl_cmd = require_any_command("oc", "kubectl")
        resource_name = str(target.resource_name or "").strip()
        if not resource_name:
            raise CommandError(
                "target discovery is inferenceservice-status-url but "
                "target.resource_name is empty"
            )
        if endpoint_scope == "internal":
            return _inferenceservice_internal_url(resource_name, namespace)
        payload = run_json_command(
            [
                kubectl_cmd,
                "get",
                "inferenceservice",
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
                f"InferenceService {resource_name} in namespace {namespace} "
                "does not have status.url yet"
            )
        return _with_default_port(url.rstrip("/"), 8080)

    raise CommandError(f"unsupported target discovery strategy: {target.discovery}")


def create_manifest(manifest_yaml: str, namespace: str) -> dict[str, Any]:
    require_command("oc")
    result = run_command(
        ["oc", "create", "-n", namespace, "-f", "-", "-o", "json"],
        input_text=manifest_yaml,
        capture_output=True,
    )
    return json.loads(result.stdout)


def load_target_kubeconfig_host_aliases(
    namespace: str, secret_name: str
) -> dict[str, str]:
    kubectl_cmd = require_any_command("oc", "kubectl")
    result = run_command(
        [kubectl_cmd, "get", "secret", secret_name, "-n", namespace, "-o", "json"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return {}
    payload = json.loads(result.stdout or "{}")
    raw = str(
        payload.get("metadata", {})
        .get("annotations", {})
        .get(TARGET_KUBECONFIG_HOST_ALIASES_ANNOTATION, "")
        or ""
    ).strip()
    if not raw:
        return {}
    try:
        aliases = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CommandError(
            f"target kubeconfig Secret {secret_name} in namespace {namespace} has invalid host alias metadata"
        ) from exc
    if not isinstance(aliases, dict):
        raise CommandError(
            f"target kubeconfig Secret {secret_name} in namespace {namespace} has invalid host alias metadata"
        )
    return {
        str(hostname).strip(): str(ip_address).strip()
        for hostname, ip_address in aliases.items()
        if str(hostname).strip() and str(ip_address).strip()
    }


def get_current_namespace() -> str:
    require_command("oc")
    result = run_command(["oc", "project", "-q"], capture_output=True)
    return result.stdout.strip()
