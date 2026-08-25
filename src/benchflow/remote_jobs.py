from __future__ import annotations

import json
import os
import secrets
import shutil
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import yaml

from .cluster import (
    CommandError,
    create_manifest,
    require_any_command,
    require_command,
    run_command,
    run_json_command,
    use_kubeconfig,
)
from .models import ResolvedRunPlan, sanitize_name
from .ui import detail

DEFAULT_REMOTE_IMAGE = "ghcr.io/albertoperdomo2/benchflow:latest"
REMOTE_RESULTS_PVC = "benchmark-results"
REMOTE_RESULTS_MOUNT = "/benchmark-results"
REMOTE_RESULTS_ROOT = f"{REMOTE_RESULTS_MOUNT}/remote-jobs"
REMOTE_API_RETRY_TIMEOUT_SECONDS = 600
REMOTE_API_RETRY_MAX_DELAY_SECONDS = 30

_TRANSIENT_KUBERNETES_ERROR_MARKERS = (
    "connection refused",
    "connection reset",
    "context deadline exceeded",
    "end of file",
    "error from server (internalerror)",
    "error from server (serviceunavailable)",
    "error from server (timeout)",
    "failed calling webhook",
    "i/o timeout",
    "internal error occurred",
    "no endpoints available for service",
    "server was unable to return a response in the time allotted",
    "service unavailable",
    "temporarily unavailable",
    "too many requests",
    "tls handshake timeout",
    "unexpected eof",
)

_PASSTHROUGH_ENV = (
    "HF_TOKEN",
    "HF_HUB_DOWNLOAD_TIMEOUT",
    "HF_HUB_ETAG_TIMEOUT",
    "HF_HUB_ENABLE_HF_TRANSFER",
    "MLFLOW_WORKSPACE",
    "MLFLOW_WORKSPACE_STORE_URI",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
    "MLFLOW_TRACKING_INSECURE_TLS",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_DEFAULT_REGION",
    "MLFLOW_S3_BUCKET_NAME",
)


@dataclass(frozen=True, slots=True)
class RemoteJobResult:
    job_name: str
    pod_name: str


class RemoteJobFailed(CommandError):
    def __init__(self, *, job_name: str, pod_name: str = "") -> None:
        super().__init__(f"remote job {job_name} failed")
        self.job_name = job_name
        self.pod_name = pod_name


def is_transient_kubernetes_error(exc: BaseException) -> bool:
    message = str(exc).strip().lower()
    return any(marker in message for marker in _TRANSIENT_KUBERNETES_ERROR_MARKERS)


def _retry_transient_kubernetes_operation(
    operation: Callable[[], Any],
    *,
    description: str,
    timeout_seconds: float = REMOTE_API_RETRY_TIMEOUT_SECONDS,
) -> Any:
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    attempt = 0
    while True:
        attempt += 1
        try:
            return operation()
        except CommandError as exc:
            if not is_transient_kubernetes_error(exc):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CommandError(
                    f"{description} did not recover within {timeout_seconds:.0f}s: {exc}"
                ) from exc
            delay = min(
                REMOTE_API_RETRY_MAX_DELAY_SECONDS,
                2 ** min(attempt - 1, 5),
                remaining,
            )
            detail(
                f"Transient Kubernetes error while {description}; "
                f"retrying in {delay:.0f}s: {exc}"
            )
            time.sleep(delay)


def _resource_payload_or_none(
    *, namespace: str, kind: str, name: str
) -> dict[str, Any] | None:
    kubectl_cmd = require_any_command("oc", "kubectl")
    result = run_command(
        [kubectl_cmd, "get", kind, name, "-n", namespace, "-o", "json"],
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return json.loads(result.stdout or "{}")
    details = (result.stderr or result.stdout or "command failed").strip()
    if "notfound" in details.lower() or "not found" in details.lower():
        return None
    raise CommandError(
        f"{kubectl_cmd} get {kind} {name} -n {namespace} -o json: {details}"
    )


def _create_manifest_idempotently(
    manifest_yaml: str,
    *,
    namespace: str,
    kind: str,
    name: str,
) -> None:
    def create_or_confirm() -> None:
        try:
            create_manifest(manifest_yaml, namespace)
            return
        except CommandError as exc:
            try:
                existing = _resource_payload_or_none(
                    namespace=namespace, kind=kind, name=name
                )
            except CommandError:
                if is_transient_kubernetes_error(exc):
                    raise exc
                raise
            if existing is not None:
                detail(
                    f"Confirmed {kind} {name} exists after an ambiguous create response"
                )
                return
            raise

    _retry_transient_kubernetes_operation(
        create_or_confirm,
        description=f"creating {kind} {name} in namespace {namespace}",
    )


def _remote_image() -> str:
    return (
        os.environ.get("BENCHFLOW_REMOTE_IMAGE")
        or os.environ.get("BENCHFLOW_IMAGE")
        or DEFAULT_REMOTE_IMAGE
    )


def _remote_run_plan_json(plan: ResolvedRunPlan) -> str:
    payload = plan.to_dict()
    payload["target_cluster"] = {"kubeconfig": "", "kubeconfig_secret": ""}
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def remote_run_plan_json(plan: ResolvedRunPlan) -> str:
    return _remote_run_plan_json(plan)


def _remote_env(extra_env: dict[str, str] | None = None) -> list[dict[str, str]]:
    env: list[dict[str, str]] = []
    for name in _PASSTHROUGH_ENV:
        value = os.environ.get(name)
        if value:
            env.append({"name": name, "value": value})
    for name, value in (extra_env or {}).items():
        if value:
            env.append({"name": str(name), "value": str(value)})
    return env


def _create_remote_job(
    plan: ResolvedRunPlan,
    *,
    job_name: str,
    job_kind: str,
    args: list[str],
    env: dict[str, str] | None = None,
    volume_mounts: list[dict[str, Any]] | None = None,
    volumes: list[dict[str, Any]] | None = None,
) -> None:
    safe_kind = sanitize_name(job_kind, max_length=20)
    manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": plan.deployment.namespace,
            "labels": {
                "app.kubernetes.io/name": "benchflow",
                "benchflow.io/experiment": plan.metadata.name,
                "benchflow.io/remote-job-kind": safe_kind,
            },
        },
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": plan.ttl_seconds_after_finished,
            "template": {
                "metadata": {
                    "labels": {
                        "app.kubernetes.io/name": "benchflow",
                        "benchflow.io/experiment": plan.metadata.name,
                        "benchflow.io/remote-job-kind": safe_kind,
                    }
                },
                "spec": {
                    "restartPolicy": "Never",
                    "serviceAccountName": plan.service_account,
                    "containers": [
                        {
                            "name": "main",
                            "image": _remote_image(),
                            "imagePullPolicy": "Always",
                            "command": ["bflow"],
                            "args": args,
                            "env": _remote_env(env),
                            "volumeMounts": volume_mounts or [],
                        }
                    ],
                    "volumes": volumes or [],
                },
            },
        },
    }
    with use_kubeconfig(plan.target_cluster.kubeconfig):
        _create_manifest_idempotently(
            yaml.safe_dump(manifest, sort_keys=False),
            namespace=plan.deployment.namespace,
            kind="job",
            name=job_name,
        )
    detail(f"Created remote {job_kind} job {job_name}")


def _generate_remote_job_name(plan: ResolvedRunPlan, job_kind: str) -> str:
    safe_kind = sanitize_name(job_kind, max_length=20)
    safe_name = sanitize_name(plan.metadata.name, max_length=20)
    suffix = secrets.token_hex(2)
    return f"benchflow-{safe_kind}-{safe_name}-{suffix}"


def generate_remote_job_name(plan: ResolvedRunPlan, job_kind: str) -> str:
    return _generate_remote_job_name(plan, job_kind)


def remote_job_results_dir(job_name: str) -> str:
    return f"{REMOTE_RESULTS_ROOT}/{job_name}"


def remote_job_benchmark_dir(job_name: str) -> str:
    return f"{remote_job_results_dir(job_name)}/benchmark"


def remote_job_artifacts_dir(job_name: str) -> str:
    return f"{remote_job_results_dir(job_name)}/artifacts"


def _merge_volume_mounts(
    current: list[dict[str, Any]] | None,
    extra: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = list(current or [])
    existing_names = {str(item.get("name") or "") for item in merged}
    for item in extra:
        if str(item.get("name") or "") not in existing_names:
            merged.append(item)
    return merged


def _merge_volumes(
    current: list[dict[str, Any]] | None,
    extra: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = list(current or [])
    existing_names = {str(item.get("name") or "") for item in merged}
    for item in extra:
        if str(item.get("name") or "") not in existing_names:
            merged.append(item)
    return merged


def _results_volume_mounts() -> list[dict[str, Any]]:
    return [{"name": "benchmark-results", "mountPath": REMOTE_RESULTS_MOUNT}]


def _results_volumes() -> list[dict[str, Any]]:
    return [
        {
            "name": "benchmark-results",
            "persistentVolumeClaim": {"claimName": REMOTE_RESULTS_PVC},
        }
    ]


def _list_job_pods(namespace: str, job_name: str, kubeconfig: str) -> list[str]:
    kubectl_cmd = require_any_command("oc", "kubectl")
    with use_kubeconfig(kubeconfig):
        payload = run_json_command(
            [
                kubectl_cmd,
                "get",
                "pods",
                "-n",
                namespace,
                "-l",
                f"job-name={job_name}",
                "-o",
                "json",
            ]
        )
    return [
        str(item.get("metadata", {}).get("name") or "")
        for item in payload.get("items", [])
        if str(item.get("metadata", {}).get("name") or "").strip()
    ]


def _remote_job_logs(namespace: str, pod_name: str, kubeconfig: str) -> str:
    kubectl_cmd = require_any_command("oc", "kubectl")
    with use_kubeconfig(kubeconfig):
        result = run_command(
            [kubectl_cmd, "logs", pod_name, "-n", namespace, "-c", "main"],
            capture_output=True,
            check=False,
        )
    return (result.stdout or result.stderr or "").strip()


def wait_for_remote_job(
    plan: ResolvedRunPlan,
    *,
    job_name: str,
    timeout_seconds: int | None = 3600,
) -> RemoteJobResult:
    kubectl_cmd = require_any_command("oc", "kubectl")
    deadline = time.time() + timeout_seconds if timeout_seconds is not None else None
    last_pod_name = ""
    with use_kubeconfig(plan.target_cluster.kubeconfig):
        while deadline is None or time.time() < deadline:
            remaining = (
                REMOTE_API_RETRY_TIMEOUT_SECONDS
                if deadline is None
                else max(
                    1.0, min(REMOTE_API_RETRY_TIMEOUT_SECONDS, deadline - time.time())
                )
            )
            payload = _retry_transient_kubernetes_operation(
                lambda: run_json_command(
                    [
                        kubectl_cmd,
                        "get",
                        "job",
                        job_name,
                        "-n",
                        plan.deployment.namespace,
                        "-o",
                        "json",
                    ]
                ),
                description=f"reading remote job {job_name}",
                timeout_seconds=remaining,
            )
            status = payload.get("status", {}) or {}
            pod_names = _retry_transient_kubernetes_operation(
                lambda: _list_job_pods(
                    plan.deployment.namespace,
                    job_name,
                    plan.target_cluster.kubeconfig,
                ),
                description=f"listing pods for remote job {job_name}",
                timeout_seconds=remaining,
            )
            if pod_names:
                last_pod_name = pod_names[0]
            if int(status.get("succeeded", 0) or 0) > 0:
                if not last_pod_name:
                    raise CommandError(
                        f"remote job {job_name} succeeded but no pod was found"
                    )
                return RemoteJobResult(job_name=job_name, pod_name=last_pod_name)
            if int(status.get("failed", 0) or 0) > 0:
                logs = (
                    _remote_job_logs(
                        plan.deployment.namespace,
                        last_pod_name,
                        plan.target_cluster.kubeconfig,
                    )
                    if last_pod_name
                    else ""
                )
                detail(logs) if logs else None
                raise RemoteJobFailed(job_name=job_name, pod_name=last_pod_name)
            time.sleep(3)
    raise CommandError(f"timed out waiting for remote job {job_name}")


def _create_reader_pod(plan: ResolvedRunPlan, *, pod_name: str) -> None:
    manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": pod_name,
            "namespace": plan.deployment.namespace,
            "labels": {
                "app.kubernetes.io/name": "benchflow",
                "benchflow.io/experiment": plan.metadata.name,
                "benchflow.io/remote-job-kind": "reader",
            },
        },
        "spec": {
            "restartPolicy": "Never",
            "serviceAccountName": plan.service_account,
            "containers": [
                {
                    "name": "main",
                    "image": _remote_image(),
                    "command": ["sh", "-c", "sleep 600"],
                    "volumeMounts": _results_volume_mounts(),
                }
            ],
            "volumes": _results_volumes(),
        },
    }
    with use_kubeconfig(plan.target_cluster.kubeconfig):
        _create_manifest_idempotently(
            yaml.safe_dump(manifest, sort_keys=False),
            namespace=plan.deployment.namespace,
            kind="pod",
            name=pod_name,
        )


def _wait_for_reader_pod(
    plan: ResolvedRunPlan, *, pod_name: str, timeout_seconds: int = 600
) -> None:
    kubectl_cmd = require_any_command("oc", "kubectl")
    deadline = time.time() + timeout_seconds
    with use_kubeconfig(plan.target_cluster.kubeconfig):
        while time.time() < deadline:
            payload = _retry_transient_kubernetes_operation(
                lambda: run_json_command(
                    [
                        kubectl_cmd,
                        "get",
                        "pod",
                        pod_name,
                        "-n",
                        plan.deployment.namespace,
                        "-o",
                        "json",
                    ]
                ),
                description=f"reading remote reader pod {pod_name}",
                timeout_seconds=max(
                    1.0, min(REMOTE_API_RETRY_TIMEOUT_SECONDS, deadline - time.time())
                ),
            )
            phase = str(payload.get("status", {}).get("phase") or "")
            conditions = payload.get("status", {}).get("conditions") or []
            ready = False
            if isinstance(conditions, list):
                for condition in conditions:
                    if not isinstance(condition, dict):
                        continue
                    if condition.get("type") == "Ready":
                        ready = str(condition.get("status") or "").lower() == "true"
                        break
            if phase == "Running" and ready:
                return
            if phase in {"Failed", "Succeeded"}:
                raise CommandError(
                    f"remote reader pod {pod_name} exited before becoming ready"
                )
            time.sleep(2)
    raise CommandError(f"timed out waiting for remote reader pod {pod_name}")


def _reader_supports_rsync(plan: ResolvedRunPlan, *, pod_name: str) -> bool:
    kubectl_cmd = require_any_command("oc", "kubectl")
    with use_kubeconfig(plan.target_cluster.kubeconfig):
        result = run_command(
            [
                kubectl_cmd,
                "exec",
                "-n",
                plan.deployment.namespace,
                pod_name,
                "-c",
                "main",
                "--",
                "sh",
                "-c",
                "command -v rsync >/dev/null 2>&1",
            ],
            capture_output=True,
            check=False,
        )
    return result.returncode == 0


def _copy_remote_results_directory_with_rsync(
    plan: ResolvedRunPlan,
    *,
    reader_pod: str,
    remote_path: str,
    local_dir: Path,
) -> None:
    require_command("oc")
    require_command("rsync")
    source = f"{reader_pod}:{remote_path.rstrip('/')}/"
    with use_kubeconfig(plan.target_cluster.kubeconfig):
        run_command(
            [
                "rsync",
                "-rltz",
                "--no-perms",
                "--no-owner",
                "--no-group",
                "--blocking-io",
                "--omit-dir-times",
                f"--rsh=oc rsh -n {plan.deployment.namespace} -c main",
                source,
                str(local_dir),
            ]
        )


def _copy_remote_results_directory_with_oc_cp(
    plan: ResolvedRunPlan,
    *,
    reader_pod: str,
    remote_path: str,
    local_dir: Path,
) -> None:
    kubectl_cmd = require_any_command("oc", "kubectl")
    with use_kubeconfig(plan.target_cluster.kubeconfig):
        run_command(
            [
                kubectl_cmd,
                "cp",
                f"{plan.deployment.namespace}/{reader_pod}:{remote_path}/.",
                str(local_dir),
            ]
        )


def _validated_remote_results_path(remote_path: str) -> str:
    target = PurePosixPath(str(remote_path).strip())
    root = PurePosixPath(REMOTE_RESULTS_ROOT)
    if not target.is_absolute() or target == root or root not in target.parents:
        raise CommandError(
            f"refusing to delete path outside a remote job result directory: {remote_path}"
        )
    return target.as_posix()


def _delete_remote_path_from_reader(
    plan: ResolvedRunPlan,
    *,
    reader_pod: str,
    remote_path: str,
) -> None:
    kubectl_cmd = require_any_command("oc", "kubectl")
    validated_path = _validated_remote_results_path(remote_path)

    def delete_once() -> None:
        with use_kubeconfig(plan.target_cluster.kubeconfig):
            run_command(
                [
                    kubectl_cmd,
                    "exec",
                    "-n",
                    plan.deployment.namespace,
                    reader_pod,
                    "-c",
                    "main",
                    "--",
                    "sh",
                    "-c",
                    (
                        "set -eu; target=$1; "
                        '[ ! -e "$target" ] && exit 0; '
                        '[ -d "$target" ] && [ ! -L "$target" ]; '
                        'rm -rf -- "$target"'
                    ),
                    "benchflow-delete",
                    validated_path,
                ]
            )

    _retry_transient_kubernetes_operation(
        delete_once,
        description=f"deleting uploaded remote results at {validated_path}",
    )


def copy_remote_results_directory(
    plan: ResolvedRunPlan,
    *,
    remote_path: str,
    local_dir: Path,
    cleanup: bool = True,
) -> None:
    kubectl_cmd = require_any_command("oc", "kubectl")
    reader_pod = _generate_remote_job_name(plan, "reader")
    local_dir.mkdir(parents=True, exist_ok=True)
    _create_reader_pod(plan, pod_name=reader_pod)
    try:
        _wait_for_reader_pod(plan, pod_name=reader_pod)
        used_rsync = False
        attempted_rsync = False
        rsync_available = shutil.which("rsync") is not None
        oc_available = shutil.which("oc") is not None
        if (
            rsync_available
            and oc_available
            and _reader_supports_rsync(plan, pod_name=reader_pod)
        ):
            attempted_rsync = True
            try:
                detail(
                    f"Copying remote results from {reader_pod} with rsync over oc rsh"
                )
                _retry_transient_kubernetes_operation(
                    lambda: _copy_remote_results_directory_with_rsync(
                        plan,
                        reader_pod=reader_pod,
                        remote_path=remote_path,
                        local_dir=local_dir,
                    ),
                    description=f"copying remote results from {reader_pod} with rsync",
                )
                used_rsync = True
            except CommandError as exc:
                detail(f"rsync copy failed, falling back to oc cp: {exc}")
        if not used_rsync:
            if attempted_rsync:
                pass
            elif not rsync_available:
                detail("rsync not available locally, falling back to oc cp")
            elif not oc_available:
                detail(
                    "oc not available locally for rsync transport, falling back to oc cp"
                )
            else:
                detail("reader pod does not have rsync, falling back to oc cp")
            _retry_transient_kubernetes_operation(
                lambda: _copy_remote_results_directory_with_oc_cp(
                    plan,
                    reader_pod=reader_pod,
                    remote_path=remote_path,
                    local_dir=local_dir,
                ),
                description=f"copying remote results from {reader_pod} with oc cp",
            )
        kubectl_cmd = require_any_command("oc", "kubectl")
        with use_kubeconfig(plan.target_cluster.kubeconfig):
            if cleanup:
                _delete_remote_path_from_reader(
                    plan,
                    reader_pod=reader_pod,
                    remote_path=remote_path,
                )
    finally:
        with use_kubeconfig(plan.target_cluster.kubeconfig):
            run_command(
                [
                    kubectl_cmd,
                    "delete",
                    "pod",
                    reader_pod,
                    "-n",
                    plan.deployment.namespace,
                    "--ignore-not-found",
                    "--wait=false",
                ],
                check=False,
            )


def delete_remote_results_directory(
    plan: ResolvedRunPlan,
    *,
    remote_path: str,
) -> None:
    reader_pod = _generate_remote_job_name(plan, "reader")
    _create_reader_pod(plan, pod_name=reader_pod)
    try:
        _wait_for_reader_pod(plan, pod_name=reader_pod)
        _delete_remote_path_from_reader(
            plan,
            reader_pod=reader_pod,
            remote_path=remote_path,
        )
    finally:
        kubectl_cmd = require_any_command("oc", "kubectl")
        with use_kubeconfig(plan.target_cluster.kubeconfig):
            run_command(
                [
                    kubectl_cmd,
                    "delete",
                    "pod",
                    reader_pod,
                    "-n",
                    plan.deployment.namespace,
                    "--ignore-not-found",
                    "--wait=false",
                ],
                check=False,
            )


def run_remote_job(
    plan: ResolvedRunPlan,
    *,
    job_kind: str,
    args: list[str] | None = None,
    args_builder: Callable[[str], list[str]] | None = None,
    env: dict[str, str] | None = None,
    volume_mounts: list[dict[str, Any]] | None = None,
    volumes: list[dict[str, Any]] | None = None,
    timeout_seconds: int | None = 3600,
    mount_results_pvc: bool = False,
    job_name: str | None = None,
) -> RemoteJobResult:
    if (args is None) == (args_builder is None):
        raise CommandError(
            "provide exactly one of args or args_builder for remote jobs"
        )
    job_name = str(job_name or _generate_remote_job_name(plan, job_kind)).strip()
    if not job_name:
        raise CommandError("remote job name must not be empty")
    resolved_args = list(
        args_builder(job_name) if args_builder is not None else args or []
    )
    resolved_volume_mounts = volume_mounts
    resolved_volumes = volumes
    if mount_results_pvc:
        resolved_volume_mounts = _merge_volume_mounts(
            resolved_volume_mounts, _results_volume_mounts()
        )
        resolved_volumes = _merge_volumes(resolved_volumes, _results_volumes())
    _create_remote_job(
        plan,
        job_name=job_name,
        job_kind=job_kind,
        args=resolved_args,
        env=env,
        volume_mounts=resolved_volume_mounts,
        volumes=resolved_volumes,
    )
    return wait_for_remote_job(plan, job_name=job_name, timeout_seconds=timeout_seconds)
