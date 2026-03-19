from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .cluster import (
    CommandError,
    create_manifest,
    require_any_command,
    run_command,
    run_json_command,
    use_kubeconfig,
)
from .models import ResolvedRunPlan, sanitize_name
from .ui import detail

DEFAULT_REMOTE_IMAGE = "ghcr.io/albertoperdomo2/benchflow/benchflow:latest"
REMOTE_BENCHMARK_DIR = "/tmp/benchflow-remote/benchmark"
REMOTE_ARTIFACTS_DIR = "/tmp/benchflow-remote/artifacts"

_PASSTHROUGH_ENV = (
    "HF_TOKEN",
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
    job_kind: str,
    args: list[str],
    env: dict[str, str] | None = None,
    volume_mounts: list[dict[str, Any]] | None = None,
    volumes: list[dict[str, Any]] | None = None,
) -> str:
    safe_kind = sanitize_name(job_kind, max_length=20)
    safe_name = sanitize_name(plan.metadata.name, max_length=20)
    manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "generateName": f"benchflow-{safe_kind}-{safe_name}-",
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
        created = create_manifest(
            yaml.safe_dump(manifest, sort_keys=False),
            plan.deployment.namespace,
        )
    job_name = str(created.get("metadata", {}).get("name") or "").strip()
    if not job_name:
        raise CommandError("remote job submission returned no job name")
    detail(f"Created remote {job_kind} job {job_name}")
    return job_name


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
    timeout_seconds: int = 3600,
) -> RemoteJobResult:
    kubectl_cmd = require_any_command("oc", "kubectl")
    deadline = time.time() + timeout_seconds
    last_pod_name = ""
    with use_kubeconfig(plan.target_cluster.kubeconfig):
        while time.time() < deadline:
            payload = run_json_command(
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
            )
            status = payload.get("status", {}) or {}
            pod_names = _list_job_pods(
                plan.deployment.namespace, job_name, plan.target_cluster.kubeconfig
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


def copy_remote_directory(
    plan: ResolvedRunPlan,
    *,
    pod_name: str,
    remote_path: str,
    local_dir: Path,
) -> None:
    kubectl_cmd = require_any_command("oc", "kubectl")
    local_dir.mkdir(parents=True, exist_ok=True)
    with use_kubeconfig(plan.target_cluster.kubeconfig):
        run_command(
            [
                kubectl_cmd,
                "cp",
                f"{plan.deployment.namespace}/{pod_name}:{remote_path}/.",
                str(local_dir),
            ]
        )


def run_remote_job(
    plan: ResolvedRunPlan,
    *,
    job_kind: str,
    args: list[str],
    env: dict[str, str] | None = None,
    volume_mounts: list[dict[str, Any]] | None = None,
    volumes: list[dict[str, Any]] | None = None,
    timeout_seconds: int = 3600,
) -> RemoteJobResult:
    job_name = _create_remote_job(
        plan,
        job_kind=job_kind,
        args=args,
        env=env,
        volume_mounts=volume_mounts,
        volumes=volumes,
    )
    return wait_for_remote_job(plan, job_name=job_name, timeout_seconds=timeout_seconds)
