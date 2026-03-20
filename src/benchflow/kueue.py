from __future__ import annotations

import base64
import json
import math
import tempfile
import time
from datetime import datetime, timezone
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
from .contracts import ExecutionSummary, ResolvedRunPlan, ValidationError
from .models import sanitize_name
from .ui import detail, step, success, warning

KUEUE_NAMESPACE = "kueue-system"
KUEUE_API_VERSION = "kueue.x-k8s.io/v1beta2"
REMOTE_GPU_RESOURCE = "benchflow.io/remote-gpu"
REMOTE_CAPACITY_ADMISSION_CHECK = "benchflow-remote-capacity"
REMOTE_CAPACITY_CONTROLLER_NAME = "benchflow.io/remote-capacity"
REMOTE_CAPACITY_CONTROLLER_DEPLOYMENT = "benchflow-remote-capacity-controller"
DEFAULT_RESOURCE_FLAVOR = "default-flavor"
LOCAL_CLUSTER_QUEUE = "local"
DEFAULT_CONTROLLER_IMAGE = "ghcr.io/albertoperdomo2/benchflow/benchflow:latest"
KUEUE_SKIP_RESERVATION_LABEL = "benchflow.io/kueue-skip-reservation"
REQUESTED_GPUS_LABEL = "benchflow.io/requested-gpus"
CLUSTER_QUEUE_LABEL = "benchflow.io/cluster-name"
TARGET_KUBECONFIG_SECRET_LABEL = "benchflow.io/target-kubeconfig-secret"
EXECUTION_NAME_LABEL = "benchflow.io/execution-name"
SUBMISSION_CONFIGMAP_LABEL = "benchflow.io/submission-configmap"
SUBMISSION_MANIFEST_KEY = "manifest.json"


def _now_rfc3339() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _duration_seconds(value: str) -> int:
    cleaned = str(value).strip().lower()
    if not cleaned:
        raise ValidationError("duration must not be empty")
    total = 0.0
    current = ""
    for char in cleaned:
        if char.isdigit() or char == ".":
            current += char
            continue
        if not current:
            raise ValidationError(f"invalid duration: {value!r}")
        magnitude = float(current)
        current = ""
        if char == "h":
            total += magnitude * 3600.0
        elif char == "m":
            total += magnitude * 60.0
        elif char == "s":
            total += magnitude
        else:
            raise ValidationError(f"invalid duration: {value!r}")
    if current:
        total += float(current)
    return max(1, math.ceil(total))


def cluster_queue_name(cluster_name: str) -> str:
    normalized = str(cluster_name).strip()
    if not normalized:
        raise ValidationError("cluster queue name must not be empty")
    return sanitize_name(normalized, max_length=63)


def cluster_name_from_plan(plan: ResolvedRunPlan) -> str:
    if plan.target_cluster.kubeconfig_secret:
        return cluster_queue_name(plan.target_cluster.kubeconfig_secret)
    return LOCAL_CLUSTER_QUEUE


def cluster_name_from_plans(plans: list[ResolvedRunPlan]) -> str:
    if not plans:
        raise ValidationError("at least one RunPlan is required")
    names = {cluster_name_from_plan(plan) for plan in plans}
    if len(names) != 1:
        raise ValidationError("matrix runs must resolve to one cluster queue")
    return next(iter(names))


def target_kubeconfig_secret_from_plan(plan: ResolvedRunPlan) -> str:
    return str(plan.target_cluster.kubeconfig_secret or "").strip()


def target_kubeconfig_secret_from_plans(plans: list[ResolvedRunPlan]) -> str:
    if not plans:
        raise ValidationError("at least one RunPlan is required")
    secret_names = {target_kubeconfig_secret_from_plan(plan) for plan in plans}
    if len(secret_names) != 1:
        raise ValidationError(
            "matrix runs must resolve to one target kubeconfig Secret"
        )
    return next(iter(secret_names))


def requested_gpus(plan: ResolvedRunPlan) -> int:
    if not (plan.stages.deploy or plan.stages.benchmark):
        return 0
    replicas = max(1, int(plan.deployment.runtime.replicas or 1))
    tensor_parallelism = max(1, int(plan.deployment.runtime.tensor_parallelism or 1))
    return replicas * tensor_parallelism


def requested_gpus_for_matrix(plans: list[ResolvedRunPlan]) -> int:
    if not plans:
        raise ValidationError("at least one RunPlan is required")
    return max(requested_gpus(plan) for plan in plans)


def execution_labels_for_plan(
    plan: ResolvedRunPlan,
    *,
    skip_reservation: bool = False,
) -> dict[str, str]:
    labels = {
        CLUSTER_QUEUE_LABEL: cluster_name_from_plan(plan),
        REQUESTED_GPUS_LABEL: str(requested_gpus(plan)),
        KUEUE_SKIP_RESERVATION_LABEL: str(skip_reservation).lower(),
    }
    secret_name = target_kubeconfig_secret_from_plan(plan)
    if secret_name:
        labels[TARGET_KUBECONFIG_SECRET_LABEL] = secret_name
    return labels


def execution_labels_for_matrix(
    plans: list[ResolvedRunPlan],
    *,
    skip_reservation: bool = False,
) -> dict[str, str]:
    labels = {
        CLUSTER_QUEUE_LABEL: cluster_name_from_plans(plans),
        REQUESTED_GPUS_LABEL: str(
            0 if skip_reservation else requested_gpus_for_matrix(plans)
        ),
        KUEUE_SKIP_RESERVATION_LABEL: str(skip_reservation).lower(),
    }
    secret_name = target_kubeconfig_secret_from_plans(plans)
    if secret_name:
        labels[TARGET_KUBECONFIG_SECRET_LABEL] = secret_name
    return labels


def reservation_required_for_labels(labels: dict[str, str] | None) -> bool:
    if not labels:
        return False
    return (
        str(labels.get(KUEUE_SKIP_RESERVATION_LABEL, "false")).strip().lower() != "true"
    )


def requested_gpus_from_labels(labels: dict[str, str] | None) -> int:
    if not labels:
        return 0
    try:
        return int(str(labels.get(REQUESTED_GPUS_LABEL) or "0"))
    except ValueError:
        return 0


def queue_name_from_labels(labels: dict[str, str] | None) -> str:
    raw = "" if not labels else str(labels.get(CLUSTER_QUEUE_LABEL) or "").strip()
    return cluster_queue_name(raw or LOCAL_CLUSTER_QUEUE)


def target_secret_from_labels(labels: dict[str, str] | None) -> str:
    if not labels:
        return ""
    return str(labels.get(TARGET_KUBECONFIG_SECRET_LABEL) or "").strip()


def execution_name_from_labels(labels: dict[str, str] | None) -> str:
    if not labels:
        return ""
    return str(labels.get(EXECUTION_NAME_LABEL) or "").strip()


def submission_configmap_name_from_labels(labels: dict[str, str] | None) -> str:
    if not labels:
        return ""
    return str(labels.get(SUBMISSION_CONFIGMAP_LABEL) or "").strip()


def ensure_queue_registration(namespace: str, cluster_name: str) -> None:
    kubectl_cmd = require_any_command("oc", "kubectl")
    normalized_name = cluster_queue_name(cluster_name)
    admissioncheck_result = run_command(
        [
            kubectl_cmd,
            "get",
            "admissioncheck",
            REMOTE_CAPACITY_ADMISSION_CHECK,
            "-o",
            "json",
        ],
        capture_output=True,
        check=False,
    )
    if admissioncheck_result.returncode != 0:
        raise CommandError(
            "BenchFlow Kueue support is not installed in the management cluster; "
            "run bflow bootstrap in the management cluster first"
        )
    localqueue_result = run_command(
        [
            kubectl_cmd,
            "get",
            "localqueue",
            normalized_name,
            "-n",
            namespace,
            "-o",
            "json",
        ],
        capture_output=True,
        check=False,
    )
    if localqueue_result.returncode != 0:
        raise CommandError(
            f"no Kueue LocalQueue named {normalized_name!r} found in namespace {namespace}; "
            f"bootstrap the cluster registration for {cluster_name!r} first"
        )
    admissioncheck = json.loads(admissioncheck_result.stdout or "{}")
    active = False
    for condition in admissioncheck.get("status", {}).get("conditions", []) or []:
        if (
            str(condition.get("type") or "") == "Active"
            and str(condition.get("status") or "") == "True"
        ):
            active = True
            break
    if not active:
        raise CommandError(
            "BenchFlow remote-capacity AdmissionCheck exists but is not Active yet; "
            "wait for the controller deployment to become ready and retry"
        )


def _workload_name(prefix: str) -> str:
    base = sanitize_name(prefix, max_length=45)
    suffix = sanitize_name(str(int(time.time() * 1000)), max_length=12)
    return f"{base}-{suffix}"


def _workload_json(
    *,
    namespace: str,
    cluster_name: str,
    execution_prefix: str,
    execution_name: str,
    submission_configmap_name: str,
    requested_gpu_count: int,
    max_execution_seconds: int,
    execution_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    requests = (
        {REMOTE_GPU_RESOURCE: str(requested_gpu_count)} if requested_gpu_count else {}
    )
    name = _workload_name(f"{execution_prefix}-reservation")
    labels = {
        "app.kubernetes.io/name": "benchflow",
        "benchflow.io/managed-by": "benchflow",
        CLUSTER_QUEUE_LABEL: cluster_name,
        REQUESTED_GPUS_LABEL: str(requested_gpu_count),
        EXECUTION_NAME_LABEL: execution_name,
        SUBMISSION_CONFIGMAP_LABEL: submission_configmap_name,
    }
    for key, value in (execution_labels or {}).items():
        cleaned_key = str(key).strip()
        cleaned_value = str(value).strip()
        if cleaned_key and cleaned_value:
            labels[cleaned_key] = cleaned_value
    return {
        "apiVersion": KUEUE_API_VERSION,
        "kind": "Workload",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": labels,
        },
        "spec": {
            "active": True,
            "queueName": cluster_name,
            "maximumExecutionTimeSeconds": max_execution_seconds,
            "podSets": [
                {
                    "name": "main",
                    "count": 1,
                    "template": {
                        "spec": {
                            "restartPolicy": "Never",
                            "containers": [
                                {
                                    "name": "reservation",
                                    "image": "registry.k8s.io/pause:3.10",
                                    "resources": {"requests": requests}
                                    if requests
                                    else {},
                                }
                            ],
                        }
                    },
                }
            ],
        },
    }


def create_reservation_workload(
    *,
    namespace: str,
    cluster_name: str,
    execution_prefix: str,
    execution_name: str,
    submission_configmap_name: str,
    requested_gpu_count: int,
    execution_timeout: str,
    execution_labels: dict[str, str] | None = None,
) -> str:
    kubectl_cmd = require_any_command("oc", "kubectl")
    ensure_queue_registration(namespace, cluster_name)
    workload = _workload_json(
        namespace=namespace,
        cluster_name=cluster_name,
        execution_prefix=execution_prefix,
        execution_name=execution_name,
        submission_configmap_name=submission_configmap_name,
        requested_gpu_count=requested_gpu_count,
        max_execution_seconds=_duration_seconds(execution_timeout),
        execution_labels=execution_labels,
    )
    result = run_command(
        [kubectl_cmd, "create", "-n", namespace, "-f", "-", "-o", "json"],
        input_text=json.dumps(workload),
        capture_output=True,
    )
    payload = json.loads(result.stdout or "{}")
    name = str(payload.get("metadata", {}).get("name") or "")
    if not name:
        raise CommandError("reservation workload creation returned no name")
    detail(f"Created Kueue reservation workload {name} for cluster {cluster_name}")
    return name


def _workload_status(namespace: str, name: str) -> dict[str, Any] | None:
    kubectl_cmd = require_any_command("oc", "kubectl")
    result = run_command(
        [kubectl_cmd, "get", "workload", name, "-n", namespace, "-o", "json"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout or "{}")


def wait_for_reservation(
    *,
    namespace: str,
    workload_name: str,
    timeout_seconds: int = 86400,
) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        payload = _workload_status(namespace, workload_name)
        if payload is None:
            raise CommandError(f"reservation workload {workload_name} disappeared")
        status = payload.get("status", {}) or {}
        if status.get("admission"):
            success(f"Kueue admitted reservation workload {workload_name}")
            return
        for check_state in status.get("admissionChecks", []) or []:
            state = str(check_state.get("state") or "")
            if state == "Rejected":
                message = str(check_state.get("message") or "rejected")
                raise CommandError(
                    f"reservation workload {workload_name} was rejected: {message}"
                )
        time.sleep(3)
    raise CommandError(f"timed out waiting for reservation workload {workload_name}")


def delete_reservation_workload(namespace: str, workload_name: str) -> None:
    kubectl_cmd = require_any_command("oc", "kubectl")
    run_command(
        [
            kubectl_cmd,
            "delete",
            "workload",
            workload_name,
            "-n",
            namespace,
            "--ignore-not-found",
            "--wait=false",
        ],
        check=False,
    )


def submission_configmap_name(execution_name: str) -> str:
    return sanitize_name(f"{execution_name}-submission", max_length=63)


def create_submission_configmap(
    *,
    namespace: str,
    execution_name: str,
    manifest: dict[str, Any],
) -> str:
    configmap_name = submission_configmap_name(execution_name)
    payload = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": configmap_name,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/name": "benchflow",
                "benchflow.io/managed-by": "benchflow",
                EXECUTION_NAME_LABEL: execution_name,
                SUBMISSION_CONFIGMAP_LABEL: configmap_name,
            },
        },
        "data": {
            SUBMISSION_MANIFEST_KEY: json.dumps(
                manifest, separators=(",", ":"), sort_keys=True
            )
        },
    }
    create_manifest(
        json.dumps(payload, separators=(",", ":"), sort_keys=True), namespace
    )
    detail(f"Stored queued execution manifest in ConfigMap {configmap_name}")
    return configmap_name


def _submission_configmap_payload(namespace: str, name: str) -> dict[str, Any] | None:
    kubectl_cmd = require_any_command("oc", "kubectl")
    result = run_command(
        [kubectl_cmd, "get", "configmap", name, "-n", namespace, "-o", "json"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout or "{}")


def delete_submission_configmap(namespace: str, configmap_name: str) -> None:
    if not configmap_name:
        return
    kubectl_cmd = require_any_command("oc", "kubectl")
    run_command(
        [
            kubectl_cmd,
            "delete",
            "configmap",
            configmap_name,
            "-n",
            namespace,
            "--ignore-not-found",
            "--wait=false",
        ],
        check=False,
    )


def list_reservation_workloads(namespace: str) -> list[dict[str, Any]]:
    kubectl_cmd = require_any_command("oc", "kubectl")
    payload = run_json_command(
        [
            kubectl_cmd,
            "get",
            "workloads",
            "-n",
            namespace,
            "-l",
            "app.kubernetes.io/name=benchflow,benchflow.io/managed-by=benchflow",
            "-o",
            "json",
        ]
    )
    return list(payload.get("items", []) or [])


def reservation_workload_by_execution_name(
    namespace: str, execution_name: str
) -> dict[str, Any] | None:
    for workload in list_reservation_workloads(namespace):
        if _workload_execution_name(workload) == execution_name:
            return workload
    return None


def _workload_status_summary(workload: dict[str, Any]) -> tuple[str, bool, bool, str]:
    status = workload.get("status", {}) or {}
    if status.get("admission"):
        return ("Starting", False, False, "admitted; waiting for PipelineRun start")
    for check_state in status.get("admissionChecks", []) or []:
        state = str(check_state.get("state") or "")
        message = str(check_state.get("message") or "")
        if state == "Rejected":
            return ("Rejected", True, False, message)
        if state in {"Ready", "Retry", "Pending"}:
            return ("Queued", False, False, message)
    return ("Queued", False, False, "")


def summarize_reservation_workload(workload: dict[str, Any]) -> ExecutionSummary:
    metadata = workload.get("metadata", {}) or {}
    labels = metadata.get("labels", {}) or {}
    status, finished, succeeded, message = _workload_status_summary(workload)
    return ExecutionSummary(
        name=_workload_execution_name(workload) or str(metadata.get("name", "")),
        namespace=str(metadata.get("namespace", "")),
        experiment=str(labels.get("benchflow.io/experiment", "")),
        platform=str(labels.get("benchflow.io/platform", "")),
        mode=str(labels.get("benchflow.io/mode", "")),
        backend="tekton",
        status=status,
        finished=finished,
        succeeded=succeeded,
        start_time=str(metadata.get("creationTimestamp", "")),
        completion_time="",
        message=message,
    )


def discover_cluster_gpu_capacity(kubeconfig: str | Path | None = None) -> int:
    kubectl_cmd = require_any_command("oc", "kubectl")
    with use_kubeconfig(kubeconfig):
        payload = run_json_command([kubectl_cmd, "get", "nodes", "-o", "json"])
    total = 0
    for item in payload.get("items", []) or []:
        spec = item.get("spec", {}) or {}
        if spec.get("unschedulable"):
            continue
        ready = False
        for condition in item.get("status", {}).get("conditions", []) or []:
            if (
                condition.get("type") == "Ready"
                and str(condition.get("status") or "") == "True"
            ):
                ready = True
                break
        if not ready:
            continue
        allocatable = item.get("status", {}).get("allocatable", {}) or {}
        try:
            total += int(str(allocatable.get("nvidia.com/gpu") or "0"))
        except ValueError:
            continue
    return total


def discover_live_gpu_usage(kubeconfig: str | Path | None = None) -> int:
    kubectl_cmd = require_any_command("oc", "kubectl")
    with use_kubeconfig(kubeconfig):
        payload = run_json_command([kubectl_cmd, "get", "pods", "-A", "-o", "json"])
    total = 0
    for item in payload.get("items", []) or []:
        metadata = item.get("metadata", {}) or {}
        if metadata.get("deletionTimestamp"):
            continue
        phase = str(item.get("status", {}).get("phase") or "")
        if phase not in {"Pending", "Running"}:
            continue
        spec = item.get("spec", {}) or {}
        for container in spec.get("containers", []) or []:
            resources = container.get("resources", {}) or {}
            requests = resources.get("requests", {}) or {}
            limits = resources.get("limits", {}) or {}
            raw_value = requests.get("nvidia.com/gpu", limits.get("nvidia.com/gpu", 0))
            try:
                total += int(str(raw_value or "0"))
            except ValueError:
                continue
    return total


def ensure_cluster_queue_resources(
    *,
    namespace: str,
    cluster_name: str,
    gpu_capacity: int,
) -> None:
    kubectl_cmd = require_any_command("oc", "kubectl")
    queue_name = cluster_queue_name(cluster_name)
    documents = [
        {
            "apiVersion": KUEUE_API_VERSION,
            "kind": "ResourceFlavor",
            "metadata": {"name": DEFAULT_RESOURCE_FLAVOR},
        },
        {
            "apiVersion": KUEUE_API_VERSION,
            "kind": "ClusterQueue",
            "metadata": {
                "name": queue_name,
                "labels": {
                    "app.kubernetes.io/name": "benchflow",
                    CLUSTER_QUEUE_LABEL: queue_name,
                },
            },
            "spec": {
                "namespaceSelector": {},
                "queueingStrategy": "BestEffortFIFO",
                "resourceGroups": [
                    {
                        "coveredResources": [REMOTE_GPU_RESOURCE],
                        "flavors": [
                            {
                                "name": DEFAULT_RESOURCE_FLAVOR,
                                "resources": [
                                    {
                                        "name": REMOTE_GPU_RESOURCE,
                                        "nominalQuota": str(max(0, gpu_capacity)),
                                    }
                                ],
                            }
                        ],
                    }
                ],
                "admissionChecksStrategy": {
                    "admissionChecks": [
                        {
                            "name": REMOTE_CAPACITY_ADMISSION_CHECK,
                            "onFlavors": [DEFAULT_RESOURCE_FLAVOR],
                        }
                    ]
                },
            },
        },
        {
            "apiVersion": KUEUE_API_VERSION,
            "kind": "LocalQueue",
            "metadata": {
                "namespace": namespace,
                "name": queue_name,
                "labels": {
                    "app.kubernetes.io/name": "benchflow",
                    CLUSTER_QUEUE_LABEL: queue_name,
                },
            },
            "spec": {"clusterQueue": queue_name},
        },
    ]
    manifest = yaml.safe_dump_all(documents, sort_keys=False)
    run_command([kubectl_cmd, "apply", "-f", "-"], input_text=manifest)
    success(
        f"Registered Kueue LocalQueue and ClusterQueue for {queue_name} with {gpu_capacity} GPU(s)"
    )


def _patch_admission_check_active(name: str) -> None:
    kubectl_cmd = require_any_command("oc", "kubectl")
    payload = run_json_command(
        [kubectl_cmd, "get", "admissioncheck", name, "-o", "json"]
    )
    conditions = list(payload.get("status", {}).get("conditions", []) or [])
    for condition in conditions:
        if (
            str(condition.get("type") or "") == "Active"
            and str(condition.get("status") or "") == "True"
        ):
            return
    active = {
        "type": "Active",
        "status": "True",
        "reason": "ControllerReady",
        "message": "BenchFlow remote capacity controller is running",
        "lastTransitionTime": _now_rfc3339(),
        "observedGeneration": int(
            payload.get("metadata", {}).get("generation", 1) or 1
        ),
    }
    updated = [
        condition for condition in conditions if condition.get("type") != "Active"
    ]
    updated.append(active)
    run_command(
        [
            kubectl_cmd,
            "patch",
            "admissioncheck",
            name,
            "--subresource=status",
            "--type",
            "merge",
            "-p",
            json.dumps({"status": {"conditions": updated}}),
        ]
    )


def _workload_requests_gpus(workload: dict[str, Any]) -> int:
    labels = workload.get("metadata", {}).get("labels", {}) or {}
    return requested_gpus_from_labels(labels)


def _workload_execution_name(workload: dict[str, Any]) -> str:
    labels = workload.get("metadata", {}).get("labels", {}) or {}
    return execution_name_from_labels(labels)


def _workload_submission_configmap_name(workload: dict[str, Any]) -> str:
    labels = workload.get("metadata", {}).get("labels", {}) or {}
    return submission_configmap_name_from_labels(labels)


def _pipeline_run_payload(namespace: str, name: str) -> dict[str, Any] | None:
    kubectl_cmd = require_any_command("oc", "kubectl")
    result = run_command(
        [kubectl_cmd, "get", "pipelinerun", name, "-n", namespace, "-o", "json"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout or "{}")


def _pipeline_run_finished(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    status = payload.get("status", {}) or {}
    if status.get("completionTime"):
        return True
    for condition in status.get("conditions", []) or []:
        if condition.get("type") == "Succeeded" and str(
            condition.get("status") or ""
        ) in {"True", "False"}:
            reason = str(condition.get("reason") or "")
            if reason in {"Succeeded", "Failed", "Cancelled", "PipelineRunTimeout"}:
                return True
    return False


def _create_execution_from_workload(namespace: str, workload: dict[str, Any]) -> None:
    execution_name = _workload_execution_name(workload)
    if not execution_name:
        return
    if _pipeline_run_payload(namespace, execution_name) is not None:
        return
    configmap_name = _workload_submission_configmap_name(workload)
    if not configmap_name:
        raise CommandError(
            f"reservation workload {workload.get('metadata', {}).get('name', '')} "
            "is missing its submission ConfigMap label"
        )
    configmap = _submission_configmap_payload(namespace, configmap_name)
    if configmap is None:
        raise CommandError(
            f"submission ConfigMap {configmap_name} for execution {execution_name} was not found"
        )
    manifest_text = str(
        (configmap.get("data", {}) or {}).get(SUBMISSION_MANIFEST_KEY) or ""
    ).strip()
    if not manifest_text:
        raise CommandError(
            f"submission ConfigMap {configmap_name} is missing {SUBMISSION_MANIFEST_KEY}"
        )
    try:
        manifest = json.loads(manifest_text)
    except json.JSONDecodeError as exc:
        raise CommandError(
            f"submission ConfigMap {configmap_name} does not contain valid JSON"
        ) from exc
    metadata = manifest.setdefault("metadata", {})
    metadata["name"] = execution_name
    metadata["namespace"] = namespace
    metadata.pop("generateName", None)
    create_manifest(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True), namespace
    )
    detail(f"Created PipelineRun {execution_name} from reservation workload")
    delete_submission_configmap(namespace, configmap_name)


def _patch_workload_check(
    namespace: str,
    workload: dict[str, Any],
    *,
    state: str,
    message: str,
    requeue_after_seconds: int | None = None,
) -> None:
    kubectl_cmd = require_any_command("oc", "kubectl")
    status = workload.get("status", {}) or {}
    current_checks = list(status.get("admissionChecks", []) or [])
    existing = next(
        (
            entry
            for entry in current_checks
            if entry.get("name") == REMOTE_CAPACITY_ADMISSION_CHECK
        ),
        None,
    )
    desired = {
        "name": REMOTE_CAPACITY_ADMISSION_CHECK,
        "state": state,
        "lastTransitionTime": _now_rfc3339(),
        "message": message,
    }
    if state == "Retry" and requeue_after_seconds is not None:
        desired["requeueAfterSeconds"] = int(requeue_after_seconds)
    if existing:
        same_state = existing.get("state") == desired["state"]
        same_message = str(existing.get("message") or "") == desired["message"]
        same_retry = int(existing.get("requeueAfterSeconds") or 0) == int(
            desired.get("requeueAfterSeconds") or 0
        )
        if same_state and same_message and same_retry:
            return
    updated = [
        entry
        for entry in current_checks
        if entry.get("name") != REMOTE_CAPACITY_ADMISSION_CHECK
    ]
    updated.append(desired)
    run_command(
        [
            kubectl_cmd,
            "patch",
            "workload",
            str(workload.get("metadata", {}).get("name") or ""),
            "-n",
            namespace,
            "--subresource=status",
            "--type",
            "merge",
            "-p",
            json.dumps({"status": {"admissionChecks": updated}}),
        ]
    )


def _kubeconfig_path_for_secret(namespace: str, secret_name: str) -> Path:
    kubectl_cmd = require_any_command("oc", "kubectl")
    secret = run_json_command(
        [kubectl_cmd, "get", "secret", secret_name, "-n", namespace, "-o", "json"]
    )
    encoded = str(secret.get("data", {}).get("kubeconfig") or "")
    if not encoded:
        raise CommandError(f"kubeconfig secret {secret_name} is missing key kubeconfig")
    with tempfile.NamedTemporaryFile(
        prefix=f"benchflow-{secret_name}-",
        suffix=".kubeconfig",
        delete=False,
    ) as handle:
        handle.write(base64.b64decode(encoded))
        return Path(handle.name)


def run_remote_capacity_controller(
    *,
    namespace: str,
    poll_interval_seconds: int = 10,
) -> None:
    step(f"Starting BenchFlow remote capacity controller for namespace {namespace}")
    while True:
        try:
            _patch_admission_check_active(REMOTE_CAPACITY_ADMISSION_CHECK)
            for workload in list_reservation_workloads(namespace):
                metadata = workload.get("metadata", {}) or {}
                workload_name = str(metadata.get("name") or "")
                labels = metadata.get("labels", {}) or {}
                execution_name = _workload_execution_name(workload)
                if execution_name:
                    payload = _pipeline_run_payload(namespace, execution_name)
                    if _pipeline_run_finished(payload):
                        detail(
                            f"Releasing Kueue reservation {workload_name} for finished PipelineRun {execution_name}"
                        )
                        delete_submission_configmap(
                            namespace, _workload_submission_configmap_name(workload)
                        )
                        delete_reservation_workload(namespace, workload_name)
                        continue

                status = workload.get("status", {}) or {}
                if status.get("admission"):
                    if execution_name and payload is None:
                        try:
                            _create_execution_from_workload(namespace, workload)
                        except CommandError as exc:
                            warning(
                                "remote capacity controller could not create "
                                f"PipelineRun for reservation {workload_name}: {exc}"
                            )
                    continue

                cluster_name = queue_name_from_labels(labels)
                requested = _workload_requests_gpus(workload)
                kubeconfig_secret = target_secret_from_labels(labels) or (
                    "" if cluster_name == LOCAL_CLUSTER_QUEUE else cluster_name
                )
                kubeconfig_path: Path | None = None
                try:
                    if kubeconfig_secret:
                        kubeconfig_path = _kubeconfig_path_for_secret(
                            namespace, kubeconfig_secret
                        )
                    capacity = discover_cluster_gpu_capacity(kubeconfig_path)
                    usage = discover_live_gpu_usage(kubeconfig_path)
                except CommandError as exc:
                    _patch_workload_check(
                        namespace,
                        workload,
                        state="Retry",
                        message=str(exc),
                        requeue_after_seconds=30,
                    )
                    continue
                finally:
                    if kubeconfig_path is not None and kubeconfig_path.exists():
                        kubeconfig_path.unlink(missing_ok=True)

                available = max(0, capacity - usage)
                if available >= requested:
                    _patch_workload_check(
                        namespace,
                        workload,
                        state="Ready",
                        message=(
                            f"target cluster {cluster_name} has {available} GPU(s) available "
                            f"for a {requested}-GPU workload"
                        ),
                    )
                    continue
                _patch_workload_check(
                    namespace,
                    workload,
                    state="Retry",
                    message=(
                        f"target cluster {cluster_name} currently has {available}/{capacity} GPU(s) "
                        f"free; workload requires {requested}"
                    ),
                    requeue_after_seconds=30,
                )
        except CommandError as exc:
            warning(f"remote capacity controller reconciliation failed: {exc}")
        time.sleep(max(1, int(poll_interval_seconds)))
