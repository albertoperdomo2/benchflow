from __future__ import annotations

import json
import shutil
import subprocess
import time
from typing import Any

from ..cluster import CommandError, require_command, run_command, run_json_command
from ..models import ResolvedRunPlan, ValidationError
from ..ui import detail, step, success, warning
from .base import ExecutionSummary


def _common_labels(plan: ResolvedRunPlan, *, backend: str) -> dict[str, str]:
    return {
        "app.kubernetes.io/name": "benchflow",
        "benchflow.io/experiment": plan.metadata.name,
        "benchflow.io/platform": plan.deployment.platform,
        "benchflow.io/mode": plan.deployment.mode,
        "benchflow.io/execution-backend": backend,
    }


def render_workflow(
    plan: ResolvedRunPlan,
    *,
    workflow_name: str,
    setup_mode: str,
    teardown: bool,
) -> dict[str, Any]:
    run_plan_json = json.dumps(plan.to_dict(), separators=(",", ":"), sort_keys=True)
    return {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "Workflow",
        "metadata": {
            "generateName": f"{plan.metadata.name}-",
            "labels": _common_labels(plan, backend="argo"),
        },
        "spec": {
            "workflowTemplateRef": {"name": workflow_name},
            "serviceAccountName": plan.service_account,
            "ttlStrategy": {
                "secondsAfterCompletion": plan.ttl_seconds_after_finished,
            },
            "arguments": {
                "parameters": [
                    {"name": "RUN_PLAN", "value": run_plan_json},
                    {
                        "name": "BENCHFLOW_IMAGE",
                        "value": "ghcr.io/albertoperdomo2/benchflow/benchflow:latest",
                    },
                    {
                        "name": "MODELS_STORAGE_PVC",
                        "value": plan.deployment.model_storage.pvc_name,
                    },
                    {"name": "SETUP_MODE", "value": setup_mode},
                    {"name": "TEARDOWN", "value": str(teardown).lower()},
                ]
            },
            "volumes": [
                {
                    "name": "results",
                    "persistentVolumeClaim": {"claimName": "benchmark-results"},
                },
                {
                    "name": "models-storage",
                    "persistentVolumeClaim": {
                        "claimName": plan.deployment.model_storage.pvc_name
                    },
                },
                {"name": "source", "emptyDir": {}},
                {"name": "workflow-state", "emptyDir": {}},
            ],
        },
    }


def render_matrix_workflow(
    plans: list[ResolvedRunPlan],
    *,
    workflow_name: str,
    child_workflow_name: str,
) -> dict[str, Any]:
    if not plans:
        raise ValidationError("matrix submission requires at least one RunPlan")

    namespaces = {plan.deployment.namespace for plan in plans}
    service_accounts = {plan.service_account for plan in plans}
    ttl_values = {plan.ttl_seconds_after_finished for plan in plans}
    backends = {plan.execution.backend for plan in plans}
    if len(namespaces) != 1:
        raise ValidationError(
            "matrix submission requires all runs to target one namespace"
        )
    if len(service_accounts) != 1:
        raise ValidationError(
            "matrix submission requires all runs to use the same service account"
        )
    if len(ttl_values) != 1:
        raise ValidationError("matrix submission requires a consistent TTL")
    if backends != {"argo"}:
        raise ValidationError(
            "Argo matrix submission requires all child RunPlans to use the argo backend"
        )

    first_plan = plans[0]
    run_plans_json = json.dumps(
        [plan.to_dict() for plan in plans], separators=(",", ":"), sort_keys=True
    )

    return {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "Workflow",
        "metadata": {
            "generateName": f"{first_plan.metadata.name}-matrix-",
            "labels": {
                "app.kubernetes.io/name": "benchflow",
                "benchflow.io/experiment": first_plan.metadata.name,
                "benchflow.io/platform": "matrix",
                "benchflow.io/mode": "matrix",
                "benchflow.io/execution-backend": "argo",
            },
        },
        "spec": {
            "workflowTemplateRef": {"name": workflow_name},
            "serviceAccountName": next(iter(service_accounts)),
            "ttlStrategy": {
                "secondsAfterCompletion": next(iter(ttl_values)),
            },
            "arguments": {
                "parameters": [
                    {"name": "RUN_PLANS", "value": run_plans_json},
                    {
                        "name": "BENCHFLOW_IMAGE",
                        "value": "ghcr.io/albertoperdomo2/benchflow/benchflow:latest",
                    },
                    {"name": "CHILD_WORKFLOW_NAME", "value": child_workflow_name},
                ]
            },
        },
    }


def _get_workflow(namespace: str, name: str) -> dict[str, Any] | None:
    require_command("oc")
    result = run_command(
        ["oc", "get", "workflow", name, "-n", namespace, "-o", "json"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)


def _workflow_state(workflow: dict[str, Any]) -> tuple[str, bool, bool, str]:
    status = workflow.get("status", {}) or {}
    phase = str(status.get("phase", "") or "Pending")
    message = str(status.get("message", "") or "")
    if phase == "Succeeded":
        return ("Succeeded", True, True, message)
    if phase in {"Failed", "Error"}:
        return (phase, True, False, message)
    return (phase or "Running", False, False, message)


def _list_workflows(
    namespace: str, *, label_selector: str = ""
) -> list[dict[str, Any]]:
    require_command("oc")
    argv = ["oc", "get", "workflow", "-n", namespace]
    if label_selector:
        argv.extend(["-l", label_selector])
    argv.extend(["-o", "json"])
    payload = run_json_command(argv)
    items = payload.get("items", [])
    return items if isinstance(items, list) else []


def _workflow_log_stream_limit(namespace: str, name: str) -> int:
    require_command("oc")
    selector = f"workflows.argoproj.io/workflow={name}"
    payload = run_json_command(
        ["oc", "get", "pod", "-n", namespace, "-l", selector, "-o", "json"]
    )
    items = payload.get("items", [])
    if not isinstance(items, list) or not items:
        return 20

    stream_count = 0
    for item in items:
        spec = item.get("spec", {}) or {}
        containers = spec.get("containers", []) or []
        init_containers = spec.get("initContainers", []) or []
        stream_count += len(containers) + len(init_containers)
    return max(20, stream_count)


class ArgoBackend:
    name = "argo"

    def render_run(
        self,
        plan: ResolvedRunPlan,
        *,
        execution_name: str,
        setup_mode: str,
        teardown: bool,
    ) -> dict[str, Any]:
        return render_workflow(
            plan,
            workflow_name=execution_name,
            setup_mode=setup_mode,
            teardown=teardown,
        )

    def render_matrix(
        self,
        plans: list[ResolvedRunPlan],
        *,
        execution_name: str,
        child_execution_name: str,
    ) -> dict[str, Any]:
        return render_matrix_workflow(
            plans,
            workflow_name=execution_name,
            child_workflow_name=child_execution_name,
        )

    def get(self, namespace: str, name: str) -> dict[str, Any] | None:
        return _get_workflow(namespace, name)

    def list(self, namespace: str, *, label_selector: str = "") -> list[dict[str, Any]]:
        return _list_workflows(namespace, label_selector=label_selector)

    def summarize(self, resource: dict[str, Any]) -> ExecutionSummary:
        metadata = resource.get("metadata", {})
        labels = metadata.get("labels", {}) or {}
        status, finished, succeeded, message = _workflow_state(resource)
        status_payload = resource.get("status", {}) or {}
        return ExecutionSummary(
            name=str(metadata.get("name", "")),
            namespace=str(metadata.get("namespace", "")),
            experiment=str(labels.get("benchflow.io/experiment", "")),
            platform=str(labels.get("benchflow.io/platform", "")),
            mode=str(labels.get("benchflow.io/mode", "")),
            backend="argo",
            status=status,
            finished=finished,
            succeeded=succeeded,
            start_time=str(
                status_payload.get("startedAt") or metadata.get("creationTimestamp", "")
            ),
            completion_time=str(status_payload.get("finishedAt", "")),
            message=message,
        )

    def cancel(self, namespace: str, name: str) -> None:
        require_command("oc")
        run_command(
            [
                "oc",
                "patch",
                "workflow",
                name,
                "-n",
                namespace,
                "--type",
                "merge",
                "-p",
                '{"spec":{"shutdown":"Terminate"}}',
            ]
        )

    def follow(self, namespace: str, name: str, *, poll_interval: int = 5) -> bool:
        require_command("oc")

        if shutil.which("argo") is not None:
            step(f"Following Workflow {name} in namespace {namespace}")
            subprocess.run(
                ["argo", "logs", "-f", "-n", namespace, name],
                check=False,
            )
            payload = _get_workflow(namespace, name)
            if payload is None:
                raise CommandError(
                    f"Workflow {name} in namespace {namespace} was not found"
                )
            state, finished, succeeded, message = _workflow_state(payload)
            if succeeded:
                success(f"{name}: {state}")
            else:
                warning(f"{name}: {state}")
            if message:
                detail(message)
            return finished and succeeded

        step(f"Following Workflow {name} logs in namespace {namespace} with oc logs")
        max_log_requests = _workflow_log_stream_limit(namespace, name)
        subprocess.run(
            [
                "oc",
                "logs",
                "-f",
                "-n",
                namespace,
                "-l",
                f"workflows.argoproj.io/workflow={name}",
                "--all-containers=true",
                "--prefix=true",
                "--ignore-errors=true",
                f"--max-log-requests={max_log_requests}",
                "--pod-running-timeout=10m",
            ],
            check=False,
        )

        last_state: tuple[str, bool, bool, str] | None = None
        step(f"Watching Workflow {name} in namespace {namespace}")
        while True:
            payload = _get_workflow(namespace, name)
            if payload is None:
                raise CommandError(
                    f"Workflow {name} in namespace {namespace} was not found"
                )
            state = _workflow_state(payload)
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
