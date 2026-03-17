from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

from ..cluster import CommandError, require_command, run_command, run_json_command
from ..contracts import ExecutionSummary, ResolvedRunPlan, ValidationError
from ..ui import detail, step, success, warning


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


def _get_workflow_template(namespace: str, name: str) -> dict[str, Any] | None:
    require_command("oc")
    result = run_command(
        ["oc", "get", "workflowtemplate", name, "-n", namespace, "-o", "json"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)


def _workflow_step_name(node: dict[str, Any]) -> str:
    for key in ("displayName", "templateName"):
        value = str(node.get(key, "") or "").strip()
        if value and value != "pipeline":
            return value
    name = str(node.get("name", "") or "").strip()
    if "." in name:
        return name.rsplit(".", 1)[-1]
    return name


def _workflow_pod_nodes(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    status = workflow.get("status", {}) or {}
    nodes = status.get("nodes", {}) or {}
    if not isinstance(nodes, dict):
        return []
    return [
        node
        for node in nodes.values()
        if isinstance(node, dict) and node.get("type") == "Pod"
    ]


def _list_workflow_pods(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = workflow.get("metadata", {}) or {}
    namespace = str(metadata.get("namespace", "") or "").strip()
    workflow_name = str(metadata.get("name", "") or "").strip()
    if not namespace or not workflow_name:
        return []

    payload = run_json_command(
        [
            "oc",
            "get",
            "pod",
            "-n",
            namespace,
            "-l",
            f"workflows.argoproj.io/workflow={workflow_name}",
            "-o",
            "json",
        ]
    )
    items = payload.get("items", [])
    return items if isinstance(items, list) else []


def _template_step_names(template: dict[str, Any]) -> list[str]:
    spec = template.get("spec", {}) or {}
    entrypoint_name = str(spec.get("entrypoint", "") or "").strip()
    templates = spec.get("templates", []) or []
    if not entrypoint_name or not isinstance(templates, list):
        return []

    entrypoint = next(
        (
            item
            for item in templates
            if isinstance(item, dict) and item.get("name") == entrypoint_name
        ),
        None,
    )
    if entrypoint is None:
        return []

    steps = entrypoint.get("steps", []) or []
    names: list[str] = []
    for group in steps:
        if not isinstance(group, list):
            continue
        for item in group:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "") or "").strip()
            if name:
                names.append(name)

    on_exit = str(spec.get("onExit", "") or "").strip()
    if on_exit:
        names.append(on_exit)
    return names


def _workflow_step_names(workflow: dict[str, Any]) -> list[str]:
    namespace = str(workflow.get("metadata", {}).get("namespace", "") or "").strip()
    step_names: list[str] = []

    workflow_template_ref = (workflow.get("spec", {}) or {}).get(
        "workflowTemplateRef", {}
    ) or {}
    workflow_template_name = str(workflow_template_ref.get("name", "") or "").strip()
    if namespace and workflow_template_name:
        workflow_template = _get_workflow_template(namespace, workflow_template_name)
        if workflow_template is not None:
            step_names.extend(_template_step_names(workflow_template))

    discovered = []
    for node in _workflow_pod_nodes(workflow):
        step_name = _workflow_step_name(node)
        if step_name and step_name not in discovered:
            discovered.append(step_name)

    for step_name in discovered:
        if step_name not in step_names:
            step_names.append(step_name)
    return step_names


def _step_pod_names(workflow: dict[str, Any], step_name: str) -> list[str]:
    pod_names: list[str] = []
    metadata = workflow.get("metadata", {}) or {}
    workflow_name = str(metadata.get("name", "") or "").strip()

    if workflow_name:
        prefix = f"{workflow_name}-{step_name}-"
        for pod in _list_workflow_pods(workflow):
            pod_name = str(
                (pod.get("metadata", {}) or {}).get("name", "") or ""
            ).strip()
            if pod_name.startswith(prefix) and pod_name not in pod_names:
                pod_names.append(pod_name)

    if pod_names:
        return pod_names

    for node in _workflow_pod_nodes(workflow):
        if _workflow_step_name(node) != step_name:
            continue
        pod_name = str(node.get("podName") or node.get("id") or "").strip()
        if pod_name and pod_name not in pod_names:
            pod_names.append(pod_name)
    return pod_names


def _stream_pod_logs(namespace: str, pod_name: str, *, all_containers: bool) -> None:
    command = [
        "oc",
        "logs",
        "-f",
        f"pod/{pod_name}",
        "-n",
        namespace,
        "--prefix=true",
        "--ignore-errors=true",
        "--pod-running-timeout=10m",
    ]
    if all_containers:
        command.append("--all-containers=true")
    else:
        command.extend(["-c", "main"])
    subprocess.run(command, check=False)


def _node_phase(node: dict[str, Any]) -> str:
    phase = str(node.get("phase", "") or "Pending").strip()
    return phase or "Pending"


def _aggregate_step_phase(phases: list[str]) -> str:
    if not phases:
        return "Pending"
    if any(phase in {"Failed", "Error"} for phase in phases):
        return next(phase for phase in phases if phase in {"Failed", "Error"})
    if any(phase == "Running" for phase in phases):
        return "Running"
    if any(phase == "Pending" for phase in phases):
        return "Pending"
    if all(phase in {"Skipped", "Omitted"} for phase in phases):
        return "Skipped"
    if all(phase == "Succeeded" for phase in phases):
        return "Succeeded"
    return phases[-1]


def _workflow_step_statuses(workflow: dict[str, Any]) -> list[tuple[str, str]]:
    step_names = _workflow_step_names(workflow)
    step_phases: dict[str, list[str]] = {step_name: [] for step_name in step_names}

    for node in _workflow_pod_nodes(workflow):
        step_name = _workflow_step_name(node)
        if not step_name:
            continue
        step_phases.setdefault(step_name, []).append(_node_phase(node))

    return [
        (step_name, _aggregate_step_phase(step_phases.get(step_name, [])))
        for step_name in step_names
    ]


def _emit_step_phase(step_name: str, phase: str) -> None:
    message = f"{step_name}: {phase}"
    if phase == "Succeeded":
        success(message)
        return
    if phase in {"Failed", "Error"}:
        warning(message)
        return
    detail(message)


class ArgoOrchestrator:
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
        last_state: tuple[str, bool, bool, str] | None = None
        last_step_statuses: dict[str, str] = {}
        step(f"Watching Workflow {name} in namespace {namespace}")
        while True:
            payload = _get_workflow(namespace, name)
            if payload is None:
                raise CommandError(
                    f"Workflow {name} in namespace {namespace} was not found"
                )
            state = _workflow_state(payload)
            step_statuses = dict(_workflow_step_statuses(payload))

            if not last_step_statuses and step_statuses:
                detail("Selectable steps: " + ", ".join(step_statuses))

            if state != last_state:
                status, finished, succeeded, message = state
                if succeeded:
                    success(f"{name}: {status}")
                elif finished:
                    warning(f"{name}: {status}")
                else:
                    detail(f"{name}: {status}")
                if message:
                    detail(message)
                last_state = state

            for step_name, phase in step_statuses.items():
                if last_step_statuses.get(step_name) == phase:
                    continue
                _emit_step_phase(step_name, phase)
                last_step_statuses[step_name] = phase

            if state[1]:
                return state[2]
            subprocess.run(["sleep", str(poll_interval)], check=False)

    def list_steps(self, namespace: str, name: str) -> list[str]:
        workflow = _get_workflow(namespace, name)
        if workflow is None:
            raise CommandError(
                f"Workflow {name} in namespace {namespace} was not found"
            )
        return _workflow_step_names(workflow)

    def logs(
        self,
        namespace: str,
        name: str,
        *,
        step_name: str | None = None,
        all_logs: bool = False,
        all_containers: bool = False,
    ) -> None:
        workflow = _get_workflow(namespace, name)
        if workflow is None:
            raise CommandError(
                f"Workflow {name} in namespace {namespace} was not found"
            )

        if all_logs:
            if shutil.which("argo") is not None:
                step(f"Following Workflow {name} logs in namespace {namespace}")
                subprocess.run(
                    ["argo", "logs", "-f", "-n", namespace, name], check=False
                )
                return

            step(
                f"Following Workflow {name} logs in namespace {namespace} with oc logs"
            )
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
            return

        if not step_name:
            raise CommandError("step_name is required when all_logs is false")

        available_steps = _workflow_step_names(workflow)
        if step_name not in available_steps:
            choices = ", ".join(available_steps) if available_steps else "none"
            raise CommandError(
                f"unknown step {step_name!r} for workflow {name}; available steps: {choices}"
            )

        pod_names = _step_pod_names(workflow, step_name)
        if not pod_names:
            raise CommandError(
                f"workflow step {step_name!r} exists but has no pod logs yet"
            )

        for pod_name in pod_names:
            step(
                f"Following {step_name} logs from pod {pod_name} in namespace {namespace}"
            )
            _stream_pod_logs(
                namespace,
                pod_name,
                all_containers=all_containers,
            )
