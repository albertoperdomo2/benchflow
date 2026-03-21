from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
import time
from typing import Any

from ..cluster import CommandError, require_command, run_command, run_json_command
from ..contracts import ExecutionSummary, ResolvedRunPlan, ValidationError
from ..kueue import execution_labels_for_matrix, execution_labels_for_plan
from ..ui import detail, step, success, warning

_SPINNER_FRAMES = ("◐", "◓", "◑", "◒")
_ANSI_RESET = "\033[0m"
_ANSI_DIM = "\033[2m"
_ANSI_GREEN = "\033[32m"
_ANSI_RED = "\033[31m"
_ANSI_CYAN = "\033[36m"
_TARGET_KUBECONFIG_PATH = "/workspace/target-kubeconfig/kubeconfig"
_DURATION_PART_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)(?P<unit>ms|h|m|s)")


def _common_labels(plan: ResolvedRunPlan, *, backend: str) -> dict[str, str]:
    return {
        "app.kubernetes.io/name": "benchflow",
        "benchflow.io/experiment": plan.metadata.name,
        "benchflow.io/platform": plan.deployment.platform,
        "benchflow.io/mode": plan.deployment.mode,
        "benchflow.io/execution-backend": backend,
    }


def _serialized_run_plan(plan: ResolvedRunPlan) -> str:
    payload = plan.to_dict()
    target_cluster = dict(payload.get("target_cluster") or {})
    if plan.target_cluster.kubeconfig_secret:
        target_cluster["kubeconfig"] = _TARGET_KUBECONFIG_PATH
    payload["target_cluster"] = target_cluster
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _pod_host_aliases(host_aliases: dict[str, str]) -> list[dict[str, Any]]:
    by_ip: dict[str, list[str]] = {}
    for hostname, ip_address in sorted(host_aliases.items()):
        cleaned_hostname = str(hostname).strip()
        cleaned_ip = str(ip_address).strip()
        if not cleaned_hostname or not cleaned_ip:
            continue
        by_ip.setdefault(cleaned_ip, []).append(cleaned_hostname)
    return [
        {"ip": ip_address, "hostnames": hostnames}
        for ip_address, hostnames in by_ip.items()
    ]


def _parse_duration_seconds(value: str) -> int:
    cleaned = str(value).strip().lower()
    if not cleaned:
        raise ValidationError("execution timeout must not be empty")
    position = 0
    total_seconds = 0.0
    for match in _DURATION_PART_RE.finditer(cleaned):
        if match.start() != position:
            raise ValidationError(f"invalid duration: {value!r}")
        magnitude = float(match.group("value"))
        unit = match.group("unit")
        if unit == "h":
            total_seconds += magnitude * 3600.0
        elif unit == "m":
            total_seconds += magnitude * 60.0
        elif unit == "s":
            total_seconds += magnitude
        elif unit == "ms":
            total_seconds += magnitude / 1000.0
        position = match.end()
    if position != len(cleaned):
        raise ValidationError(f"invalid duration: {value!r}")
    return max(1, math.ceil(total_seconds))


def _format_duration(seconds: int) -> str:
    remaining = max(1, int(seconds))
    hours, remaining = divmod(remaining, 3600)
    minutes, seconds = divmod(remaining, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds or not parts:
        parts.append(f"{seconds}s")
    return "".join(parts)


def _matrix_timeout(plans: list[ResolvedRunPlan]) -> str:
    total_seconds = sum(
        _parse_duration_seconds(plan.execution.timeout) for plan in plans
    )
    # Leave room for the supervisor task plus any hoisted setup/teardown work.
    total_seconds += 3600
    return _format_duration(total_seconds)


def render_pipelinerun(
    plan: ResolvedRunPlan,
    *,
    pipeline_name: str,
    setup_mode: str,
    teardown: bool,
    skip_kueue_reservation: bool = False,
    benchflow_image: str | None = None,
) -> dict[str, Any]:
    if plan.target_cluster.kubeconfig and not plan.target_cluster.kubeconfig_secret:
        raise ValidationError(
            "Tekton execution requires target_cluster.kubeconfig_secret for remote target clusters; "
            "--target-kubeconfig only works with direct local BenchFlow commands"
        )
    run_plan_json = _serialized_run_plan(plan)
    workspaces: list[dict[str, Any]] = [
        {
            "name": "results",
            "persistentVolumeClaim": {"claimName": "benchmark-results"},
        },
        {
            "name": "models-storage",
            (
                "emptyDir" if plan.target_cluster.enabled() else "persistentVolumeClaim"
            ): (
                {}
                if plan.target_cluster.enabled()
                else {"claimName": plan.deployment.model_storage.pvc_name}
            ),
        },
        {"name": "source", "emptyDir": {}},
        {"name": "target-kubeconfig", "emptyDir": {}},
    ]
    if plan.target_cluster.kubeconfig_secret:
        workspaces[-1] = {
            "name": "target-kubeconfig",
            "secret": {"secretName": plan.target_cluster.kubeconfig_secret},
        }
    task_run_template: dict[str, Any] = {"serviceAccountName": plan.service_account}
    host_aliases = _pod_host_aliases(plan.target_cluster.host_aliases)
    if host_aliases:
        task_run_template["podTemplate"] = {"hostAliases": host_aliases}
    return {
        "apiVersion": "tekton.dev/v1",
        "kind": "PipelineRun",
        "metadata": {
            "generateName": f"{plan.metadata.name}-",
            "labels": {
                **_common_labels(plan, backend="tekton"),
                **execution_labels_for_plan(
                    plan, skip_reservation=skip_kueue_reservation
                ),
            },
        },
        "spec": {
            "pipelineRef": {"name": pipeline_name},
            "taskRunTemplate": task_run_template,
            "ttlSecondsAfterFinished": plan.ttl_seconds_after_finished,
            "timeouts": {"pipeline": plan.execution.timeout},
            "params": [
                {"name": "RUN_PLAN", "value": run_plan_json},
                *(
                    [{"name": "BENCHFLOW_IMAGE", "value": benchflow_image}]
                    if benchflow_image
                    else []
                ),
                {"name": "SETUP_MODE", "value": setup_mode},
                {"name": "TEARDOWN", "value": str(teardown).lower()},
            ],
            "workspaces": workspaces,
        },
    }


def render_matrix_pipelinerun(
    plans: list[ResolvedRunPlan],
    *,
    pipeline_name: str,
    child_pipeline_name: str,
    benchflow_image: str | None = None,
) -> dict[str, Any]:
    if not plans:
        raise ValidationError("matrix submission requires at least one RunPlan")

    namespaces = {plan.deployment.namespace for plan in plans}
    service_accounts = {plan.service_account for plan in plans}
    ttl_values = {plan.ttl_seconds_after_finished for plan in plans}
    kubeconfig_secrets = {plan.target_cluster.kubeconfig_secret for plan in plans}
    inline_kubeconfigs = {
        str(plan.target_cluster.kubeconfig or "").strip() for plan in plans
    }
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
    if len(kubeconfig_secrets) != 1:
        raise ValidationError(
            "matrix submission requires all runs to use the same target kubeconfig secret"
        )
    if inline_kubeconfigs - {""}:
        raise ValidationError(
            "Tekton matrix submission requires target_cluster.kubeconfig_secret for remote target clusters; "
            "--target-kubeconfig only works with direct local BenchFlow commands"
        )

    first_plan = plans[0]
    run_plans_json = json.dumps(
        [json.loads(_serialized_run_plan(plan)) for plan in plans],
        separators=(",", ":"),
        sort_keys=True,
    )
    workspaces: list[dict[str, Any]] = [{"name": "target-kubeconfig", "emptyDir": {}}]
    kubeconfig_secret = next(iter(kubeconfig_secrets))
    if kubeconfig_secret:
        workspaces[0] = {
            "name": "target-kubeconfig",
            "secret": {"secretName": kubeconfig_secret},
        }
    task_run_template: dict[str, Any] = {
        "serviceAccountName": next(iter(service_accounts))
    }
    host_aliases = _pod_host_aliases(first_plan.target_cluster.host_aliases)
    if host_aliases:
        task_run_template["podTemplate"] = {"hostAliases": host_aliases}

    return {
        "apiVersion": "tekton.dev/v1",
        "kind": "PipelineRun",
        "metadata": {
            "generateName": f"{first_plan.metadata.name}-matrix-",
            "labels": {
                "app.kubernetes.io/name": "benchflow",
                "benchflow.io/experiment": first_plan.metadata.name,
                "benchflow.io/platform": "matrix",
                "benchflow.io/mode": "matrix",
                "benchflow.io/execution-backend": "tekton",
                **execution_labels_for_matrix(plans, skip_reservation=True),
            },
        },
        "spec": {
            "pipelineRef": {"name": pipeline_name},
            "taskRunTemplate": task_run_template,
            "ttlSecondsAfterFinished": next(iter(ttl_values)),
            "timeouts": {"pipeline": _matrix_timeout(plans)},
            "params": [
                {"name": "RUN_PLANS", "value": run_plans_json},
                *(
                    [{"name": "BENCHFLOW_IMAGE", "value": benchflow_image}]
                    if benchflow_image
                    else []
                ),
                {"name": "CHILD_PIPELINE_NAME", "value": child_pipeline_name},
            ],
            "workspaces": workspaces,
        },
    }


def _get_pipelinerun(namespace: str, name: str) -> dict[str, Any] | None:
    require_command("oc")
    result = run_command(
        ["oc", "get", "pipelinerun", name, "-n", namespace, "-o", "json"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)


def _list_pipelineruns(
    namespace: str, *, label_selector: str = ""
) -> list[dict[str, Any]]:
    require_command("oc")
    argv = ["oc", "get", "pipelinerun", "-n", namespace]
    if label_selector:
        argv.extend(["-l", label_selector])
    argv.extend(["-o", "json"])
    payload = run_json_command(argv)
    items = payload.get("items", [])
    return items if isinstance(items, list) else []


def _get_pipeline(namespace: str, name: str) -> dict[str, Any] | None:
    require_command("oc")
    result = run_command(
        ["oc", "get", "pipeline", name, "-n", namespace, "-o", "json"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)


def _list_taskruns(namespace: str, pipeline_run_name: str) -> list[dict[str, Any]]:
    require_command("oc")
    payload = run_json_command(
        [
            "oc",
            "get",
            "taskrun",
            "-n",
            namespace,
            "-l",
            f"tekton.dev/pipelineRun={pipeline_run_name}",
            "-o",
            "json",
        ]
    )
    items = payload.get("items", [])
    return items if isinstance(items, list) else []


def _taskrun_state(taskrun: dict[str, Any]) -> str:
    conditions = (taskrun.get("status", {}) or {}).get("conditions", []) or []
    if not conditions:
        return "Pending"
    condition = conditions[0]
    status = str(condition.get("status", "") or "")
    if status == "True":
        return "Succeeded"
    if status == "False":
        return "Failed"
    if (taskrun.get("status", {}) or {}).get("startTime"):
        return "Running"
    return "Pending"


def _pipelinerun_state(pipelinerun: dict[str, Any]) -> tuple[str, bool, bool, str]:
    conditions = (pipelinerun.get("status", {}) or {}).get("conditions", []) or []
    if not conditions:
        return ("Pending", False, False, "")
    condition = conditions[0]
    status = str(condition.get("status", "") or "")
    reason = str(condition.get("reason", "") or "")
    message = str(condition.get("message", "") or "")
    if status == "True":
        return (reason or "Succeeded", True, True, message)
    if status == "False":
        return (reason or "Failed", True, False, message)
    if (pipelinerun.get("status", {}) or {}).get("startTime"):
        return ("Running", False, False, message)
    return ("Pending", False, False, message)


def _pipeline_task_names(pipelinerun: dict[str, Any]) -> list[str]:
    metadata = pipelinerun.get("metadata", {}) or {}
    namespace = str(metadata.get("namespace", "") or "").strip()
    spec = pipelinerun.get("spec", {}) or {}
    pipeline_ref = spec.get("pipelineRef", {}) or {}
    pipeline_name = str(pipeline_ref.get("name", "") or "").strip()
    names: list[str] = []

    if namespace and pipeline_name:
        pipeline = _get_pipeline(namespace, pipeline_name)
        if pipeline is not None:
            pipeline_spec = pipeline.get("spec", {}) or {}
            names.extend(
                str(task.get("name", "") or "").strip()
                for task in (pipeline_spec.get("tasks", []) or [])
                if str(task.get("name", "") or "").strip()
            )
            names.extend(
                str(task.get("name", "") or "").strip()
                for task in (pipeline_spec.get("finally", []) or [])
                if str(task.get("name", "") or "").strip()
            )
            if names:
                return names

    seen: list[str] = []
    child_refs = (pipelinerun.get("status", {}) or {}).get("childReferences", []) or []
    for child in child_refs:
        task_name = str(child.get("pipelineTaskName", "") or "").strip()
        if task_name and task_name not in seen:
            seen.append(task_name)
    return seen


def _task_status_pairs(pipelinerun: dict[str, Any]) -> list[tuple[str, str]]:
    metadata = pipelinerun.get("metadata", {}) or {}
    namespace = str(metadata.get("namespace", "") or "").strip()
    pipeline_run_name = str(metadata.get("name", "") or "").strip()
    step_names = _pipeline_task_names(pipelinerun)
    status_by_step = {name: "Pending" for name in step_names}

    skipped = (pipelinerun.get("status", {}) or {}).get("skippedTasks", []) or []
    for item in skipped:
        name = str(item.get("name", "") or "").strip()
        if name:
            status_by_step[name] = "Skipped"

    if namespace and pipeline_run_name:
        for taskrun in _list_taskruns(namespace, pipeline_run_name):
            labels = (taskrun.get("metadata", {}) or {}).get("labels", {}) or {}
            step_name = str(labels.get("tekton.dev/pipelineTask", "") or "").strip()
            if not step_name:
                continue
            status_by_step[step_name] = _taskrun_state(taskrun)
            if step_name not in step_names:
                step_names.append(step_name)

    return [(name, status_by_step.get(name, "Pending")) for name in step_names]


def _taskruns_for_step(
    namespace: str, pipeline_run_name: str, step_name: str
) -> list[dict[str, Any]]:
    taskruns = _list_taskruns(namespace, pipeline_run_name)
    matches = []
    for taskrun in taskruns:
        labels = (taskrun.get("metadata", {}) or {}).get("labels", {}) or {}
        if str(labels.get("tekton.dev/pipelineTask", "") or "").strip() == step_name:
            matches.append(taskrun)
    return matches


def _stream_taskrun_logs(
    namespace: str, taskrun: dict[str, Any], *, all_containers: bool
) -> None:
    taskrun_name = str(
        (taskrun.get("metadata", {}) or {}).get("name", "") or ""
    ).strip()
    if not taskrun_name:
        return
    if shutil.which("tkn") is not None:
        step(f"Following logs for TaskRun {taskrun_name} in namespace {namespace}")
        command = ["tkn", "taskrun", "logs", "-f", "-n", namespace, taskrun_name]
        if all_containers:
            command.append("--all")
        subprocess.run(command, check=False)
        return

    payload = run_json_command(
        [
            "oc",
            "get",
            "pod",
            "-n",
            namespace,
            "-l",
            f"tekton.dev/taskRun={taskrun_name}",
            "-o",
            "json",
        ]
    )
    items = payload.get("items", [])
    if not isinstance(items, list) or not items:
        raise CommandError(
            f"TaskRun {taskrun_name} exists but has no pod logs yet in namespace {namespace}"
        )
    for item in items:
        pod_name = str((item.get("metadata", {}) or {}).get("name", "") or "").strip()
        if not pod_name:
            continue
        step(f"Following logs from pod {pod_name} in namespace {namespace}")
        command = ["oc", "logs", "-f", "pod/" + pod_name, "-n", namespace, "--prefix"]
        if all_containers:
            command.append("--all-containers=true")
        subprocess.run(command, check=False)


def _pipelinerun_log_stream_limit(namespace: str, name: str) -> int:
    payload = run_json_command(
        [
            "oc",
            "get",
            "pod",
            "-n",
            namespace,
            "-l",
            f"tekton.dev/pipelineRun={name}",
            "-o",
            "json",
        ]
    )
    items = payload.get("items", [])
    if not isinstance(items, list) or not items:
        return 20
    stream_count = 0
    for item in items:
        spec = item.get("spec", {}) or {}
        stream_count += len(spec.get("containers", []) or [])
        stream_count += len(spec.get("initContainers", []) or [])
    return max(20, stream_count)


def _styled_token(value: str, ansi: str, *, interactive: bool) -> str:
    if not interactive:
        return value
    return f"{ansi}{value}{_ANSI_RESET}"


def _render_step_segment(
    step_name: str, phase: str, *, interactive: bool, spinner_frame: str
) -> str:
    if phase == "Succeeded":
        return _styled_token(f"✓ {step_name}", _ANSI_GREEN, interactive=interactive)
    if phase in {"Failed", "Error"}:
        return _styled_token(f"✗ {step_name}", _ANSI_RED, interactive=interactive)
    if phase == "Running":
        return _styled_token(
            f"{spinner_frame} {step_name}", _ANSI_CYAN, interactive=interactive
        )
    if phase in {"Skipped", "Omitted"}:
        return _styled_token(f"⊘ {step_name}", _ANSI_DIM, interactive=interactive)
    return _styled_token(step_name, _ANSI_DIM, interactive=interactive)


def _render_status_line(
    step_statuses: list[tuple[str, str]], *, interactive: bool, spinner_frame: str
) -> str:
    return " › ".join(
        _render_step_segment(
            step_name,
            phase,
            interactive=interactive,
            spinner_frame=spinner_frame,
        )
        for step_name, phase in step_statuses
    )


def _truncate_live_line(line: str) -> str:
    if "\033[" in line:
        return line
    columns = shutil.get_terminal_size((120, 20)).columns
    if len(line) <= columns:
        return line
    if columns <= 1:
        return line
    return line[: max(columns - 1, 0)]


class _TerminalWatchUI:
    def __init__(self) -> None:
        self._initialized = False

    def update(self, line: str) -> None:
        if self._initialized:
            sys.stdout.write("\033[1F")
        sys.stdout.write("\033[2K")
        sys.stdout.write(f"  {_truncate_live_line(line)}\n")
        sys.stdout.flush()
        self._initialized = True


class TektonOrchestrator:
    name = "tekton"

    def render_run(
        self,
        plan: ResolvedRunPlan,
        *,
        execution_name: str,
        setup_mode: str,
        teardown: bool,
        skip_kueue_reservation: bool = False,
        benchflow_image: str | None = None,
    ) -> dict[str, Any]:
        return render_pipelinerun(
            plan,
            pipeline_name=execution_name,
            setup_mode=setup_mode,
            teardown=teardown,
            skip_kueue_reservation=skip_kueue_reservation,
            benchflow_image=benchflow_image,
        )

    def render_matrix(
        self,
        plans: list[ResolvedRunPlan],
        *,
        execution_name: str,
        child_execution_name: str,
        benchflow_image: str | None = None,
    ) -> dict[str, Any]:
        return render_matrix_pipelinerun(
            plans,
            pipeline_name=execution_name,
            child_pipeline_name=child_execution_name,
            benchflow_image=benchflow_image,
        )

    def get(self, namespace: str, name: str) -> dict[str, Any] | None:
        return _get_pipelinerun(namespace, name)

    def list(self, namespace: str, *, label_selector: str = "") -> list[dict[str, Any]]:
        return _list_pipelineruns(namespace, label_selector=label_selector)

    def summarize(self, resource: dict[str, Any]) -> ExecutionSummary:
        metadata = resource.get("metadata", {}) or {}
        labels = metadata.get("labels", {}) or {}
        status, finished, succeeded, message = _pipelinerun_state(resource)
        status_payload = resource.get("status", {}) or {}
        return ExecutionSummary(
            name=str(metadata.get("name", "")),
            namespace=str(metadata.get("namespace", "")),
            experiment=str(labels.get("benchflow.io/experiment", "")),
            platform=str(labels.get("benchflow.io/platform", "")),
            mode=str(labels.get("benchflow.io/mode", "")),
            backend="tekton",
            status=status,
            finished=finished,
            succeeded=succeeded,
            start_time=str(
                status_payload.get("startTime") or metadata.get("creationTimestamp", "")
            ),
            completion_time=str(status_payload.get("completionTime", "")),
            message=str(message),
        )

    def cancel(self, namespace: str, name: str) -> None:
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

    def follow(self, namespace: str, name: str, *, poll_interval: int = 5) -> bool:
        step(f"Watching PipelineRun {name} in namespace {namespace}")
        interactive = sys.stdout.isatty()
        watch_ui: _TerminalWatchUI | None = _TerminalWatchUI() if interactive else None
        last_state: tuple[str, bool, bool, str] | None = None
        last_pairs: list[tuple[str, str]] | None = None
        spinner_index = 0
        while True:
            payload = _get_pipelinerun(namespace, name)
            if payload is None:
                raise CommandError(
                    f"PipelineRun {name} in namespace {namespace} was not found"
                )
            state = _pipelinerun_state(payload)
            pairs = _task_status_pairs(payload)
            spinner_frame = _SPINNER_FRAMES[spinner_index % len(_SPINNER_FRAMES)]
            spinner_index += 1

            if interactive and watch_ui is not None:
                watch_ui.update(
                    _render_status_line(
                        pairs, interactive=True, spinner_frame=spinner_frame
                    )
                )
            elif state != last_state or pairs != last_pairs:
                if pairs:
                    detail(
                        _render_status_line(
                            pairs, interactive=False, spinner_frame=spinner_frame
                        )
                    )
                label, _, _, message = state
                detail(f"{name}: {label}")
                if message:
                    detail(message)

            last_state = state
            last_pairs = pairs
            label, finished, succeeded, message = state
            if finished:
                if succeeded:
                    success(f"{name}: {label}")
                else:
                    warning(f"{name}: {label}")
                if message:
                    detail(message)
                return succeeded
            time.sleep(poll_interval)

    def list_steps(self, namespace: str, name: str) -> list[str]:
        pipelinerun = _get_pipelinerun(namespace, name)
        if pipelinerun is None:
            raise CommandError(
                f"PipelineRun {name} in namespace {namespace} was not found"
            )
        return _pipeline_task_names(pipelinerun)

    def logs(
        self,
        namespace: str,
        name: str,
        *,
        step_name: str | None = None,
        all_logs: bool = False,
        all_containers: bool = False,
    ) -> None:
        pipelinerun = _get_pipelinerun(namespace, name)
        if pipelinerun is None:
            raise CommandError(
                f"PipelineRun {name} in namespace {namespace} was not found"
            )

        if all_logs:
            if shutil.which("tkn") is not None:
                step(f"Following PipelineRun {name} logs in namespace {namespace}")
                subprocess.run(
                    ["tkn", "pipelinerun", "logs", "-f", "-n", namespace, name],
                    check=False,
                )
                return

            step(f"Following PipelineRun {name} logs in namespace {namespace}")
            max_log_requests = _pipelinerun_log_stream_limit(namespace, name)
            subprocess.run(
                [
                    "oc",
                    "logs",
                    "-f",
                    "-n",
                    namespace,
                    "-l",
                    f"tekton.dev/pipelineRun={name}",
                    "--all-containers=true",
                    "--prefix",
                    "--max-log-requests",
                    str(max_log_requests),
                ],
                check=False,
            )
            return

        available_steps = _pipeline_task_names(pipelinerun)
        if not step_name:
            raise ValidationError(
                "step logs require a step name; use list_steps first or pass --all"
            )
        if step_name not in available_steps:
            choices = ", ".join(available_steps) or "<none>"
            raise ValidationError(
                f"unknown step {step_name!r} for PipelineRun {name}; available steps: {choices}"
            )

        taskruns = _taskruns_for_step(namespace, name, step_name)
        if not taskruns:
            raise CommandError(
                f"PipelineRun step {step_name!r} exists but has no TaskRun logs yet"
            )
        for taskrun in taskruns:
            _stream_taskrun_logs(namespace, taskrun, all_containers=all_containers)
