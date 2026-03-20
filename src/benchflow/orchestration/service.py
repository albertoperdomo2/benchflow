from __future__ import annotations

import copy
import json
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any

from ..cluster import CommandError, create_manifest
from ..contracts import ExecutionContext, ResolvedRunPlan, ValidationError
from ..kueue import (
    create_reservation_workload,
    delete_reservation_workload,
    create_submission_configmap,
    delete_submission_configmap,
    list_reservation_workloads,
    reservation_workload_by_execution_name,
    queue_name_from_labels,
    requested_gpus_from_labels,
    reservation_required_for_labels,
    submission_configmap_name_from_labels,
    summarize_reservation_workload,
)
from ..loaders import load_run_plan_data, load_run_plan_file
from ..models import sanitize_name
from ..setup import load_setup_state
from ..toolbox import setup_platform, teardown_platform
from ..ui import detail, step, success, warning
from .tekton import TektonOrchestrator

DEFAULT_EXECUTION_NAME = "benchflow-e2e"
DEFAULT_MATRIX_EXECUTION_NAME = "benchflow-matrix"

_TEKTON = TektonOrchestrator()


def _materialize_execution_name(manifest: dict[str, Any]) -> tuple[dict[str, Any], str]:
    rendered = copy.deepcopy(manifest)
    metadata = rendered.setdefault("metadata", {})
    explicit_name = str(metadata.get("name") or "").strip()
    if explicit_name:
        labels = metadata.setdefault("labels", {})
        labels["benchflow.io/execution-name"] = explicit_name
        return rendered, explicit_name

    raw_prefix = str(metadata.get("generateName") or "benchflow-").strip().rstrip("-")
    prefix = sanitize_name(raw_prefix or "benchflow", max_length=56)
    execution_name = f"{prefix}-{secrets.token_hex(3)}"
    metadata["name"] = execution_name
    metadata.pop("generateName", None)
    labels = metadata.setdefault("labels", {})
    labels["benchflow.io/execution-name"] = execution_name
    return rendered, execution_name


def _pending_execution_summary(namespace: str, name: str) -> dict[str, Any] | None:
    workload = reservation_workload_by_execution_name(namespace, name)
    if workload is None:
        return None
    return summarize_reservation_workload(workload).to_dict()


def load_run_plan_from_sources(
    *,
    run_plan_file: str | None = None,
    run_plan_json: str | None = None,
) -> ResolvedRunPlan:
    if run_plan_file:
        return load_run_plan_file(Path(run_plan_file).resolve())
    if run_plan_json:
        try:
            raw = json.loads(run_plan_json)
        except json.JSONDecodeError as exc:
            raise ValidationError("invalid JSON passed to --run-plan-json") from exc
        return load_run_plan_data(raw)
    raise ValidationError("provide --run-plan-file or --run-plan-json")


def require_platform(plan: ResolvedRunPlan, platform: str) -> None:
    if plan.deployment.platform != platform:
        raise ValidationError(
            f"unsupported deployment platform: {plan.deployment.platform}; only {platform} is implemented"
        )


def render_execution_manifest(
    plan: ResolvedRunPlan,
    *,
    execution_name: str = DEFAULT_EXECUTION_NAME,
    setup_mode: str = "auto",
    teardown: bool = True,
    skip_kueue_reservation: bool = False,
    benchflow_image: str | None = None,
) -> dict[str, Any]:
    return _TEKTON.render_run(
        plan,
        execution_name=execution_name,
        setup_mode=setup_mode,
        teardown=teardown,
        skip_kueue_reservation=skip_kueue_reservation,
        benchflow_image=benchflow_image,
    )


def render_matrix_execution_manifest(
    plans: list[ResolvedRunPlan],
    *,
    execution_name: str = DEFAULT_MATRIX_EXECUTION_NAME,
    child_execution_name: str = DEFAULT_EXECUTION_NAME,
    benchflow_image: str | None = None,
) -> dict[str, Any]:
    if not plans:
        raise ValidationError("matrix submission requires at least one RunPlan")
    return _TEKTON.render_matrix(
        plans,
        execution_name=execution_name,
        child_execution_name=child_execution_name,
        benchflow_image=benchflow_image,
    )


def _execution_summaries_for_backend(namespace: str) -> list[dict[str, Any]]:
    try:
        items = _TEKTON.list(
            namespace, label_selector="app.kubernetes.io/name=benchflow"
        )
    except CommandError:
        return []
    return [_TEKTON.summarize(item).to_dict() for item in items]


def list_benchflow_executions(
    namespace: str,
    *,
    include_completed: bool = True,
) -> list[dict[str, Any]]:
    summaries = _execution_summaries_for_backend(namespace)
    active_names = {str(item.get("name") or "") for item in summaries}
    pending_entries: list[dict[str, Any]] = []
    for workload in list_reservation_workloads(namespace):
        summary = summarize_reservation_workload(workload).to_dict()
        if not summary["name"] or summary["name"] in active_names:
            continue
        pending_entries.append(summary)
    summaries.extend(pending_entries)
    summaries.sort(key=lambda item: item.get("start_time", "") or "", reverse=True)
    if not include_completed:
        summaries = [item for item in summaries if not item.get("finished")]
    return summaries


def _ensure_execution_exists(namespace: str, name: str) -> None:
    if (
        _TEKTON.get(namespace, name) is None
        and _pending_execution_summary(namespace, name) is None
    ):
        raise CommandError(
            f"no BenchFlow execution named {name!r} found in {namespace}"
        )


def get_execution(namespace: str, name: str) -> dict[str, Any]:
    payload = _TEKTON.get(namespace, name)
    if payload is None:
        raise CommandError(f"PipelineRun {name} in namespace {namespace} was not found")
    return payload


def summarize_execution(namespace: str, name: str) -> dict[str, Any]:
    payload = _TEKTON.get(namespace, name)
    if payload is not None:
        return _TEKTON.summarize(payload).to_dict()
    pending = _pending_execution_summary(namespace, name)
    if pending is not None:
        return pending
    raise CommandError(f"PipelineRun {name} in namespace {namespace} was not found")


def cancel_execution(namespace: str, name: str) -> None:
    payload = _TEKTON.get(namespace, name)
    if payload is not None:
        _TEKTON.cancel(namespace, name)
        return
    workload = reservation_workload_by_execution_name(namespace, name)
    if workload is None:
        raise CommandError(
            f"no BenchFlow execution named {name!r} found in {namespace}"
        )
    delete_submission_configmap(
        namespace,
        submission_configmap_name_from_labels(
            (workload.get("metadata", {}) or {}).get("labels", {}) or {}
        ),
    )
    delete_reservation_workload(
        namespace, str((workload.get("metadata", {}) or {}).get("name") or "")
    )


def follow_execution(
    namespace: str,
    name: str,
    *,
    poll_interval: int = 5,
) -> bool:
    _ensure_execution_exists(namespace, name)
    queued_state: tuple[str, str] | None = None
    while True:
        payload = _TEKTON.get(namespace, name)
        if payload is not None:
            return _TEKTON.follow(namespace, name, poll_interval=poll_interval)
        workload = reservation_workload_by_execution_name(namespace, name)
        if workload is None:
            raise CommandError(
                f"queued BenchFlow execution {name!r} disappeared before it started"
            )
        summary = summarize_reservation_workload(workload)
        current_state = (summary.status, summary.message)
        if current_state != queued_state:
            if summary.message:
                detail(f"{name}: {summary.status} ({summary.message})")
            else:
                detail(f"{name}: {summary.status}")
            queued_state = current_state
        time.sleep(max(1, int(poll_interval)))


def list_execution_steps(namespace: str, name: str) -> list[str]:
    if _TEKTON.get(namespace, name) is None:
        if _pending_execution_summary(namespace, name) is not None:
            raise CommandError(
                f"execution {name!r} is still queued and has no Pipeline tasks yet"
            )
    _ensure_execution_exists(namespace, name)
    return _TEKTON.list_steps(namespace, name)


def stream_execution_logs(
    namespace: str,
    name: str,
    *,
    step_name: str | None = None,
    all_logs: bool = False,
    all_containers: bool = False,
) -> None:
    if _TEKTON.get(namespace, name) is None:
        if _pending_execution_summary(namespace, name) is not None:
            raise CommandError(
                f"execution {name!r} is still queued and has no logs yet"
            )
    _ensure_execution_exists(namespace, name)
    _TEKTON.logs(
        namespace,
        name,
        step_name=step_name,
        all_logs=all_logs,
        all_containers=all_containers,
    )


def submit_execution_manifest(manifest: dict[str, Any], namespace: str) -> str:
    manifest, execution_name = _materialize_execution_name(manifest)
    labels = {
        str(key): str(value)
        for key, value in (manifest.get("metadata", {}) or {}).get("labels", {}).items()
    }
    metadata = manifest.setdefault("metadata", {})
    metadata["namespace"] = namespace
    if not reservation_required_for_labels(labels):
        submitted = create_manifest(
            json.dumps(manifest, separators=(",", ":"), sort_keys=True), namespace
        )
        name = submitted.get("metadata", {}).get("name")
        if not name:
            raise ValidationError("execution submission returned no name")
        return str(name)

    cluster_name = queue_name_from_labels(labels)
    requested_gpus = requested_gpus_from_labels(labels)
    execution_timeout = str(
        (manifest.get("spec", {}) or {}).get("timeouts", {}).get("pipeline") or "3h"
    )
    configmap_name = ""
    reservation_name = ""
    try:
        configmap_name = create_submission_configmap(
            namespace=namespace,
            execution_name=execution_name,
            manifest=manifest,
        )
        reservation_name = create_reservation_workload(
            namespace=namespace,
            cluster_name=cluster_name,
            execution_prefix=execution_name,
            execution_name=execution_name,
            submission_configmap_name=configmap_name,
            requested_gpu_count=requested_gpus,
            execution_timeout=execution_timeout,
            execution_labels=labels,
        )
        return execution_name
    except BaseException:
        if reservation_name:
            delete_reservation_workload(namespace, reservation_name)
        if configmap_name:
            delete_submission_configmap(namespace, configmap_name)
        raise


def run_matrix_supervisor(
    plans: list[ResolvedRunPlan],
    *,
    child_execution_name: str,
    benchflow_image: str | None = None,
) -> list[str]:
    if not plans:
        raise ValidationError("matrix execution requires at least one RunPlan")

    namespaces = {plan.deployment.namespace for plan in plans}
    if len(namespaces) != 1:
        raise ValidationError(
            "matrix execution requires all child RunPlans to target the same namespace"
        )

    step(
        f"Running matrix supervisor for {plans[0].metadata.name} "
        f"with {len(plans)} profile combination(s)"
    )
    failures: list[str] = []
    total = len(plans)
    setup_state: dict[str, Any] = {}
    setup_hoisted = False
    setup_state_dir: tempfile.TemporaryDirectory[str] | None = None
    setup_state_path: Path | None = None
    submitted: dict[str, str] = {}

    matrix_platform = (
        plans[0].deployment.platform
        if len({plan.deployment.platform for plan in plans}) == 1
        else ""
    )
    if (
        matrix_platform in {"llm-d", "rhoai"}
        and all(plan.stages.deploy for plan in plans)
        and all(plan.stages.cleanup for plan in plans)
    ):
        step(f"Setting up {matrix_platform} platform once for the whole matrix")
        setup_state_dir = tempfile.TemporaryDirectory(prefix="benchflow-matrix-setup-")
        setup_state_path = Path(setup_state_dir.name) / "setup-state.json"
        setup_hoisted = True
        setup_state = setup_platform(
            plans[0],
            context=ExecutionContext(state_path=setup_state_path),
        )

    try:
        for index, plan in enumerate(plans, start=1):
            descriptor = (
                f"deployment={plan.profiles.deployment}, "
                f"benchmark={plan.profiles.benchmark}, "
                f"metrics={plan.profiles.metrics}"
            )
            step(f"[{index}/{total}] Submitting child execution")
            detail(descriptor)
            manifest = render_execution_manifest(
                plan,
                execution_name=child_execution_name,
                setup_mode="skip" if setup_hoisted else "auto",
                teardown=False if setup_hoisted else True,
                skip_kueue_reservation=False,
                benchflow_image=benchflow_image,
            )
            name = submit_execution_manifest(manifest, plan.deployment.namespace)
            submitted[name] = descriptor
            detail(f"Queued execution {name} in namespace {plan.deployment.namespace}")

        if submitted:
            step(f"Watching {len(submitted)} child execution(s)")
        last_states: dict[str, tuple[str, bool, str]] = {}
        pending = set(submitted)
        while pending:
            for name in list(pending):
                summary = summarize_execution(plans[0].deployment.namespace, name)
                state = (
                    str(summary.get("status") or ""),
                    bool(summary.get("finished")),
                    str(summary.get("message") or ""),
                )
                if last_states.get(name) != state:
                    status_text = state[0]
                    message_text = state[2]
                    if message_text:
                        detail(f"{name}: {status_text} ({message_text})")
                    else:
                        detail(f"{name}: {status_text}")
                    last_states[name] = state
                if not summary.get("finished"):
                    continue
                pending.remove(name)
                if summary.get("succeeded"):
                    success(f"{name} succeeded")
                    continue
                warning(f"{name} failed")
                failures.append(name)
            if pending:
                time.sleep(5)
    finally:
        if setup_hoisted:
            step(f"Tearing down hoisted {plans[0].deployment.platform} platform setup")
            if setup_state_path is not None:
                setup_state = load_setup_state(setup_state_path)
            teardown_platform(
                plans[0],
                setup_state,
                context=ExecutionContext(state_path=setup_state_path),
            )
        if setup_state_dir is not None:
            setup_state_dir.cleanup()

    if failures:
        raise ValidationError(
            f"{len(failures)} matrix child run(s) failed: {', '.join(failures)}"
        )

    success(f"Matrix supervisor completed {total} child execution(s)")
    return failures
