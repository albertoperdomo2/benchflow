from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .backends import ArgoBackend, ExecutionBackend, ExecutionSummary
from .cluster import CommandError
from .loaders import load_run_plan_data, load_run_plan_file
from .models import ResolvedRunPlan, ValidationError

DEFAULT_EXECUTION_NAME = "benchflow-e2e"
DEFAULT_MATRIX_EXECUTION_NAME = "benchflow-matrix"

_BACKENDS: dict[str, ExecutionBackend] = {
    "argo": ArgoBackend(),
}


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


def normalize_execution_backend(name: str | None) -> str:
    backend = str(name or "argo").strip().lower()
    if backend not in _BACKENDS:
        choices = ", ".join(sorted(_BACKENDS))
        raise ValidationError(
            f"unsupported execution backend: {backend!r}; expected one of: {choices}"
        )
    return backend


def get_execution_backend(name: str | None) -> ExecutionBackend:
    return _BACKENDS[normalize_execution_backend(name)]


def render_execution_manifest(
    plan: ResolvedRunPlan,
    *,
    execution_name: str = DEFAULT_EXECUTION_NAME,
    setup_mode: str = "auto",
    teardown: bool = True,
    backend: str | None = None,
) -> dict[str, Any]:
    backend_name = normalize_execution_backend(backend or plan.execution.backend)
    plan.execution.backend = backend_name
    return get_execution_backend(backend_name).render_run(
        plan,
        execution_name=execution_name,
        setup_mode=setup_mode,
        teardown=teardown,
    )


def render_matrix_execution_manifest(
    plans: list[ResolvedRunPlan],
    *,
    execution_name: str = DEFAULT_MATRIX_EXECUTION_NAME,
    child_execution_name: str = DEFAULT_EXECUTION_NAME,
    backend: str | None = None,
) -> dict[str, Any]:
    if not plans:
        raise ValidationError("matrix submission requires at least one RunPlan")
    backend_name = normalize_execution_backend(backend or plans[0].execution.backend)
    if {normalize_execution_backend(plan.execution.backend) for plan in plans} != {
        backend_name
    }:
        raise ValidationError(
            "matrix submission requires all child RunPlans to use the same execution backend"
        )
    for plan in plans:
        plan.execution.backend = backend_name
    return get_execution_backend(backend_name).render_matrix(
        plans,
        execution_name=execution_name,
        child_execution_name=child_execution_name,
    )


def _execution_summaries_for_backend(
    backend_name: str, namespace: str
) -> list[ExecutionSummary]:
    backend = get_execution_backend(backend_name)
    try:
        items = backend.list(
            namespace, label_selector="app.kubernetes.io/name=benchflow"
        )
    except CommandError:
        return []
    return [backend.summarize(item) for item in items]


def list_benchflow_executions(
    namespace: str,
    *,
    include_completed: bool = True,
    backend: str | None = None,
) -> list[dict[str, Any]]:
    backend_names = (
        [normalize_execution_backend(backend)] if backend else list(_BACKENDS)
    )
    summaries: list[ExecutionSummary] = []
    for backend_name in backend_names:
        summaries.extend(_execution_summaries_for_backend(backend_name, namespace))
    summaries.sort(key=lambda item: item.start_time or "", reverse=True)
    if not include_completed:
        summaries = [item for item in summaries if not item.finished]
    return [item.to_dict() for item in summaries]


def _detect_execution_backend(namespace: str, name: str) -> str:
    backend_name = "argo"
    if get_execution_backend(backend_name).get(namespace, name) is None:
        raise CommandError(
            f"no BenchFlow execution named {name!r} found in {namespace}"
        )
    return backend_name


def get_execution(
    namespace: str, name: str, *, backend: str | None = None
) -> dict[str, Any]:
    backend_name = (
        normalize_execution_backend(backend)
        if backend
        else _detect_execution_backend(namespace, name)
    )
    payload = get_execution_backend(backend_name).get(namespace, name)
    if payload is None:
        raise CommandError(f"Workflow {name} in namespace {namespace} was not found")
    return payload


def summarize_execution(
    namespace: str, name: str, *, backend: str | None = None
) -> dict[str, Any]:
    backend_name = (
        normalize_execution_backend(backend)
        if backend
        else _detect_execution_backend(namespace, name)
    )
    payload = get_execution_backend(backend_name).get(namespace, name)
    if payload is None:
        raise CommandError(f"Workflow {name} in namespace {namespace} was not found")
    return get_execution_backend(backend_name).summarize(payload).to_dict()


def cancel_execution(namespace: str, name: str, *, backend: str | None = None) -> None:
    backend_name = (
        normalize_execution_backend(backend)
        if backend
        else _detect_execution_backend(namespace, name)
    )
    get_execution_backend(backend_name).cancel(namespace, name)


def follow_execution(
    namespace: str,
    name: str,
    *,
    backend: str | None = None,
    poll_interval: int = 5,
) -> bool:
    backend_name = (
        normalize_execution_backend(backend)
        if backend
        else _detect_execution_backend(namespace, name)
    )
    return get_execution_backend(backend_name).follow(
        namespace, name, poll_interval=poll_interval
    )
