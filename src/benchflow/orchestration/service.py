from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from ..cluster import CommandError, create_manifest
from ..contracts import ExecutionContext, ResolvedRunPlan, ValidationError
from ..loaders import load_run_plan_data, load_run_plan_file
from ..setup import load_setup_state
from ..toolbox import setup_platform, teardown_platform
from ..ui import detail, step, success, warning
from .argo import ArgoOrchestrator
from .base import ExecutionOrchestrator

DEFAULT_EXECUTION_NAME = "benchflow-e2e"
DEFAULT_MATRIX_EXECUTION_NAME = "benchflow-matrix"

_BACKENDS: dict[str, ExecutionOrchestrator] = {
    "argo": ArgoOrchestrator(),
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


def get_execution_backend(name: str | None) -> ExecutionOrchestrator:
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


def _execution_summaries_for_backend(namespace: str) -> list[dict[str, Any]]:
    backend = get_execution_backend("argo")
    try:
        items = backend.list(
            namespace, label_selector="app.kubernetes.io/name=benchflow"
        )
    except CommandError:
        return []
    return [backend.summarize(item).to_dict() for item in items]


def list_benchflow_executions(
    namespace: str,
    *,
    include_completed: bool = True,
) -> list[dict[str, Any]]:
    summaries = _execution_summaries_for_backend(namespace)
    summaries.sort(key=lambda item: item.get("start_time", "") or "", reverse=True)
    if not include_completed:
        summaries = [item for item in summaries if not item.get("finished")]
    return summaries


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


def list_execution_steps(
    namespace: str,
    name: str,
    *,
    backend: str | None = None,
) -> list[str]:
    backend_name = (
        normalize_execution_backend(backend)
        if backend
        else _detect_execution_backend(namespace, name)
    )
    return get_execution_backend(backend_name).list_steps(namespace, name)


def stream_execution_logs(
    namespace: str,
    name: str,
    *,
    backend: str | None = None,
    step_name: str | None = None,
    all_logs: bool = False,
) -> None:
    backend_name = (
        normalize_execution_backend(backend)
        if backend
        else _detect_execution_backend(namespace, name)
    )
    get_execution_backend(backend_name).logs(
        namespace,
        name,
        step_name=step_name,
        all_logs=all_logs,
    )


def submit_execution_manifest(manifest: dict[str, Any], namespace: str) -> str:
    submitted = create_manifest(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True), namespace
    )
    name = submitted.get("metadata", {}).get("name")
    if not name:
        raise ValidationError("execution submission returned no name")
    return str(name)


def run_matrix_supervisor(
    plans: list[ResolvedRunPlan],
    *,
    child_execution_name: str,
) -> list[str]:
    if not plans:
        raise ValidationError("matrix execution requires at least one RunPlan")

    namespaces = {plan.deployment.namespace for plan in plans}
    execution_backends = {plan.execution.backend for plan in plans}
    if len(namespaces) != 1:
        raise ValidationError(
            "matrix execution requires all child RunPlans to target the same namespace"
        )
    if len(execution_backends) != 1:
        raise ValidationError(
            "matrix execution requires all child RunPlans to use the same execution backend"
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
            )
            name = submit_execution_manifest(manifest, plan.deployment.namespace)
            detail(f"Created execution {name} in namespace {plan.deployment.namespace}")
            succeeded = follow_execution(
                plan.deployment.namespace,
                name,
                backend=plan.execution.backend,
            )
            if succeeded:
                success(f"[{index}/{total}] {name} succeeded")
                continue
            warning(f"[{index}/{total}] {name} failed")
            failures.append(name)
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
