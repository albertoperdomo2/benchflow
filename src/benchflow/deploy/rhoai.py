from __future__ import annotations

import time
from pathlib import Path

import yaml

from ..cluster import CommandError, require_any_command, run_command, run_json_command
from ..models import ResolvedRunPlan
from ..renderers.deployment import (
    render_runtime_pvc_manifests,
    render_rhoai_manifest,
    render_rhoai_profiler_configmap,
)
from ..ui import detail, step, success

_PUBLIC_ROUTE_AUTH_TIMEOUT_SECONDS = 900


def _deployment_kind(plan: ResolvedRunPlan) -> str:
    return str(plan.deployment.target.resource_kind or "LLMInferenceService").strip()


def _deployment_resource(plan: ResolvedRunPlan) -> str:
    return _deployment_kind(plan).lower()


def _deployment_manifest_filename(plan: ResolvedRunPlan) -> str:
    if _deployment_kind(plan) == "InferenceService":
        return "inferenceservice.yaml"
    return "llminferenceservice.yaml"


def _deployment_exists(
    namespace: str, release_name: str, kubectl_cmd: str, resource: str
) -> bool:
    result = run_command(
        [
            kubectl_cmd,
            "get",
            resource,
            release_name,
            "-n",
            namespace,
            "-o",
            "name",
        ],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _profiling_enabled(plan: ResolvedRunPlan) -> bool:
    return plan.execution.profiling.enabled


def _apply_runtime_pvc_manifests(plan: ResolvedRunPlan, kubectl_cmd: str) -> None:
    for manifest in render_runtime_pvc_manifests(plan):
        name = str(manifest.get("metadata", {}).get("name") or "").strip()
        step(f"Ensuring runtime PVC {name} in namespace {plan.deployment.namespace}")
        run_command(
            [kubectl_cmd, "apply", "-f", "-"],
            input_text=yaml.safe_dump(manifest, sort_keys=False),
        )


def _status_snapshot(payload: dict[str, object]) -> tuple[bool, str, str]:
    status = payload.get("status", {})
    if not isinstance(status, dict):
        return False, "", ""
    url = str(status.get("url") or "").strip()
    conditions = status.get("conditions") or []
    ready = False
    reason = ""
    if isinstance(conditions, list):
        for condition in conditions:
            if not isinstance(condition, dict):
                continue
            if condition.get("type") != "Ready":
                continue
            ready = str(condition.get("status", "")).lower() == "true"
            reason = str(condition.get("reason") or condition.get("message") or "")
            break
    return ready, url, reason


def _auth_disabled(plan: ResolvedRunPlan) -> bool:
    return not bool(plan.deployment.options.get("enable_auth", False))


def _route_authpolicy_name(release_name: str) -> str:
    return f"{release_name}-kserve-route-authn"


def _route_authpolicy_snapshot(
    payload: dict[str, object],
) -> tuple[bool, bool | None, bool | None]:
    spec = payload.get("spec", {})
    status = payload.get("status", {})
    anonymous = False
    accepted: bool | None = None
    enforced: bool | None = None

    if isinstance(spec, dict):
        rules = spec.get("rules", {})
        if isinstance(rules, dict):
            authentication = rules.get("authentication", {})
            if isinstance(authentication, dict):
                for rule in authentication.values():
                    if not isinstance(rule, dict):
                        continue
                    if "anonymous" in rule:
                        anonymous = True
                        break

    if isinstance(status, dict):
        conditions = status.get("conditions") or []
        if isinstance(conditions, list):
            for condition in conditions:
                if not isinstance(condition, dict):
                    continue
                condition_type = str(condition.get("type") or "")
                condition_status = str(condition.get("status") or "").lower() == "true"
                if condition_type == "Accepted":
                    accepted = condition_status
                elif condition_type == "Enforced":
                    enforced = condition_status

    return anonymous, accepted, enforced


def _authpolicy_resource_unavailable(result_stderr: str) -> bool:
    return (
        "the server doesn't have a resource type" in result_stderr
        or "the server does not have a resource type" in result_stderr
        or "no matches for kind" in result_stderr
    )


def _status_label(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "yes" if value else "no"


def _verify_public_route_auth(plan: ResolvedRunPlan, timeout_seconds: int) -> None:
    kubectl_cmd = require_any_command("oc", "kubectl")
    namespace = plan.deployment.namespace
    authpolicy_name = _route_authpolicy_name(plan.deployment.release_name)
    deadline = time.time() + timeout_seconds
    last_snapshot: tuple[bool, bool | None, bool | None] | None = None

    step(
        f"Waiting for RHOAI route AuthPolicy {authpolicy_name} "
        f"in namespace {namespace} to allow anonymous access"
    )
    detail(f"Timeout: {timeout_seconds}s")

    while time.time() < deadline:
        result = run_command(
            [
                kubectl_cmd,
                "get",
                "authpolicy",
                authpolicy_name,
                "-n",
                namespace,
                "-o",
                "json",
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr or ""
            if _authpolicy_resource_unavailable(stderr):
                detail(
                    "AuthPolicy resource type is unavailable; skipping anonymous "
                    "route policy status check"
                )
                return
            detail(f"AuthPolicy {authpolicy_name} not published yet")
            time.sleep(5)
            continue

        payload = run_json_command(
            [
                kubectl_cmd,
                "get",
                "authpolicy",
                authpolicy_name,
                "-n",
                namespace,
                "-o",
                "json",
            ]
        )
        snapshot = _route_authpolicy_snapshot(payload)
        if snapshot != last_snapshot:
            anonymous, accepted, enforced = snapshot
            detail(
                "Anonymous auth: "
                f"{'yes' if anonymous else 'no'}, "
                f"accepted: {_status_label(accepted)}, "
                f"enforced: {_status_label(enforced)}"
            )
            last_snapshot = snapshot

        anonymous, _, _ = snapshot
        if anonymous:
            success(
                f"RHOAI route AuthPolicy {authpolicy_name} declares anonymous access"
            )
            return
        time.sleep(5)

    raise CommandError(
        "timed out waiting for the RHOAI route AuthPolicy to allow anonymous access: "
        f"{authpolicy_name}"
    )


def _verify_deployment(plan: ResolvedRunPlan, timeout_seconds: int) -> None:
    kubectl_cmd = require_any_command("oc", "kubectl")
    namespace = plan.deployment.namespace
    release_name = plan.deployment.release_name
    resource = _deployment_resource(plan)
    resource_kind = _deployment_kind(plan)
    deadline = time.time() + timeout_seconds
    last_snapshot: tuple[bool, str, str] | None = None

    step(
        f"Waiting for RHOAI deployment {release_name} in namespace {namespace} to become ready"
    )

    while time.time() < deadline:
        payload = run_json_command(
            [
                kubectl_cmd,
                "get",
                resource,
                release_name,
                "-n",
                namespace,
                "-o",
                "json",
            ]
        )
        snapshot = _status_snapshot(payload)
        if snapshot != last_snapshot:
            ready, url, reason = snapshot
            detail(
                f"Ready: {'yes' if ready else 'no'}, "
                f"status.url: {url or 'not published yet'}, "
                f"status: {reason or 'waiting'}"
            )
            last_snapshot = snapshot

        ready, url, _ = snapshot
        if ready and url:
            success(f"RHOAI deployment {release_name} is ready and published at {url}")
            if _auth_disabled(plan) and _deployment_kind(plan) == "LLMInferenceService":
                _verify_public_route_auth(
                    plan,
                    timeout_seconds=min(
                        timeout_seconds, _PUBLIC_ROUTE_AUTH_TIMEOUT_SECONDS
                    ),
                )
            return
        time.sleep(10)

    raise CommandError(
        f"timed out waiting for {resource_kind} deployment {release_name} to become ready"
    )


def deploy_rhoai(
    plan: ResolvedRunPlan,
    *,
    manifests_dir: Path | None = None,
    skip_if_exists: bool = True,
    verify: bool = True,
    verify_timeout_seconds: int = 1800,
) -> Path:
    kubectl_cmd = require_any_command("oc", "kubectl")
    namespace = plan.deployment.namespace
    release_name = plan.deployment.release_name
    resource = _deployment_resource(plan)
    resource_kind = _deployment_kind(plan)

    if skip_if_exists and _deployment_exists(
        namespace, release_name, kubectl_cmd, resource
    ):
        success(f"Skipping deploy; {resource_kind} {release_name} already exists")
        return manifests_dir.resolve() if manifests_dir else Path.cwd()

    profiler_configmap = (
        render_rhoai_profiler_configmap(plan) if _profiling_enabled(plan) else None
    )
    manifests = [render_rhoai_manifest(plan)]
    if manifests_dir is not None:
        manifests_dir.mkdir(parents=True, exist_ok=True)
        for pvc_manifest in render_runtime_pvc_manifests(plan):
            pvc_name = str(pvc_manifest.get("metadata", {}).get("name") or "runtime")
            pvc_target = manifests_dir / f"pvc-{pvc_name}.yaml"
            pvc_target.write_text(
                yaml.safe_dump(pvc_manifest, sort_keys=False), encoding="utf-8"
            )
            detail(f"Rendered runtime PVC manifest written to {pvc_target}")
        if profiler_configmap is not None:
            profiler_target = manifests_dir / "vllm-profiler-configmap.yaml"
            profiler_target.write_text(
                yaml.safe_dump(profiler_configmap, sort_keys=False), encoding="utf-8"
            )
            detail(f"Rendered profiler ConfigMap written to {profiler_target}")
        names = [_deployment_manifest_filename(plan)]
        for manifest, name in zip(manifests, names, strict=True):
            target = manifests_dir / name
            target.write_text(
                yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
            )
            detail(f"Rendered RHOAI manifest written to {target}")

    if profiler_configmap is not None:
        configmap_name = str(
            profiler_configmap.get("metadata", {}).get("name") or "vllm-profiler"
        )
        step(f"Applying profiler ConfigMap {configmap_name} in namespace {namespace}")
        run_command(
            [kubectl_cmd, "apply", "-f", "-"],
            input_text=yaml.safe_dump(profiler_configmap, sort_keys=False),
        )
        success(f"Applied profiler ConfigMap {configmap_name} in namespace {namespace}")

    _apply_runtime_pvc_manifests(plan, kubectl_cmd)

    step(
        f"Applying RHOAI {plan.deployment.mode} deployment {release_name} "
        f"in namespace {namespace}"
    )
    for manifest in manifests:
        run_command(
            [kubectl_cmd, "apply", "-f", "-"],
            input_text=yaml.safe_dump(manifest, sort_keys=False),
        )
    success(f"Applied {resource_kind} {release_name} in namespace {namespace}")

    if verify:
        _verify_deployment(plan, verify_timeout_seconds)

    return manifests_dir.resolve() if manifests_dir else Path.cwd()
