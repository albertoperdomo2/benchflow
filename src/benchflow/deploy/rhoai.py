from __future__ import annotations

import time
from pathlib import Path

import yaml

from ..cluster import CommandError, require_any_command, run_command, run_json_command
from ..models import ResolvedRunPlan
from ..renderers.deployment import render_rhoai_manifest
from ..ui import detail, step, success


def _deployment_exists(namespace: str, release_name: str, kubectl_cmd: str) -> bool:
    result = run_command(
        [
            kubectl_cmd,
            "get",
            "llminferenceservice",
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


def _verify_deployment(plan: ResolvedRunPlan, timeout_seconds: int) -> None:
    kubectl_cmd = require_any_command("oc", "kubectl")
    namespace = plan.deployment.namespace
    release_name = plan.deployment.release_name
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
                "llminferenceservice",
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
            return
        time.sleep(10)

    raise CommandError(
        f"timed out waiting for RHOAI deployment {release_name} to become ready"
    )


def deploy_rhoai(
    plan: ResolvedRunPlan,
    *,
    manifests_dir: Path | None = None,
    skip_if_exists: bool = True,
    verify: bool = True,
    verify_timeout_seconds: int = 900,
) -> Path:
    kubectl_cmd = require_any_command("oc", "kubectl")
    namespace = plan.deployment.namespace
    release_name = plan.deployment.release_name

    if skip_if_exists and _deployment_exists(namespace, release_name, kubectl_cmd):
        success(f"Skipping deploy; LLMInferenceService {release_name} already exists")
        return manifests_dir.resolve() if manifests_dir else Path.cwd()

    manifest = render_rhoai_manifest(plan)
    if manifests_dir is not None:
        manifests_dir.mkdir(parents=True, exist_ok=True)
        target = manifests_dir / "llminferenceservice.yaml"
        target.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
        detail(f"Rendered RHOAI manifest written to {target}")

    step(
        f"Applying RHOAI {plan.deployment.mode} deployment {release_name} "
        f"in namespace {namespace}"
    )
    run_command(
        [kubectl_cmd, "apply", "-f", "-"],
        input_text=yaml.safe_dump(manifest, sort_keys=False),
    )
    success(f"Applied LLMInferenceService {release_name} in namespace {namespace}")

    if verify:
        _verify_deployment(plan, verify_timeout_seconds)

    return manifests_dir.resolve() if manifests_dir else Path.cwd()
