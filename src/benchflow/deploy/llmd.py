from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

from ..cluster import (
    CommandError,
    require_any_command,
    require_command,
    run_command,
    run_json_command,
)
from ..models import ResolvedRunPlan
from ..repository import clone_repo
from ..ui import detail, step, success


def _release_exists(namespace: str, release_name: str) -> bool:
    helm_json = run_json_command(["helm", "list", "-n", namespace, "-o", "json"])
    return any(entry.get("name") == f"ms-{release_name}" for entry in helm_json)


def _environment_name(plan: ResolvedRunPlan) -> str:
    gateway = plan.deployment.gateway
    if gateway in {"istio", "kgateway", "agentgateway", "gke", "standalone"}:
        return gateway
    return "default"


def _model_uri(plan: ResolvedRunPlan) -> str:
    storage = plan.deployment.model_storage
    return (
        f"pvc://{storage.pvc_name}{storage.cache_dir}/{plan.model.pvc_directory_name}"
    )


def _model_mount_path(plan: ResolvedRunPlan) -> str:
    storage = plan.deployment.model_storage
    return f"{storage.mount_path}{storage.cache_dir}/{plan.model.pvc_directory_name}"


def _cuda_visible_devices(tp: int) -> str:
    if tp <= 1:
        return "0"
    return ",".join(str(index) for index in range(tp))


def _port_from_values(values: dict[str, Any]) -> int:
    try:
        container = values["decode"]["containers"][0]
    except (KeyError, IndexError, TypeError):
        return 8000

    for probe_name in ("startupProbe", "readinessProbe", "livenessProbe"):
        try:
            return int(container[probe_name]["httpGet"]["port"])
        except (KeyError, TypeError, ValueError):
            continue

    try:
        for port_spec in container.get("ports", []):
            if port_spec.get("name") == "metrics":
                return int(port_spec["containerPort"])
    except (TypeError, ValueError, KeyError):
        pass

    try:
        return int(container["ports"][0]["containerPort"])
    except (KeyError, IndexError, TypeError, ValueError):
        return 8000


def _ensure_container(values: dict[str, Any]) -> dict[str, Any]:
    decode = values.setdefault("decode", {})
    containers = decode.setdefault("containers", [])
    if not containers:
        containers.append({"name": "vllm"})
    return containers[0]


def _patch_values(plan: ResolvedRunPlan, values_file: Path) -> dict[str, Any]:
    values = yaml.safe_load(values_file.read_text(encoding="utf-8")) or {}
    container = _ensure_container(values)
    decode = values.setdefault("decode", {})
    model_artifacts = values.setdefault("modelArtifacts", {})
    storage = plan.deployment.model_storage
    runtime = plan.deployment.runtime
    port = _port_from_values(values)

    model_artifacts["name"] = plan.model.name
    model_artifacts["uri"] = _model_uri(plan)
    model_artifacts["authSecretName"] = "huggingface-token"

    decode["replicas"] = runtime.replicas
    decode.setdefault("parallelism", {})
    decode["parallelism"]["tensor"] = runtime.tensor_parallelism

    env = container.setdefault("env", [])
    env = [entry for entry in env if entry.get("name") != "CUDA_VISIBLE_DEVICES"]
    env.append(
        {
            "name": "CUDA_VISIBLE_DEVICES",
            "value": _cuda_visible_devices(runtime.tensor_parallelism),
        }
    )
    container["env"] = env

    container["image"] = runtime.image
    container["modelCommand"] = "custom"
    container["command"] = ["vllm", "serve"]

    args: list[str] = [
        _model_mount_path(plan),
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(runtime.tensor_parallelism),
        "--served-model-name",
        plan.model.name,
    ]
    if plan.model.revision and plan.model.revision != "main":
        args.extend(["--revision", plan.model.revision])
    args.extend(runtime.vllm_args)
    container["args"] = args

    container.setdefault("volumeMounts", [])
    for volume_mount in container["volumeMounts"]:
        if volume_mount.get("name") == "models-storage":
            volume_mount["mountPath"] = storage.mount_path
            volume_mount["readOnly"] = True

    values_file.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return values


def _apply_pipeline_labels(
    values: dict[str, Any], release_name: str, pipeline_run_name: str
) -> None:
    if not pipeline_run_name:
        return
    decode = values.setdefault("decode", {})
    template = decode.setdefault("template", {})
    metadata = template.setdefault("metadata", {})
    labels = metadata.setdefault("labels", {})
    labels["tekton.dev/pipelineRun"] = pipeline_run_name
    labels["benchflow/managed-by"] = "pipeline"
    labels["benchflow/release"] = release_name


def _capture_manifests(
    guide_dir: Path, manifests_dir: Path, namespace: str, env: dict[str, str]
) -> None:
    manifests_dir.mkdir(parents=True, exist_ok=True)
    rendered_dir = manifests_dir / "rendered"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    rendered_path = rendered_dir / "manifests.yaml"
    values_path = guide_dir / "ms-inference-scheduling" / "values.yaml"

    result = run_command(
        ["helmfile", "-e", env["HELMFILE_ENVIRONMENT"], "template", "-n", namespace],
        cwd=guide_dir,
        env=env,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0 and result.stdout:
        rendered_path.write_text(result.stdout, encoding="utf-8")
    shutil.copy2(values_path, rendered_dir / "values.yaml")


def _pods_ready(
    namespace: str, selector: str, kubectl_cmd: str
) -> tuple[bool, int, int]:
    payload = run_json_command(
        [kubectl_cmd, "get", "pods", "-n", namespace, "-l", selector, "-o", "json"],
    )
    items = payload.get("items", [])
    total = len(items)
    ready = 0
    for item in items:
        statuses = item.get("status", {}).get("containerStatuses") or []
        if statuses and all(bool(status.get("ready")) for status in statuses):
            ready += 1
    return total > 0 and ready == total, ready, total


def _gateway_exists(namespace: str, release_name: str, kubectl_cmd: str) -> bool:
    result = run_command(
        [
            kubectl_cmd,
            "get",
            "gateway",
            f"infra-{release_name}-inference-gateway",
            "-n",
            namespace,
            "-o",
            "name",
        ],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _verify_deployment(plan: ResolvedRunPlan, timeout_seconds: int) -> None:
    kubectl_cmd = require_any_command("oc", "kubectl")
    namespace = plan.deployment.namespace
    release_name = plan.deployment.release_name
    deadline = time.time() + timeout_seconds
    last_snapshot: tuple[int, int, bool] | None = None

    step(
        f"Waiting for llm-d deployment {release_name} in namespace {namespace} to become ready"
    )

    while time.time() < deadline:
        epp_ready, epp_ready_count, epp_total = _pods_ready(
            namespace, f"inferencepool=gaie-{release_name}-epp", kubectl_cmd
        )
        ms_ready, ms_ready_count, ms_total = _pods_ready(
            namespace, "llm-d.ai/inference-serving=true", kubectl_cmd
        )
        if not ms_ready:
            ms_ready, ms_ready_count, ms_total = _pods_ready(
                namespace, "llm-d.ai/inferenceServing=true", kubectl_cmd
            )
        gateway_ready = _gateway_exists(namespace, release_name, kubectl_cmd)
        snapshot = (epp_ready_count, ms_ready_count, gateway_ready)

        if snapshot != last_snapshot:
            detail(
                f"EPP pods ready: {epp_ready_count}/{epp_total}, "
                f"model-service pods ready: {ms_ready_count}/{ms_total}, "
                f"gateway present: {'yes' if gateway_ready else 'no'}"
            )
            last_snapshot = snapshot

        if epp_ready and ms_ready and gateway_ready:
            success(
                f"llm-d deployment {release_name} is ready "
                f"(EPP {epp_ready_count}/{epp_total}, model-service {ms_ready_count}/{ms_total})"
            )
            return
        time.sleep(10)

    raise CommandError(
        f"timed out waiting for llm-d deployment {release_name} to become ready"
    )


def deploy_llmd(
    plan: ResolvedRunPlan,
    *,
    workspace_dir: Path | None = None,
    manifests_dir: Path | None = None,
    pipeline_run_name: str = "",
    skip_if_exists: bool = True,
    verify: bool = True,
    verify_timeout_seconds: int = 900,
) -> Path:
    require_command("helm")
    require_command("helmfile")
    require_any_command("oc", "kubectl")

    if skip_if_exists and _release_exists(
        plan.deployment.namespace, plan.deployment.release_name
    ):
        success(
            f"Skipping deploy; Helm release ms-{plan.deployment.release_name} already exists"
        )
        return workspace_dir.resolve() if workspace_dir else Path.cwd()

    created_tempdir = workspace_dir is None
    checkout_root = (
        workspace_dir
        if workspace_dir is not None
        else Path(tempfile.mkdtemp(prefix="benchflow-llmd-"))
    )
    checkout_dir = checkout_root / "llm-d-repo"
    step(
        f"Cloning llm-d guide from {plan.deployment.repo_url} at {plan.deployment.repo_ref}"
    )
    clone_repo(
        url=plan.deployment.repo_url,
        revision=plan.deployment.repo_ref,
        output_dir=checkout_dir,
        delete_existing=True,
    )

    guide_dir = checkout_dir / "guides" / "inference-scheduling"
    values_file = guide_dir / "ms-inference-scheduling" / "values.yaml"
    if not values_file.exists():
        raise CommandError(f"expected llm-d guide file not found: {values_file}")

    step(f"Patching llm-d guide values for release {plan.deployment.release_name}")
    detail(f"Guide directory: {guide_dir}")
    values = _patch_values(plan, values_file)
    _apply_pipeline_labels(values, plan.deployment.release_name, pipeline_run_name)
    values_file.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")

    env = {
        **os.environ,
        "HOME": "/tmp",
        "HELM_CACHE_HOME": "/tmp/.cache/helm",
        "HELM_CONFIG_HOME": "/tmp/.config/helm",
        "HELM_DATA_HOME": "/tmp/.local/share/helm",
        "HELM_PLUGINS": "/tmp/.local/share/helm/plugins",
        "RELEASE_NAME_POSTFIX": plan.deployment.release_name,
        "HELMFILE_ENVIRONMENT": _environment_name(plan),
    }

    step("Initializing helmfile plugins")
    run_command(["helmfile", "init", "--force"], cwd=guide_dir, env=env)

    if manifests_dir is not None:
        step(f"Capturing rendered manifests in {manifests_dir}")
        _capture_manifests(guide_dir, manifests_dir, plan.deployment.namespace, env)

    step(
        f"Applying helmfile environment {env['HELMFILE_ENVIRONMENT']} "
        f"into namespace {plan.deployment.namespace}"
    )
    run_command(
        [
            "helmfile",
            "-e",
            env["HELMFILE_ENVIRONMENT"],
            "apply",
            "-n",
            plan.deployment.namespace,
            "--skip-deps=false",
            "--suppress-secrets",
        ],
        cwd=guide_dir,
        env=env,
    )
    success(
        f"Applied llm-d releases for {plan.deployment.release_name} in namespace "
        f"{plan.deployment.namespace}"
    )

    if verify:
        _verify_deployment(plan, verify_timeout_seconds)

    if created_tempdir:
        return checkout_root
    return checkout_root.resolve()
