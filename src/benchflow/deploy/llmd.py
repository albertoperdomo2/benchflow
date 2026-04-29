from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

from ..assets import render_jinja_text
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

_LLMD_INFERENCE_SERVING_LABEL = "llm-d.ai/inferenceServing"
_LLMD_MODEL_LABEL = "llm-d.ai/model"
_BENCHFLOW_RELEASE_LABEL = "benchflow.io/release"
_BENCHFLOW_EPP_CONFIG_FILE = "benchflow-epp-config.yaml"


def _llmd_guide_layout(plan: ResolvedRunPlan) -> dict[str, str]:
    mode = str(plan.deployment.mode or "").strip()
    if mode == "precise-prefix-cache":
        return {
            "guide_dirname": "precise-prefix-cache-aware",
            "model_values_relpath": "ms-kv-events/values.yaml",
            "scheduler_values_relpath": "gaie-kv-events/values.yaml",
        }
    return {
        "guide_dirname": "inference-scheduling",
        "model_values_relpath": "ms-inference-scheduling/values.yaml",
        "scheduler_values_relpath": "gaie-inference-scheduling/values.yaml",
    }


def _release_exists(namespace: str, release_name: str) -> bool:
    helm_json = run_json_command(["helm", "list", "-n", namespace, "-o", "json"])
    return any(entry.get("name") == f"ms-{release_name}" for entry in helm_json)


def _gaie_service_account_name(release_name: str) -> str:
    return f"gaie-{release_name}-epp"


def _gaie_rbac_name(release_name: str) -> str:
    suffix = hashlib.sha1(release_name.encode("utf-8")).hexdigest()[:10]
    return f"benchflow-gaie-epp-rbac-{suffix}"


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


def _apply_runtime_resources(container: dict[str, Any], plan: ResolvedRunPlan) -> None:
    runtime_resources = plan.deployment.runtime.resources
    if not runtime_resources.requests and not runtime_resources.limits:
        return

    resources = container.setdefault("resources", {})
    requests = resources.setdefault("requests", {})
    limits = resources.setdefault("limits", {})
    requests.update(runtime_resources.requests)
    limits.update(runtime_resources.limits)


def _release_match_labels(release_name: str) -> dict[str, str]:
    return {
        _LLMD_INFERENCE_SERVING_LABEL: "true",
        _LLMD_MODEL_LABEL: release_name,
    }


def _llmd_inference_pool_backend_group(repo_ref: str) -> str:
    # Temporary compatibility fix: llm-d v0.4.x inference-scheduling still
    # references the legacy x-k8s InferencePool API group, while newer refs such
    # as v0.6.0 route to the promoted inference.networking.k8s.io group.
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)(?:[-+].*)?", repo_ref.strip())
    if match is None:
        return "inference.networking.k8s.io"
    version = tuple(int(part) for part in match.groups())
    if version <= (0, 4, 0):
        return "inference.networking.x-k8s.io"
    return "inference.networking.k8s.io"


def _split_image_reference(image: str) -> tuple[str, str, str]:
    trimmed = image.strip()
    if not trimmed:
        raise CommandError("scheduler image override is empty")
    if "@" in trimmed:
        raise CommandError(
            "scheduler image override must use a tag, not a digest, because the "
            "llm-d guide expects separate hub/name/tag values"
        )
    last_slash = trimmed.rfind("/")
    last_colon = trimmed.rfind(":")
    if last_colon <= last_slash:
        raise CommandError(
            "scheduler image override must be a fully qualified image reference "
            "in the form <registry>/<path>/<name>:<tag>"
        )
    name_part = trimmed[:last_colon]
    tag = trimmed[last_colon + 1 :]
    hub, _, name = name_part.rpartition("/")
    if not hub or not name or not tag:
        raise CommandError(
            "scheduler image override must be a fully qualified image reference "
            "in the form <registry>/<path>/<name>:<tag>"
        )
    return hub, name, tag


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
    labels = model_artifacts.setdefault("labels", {})
    if not isinstance(labels, dict):
        labels = {}
        model_artifacts["labels"] = labels
    labels.update(_release_match_labels(plan.deployment.release_name))

    decode["replicas"] = runtime.replicas
    decode.setdefault("parallelism", {})
    decode["parallelism"]["tensor"] = runtime.tensor_parallelism
    if runtime.node_selector:
        decode["nodeSelector"] = dict(runtime.node_selector)
    if runtime.affinity:
        decode["affinity"] = dict(runtime.affinity)
    if runtime.tolerations:
        decode["tolerations"] = list(runtime.tolerations)

    env = container.setdefault("env", [])
    managed_env_names = {"CUDA_VISIBLE_DEVICES", *runtime.env.keys()}
    env = [entry for entry in env if entry.get("name") not in managed_env_names]
    env.append(
        {
            "name": "CUDA_VISIBLE_DEVICES",
            "value": _cuda_visible_devices(runtime.tensor_parallelism),
        }
    )
    for key, value in sorted(runtime.env.items()):
        env.append({"name": key, "value": value})
    container["env"] = env

    if runtime.image:
        container["image"] = runtime.image
    _apply_runtime_resources(container, plan)
    if plan.deployment.mode == "precise-prefix-cache":
        existing_args = list(container.get("args") or [])
        kv_events_config: dict[str, Any] | None = None
        preserved_args: list[str] = []
        index = 0
        while index < len(existing_args):
            item = str(existing_args[index])
            if item == "--kv-events-config" and index + 1 < len(existing_args):
                try:
                    kv_events_config = json.loads(str(existing_args[index + 1]))
                except json.JSONDecodeError:
                    kv_events_config = None
                index += 2
                continue
            preserved_args.append(item)
            index += 1

        if kv_events_config is None:
            kv_events_config = {
                "enable_kv_cache_events": True,
                "publisher": "zmq",
                "endpoint": (
                    "tcp://gaie-$(GAIE_RELEASE_NAME_POSTFIX)-epp."
                    "$(NAMESPACE).svc.cluster.local:5557"
                ),
                "topic": f"kv@$(POD_IP):{port}@{plan.model.name}",
            }
        else:
            kv_events_config["endpoint"] = (
                "tcp://gaie-$(GAIE_RELEASE_NAME_POSTFIX)-epp."
                "$(NAMESPACE).svc.cluster.local:5557"
            )
            kv_events_config["topic"] = f"kv@$(POD_IP):{port}@{plan.model.name}"

        args = [
            _model_mount_path(plan),
            "--port",
            str(port),
            "--tensor-parallel-size",
            str(runtime.tensor_parallelism),
            "--served-model-name",
            plan.model.name,
            *preserved_args,
            "--kv-events-config",
            json.dumps(kv_events_config, separators=(",", ":")),
            *runtime.vllm_args,
        ]
        container["modelCommand"] = "custom"
        container["command"] = ["vllm", "serve"]
        container["args"] = args
    else:
        container["modelCommand"] = "custom"
        container["command"] = ["vllm", "serve"]

        args = [
            _model_mount_path(plan),
            "--port",
            str(port),
            "--tensor-parallel-size",
            str(runtime.tensor_parallelism),
            "--served-model-name",
            plan.model.name,
            *runtime.vllm_args,
        ]
        container["args"] = args

    container.setdefault("volumeMounts", [])
    for volume_mount in container["volumeMounts"]:
        if volume_mount.get("name") == "models-storage":
            volume_mount["mountPath"] = storage.mount_path
            volume_mount["readOnly"] = True

    values_file.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return values


def _llmd_custom_epp_config_context(plan: ResolvedRunPlan) -> dict[str, Any]:
    runtime = plan.deployment.runtime
    return {
        "release_name": plan.deployment.release_name,
        "namespace": plan.deployment.namespace,
        "mode": plan.deployment.mode,
        "repo_ref": plan.deployment.repo_ref,
        "gateway": plan.deployment.gateway,
        "scheduler_image": plan.deployment.scheduler_image,
        "model_name": plan.model.name,
        "model_uri": _model_uri(plan),
        "model_mount_path": _model_mount_path(plan),
        "replicas": runtime.replicas,
        "tensor_parallelism": runtime.tensor_parallelism,
        "runtime_image": runtime.image,
        "runtime_env": runtime.env,
        "runtime_vllm_args": runtime.vllm_args,
        "runtime_node_selector": runtime.node_selector,
        "runtime_affinity": runtime.affinity,
        "runtime_tolerations": runtime.tolerations,
        "runtime_image_pull_secrets": runtime.image_pull_secrets,
        "runtime_resources": runtime.resources,
    }


def _render_llmd_custom_epp_config(plan: ResolvedRunPlan) -> str:
    raw_config = plan.deployment.options.get("epp_config")
    if raw_config is None or str(raw_config).strip() == "":
        return ""
    if not isinstance(raw_config, str):
        raise CommandError(
            "deployment profile options.epp_config must be a YAML string"
        )

    rendered = render_jinja_text(
        raw_config, _llmd_custom_epp_config_context(plan)
    ).strip()
    try:
        parsed = yaml.safe_load(rendered)
    except yaml.YAMLError as exc:
        raise CommandError(
            "deployment profile options.epp_config must render valid YAML"
        ) from exc
    if not isinstance(parsed, dict):
        raise CommandError(
            "deployment profile options.epp_config must render to a YAML mapping"
        )
    if parsed.get("kind") != "EndpointPickerConfig":
        raise CommandError(
            "deployment profile options.epp_config must render an EndpointPickerConfig"
        )
    return rendered + "\n"


def _llmd_epp_verbosity(plan: ResolvedRunPlan) -> int | None:
    raw_value = plan.deployment.options.get("epp_verbosity")
    if raw_value is None or str(raw_value).strip() == "":
        return None
    if isinstance(raw_value, bool):
        raise CommandError(
            "deployment profile options.epp_verbosity must be an integer"
        )
    try:
        verbosity = int(str(raw_value).strip())
    except ValueError as exc:
        raise CommandError(
            "deployment profile options.epp_verbosity must be an integer"
        ) from exc
    if verbosity < 0:
        raise CommandError(
            "deployment profile options.epp_verbosity must be greater than or "
            "equal to 0"
        )
    return verbosity


def _patch_scheduler_values(plan: ResolvedRunPlan, values_file: Path) -> None:
    values = yaml.safe_load(values_file.read_text(encoding="utf-8")) or {}
    inference_extension = values.setdefault("inferenceExtension", {})
    monitoring = inference_extension.setdefault("monitoring", {})
    secret_name = f"{plan.deployment.release_name}-gateway-sa-metrics-reader-secret"

    # Older guide values used monitoring.secret.name, while the v1.2
    # inferencepool chart reads monitoring.prometheus.auth.secretName.
    secret = monitoring.setdefault("secret", {})
    secret["name"] = secret_name
    prometheus = monitoring.setdefault("prometheus", {})
    auth = prometheus.setdefault("auth", {})
    auth["secretName"] = secret_name
    for env_entry in inference_extension.get("env", []) or []:
        if str(env_entry.get("name") or "") == "HF_TOKEN" and isinstance(
            env_entry.get("valueFrom"), dict
        ):
            secret_ref = (env_entry.get("valueFrom", {}) or {}).get(
                "secretKeyRef", {}
            ) or {}
            secret_ref["name"] = "huggingface-token"
            env_entry["valueFrom"]["secretKeyRef"] = secret_ref
    inference_pool = values.setdefault("inferencePool", {})
    model_servers = inference_pool.setdefault("modelServers", {})
    match_labels = model_servers.setdefault("matchLabels", {})
    if not isinstance(match_labels, dict):
        match_labels = {}
        model_servers["matchLabels"] = match_labels
    match_labels.update(_release_match_labels(plan.deployment.release_name))

    epp_verbosity = _llmd_epp_verbosity(plan)
    if epp_verbosity is not None:
        flags = inference_extension.get("flags")
        if not isinstance(flags, dict):
            flags = {}
            inference_extension["flags"] = flags
        flags["v"] = epp_verbosity

    custom_epp_config = _render_llmd_custom_epp_config(plan)
    if custom_epp_config:
        inference_extension["pluginsConfigFile"] = _BENCHFLOW_EPP_CONFIG_FILE
        plugins_custom_config = inference_extension.setdefault(
            "pluginsCustomConfig", {}
        )
        if not isinstance(plugins_custom_config, dict):
            plugins_custom_config = {}
            inference_extension["pluginsCustomConfig"] = plugins_custom_config
        plugins_custom_config[_BENCHFLOW_EPP_CONFIG_FILE] = custom_epp_config
    elif plan.deployment.mode == "precise-prefix-cache":
        plugins_config_name = str(
            inference_extension.get("pluginsConfigFile") or ""
        ).strip()
        plugins_custom_config = inference_extension.setdefault(
            "pluginsCustomConfig", {}
        )
        raw_plugins_config = str(plugins_custom_config.get(plugins_config_name) or "")
        if raw_plugins_config:
            plugins_payload = yaml.safe_load(raw_plugins_config) or {}
            for plugin in plugins_payload.get("plugins", []) or []:
                if str(plugin.get("type") or "") == "tokenizer":
                    parameters = plugin.setdefault("parameters", {})
                    parameters["modelName"] = plan.model.name
                if str(plugin.get("type") or "") == "precise-prefix-cache-scorer":
                    parameters = plugin.setdefault("parameters", {})
                    indexer_config = parameters.setdefault("indexerConfig", {})
                    tokenizers_pool = indexer_config.setdefault(
                        "tokenizersPoolConfig", {}
                    )
                    tokenizers_pool["modelName"] = plan.model.name
            plugins_custom_config[plugins_config_name] = yaml.safe_dump(
                plugins_payload, sort_keys=False
            )

    if plan.deployment.scheduler_image:
        image = inference_extension.setdefault("image", {})
        hub, name, tag = _split_image_reference(plan.deployment.scheduler_image)
        image["hub"] = hub
        image["name"] = name
        image["tag"] = tag
    values_file.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")


def _apply_pipeline_labels(
    values: dict[str, Any],
    release_name: str,
    execution_name: str,
    *,
    execution_backend: str,
) -> None:
    if not execution_name:
        return
    decode = values.setdefault("decode", {})
    template = decode.setdefault("template", {})
    metadata = template.setdefault("metadata", {})
    labels = metadata.setdefault("labels", {})
    labels["benchflow.io/execution-run"] = execution_name
    labels["benchflow.io/execution-backend"] = execution_backend
    labels["benchflow/managed-by"] = "pipeline"
    labels["benchflow/release"] = release_name
    labels[_BENCHFLOW_RELEASE_LABEL] = release_name


def _capture_manifests(
    guide_dir: Path,
    manifests_dir: Path,
    namespace: str,
    env: dict[str, str],
    *,
    model_values_relpath: str,
) -> None:
    manifests_dir.mkdir(parents=True, exist_ok=True)
    rendered_dir = manifests_dir / "rendered"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    rendered_path = rendered_dir / "manifests.yaml"
    values_path = guide_dir / Path(model_values_relpath)

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


def _create_httproute(plan: ResolvedRunPlan, kubectl_cmd: str) -> None:
    inference_pool_backend_group = _llmd_inference_pool_backend_group(
        plan.deployment.repo_ref
    )
    route = {
        "apiVersion": "gateway.networking.k8s.io/v1",
        "kind": "HTTPRoute",
        "metadata": {
            "name": f"llm-d-{plan.deployment.release_name}",
            "namespace": plan.deployment.namespace,
            "labels": {
                "app.kubernetes.io/name": "benchflow",
                "benchflow.io/platform": "llm-d",
                "benchflow.io/release": plan.deployment.release_name,
            },
        },
        "spec": {
            "parentRefs": [
                {
                    "group": "gateway.networking.k8s.io",
                    "kind": "Gateway",
                    "name": f"infra-{plan.deployment.release_name}-inference-gateway",
                }
            ],
            "rules": [
                {
                    "backendRefs": [
                        {
                            "group": inference_pool_backend_group,
                            "kind": "InferencePool",
                            "name": f"gaie-{plan.deployment.release_name}",
                            "port": 8000,
                            "weight": 1,
                        }
                    ],
                    "matches": [{"path": {"type": "PathPrefix", "value": "/"}}],
                }
            ],
        },
    }
    run_command(
        [kubectl_cmd, "apply", "-f", "-"],
        input_text=yaml.safe_dump(route, sort_keys=False),
    )


def _ensure_gaie_rbac(plan: ResolvedRunPlan, kubectl_cmd: str) -> None:
    namespace = plan.deployment.namespace
    release_name = plan.deployment.release_name
    resource_name = _gaie_rbac_name(release_name)
    service_account_name = _gaie_service_account_name(release_name)
    document = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "Role",
        "metadata": {
            "name": resource_name,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/name": "benchflow",
                "benchflow.io/platform": "llm-d",
                "benchflow.io/release": release_name,
                "benchflow.io/managed-by": "benchflow",
            },
        },
        "rules": [
            {
                "apiGroups": ["inference.networking.x-k8s.io"],
                "resources": ["inferencemodelrewrites"],
                "verbs": ["get", "list", "watch"],
            }
        ],
    }
    binding = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "RoleBinding",
        "metadata": {
            "name": resource_name,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/name": "benchflow",
                "benchflow.io/platform": "llm-d",
                "benchflow.io/release": release_name,
                "benchflow.io/managed-by": "benchflow",
            },
        },
        "subjects": [
            {
                "kind": "ServiceAccount",
                "name": service_account_name,
                "namespace": namespace,
            }
        ],
        "roleRef": {
            "apiGroup": "rbac.authorization.k8s.io",
            "kind": "Role",
            "name": resource_name,
        },
    }
    step(f"Applying supplemental GAIE RBAC for {service_account_name}")
    run_command(
        [kubectl_cmd, "apply", "-f", "-"],
        input_text="---\n".join(
            [
                yaml.safe_dump(document, sort_keys=False),
                yaml.safe_dump(binding, sort_keys=False),
            ]
        ),
    )


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


def _httproute_exists(namespace: str, release_name: str, kubectl_cmd: str) -> bool:
    result = run_command(
        [
            kubectl_cmd,
            "get",
            "httproute",
            f"llm-d-{release_name}",
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
    last_snapshot: tuple[int, int, bool, bool] | None = None

    step(
        f"Waiting for llm-d deployment {release_name} in namespace {namespace} to become ready"
    )

    while time.time() < deadline:
        epp_ready, epp_ready_count, epp_total = _pods_ready(
            namespace, f"inferencepool=gaie-{release_name}-epp", kubectl_cmd
        )
        ms_ready, ms_ready_count, ms_total = _pods_ready(
            namespace,
            f"{_LLMD_INFERENCE_SERVING_LABEL}=true,{_LLMD_MODEL_LABEL}={release_name}",
            kubectl_cmd,
        )
        gateway_ready = _gateway_exists(namespace, release_name, kubectl_cmd)
        httproute_ready = _httproute_exists(namespace, release_name, kubectl_cmd)
        snapshot = (epp_ready_count, ms_ready_count, gateway_ready, httproute_ready)

        if snapshot != last_snapshot:
            detail(
                f"EPP pods ready: {epp_ready_count}/{epp_total}, "
                f"model-service pods ready: {ms_ready_count}/{ms_total}, "
                f"gateway present: {'yes' if gateway_ready else 'no'}, "
                f"httproute present: {'yes' if httproute_ready else 'no'}"
            )
            last_snapshot = snapshot

        if epp_ready and ms_ready and gateway_ready and httproute_ready:
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
    execution_name: str = "",
    skip_if_exists: bool = True,
    verify: bool = True,
    verify_timeout_seconds: int = 900,
) -> Path:
    require_command("helm")
    require_command("helmfile")
    kubectl_cmd = require_any_command("oc", "kubectl")

    if skip_if_exists and _release_exists(
        plan.deployment.namespace, plan.deployment.release_name
    ):
        _ensure_gaie_rbac(plan, kubectl_cmd)
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

    guide_layout = _llmd_guide_layout(plan)
    guide_dir = checkout_dir / "guides" / guide_layout["guide_dirname"]
    values_file = guide_dir / Path(guide_layout["model_values_relpath"])
    scheduler_values_file = guide_dir / Path(guide_layout["scheduler_values_relpath"])
    if not values_file.exists():
        raise CommandError(f"expected llm-d guide file not found: {values_file}")
    if not scheduler_values_file.exists():
        raise CommandError(
            f"expected llm-d guide file not found: {scheduler_values_file}"
        )

    step(f"Patching llm-d guide values for release {plan.deployment.release_name}")
    detail(f"Guide directory: {guide_dir}")
    values = _patch_values(plan, values_file)
    _patch_scheduler_values(plan, scheduler_values_file)
    _apply_pipeline_labels(
        values,
        plan.deployment.release_name,
        execution_name,
        execution_backend="tekton",
    )
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
        _capture_manifests(
            guide_dir,
            manifests_dir,
            plan.deployment.namespace,
            env,
            model_values_relpath=guide_layout["model_values_relpath"],
        )

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
    _ensure_gaie_rbac(plan, kubectl_cmd)
    step(f"Applying HTTPRoute llm-d-{plan.deployment.release_name}")
    _create_httproute(plan, kubectl_cmd)
    success(
        f"Applied llm-d releases for {plan.deployment.release_name} in namespace "
        f"{plan.deployment.namespace}"
    )

    if verify:
        _verify_deployment(plan, verify_timeout_seconds)

    if created_tempdir:
        return checkout_root
    return checkout_root.resolve()
