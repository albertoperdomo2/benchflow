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
from ..models import ResolvedRunPlan, sanitize_name
from ..platform_state import (
    load_cluster_platform_state,
    persist_cluster_platform_state,
    setup_key_for_plan,
)
from ..repository import clone_repo
from ..storage_offloading import (
    STORAGE_OFFLOADING_TYPE_HOST_PATH,
    STORAGE_OFFLOADING_TYPE_PVC,
    storage_offloading_config as _storage_offloading_config,
)
from ..ui import detail, step, success

_LLMD_INFERENCE_SERVING_LABEL = "llm-d.ai/inferenceServing"
_LLMD_MODEL_LABEL = "llm-d.ai/model"
_BENCHFLOW_GUIDE_LABEL = "llm-d.ai/guide"
_BENCHFLOW_RELEASE_LABEL = "benchflow.io/release"
_BENCHFLOW_EPP_CONFIG_FILE = "benchflow-epp-config.yaml"
_STORAGE_OFFLOADING_VOLUME_NAME = "storage-offloading"
_STORAGE_OFFLOADING_SERVICE_FILE = "benchflow-storage-events-service.yaml"
_MODELSERVER_PODMONITOR_FILE = "benchflow-modelserver-podmonitor.yaml"


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


def _llmd_recipe_layout_available(checkout_dir: Path) -> bool:
    return _llmd_recipe_scheduler_layout_available(
        checkout_dir
    ) or _llmd_recipe_router_layout_available(checkout_dir)


def _llmd_recipe_scheduler_layout_available(checkout_dir: Path) -> bool:
    return (
        checkout_dir / "guides" / "recipes" / "scheduler" / "base.values.yaml"
    ).exists()


def _llmd_recipe_router_layout_available(checkout_dir: Path) -> bool:
    return (
        checkout_dir / "guides" / "recipes" / "router" / "base.values.yaml"
    ).exists()


def _record_llmd_repo_head(
    plan: ResolvedRunPlan, kubectl_cmd: str, repo_head: str
) -> None:
    if str(plan.deployment.repo_ref or "").strip() != "main":
        return
    if not repo_head:
        return

    cluster_state = load_cluster_platform_state(kubectl_cmd, plan.deployment.namespace)
    setup_state = dict(cluster_state.get("setup_state") or {})
    if setup_state and setup_state.get("platform") not in {"llm-d", None}:
        return

    setup_state = {
        **setup_state,
        "platform": "llm-d",
        "repo_url": plan.deployment.repo_url,
        "repo_ref": plan.deployment.repo_ref,
        "repo_head": repo_head,
        "gateway": plan.deployment.gateway,
    }
    persist_cluster_platform_state(
        kubectl_cmd,
        plan.deployment.namespace,
        {
            "installed_key": str(cluster_state.get("installed_key") or "").strip()
            or setup_key_for_plan(plan),
            "setup_state": setup_state,
        },
    )


def _llmd_recipe_guide_name(plan: ResolvedRunPlan, *, router_chart: bool) -> str:
    mode = str(plan.deployment.mode or "").strip()
    if mode == "precise-prefix-cache":
        return (
            "precise-prefix-cache-routing"
            if router_chart
            else "precise-prefix-cache-aware"
        )
    return "optimized-baseline"


def _llmd_recipe_scheduler_values_path(
    checkout_dir: Path, plan: ResolvedRunPlan, *, router_chart: bool
) -> Path:
    guide_name = _llmd_recipe_guide_name(plan, router_chart=router_chart)
    subdir = "router" if router_chart else "scheduler"
    return checkout_dir / "guides" / guide_name / subdir / f"{guide_name}.values.yaml"


def _llmd_recipe_modelserver_overlay_dir(
    checkout_dir: Path, plan: ResolvedRunPlan, *, router_chart: bool
) -> Path:
    guide_name = _llmd_recipe_guide_name(plan, router_chart=router_chart)
    backend_dir = _llmd_recipe_modelserver_backend_dir(plan)
    provider = (
        str(plan.deployment.options.get("infra_provider") or "base").strip().lower()
    )
    if backend_dir.startswith("gpu/"):
        if provider == "gke":
            return (
                checkout_dir
                / "guides"
                / guide_name
                / "modelserver"
                / "gpu"
                / backend_dir.split("/", 1)[1]
                / "gke"
            )
        return (
            checkout_dir
            / "guides"
            / guide_name
            / "modelserver"
            / "gpu"
            / backend_dir.split("/", 1)[1]
            / "base"
        )
    return checkout_dir / "guides" / guide_name / "modelserver" / backend_dir


def _llmd_recipe_gateway_dir(checkout_dir: Path) -> Path:
    return checkout_dir / "guides" / "recipes" / "gateway" / "istio"


def _llmd_recipe_scheduler_release_name(plan: ResolvedRunPlan) -> str:
    return f"gaie-{plan.deployment.release_name}"


def _llmd_recipe_scheduler_release_name_for(release_name: str) -> str:
    return f"gaie-{release_name}"


def _llmd_router_chart_label_key(gateway_mode: str) -> str:
    return (
        "llm-d-router-standalone"
        if gateway_mode == "standalone"
        else "llm-d-router-gateway"
    )


def _llmd_router_epp_selector(release_name: str, gateway_mode: str) -> str:
    label_key = _llmd_router_chart_label_key(gateway_mode)
    epp_name = f"{_llmd_recipe_scheduler_release_name_for(release_name)}-epp"
    return f"{label_key}={epp_name}"


def _llmd_router_epp_selectors(release_name: str, gateway_mode: str) -> list[str]:
    epp_name = f"{_llmd_recipe_scheduler_release_name_for(release_name)}-epp"
    selectors = [_llmd_router_epp_selector(release_name, gateway_mode)]
    for label_key in ("llm-d-router-gateway", "llm-d-router-standalone"):
        selector = f"{label_key}={epp_name}"
        if selector not in selectors:
            selectors.append(selector)
    return selectors


def _llmd_recipe_standalone_envoy_configmap_name(plan: ResolvedRunPlan) -> str:
    return f"gaie-{plan.deployment.release_name}-envoy"


def _llmd_recipe_modelserver_backend_dir(plan: ResolvedRunPlan) -> str:
    accelerator = (
        str(
            plan.mlflow.tags.get("accelerator")
            or plan.deployment.options.get("accelerator")
            or ""
        )
        .strip()
        .upper()
    )
    if not accelerator:
        return "gpu/vllm"
    if "AMD" in accelerator or accelerator.startswith("MI"):
        return "amd/vllm"
    if "HPU" in accelerator or "GAUDI" in accelerator:
        return "hpu/vllm"
    if "XPU" in accelerator or "INTEL" in accelerator:
        return "xpu/vllm"
    if "CPU" in accelerator:
        return "cpu/vllm"
    if "TPU" in accelerator:
        if "V6" in accelerator:
            return "tpu-v6/vllm"
        return "tpu-v7/vllm"
    return "gpu/vllm"


def _llmd_model_label_value(plan: ResolvedRunPlan) -> str:
    # Kubernetes label values must be DNS-like and cannot contain the raw model
    # identifier when it includes characters such as "/".
    return sanitize_name(plan.model.resolved_name(), max_length=63)


def _llmd_inference_model_api_group(repo_ref: str) -> str:
    # Temporary compatibility fix: llm-d v0.4.x still uses the legacy x-k8s
    # API group, while newer refs route through the promoted
    # inference.networking.k8s.io group.
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)(?:[-+].*)?", repo_ref.strip())
    if match is None:
        return "inference.networking.k8s.io"
    version = tuple(int(part) for part in match.groups())
    if version <= (0, 4, 0):
        return "inference.networking.x-k8s.io"
    return "inference.networking.k8s.io"


def _release_exists(namespace: str, release_name: str) -> bool:
    helm_json = run_json_command(["helm", "list", "-n", namespace, "-o", "json"])
    expected = {f"ms-{release_name}", f"gaie-{release_name}"}
    return any(entry.get("name") in expected for entry in helm_json)


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


def _recipe_modelserver_podmonitor_manifest(
    plan: ResolvedRunPlan, guide_name: str
) -> dict[str, Any]:
    release_name = plan.deployment.release_name
    return {
        "apiVersion": "monitoring.coreos.com/v1",
        "kind": "PodMonitor",
        "metadata": {
            "name": "modelserver",
            "labels": {
                "app.kubernetes.io/name": "benchflow",
                "benchflow.io/platform": "llm-d",
                _BENCHFLOW_RELEASE_LABEL: release_name,
            },
        },
        "spec": {
            "selector": {
                "matchLabels": {
                    _BENCHFLOW_RELEASE_LABEL: release_name,
                    _BENCHFLOW_GUIDE_LABEL: guide_name,
                }
            },
            "podMetricsEndpoints": [
                {
                    "port": "modelserver",
                    "path": "/metrics",
                    "interval": "30s",
                }
            ],
        },
    }


def _recipe_epp_podmonitor_manifest(
    plan: ResolvedRunPlan, *, router_chart: bool
) -> dict[str, Any]:
    release_name = plan.deployment.release_name
    epp_name = f"{_llmd_recipe_scheduler_release_name(plan)}-epp"
    selector = (
        {
            "matchLabels": {
                _llmd_router_chart_label_key(
                    str(plan.deployment.gateway or "").strip()
                ): epp_name
            }
        }
        if router_chart
        else {"matchLabels": {"inferencepool": epp_name}}
    )
    return {
        "apiVersion": "monitoring.coreos.com/v1",
        "kind": "PodMonitor",
        "metadata": {
            "name": epp_name,
            "namespace": plan.deployment.namespace,
            "labels": {
                "app.kubernetes.io/name": "benchflow",
                "benchflow.io/platform": "llm-d",
                _BENCHFLOW_RELEASE_LABEL: release_name,
            },
        },
        "spec": {
            "selector": selector,
            "podMetricsEndpoints": [
                {
                    "port": "metrics",
                    "path": "/metrics",
                    "interval": "30s",
                }
            ],
        },
    }


def _apply_recipe_epp_podmonitor(
    plan: ResolvedRunPlan, kubectl_cmd: str, *, router_chart: bool
) -> None:
    if router_chart:
        return
    manifest = _recipe_epp_podmonitor_manifest(plan, router_chart=router_chart)
    step(f"Applying llm-d EPP PodMonitor {manifest['metadata']['name']}")
    run_command(
        [kubectl_cmd, "apply", "-f", "-"],
        input_text=yaml.safe_dump(manifest, sort_keys=False),
    )


def _ensure_storage_offloading_pvc(
    plan: ResolvedRunPlan, kubectl_cmd: str, config: dict[str, Any]
) -> None:
    if config["type"] != STORAGE_OFFLOADING_TYPE_PVC:
        return
    pvc_name = config["pvc_name"]
    step(f"Ensuring shared storage offloading PVC {pvc_name}")
    spec: dict[str, Any] = {
        "accessModes": [config["access_mode"]],
        "resources": {"requests": {"storage": config["size"]}},
    }
    if config["storage_class"]:
        spec["storageClassName"] = config["storage_class"]
    manifest = {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {
            "name": pvc_name,
            "namespace": plan.deployment.namespace,
            "labels": {
                "app.kubernetes.io/name": "benchflow",
                "benchflow.io/purpose": "llm-d-storage-offloading",
            },
        },
        "spec": spec,
    }
    run_command(
        [kubectl_cmd, "apply", "-f", "-"],
        input_text=yaml.safe_dump(manifest, sort_keys=False),
    )


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


def _recipe_release_match_labels(release_name: str) -> dict[str, str]:
    return {_BENCHFLOW_RELEASE_LABEL: release_name}


def _append_kustomize_resource(kustomization: dict[str, Any], resource: str) -> None:
    resources = kustomization.get("resources")
    if resources is None:
        resources = []
        kustomization["resources"] = resources
    if not isinstance(resources, list):
        raise CommandError(
            "expected llm-d modelserver kustomization resources to be a list"
        )
    if resource not in resources:
        resources.append(resource)


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


def _image_reference_components(image: str) -> dict[str, str]:
    trimmed = image.strip()
    hub, name, tag = _split_image_reference(trimmed)
    name_part = trimmed.rsplit(":", 1)[0]
    registry, separator, repository = name_part.partition("/")
    if not separator:
        raise CommandError(
            "scheduler image override must be a fully qualified image reference "
            "in the form <registry>/<path>/<name>:<tag>"
        )
    if not registry or not repository or not tag:
        raise CommandError(
            "scheduler image override must be a fully qualified image reference "
            "in the form <registry>/<path>/<name>:<tag>"
        )
    return {
        "hub": hub,
        "name": name,
        "tag": tag,
        "registry": registry,
        "repository": repository,
    }


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
    storage_offloading = _storage_offloading_config(plan) or {}
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
        "storage_offloading": storage_offloading,
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


def _patch_scheduler_values(
    plan: ResolvedRunPlan,
    values_file: Path,
    *,
    recipe_layout: bool,
    router_chart: bool = False,
) -> None:
    values = yaml.safe_load(values_file.read_text(encoding="utf-8")) or {}
    if router_chart:
        router = values.setdefault("router", {})
        epp = router.setdefault("epp", {})
        model_servers = router.setdefault("modelServers", {})
        match_labels = model_servers.setdefault("matchLabels", {})
        if not isinstance(match_labels, dict):
            match_labels = {}
            model_servers["matchLabels"] = match_labels
        match_labels.update(_recipe_release_match_labels(plan.deployment.release_name))

        epp_verbosity = _llmd_epp_verbosity(plan)
        if epp_verbosity is not None:
            flags = epp.get("flags")
            if not isinstance(flags, dict):
                flags = {}
                epp["flags"] = flags
            flags["v"] = epp_verbosity

        custom_epp_config = _render_llmd_custom_epp_config(plan)
        if custom_epp_config:
            epp["pluginsConfigFile"] = _BENCHFLOW_EPP_CONFIG_FILE
            plugins_custom_config = epp.setdefault("pluginsCustomConfig", {})
            if not isinstance(plugins_custom_config, dict):
                plugins_custom_config = {}
                epp["pluginsCustomConfig"] = plugins_custom_config
            plugins_custom_config[_BENCHFLOW_EPP_CONFIG_FILE] = custom_epp_config
        elif plan.deployment.mode == "precise-prefix-cache":
            tokenizer = router.setdefault("tokenizer", {})
            tokenizer["modelName"] = plan.model.name
            plugins_config_name = str(epp.get("pluginsConfigFile") or "").strip()
            plugins_custom_config = epp.setdefault("pluginsCustomConfig", {})
            raw_plugins_config = str(
                plugins_custom_config.get(plugins_config_name) or ""
            )
            if raw_plugins_config:
                plugins_payload = yaml.safe_load(raw_plugins_config) or {}
                for plugin in plugins_payload.get("plugins", []) or []:
                    plugin_type = str(plugin.get("type") or "")
                    parameters = plugin.setdefault("parameters", {})
                    if plugin_type == "token-producer":
                        parameters["modelName"] = plan.model.name
                plugins_custom_config[plugins_config_name] = yaml.safe_dump(
                    plugins_payload, sort_keys=False
                )

        for env_entry in epp.get("env", []) or []:
            if str(env_entry.get("name") or "") != "HF_TOKEN":
                continue
            value_from = env_entry.get("valueFrom")
            if not isinstance(value_from, dict):
                continue
            secret_ref = value_from.get("secretKeyRef")
            if not isinstance(secret_ref, dict):
                continue
            secret_ref["name"] = "huggingface-token"

        if plan.deployment.scheduler_image:
            image = epp.setdefault("image", {})
            components = _image_reference_components(plan.deployment.scheduler_image)
            image.update(
                {
                    "registry": components["registry"],
                    "repository": components["repository"],
                    "tag": components["tag"],
                }
            )
        values_file.write_text(
            yaml.safe_dump(values, sort_keys=False), encoding="utf-8"
        )
        return

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
    if recipe_layout:
        match_labels.update(_recipe_release_match_labels(plan.deployment.release_name))
    else:
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
        components = _image_reference_components(plan.deployment.scheduler_image)
        image.update(
            {
                "hub": components["hub"],
                "name": components["name"],
                "tag": components["tag"],
                "registry": components["registry"],
                "repository": components["repository"],
            }
        )
    values_file.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")


def _recipe_modelserver_container(values: dict[str, Any]) -> dict[str, Any]:
    spec = values.setdefault("spec", {})
    template = spec.setdefault("template", {})
    pod_spec = template.setdefault("spec", {})
    containers = pod_spec.setdefault("containers", [])
    if not containers:
        containers.append({"name": "modelserver"})
    return containers[0]


def _ensure_container_port(container: dict[str, Any], name: str, port: int) -> None:
    ports = container.setdefault("ports", [])
    for port_spec in ports:
        if str(port_spec.get("name") or "") == name:
            port_spec["containerPort"] = port
            port_spec["protocol"] = "TCP"
            return
    ports.append({"name": name, "containerPort": port, "protocol": "TCP"})


def _apply_recipe_storage_offloading(
    *,
    overlay_dir: Path,
    kustomization: dict[str, Any],
    patch: dict[str, Any],
    container: dict[str, Any],
    plan: ResolvedRunPlan,
    guide_name: str,
    config: dict[str, Any],
) -> None:
    storage_events_port = int(config["storage_events_port"])
    storage_type = str(config["type"])
    volume_mounts = container.setdefault("volumeMounts", [])
    if not any(
        str(volume_mount.get("name") or "") == _STORAGE_OFFLOADING_VOLUME_NAME
        for volume_mount in volume_mounts
    ):
        volume_mounts.append(
            {
                "name": _STORAGE_OFFLOADING_VOLUME_NAME,
                "mountPath": config["mount_path"],
                "readOnly": False,
            }
        )

    pod_spec = (
        patch.setdefault("spec", {}).setdefault("template", {}).setdefault("spec", {})
    )
    volumes = pod_spec.setdefault("volumes", [])
    if not any(
        str(volume.get("name") or "") == _STORAGE_OFFLOADING_VOLUME_NAME
        for volume in volumes
    ):
        if storage_type == STORAGE_OFFLOADING_TYPE_PVC:
            source = {"persistentVolumeClaim": {"claimName": config["pvc_name"]}}
        elif storage_type == STORAGE_OFFLOADING_TYPE_HOST_PATH:
            source = {
                "hostPath": {
                    "path": config["host_path"],
                    "type": config["host_path_type"],
                }
            }
        else:
            raise CommandError(f"unsupported storage_offloading type: {storage_type}")
        volumes.append({"name": _STORAGE_OFFLOADING_VOLUME_NAME, **source})

    _ensure_container_port(container, "storage-events", storage_events_port)

    service_path = overlay_dir / _STORAGE_OFFLOADING_SERVICE_FILE
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": "storage-events",
            "labels": {
                "app.kubernetes.io/name": "benchflow",
                _BENCHFLOW_RELEASE_LABEL: plan.deployment.release_name,
            },
        },
        "spec": {
            "selector": {
                _BENCHFLOW_RELEASE_LABEL: plan.deployment.release_name,
                _BENCHFLOW_GUIDE_LABEL: guide_name,
            },
            "ports": [
                {
                    "name": "storage-events",
                    "port": storage_events_port,
                    "targetPort": "storage-events",
                    "protocol": "TCP",
                }
            ],
        },
    }
    service_path.write_text(yaml.safe_dump(service, sort_keys=False), encoding="utf-8")

    _append_kustomize_resource(kustomization, _STORAGE_OFFLOADING_SERVICE_FILE)


def _patch_recipe_modelserver_overlay(
    plan: ResolvedRunPlan, overlay_dir: Path, *, router_chart: bool
) -> None:
    guide_name = _llmd_recipe_guide_name(plan, router_chart=router_chart)
    kustomization_path = overlay_dir / "kustomization.yaml"
    patch_path = overlay_dir / "patch-vllm.yaml"
    storage = plan.deployment.model_storage
    storage_offloading = _storage_offloading_config(plan)

    kustomization = yaml.safe_load(kustomization_path.read_text(encoding="utf-8"))
    if not isinstance(kustomization, dict):
        raise CommandError(
            f"expected llm-d modelserver kustomization not found: {kustomization_path}"
        )
    kustomization["namePrefix"] = f"ms-{plan.deployment.release_name}-"
    images = kustomization.setdefault("images", [])
    if images:
        image = images[0]
        if plan.deployment.runtime.image:
            hub, name, tag = _split_image_reference(plan.deployment.runtime.image)
            image["newName"] = f"{hub}/{name}"
            image["newTag"] = tag
    labels = kustomization.setdefault("labels", [])
    if labels:
        label_entry = labels[0]
        if isinstance(label_entry, dict):
            pairs = label_entry.setdefault("pairs", {})
            if isinstance(pairs, dict):
                pairs.update(
                    {
                        _BENCHFLOW_GUIDE_LABEL: guide_name,
                        _LLMD_MODEL_LABEL: _llmd_model_label_value(plan),
                        _BENCHFLOW_RELEASE_LABEL: plan.deployment.release_name,
                    }
                )
            fields = label_entry.setdefault("fields", [])
            if isinstance(fields, list):
                service_field = {
                    "version": "v1",
                    "kind": "Service",
                    "path": "metadata/labels",
                    "create": True,
                }
                deployment_field = {
                    "version": "apps/v1",
                    "kind": "Deployment",
                    "path": "metadata/labels",
                    "create": True,
                }
                service_account_field = {
                    "version": "v1",
                    "kind": "ServiceAccount",
                    "path": "metadata/labels",
                    "create": True,
                }
                for field in (
                    service_field,
                    deployment_field,
                    service_account_field,
                ):
                    if field not in fields:
                        fields.append(field)
    patch = yaml.safe_load(patch_path.read_text(encoding="utf-8"))
    if not isinstance(patch, dict):
        raise CommandError(f"expected llm-d modelserver patch not found: {patch_path}")
    patch_metadata = patch.setdefault("metadata", {})
    patch_labels = patch_metadata.setdefault("labels", {})
    if not isinstance(patch_labels, dict):
        patch_labels = {}
        patch_metadata["labels"] = patch_labels
    patch_labels.update(
        {
            _BENCHFLOW_GUIDE_LABEL: guide_name,
            _LLMD_MODEL_LABEL: _llmd_model_label_value(plan),
            _BENCHFLOW_RELEASE_LABEL: plan.deployment.release_name,
        }
    )
    runtime = plan.deployment.runtime
    spec = patch.setdefault("spec", {})
    spec["replicas"] = runtime.replicas
    container = _recipe_modelserver_container(patch)
    args = [
        _model_mount_path(plan),
        "--disable-access-log-for-endpoints=/health,/metrics,/v1/models",
        f"--tensor-parallel-size={runtime.tensor_parallelism}",
        "--served-model-name",
        plan.model.name,
    ]
    env: list[dict[str, Any]] = []
    if plan.deployment.mode == "precise-prefix-cache":
        env.extend(
            [
                {
                    "name": "NAMESPACE",
                    "valueFrom": {"fieldRef": {"fieldPath": "metadata.namespace"}},
                },
                {
                    "name": "POD_IP",
                    "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}},
                },
                {"name": "POD_PORT", "value": "8000"},
                {"name": "KV_EVENTS_ENDPOINT", "value": "tcp://*:5556"},
                {"name": "DO_NOT_TRACK", "value": "1"},
            ]
        )
        args.extend(
            [
                "--block-size=64",
                "--kv-events-config",
                json.dumps(
                    {
                        "enable_kv_cache_events": True,
                        "publisher": "zmq",
                        "endpoint": "$(KV_EVENTS_ENDPOINT)",
                        "topic": (f"kv@$(POD_IP):$(POD_PORT)@{plan.model.name}"),
                    },
                    separators=(",", ":"),
                ),
            ]
        )

    managed_env_names = {"CUDA_VISIBLE_DEVICES", *runtime.env.keys()}
    if plan.deployment.mode == "precise-prefix-cache":
        managed_env_names.update(
            {"NAMESPACE", "POD_IP", "POD_PORT", "KV_EVENTS_ENDPOINT", "DO_NOT_TRACK"}
        )
    existing_env = [
        entry
        for entry in list(container.get("env") or [])
        if str(entry.get("name") or "") not in managed_env_names
    ]
    hf_token_rewritten = False
    for env_entry in existing_env:
        if str(env_entry.get("name") or "") != "HF_TOKEN":
            continue
        value_from = env_entry.get("valueFrom")
        if not isinstance(value_from, dict):
            continue
        secret_ref = value_from.get("secretKeyRef")
        if not isinstance(secret_ref, dict):
            continue
        secret_ref["name"] = "huggingface-token"
        hf_token_rewritten = True
    existing_env.append(
        {
            "name": "CUDA_VISIBLE_DEVICES",
            "value": _cuda_visible_devices(runtime.tensor_parallelism),
        }
    )
    existing_env.extend(
        {"name": key, "value": value} for key, value in sorted(runtime.env.items())
    )
    if not hf_token_rewritten:
        existing_env.append(
            {
                "name": "HF_TOKEN",
                "valueFrom": {
                    "secretKeyRef": {"name": "huggingface-token", "key": "HF_TOKEN"}
                },
            }
        )
    existing_env.extend(env)

    container["command"] = ["vllm", "serve"]
    container["args"] = args + list(runtime.vllm_args)
    container["env"] = existing_env
    _apply_runtime_resources(container, plan)

    volume_mounts = container.setdefault("volumeMounts", [])
    if not any(
        str(volume_mount.get("name") or "") == "models-storage"
        for volume_mount in volume_mounts
    ):
        volume_mounts.append(
            {
                "name": "models-storage",
                "mountPath": storage.mount_path,
                "readOnly": True,
            }
        )

    volumes = (
        patch.setdefault("spec", {})
        .setdefault("template", {})
        .setdefault("spec", {})
        .setdefault("volumes", [])
    )
    if not any(str(volume.get("name") or "") == "models-storage" for volume in volumes):
        volumes.append(
            {
                "name": "models-storage",
                "persistentVolumeClaim": {"claimName": storage.pvc_name},
            }
        )

    pod_spec = (
        patch.setdefault("spec", {}).setdefault("template", {}).setdefault("spec", {})
    )
    if runtime.node_selector:
        pod_spec["nodeSelector"] = dict(runtime.node_selector)
    if runtime.affinity:
        pod_spec["affinity"] = dict(runtime.affinity)
    if runtime.tolerations:
        pod_spec["tolerations"] = list(runtime.tolerations)
    if runtime.image_pull_secrets:
        pod_spec["imagePullSecrets"] = list(runtime.image_pull_secrets)

    podmonitor_path = overlay_dir / _MODELSERVER_PODMONITOR_FILE
    podmonitor_path.write_text(
        yaml.safe_dump(
            _recipe_modelserver_podmonitor_manifest(plan, guide_name),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _append_kustomize_resource(kustomization, _MODELSERVER_PODMONITOR_FILE)

    if storage_offloading:
        _apply_recipe_storage_offloading(
            overlay_dir=overlay_dir,
            kustomization=kustomization,
            patch=patch,
            container=container,
            plan=plan,
            guide_name=guide_name,
            config=storage_offloading,
        )

    kustomization_path.write_text(
        yaml.safe_dump(kustomization, sort_keys=False), encoding="utf-8"
    )
    patch_path.write_text(yaml.safe_dump(patch, sort_keys=False), encoding="utf-8")


def _patch_recipe_gateway(plan: ResolvedRunPlan, gateway_dir: Path) -> None:
    gateway_config_name = f"infra-{plan.deployment.release_name}-inference-gateway"
    shared_gateway_labels = {
        "app.kubernetes.io/name": "benchflow",
        "benchflow.io/platform": "llm-d",
    }
    release_labels = {
        **shared_gateway_labels,
        _BENCHFLOW_RELEASE_LABEL: plan.deployment.release_name,
    }
    gateway_path = gateway_dir / "gateway.yaml"
    configmap_path = gateway_dir / "configmap.yaml"
    for path in (gateway_path, configmap_path):
        manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise CommandError(f"expected llm-d gateway manifest not found: {path}")
        kind = str(manifest.get("kind") or "")
        metadata = manifest.setdefault("metadata", {})
        if kind == "ConfigMap":
            metadata["name"] = gateway_config_name
        manifest_labels = metadata.setdefault("labels", {})
        if not isinstance(manifest_labels, dict):
            manifest_labels = {}
            metadata["labels"] = manifest_labels
        # The recipe scheduler chart creates HTTPRoutes that target the upstream
        # shared Gateway name. Keep that name stable and only make the Gateway
        # point at the release-specific infrastructure ConfigMap.
        manifest_labels.update(
            shared_gateway_labels if kind == "Gateway" else release_labels
        )
        if kind == "Gateway":
            infrastructure = manifest.setdefault("spec", {}).setdefault(
                "infrastructure", {}
            )
            parameters_ref = infrastructure.setdefault("parametersRef", {})
            parameters_ref["name"] = gateway_config_name
        path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


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


def _capture_recipe_inputs(
    *,
    plan: ResolvedRunPlan,
    scheduler_values_file: Path,
    gateway_dir: Path | None,
    overlay_dir: Path,
    manifests_dir: Path,
    router_chart: bool,
) -> None:
    manifests_dir.mkdir(parents=True, exist_ok=True)
    rendered_dir = manifests_dir / "rendered"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(scheduler_values_file, rendered_dir / "scheduler.values.yaml")
    if not router_chart:
        (rendered_dir / "epp-podmonitor.yaml").write_text(
            yaml.safe_dump(
                _recipe_epp_podmonitor_manifest(plan, router_chart=router_chart),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
    source_dirs = [(overlay_dir, "modelserver")]
    if gateway_dir is not None:
        source_dirs.insert(0, (gateway_dir, "gateway"))
    for source_dir, target_name in source_dirs:
        target_dir = rendered_dir / target_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for path in source_dir.iterdir():
            if path.is_file():
                shutil.copy2(path, target_dir / path.name)


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
    api_group = _llmd_inference_model_api_group(plan.deployment.repo_ref)
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
                "apiGroups": [api_group],
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


def _patch_standalone_envoy_volume(
    plan: ResolvedRunPlan, kubectl_cmd: str, *, skip_if_missing: bool = False
) -> None:
    deployment_name = f"{_llmd_recipe_scheduler_release_name(plan)}-epp"
    configmap_name = _llmd_recipe_standalone_envoy_configmap_name(plan)
    patch = {
        "spec": {
            "template": {
                "spec": {
                    "volumes": [
                        {
                            "name": "config",
                            "configMap": {
                                "name": configmap_name,
                                "items": [{"key": "envoy.yaml", "path": "envoy.yaml"}],
                            },
                        }
                    ]
                }
            }
        }
    }
    step(f"Patching llm-d standalone Envoy volume to use ConfigMap {configmap_name}")
    result = run_command(
        [
            kubectl_cmd,
            "patch",
            "deployment",
            deployment_name,
            "-n",
            plan.deployment.namespace,
            "--type=strategic",
            "-p",
            json.dumps(patch, separators=(",", ":")),
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return
    if skip_if_missing and "not found" in (result.stderr or "").lower():
        return
    raise CommandError(
        f"failed to patch llm-d standalone Envoy volume for {deployment_name}: "
        f"{(result.stderr or result.stdout or '').strip()}"
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


def _pods_ready_any(
    namespace: str, selectors: list[str], kubectl_cmd: str
) -> tuple[bool, int, int]:
    for selector in selectors:
        ready, ready_count, total = _pods_ready(namespace, selector, kubectl_cmd)
        if total > 0:
            return ready, ready_count, total
    return False, 0, 0


def _gateway_exists(
    namespace: str, release_name: str, kubectl_cmd: str, *, recipe_layout: bool
) -> bool:
    gateway_names = (
        ["llm-d-inference-gateway", f"infra-{release_name}-inference-gateway"]
        if recipe_layout
        else [f"infra-{release_name}-inference-gateway"]
    )
    for gateway_name in gateway_names:
        result = run_command(
            [
                kubectl_cmd,
                "get",
                "gateway",
                gateway_name,
                "-n",
                namespace,
                "-o",
                "name",
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return True
    return False


def _httproute_exists(
    namespace: str,
    release_name: str,
    kubectl_cmd: str,
    *,
    recipe_layout: bool,
    router_chart: bool,
) -> bool:
    if router_chart:
        result = run_command(
            [
                kubectl_cmd,
                "get",
                "httproute",
                "-n",
                namespace,
                "-l",
                "app.kubernetes.io/instance="
                f"{_llmd_recipe_scheduler_release_name_for(release_name)}",
                "-o",
                "name",
            ],
            capture_output=True,
            check=False,
        )
        return result.returncode == 0 and bool(str(result.stdout or "").strip())
    route_name = f"gaie-{release_name}" if recipe_layout else f"llm-d-{release_name}"
    result = run_command(
        [
            kubectl_cmd,
            "get",
            "httproute",
            route_name,
            "-n",
            namespace,
            "-o",
            "name",
        ],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _verify_deployment(
    plan: ResolvedRunPlan,
    timeout_seconds: int,
    *,
    recipe_layout: bool,
    router_chart: bool = False,
) -> None:
    kubectl_cmd = require_any_command("oc", "kubectl")
    namespace = plan.deployment.namespace
    release_name = plan.deployment.release_name
    gateway_mode = str(plan.deployment.gateway or "").strip()
    standalone = recipe_layout and gateway_mode == "standalone"
    deadline = time.time() + timeout_seconds
    last_snapshot: tuple[int, int, bool, bool] | None = None

    step(
        f"Waiting for llm-d deployment {release_name} in namespace {namespace} to become ready"
    )

    while time.time() < deadline:
        epp_selectors = (
            _llmd_router_epp_selectors(release_name, gateway_mode)
            if router_chart
            else [f"inferencepool=gaie-{release_name}-epp"]
        )
        epp_ready, epp_ready_count, epp_total = _pods_ready_any(
            namespace, epp_selectors, kubectl_cmd
        )
        if recipe_layout:
            guide_name = _llmd_recipe_guide_name(plan, router_chart=router_chart)
            model_selector = (
                f"{_BENCHFLOW_RELEASE_LABEL}={release_name},"
                f"{_BENCHFLOW_GUIDE_LABEL}={guide_name}"
            )
        else:
            model_selector = (
                f"{_LLMD_INFERENCE_SERVING_LABEL}=true,"
                f"{_LLMD_MODEL_LABEL}={release_name}"
            )
        ms_ready, ms_ready_count, ms_total = _pods_ready(
            namespace,
            model_selector,
            kubectl_cmd,
        )
        gateway_ready = standalone or _gateway_exists(
            namespace, release_name, kubectl_cmd, recipe_layout=recipe_layout
        )
        httproute_ready = standalone or _httproute_exists(
            namespace,
            release_name,
            kubectl_cmd,
            recipe_layout=recipe_layout,
            router_chart=router_chart,
        )
        snapshot = (
            epp_ready_count,
            ms_ready_count,
            gateway_ready,
            httproute_ready,
        )

        if snapshot != last_snapshot:
            if standalone:
                detail(
                    f"EPP pods ready: {epp_ready_count}/{epp_total}, "
                    f"model-service pods ready: {ms_ready_count}/{ms_total}"
                )
            else:
                httproute_text = (
                    f"httproute present: {'yes' if httproute_ready else 'no'}"
                )
                detail(
                    f"EPP pods ready: {epp_ready_count}/{epp_total}, "
                    f"model-service pods ready: {ms_ready_count}/{ms_total}, "
                    f"gateway present: {'yes' if gateway_ready else 'no'}, "
                    f"{httproute_text}"
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
    verify_timeout_seconds: int = 1800,
) -> Path:
    require_command("helm")
    kubectl_cmd = require_any_command("oc", "kubectl")

    if skip_if_exists and _release_exists(
        plan.deployment.namespace, plan.deployment.release_name
    ):
        _ensure_gaie_rbac(plan, kubectl_cmd)
        if (
            str(plan.deployment.gateway or "").strip() == "standalone"
            and str(plan.deployment.repo_ref or "").strip() != "main"
        ):
            _patch_standalone_envoy_volume(plan, kubectl_cmd, skip_if_missing=True)
        success(
            "Skipping deploy; llm-d Helm release already exists for "
            f"{plan.deployment.release_name}"
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
    repo_head = clone_repo(
        url=plan.deployment.repo_url,
        revision=plan.deployment.repo_ref,
        output_dir=checkout_dir,
        delete_existing=True,
    )
    _record_llmd_repo_head(plan, kubectl_cmd, repo_head)

    recipe_layout = _llmd_recipe_layout_available(checkout_dir)
    router_chart = _llmd_recipe_router_layout_available(checkout_dir)
    storage_offloading = _storage_offloading_config(plan)
    if recipe_layout:
        gateway_mode = str(plan.deployment.gateway or "").strip()
        if gateway_mode not in {"istio", "standalone"}:
            raise CommandError(
                "llm-d recipe layout currently supports gateway=istio or "
                f"gateway=standalone, got {plan.deployment.gateway}"
            )
        guide_name = _llmd_recipe_guide_name(plan, router_chart=router_chart)
        guide_dir = checkout_dir / "guides" / guide_name
        scheduler_values_file = _llmd_recipe_scheduler_values_path(
            checkout_dir, plan, router_chart=router_chart
        )
        gateway_dir = (
            None
            if gateway_mode == "standalone"
            else _llmd_recipe_gateway_dir(checkout_dir)
        )
        overlay_dir = _llmd_recipe_modelserver_overlay_dir(
            checkout_dir, plan, router_chart=router_chart
        )
        if not scheduler_values_file.exists():
            raise CommandError(
                f"expected llm-d guide file not found: {scheduler_values_file}"
            )
        if gateway_dir is not None and not gateway_dir.exists():
            raise CommandError(
                f"expected llm-d gateway directory not found: {gateway_dir}"
            )
        if not overlay_dir.exists():
            raise CommandError(
                f"expected llm-d modelserver overlay not found: {overlay_dir}"
            )
        if plan.deployment.mode == "precise-prefix-cache":
            require_command("kustomize")

        step(f"Patching llm-d recipe values for release {plan.deployment.release_name}")
        detail(f"Guide directory: {guide_dir}")
        _patch_scheduler_values(
            plan,
            scheduler_values_file,
            recipe_layout=True,
            router_chart=router_chart,
        )
        if gateway_dir is not None:
            _patch_recipe_gateway(plan, gateway_dir)
        _patch_recipe_modelserver_overlay(plan, overlay_dir, router_chart=router_chart)
        if storage_offloading:
            _ensure_storage_offloading_pvc(plan, kubectl_cmd, storage_offloading)

        env = {
            **os.environ,
            "HOME": "/tmp",
            "HELM_CACHE_HOME": "/tmp/.cache/helm",
            "HELM_CONFIG_HOME": "/tmp/.config/helm",
            "HELM_DATA_HOME": "/tmp/.local/share/helm",
            "HELM_PLUGINS": "/tmp/.local/share/helm/plugins",
            "RELEASE_NAME_POSTFIX": plan.deployment.release_name,
        }
        if router_chart:
            chart_ref = (
                "oci://ghcr.io/llm-d/charts/llm-d-router-standalone-dev"
                if gateway_mode == "standalone"
                else "oci://ghcr.io/llm-d/charts/llm-d-router-gateway-dev"
            )
            helm_args = [
                "helm",
                "install",
                _llmd_recipe_scheduler_release_name(plan),
                chart_ref,
                "-f",
                str(
                    checkout_dir / "guides" / "recipes" / "router" / "base.values.yaml"
                ),
                "-f",
                str(
                    checkout_dir
                    / "guides"
                    / "recipes"
                    / "router"
                    / "features"
                    / "monitoring.values.yaml"
                ),
                "-f",
                str(scheduler_values_file),
                "--set",
                f"provider.name={'none' if gateway_mode == 'standalone' else 'istio'}",
                "-n",
                plan.deployment.namespace,
                "--version",
                "v0",
            ]
            if gateway_mode != "standalone":
                helm_args.extend(
                    [
                        "-f",
                        str(
                            checkout_dir
                            / "guides"
                            / "recipes"
                            / "router"
                            / "features"
                            / "httproute-flags.yaml"
                        ),
                    ]
                )
        else:
            chart_name = (
                "standalone" if gateway_mode == "standalone" else "inferencepool"
            )
            helm_args = [
                "helm",
                "install",
                _llmd_recipe_scheduler_release_name(plan),
                "oci://registry.k8s.io/gateway-api-inference-extension/charts/"
                f"{chart_name}",
                "-f",
                str(
                    checkout_dir
                    / "guides"
                    / "recipes"
                    / "scheduler"
                    / "base.values.yaml"
                ),
                "-f",
                str(scheduler_values_file),
                "-n",
                plan.deployment.namespace,
                "--version",
                "v1.5.0",
            ]
            if gateway_mode != "standalone":
                helm_args.extend(
                    [
                        "-f",
                        str(
                            checkout_dir
                            / "guides"
                            / "recipes"
                            / "scheduler"
                            / "features"
                            / "httproute-flags.yaml"
                        ),
                        "--set",
                        "provider.name=istio",
                        "--set",
                        "experimentalHttpRoute.enabled=true",
                        "--set",
                        "experimentalHttpRoute.inferenceGatewayName=llm-d-inference-gateway",
                    ]
                )
            else:
                helm_args.extend(
                    [
                        "--set",
                        "inferenceExtension.sidecar.configMap.name="
                        f"{_llmd_recipe_standalone_envoy_configmap_name(plan)}",
                    ]
                )
        if plan.deployment.mode == "precise-prefix-cache" and not router_chart:
            helm_args.extend(
                [
                    "--post-renderer",
                    str(
                        checkout_dir
                        / "guides"
                        / "precise-prefix-cache-aware"
                        / "scheduler"
                        / "patches"
                        / "uds-tokenizer"
                        / "post-renderer.sh"
                    ),
                ]
            )

        if manifests_dir is not None:
            step(f"Capturing rendered manifests in {manifests_dir}")
            _capture_recipe_inputs(
                plan=plan,
                scheduler_values_file=scheduler_values_file,
                gateway_dir=gateway_dir,
                overlay_dir=overlay_dir,
                manifests_dir=manifests_dir,
                router_chart=router_chart,
            )

        step(
            f"Installing llm-d recipe scheduler {helm_args[2]} "
            f"into namespace {plan.deployment.namespace}"
        )
        run_command(helm_args, cwd=guide_dir, env=env)
        _apply_recipe_epp_podmonitor(plan, kubectl_cmd, router_chart=router_chart)

        if gateway_mode == "standalone" and not router_chart:
            # The official standalone chart renders the Envoy ConfigMap name from
            # values, but v1.5.0 still hard-codes the mounted volume reference to
            # "envoy". Patch the Deployment so parallel BenchFlow releases do not
            # fight over one shared ConfigMap.
            _patch_standalone_envoy_volume(plan, kubectl_cmd)

        if gateway_dir is not None:
            step(f"Applying llm-d gateway resources from {gateway_dir}")
            run_command(
                [
                    kubectl_cmd,
                    "apply",
                    "-n",
                    plan.deployment.namespace,
                    "-k",
                    str(gateway_dir),
                ]
            )
        step(f"Applying llm-d modelserver overlay from {overlay_dir}")
        run_command(
            [
                kubectl_cmd,
                "apply",
                "-n",
                plan.deployment.namespace,
                "-k",
                str(overlay_dir),
            ]
        )
        success(
            f"Applied llm-d recipe resources for {plan.deployment.release_name} "
            f"in namespace {plan.deployment.namespace}"
        )
    else:
        if storage_offloading:
            raise CommandError(
                "llm-d storage_offloading requires the upstream recipe layout"
            )
        guide_layout = _llmd_guide_layout(plan)
        guide_dir = checkout_dir / "guides" / guide_layout["guide_dirname"]
        values_file = guide_dir / Path(guide_layout["model_values_relpath"])
        scheduler_values_file = guide_dir / Path(
            guide_layout["scheduler_values_relpath"]
        )
        if not values_file.exists():
            raise CommandError(f"expected llm-d guide file not found: {values_file}")
        if not scheduler_values_file.exists():
            raise CommandError(
                f"expected llm-d guide file not found: {scheduler_values_file}"
            )
        require_command("helmfile")

        step(f"Patching llm-d guide values for release {plan.deployment.release_name}")
        detail(f"Guide directory: {guide_dir}")
        values = _patch_values(plan, values_file)
        _patch_scheduler_values(plan, scheduler_values_file, recipe_layout=False)
        _apply_pipeline_labels(
            values,
            plan.deployment.release_name,
            execution_name,
            execution_backend="tekton",
        )
        values_file.write_text(
            yaml.safe_dump(values, sort_keys=False), encoding="utf-8"
        )

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
        _verify_deployment(
            plan,
            verify_timeout_seconds,
            recipe_layout=recipe_layout,
            router_chart=router_chart,
        )

    if created_tempdir:
        return checkout_root
    return checkout_root.resolve()
