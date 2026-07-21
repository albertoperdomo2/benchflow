from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from ..assets import asset_text, render_jinja_text, render_jinja_yaml_document
from ..models import ResolvedRunPlan, ValidationError, model_storage_relative_path

RHOAI_PROFILER_CONFIGMAP_SUFFIX = "vllm-profiler"
RHOAI_PROFILER_MOUNT_PATH = "/home/vllm/profiler"
RHOAI_PROFILER_OUTPUT_DIR = "/tmp/benchflow-profiler"
RAHIIS_PROGRESS_DEADLINE_SECONDS = 1800


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _base_labels(plan: ResolvedRunPlan) -> dict[str, str]:
    labels = {
        "app.kubernetes.io/name": "benchflow",
        "benchflow.io/experiment": plan.metadata.name,
        "benchflow.io/platform": plan.deployment.platform,
        "benchflow.io/mode": plan.deployment.mode,
    }
    placement = plan.deployment.runtime.placement
    if placement.mode == "same-node":
        labels["benchflow.io/placement-pool"] = placement.spread_pool
        labels["benchflow.io/placement-group"] = plan.deployment.release_name
    return labels


def _append_affinity_terms(
    affinity: dict[str, Any], section: str, key: str, terms: list[dict[str, Any]]
) -> None:
    pod_affinity = affinity.setdefault(section, {})
    existing = pod_affinity.get(key)
    if existing is None:
        pod_affinity[key] = terms
        return
    if isinstance(existing, list):
        for term in terms:
            if term not in existing:
                existing.append(term)
        return
    raise ValidationError(f"runtime.affinity.{section}.{key} must be a list")


def _runtime_affinity(plan: ResolvedRunPlan) -> dict[str, Any]:
    affinity = deepcopy(plan.deployment.runtime.affinity)
    placement = plan.deployment.runtime.placement
    # Sequential controls matrix submission order in the orchestration layer.
    # It must not constrain unrelated matrices at pod scheduling time.
    if placement.mode != "same-node":
        return affinity

    # LLMInferenceService exposes only a PodSpec template, not PodTemplate metadata,
    # so custom BenchFlow placement labels cannot be propagated to model pods.
    # Use stable RHOAI/KServe-generated workload labels instead.
    workload_selector = {
        "app.kubernetes.io/part-of": "llminferenceservice",
        "kserve.io/component": "workload",
    }
    workload_identity_label = "app.kubernetes.io/name"
    _append_affinity_terms(
        affinity,
        "podAffinity",
        "requiredDuringSchedulingIgnoredDuringExecution",
        [
            {
                "topologyKey": "kubernetes.io/hostname",
                "labelSelector": {
                    "matchLabels": workload_selector,
                },
                "matchLabelKeys": [workload_identity_label],
            }
        ],
    )
    _append_affinity_terms(
        affinity,
        "podAntiAffinity",
        "requiredDuringSchedulingIgnoredDuringExecution",
        [
            {
                "topologyKey": "kubernetes.io/hostname",
                "labelSelector": {
                    "matchLabels": workload_selector,
                },
                "mismatchLabelKeys": [workload_identity_label],
            }
        ],
    )
    return affinity


def _validate_rhoai_profiling(plan: ResolvedRunPlan) -> None:
    if not plan.execution.profiling.enabled:
        return
    if plan.deployment.platform != "rhoai":
        raise ValidationError(
            "execution.profiling is currently supported only for rhoai deployments"
        )


def _model_path(plan: ResolvedRunPlan) -> str:
    return f"/{model_storage_relative_path(plan.deployment.model_storage, plan.model)}"


def render_llmd_values(plan: ResolvedRunPlan) -> dict[str, Any]:
    return {
        "releaseName": plan.deployment.release_name,
        "platform": plan.deployment.platform,
        "mode": plan.deployment.mode,
        "namespace": plan.deployment.namespace,
        "repoRef": plan.deployment.repo_ref,
        "platformChannel": plan.deployment.platform_channel,
        "gateway": plan.deployment.gateway,
        "schedulerProfile": plan.deployment.scheduler_profile,
        "schedulerImage": plan.deployment.scheduler_image,
        "modelArtifacts": {
            "name": plan.model.name,
            "uri": f"pvc://{plan.deployment.model_storage.pvc_name}{_model_path(plan)}",
        },
        "runtime": {
            "image": plan.deployment.runtime.image,
            "replicas": plan.deployment.runtime.replicas,
            "tensorParallelism": plan.deployment.runtime.tensor_parallelism,
            "vllmArgs": plan.deployment.runtime.vllm_args,
            "env": plan.deployment.runtime.env,
            "sharedMemorySize": plan.deployment.runtime.shared_memory_size,
            "serviceAccountName": plan.deployment.runtime.service_account_name,
            "nodeSelector": plan.deployment.runtime.node_selector,
            "affinity": plan.deployment.runtime.affinity,
            "tolerations": plan.deployment.runtime.tolerations,
            "imagePullSecrets": plan.deployment.runtime.image_pull_secrets,
            "hostPaths": [
                {
                    "name": host_path.name,
                    "hostPath": host_path.host_path,
                    "mountPath": host_path.mount_path,
                    "type": host_path.type,
                    "readOnly": host_path.read_only,
                }
                for host_path in plan.deployment.runtime.host_paths
            ],
            "pvcMounts": [
                {
                    "name": pvc_mount.name,
                    "claimName": pvc_mount.claim_name,
                    "mountPath": pvc_mount.mount_path,
                    "readOnly": pvc_mount.read_only,
                    "create": pvc_mount.create,
                    "storageClassName": pvc_mount.storage_class_name,
                    "size": pvc_mount.size,
                    "accessModes": list(pvc_mount.access_modes),
                }
                for pvc_mount in plan.deployment.runtime.pvc_mounts
            ],
            "resources": {
                "limits": dict(plan.deployment.runtime.resources.limits),
                "requests": dict(plan.deployment.runtime.resources.requests),
                "removeLimits": list(plan.deployment.runtime.resources.remove_limits),
                "removeRequests": list(
                    plan.deployment.runtime.resources.remove_requests
                ),
            },
        },
        "options": plan.deployment.options,
    }


def _runtime_resource_requirements(
    plan: ResolvedRunPlan, *, include_gpu: bool
) -> dict[str, dict[str, str]]:
    resources = {
        "limits": dict(plan.deployment.runtime.resources.limits),
        "requests": dict(plan.deployment.runtime.resources.requests),
    }
    if include_gpu:
        gpu_count = str(plan.deployment.runtime.tensor_parallelism)
        resources["limits"]["nvidia.com/gpu"] = gpu_count
        resources["requests"]["nvidia.com/gpu"] = gpu_count
    return resources


def _runtime_host_path_volume_mounts(plan: ResolvedRunPlan) -> list[dict[str, Any]]:
    mounts: list[dict[str, Any]] = []
    for host_path in plan.deployment.runtime.host_paths:
        mounts.append(
            {
                "name": host_path.name,
                "mountPath": host_path.mount_path,
                "readOnly": host_path.read_only,
            }
        )
    return mounts


def _runtime_pvc_volume_mounts(plan: ResolvedRunPlan) -> list[dict[str, Any]]:
    mounts: list[dict[str, Any]] = []
    for pvc_mount in plan.deployment.runtime.pvc_mounts:
        mounts.append(
            {
                "name": pvc_mount.name,
                "mountPath": pvc_mount.mount_path,
                "readOnly": pvc_mount.read_only,
            }
        )
    return mounts


def _runtime_host_path_volumes(plan: ResolvedRunPlan) -> list[dict[str, Any]]:
    volumes: list[dict[str, Any]] = []
    for host_path in plan.deployment.runtime.host_paths:
        host_path_spec = {"path": host_path.host_path, "type": host_path.type}
        volumes.append({"name": host_path.name, "hostPath": host_path_spec})
    return volumes


def _runtime_pvc_volumes(plan: ResolvedRunPlan) -> list[dict[str, Any]]:
    volumes: list[dict[str, Any]] = []
    for pvc_mount in plan.deployment.runtime.pvc_mounts:
        volumes.append(
            {
                "name": pvc_mount.name,
                "persistentVolumeClaim": {"claimName": pvc_mount.claim_name},
            }
        )
    return volumes


def render_runtime_pvc_manifests(plan: ResolvedRunPlan) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for pvc_mount in plan.deployment.runtime.pvc_mounts:
        if not pvc_mount.create:
            continue
        spec: dict[str, Any] = {
            "accessModes": list(pvc_mount.access_modes),
            "resources": {"requests": {"storage": pvc_mount.size}},
        }
        if pvc_mount.storage_class_name:
            spec["storageClassName"] = pvc_mount.storage_class_name
        manifests.append(
            {
                "apiVersion": "v1",
                "kind": "PersistentVolumeClaim",
                "metadata": {
                    "name": pvc_mount.claim_name,
                    "namespace": plan.deployment.namespace,
                    "labels": {
                        **_base_labels(plan),
                        "benchflow.io/purpose": "runtime-pvc-mount",
                    },
                },
                "spec": spec,
            }
        )
    return manifests


def _rhoai_uses_isvc(plan: ResolvedRunPlan) -> bool:
    return plan.deployment.mode == "isvc"


def _rhoai_runtime_env(plan: ResolvedRunPlan) -> list[dict[str, Any]]:
    return [
        {"name": key, "value": value}
        for key, value in sorted(plan.deployment.runtime.env.items())
    ]


def _rhoai_basic_model_path(plan: ResolvedRunPlan) -> str:
    mount_root = plan.deployment.model_storage.mount_path.rstrip("/")
    return f"{mount_root}/{model_storage_relative_path(plan.deployment.model_storage, plan.model)}"


def _rhoai_basic_runtime_env(plan: ResolvedRunPlan) -> list[dict[str, Any]]:
    mount_root = plan.deployment.model_storage.mount_path.rstrip("/")
    cache_dir = f"{mount_root}{plan.deployment.model_storage.cache_dir.rstrip('/')}"
    env = {
        "HOME": "/tmp/vllm-home",
        "HF_HOME": cache_dir,
        "TRANSFORMERS_CACHE": f"{cache_dir}/hub",
        "HF_HUB_CACHE": f"{cache_dir}/hub",
        **plan.deployment.runtime.env,
    }
    return [{"name": key, "value": value} for key, value in sorted(env.items())]


def _rhoai_vllm_args(plan: ResolvedRunPlan) -> list[str]:
    model_path = f"/mnt/models{_model_path(plan)}"
    return [
        "--port=8000",
        "--host=0.0.0.0",
        f"--model={model_path}",
        f"--served-model-name={plan.model.name}",
        f"--tensor-parallel-size={plan.deployment.runtime.tensor_parallelism}",
        "--enable-ssl-refresh",
        "--ssl-certfile=/var/run/kserve/tls/tls.crt",
        "--ssl-keyfile=/var/run/kserve/tls/tls.key",
    ] + plan.deployment.runtime.vllm_args


def _rhoai_basic_vllm_args(plan: ResolvedRunPlan) -> list[str]:
    return [
        "--port=8080",
        "--host=0.0.0.0",
        f"--model={_rhoai_basic_model_path(plan)}",
        f"--served-model-name={plan.model.name}",
        f"--tensor-parallel-size={plan.deployment.runtime.tensor_parallelism}",
    ] + plan.deployment.runtime.vllm_args


def _rhoai_precise_tokenizer_model_path(plan: ResolvedRunPlan) -> str:
    return f"/mnt/models/base{_model_path(plan)}"


def _rhoai_custom_epp_config_lines(
    plan: ResolvedRunPlan, context: dict[str, Any]
) -> list[str]:
    raw_config = plan.deployment.options.get("epp_config")
    if raw_config is None or str(raw_config).strip() == "":
        return []
    if not isinstance(raw_config, str):
        raise ValidationError(
            "deployment profile options.epp_config must be a YAML string"
        )

    rendered = render_jinja_text(raw_config, context).strip()
    parsed = yaml.safe_load(rendered)
    if not isinstance(parsed, dict):
        raise ValidationError(
            "deployment profile options.epp_config must render to a YAML mapping"
        )
    if parsed.get("kind") != "EndpointPickerConfig":
        raise ValidationError(
            "deployment profile options.epp_config must render an EndpointPickerConfig"
        )
    return rendered.splitlines()


def _rhoai_epp_verbosity(plan: ResolvedRunPlan) -> int | None:
    raw_value = plan.deployment.options.get("epp_verbosity")
    if raw_value is None or str(raw_value).strip() == "":
        return None
    if isinstance(raw_value, bool):
        raise ValidationError(
            "deployment profile options.epp_verbosity must be an integer"
        )
    try:
        verbosity = int(str(raw_value).strip())
    except ValueError as exc:
        raise ValidationError(
            "deployment profile options.epp_verbosity must be an integer"
        ) from exc
    if verbosity < 0:
        raise ValidationError(
            "deployment profile options.epp_verbosity must be greater than or "
            "equal to 0"
        )
    return verbosity


def _rhoai_startup_probe(plan: ResolvedRunPlan) -> dict[str, Any] | None:
    default_probe = {
        "httpGet": {
            "path": "/health",
            "port": 8000,
            "scheme": "HTTPS",
        },
        "failureThreshold": 120,
        "periodSeconds": 10,
        "timeoutSeconds": 1,
    }
    raw_probe = plan.deployment.options.get("startup_probe")
    if raw_probe is None:
        return default_probe
    if raw_probe is False:
        return None
    if isinstance(raw_probe, str):
        raw_probe = yaml.safe_load(raw_probe)
    if not isinstance(raw_probe, dict):
        raise ValidationError(
            "deployment profile options.startup_probe must be a mapping"
        )
    return _deep_merge(default_probe, raw_probe)


def _yaml_lines(value: dict[str, Any] | None) -> list[str]:
    if value is None:
        return []
    return yaml.safe_dump(value, sort_keys=False).rstrip().splitlines()


def _rhoai_validate_isvc(plan: ResolvedRunPlan) -> None:
    if not _rhoai_uses_isvc(plan):
        return
    if not plan.deployment.runtime.image:
        raise ValidationError("rhoai isvc deployments require deployment.runtime.image")
    if plan.deployment.scheduler_image:
        raise ValidationError("rhoai isvc deployments do not support scheduler_image")
    if str(plan.deployment.options.get("epp_config") or "").strip():
        raise ValidationError(
            "rhoai isvc deployments do not support options.epp_config"
        )
    if _rhoai_epp_verbosity(plan) is not None:
        raise ValidationError(
            "rhoai isvc deployments do not support options.epp_verbosity"
        )


def _rhoai_llminferenceservice_template_context(
    plan: ResolvedRunPlan,
) -> dict[str, Any]:
    _validate_rhoai_profiling(plan)
    has_custom_epp_config = bool(
        str(plan.deployment.options.get("epp_config") or "").strip()
    )
    epp_verbosity = _rhoai_epp_verbosity(plan)
    custom_scheduler_enabled = (
        plan.deployment.mode
        in {
            "approximate-prefix-cache",
            "precise-prefix-cache",
        }
        or has_custom_epp_config
        or epp_verbosity is not None
    )
    scheduler_config_enabled = (
        plan.deployment.mode
        in {
            "approximate-prefix-cache",
            "precise-prefix-cache",
        }
        or has_custom_epp_config
    )
    context: dict[str, Any] = {
        "release_name": plan.deployment.release_name,
        "namespace": plan.deployment.namespace,
        "labels": _base_labels(plan),
        "enable_auth": str(plan.deployment.options.get("enable_auth", False)).lower(),
        "model_name": plan.model.name,
        "model_uri": f"pvc://{plan.deployment.model_storage.pvc_name}",
        "replicas": plan.deployment.runtime.replicas,
        "runtime_image": plan.deployment.runtime.image,
        "scheduler_image": plan.deployment.scheduler_image,
        "runtime_args": _rhoai_vllm_args(plan),
        "runtime_env": _rhoai_runtime_env(plan),
        "runtime_node_selector": plan.deployment.runtime.node_selector,
        "runtime_affinity": _runtime_affinity(plan),
        "runtime_tolerations": plan.deployment.runtime.tolerations,
        "runtime_image_pull_secrets": plan.deployment.runtime.image_pull_secrets,
        "runtime_resources": _runtime_resource_requirements(plan, include_gpu=True),
        "runtime_service_account_name": plan.deployment.runtime.service_account_name,
        "runtime_shared_memory_size": plan.deployment.runtime.shared_memory_size,
        "runtime_host_path_mounts": _runtime_host_path_volume_mounts(plan),
        "runtime_host_path_volumes": _runtime_host_path_volumes(plan),
        "runtime_pvc_mounts": _runtime_pvc_volume_mounts(plan),
        "runtime_pvc_volumes": _runtime_pvc_volumes(plan),
        "startup_probe_lines": _yaml_lines(_rhoai_startup_probe(plan)),
        "gpu_count": str(plan.deployment.runtime.tensor_parallelism),
        "custom_scheduler_enabled": custom_scheduler_enabled,
        "scheduler_config_enabled": scheduler_config_enabled,
        "epp_verbosity": epp_verbosity,
        "approximate_prefix_cache_enabled": (
            plan.deployment.mode == "approximate-prefix-cache"
        ),
        "precise_prefix_cache_enabled": plan.deployment.mode == "precise-prefix-cache",
        "precise_prefix_cache_tokenizer_model_path": (
            _rhoai_precise_tokenizer_model_path(plan)
        ),
        "profiling_enabled": plan.execution.profiling.enabled,
        "profiler_call_ranges": plan.execution.profiling.call_ranges,
        "profiler_idle_seconds": plan.execution.profiling.idle_seconds,
        "profiler_configmap_name": rhoai_profiler_configmap_name(plan),
        "profiler_mount_path": RHOAI_PROFILER_MOUNT_PATH,
    }
    context["custom_epp_config_lines"] = _rhoai_custom_epp_config_lines(plan, context)
    return context


def _rhoai_inferenceservice_template_context(plan: ResolvedRunPlan) -> dict[str, Any]:
    _validate_rhoai_profiling(plan)
    _rhoai_validate_isvc(plan)
    return {
        "release_name": plan.deployment.release_name,
        "namespace": plan.deployment.namespace,
        "labels": _base_labels(plan),
        "enable_auth": str(plan.deployment.options.get("enable_auth", False)).lower(),
        "replicas": plan.deployment.runtime.replicas,
        "runtime_image": plan.deployment.runtime.image,
        "runtime_args": _rhoai_basic_vllm_args(plan),
        "runtime_env": _rhoai_basic_runtime_env(plan),
        "runtime_node_selector": plan.deployment.runtime.node_selector,
        "runtime_affinity": _runtime_affinity(plan),
        "runtime_tolerations": plan.deployment.runtime.tolerations,
        "runtime_image_pull_secrets": plan.deployment.runtime.image_pull_secrets,
        "runtime_resources": _runtime_resource_requirements(plan, include_gpu=True),
        "runtime_service_account_name": plan.deployment.runtime.service_account_name,
        "runtime_shared_memory_size": plan.deployment.runtime.shared_memory_size,
        "runtime_host_path_mounts": _runtime_host_path_volume_mounts(plan),
        "runtime_host_path_volumes": _runtime_host_path_volumes(plan),
        "runtime_pvc_mounts": _runtime_pvc_volume_mounts(plan),
        "runtime_pvc_volumes": _runtime_pvc_volumes(plan),
        "model_storage_pvc_name": plan.deployment.model_storage.pvc_name,
        "model_storage_mount_path": plan.deployment.model_storage.mount_path,
        "profiling_enabled": plan.execution.profiling.enabled,
        "profiler_call_ranges": plan.execution.profiling.call_ranges,
        "profiler_idle_seconds": plan.execution.profiling.idle_seconds,
        "profiler_configmap_name": rhoai_profiler_configmap_name(plan),
        "profiler_mount_path": RHOAI_PROFILER_MOUNT_PATH,
    }


def render_rhoai_manifest(plan: ResolvedRunPlan) -> dict[str, Any]:
    if _rhoai_uses_isvc(plan):
        return render_jinja_yaml_document(
            "deployment/rhoai/inferenceservice.yaml.j2",
            _rhoai_inferenceservice_template_context(plan),
        )
    if plan.deployment.mode not in {
        "distributed-default",
        "approximate-prefix-cache",
        "precise-prefix-cache",
    }:
        raise ValueError(f"unsupported RHOAI deployment mode: {plan.deployment.mode}")
    return render_jinja_yaml_document(
        "deployment/rhoai/llminferenceservice.yaml.j2",
        _rhoai_llminferenceservice_template_context(plan),
    )


def rhoai_profiler_configmap_name(plan: ResolvedRunPlan) -> str:
    return f"{plan.deployment.release_name}-{RHOAI_PROFILER_CONFIGMAP_SUFFIX}"


def render_rhoai_profiler_configmap(plan: ResolvedRunPlan) -> dict[str, Any]:
    _validate_rhoai_profiling(plan)
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": rhoai_profiler_configmap_name(plan),
            "namespace": plan.deployment.namespace,
            "labels": {
                **_base_labels(plan),
                "app.kubernetes.io/component": "vllm-profiler",
            },
        },
        "data": {
            "sitecustomize.py": asset_text("deployment/rhoai/profiler/sitecustomize.py")
        },
    }


def rhaiis_raw_vllm_deployment_name(plan: ResolvedRunPlan) -> str:
    return f"{plan.deployment.release_name}-vllm"


def rhaiis_raw_vllm_service_name(plan: ResolvedRunPlan) -> str:
    return plan.deployment.release_name


def rhaiis_raw_vllm_servicemonitor_name(plan: ResolvedRunPlan) -> str:
    return f"{plan.deployment.release_name}-vllm"


def _rhaiis_raw_vllm_labels(plan: ResolvedRunPlan) -> dict[str, str]:
    return {
        **_base_labels(plan),
        "app.kubernetes.io/component": "raw-vllm",
        "app.kubernetes.io/instance": plan.deployment.release_name,
        "benchflow.io/release": plan.deployment.release_name,
    }


def _rhaiis_raw_vllm_selector_labels(plan: ResolvedRunPlan) -> dict[str, str]:
    return {
        "app.kubernetes.io/component": "raw-vllm",
        "app.kubernetes.io/instance": plan.deployment.release_name,
        "benchflow.io/release": plan.deployment.release_name,
    }


def _rhaiis_raw_vllm_model_path(plan: ResolvedRunPlan) -> str:
    mount_root = plan.deployment.model_storage.mount_path.rstrip("/")
    return f"{mount_root}/{model_storage_relative_path(plan.deployment.model_storage, plan.model)}"


def _rhaiis_raw_vllm_runtime_env(plan: ResolvedRunPlan) -> list[dict[str, Any]]:
    mount_root = plan.deployment.model_storage.mount_path.rstrip("/")
    cache_dir = f"{mount_root}{plan.deployment.model_storage.cache_dir.rstrip('/')}"
    env = {
        "HOME": "/tmp/vllm-home",
        "HF_HOME": cache_dir,
        "TRANSFORMERS_CACHE": f"{cache_dir}/hub",
        "HF_HUB_CACHE": f"{cache_dir}/hub",
        **plan.deployment.runtime.env,
    }
    return [{"name": key, "value": value} for key, value in sorted(env.items())]


def render_rhaiis_raw_vllm_manifests(plan: ResolvedRunPlan) -> list[dict[str, Any]]:
    if not plan.deployment.runtime.image:
        raise ValidationError(
            "rhaiis raw-vllm deployments require deployment.runtime.image"
        )

    labels = _rhaiis_raw_vllm_labels(plan)
    selector_labels = _rhaiis_raw_vllm_selector_labels(plan)
    container_spec: dict[str, Any] = {
        "name": "vllm",
        "image": plan.deployment.runtime.image,
        "command": ["python3", "-m", "vllm.entrypoints.openai.api_server"],
        "args": [
            f"--model={_rhaiis_raw_vllm_model_path(plan)}",
            f"--served-model-name={plan.model.name}",
            f"--tensor-parallel-size={plan.deployment.runtime.tensor_parallelism}",
            "--port=8000",
            "--host=0.0.0.0",
            *plan.deployment.runtime.vllm_args,
        ],
        "env": _rhaiis_raw_vllm_runtime_env(plan),
        "ports": [{"containerPort": 8000, "name": "http", "protocol": "TCP"}],
        "readinessProbe": {
            "httpGet": {"path": "/health", "port": "http"},
            "periodSeconds": 10,
            "timeoutSeconds": 5,
            "failureThreshold": 3,
        },
        "resources": _runtime_resource_requirements(plan, include_gpu=True),
        "volumeMounts": [
            {
                "name": "model-storage",
                "mountPath": plan.deployment.model_storage.mount_path,
            },
            *(
                [{"name": "dshm", "mountPath": "/dev/shm"}]
                if plan.deployment.runtime.shared_memory_size
                else []
            ),
            *_runtime_host_path_volume_mounts(plan),
            *_runtime_pvc_volume_mounts(plan),
        ],
    }

    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": rhaiis_raw_vllm_deployment_name(plan),
            "namespace": plan.deployment.namespace,
            "labels": labels,
        },
        "spec": {
            "progressDeadlineSeconds": RAHIIS_PROGRESS_DEADLINE_SECONDS,
            "replicas": plan.deployment.runtime.replicas,
            "selector": {"matchLabels": selector_labels},
            "template": {
                "metadata": {"labels": {**labels, **selector_labels}},
                "spec": {
                    "containers": [container_spec],
                    "volumes": [
                        {
                            "name": "model-storage",
                            "persistentVolumeClaim": {
                                "claimName": plan.deployment.model_storage.pvc_name
                            },
                        },
                        *(
                            [
                                {
                                    "name": "dshm",
                                    "emptyDir": {
                                        "medium": "Memory",
                                        "sizeLimit": (
                                            plan.deployment.runtime.shared_memory_size
                                        ),
                                    },
                                }
                            ]
                            if plan.deployment.runtime.shared_memory_size
                            else []
                        ),
                        *_runtime_host_path_volumes(plan),
                        *_runtime_pvc_volumes(plan),
                    ],
                },
            },
        },
    }

    pod_spec = deployment["spec"]["template"]["spec"]
    if plan.deployment.runtime.node_selector:
        pod_spec["nodeSelector"] = dict(plan.deployment.runtime.node_selector)
    if plan.deployment.runtime.affinity:
        pod_spec["affinity"] = dict(plan.deployment.runtime.affinity)
    if plan.deployment.runtime.tolerations:
        pod_spec["tolerations"] = list(plan.deployment.runtime.tolerations)
    if plan.deployment.runtime.service_account_name:
        pod_spec["serviceAccountName"] = plan.deployment.runtime.service_account_name

    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": rhaiis_raw_vllm_service_name(plan),
            "namespace": plan.deployment.namespace,
            "labels": labels,
        },
        "spec": {
            "type": "ClusterIP",
            "selector": selector_labels,
            "ports": [
                {
                    "name": "http",
                    "port": 8000,
                    "protocol": "TCP",
                    "targetPort": "http",
                }
            ],
        },
    }

    servicemonitor = {
        "apiVersion": "monitoring.coreos.com/v1",
        "kind": "ServiceMonitor",
        "metadata": {
            "name": rhaiis_raw_vllm_servicemonitor_name(plan),
            "namespace": plan.deployment.namespace,
            "labels": labels,
        },
        "spec": {
            "selector": {"matchLabels": selector_labels},
            "namespaceSelector": {"matchNames": [plan.deployment.namespace]},
            "endpoints": [
                {
                    "path": "/metrics",
                    "port": "http",
                    "scheme": "http",
                }
            ],
        },
    }

    return [deployment, service, servicemonitor]


def write_deployment_assets(plan: ResolvedRunPlan, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if plan.deployment.platform == "llm-d":
        for pvc_manifest in render_runtime_pvc_manifests(plan):
            pvc_name = str(pvc_manifest.get("metadata", {}).get("name") or "runtime")
            target = output_dir / f"pvc-{pvc_name}.yaml"
            target.write_text(
                yaml.safe_dump(pvc_manifest, sort_keys=False), encoding="utf-8"
            )
            written.append(target)
        target = output_dir / "llm-d-values.yaml"
        target.write_text(
            yaml.safe_dump(render_llmd_values(plan), sort_keys=False), encoding="utf-8"
        )
        written.append(target)
        return written

    if plan.deployment.platform == "rhoai":
        for pvc_manifest in render_runtime_pvc_manifests(plan):
            pvc_name = str(pvc_manifest.get("metadata", {}).get("name") or "runtime")
            target = output_dir / f"pvc-{pvc_name}.yaml"
            target.write_text(
                yaml.safe_dump(pvc_manifest, sort_keys=False), encoding="utf-8"
            )
            written.append(target)
        if plan.execution.profiling.enabled:
            profiler_target = output_dir / "vllm-profiler-configmap.yaml"
            profiler_target.write_text(
                yaml.safe_dump(render_rhoai_profiler_configmap(plan), sort_keys=False),
                encoding="utf-8",
            )
            written.append(profiler_target)
        target = output_dir / "llminferenceservice.yaml"
        target.write_text(
            yaml.safe_dump(render_rhoai_manifest(plan), sort_keys=False),
            encoding="utf-8",
        )
        written.append(target)
        return written

    if plan.deployment.platform == "rhaiis":
        for pvc_manifest in render_runtime_pvc_manifests(plan):
            pvc_name = str(pvc_manifest.get("metadata", {}).get("name") or "runtime")
            target = output_dir / f"pvc-{pvc_name}.yaml"
            target.write_text(
                yaml.safe_dump(pvc_manifest, sort_keys=False), encoding="utf-8"
            )
            written.append(target)
        manifests = render_rhaiis_raw_vllm_manifests(plan)
        names = ["deployment.yaml", "service.yaml", "servicemonitor.yaml"]
        for manifest, name in zip(manifests, names, strict=True):
            target = output_dir / name
            target.write_text(
                yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
            )
            written.append(target)
        return written

    raise ValidationError(
        f"unsupported deployment platform for rendered assets: {plan.deployment.platform}"
    )
