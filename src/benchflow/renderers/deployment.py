from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..assets import asset_text, render_jinja_text, render_jinja_yaml_document
from ..models import ResolvedRunPlan, ValidationError

RHOAI_PROFILER_CONFIGMAP_SUFFIX = "vllm-profiler"
RHOAI_PROFILER_MOUNT_PATH = "/home/vllm/profiler"
RHOAI_PROFILER_OUTPUT_DIR = "/tmp/benchflow-profiler"


def _base_labels(plan: ResolvedRunPlan) -> dict[str, str]:
    return {
        "app.kubernetes.io/name": "benchflow",
        "benchflow.io/experiment": plan.metadata.name,
        "benchflow.io/platform": plan.deployment.platform,
        "benchflow.io/mode": plan.deployment.mode,
    }


def _validate_rhoai_profiling(plan: ResolvedRunPlan) -> None:
    if not plan.execution.profiling.enabled:
        return
    if plan.deployment.platform != "rhoai":
        raise ValidationError(
            "execution.profiling is currently supported only for rhoai deployments"
        )


def _model_path(plan: ResolvedRunPlan) -> str:
    return f"{plan.deployment.model_storage.cache_dir}/{plan.model.pvc_directory_name}"


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
            "nodeSelector": plan.deployment.runtime.node_selector,
            "affinity": plan.deployment.runtime.affinity,
            "tolerations": plan.deployment.runtime.tolerations,
            "resources": plan.deployment.runtime.resources,
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


def _rhoai_runtime_env(plan: ResolvedRunPlan) -> list[dict[str, Any]]:
    return [
        {"name": key, "value": value}
        for key, value in sorted(plan.deployment.runtime.env.items())
    ]


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


def _rhoai_template_context(plan: ResolvedRunPlan) -> dict[str, Any]:
    _validate_rhoai_profiling(plan)
    has_custom_epp_config = bool(
        str(plan.deployment.options.get("epp_config") or "").strip()
    )
    custom_scheduler_enabled = (
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
        "runtime_affinity": plan.deployment.runtime.affinity,
        "runtime_tolerations": plan.deployment.runtime.tolerations,
        "runtime_resources": _runtime_resource_requirements(plan, include_gpu=True),
        "gpu_count": str(plan.deployment.runtime.tensor_parallelism),
        "custom_scheduler_enabled": custom_scheduler_enabled,
        "approximate_prefix_cache_enabled": (
            plan.deployment.mode == "approximate-prefix-cache"
        ),
        "precise_prefix_cache_enabled": plan.deployment.mode == "precise-prefix-cache",
        "precise_prefix_cache_tokenizer_model_path": (
            _rhoai_precise_tokenizer_model_path(plan)
        ),
        "profiling_enabled": plan.execution.profiling.enabled,
        "profiler_call_ranges": plan.execution.profiling.call_ranges,
        "profiler_configmap_name": rhoai_profiler_configmap_name(plan),
        "profiler_mount_path": RHOAI_PROFILER_MOUNT_PATH,
    }
    context["custom_epp_config_lines"] = _rhoai_custom_epp_config_lines(plan, context)
    return context


def render_rhoai_manifest(plan: ResolvedRunPlan) -> dict[str, Any]:
    if plan.deployment.mode not in {
        "distributed-default",
        "approximate-prefix-cache",
        "precise-prefix-cache",
    }:
        raise ValueError(f"unsupported RHOAI deployment mode: {plan.deployment.mode}")
    return render_jinja_yaml_document(
        "deployment/rhoai/llminferenceservice.yaml.j2",
        _rhoai_template_context(plan),
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
    cache_dir = plan.deployment.model_storage.cache_dir.rstrip("/")
    return f"{mount_root}{cache_dir}/{plan.model.pvc_directory_name}"


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
        "resources": _runtime_resource_requirements(plan, include_gpu=True),
        "volumeMounts": [
            {
                "name": "model-storage",
                "mountPath": plan.deployment.model_storage.mount_path,
            }
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
                        }
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

    return [deployment, service]


def write_deployment_assets(plan: ResolvedRunPlan, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if plan.deployment.platform == "llm-d":
        target = output_dir / "llm-d-values.yaml"
        target.write_text(
            yaml.safe_dump(render_llmd_values(plan), sort_keys=False), encoding="utf-8"
        )
        written.append(target)
        return written

    if plan.deployment.platform == "rhoai":
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
        manifests = render_rhaiis_raw_vllm_manifests(plan)
        names = ["deployment.yaml", "service.yaml"]
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
