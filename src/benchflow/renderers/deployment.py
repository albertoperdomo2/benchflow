from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..assets import asset_text, render_jinja_yaml_document
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
        },
        "options": plan.deployment.options,
    }


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


def _rhoai_template_context(plan: ResolvedRunPlan) -> dict[str, Any]:
    _validate_rhoai_profiling(plan)
    custom_scheduler_enabled = plan.deployment.mode in {
        "approximate-prefix-cache",
        "precise-prefix-cache",
    }
    return {
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
        "gpu_count": str(plan.deployment.runtime.tensor_parallelism),
        "custom_scheduler_enabled": custom_scheduler_enabled,
        "approximate_prefix_cache_enabled": (
            plan.deployment.mode == "approximate-prefix-cache"
        ),
        "precise_prefix_cache_enabled": plan.deployment.mode == "precise-prefix-cache",
        "profiling_enabled": plan.execution.profiling.enabled,
        "profiler_call_ranges": plan.execution.profiling.call_ranges,
        "profiler_configmap_name": rhoai_profiler_configmap_name(plan),
        "profiler_mount_path": RHOAI_PROFILER_MOUNT_PATH,
    }


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


def render_rhoai_raw_kserve_manifest(plan: ResolvedRunPlan) -> dict[str, Any]:
    _validate_rhoai_profiling(plan)
    if not plan.deployment.runtime.image:
        raise ValidationError(
            "rhoai raw-kserve deployments require deployment.runtime.image"
        )

    env = _rhoai_runtime_env(plan)
    env.append(
        {
            "name": "STORAGE_URI",
            "value": (
                f"pvc://{plan.deployment.model_storage.pvc_name}{_model_path(plan)}"
            ),
        }
    )
    if plan.execution.profiling.enabled:
        env.extend(
            [
                {
                    "name": "POD_NAME",
                    "valueFrom": {
                        "fieldRef": {
                            "apiVersion": "v1",
                            "fieldPath": "metadata.name",
                        }
                    },
                },
                {"name": "PYTHONPATH", "value": RHOAI_PROFILER_MOUNT_PATH},
                {
                    "name": "VLLM_PROFILER_RANGES",
                    "value": plan.execution.profiling.call_ranges,
                },
                {
                    "name": "VLLM_PROFILER_OUTPUT_DIR",
                    "value": RHOAI_PROFILER_OUTPUT_DIR,
                },
            ]
        )

    container_spec: dict[str, Any] = {
        "name": "kserve-container",
        "image": plan.deployment.runtime.image,
        "command": ["python3", "-m", "vllm.entrypoints.openai.api_server"],
        "args": [
            "--port=8000",
            "--host=0.0.0.0",
            "--model=/mnt/models",
            f"--served-model-name={plan.model.name}",
            f"--tensor-parallel-size={plan.deployment.runtime.tensor_parallelism}",
            *plan.deployment.runtime.vllm_args,
        ],
        "resources": {
            "limits": {
                "nvidia.com/gpu": str(plan.deployment.runtime.tensor_parallelism)
            },
            "requests": {
                "nvidia.com/gpu": str(plan.deployment.runtime.tensor_parallelism)
            },
        },
        "ports": [{"containerPort": 8000, "protocol": "TCP"}],
    }
    if env:
        container_spec["env"] = env
    if plan.execution.profiling.enabled:
        container_spec["volumeMounts"] = [
            {
                "mountPath": RHOAI_PROFILER_MOUNT_PATH,
                "name": "vllm-profiler",
                "readOnly": True,
            }
        ]

    predictor_spec: dict[str, Any] = {
        "minReplicas": plan.deployment.runtime.replicas,
        "maxReplicas": plan.deployment.runtime.replicas,
        "containers": [container_spec],
    }
    if plan.execution.profiling.enabled:
        predictor_spec["volumes"] = [
            {
                "name": "vllm-profiler",
                "configMap": {"name": rhoai_profiler_configmap_name(plan)},
            }
        ]

    return {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": plan.deployment.release_name,
            "namespace": plan.deployment.namespace,
            "labels": _base_labels(plan),
            "annotations": {
                "serving.kserve.io/deploymentMode": "RawDeployment",
                "serving.kserve.io/enable-prometheus-scraping": "true",
                "serving.knative.dev/autoscaleClass": "external",
            },
        },
        "spec": {
            "predictor": predictor_spec,
        },
    }


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


def render_rhaiis_manifests(plan: ResolvedRunPlan) -> list[dict[str, Any]]:
    storage_uri = f"pvc://{plan.deployment.model_storage.pvc_name}{_model_path(plan)}"
    runtime_name = plan.deployment.release_name
    args = [
        "--model=/mnt/models",
        f"--served-model-name={plan.model.name}",
        "--port=8080",
        f"--tensor-parallel-size={plan.deployment.runtime.tensor_parallelism}",
    ] + plan.deployment.runtime.vllm_args
    env = [
        {"name": key, "value": value}
        for key, value in sorted(plan.deployment.runtime.env.items())
    ]
    if plan.execution.profiling.enabled:
        env.extend(
            [
                {
                    "name": "POD_NAME",
                    "valueFrom": {
                        "fieldRef": {
                            "apiVersion": "v1",
                            "fieldPath": "metadata.name",
                        }
                    },
                },
                {"name": "PYTHONPATH", "value": RHOAI_PROFILER_MOUNT_PATH},
                {
                    "name": "VLLM_PROFILER_RANGES",
                    "value": plan.execution.profiling.call_ranges,
                },
                {
                    "name": "VLLM_PROFILER_OUTPUT_DIR",
                    "value": RHOAI_PROFILER_OUTPUT_DIR,
                },
            ]
        )

    container_spec: dict[str, Any] = {
        "name": "kserve-container",
        "image": plan.deployment.runtime.image,
        "command": ["python", "-m", "vllm.entrypoints.openai.api_server"],
        "args": args,
        "env": env,
        "ports": [{"containerPort": 8080, "protocol": "TCP"}],
    }
    volumes: list[dict[str, Any]] = []
    if plan.execution.profiling.enabled:
        container_spec["volumeMounts"] = [
            {
                "mountPath": RHOAI_PROFILER_MOUNT_PATH,
                "name": "vllm-profiler",
                "readOnly": True,
            }
        ]
        volumes.append(
            {
                "name": "vllm-profiler",
                "configMap": {"name": rhoai_profiler_configmap_name(plan)},
            }
        )

    serving_runtime = {
        "apiVersion": "serving.kserve.io/v1alpha1",
        "kind": "ServingRuntime",
        "metadata": {
            "name": runtime_name,
            "namespace": plan.deployment.namespace,
            "labels": _base_labels(plan),
        },
        "spec": {
            "containers": [container_spec],
            "multiModel": False,
            "supportedModelFormats": [{"name": "pytorch", "autoSelect": True}],
        },
    }
    if volumes:
        serving_runtime["spec"]["volumes"] = volumes

    inference_service = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": runtime_name,
            "namespace": plan.deployment.namespace,
            "labels": _base_labels(plan),
            "annotations": {
                "serving.kserve.io/deploymentMode": "RawDeployment",
                "serving.kserve.io/enable-prometheus-scraping": "true",
            },
        },
        "spec": {
            "predictor": {
                "minReplicas": plan.deployment.runtime.replicas,
                "model": {
                    "modelFormat": {"name": "pytorch"},
                    "runtime": runtime_name,
                    "storageUri": storage_uri,
                    "resources": {
                        "limits": {
                            "nvidia.com/gpu": str(
                                plan.deployment.runtime.tensor_parallelism
                            )
                        },
                        "requests": {
                            "nvidia.com/gpu": str(
                                plan.deployment.runtime.tensor_parallelism
                            )
                        },
                    },
                },
            }
        },
    }

    return [serving_runtime, inference_service]


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
        if plan.deployment.mode == "raw-kserve":
            target = output_dir / "inferenceservice.yaml"
            target.write_text(
                yaml.safe_dump(render_rhoai_raw_kserve_manifest(plan), sort_keys=False),
                encoding="utf-8",
            )
            written.append(target)
            if plan.execution.profiling.enabled:
                profiler_target = output_dir / "vllm-profiler-configmap.yaml"
                profiler_target.write_text(
                    yaml.safe_dump(
                        render_rhoai_profiler_configmap(plan), sort_keys=False
                    ),
                    encoding="utf-8",
                )
                written.append(profiler_target)
            return written
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

    manifests = render_rhaiis_manifests(plan)
    names = ["servingruntime.yaml", "inferenceservice.yaml"]
    for manifest, name in zip(manifests, names, strict=True):
        target = output_dir / name
        target.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
        written.append(target)
    return written
