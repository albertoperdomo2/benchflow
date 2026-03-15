from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..assets import render_jinja_yaml_document
from ..models import ResolvedRunPlan


def _base_labels(plan: ResolvedRunPlan) -> dict[str, str]:
    return {
        "app.kubernetes.io/name": "benchflow",
        "benchflow.io/experiment": plan.metadata.name,
        "benchflow.io/platform": plan.deployment.platform,
        "benchflow.io/mode": plan.deployment.mode,
    }


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
    return {
        "release_name": plan.deployment.release_name,
        "namespace": plan.deployment.namespace,
        "labels": _base_labels(plan),
        "enable_auth": str(plan.deployment.options.get("enable_auth", False)).lower(),
        "model_name": plan.model.name,
        "model_uri": f"pvc://{plan.deployment.model_storage.pvc_name}",
        "replicas": plan.deployment.runtime.replicas,
        "runtime_image": plan.deployment.runtime.image,
        "runtime_args": _rhoai_vllm_args(plan),
        "runtime_env": _rhoai_runtime_env(plan),
        "gpu_count": str(plan.deployment.runtime.tensor_parallelism),
        "precise_prefix_cache_enabled": plan.deployment.mode == "precise-prefix-cache",
    }


def render_rhoai_manifest(plan: ResolvedRunPlan) -> dict[str, Any]:
    if plan.deployment.mode not in {"kserve", "precise-prefix-cache"}:
        raise ValueError(f"unsupported RHOAI deployment mode: {plan.deployment.mode}")
    return render_jinja_yaml_document(
        "deployment/rhoai/llminferenceservice.yaml.j2",
        _rhoai_template_context(plan),
    )


def render_rhaiis_manifests(plan: ResolvedRunPlan) -> list[dict[str, Any]]:
    storage_uri = f"pvc://{plan.deployment.model_storage.pvc_name}{_model_path(plan)}"
    runtime_name = plan.deployment.release_name
    args = [
        "--model=/mnt/models",
        f"--served-model-name={plan.model.name}",
        "--port=8080",
        f"--tensor-parallel-size={plan.deployment.runtime.tensor_parallelism}",
    ] + plan.deployment.runtime.vllm_args

    serving_runtime = {
        "apiVersion": "serving.kserve.io/v1alpha1",
        "kind": "ServingRuntime",
        "metadata": {
            "name": runtime_name,
            "namespace": plan.deployment.namespace,
            "labels": _base_labels(plan),
        },
        "spec": {
            "containers": [
                {
                    "name": "kserve-container",
                    "image": plan.deployment.runtime.image,
                    "command": ["python", "-m", "vllm.entrypoints.openai.api_server"],
                    "args": args,
                    "ports": [{"containerPort": 8080, "protocol": "TCP"}],
                }
            ],
            "multiModel": False,
            "supportedModelFormats": [{"name": "pytorch", "autoSelect": True}],
        },
    }

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
