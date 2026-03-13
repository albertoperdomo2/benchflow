from __future__ import annotations

from .loaders import ProfileCatalog
from .models import (
    Experiment,
    ProfileRefs,
    ResolvedDeployment,
    ResolvedRunPlan,
    TargetSpec,
    sanitize_name,
)


def _target_for(
    platform: str, release_name: str, namespace: str, gateway: str, path: str
) -> TargetSpec:
    if platform == "llm-d":
        if gateway == "standalone":
            base_url = f"http://ms-{release_name}.{namespace}.svc.cluster.local:8000"
        elif gateway == "kgateway":
            base_url = f"http://infra-{release_name}-inference-gateway.{namespace}.svc.cluster.local:80"
        else:
            base_url = f"http://infra-{release_name}-inference-gateway-istio.{namespace}.svc.cluster.local:80"
        return TargetSpec(discovery="static", base_url=base_url, path=path)

    if platform == "rhoai":
        return TargetSpec(
            discovery="llminferenceservice-status-url",
            resource_kind="LLMInferenceService",
            resource_name=release_name,
            path=path,
        )

    return TargetSpec(
        discovery="static",
        base_url=f"http://{release_name}-predictor.{namespace}.svc.cluster.local:8080",
        path=path,
    )


def resolve_run_plan(
    experiment: Experiment, catalog: ProfileCatalog
) -> ResolvedRunPlan:
    deployment_profile = catalog.require_deployment(experiment.spec.deployment_profile)
    benchmark_profile = catalog.require_benchmark(experiment.spec.benchmark_profile)
    metrics_profile = catalog.require_metrics(experiment.spec.metrics_profile)

    release_name = sanitize_name(experiment.metadata.name, max_length=42)
    namespace = deployment_profile.spec.namespace or experiment.spec.namespace

    target = _target_for(
        platform=deployment_profile.spec.platform,
        release_name=release_name,
        namespace=namespace,
        gateway=deployment_profile.spec.gateway,
        path=deployment_profile.spec.endpoint_path,
    )

    deployment = ResolvedDeployment(
        platform=deployment_profile.spec.platform,
        mode=deployment_profile.spec.mode,
        namespace=namespace,
        release_name=release_name,
        runtime=deployment_profile.spec.runtime,
        model_storage=deployment_profile.spec.model_storage,
        repo_url=deployment_profile.spec.repo_url,
        repo_ref=deployment_profile.spec.repo_ref,
        gateway=deployment_profile.spec.gateway,
        scheduler_profile=deployment_profile.spec.scheduler_profile,
        options=deployment_profile.spec.options,
        target=target,
    )

    tags = dict(experiment.spec.mlflow.tags)
    tags.setdefault("deployment_platform", deployment.platform)
    tags.setdefault("deployment_mode", deployment.mode)
    tags.setdefault("benchmark_profile", benchmark_profile.metadata.name)

    mlflow = experiment.spec.mlflow
    mlflow.tags = tags

    return ResolvedRunPlan(
        api_version="benchflow.io/v1alpha1",
        kind="RunPlan",
        metadata=experiment.metadata,
        profiles=ProfileRefs(
            deployment=deployment_profile.metadata.name,
            benchmark=benchmark_profile.metadata.name,
            metrics=metrics_profile.metadata.name,
        ),
        model=experiment.spec.model,
        deployment=deployment,
        benchmark=benchmark_profile.spec,
        metrics=metrics_profile.spec,
        stages=experiment.spec.stages,
        mlflow=mlflow,
        service_account=experiment.spec.service_account,
        ttl_seconds_after_finished=experiment.spec.ttl_seconds_after_finished,
    )
