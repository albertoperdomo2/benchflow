from __future__ import annotations

from dataclasses import replace

from .loaders import ProfileCatalog
from .models import (
    Experiment,
    MlflowSpec,
    ProfileRefs,
    ResolvedDeployment,
    ResolvedRunPlan,
    RuntimeSpec,
    TargetSpec,
    ValidationError,
    normalize_model_names,
    normalize_profile_refs,
    sanitize_name,
)

_MATRIX_CHILD_INDEX_LABEL = "benchflow.io/matrix-child-index"
_MAX_MODEL_LEN_FLAG = "--max-model-len"


def _validate_profiling_support(*, platform: str, profiling_enabled: bool) -> None:
    if not profiling_enabled:
        return
    if platform != "rhoai":
        raise ValidationError(
            "execution.profiling is currently supported only for the rhoai platform"
        )


def _release_name_for(experiment: Experiment) -> str:
    child_index = str(
        (experiment.metadata.labels or {}).get(_MATRIX_CHILD_INDEX_LABEL) or ""
    ).strip()
    if not child_index:
        return sanitize_name(experiment.metadata.name, max_length=42)
    suffix = f"m{child_index}"
    prefix = sanitize_name(
        experiment.metadata.name,
        max_length=max(1, 42 - len(suffix) - 1),
    )
    return f"{prefix}-{suffix}"


def _target_for(
    platform: str, mode: str, release_name: str, namespace: str, gateway: str, path: str
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
        if mode == "raw-kserve":
            return TargetSpec(
                discovery="static",
                base_url=(
                    f"http://{release_name}-frontend.{namespace}.svc.cluster.local:8000"
                ),
                path=path,
            )
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


def _scalar_override(value, field_name: str):
    if isinstance(value, list):
        if len(value) != 1:
            raise ValidationError(
                f"{field_name} resolved to multiple values; expected a single combination"
            )
        return value[0]
    return value


def _parse_positive_int(value: str, *, field_name: str) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{field_name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValidationError(f"{field_name} must be a positive integer")
    return parsed


def _strip_max_model_len(
    args: list[str], *, field_name: str
) -> tuple[list[str], int | None]:
    cleaned: list[str] = []
    resolved: int | None = None
    index = 0
    while index < len(args):
        item = str(args[index]).strip()
        if item.startswith(f"{_MAX_MODEL_LEN_FLAG}="):
            resolved = _parse_positive_int(item.split("=", 1)[1], field_name=field_name)
            index += 1
            continue
        if item == _MAX_MODEL_LEN_FLAG:
            if index + 1 >= len(args):
                raise ValidationError(f"{field_name} is missing a value")
            resolved = _parse_positive_int(args[index + 1], field_name=field_name)
            index += 2
            continue
        cleaned.append(str(args[index]))
        index += 1
    return cleaned, resolved


def _resolve_vllm_args(
    *,
    deployment_args: list[str],
    override_args: list[str],
    benchmark_min_max_model_len: int | None,
) -> list[str]:
    base_args, deployment_max_model_len = _strip_max_model_len(
        deployment_args,
        field_name="deployment runtime max-model-len",
    )
    extra_args, override_max_model_len = _strip_max_model_len(
        override_args,
        field_name="spec.overrides.runtime.vllm_args max-model-len",
    )

    if (
        override_max_model_len is not None
        and benchmark_min_max_model_len is not None
        and override_max_model_len < benchmark_min_max_model_len
    ):
        raise ValidationError(
            "spec.overrides.runtime.vllm_args sets --max-model-len below "
            "benchmark requirements.min_max_model_len"
        )

    candidates = [
        value
        for value in (
            deployment_max_model_len,
            benchmark_min_max_model_len,
            override_max_model_len,
        )
        if value is not None
    ]
    resolved_max_model_len = max(candidates) if candidates else None

    resolved_args = [*base_args, *extra_args]
    if resolved_max_model_len is not None:
        resolved_args.append(f"{_MAX_MODEL_LEN_FLAG}={resolved_max_model_len}")
    return resolved_args


def resolve_run_plan(
    experiment: Experiment, catalog: ProfileCatalog
) -> ResolvedRunPlan:
    deployment_profile_names = normalize_profile_refs(
        experiment.spec.deployment_profile, "spec.deployment_profile"
    )
    benchmark_profile_names = normalize_profile_refs(
        experiment.spec.benchmark_profile, "spec.benchmark_profile"
    )
    metrics_profile_names = normalize_profile_refs(
        experiment.spec.metrics_profile, "spec.metrics_profile"
    )

    if len(deployment_profile_names) != 1:
        raise ValidationError(
            "resolve_run_plan requires exactly one deployment profile"
        )
    if len(benchmark_profile_names) != 1:
        raise ValidationError("resolve_run_plan requires exactly one benchmark profile")
    if len(metrics_profile_names) != 1:
        raise ValidationError("resolve_run_plan requires exactly one metrics profile")

    deployment_profile = catalog.require_deployment(deployment_profile_names[0])
    benchmark_profile = catalog.require_benchmark(benchmark_profile_names[0])
    metrics_profile = catalog.require_metrics(metrics_profile_names[0])
    model_names = normalize_model_names(experiment.spec.model.name, "spec.model.name")
    if len(model_names) != 1:
        raise ValidationError("resolve_run_plan requires exactly one model name")
    model_name = model_names[0]

    release_name = _release_name_for(experiment)
    namespace = deployment_profile.spec.namespace or experiment.spec.namespace
    overrides = experiment.spec.overrides

    runtime_image_override = _scalar_override(
        overrides.images.runtime, "spec.overrides.images.runtime"
    )
    scheduler_image_override = _scalar_override(
        overrides.images.scheduler, "spec.overrides.images.scheduler"
    )
    replicas_override = _scalar_override(
        overrides.scale.replicas, "spec.overrides.scale.replicas"
    )
    tp_override = _scalar_override(
        overrides.scale.tensor_parallelism, "spec.overrides.scale.tensor_parallelism"
    )

    runtime = RuntimeSpec(
        image=str(runtime_image_override or deployment_profile.spec.runtime.image),
        replicas=int(
            replicas_override
            if replicas_override is not None
            else deployment_profile.spec.runtime.replicas
        ),
        tensor_parallelism=int(
            tp_override
            if tp_override is not None
            else deployment_profile.spec.runtime.tensor_parallelism
        ),
        vllm_args=_resolve_vllm_args(
            deployment_args=deployment_profile.spec.runtime.vllm_args,
            override_args=experiment.spec.overrides.runtime.vllm_args,
            benchmark_min_max_model_len=benchmark_profile.spec.requirements.min_max_model_len,
        ),
        env={
            **deployment_profile.spec.runtime.env,
            **experiment.spec.overrides.runtime.env,
        },
    )
    _validate_profiling_support(
        platform=deployment_profile.spec.platform,
        profiling_enabled=experiment.spec.execution.profiling.enabled,
    )

    repo_ref = deployment_profile.spec.repo_ref
    if deployment_profile.spec.platform == "llm-d":
        repo_ref_override = _scalar_override(
            overrides.llm_d.repo_ref, "spec.overrides.llm_d.repo_ref"
        )
        if repo_ref_override:
            repo_ref = str(repo_ref_override)

    scheduler_image = str(
        scheduler_image_override or deployment_profile.spec.scheduler_image
    )
    if scheduler_image and deployment_profile.spec.platform not in {"llm-d", "rhoai"}:
        raise ValidationError(
            f"scheduler image override is not supported for platform "
            f"{deployment_profile.spec.platform!r}"
        )

    options = dict(deployment_profile.spec.options)
    if (
        deployment_profile.spec.platform == "rhoai"
        and overrides.rhoai.enable_auth is not None
    ):
        options["enable_auth"] = overrides.rhoai.enable_auth

    benchmark = replace(benchmark_profile.spec)
    if overrides.benchmark.rates is not None:
        benchmark.rates = list(overrides.benchmark.rates)
    if overrides.benchmark.max_seconds is not None:
        benchmark.max_seconds = overrides.benchmark.max_seconds
    if overrides.benchmark.max_requests is not None:
        benchmark.max_requests = overrides.benchmark.max_requests
    if overrides.benchmark.request_type is not None:
        benchmark.request_type = overrides.benchmark.request_type

    target = _target_for(
        platform=deployment_profile.spec.platform,
        mode=deployment_profile.spec.mode,
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
        runtime=runtime,
        model_storage=deployment_profile.spec.model_storage,
        repo_url=deployment_profile.spec.repo_url,
        repo_ref=repo_ref,
        gateway=deployment_profile.spec.gateway,
        scheduler_profile=deployment_profile.spec.scheduler_profile,
        scheduler_image=scheduler_image,
        options=options,
        target=target,
    )

    tags = dict(experiment.spec.mlflow.tags)
    tags.setdefault("deployment_type", f"{deployment.platform}-{deployment.mode}")
    tags.setdefault("deployment_profile", deployment_profile.metadata.name)
    tags.setdefault("benchmark_profile", benchmark_profile.metadata.name)
    tags.setdefault("metrics_profile", metrics_profile.metadata.name)

    model_name_fragment = (
        model_name.lower().replace("/", "-").replace(".", "").strip("-")
    )
    default_experiment_name = (
        f"{model_name_fragment}-{benchmark_profile.metadata.name}"
        if model_name_fragment
        else benchmark_profile.metadata.name
    )
    mlflow = MlflowSpec(
        experiment=experiment.spec.mlflow.experiment.strip() or default_experiment_name,
        version=experiment.spec.mlflow.version.strip(),
        tags=tags,
    )

    return ResolvedRunPlan(
        api_version="benchflow.io/v1alpha1",
        kind="RunPlan",
        metadata=experiment.metadata,
        profiles=ProfileRefs(
            deployment=deployment_profile.metadata.name,
            benchmark=benchmark_profile.metadata.name,
            metrics=metrics_profile.metadata.name,
        ),
        execution=experiment.spec.execution,
        target_cluster=experiment.spec.target_cluster,
        model=experiment.spec.model.__class__(name=model_name),
        deployment=deployment,
        benchmark=benchmark,
        metrics=metrics_profile.spec,
        stages=experiment.spec.stages,
        mlflow=mlflow,
        service_account=experiment.spec.service_account,
        ttl_seconds_after_finished=experiment.spec.ttl_seconds_after_finished,
    )
