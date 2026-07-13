from __future__ import annotations

from copy import deepcopy

from .loaders import ProfileCatalog
from .llmd_layout import uses_recipe_layout as _llmd_uses_recipe_layout
from .models import (
    Experiment,
    MlflowSpec,
    OverrideBenchmarkSpec,
    OverrideImagesSpec,
    OverrideLlmdSpec,
    OverrideRhoaiSpec,
    OverrideRuntimeSpec,
    OverrideScaleSpec,
    OverrideSpec,
    ProfileRefs,
    ResolvedDeployment,
    ResolvedRunPlan,
    RuntimeResourcesSpec,
    RuntimeSpec,
    StageSpec,
    TargetSpec,
    ValidationError,
    normalize_model_names,
    normalize_profile_refs,
    sanitize_name,
)

_MATRIX_CHILD_INDEX_LABEL = "benchflow.io/matrix-child-index"
_MAX_MODEL_LEN_FLAG = "--max-model-len"
_UNSUPPORTED_BENCHMARK_ENV = {
    "GUIDELLM_OUTPUT_PATH": (
        "GUIDELLM_OUTPUT_PATH is not supported in benchmark env overrides; "
        "BenchFlow manages exact benchmark output file paths. "
        "Use the BenchFlow-managed output directory or GUIDELLM_OUTPUT_DIR instead."
    )
}


def _guidellm_mapping(args: dict[str, object], key: str) -> dict[str, object]:
    value = args.get(key)
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        return {"kind": value.strip()}
    return {}


def _guidellm_load_field(profile: dict[str, object]) -> str:
    for field in ("streams", "rate", "max_concurrency", "sweep_size"):
        if field in profile:
            return field
    kind = str(profile.get("kind", "") or "").strip()
    if kind == "poisson":
        return "rate"
    if kind == "throughput":
        return "max_concurrency"
    return "streams"


def _set_guidellm_load_values(args: dict[str, object], values: list[int]) -> None:
    profile = _guidellm_mapping(args, "profile")
    profile.setdefault("kind", "concurrent")
    args["profile"] = profile
    args.pop("override", None)
    args.pop("overrides", None)
    args["override"] = {f"profile.{_guidellm_load_field(profile)}": list(values)}


def _set_guidellm_backend_field(
    args: dict[str, object],
    key: str,
    value: object,
) -> None:
    backend = _guidellm_mapping(args, "backend")
    backend.setdefault("kind", "openai_http")
    backend[key] = value
    args["backend"] = backend


def _set_aiperf_request_type(args: dict[str, object], request_type: str) -> None:
    cleaned = request_type.strip()
    normalized = cleaned.lower().replace("_", "-")
    if normalized in {
        "/v1/completions",
        "v1/completions",
        "completions",
        "completion",
        "text-completions",
        "text-completion",
    }:
        args["endpoint_type"] = "completions"
        args["endpoint_path"] = "/v1/completions"
        return
    if normalized in {
        "/v1/chat/completions",
        "v1/chat/completions",
        "chat",
        "chat-completions",
        "chat-completion",
    }:
        args["endpoint_type"] = "chat"
        args["endpoint_path"] = "/v1/chat/completions"
        return
    raise ValidationError(
        "unsupported AIPerf benchmark.request_type override: "
        f"{request_type!r}; supported values are /v1/completions and "
        "/v1/chat/completions"
    )


def _set_guidellm_constraint(
    args: dict[str, object],
    *,
    kind: str,
    field: str,
    value: object,
) -> None:
    existing = args.pop("constraint", args.pop("constraints", []))
    constraints = existing if isinstance(existing, list) else [existing]
    updated = [
        item
        for item in constraints
        if not (isinstance(item, dict) and item.get("kind") == kind)
    ]
    updated.append({"kind": kind, field: value})
    args["constraint"] = updated


def _image_tag(image: str) -> str:
    cleaned = str(image).strip()
    if not cleaned:
        return ""
    without_digest = cleaned.split("@", 1)[0]
    last_slash = without_digest.rfind("/")
    last_colon = without_digest.rfind(":")
    if last_colon <= last_slash:
        return ""
    return without_digest[last_colon + 1 :].strip()


def _rhaiis_platform_version(image: str) -> str:
    tag = _image_tag(image)
    if not tag:
        return ""
    return f"RHAIIS-{tag}"


def _validate_profiling_support(*, platform: str, profiling_enabled: bool) -> None:
    if not profiling_enabled:
        return
    if platform != "rhoai":
        raise ValidationError(
            "execution.profiling is currently supported only for the rhoai platform"
        )


def _validate_existing_target_support(experiment: Experiment) -> None:
    if not experiment.spec.target.enabled():
        return
    return


def _resolved_stage_spec(experiment: Experiment) -> StageSpec:
    if not experiment.spec.target.enabled():
        return experiment.spec.stages
    return StageSpec(
        download=False,
        deploy=False,
        benchmark=experiment.spec.stages.benchmark,
        collect=bool(str(experiment.spec.target.metrics_release_name or "").strip()),
        cleanup=False,
    )


def _validate_benchmark_env(env: dict[str, str]) -> None:
    for name, message in _UNSUPPORTED_BENCHMARK_ENV.items():
        if str(env.get(name, "")).strip():
            raise ValidationError(message)


def _validate_runtime_volume_mounts(runtime: RuntimeSpec) -> None:
    names: dict[str, str] = {}
    mount_paths: dict[str, str] = {}
    entries = [
        *(
            (host_path.name, host_path.mount_path, "runtime.host_paths")
            for host_path in runtime.host_paths
        ),
        *(
            (pvc_mount.name, pvc_mount.mount_path, "runtime.pvc_mounts")
            for pvc_mount in runtime.pvc_mounts
        ),
    ]
    for name, mount_path, source in entries:
        if name in names:
            raise ValidationError(
                f"{source} volume name {name!r} conflicts with {names[name]}"
            )
        if mount_path in mount_paths:
            raise ValidationError(
                f"{source} mount_path {mount_path!r} conflicts with "
                f"{mount_paths[mount_path]}"
            )
        names[name] = source
        mount_paths[mount_path] = source


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
    platform: str,
    mode: str,
    release_name: str,
    namespace: str,
    gateway: str,
    path: str,
    repo_ref: str,
    endpoint_scope: str,
) -> TargetSpec:
    if platform == "llm-d":
        if gateway == "standalone":
            if _llmd_uses_recipe_layout(repo_ref):
                base_url = (
                    f"http://gaie-{release_name}-epp.{namespace}.svc.cluster.local:80"
                )
            else:
                base_url = (
                    f"http://ms-{release_name}.{namespace}.svc.cluster.local:8000"
                )
        else:
            gateway_name = (
                "llm-d-inference-gateway"
                if _llmd_uses_recipe_layout(repo_ref)
                else f"infra-{release_name}-inference-gateway"
            )
            return TargetSpec(
                discovery="gateway-status-url",
                resource_kind="Gateway",
                resource_name=gateway_name,
                path=path,
                endpoint_scope=endpoint_scope,
            )
        return TargetSpec(
            discovery="static",
            base_url=base_url,
            path=path,
            endpoint_scope=endpoint_scope,
        )

    if platform == "rhoai":
        if mode == "isvc":
            return TargetSpec(
                discovery="inferenceservice-status-url",
                resource_kind="InferenceService",
                resource_name=release_name,
                path=path,
                endpoint_scope=endpoint_scope,
            )
        return TargetSpec(
            discovery="llminferenceservice-status-url",
            resource_kind="LLMInferenceService",
            resource_name=release_name,
            path=path,
            endpoint_scope=endpoint_scope,
        )

    if platform == "rhaiis" and mode == "raw-vllm":
        return TargetSpec(
            discovery="static",
            base_url=f"http://{release_name}.{namespace}.svc.cluster.local:8000",
            path=path,
            endpoint_scope=endpoint_scope,
        )

    return TargetSpec(
        discovery="static",
        base_url=f"http://{release_name}-predictor.{namespace}.svc.cluster.local:8080",
        path=path,
        endpoint_scope=endpoint_scope,
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
    benchmark_min_max_model_len: int | None,
) -> list[str]:
    base_args, deployment_max_model_len = _strip_max_model_len(
        deployment_args,
        field_name="deployment runtime max-model-len",
    )
    candidates = [
        value
        for value in (
            deployment_max_model_len,
            benchmark_min_max_model_len,
        )
        if value is not None
    ]
    resolved_max_model_len = max(candidates) if candidates else None

    resolved_args = list(base_args)
    if resolved_max_model_len is not None:
        resolved_args.append(f"{_MAX_MODEL_LEN_FLAG}={resolved_max_model_len}")
    return resolved_args


def _scalar_model_override(value, fallback):
    return deepcopy(value) if value is not None else deepcopy(fallback)


def _optional_env_override(
    base: dict[str, str] | None, model: dict[str, str] | None
) -> dict[str, str] | None:
    if model is None:
        return deepcopy(base) if base is not None else None
    return {**(base or {}), **model}


def _merge_model_override(
    base: OverrideSpec, model_override: OverrideSpec | None
) -> OverrideSpec:
    if model_override is None:
        return deepcopy(base)
    return OverrideSpec(
        images=OverrideImagesSpec(
            runtime=_scalar_model_override(
                model_override.images.runtime, base.images.runtime
            ),
            scheduler=_scalar_model_override(
                model_override.images.scheduler, base.images.scheduler
            ),
        ),
        scale=OverrideScaleSpec(
            replicas=_scalar_model_override(
                model_override.scale.replicas, base.scale.replicas
            ),
            tensor_parallelism=_scalar_model_override(
                model_override.scale.tensor_parallelism,
                base.scale.tensor_parallelism,
            ),
        ),
        runtime=OverrideRuntimeSpec(
            env={**base.runtime.env, **model_override.runtime.env},
            vllm_args=(
                list(model_override.runtime.vllm_args)
                if model_override.runtime.vllm_args is not None
                else (
                    list(base.runtime.vllm_args)
                    if base.runtime.vllm_args is not None
                    else None
                )
            ),
            vllm_extra_args=[
                *base.runtime.vllm_extra_args,
                *model_override.runtime.vllm_extra_args,
            ],
            host_paths=_scalar_model_override(
                model_override.runtime.host_paths,
                base.runtime.host_paths,
            ),
            service_account_name=_scalar_model_override(
                model_override.runtime.service_account_name,
                base.runtime.service_account_name,
            ),
            node_selector=_scalar_model_override(
                model_override.runtime.node_selector,
                base.runtime.node_selector,
            ),
            affinity=_scalar_model_override(
                model_override.runtime.affinity,
                base.runtime.affinity,
            ),
            placement=_scalar_model_override(
                model_override.runtime.placement,
                base.runtime.placement,
            ),
            tolerations=_scalar_model_override(
                model_override.runtime.tolerations,
                base.runtime.tolerations,
            ),
            resources=_scalar_model_override(
                model_override.runtime.resources,
                base.runtime.resources,
            ),
        ),
        benchmark=OverrideBenchmarkSpec(
            rates=_scalar_model_override(
                model_override.benchmark.rates,
                base.benchmark.rates,
            ),
            max_seconds=_scalar_model_override(
                model_override.benchmark.max_seconds,
                base.benchmark.max_seconds,
            ),
            max_requests=_scalar_model_override(
                model_override.benchmark.max_requests,
                base.benchmark.max_requests,
            ),
            request_type=_scalar_model_override(
                model_override.benchmark.request_type,
                base.benchmark.request_type,
            ),
            env=_optional_env_override(
                base.benchmark.env,
                model_override.benchmark.env,
            ),
        ),
        llm_d=OverrideLlmdSpec(
            repo_ref=_scalar_model_override(
                model_override.llm_d.repo_ref,
                base.llm_d.repo_ref,
            ),
        ),
        rhoai=OverrideRhoaiSpec(
            enable_auth=_scalar_model_override(
                model_override.rhoai.enable_auth,
                base.rhoai.enable_auth,
            ),
        ),
    )


def _resolve_runtime_resources(
    profile: RuntimeResourcesSpec, override: RuntimeResourcesSpec | None
) -> RuntimeResourcesSpec:
    if override is None:
        return deepcopy(profile)
    requests = dict(profile.requests)
    limits = dict(profile.limits)
    for key in override.remove_requests:
        requests.pop(key, None)
    for key in override.remove_limits:
        limits.pop(key, None)
    requests.update(override.requests)
    limits.update(override.limits)
    return RuntimeResourcesSpec(
        requests=requests,
        limits=limits,
        remove_requests=list(
            dict.fromkeys([*profile.remove_requests, *override.remove_requests])
        ),
        remove_limits=list(
            dict.fromkeys([*profile.remove_limits, *override.remove_limits])
        ),
    )


def resolve_run_plan(
    experiment: Experiment, catalog: ProfileCatalog
) -> ResolvedRunPlan:
    _validate_existing_target_support(experiment)
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
    overrides = _merge_model_override(
        experiment.spec.overrides,
        experiment.spec.model_overrides.get(model_name),
    )

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
            deployment_args=[
                *(
                    overrides.runtime.vllm_args
                    if overrides.runtime.vllm_args is not None
                    else deployment_profile.spec.runtime.vllm_args
                ),
                *overrides.runtime.vllm_extra_args,
            ],
            benchmark_min_max_model_len=benchmark_profile.spec.requirements.min_max_model_len,
        ),
        env={
            **deployment_profile.spec.runtime.env,
            **overrides.runtime.env,
        },
        shared_memory_size=deployment_profile.spec.runtime.shared_memory_size,
        host_paths=(
            deepcopy(overrides.runtime.host_paths)
            if overrides.runtime.host_paths is not None
            else deepcopy(deployment_profile.spec.runtime.host_paths)
        ),
        pvc_mounts=deepcopy(deployment_profile.spec.runtime.pvc_mounts),
        service_account_name=str(
            overrides.runtime.service_account_name
            or deployment_profile.spec.runtime.service_account_name
        ).strip(),
        node_selector=(
            dict(overrides.runtime.node_selector)
            if overrides.runtime.node_selector is not None
            else dict(deployment_profile.spec.runtime.node_selector)
        ),
        affinity=(
            deepcopy(overrides.runtime.affinity)
            if overrides.runtime.affinity is not None
            else deepcopy(deployment_profile.spec.runtime.affinity)
        ),
        placement=(
            deepcopy(overrides.runtime.placement)
            if overrides.runtime.placement is not None
            else deepcopy(deployment_profile.spec.runtime.placement)
        ),
        tolerations=(
            deepcopy(overrides.runtime.tolerations)
            if overrides.runtime.tolerations is not None
            else deepcopy(deployment_profile.spec.runtime.tolerations)
        ),
        image_pull_secrets=deepcopy(deployment_profile.spec.runtime.image_pull_secrets),
        resources=(
            _resolve_runtime_resources(
                deployment_profile.spec.runtime.resources,
                overrides.runtime.resources,
            )
        ),
    )
    if runtime.placement.mode and deployment_profile.spec.platform != "rhoai":
        raise ValidationError(
            "runtime.placement is currently supported only for rhoai LLMInferenceService deployments"
        )
    if runtime.placement.mode and deployment_profile.spec.mode == "isvc":
        raise ValidationError(
            "runtime.placement is not supported for rhoai isvc deployments"
        )
    if runtime.host_paths and deployment_profile.spec.platform not in {
        "llm-d",
        "rhoai",
        "rhaiis",
    }:
        raise ValidationError(
            "runtime.host_paths is currently supported only for llm-d, rhoai, and rhaiis deployments"
        )
    if runtime.pvc_mounts and deployment_profile.spec.platform not in {
        "llm-d",
        "rhoai",
        "rhaiis",
    }:
        raise ValidationError(
            "runtime.pvc_mounts is currently supported only for llm-d, rhoai, and rhaiis deployments"
        )
    _validate_runtime_volume_mounts(runtime)
    _validate_profiling_support(
        platform=deployment_profile.spec.platform,
        profiling_enabled=experiment.spec.execution.profiling.enabled,
    )

    repo_ref = deployment_profile.spec.repo_ref
    platform_version = str(deployment_profile.spec.platform_version or "").strip()
    if deployment_profile.spec.platform == "llm-d":
        repo_ref_override = _scalar_override(
            overrides.llm_d.repo_ref, "spec.overrides.llm_d.repo_ref"
        )
        if repo_ref_override:
            repo_ref = str(repo_ref_override)
        if not platform_version:
            platform_version = repo_ref
    elif deployment_profile.spec.platform == "rhaiis" and not platform_version:
        platform_version = _rhaiis_platform_version(runtime.image)

    scheduler_image = str(
        scheduler_image_override or deployment_profile.spec.scheduler_image
    )
    if scheduler_image and deployment_profile.spec.platform not in {"llm-d", "rhoai"}:
        raise ValidationError(
            f"scheduler image override is not supported for platform "
            f"{deployment_profile.spec.platform!r}"
        )
    if (
        scheduler_image
        and deployment_profile.spec.platform == "rhoai"
        and deployment_profile.spec.mode == "isvc"
    ):
        raise ValidationError(
            "scheduler image override is not supported for rhoai isvc deployments"
        )

    options = dict(deployment_profile.spec.options)
    if (
        deployment_profile.spec.platform == "rhoai"
        and overrides.rhoai.enable_auth is not None
    ):
        options["enable_auth"] = overrides.rhoai.enable_auth

    benchmark = deepcopy(benchmark_profile.spec)
    if overrides.benchmark.rates is not None:
        if benchmark.tool == "guidellm":
            _set_guidellm_load_values(
                benchmark.guidellm.args,
                list(overrides.benchmark.rates),
            )
    if overrides.benchmark.max_seconds is not None:
        if benchmark.tool == "aiperf":
            benchmark.aiperf.max_seconds = overrides.benchmark.max_seconds
        else:
            _set_guidellm_constraint(
                benchmark.guidellm.args,
                kind="max_duration",
                field="seconds",
                value=overrides.benchmark.max_seconds,
            )
    if overrides.benchmark.max_requests is not None:
        if benchmark.tool == "guidellm":
            _set_guidellm_constraint(
                benchmark.guidellm.args,
                kind="max_requests",
                field="count",
                value=overrides.benchmark.max_requests,
            )
    if overrides.benchmark.request_type is not None:
        if benchmark.tool == "guidellm":
            _set_guidellm_backend_field(
                benchmark.guidellm.args,
                "request_format",
                overrides.benchmark.request_type,
            )
        elif benchmark.tool == "aiperf":
            _set_aiperf_request_type(
                benchmark.aiperf.args,
                overrides.benchmark.request_type,
            )
    if overrides.benchmark.env is not None:
        benchmark.env = {
            **benchmark_profile.spec.env,
            **overrides.benchmark.env,
        }
    _validate_benchmark_env(benchmark.env)

    target = (
        TargetSpec(
            discovery="static",
            base_url=experiment.spec.target.base_url.rstrip("/"),
            path=experiment.spec.target.path,
            metrics_release_name=experiment.spec.target.metrics_release_name,
            endpoint_scope=experiment.spec.target.endpoint_scope or "external",
        )
        if experiment.spec.target.enabled()
        else _target_for(
            platform=deployment_profile.spec.platform,
            mode=deployment_profile.spec.mode,
            release_name=release_name,
            namespace=namespace,
            gateway=deployment_profile.spec.gateway,
            path=deployment_profile.spec.endpoint_path,
            repo_ref=repo_ref,
            endpoint_scope=(
                experiment.spec.target.endpoint_scope
                or deployment_profile.spec.endpoint_scope
            ),
        )
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
        platform_version=platform_version,
        platform_channel=deployment_profile.spec.platform_channel,
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
        stages=_resolved_stage_spec(experiment),
        mlflow=mlflow,
        service_account=experiment.spec.service_account,
        ttl_seconds_after_finished=experiment.spec.ttl_seconds_after_finished,
    )
