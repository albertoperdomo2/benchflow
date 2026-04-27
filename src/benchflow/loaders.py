from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .models import (
    AiperfBenchmarkSpec,
    BenchmarkProfile,
    BenchmarkRequirementsSpec,
    BenchmarkProfileSpec,
    ClusterTargetSpec,
    DeploymentProfile,
    DeploymentProfileSpec,
    ExecutionSpec,
    Experiment,
    ExperimentSpec,
    ExperimentTargetSpec,
    GuidellmBenchmarkSpec,
    MetricsProfile,
    MetricsProfileSpec,
    MlflowSpec,
    ModelStorageSpec,
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
    _require,
    _as_bool,
    normalize_model_names,
    normalize_profile_refs,
    parse_metadata,
    parse_model_spec,
)

_AIPERF_REQUIRED_FIELDS = {
    "dataset_url",
    "dataset_type",
    "endpoint_type",
}


def _string_or_list(raw: Any, field_name: str) -> str | list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        cleaned = raw.strip()
        return cleaned or None
    if isinstance(raw, list):
        values = [str(item).strip() for item in raw if str(item).strip()]
        if not values:
            raise ValidationError(f"{field_name} must not be an empty list")
        return values
    raise ValidationError(
        f"{field_name} must be a string or list of strings, got: {raw!r}"
    )


def _int_or_list(raw: Any, field_name: str) -> int | list[int] | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ValidationError(f"{field_name} must be an integer or list of integers")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, list):
        try:
            values = [int(item) for item in raw]
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"{field_name} must be a list of integers, got: {raw!r}"
            ) from exc
        if not values:
            raise ValidationError(f"{field_name} must not be an empty list")
        return values
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"{field_name} must be an integer or list of integers, got: {raw!r}"
        ) from exc


def _int_list(raw: Any, field_name: str) -> list[int] | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ValidationError(f"{field_name} must be an integer or list of integers")
    if isinstance(raw, int):
        if raw <= 0:
            raise ValidationError(f"{field_name} must contain only positive integers")
        return [raw]
    if isinstance(raw, list):
        try:
            values = [int(item) for item in raw]
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"{field_name} must be a list of integers, got: {raw!r}"
            ) from exc
        if not values:
            raise ValidationError(f"{field_name} must not be an empty list")
        if any(value <= 0 for value in values):
            raise ValidationError(f"{field_name} must contain only positive integers")
        return values
    try:
        parsed = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"{field_name} must be an integer or list of integers, got: {raw!r}"
        ) from exc
    if parsed <= 0:
        raise ValidationError(f"{field_name} must contain only positive integers")
    return [parsed]


def _positive_int(raw: Any, field_name: str) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ValidationError(f"{field_name} must be a positive integer")
    try:
        parsed = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{field_name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValidationError(f"{field_name} must be a positive integer")
    return parsed


def _optional_positive_int(raw: Any, field_name: str) -> int | None:
    if raw is None or str(raw).strip() == "":
        return None
    return _positive_int(raw, field_name)


def _nonempty_string(raw: Any, field_name: str) -> str | None:
    if raw is None:
        return None
    cleaned = str(raw).strip()
    if not cleaned:
        raise ValidationError(f"{field_name} must not be empty")
    return cleaned


def _string_mapping(raw: Any, field_name: str) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValidationError(f"{field_name} must be a mapping")
    return {
        str(key): str(value)
        for key, value in raw.items()
        if str(key).strip() and str(value).strip()
    }


def _mapping(raw: Any, field_name: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValidationError(f"{field_name} must be a mapping")
    return dict(raw)


def _resource_mapping(raw: Any, field_name: str) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValidationError(f"{field_name} must be a mapping")
    values: dict[str, str] = {}
    for key, value in raw.items():
        cleaned_key = str(key).strip()
        cleaned_value = str(value).strip()
        if cleaned_key and cleaned_value:
            values[cleaned_key] = cleaned_value
    return values


def _runtime_resources_from_dict(
    raw: dict[str, Any] | None, field_name: str
) -> RuntimeResourcesSpec:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValidationError(f"{field_name} must be a mapping")
    return RuntimeResourcesSpec(
        requests=_resource_mapping(raw.get("requests"), f"{field_name}.requests"),
        limits=_resource_mapping(raw.get("limits"), f"{field_name}.limits"),
    )


def _mapping_list(raw: Any, field_name: str) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValidationError(f"{field_name} must be a list")
    values: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValidationError(f"{field_name}[{index}] must be a mapping")
        values.append(dict(item))
    return values


def _local_object_reference_list(raw: Any, field_name: str) -> list[dict[str, str]]:
    values = _mapping_list(raw, field_name)
    refs: list[dict[str, str]] = []
    for index, item in enumerate(values):
        name = str(item.get("name", "") or "").strip()
        if not name:
            raise ValidationError(f"{field_name}[{index}].name must not be empty")
        refs.append({"name": name})
    return refs


def _overrides_from_dict(raw: dict[str, Any] | None) -> OverrideSpec:
    raw = raw or {}
    images = raw.get("images") or {}
    scale = raw.get("scale") or {}
    runtime = raw.get("runtime") or {}
    benchmark = raw.get("benchmark") or {}
    llm_d = raw.get("llm_d") or {}
    rhoai = raw.get("rhoai") or {}

    return OverrideSpec(
        images=OverrideImagesSpec(
            runtime=_string_or_list(
                images.get("runtime"), "spec.overrides.images.runtime"
            ),
            scheduler=_string_or_list(
                images.get("scheduler"), "spec.overrides.images.scheduler"
            ),
        ),
        scale=OverrideScaleSpec(
            replicas=_int_or_list(
                scale.get("replicas"), "spec.overrides.scale.replicas"
            ),
            tensor_parallelism=_int_or_list(
                scale.get("tensor_parallelism"),
                "spec.overrides.scale.tensor_parallelism",
            ),
        ),
        runtime=OverrideRuntimeSpec(
            vllm_args=[str(item) for item in (runtime.get("vllm_args") or [])],
            env={
                str(key): str(value)
                for key, value in (runtime.get("env") or {}).items()
            },
            node_selector=(
                _string_mapping(
                    runtime.get("node_selector"),
                    "spec.overrides.runtime.node_selector",
                )
                if "node_selector" in runtime
                else None
            ),
            affinity=(
                _mapping(runtime.get("affinity"), "spec.overrides.runtime.affinity")
                if "affinity" in runtime
                else None
            ),
            tolerations=(
                _mapping_list(
                    runtime.get("tolerations"),
                    "spec.overrides.runtime.tolerations",
                )
                if "tolerations" in runtime
                else None
            ),
            resources=(
                _runtime_resources_from_dict(
                    runtime.get("resources"), "spec.overrides.runtime.resources"
                )
                if "resources" in runtime
                else None
            ),
        ),
        benchmark=OverrideBenchmarkSpec(
            rates=_int_list(benchmark.get("rates"), "spec.overrides.benchmark.rates"),
            max_seconds=_positive_int(
                benchmark.get("max_seconds"),
                "spec.overrides.benchmark.max_seconds",
            ),
            max_requests=_nonempty_string(
                benchmark.get("max_requests"),
                "spec.overrides.benchmark.max_requests",
            ),
            request_type=_nonempty_string(
                benchmark.get("request_type"),
                "spec.overrides.benchmark.request_type",
            ),
            env=(
                _string_mapping(
                    benchmark.get("env"),
                    "spec.overrides.benchmark.env",
                )
                if "env" in benchmark
                else None
            ),
        ),
        llm_d=OverrideLlmdSpec(
            repo_ref=_string_or_list(
                llm_d.get("repo_ref"), "spec.overrides.llm_d.repo_ref"
            )
        ),
        rhoai=OverrideRhoaiSpec(
            enable_auth=(
                _as_bool(rhoai.get("enable_auth"), False)
                if "enable_auth" in rhoai
                else None
            )
        ),
    )


def _target_cluster_from_dict(raw: dict[str, Any] | None) -> ClusterTargetSpec:
    raw = raw or {}
    host_aliases_raw = raw.get("host_aliases") or {}
    if not isinstance(host_aliases_raw, dict):
        raise ValidationError("target_cluster.host_aliases must be a mapping")
    return ClusterTargetSpec(
        kubeconfig=str(raw.get("kubeconfig", "") or ""),
        kubeconfig_secret=str(raw.get("kubeconfig_secret", "") or ""),
        host_aliases={
            str(hostname).strip(): str(ip_address).strip()
            for hostname, ip_address in host_aliases_raw.items()
            if str(hostname).strip() and str(ip_address).strip()
        },
    )


def _experiment_target_from_dict(raw: dict[str, Any] | None) -> ExperimentTargetSpec:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValidationError("target must be a mapping")
    base_url = str(raw.get("base_url", "") or "").strip()
    path = str(raw.get("path", "/v1/models") or "/v1/models").strip()
    metrics_release_name = str(raw.get("metrics_release_name", "") or "").strip()
    if not path:
        raise ValidationError("target.path must not be empty")
    if raw and not base_url:
        raise ValidationError("target.base_url must not be empty")
    return ExperimentTargetSpec(
        base_url=base_url,
        path=path,
        metrics_release_name=metrics_release_name,
    )


def load_yaml_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValidationError(f"{path} does not contain a mapping document")
    return data


def _runtime_from_dict(raw: dict[str, Any] | None) -> RuntimeSpec:
    raw = raw or {}
    env = {str(key): str(value) for key, value in (raw.get("env") or {}).items()}
    image_pull_secrets = raw.get("image_pull_secrets")
    if image_pull_secrets is None:
        image_pull_secrets = raw.get("imagePullSecrets")
    return RuntimeSpec(
        image=str(raw.get("image", "")),
        replicas=int(raw.get("replicas", 1)),
        tensor_parallelism=int(raw.get("tensor_parallelism", 1)),
        vllm_args=[str(item) for item in (raw.get("vllm_args") or [])],
        env=env,
        node_selector=_string_mapping(
            raw.get("node_selector"), "spec.runtime.node_selector"
        ),
        affinity=_mapping(raw.get("affinity"), "spec.runtime.affinity"),
        tolerations=_mapping_list(raw.get("tolerations"), "spec.runtime.tolerations"),
        image_pull_secrets=_local_object_reference_list(
            image_pull_secrets, "spec.runtime.image_pull_secrets"
        ),
        resources=_runtime_resources_from_dict(
            raw.get("resources"), "spec.runtime.resources"
        ),
    )


def _storage_from_dict(raw: dict[str, Any] | None) -> ModelStorageSpec:
    raw = raw or {}
    return ModelStorageSpec(
        pvc_name=str(raw.get("pvc_name", "models-storage")),
        cache_dir=str(raw.get("cache_dir", "/models")),
        mount_path=str(raw.get("mount_path", "/model-cache")),
    )


def _benchmark_requirements_from_dict(
    raw: dict[str, Any] | None,
) -> BenchmarkRequirementsSpec:
    raw = raw or {}
    min_max_model_len = raw.get("min_max_model_len")
    if min_max_model_len is None:
        return BenchmarkRequirementsSpec()
    try:
        resolved = int(min_max_model_len)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "spec.requirements.min_max_model_len must be an integer"
        ) from exc
    if resolved <= 0:
        raise ValidationError(
            "spec.requirements.min_max_model_len must be greater than zero"
        )
    return BenchmarkRequirementsSpec(min_max_model_len=resolved)


def load_experiment(path: Path) -> Experiment:
    raw = load_yaml_file(path)
    if raw.get("kind") != "Experiment":
        raise ValidationError(f"{path} is not an Experiment")

    metadata = parse_metadata(raw)
    spec = raw.get("spec") or {}

    experiment_spec = ExperimentSpec(
        model=parse_model_spec(spec.get("model") or {}),
        deployment_profile=normalize_profile_refs(
            spec.get("deployment_profile") or "", "spec.deployment_profile"
        ),
        benchmark_profile=normalize_profile_refs(
            spec.get("benchmark_profile") or "", "spec.benchmark_profile"
        ),
        metrics_profile=normalize_profile_refs(
            spec.get("metrics_profile", "detailed"), "spec.metrics_profile"
        ),
        namespace=str(spec.get("namespace", "benchflow")),
        service_account=str(spec.get("service_account", "benchflow-runner")),
        ttl_seconds_after_finished=int(spec.get("ttl_seconds_after_finished", 3600)),
        stages=StageSpec.from_dict(spec.get("stages")),
        mlflow=MlflowSpec.from_dict(spec.get("mlflow")),
        execution=ExecutionSpec.from_dict(spec.get("execution")),
        target=_experiment_target_from_dict(spec.get("target")),
        target_cluster=_target_cluster_from_dict(spec.get("target_cluster")),
        overrides=_overrides_from_dict(spec.get("overrides")),
    )

    return Experiment(
        api_version=str(raw.get("apiVersion", "benchflow.io/v1alpha1")),
        kind="Experiment",
        metadata=metadata,
        spec=experiment_spec,
    )


def _guidellm_benchmark_from_dict(raw: dict[str, Any]) -> GuidellmBenchmarkSpec:
    return GuidellmBenchmarkSpec(
        backend_type=str(raw.get("backend_type", "openai_http")),
        request_type=str(raw.get("request_type", "") or "").strip(),
        rate_type=str(raw.get("rate_type", "concurrent")),
        rates=[int(item) for item in (raw.get("rates") or [])],
        data=str(raw.get("data", "prompt_tokens=1000,output_tokens=1000")),
        max_seconds=int(raw.get("max_seconds", 600)),
        max_requests=str(raw["max_requests"]).strip()
        if raw.get("max_requests") is not None
        else None,
    )


def _aiperf_benchmark_from_dict(raw: dict[str, Any]) -> AiperfBenchmarkSpec:
    missing = [
        field_name
        for field_name in sorted(_AIPERF_REQUIRED_FIELDS)
        if not str(raw.get(field_name, "") or "").strip()
    ]
    if missing:
        joined = ", ".join(f"spec.aiperf.{field_name}" for field_name in missing)
        raise ValidationError(f"aiperf benchmark profile is missing {joined}")

    return AiperfBenchmarkSpec(
        dataset_url=str(raw.get("dataset_url", "") or "").strip(),
        dataset_name=str(raw.get("dataset_name", "") or "").strip(),
        dataset_type=str(raw.get("dataset_type", "") or "").strip(),
        endpoint_type=str(raw.get("endpoint_type", "") or "").strip(),
        endpoint_path=str(raw.get("endpoint_path", "") or "").strip(),
        tokenizer=str(raw.get("tokenizer", "") or "").strip(),
        streaming=_as_bool(raw.get("streaming"), True),
        fixed_schedule=_as_bool(raw.get("fixed_schedule"), True),
        fixed_schedule_auto_offset=_as_bool(
            raw.get("fixed_schedule_auto_offset"), True
        ),
        synthesis_max_isl=_optional_positive_int(
            raw.get("synthesis_max_isl"), "spec.aiperf.synthesis_max_isl"
        ),
        fixed_schedule_end_offset=_optional_positive_int(
            raw.get("fixed_schedule_end_offset"),
            "spec.aiperf.fixed_schedule_end_offset",
        ),
        dataset_cap=_optional_positive_int(
            raw.get("dataset_cap"), "spec.aiperf.dataset_cap"
        ),
        export_level=str(raw.get("export_level", "") or "").strip(),
        export_http_trace=_as_bool(raw.get("export_http_trace"), False),
        max_seconds=int(raw.get("max_seconds", 7200)),
    )


def _benchmark_profile_spec_from_dict(raw: dict[str, Any]) -> BenchmarkProfileSpec:
    tool = str(raw.get("tool", "guidellm") or "guidellm").strip()
    if tool not in {"guidellm", "aiperf"}:
        raise ValidationError(
            f"unsupported benchmark tool: {tool}; supported tools are guidellm and aiperf"
        )
    env = {str(key): str(value) for key, value in (raw.get("env") or {}).items()}
    guidellm_raw = raw.get("guidellm")
    if guidellm_raw is None:
        guidellm_raw = raw
    if not isinstance(guidellm_raw, dict):
        raise ValidationError("spec.guidellm must be a mapping")

    aiperf_raw = raw.get("aiperf")
    if aiperf_raw is None and raw.get("options") is not None:
        aiperf_raw = {
            **dict(raw.get("options") or {}),
            "max_seconds": raw.get("max_seconds", 7200),
        }
    if aiperf_raw is None:
        aiperf_raw = {}
    if not isinstance(aiperf_raw, dict):
        raise ValidationError("spec.aiperf must be a mapping")

    return BenchmarkProfileSpec(
        tool=tool,
        env=env,
        guidellm=_guidellm_benchmark_from_dict(guidellm_raw),
        aiperf=_aiperf_benchmark_from_dict(aiperf_raw)
        if tool == "aiperf"
        else AiperfBenchmarkSpec(),
        requirements=_benchmark_requirements_from_dict(raw.get("requirements")),
    )


def load_deployment_profile(path: Path) -> DeploymentProfile:
    raw = load_yaml_file(path)
    if raw.get("kind") != "DeploymentProfile":
        raise ValidationError(f"{path} is not a DeploymentProfile")

    metadata = parse_metadata(raw)
    spec = raw.get("spec") or {}
    profile_spec = DeploymentProfileSpec(
        platform=str(spec.get("platform", "")),
        mode=str(spec.get("mode", "")),
        runtime=_runtime_from_dict(spec.get("runtime")),
        model_storage=_storage_from_dict(spec.get("model_storage")),
        namespace=spec.get("namespace"),
        repo_url=str(spec.get("repo_url", "https://github.com/llm-d/llm-d.git")),
        repo_ref=str(spec.get("repo_ref", "main")),
        platform_version=str(spec.get("platform_version", "")),
        platform_channel=str(spec.get("platform_channel", "")),
        gateway=str(spec.get("gateway", "istio")),
        endpoint_path=str(spec.get("endpoint_path", "/v1/models")),
        scheduler_profile=str(spec.get("scheduler_profile", "")),
        scheduler_image=str(spec.get("scheduler_image", "")),
        options=dict(spec.get("options") or {}),
    )
    if not profile_spec.platform:
        raise ValidationError(f"{path} is missing spec.platform")
    if not profile_spec.mode:
        raise ValidationError(f"{path} is missing spec.mode")

    return DeploymentProfile(
        api_version=str(raw.get("apiVersion", "benchflow.io/v1alpha1")),
        kind="DeploymentProfile",
        metadata=metadata,
        spec=profile_spec,
    )


def load_benchmark_profile(path: Path) -> BenchmarkProfile:
    raw = load_yaml_file(path)
    if raw.get("kind") != "BenchmarkProfile":
        raise ValidationError(f"{path} is not a BenchmarkProfile")

    metadata = parse_metadata(raw)
    spec = raw.get("spec") or {}
    profile_spec = _benchmark_profile_spec_from_dict(spec)
    return BenchmarkProfile(
        api_version=str(raw.get("apiVersion", "benchflow.io/v1alpha1")),
        kind="BenchmarkProfile",
        metadata=metadata,
        spec=profile_spec,
    )


def load_metrics_profile(path: Path) -> MetricsProfile:
    raw = load_yaml_file(path)
    if raw.get("kind") != "MetricsProfile":
        raise ValidationError(f"{path} is not a MetricsProfile")

    metadata = parse_metadata(raw)
    spec = raw.get("spec") or {}
    profile_spec = MetricsProfileSpec(
        prometheus_url=str(spec.get("prometheus_url", "")),
        query_step=str(spec.get("query_step", "15s")),
        query_timeout=str(spec.get("query_timeout", "30s")),
        verify_tls=_as_bool(spec.get("verify_tls"), False),
        queries={
            str(key): str(value) for key, value in (spec.get("queries") or {}).items()
        },
    )
    if not profile_spec.prometheus_url:
        raise ValidationError(f"{path} is missing spec.prometheus_url")

    return MetricsProfile(
        api_version=str(raw.get("apiVersion", "benchflow.io/v1alpha1")),
        kind="MetricsProfile",
        metadata=metadata,
        spec=profile_spec,
    )


def load_run_plan_data(raw: dict[str, Any]) -> ResolvedRunPlan:
    if raw.get("kind") != "RunPlan":
        raise ValidationError("document is not a RunPlan")

    metadata = parse_metadata(raw)
    model = parse_model_spec(raw.get("model") or {})
    model_names = normalize_model_names(model.name, "model.name")
    if len(model_names) != 1:
        raise ValidationError("RunPlan model.name must contain exactly one value")
    model = model.__class__(name=model_names[0])

    profiles_raw = raw.get("profiles") or {}
    profiles = ProfileRefs(
        deployment=str(_require(profiles_raw.get("deployment"), "profiles.deployment")),
        benchmark=str(_require(profiles_raw.get("benchmark"), "profiles.benchmark")),
        metrics=str(_require(profiles_raw.get("metrics"), "profiles.metrics")),
    )

    deployment_raw = raw.get("deployment") or {}
    target_raw = deployment_raw.get("target") or {}
    deployment = ResolvedDeployment(
        platform=str(_require(deployment_raw.get("platform"), "deployment.platform")),
        mode=str(_require(deployment_raw.get("mode"), "deployment.mode")),
        namespace=str(
            _require(deployment_raw.get("namespace"), "deployment.namespace")
        ),
        release_name=str(
            _require(deployment_raw.get("release_name"), "deployment.release_name")
        ),
        runtime=_runtime_from_dict(deployment_raw.get("runtime")),
        model_storage=_storage_from_dict(deployment_raw.get("model_storage")),
        repo_url=str(
            deployment_raw.get("repo_url", "https://github.com/llm-d/llm-d.git")
        ),
        repo_ref=str(deployment_raw.get("repo_ref", "main")),
        platform_version=str(deployment_raw.get("platform_version", "")),
        platform_channel=str(deployment_raw.get("platform_channel", "")),
        gateway=str(deployment_raw.get("gateway", "istio")),
        scheduler_profile=str(deployment_raw.get("scheduler_profile", "")),
        scheduler_image=str(deployment_raw.get("scheduler_image", "")),
        options=dict(deployment_raw.get("options") or {}),
        target=TargetSpec(
            discovery=str(
                _require(target_raw.get("discovery"), "deployment.target.discovery")
            ),
            base_url=str(target_raw.get("base_url", "")),
            resource_kind=str(target_raw.get("resource_kind", "")),
            resource_name=str(target_raw.get("resource_name", "")),
            path=str(target_raw.get("path", "/v1/models")),
            metrics_release_name=str(target_raw.get("metrics_release_name", "")),
        ),
    )

    benchmark_raw = raw.get("benchmark") or {}
    benchmark = _benchmark_profile_spec_from_dict(benchmark_raw)

    metrics_raw = raw.get("metrics") or {}
    metrics = MetricsProfileSpec(
        prometheus_url=str(
            _require(metrics_raw.get("prometheus_url"), "metrics.prometheus_url")
        ),
        query_step=str(metrics_raw.get("query_step", "15s")),
        query_timeout=str(metrics_raw.get("query_timeout", "30s")),
        verify_tls=_as_bool(metrics_raw.get("verify_tls"), False),
        queries={
            str(key): str(value)
            for key, value in (metrics_raw.get("queries") or {}).items()
        },
    )

    return ResolvedRunPlan(
        api_version=str(raw.get("apiVersion", "benchflow.io/v1alpha1")),
        kind="RunPlan",
        metadata=metadata,
        profiles=profiles,
        execution=ExecutionSpec.from_dict(raw.get("execution")),
        target_cluster=_target_cluster_from_dict(raw.get("target_cluster")),
        model=model,
        deployment=deployment,
        benchmark=benchmark,
        metrics=metrics,
        stages=StageSpec.from_dict(raw.get("stages")),
        mlflow=MlflowSpec.from_dict(raw.get("mlflow")),
        service_account=str(raw.get("service_account", "benchflow-runner")),
        ttl_seconds_after_finished=int(raw.get("ttl_seconds_after_finished", 3600)),
    )


def load_run_plan_file(path: Path) -> ResolvedRunPlan:
    return load_run_plan_data(load_yaml_file(path))


@dataclass(slots=True)
class ProfileIndexEntry:
    name: str
    kind: str
    path: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def list_profile_entries(profiles_dir: Path) -> list[ProfileIndexEntry]:
    entries: list[ProfileIndexEntry] = []

    for path in sorted(profiles_dir.rglob("*.yaml")):
        raw = load_yaml_file(path)
        kind = raw.get("kind")
        metadata = raw.get("metadata") or {}
        name = metadata.get("name")
        if not isinstance(name, str) or not name:
            continue

        spec = raw.get("spec") or {}
        relative_path = str(path.relative_to(profiles_dir))

        if kind == "DeploymentProfile":
            details = {
                "platform": str(spec.get("platform", "")),
                "mode": str(spec.get("mode", "")),
            }
            entries.append(
                ProfileIndexEntry(
                    name=name, kind="deployment", path=relative_path, details=details
                )
            )
        elif kind == "BenchmarkProfile":
            tool = str(spec.get("tool", "guidellm") or "guidellm")
            guidellm = spec.get("guidellm") or {}
            aiperf = spec.get("aiperf") or {}
            details = {"tool": tool}
            if tool == "aiperf":
                details["endpoint_type"] = str(aiperf.get("endpoint_type", ""))
                details["dataset_type"] = str(aiperf.get("dataset_type", ""))
            else:
                details["rate_type"] = str(
                    guidellm.get("rate_type", spec.get("rate_type", ""))
                )
            entries.append(
                ProfileIndexEntry(
                    name=name, kind="benchmark", path=relative_path, details=details
                )
            )
        elif kind == "MetricsProfile":
            details = {
                "prometheus_url": str(spec.get("prometheus_url", "")),
                "query_count": len(spec.get("queries") or {}),
            }
            entries.append(
                ProfileIndexEntry(
                    name=name, kind="metrics", path=relative_path, details=details
                )
            )

    return entries


@dataclass(slots=True)
class ProfileCatalog:
    deployments: dict[str, DeploymentProfile]
    benchmarks: dict[str, BenchmarkProfile]
    metrics: dict[str, MetricsProfile]

    @classmethod
    def load(cls, profiles_dir: Path) -> "ProfileCatalog":
        deployments: dict[str, DeploymentProfile] = {}
        benchmarks: dict[str, BenchmarkProfile] = {}
        metrics: dict[str, MetricsProfile] = {}

        for path in sorted(profiles_dir.rglob("*.yaml")):
            raw = load_yaml_file(path)
            kind = raw.get("kind")
            if kind == "DeploymentProfile":
                profile = load_deployment_profile(path)
                deployments[profile.metadata.name] = profile
            elif kind == "BenchmarkProfile":
                profile = load_benchmark_profile(path)
                benchmarks[profile.metadata.name] = profile
            elif kind == "MetricsProfile":
                profile = load_metrics_profile(path)
                metrics[profile.metadata.name] = profile

        return cls(deployments=deployments, benchmarks=benchmarks, metrics=metrics)

    def require_deployment(self, name: str) -> DeploymentProfile:
        try:
            return self.deployments[name]
        except KeyError as exc:
            raise ValidationError(f"unknown deployment profile: {name}") from exc

    def require_benchmark(self, name: str) -> BenchmarkProfile:
        try:
            return self.benchmarks[name]
        except KeyError as exc:
            raise ValidationError(f"unknown benchmark profile: {name}") from exc

    def require_metrics(self, name: str) -> MetricsProfile:
        try:
            return self.metrics[name]
        except KeyError as exc:
            raise ValidationError(f"unknown metrics profile: {name}") from exc
