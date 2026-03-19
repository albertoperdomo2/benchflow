from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .models import (
    BenchmarkProfile,
    BenchmarkProfileSpec,
    DeploymentProfile,
    DeploymentProfileSpec,
    ExecutionSpec,
    Experiment,
    ExperimentSpec,
    MetricsProfile,
    MetricsProfileSpec,
    MlflowSpec,
    ModelStorageSpec,
    OverrideImagesSpec,
    OverrideLlmdSpec,
    OverrideRhoaiSpec,
    OverrideRuntimeSpec,
    OverrideScaleSpec,
    OverrideSpec,
    ProfileRefs,
    ResolvedDeployment,
    ResolvedRunPlan,
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


def _overrides_from_dict(raw: dict[str, Any] | None) -> OverrideSpec:
    raw = raw or {}
    images = raw.get("images") or {}
    scale = raw.get("scale") or {}
    runtime = raw.get("runtime") or {}
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


def load_yaml_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValidationError(f"{path} does not contain a mapping document")
    return data


def _runtime_from_dict(raw: dict[str, Any] | None) -> RuntimeSpec:
    raw = raw or {}
    env = {str(key): str(value) for key, value in (raw.get("env") or {}).items()}
    return RuntimeSpec(
        image=str(raw.get("image", "")),
        replicas=int(raw.get("replicas", 1)),
        tensor_parallelism=int(raw.get("tensor_parallelism", 1)),
        vllm_args=[str(item) for item in (raw.get("vllm_args") or [])],
        env=env,
    )


def _storage_from_dict(raw: dict[str, Any] | None) -> ModelStorageSpec:
    raw = raw or {}
    return ModelStorageSpec(
        pvc_name=str(raw.get("pvc_name", "models-storage")),
        cache_dir=str(raw.get("cache_dir", "/models")),
        mount_path=str(raw.get("mount_path", "/model-cache")),
    )


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
        overrides=_overrides_from_dict(spec.get("overrides")),
    )

    return Experiment(
        api_version=str(raw.get("apiVersion", "benchflow.io/v1alpha1")),
        kind="Experiment",
        metadata=metadata,
        spec=experiment_spec,
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
    rates = [int(item) for item in (spec.get("rates") or [])]
    env = {str(key): str(value) for key, value in (spec.get("env") or {}).items()}
    profile_spec = BenchmarkProfileSpec(
        tool=str(spec.get("tool", "guidellm")),
        backend_type=str(spec.get("backend_type", "openai_http")),
        rate_type=str(spec.get("rate_type", "concurrent")),
        rates=rates,
        data=str(spec.get("data", "prompt_tokens=1000,output_tokens=1000")),
        max_seconds=int(spec.get("max_seconds", 600)),
        max_requests=int(spec["max_requests"])
        if spec.get("max_requests") is not None
        else None,
        env=env,
    )
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
        ),
    )

    benchmark_raw = raw.get("benchmark") or {}
    benchmark = BenchmarkProfileSpec(
        tool=str(benchmark_raw.get("tool", "guidellm")),
        backend_type=str(benchmark_raw.get("backend_type", "openai_http")),
        rate_type=str(benchmark_raw.get("rate_type", "concurrent")),
        rates=[int(item) for item in (benchmark_raw.get("rates") or [])],
        data=str(benchmark_raw.get("data", "prompt_tokens=1000,output_tokens=1000")),
        max_seconds=int(benchmark_raw.get("max_seconds", 600)),
        max_requests=int(benchmark_raw["max_requests"])
        if benchmark_raw.get("max_requests") is not None
        else None,
        env={
            str(key): str(value)
            for key, value in (benchmark_raw.get("env") or {}).items()
        },
    )

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
            details = {
                "tool": str(spec.get("tool", "")),
                "rate_type": str(spec.get("rate_type", "")),
            }
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
