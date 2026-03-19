from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class ValidationError(ValueError):
    """Raised when a config document is malformed."""


def _require(value: Any, field_name: str) -> Any:
    if value in (None, "", []):
        raise ValidationError(f"missing required field: {field_name}")
    return value


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    raise ValidationError(f"invalid boolean value: {value!r}")


def sanitize_name(value: str, max_length: int = 42) -> str:
    cleaned = value.lower().replace("/", "-").replace(".", "")
    cleaned = cleaned.strip("-")
    return cleaned[:max_length]


def normalize_profile_refs(value: str | list[str], field_name: str) -> list[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            raise ValidationError(f"missing required field: {field_name}")
        return [cleaned]
    if isinstance(value, list):
        cleaned_values = [str(item).strip() for item in value if str(item).strip()]
        if not cleaned_values:
            raise ValidationError(f"missing required field: {field_name}")
        return cleaned_values
    raise ValidationError(
        f"{field_name} must be a string or a list of strings, got: {value!r}"
    )


def normalize_model_names(value: str | list[str], field_name: str) -> list[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            raise ValidationError(f"missing required field: {field_name}")
        return [cleaned]
    if isinstance(value, list):
        cleaned_values = [str(item).strip() for item in value if str(item).strip()]
        if not cleaned_values:
            raise ValidationError(f"missing required field: {field_name}")
        return cleaned_values
    raise ValidationError(
        f"{field_name} must be a string or a list of strings, got: {value!r}"
    )


@dataclass(slots=True)
class Metadata:
    name: str
    labels: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ModelSpec:
    name: str | list[str]

    def resolved_name(self) -> str:
        if isinstance(self.name, list):
            if len(self.name) != 1:
                raise ValidationError("resolved model name requires exactly one value")
            return self.name[0]
        return self.name

    @property
    def pvc_directory_name(self) -> str:
        return self.resolved_name().replace("/", "-")

    @property
    def resource_name(self) -> str:
        return sanitize_name(self.resolved_name())


@dataclass(slots=True)
class StageSpec:
    download: bool = True
    deploy: bool = True
    benchmark: bool = True
    collect: bool = True
    cleanup: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "StageSpec":
        raw = raw or {}
        return cls(
            download=_as_bool(raw.get("download"), True),
            deploy=_as_bool(raw.get("deploy"), True),
            benchmark=_as_bool(raw.get("benchmark"), True),
            collect=_as_bool(raw.get("collect"), True),
            cleanup=_as_bool(raw.get("cleanup"), True),
        )


@dataclass(slots=True)
class MlflowSpec:
    experiment: str = ""
    tags: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "MlflowSpec":
        raw = raw or {}
        tags = {str(key): str(value) for key, value in (raw.get("tags") or {}).items()}
        return cls(experiment=str(raw.get("experiment", "") or ""), tags=tags)


@dataclass(slots=True)
class ExecutionSpec:
    backend: str = "tekton"
    timeout: str = "1h"

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "ExecutionSpec":
        raw = raw or {}
        timeout = str(raw.get("timeout", "1h") or "1h").strip()
        if not timeout:
            raise ValidationError("execution.timeout must not be empty")
        return cls(timeout=timeout)


@dataclass(slots=True)
class ClusterTargetSpec:
    kubeconfig: str = ""
    kubeconfig_secret: str = ""

    def enabled(self) -> bool:
        return bool(self.kubeconfig or self.kubeconfig_secret)


@dataclass(slots=True)
class OverrideImagesSpec:
    runtime: str | list[str] | None = None
    scheduler: str | list[str] | None = None


@dataclass(slots=True)
class OverrideScaleSpec:
    replicas: int | list[int] | None = None
    tensor_parallelism: int | list[int] | None = None


@dataclass(slots=True)
class OverrideRuntimeSpec:
    vllm_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class OverrideLlmdSpec:
    repo_ref: str | list[str] | None = None


@dataclass(slots=True)
class OverrideRhoaiSpec:
    enable_auth: bool | None = None


@dataclass(slots=True)
class OverrideSpec:
    images: OverrideImagesSpec = field(default_factory=OverrideImagesSpec)
    scale: OverrideScaleSpec = field(default_factory=OverrideScaleSpec)
    runtime: OverrideRuntimeSpec = field(default_factory=OverrideRuntimeSpec)
    llm_d: OverrideLlmdSpec = field(default_factory=OverrideLlmdSpec)
    rhoai: OverrideRhoaiSpec = field(default_factory=OverrideRhoaiSpec)


@dataclass(slots=True)
class ExperimentSpec:
    model: ModelSpec
    deployment_profile: list[str]
    benchmark_profile: list[str]
    metrics_profile: list[str] = field(default_factory=lambda: ["detailed"])
    namespace: str = "benchflow"
    service_account: str = "benchflow-runner"
    ttl_seconds_after_finished: int = 3600
    stages: StageSpec = field(default_factory=StageSpec)
    mlflow: MlflowSpec = field(default_factory=MlflowSpec)
    execution: ExecutionSpec = field(default_factory=ExecutionSpec)
    target_cluster: ClusterTargetSpec = field(default_factory=ClusterTargetSpec)
    overrides: OverrideSpec = field(default_factory=OverrideSpec)


@dataclass(slots=True)
class Experiment:
    api_version: str
    kind: str
    metadata: Metadata
    spec: ExperimentSpec


@dataclass(slots=True)
class RuntimeSpec:
    image: str = ""
    replicas: int = 1
    tensor_parallelism: int = 1
    vllm_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ModelStorageSpec:
    pvc_name: str = "models-storage"
    cache_dir: str = "/models"
    mount_path: str = "/model-cache"


@dataclass(slots=True)
class DeploymentProfileSpec:
    platform: str
    mode: str
    runtime: RuntimeSpec = field(default_factory=RuntimeSpec)
    model_storage: ModelStorageSpec = field(default_factory=ModelStorageSpec)
    namespace: str | None = None
    repo_url: str = "https://github.com/llm-d/llm-d.git"
    repo_ref: str = "main"
    gateway: str = "istio"
    endpoint_path: str = "/v1/models"
    scheduler_profile: str = ""
    scheduler_image: str = ""
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DeploymentProfile:
    api_version: str
    kind: str
    metadata: Metadata
    spec: DeploymentProfileSpec


@dataclass(slots=True)
class BenchmarkProfileSpec:
    tool: str = "guidellm"
    backend_type: str = "openai_http"
    rate_type: str = "concurrent"
    rates: list[int] = field(default_factory=list)
    data: str = "prompt_tokens=1000,output_tokens=1000"
    max_seconds: int = 600
    max_requests: int | None = None
    env: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkProfile:
    api_version: str
    kind: str
    metadata: Metadata
    spec: BenchmarkProfileSpec


@dataclass(slots=True)
class MetricsProfileSpec:
    prometheus_url: str
    query_step: str
    query_timeout: str
    verify_tls: bool = False
    queries: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class MetricsProfile:
    api_version: str
    kind: str
    metadata: Metadata
    spec: MetricsProfileSpec


@dataclass(slots=True)
class TargetSpec:
    discovery: str
    base_url: str = ""
    resource_kind: str = ""
    resource_name: str = ""
    path: str = "/v1/models"


@dataclass(slots=True)
class ResolvedDeployment:
    platform: str
    mode: str
    namespace: str
    release_name: str
    runtime: RuntimeSpec
    model_storage: ModelStorageSpec
    repo_url: str
    repo_ref: str
    gateway: str
    scheduler_profile: str
    scheduler_image: str
    options: dict[str, Any]
    target: TargetSpec


@dataclass(slots=True)
class ProfileRefs:
    deployment: str
    benchmark: str
    metrics: str


@dataclass(slots=True)
class ResolvedRunPlan:
    api_version: str
    kind: str
    metadata: Metadata
    profiles: ProfileRefs
    execution: ExecutionSpec
    target_cluster: ClusterTargetSpec
    model: ModelSpec
    deployment: ResolvedDeployment
    benchmark: BenchmarkProfileSpec
    metrics: MetricsProfileSpec
    stages: StageSpec
    mlflow: MlflowSpec
    service_account: str
    ttl_seconds_after_finished: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_metadata(raw: dict[str, Any]) -> Metadata:
    metadata = raw.get("metadata") or {}
    return Metadata(
        name=str(_require(metadata.get("name"), "metadata.name")),
        labels={
            str(key): str(value)
            for key, value in (metadata.get("labels") or {}).items()
        },
    )


def parse_model_spec(raw: dict[str, Any]) -> ModelSpec:
    name = raw.get("name")
    if isinstance(name, str):
        cleaned = str(_require(name, "spec.model.name")).strip()
        return ModelSpec(name=cleaned)
    return ModelSpec(name=normalize_model_names(name, "spec.model.name"))
