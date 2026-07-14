from __future__ import annotations

import json
import re
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
    GuidellmPreWarmupSpec,
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
    RuntimeHostPathSpec,
    RuntimePlacementSpec,
    RuntimePVCMountSpec,
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

_AIPERF_REQUIRED_FIELDS = {"endpoint_type"}
_AIPERF_RESERVED_FIELDS = {"dataset_url", "dataset_name", "dataset_cap", "max_seconds"}
_GUIDELLM_RESERVED_FIELDS = {"pre_warmup"}
_HOST_PATH_TYPES = {
    "",
    "DirectoryOrCreate",
    "Directory",
    "FileOrCreate",
    "File",
    "Socket",
    "CharDevice",
    "BlockDevice",
}
_RESERVED_RUNTIME_VOLUME_NAMES = {"dshm", "model-storage", "vllm-profiler"}
_HOST_PATH_RESERVED_VOLUME_NAMES = _RESERVED_RUNTIME_VOLUME_NAMES
_KUBERNETES_VOLUME_NAME_RE = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")


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


def _string_list(raw: Any, field_name: str) -> list[str] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValidationError(f"{field_name} must be a list of strings")
    return [str(item).strip() for item in raw if str(item).strip()]


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


def _reject_removed_runtime_security_fields(raw: Any, field_name: str) -> None:
    if not isinstance(raw, dict):
        return
    removed = [field for field in ("fs_group", "supplemental_groups") if field in raw]
    if removed:
        fields = ", ".join(f"{field_name}.{field}" for field in removed)
        raise ValidationError(
            f"{fields} are no longer supported; BenchFlow derives OpenShift "
            "runtime UID, groups, and MCS labels automatically"
        )


def _raw_value(raw: Any) -> Any | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        cleaned = raw.strip()
        return cleaned or None
    if isinstance(raw, (dict, list)):
        return json.dumps(raw, separators=(",", ":"))
    return raw


def _passthrough_value(raw: Any, field_name: str) -> Any | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        cleaned = raw.strip()
        return cleaned or None
    if isinstance(raw, (dict, list, bool, int, float)):
        return raw
    raise ValidationError(f"{field_name} has unsupported value: {raw!r}")


def _nonempty_string(raw: Any, field_name: str) -> str | None:
    if raw is None:
        return None
    cleaned = str(raw).strip()
    if not cleaned:
        raise ValidationError(f"{field_name} must not be empty")
    return cleaned


def _optional_string(raw: Any) -> str:
    if raw is None:
        return ""
    return str(raw).strip()


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


def _resource_key_list(raw: Any, field_name: str) -> list[str]:
    values = _string_list(raw, field_name)
    if values is None:
        return []
    return list(dict.fromkeys(values))


def _runtime_resources_from_dict(
    raw: dict[str, Any] | None, field_name: str
) -> RuntimeResourcesSpec:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValidationError(f"{field_name} must be a mapping")
    requests = _resource_mapping(raw.get("requests"), f"{field_name}.requests")
    limits = _resource_mapping(raw.get("limits"), f"{field_name}.limits")
    remove_requests = _resource_key_list(
        raw.get("remove_requests"), f"{field_name}.remove_requests"
    )
    remove_limits = _resource_key_list(
        raw.get("remove_limits"), f"{field_name}.remove_limits"
    )
    request_conflicts = sorted(set(requests).intersection(remove_requests))
    limit_conflicts = sorted(set(limits).intersection(remove_limits))
    if request_conflicts:
        conflicts = ", ".join(request_conflicts)
        raise ValidationError(
            f"{field_name} cannot set and remove request resources: {conflicts}"
        )
    if limit_conflicts:
        conflicts = ", ".join(limit_conflicts)
        raise ValidationError(
            f"{field_name} cannot set and remove limit resources: {conflicts}"
        )
    return RuntimeResourcesSpec(
        requests=requests,
        limits=limits,
        remove_requests=remove_requests,
        remove_limits=remove_limits,
    )


def _runtime_placement_from_dict(
    raw: dict[str, Any] | None, field_name: str
) -> RuntimePlacementSpec:
    if raw is None:
        return RuntimePlacementSpec()
    if not isinstance(raw, dict):
        raise ValidationError(f"{field_name} must be a mapping")
    mode = str(raw.get("mode", "") or "").strip()
    spread_pool = str(raw.get("spread_pool", "") or "").strip()
    allowed_modes = {"same-node", "sequential"}
    if mode and mode not in allowed_modes:
        allowed = ", ".join(sorted(allowed_modes))
        raise ValidationError(f"{field_name}.mode must be one of: {allowed}")
    if mode == "same-node" and not spread_pool:
        raise ValidationError(
            f"{field_name}.spread_pool is required when mode is 'same-node'"
        )
    if spread_pool and not mode:
        raise ValidationError(f"{field_name}.mode is required when spread_pool is set")
    return RuntimePlacementSpec(mode=mode, spread_pool=spread_pool)


def _runtime_host_paths_from_dict(
    raw: Any, field_name: str
) -> list[RuntimeHostPathSpec]:
    entries = _mapping_list(raw, field_name)
    host_paths: list[RuntimeHostPathSpec] = []
    names: set[str] = set()
    mount_paths: set[str] = set()
    for index, item in enumerate(entries):
        item_field = f"{field_name}[{index}]"
        name = _nonempty_string(item.get("name"), f"{item_field}.name")
        host_path = _nonempty_string(item.get("host_path"), f"{item_field}.host_path")
        mount_path = _nonempty_string(
            item.get("mount_path"), f"{item_field}.mount_path"
        )
        if name is None:
            raise ValidationError(f"{item_field}.name is required")
        if host_path is None:
            raise ValidationError(f"{item_field}.host_path is required")
        if mount_path is None:
            raise ValidationError(f"{item_field}.mount_path is required")
        if len(name) > 63 or not _KUBERNETES_VOLUME_NAME_RE.match(name):
            raise ValidationError(
                f"{item_field}.name must be a valid Kubernetes volume name"
            )
        if name in _HOST_PATH_RESERVED_VOLUME_NAMES:
            raise ValidationError(
                f"{item_field}.name uses reserved BenchFlow volume name {name!r}"
            )
        if name in names:
            raise ValidationError(f"{item_field}.name duplicates host path {name!r}")
        if not host_path.startswith("/"):
            raise ValidationError(f"{item_field}.host_path must be an absolute path")
        if not mount_path.startswith("/"):
            raise ValidationError(f"{item_field}.mount_path must be an absolute path")
        if mount_path in mount_paths:
            raise ValidationError(
                f"{item_field}.mount_path duplicates mount path {mount_path!r}"
            )
        host_path_type = str(item.get("type", "") or "").strip()
        if host_path_type not in _HOST_PATH_TYPES:
            allowed = ", ".join(value for value in sorted(_HOST_PATH_TYPES) if value)
            raise ValidationError(f"{item_field}.type must be one of: {allowed}")
        host_paths.append(
            RuntimeHostPathSpec(
                name=name,
                host_path=host_path,
                mount_path=mount_path,
                type=host_path_type,
                read_only=_as_bool(item.get("read_only"), False),
            )
        )
        names.add(name)
        mount_paths.add(mount_path)
    return host_paths


def _runtime_pvc_mounts_from_dict(
    raw: Any, field_name: str
) -> list[RuntimePVCMountSpec]:
    entries = _mapping_list(raw, field_name)
    pvc_mounts: list[RuntimePVCMountSpec] = []
    names: set[str] = set()
    mount_paths: set[str] = set()
    for index, item in enumerate(entries):
        item_field = f"{field_name}[{index}]"
        name = _nonempty_string(item.get("name"), f"{item_field}.name")
        claim_name = _nonempty_string(
            item.get("claim_name") or item.get("pvc_name"),
            f"{item_field}.claim_name",
        )
        mount_path = _nonempty_string(
            item.get("mount_path"), f"{item_field}.mount_path"
        )
        if name is None:
            raise ValidationError(f"{item_field}.name is required")
        if claim_name is None:
            raise ValidationError(f"{item_field}.claim_name is required")
        if mount_path is None:
            raise ValidationError(f"{item_field}.mount_path is required")
        if len(name) > 63 or not _KUBERNETES_VOLUME_NAME_RE.match(name):
            raise ValidationError(
                f"{item_field}.name must be a valid Kubernetes volume name"
            )
        if name in _RESERVED_RUNTIME_VOLUME_NAMES:
            raise ValidationError(
                f"{item_field}.name uses reserved BenchFlow volume name {name!r}"
            )
        if name in names:
            raise ValidationError(f"{item_field}.name duplicates PVC mount {name!r}")
        if len(claim_name) > 253 or not _KUBERNETES_VOLUME_NAME_RE.match(claim_name):
            raise ValidationError(
                f"{item_field}.claim_name must be a valid Kubernetes PVC name"
            )
        if not mount_path.startswith("/"):
            raise ValidationError(f"{item_field}.mount_path must be an absolute path")
        if mount_path in mount_paths:
            raise ValidationError(
                f"{item_field}.mount_path duplicates mount path {mount_path!r}"
            )

        create = _as_bool(item.get("create"), False)
        storage_class_name = str(
            item.get("storage_class_name") or item.get("storage_class") or ""
        ).strip()
        size = str(item.get("size") or "").strip()
        access_modes_raw = item.get("access_modes")
        if access_modes_raw is None and item.get("access_mode") is not None:
            access_modes_raw = [item.get("access_mode")]
        access_modes = [
            _nonempty_string(value, f"{item_field}.access_modes[]") or ""
            for value in (access_modes_raw or [])
        ]
        access_modes = [value for value in access_modes if value]
        if create:
            if not size:
                raise ValidationError(f"{item_field}.size is required when create=true")
            if not access_modes:
                raise ValidationError(
                    f"{item_field}.access_modes is required when create=true"
                )

        pvc_mounts.append(
            RuntimePVCMountSpec(
                name=name,
                claim_name=claim_name,
                mount_path=mount_path,
                read_only=_as_bool(item.get("read_only"), False),
                create=create,
                storage_class_name=storage_class_name,
                size=size,
                access_modes=access_modes,
            )
        )
        names.add(name)
        mount_paths.add(mount_path)
    return pvc_mounts


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


def _overrides_from_dict(
    raw: dict[str, Any] | None, *, field_name: str = "spec.overrides"
) -> OverrideSpec:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValidationError(f"{field_name} must be a mapping")
    images = raw.get("images") or {}
    scale = raw.get("scale") or {}
    runtime = raw.get("runtime") or {}
    _reject_removed_runtime_security_fields(runtime, f"{field_name}.runtime")
    benchmark = raw.get("benchmark") or {}
    llm_d = raw.get("llm_d") or {}
    rhoai = raw.get("rhoai") or {}

    return OverrideSpec(
        images=OverrideImagesSpec(
            runtime=_string_or_list(
                images.get("runtime"), f"{field_name}.images.runtime"
            ),
            scheduler=_string_or_list(
                images.get("scheduler"), f"{field_name}.images.scheduler"
            ),
        ),
        scale=OverrideScaleSpec(
            replicas=_int_or_list(
                scale.get("replicas"), f"{field_name}.scale.replicas"
            ),
            tensor_parallelism=_int_or_list(
                scale.get("tensor_parallelism"),
                f"{field_name}.scale.tensor_parallelism",
            ),
        ),
        runtime=OverrideRuntimeSpec(
            env={
                str(key): str(value)
                for key, value in (runtime.get("env") or {}).items()
            },
            vllm_args=(
                _string_list(
                    runtime.get("vllm_args"), f"{field_name}.runtime.vllm_args"
                )
                if "vllm_args" in runtime
                else None
            ),
            vllm_extra_args=(
                _string_list(
                    runtime.get("vllm_extra_args"),
                    f"{field_name}.runtime.vllm_extra_args",
                )
                or []
            ),
            host_paths=(
                _runtime_host_paths_from_dict(
                    runtime.get("host_paths"), f"{field_name}.runtime.host_paths"
                )
                if "host_paths" in runtime
                else None
            ),
            service_account_name=(
                _nonempty_string(
                    runtime.get("service_account_name"),
                    f"{field_name}.runtime.service_account_name",
                )
                if "service_account_name" in runtime
                else None
            ),
            node_selector=(
                _string_mapping(
                    runtime.get("node_selector"),
                    f"{field_name}.runtime.node_selector",
                )
                if "node_selector" in runtime
                else None
            ),
            affinity=(
                _mapping(runtime.get("affinity"), f"{field_name}.runtime.affinity")
                if "affinity" in runtime
                else None
            ),
            placement=(
                _runtime_placement_from_dict(
                    runtime.get("placement"), f"{field_name}.runtime.placement"
                )
                if "placement" in runtime
                else None
            ),
            tolerations=(
                _mapping_list(
                    runtime.get("tolerations"),
                    f"{field_name}.runtime.tolerations",
                )
                if "tolerations" in runtime
                else None
            ),
            resources=(
                _runtime_resources_from_dict(
                    runtime.get("resources"), f"{field_name}.runtime.resources"
                )
                if "resources" in runtime
                else None
            ),
        ),
        benchmark=OverrideBenchmarkSpec(
            rates=_int_list(benchmark.get("rates"), f"{field_name}.benchmark.rates"),
            max_seconds=_positive_int(
                benchmark.get("max_seconds"),
                f"{field_name}.benchmark.max_seconds",
            ),
            max_requests=_nonempty_string(
                benchmark.get("max_requests"),
                f"{field_name}.benchmark.max_requests",
            ),
            request_type=_nonempty_string(
                benchmark.get("request_type"),
                f"{field_name}.benchmark.request_type",
            ),
            env=(
                _string_mapping(
                    benchmark.get("env"),
                    f"{field_name}.benchmark.env",
                )
                if "env" in benchmark
                else None
            ),
        ),
        llm_d=OverrideLlmdSpec(
            repo_ref=_string_or_list(
                llm_d.get("repo_ref"), f"{field_name}.llm_d.repo_ref"
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


def _reject_model_override_axes(override: OverrideSpec, *, field_name: str) -> None:
    axis_fields = {
        "images.runtime": override.images.runtime,
        "images.scheduler": override.images.scheduler,
        "scale.replicas": override.scale.replicas,
        "scale.tensor_parallelism": override.scale.tensor_parallelism,
        "llm_d.repo_ref": override.llm_d.repo_ref,
    }
    for suffix, value in axis_fields.items():
        if isinstance(value, list):
            raise ValidationError(
                f"{field_name}.{suffix} must be a scalar; "
                "model_overrides do not define matrix axes"
            )


def _model_overrides_from_dict(raw: Any) -> dict[str, OverrideSpec]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValidationError("spec.model_overrides must be a mapping")
    overrides: dict[str, OverrideSpec] = {}
    for model_name, override_raw in raw.items():
        cleaned_model_name = str(model_name).strip()
        if not cleaned_model_name:
            raise ValidationError("spec.model_overrides keys must not be empty")
        field_name = f"spec.model_overrides.{cleaned_model_name}"
        override = _overrides_from_dict(
            override_raw,
            field_name=field_name,
        )
        _reject_model_override_axes(override, field_name=field_name)
        overrides[cleaned_model_name] = override
    return overrides


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


def _endpoint_scope(raw: object, field_name: str, *, default: str = "external") -> str:
    value = str(raw or default).strip() or default
    if not value:
        return ""
    if value not in {"external", "internal"}:
        raise ValidationError(f"{field_name} must be 'external' or 'internal'")
    return value


def _experiment_target_from_dict(raw: dict[str, Any] | None) -> ExperimentTargetSpec:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValidationError("target must be a mapping")
    base_url = str(raw.get("base_url", "") or "").strip()
    path = str(raw.get("path", "/v1/models") or "/v1/models").strip()
    metrics_release_name = str(raw.get("metrics_release_name", "") or "").strip()
    endpoint_scope = _endpoint_scope(
        raw.get("endpoint_scope"), "target.endpoint_scope", default=""
    )
    if not path:
        raise ValidationError("target.path must not be empty")
    if raw and not base_url and set(raw) - {"endpoint_scope"}:
        raise ValidationError("target.base_url must not be empty")
    return ExperimentTargetSpec(
        base_url=base_url,
        path=path,
        metrics_release_name=metrics_release_name,
        endpoint_scope=endpoint_scope,
    )


def load_yaml_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValidationError(f"{path} does not contain a mapping document")
    return data


def _runtime_from_dict(raw: dict[str, Any] | None) -> RuntimeSpec:
    raw = raw or {}
    _reject_removed_runtime_security_fields(raw, "spec.runtime")
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
        shared_memory_size=_optional_string(raw.get("shared_memory_size")),
        host_paths=_runtime_host_paths_from_dict(
            raw.get("host_paths"), "spec.runtime.host_paths"
        ),
        pvc_mounts=_runtime_pvc_mounts_from_dict(
            raw.get("pvc_mounts"), "spec.runtime.pvc_mounts"
        ),
        service_account_name=str(raw.get("service_account_name", "") or "").strip(),
        node_selector=_string_mapping(
            raw.get("node_selector"), "spec.runtime.node_selector"
        ),
        affinity=_mapping(raw.get("affinity"), "spec.runtime.affinity"),
        placement=_runtime_placement_from_dict(
            raw.get("placement"), "spec.runtime.placement"
        ),
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
        model_overrides=_model_overrides_from_dict(spec.get("model_overrides")),
    )

    return Experiment(
        api_version=str(raw.get("apiVersion", "benchflow.io/v1alpha1")),
        kind="Experiment",
        metadata=metadata,
        spec=experiment_spec,
    )


def _guidellm_benchmark_from_dict(raw: dict[str, Any]) -> GuidellmBenchmarkSpec:
    args: dict[str, Any] = {}
    for key, value in raw.items():
        if key in _GUIDELLM_RESERVED_FIELDS:
            continue
        field_name = f"spec.guidellm.{key}"
        normalized = _passthrough_value(value, field_name)
        if normalized is not None:
            args[key] = normalized
    return GuidellmBenchmarkSpec(
        args=args,
        pre_warmup=_guidellm_pre_warmup_from_dict(raw.get("pre_warmup")),
    )


def _guidellm_pre_warmup_from_dict(raw: Any) -> GuidellmPreWarmupSpec:
    if raw is None:
        return GuidellmPreWarmupSpec()
    if not isinstance(raw, dict):
        raise ValidationError("spec.guidellm.pre_warmup must be a mapping")

    enabled = _as_bool(raw.get("enabled"), True)
    raw_rate = raw.get("rate")
    raw_rates = raw.get("rates")
    if raw_rate is not None and raw_rates is not None:
        raise ValidationError("spec.guidellm.pre_warmup cannot set both rate and rates")
    rate = _positive_int(raw_rate, "spec.guidellm.pre_warmup.rate")
    if raw_rates is not None:
        rates = _int_list(raw_rates, "spec.guidellm.pre_warmup.rates")
        if rates is None or len(rates) != 1:
            raise ValidationError(
                "spec.guidellm.pre_warmup.rates must contain exactly one value"
            )
        rate = rates[0]
    if enabled and rate is None:
        raise ValidationError(
            "spec.guidellm.pre_warmup requires a single rate; use rate: 15 or rates: [15]"
        )

    args: dict[str, Any] = {}
    for key, value in raw.items():
        if key in {"enabled", "rates"}:
            continue
        field_name = f"spec.guidellm.pre_warmup.{key}"
        if key == "rate":
            normalized = _positive_int(value, field_name)
        else:
            normalized = _passthrough_value(value, field_name)
        if normalized is not None:
            args[key] = normalized
    if rate is not None:
        args["rate"] = rate
    return GuidellmPreWarmupSpec(enabled=enabled, args=args)


def _aiperf_benchmark_from_dict(raw: dict[str, Any]) -> AiperfBenchmarkSpec:
    args: dict[str, Any] = {}
    for key, value in raw.items():
        if key in _AIPERF_RESERVED_FIELDS:
            continue
        field_name = f"spec.aiperf.{key}"
        if key in {
            "endpoint_type",
            "endpoint_path",
            "tokenizer",
            "dataset_type",
            "public_dataset",
            "export_level",
        }:
            normalized = _nonempty_string(value, field_name)
        elif key in {"streaming", "fixed_schedule", "fixed_schedule_auto_offset"}:
            normalized = _as_bool(value, True)
        elif key == "export_http_trace":
            normalized = _as_bool(value, False)
        elif key in {"synthesis_max_isl", "fixed_schedule_end_offset"}:
            normalized = _optional_positive_int(value, field_name)
        else:
            normalized = _passthrough_value(value, field_name)
        if normalized is not None:
            args[key] = normalized
    public_dataset = str(args.get("public_dataset", "") or "").strip()
    dataset_url = str(raw.get("dataset_url", "") or "").strip()
    dataset_type = str(args.get("dataset_type", "") or "").strip()
    args.setdefault("streaming", True)
    if not public_dataset:
        args.setdefault("fixed_schedule", True)
        args.setdefault("fixed_schedule_auto_offset", True)
    missing = [
        field_name
        for field_name in sorted(_AIPERF_REQUIRED_FIELDS)
        if not str(args.get(field_name, "") or "").strip()
    ]
    if missing:
        joined = ", ".join(f"spec.aiperf.{field_name}" for field_name in missing)
        raise ValidationError(f"aiperf benchmark profile is missing {joined}")
    if public_dataset:
        if dataset_url:
            raise ValidationError(
                "spec.aiperf.public_dataset cannot be combined with spec.aiperf.dataset_url"
            )
        if dataset_type:
            raise ValidationError(
                "spec.aiperf.public_dataset cannot be combined with spec.aiperf.dataset_type"
            )
        if raw.get("dataset_cap") is not None:
            raise ValidationError(
                "spec.aiperf.dataset_cap is not supported with spec.aiperf.public_dataset"
            )
    else:
        required_dataset_fields = []
        if not dataset_url:
            required_dataset_fields.append("spec.aiperf.dataset_url")
        if not dataset_type:
            required_dataset_fields.append("spec.aiperf.dataset_type")
        if required_dataset_fields:
            raise ValidationError(
                "aiperf benchmark profile is missing "
                + ", ".join(required_dataset_fields)
            )

    return AiperfBenchmarkSpec(
        args=args,
        dataset_url=dataset_url,
        dataset_name=str(raw.get("dataset_name", "") or "").strip(),
        dataset_cap=_optional_positive_int(
            raw.get("dataset_cap"), "spec.aiperf.dataset_cap"
        ),
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
    if tool == "guidellm" and guidellm_raw is None:
        raise ValidationError("spec.guidellm is required when spec.tool is guidellm")
    if guidellm_raw is None:
        guidellm_raw = {}
    if not isinstance(guidellm_raw, dict):
        raise ValidationError("spec.guidellm must be a mapping")

    aiperf_raw = raw.get("aiperf")
    if tool == "aiperf" and aiperf_raw is None:
        raise ValidationError("spec.aiperf is required when spec.tool is aiperf")
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


def _resolved_guidellm_pre_warmup_from_dict(raw: Any) -> GuidellmPreWarmupSpec:
    if raw is None:
        return GuidellmPreWarmupSpec()
    if not isinstance(raw, dict):
        raise ValidationError("benchmark.guidellm.pre_warmup must be a mapping")
    enabled = _as_bool(raw.get("enabled"), False)
    args_raw = raw.get("args")
    if args_raw is None:
        return _guidellm_pre_warmup_from_dict(raw)
    if not isinstance(args_raw, dict):
        raise ValidationError("benchmark.guidellm.pre_warmup.args must be a mapping")
    args = {
        str(key): _passthrough_value(value, f"benchmark.guidellm.pre_warmup.args.{key}")
        for key, value in args_raw.items()
    }
    return GuidellmPreWarmupSpec(
        enabled=enabled, args={k: v for k, v in args.items() if v is not None}
    )


def _resolved_guidellm_benchmark_from_dict(
    raw: dict[str, Any],
) -> GuidellmBenchmarkSpec:
    args_raw = raw.get("args")
    if args_raw is None:
        return _guidellm_benchmark_from_dict(raw)
    if not isinstance(args_raw, dict):
        raise ValidationError("benchmark.guidellm.args must be a mapping")
    args = {
        str(key): _passthrough_value(value, f"benchmark.guidellm.args.{key}")
        for key, value in args_raw.items()
    }
    return GuidellmBenchmarkSpec(
        args={k: v for k, v in args.items() if v is not None},
        pre_warmup=_resolved_guidellm_pre_warmup_from_dict(raw.get("pre_warmup")),
    )


def _resolved_aiperf_benchmark_from_dict(raw: dict[str, Any]) -> AiperfBenchmarkSpec:
    args_raw = raw.get("args")
    if args_raw is None:
        return _aiperf_benchmark_from_dict(raw)
    if not isinstance(args_raw, dict):
        raise ValidationError("benchmark.aiperf.args must be a mapping")
    args = {
        str(key): _passthrough_value(value, f"benchmark.aiperf.args.{key}")
        for key, value in args_raw.items()
    }
    return AiperfBenchmarkSpec(
        args={k: v for k, v in args.items() if v is not None},
        dataset_url=str(raw.get("dataset_url", "") or "").strip(),
        dataset_name=str(raw.get("dataset_name", "") or "").strip(),
        dataset_cap=_optional_positive_int(
            raw.get("dataset_cap"), "benchmark.aiperf.dataset_cap"
        ),
        max_seconds=int(raw.get("max_seconds", 7200)),
    )


def _resolved_benchmark_profile_spec_from_dict(
    raw: dict[str, Any],
) -> BenchmarkProfileSpec:
    tool = str(raw.get("tool", "guidellm") or "guidellm").strip()
    if tool not in {"guidellm", "aiperf"}:
        raise ValidationError(
            f"unsupported benchmark tool: {tool}; supported tools are guidellm and aiperf"
        )
    env = {str(key): str(value) for key, value in (raw.get("env") or {}).items()}
    guidellm_raw = raw.get("guidellm") or {}
    aiperf_raw = raw.get("aiperf") or {}
    if not isinstance(guidellm_raw, dict):
        raise ValidationError("benchmark.guidellm must be a mapping")
    if not isinstance(aiperf_raw, dict):
        raise ValidationError("benchmark.aiperf must be a mapping")
    return BenchmarkProfileSpec(
        tool=tool,
        env=env,
        guidellm=_resolved_guidellm_benchmark_from_dict(guidellm_raw),
        aiperf=_resolved_aiperf_benchmark_from_dict(aiperf_raw)
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
        endpoint_scope=_endpoint_scope(
            spec.get("endpoint_scope"), "spec.endpoint_scope"
        ),
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
            endpoint_scope=_endpoint_scope(
                target_raw.get("endpoint_scope"), "deployment.target.endpoint_scope"
            ),
        ),
    )

    benchmark_raw = raw.get("benchmark") or {}
    benchmark = _resolved_benchmark_profile_spec_from_dict(benchmark_raw)

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
                profile = guidellm.get("profile") or {}
                if isinstance(profile, dict):
                    details["profile"] = str(profile.get("kind", "") or "")
                else:
                    details["profile"] = str(profile or "")
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
