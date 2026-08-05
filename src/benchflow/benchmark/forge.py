"""Forge MLflow artifact adapter for OpenShift AI comparison reports."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

from ..mlflow_compat import configure_mlflow_tracking, create_mlflow_client
from ..models import ValidationError
from . import runtime

logger = logging.getLogger(__name__)

_TEST_LABELS_SUFFIX = "/000__llmd_test/__test_labels__.yaml"


def is_forge_artifact_paths(artifact_paths: set[str]) -> bool:
    """Return whether an MLflow artifact tree uses Forge's test layout."""
    return any(
        "01__test/" in path and path.endswith(_TEST_LABELS_SUFFIX)
        for path in artifact_paths
    )


def is_forge_run(tags: dict[str, Any]) -> bool:
    """Return whether MLflow metadata identifies a Forge multi-run export."""
    return str(tags.get("forge.multi_run") or "").strip().lower() == "true"


def _load_yaml(path: str) -> dict[str, Any]:
    with Path(path).open() as stream:
        payload = yaml.safe_load(stream)
    return payload if isinstance(payload, dict) else {}


def _download_yaml(
    artifact_uri: str, artifact_path: str, cache_dir: Path
) -> dict[str, Any]:
    downloaded_path = runtime._download_run_artifact(
        artifact_uri, artifact_path, str(cache_dir)
    )
    return _load_yaml(downloaded_path)


def _forge_test_directories(artifact_uri: str) -> list[str]:
    """List Forge tests without crawling every artifact produced by each test."""
    repo = get_artifact_repository(artifact_uri)
    for root_path in ("01__test", "artifacts/01__test"):
        try:
            entries = repo.list_artifacts(root_path)
        except Exception:  # noqa: BLE001
            continue
        test_directories = sorted(
            entry.path
            for entry in entries
            if entry.is_dir and re.fullmatch(r"\d+__llmd__.+", Path(entry.path).name)
        )
        if test_directories:
            return test_directories
    raise ValidationError(
        "Forge MLflow run does not contain an 01__test artifact directory"
    )


def _benchmark_index_path(artifact_uri: str, test_root: str) -> str:
    """Locate Forge's benchmark task without listing its result artifacts."""
    repo = get_artifact_repository(artifact_uri)
    test_path = f"{test_root}/000__llmd_test"
    entries = repo.list_artifacts(test_path)
    benchmark_tasks = sorted(
        entry.path
        for entry in entries
        if entry.is_dir and "__benchmark_" in Path(entry.path).name
    )
    if len(benchmark_tasks) != 1:
        raise ValidationError(
            f"Forge test {Path(test_root).name} must contain exactly one benchmark task; "
            f"found {len(benchmark_tasks)}"
        )
    return (
        f"{benchmark_tasks[0]}/000__run_guidellm_benchmark/artifacts/results/index.json"
    )


def _accelerator_from_artifact(
    artifact_uri: str, test_root: str, cache_dir: Path
) -> str:
    """Extract the GPU SKU from Forge's captured deployment node assignments."""
    repo = get_artifact_repository(artifact_uri)
    test_path = f"{test_root}/000__llmd_test"
    capture_tasks = sorted(
        entry.path
        for entry in repo.list_artifacts(test_path)
        if entry.is_dir and "__capture_llmisvc_state" in Path(entry.path).name
    )
    if len(capture_tasks) != 1:
        raise ValidationError(
            f"Forge test {Path(test_root).name} must contain exactly one captured "
            f"LLMInferenceService state task; found {len(capture_tasks)}"
        )

    artifact_path = f"{capture_tasks[0]}/artifacts/namespace.pods.status"
    downloaded_path = runtime._download_run_artifact(
        artifact_uri, artifact_path, str(cache_dir)
    )
    try:
        pod_status = Path(downloaded_path).read_text()
    except OSError as exc:
        raise ValidationError(
            f"cannot read Forge deployment node assignments {artifact_path}: {exc}"
        ) from exc

    accelerator_matches = re.findall(
        r"(?:^|-)gpu-(h\d+|a\d+(?:-\d+gb)?|mi\d+[a-z0-9]*|l\d+|b\d+)(?:-|\s|$)",
        pod_status,
        flags=re.IGNORECASE,
    )
    accelerators = {match.upper() for match in accelerator_matches}
    if len(accelerators) != 1:
        rendered = ", ".join(sorted(accelerators)) or "none"
        raise ValidationError(
            f"Forge deployment node assignments for {Path(test_root).name} must "
            f"identify exactly one GPU SKU; found {rendered}"
        )
    return accelerators.pop()


def _result_artifact_paths(
    artifact_uri: str, index_path: str, cache_dir: Path
) -> list[str]:
    """Return all GuideLLM JSON result paths declared by Forge's index."""
    downloaded_path = runtime._download_run_artifact(
        artifact_uri, index_path, str(cache_dir)
    )
    try:
        with Path(downloaded_path).open() as stream:
            index_data = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(
            f"invalid Forge benchmark index {index_path}: {exc}"
        ) from exc

    result_paths: list[str] = []
    for entry in index_data.get("runs", []):
        if not isinstance(entry, dict):
            continue
        local_path = str(entry.get("local_path") or "")
        if local_path:
            result_paths.append(
                str(
                    Path(index_path).parent
                    / local_path.removeprefix("artifacts/results/")
                )
            )
    return result_paths


def _deployment_values(
    config: dict[str, Any], deployment_profile: str
) -> dict[str, Any]:
    deployments = config.get("deployments")
    if not isinstance(deployments, dict):
        return {}
    defaults = deployments.get("defaults")
    profiles = deployments.get("profiles")
    profile = profiles.get(deployment_profile) if isinstance(profiles, dict) else None
    values: dict[str, Any] = {}
    if isinstance(defaults, dict):
        values.update(defaults)
    if isinstance(profile, dict):
        values.update(profile)
    return values


def _workload_data_profile(config: dict[str, Any], workload: str) -> dict[str, Any]:
    """Return the static GuideLLM data-profile settings for a Forge workload."""
    workloads = config.get("workloads")
    benchmarks = workloads.get("benchmarks") if isinstance(workloads, dict) else None
    benchmark = benchmarks.get(workload) if isinstance(benchmarks, dict) else None
    args = benchmark.get("args") if isinstance(benchmark, dict) else None
    data = args.get("data") if isinstance(args, dict) else None

    if isinstance(data, dict):
        return {str(key): value for key, value in data.items() if value is not None}
    if not isinstance(data, str):
        return {}

    profile: dict[str, Any] = {}
    for item in data.split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            profile[key] = value
    return profile


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _compose_version(
    params: dict[str, Any], tags: dict[str, Any], model: str, deployment_profile: str
) -> str:
    base_version = str(
        params.get("version")
        or tags.get("version")
        or params.get("image_tag")
        or model
        or "unknown"
    ).strip()
    suffix = deployment_profile.strip()
    return f"{base_version}-{suffix}" if suffix else base_version


def fetch_mlflow_runs(
    run_ids: list[str],
    mlflow_tracking_uri: str | None = None,
    accelerator: str | None = None,
    forge_workload: str | None = None,
) -> list[dict[str, Any]]:
    """Expand each Forge test in an MLflow run into a report comparison input."""
    configure_mlflow_tracking(mlflow_tracking_uri)
    client = create_mlflow_client(mlflow_tracking_uri)
    runs_data: list[dict[str, Any]] = []
    if accelerator:
        logger.info(
            "Ignoring --accelerator=%s for Forge runs; using captured deployment nodes",
            accelerator,
        )

    for mlflow_run_id in run_ids:
        run = client.get_run(mlflow_run_id)
        params = dict(run.data.params)
        tags = dict(run.data.tags)
        artifact_uri = run.info.artifact_uri
        test_roots = _forge_test_directories(artifact_uri)

        logger.info(
            "Expanding Forge MLflow run %s into %d test result(s)",
            mlflow_run_id,
            len(test_roots),
        )
        for test_root in test_roots:
            test_id = Path(test_root).name
            cache_dir = Path("/tmp/mlflow") / mlflow_run_id / "forge" / test_id
            cache_dir.mkdir(parents=True, exist_ok=True)
            labels_path = f"{test_root}{_TEST_LABELS_SUFFIX}"
            labels_data = _download_yaml(artifact_uri, labels_path, cache_dir)
            labels = labels_data.get("labels")
            labels = labels if isinstance(labels, dict) else {}
            workload = str(labels.get("guidellm_loadshape") or "").strip()
            if forge_workload and workload != forge_workload:
                continue
            deployment_profile = str(labels.get("deployment_profile") or "").strip()
            model = str(labels.get("model_name") or params.get("model") or "unknown")

            config_path = f"{test_root}/000__llmd_test/config.yaml"
            config = _download_yaml(artifact_uri, config_path, cache_dir)
            deployment_values = _deployment_values(config, deployment_profile)
            workload_data_profile = _workload_data_profile(config, workload)
            prompt_tokens = _as_int(workload_data_profile.get("prompt_tokens"))
            output_tokens = _as_int(workload_data_profile.get("output_tokens"))
            if prompt_tokens is None or output_tokens is None:
                raise ValidationError(
                    f"Forge workload {forge_workload} in test {test_id} does not "
                    "define prompt_tokens and output_tokens"
                )
            tp_size = _as_int(deployment_values.get("tensor_parallelism"))
            tp_size = tp_size or 1
            replicas = _as_int(deployment_values.get("replicas")) or 1
            accelerator_name = _accelerator_from_artifact(
                artifact_uri, test_root, cache_dir
            )
            composed_version = _compose_version(params, tags, model, deployment_profile)
            index_path = _benchmark_index_path(artifact_uri, test_root)
            result_paths = _result_artifact_paths(artifact_uri, index_path, cache_dir)
            if not result_paths:
                raise ValidationError(
                    f"Forge test {test_id} in MLflow run {mlflow_run_id} has no GuideLLM results"
                )

            benchmarks: list[dict[str, Any]] = []
            for result_path in result_paths:
                downloaded_path = runtime._download_run_artifact(
                    artifact_uri, result_path, str(cache_dir)
                )
                try:
                    with Path(downloaded_path).open() as stream:
                        payload = json.load(stream)
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValidationError(
                        f"invalid Forge benchmark result {result_path}: {exc}"
                    ) from exc
                result_benchmarks = payload.get("benchmarks")
                if isinstance(result_benchmarks, list):
                    benchmarks.extend(result_benchmarks)

            if not benchmarks:
                raise ValidationError(
                    f"Forge test {test_id} in MLflow run {mlflow_run_id} has no benchmark data"
                )

            combined_path = cache_dir / "combined_benchmarks.json"
            combined_path.write_text(json.dumps({"benchmarks": benchmarks}))
            runs_data.append(
                {
                    "run_id": f"{mlflow_run_id}:{test_id}",
                    "params": {
                        "model": model,
                        "accelerator": accelerator_name,
                        "tp": str(tp_size),
                        "replicas": str(replicas),
                        **workload_data_profile,
                    },
                    "tags": {
                        "deployment_profile": deployment_profile,
                        "forge_test_id": test_id,
                        "forge_workload": workload,
                    },
                    "artifact_uri": artifact_uri,
                    "composed_version": composed_version,
                    "artifact_path": str(combined_path),
                    "workload_data_profile": workload_data_profile,
                }
            )

    if not runs_data:
        scope = f" matching Forge workload {forge_workload}" if forge_workload else ""
        raise ValidationError(f"No Forge tests found{scope}")
    workloads = {str(run_data["tags"]["forge_workload"]) for run_data in runs_data}
    if len(workloads) > 1:
        raise ValidationError(
            "Forge comparison reports require --forge-workload when a run contains "
            f"multiple workloads; available: {', '.join(sorted(workloads))}"
        )
    models = {str(run_data["params"]["model"]) for run_data in runs_data}
    if len(models) != 1:
        raise ValidationError(
            "Forge comparison reports require one model per report; "
            f"found: {', '.join(sorted(models))}"
        )
    return runs_data


def generate_report(
    *,
    mlflow_run_ids: list[str],
    mlflow_tracking_uri: str | None = None,
    accelerator: str | None = None,
    forge_workload: str | None = None,
    versions: list[str] | None = None,
    version_overrides: dict[str, str] | None = None,
    additional_csv_files: list[str] | None = None,
    notes: list[str] | None = None,
    repeat_section_legends: bool = False,
    include_total_throughput: bool = False,
    baseline_version: str | None = None,
    metrics_yaml_path: Path | None = None,
    output_dir: Path | None = None,
    output_file: Path | None = None,
) -> Path:
    """Generate the existing OpenShift AI comparison report from Forge results."""
    runs_data = fetch_mlflow_runs(
        mlflow_run_ids,
        mlflow_tracking_uri,
        accelerator,
        forge_workload,
    )
    workload_data_profile = None
    if forge_workload:
        profiles = {
            tuple(sorted(dict(run_data.get("workload_data_profile") or {}).items()))
            for run_data in runs_data
        }
        if not profiles or len(profiles) != 1:
            raise ValidationError(
                f"Forge workload {forge_workload} does not have one canonical "
                "data profile across its selected tests"
            )
        workload_data_profile = dict(profiles.pop())
        if not workload_data_profile:
            raise ValidationError(
                f"Forge workload {forge_workload} has no canonical data profile"
            )
    html_path = runtime.generate_plot_only_report(
        runs_data=runs_data,
        versions=versions,
        mlflow_tracking_uri=mlflow_tracking_uri,
        additional_csv_files=additional_csv_files,
        versions_override=version_overrides or {},
        output_dir=str(output_dir) if output_dir else None,
        output_file=str(output_file) if output_file else None,
        notes=notes or [],
        repeat_section_legends=repeat_section_legends,
        include_total_throughput=include_total_throughput,
        baseline_version=baseline_version,
        metrics_yaml_path=str(metrics_yaml_path) if metrics_yaml_path else None,
        workload_data_profile=workload_data_profile,
        # A Forge run deliberately contains a workload/deployment matrix.
        force=True,
    )
    if not html_path:
        raise ValidationError(
            "Forge comparison report generation returned no output path"
        )
    return Path(html_path)
