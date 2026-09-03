from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import mlflow
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

from .cluster import require_any_command, run_command
from .mlflow_compat import create_mlflow_client, configure_mlflow_tracking
from .models import ResolvedRunPlan
from .remote_jobs import copy_remote_results_directory, delete_remote_results_directory
from .reports import generate_post_run_reports
from .ui import detail, step, success, warning

MLFLOW_MULTIPART_MINIMUM_FILE_SIZE = 64 * 1024 * 1024
MLFLOW_MULTIPART_CHUNK_SIZE = 64 * 1024 * 1024
MLFLOW_UPLOAD_ATTEMPTS = 3

_TRANSIENT_MLFLOW_ERROR_MARKERS = (
    "connection aborted",
    "connection refused",
    "connection reset",
    "gateway timeout",
    "max retries exceeded",
    "read timed out",
    "remote end closed connection",
    "responseerror",
    "service unavailable",
    "temporarily unavailable",
    "timeout",
    "too many 429",
    "too many 500",
    "too many 502",
    "too many 503",
    "too many 504",
)


class ArtifactUploadFailed(RuntimeError):
    pass


def _configure_resilient_artifact_uploads() -> None:
    os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD", "true")
    os.environ.setdefault(
        "MLFLOW_MULTIPART_UPLOAD_MINIMUM_FILE_SIZE",
        str(MLFLOW_MULTIPART_MINIMUM_FILE_SIZE),
    )
    os.environ.setdefault(
        "MLFLOW_MULTIPART_UPLOAD_CHUNK_SIZE", str(MLFLOW_MULTIPART_CHUNK_SIZE)
    )
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "600")


def is_transient_mlflow_error(exc: BaseException) -> bool:
    message = str(exc).strip().lower()
    return any(marker in message for marker in _TRANSIENT_MLFLOW_ERROR_MARKERS)


def _retry_mlflow_write(operation: Callable[[], None], *, description: str) -> None:
    for attempt in range(1, MLFLOW_UPLOAD_ATTEMPTS + 1):
        try:
            operation()
            return
        except Exception as exc:  # noqa: BLE001
            if not is_transient_mlflow_error(exc) or attempt >= MLFLOW_UPLOAD_ATTEMPTS:
                raise
            delay = 30 * attempt
            warning(
                f"Transient MLflow error while {description}; "
                f"retrying in {delay}s: {exc}"
            )
            time.sleep(delay)


def _iso8601_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _mlflow_tracking_uri() -> str:
    return str(os.environ.get("MLFLOW_TRACKING_URI", "")).strip()


def _run_name(plan: ResolvedRunPlan, execution_name: str = "") -> str:
    return (
        str(execution_name or "").strip()
        or plan.deployment.release_name
        or f"{plan.model.name.split('/')[-1]}-{_iso8601_now()}"
    )


def _initial_run_tags(
    plan: ResolvedRunPlan, execution_name: str = ""
) -> dict[str, str]:
    tags = dict(plan.mlflow.tags)
    tags.update(
        {
            "benchmark_tool": plan.benchmark.tool,
            "benchflow.run_initialized": "true",
            "deployment_profile": plan.profiles.deployment,
            "deployment_type": f"{plan.deployment.platform}-{plan.deployment.mode}",
            "model": plan.model.name,
            "version": plan.mlflow.version,
            # MlflowClient.create_run does not add these system tags like
            # mlflow.start_run does, leaving the UI Source column empty.
            "mlflow.source.name": "bflow",
            "mlflow.source.type": "LOCAL",
        }
    )
    if execution_name:
        tags["execution_name"] = execution_name
    tags["mlflow.runName"] = _run_name(plan, execution_name)
    return {key: value for key, value in tags.items() if value}


def initialize_mlflow_run(
    plan: ResolvedRunPlan,
    *,
    execution_name: str = "",
) -> tuple[str, str]:
    run_start_time = _iso8601_now()
    explicit_tracking_uri = _mlflow_tracking_uri()
    if not explicit_tracking_uri:
        warning(
            "Skipping MLflow run initialization because MLFLOW_TRACKING_URI is not configured"
        )
        return "", run_start_time

    configure_mlflow_tracking(explicit_tracking_uri)
    experiment_name = plan.mlflow.experiment or "Default"
    experiment = mlflow.set_experiment(experiment_name)
    client = create_mlflow_client(explicit_tracking_uri)
    run = client.create_run(
        experiment.experiment_id,
        tags=_initial_run_tags(plan, execution_name),
    )
    detail(f"Initialized MLflow run {run.info.run_id} in experiment {experiment_name}")
    return run.info.run_id, run_start_time


def mark_mlflow_run(
    *,
    mlflow_run_id: str,
    status: str,
    reason: str = "",
) -> None:
    explicit_tracking_uri = _mlflow_tracking_uri()
    if not mlflow_run_id or not explicit_tracking_uri:
        return
    client = create_mlflow_client(explicit_tracking_uri)
    normalized_status = status.strip().upper() or "FAILED"
    failures: list[str] = []
    if reason:
        try:
            _retry_mlflow_write(
                lambda: client.set_tag(
                    mlflow_run_id, "benchflow.finalizer_reason", reason
                ),
                description=f"setting finalizer reason on MLflow run {mlflow_run_id}",
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"finalizer reason tag: {exc}")
    try:
        _retry_mlflow_write(
            lambda: client.set_tag(
                mlflow_run_id, "benchflow.final_status", normalized_status
            ),
            description=f"setting final status on MLflow run {mlflow_run_id}",
        )
    except Exception as exc:  # noqa: BLE001
        failures.append(f"final status tag: {exc}")
    try:
        _retry_mlflow_write(
            lambda: client.set_terminated(mlflow_run_id, status=normalized_status),
            description=f"terminating MLflow run {mlflow_run_id}",
        )
    except Exception as exc:  # noqa: BLE001
        failures.append(f"run termination: {exc}")
    if failures:
        raise ArtifactUploadFailed(
            f"failed to finalize MLflow run {mlflow_run_id}: {'; '.join(failures)}"
        )


def _discover_grafana_base_url(namespace: str) -> str:
    kubectl_cmd = require_any_command("oc", "kubectl")
    for candidate_namespace in (
        f"{namespace}-grafana",
        namespace,
        "benchflow",
        "grafana",
        "openshift-monitoring",
    ):
        if not candidate_namespace:
            continue
        result = run_command(
            [
                kubectl_cmd,
                "get",
                "route",
                "-n",
                candidate_namespace,
                "-o",
                'jsonpath={range .items[*]}{.metadata.name}{"\\t"}{.spec.host}{"\\n"}{end}',
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            continue
        for line in result.stdout.splitlines():
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            name, host = parts
            if "grafana" in name and host:
                return f"https://{host}"
    return ""


def _build_grafana_url(
    plan: ResolvedRunPlan,
    run_id: str,
    benchmark_start_time: str,
    benchmark_end_time: str,
    grafana_base_url: str,
) -> str:
    if not grafana_base_url or not benchmark_start_time or not benchmark_end_time:
        return ""
    start_dt = datetime.fromisoformat(benchmark_start_time.replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(benchmark_end_time.replace("Z", "+00:00"))
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    return (
        f"{grafana_base_url}/d/benchflow"
        f"?var-run_id={run_id}"
        f"&var-namespace={plan.deployment.namespace}"
        f"&var-release={plan.deployment.target.scoped_release_name(plan.deployment.release_name)}"
        f"&from={start_ms}"
        f"&to={end_ms}"
    )


def _cleanup_dir_contents(
    directory: Path, *, preserve_names: set[str] | None = None
) -> None:
    if not directory.exists():
        return
    preserved = preserve_names or set()
    for item in directory.iterdir():
        if item.name in preserved:
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def _count_files(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for path in directory.rglob("*") if path.is_file())


def _load_json_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8") or "{}")
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _merge_artifact_tree(source_dir: Path, target_dir: Path) -> None:
    for child in source_dir.iterdir():
        target = target_dir / child.name
        if child.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            _merge_artifact_tree(child, target)
            continue
        if child.name == "metadata.json" and target.exists():
            merged = {**_load_json_file(child), **_load_json_file(target)}
            target.write_text(json.dumps(merged, indent=2), encoding="utf-8")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(child, target)


def _materialize_remote_tree_if_needed(
    plan: ResolvedRunPlan,
    *,
    reference_path: Path,
    local_dir: Path,
    label: str,
) -> bool:
    if not reference_path.exists():
        return False

    reference = _load_json_file(reference_path)
    remote_path = str(reference.get("remote_path") or "").strip()
    if not remote_path:
        return False
    if str(reference.get("status") or "").strip() in {"materialized", "uploaded"}:
        return False

    step(f"Copying remote collected {label} from {remote_path}")
    temp_root = Path(tempfile.mkdtemp(prefix="benchflow-remote-artifacts-"))
    try:
        copy_remote_results_directory(
            plan,
            remote_path=remote_path,
            local_dir=temp_root,
            cleanup=False,
        )
        _merge_artifact_tree(temp_root, local_dir)
        reference["status"] = "materialized"
        reference["uploaded_to_mlflow"] = False
        reference_path.write_text(json.dumps(reference, indent=2), encoding="utf-8")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
    return True


def _materialize_remote_artifacts_if_needed(
    plan: ResolvedRunPlan,
    artifacts_dir: Path,
) -> bool:
    return _materialize_remote_tree_if_needed(
        plan,
        reference_path=artifacts_dir / "remote-target-artifacts.json",
        local_dir=artifacts_dir,
        label="artifacts",
    )


def _materialize_remote_metrics_if_needed(
    plan: ResolvedRunPlan,
    artifacts_dir: Path,
) -> bool:
    metrics_dir = artifacts_dir / "metrics"
    if (metrics_dir / "metrics_summary.json").exists():
        return False
    return _materialize_remote_tree_if_needed(
        plan,
        reference_path=metrics_dir / "remote-target-metrics.json",
        local_dir=metrics_dir,
        label="metrics",
    )


def _remote_materialization_references(artifacts_dir: Path) -> list[Path]:
    return [
        artifacts_dir / "remote-target-artifacts.json",
        artifacts_dir / "metrics" / "remote-target-metrics.json",
    ]


def _mark_remote_materializations_uploaded(artifacts_dir: Path) -> list[str]:
    remote_paths: list[str] = []
    for reference_path in _remote_materialization_references(artifacts_dir):
        if not reference_path.exists():
            continue
        reference = _load_json_file(reference_path)
        remote_path = str(reference.get("remote_path") or "").strip().rstrip("/")
        if not remote_path:
            continue
        reference["status"] = "uploaded"
        reference["uploaded_to_mlflow"] = True
        reference_path.write_text(json.dumps(reference, indent=2), encoding="utf-8")
        remote_paths.append(remote_path)

    # Metrics commonly live below the artifact collection directory. Deleting
    # the parent once is sufficient and avoids creating another reader pod.
    roots: list[str] = []
    for candidate in sorted(set(remote_paths), key=lambda value: (len(value), value)):
        if any(candidate == root or candidate.startswith(f"{root}/") for root in roots):
            continue
        roots.append(candidate)
    return roots


def _cleanup_uploaded_remote_materializations(
    plan: ResolvedRunPlan,
    *,
    artifacts_dir: Path,
) -> None:
    for remote_path in _mark_remote_materializations_uploaded(artifacts_dir):
        try:
            delete_remote_results_directory(plan, remote_path=remote_path)
        except Exception as exc:  # noqa: BLE001
            warning(
                f"MLflow upload is complete, but remote artifacts remain at "
                f"{remote_path}: {exc}"
            )


def _write_run_plan_artifact(plan: ResolvedRunPlan, artifacts_dir: Path) -> Path:
    metadata_dir = artifacts_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    run_plan_path = metadata_dir / "run-plan.json"
    run_plan_path.write_text(
        json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    detail(f"Wrote RunPlan artifact to {run_plan_path}")
    return run_plan_path


def _list_run_artifact_paths(
    client, mlflow_run_id: str, root_path: str = ""
) -> set[str]:
    run = client.get_run(mlflow_run_id)
    repo = get_artifact_repository(run.info.artifact_uri)
    pending = [root_path.strip("/")]
    discovered: set[str] = set()

    while pending:
        current_path = pending.pop()
        for entry in repo.list_artifacts(current_path):
            if entry.is_dir:
                pending.append(entry.path)
                continue
            discovered.add(entry.path)

    return discovered


def _benchmark_workspace_artifact_root(relative_path: Path) -> str:
    suffix = relative_path.suffix.lower()
    if suffix == ".log":
        return "logs"
    if suffix in {".html", ".htm"}:
        return "reports"
    if suffix in {".json", ".csv"}:
        return "results"
    return "benchmark"


def _upload_missing_benchmark_artifacts(
    *,
    client,
    mlflow_run_id: str,
    benchmark_dir: Path,
) -> int:
    if not benchmark_dir.exists():
        return 0

    candidate_files: list[tuple[Path, str, str]] = []
    for file_path in sorted(
        path for path in benchmark_dir.rglob("*") if path.is_file()
    ):
        relative_path = file_path.relative_to(benchmark_dir)
        if relative_path.name.startswith("."):
            continue
        artifact_root = _benchmark_workspace_artifact_root(relative_path)
        artifact_dir = (
            Path(artifact_root) / relative_path.parent
            if relative_path.parent != Path(".")
            else Path(artifact_root)
        )
        artifact_dir_str = artifact_dir.as_posix().strip("/")
        final_artifact_path = (
            f"{artifact_dir_str}/{relative_path.name}"
            if artifact_dir_str
            else relative_path.name
        )
        candidate_files.append((file_path, artifact_dir_str, final_artifact_path))

    if not candidate_files:
        return 0

    existing_paths: set[str] = set()
    for artifact_root in sorted(
        {
            final_path.split("/", 1)[0]
            for _, _, final_path in candidate_files
            if "/" in final_path
        }
    ):
        try:
            existing_paths.update(
                _list_run_artifact_paths(client, mlflow_run_id, artifact_root)
            )
        except Exception as exc:  # noqa: BLE001
            warning(
                "Failed to inspect existing MLflow artifacts under "
                f"{artifact_root}: {exc}. Uploading benchmark workspace fallback files anyway."
            )
            existing_paths.clear()
            break

    uploaded_count = 0
    for file_path, artifact_dir, final_artifact_path in candidate_files:
        if final_artifact_path in existing_paths:
            continue
        detail(
            f"Uploading fallback benchmark artifact {file_path.name}"
            + (f" to {artifact_dir}" if artifact_dir else "")
        )
        _upload_artifact_with_recovery(
            client,
            mlflow_run_id=mlflow_run_id,
            file_path=file_path,
            artifact_dir=artifact_dir,
            remote_path=final_artifact_path,
        )
        uploaded_count += 1

    return uploaded_count


def _artifact_candidates(
    artifacts_dir: Path,
    *,
    prefix: str,
    excluded: set[str],
) -> list[tuple[Path, str, str]]:
    candidates: list[tuple[Path, str, str]] = []
    if not artifacts_dir.exists():
        return candidates
    for file_path in sorted(
        path for path in artifacts_dir.rglob("*") if path.is_file()
    ):
        relative_path = file_path.relative_to(artifacts_dir)
        if relative_path.parts and relative_path.parts[0] in excluded:
            continue
        remote_path = "/".join(
            part for part in (prefix, relative_path.as_posix()) if part
        )
        remote_dir = str(Path(remote_path).parent.as_posix())
        if remote_dir == ".":
            remote_dir = ""
        candidates.append((file_path, remote_dir, remote_path))
    return candidates


def _run_artifact_sizes(client, mlflow_run_id: str) -> dict[str, int]:
    pending = [""]
    discovered: dict[str, int] = {}
    while pending:
        current_path = pending.pop()
        for entry in client.list_artifacts(mlflow_run_id, current_path):
            if entry.is_dir:
                pending.append(entry.path)
            else:
                discovered[entry.path] = int(entry.file_size or 0)
    return discovered


def _committed_artifact_size(
    client,
    *,
    mlflow_run_id: str,
    remote_path: str,
) -> int | None:
    parent = str(Path(remote_path).parent.as_posix())
    if parent == ".":
        parent = ""
    for entry in client.list_artifacts(mlflow_run_id, parent):
        if not entry.is_dir and entry.path == remote_path:
            return int(entry.file_size or 0)
    return None


def _upload_artifact_with_recovery(
    client,
    *,
    mlflow_run_id: str,
    file_path: Path,
    artifact_dir: str,
    remote_path: str,
) -> None:
    expected_size = file_path.stat().st_size
    for attempt in range(1, MLFLOW_UPLOAD_ATTEMPTS + 1):
        try:
            client.log_artifact(
                mlflow_run_id,
                str(file_path),
                artifact_path=artifact_dir or None,
            )
            return
        except Exception as exc:  # noqa: BLE001
            try:
                committed_size = _committed_artifact_size(
                    client,
                    mlflow_run_id=mlflow_run_id,
                    remote_path=remote_path,
                )
            except Exception:  # noqa: BLE001
                committed_size = None
            if committed_size == expected_size:
                detail(f"MLflow committed {remote_path} despite an ambiguous response")
                return
            if not is_transient_mlflow_error(exc) or attempt >= MLFLOW_UPLOAD_ATTEMPTS:
                raise
            delay = 30 * attempt
            warning(
                f"Transient MLflow error uploading {remote_path}; "
                f"retrying in {delay}s: {exc}"
            )
            time.sleep(delay)


def upload_artifact_directory_to_mlflow(
    *,
    mlflow_run_id: str,
    artifacts_dir: Path,
    artifact_path_prefix: str = "",
    cleanup_after_upload: bool = False,
    preserve_names: set[str] | None = None,
    exclude_names: set[str] | None = None,
) -> Path:
    _configure_resilient_artifact_uploads()
    explicit_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if explicit_tracking_uri:
        configure_mlflow_tracking(explicit_tracking_uri)
    if not mlflow_run_id or not explicit_tracking_uri:
        warning(
            "Skipping MLflow upload because "
            + (
                "the run ID is missing"
                if not mlflow_run_id
                else "MLFLOW_TRACKING_URI is not configured"
            )
        )
        return artifacts_dir

    client = create_mlflow_client(explicit_tracking_uri)
    prefix = artifact_path_prefix.strip("/")
    excluded = exclude_names or set()
    candidates = _artifact_candidates(artifacts_dir, prefix=prefix, excluded=excluded)
    try:
        existing_sizes = _run_artifact_sizes(client, mlflow_run_id)
    except Exception as exc:  # noqa: BLE001
        warning(
            f"Could not inventory existing MLflow artifacts; uploading all candidates: {exc}"
        )
        existing_sizes = {}

    artifact_count = 0
    skipped_count = 0
    failures: list[str] = []
    for index, (file_path, artifact_dir, remote_path) in enumerate(candidates, start=1):
        expected_size = file_path.stat().st_size
        if existing_sizes.get(remote_path) == expected_size:
            skipped_count += 1
            continue
        detail(
            f"Uploading artifact {index}/{len(candidates)}: "
            f"{remote_path} ({expected_size} bytes)"
        )
        try:
            _upload_artifact_with_recovery(
                client,
                mlflow_run_id=mlflow_run_id,
                file_path=file_path,
                artifact_dir=artifact_dir,
                remote_path=remote_path,
            )
            artifact_count += 1
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{remote_path}: {exc}")
            warning(
                f"Failed to upload {remote_path}; continuing with other files: {exc}"
            )

    detail(
        f"Uploaded {artifact_count} artifact file(s), skipped {skipped_count} "
        f"already complete file(s) from {artifacts_dir}"
        + (f" to {prefix}" if prefix else "")
    )
    if failures:
        preview = "; ".join(failures[:5])
        if len(failures) > 5:
            preview += f"; and {len(failures) - 5} more"
        raise ArtifactUploadFailed(
            f"failed to upload {len(failures)} MLflow artifact file(s): {preview}"
        )
    if cleanup_after_upload and artifacts_dir.exists():
        _cleanup_dir_contents(artifacts_dir, preserve_names=preserve_names)
    return artifacts_dir


def upload_to_mlflow(
    plan: ResolvedRunPlan,
    *,
    mlflow_run_id: str,
    benchmark_start_time: str,
    benchmark_end_time: str,
    artifacts_dir: Path,
    grafana_url: str = "",
) -> Path:
    _configure_resilient_artifact_uploads()
    explicit_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if explicit_tracking_uri:
        configure_mlflow_tracking(explicit_tracking_uri)
    step(
        f"Uploading artifacts to MLflow run {mlflow_run_id or '<missing>'} "
        f"for release {plan.deployment.release_name}"
    )
    if not mlflow_run_id or not explicit_tracking_uri:
        warning(
            "Skipping MLflow upload because "
            + (
                "the run ID is missing"
                if not mlflow_run_id
                else "MLFLOW_TRACKING_URI is not configured"
            )
        )
        return artifacts_dir

    client = create_mlflow_client(explicit_tracking_uri)
    detail(f"MLflow tracking URI: {explicit_tracking_uri}")
    materialization_failures: list[str] = []
    for label, materialize in (
        ("artifacts", _materialize_remote_artifacts_if_needed),
        ("metrics", _materialize_remote_metrics_if_needed),
    ):
        try:
            materialize(plan, artifacts_dir)
        except Exception as exc:  # noqa: BLE001
            materialization_failures.append(f"{label}: {exc}")
            warning(
                f"Could not materialize remote {label}; "
                f"continuing with the durable local bundle: {exc}"
            )
    grafana_base_url = grafana_url or _discover_grafana_base_url(
        plan.deployment.namespace
    )
    full_grafana_url = _build_grafana_url(
        plan,
        mlflow_run_id,
        benchmark_start_time,
        benchmark_end_time,
        grafana_base_url,
    )
    if full_grafana_url:
        detail(f"Setting Grafana URL tag: {full_grafana_url}")
        client.set_tag(mlflow_run_id, "grafana_url", full_grafana_url)

    artifact_count = 0
    fallback_count = 0
    if artifacts_dir.exists():
        _write_run_plan_artifact(plan, artifacts_dir)
        benchmark_dir = artifacts_dir / "benchmark"
        if benchmark_dir.exists():
            fallback_count = _upload_missing_benchmark_artifacts(
                client=client,
                mlflow_run_id=mlflow_run_id,
                benchmark_dir=benchmark_dir,
            )
            if fallback_count:
                detail(
                    "Uploaded "
                    f"{fallback_count} fallback benchmark artifact file(s) from {benchmark_dir}"
                )
            else:
                detail(
                    "No fallback benchmark artifacts were needed; MLflow already "
                    "contained the benchmark results and console output"
                )
        generate_post_run_reports(artifacts_dir=artifacts_dir)
        before_cleanup = _count_files(artifacts_dir)
        upload_artifact_directory_to_mlflow(
            mlflow_run_id=mlflow_run_id,
            artifacts_dir=artifacts_dir,
            cleanup_after_upload=False,
            exclude_names={"benchmark"},
        )
        artifact_count = fallback_count + (
            before_cleanup - _count_files(artifacts_dir / "benchmark")
            if (artifacts_dir / "benchmark").exists()
            else before_cleanup
        )
    detail(
        f"Uploaded {artifact_count} additional artifact file(s) from {artifacts_dir}"
    )
    if materialization_failures:
        raise ArtifactUploadFailed(
            "MLflow received the available local artifacts, but remote materialization "
            f"was incomplete: {'; '.join(materialization_failures)}"
        )
    _cleanup_uploaded_remote_materializations(plan, artifacts_dir=artifacts_dir)
    success(f"MLflow upload complete for run {mlflow_run_id}")
    if artifacts_dir.exists():
        _cleanup_dir_contents(artifacts_dir, preserve_names={"platform-state"})
    return artifacts_dir
