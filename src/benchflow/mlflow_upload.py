from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

from .benchmark import generate_run_report
from .cluster import require_any_command, run_command
from .models import ResolvedRunPlan
from .remote_jobs import copy_remote_results_directory
from .ui import detail, step, success, warning


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
            shutil.copytree(child, target, dirs_exist_ok=True)
            continue
        if child.name == "metadata.json" and target.exists():
            merged = {**_load_json_file(child), **_load_json_file(target)}
            target.write_text(json.dumps(merged, indent=2), encoding="utf-8")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(child, target)


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

    step(f"Copying remote collected {label} from {remote_path}")
    temp_root = Path(tempfile.mkdtemp(prefix="benchflow-remote-artifacts-"))
    try:
        copy_remote_results_directory(
            plan,
            remote_path=remote_path,
            local_dir=temp_root,
        )
        _merge_artifact_tree(temp_root, local_dir)
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


def _generate_post_run_report(artifacts_dir: Path) -> Path | None:
    try:
        report_path = generate_run_report(artifacts_dir=artifacts_dir)
    except Exception as exc:  # noqa: BLE001
        warning(f"Skipping post-run report generation: {exc}")
        return None
    detail(f"Generated post-run report at {report_path}")
    return report_path


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
        client.log_artifact(
            mlflow_run_id,
            str(file_path),
            artifact_path=artifact_dir or None,
        )
        uploaded_count += 1

    return uploaded_count


def upload_artifact_directory_to_mlflow(
    *,
    mlflow_run_id: str,
    artifacts_dir: Path,
    artifact_path_prefix: str = "",
    cleanup_after_upload: bool = False,
    preserve_names: set[str] | None = None,
    exclude_names: set[str] | None = None,
) -> Path:
    import mlflow

    explicit_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if explicit_tracking_uri:
        mlflow.set_tracking_uri(explicit_tracking_uri)
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

    client = mlflow.tracking.MlflowClient()
    prefix = artifact_path_prefix.strip("/")
    excluded = exclude_names or set()
    artifact_count = 0

    if artifacts_dir.exists():
        for child in sorted(artifacts_dir.iterdir()):
            if child.name in excluded:
                continue
            if child.is_dir():
                child_count = _count_files(child)
                if child_count == 0:
                    continue
                artifact_path = "/".join(part for part in (prefix, child.name) if part)
                detail(
                    f"Uploading directory {child.name} ({child_count} file(s)) to MLflow"
                    + (f" under {artifact_path}" if artifact_path else "")
                )
                client.log_artifacts(
                    mlflow_run_id,
                    str(child),
                    artifact_path=artifact_path or None,
                )
                artifact_count += child_count
            elif child.is_file():
                detail(
                    f"Uploading file {child.name} to MLflow"
                    + (f" under {prefix}" if prefix else "")
                )
                client.log_artifact(
                    mlflow_run_id,
                    str(child),
                    artifact_path=prefix or None,
                )
                artifact_count += 1

    detail(
        f"Uploaded {artifact_count} artifact file(s) from {artifacts_dir}"
        + (f" to {prefix}" if prefix else "")
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
    import mlflow

    explicit_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if explicit_tracking_uri:
        mlflow.set_tracking_uri(explicit_tracking_uri)
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

    client = mlflow.tracking.MlflowClient()
    detail(f"MLflow tracking URI: {explicit_tracking_uri}")
    _materialize_remote_artifacts_if_needed(plan, artifacts_dir)
    _materialize_remote_metrics_if_needed(plan, artifacts_dir)
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
        _generate_post_run_report(artifacts_dir)
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
    success(f"MLflow upload complete for run {mlflow_run_id}")
    if artifacts_dir.exists():
        _cleanup_dir_contents(artifacts_dir, preserve_names={"platform-state"})
    return artifacts_dir
