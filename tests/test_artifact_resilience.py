from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from benchflow import mlflow_upload, remote_jobs
from benchflow.cluster import CommandError


class _ArtifactEntry:
    def __init__(self, path: str, size: int, *, is_dir: bool = False) -> None:
        self.path = path
        self.file_size = size
        self.is_dir = is_dir


def test_transient_kubernetes_webhook_failure_is_retryable() -> None:
    error = CommandError(
        'failed calling webhook "mpod.kb.io": no endpoints available for service '
        '"kueue-webhook-service"'
    )

    assert remote_jobs.is_transient_kubernetes_error(error)


def test_remote_cleanup_path_must_be_a_job_result_descendant() -> None:
    assert (
        remote_jobs._validated_remote_results_path(
            "/benchmark-results/remote-jobs/benchflow-artifacts-123/artifacts"
        )
        == "/benchmark-results/remote-jobs/benchflow-artifacts-123/artifacts"
    )
    for unsafe_path in (
        "/benchmark-results/remote-jobs",
        "/benchmark-results",
        "/tmp/artifacts",
        "relative/artifacts",
    ):
        with pytest.raises(CommandError, match="refusing to delete"):
            remote_jobs._validated_remote_results_path(unsafe_path)


def test_ambiguous_create_accepts_an_existing_named_resource(monkeypatch) -> None:
    create_calls = 0

    def fail_create(*_args, **_kwargs) -> None:
        nonlocal create_calls
        create_calls += 1
        raise CommandError("error from server (Timeout): response deadline exceeded")

    monkeypatch.setattr(remote_jobs, "create_manifest", fail_create)
    monkeypatch.setattr(
        remote_jobs,
        "_resource_payload_or_none",
        lambda **_kwargs: {"metadata": {"name": "durable-job"}},
    )

    remote_jobs._create_manifest_idempotently(
        "kind: Job",
        namespace="benchflow",
        kind="job",
        name="durable-job",
    )

    assert create_calls == 1


def test_ambiguous_mlflow_upload_is_success_when_exact_size_was_committed(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "trace.jsonl"
    artifact.write_bytes(b"trace payload")

    class Client:
        def log_artifact(self, *_args, **_kwargs) -> None:
            raise RuntimeError("too many 504 error responses")

        def list_artifacts(self, _run_id: str, path: str):
            assert path == "runtime-artifacts/oracle-trace"
            return [
                _ArtifactEntry(
                    "runtime-artifacts/oracle-trace/trace.jsonl",
                    artifact.stat().st_size,
                )
            ]

    mlflow_upload._upload_artifact_with_recovery(
        Client(),
        mlflow_run_id="run-123",
        file_path=artifact,
        artifact_dir="runtime-artifacts/oracle-trace",
        remote_path="runtime-artifacts/oracle-trace/trace.jsonl",
    )


def test_directory_upload_continues_after_one_file_fails(
    monkeypatch, tmp_path: Path
) -> None:
    (tmp_path / "a.json").write_text("a", encoding="utf-8")
    (tmp_path / "b.json").write_text("b", encoding="utf-8")
    attempted: list[str] = []

    class Client:
        def list_artifacts(self, _run_id: str, _path: str):
            return []

        def log_artifact(self, _run_id: str, local_file: str, **_kwargs) -> None:
            name = Path(local_file).name
            attempted.append(name)
            if name == "a.json":
                raise RuntimeError("permanent upload failure")

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.test")
    monkeypatch.setattr(mlflow_upload, "configure_mlflow_tracking", lambda _uri: None)
    monkeypatch.setattr(mlflow_upload, "create_mlflow_client", lambda _uri: Client())

    with pytest.raises(mlflow_upload.ArtifactUploadFailed, match="a.json"):
        mlflow_upload.upload_artifact_directory_to_mlflow(
            mlflow_run_id="run-123",
            artifacts_dir=tmp_path,
        )

    assert attempted == ["a.json", "b.json"]
    assert (tmp_path / "a.json").exists()
    assert (tmp_path / "b.json").exists()


def test_remote_source_is_deleted_only_after_references_are_marked_uploaded(
    monkeypatch, tmp_path: Path
) -> None:
    artifact_reference = tmp_path / "remote-target-artifacts.json"
    metrics_reference = tmp_path / "metrics" / "remote-target-metrics.json"
    metrics_reference.parent.mkdir()
    artifact_root = "/benchmark-results/remote-jobs/collection/artifacts"
    artifact_reference.write_text(
        json.dumps({"remote_path": artifact_root, "status": "materialized"}),
        encoding="utf-8",
    )
    metrics_reference.write_text(
        json.dumps(
            {"remote_path": f"{artifact_root}/metrics", "status": "materialized"}
        ),
        encoding="utf-8",
    )
    deleted: list[str] = []
    monkeypatch.setattr(
        mlflow_upload,
        "delete_remote_results_directory",
        lambda _plan, *, remote_path: deleted.append(remote_path),
    )

    mlflow_upload._cleanup_uploaded_remote_materializations(
        SimpleNamespace(), artifacts_dir=tmp_path
    )

    assert deleted == [artifact_root]
    assert json.loads(artifact_reference.read_text())["status"] == "uploaded"
    assert json.loads(metrics_reference.read_text())["uploaded_to_mlflow"] is True


def test_mlflow_finalization_retries_transient_writes_and_attempts_every_field(
    monkeypatch,
) -> None:
    calls: list[tuple[str, str]] = []
    status_attempts = 0

    class Client:
        def set_tag(self, _run_id: str, key: str, value: str) -> None:
            nonlocal status_attempts
            calls.append((key, value))
            if key == "benchflow.final_status":
                status_attempts += 1
                if status_attempts == 1:
                    raise RuntimeError("too many 504 error responses")

        def set_terminated(self, _run_id: str, *, status: str) -> None:
            calls.append(("terminated", status))

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.test")
    monkeypatch.setattr(mlflow_upload, "create_mlflow_client", lambda _uri: Client())
    monkeypatch.setattr(mlflow_upload.time, "sleep", lambda _seconds: None)

    mlflow_upload.mark_mlflow_run(
        mlflow_run_id="run-123",
        status="failed",
        reason="upload did not complete",
    )

    assert status_attempts == 2
    assert calls[-1] == ("terminated", "FAILED")


def test_artifact_pipeline_tasks_have_bounded_retries() -> None:
    pipeline_path = Path(__file__).resolve().parents[1] / "tekton/pipelines/e2e.yaml"
    pipeline = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    tasks = {task["name"]: task for task in pipeline["spec"]["tasks"]}
    finally_tasks = {task["name"]: task for task in pipeline["spec"]["finally"]}

    for task_name in ("collect-artifacts", "collect-metrics", "upload-to-mlflow"):
        assert tasks[task_name]["retries"] == 2
    assert finally_tasks["finalize-mlflow-run"]["retries"] == 2
