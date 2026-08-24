from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow import artifacts
from benchflow.loaders import load_deployment_profile
from benchflow.models import RuntimeArtifactDirectorySpec, ValidationError


def _write_deployment_profile(tmp_path: Path, artifact_yaml: str) -> Path:
    profile_path = tmp_path / "deployment.yaml"
    profile_path.write_text(
        "\n".join(
            [
                "apiVersion: benchflow.io/v1alpha1",
                "kind: DeploymentProfile",
                "metadata:",
                "  name: runtime-artifacts-test",
                "spec:",
                "  platform: rhoai",
                "  mode: distributed-default",
                "  runtime:",
                "    artifact_directories:",
                artifact_yaml,
            ]
        ),
        encoding="utf-8",
    )
    return profile_path


def test_runtime_artifact_directories_load_from_profile(tmp_path: Path) -> None:
    profile = load_deployment_profile(
        _write_deployment_profile(
            tmp_path,
            "      - name: oracle-trace\n        path: /tmp//vllm-kv-oracle-traces/",
        )
    )

    assert profile.spec.runtime.artifact_directories == [
        RuntimeArtifactDirectorySpec(
            name="oracle-trace", path="/tmp/vllm-kv-oracle-traces"
        )
    ]


@pytest.mark.parametrize(
    "artifact_yaml, message",
    [
        (
            "      - name: requests\n        path: relative/path",
            "path must be an absolute path",
        ),
        (
            "      - name: secrets\n        path: /var/run/secrets/tokens",
            "must not target a root or runtime secret directory",
        ),
        (
            "      - name: duplicate\n"
            "        path: /tmp/one\n"
            "      - name: duplicate\n"
            "        path: /tmp/two",
            "duplicates artifact directory",
        ),
    ],
)
def test_runtime_artifact_directory_validation(
    tmp_path: Path, artifact_yaml: str, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        load_deployment_profile(_write_deployment_profile(tmp_path, artifact_yaml))


def test_collect_runtime_artifact_directory(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run_command(args, **_kwargs):
        calls.append(args)
        if args[1] == "exec":
            return SimpleNamespace(returncode=0, stdout="oracle-trace-engine.jsonl\n")
        target_dir = Path(args[-1])
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "oracle-trace-engine.jsonl").write_text("{}\n")
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(artifacts, "run_command", fake_run_command)

    count = artifacts._collect_runtime_artifact_directory(
        "oc",
        "benchflow",
        "model-pod",
        "main",
        tmp_path,
        RuntimeArtifactDirectorySpec(
            name="oracle-trace", path="/tmp/vllm-kv-oracle-traces"
        ),
    )

    assert count == 1
    assert calls[0][1] == "exec"
    assert "find /tmp/vllm-kv-oracle-traces -type f -print -quit" in calls[0][-1]
    assert calls[1][1] == "cp"
    assert calls[1][-2] == "model-pod:/tmp/vllm-kv-oracle-traces/."
    assert (
        tmp_path
        / "runtime-artifacts"
        / "oracle-trace"
        / "model-pod"
        / "oracle-trace-engine.jsonl"
    ).is_file()


def test_collect_runtime_artifact_directory_is_optional(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        artifacts,
        "run_command",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=""),
    )

    assert (
        artifacts._collect_runtime_artifact_directory(
            "oc",
            "benchflow",
            "model-pod",
            "main",
            tmp_path,
            RuntimeArtifactDirectorySpec(name="requests", path="/tmp/requests"),
        )
        == 0
    )


@pytest.mark.parametrize(
    "containers, expected",
    [
        (["queue-proxy", "main"], "main"),
        (["metrics", "vllm"], "vllm"),
        (["model-server"], "model-server"),
        (["one", "two"], ""),
    ],
)
def test_model_runtime_container_detection(
    containers: list[str], expected: str
) -> None:
    pod = {"spec": {"containers": [{"name": name} for name in containers]}}

    assert artifacts._model_runtime_container_name(pod) == expected
