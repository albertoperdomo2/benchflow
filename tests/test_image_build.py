from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner
import pytest
import yaml

from benchflow.cli import cli
from benchflow.cluster import CommandError
from benchflow.models import ValidationError
from benchflow.toolbox.image import (
    ASSET_DIRECTORY,
    build_lmcache_image,
    load_image_build_config,
)
from benchflow.assets import asset_text


IMAGE_ID = "sha256:" + "a" * 64
REGISTRY_DIGEST = "sha256:" + "b" * 64
CONFIG = {
    "base_image": "vllm/vllm-openai:v0.32.0",
    "lmcache_version": "0.5.7",
    "destination": "quay.io/example/vllm-lmcache:test",
}


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "serving.yaml"
    path.write_text(yaml.safe_dump(CONFIG))
    return path


def test_config_accepts_digest_and_normalizes_upstream_name(config_path: Path) -> None:
    config = load_image_build_config(config_path)
    assert config["base_image"] == "docker.io/" + CONFIG["base_image"]
    config["base_image"] = "quay.io/example/base@" + IMAGE_ID
    config_path.write_text(yaml.safe_dump(config))
    assert load_image_build_config(config_path) == config


@pytest.mark.parametrize(
    "changes",
    [
        {"destination": "quay.io/example/runtime:latest"},
        {"destination": "ghcr.io/example/runtime:v1"},
        {"destination": "quay.io/example/runtime"},
        {"base_image": "vllm/vllm-openai:latest"},
        {"base_image": "vllm/vllm-openai"},
        {"base_image": "registry.example/base:v1\nRUN evil"},
        {"lmcache_version": "latest"},
        {"lmcache_version": "0.5.7; echo evil"},
        {"lmcache_version": 0.5},
        {"password": "secret"},
    ],
)
def test_invalid_config_is_rejected_before_podman(config_path: Path, changes) -> None:
    config_path.write_text(yaml.safe_dump({**CONFIG, **changes}))
    with patch("benchflow.toolbox.image.run_command") as run:
        with pytest.raises(ValidationError):
            build_lmcache_image(config_path)
    run.assert_not_called()


def test_dry_run_needs_no_podman_or_credentials(config_path: Path) -> None:
    with (
        patch("benchflow.toolbox.image.require_command") as require,
        patch("benchflow.toolbox.image.run_command") as run,
    ):
        result = CliRunner().invoke(
            cli, ["image", "build", str(config_path), "--dry-run"]
        )
    assert result.exit_code == 0, result.output
    assert "--no-build-isolation" in asset_text(f"{ASSET_DIRECTORY}/install.py")
    assert "FROM ${BASE_IMAGE}" in result.output
    assert "podman push" in result.output
    assert "BASE_IMAGE=docker.io/vllm/vllm-openai:v0.32.0" in result.output
    assert "<built-image-id>" in result.output
    require.assert_not_called()
    run.assert_not_called()
    assert not config_path.with_suffix(".image.json").exists()


class Podman:
    def __init__(self, *, failure: str = "", arch: str = "amd64") -> None:
        self.calls: list[tuple[list[str], dict]] = []
        self.failure = failure
        self.arch = arch
        self.context: Path | None = None

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        action = command[1]
        if action == "build":
            self.context = Path(command[-1])
        if action == self.failure:
            raise CommandError(f"simulated {action} failure")
        stdout = ""
        if action == "info":
            # Current Podman reports server OS under version, not host.
            stdout = json.dumps(
                {"host": {"arch": self.arch}, "version": {"Os": "linux"}}
            )
        elif action == "login":
            stdout = "robot-user"
        elif action == "build":
            self.context = Path(command[-1])
            assert {path.name for path in self.context.iterdir()} == {
                "Containerfile",
                "install.py",
                "check.py",
            }
            (self.context / "image.id").write_text(IMAGE_ID)
        elif action == "run":
            assert command[command.index("--platform") + 1] == "linux/amd64"
            if "-c" in command:
                if self.failure == "probe":
                    raise CommandError("exec format error")
                assert (
                    command[command.index("python3") + 1]
                    == "docker.io/" + CONFIG["base_image"]
                )
                return subprocess.CompletedProcess(
                    command, 0, "amd64 base image ready", ""
                )
            assert command[-2] == IMAGE_ID
            stdout = "library log before provenance\n" + json.dumps(
                {"lmcache": "0.5.7"}
            )
        elif action == "push":
            assert command[-2] == IMAGE_ID
            assert command[-1] == "docker://" + CONFIG["destination"]
            Path(command[command.index("--digestfile") + 1]).write_text(REGISTRY_DIGEST)
        return subprocess.CompletedProcess(command, 0, stdout, "")


def test_build_checks_then_pushes_exact_id_and_records_registry_digest(
    config_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("QUAY_USERNAME", raising=False)
    monkeypatch.delenv("QUAY_PASSWORD", raising=False)
    podman = Podman()
    with (
        patch("benchflow.toolbox.image.require_command"),
        patch("benchflow.toolbox.image.run_command", side_effect=podman),
    ):
        result = build_lmcache_image(config_path)
    assert [command[1] for command, _ in podman.calls] == [
        "info",
        "login",
        "build",
        "run",
        "push",
    ]
    assert result["image"] == "quay.io/example/vllm-lmcache@" + REGISTRY_DIGEST
    assert json.loads(config_path.with_suffix(".image.json").read_text()) == result
    assert podman.context is not None and not podman.context.exists()


@pytest.mark.parametrize("failure", ["build", "run", "push"])
def test_failures_stop_publication_and_preserve_previous_record(
    config_path: Path, monkeypatch, failure: str
) -> None:
    monkeypatch.delenv("QUAY_USERNAME", raising=False)
    monkeypatch.delenv("QUAY_PASSWORD", raising=False)
    record = config_path.with_suffix(".image.json")
    record.write_text("previous result")
    podman = Podman(failure=failure)
    with (
        patch("benchflow.toolbox.image.require_command"),
        patch("benchflow.toolbox.image.run_command", side_effect=podman),
        pytest.raises(CommandError, match=failure),
    ):
        build_lmcache_image(config_path)
    assert record.read_text() == "previous result"
    if failure != "push":
        assert "push" not in [command[1] for command, _ in podman.calls]
    assert podman.context is not None and not podman.context.exists()


def test_arm_builder_checks_emulation_and_publishes_amd64(
    config_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("QUAY_USERNAME", raising=False)
    monkeypatch.delenv("QUAY_PASSWORD", raising=False)
    podman = Podman(arch="arm64")
    with (
        patch("benchflow.toolbox.image.require_command"),
        patch("benchflow.toolbox.image.run_command", side_effect=podman),
    ):
        result = build_lmcache_image(config_path)
    assert [command[1] for command, _ in podman.calls] == [
        "info",
        "login",
        "run",
        "build",
        "run",
        "push",
    ]
    assert result["platform"] == "linux/amd64"


def test_failed_emulation_probe_stops_before_build_or_push(
    config_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("QUAY_USERNAME", raising=False)
    monkeypatch.delenv("QUAY_PASSWORD", raising=False)
    podman = Podman(arch="arm64", failure="probe")
    with (
        patch("benchflow.toolbox.image.require_command"),
        patch("benchflow.toolbox.image.run_command", side_effect=podman),
        pytest.raises(CommandError, match="Rosetta/QEMU.*exec format error"),
    ):
        build_lmcache_image(config_path)
    assert [command[1] for command, _ in podman.calls] == ["info", "login", "run"]
    assert not config_path.with_suffix(".image.json").exists()


def test_robot_password_uses_stdin_and_is_not_written(
    config_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("QUAY_USERNAME", "organization+builder")
    monkeypatch.setenv("QUAY_PASSWORD", "private-password")
    podman = Podman()
    with (
        patch("benchflow.toolbox.image.require_command"),
        patch("benchflow.toolbox.image.run_command", side_effect=podman),
    ):
        build_lmcache_image(config_path)
    login, options = podman.calls[1]
    assert "--password-stdin" in login
    assert options["input_text"] == "private-password"
    assert all("private-password" not in str(command) for command, _ in podman.calls)
    assert "private-password" not in config_path.with_suffix(".image.json").read_text()


def _installer():
    path = (
        Path(__file__).parents[1]
        / "src/benchflow/assets"
        / ASSET_DIRECTORY
        / "install.py"
    )
    spec = importlib.util.spec_from_file_location("lmcache_image_install", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_duplicate_metadata_preserves_python_search_precedence() -> None:
    installer = _installer()
    distributions = [
        SimpleNamespace(metadata={"Name": "PyJWT"}, version="2.10.1"),
        SimpleNamespace(metadata={"Name": "pyjwt"}, version="2.7.0"),
    ]
    with patch.object(installer.metadata, "distributions", return_value=distributions):
        assert installer.installed_versions() == {"pyjwt": "2.10.1"}


def test_dependency_checks_separate_inherited_conflicts_and_check_lmcache() -> None:
    installer = _installer()
    requirements = {
        "old-exporter": ["sdk~=1.45", "extra-only; extra == 'optional'"],
        "sdk": [],
        "lmcache": ["torch>=2.13"],
        "torch": [],
    }
    before = {
        "old-exporter": "0.66",
        "sdk": "1.40",
        "lmcache": "0.5.6",
        "torch": "2.12",
    }
    with patch.object(installer.metadata, "requires", side_effect=requirements.get):
        baseline = installer.dependency_conflicts(before, exclude_lmcache=True)
        assert baseline == {("old-exporter", "sdk~=1.45")}
        after = installer.dependency_conflicts(before)
        assert after - baseline == {("lmcache", "torch>=2.13")}


def test_core_stack_constraints_include_nvidia_override() -> None:
    installer = _installer()
    assert installer.protected_package("vllm")
    assert installer.protected_package("torch")
    assert installer.protected_package("nvidia-nccl-cu13")
    assert installer.protected_package("triton")
    assert not installer.protected_package("lmcache")
    assert not installer.protected_package("grpcio-tools")


def test_source_build_uses_existing_torch_abi_and_nvidia_constraints(
    tmp_path: Path, monkeypatch
) -> None:
    installer = _installer()
    before = {
        "vllm": "0.32.0",
        "torch": "2.13.0",
        "nvidia-nccl-cu13": "2.30.7",
        "lmcache": "0.5.6",
    }
    after = {**before, "lmcache": "0.5.7"}
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            version=SimpleNamespace(cuda="13.0"),
            _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=False),
        ),
    )
    monkeypatch.setenv("UV_OVERRIDE", "/etc/uv-overrides.txt")
    monkeypatch.setattr(installer, "PROVENANCE_PATH", tmp_path / "image.json")
    commands = []
    real_rmtree = installer.shutil.rmtree

    def cleanup(path, *args, **kwargs):
        if str(path) != "/var/lib/apt/lists":
            return real_rmtree(path, *args, **kwargs)

    def run(command, **kwargs):
        commands.append(command)
        if command[:3] == ["uv", "pip", "install"]:
            constraints = Path(command[command.index("--constraint") + 1]).read_text()
            assert "torch==2.13.0" in constraints
            assert "nvidia-nccl-cu13==2.30.7" in constraints
            assert "lmcache==" not in constraints
            assert kwargs["env"]["ENABLE_CXX11_ABI"] == "0"
            assert kwargs["env"]["UV_OVERRIDE"] == "/etc/uv-overrides.txt"
            assert kwargs["env"]["CUDA_HOME"] == "/usr/local/cuda-13.0"
        return subprocess.CompletedProcess(command, 0)

    with (
        patch.object(installer, "installed_versions", side_effect=[before, after]),
        patch.object(installer, "dependency_conflicts", return_value=set()),
        patch.object(installer.shutil, "which", return_value="/usr/bin/tool"),
        patch.object(installer.shutil, "rmtree", side_effect=cleanup),
        patch.object(installer.subprocess, "run", side_effect=run),
        patch.object(
            installer.subprocess, "check_output", return_value="source-commit\n"
        ),
    ):
        installer.install("0.5.7")
    assert any("--no-build-isolation" in command for command in commands)
    apt = next(command for command in commands if command[:2] == ["apt-get", "install"])
    assert "libcusparse-dev-13-0" in apt and "libcusolver-dev-13-0" in apt
    provenance = json.loads((tmp_path / "image.json").read_text())
    assert provenance["lmcache_commit"] == "source-commit"
    assert provenance["cxx11_abi"] == "0"
