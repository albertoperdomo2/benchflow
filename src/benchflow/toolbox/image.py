"""Explicit local serving-image preparation; never called by workflow tasks."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import tempfile
from typing import Any

import yaml

from ..assets import asset_text
from ..cluster import CommandError, require_command, run_command
from ..models import ValidationError
from ..ui import detail, emit


ASSET_DIRECTORY = "images/vllm-lmcache"
CHECK_SCRIPT = "/usr/local/share/benchflow/lmcache-image-check.py"
_DIGEST = re.compile(r"sha256:[a-f0-9]{64}")
_REPOSITORY = r"[a-z0-9]+(?:[._-][a-z0-9]+)*"
_TAG = r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}"


def _amd64_probe_command(config: dict[str, str]) -> list[str]:
    return [
        "podman",
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "--network=none",
        "--entrypoint",
        "python3",
        config["base_image"],
        "-c",
        "import platform, torch; "
        "assert platform.machine() in ('x86_64', 'amd64'), platform.machine(); "
        "print('amd64 base image ready; PyTorch ' + torch.__version__)",
    ]


def _ensure_amd64_execution(config: dict[str, str]) -> None:
    detail("Checking amd64 base-image execution through Podman's emulator")
    try:
        run_command(_amd64_probe_command(config), capture_output=True)
    except CommandError as exc:
        raise CommandError(
            "Cannot execute the configured linux/amd64 base image. "
            "The Podman server needs working Rosetta/QEMU emulation or native amd64. "
            f"Underlying error: {exc}"
        ) from exc
    detail("amd64 base-image check passed; compilation will run through emulation")


def load_image_build_config(path: Path) -> dict[str, str]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValidationError(f"invalid image build YAML: {exc}") from exc
    fields = {"base_image", "lmcache_version", "destination"}
    if not isinstance(data, dict) or set(data) != fields:
        raise ValidationError(
            "image build config requires exactly base_image, lmcache_version, destination"
        )
    if any(not isinstance(value, str) or not value.strip() for value in data.values()):
        raise ValidationError("image build config values must be nonempty strings")
    data = {key: value.strip() for key, value in data.items()}
    base = data["base_image"]
    if base.startswith("vllm/"):
        base = "docker.io/" + base
    reference = (
        rf"[a-z0-9.-]+(?::[0-9]+)?/(?:{_REPOSITORY}/)*{_REPOSITORY}"
        rf"(?::{_TAG}|@sha256:[a-f0-9]{{64}})"
    )
    if not re.fullmatch(reference, base) or base.rsplit(":", 1)[-1].lower() == "latest":
        raise ValidationError(
            "base_image must be a registry-qualified image with an explicit version "
            "tag or sha256 digest (not latest)"
        )
    data["base_image"] = base
    if not re.fullmatch(
        r"[0-9]+\.[0-9]+\.[0-9]+(?:(?:a|b|rc)[0-9]+)?", data["lmcache_version"]
    ):
        raise ValidationError(
            "lmcache_version must be an exact release version, for example '0.5.7'"
        )
    destination = data["destination"]
    if (
        not re.fullmatch(rf"quay\.io/{_REPOSITORY}/{_REPOSITORY}:{_TAG}", destination)
        or destination.rsplit(":", 1)[-1].lower() == "latest"
    ):
        raise ValidationError(
            "destination must be quay.io/<organization>/<repository>:<explicit-tag> "
            "(not latest)"
        )
    return data


def _commands(config: dict[str, str], context: Path) -> list[list[str]]:
    return [
        [
            "podman",
            "build",
            "--platform",
            "linux/amd64",
            "--pull=always",
            "--file",
            str(context / "Containerfile"),
            "--iidfile",
            str(context / "image.id"),
            "--build-arg",
            f"BASE_IMAGE={config['base_image']}",
            "--build-arg",
            f"LMCACHE_VERSION={config['lmcache_version']}",
            "--tag",
            config["destination"],
            str(context),
        ],
        [
            "podman",
            "run",
            "--rm",
            "--platform",
            "linux/amd64",
            "--network=none",
            "--entrypoint",
            "python3",
            "<built-image-id>",
            CHECK_SCRIPT,
        ],
        [
            "podman",
            "push",
            "--digestfile",
            str(context / "image.digest"),
            "<built-image-id>",
            "docker://" + config["destination"],
        ],
    ]


def _quay_login() -> None:
    username = os.environ.get("QUAY_USERNAME", "")
    password = os.environ.get("QUAY_PASSWORD", "")
    if bool(username) != bool(password):
        raise ValidationError(
            "set both QUAY_USERNAME and QUAY_PASSWORD, or use existing Podman credentials"
        )
    if username:
        run_command(
            ["podman", "login", "--username", username, "--password-stdin", "quay.io"],
            input_text=password,
            capture_output=True,
        )
    else:
        result = run_command(
            ["podman", "login", "--get-login", "quay.io"],
            capture_output=True,
            check=False,
        )
        if result.returncode or not result.stdout.strip():
            raise CommandError(
                "Quay credentials are missing; set QUAY_USERNAME and QUAY_PASSWORD "
                "or run podman login quay.io"
            )


def build_lmcache_image(
    config_path: Path, *, dry_run: bool = False
) -> dict[str, Any] | None:
    config = load_image_build_config(config_path)
    if dry_run:
        emit(asset_text(f"{ASSET_DIRECTORY}/Containerfile"))
        detail("Requires a Linux Podman server, amd64 execution, and Quay credentials.")
        detail("On non-amd64 servers, check base-image execution before building:")
        emit(shlex.join(_amd64_probe_command(config)))
        for command in _commands(config, Path("<temporary-build-context>")):
            emit(shlex.join(command))
        detail(f"Image record: {config_path.resolve().with_suffix('.image.json')}")
        return None

    require_command("podman")
    info = json.loads(
        run_command(["podman", "info", "--format", "json"], capture_output=True).stdout
    )
    host = info.get("host", {})
    server_version = info.get("version", {})
    server_os = (
        host.get("os")
        or server_version.get("Os")
        or str(server_version.get("OsArch", "")).split("/")[0]
    )
    if server_os != "linux":
        raise ValidationError("local LMCache builds require a Linux Podman server")
    _quay_login()
    if host.get("arch") != "amd64":
        _ensure_amd64_execution(config)
    with tempfile.TemporaryDirectory(prefix="benchflow-lmcache-image-") as temporary:
        context = Path(temporary)
        for name in ("Containerfile", "install.py", "check.py"):
            (context / name).write_text(
                asset_text(f"{ASSET_DIRECTORY}/{name}"), encoding="utf-8"
            )
        build, check, push = _commands(config, context)
        detail(f"Building {config['destination']} from {config['base_image']}")
        run_command(build)
        image_id = (context / "image.id").read_text().strip()
        if not _DIGEST.fullmatch(image_id):
            raise CommandError(
                "Podman did not return a valid built image ID; image was not pushed"
            )
        check[check.index("<built-image-id>")] = image_id
        push[push.index("<built-image-id>")] = image_id
        detail(
            "Checking native imports, connector, and server flags without GPU access"
        )
        checked = run_command(check, capture_output=True)
        try:
            software = json.loads(checked.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise CommandError(
                "Image check did not return software provenance; image was not pushed"
            ) from exc
        if (
            not isinstance(software, dict)
            or software.get("lmcache") != config["lmcache_version"]
        ):
            raise CommandError(
                "Image check did not confirm the requested LMCache version; image was not pushed"
            )
        detail(f"Publishing {config['destination']}")
        run_command(push)
        digest = (context / "image.digest").read_text().strip()
        if not _DIGEST.fullmatch(digest):
            raise CommandError(
                "Push completed but Podman did not return a valid registry digest"
            )
    result = {
        **config,
        "image": config["destination"].rsplit(":", 1)[0] + "@" + digest,
        "image_id": image_id,
        "platform": "linux/amd64",
        "software": software,
    }
    config_path.resolve().with_suffix(".image.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
