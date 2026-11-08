"""Build inside an Ubuntu/NVIDIA vLLM image, using its existing Python stack."""

from __future__ import annotations

from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


PROVENANCE_PATH = Path("/usr/local/share/benchflow/lmcache-image-build.json")


def installed_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in metadata.distributions():
        name = re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower()
        # First match wins, matching Python's search-path precedence.
        versions.setdefault(name, distribution.version)
    return versions


def dependency_conflicts(
    versions: dict[str, str], *, exclude_lmcache: bool = False
) -> set[tuple[str, str]]:
    from packaging.requirements import Requirement

    conflicts = set()
    for name in versions:
        if exclude_lmcache and name == "lmcache":
            continue  # The base's copy is replaced by the requested release.
        for value in metadata.requires(name) or []:
            requirement = Requirement(value)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            dependency = re.sub(r"[-_.]+", "-", requirement.name).lower()
            found = versions.get(dependency)
            if found is None or not requirement.specifier.contains(
                found, prereleases=True
            ):
                conflicts.add((name, str(requirement)))
    return conflicts


def protected_package(name: str) -> bool:
    return name in {
        "vllm",
        "torch",
        "torchvision",
        "torchaudio",
        "triton",
    } or name.startswith("nvidia-")


def install(version: str) -> None:
    import torch
    from packaging.version import Version

    before = installed_versions()
    if "vllm" not in before or not torch.version.cuda or not shutil.which("uv"):
        raise RuntimeError("base image must contain vLLM, CUDA PyTorch, and uv")
    if Version(before["vllm"]) < Version("0.20.0"):
        raise RuntimeError("external MP connector requires vLLM >= 0.20.0")
    if "lmcache-cli" in before:
        raise RuntimeError(
            "base image contains lmcache-cli, which conflicts with the lmcache CLI"
        )
    baseline_conflicts = dependency_conflicts(before, exclude_lmcache=True)
    cuda_series = tuple(int(value) for value in torch.version.cuda.split(".")[:2])
    if cuda_series < (12, 1) or cuda_series[0] not in (12, 13):
        raise RuntimeError(f"unsupported base CUDA runtime: {torch.version.cuda}")
    if not shutil.which("apt-get"):
        raise RuntimeError(
            "source builds require an Ubuntu vLLM base with NVIDIA CUDA apt repositories"
        )
    cuda_package = "-".join(str(value) for value in cuda_series)
    environment = {
        **os.environ,
        "DEBIAN_FRONTEND": "noninteractive",
        "BUILD_WITH_CUDA": "1",
        "LMCACHE_CUDA_MAJOR": str(cuda_series[0]),
        "CUDA_HOME": f"/usr/local/cuda-{'.'.join(str(value) for value in cuda_series)}",
        "MAX_JOBS": "2",
        "NVCC_THREADS": "1",
        "ENABLE_CXX11_ABI": str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
        # A GPU is not needed for compilation. Include Ampere, Ada, Hopper,
        # and, on toolchains that support it, Blackwell.
        "TORCH_CUDA_ARCH_LIST": (
            "8.0;8.6;8.9;9.0;10.0;12.0+PTX"
            if cuda_series >= (12, 8)
            else "8.0;8.6;8.9;9.0+PTX"
        ),
    }
    subprocess.run(["apt-get", "update"], check=True, env=environment)
    subprocess.run(
        [
            "apt-get",
            "install",
            "-y",
            "--no-install-recommends",
            "git",
            "ca-certificates",
            "build-essential",
            f"cuda-nvcc-{cuda_package}",
            f"cuda-cudart-dev-{cuda_package}",
            f"libcublas-dev-{cuda_package}",
            f"libcusparse-dev-{cuda_package}",
            f"libcusolver-dev-{cuda_package}",
        ],
        check=True,
        env=environment,
    )
    environment["PATH"] = environment["CUDA_HOME"] + "/bin:" + environment["PATH"]
    with tempfile.TemporaryDirectory(prefix="lmcache-source-") as temporary:
        root = Path(temporary)
        source = root / "source"
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                "v" + version,
                "https://github.com/LMCache/LMCache.git",
                str(source),
            ],
            check=True,
        )
        constraints = root / "base-constraints.txt"
        constraints.write_text(
            "".join(
                f"{name}=={value}\n"
                for name, value in sorted(before.items())
                if protected_package(name)
            )
        )
        uv = [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--constraint",
            str(constraints),
        ]
        # Respect UV_OVERRIDE from the base, including intentional NCCL pins.
        subprocess.run(
            [*uv, "-r", str(source / "requirements/build.txt")],
            check=True,
            env=environment,
        )
        subprocess.run(
            [
                *uv,
                "--no-build-isolation",
                "--reinstall-package",
                "lmcache",
                str(source),
            ],
            check=True,
            env=environment,
        )
        after = installed_versions()
        changed = [
            name
            for name, value in before.items()
            if protected_package(name) and after.get(name) != value
        ]
        if changed:
            raise RuntimeError(
                f"installation changed protected base packages: {', '.join(changed)}"
            )
        if after.get("lmcache") != version:
            raise RuntimeError(
                f"expected LMCache {version}, installed {after.get('lmcache')}"
            )
        new_conflicts = dependency_conflicts(after) - baseline_conflicts
        if new_conflicts:
            raise RuntimeError(
                f"installation introduced dependency conflicts: {sorted(new_conflicts)}"
            )
        provenance = {
            "lmcache_commit": subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
            ).strip(),
            "requested_lmcache_version": version,
            "cuda_arch_list": environment["TORCH_CUDA_ARCH_LIST"],
            "cxx11_abi": environment["ENABLE_CXX11_ABI"],
            "base_dependency_conflicts": sorted(baseline_conflicts),
            "base_packages": {
                name: value for name, value in before.items() if protected_package(name)
            },
        }
        PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2) + "\n")
    subprocess.run(["uv", "cache", "clean"], check=True)
    shutil.rmtree("/var/lib/apt/lists", ignore_errors=True)


if __name__ == "__main__":
    install(sys.argv[1])
