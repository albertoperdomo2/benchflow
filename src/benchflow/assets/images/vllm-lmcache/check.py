"""CPU-only checks before publication; GPU readback is a separate cluster gate."""

from importlib import metadata
import json
from pathlib import Path
import subprocess


def check() -> dict:
    import torch
    import vllm
    import lmcache.cuda_ops  # noqa: F401
    from cuda.bindings import driver  # noqa: F401
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
    from packaging.version import Version

    if Version(vllm.__version__) < Version("0.20.0"):
        raise RuntimeError(
            "BenchFlow requires vLLM >= 0.20.0 for the external MP connector"
        )
    if LMCacheMPConnector is None or not torch.version.cuda:
        raise RuntimeError("missing CUDA runtime or external LMCache connector")
    result = subprocess.run(
        ["lmcache", "server", "--help"], check=True, capture_output=True, text=True
    )
    required = {
        "--host",
        "--port",
        "--http-host",
        "--http-port",
        "--l1-size-gb",
        "--chunk-size",
        "--eviction-policy",
        "--isolated-ipc",
    }
    missing = sorted(flag for flag in required if flag not in result.stdout)
    if missing:
        raise RuntimeError(f"LMCache server lacks required flags: {', '.join(missing)}")
    provenance = json.loads(
        Path("/usr/local/share/benchflow/lmcache-image-build.json").read_text()
    )
    return {
        "vllm": metadata.version("vllm"),
        "lmcache": metadata.version("lmcache"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "lmcache_commit": provenance["lmcache_commit"],
        "cuda_arch_list": provenance["cuda_arch_list"],
        "cxx11_abi": provenance["cxx11_abi"],
        "base_dependency_conflicts": provenance["base_dependency_conflicts"],
    }


if __name__ == "__main__":
    print(json.dumps(check()))
