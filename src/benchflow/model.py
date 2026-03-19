from __future__ import annotations

import os
import shutil
from pathlib import Path

from .cluster import CommandError
from .models import ResolvedRunPlan
from .ui import detail, step, success, warning


def _configure_huggingface_runtime() -> Path:
    cache_root = Path("/tmp/benchflow-hf")
    home_dir = cache_root / "home"
    hf_home = cache_root / "huggingface"
    xdg_cache_home = cache_root / "xdg-cache"

    for path in (home_dir, hf_home, xdg_cache_home):
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HOME", str(home_dir))
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("HF_XET_CACHE", str(hf_home / "xet"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))

    return hf_home / "hub"


def _has_model_weights(target_dir: Path) -> bool:
    if not target_dir.is_dir():
        return False
    for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf"):
        if any(target_dir.glob(pattern)):
            return True
    return False


def download_model(
    plan: ResolvedRunPlan,
    *,
    models_storage_path: Path,
    skip_if_exists: bool = True,
) -> Path:
    target_dir = (
        models_storage_path
        / plan.deployment.model_storage.cache_dir.lstrip("/")
        / plan.model.pvc_directory_name
    )
    step(f"Preparing model cache for {plan.model.name}")
    detail(f"Target directory: {target_dir}")
    if skip_if_exists and _has_model_weights(target_dir):
        success(
            f"Skipping download; cached model weights already exist at {target_dir}"
        )
        return target_dir
    if target_dir.exists():
        warning(f"Removing incomplete cached model directory at {target_dir}")
        shutil.rmtree(target_dir)

    target_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download

        cache_dir = _configure_huggingface_runtime()
        detail(f"Hugging Face cache directory: {cache_dir}")
        step(f"Downloading {plan.model.name}")

        snapshot_download(
            repo_id=plan.model.name,
            local_dir=str(target_dir),
            cache_dir=str(cache_dir),
            token=os.environ.get("HF_TOKEN"),
            local_dir_use_symlinks=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise CommandError(
            f"failed to download model {plan.model.name}: {exc}"
        ) from exc
    if not _has_model_weights(target_dir):
        raise CommandError(
            f"download completed but no model weights were found in {target_dir}"
        )
    success(f"Downloaded model weights to {target_dir}")

    return target_dir
