from __future__ import annotations

import os
from pathlib import Path

from .cluster import CommandError
from .models import ResolvedRunPlan


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
    if skip_if_exists and target_dir.exists():
        return target_dir

    target_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download

        cache_dir = _configure_huggingface_runtime()

        snapshot_download(
            repo_id=plan.model.name,
            revision=plan.model.revision,
            local_dir=str(target_dir),
            cache_dir=str(cache_dir),
            token=os.environ.get("HF_TOKEN"),
            local_dir_use_symlinks=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise CommandError(
            f"failed to download model {plan.model.name}: {exc}"
        ) from exc

    return target_dir
