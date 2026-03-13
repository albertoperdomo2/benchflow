from __future__ import annotations

import os
from pathlib import Path

from .cluster import CommandError
from .models import ResolvedRunPlan


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

        snapshot_download(
            repo_id=plan.model.name,
            revision=plan.model.revision,
            local_dir=str(target_dir),
            token=os.environ.get("HF_TOKEN"),
            local_dir_use_symlinks=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise CommandError(
            f"failed to download model {plan.model.name}: {exc}"
        ) from exc

    return target_dir
