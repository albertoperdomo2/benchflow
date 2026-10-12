from __future__ import annotations

import os
import secrets
import shutil
import time
from contextlib import contextmanager
from fcntl import LOCK_EX, LOCK_NB, LOCK_UN, flock
from pathlib import Path
from typing import Iterator

from .cluster import CommandError
from .models import ResolvedRunPlan, model_storage_relative_path
from .ui import detail, step, success, warning


_MODEL_CACHE_LOCK_POLL_SECONDS = 5


def _cache_access_error(path: Path, exc: OSError) -> CommandError:
    return CommandError(
        "model cache is not writable: "
        f"path={path}, uid={os.geteuid()}, gid={os.getegid()}: {exc}. "
        "Provision the backing volume for the workload UID/GID and SELinux "
        "context before retrying"
    )


@contextmanager
def _model_cache_lock(target_dir: Path) -> Iterator[None]:
    """Serialize population of one model cache on the backing filesystem."""
    parent = target_dir.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        lock_file = (parent / f".{target_dir.name}.benchflow.lock").open("a+")
    except OSError as exc:
        raise _cache_access_error(parent, exc) from exc

    waiting_reported = False
    try:
        while True:
            try:
                flock(lock_file.fileno(), LOCK_EX | LOCK_NB)
                break
            except BlockingIOError:
                if not waiting_reported:
                    detail(f"Waiting for another download to populate {target_dir}")
                    waiting_reported = True
                time.sleep(_MODEL_CACHE_LOCK_POLL_SECONDS)
        yield
    finally:
        flock(lock_file.fileno(), LOCK_UN)
        lock_file.close()


def _remove_directory(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except OSError as exc:
        raise _cache_access_error(path, exc) from exc


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
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
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


def _has_tokenizer_files(target_dir: Path) -> bool:
    if not target_dir.is_dir():
        return False
    if not (target_dir / "tokenizer_config.json").is_file():
        return False
    if (target_dir / "tokenizer.json").is_file():
        return True
    if (target_dir / "tokenizer.model").is_file():
        return True
    return (target_dir / "vocab.json").is_file() and (
        target_dir / "merges.txt"
    ).is_file()


def _is_complete_model_cache(target_dir: Path) -> bool:
    if not target_dir.is_dir():
        return False
    if not (target_dir / "config.json").is_file():
        return False
    if not _has_model_weights(target_dir):
        return False
    if not _has_tokenizer_files(target_dir):
        return False
    return True


def download_model(
    plan: ResolvedRunPlan,
    *,
    models_storage_path: Path,
    skip_if_exists: bool = True,
) -> Path:
    target_dir = models_storage_path / model_storage_relative_path(
        plan.deployment.model_storage, plan.model
    )
    step(f"Preparing model cache for {plan.model.name}")
    detail(f"Target directory: {target_dir}")
    with _model_cache_lock(target_dir):
        # A concurrent matrix child may have completed the cache while this
        # process waited for the filesystem lock, so check only after locking.
        if skip_if_exists and _is_complete_model_cache(target_dir):
            success(
                "Skipping download; complete cached model already exists at "
                f"{target_dir}"
            )
            return target_dir

        staging_dir = target_dir.parent / (
            f".{target_dir.name}.benchflow-partial-{secrets.token_hex(4)}"
        )
        try:
            cache_dir = _configure_huggingface_runtime()
            from huggingface_hub import snapshot_download

            detail(f"Hugging Face cache directory: {cache_dir}")
            detail(
                "Hugging Face timeouts: "
                f"download={os.environ['HF_HUB_DOWNLOAD_TIMEOUT']}s, "
                f"etag={os.environ['HF_HUB_ETAG_TIMEOUT']}s"
            )
            step(f"Downloading {plan.model.name}")

            snapshot_download(
                repo_id=plan.model.name,
                local_dir=str(staging_dir),
                cache_dir=str(cache_dir),
                etag_timeout=float(os.environ["HF_HUB_ETAG_TIMEOUT"]),
                token=os.environ.get("HF_TOKEN"),
            )
            if not _has_model_weights(staging_dir):
                raise CommandError(
                    "download completed but no model weights were found in "
                    f"{staging_dir}"
                )
            if not _has_tokenizer_files(staging_dir):
                raise CommandError(
                    "download completed but tokenizer assets were not found in "
                    f"{staging_dir}"
                )

            if target_dir.exists():
                warning(f"Replacing incomplete cached model at {target_dir}")
                _remove_directory(target_dir)
            try:
                staging_dir.replace(target_dir)
            except OSError as exc:
                raise _cache_access_error(target_dir, exc) from exc
        except CommandError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise CommandError(
                f"failed to download model {plan.model.name}: {exc}"
            ) from exc
        finally:
            if staging_dir.exists():
                _remove_directory(staging_dir)

        success(f"Downloaded model weights to {target_dir}")

    return target_dir
