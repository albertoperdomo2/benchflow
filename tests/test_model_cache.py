from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from benchflow import model
from benchflow.models import ModelSpec, ModelStorageSpec


def _plan() -> SimpleNamespace:
    return SimpleNamespace(
        model=ModelSpec(name="org/model"),
        deployment=SimpleNamespace(model_storage=ModelStorageSpec()),
    )


def _write_complete_model(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")


def test_download_publishes_complete_model_atomically(
    monkeypatch, tmp_path: Path
) -> None:
    downloaded_to: list[Path] = []

    def snapshot_download(**kwargs: object) -> None:
        destination = Path(str(kwargs["local_dir"]))
        downloaded_to.append(destination)
        _write_complete_model(destination)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )

    target = model.download_model(_plan(), models_storage_path=tmp_path)

    assert target == tmp_path / "models/org-model"
    assert model._is_complete_model_cache(target)
    assert downloaded_to[0] != target
    assert ".benchflow-partial-" in downloaded_to[0].name
    assert not downloaded_to[0].exists()


def test_download_rechecks_complete_cache_after_lock(
    monkeypatch, tmp_path: Path
) -> None:
    target = tmp_path / "models/org-model"
    _write_complete_model(target)

    def unexpected_download(**_kwargs: object) -> None:
        raise AssertionError("complete model cache should not be downloaded again")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=unexpected_download),
    )

    assert model.download_model(_plan(), models_storage_path=tmp_path) == target
