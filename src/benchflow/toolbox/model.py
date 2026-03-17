from __future__ import annotations

from pathlib import Path

from ..contracts import ExecutionContext, ResolvedRunPlan, ValidationError
from ..model import download_model


def download_cached_model(
    plan: ResolvedRunPlan,
    *,
    context: ExecutionContext,
    skip_if_exists: bool = True,
) -> Path:
    if context.models_storage_path is None:
        raise ValidationError("model download requires a models storage path")
    return download_model(
        plan,
        models_storage_path=context.models_storage_path,
        skip_if_exists=skip_if_exists,
    )
