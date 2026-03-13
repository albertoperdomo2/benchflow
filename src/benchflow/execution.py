from __future__ import annotations

import json
from pathlib import Path

from .loaders import load_run_plan_data, load_run_plan_file
from .models import ResolvedRunPlan, ValidationError


def load_run_plan_from_sources(
    *,
    run_plan_file: str | None = None,
    run_plan_json: str | None = None,
) -> ResolvedRunPlan:
    if run_plan_file:
        return load_run_plan_file(Path(run_plan_file).resolve())
    if run_plan_json:
        try:
            raw = json.loads(run_plan_json)
        except json.JSONDecodeError as exc:
            raise ValidationError("invalid JSON passed to --run-plan-json") from exc
        return load_run_plan_data(raw)
    raise ValidationError("provide --run-plan-file or --run-plan-json")


def require_platform(plan: ResolvedRunPlan, platform: str) -> None:
    if plan.deployment.platform != platform:
        raise ValidationError(
            f"unsupported deployment platform: {plan.deployment.platform}; only {platform} is implemented"
        )
