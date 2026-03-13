from __future__ import annotations

from ..models import ResolvedRunPlan


def cleanup_rhoai(plan: ResolvedRunPlan) -> None:
    raise NotImplementedError(
        f"RHOAI cleanup is not implemented yet for {plan.metadata.name}"
    )
