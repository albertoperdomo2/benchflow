from __future__ import annotations

from ..models import ResolvedRunPlan


def cleanup_rhaiis(plan: ResolvedRunPlan) -> None:
    raise NotImplementedError(
        f"RHAIIS cleanup is not implemented yet for {plan.metadata.name}"
    )
