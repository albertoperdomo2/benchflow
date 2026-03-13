from __future__ import annotations

from ..models import ResolvedRunPlan


def deploy_rhoai(plan: ResolvedRunPlan) -> None:
    raise NotImplementedError(
        f"RHOAI deployment is not implemented yet for {plan.metadata.name}"
    )
