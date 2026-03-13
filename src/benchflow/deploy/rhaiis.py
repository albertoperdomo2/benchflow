from __future__ import annotations

from ..models import ResolvedRunPlan


def deploy_rhaiis(plan: ResolvedRunPlan) -> None:
    raise NotImplementedError(
        f"RHAIIS deployment is not implemented yet for {plan.metadata.name}"
    )
