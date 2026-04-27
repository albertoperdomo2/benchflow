from __future__ import annotations

from ..accelerator import discover_plan_accelerator
from ..cluster import CommandError
from ..models import ResolvedRunPlan
from ..setup.rhoai import (
    discover_rhoai_mlflow_version,
    normalize_rhoai_platform_version,
)
from ..ui import warning


class BenchmarkRunFailed(CommandError):
    def __init__(
        self,
        message: str,
        *,
        run_id: str = "",
        start_time: str = "",
        end_time: str = "",
    ) -> None:
        super().__init__(message)
        self.run_id = run_id
        self.start_time = start_time
        self.end_time = end_time


def benchmark_version_from_plan(plan: ResolvedRunPlan) -> str:
    explicit_version = str(plan.mlflow.version or "").strip()
    if explicit_version:
        return explicit_version
    if plan.deployment.platform == "llm-d":
        return f"llm-d-{plan.deployment.repo_ref}"
    if plan.deployment.platform == "rhoai":
        kubeconfig = str(plan.target_cluster.kubeconfig or "").strip() or None
        try:
            return discover_rhoai_mlflow_version(kubeconfig=kubeconfig)
        except CommandError:
            pass
        return normalize_rhoai_platform_version(plan.deployment.platform_version)
    return f"{plan.deployment.platform}-{plan.deployment.mode}"


def resolved_accelerator(plan: ResolvedRunPlan) -> str:
    try:
        return discover_plan_accelerator(plan)
    except CommandError as exc:
        warning(
            "Could not auto-discover accelerator from the cluster; "
            f"falling back to unknown: {exc}"
        )
        return "unknown"
