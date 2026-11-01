from __future__ import annotations

from copy import deepcopy
from typing import Any

from .models import ResolvedRunPlan, ValidationError

PLACEMENT_POOL_NODE_LABEL = "benchflow.io/placement-pool"


def runtime_node_selector(plan: ResolvedRunPlan) -> dict[str, str]:
    """Return the effective model-server node selector for a run."""
    node_selector = dict(plan.deployment.runtime.node_selector)
    placement = plan.deployment.runtime.placement
    if placement.mode != "topology-spread":
        return node_selector

    configured_pool = node_selector.get(PLACEMENT_POOL_NODE_LABEL)
    if configured_pool and configured_pool != placement.spread_pool:
        raise ValidationError(
            "runtime.node_selector conflicts with runtime.placement.spread_pool: "
            f"{PLACEMENT_POOL_NODE_LABEL}={configured_pool!r}, expected "
            f"{placement.spread_pool!r}"
        )
    node_selector[PLACEMENT_POOL_NODE_LABEL] = placement.spread_pool
    return node_selector


def runtime_topology_spread_constraints(
    plan: ResolvedRunPlan,
) -> list[dict[str, Any]]:
    """Build the strict, release-scoped hostname spread policy for model pods."""
    if plan.deployment.runtime.placement.mode != "topology-spread":
        return []

    platform = plan.deployment.platform
    if platform == "llm-d":
        label_selector: dict[str, Any] = {
            "matchLabels": {
                "llm-d.ai/inferenceServing": "true",
                "llm-d.ai/model": plan.deployment.release_name,
            }
        }
        match_label_keys: list[str] = []
    elif platform == "rhoai":
        # RHOAI serving CRDs expose a PodSpec rather than PodTemplate metadata,
        # so release labels cannot be injected directly. RHOAI/KServe supplies
        # stable workload identity labels; matchLabelKeys scopes the constraint
        # to replicas of the same generated workload.
        if plan.deployment.target.resource_kind == "InferenceService":
            label_selector = {"matchLabels": {}}
            match_label_keys = ["serving.kserve.io/inferenceservice"]
        else:
            label_selector = {
                "matchLabels": {
                    "app.kubernetes.io/part-of": "llminferenceservice",
                    "kserve.io/component": "workload",
                }
            }
            match_label_keys = ["app.kubernetes.io/name"]
    elif platform == "rhaiis":
        label_selector = {
            "matchLabels": {
                "app.kubernetes.io/component": "raw-vllm",
                "benchflow.io/release": plan.deployment.release_name,
            }
        }
        match_label_keys = []
    else:
        raise ValidationError(
            "runtime.placement.mode 'topology-spread' is not supported for "
            f"platform {platform!r}"
        )

    constraint: dict[str, Any] = {
        "maxSkew": 1,
        "minDomains": 2,
        "topologyKey": "kubernetes.io/hostname",
        "whenUnsatisfiable": "DoNotSchedule",
        "labelSelector": deepcopy(label_selector),
    }
    if match_label_keys:
        constraint["matchLabelKeys"] = match_label_keys
    return [constraint]
