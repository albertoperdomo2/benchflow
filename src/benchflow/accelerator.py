from __future__ import annotations

import re
from typing import Any

from .cluster import CommandError, require_any_command, run_json_command, use_kubeconfig
from .models import ResolvedRunPlan
from .ui import detail, warning

_KNOWN_ACCELERATORS = (
    "MI300X",
    "MI250X",
    "RTX6000ADA",
    "A10G",
    "L40S",
    "H200",
    "H100",
    "A100",
    "A30",
    "A10",
    "L40",
    "L4",
    "T4",
    "V100",
)


def _release_token_matches(candidate: str, release_name: str) -> bool:
    if candidate == release_name:
        return True
    for separator in ("-", ".", "/", "_"):
        if candidate.startswith(f"{release_name}{separator}"):
            return True
        if candidate.endswith(f"{separator}{release_name}"):
            return True
        if f"{separator}{release_name}{separator}" in candidate:
            return True
    return False


def _matches_release(metadata: dict[str, object], release_name: str) -> bool:
    name = metadata.get("name")
    if isinstance(name, str) and _release_token_matches(name, release_name):
        return True

    labels = metadata.get("labels")
    if isinstance(labels, dict):
        for value in labels.values():
            if isinstance(value, str) and _release_token_matches(value, release_name):
                return True

    owner_references = metadata.get("ownerReferences")
    if isinstance(owner_references, list):
        for owner in owner_references:
            if not isinstance(owner, dict):
                continue
            owner_name = owner.get("name")
            if isinstance(owner_name, str) and _release_token_matches(
                owner_name, release_name
            ):
                return True

    return False


def _pod_type(pod_name: str) -> str:
    lowered = pod_name.lower()
    if any(token in lowered for token in ("gaie", "scheduler", "epp", "kserve-router")):
        return "gaie"
    if any(
        token in lowered
        for token in (
            "ms-",
            "model-service",
            "inference",
            "decode",
            "prefill",
            "predictor",
            "kserve-",
            "vllm",
        )
    ):
        return "model"
    return "infra"


def _normalize_accelerator_label(raw: str) -> str:
    candidate = str(raw or "").strip()
    if not candidate:
        raise CommandError("GPU product label is empty")

    compact = re.sub(r"[^A-Za-z0-9]+", "", candidate).upper()
    for known in _KNOWN_ACCELERATORS:
        if known in compact:
            return known

    simplified = re.sub(
        r"^(NVIDIA|AMD|INTEL)[-_ ]+", "", candidate, flags=re.IGNORECASE
    )
    simplified = re.sub(r"[-_]+", " ", simplified).strip()
    return simplified or candidate


def _matching_model_pods(plan: ResolvedRunPlan) -> list[dict[str, Any]]:
    kubectl_cmd = require_any_command("oc", "kubectl")
    kubeconfig = str(plan.target_cluster.kubeconfig or "").strip() or None
    metrics_release_name = plan.deployment.target.scoped_release_name(
        plan.deployment.release_name
    )
    with use_kubeconfig(kubeconfig):
        payload = run_json_command(
            [
                kubectl_cmd,
                "get",
                "pods",
                "-n",
                plan.deployment.namespace,
                "-o",
                "json",
            ]
        )

    items = payload.get("items", [])
    if not isinstance(items, list):
        return []

    matched: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        pod_name = str(metadata.get("name") or "")
        if not pod_name or _pod_type(pod_name) != "model":
            continue
        if _matches_release(metadata, metrics_release_name):
            matched.append(item)
    return matched


def _node_gpu_product(node_name: str, kubeconfig: str | None) -> str:
    kubectl_cmd = require_any_command("oc", "kubectl")
    with use_kubeconfig(kubeconfig):
        payload = run_json_command(
            [kubectl_cmd, "get", "node", node_name, "-o", "json"]
        )
    labels = payload.get("metadata", {}).get("labels", {}) or {}
    if not isinstance(labels, dict):
        raise CommandError(f"node {node_name} does not expose metadata.labels")

    for key in ("nvidia.com/gpu.product", "amd.com/gpu.product"):
        value = labels.get(key)
        if isinstance(value, str) and value.strip():
            return value

    raise CommandError(
        f"node {node_name} does not expose a supported GPU product label"
    )


def discover_plan_accelerator(plan: ResolvedRunPlan) -> str:
    explicit = str(
        plan.mlflow.tags.get("accelerator")
        or plan.deployment.options.get("accelerator")
        or ""
    ).strip()
    if explicit:
        return explicit

    pods = _matching_model_pods(plan)
    if not pods:
        warning(
            "Could not auto-discover accelerator: no matching model pods were found; "
            "falling back to unknown"
        )
        return "unknown"

    kubeconfig = str(plan.target_cluster.kubeconfig or "").strip() or None
    node_names = sorted(
        {
            str((item.get("spec", {}) or {}).get("nodeName") or "").strip()
            for item in pods
            if str((item.get("spec", {}) or {}).get("nodeName") or "").strip()
        }
    )
    if not node_names:
        warning(
            "Could not auto-discover accelerator: matching model pods are not "
            "scheduled on nodes yet; falling back to unknown"
        )
        return "unknown"

    accelerators = {
        _normalize_accelerator_label(_node_gpu_product(node_name, kubeconfig))
        for node_name in node_names
    }
    if len(accelerators) != 1:
        raise CommandError(
            "auto-discovered multiple accelerator types for the run: "
            + ", ".join(sorted(accelerators))
        )

    accelerator = next(iter(accelerators))
    detail(f"Auto-discovered accelerator: {accelerator}")
    return accelerator
