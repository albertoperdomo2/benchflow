"""Whole-node target-cluster reservations for BenchFlow deployments."""

from __future__ import annotations

import copy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json

from .cluster import (
    CommandError,
    require_any_command,
    run_command,
    run_json_command,
    use_kubeconfig,
)
from .models import ResolvedRunPlan
from .ui import detail

NODE_EXCLUSIVE_RELEASE_LABEL = "benchflow.io/node-exclusive-release"
NODE_EXCLUSIVE_NODE_LABEL = "benchflow.io/node"


def _lease_name(node: str) -> str:
    return f"benchflow-node-slot-{hashlib.sha1(node.encode()).hexdigest()[:16]}"


def _needed(plan: ResolvedRunPlan) -> int:
    runtime = plan.deployment.runtime
    return runtime.replicas * runtime.tensor_parallelism * runtime.pipeline_parallelism


def _node_gpu_count(item: dict) -> int:
    labels = (item.get("metadata") or {}).get("labels") or {}
    try:
        count = int(
            ((item.get("status") or {}).get("allocatable") or {}).get(
                "nvidia.com/gpu", 0
            )
        )
    except (TypeError, ValueError):
        count = 0
    if count:
        return count
    try:
        return int(labels.get("nvidia.com/gpu.count", 0))
    except (TypeError, ValueError):
        return 0


def _gpu_workload_nodes(kubectl: str) -> set[str]:
    payload = run_json_command([kubectl, "get", "pods", "-A", "-o", "json"])
    occupied: set[str] = set()
    for item in payload.get("items", []) or []:
        metadata = item.get("metadata", {}) or {}
        if metadata.get("deletionTimestamp"):
            continue
        status = item.get("status", {}) or {}
        if str(status.get("phase") or "") not in {"Pending", "Running"}:
            continue
        spec = item.get("spec", {}) or {}
        node = str(spec.get("nodeName") or "").strip()
        if not node:
            continue
        has_gpu = bool(spec.get("resourceClaims"))
        for container in [
            *(spec.get("initContainers", []) or []),
            *(spec.get("containers", []) or []),
        ]:
            resources = container.get("resources", {}) or {}
            for values in (
                resources.get("requests", {}) or {},
                resources.get("limits", {}) or {},
            ):
                try:
                    has_gpu = (
                        has_gpu or int(str(values.get("nvidia.com/gpu") or "0")) > 0
                    )
                except ValueError:
                    continue
        if has_gpu:
            occupied.add(node)
    return occupied


def _affinity_plan(plan: ResolvedRunPlan, nodes: list[str]) -> ResolvedRunPlan:
    affinity = copy.deepcopy(plan.deployment.runtime.affinity)
    terms = (
        affinity.setdefault("nodeAffinity", {})
        .setdefault("requiredDuringSchedulingIgnoredDuringExecution", {})
        .setdefault("nodeSelectorTerms", [])
    )
    hostname_expression = {
        "key": "kubernetes.io/hostname",
        "operator": "In",
        "values": nodes,
    }
    if not terms:
        terms.append({"matchExpressions": [hostname_expression]})
    else:
        for term in terms:
            expressions = term.setdefault("matchExpressions", [])
            existing = next(
                (
                    expression
                    for expression in expressions
                    if expression.get("key") == "kubernetes.io/hostname"
                    and expression.get("operator") == "In"
                ),
                None,
            )
            if existing is None:
                expressions.append(copy.deepcopy(hostname_expression))
            else:
                existing["values"] = list(nodes)
    return replace(
        plan,
        deployment=replace(
            plan.deployment,
            runtime=replace(plan.deployment.runtime, affinity=affinity),
        ),
    )


def release_reservation(*, namespace: str, release: str, kubeconfig: str = "") -> None:
    if not release:
        return
    with use_kubeconfig(kubeconfig):
        kubectl = require_any_command("oc", "kubectl")
        run_command(
            [
                kubectl,
                "delete",
                "lease",
                "-n",
                namespace,
                "-l",
                f"{NODE_EXCLUSIVE_RELEASE_LABEL}={release}",
                "--ignore-not-found=true",
            ],
            capture_output=True,
        )


def release_nodes(plan: ResolvedRunPlan) -> None:
    if plan.deployment.runtime.placement.mode != "node-exclusive":
        return
    release_reservation(
        namespace=plan.deployment.namespace,
        release=plan.deployment.release_name,
        kubeconfig=plan.target_cluster.kubeconfig,
    )


def reserve_nodes(plan: ResolvedRunPlan) -> ResolvedRunPlan | None:
    """Try once to reserve complete target nodes and inject required affinity."""
    if plan.deployment.runtime.placement.mode != "node-exclusive":
        return plan
    with use_kubeconfig(plan.target_cluster.kubeconfig):
        return _reserve_nodes(plan)


def _reserve_nodes(plan: ResolvedRunPlan) -> ResolvedRunPlan | None:
    kubectl = require_any_command("oc", "kubectl")
    need = _needed(plan)
    namespace = plan.deployment.namespace
    release = plan.deployment.release_name
    pool = plan.deployment.runtime.placement.spread_pool

    nodes_payload = run_json_command([kubectl, "get", "nodes", "-o", "json"])
    leases_payload = run_json_command(
        [
            kubectl,
            "get",
            "lease",
            "-n",
            namespace,
            "-l",
            "app.kubernetes.io/name=benchflow",
            "-o",
            "json",
        ]
    )
    occupied = _gpu_workload_nodes(kubectl)
    leases_by_node = {
        str(
            (item.get("metadata", {}) or {})
            .get("labels", {})
            .get(NODE_EXCLUSIVE_NODE_LABEL)
            or ""
        ): item
        for item in leases_payload.get("items", []) or []
    }

    eligible: list[tuple[str, int]] = []
    owned: list[tuple[str, int]] = []
    for item in nodes_payload.get("items", []) or []:
        metadata = item.get("metadata", {}) or {}
        labels = metadata.get("labels", {}) or {}
        spec = item.get("spec", {}) or {}
        name = str(metadata.get("name") or "")
        ready = any(
            condition.get("type") == "Ready" and condition.get("status") == "True"
            for condition in ((item.get("status") or {}).get("conditions") or [])
        )
        gpus = _node_gpu_count(item)
        if (
            not name
            or not ready
            or spec.get("unschedulable")
            or labels.get("benchflow.io/placement-pool") != pool
            or not gpus
        ):
            continue
        lease = leases_by_node.get(name)
        holder = str((lease or {}).get("spec", {}).get("holderIdentity") or "")
        if holder == release:
            owned.append((name, gpus))
        elif lease is None and name not in occupied:
            eligible.append((name, gpus))

    claimed = list(owned)
    if sum(value for _, value in claimed) < need:
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        for name, gpus in eligible:
            doc = {
                "apiVersion": "coordination.k8s.io/v1",
                "kind": "Lease",
                "metadata": {
                    "name": _lease_name(name),
                    "namespace": namespace,
                    "labels": {
                        "app.kubernetes.io/name": "benchflow",
                        NODE_EXCLUSIVE_RELEASE_LABEL: release,
                        NODE_EXCLUSIVE_NODE_LABEL: name,
                    },
                },
                "spec": {
                    "holderIdentity": release,
                    "leaseDurationSeconds": 43200,
                    "acquireTime": now,
                    "renewTime": now,
                },
            }
            result = run_command(
                [kubectl, "create", "-f", "-"],
                input_text=json.dumps(doc),
                capture_output=True,
                check=False,
            )
            if result.returncode == 0:
                claimed.append((name, gpus))
            elif "AlreadyExists" in f"{result.stdout}\n{result.stderr}":
                existing = run_json_command(
                    [
                        kubectl,
                        "get",
                        "lease",
                        _lease_name(name),
                        "-n",
                        namespace,
                        "-o",
                        "json",
                    ]
                )
                if (
                    str((existing.get("spec", {}) or {}).get("holderIdentity") or "")
                    == release
                ):
                    claimed.append((name, gpus))
            else:
                release_nodes(plan)
                raise CommandError(
                    f"failed to reserve target node {name}: {result.stderr or result.stdout}"
                )
            if sum(value for _, value in claimed) >= need:
                break

    if sum(value for _, value in claimed) < need:
        release_nodes(plan)
        return None

    names = [node for node, _ in claimed]
    detail(f"Reserved exclusive node(s): {', '.join(names)}")
    return _affinity_plan(plan, names)
