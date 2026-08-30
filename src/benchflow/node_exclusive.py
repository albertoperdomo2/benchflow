"""Whole-node target-cluster reservations for BenchFlow deployments."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import time

from .cluster import (
    CommandError,
    require_any_command,
    run_command,
    run_json_command,
    use_kubeconfig,
)
from .models import ResolvedRunPlan
from .ui import detail, step

_LABEL = "benchflow.io/node-exclusive-release"


def _lease_name(node: str) -> str:
    return f"benchflow-node-slot-{hashlib.sha1(node.encode()).hexdigest()[:16]}"


def _needed(plan: ResolvedRunPlan) -> int:
    runtime = plan.deployment.runtime
    return runtime.replicas * runtime.tensor_parallelism * runtime.pipeline_parallelism


def release_nodes(plan: ResolvedRunPlan) -> None:
    if plan.deployment.runtime.placement.mode != "node-exclusive":
        return
    kubectl = require_any_command("oc", "kubectl")
    run_command(
        [
            kubectl,
            "delete",
            "lease",
            "-n",
            plan.deployment.namespace,
            "-l",
            f"{_LABEL}={plan.deployment.release_name}",
            "--ignore-not-found=true",
        ],
        capture_output=True,
        check=False,
    )


def allocate_nodes(
    plan: ResolvedRunPlan,
    timeout_seconds: int,
    *,
    kubeconfig: str = "",
) -> ResolvedRunPlan:
    with use_kubeconfig(kubeconfig):
        return _allocate_nodes(plan, timeout_seconds)


def _allocate_nodes(plan: ResolvedRunPlan, timeout_seconds: int) -> ResolvedRunPlan:
    if plan.deployment.runtime.placement.mode != "node-exclusive":
        return plan
    kubectl = require_any_command("oc", "kubectl")
    need, namespace, release = (
        _needed(plan),
        plan.deployment.namespace,
        plan.deployment.release_name,
    )
    pool = plan.deployment.runtime.placement.spread_pool
    step(f"Reserving exclusive nodes for {release} ({need} GPU(s))")
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        payload = run_json_command([kubectl, "get", "nodes", "-o", "json"])
        claimed: list[tuple[str, int]] = []
        for item in payload.get("items", []):
            labels = (item.get("metadata") or {}).get("labels") or {}
            name = str((item.get("metadata") or {}).get("name") or "")
            ready = any(
                c.get("type") == "Ready" and c.get("status") == "True"
                for c in ((item.get("status") or {}).get("conditions") or [])
            )
            if (
                not name
                or not ready
                or labels.get("benchflow.io/placement-pool") != pool
            ):
                continue
            try:
                gpus = int(
                    ((item.get("status") or {}).get("allocatable") or {}).get(
                        "nvidia.com/gpu", 0
                    )
                )
            except (TypeError, ValueError):
                gpus = 0
            if not gpus:
                # NVIDIA DRA clusters advertise physical GPU count through GFD
                # labels while the legacy extended resource remains zero.
                try:
                    gpus = int(labels.get("nvidia.com/gpu.count", 0))
                except (TypeError, ValueError):
                    gpus = 0
            if not gpus:
                continue
            doc = {
                "apiVersion": "coordination.k8s.io/v1",
                "kind": "Lease",
                "metadata": {
                    "name": _lease_name(name),
                    "namespace": namespace,
                    "labels": {
                        "app.kubernetes.io/name": "benchflow",
                        _LABEL: release,
                        "benchflow.io/node": name,
                    },
                },
                "spec": {"holderIdentity": release, "leaseDurationSeconds": 43200},
            }
            result = run_command(
                [kubectl, "create", "-f", "-"],
                input_text=json.dumps(doc),
                capture_output=True,
                check=False,
            )
            if result.returncode == 0:
                claimed.append((name, gpus))
                if sum(value for _, value in claimed) >= need:
                    names = [node for node, _ in claimed]
                    affinity = dict(plan.deployment.runtime.affinity)
                    terms = (
                        affinity.setdefault("nodeAffinity", {})
                        .setdefault(
                            "requiredDuringSchedulingIgnoredDuringExecution", {}
                        )
                        .setdefault("nodeSelectorTerms", [])
                    )
                    terms.append(
                        {
                            "matchExpressions": [
                                {
                                    "key": "kubernetes.io/hostname",
                                    "operator": "In",
                                    "values": names,
                                }
                            ]
                        }
                    )
                    detail(f"Reserved exclusive node(s): {', '.join(names)}")
                    return replace(
                        plan,
                        deployment=replace(
                            plan.deployment,
                            runtime=replace(plan.deployment.runtime, affinity=affinity),
                        ),
                    )
            elif "AlreadyExists" not in f"{result.stdout}\n{result.stderr}":
                raise CommandError(
                    f"failed to reserve target node {name}: {result.stderr or result.stdout}"
                )
        release_nodes(plan)
        time.sleep(10)
    raise CommandError(
        f"timed out reserving {need} GPU(s) from node-exclusive pool {pool!r}"
    )
