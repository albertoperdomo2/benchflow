from __future__ import annotations

import time
from typing import Any

from .cluster import CommandError
from .kueue import REMOTE_CAPACITY_CONTROLLER_DEPLOYMENT, ensure_cluster_queue_resources
from .ui import step, success


def kueue_crds_present(installer: Any) -> bool:
    return all(
        installer._resource_exists("get", "crd", name)
        for name in (
            "admissionchecks.kueue.x-k8s.io",
            "clusterqueues.kueue.x-k8s.io",
            "localqueues.kueue.x-k8s.io",
            "resourceflavors.kueue.x-k8s.io",
            "workloads.kueue.x-k8s.io",
        )
    )


def kueue_ready(installer: Any) -> bool:
    if not kueue_crds_present(installer):
        return False
    if not installer._resource_exists(
        "get",
        "deployment",
        "kueue-controller-manager",
        "-n",
        installer.kueue_namespace,
    ):
        return False
    deployment = installer._oc_json(
        "get",
        "deployment",
        "kueue-controller-manager",
        "-n",
        installer.kueue_namespace,
        retry=True,
        description="reading deployment/kueue-controller-manager",
    )
    status = deployment.get("status", {}) or {}
    ready = int(status.get("readyReplicas", 0) or 0)
    desired = int(status.get("replicas", 0) or 0)
    return desired > 0 and ready >= desired


def install_kueue_if_needed(installer: Any) -> None:
    if kueue_ready(installer):
        success("Kueue CRDs and controller are already present")
        return

    installer.ensure_namespace(installer.kueue_namespace)
    if not kueue_crds_present(installer):
        step(f"Installing Kueue in {installer.kueue_namespace}")
        installer._oc(
            "apply",
            "--server-side",
            "-f",
            "https://github.com/kubernetes-sigs/kueue/releases/download/v0.16.4/manifests.yaml",
            retry=True,
            description="installing Kueue",
            echo_output=True,
        )

    step("Waiting for Kueue CRDs")
    for crd_name in (
        "crd/admissionchecks.kueue.x-k8s.io",
        "crd/clusterqueues.kueue.x-k8s.io",
        "crd/localqueues.kueue.x-k8s.io",
        "crd/resourceflavors.kueue.x-k8s.io",
        "crd/workloads.kueue.x-k8s.io",
    ):
        installer._wait_for_resource(
            resource=crd_name,
            namespace=None,
            timeout_seconds=600,
            label=crd_name,
        )

    step("Waiting for the Kueue controller manager")
    installer._wait_for_resource(
        resource="deployment/kueue-controller-manager",
        namespace=installer.kueue_namespace,
        timeout_seconds=600,
        label=f"deployment/kueue-controller-manager in namespace {installer.kueue_namespace}",
    )
    installer._oc(
        "wait",
        "--for=condition=available",
        "--timeout=10m",
        "deployment/kueue-controller-manager",
        "-n",
        installer.kueue_namespace,
        retry=True,
        description="waiting for deployment/kueue-controller-manager",
    )


def apply_kueue_support_resources(installer: Any) -> None:
    step("Applying BenchFlow Kueue support resources")
    installer._apply_asset_documents(
        "operators/kueue/controller.yaml",
        namespace=None,
        description="applying BenchFlow Kueue support resources",
        variables=installer._base_asset_variables(),
    )
    wait_for_kueue_support_ready(installer, timeout_seconds=600)


def wait_for_kueue_support_ready(installer: Any, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if not installer._resource_exists(
            "get",
            "deployment",
            REMOTE_CAPACITY_CONTROLLER_DEPLOYMENT,
            "-n",
            installer.options.namespace,
        ):
            time.sleep(5)
            continue
        deployment = installer._oc_json(
            "get",
            "deployment",
            REMOTE_CAPACITY_CONTROLLER_DEPLOYMENT,
            "-n",
            installer.options.namespace,
            retry=True,
            description=f"reading deployment/{REMOTE_CAPACITY_CONTROLLER_DEPLOYMENT}",
        )
        status = deployment.get("status", {}) or {}
        ready = int(status.get("readyReplicas", 0) or 0)
        desired = int(status.get("replicas", 0) or 0)
        if desired > 0 and ready >= desired:
            return
        time.sleep(5)
    raise CommandError(
        "timed out waiting for the BenchFlow remote-capacity controller deployment"
    )


def register_kueue_cluster_queue(
    installer: Any, *, cluster_name: str, gpu_capacity: int
) -> None:
    ensure_cluster_queue_resources(
        namespace=installer.options.namespace,
        cluster_name=cluster_name,
        gpu_capacity=gpu_capacity,
    )
