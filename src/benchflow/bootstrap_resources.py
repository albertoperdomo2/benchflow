from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yaml

from .cluster import CommandError
from .ui import detail, step


DCGM_EXPORTER_SERVICE_TIMEOUT_SECONDS = 300


def install_real_secrets(installer: Any) -> None:
    secrets_dir = installer.repo_root / "config" / "cluster" / "secrets"
    found = False
    for secret_file in sorted(secrets_dir.glob("*.yaml")):
        if secret_file.name.endswith(".example.yaml"):
            continue
        found = True
        step(f"Applying secret {secret_file.name}")
        installer._oc(
            "apply",
            "-n",
            installer.options.namespace,
            "-f",
            str(secret_file),
            retry=True,
            description=f"applying secret {secret_file.name}",
            echo_output=True,
        )
    if not found:
        detail("No non-example secrets found under config/cluster/secrets")


def apply_manifest_tree(installer: Any, root_dir: Path, label: str) -> None:
    step(f"Applying {label}")
    for manifest in sorted(root_dir.rglob("*.yaml")):
        detail(str(manifest.relative_to(installer.repo_root)))
        installer._oc(
            "apply",
            "-n",
            installer.options.namespace,
            "-f",
            str(manifest),
            retry=True,
            description=f"applying {manifest.relative_to(installer.repo_root)}",
            echo_output=True,
        )


def apply_workspace_pvcs(installer: Any) -> None:
    variables = {
        "MODELS_STORAGE_ACCESS_MODE": installer.options.models_storage_access_mode,
        "MODELS_STORAGE_CLASS": installer.options.models_storage_class,
        "MODELS_STORAGE_SIZE": installer.options.models_storage_size,
        "RESULTS_STORAGE_ACCESS_MODE": installer.options.results_storage_access_mode,
        "RESULTS_STORAGE_CLASS": installer.options.results_storage_class,
        "RESULTS_STORAGE_SIZE": installer.options.results_storage_size,
    }
    documents = installer._render_asset_documents("workspaces/pvcs.yaml", variables)
    selected: list[dict[str, Any]] = []
    for document in documents:
        name = str(document.get("metadata", {}).get("name") or "")
        if name == "models-storage" and installer.options.install_models_storage:
            selected.append(document)
        elif name == "benchmark-results" and installer.options.install_results_storage:
            selected.append(document)
    if not selected:
        detail("Skipping workspace PVCs for this bootstrap mode")
        return
    step("Applying workspace PVCs")
    installer._apply_documents(
        selected,
        namespace=installer.options.namespace,
        description="applying workspace PVCs",
    )


def apply_cluster_monitoring_rbac(installer: Any) -> None:
    step("Applying cluster monitoring RBAC")
    if not installer._resource_exists("get", "clusterrole", "cluster-monitoring-view"):
        raise CommandError(
            "required ClusterRole not found: cluster-monitoring-view. "
            "This BenchFlow MVP expects OpenShift cluster monitoring to be available."
        )

    installer._apply_asset_documents(
        "rbac/runner-cluster-monitoring-view.yaml",
        namespace=None,
        description="applying cluster monitoring RBAC",
        variables=installer._base_asset_variables(),
    )


def apply_cluster_monitoring_config(installer: Any) -> None:
    step("Enabling user workload monitoring")
    if not installer._resource_exists("get", "namespace", "openshift-monitoring"):
        raise CommandError(
            "required namespace not found: openshift-monitoring. "
            "BenchFlow expects OpenShift cluster monitoring to be available."
        )

    data: dict[str, str] = {}
    if installer._resource_exists(
        "get",
        "configmap",
        "cluster-monitoring-config",
        "-n",
        "openshift-monitoring",
    ):
        configmap = installer._oc_json(
            "get",
            "configmap",
            "cluster-monitoring-config",
            "-n",
            "openshift-monitoring",
            retry=True,
            description="reading cluster monitoring config",
        )
        data = dict(configmap.get("data") or {})

    config = yaml.safe_load(data.get("config.yaml") or "{}") or {}
    if not isinstance(config, dict):
        raise CommandError(
            "openshift-monitoring/cluster-monitoring-config data.config.yaml must be a YAML mapping"
        )

    if config.get("enableUserWorkload") is True:
        detail("User workload monitoring is already enabled")
    else:
        config["enableUserWorkload"] = True
        data["config.yaml"] = yaml.safe_dump(config, sort_keys=False)
        installer._apply_documents(
            [
                {
                    "apiVersion": "v1",
                    "kind": "ConfigMap",
                    "metadata": {
                        "name": "cluster-monitoring-config",
                        "namespace": "openshift-monitoring",
                    },
                    "data": data,
                }
            ],
            namespace=None,
            description="enabling user workload monitoring",
        )


def _dcgm_exporter_metrics_enabled(cluster_policy: dict[str, Any]) -> bool:
    spec = cluster_policy.get("spec") or {}
    if not isinstance(spec, dict):
        return False
    dcgm = spec.get("dcgm") or {}
    dcgm_exporter = spec.get("dcgmExporter") or {}
    service_monitor = (
        dcgm_exporter.get("serviceMonitor") if isinstance(dcgm_exporter, dict) else {}
    ) or {}
    return (
        isinstance(dcgm, dict)
        and isinstance(dcgm_exporter, dict)
        and isinstance(service_monitor, dict)
        and dcgm.get("enabled") is True
        and dcgm_exporter.get("enabled") is True
        and service_monitor.get("enabled") is True
    )


def _ensure_dcgm_exporter_metrics_enabled(installer: Any) -> bool:
    if not installer._resource_exists("get", "crd", "clusterpolicies.nvidia.com"):
        detail(
            "Skipping DCGM exporter configuration because ClusterPolicy CRD is unavailable"
        )
        return False
    if not installer._resource_exists(
        "get", "clusterpolicy.nvidia.com", "gpu-cluster-policy"
    ):
        detail(
            "Skipping DCGM exporter configuration because ClusterPolicy/gpu-cluster-policy does not exist"
        )
        return False

    cluster_policy = installer._oc_json(
        "get",
        "clusterpolicy.nvidia.com",
        "gpu-cluster-policy",
        retry=True,
        description="reading NVIDIA GPU ClusterPolicy",
    )
    if _dcgm_exporter_metrics_enabled(cluster_policy):
        detail(
            "DCGM exporter metrics are already enabled in the NVIDIA GPU ClusterPolicy"
        )
        return True

    step("Enabling DCGM exporter metrics in the NVIDIA GPU ClusterPolicy")
    patch = {
        "spec": {
            "dcgm": {"enabled": True},
            "dcgmExporter": {
                "enabled": True,
                "serviceMonitor": {"enabled": True},
            },
        }
    }
    installer._oc(
        "patch",
        "clusterpolicy.nvidia.com",
        "gpu-cluster-policy",
        "--type=merge",
        "-p",
        json.dumps(patch),
        retry=True,
        description="enabling DCGM exporter metrics",
        echo_output=True,
    )
    return True


def _wait_for_dcgm_exporter_service(installer: Any) -> bool:
    deadline = time.time() + DCGM_EXPORTER_SERVICE_TIMEOUT_SECONDS
    while time.time() < deadline:
        if installer._resource_exists(
            "get",
            "service",
            "nvidia-dcgm-exporter",
            "-n",
            installer.gpu_operator_namespace,
        ):
            return True
        time.sleep(5)
    return False


def apply_gpu_metrics_monitoring(installer: Any) -> None:
    dcgm_exporter_expected = False
    if installer.options.install_accelerator_prerequisites:
        dcgm_exporter_expected = _ensure_dcgm_exporter_metrics_enabled(installer)
    else:
        detail(
            "Skipping DCGM exporter configuration because accelerator prerequisite installation is disabled"
        )

    if not installer._resource_exists(
        "get", "crd", "servicemonitors.monitoring.coreos.com"
    ):
        detail("Skipping DCGM ServiceMonitor because ServiceMonitor CRD is unavailable")
        return
    if not installer._resource_exists(
        "get", "namespace", installer.gpu_operator_namespace
    ):
        detail(
            f"Skipping DCGM ServiceMonitor because namespace {installer.gpu_operator_namespace} does not exist"
        )
        return
    if not installer._resource_exists(
        "get",
        "service",
        "nvidia-dcgm-exporter",
        "-n",
        installer.gpu_operator_namespace,
    ):
        if dcgm_exporter_expected:
            detail(
                "Waiting for service/nvidia-dcgm-exporter after enabling DCGM exporter metrics"
            )
            if _wait_for_dcgm_exporter_service(installer):
                detail("DCGM exporter service is present")
            else:
                detail(
                    "Skipping DCGM ServiceMonitor because service/nvidia-dcgm-exporter did not become available"
                )
                return
        else:
            detail(
                "Skipping DCGM ServiceMonitor because service/nvidia-dcgm-exporter does not exist"
            )
            return

    if not installer._resource_exists(
        "get",
        "service",
        "nvidia-dcgm-exporter",
        "-n",
        installer.gpu_operator_namespace,
    ):
        detail(
            "Skipping DCGM ServiceMonitor because service/nvidia-dcgm-exporter does not exist"
        )
        return

    step("Applying DCGM metrics ServiceMonitor")
    installer._apply_asset_documents(
        "monitoring/nvidia-dcgm-exporter-servicemonitor.yaml",
        namespace=None,
        description="applying DCGM metrics ServiceMonitor",
        variables=installer._base_asset_variables(),
    )


def apply_runner_rbac(installer: Any) -> None:
    step("Applying runner RBAC")
    installer._apply_asset_documents(
        "rbac/runner-namespaced.yaml",
        namespace=installer.options.namespace,
        description="applying runner RBAC",
        variables=installer._base_asset_variables(),
    )
    if installer._resource_exists("get", "namespace", "istio-system"):
        installer._apply_asset_documents(
            "rbac/runner-istio-system.yaml",
            namespace=None,
            description="applying istio-system runner RBAC",
            variables=installer._base_asset_variables(),
        )
    else:
        detail(
            "Skipping istio-system runner RBAC because namespace istio-system does not exist"
        )
    if installer._resource_exists("get", "namespace", "rook-ceph"):
        installer._apply_asset_documents(
            "rbac/runner-rook-ceph-diagnostics.yaml",
            namespace=None,
            description="applying rook-ceph diagnostics RBAC",
            variables=installer._base_asset_variables(),
        )
    else:
        detail(
            "Skipping rook-ceph diagnostics RBAC because namespace rook-ceph does not exist"
        )
    installer._apply_asset_documents(
        "rbac/runner-cluster.yaml",
        namespace=None,
        description="applying runner cluster RBAC",
        variables=installer._base_asset_variables(),
    )


def apply_hostpath_runtime_security(installer: Any) -> None:
    step("Applying managed hostPath runtime security")
    installer._apply_asset_documents(
        "rbac/hostpath-runtime.yaml",
        namespace=None,
        description="applying managed hostPath runtime security",
        variables=installer._base_asset_variables(),
    )


def apply_namespaced_resources(installer: Any) -> None:
    step("Applying namespace RBAC")
    installer._apply_asset_documents(
        "rbac/runner-base.yaml",
        namespace=installer.options.namespace,
        description="applying namespace service account",
        variables=installer._base_asset_variables(),
    )
    apply_hostpath_runtime_security(installer)
    apply_runner_rbac(installer)
    apply_cluster_monitoring_rbac(installer)
    apply_cluster_monitoring_config(installer)
    apply_gpu_metrics_monitoring(installer)
    apply_workspace_pvcs(installer)
    if installer.options.install_tekton:
        apply_manifest_tree(
            installer, installer.repo_root / "tekton" / "tasks", "Tekton tasks"
        )
        apply_manifest_tree(
            installer,
            installer.repo_root / "tekton" / "pipelines",
            "Tekton pipelines",
        )
