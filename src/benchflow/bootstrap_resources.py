from __future__ import annotations

from pathlib import Path
from typing import Any

from .cluster import CommandError
from .ui import detail, step


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
    installer._apply_asset_documents(
        "rbac/runner-cluster.yaml",
        namespace=None,
        description="applying runner cluster RBAC",
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
    apply_runner_rbac(installer)
    apply_cluster_monitoring_rbac(installer)
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
