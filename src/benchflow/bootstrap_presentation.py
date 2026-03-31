from __future__ import annotations

from typing import Any

from .ui import detail, panel, rule, step


def _gpu_prerequisites_text(options: Any) -> str:
    if options.install_accelerator_prerequisites:
        return "NFD operator + instance, GPU operator + ClusterPolicy"
    return "not installed in this bootstrap mode"


def _models_storage_text(options: Any) -> str:
    if not options.install_models_storage:
        return "not installed in this bootstrap mode"
    suffix = (
        f" via {options.models_storage_class}" if options.models_storage_class else ""
    )
    return f"{options.models_storage_access_mode} {options.models_storage_size}{suffix}"


def _results_storage_text(options: Any) -> str:
    if not options.install_results_storage:
        return "not installed in this bootstrap mode"
    suffix = (
        f" via {options.results_storage_class}" if options.results_storage_class else ""
    )
    return f"ReadWriteOnce {options.results_storage_size}{suffix}"


def print_bootstrap_intro(
    *,
    bootstrap_mode: str,
    options: Any,
    grafana_namespace: str,
    kueue_namespace: str,
    nfd_namespace: str,
    gpu_operator_namespace: str,
    default_storage_class: str | None,
) -> None:
    rule("BenchFlow Bootstrap")
    panel(
        "Configuration",
        (
            ("Bootstrap mode", bootstrap_mode),
            ("Namespace", options.namespace),
            ("Grafana namespace", grafana_namespace),
            ("Kueue namespace", kueue_namespace),
            ("NFD namespace", nfd_namespace),
            ("GPU operator namespace", gpu_operator_namespace),
            ("GPU prerequisites", _gpu_prerequisites_text(options)),
            ("Install Tekton if missing", str(options.install_tekton).lower()),
            ("Install Kueue if missing", str(options.install_kueue).lower()),
            ("Install Grafana if missing", str(options.install_grafana).lower()),
            ("OpenShift Pipelines channel", options.tekton_channel),
            (
                "Target kubeconfig",
                options.target_kubeconfig or "current cluster context",
            ),
            ("models-storage", _models_storage_text(options)),
            ("benchmark-results", _results_storage_text(options)),
            (
                "metrics access",
                "cluster-monitoring-view -> benchflow-runner, benchflow-grafana",
            ),
        ),
    )
    if options.install_models_storage and options.models_storage_class is None:
        detail(f"default StorageClass for models-storage: {default_storage_class}")
    if (
        options.install_models_storage
        and options.models_storage_access_mode == "ReadWriteOnce"
    ):
        detail(
            "note: the shipped qwen smoke profile is single-replica and matches ReadWriteOnce"
        )


def print_bootstrap_summary(
    *,
    bootstrap_mode: str,
    options: Any,
    grafana_namespace: str,
    kueue_namespace: str,
    nfd_namespace: str,
    gpu_operator_namespace: str,
    grafana_host: str | None,
    grafana_admin_secret_name: str,
) -> None:
    rule("Bootstrap Complete")
    panel(
        "BenchFlow",
        (
            ("Bootstrap mode", bootstrap_mode),
            ("Namespace", options.namespace),
            ("Grafana namespace", grafana_namespace),
            ("Kueue namespace", kueue_namespace),
            ("NFD namespace", nfd_namespace),
            ("GPU operator namespace", gpu_operator_namespace),
            ("GPU prerequisites", _gpu_prerequisites_text(options)),
            ("Tekton install attempted", str(options.install_tekton).lower()),
            ("Kueue install attempted", str(options.install_kueue).lower()),
            ("Grafana install attempted", str(options.install_grafana).lower()),
            ("OpenShift Pipelines channel", options.tekton_channel),
            (
                "Target kubeconfig",
                options.target_kubeconfig or "current cluster context",
            ),
            ("models-storage", _models_storage_text(options)),
            ("benchmark-results", _results_storage_text(options)),
            (
                "metrics access",
                "cluster-monitoring-view bound to benchflow-runner and benchflow-grafana",
            ),
            (
                "Grafana route",
                f"https://{grafana_host}" if grafana_host else "not detected yet",
            ),
        ),
    )
    if grafana_host:
        detail(
            "Grafana admin password: "
            f"oc get secret -n {grafana_namespace} {grafana_admin_secret_name} "
            '-o go-template=\'{{index .data "admin-password" | base64decode}}{{"\\n"}}\''
        )
    step("Required secrets if you have not already created them")
    detail("config/cluster/secrets/huggingface-token.example.yaml")
    detail("config/cluster/secrets/mlflow-auth.example.yaml")
    detail("config/cluster/secrets/mlflow-s3-creds.example.yaml")
    step("Example run")
    detail("pip install -e .")
    if bootstrap_mode == "single-cluster":
        detail(
            f"bflow experiment run experiments/smoke/qwen3-06b-llm-d-smoke.yaml --namespace {options.namespace}"
        )
    elif bootstrap_mode == "management-cluster":
        detail(
            "bflow bootstrap --target-kubeconfig ~/.kube/target-cluster --cluster-name target-cluster"
        )
        detail(
            f"bflow experiment run experiments/smoke/qwen3-06b-rhoai-distributed-default-smoke.yaml --namespace {options.namespace} --cluster-name target-cluster"
        )
    else:
        detail(
            f"bflow experiment run experiments/smoke/qwen3-06b-rhoai-distributed-default-smoke.yaml --namespace {options.namespace} --cluster-name <management-secret-name>"
        )
