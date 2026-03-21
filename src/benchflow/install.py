from __future__ import annotations

import json
import os
import secrets
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .assets import asset_text, render_yaml_documents
from .cluster import (
    CommandError,
    TARGET_KUBECONFIG_HOST_ALIASES_ANNOTATION,
    discover_repo_root,
    require_command,
)
from .kueue import (
    DEFAULT_CONTROLLER_IMAGE,
    KUEUE_NAMESPACE,
    REMOTE_CAPACITY_CONTROLLER_DEPLOYMENT,
    ensure_cluster_queue_resources,
)
from .ui import detail, emit, panel, rule, step, success, warning


CONNECTIVITY_MARKERS = (
    "Unable to connect to the server",
    "no such host",
    "dial tcp",
    "i/o timeout",
    "context deadline exceeded",
    "http2: client connection lost",
    "TLS handshake timeout",
    "Client.Timeout exceeded while awaiting headers",
    "server closed idle connection",
    "the server is currently unable to handle the request",
    "ServiceUnavailable",
    "connection refused",
    "EOF",
)


@dataclass
class BootstrapOptions:
    namespace: str = "benchflow"
    single_cluster: bool = False
    install_grafana: bool = True
    install_tekton: bool = True
    install_kueue: bool = True
    install_accelerator_prerequisites: bool = True
    install_models_storage: bool = True
    install_results_storage: bool = True
    tekton_channel: str = "latest"
    target_kubeconfig: str | None = None
    benchflow_image: str = DEFAULT_CONTROLLER_IMAGE
    models_storage_access_mode: str = "ReadWriteOnce"
    models_storage_size: str = "250Gi"
    models_storage_class: str | None = None
    results_storage_size: str = "20Gi"
    results_storage_class: str | None = None


class Installer:
    kueue_namespace = KUEUE_NAMESPACE
    pipelines_operator_namespace = "openshift-operators"
    pipelines_runtime_namespace = "openshift-pipelines"
    nfd_namespace = "openshift-nfd"
    gpu_operator_namespace = "nvidia-gpu-operator"
    nfd_package_name = "nfd"
    gpu_operator_package_name = "gpu-operator-certified"
    grafana_admin_secret_name = "grafana-admin-credentials"
    grafana_datasource_service_account = "benchflow-grafana"
    grafana_datasource_token_secret = "benchflow-grafana-datasource-token"

    def __init__(self, repo_root: Path, options: BootstrapOptions) -> None:
        self.repo_root = repo_root.resolve()
        self.options = options
        self._default_storage_class_name: str | None = None

    @property
    def grafana_namespace(self) -> str:
        return f"{self.options.namespace}-grafana"

    @property
    def bootstrap_mode(self) -> str:
        if self.options.single_cluster:
            return "single-cluster"
        if self.options.target_kubeconfig:
            return "remote-target"
        return "management-cluster"

    def run(self) -> int:
        self.ensure_cluster_access()
        if self.options.install_models_storage:
            self.ensure_storage_class(
                self.options.models_storage_class, "models-storage"
            )
            self.ensure_default_storage_class_if_needed(
                self.options.models_storage_class, "models-storage"
            )
        if self.options.install_results_storage:
            self.ensure_storage_class(
                self.options.results_storage_class, "benchmark-results"
            )
            self.ensure_default_storage_class_if_needed(
                self.options.results_storage_class, "benchmark-results"
            )
        self.ensure_namespace(self.options.namespace)
        if self.options.install_grafana:
            self.ensure_namespace(self.grafana_namespace)

        self.print_intro()

        if self.options.install_accelerator_prerequisites:
            self.install_accelerator_prerequisites()
        if self.options.install_tekton:
            self.install_tekton_if_needed()
            self.configure_tekton_scc()
        if self.options.install_kueue:
            self.install_kueue_if_needed()
        self.install_grafana_if_needed()
        self.install_real_secrets()
        self.apply_namespaced_resources()
        if self.options.install_kueue:
            self.apply_kueue_support_resources()
        self.apply_grafana_stack()
        self.print_summary()
        return 0

    def print_intro(self) -> None:
        options = self.options
        rule("BenchFlow Bootstrap")
        panel(
            "Configuration",
            (
                ("Bootstrap mode", self.bootstrap_mode),
                ("Namespace", options.namespace),
                ("Grafana namespace", self.grafana_namespace),
                ("Kueue namespace", self.kueue_namespace),
                ("NFD namespace", self.nfd_namespace),
                ("GPU operator namespace", self.gpu_operator_namespace),
                (
                    "GPU prerequisites",
                    (
                        "NFD operator + instance, GPU operator + ClusterPolicy"
                        if options.install_accelerator_prerequisites
                        else "not installed in this bootstrap mode"
                    ),
                ),
                ("Install Tekton if missing", str(options.install_tekton).lower()),
                ("Install Kueue if missing", str(options.install_kueue).lower()),
                ("Install Grafana if missing", str(options.install_grafana).lower()),
                ("OpenShift Pipelines channel", options.tekton_channel),
                (
                    "Target kubeconfig",
                    options.target_kubeconfig or "current cluster context",
                ),
                (
                    "models-storage",
                    (
                        f"{options.models_storage_access_mode} {options.models_storage_size}"
                        f"{f' via {options.models_storage_class}' if options.models_storage_class else ''}"
                        if options.install_models_storage
                        else "not installed in this bootstrap mode"
                    ),
                ),
                (
                    "benchmark-results",
                    (
                        f"ReadWriteOnce {options.results_storage_size}"
                        f"{f' via {options.results_storage_class}' if options.results_storage_class else ''}"
                        if options.install_results_storage
                        else "not installed in this bootstrap mode"
                    ),
                ),
                (
                    "metrics access",
                    "cluster-monitoring-view -> benchflow-runner, benchflow-grafana",
                ),
            ),
        )
        if options.install_models_storage and options.models_storage_class is None:
            detail(
                f"default StorageClass for models-storage: {self.default_storage_class()}"
            )
        if (
            options.install_models_storage
            and options.models_storage_access_mode == "ReadWriteOnce"
        ):
            detail(
                "note: the shipped qwen smoke profile is single-replica and matches ReadWriteOnce"
            )

    def print_summary(self) -> None:
        options = self.options
        grafana_host: str | None = None
        try:
            grafana_host = self.discover_grafana_route_host()
        except CommandError as exc:
            warning(f"Could not query the Grafana route for the final summary: {exc}")
        rule("Bootstrap Complete")
        panel(
            "BenchFlow",
            (
                ("Bootstrap mode", self.bootstrap_mode),
                ("Namespace", options.namespace),
                ("Grafana namespace", self.grafana_namespace),
                ("Kueue namespace", self.kueue_namespace),
                ("NFD namespace", self.nfd_namespace),
                ("GPU operator namespace", self.gpu_operator_namespace),
                (
                    "GPU prerequisites",
                    (
                        "NFD operator + instance, GPU operator + ClusterPolicy"
                        if options.install_accelerator_prerequisites
                        else "not installed in this bootstrap mode"
                    ),
                ),
                ("Tekton install attempted", str(options.install_tekton).lower()),
                ("Kueue install attempted", str(options.install_kueue).lower()),
                ("Grafana install attempted", str(options.install_grafana).lower()),
                ("OpenShift Pipelines channel", options.tekton_channel),
                (
                    "Target kubeconfig",
                    options.target_kubeconfig or "current cluster context",
                ),
                (
                    "models-storage",
                    (
                        f"{options.models_storage_access_mode} {options.models_storage_size}"
                        f"{f' via {options.models_storage_class}' if options.models_storage_class else ''}"
                        if options.install_models_storage
                        else "not installed in this bootstrap mode"
                    ),
                ),
                (
                    "benchmark-results",
                    (
                        f"ReadWriteOnce {options.results_storage_size}"
                        f"{f' via {options.results_storage_class}' if options.results_storage_class else ''}"
                        if options.install_results_storage
                        else "not installed in this bootstrap mode"
                    ),
                ),
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
                f"oc get secret -n {self.grafana_namespace} {self.grafana_admin_secret_name} "
                '-o go-template=\'{{index .data "admin-password" | base64decode}}{{"\\n"}}\''
            )
        step("Required secrets if you have not already created them")
        detail("config/cluster/secrets/huggingface-token.example.yaml")
        detail("config/cluster/secrets/mlflow-auth.example.yaml")
        detail("config/cluster/secrets/mlflow-s3-creds.example.yaml")
        step("Example run")
        detail("pip install -e .")
        if self.bootstrap_mode == "single-cluster":
            detail(
                f"bflow experiment run experiments/smoke/qwen3-06b-llm-d-smoke.yaml --namespace {options.namespace}"
            )
        elif self.bootstrap_mode == "management-cluster":
            detail(
                "bflow bootstrap --target-kubeconfig ~/.kube/target-cluster --cluster-name target-cluster"
            )
            detail(
                f"bflow experiment run experiments/smoke/qwen3-06b-rhoai-kserve-smoke.yaml --namespace {options.namespace} --cluster-name target-cluster"
            )
        else:
            detail(
                f"bflow experiment run experiments/smoke/qwen3-06b-rhoai-kserve-smoke.yaml --namespace {options.namespace} --cluster-name <management-secret-name>"
            )

    def _is_connectivity_error(self, output: str) -> bool:
        return any(marker in output for marker in CONNECTIVITY_MARKERS)

    def _run(
        self,
        argv: list[str],
        *,
        input_text: str | None = None,
        retry: bool = False,
        description: str | None = None,
        echo_output: bool = False,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        attempts = int((5 if retry else 1))
        delay_seconds = 2
        last_output = ""

        for attempt in range(1, attempts + 1):
            result = subprocess.run(
                argv,
                cwd=str(self.repo_root),
                env={
                    **os.environ,
                    **(
                        {"KUBECONFIG": self.options.target_kubeconfig}
                        if self.options.target_kubeconfig
                        else {}
                    ),
                },
                input=input_text,
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode == 0:
                if echo_output and result.stdout:
                    emit(result.stdout, end="")
                return result

            output = (result.stderr or result.stdout or "").strip()
            last_output = output

            if retry and self._is_connectivity_error(output) and attempt < attempts:
                current = description or "running command"
                warning(
                    f"Transient cluster API error while {current}; retrying ({attempt}/{attempts})..."
                )
                if output:
                    detail(output)
                time.sleep(delay_seconds)
                delay_seconds *= 2
                continue

            if check:
                if retry and self._is_connectivity_error(output):
                    current = description or "running command"
                    raise CommandError(
                        f"{current} failed after {attempts} attempts due to cluster API connectivity issues: {output}"
                    )
                raise CommandError(f"{' '.join(argv)}: {output or 'command failed'}")
            return result

        raise CommandError(last_output or f"{' '.join(argv)}: command failed")

    def _oc(
        self,
        *args: str,
        input_text: str | None = None,
        retry: bool = False,
        description: str | None = None,
        echo_output: bool = False,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        return self._run(
            ["oc", *args],
            input_text=input_text,
            retry=retry,
            description=description,
            echo_output=echo_output,
            check=check,
        )

    def _oc_json(
        self, *args: str, retry: bool = False, description: str | None = None
    ) -> Any:
        result = self._oc(*args, "-o", "json", retry=retry, description=description)
        return json.loads(result.stdout or "{}")

    def _resource_exists(self, *args: str) -> bool:
        result = self._oc(*args, check=False)
        output = (result.stderr or result.stdout or "").strip()
        if result.returncode != 0 and self._is_connectivity_error(output):
            raise CommandError(
                f"cluster API is unreachable while running: oc {' '.join(args)}\n{output}"
            )
        return result.returncode == 0

    def _apply_documents(
        self,
        documents: list[dict[str, Any]],
        *,
        namespace: str | None,
        description: str,
    ) -> None:
        manifest = yaml.safe_dump_all(documents, sort_keys=False)
        args = ["apply"]
        if namespace is not None:
            args.extend(["-n", namespace])
        args.extend(["-f", "-"])
        self._oc(
            *args,
            input_text=manifest,
            retry=True,
            description=description,
            echo_output=True,
        )

    def _asset_path(self, relative_path: str | Path) -> Path:
        return Path("bootstrap") / Path(relative_path)

    def _asset_text(self, relative_path: str | Path) -> str:
        return asset_text(self._asset_path(relative_path))

    def _render_asset_documents(
        self, relative_path: str | Path, variables: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        return render_yaml_documents(
            self._asset_path(relative_path),
            variables or {},
        )

    def _apply_asset_documents(
        self,
        relative_path: str | Path,
        *,
        namespace: str | None,
        description: str,
        variables: dict[str, Any] | None = None,
    ) -> None:
        self._apply_documents(
            self._render_asset_documents(relative_path, variables),
            namespace=namespace,
            description=description,
        )

    def _base_asset_variables(self) -> dict[str, Any]:
        return {
            "BENCHFLOW_NAMESPACE": self.options.namespace,
            "BENCHFLOW_IMAGE": self.options.benchflow_image,
            "BENCHFLOW_CONTROLLER_HOST_ALIASES": self._controller_host_aliases(),
            "GRAFANA_NAMESPACE": self.grafana_namespace,
            "GRAFANA_SERVICE_ACCOUNT": self.grafana_datasource_service_account,
            "GRAFANA_DATASOURCE_TOKEN_SECRET": self.grafana_datasource_token_secret,
            "GRAFANA_ADMIN_SECRET_NAME": self.grafana_admin_secret_name,
            "NFD_NAMESPACE": self.nfd_namespace,
            "GPU_OPERATOR_NAMESPACE": self.gpu_operator_namespace,
        }

    def _controller_host_aliases(self) -> list[dict[str, Any]]:
        if not self._resource_exists("get", "namespace", self.options.namespace):
            return []
        try:
            payload = self._oc_json(
                "get",
                "secrets",
                "-n",
                self.options.namespace,
                "-l",
                "benchflow.io/secret-kind=target-kubeconfig",
                retry=True,
                description="reading target kubeconfig Secrets",
            )
        except CommandError:
            return []

        by_ip: dict[str, set[str]] = {}
        for item in payload.get("items", []) or []:
            annotations = item.get("metadata", {}).get("annotations", {}) or {}
            raw = str(
                annotations.get(TARGET_KUBECONFIG_HOST_ALIASES_ANNOTATION, "") or ""
            ).strip()
            if not raw:
                continue
            try:
                host_aliases = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(host_aliases, dict):
                continue
            for hostname, ip_address in host_aliases.items():
                cleaned_hostname = str(hostname).strip()
                cleaned_ip = str(ip_address).strip()
                if not cleaned_hostname or not cleaned_ip:
                    continue
                by_ip.setdefault(cleaned_ip, set()).add(cleaned_hostname)

        return [
            {"ip": ip_address, "hostnames": sorted(hostnames)}
            for ip_address, hostnames in sorted(by_ip.items())
        ]

    def _wait_for_resource(
        self, *, resource: str, namespace: str | None, timeout_seconds: int, label: str
    ) -> None:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            args = ["get", resource]
            if namespace is not None:
                args.extend(["-n", namespace])
            if self._resource_exists(*args):
                return
            time.sleep(5)
        raise CommandError(f"timed out waiting for {label}")

    def _wait_for_secret_key(
        self, *, name: str, key: str, namespace: str, timeout_seconds: int
    ) -> None:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            try:
                secret = self._oc_json(
                    "get",
                    "secret",
                    name,
                    "-n",
                    namespace,
                    retry=True,
                    description=f"reading secret/{name}",
                )
            except CommandError:
                time.sleep(5)
                continue
            value = secret.get("data", {}).get(key)
            if value:
                return
            time.sleep(5)
        raise CommandError(f"timed out waiting for secret/{name} key {key}")

    def ensure_cluster_access(self) -> None:
        require_command("oc")
        self._oc("whoami", retry=True, description="verifying cluster access")

    def ensure_namespace(self, namespace_name: str) -> None:
        if not self._resource_exists("get", "namespace", namespace_name):
            step(f"Creating namespace {namespace_name}")
            self._oc(
                "create",
                "namespace",
                namespace_name,
                retry=True,
                description=f"creating namespace {namespace_name}",
                echo_output=True,
            )
            return

        namespace = self._oc_json(
            "get",
            "namespace",
            namespace_name,
            retry=True,
            description=f"reading namespace/{namespace_name}",
        )
        deletion_timestamp = namespace.get("metadata", {}).get("deletionTimestamp")
        if not deletion_timestamp:
            return

        step(f"Waiting for namespace {namespace_name} to finish terminating")
        deadline = time.time() + 600
        while time.time() < deadline:
            if not self._resource_exists("get", "namespace", namespace_name):
                step(f"Creating namespace {namespace_name}")
                self._oc(
                    "create",
                    "namespace",
                    namespace_name,
                    retry=True,
                    description=f"creating namespace {namespace_name}",
                    echo_output=True,
                )
                return
            time.sleep(5)
        raise CommandError(
            f"timed out waiting for namespace {namespace_name} to finish terminating"
        )

    def ensure_storage_class(self, storage_class: str | None, label: str) -> None:
        if storage_class is None:
            return
        if not self._resource_exists("get", "storageclass", storage_class):
            raise CommandError(f"{label} StorageClass not found: {storage_class}")

    def default_storage_class(self) -> str:
        if self._default_storage_class_name is not None:
            return self._default_storage_class_name

        storage_classes = self._oc_json(
            "get",
            "storageclass",
            retry=True,
            description="discovering the default StorageClass",
        )
        for item in storage_classes.get("items", []):
            annotations = item.get("metadata", {}).get("annotations", {})
            if annotations.get("storageclass.kubernetes.io/is-default-class") == "true":
                self._default_storage_class_name = item["metadata"]["name"]
                return self._default_storage_class_name
        raise CommandError("no default StorageClass was found")

    def ensure_default_storage_class_if_needed(
        self, storage_class: str | None, label: str
    ) -> None:
        if storage_class is None:
            self.default_storage_class()
            return

    def tekton_ready(self) -> bool:
        return all(
            self._resource_exists("get", "crd", name)
            for name in (
                "tasks.tekton.dev",
                "pipelines.tekton.dev",
                "pipelineruns.tekton.dev",
            )
        )

    def _kueue_crds_present(self) -> bool:
        return all(
            self._resource_exists("get", "crd", name)
            for name in (
                "admissionchecks.kueue.x-k8s.io",
                "clusterqueues.kueue.x-k8s.io",
                "localqueues.kueue.x-k8s.io",
                "resourceflavors.kueue.x-k8s.io",
                "workloads.kueue.x-k8s.io",
            )
        )

    def kueue_ready(self) -> bool:
        if not self._kueue_crds_present():
            return False
        if not self._resource_exists(
            "get",
            "deployment",
            "kueue-controller-manager",
            "-n",
            self.kueue_namespace,
        ):
            return False
        deployment = self._oc_json(
            "get",
            "deployment",
            "kueue-controller-manager",
            "-n",
            self.kueue_namespace,
            retry=True,
            description="reading deployment/kueue-controller-manager",
        )
        status = deployment.get("status", {}) or {}
        ready = int(status.get("readyReplicas", 0) or 0)
        desired = int(status.get("replicas", 0) or 0)
        return desired > 0 and ready >= desired

    def _print_olm_diagnostics(
        self, *, subscription_name: str, namespace: str, catalog_source: str
    ) -> None:
        warning("Operator OLM diagnostics")
        for argv in (
            ["get", "subscription", subscription_name, "-n", namespace, "-o", "yaml"],
            ["get", "csv", "-n", namespace],
            ["get", "installplan", "-n", namespace],
            [
                "get",
                "catalogsource",
                catalog_source,
                "-n",
                "openshift-marketplace",
                "-o",
                "yaml",
            ],
            ["get", "pods", "-n", "openshift-operator-lifecycle-manager"],
        ):
            result = self._oc(*argv, check=False)
            output = result.stdout or result.stderr or ""
            if output:
                emit(output, end="" if output.endswith("\n") else "\n", stderr=True)

    def _wait_for_subscription_current_csv(
        self, *, subscription_name: str, namespace: str, timeout_seconds: int
    ) -> str:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            try:
                subscription = self._oc_json(
                    "get",
                    "subscription",
                    subscription_name,
                    "-n",
                    namespace,
                    retry=True,
                    description=f"reading subscription/{subscription_name}",
                )
            except CommandError:
                time.sleep(5)
                continue
            conditions = subscription.get("status", {}).get("conditions", [])
            resolution_failed = next(
                (
                    condition
                    for condition in conditions
                    if condition.get("type") == "ResolutionFailed"
                    and condition.get("status") == "True"
                ),
                None,
            )
            if resolution_failed is not None:
                message = resolution_failed.get(
                    "message", "subscription resolution failed"
                )
                raise CommandError(
                    f"subscription/{subscription_name} in namespace {namespace} failed to resolve: {message}"
                )
            current_csv = subscription.get("status", {}).get("currentCSV", "")
            if current_csv:
                return str(current_csv)
            time.sleep(5)
        raise CommandError(
            f"timed out waiting for subscription/{subscription_name} to resolve a CSV"
        )

    def _get_subscription(
        self, *, subscription_name: str, namespace: str, description: str | None = None
    ) -> dict[str, Any]:
        return self._oc_json(
            "get",
            "subscription",
            subscription_name,
            "-n",
            namespace,
            retry=True,
            description=description or f"reading subscription/{subscription_name}",
        )

    def _get_packagemanifest(self, package_name: str) -> dict[str, Any]:
        return self._oc_json(
            "get",
            "packagemanifest",
            package_name,
            "-n",
            "openshift-marketplace",
            retry=True,
            description=f"reading packagemanifest/{package_name}",
        )

    def _default_channel_for_package(self, package_name: str) -> str:
        package = self._get_packagemanifest(package_name)
        channel = package.get("status", {}).get("defaultChannel", "")
        if not channel:
            raise CommandError(
                f"packagemanifest/{package_name} does not expose a default channel"
            )
        return str(channel)

    def _catalog_source_for_package(self, package_name: str) -> tuple[str, str]:
        package = self._get_packagemanifest(package_name)
        status = package.get("status", {}) or {}
        catalog_source = status.get("catalogSource", "")
        catalog_namespace = status.get("catalogSourceNamespace", "")
        if not catalog_source or not catalog_namespace:
            raise CommandError(
                f"packagemanifest/{package_name} does not expose catalog source details"
            )
        return str(catalog_source), str(catalog_namespace)

    def _operatorgroups_in_namespace(self, namespace: str) -> list[dict[str, Any]]:
        operatorgroups = self._oc_json(
            "get",
            "operatorgroup",
            "-n",
            namespace,
            retry=True,
            description=f"reading operatorgroups in namespace {namespace}",
        )
        return list(operatorgroups.get("items", []))

    def _reuse_or_create_operatorgroup(
        self, *, namespace: str, operatorgroup_name: str
    ) -> bool:
        operatorgroups = self._operatorgroups_in_namespace(namespace)
        if not operatorgroups:
            return False

        if len(operatorgroups) > 1:
            names = ", ".join(
                str(item.get("metadata", {}).get("name", "unknown"))
                for item in operatorgroups
            )
            raise CommandError(
                f"namespace {namespace} already has multiple OperatorGroups ({names}); "
                "BenchFlow requires exactly one. Clean the namespace and rerun bootstrap."
            )

        operatorgroup = operatorgroups[0]
        target_namespaces = (
            operatorgroup.get("spec", {}).get("targetNamespaces", []) or []
        )
        if target_namespaces not in ([], [namespace]):
            raise CommandError(
                f"existing OperatorGroup {operatorgroup.get('metadata', {}).get('name', 'unknown')} "
                f"in namespace {namespace} targets {target_namespaces}, expected [{namespace}]"
            )
        detail(
            "Reusing existing OperatorGroup "
            f"{operatorgroup.get('metadata', {}).get('name', 'unknown')} in namespace {namespace}"
        )
        return True

    def _install_operator_from_package(
        self,
        *,
        package_name: str,
        namespace: str,
        subscription_name: str,
        operatorgroup_name: str,
        asset_path: str,
    ) -> str:
        channel = self._default_channel_for_package(package_name)
        catalog_source, catalog_namespace = self._catalog_source_for_package(
            package_name
        )
        self.ensure_namespace(namespace)
        has_existing_operatorgroup = self._reuse_or_create_operatorgroup(
            namespace=namespace, operatorgroup_name=operatorgroup_name
        )
        documents = self._render_asset_documents(
            asset_path,
            {
                "OPERATOR_NAMESPACE": namespace,
                "OPERATORGROUP_NAME": operatorgroup_name,
                "SUBSCRIPTION_NAME": subscription_name,
                "PACKAGE_NAME": package_name,
                "CHANNEL": channel,
                "SOURCE": catalog_source,
                "SOURCE_NAMESPACE": catalog_namespace,
            },
        )
        if has_existing_operatorgroup:
            documents = [
                document
                for document in documents
                if document.get("kind") != "OperatorGroup"
            ]
        self._apply_documents(
            documents,
            namespace=None,
            description=f"installing operator package {package_name}",
        )
        step(f"Waiting for the {package_name} subscription to resolve")
        csv_name = self._wait_for_subscription_current_csv(
            subscription_name=subscription_name,
            namespace=namespace,
            timeout_seconds=600,
        )
        step(f"Waiting for CSV {csv_name} to succeed")
        self._wait_for_csv_succeeded(
            subscription_name=subscription_name,
            namespace=namespace,
            csv_name=csv_name,
            timeout_seconds=900,
            csv_prefix=f"{package_name}.",
            catalog_source=catalog_source,
        )
        return csv_name

    def _approve_pending_installplan(
        self,
        *,
        subscription_name: str,
        namespace: str,
        csv_prefix: str,
        catalog_source: str,
        expected_csv_name: str | None = None,
    ) -> None:
        subscription = self._oc_json(
            "get",
            "subscription",
            subscription_name,
            "-n",
            namespace,
            retry=True,
            description=f"checking InstallPlan state for subscription/{subscription_name}",
        )
        conditions = subscription.get("status", {}).get("conditions", [])
        pending = next(
            (
                condition
                for condition in conditions
                if condition.get("type") == "InstallPlanPending"
            ),
            None,
        )
        if (
            not pending
            or pending.get("status") != "True"
            or pending.get("reason") != "RequiresApproval"
        ):
            return

        installplan_name = (
            subscription.get("status", {}).get("installPlanRef", {}) or {}
        ).get("name")
        if not installplan_name:
            return

        installplan = self._oc_json(
            "get",
            "installplan",
            installplan_name,
            "-n",
            namespace,
            retry=True,
            description=f"reading installplan/{installplan_name}",
        )
        csv_names = installplan.get("spec", {}).get("clusterServiceVersionNames", [])
        for csv_name in csv_names:
            if expected_csv_name is not None:
                if str(csv_name) != expected_csv_name:
                    self._print_olm_diagnostics(
                        subscription_name=subscription_name,
                        namespace=namespace,
                        catalog_source=catalog_source,
                    )
                    raise CommandError(
                        f"refusing to auto-approve InstallPlan {installplan_name}: expected {expected_csv_name}, got {csv_name}"
                    )
                continue
            if not str(csv_name).startswith(csv_prefix):
                self._print_olm_diagnostics(
                    subscription_name=subscription_name,
                    namespace=namespace,
                    catalog_source=catalog_source,
                )
                raise CommandError(
                    f"refusing to auto-approve InstallPlan {installplan_name}: unexpected CSV {csv_name}"
                )

        step(f"Approving pending InstallPlan {installplan_name}")
        self._oc(
            "patch",
            "installplan",
            installplan_name,
            "-n",
            namespace,
            "--type",
            "merge",
            "-p",
            '{"spec":{"approved":true}}',
            retry=True,
            description=f"approving installplan/{installplan_name}",
            echo_output=True,
        )

    def _wait_for_csv_succeeded(
        self,
        *,
        subscription_name: str,
        namespace: str,
        csv_name: str,
        timeout_seconds: int,
        csv_prefix: str,
        catalog_source: str,
        expected_csv_name: str | None = None,
    ) -> None:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            self._approve_pending_installplan(
                subscription_name=subscription_name,
                namespace=namespace,
                csv_prefix=csv_prefix,
                catalog_source=catalog_source,
                expected_csv_name=expected_csv_name,
            )
            try:
                csv = self._oc_json(
                    "get",
                    "csv",
                    csv_name,
                    "-n",
                    namespace,
                    retry=True,
                    description=f"reading csv/{csv_name}",
                )
            except CommandError:
                time.sleep(5)
                continue
            phase = csv.get("status", {}).get("phase", "")
            if phase == "Succeeded":
                return
            time.sleep(5)

        self._print_olm_diagnostics(
            subscription_name=subscription_name,
            namespace=namespace,
            catalog_source=catalog_source,
        )
        raise CommandError(f"timed out waiting for CSV {csv_name} to reach Succeeded")

    def install_accelerator_prerequisites(self) -> None:
        step("Installing accelerator prerequisites for GPU inference")
        self.install_nfd_operator_and_instance()
        self.install_gpu_operator_and_cluster_policy()

    def nfd_ready(self) -> bool:
        return self._resource_exists(
            "get",
            "nodefeaturediscovery.nfd.openshift.io",
            "nfd-instance",
            "-n",
            self.nfd_namespace,
        )

    def gpu_operator_ready(self) -> bool:
        if not self._resource_exists(
            "get", "clusterpolicy.nvidia.com", "gpu-cluster-policy"
        ):
            return False
        cluster_policy = self._oc_json(
            "get",
            "clusterpolicy.nvidia.com",
            "gpu-cluster-policy",
            retry=True,
            description="reading ClusterPolicy/gpu-cluster-policy",
        )
        status = str(cluster_policy.get("status", {}).get("state", ""))
        return status.lower() == "ready"

    def install_nfd_operator_and_instance(self) -> None:
        if self.nfd_ready():
            success("Node Feature Discovery instance already present")
            return
        self._install_operator_from_package(
            package_name=self.nfd_package_name,
            namespace=self.nfd_namespace,
            subscription_name="nfd",
            operatorgroup_name="nfd",
            asset_path="operators/nfd/operator.yaml",
        )
        self._wait_for_resource(
            resource="crd/nodefeaturediscoveries.nfd.openshift.io",
            namespace=None,
            timeout_seconds=600,
            label="CRD nodefeaturediscoveries.nfd.openshift.io",
        )
        step("Applying the Node Feature Discovery instance")
        self._apply_asset_documents(
            "operators/nfd/instance.yaml",
            namespace=None,
            description="applying NodeFeatureDiscovery instance",
            variables=self._base_asset_variables(),
        )
        success("Node Feature Discovery operator and instance are present")

    def install_gpu_operator_and_cluster_policy(self) -> None:
        if self.gpu_operator_ready():
            success("NVIDIA GPU ClusterPolicy already present and ready")
            return
        self._install_operator_from_package(
            package_name=self.gpu_operator_package_name,
            namespace=self.gpu_operator_namespace,
            subscription_name="gpu-operator-certified",
            operatorgroup_name="gpu-operator-certified",
            asset_path="operators/gpu/operator.yaml",
        )
        self._wait_for_resource(
            resource="crd/clusterpolicies.nvidia.com",
            namespace=None,
            timeout_seconds=600,
            label="CRD clusterpolicies.nvidia.com",
        )
        step("Applying the NVIDIA GPU ClusterPolicy")
        self._apply_asset_documents(
            "operators/gpu/cluster-policy.yaml",
            namespace=None,
            description="applying NVIDIA GPU ClusterPolicy",
            variables=self._base_asset_variables(),
        )
        success("NVIDIA GPU Operator and ClusterPolicy are present")

    def install_tekton_if_needed(self) -> None:
        if self.tekton_ready():
            success("Tekton CRDs already present")
            return

        step(
            f"Installing OpenShift Pipelines operator in {self.pipelines_operator_namespace}"
        )
        self._apply_documents(
            self._render_asset_documents(
                "operators/tekton/subscription.yaml",
                {"TEKTON_CHANNEL": self.options.tekton_channel},
            ),
            namespace=None,
            description="applying OpenShift Pipelines subscription",
        )

        step("Waiting for the Tekton subscription to resolve")
        tekton_csv = self._wait_for_subscription_current_csv(
            subscription_name="openshift-pipelines-operator",
            namespace=self.pipelines_operator_namespace,
            timeout_seconds=600,
        )
        step(f"Waiting for CSV {tekton_csv} to succeed")
        self._wait_for_csv_succeeded(
            subscription_name="openshift-pipelines-operator",
            namespace=self.pipelines_operator_namespace,
            csv_name=tekton_csv,
            timeout_seconds=600,
            csv_prefix="openshift-pipelines-operator-rh.",
            catalog_source="redhat-operators",
        )

        step("Waiting for Tekton CRDs")
        self._wait_for_resource(
            resource="crd/tasks.tekton.dev",
            namespace=None,
            timeout_seconds=600,
            label="CRD tasks.tekton.dev",
        )
        self._wait_for_resource(
            resource="crd/pipelines.tekton.dev",
            namespace=None,
            timeout_seconds=600,
            label="CRD pipelines.tekton.dev",
        )
        self._wait_for_resource(
            resource="crd/pipelineruns.tekton.dev",
            namespace=None,
            timeout_seconds=600,
            label="CRD pipelineruns.tekton.dev",
        )

        step("Waiting for Tekton service accounts")
        for sa_name in (
            "serviceaccount/tekton-pipelines-controller",
            "serviceaccount/tekton-events-controller",
            "serviceaccount/tekton-pipelines-webhook",
        ):
            self._wait_for_resource(
                resource=sa_name,
                namespace=self.pipelines_runtime_namespace,
                timeout_seconds=600,
                label=f"{sa_name} in namespace {self.pipelines_runtime_namespace}",
            )

        self.configure_tekton_scc()

        step("Waiting for Tekton controllers")
        for deployment in (
            "deployment/tekton-pipelines-controller",
            "deployment/tekton-pipelines-webhook",
        ):
            self._wait_for_resource(
                resource=deployment,
                namespace=self.pipelines_runtime_namespace,
                timeout_seconds=600,
                label=f"{deployment} in namespace {self.pipelines_runtime_namespace}",
            )
            self._oc(
                "wait",
                "--for=condition=available",
                "--timeout=10m",
                deployment,
                "-n",
                self.pipelines_runtime_namespace,
                retry=True,
                description=f"waiting for {deployment} in {self.pipelines_runtime_namespace}",
            )

    def install_kueue_if_needed(self) -> None:
        if self.kueue_ready():
            success("Kueue CRDs and controller are already present")
            return

        self.ensure_namespace(self.kueue_namespace)
        if not self._kueue_crds_present():
            step(f"Installing Kueue in {self.kueue_namespace}")
            self._oc(
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
            self._wait_for_resource(
                resource=crd_name,
                namespace=None,
                timeout_seconds=600,
                label=crd_name,
            )

        step("Waiting for the Kueue controller manager")
        self._wait_for_resource(
            resource="deployment/kueue-controller-manager",
            namespace=self.kueue_namespace,
            timeout_seconds=600,
            label=f"deployment/kueue-controller-manager in namespace {self.kueue_namespace}",
        )
        self._oc(
            "wait",
            "--for=condition=available",
            "--timeout=10m",
            "deployment/kueue-controller-manager",
            "-n",
            self.kueue_namespace,
            retry=True,
            description="waiting for deployment/kueue-controller-manager",
        )

    def apply_kueue_support_resources(self) -> None:
        step("Applying BenchFlow Kueue support resources")
        self._apply_asset_documents(
            "operators/kueue/controller.yaml",
            namespace=None,
            description="applying BenchFlow Kueue support resources",
            variables=self._base_asset_variables(),
        )
        self.wait_for_kueue_support_ready(timeout_seconds=600)

    def wait_for_kueue_support_ready(self, timeout_seconds: int) -> None:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if not self._resource_exists(
                "get",
                "deployment",
                REMOTE_CAPACITY_CONTROLLER_DEPLOYMENT,
                "-n",
                self.options.namespace,
            ):
                time.sleep(5)
                continue
            deployment = self._oc_json(
                "get",
                "deployment",
                REMOTE_CAPACITY_CONTROLLER_DEPLOYMENT,
                "-n",
                self.options.namespace,
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
        self, *, cluster_name: str, gpu_capacity: int
    ) -> None:
        ensure_cluster_queue_resources(
            namespace=self.options.namespace,
            cluster_name=cluster_name,
            gpu_capacity=gpu_capacity,
        )

    def configure_tekton_scc(self) -> None:
        if not self._resource_exists(
            "get", "namespace", self.pipelines_runtime_namespace
        ):
            detail(
                f"Skipping Tekton SCC configuration because {self.pipelines_runtime_namespace} does not exist yet"
            )
            return

        step("Configuring Tekton SCCs")
        for service_account in (
            "tekton-pipelines-controller",
            "tekton-events-controller",
            "tekton-pipelines-webhook",
        ):
            result = self._oc(
                "adm",
                "policy",
                "add-scc-to-user",
                "privileged",
                "-z",
                service_account,
                "-n",
                self.pipelines_runtime_namespace,
                retry=True,
                description=f"granting privileged SCC to {service_account}",
                check=False,
            )
            if result.returncode != 0:
                warning(f"Could not grant privileged SCC to {service_account}")
                if result.stderr:
                    emit(
                        result.stderr,
                        end="" if result.stderr.endswith("\n") else "\n",
                        stderr=True,
                    )
            else:
                success(f"Granted privileged SCC to {service_account}")

    def install_grafana_if_needed(self) -> None:
        if not self.options.install_grafana:
            detail(
                "Skipping Grafana install because install_grafana=false for this bootstrap run"
            )
            return
        success(
            f"Grafana will be installed directly in namespace {self.grafana_namespace}"
        )

    def install_real_secrets(self) -> None:
        secrets_dir = self.repo_root / "config" / "cluster" / "secrets"
        found = False
        for secret_file in sorted(secrets_dir.glob("*.yaml")):
            if secret_file.name.endswith(".example.yaml"):
                continue
            found = True
            step(f"Applying secret {secret_file.name}")
            self._oc(
                "apply",
                "-n",
                self.options.namespace,
                "-f",
                str(secret_file),
                retry=True,
                description=f"applying secret {secret_file.name}",
                echo_output=True,
            )
        if not found:
            detail("No non-example secrets found under config/cluster/secrets")

    def apply_manifest_tree(self, root_dir: Path, label: str) -> None:
        step(f"Applying {label}")
        for manifest in sorted(root_dir.rglob("*.yaml")):
            detail(str(manifest.relative_to(self.repo_root)))
            self._oc(
                "apply",
                "-n",
                self.options.namespace,
                "-f",
                str(manifest),
                retry=True,
                description=f"applying {manifest.relative_to(self.repo_root)}",
                echo_output=True,
            )

    def apply_workspace_pvcs(self) -> None:
        variables = {
            "MODELS_STORAGE_ACCESS_MODE": self.options.models_storage_access_mode,
            "MODELS_STORAGE_CLASS": self.options.models_storage_class,
            "MODELS_STORAGE_SIZE": self.options.models_storage_size,
            "RESULTS_STORAGE_CLASS": self.options.results_storage_class,
            "RESULTS_STORAGE_SIZE": self.options.results_storage_size,
        }
        documents = self._render_asset_documents("workspaces/pvcs.yaml", variables)
        selected: list[dict[str, Any]] = []
        for document in documents:
            name = str(document.get("metadata", {}).get("name") or "")
            if name == "models-storage" and self.options.install_models_storage:
                selected.append(document)
            elif name == "benchmark-results" and self.options.install_results_storage:
                selected.append(document)
        if not selected:
            detail("Skipping workspace PVCs for this bootstrap mode")
            return
        step("Applying workspace PVCs")
        self._apply_documents(
            selected,
            namespace=self.options.namespace,
            description="applying workspace PVCs",
        )

    def apply_cluster_monitoring_rbac(self) -> None:
        step("Applying cluster monitoring RBAC")
        if not self._resource_exists("get", "clusterrole", "cluster-monitoring-view"):
            raise CommandError(
                "required ClusterRole not found: cluster-monitoring-view. "
                "This BenchFlow MVP expects OpenShift cluster monitoring to be available."
            )

        self._apply_asset_documents(
            "rbac/runner-cluster-monitoring-view.yaml",
            namespace=None,
            description="applying cluster monitoring RBAC",
            variables=self._base_asset_variables(),
        )

    def apply_runner_rbac(self) -> None:
        step("Applying runner RBAC")
        self._apply_asset_documents(
            "rbac/runner-namespaced.yaml",
            namespace=self.options.namespace,
            description="applying runner RBAC",
            variables=self._base_asset_variables(),
        )
        if self._resource_exists("get", "namespace", "istio-system"):
            self._apply_asset_documents(
                "rbac/runner-istio-system.yaml",
                namespace=None,
                description="applying istio-system runner RBAC",
                variables=self._base_asset_variables(),
            )
        else:
            detail(
                "Skipping istio-system runner RBAC because namespace istio-system does not exist"
            )
        self._apply_asset_documents(
            "rbac/runner-cluster.yaml",
            namespace=None,
            description="applying runner cluster RBAC",
            variables=self._base_asset_variables(),
        )

    def apply_namespaced_resources(self) -> None:
        step("Applying namespace RBAC")
        self._apply_asset_documents(
            "rbac/runner-base.yaml",
            namespace=self.options.namespace,
            description="applying namespace service account",
            variables=self._base_asset_variables(),
        )
        self.apply_runner_rbac()
        self.apply_cluster_monitoring_rbac()
        self.apply_workspace_pvcs()
        if self.options.install_tekton:
            self.apply_manifest_tree(
                self.repo_root / "tekton" / "tasks", "Tekton tasks"
            )
            self.apply_manifest_tree(
                self.repo_root / "tekton" / "pipelines", "Tekton pipelines"
            )

    def discover_grafana_route_host(self) -> str | None:
        if not self._resource_exists("get", "route", "-n", self.grafana_namespace):
            return None
        routes = self._oc_json(
            "get",
            "route",
            "-n",
            self.grafana_namespace,
            retry=True,
            description="reading routes",
        )
        for item in routes.get("items", []):
            name = item.get("metadata", {}).get("name", "")
            host = item.get("spec", {}).get("host", "")
            if "grafana" in name and host:
                return str(host)
        return None

    def wait_for_grafana_route(self, timeout_seconds: int) -> None:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            host = self.discover_grafana_route_host()
            if host:
                return
            time.sleep(5)
        raise CommandError("timed out waiting for the Grafana route")

    def wait_for_grafana_ready(self, timeout_seconds: int) -> None:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            if not self._resource_exists(
                "get", "deployment", "grafana", "-n", self.grafana_namespace
            ):
                time.sleep(5)
                continue
            deployment = self._oc_json(
                "get",
                "deployment",
                "grafana",
                "-n",
                self.grafana_namespace,
                retry=True,
                description="reading deployment/grafana",
            )
            status = deployment.get("status", {}) or {}
            ready = int(status.get("readyReplicas", 0) or 0)
            desired = int(status.get("replicas", 0) or 0)
            if desired > 0 and ready >= desired:
                return
            time.sleep(5)
        raise CommandError("timed out waiting for deployment/grafana to become ready")

    def apply_grafana_stack(self) -> None:
        if not self.options.install_grafana:
            return

        step("Applying Grafana monitoring RBAC")
        self._apply_asset_documents(
            "operators/grafana/rbac.yaml",
            namespace=None,
            description="applying Grafana service account resources",
            variables=self._base_asset_variables(),
        )
        self._wait_for_secret_key(
            name=self.grafana_datasource_token_secret,
            key="token",
            namespace=self.grafana_namespace,
            timeout_seconds=300,
        )
        if not self._resource_exists(
            "get",
            "secret",
            self.grafana_admin_secret_name,
            "-n",
            self.grafana_namespace,
        ):
            self._apply_asset_documents(
                "operators/grafana/admin-secret.yaml",
                namespace=self.grafana_namespace,
                description="applying Grafana admin credentials",
                variables={
                    **self._base_asset_variables(),
                    "GRAFANA_ADMIN_USER": "admin",
                    "GRAFANA_ADMIN_PASSWORD": secrets.token_urlsafe(24),
                },
            )

        step("Applying Grafana deployment, route, datasource, and dashboards")
        grafana_stack_variables = {
            **self._base_asset_variables(),
            "GRAFANA_DATASOURCES_YAML": self._asset_text(
                "operators/grafana/datasources.yaml"
            ),
            "GRAFANA_DASHBOARDS_YAML": self._asset_text(
                "operators/grafana/dashboard-providers.yaml"
            ),
            "GRAFANA_LIVE_DASHBOARD_JSON": self._asset_text(
                "operators/grafana/benchflow-live-dashboard.json"
            ),
        }
        self._apply_documents(
            [
                *self._render_asset_documents(
                    "operators/grafana/provisioning-configmap.yaml",
                    grafana_stack_variables,
                ),
                *self._render_asset_documents(
                    "operators/grafana/dashboards-configmap.yaml",
                    grafana_stack_variables,
                ),
                *self._render_asset_documents(
                    "operators/grafana/deployment.yaml",
                    grafana_stack_variables,
                ),
                *self._render_asset_documents(
                    "operators/grafana/service.yaml",
                    grafana_stack_variables,
                ),
                *self._render_asset_documents(
                    "operators/grafana/route.yaml",
                    grafana_stack_variables,
                ),
            ],
            namespace=self.grafana_namespace,
            description="applying Grafana stack",
        )
        step("Waiting for Grafana route")
        self.wait_for_grafana_route(timeout_seconds=600)
        step("Waiting for Grafana to become ready")
        self.wait_for_grafana_ready(timeout_seconds=600)


def run_bootstrap(repo_root: Path, options: BootstrapOptions) -> int:
    installer = Installer(repo_root, options)
    return installer.run()


def reconcile_management_cluster_queue(
    repo_root: Path,
    *,
    namespace: str,
    cluster_name: str,
    gpu_capacity: int,
    benchflow_image: str = DEFAULT_CONTROLLER_IMAGE,
) -> None:
    installer = Installer(
        repo_root,
        BootstrapOptions(
            namespace=namespace,
            install_grafana=False,
            install_tekton=False,
            install_kueue=True,
            install_accelerator_prerequisites=False,
            install_models_storage=False,
            install_results_storage=False,
            benchflow_image=benchflow_image,
        ),
    )
    installer.ensure_cluster_access()
    installer.ensure_namespace(namespace)
    installer.install_kueue_if_needed()
    installer.apply_kueue_support_resources()
    installer.register_kueue_cluster_queue(
        cluster_name=cluster_name,
        gpu_capacity=gpu_capacity,
    )


__all__ = [
    "BootstrapOptions",
    "Installer",
    "discover_repo_root",
    "reconcile_management_cluster_queue",
    "run_bootstrap",
]
