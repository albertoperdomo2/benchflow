from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .bootstrap_grafana import (
    apply_grafana_stack as apply_bootstrap_grafana_stack,
    discover_grafana_route_host as discover_bootstrap_grafana_route_host,
    install_grafana_if_needed as maybe_install_bootstrap_grafana,
    wait_for_grafana_ready as wait_for_bootstrap_grafana_ready,
    wait_for_grafana_route as wait_for_bootstrap_grafana_route,
)
from .bootstrap_kueue import (
    apply_kueue_support_resources as apply_bootstrap_kueue_support_resources,
    install_kueue_if_needed as install_bootstrap_kueue_if_needed,
    kueue_crds_present as bootstrap_kueue_crds_present,
    kueue_ready as bootstrap_kueue_ready,
    register_kueue_cluster_queue as register_bootstrap_kueue_cluster_queue,
    wait_for_kueue_support_ready as wait_for_bootstrap_kueue_support_ready,
)
from .bootstrap_operators import (
    approve_pending_installplan as approve_bootstrap_pending_installplan,
    catalog_source_for_package as bootstrap_catalog_source_for_package,
    configure_tekton_scc as configure_bootstrap_tekton_scc,
    default_channel_for_package as bootstrap_default_channel_for_package,
    get_packagemanifest as bootstrap_get_packagemanifest,
    get_subscription as bootstrap_get_subscription,
    gpu_operator_ready as bootstrap_gpu_operator_ready,
    install_accelerator_prerequisites as install_bootstrap_accelerator_prerequisites,
    install_gpu_operator_and_cluster_policy as install_bootstrap_gpu_operator_and_cluster_policy,
    install_nfd_operator_and_instance as install_bootstrap_nfd_operator_and_instance,
    install_operator_from_package as install_bootstrap_operator_from_package,
    install_tekton_if_needed as install_bootstrap_tekton_if_needed,
    nfd_ready as bootstrap_nfd_ready,
    operatorgroups_in_namespace as bootstrap_operatorgroups_in_namespace,
    print_olm_diagnostics as print_bootstrap_olm_diagnostics,
    reuse_or_create_operatorgroup as reuse_bootstrap_or_create_operatorgroup,
    wait_for_csv_succeeded as wait_for_bootstrap_csv_succeeded,
    wait_for_subscription_current_csv as wait_for_bootstrap_subscription_current_csv,
)
from .assets import asset_text, render_yaml_documents
from .bootstrap_presentation import (
    print_bootstrap_intro,
    print_bootstrap_summary,
)
from .bootstrap_resources import (
    apply_cluster_monitoring_rbac as apply_bootstrap_cluster_monitoring_rbac,
    apply_manifest_tree as apply_bootstrap_manifest_tree,
    apply_namespaced_resources as apply_bootstrap_namespaced_resources,
    apply_runner_rbac as apply_bootstrap_runner_rbac,
    apply_workspace_pvcs as apply_bootstrap_workspace_pvcs,
    install_real_secrets as install_bootstrap_real_secrets,
)
from .cluster import (
    CommandError,
    TARGET_KUBECONFIG_HOST_ALIASES_ANNOTATION,
    discover_repo_root,
    require_command,
)
from .kueue import DEFAULT_CONTROLLER_IMAGE, KUEUE_NAMESPACE
from .ui import detail, emit, step, ui_scope, warning


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
    cluster_name: str | None = None


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

    @property
    def ui_label(self) -> str:
        if self.options.single_cluster:
            return "[single-cluster]"
        if self.options.target_kubeconfig:
            target_name = str(self.options.cluster_name or "target-cluster").strip()
            return f"[target-cluster:{target_name}]"
        return "[management-cluster]"

    def run(self, *, emit_summary: bool = True) -> int:
        with ui_scope(self.ui_label):
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
            if emit_summary:
                self.print_summary()
        return 0

    def print_intro(self) -> None:
        print_bootstrap_intro(
            bootstrap_mode=self.bootstrap_mode,
            options=self.options,
            grafana_namespace=self.grafana_namespace,
            kueue_namespace=self.kueue_namespace,
            nfd_namespace=self.nfd_namespace,
            gpu_operator_namespace=self.gpu_operator_namespace,
            default_storage_class=(
                self.default_storage_class()
                if self.options.install_models_storage
                and self.options.models_storage_class is None
                else None
            ),
        )

    def print_summary(self) -> None:
        grafana_host: str | None = None
        try:
            grafana_host = self.discover_grafana_route_host()
        except CommandError as exc:
            warning(f"Could not query the Grafana route for the final summary: {exc}")
        print_bootstrap_summary(
            bootstrap_mode=self.bootstrap_mode,
            options=self.options,
            grafana_namespace=self.grafana_namespace,
            kueue_namespace=self.kueue_namespace,
            nfd_namespace=self.nfd_namespace,
            gpu_operator_namespace=self.gpu_operator_namespace,
            grafana_host=grafana_host,
            grafana_admin_secret_name=self.grafana_admin_secret_name,
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
        return bootstrap_kueue_crds_present(self)

    def kueue_ready(self) -> bool:
        return bootstrap_kueue_ready(self)

    def _print_olm_diagnostics(
        self, *, subscription_name: str, namespace: str, catalog_source: str
    ) -> None:
        print_bootstrap_olm_diagnostics(
            self,
            subscription_name=subscription_name,
            namespace=namespace,
            catalog_source=catalog_source,
        )

    def _wait_for_subscription_current_csv(
        self, *, subscription_name: str, namespace: str, timeout_seconds: int
    ) -> str:
        return wait_for_bootstrap_subscription_current_csv(
            self,
            subscription_name=subscription_name,
            namespace=namespace,
            timeout_seconds=timeout_seconds,
        )

    def _get_subscription(
        self, *, subscription_name: str, namespace: str, description: str | None = None
    ) -> dict[str, Any]:
        return bootstrap_get_subscription(
            self,
            subscription_name=subscription_name,
            namespace=namespace,
            description=description,
        )

    def _get_packagemanifest(self, package_name: str) -> dict[str, Any]:
        return bootstrap_get_packagemanifest(self, package_name)

    def _default_channel_for_package(self, package_name: str) -> str:
        return bootstrap_default_channel_for_package(self, package_name)

    def _catalog_source_for_package(self, package_name: str) -> tuple[str, str]:
        return bootstrap_catalog_source_for_package(self, package_name)

    def _operatorgroups_in_namespace(self, namespace: str) -> list[dict[str, Any]]:
        return bootstrap_operatorgroups_in_namespace(self, namespace)

    def _reuse_or_create_operatorgroup(
        self, *, namespace: str, operatorgroup_name: str
    ) -> bool:
        return reuse_bootstrap_or_create_operatorgroup(
            self,
            namespace=namespace,
            operatorgroup_name=operatorgroup_name,
        )

    def _install_operator_from_package(
        self,
        *,
        package_name: str,
        namespace: str,
        subscription_name: str,
        operatorgroup_name: str,
        asset_path: str,
    ) -> str:
        return install_bootstrap_operator_from_package(
            self,
            package_name=package_name,
            namespace=namespace,
            subscription_name=subscription_name,
            operatorgroup_name=operatorgroup_name,
            asset_path=asset_path,
        )

    def _approve_pending_installplan(
        self,
        *,
        subscription_name: str,
        namespace: str,
        csv_prefix: str,
        catalog_source: str,
        expected_csv_name: str | None = None,
    ) -> None:
        approve_bootstrap_pending_installplan(
            self,
            subscription_name=subscription_name,
            namespace=namespace,
            csv_prefix=csv_prefix,
            catalog_source=catalog_source,
            expected_csv_name=expected_csv_name,
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
        wait_for_bootstrap_csv_succeeded(
            self,
            subscription_name=subscription_name,
            namespace=namespace,
            csv_name=csv_name,
            timeout_seconds=timeout_seconds,
            csv_prefix=csv_prefix,
            catalog_source=catalog_source,
            expected_csv_name=expected_csv_name,
        )

    def install_accelerator_prerequisites(self) -> None:
        install_bootstrap_accelerator_prerequisites(self)

    def nfd_ready(self) -> bool:
        return bootstrap_nfd_ready(self)

    def gpu_operator_ready(self) -> bool:
        return bootstrap_gpu_operator_ready(self)

    def install_nfd_operator_and_instance(self) -> None:
        install_bootstrap_nfd_operator_and_instance(self)

    def install_gpu_operator_and_cluster_policy(self) -> None:
        install_bootstrap_gpu_operator_and_cluster_policy(self)

    def install_tekton_if_needed(self) -> None:
        install_bootstrap_tekton_if_needed(self)

    def install_kueue_if_needed(self) -> None:
        install_bootstrap_kueue_if_needed(self)

    def apply_kueue_support_resources(self) -> None:
        apply_bootstrap_kueue_support_resources(self)

    def wait_for_kueue_support_ready(self, timeout_seconds: int) -> None:
        wait_for_bootstrap_kueue_support_ready(self, timeout_seconds=timeout_seconds)

    def register_kueue_cluster_queue(
        self, *, cluster_name: str, gpu_capacity: int
    ) -> None:
        register_bootstrap_kueue_cluster_queue(
            self,
            cluster_name=cluster_name,
            gpu_capacity=gpu_capacity,
        )

    def configure_tekton_scc(self) -> None:
        configure_bootstrap_tekton_scc(self)

    def install_grafana_if_needed(self) -> None:
        maybe_install_bootstrap_grafana(self)

    def install_real_secrets(self) -> None:
        install_bootstrap_real_secrets(self)

    def apply_manifest_tree(self, root_dir: Path, label: str) -> None:
        apply_bootstrap_manifest_tree(self, root_dir, label)

    def apply_workspace_pvcs(self) -> None:
        apply_bootstrap_workspace_pvcs(self)

    def apply_cluster_monitoring_rbac(self) -> None:
        apply_bootstrap_cluster_monitoring_rbac(self)

    def apply_runner_rbac(self) -> None:
        apply_bootstrap_runner_rbac(self)

    def apply_namespaced_resources(self) -> None:
        apply_bootstrap_namespaced_resources(self)

    def discover_grafana_route_host(self) -> str | None:
        return discover_bootstrap_grafana_route_host(self)

    def wait_for_grafana_route(self, timeout_seconds: int) -> None:
        wait_for_bootstrap_grafana_route(self, timeout_seconds)

    def wait_for_grafana_ready(self, timeout_seconds: int) -> None:
        wait_for_bootstrap_grafana_ready(self, timeout_seconds)

    def apply_grafana_stack(self) -> None:
        apply_bootstrap_grafana_stack(self)


def run_bootstrap(
    repo_root: Path, options: BootstrapOptions, *, emit_summary: bool = True
) -> int:
    installer = Installer(repo_root, options)
    return installer.run(emit_summary=emit_summary)


def reconcile_management_cluster_queue(
    repo_root: Path,
    *,
    namespace: str,
    cluster_name: str,
    gpu_capacity: int,
    benchflow_image: str = DEFAULT_CONTROLLER_IMAGE,
    ui_label: str = "[management-cluster]",
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
    with ui_scope(ui_label):
        step(f"Registering BenchFlow Kueue capacity for {cluster_name}")
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
