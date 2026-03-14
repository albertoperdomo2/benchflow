from __future__ import annotations

import base64
import json
import secrets
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from .cluster import CommandError, discover_repo_root, require_command
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
class InstallOptions:
    namespace: str = "benchflow"
    install_tekton: bool = True
    install_grafana: bool = True
    tekton_channel: str = "latest"
    models_storage_access_mode: str = "ReadWriteOnce"
    models_storage_size: str = "250Gi"
    models_storage_class: str | None = None
    results_storage_size: str = "20Gi"
    results_storage_class: str | None = None


class Installer:
    pipelines_operator_namespace = "openshift-operators"
    pipelines_runtime_namespace = "openshift-pipelines"
    grafana_admin_secret_name = "grafana-admin-credentials"
    grafana_datasource_service_account = "benchflow-grafana"
    grafana_datasource_token_secret = "benchflow-grafana-datasource-token"

    def __init__(self, repo_root: Path, options: InstallOptions) -> None:
        self.repo_root = repo_root.resolve()
        self.options = options
        self._default_storage_class_name: str | None = None

    @property
    def grafana_namespace(self) -> str:
        return f"{self.options.namespace}-grafana"

    def run(self) -> int:
        self.ensure_cluster_access()
        self.ensure_storage_class(self.options.models_storage_class, "models-storage")
        self.ensure_storage_class(
            self.options.results_storage_class, "benchmark-results"
        )
        self.ensure_default_storage_class_if_needed(
            self.options.models_storage_class, "models-storage"
        )
        self.ensure_default_storage_class_if_needed(
            self.options.results_storage_class, "benchmark-results"
        )
        self.ensure_namespace(self.options.namespace)
        if self.options.install_grafana:
            self.ensure_namespace(self.grafana_namespace)

        self.print_intro()

        self.install_tekton_if_needed()
        self.configure_tekton_scc()
        self.install_grafana_if_needed()
        self.install_real_secrets()
        self.apply_namespaced_resources()
        self.apply_grafana_stack()
        self.print_summary()
        return 0

    def print_intro(self) -> None:
        options = self.options
        rule("BenchFlow Install")
        panel(
            "Configuration",
            (
                ("Namespace", options.namespace),
                ("Grafana namespace", self.grafana_namespace),
                ("Install Tekton if missing", str(options.install_tekton).lower()),
                ("Install Grafana if missing", str(options.install_grafana).lower()),
                ("OpenShift Pipelines channel", options.tekton_channel),
                (
                    "models-storage",
                    f"{options.models_storage_access_mode} {options.models_storage_size}"
                    f"{f' via {options.models_storage_class}' if options.models_storage_class else ''}",
                ),
                (
                    "benchmark-results",
                    f"ReadWriteOnce {options.results_storage_size}"
                    f"{f' via {options.results_storage_class}' if options.results_storage_class else ''}",
                ),
                (
                    "metrics access",
                    "cluster-monitoring-view -> benchflow-runner, benchflow-grafana",
                ),
            ),
        )
        if options.models_storage_class is None:
            detail(
                f"default StorageClass for models-storage: {self.default_storage_class()}"
            )
        if options.models_storage_access_mode == "ReadWriteOnce":
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
        rule("Install Complete")
        panel(
            "BenchFlow",
            (
                ("Namespace", options.namespace),
                ("Grafana namespace", self.grafana_namespace),
                ("Tekton install attempted", str(options.install_tekton).lower()),
                ("Grafana install attempted", str(options.install_grafana).lower()),
                ("OpenShift Pipelines channel", options.tekton_channel),
                (
                    "models-storage",
                    f"{options.models_storage_access_mode} {options.models_storage_size}"
                    f"{f' via {options.models_storage_class}' if options.models_storage_class else ''}",
                ),
                (
                    "benchmark-results",
                    f"ReadWriteOnce {options.results_storage_size}"
                    f"{f' via {options.results_storage_class}' if options.results_storage_class else ''}",
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
        detail(
            f"bflow experiment run experiments/smoke/qwen3-06b-llm-d-smoke.yaml --namespace {options.namespace}"
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

    def install_tekton_if_needed(self) -> None:
        if self.tekton_ready():
            success("Tekton CRDs already present")
            return

        if not self.options.install_tekton:
            raise CommandError(
                "Tekton is not installed and --skip-tekton-install was requested"
            )

        step(
            f"Installing OpenShift Pipelines operator in {self.pipelines_operator_namespace}"
        )
        subscription_path = (
            self.repo_root
            / "config"
            / "cluster"
            / "operators"
            / "openshift-pipelines-subscription.yaml"
        )
        subscription = yaml.safe_load(subscription_path.read_text(encoding="utf-8"))
        subscription["spec"]["channel"] = self.options.tekton_channel
        self._apply_documents(
            [subscription],
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
                "Skipping Grafana install because --skip-grafana-install was requested"
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
        step("Applying workspace PVCs")
        documents = [
            {
                "apiVersion": "v1",
                "kind": "PersistentVolumeClaim",
                "metadata": {
                    "name": "models-storage",
                    "labels": {
                        "app.kubernetes.io/name": "benchflow",
                        "benchflow.io/purpose": "model-cache",
                    },
                },
                "spec": {
                    "accessModes": [self.options.models_storage_access_mode],
                    "resources": {
                        "requests": {"storage": self.options.models_storage_size}
                    },
                },
            },
            {
                "apiVersion": "v1",
                "kind": "PersistentVolumeClaim",
                "metadata": {
                    "name": "benchmark-results",
                    "labels": {
                        "app.kubernetes.io/name": "benchflow",
                        "benchflow.io/purpose": "benchmark-results",
                    },
                },
                "spec": {
                    "accessModes": ["ReadWriteOnce"],
                    "resources": {
                        "requests": {"storage": self.options.results_storage_size}
                    },
                },
            },
        ]
        if self.options.models_storage_class:
            documents[0]["spec"]["storageClassName"] = self.options.models_storage_class
        if self.options.results_storage_class:
            documents[1]["spec"]["storageClassName"] = (
                self.options.results_storage_class
            )

        self._apply_documents(
            documents,
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

        documents = [
            {
                "apiVersion": "rbac.authorization.k8s.io/v1",
                "kind": "ClusterRoleBinding",
                "metadata": {
                    "name": f"benchflow-runner-cluster-monitoring-view-{self.options.namespace}",
                    "labels": {
                        "app.kubernetes.io/name": "benchflow",
                        "benchflow.io/purpose": "metrics",
                    },
                },
                "subjects": [
                    {
                        "kind": "ServiceAccount",
                        "name": "benchflow-runner",
                        "namespace": self.options.namespace,
                    }
                ],
                "roleRef": {
                    "apiGroup": "rbac.authorization.k8s.io",
                    "kind": "ClusterRole",
                    "name": "cluster-monitoring-view",
                },
            }
        ]
        self._apply_documents(
            documents, namespace=None, description="applying cluster monitoring RBAC"
        )

    def apply_runner_rbac(self) -> None:
        step("Applying runner RBAC")
        namespaced_documents = [
            {
                "apiVersion": "rbac.authorization.k8s.io/v1",
                "kind": "Role",
                "metadata": {
                    "name": "benchflow-runner-llmd-apis",
                    "labels": {
                        "app.kubernetes.io/name": "benchflow",
                        "benchflow.io/purpose": "runner-rbac",
                    },
                },
                "rules": [
                    {
                        "apiGroups": ["gateway.networking.k8s.io"],
                        "resources": ["*"],
                        "verbs": ["*"],
                    },
                    {
                        "apiGroups": ["inference.networking.x-k8s.io"],
                        "resources": ["*"],
                        "verbs": ["*"],
                    },
                    {
                        "apiGroups": ["llm-d.ai"],
                        "resources": ["*"],
                        "verbs": ["*"],
                    },
                    {
                        "apiGroups": ["monitoring.coreos.com"],
                        "resources": ["*"],
                        "verbs": ["*"],
                    },
                    {
                        "apiGroups": ["telemetry.istio.io"],
                        "resources": ["*"],
                        "verbs": ["*"],
                    },
                    {
                        "apiGroups": ["networking.istio.io"],
                        "resources": ["*"],
                        "verbs": ["*"],
                    },
                ],
            },
            {
                "apiVersion": "rbac.authorization.k8s.io/v1",
                "kind": "RoleBinding",
                "metadata": {
                    "name": "benchflow-runner-admin",
                    "labels": {
                        "app.kubernetes.io/name": "benchflow",
                        "benchflow.io/purpose": "runner-rbac",
                    },
                },
                "subjects": [
                    {
                        "kind": "ServiceAccount",
                        "name": "benchflow-runner",
                        "namespace": self.options.namespace,
                    }
                ],
                "roleRef": {
                    "apiGroup": "rbac.authorization.k8s.io",
                    "kind": "ClusterRole",
                    "name": "admin",
                },
            },
            {
                "apiVersion": "rbac.authorization.k8s.io/v1",
                "kind": "RoleBinding",
                "metadata": {
                    "name": "benchflow-runner-llmd-apis",
                    "labels": {
                        "app.kubernetes.io/name": "benchflow",
                        "benchflow.io/purpose": "runner-rbac",
                    },
                },
                "subjects": [
                    {
                        "kind": "ServiceAccount",
                        "name": "benchflow-runner",
                        "namespace": self.options.namespace,
                    }
                ],
                "roleRef": {
                    "apiGroup": "rbac.authorization.k8s.io",
                    "kind": "Role",
                    "name": "benchflow-runner-llmd-apis",
                },
            },
        ]
        self._apply_documents(
            namespaced_documents,
            namespace=self.options.namespace,
            description="applying runner RBAC",
        )

        cluster_documents = [
            {
                "apiVersion": "rbac.authorization.k8s.io/v1",
                "kind": "ClusterRole",
                "metadata": {
                    "name": f"benchflow-runner-llmd-cluster-rbac-{self.options.namespace}",
                    "labels": {
                        "app.kubernetes.io/name": "benchflow",
                        "benchflow.io/purpose": "runner-rbac",
                    },
                },
                "rules": [
                    {
                        "apiGroups": [""],
                        "resources": ["namespaces"],
                        "resourceNames": [self.options.namespace],
                        "verbs": ["get", "patch", "update"],
                    },
                    {
                        "apiGroups": [""],
                        "resources": ["namespaces"],
                        "verbs": ["create"],
                    },
                    {
                        "apiGroups": ["rbac.authorization.k8s.io"],
                        "resources": ["clusterroles", "clusterrolebindings"],
                        "verbs": ["*"],
                    },
                ],
            },
            {
                "apiVersion": "rbac.authorization.k8s.io/v1",
                "kind": "ClusterRoleBinding",
                "metadata": {
                    "name": f"benchflow-runner-llmd-cluster-rbac-{self.options.namespace}",
                    "labels": {
                        "app.kubernetes.io/name": "benchflow",
                        "benchflow.io/purpose": "runner-rbac",
                    },
                },
                "subjects": [
                    {
                        "kind": "ServiceAccount",
                        "name": "benchflow-runner",
                        "namespace": self.options.namespace,
                    }
                ],
                "roleRef": {
                    "apiGroup": "rbac.authorization.k8s.io",
                    "kind": "ClusterRole",
                    "name": f"benchflow-runner-llmd-cluster-rbac-{self.options.namespace}",
                },
            },
        ]
        self._apply_documents(
            cluster_documents,
            namespace=None,
            description="applying runner cluster RBAC",
        )

    def apply_namespaced_resources(self) -> None:
        step("Applying namespace RBAC")
        rbac_dir = self.repo_root / "config" / "cluster" / "rbac"
        self._oc(
            "apply",
            "-n",
            self.options.namespace,
            "-f",
            str(rbac_dir),
            retry=True,
            description="applying namespace RBAC",
            echo_output=True,
        )
        self.apply_runner_rbac()
        self.apply_cluster_monitoring_rbac()
        self.apply_workspace_pvcs()
        self.apply_manifest_tree(self.repo_root / "tekton" / "tasks", "Tekton tasks")
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
        self._apply_documents(
            [
                {
                    "apiVersion": "v1",
                    "kind": "ServiceAccount",
                    "metadata": {
                        "name": self.grafana_datasource_service_account,
                        "labels": {
                            "app.kubernetes.io/name": "benchflow",
                            "benchflow.io/component": "grafana",
                        },
                    },
                },
                {
                    "apiVersion": "rbac.authorization.k8s.io/v1",
                    "kind": "Role",
                    "metadata": {
                        "name": "benchflow-grafana-route-reader",
                        "labels": {
                            "app.kubernetes.io/name": "benchflow",
                            "benchflow.io/component": "grafana",
                        },
                    },
                    "rules": [
                        {
                            "apiGroups": ["route.openshift.io"],
                            "resources": ["routes"],
                            "verbs": ["get", "list"],
                        }
                    ],
                },
                {
                    "apiVersion": "rbac.authorization.k8s.io/v1",
                    "kind": "RoleBinding",
                    "metadata": {
                        "name": "benchflow-grafana-route-reader",
                        "labels": {
                            "app.kubernetes.io/name": "benchflow",
                            "benchflow.io/component": "grafana",
                        },
                    },
                    "subjects": [
                        {
                            "kind": "ServiceAccount",
                            "name": "benchflow-runner",
                            "namespace": self.options.namespace,
                        }
                    ],
                    "roleRef": {
                        "apiGroup": "rbac.authorization.k8s.io",
                        "kind": "Role",
                        "name": "benchflow-grafana-route-reader",
                    },
                },
                {
                    "apiVersion": "v1",
                    "kind": "Secret",
                    "metadata": {
                        "name": self.grafana_datasource_token_secret,
                        "annotations": {
                            "kubernetes.io/service-account.name": self.grafana_datasource_service_account
                        },
                        "labels": {
                            "app.kubernetes.io/name": "benchflow",
                            "benchflow.io/component": "grafana",
                        },
                    },
                    "type": "kubernetes.io/service-account-token",
                },
            ],
            namespace=self.grafana_namespace,
            description="applying Grafana service account resources",
        )
        self._apply_documents(
            [
                {
                    "apiVersion": "rbac.authorization.k8s.io/v1",
                    "kind": "ClusterRoleBinding",
                    "metadata": {
                        "name": f"benchflow-grafana-cluster-monitoring-view-{self.options.namespace}",
                        "labels": {
                            "app.kubernetes.io/name": "benchflow",
                            "benchflow.io/component": "grafana",
                        },
                    },
                    "subjects": [
                        {
                            "kind": "ServiceAccount",
                            "name": self.grafana_datasource_service_account,
                            "namespace": self.grafana_namespace,
                        }
                    ],
                    "roleRef": {
                        "apiGroup": "rbac.authorization.k8s.io",
                        "kind": "ClusterRole",
                        "name": "cluster-monitoring-view",
                    },
                }
            ],
            namespace=None,
            description="applying Grafana cluster monitoring RBAC",
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
            self._apply_documents(
                [
                    {
                        "apiVersion": "v1",
                        "kind": "Secret",
                        "metadata": {
                            "name": self.grafana_admin_secret_name,
                            "labels": {
                                "app.kubernetes.io/name": "benchflow",
                                "benchflow.io/component": "grafana",
                            },
                        },
                        "type": "Opaque",
                        "stringData": {
                            "admin-user": "admin",
                            "admin-password": secrets.token_urlsafe(24),
                        },
                    }
                ],
                namespace=self.grafana_namespace,
                description="applying Grafana admin credentials",
            )

        dashboard_live_json = (
            self.repo_root
            / "config"
            / "monitoring"
            / "grafana-dashboard-benchflow.json"
        ).read_text(encoding="utf-8")
        dashboard_archive_json = (
            self.repo_root
            / "config"
            / "monitoring"
            / "grafana-dashboard-benchflow-archive.json"
        ).read_text(encoding="utf-8")
        archive_base_url = ""
        if self._resource_exists(
            "get", "secret", "mlflow-s3-secret", "-n", self.options.namespace
        ):
            mlflow_secret = self._oc_json(
                "get",
                "secret",
                "mlflow-s3-secret",
                "-n",
                self.options.namespace,
                retry=True,
                description="reading secret/mlflow-s3-secret",
            )
            secret_data = mlflow_secret.get("data", {}) or {}
            encoded = str(secret_data.get("archive-base-url", "") or "")
            if encoded:
                archive_base_url = base64.b64decode(encoded).decode("utf-8")
        if not archive_base_url:
            warning(
                "mlflow-s3-secret does not define archive-base-url; "
                "the archive dashboard will need that HTTP(S) MLflow artifact root "
                "configured before Infinity queries can succeed"
            )
            archive_base_url = "https://example.invalid/mlflow"
        archive_allowed_host = (
            f"{urlsplit(archive_base_url).scheme}://{urlsplit(archive_base_url).netloc}"
            if urlsplit(archive_base_url).netloc
            else archive_base_url
        )

        datasources_yaml = yaml.safe_dump(
            {
                "apiVersion": 1,
                "datasources": [
                    {
                        "name": "openshift-monitoring",
                        "uid": "openshift-monitoring",
                        "type": "prometheus",
                        "access": "proxy",
                        "url": "https://thanos-querier.openshift-monitoring.svc:9091",
                        "isDefault": True,
                        "jsonData": {
                            "timeInterval": "15s",
                            "tlsSkipVerify": True,
                            "httpHeaderName1": "Authorization",
                        },
                        "secureJsonData": {
                            "httpHeaderValue1": "Bearer ${GRAFANA_THANOS_TOKEN}",
                        },
                    },
                    {
                        "name": "benchflow-archive",
                        "uid": "benchflow-archive",
                        "type": "yesoreyeram-infinity-datasource",
                        "access": "proxy",
                        "url": "${BENCHFLOW_ARCHIVE_BASE_URL}",
                        "jsonData": {
                            "allowedHosts": [archive_allowed_host],
                        },
                    },
                ],
            },
            sort_keys=False,
        )
        dashboards_yaml = yaml.safe_dump(
            {
                "apiVersion": 1,
                "providers": [
                    {
                        "name": "benchflow",
                        "orgId": 1,
                        "folder": "BenchFlow",
                        "type": "file",
                        "disableDeletion": False,
                        "editable": True,
                        "options": {"path": "/var/lib/grafana/dashboards"},
                    }
                ],
            },
            sort_keys=False,
        )
        documents = [
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {
                    "name": "grafana-provisioning",
                    "labels": {
                        "app.kubernetes.io/name": "benchflow",
                        "benchflow.io/component": "grafana",
                    },
                },
                "data": {
                    "datasources.yaml": datasources_yaml,
                    "dashboards.yaml": dashboards_yaml,
                },
            },
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {
                    "name": "grafana-dashboards",
                    "labels": {
                        "app.kubernetes.io/name": "benchflow",
                        "benchflow.io/component": "grafana",
                    },
                },
                "data": {
                    "benchflow-live.json": dashboard_live_json,
                    "benchflow-archive.json": dashboard_archive_json,
                },
            },
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": "grafana",
                    "labels": {
                        "app.kubernetes.io/name": "benchflow",
                        "benchflow.io/component": "grafana",
                    },
                },
                "spec": {
                    "replicas": 1,
                    "selector": {
                        "matchLabels": {
                            "app.kubernetes.io/name": "benchflow",
                            "benchflow.io/component": "grafana",
                        }
                    },
                    "template": {
                        "metadata": {
                            "labels": {
                                "app.kubernetes.io/name": "benchflow",
                                "benchflow.io/component": "grafana",
                            }
                        },
                        "spec": {
                            "serviceAccountName": self.grafana_datasource_service_account,
                            "containers": [
                                {
                                    "name": "grafana",
                                    "image": "grafana/grafana:11.5.2",
                                    "ports": [{"containerPort": 3000, "name": "http"}],
                                    "env": [
                                        {
                                            "name": "GF_SECURITY_ADMIN_USER",
                                            "valueFrom": {
                                                "secretKeyRef": {
                                                    "name": self.grafana_admin_secret_name,
                                                    "key": "admin-user",
                                                }
                                            },
                                        },
                                        {
                                            "name": "GF_SECURITY_ADMIN_PASSWORD",
                                            "valueFrom": {
                                                "secretKeyRef": {
                                                    "name": self.grafana_admin_secret_name,
                                                    "key": "admin-password",
                                                }
                                            },
                                        },
                                        {
                                            "name": "GF_INSTALL_PLUGINS",
                                            "value": "yesoreyeram-infinity-datasource",
                                        },
                                        {
                                            "name": "GRAFANA_THANOS_TOKEN",
                                            "valueFrom": {
                                                "secretKeyRef": {
                                                    "name": self.grafana_datasource_token_secret,
                                                    "key": "token",
                                                }
                                            },
                                        },
                                        {
                                            "name": "BENCHFLOW_ARCHIVE_BASE_URL",
                                            "value": archive_base_url,
                                        },
                                    ],
                                    "volumeMounts": [
                                        {
                                            "name": "grafana-data",
                                            "mountPath": "/var/lib/grafana",
                                        },
                                        {
                                            "name": "provisioning",
                                            "mountPath": "/etc/grafana/provisioning/datasources/datasources.yaml",
                                            "subPath": "datasources.yaml",
                                        },
                                        {
                                            "name": "provisioning",
                                            "mountPath": "/etc/grafana/provisioning/dashboards/dashboards.yaml",
                                            "subPath": "dashboards.yaml",
                                        },
                                        {
                                            "name": "dashboards",
                                            "mountPath": "/var/lib/grafana/dashboards",
                                        },
                                    ],
                                    "readinessProbe": {
                                        "httpGet": {
                                            "path": "/api/health",
                                            "port": "http",
                                        },
                                        "initialDelaySeconds": 10,
                                        "periodSeconds": 10,
                                    },
                                    "livenessProbe": {
                                        "httpGet": {
                                            "path": "/api/health",
                                            "port": "http",
                                        },
                                        "initialDelaySeconds": 30,
                                        "periodSeconds": 30,
                                    },
                                }
                            ],
                            "volumes": [
                                {
                                    "name": "grafana-data",
                                    "emptyDir": {},
                                },
                                {
                                    "name": "provisioning",
                                    "configMap": {"name": "grafana-provisioning"},
                                },
                                {
                                    "name": "dashboards",
                                    "configMap": {"name": "grafana-dashboards"},
                                },
                            ],
                        },
                    },
                },
            },
            {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": "grafana",
                    "labels": {
                        "app.kubernetes.io/name": "benchflow",
                        "benchflow.io/component": "grafana",
                    },
                },
                "spec": {
                    "selector": {
                        "app.kubernetes.io/name": "benchflow",
                        "benchflow.io/component": "grafana",
                    },
                    "ports": [{"name": "http", "port": 3000, "targetPort": "http"}],
                },
            },
            {
                "apiVersion": "route.openshift.io/v1",
                "kind": "Route",
                "metadata": {
                    "name": "grafana",
                    "labels": {
                        "app.kubernetes.io/name": "benchflow",
                        "benchflow.io/component": "grafana",
                    },
                },
                "spec": {
                    "to": {"kind": "Service", "name": "grafana"},
                    "port": {"targetPort": "http"},
                    "tls": {
                        "termination": "edge",
                        "insecureEdgeTerminationPolicy": "Redirect",
                    },
                },
            },
        ]

        step("Applying Grafana deployment, route, datasource, and dashboards")
        self._apply_documents(
            documents,
            namespace=self.grafana_namespace,
            description="applying Grafana stack",
        )
        step("Waiting for Grafana route")
        self.wait_for_grafana_route(timeout_seconds=600)
        step("Waiting for Grafana to become ready")
        self.wait_for_grafana_ready(timeout_seconds=600)


def run_install(repo_root: Path, options: InstallOptions) -> int:
    installer = Installer(repo_root, options)
    return installer.run()


__all__ = ["InstallOptions", "Installer", "run_install", "discover_repo_root"]
