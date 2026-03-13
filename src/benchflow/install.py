from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    grafana_channel: str = "v5"
    grafana_starting_csv: str = "grafana-operator.v5.21.2"
    models_storage_access_mode: str = "ReadWriteOnce"
    models_storage_size: str = "250Gi"
    models_storage_class: str | None = None
    results_storage_size: str = "20Gi"
    results_storage_class: str | None = None


class Installer:
    pipelines_operator_namespace = "openshift-operators"
    pipelines_runtime_namespace = "openshift-pipelines"
    grafana_operator_name = "grafana-operator"
    grafana_admin_secret_name = "grafana-admin-credentials"
    grafana_datasource_service_account = "benchflow-grafana-datasource"
    grafana_datasource_token_secret = "benchflow-grafana-datasource-token"

    def __init__(self, repo_root: Path, options: InstallOptions) -> None:
        self.repo_root = repo_root.resolve()
        self.options = options
        self._default_storage_class_name: str | None = None

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
        self.ensure_namespace()

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
                ("Install Tekton if missing", str(options.install_tekton).lower()),
                ("Install Grafana if missing", str(options.install_grafana).lower()),
                ("OpenShift Pipelines channel", options.tekton_channel),
                ("Grafana operator channel", options.grafana_channel),
                ("Grafana operator CSV", options.grafana_starting_csv),
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
                    "cluster-monitoring-view -> benchflow-runner, benchflow-grafana-datasource",
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
                ("Tekton install attempted", str(options.install_tekton).lower()),
                ("Grafana install attempted", str(options.install_grafana).lower()),
                ("OpenShift Pipelines channel", options.tekton_channel),
                ("Grafana operator channel", options.grafana_channel),
                ("Grafana operator CSV", options.grafana_starting_csv),
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
                    "cluster-monitoring-view bound to benchflow-runner and benchflow-grafana-datasource",
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
                f"oc get secret -n {options.namespace} {self.grafana_admin_secret_name} "
                '-o go-template=\'{{index .data "GF_SECURITY_ADMIN_PASSWORD" | base64decode}}{{"\\n"}}\''
            )
        step("Required secrets if you have not already created them")
        detail("config/cluster/secrets/huggingface-token.example.yaml")
        detail("config/cluster/secrets/mlflow-auth.example.yaml")
        detail("config/cluster/secrets/mlflow-s3-creds.example.yaml")
        step("Example run")
        detail("pip install -e .")
        detail(
            f"bflow experiment run experiments/examples/qwen3-06b-llm-d-smoke.yaml --namespace {options.namespace}"
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

    def ensure_namespace(self) -> None:
        if not self._resource_exists("get", "namespace", self.options.namespace):
            step(f"Creating namespace {self.options.namespace}")
            self._oc(
                "create",
                "namespace",
                self.options.namespace,
                retry=True,
                description=f"creating namespace {self.options.namespace}",
                echo_output=True,
            )
            return

        namespace = self._oc_json(
            "get",
            "namespace",
            self.options.namespace,
            retry=True,
            description=f"reading namespace/{self.options.namespace}",
        )
        deletion_timestamp = namespace.get("metadata", {}).get("deletionTimestamp")
        if not deletion_timestamp:
            return

        step(f"Waiting for namespace {self.options.namespace} to finish terminating")
        deadline = time.time() + 600
        while time.time() < deadline:
            if not self._resource_exists("get", "namespace", self.options.namespace):
                step(f"Creating namespace {self.options.namespace}")
                self._oc(
                    "create",
                    "namespace",
                    self.options.namespace,
                    retry=True,
                    description=f"creating namespace {self.options.namespace}",
                    echo_output=True,
                )
                return
            time.sleep(5)
        raise CommandError(
            f"timed out waiting for namespace {self.options.namespace} to finish terminating"
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
            description=description
            or f"reading subscription/{subscription_name}",
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
        subscription_already_present = self._resource_exists(
            "get",
            "subscription",
            self.grafana_operator_name,
            "-n",
            self.options.namespace,
        )
        if subscription_already_present:
            success(
                f"Grafana operator subscription already present in {self.options.namespace}; reconciling desired version"
            )
        else:
            if not self.options.install_grafana:
                raise CommandError(
                    "Grafana operator is not installed in the target namespace and "
                    "--skip-grafana-install was requested"
                )
            step(f"Installing Grafana operator in {self.options.namespace}")

        operator_group = {
            "apiVersion": "operators.coreos.com/v1",
            "kind": "OperatorGroup",
            "metadata": {
                "name": "benchflow-grafana",
                "namespace": self.options.namespace,
            },
            "spec": {"targetNamespaces": [self.options.namespace]},
        }
        subscription = {
            "apiVersion": "operators.coreos.com/v1alpha1",
            "kind": "Subscription",
            "metadata": {
                "name": self.grafana_operator_name,
                "namespace": self.options.namespace,
            },
            "spec": {
                "channel": self.options.grafana_channel,
                "installPlanApproval": "Manual",
                "name": self.grafana_operator_name,
                "source": "community-operators",
                "sourceNamespace": "openshift-marketplace",
                "startingCSV": self.options.grafana_starting_csv,
            },
        }
        self._apply_documents(
            [operator_group, subscription],
            namespace=self.options.namespace,
            description="applying Grafana operator resources",
        )

        step("Waiting for the Grafana subscription to resolve")
        grafana_csv = self._wait_for_subscription_current_csv(
            subscription_name=self.grafana_operator_name,
            namespace=self.options.namespace,
            timeout_seconds=600,
        )
        subscription = self._get_subscription(
            subscription_name=self.grafana_operator_name,
            namespace=self.options.namespace,
            description=f"reading subscription/{self.grafana_operator_name} after resolution",
        )
        installed_grafana_csv = str(
            subscription.get("status", {}).get("installedCSV", "") or ""
        )
        subscription_state = str(subscription.get("status", {}).get("state", "") or "")
        expected_csv_name = self.options.grafana_starting_csv
        if grafana_csv != self.options.grafana_starting_csv:
            if subscription_already_present:
                if (
                    installed_grafana_csv == self.options.grafana_starting_csv
                    and subscription_state == "UpgradePending"
                ):
                    warning(
                        "Grafana operator is already installed as "
                        f"{installed_grafana_csv}; a newer unapproved upgrade "
                        f"({grafana_csv}) is pending, but BenchFlow will continue "
                        f"with the pinned installed version"
                    )
                    grafana_csv = installed_grafana_csv
                else:
                    warning(
                        "Grafana operator is already installed as "
                        f"{grafana_csv}; BenchFlow will keep the existing version "
                        f"instead of forcing a downgrade to {self.options.grafana_starting_csv}"
                    )
                expected_csv_name = grafana_csv
            else:
                self._print_olm_diagnostics(
                    subscription_name=self.grafana_operator_name,
                    namespace=self.options.namespace,
                    catalog_source="community-operators",
                )
                raise CommandError(
                    "Grafana subscription resolved to an unexpected CSV: "
                    f"wanted {self.options.grafana_starting_csv}, got {grafana_csv}"
                )
        if (
            subscription_already_present
            and grafana_csv == self.options.grafana_starting_csv
            and installed_grafana_csv == self.options.grafana_starting_csv
            and subscription_state == "UpgradePending"
        ):
            success(
                f"Grafana operator {installed_grafana_csv} is already installed; skipping CSV phase wait despite pending upgrade"
            )
        else:
            step(f"Waiting for CSV {grafana_csv} to succeed")
            self._wait_for_csv_succeeded(
                subscription_name=self.grafana_operator_name,
                namespace=self.options.namespace,
                csv_name=grafana_csv,
                timeout_seconds=600,
                csv_prefix="grafana-operator.",
                catalog_source="community-operators",
                expected_csv_name=expected_csv_name,
            )

        step("Waiting for Grafana CRDs")
        for crd in (
            "crd/grafanas.grafana.integreatly.org",
            "crd/grafanadashboards.grafana.integreatly.org",
            "crd/grafanadatasources.grafana.integreatly.org",
        ):
            self._wait_for_resource(
                resource=crd, namespace=None, timeout_seconds=600, label=crd
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
        if not self._resource_exists("get", "route", "-n", self.options.namespace):
            return None
        routes = self._oc_json(
            "get",
            "route",
            "-n",
            self.options.namespace,
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
        last_message = ""
        while time.time() < deadline:
            if not self._resource_exists(
                "get", "grafana", "grafana", "-n", self.options.namespace
            ):
                time.sleep(5)
                continue
            grafana = self._oc_json(
                "get",
                "grafana",
                "grafana",
                "-n",
                self.options.namespace,
                retry=True,
                description="reading grafana/grafana",
            )
            status = grafana.get("status", {})
            stage = str(status.get("stage", "")).lower()
            stage_status = str(status.get("stageStatus", "")).lower()
            last_message = str(status.get("lastMessage", ""))
            if stage == "complete" and stage_status == "success":
                return
            time.sleep(5)
        suffix = f": {last_message}" if last_message else ""
        raise CommandError(
            f"timed out waiting for grafana/grafana to become ready{suffix}"
        )

    def apply_grafana_stack(self) -> None:
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
            namespace=self.options.namespace,
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
                            "namespace": self.options.namespace,
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
            namespace=self.options.namespace,
            timeout_seconds=300,
        )

        step("Applying Grafana instance and route")
        self._apply_documents(
            [
                {
                    "apiVersion": "grafana.integreatly.org/v1beta1",
                    "kind": "Grafana",
                    "metadata": {
                        "name": "grafana",
                        "labels": {
                            "dashboards": "benchflow",
                            "folders": "benchflow",
                            "app.kubernetes.io/name": "benchflow",
                            "benchflow.io/component": "grafana",
                        },
                    },
                    "spec": {
                        "config": {
                            "log": {"mode": "console"},
                            "auth": {"disable_login_form": "false"},
                        },
                        "route": {"spec": {}},
                    },
                }
            ],
            namespace=self.options.namespace,
            description="applying Grafana instance",
        )
        step("Waiting for Grafana route")
        self.wait_for_grafana_route(timeout_seconds=600)
        step("Waiting for Grafana to become ready")
        self.wait_for_grafana_ready(timeout_seconds=600)

        dashboard_json = (
            self.repo_root
            / "config"
            / "monitoring"
            / "grafana-dashboard-benchflow.json"
        ).read_text(encoding="utf-8")
        documents = [
            {
                "apiVersion": "grafana.integreatly.org/v1beta1",
                "kind": "GrafanaDatasource",
                "metadata": {
                    "name": "openshift-monitoring",
                    "labels": {
                        "app.kubernetes.io/name": "benchflow",
                        "benchflow.io/component": "grafana",
                    },
                },
                "spec": {
                    "instanceSelector": {"matchLabels": {"dashboards": "benchflow"}},
                    "valuesFrom": [
                        {
                            "targetPath": "secureJsonData.httpHeaderValue1",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "name": self.grafana_datasource_token_secret,
                                    "key": "token",
                                }
                            },
                        }
                    ],
                    "datasource": {
                        "name": "openshift-monitoring",
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
                            "httpHeaderValue1": "Bearer ${token}",
                        },
                    },
                },
            },
            {
                "apiVersion": "grafana.integreatly.org/v1beta1",
                "kind": "GrafanaDashboard",
                "metadata": {
                    "name": "benchflow",
                    "labels": {
                        "app.kubernetes.io/name": "benchflow",
                        "benchflow.io/component": "grafana",
                    },
                },
                "spec": {
                    "instanceSelector": {"matchLabels": {"dashboards": "benchflow"}},
                    "json": dashboard_json,
                },
            },
        ]

        step("Applying Grafana datasource and dashboard")
        self._apply_documents(
            documents,
            namespace=self.options.namespace,
            description="applying Grafana datasource and dashboard",
        )


def run_install(repo_root: Path, options: InstallOptions) -> int:
    installer = Installer(repo_root, options)
    return installer.run()


__all__ = ["InstallOptions", "Installer", "run_install", "discover_repo_root"]
