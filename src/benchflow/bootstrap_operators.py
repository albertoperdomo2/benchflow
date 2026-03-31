from __future__ import annotations

import time
from typing import Any

from .cluster import CommandError
from .ui import detail, emit, step, success, warning


def print_olm_diagnostics(
    installer: Any, *, subscription_name: str, namespace: str, catalog_source: str
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
        result = installer._oc(*argv, check=False)
        output = result.stdout or result.stderr or ""
        if output:
            emit(output, end="" if output.endswith("\n") else "\n", stderr=True)


def wait_for_subscription_current_csv(
    installer: Any, *, subscription_name: str, namespace: str, timeout_seconds: int
) -> str:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            subscription = installer._oc_json(
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
            message = resolution_failed.get("message", "subscription resolution failed")
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


def get_subscription(
    installer: Any,
    *,
    subscription_name: str,
    namespace: str,
    description: str | None = None,
) -> dict[str, Any]:
    return installer._oc_json(
        "get",
        "subscription",
        subscription_name,
        "-n",
        namespace,
        retry=True,
        description=description or f"reading subscription/{subscription_name}",
    )


def get_packagemanifest(installer: Any, package_name: str) -> dict[str, Any]:
    return installer._oc_json(
        "get",
        "packagemanifest",
        package_name,
        "-n",
        "openshift-marketplace",
        retry=True,
        description=f"reading packagemanifest/{package_name}",
    )


def default_channel_for_package(installer: Any, package_name: str) -> str:
    package = get_packagemanifest(installer, package_name)
    channel = package.get("status", {}).get("defaultChannel", "")
    if not channel:
        raise CommandError(
            f"packagemanifest/{package_name} does not expose a default channel"
        )
    return str(channel)


def catalog_source_for_package(installer: Any, package_name: str) -> tuple[str, str]:
    package = get_packagemanifest(installer, package_name)
    status = package.get("status", {}) or {}
    catalog_source = status.get("catalogSource", "")
    catalog_namespace = status.get("catalogSourceNamespace", "")
    if not catalog_source or not catalog_namespace:
        raise CommandError(
            f"packagemanifest/{package_name} does not expose catalog source details"
        )
    return str(catalog_source), str(catalog_namespace)


def operatorgroups_in_namespace(installer: Any, namespace: str) -> list[dict[str, Any]]:
    operatorgroups = installer._oc_json(
        "get",
        "operatorgroup",
        "-n",
        namespace,
        retry=True,
        description=f"reading operatorgroups in namespace {namespace}",
    )
    return list(operatorgroups.get("items", []))


def reuse_or_create_operatorgroup(
    installer: Any, *, namespace: str, operatorgroup_name: str
) -> bool:
    operatorgroups = operatorgroups_in_namespace(installer, namespace)
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
    target_namespaces = operatorgroup.get("spec", {}).get("targetNamespaces", []) or []
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


def install_operator_from_package(
    installer: Any,
    *,
    package_name: str,
    namespace: str,
    subscription_name: str,
    operatorgroup_name: str,
    asset_path: str,
) -> str:
    channel = default_channel_for_package(installer, package_name)
    catalog_source, catalog_namespace = catalog_source_for_package(
        installer, package_name
    )
    installer.ensure_namespace(namespace)
    has_existing_operatorgroup = reuse_or_create_operatorgroup(
        installer, namespace=namespace, operatorgroup_name=operatorgroup_name
    )
    documents = installer._render_asset_documents(
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
    installer._apply_documents(
        documents,
        namespace=None,
        description=f"installing operator package {package_name}",
    )
    step(f"Waiting for the {package_name} subscription to resolve")
    csv_name = wait_for_subscription_current_csv(
        installer,
        subscription_name=subscription_name,
        namespace=namespace,
        timeout_seconds=600,
    )
    step(f"Waiting for CSV {csv_name} to succeed")
    wait_for_csv_succeeded(
        installer,
        subscription_name=subscription_name,
        namespace=namespace,
        csv_name=csv_name,
        timeout_seconds=900,
        csv_prefix=f"{package_name}.",
        catalog_source=catalog_source,
    )
    return csv_name


def approve_pending_installplan(
    installer: Any,
    *,
    subscription_name: str,
    namespace: str,
    csv_prefix: str,
    catalog_source: str,
    expected_csv_name: str | None = None,
) -> None:
    subscription = installer._oc_json(
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

    installplan = installer._oc_json(
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
                print_olm_diagnostics(
                    installer,
                    subscription_name=subscription_name,
                    namespace=namespace,
                    catalog_source=catalog_source,
                )
                raise CommandError(
                    f"refusing to auto-approve InstallPlan {installplan_name}: expected {expected_csv_name}, got {csv_name}"
                )
            continue
        if not str(csv_name).startswith(csv_prefix):
            print_olm_diagnostics(
                installer,
                subscription_name=subscription_name,
                namespace=namespace,
                catalog_source=catalog_source,
            )
            raise CommandError(
                f"refusing to auto-approve InstallPlan {installplan_name}: unexpected CSV {csv_name}"
            )

    step(f"Approving pending InstallPlan {installplan_name}")
    installer._oc(
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


def wait_for_csv_succeeded(
    installer: Any,
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
        approve_pending_installplan(
            installer,
            subscription_name=subscription_name,
            namespace=namespace,
            csv_prefix=csv_prefix,
            catalog_source=catalog_source,
            expected_csv_name=expected_csv_name,
        )
        try:
            csv = installer._oc_json(
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

    print_olm_diagnostics(
        installer,
        subscription_name=subscription_name,
        namespace=namespace,
        catalog_source=catalog_source,
    )
    raise CommandError(f"timed out waiting for CSV {csv_name} to reach Succeeded")


def install_accelerator_prerequisites(installer: Any) -> None:
    step("Installing accelerator prerequisites for GPU inference")
    install_nfd_operator_and_instance(installer)
    install_gpu_operator_and_cluster_policy(installer)


def nfd_ready(installer: Any) -> bool:
    return installer._resource_exists(
        "get",
        "nodefeaturediscovery.nfd.openshift.io",
        "nfd-instance",
        "-n",
        installer.nfd_namespace,
    )


def gpu_operator_ready(installer: Any) -> bool:
    if not installer._resource_exists(
        "get", "clusterpolicy.nvidia.com", "gpu-cluster-policy"
    ):
        return False
    cluster_policy = installer._oc_json(
        "get",
        "clusterpolicy.nvidia.com",
        "gpu-cluster-policy",
        retry=True,
        description="reading ClusterPolicy/gpu-cluster-policy",
    )
    status = str(cluster_policy.get("status", {}).get("state", ""))
    return status.lower() == "ready"


def install_nfd_operator_and_instance(installer: Any) -> None:
    if nfd_ready(installer):
        success("Node Feature Discovery instance already present")
        return
    install_operator_from_package(
        installer,
        package_name=installer.nfd_package_name,
        namespace=installer.nfd_namespace,
        subscription_name="nfd",
        operatorgroup_name="nfd",
        asset_path="operators/nfd/operator.yaml",
    )
    installer._wait_for_resource(
        resource="crd/nodefeaturediscoveries.nfd.openshift.io",
        namespace=None,
        timeout_seconds=600,
        label="CRD nodefeaturediscoveries.nfd.openshift.io",
    )
    step("Applying the Node Feature Discovery instance")
    installer._apply_asset_documents(
        "operators/nfd/instance.yaml",
        namespace=None,
        description="applying NodeFeatureDiscovery instance",
        variables=installer._base_asset_variables(),
    )
    success("Node Feature Discovery operator and instance are present")


def install_gpu_operator_and_cluster_policy(installer: Any) -> None:
    if gpu_operator_ready(installer):
        success("NVIDIA GPU ClusterPolicy already present and ready")
        return
    install_operator_from_package(
        installer,
        package_name=installer.gpu_operator_package_name,
        namespace=installer.gpu_operator_namespace,
        subscription_name="gpu-operator-certified",
        operatorgroup_name="gpu-operator-certified",
        asset_path="operators/gpu/operator.yaml",
    )
    installer._wait_for_resource(
        resource="crd/clusterpolicies.nvidia.com",
        namespace=None,
        timeout_seconds=600,
        label="CRD clusterpolicies.nvidia.com",
    )
    step("Applying the NVIDIA GPU ClusterPolicy")
    installer._apply_asset_documents(
        "operators/gpu/cluster-policy.yaml",
        namespace=None,
        description="applying NVIDIA GPU ClusterPolicy",
        variables=installer._base_asset_variables(),
    )
    success("NVIDIA GPU Operator and ClusterPolicy are present")


def install_tekton_if_needed(installer: Any) -> None:
    if installer.tekton_ready():
        success("Tekton CRDs already present")
        return

    step(
        f"Installing OpenShift Pipelines operator in {installer.pipelines_operator_namespace}"
    )
    installer._apply_documents(
        installer._render_asset_documents(
            "operators/tekton/subscription.yaml",
            {"TEKTON_CHANNEL": installer.options.tekton_channel},
        ),
        namespace=None,
        description="applying OpenShift Pipelines subscription",
    )

    step("Waiting for the Tekton subscription to resolve")
    tekton_csv = wait_for_subscription_current_csv(
        installer,
        subscription_name="openshift-pipelines-operator",
        namespace=installer.pipelines_operator_namespace,
        timeout_seconds=600,
    )
    step(f"Waiting for CSV {tekton_csv} to succeed")
    wait_for_csv_succeeded(
        installer,
        subscription_name="openshift-pipelines-operator",
        namespace=installer.pipelines_operator_namespace,
        csv_name=tekton_csv,
        timeout_seconds=600,
        csv_prefix="openshift-pipelines-operator-rh.",
        catalog_source="redhat-operators",
    )

    step("Waiting for Tekton CRDs")
    for crd_name in (
        "crd/tasks.tekton.dev",
        "crd/pipelines.tekton.dev",
        "crd/pipelineruns.tekton.dev",
    ):
        installer._wait_for_resource(
            resource=crd_name,
            namespace=None,
            timeout_seconds=600,
            label=crd_name,
        )

    step("Waiting for Tekton service accounts")
    for sa_name in (
        "serviceaccount/tekton-pipelines-controller",
        "serviceaccount/tekton-events-controller",
        "serviceaccount/tekton-pipelines-webhook",
    ):
        installer._wait_for_resource(
            resource=sa_name,
            namespace=installer.pipelines_runtime_namespace,
            timeout_seconds=600,
            label=f"{sa_name} in namespace {installer.pipelines_runtime_namespace}",
        )

    configure_tekton_scc(installer)

    step("Waiting for Tekton controllers")
    for deployment in (
        "deployment/tekton-pipelines-controller",
        "deployment/tekton-pipelines-webhook",
    ):
        installer._wait_for_resource(
            resource=deployment,
            namespace=installer.pipelines_runtime_namespace,
            timeout_seconds=600,
            label=f"{deployment} in namespace {installer.pipelines_runtime_namespace}",
        )
        installer._oc(
            "wait",
            "--for=condition=available",
            "--timeout=10m",
            deployment,
            "-n",
            installer.pipelines_runtime_namespace,
            retry=True,
            description=f"waiting for {deployment} in {installer.pipelines_runtime_namespace}",
        )


def configure_tekton_scc(installer: Any) -> None:
    if not installer._resource_exists(
        "get", "namespace", installer.pipelines_runtime_namespace
    ):
        detail(
            f"Skipping Tekton SCC configuration because {installer.pipelines_runtime_namespace} does not exist yet"
        )
        return

    step("Configuring Tekton SCCs")
    for service_account in (
        "tekton-pipelines-controller",
        "tekton-events-controller",
        "tekton-pipelines-webhook",
    ):
        result = installer._oc(
            "adm",
            "policy",
            "add-scc-to-user",
            "privileged",
            "-z",
            service_account,
            "-n",
            installer.pipelines_runtime_namespace,
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
