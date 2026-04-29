from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import yaml

from ..assets import render_yaml_documents
from ..cluster import (
    CommandError,
    require_any_command,
    run_command,
    run_json_command,
    use_kubeconfig,
)
from ..models import ResolvedRunPlan
from ..ui import detail, step, success

RHOAI_OPERATOR_NAMESPACE = "redhat-ods-operator"
RHOAI_OPERATOR_SUBSCRIPTION_NAME = "rhods-operator"
RHOAI_OPERATORGROUP_NAME = "redhat-ods-operator"
RHOAI_OPERATOR_PACKAGE_NAME = "rhods-operator"
DEFAULT_RHOAI_CHANNEL = "fast-3.x"
DEFAULT_RHOAI_PLATFORM_VERSION = "RHOAI-3.3"
RHOAI_DSC_NAME = "default-dsc"
RHOAI_APPLICATIONS_NAMESPACE = "redhat-ods-applications"
RHOAI_GATEWAYCLASS_NAME = "openshift-ai-inference"
RHOAI_GATEWAY_NAME = "openshift-ai-inference"
RHOAI_GATEWAY_NAMESPACE = "openshift-ingress"
RHOAI_GATEWAY_DEFAULT_TLS_SECRET = "default-gateway-tls"
RHOAI_GATEWAY_FALLBACK_TLS_SECRET = "data-science-gateway-service-tls"
_RHOAI_VERSION_RE = re.compile(r"(?P<major>\d+)\.(?P<minor>\d+)")
_RHOAI_EXACT_VERSION_RE = re.compile(r"(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)")
_RHOAI_EA_RE = re.compile(r"(?i)(?:^|[-_.])ea[-_.]?(?P<index>\d+)(?:$|[-_.])")


def _empty_state(platform_version: str) -> dict[str, Any]:
    return {
        "apiVersion": "benchflow.io/v1alpha1",
        "kind": "SetupState",
        "platform": "rhoai",
        "platform_version": platform_version,
        "operator_namespace_created": False,
        "operator_subscription_managed": False,
        "operatorgroup_managed": False,
        "operator_csv_name": "",
        "datasciencecluster_managed": False,
        "gatewayclass_managed": False,
        "gateway_managed": False,
    }


def _persist_state(state: dict[str, Any], state_path: Path | None) -> None:
    if state_path is None:
        return
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _resource_exists(argv: list[str]) -> bool:
    result = run_command(argv, capture_output=True, check=False)
    return result.returncode == 0


def _namespace_manifest(kubectl_cmd: str, name: str) -> dict[str, Any] | None:
    result = run_command(
        [kubectl_cmd, "get", "namespace", name, "-o", "json"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout or "{}")


def _wait_for_namespace_absent(
    kubectl_cmd: str, name: str, *, timeout_seconds: int
) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _namespace_manifest(kubectl_cmd, name) is None:
            return
        time.sleep(5)
    raise CommandError(f"timed out waiting for namespace {name} to be deleted")


def _ensure_namespace(kubectl_cmd: str, name: str) -> bool:
    manifest = _namespace_manifest(kubectl_cmd, name)
    if manifest is not None:
        deletion_timestamp = manifest.get("metadata", {}).get("deletionTimestamp") or ""
        if not deletion_timestamp:
            return False
        detail(f"Waiting for namespace {name} to finish terminating")
        _wait_for_namespace_absent(kubectl_cmd, name, timeout_seconds=900)
    run_command([kubectl_cmd, "create", "namespace", name])
    return True


def _operator_subscription_exists(kubectl_cmd: str) -> bool:
    return _resource_exists(
        [
            kubectl_cmd,
            "get",
            "subscription",
            RHOAI_OPERATOR_SUBSCRIPTION_NAME,
            "-n",
            RHOAI_OPERATOR_NAMESPACE,
            "-o",
            "name",
        ]
    )


def _datasciencecluster_exists(kubectl_cmd: str) -> bool:
    return _resource_exists(
        [kubectl_cmd, "get", "datasciencecluster", RHOAI_DSC_NAME, "-o", "name"]
    )


def _gatewayclass_exists(kubectl_cmd: str) -> bool:
    return _resource_exists(
        [kubectl_cmd, "get", "gatewayclass", RHOAI_GATEWAYCLASS_NAME, "-o", "name"]
    )


def _gateway_exists(kubectl_cmd: str) -> bool:
    return _resource_exists(
        [
            kubectl_cmd,
            "get",
            "gateway",
            RHOAI_GATEWAY_NAME,
            "-n",
            RHOAI_GATEWAY_NAMESPACE,
            "-o",
            "name",
        ]
    )


def _infer_rhoai_gateway_hostname(kubectl_cmd: str) -> str | None:
    route = run_command(
        [
            kubectl_cmd,
            "get",
            "route",
            "data-science-gateway",
            "-n",
            RHOAI_GATEWAY_NAMESPACE,
            "-o",
            "json",
        ],
        capture_output=True,
        check=False,
    )
    if route.returncode != 0:
        return None
    payload = json.loads(route.stdout or "{}")
    host = str(payload.get("spec", {}).get("host") or "").strip()
    if not host:
        return None
    _, separator, remainder = host.partition(".")
    if not separator or not remainder:
        return None
    return f"inference-gateway.{remainder}"


def _rhoai_gateway_tls_secret_name(kubectl_cmd: str) -> str:
    for secret_name in (
        RHOAI_GATEWAY_DEFAULT_TLS_SECRET,
        RHOAI_GATEWAY_FALLBACK_TLS_SECRET,
    ):
        if _resource_exists(
            [
                kubectl_cmd,
                "get",
                "secret",
                secret_name,
                "-n",
                RHOAI_GATEWAY_NAMESPACE,
                "-o",
                "name",
            ]
        ):
            return secret_name
    raise CommandError(
        f"could not find a Gateway TLS secret in {RHOAI_GATEWAY_NAMESPACE}; "
        f"expected {RHOAI_GATEWAY_DEFAULT_TLS_SECRET} or {RHOAI_GATEWAY_FALLBACK_TLS_SECRET}"
    )


def _package_source(package: dict[str, Any], package_name: str) -> tuple[str, str]:
    status = package.get("status", {})
    source = str(status.get("catalogSource") or "")
    source_namespace = str(status.get("catalogSourceNamespace") or "")
    if not source or not source_namespace:
        raise CommandError(
            f"packagemanifest/{package_name} does not expose catalog source details"
        )
    return source, source_namespace


def _candidate_package_manifests(
    kubectl_cmd: str, package_name: str
) -> list[dict[str, Any]]:
    packages = run_json_command(
        [
            kubectl_cmd,
            "get",
            "packagemanifest",
            "-n",
            "openshift-marketplace",
            "-o",
            "json",
        ]
    )
    return [
        item
        for item in packages.get("items", []) or []
        if str(item.get("metadata", {}).get("name") or "") == package_name
    ]


def _operatorgroups_in_namespace(
    kubectl_cmd: str, namespace: str
) -> list[dict[str, Any]]:
    payload = run_json_command(
        [kubectl_cmd, "get", "operatorgroup", "-n", namespace, "-o", "json"]
    )
    return list(payload.get("items", []) or [])


def _parse_version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", value)
    if not match:
        return tuple()
    return tuple(int(part) for part in match.groups())


def _normalize_rhoai_mlflow_version(value: str) -> str | None:
    candidate = str(value or "").strip()
    if not candidate:
        return None

    version_match = _RHOAI_VERSION_RE.search(candidate)
    if version_match is None:
        return None

    label = f"RHOAI-{version_match.group('major')}.{version_match.group('minor')}"
    ea_match = _RHOAI_EA_RE.search(candidate)
    if ea_match is None:
        return label
    return f"{label}-EA{ea_match.group('index')}"


def normalize_rhoai_platform_version(value: str) -> str:
    normalized = _normalize_rhoai_mlflow_version(value)
    return normalized or DEFAULT_RHOAI_PLATFORM_VERSION


def _requested_rhoai_version(
    value: str,
) -> tuple[str, tuple[int, ...] | None, str | None]:
    requested = str(value or "").strip() or DEFAULT_RHOAI_PLATFORM_VERSION
    version_match = _RHOAI_VERSION_RE.search(requested)
    if version_match is None:
        raise CommandError(f"invalid requested RHOAI version: {value!r}")
    exact_match = _RHOAI_EXACT_VERSION_RE.search(requested)
    ea_match = _RHOAI_EA_RE.search(requested)
    requested_exact_version = (
        tuple(int(exact_match.group(name)) for name in ("major", "minor", "patch"))
        if exact_match is not None
        else None
    )
    return (
        f"{version_match.group('major')}.{version_match.group('minor')}.",
        requested_exact_version,
        ea_match.group("index") if ea_match is not None else None,
    )


def discover_rhoai_mlflow_version(kubeconfig: str | Path | None = None) -> str:
    kubectl_cmd = require_any_command("oc", "kubectl")
    with use_kubeconfig(kubeconfig):
        subscription = run_json_command(
            [
                kubectl_cmd,
                "get",
                "subscription",
                RHOAI_OPERATOR_SUBSCRIPTION_NAME,
                "-n",
                RHOAI_OPERATOR_NAMESPACE,
                "-o",
                "json",
            ]
        )

    candidates = (
        (subscription.get("status", {}).get("currentCSVDesc", {}) or {}).get("version"),
        subscription.get("status", {}).get("currentCSV"),
        subscription.get("status", {}).get("installedCSV"),
    )
    for candidate in candidates:
        version = _normalize_rhoai_mlflow_version(str(candidate or ""))
        if version:
            return version

    raise CommandError(
        "could not derive a RHOAI MLflow version from subscription/rhods-operator"
    )


def _rhoai_channel(plan: ResolvedRunPlan) -> str:
    return str(plan.deployment.platform_channel or "").strip() or DEFAULT_RHOAI_CHANNEL


def _rhoai_channel_candidates(
    package: dict[str, Any],
    *,
    requested_series: str,
    requested_exact_version: tuple[int, ...] | None,
    requested_ea: str | None,
    channel_name: str,
    package_name: str,
) -> list[tuple[tuple[int, ...], str, str, str]]:
    source, source_namespace = _package_source(package, package_name)
    channels = package.get("status", {}).get("channels", []) or []
    channel = next(
        (item for item in channels if item.get("name") == channel_name), None
    )
    if not isinstance(channel, dict):
        return []

    candidates: list[tuple[tuple[int, ...], str, str, str]] = []
    current_csv = str(channel.get("currentCSV") or "")
    current_version = str(
        (channel.get("currentCSVDesc", {}) or {}).get("version") or ""
    )
    if current_csv and current_version.startswith(requested_series):
        normalized_current = _normalize_rhoai_mlflow_version(current_version)
        normalized_ea = (
            _RHOAI_EA_RE.search(normalized_current or "").group("index")
            if normalized_current and _RHOAI_EA_RE.search(normalized_current)
            else None
        )
        if normalized_ea == requested_ea:
            version_tuple = _parse_version_tuple(current_version)
            if version_tuple and (
                requested_exact_version is None
                or version_tuple == requested_exact_version
            ):
                candidates.append(
                    (version_tuple, current_csv, source, source_namespace)
                )

    for entry in channel.get("entries", []) or []:
        if not isinstance(entry, dict):
            continue
        version = str(entry.get("version") or "")
        name = str(entry.get("name") or "")
        if not version.startswith(requested_series):
            continue
        normalized_version = _normalize_rhoai_mlflow_version(version)
        normalized_ea = (
            _RHOAI_EA_RE.search(normalized_version or "").group("index")
            if normalized_version and _RHOAI_EA_RE.search(normalized_version)
            else None
        )
        if normalized_ea != requested_ea:
            continue
        version_tuple = _parse_version_tuple(version)
        if not version_tuple or not name:
            continue
        if (
            requested_exact_version is not None
            and version_tuple != requested_exact_version
        ):
            continue
        candidates.append((version_tuple, name, source, source_namespace))
    return candidates


def _resolve_rhoai_operator_package(
    kubectl_cmd: str, requested_version: str, channel_name: str
) -> tuple[str, str, str]:
    requested_series, requested_exact_version, requested_ea = _requested_rhoai_version(
        requested_version
    )
    packages = _candidate_package_manifests(kubectl_cmd, RHOAI_OPERATOR_PACKAGE_NAME)
    if not packages:
        raise CommandError(
            f"packagemanifest/{RHOAI_OPERATOR_PACKAGE_NAME} was not found"
        )

    candidates: list[tuple[tuple[int, ...], str, str, str]] = []
    packages_with_channel = 0
    for package in packages:
        package_candidates = _rhoai_channel_candidates(
            package,
            requested_series=requested_series,
            requested_exact_version=requested_exact_version,
            requested_ea=requested_ea,
            channel_name=channel_name,
            package_name=RHOAI_OPERATOR_PACKAGE_NAME,
        )
        if package_candidates:
            packages_with_channel += 1
            candidates.extend(package_candidates)
            continue
        channels = package.get("status", {}).get("channels", []) or []
        if any(item.get("name") == channel_name for item in channels):
            packages_with_channel += 1

    if packages_with_channel == 0:
        raise CommandError(
            f"packagemanifest/{RHOAI_OPERATOR_PACKAGE_NAME} does not expose channel {channel_name}"
        )

    if not candidates:
        raise CommandError(
            f"channel {channel_name} for {RHOAI_OPERATOR_PACKAGE_NAME} does not expose "
            f"a CSV matching {requested_version}"
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, starting_csv, source, source_namespace = candidates[0]
    return source, source_namespace, starting_csv


def rhoai_platform_present(kubectl_cmd: str) -> bool:
    return _operator_subscription_exists(kubectl_cmd) or _datasciencecluster_exists(
        kubectl_cmd
    )


def reset_rhoai_platform() -> None:
    kubectl_cmd = require_any_command("oc", "kubectl")

    subscription = run_command(
        [
            kubectl_cmd,
            "get",
            "subscription",
            RHOAI_OPERATOR_SUBSCRIPTION_NAME,
            "-n",
            RHOAI_OPERATOR_NAMESPACE,
            "-o",
            "json",
        ],
        capture_output=True,
        check=False,
    )
    csv_candidates: list[str] = []
    if subscription.returncode == 0:
        payload = json.loads(subscription.stdout or "{}")
        csv_candidates = [
            str(candidate or "").strip()
            for candidate in (
                payload.get("status", {}).get("currentCSV"),
                payload.get("status", {}).get("installedCSV"),
            )
            if str(candidate or "").strip()
        ]

    run_command(
        [
            kubectl_cmd,
            "delete",
            "gatewayclass",
            RHOAI_GATEWAYCLASS_NAME,
            "--ignore-not-found=true",
        ],
        check=False,
    )
    run_command(
        [
            kubectl_cmd,
            "delete",
            "gateway",
            RHOAI_GATEWAY_NAME,
            "-n",
            RHOAI_GATEWAY_NAMESPACE,
            "--ignore-not-found=true",
        ],
        check=False,
    )
    run_command(
        [
            kubectl_cmd,
            "delete",
            "datasciencecluster",
            RHOAI_DSC_NAME,
            "--ignore-not-found=true",
        ],
        check=False,
    )
    run_command(
        [
            kubectl_cmd,
            "delete",
            "subscription",
            RHOAI_OPERATOR_SUBSCRIPTION_NAME,
            "-n",
            RHOAI_OPERATOR_NAMESPACE,
            "--ignore-not-found=true",
        ],
        check=False,
    )
    for csv_name in csv_candidates:
        run_command(
            [
                kubectl_cmd,
                "delete",
                "csv",
                csv_name,
                "-n",
                RHOAI_OPERATOR_NAMESPACE,
                "--ignore-not-found=true",
            ],
            check=False,
        )

    run_command(
        [
            kubectl_cmd,
            "delete",
            "operatorgroup",
            RHOAI_OPERATORGROUP_NAME,
            "-n",
            RHOAI_OPERATOR_NAMESPACE,
            "--ignore-not-found=true",
        ],
        check=False,
    )
    run_command(
        [
            kubectl_cmd,
            "delete",
            "namespace",
            RHOAI_OPERATOR_NAMESPACE,
            "--ignore-not-found=true",
            "--wait=false",
        ],
        check=False,
    )
    detail(f"Waiting for namespace {RHOAI_OPERATOR_NAMESPACE} to be deleted")
    _wait_for_namespace_absent(
        kubectl_cmd, RHOAI_OPERATOR_NAMESPACE, timeout_seconds=900
    )
    success("RHOAI platform prerequisites have been reset")


def _apply_documents(
    kubectl_cmd: str, documents: list[dict[str, Any]], *, namespace: str | None = None
) -> None:
    manifest = yaml.safe_dump_all(documents, sort_keys=False)

    argv = [kubectl_cmd, "apply", "-f", "-"]
    if namespace is not None:
        argv[2:2] = ["-n", namespace]
    run_command(argv, input_text=manifest)


def _wait_for_subscription_current_csv(
    kubectl_cmd: str, *, namespace: str, timeout_seconds: int
) -> str:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        payload = run_json_command(
            [
                kubectl_cmd,
                "get",
                "subscription",
                RHOAI_OPERATOR_SUBSCRIPTION_NAME,
                "-n",
                namespace,
                "-o",
                "json",
            ]
        )
        current_csv = str(payload.get("status", {}).get("currentCSV") or "").strip()
        if current_csv:
            return current_csv
        time.sleep(5)
    raise CommandError("timed out waiting for the RHOAI subscription to resolve")


def _wait_for_csv_succeeded(
    kubectl_cmd: str, *, namespace: str, csv_name: str, timeout_seconds: int
) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        result = run_command(
            [kubectl_cmd, "get", "csv", csv_name, "-n", namespace, "-o", "json"],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            payload = json.loads(result.stdout or "{}")
            if str(payload.get("status", {}).get("phase") or "") == "Succeeded":
                return
        _approve_pending_installplan(
            kubectl_cmd,
            namespace=namespace,
            expected_csv_name=csv_name,
        )
        time.sleep(5)
    raise CommandError(f"timed out waiting for CSV {csv_name} to reach Succeeded")


def _approve_pending_installplan(
    kubectl_cmd: str, *, namespace: str, expected_csv_name: str
) -> None:
    subscription = run_json_command(
        [
            kubectl_cmd,
            "get",
            "subscription",
            RHOAI_OPERATOR_SUBSCRIPTION_NAME,
            "-n",
            namespace,
            "-o",
            "json",
        ]
    )
    conditions = subscription.get("status", {}).get("conditions", []) or []
    pending = next(
        (
            condition
            for condition in conditions
            if condition.get("type") == "InstallPlanPending"
        ),
        None,
    )
    if (
        not isinstance(pending, dict)
        or pending.get("status") != "True"
        or pending.get("reason") != "RequiresApproval"
    ):
        return

    installplan_name = (
        subscription.get("status", {}).get("installPlanRef", {}) or {}
    ).get("name")
    if not installplan_name:
        return

    installplan = run_json_command(
        [
            kubectl_cmd,
            "get",
            "installplan",
            str(installplan_name),
            "-n",
            namespace,
            "-o",
            "json",
        ]
    )
    csv_names = installplan.get("spec", {}).get("clusterServiceVersionNames", []) or []
    for csv_name in csv_names:
        if str(csv_name) != expected_csv_name:
            raise CommandError(
                f"refusing to auto-approve InstallPlan {installplan_name}: "
                f"expected {expected_csv_name}, got {csv_name}"
            )

    step(f"Approving pending InstallPlan {installplan_name}")
    run_command(
        [
            kubectl_cmd,
            "patch",
            "installplan",
            str(installplan_name),
            "-n",
            namespace,
            "--type",
            "merge",
            "-p",
            '{"spec":{"approved":true}}',
        ]
    )


def _wait_for_resource(
    kubectl_cmd: str, argv: list[str], *, timeout_seconds: int, label: str
) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _resource_exists([kubectl_cmd, *argv]):
            return
        time.sleep(5)
    raise CommandError(f"timed out waiting for {label}")


def _wait_for_labeled_pods_ready(
    kubectl_cmd: str,
    *,
    namespace: str,
    label_selector: str,
    timeout_seconds: int,
    label: str,
) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        payload = run_json_command(
            [
                kubectl_cmd,
                "get",
                "pods",
                "-n",
                namespace,
                "-l",
                label_selector,
                "-o",
                "json",
            ]
        )
        items = payload.get("items", []) or []
        if not items:
            time.sleep(5)
            continue
        ready_pods = 0
        for item in items:
            conditions = item.get("status", {}).get("conditions", []) or []
            if any(
                condition.get("type") == "Ready" and condition.get("status") == "True"
                for condition in conditions
            ):
                ready_pods += 1
        if ready_pods:
            return
        time.sleep(5)
    raise CommandError(f"timed out waiting for {label}")


def _wait_for_gateway_ready(
    kubectl_cmd: str,
    *,
    namespace: str,
    name: str,
    timeout_seconds: int,
) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        payload = run_json_command(
            [kubectl_cmd, "get", "gateway", name, "-n", namespace, "-o", "json"]
        )
        conditions = payload.get("status", {}).get("conditions", []) or []
        accepted = any(
            item.get("type") == "Accepted" and item.get("status") == "True"
            for item in conditions
        )
        programmed = any(
            item.get("type") == "Programmed" and item.get("status") == "True"
            for item in conditions
        )
        if accepted and programmed:
            return
        time.sleep(5)
    raise CommandError(
        f"timed out waiting for Gateway {namespace}/{name} to become ready"
    )


def setup_rhoai(
    plan: ResolvedRunPlan, *, state_path: Path | None = None
) -> dict[str, Any]:
    kubectl_cmd = require_any_command("oc", "kubectl")
    requested_version = (
        str(plan.deployment.platform_version or "").strip()
        or DEFAULT_RHOAI_PLATFORM_VERSION
    )
    channel_name = _rhoai_channel(plan)
    state = _empty_state(requested_version)
    state["platform_channel"] = channel_name
    _persist_state(state, state_path)

    if _operator_subscription_exists(kubectl_cmd):
        detail("RHOAI operator subscription already present")
    else:
        step("Installing the RHOAI operator")
        state["operator_namespace_created"] = _ensure_namespace(
            kubectl_cmd, RHOAI_OPERATOR_NAMESPACE
        )
        _persist_state(state, state_path)
        source, source_namespace, starting_csv = _resolve_rhoai_operator_package(
            kubectl_cmd, requested_version, channel_name
        )
        detail(
            f"Installing {RHOAI_OPERATOR_PACKAGE_NAME} from {source}/{channel_name} "
            f"with startingCSV {starting_csv}"
        )
        documents = render_yaml_documents(
            "setup/rhoai/operator.yaml",
            {
                "SOURCE": source,
                "SOURCE_NAMESPACE": source_namespace,
                "CHANNEL": channel_name,
                "STARTING_CSV": starting_csv,
            },
        )
        operatorgroups = _operatorgroups_in_namespace(
            kubectl_cmd, RHOAI_OPERATOR_NAMESPACE
        )
        if len(operatorgroups) > 1:
            names = ", ".join(
                str(item.get("metadata", {}).get("name", "unknown"))
                for item in operatorgroups
            )
            raise CommandError(
                f"namespace {RHOAI_OPERATOR_NAMESPACE} already has multiple OperatorGroups "
                f"({names}); clean the namespace and rerun setup"
            )
        if operatorgroups:
            existing_name = str(
                operatorgroups[0].get("metadata", {}).get("name", "unknown")
            )
            detail(f"Reusing existing OperatorGroup {existing_name}")
            documents = [
                document
                for document in documents
                if document.get("kind") != "OperatorGroup"
            ]
        else:
            state["operatorgroup_managed"] = True
        _apply_documents(kubectl_cmd, documents)
        state["operator_subscription_managed"] = True
        _persist_state(state, state_path)
        csv_name = _wait_for_subscription_current_csv(
            kubectl_cmd, namespace=RHOAI_OPERATOR_NAMESPACE, timeout_seconds=900
        )
        state["operator_csv_name"] = csv_name
        _persist_state(state, state_path)
        _wait_for_csv_succeeded(
            kubectl_cmd,
            namespace=RHOAI_OPERATOR_NAMESPACE,
            csv_name=csv_name,
            timeout_seconds=1800,
        )

    _wait_for_resource(
        kubectl_cmd,
        [
            "get",
            "crd",
            "datascienceclusters.datasciencecluster.opendatahub.io",
            "-o",
            "name",
        ],
        timeout_seconds=600,
        label="RHOAI DataScienceCluster CRD",
    )

    if _datasciencecluster_exists(kubectl_cmd):
        detail("DataScienceCluster default-dsc already present")
    else:
        step("Creating the DataScienceCluster")
        documents = render_yaml_documents("setup/rhoai/datasciencecluster.yaml", {})
        _apply_documents(kubectl_cmd, documents)
        state["datasciencecluster_managed"] = True
        _persist_state(state, state_path)

    _wait_for_resource(
        kubectl_cmd,
        ["get", "datasciencecluster", RHOAI_DSC_NAME, "-o", "name"],
        timeout_seconds=600,
        label="DataScienceCluster default-dsc",
    )
    _wait_for_resource(
        kubectl_cmd,
        ["get", "namespace", RHOAI_APPLICATIONS_NAMESPACE, "-o", "name"],
        timeout_seconds=900,
        label=f"namespace {RHOAI_APPLICATIONS_NAMESPACE}",
    )
    step("Waiting for RHOAI serving controllers to become ready")
    _wait_for_labeled_pods_ready(
        kubectl_cmd,
        namespace=RHOAI_APPLICATIONS_NAMESPACE,
        label_selector="app=odh-model-controller",
        timeout_seconds=900,
        label="odh-model-controller pods",
    )
    _wait_for_labeled_pods_ready(
        kubectl_cmd,
        namespace=RHOAI_APPLICATIONS_NAMESPACE,
        label_selector="control-plane=kserve-controller-manager",
        timeout_seconds=900,
        label="kserve-controller-manager pods",
    )

    gatewayclass_documents = render_yaml_documents("setup/rhoai/gatewayclass.yaml", {})
    gatewayclass_document = gatewayclass_documents[0]
    if _gatewayclass_exists(kubectl_cmd):
        step("Reconciling the RHOAI GatewayClass")
    else:
        step("Creating the RHOAI GatewayClass")
        state["gatewayclass_managed"] = True
        _persist_state(state, state_path)
    _apply_documents(kubectl_cmd, [gatewayclass_document])

    _wait_for_resource(
        kubectl_cmd,
        ["get", "gatewayclass", RHOAI_GATEWAYCLASS_NAME, "-o", "name"],
        timeout_seconds=900,
        label=f"GatewayClass {RHOAI_GATEWAYCLASS_NAME}",
    )
    gateway_documents = render_yaml_documents(
        "setup/rhoai/gateway.yaml",
        {
            "HOSTNAME": _infer_rhoai_gateway_hostname(kubectl_cmd),
            "TLS_SECRET_NAME": _rhoai_gateway_tls_secret_name(kubectl_cmd),
        },
    )
    gateway_document = gateway_documents[0]

    if _gateway_exists(kubectl_cmd):
        step("Reconciling the RHOAI Gateway")
    else:
        step("Creating the RHOAI Gateway")
        state["gateway_managed"] = True
        _persist_state(state, state_path)
    _apply_documents(kubectl_cmd, [gateway_document])

    step("Waiting for the RHOAI Gateway to become ready")
    _wait_for_gateway_ready(
        kubectl_cmd,
        namespace=RHOAI_GATEWAY_NAMESPACE,
        name=RHOAI_GATEWAY_NAME,
        timeout_seconds=900,
    )

    success("RHOAI platform prerequisites are ready")
    return state


def teardown_rhoai(plan: ResolvedRunPlan | None, state: dict[str, Any]) -> None:
    if not state or state.get("platform") != "rhoai":
        detail("No RHOAI setup state found; skipping teardown")
        return

    kubectl_cmd = require_any_command("oc", "kubectl")

    if state.get("gateway_managed"):
        step("Removing the RHOAI Gateway")
        run_command(
            [
                kubectl_cmd,
                "delete",
                "gateway",
                RHOAI_GATEWAY_NAME,
                "-n",
                RHOAI_GATEWAY_NAMESPACE,
                "--ignore-not-found=true",
            ]
        )

    if state.get("gatewayclass_managed"):
        step("Removing the RHOAI GatewayClass")
        run_command(
            [
                kubectl_cmd,
                "delete",
                "gatewayclass",
                RHOAI_GATEWAYCLASS_NAME,
                "--ignore-not-found=true",
            ]
        )

    if state.get("datasciencecluster_managed"):
        step("Removing the DataScienceCluster")
        run_command(
            [
                kubectl_cmd,
                "delete",
                "datasciencecluster",
                RHOAI_DSC_NAME,
                "--ignore-not-found=true",
            ]
        )

    if state.get("operator_subscription_managed"):
        step("Removing the RHOAI operator subscription")
        run_command(
            [
                kubectl_cmd,
                "delete",
                "subscription",
                RHOAI_OPERATOR_SUBSCRIPTION_NAME,
                "-n",
                RHOAI_OPERATOR_NAMESPACE,
                "--ignore-not-found=true",
            ]
        )

    csv_name = str(state.get("operator_csv_name") or "").strip()
    if csv_name:
        step(f"Removing CSV {csv_name}")
        run_command(
            [
                kubectl_cmd,
                "delete",
                "csv",
                csv_name,
                "-n",
                RHOAI_OPERATOR_NAMESPACE,
                "--ignore-not-found=true",
            ],
            check=False,
        )

    if state.get("operatorgroup_managed"):
        step("Removing the RHOAI OperatorGroup")
        run_command(
            [
                kubectl_cmd,
                "delete",
                "operatorgroup",
                RHOAI_OPERATORGROUP_NAME,
                "-n",
                RHOAI_OPERATOR_NAMESPACE,
                "--ignore-not-found=true",
            ],
            check=False,
        )

    if state.get("operator_namespace_created"):
        step(f"Removing namespace {RHOAI_OPERATOR_NAMESPACE}")
        run_command(
            [
                kubectl_cmd,
                "delete",
                "namespace",
                RHOAI_OPERATOR_NAMESPACE,
                "--wait=false",
                "--ignore-not-found=true",
            ],
            check=False,
        )

    success("RHOAI platform setup has been torn down")
