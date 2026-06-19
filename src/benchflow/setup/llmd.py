from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import yaml

from ..assets import render_yaml_documents
from ..cluster import (
    CommandError,
    require_any_command,
    require_command,
    run_command,
    run_json_command,
)
from ..models import ResolvedRunPlan
from ..repository import clone_repo
from ..ui import detail, step, success

RECIPE_LAYOUT_ISTIO_VERSION = "1.29.2"


def _empty_state(plan: ResolvedRunPlan) -> dict[str, Any]:
    return {
        "apiVersion": "benchflow.io/v1alpha1",
        "kind": "SetupState",
        "platform": "llm-d",
        "gateway": plan.deployment.gateway,
        "repo_url": plan.deployment.repo_url,
        "repo_ref": plan.deployment.repo_ref,
        "repo_head": "",
        "namespace": plan.deployment.namespace,
        "gateway_dependencies_managed": False,
        "istio_releases_managed": False,
        "patched_istio_crds": [],
        "restorable_manifests": [],
    }


def _persist_state(state: dict[str, Any], state_path: Path | None) -> None:
    if state_path is None:
        return
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def load_setup_state(state_path: Path | None) -> dict[str, Any]:
    if state_path is None or not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))


def _gateway_provider_dir(checkout_root: Path) -> Path:
    return checkout_root / "guides" / "recipes" / "gateway"


def _legacy_gateway_provider_dir(checkout_root: Path) -> Path:
    return checkout_root / "guides" / "prereq" / "gateway-provider"


def _uses_legacy_gateway_provider(checkout_root: Path) -> bool:
    gateway_provider_dir = _legacy_gateway_provider_dir(checkout_root)
    return (
        gateway_provider_dir.exists()
        and (gateway_provider_dir / "install-gateway-provider-dependencies.sh").exists()
    )


def _clone_llmd_repo_source(
    *,
    repo_url: str,
    repo_ref: str,
    workspace_dir: Path | None,
) -> tuple[Path, bool, str]:
    created_tempdir = workspace_dir is None
    checkout_root = (
        workspace_dir
        if workspace_dir is not None
        else Path(tempfile.mkdtemp(prefix="benchflow-llmd-setup-"))
    )
    checkout_dir = checkout_root / "llm-d-repo"
    step(f"Cloning llm-d platform setup sources from {repo_url} at {repo_ref}")
    clone_repo(
        url=repo_url,
        revision=repo_ref,
        output_dir=checkout_dir,
        delete_existing=True,
    )
    head = run_command(
        ["git", "rev-parse", "HEAD"], cwd=checkout_dir, capture_output=True
    ).stdout.strip()
    return checkout_dir, created_tempdir, head


def _clone_llmd_repo(
    plan: ResolvedRunPlan, workspace_dir: Path | None
) -> tuple[Path, bool, str]:
    return _clone_llmd_repo_source(
        repo_url=plan.deployment.repo_url,
        repo_ref=plan.deployment.repo_ref,
        workspace_dir=workspace_dir,
    )


def resolve_llmd_repo_head(
    *,
    repo_url: str,
    repo_ref: str,
    workspace_dir: Path | None = None,
) -> str:
    checkout_dir, created_tempdir, repo_head = _clone_llmd_repo_source(
        repo_url=repo_url,
        repo_ref=repo_ref,
        workspace_dir=workspace_dir,
    )
    try:
        return repo_head
    finally:
        if created_tempdir:
            shutil.rmtree(checkout_dir.parent, ignore_errors=True)


def _resource_manifest(
    kubectl_cmd: str, kind: str, name: str, namespace: str
) -> dict[str, Any] | None:
    result = run_command(
        [
            kubectl_cmd,
            "get",
            kind,
            name,
            "-n",
            namespace,
            "-o",
            "yaml",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None

    manifest = yaml.safe_load(result.stdout or "")
    if not isinstance(manifest, dict):
        return None
    manifest.pop("status", None)
    metadata = manifest.get("metadata", {}) or {}
    for key in (
        "creationTimestamp",
        "deletionGracePeriodSeconds",
        "deletionTimestamp",
        "generation",
        "managedFields",
        "resourceVersion",
        "selfLink",
        "uid",
    ):
        metadata.pop(key, None)
    if not metadata.get("annotations"):
        metadata.pop("annotations", None)
    if not metadata.get("labels"):
        metadata.pop("labels", None)
    manifest["metadata"] = metadata
    return manifest


def _delete_if_exists(kubectl_cmd: str, kind: str, name: str, namespace: str) -> None:
    run_command(
        [
            kubectl_cmd,
            "delete",
            kind,
            name,
            "-n",
            namespace,
            "--ignore-not-found=true",
        ]
    )


def _namespace_exists(kubectl_cmd: str, namespace: str) -> bool:
    result = run_command(
        [kubectl_cmd, "get", "namespace", namespace, "-o", "name"],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _ensure_namespace(kubectl_cmd: str, namespace: str) -> None:
    if _namespace_exists(kubectl_cmd, namespace):
        return
    run_command([kubectl_cmd, "create", "namespace", namespace])


def _apply_runner_rbac_in_istio_system(
    kubectl_cmd: str, benchflow_namespace: str
) -> None:
    documents = render_yaml_documents(
        "bootstrap/rbac/runner-istio-system.yaml",
        {"BENCHFLOW_NAMESPACE": benchflow_namespace},
    )
    run_command(
        [kubectl_cmd, "apply", "-f", "-"],
        input_text=yaml.safe_dump_all(documents, sort_keys=False),
    )


def _helm_release_names(namespace: str) -> set[str]:
    try:
        payload = run_json_command(["helm", "list", "-n", namespace, "-o", "json"])
    except CommandError as exc:
        if "secrets is forbidden" in str(exc):
            raise CommandError(
                f"missing RBAC to inspect Helm releases in namespace {namespace}; "
                "rerun `bflow bootstrap` to reconcile the runner permissions"
            ) from exc
        raise
    return {str(item.get("name") or "") for item in payload}


def _llmd_inference_pool_crd_name(repo_ref: str) -> str:
    # Temporary compatibility fix: llm-d v0.4.x inference-scheduling still
    # expects the legacy x-k8s InferencePool CRD, while newer refs rely on the
    # promoted inference.networking.k8s.io API group.
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)(?:[-+].*)?", repo_ref.strip())
    if match is None:
        return "inferencepools.inference.networking.k8s.io"
    version = tuple(int(part) for part in match.groups())
    if version <= (0, 4, 0):
        return "inferencepools.inference.networking.x-k8s.io"
    return "inferencepools.inference.networking.k8s.io"


def _gateway_dependencies_present(kubectl_cmd: str, repo_ref: str) -> bool:
    required_crds = (
        "gateways.gateway.networking.k8s.io",
        _llmd_inference_pool_crd_name(repo_ref),
    )
    for crd_name in required_crds:
        result = run_command(
            [kubectl_cmd, "get", "crd", crd_name, "-o", "name"],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return False
    return True


def llmd_platform_present(kubectl_cmd: str) -> bool:
    if shutil.which("helm") is not None:
        if {"istio-base", "istiod"}.issubset(_helm_release_names("istio-system")):
            return True
    for crd_name in (
        "inferencepools.inference.networking.x-k8s.io",
        "inferencepools.inference.networking.k8s.io",
    ):
        result = run_command(
            [kubectl_cmd, "get", "crd", crd_name, "-o", "name"],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return True
    return False


def _istio_crd_metadata(kubectl_cmd: str) -> list[dict[str, Any]]:
    payload = run_json_command([kubectl_cmd, "get", "crd", "-o", "json"])
    snapshots: list[dict[str, Any]] = []
    for item in payload.get("items", []):
        metadata = item.get("metadata", {}) or {}
        name = str(metadata.get("name") or "")
        if "istio.io" not in name:
            continue
        snapshots.append(
            {
                "name": name,
                "labels": dict(metadata.get("labels") or {}),
                "annotations": dict(metadata.get("annotations") or {}),
            }
        )
    return snapshots


def _patch_istio_crds_for_helm(
    kubectl_cmd: str, state: dict[str, Any], state_path: Path | None
) -> None:
    snapshots = _istio_crd_metadata(kubectl_cmd)
    state["patched_istio_crds"] = snapshots
    _persist_state(state, state_path)
    for snapshot in snapshots:
        run_command(
            [
                kubectl_cmd,
                "patch",
                "crd",
                snapshot["name"],
                "--type=merge",
                "-p",
                '{"metadata":{"labels":{"app.kubernetes.io/managed-by":"Helm"},"annotations":{"meta.helm.sh/release-name":"istio-base","meta.helm.sh/release-namespace":"istio-system"}}}',
            ]
        )


def _restore_istio_crd_metadata(
    kubectl_cmd: str, snapshots: list[dict[str, Any]]
) -> None:
    for snapshot in snapshots:
        labels = snapshot.get("labels") or {}
        annotations = snapshot.get("annotations") or {}
        patch = {
            "metadata": {
                "labels": {
                    "app.kubernetes.io/managed-by": labels.get(
                        "app.kubernetes.io/managed-by"
                    )
                },
                "annotations": {
                    "meta.helm.sh/release-name": annotations.get(
                        "meta.helm.sh/release-name"
                    ),
                    "meta.helm.sh/release-namespace": annotations.get(
                        "meta.helm.sh/release-namespace"
                    ),
                },
            }
        }
        run_command(
            [
                kubectl_cmd,
                "patch",
                "crd",
                str(snapshot["name"]),
                "--type=merge",
                "-p",
                json.dumps(patch, separators=(",", ":")),
            ],
            check=False,
        )


def _restore_manifest(kubectl_cmd: str, manifest: dict[str, Any]) -> None:
    run_command(
        [kubectl_cmd, "apply", "-f", "-"],
        input_text=yaml.safe_dump(manifest, sort_keys=False),
    )


def _run_gateway_provider_script(gateway_provider_dir: Path, mode: str) -> None:
    run_command(
        ["bash", "./install-gateway-crds.sh", mode],
        cwd=gateway_provider_dir,
    )


def _run_legacy_gateway_provider_script(
    gateway_provider_dir: Path, action: str
) -> None:
    run_command(
        ["bash", "./install-gateway-provider-dependencies.sh", action],
        cwd=gateway_provider_dir,
    )


def _run_istio_helmfile(gateway_provider_dir: Path, action: str) -> None:
    helmfile_path = gateway_provider_dir / "istio.helmfile.yaml"
    if not helmfile_path.exists():
        raise CommandError(
            f"checked-out llm-d ref does not ship istio.helmfile.yaml in {gateway_provider_dir}; "
            "automatic repo-managed Istio installation is not available for this ref"
        )
    run_command(
        ["helmfile", "-f", "istio.helmfile.yaml", action],
        cwd=gateway_provider_dir,
    )


def _istio_gateway_api_extension_enabled(kubectl_cmd: str) -> bool:
    result = run_command(
        [
            kubectl_cmd,
            "get",
            "deployment",
            "istiod",
            "-n",
            "istio-system",
            "-o",
            "json",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    payload = json.loads(result.stdout or "{}")
    containers = (
        payload.get("spec", {})
        .get("template", {})
        .get("spec", {})
        .get("containers", [])
    )
    for container in containers:
        if not isinstance(container, dict):
            continue
        for env_var in container.get("env", []) or []:
            if not isinstance(env_var, dict):
                continue
            if str(env_var.get("name") or "").strip() != (
                "ENABLE_GATEWAY_API_INFERENCE_EXTENSION"
            ):
                continue
            return str(env_var.get("value") or "").strip().lower() in {
                "1",
                "true",
                "yes",
            }
    return False


def _ensure_preinstalled_istio_for_recipe_layout(kubectl_cmd: str) -> None:
    if not _istio_gateway_api_extension_enabled(kubectl_cmd):
        raise CommandError(
            "current llm-d refs require Istio to be preinstalled outside BenchFlow "
            "with ENABLE_GATEWAY_API_INFERENCE_EXTENSION=true; install it with "
            "istioctl before using gateway=istio"
        )
    _wait_for_istiod(kubectl_cmd, timeout_seconds=120)


def _wait_for_istiod(kubectl_cmd: str, timeout_seconds: int) -> None:
    run_command(
        [
            kubectl_cmd,
            "wait",
            "--for=condition=ready",
            "pod",
            "-l",
            "app=istiod",
            "-n",
            "istio-system",
            f"--timeout={timeout_seconds}s",
        ]
    )


def _istiod_present(kubectl_cmd: str) -> bool:
    result = run_command(
        [
            kubectl_cmd,
            "get",
            "deployment",
            "istiod",
            "-n",
            "istio-system",
            "-o",
            "name",
        ],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _download_istio_install_script(target_path: Path) -> None:
    try:
        with urlopen("https://istio.io/downloadIstio", timeout=60) as response:
            target_path.write_bytes(response.read())
    except URLError as exc:
        raise CommandError(
            "failed to download the official Istio installer from https://istio.io/downloadIstio"
        ) from exc


def _with_env(base: dict[str, str], extra: dict[str, str]) -> dict[str, str]:
    env = dict(base)
    env.update(extra)
    return env


def _download_istioctl(workspace_dir: Path) -> Path:
    install_script = workspace_dir / "downloadIstio"
    _download_istio_install_script(install_script)
    run_command(
        ["bash", str(install_script)],
        cwd=workspace_dir,
        env=_with_env(os.environ, {"ISTIO_VERSION": RECIPE_LAYOUT_ISTIO_VERSION}),
    )
    istioctl_path = (
        workspace_dir / f"istio-{RECIPE_LAYOUT_ISTIO_VERSION}" / "bin" / "istioctl"
    )
    if not istioctl_path.exists():
        raise CommandError(
            f"expected istioctl not found after download: {istioctl_path}"
        )
    return istioctl_path


def _install_recipe_layout_istio(kubectl_cmd: str) -> None:
    tempdir = Path(tempfile.mkdtemp(prefix="benchflow-istioctl-"))
    try:
        istioctl_path = _download_istioctl(tempdir)
        run_command(
            [
                str(istioctl_path),
                "install",
                "-y",
                "--set",
                "values.pilot.env.ENABLE_GATEWAY_API_INFERENCE_EXTENSION=true",
            ]
        )
        _wait_for_istiod(kubectl_cmd, timeout_seconds=120)
        if not _istio_gateway_api_extension_enabled(kubectl_cmd):
            raise CommandError(
                "Istio installed but ENABLE_GATEWAY_API_INFERENCE_EXTENSION was not enabled"
            )
    finally:
        shutil.rmtree(tempdir, ignore_errors=True)


def _uninstall_recipe_layout_istio() -> None:
    tempdir = Path(tempfile.mkdtemp(prefix="benchflow-istioctl-"))
    try:
        istioctl_path = _download_istioctl(tempdir)
        run_command([str(istioctl_path), "uninstall", "--purge", "-y"])
    finally:
        shutil.rmtree(tempdir, ignore_errors=True)


def reset_llmd_platform(
    *,
    repo_url: str,
    repo_ref: str,
    gateway: str = "istio",
    workspace_dir: Path | None = None,
) -> None:
    if gateway not in {"istio", "standalone"}:
        raise CommandError(
            "llm-d cleanup currently supports gateway=istio or "
            f"gateway=standalone, got {gateway}"
        )

    require_command("bash")
    require_command("git")
    kubectl_cmd = require_any_command("oc", "kubectl")

    checkout_dir, created_tempdir, _repo_head = _clone_llmd_repo_source(
        repo_url=repo_url,
        repo_ref=repo_ref,
        workspace_dir=workspace_dir,
    )
    try:
        legacy_gateway_provider = _uses_legacy_gateway_provider(checkout_dir)
        gateway_provider_dir = (
            _legacy_gateway_provider_dir(checkout_dir)
            if legacy_gateway_provider
            else _gateway_provider_dir(checkout_dir)
        )
        if (
            gateway == "istio"
            and legacy_gateway_provider
            and {
                "istio-base",
                "istiod",
            }.issubset(_helm_release_names("istio-system"))
        ):
            require_command("helm")
            require_command("helmfile")
            step("Removing upstream Istio before switching platforms")
            _run_istio_helmfile(gateway_provider_dir, "destroy")
        if llmd_platform_present(kubectl_cmd):
            step("Removing Gateway API and GAIE CRDs before switching platforms")
            if legacy_gateway_provider:
                _run_legacy_gateway_provider_script(gateway_provider_dir, "delete")
            else:
                _run_gateway_provider_script(gateway_provider_dir, "delete")
        success("llm-d platform prerequisites have been reset")
    finally:
        if created_tempdir:
            shutil.rmtree(checkout_dir.parent, ignore_errors=True)


def setup_llmd(
    plan: ResolvedRunPlan,
    *,
    workspace_dir: Path | None = None,
    state_path: Path | None = None,
) -> dict[str, Any]:
    gateway = str(plan.deployment.gateway or "").strip()
    if gateway not in {"istio", "standalone"}:
        raise CommandError(
            "llm-d setup currently supports gateway=istio or "
            f"gateway=standalone, got {plan.deployment.gateway}"
        )

    require_command("bash")
    require_command("git")
    kubectl_cmd = require_any_command("oc", "kubectl")

    state = _empty_state(plan)
    _persist_state(state, state_path)

    checkout_dir, created_tempdir, repo_head = _clone_llmd_repo(plan, workspace_dir)
    try:
        legacy_gateway_provider = _uses_legacy_gateway_provider(checkout_dir)
        gateway_provider_dir = (
            _legacy_gateway_provider_dir(checkout_dir)
            if legacy_gateway_provider
            else _gateway_provider_dir(checkout_dir)
        )
        if not gateway_provider_dir.exists():
            raise CommandError(
                f"expected llm-d gateway setup directory not found: {gateway_provider_dir}"
            )
        detail(f"Gateway setup directory: {gateway_provider_dir}")
        state["repo_head"] = repo_head
        _persist_state(state, state_path)

        gateway_dependencies_present_before = _gateway_dependencies_present(
            kubectl_cmd, plan.deployment.repo_ref
        )
        if gateway_dependencies_present_before:
            detail("Gateway API and GAIE CRD markers already present")
        else:
            step("Installing Gateway API and GAIE CRDs")
            if legacy_gateway_provider:
                _run_legacy_gateway_provider_script(gateway_provider_dir, "apply")
            else:
                _run_gateway_provider_script(gateway_provider_dir, "apply")
            state["gateway_dependencies_managed"] = True
            _persist_state(state, state_path)

        if gateway == "standalone":
            success("llm-d standalone platform prerequisites are ready")
            return state

        step("Ensuring llm-d Istio namespace and runner RBAC")
        _ensure_namespace(kubectl_cmd, "istio-system")
        _apply_runner_rbac_in_istio_system(kubectl_cmd, plan.deployment.namespace)

        if not legacy_gateway_provider:
            if _istiod_present(kubectl_cmd):
                step("Validating existing Istio for current llm-d gateway flow")
                _ensure_preinstalled_istio_for_recipe_layout(kubectl_cmd)
            else:
                step("Installing Istio for current llm-d gateway flow")
                _install_recipe_layout_istio(kubectl_cmd)
                state["istio_releases_managed"] = True
                _persist_state(state, state_path)
            success("llm-d platform prerequisites are ready")
            return state

        if _istiod_present(kubectl_cmd):
            step("Validating existing Istio for legacy llm-d gateway flow")
            _ensure_preinstalled_istio_for_recipe_layout(kubectl_cmd)
            success("llm-d platform prerequisites are ready")
            return state

        require_command("helm")
        require_command("helmfile")
        istio_releases_present_before = {
            "istio-base",
            "istiod",
        }.issubset(_helm_release_names("istio-system"))
        if istio_releases_present_before:
            detail("Upstream Istio releases already present in istio-system")
            success("llm-d platform prerequisites are ready")
            return state

        step("Snapshotting and removing conflicting OpenShift Service Mesh resources")
        smm_manifest = _resource_manifest(
            kubectl_cmd, "servicemeshmember", "default", plan.deployment.namespace
        )
        if smm_manifest is not None:
            state["restorable_manifests"].append(smm_manifest)
            _persist_state(state, state_path)
            _delete_if_exists(
                kubectl_cmd, "servicemeshmember", "default", plan.deployment.namespace
            )

        smcp_manifest = _resource_manifest(
            kubectl_cmd, "servicemeshcontrolplane", "data-science-smcp", "istio-system"
        )
        if smcp_manifest is not None:
            state["restorable_manifests"].append(smcp_manifest)
            _persist_state(state, state_path)
            _delete_if_exists(
                kubectl_cmd,
                "servicemeshcontrolplane",
                "data-science-smcp",
                "istio-system",
            )
            detail("Waiting 30 seconds for Service Mesh cleanup to settle")
            time.sleep(30)

        step("Patching Istio CRDs for Helm ownership")
        _patch_istio_crds_for_helm(kubectl_cmd, state, state_path)

        step("Installing upstream Istio with Gateway API Inference Extension")
        _run_istio_helmfile(gateway_provider_dir, "sync")
        state["istio_releases_managed"] = True
        _persist_state(state, state_path)

        step("Waiting for upstream Istio to become ready")
        _wait_for_istiod(kubectl_cmd, timeout_seconds=120)
        success("llm-d platform prerequisites are ready")
        return state
    finally:
        if created_tempdir:
            shutil.rmtree(checkout_dir.parent, ignore_errors=True)


def teardown_llmd(
    plan: ResolvedRunPlan | None,
    state: dict[str, Any],
    *,
    workspace_dir: Path | None = None,
) -> None:
    if not state or state.get("platform") != "llm-d":
        detail("No llm-d setup state found; skipping teardown")
        return
    gateway = str(
        state.get("gateway")
        or (plan.deployment.gateway if plan is not None else "")
        or "istio"
    ).strip()
    if gateway not in {"istio", "standalone"}:
        detail(f"Skipping llm-d teardown because gateway={gateway} is not managed here")
        return

    require_command("bash")
    require_command("git")
    kubectl_cmd = require_any_command("oc", "kubectl")

    repo_url = str(
        state.get("repo_url")
        or (plan.deployment.repo_url if plan is not None else "")
        or "https://github.com/llm-d/llm-d.git"
    ).strip()
    repo_ref = str(
        state.get("repo_ref")
        or (plan.deployment.repo_ref if plan is not None else "")
        or "main"
    ).strip()

    checkout_dir, created_tempdir, _repo_head = _clone_llmd_repo_source(
        repo_url=repo_url,
        repo_ref=repo_ref,
        workspace_dir=workspace_dir,
    )
    try:
        legacy_gateway_provider = _uses_legacy_gateway_provider(checkout_dir)
        gateway_provider_dir = (
            _legacy_gateway_provider_dir(checkout_dir)
            if legacy_gateway_provider
            else _gateway_provider_dir(checkout_dir)
        )

        if gateway == "istio" and state.get("istio_releases_managed"):
            step("Removing upstream Istio installed during llm-d setup")
            if legacy_gateway_provider:
                require_command("helm")
                require_command("helmfile")
                _run_istio_helmfile(gateway_provider_dir, "destroy")
            else:
                _uninstall_recipe_layout_istio()

        patched_crds = list(state.get("patched_istio_crds") or [])
        if patched_crds:
            step("Restoring original Istio CRD metadata")
            _restore_istio_crd_metadata(kubectl_cmd, patched_crds)

        if state.get("gateway_dependencies_managed"):
            step("Removing Gateway API and GAIE CRDs installed during llm-d setup")
            if legacy_gateway_provider:
                _run_legacy_gateway_provider_script(gateway_provider_dir, "delete")
            else:
                _run_gateway_provider_script(gateway_provider_dir, "delete")

        restorable_manifests = list(state.get("restorable_manifests") or [])
        if restorable_manifests:
            step("Restoring previously removed Service Mesh resources")
            for manifest in restorable_manifests:
                _restore_manifest(kubectl_cmd, manifest)

        success("llm-d platform setup has been torn down")
    finally:
        if created_tempdir:
            shutil.rmtree(checkout_dir.parent, ignore_errors=True)
