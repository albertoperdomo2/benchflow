from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .cluster import CommandError, require_any_command, run_command, run_json_command
from .models import ResolvedRunPlan
from .ui import detail, step, success

RHOAI_PROFILER_OUTPUT_DIR = "/tmp/benchflow-profiler"


def _release_token_matches(candidate: str, release_name: str) -> bool:
    if candidate == release_name:
        return True
    for separator in ("-", ".", "/", "_"):
        if candidate.startswith(f"{release_name}{separator}"):
            return True
        if candidate.endswith(f"{separator}{release_name}"):
            return True
        if f"{separator}{release_name}{separator}" in candidate:
            return True
    return False


def _matches_release(metadata: dict[str, object], release_name: str) -> bool:
    name = metadata.get("name")
    if isinstance(name, str) and _release_token_matches(name, release_name):
        return True

    labels = metadata.get("labels")
    if isinstance(labels, dict):
        for value in labels.values():
            if isinstance(value, str) and _release_token_matches(value, release_name):
                return True

    owner_references = metadata.get("ownerReferences")
    if isinstance(owner_references, list):
        for owner in owner_references:
            if not isinstance(owner, dict):
                continue
            owner_name = owner.get("name")
            if isinstance(owner_name, str) and _release_token_matches(
                owner_name, release_name
            ):
                return True

    return False


def _pod_type(pod_name: str) -> str:
    lowered = pod_name.lower()
    if any(token in lowered for token in ("gaie", "scheduler", "epp", "kserve-router")):
        return "gaie"
    if any(
        token in lowered
        for token in (
            "ms-",
            "model-service",
            "inference",
            "decode",
            "prefill",
            "predictor",
            "kserve-",
        )
    ):
        return "model"
    return "infra"


def _collect_pod_logs(
    kubectl_cmd: str, namespace: str, pod_name: str, log_dir: Path
) -> bool:
    try:
        payload = run_json_command(
            [kubectl_cmd, "get", "pod", pod_name, "-n", namespace, "-o", "json"]
        )
    except CommandError:
        return False
    containers = [
        entry.get("name")
        for entry in payload.get("spec", {}).get("containers", [])
        if entry.get("name")
    ]
    has_logs = False
    for container in containers:
        log_file = log_dir / f"{pod_name}_{container}.log"
        result = run_command(
            [kubectl_cmd, "logs", pod_name, "-c", container, "-n", namespace],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            log_file.write_text(result.stdout, encoding="utf-8")
            has_logs = True
        elif log_file.exists():
            log_file.unlink()
    return has_logs


def _ensure_artifact_layout(artifacts_dir: Path) -> None:
    for relative in (
        "logs/pipeline",
        "logs/model",
        "logs/gaie",
        "logs/infra",
        "manifests",
        "profiling",
    ):
        (artifacts_dir / relative).mkdir(parents=True, exist_ok=True)


def _collect_profiling_artifacts(
    kubectl_cmd: str,
    namespace: str,
    pod_name: str,
    artifacts_dir: Path,
) -> int:
    probe = run_command(
        [
            kubectl_cmd,
            "exec",
            pod_name,
            "-c",
            "main",
            "-n",
            namespace,
            "--",
            "sh",
            "-lc",
            (
                f"test -d {RHOAI_PROFILER_OUTPUT_DIR} && "
                f"find {RHOAI_PROFILER_OUTPUT_DIR} -maxdepth 1 -type f | head -n 1"
            ),
        ],
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        return 0

    target_dir = artifacts_dir / "profiling" / pod_name
    target_dir.mkdir(parents=True, exist_ok=True)
    copy_result = run_command(
        [
            kubectl_cmd,
            "cp",
            "-c",
            "main",
            "-n",
            namespace,
            f"{pod_name}:{RHOAI_PROFILER_OUTPUT_DIR}/.",
            str(target_dir),
        ],
        capture_output=True,
        check=False,
    )
    if copy_result.returncode != 0:
        detail(f"Failed to copy profiling artifacts from pod {pod_name}")
        return 0
    return sum(1 for path in target_dir.rglob("*") if path.is_file())


def collect_execution_logs(
    plan: ResolvedRunPlan,
    *,
    artifacts_dir: Path,
    execution_name: str,
) -> int:
    if not execution_name:
        return 0
    kubectl_cmd = require_any_command("oc", "kubectl")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    _ensure_artifact_layout(artifacts_dir)

    namespace = plan.deployment.namespace
    detail("Collecting execution pod logs")
    seen_execution_pods: set[str] = set()
    payload = run_json_command(
        [
            kubectl_cmd,
            "get",
            "pods",
            "-n",
            namespace,
            "-l",
            f"tekton.dev/pipelineRun={execution_name}",
            "-o",
            "json",
        ]
    )
    for item in payload.get("items", []):
        pod_name = item.get("metadata", {}).get("name", "")
        if pod_name:
            seen_execution_pods.add(pod_name)

    execution_pod_count = 0
    for pod_name in sorted(seen_execution_pods):
        if _collect_pod_logs(
            kubectl_cmd, namespace, pod_name, artifacts_dir / "logs" / "pipeline"
        ):
            execution_pod_count += 1
    detail(f"Collected logs from {execution_pod_count} execution pod(s)")
    return execution_pod_count


def collect_artifacts(
    plan: ResolvedRunPlan,
    *,
    artifacts_dir: Path,
    execution_name: str = "",
    include_execution_logs: bool = True,
    include_workload: bool = True,
    include_manifests: bool = True,
) -> Path:
    kubectl_cmd = require_any_command("oc", "kubectl")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    step(
        f"Collecting artifacts for release {plan.deployment.release_name} "
        f"in namespace {plan.deployment.namespace}"
    )
    if execution_name:
        detail(f"Execution: {execution_name}")
    detail(f"Artifacts directory: {artifacts_dir}")
    _ensure_artifact_layout(artifacts_dir)

    namespace = plan.deployment.namespace
    release_name = plan.deployment.release_name

    execution_pod_count = 0
    execution_pods: list[str] = []
    if include_execution_logs and execution_name:
        detail("Collecting execution pod logs")
        seen_execution_pods: set[str] = set()
        payload = run_json_command(
            [
                kubectl_cmd,
                "get",
                "pods",
                "-n",
                namespace,
                "-l",
                f"tekton.dev/pipelineRun={execution_name}",
                "-o",
                "json",
            ]
        )
        for item in payload.get("items", []):
            pod_name = item.get("metadata", {}).get("name", "")
            if pod_name:
                seen_execution_pods.add(pod_name)
        execution_pods = sorted(seen_execution_pods)
        for pod_name in execution_pods:
            if _collect_pod_logs(
                kubectl_cmd, namespace, pod_name, artifacts_dir / "logs" / "pipeline"
            ):
                execution_pod_count += 1
        detail(f"Collected logs from {execution_pod_count} execution pod(s)")

    model_count = 0
    gaie_count = 0
    infra_count = 0
    model_pod_names: list[str] = []
    if include_workload:
        payload = run_json_command(
            [kubectl_cmd, "get", "pods", "-n", namespace, "-o", "json"]
        )
        for item in payload.get("items", []):
            metadata = item.get("metadata", {})
            if not isinstance(metadata, dict) or not _matches_release(
                metadata, release_name
            ):
                continue
            pod_name = metadata.get("name", "")
            if not pod_name or pod_name.endswith("-pod") or pod_name in execution_pods:
                continue
            pod_type = _pod_type(pod_name)
            if pod_type == "model":
                model_pod_names.append(pod_name)
            if _collect_pod_logs(
                kubectl_cmd, namespace, pod_name, artifacts_dir / "logs" / pod_type
            ):
                if pod_type == "model":
                    model_count += 1
                elif pod_type == "gaie":
                    gaie_count += 1
                else:
                    infra_count += 1
        detail(
            f"Collected workload logs from {model_count} model pod(s), "
            f"{gaie_count} gaie pod(s), and {infra_count} infra pod(s)"
        )

    profiling_pods: list[str] = []
    profiling_file_count = 0
    if (
        include_workload
        and plan.deployment.platform == "rhoai"
        and plan.execution.profiling.enabled
        and model_pod_names
    ):
        for pod_name in sorted(model_pod_names):
            collected = _collect_profiling_artifacts(
                kubectl_cmd,
                namespace,
                pod_name,
                artifacts_dir,
            )
            if collected <= 0:
                continue
            profiling_pods.append(pod_name)
            profiling_file_count += collected
        detail(
            f"Collected {profiling_file_count} profiling artifact file(s) "
            f"from {len(profiling_pods)} pod(s)"
        )

    manifest_root = artifacts_dir / "manifests"
    manifest_count = 0
    if include_manifests:
        detail("Collecting Kubernetes manifests for deployed resources")
        for resource_type in (
            "deployments",
            "pods",
            "services",
            "configmaps",
            "servingruntimes",
            "inferenceservices",
            "gateways",
            "inferencepool",
            "llminferenceservices",
            "httproutes",
        ):
            get_result = run_command(
                [kubectl_cmd, "get", resource_type, "-n", namespace, "-o", "json"],
                capture_output=True,
                check=False,
            )
            if get_result.returncode != 0:
                continue
            payload = json.loads(get_result.stdout or "{}")
            items = payload.get("items", [])
            if not items:
                continue
            resource_dir = manifest_root / resource_type
            resource_dir.mkdir(parents=True, exist_ok=True)
            for item in items:
                metadata = item.get("metadata", {})
                if not isinstance(metadata, dict) or not _matches_release(
                    metadata, release_name
                ):
                    continue
                name = metadata.get("name")
                if not name:
                    continue
                result = run_command(
                    [
                        kubectl_cmd,
                        "get",
                        resource_type,
                        name,
                        "-n",
                        namespace,
                        "-o",
                        "yaml",
                    ],
                    capture_output=True,
                    check=False,
                )
                if result.returncode == 0 and result.stdout.strip():
                    (resource_dir / f"{name}.yaml").write_text(
                        result.stdout, encoding="utf-8"
                    )
                    manifest_count += 1

    metadata = {
        "namespace": namespace,
        "release": plan.deployment.release_name,
        "execution_name": execution_name,
        "execution_pods": execution_pod_count,
        "model_pods": model_count,
        "gaie_pods": gaie_count,
        "infra_pods": infra_count,
        "profiling_pods": profiling_pods,
        "profiling_files": profiling_file_count,
        "manifest_files": manifest_count,
        "timestamp": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
    }
    (artifacts_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    success(
        f"Artifacts collected in {artifacts_dir} ({manifest_count} manifest file(s))"
    )
    return artifacts_dir
