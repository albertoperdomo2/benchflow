from __future__ import annotations

import json
import shlex
from datetime import datetime, timezone
from pathlib import Path

from .benchmark import runtime as runtime_module
from .benchmark import benchmark_version_from_plan
from .cluster import CommandError, require_any_command, run_command, run_json_command
from .models import ResolvedRunPlan
from .storage_offloading import storage_offloading_config
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
            "vllm",
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
    containers = _pod_container_names(payload)
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


def _pod_container_names(pod: dict) -> list[str]:
    containers = pod.get("spec", {}).get("containers", [])
    if not isinstance(containers, list):
        return []
    return [
        str(entry.get("name") or "")
        for entry in containers
        if isinstance(entry, dict) and entry.get("name")
    ]


def _ensure_artifact_layout(artifacts_dir: Path) -> None:
    for relative in (
        "logs/pipeline",
        "logs/model",
        "logs/gaie",
        "logs/infra",
        "logs/storage-offloading",
        "manifests",
        "platform-state",
        "profiling",
    ):
        (artifacts_dir / relative).mkdir(parents=True, exist_ok=True)


def _write_command_snapshot(
    kubectl_cmd: str,
    output_dir: Path,
    name: str,
    command: list[str],
) -> bool:
    result = run_command([kubectl_cmd, *command], capture_output=True, check=False)
    output = result.stdout or result.stderr or ""
    if not output.strip():
        return False
    suffix = "txt" if result.returncode == 0 else "error.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}.{suffix}").write_text(output, encoding="utf-8")
    return result.returncode == 0


def _collect_platform_state(
    kubectl_cmd: str,
    namespace: str,
    release_name: str,
    artifacts_dir: Path,
) -> int:
    state_dir = artifacts_dir / "platform-state"
    snapshot_count = 0
    for name, command in (
        ("pods-wide", ["get", "pods", "-n", namespace, "-o", "wide"]),
        ("pods-json", ["get", "pods", "-n", namespace, "-o", "json"]),
        (
            "workloads",
            [
                "get",
                "deployments,statefulsets,replicasets,jobs,pods,services",
                "-n",
                namespace,
                "-o",
                "wide",
            ],
        ),
        ("httproutes", ["get", "httproutes", "-n", namespace, "-o", "wide"]),
        ("gateways", ["get", "gateways", "-n", namespace, "-o", "wide"]),
        (
            "inferenceservices",
            ["get", "inferenceservices", "-n", namespace, "-o", "wide"],
        ),
        (
            "llminferenceservices",
            ["get", "llminferenceservices", "-n", namespace, "-o", "wide"],
        ),
        (
            "events",
            [
                "get",
                "events",
                "-n",
                namespace,
                "--sort-by=.lastTimestamp",
            ],
        ),
    ):
        if _write_command_snapshot(kubectl_cmd, state_dir, name, command):
            snapshot_count += 1

    pods_payload = run_command(
        [kubectl_cmd, "get", "pods", "-n", namespace, "-o", "json"],
        capture_output=True,
        check=False,
    )
    if pods_payload.returncode != 0 or not pods_payload.stdout.strip():
        return snapshot_count
    try:
        payload = json.loads(pods_payload.stdout)
    except json.JSONDecodeError:
        return snapshot_count

    describe_dir = state_dir / "describe"
    for item in payload.get("items", []):
        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict) or not _matches_release(
            metadata, release_name
        ):
            continue
        pod_name = str(metadata.get("name") or "")
        if not pod_name:
            continue
        if _write_command_snapshot(
            kubectl_cmd,
            describe_dir,
            f"pod-{pod_name}",
            ["describe", "pod", pod_name, "-n", namespace],
        ):
            snapshot_count += 1
    return snapshot_count


def _collect_storage_offloading_dir_size(
    kubectl_cmd: str,
    namespace: str,
    pod_name: str,
    artifacts_dir: Path,
    mount_path: str,
) -> bool:
    try:
        payload = run_json_command(
            [kubectl_cmd, "get", "pod", pod_name, "-n", namespace, "-o", "json"]
        )
    except CommandError:
        return False

    containers = _pod_container_names(payload)
    if not containers:
        return False

    quoted_path = shlex.quote(mount_path)
    script = (
        "set -u; "
        f"target={quoted_path}; "
        'echo "path=${target}"; '
        'if [ ! -d "${target}" ]; then echo "status=missing"; exit 0; fi; '
        'echo "status=present"; '
        'du -sh "${target}" 2>/dev/null || true; '
        'du -sb "${target}" 2>/dev/null || true; '
        'find "${target}" -mindepth 1 -maxdepth 2 '
        "-exec du -sh {} + 2>/dev/null | sort -h | tail -50 || true"
    )
    container = containers[0]
    result = run_command(
        [
            kubectl_cmd,
            "exec",
            pod_name,
            "-c",
            container,
            "-n",
            namespace,
            "--",
            "sh",
            "-lc",
            script,
        ],
        capture_output=True,
        check=False,
    )

    output = result.stdout or result.stderr or ""
    if not output.strip():
        return False

    log_dir = artifacts_dir / "logs" / "storage-offloading"
    log_dir.mkdir(parents=True, exist_ok=True)
    suffix = "log" if result.returncode == 0 else "error.log"
    (log_dir / f"{pod_name}_{container}_dir-size.{suffix}").write_text(
        output,
        encoding="utf-8",
    )
    return result.returncode == 0


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
    platform_state_count = _collect_platform_state(
        kubectl_cmd,
        namespace,
        release_name,
        artifacts_dir,
    )
    detail(f"Collected {platform_state_count} platform state snapshot(s)")

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
    storage_offloading_log_count = 0
    storage_offloading = storage_offloading_config(plan)
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
            if (
                pod_type == "model"
                and storage_offloading
                and _collect_storage_offloading_dir_size(
                    kubectl_cmd,
                    namespace,
                    pod_name,
                    artifacts_dir,
                    str(storage_offloading["directory_path"]),
                )
            ):
                storage_offloading_log_count += 1
        detail(
            f"Collected workload logs from {model_count} model pod(s), "
            f"{gaie_count} gaie pod(s), and {infra_count} infra pod(s)"
        )
        if storage_offloading:
            detail(
                "Collected storage offloading directory size logs from "
                f"{storage_offloading_log_count} model pod(s)"
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
            "podmonitors",
            "servicemonitors",
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

    guidellm_data = (
        runtime_module.guidellm_data_mapping(plan.benchmark.guidellm.args)
        if plan.benchmark.tool == "guidellm"
        else {}
    )
    guidellm_backend = (
        runtime_module.guidellm_backend_mapping(plan.benchmark.guidellm.args)
        if plan.benchmark.tool == "guidellm"
        else {}
    )

    metadata = {
        "namespace": namespace,
        "release": plan.deployment.release_name,
        "execution_name": execution_name,
        "model_name": plan.model.resolved_name(),
        "platform": plan.deployment.platform,
        "mode": plan.deployment.mode,
        "version": benchmark_version_from_plan(plan),
        "accelerator": str(
            plan.mlflow.tags.get("accelerator")
            or plan.deployment.options.get("accelerator")
            or ""
        ),
        "runtime_args": " ".join(plan.deployment.runtime.vllm_args),
        "replicas": plan.deployment.runtime.replicas,
        "tp": plan.deployment.runtime.tensor_parallelism,
        "data_spec": (
            json.dumps(guidellm_data, separators=(",", ":"))
            if plan.benchmark.tool == "guidellm"
            else (
                plan.benchmark.aiperf.args.get("public_dataset")
                or plan.benchmark.aiperf.dataset_name
                or plan.benchmark.aiperf.dataset_url
            )
        ),
        "profile": plan.profiles.benchmark,
        "backend": (
            guidellm_backend.get("kind", "openai_http")
            if plan.benchmark.tool == "guidellm"
            else "openai_http"
        ),
        "execution_pods": execution_pod_count,
        "model_pods": model_count,
        "gaie_pods": gaie_count,
        "infra_pods": infra_count,
        "profiling_pods": profiling_pods,
        "profiling_files": profiling_file_count,
        "storage_offloading_logs": storage_offloading_log_count,
        "manifest_files": manifest_count,
        "platform_state_files": platform_state_count,
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
