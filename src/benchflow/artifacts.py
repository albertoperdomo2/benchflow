from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .cluster import CommandError, require_any_command, run_command, run_json_command
from .models import ResolvedRunPlan
from .ui import detail, step, success


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


def collect_artifacts(
    plan: ResolvedRunPlan,
    *,
    artifacts_dir: Path,
    pipeline_run_name: str = "",
) -> Path:
    kubectl_cmd = require_any_command("oc", "kubectl")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    step(
        f"Collecting artifacts for release {plan.deployment.release_name} "
        f"in namespace {plan.deployment.namespace}"
    )
    if pipeline_run_name:
        detail(f"PipelineRun: {pipeline_run_name}")
    detail(f"Artifacts directory: {artifacts_dir}")
    for relative in (
        "logs/pipeline",
        "logs/model",
        "logs/gaie",
        "logs/infra",
        "manifests",
    ):
        (artifacts_dir / relative).mkdir(parents=True, exist_ok=True)

    namespace = plan.deployment.namespace

    pipeline_pods: list[str] = []
    pipeline_count = 0
    if pipeline_run_name:
        detail("Collecting Tekton task pod logs")
        payload = run_json_command(
            [
                kubectl_cmd,
                "get",
                "pods",
                "-n",
                namespace,
                "-l",
                f"tekton.dev/pipelineRun={pipeline_run_name}",
                "-o",
                "json",
            ]
        )
        pipeline_pods = [item["metadata"]["name"] for item in payload.get("items", [])]
        for pod_name in pipeline_pods:
            if _collect_pod_logs(
                kubectl_cmd, namespace, pod_name, artifacts_dir / "logs" / "pipeline"
            ):
                pipeline_count += 1
        detail(f"Collected logs from {pipeline_count} pipeline pod(s)")

    payload = run_json_command(
        [kubectl_cmd, "get", "pods", "-n", namespace, "-o", "json"]
    )
    model_count = 0
    gaie_count = 0
    infra_count = 0
    for item in payload.get("items", []):
        pod_name = item.get("metadata", {}).get("name", "")
        if not pod_name or pod_name.endswith("-pod") or pod_name in pipeline_pods:
            continue
        pod_type = _pod_type(pod_name)
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

    manifest_root = artifacts_dir / "manifests"
    manifest_count = 0
    detail("Collecting Kubernetes manifests for deployed resources")
    for resource_type in (
        "deployments",
        "pods",
        "services",
        "configmaps",
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
            name = item.get("metadata", {}).get("name")
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
        "pipeline_run": pipeline_run_name,
        "pipeline_pods": pipeline_count,
        "model_pods": model_count,
        "gaie_pods": gaie_count,
        "infra_pods": infra_count,
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
