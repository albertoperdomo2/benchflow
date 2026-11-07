"""The node-local LMCache MP CPU tier managed by platform setup.

One configuration and one benchmark release own the cache at a time. The setup
key participates in the existing Kueue platform waves; a new release restarts
the servers rather than inheriting another benchmark's KV state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .cluster import CommandError, run_command
from .models import ResolvedDeployment, ResolvedRunPlan, ValidationError
from .lmcache_gpu import ensure_lmcache_gpu_access, release_lmcache_gpu_access

NAME = "benchflow-lmcache"
LABELS = {"app.kubernetes.io/name": NAME, "benchflow.io/managed-by": "benchflow"}
LMCACHE_VERSION = "0.5.7"
VLLM_VERSION = "0.32.0"
DEFAULT_RUNTIME_IMAGE = (
    "ghcr.io/albertoperdomo2/benchflow:lmcache-vllm0.32.0-lmcache0.5.7"
)


@dataclass(frozen=True)
class LMCacheSpec:
    cpu_gib: int = 64
    chunk_size: int = 256
    verify_readback: bool = False


def lmcache_spec(deployment: ResolvedDeployment) -> LMCacheSpec | None:
    raw = deployment.options.get("lmcache")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValidationError("options.lmcache must be a mapping")
    unknown = set(raw) - {"cpu_gib", "chunk_size", "verify_readback"}
    if unknown:
        raise ValidationError(
            f"unsupported options.lmcache fields: {', '.join(sorted(unknown))}"
        )
    if (deployment.platform, deployment.mode) not in {
        ("rhoai", "distributed-default"),
        ("llm-d", "inference-scheduling"),
    }:
        raise ValidationError(
            "LMCache supports rhoai distributed-default and llm-d inference-scheduling only"
        )
    runtime = deployment.runtime
    if runtime.node_selector.get("nvidia.com/gpu.present", "true") != "true":
        raise ValidationError(
            "LMCache requires runtime.node_selector nvidia.com/gpu.present=true when specified"
        )
    if runtime.replicas != 1 or runtime.pipeline_parallelism != 1:
        raise ValidationError(
            "LMCache CPU offload currently requires one replica and pipeline_parallelism=1"
        )
    if deployment.options.get("mooncake_store") or deployment.options.get(
        "storage_offloading"
    ):
        raise ValidationError(
            "LMCache cannot be combined with Mooncake or native storage_offloading"
        )
    image = runtime.image.strip()
    tag = image.rsplit("/", 1)[-1]
    if not image or (
        "@sha256:" not in image
        and (":" not in tag or any(x in tag.lower() for x in ("latest", "nightly")))
    ):
        raise ValidationError(
            "LMCache requires an explicit versioned runtime.image (prefer a digest) containing vLLM 0.32.0 and LMCache 0.5.7"
        )
    for key in ("cpu_gib", "chunk_size"):
        value = raw.get(key, getattr(LMCacheSpec(), key))
        if type(value) is not int or value <= 0:
            raise ValidationError(f"options.lmcache.{key} must be a positive integer")
    if type(raw.get("verify_readback", False)) is not bool:
        raise ValidationError("options.lmcache.verify_readback must be a boolean")
    for arg in runtime.vllm_args:
        flag = arg.split("=", 1)[0]
        if flag in {
            "--enable-sleep-mode",
            "--cpu-offload-gb",
            "--kv-offloading-backend",
            "--kv-offloading-size",
            "--distributed-executor-backend",
        }:
            raise ValidationError(
                f"LMCache does not support {flag} in this single-node CPU path"
            )
    if any(
        re.search(
            r"expandable_segments\s*:\s*true", runtime.env.get(key, ""), re.IGNORECASE
        )
        for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")
    ):
        raise ValidationError(
            "LMCache isolated IPC requires cudaMalloc allocations; expandable_segments is unsupported"
        )
    return LMCacheSpec(
        raw.get("cpu_gib", 64),
        raw.get("chunk_size", 256),
        raw.get("verify_readback", False),
    )


def connector_config(deployment: ResolvedDeployment) -> dict[str, Any]:
    return {
        "kv_connector": "LMCacheMPConnector",
        "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "lmcache.mp.host": f"tcp://{NAME}.{deployment.namespace}.svc.cluster.local",
            "lmcache.mp.port": 5555,
            "lmcache.mp.isolated_ipc": True,
        },
    }


def configure_lmcache(
    deployment: ResolvedDeployment, *, model_name: str = ""
) -> ResolvedDeployment:
    """Resolve once into the RunPlan; also validate deserialized RunPlans."""
    if (
        deployment.options.get("lmcache") is not None
        and not deployment.runtime.image.strip()
    ):
        deployment = replace(
            deployment, runtime=replace(deployment.runtime, image=DEFAULT_RUNTIME_IMAGE)
        )
    if lmcache_spec(deployment) is None:
        return deployment
    if model_name and model_name not in {"Qwen/Qwen3-0.6B", "Qwen/Qwen3-32B"}:
        raise ValidationError(
            "The initial LMCache path is limited to Qwen/Qwen3-0.6B and Qwen/Qwen3-32B; other model layouts need validation"
        )
    expected = connector_config(deployment)
    args = list(deployment.runtime.vllm_args)
    found = 0
    for index, arg in enumerate(args):
        if arg.split("=", 1)[0] != "--kv-transfer-config":
            continue
        found += 1
        value = (
            arg.split("=", 1)[1]
            if "=" in arg
            else (args[index + 1] if index + 1 < len(args) else "")
        )
        try:
            actual = json.loads(value)
        except (ValueError, TypeError) as exc:
            raise ValidationError(
                "LMCache manages --kv-transfer-config; remove the profile's connector argument"
            ) from exc
        if actual != expected or found > 1:
            raise ValidationError(
                "LMCache manages --kv-transfer-config; remove the conflicting connector argument"
            )
    if not found:
        args.append(
            "--kv-transfer-config="
            + json.dumps(expected, separators=(",", ":"), sort_keys=True)
        )
    if deployment.runtime.env.get("PYTHONHASHSEED", "0") != "0":
        raise ValidationError("LMCache requires PYTHONHASHSEED=0")
    return replace(
        deployment,
        runtime=replace(
            deployment.runtime,
            vllm_args=args,
            env={**deployment.runtime.env, "PYTHONHASHSEED": "0"},
        ),
    )


def lmcache_identity(plan: ResolvedRunPlan) -> str:
    if lmcache_spec(plan.deployment) is None:
        return ""
    # Rendering is pure: hash the resolved setup, including its benchmark owner.
    payload = json.dumps(
        {
            "documents": lmcache_documents(plan),
            "model": plan.model.name,
            "runtime": asdict(plan.deployment.runtime),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def lmcache_documents(plan: ResolvedRunPlan) -> list[dict[str, Any]]:
    deployment = configure_lmcache(plan.deployment, model_name=plan.model.name)
    spec = lmcache_spec(deployment)
    if spec is None:
        return []
    runtime = deployment.runtime
    namespace = deployment.namespace
    metadata = {"name": NAME, "namespace": namespace, "labels": LABELS}
    # Use the serving image on both sides: torch/CUDA/Python and LMCache ABI
    # cannot silently drift between independently selected images.
    preflight = (
        "import importlib.metadata as m,json,os,torch; "
        f"assert m.version('lmcache') == '{LMCACHE_VERSION}', 'expected LMCache {LMCACHE_VERSION}'; "
        f"assert m.version('vllm') == '{VLLM_VERSION}', 'expected vLLM {VLLM_VERSION}'; "
        "import cuda.bindings.driver; "
        "from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector; "
        "assert torch.cuda.is_available(), 'LMCache server cannot access CUDA'; "
        "print(json.dumps({'lmcache':m.version('lmcache'),'vllm':m.version('vllm'),"
        "'torch':torch.__version__,'cuda':torch.version.cuda,'gpus':torch.cuda.device_count()}),flush=True); "
        "os.execvp('lmcache', ['lmcache','server'] + __import__('sys').argv[1:])"
    )
    pod_spec: dict[str, Any] = {
        "runtimeClassName": "nvidia",
        "automountServiceAccountToken": False,
        "nodeSelector": {**runtime.node_selector, "nvidia.com/gpu.present": "true"},
        "tolerations": runtime.tolerations,
        "imagePullSecrets": runtime.image_pull_secrets,
        "containers": [
            {
                "name": "lmcache-server",
                "image": runtime.image,
                "command": ["python3", "-c", preflight],
                "args": [
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "5555",
                    "--http-host",
                    "0.0.0.0",
                    "--http-port",
                    "8080",
                    "--l1-size-gb",
                    str(spec.cpu_gib),
                    "--chunk-size",
                    str(spec.chunk_size),
                    "--eviction-policy",
                    "LRU",
                    "--isolated-ipc",
                ],
                "env": [
                    {"name": "NVIDIA_VISIBLE_DEVICES", "value": "all"},
                    {"name": "NVIDIA_DRIVER_CAPABILITIES", "value": "compute,utility"},
                    {"name": "HOME", "value": "/tmp"},
                    {"name": "PYTHONHASHSEED", "value": "0"},
                ],
                "ports": [
                    {"name": "transport", "containerPort": 5555},
                    {"name": "http", "containerPort": 8080},
                ],
                "resources": {
                    "requests": {"cpu": "2", "memory": f"{spec.cpu_gib + 8}Gi"},
                    "limits": {"memory": f"{spec.cpu_gib + 8}Gi"},
                },
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                },
                "startupProbe": {
                    "httpGet": {"path": "/healthcheck", "port": "http"},
                    "failureThreshold": 60,
                    "periodSeconds": 10,
                },
                "readinessProbe": {
                    "httpGet": {"path": "/healthcheck", "port": "http"},
                    "periodSeconds": 10,
                },
            }
        ],
    }
    if runtime.affinity:
        pod_spec["affinity"] = runtime.affinity
    return [
        {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": metadata,
            "spec": {
                "podSelector": {"matchLabels": LABELS},
                "policyTypes": ["Ingress"],
                "ingress": [
                    {
                        "from": [{"podSelector": {}}],
                        "ports": [{"port": 5555}, {"port": 8080}],
                    },
                    {
                        "from": [
                            {
                                "namespaceSelector": {
                                    "matchLabels": {"kubernetes.io/metadata.name": ns}
                                }
                            }
                            for ns in (
                                "openshift-monitoring",
                                "openshift-user-workload-monitoring",
                            )
                        ],
                        "ports": [{"port": 8080}],
                    },
                ],
            },
        },
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": metadata,
            "spec": {
                "selector": LABELS,
                "internalTrafficPolicy": "Local",
                "ports": [
                    {"name": "transport", "port": 5555, "targetPort": "transport"},
                    {"name": "http", "port": 8080, "targetPort": "http"},
                ],
            },
        },
        {
            "apiVersion": "apps/v1",
            "kind": "DaemonSet",
            "metadata": metadata,
            "spec": {
                "selector": {"matchLabels": LABELS},
                "updateStrategy": {
                    "type": "RollingUpdate",
                    "rollingUpdate": {"maxUnavailable": 1},
                },
                "template": {
                    "metadata": {
                        "labels": LABELS,
                        "annotations": {
                            "benchflow.io/cache-owner": deployment.release_name
                        },
                    },
                    "spec": pod_spec,
                },
            },
        },
        {
            "apiVersion": "monitoring.coreos.com/v1",
            "kind": "ServiceMonitor",
            "metadata": metadata,
            "spec": {
                "selector": {"matchLabels": LABELS},
                "endpoints": [{"port": "http", "path": "/metrics", "interval": "15s"}],
            },
        },
    ]


def remove_lmcache(kubectl_cmd: str, namespace: str) -> None:
    # Wait for the old CUDA process to exit before replacing its cache owner.
    run_command(
        [
            kubectl_cmd,
            "delete",
            "daemonset,service,networkpolicy,servicemonitor",
            NAME,
            "-n",
            namespace,
            "--ignore-not-found=true",
            "--cascade=foreground",
            "--wait=true",
            "--timeout=300s",
        ]
    )


def ensure_lmcache(plan: ResolvedRunPlan, kubectl_cmd: str) -> None:
    spec = lmcache_spec(plan.deployment)
    if spec is None:
        remove_lmcache(kubectl_cmd, plan.deployment.namespace)
        release_lmcache_gpu_access(kubectl_cmd, plan.deployment.namespace)
        return
    gpu_access = ensure_lmcache_gpu_access(kubectl_cmd, plan.deployment.namespace)
    selector = {
        **plan.deployment.runtime.node_selector,
        "nvidia.com/gpu.present": "true",
    }
    nodes = run_command(
        [
            kubectl_cmd,
            "get",
            "nodes",
            "-l",
            ",".join(f"{k}={v}" for k, v in sorted(selector.items())),
            "-o",
            "json",
        ],
        capture_output=True,
    )
    items = json.loads(nodes.stdout).get("items", [])
    if not items:
        raise CommandError(
            "LMCache found no GPU nodes matching the runtime node selector"
        )
    for node in items:
        labels = node.get("metadata", {}).get("labels", {})
        if (
            labels.get("nvidia.com/mig.config", "all-disabled")
            not in {"", "all-disabled"}
            or int(labels.get("nvidia.com/gpu.replicas", "1")) > 1
        ):
            raise CommandError(
                "LMCache CPU offload currently requires whole NVIDIA GPUs without MIG or time slicing"
            )
    # Apply ingress restrictions before making the transport reachable.
    for document in lmcache_documents(plan):
        if document["kind"] == "DaemonSet" and gpu_access == "nri":
            template = document["spec"]["template"]
            template["spec"].pop("runtimeClassName", None)
            template["metadata"]["annotations"][
                "nvidia.cdi.k8s.io/container.lmcache-server"
            ] = "management.nvidia.com/gpu=all"
        run_command([kubectl_cmd, "apply", "-f", "-"], input_text=json.dumps(document))
    run_command(
        [
            kubectl_cmd,
            "rollout",
            "status",
            f"daemonset/{NAME}",
            "-n",
            plan.deployment.namespace,
            "--timeout=600s",
        ]
    )
    result = run_command(
        [
            kubectl_cmd,
            "get",
            "daemonset",
            NAME,
            "-n",
            plan.deployment.namespace,
            "-o",
            "json",
        ],
        capture_output=True,
    )
    status = json.loads(result.stdout).get("status", {})
    desired = status.get("desiredNumberScheduled", 0)
    if desired < 1 or status.get("numberReady", 0) != desired:
        raise CommandError(
            "LMCache DaemonSet has no ready server on every selected GPU node"
        )


def require_lmcache(plan: ResolvedRunPlan, kubectl_cmd: str) -> None:
    if lmcache_spec(plan.deployment) is None:
        return
    from .platform_state import load_cluster_platform_state, setup_key_for_plan

    state = load_cluster_platform_state(kubectl_cmd, plan.deployment.namespace)
    if state.get("installed_key") != setup_key_for_plan(plan):
        raise ValidationError(
            "LMCache platform state does not match this RunPlan; run platform setup before deploying"
        )


def lmcache_readback_snapshot(
    plan: ResolvedRunPlan, kubectl_cmd: str
) -> dict[str, Any]:
    result = run_command(
        [
            kubectl_cmd,
            "get",
            "pods",
            "-n",
            plan.deployment.namespace,
            "-l",
            f"app.kubernetes.io/name={NAME}",
            "-o",
            "json",
        ],
        capture_output=True,
    )
    pods = json.loads(result.stdout).get("items", [])
    if not pods:
        raise CommandError("LMCache smoke test found no server pods")
    snapshot = {}
    for pod in pods:
        metadata = pod["metadata"]
        metrics = run_command(
            [
                kubectl_cmd,
                "exec",
                "-n",
                plan.deployment.namespace,
                metadata["name"],
                "-c",
                "lmcache-server",
                "--",
                "python3",
                "-c",
                "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8080/metrics', timeout=15).read().decode())",
            ],
            capture_output=True,
        )
        counts = {"stores": 0.0, "loads": 0.0}
        for line in metrics.stdout.splitlines():
            match = re.match(
                r"(lmcache_mp_l1_write_chunks_total|lmcache_mp_num_chunks_loaded_total)(?:\{.*\})?\s+([0-9.eE+-]+)(?:\s|$)",
                line,
            )
            if match:
                counts["stores" if "l1_write" in match[1] else "loads"] += float(
                    match[2]
                )
        restarts = sum(
            status.get("restartCount", 0)
            for status in pod.get("status", {}).get("containerStatuses", [])
            if status.get("name") == "lmcache-server"
        )
        snapshot[metadata["name"]] = {
            "uid": metadata["uid"],
            "restarts": restarts,
            **counts,
        }
    return snapshot


def verify_lmcache_readback(
    plan: ResolvedRunPlan,
    kubectl_cmd: str,
    before: dict[str, Any],
    output_dir: Path | None,
) -> None:
    after = lmcache_readback_snapshot(plan, kubectl_cmd)
    stable = before.keys() == after.keys() and all(
        before[name]["uid"] == after[name]["uid"]
        and before[name].get("restarts", 0) == after[name].get("restarts", 0)
        for name in before
    )
    delta = {
        key: sum(value[key] for value in after.values())
        - sum(value[key] for value in before.values())
        for key in ("stores", "loads")
    }
    valid = stable and delta["stores"] > 0 and delta["loads"] > 0
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "lmcache-readback.json").write_text(
            json.dumps(
                {
                    "before": before,
                    "after": after,
                    "delta": delta,
                    "servers_stable": stable,
                    "passed": valid,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if not valid:
        raise CommandError(
            f"LMCache smoke test failed: stable servers={stable}, new stored chunks={delta['stores']}, new successfully loaded chunks={delta['loads']}"
        )
