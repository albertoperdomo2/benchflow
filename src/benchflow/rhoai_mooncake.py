from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .models import ResolvedRunPlan, ValidationError

MOONCAKE_CONFIG_MOUNT_PATH = "/etc/benchflow/mooncake"
MOONCAKE_CONFIG_FILENAME = "mooncake_config.json"
MOONCAKE_MASTER_PORT = 50051
MOONCAKE_STORE_PORT = 50053


@dataclass(frozen=True, slots=True)
class RhoaiMooncakeSpec:
    mode: str
    global_segment_size: str
    local_buffer_size: str
    protocol: str
    device_name: str
    store_global_segment_size: str = ""
    host_path_name: str = ""
    offload_path: str = ""
    offload_size_limit_bytes: str = ""

    @property
    def is_nvme(self) -> bool:
        return self.mode == "standalone-store"


def _scoped_name(release_name: str, suffix: str) -> str:
    candidate = f"{release_name}-{suffix}"
    if len(candidate) <= 63:
        return candidate
    digest = hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:10]
    prefix_length = 63 - len(suffix) - len(digest) - 2
    return f"{release_name[:prefix_length].rstrip('-')}-{digest}-{suffix}"


def mooncake_configmap_name(plan: ResolvedRunPlan) -> str:
    return _scoped_name(plan.deployment.release_name, "mooncake-config")


def mooncake_master_name(plan: ResolvedRunPlan) -> str:
    return _scoped_name(plan.deployment.release_name, "mooncake-master")


def mooncake_master_service_name(plan: ResolvedRunPlan) -> str:
    return _scoped_name(plan.deployment.release_name, "mooncake-master")


def mooncake_config_path() -> str:
    return f"{MOONCAKE_CONFIG_MOUNT_PATH}/{MOONCAKE_CONFIG_FILENAME}"


def mooncake_nvme_release_directory(plan: ResolvedRunPlan) -> str | None:
    spec = rhoai_mooncake_spec(plan)
    if spec is None or not spec.is_nvme:
        return None
    return f"{spec.offload_path.rstrip('/')}/{plan.deployment.release_name}"


def rhoai_mooncake_spec(plan: ResolvedRunPlan) -> RhoaiMooncakeSpec | None:
    raw = plan.deployment.options.get("mooncake_store")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValidationError("options.mooncake_store must be a mapping")
    if plan.deployment.platform != "rhoai" or plan.deployment.mode == "isvc":
        raise ValidationError(
            "options.mooncake_store is supported only for rhoai "
            "LLMInferenceService deployments"
        )
    if not plan.deployment.runtime.image:
        raise ValidationError("Mooncake deployments require runtime.image")

    mode = str(raw.get("mode") or "").strip()
    if mode not in {"embedded", "standalone-store"}:
        raise ValidationError(
            "options.mooncake_store.mode must be 'embedded' or 'standalone-store'"
        )
    protocol = str(raw.get("protocol") or "").strip()
    if protocol not in {"rdma", "tcp"}:
        raise ValidationError("options.mooncake_store.protocol must be 'rdma' or 'tcp'")
    local_buffer_size = str(raw.get("local_buffer_size") or "").strip()
    if not local_buffer_size:
        raise ValidationError("options.mooncake_store.local_buffer_size is required")
    device_name = str(raw.get("device_name") or "").strip()

    if mode == "embedded":
        global_segment_size = str(raw.get("global_segment_size") or "").strip()
        if not global_segment_size:
            raise ValidationError(
                "options.mooncake_store.global_segment_size is required for "
                "embedded mode"
            )
        return RhoaiMooncakeSpec(
            mode=mode,
            global_segment_size=global_segment_size,
            local_buffer_size=local_buffer_size,
            protocol=protocol,
            device_name=device_name,
        )

    store_global_segment_size = str(raw.get("store_global_segment_size") or "").strip()
    host_path_name = str(raw.get("host_path_name") or "").strip()
    offload_path = str(raw.get("offload_path") or "").strip()
    offload_size_limit_bytes = str(raw.get("offload_size_limit_bytes") or "").strip()
    if not store_global_segment_size:
        raise ValidationError(
            "options.mooncake_store.store_global_segment_size is required for "
            "standalone-store mode"
        )
    if not host_path_name:
        raise ValidationError(
            "options.mooncake_store.host_path_name is required for "
            "standalone-store mode"
        )
    if not offload_path.startswith("/"):
        raise ValidationError(
            "options.mooncake_store.offload_path must be an absolute path"
        )
    if not offload_size_limit_bytes.isdigit() or int(offload_size_limit_bytes) <= 0:
        raise ValidationError(
            "options.mooncake_store.offload_size_limit_bytes must be a positive integer"
        )

    matching_mounts = [
        mount
        for mount in plan.deployment.runtime.host_paths
        if mount.name == host_path_name and not mount.read_only
    ]
    if len(matching_mounts) != 1:
        raise ValidationError(
            "options.mooncake_store.host_path_name must reference exactly one "
            "writable runtime.host_paths entry"
        )
    if matching_mounts[0].mount_path != offload_path:
        raise ValidationError(
            "options.mooncake_store.offload_path must match the referenced "
            "runtime.host_paths mount_path"
        )
    return RhoaiMooncakeSpec(
        mode=mode,
        global_segment_size="0",
        local_buffer_size=local_buffer_size,
        protocol=protocol,
        device_name=device_name,
        store_global_segment_size=store_global_segment_size,
        host_path_name=host_path_name,
        offload_path=offload_path,
        offload_size_limit_bytes=offload_size_limit_bytes,
    )


def _labels(plan: ResolvedRunPlan, component: str) -> dict[str, str]:
    return {
        "app.kubernetes.io/name": mooncake_master_name(plan),
        "app.kubernetes.io/component": component,
        "app.kubernetes.io/managed-by": "benchflow",
        "benchflow.io/experiment": plan.metadata.name,
        "benchflow.io/release": plan.deployment.release_name,
    }


def _master_address(plan: ResolvedRunPlan) -> str:
    return (
        f"{mooncake_master_service_name(plan)}."
        f"{plan.deployment.namespace}.svc:{MOONCAKE_MASTER_PORT}"
    )


def render_rhoai_mooncake_manifests(plan: ResolvedRunPlan) -> list[dict[str, Any]]:
    spec = rhoai_mooncake_spec(plan)
    if spec is None:
        return []
    master_name = mooncake_master_name(plan)
    master_labels = _labels(plan, "mooncake-master")
    config = {
        "mode": spec.mode,
        "metadata_server": "P2PHANDSHAKE",
        "master_server_address": _master_address(plan),
        "global_segment_size": spec.global_segment_size,
        "local_buffer_size": spec.local_buffer_size,
        "protocol": spec.protocol,
        "device_name": spec.device_name,
        "enable_offload": spec.is_nvme,
    }
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": mooncake_configmap_name(plan),
            "namespace": plan.deployment.namespace,
            "labels": _labels(plan, "mooncake-config"),
        },
        "data": {MOONCAKE_CONFIG_FILENAME: json.dumps(config, indent=2) + "\n"},
    }
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": mooncake_master_service_name(plan),
            "namespace": plan.deployment.namespace,
            "labels": master_labels,
        },
        "spec": {
            "selector": master_labels,
            "ports": [
                {
                    "name": "rpc",
                    "port": MOONCAKE_MASTER_PORT,
                    "targetPort": "rpc",
                    "protocol": "TCP",
                }
            ],
        },
    }
    master_flags = [
        f"--rpc_port={MOONCAKE_MASTER_PORT}",
        "--rpc_address=0.0.0.0",
    ]
    if spec.is_nvme:
        master_flags += [
            "--enable_offload=true",
            "--offload_on_evict=true",
            "--enable_disk_eviction=true",
        ]
    master_script = (
        'MOONCAKE_DIR=/usr/local/lib/python3.12/dist-packages/mooncake\n'
        'export LD_LIBRARY_PATH="$MOONCAKE_DIR:'
        '/usr/local/lib/python3.12/dist-packages/'
        'mooncake_transfer_engine.libs:${LD_LIBRARY_PATH:-}"\n'
        'exec mooncake_master ' + ' '.join(master_flags) + '\n'
    )
    pod_spec: dict[str, Any] = {
        "containers": [
            {
                "name": "mooncake-master",
                "image": plan.deployment.runtime.image,
                "command": ["/bin/bash", "-c"],
                "args": [master_script],
                "ports": [
                    {
                        "name": "rpc",
                        "containerPort": MOONCAKE_MASTER_PORT,
                        "protocol": "TCP",
                    }
                ],
                "readinessProbe": {
                    "tcpSocket": {"port": "rpc"},
                    "initialDelaySeconds": 2,
                    "periodSeconds": 5,
                },
                "livenessProbe": {
                    "tcpSocket": {"port": "rpc"},
                    "initialDelaySeconds": 10,
                    "periodSeconds": 10,
                },
                "resources": {},
            }
        ]
    }
    if plan.deployment.runtime.image_pull_secrets:
        pod_spec["imagePullSecrets"] = list(plan.deployment.runtime.image_pull_secrets)
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": master_name,
            "namespace": plan.deployment.namespace,
            "labels": master_labels,
        },
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": master_labels},
            "template": {
                "metadata": {"labels": master_labels},
                "spec": pod_spec,
            },
        },
    }
    return [configmap, service, deployment]


def rhoai_mooncake_model_volume(plan: ResolvedRunPlan) -> dict[str, Any] | None:
    if rhoai_mooncake_spec(plan) is None:
        return None
    return {
        "name": "mooncake-config",
        "configMap": {"name": mooncake_configmap_name(plan)},
    }


def rhoai_mooncake_model_volume_mount(plan: ResolvedRunPlan) -> dict[str, Any] | None:
    if rhoai_mooncake_spec(plan) is None:
        return None
    return {
        "name": "mooncake-config",
        "mountPath": MOONCAKE_CONFIG_MOUNT_PATH,
        "readOnly": True,
    }


def rhoai_mooncake_model_env(plan: ResolvedRunPlan) -> list[dict[str, str]]:
    spec = rhoai_mooncake_spec(plan)
    if spec is None:
        return []
    env = [{"name": "MOONCAKE_CONFIG_PATH", "value": mooncake_config_path()}]
    if spec.is_nvme:
        env.append(
            {
                "name": "MOONCAKE_PREFERRED_SEGMENT",
                "value": f"127.0.0.1:{MOONCAKE_STORE_PORT}",
            }
        )
    return env


def rhoai_mooncake_store_sidecar(plan: ResolvedRunPlan) -> dict[str, Any] | None:
    spec = rhoai_mooncake_spec(plan)
    if spec is None or not spec.is_nvme:
        return None
    release_dir = mooncake_nvme_release_directory(plan)
    if release_dir is None:
        raise AssertionError("NVMe Mooncake sidecar is missing its release directory")
    pod_dir = f"{release_dir}/$POD_NAME"
    command = "\n".join(
        [
            "set -eu",
            f"release_dir={json.dumps(release_dir)}",
            f"pod_dir={json.dumps(pod_dir)}",
            'mkdir -p -m 1777 "$release_dir"',
            'rm -rf -- "$pod_dir"',
            'mkdir -p "$pod_dir"',
            'export MOONCAKE_OFFLOAD_FILE_STORAGE_PATH="$pod_dir"',
            "exec mooncake_client "
            f"--port={MOONCAKE_STORE_PORT} "
            '--master_server_address="$MOONCAKE_MASTER" '
            '--metadata_server="$MOONCAKE_TE_META_DATA_SERVER" '
            '--protocol="$MOONCAKE_PROTOCOL" '
            '--device_name="$MOONCAKE_DEVICE" '
            '--global_segment_size="$MOONCAKE_GLOBAL_SEGMENT_SIZE" '
            "--local_buffer_size=0 --enable_offload=true",
        ]
    )
    return {
        "name": "mooncake-store",
        "image": plan.deployment.runtime.image,
        "command": ["/bin/sh", "-ec"],
        "args": [command],
        "ports": [
            {
                "name": "store",
                "containerPort": MOONCAKE_STORE_PORT,
                "protocol": "TCP",
            }
        ],
        "env": [
            {
                "name": "POD_NAME",
                "valueFrom": {
                    "fieldRef": {"apiVersion": "v1", "fieldPath": "metadata.name"}
                },
            },
            {
                "name": "MOONCAKE_LOCAL_HOSTNAME",
                "valueFrom": {
                    "fieldRef": {"apiVersion": "v1", "fieldPath": "status.podIP"}
                },
            },
            {"name": "MOONCAKE_TE_META_DATA_SERVER", "value": "P2PHANDSHAKE"},
            {"name": "MOONCAKE_MASTER", "value": _master_address(plan)},
            {"name": "MOONCAKE_PROTOCOL", "value": spec.protocol},
            {"name": "MOONCAKE_DEVICE", "value": spec.device_name},
            {
                "name": "MOONCAKE_GLOBAL_SEGMENT_SIZE",
                "value": spec.store_global_segment_size,
            },
            {"name": "MOONCAKE_LOCAL_BUFFER_SIZE", "value": "0"},
            {
                "name": "MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES",
                "value": spec.offload_size_limit_bytes,
            },
        ],
        "volumeMounts": [{"name": spec.host_path_name, "mountPath": spec.offload_path}],
        "lifecycle": {
            "preStop": {
                "exec": {
                    "command": [
                        "/bin/sh",
                        "-ec",
                        f"rm -rf -- {json.dumps(pod_dir)}",
                    ]
                }
            }
        },
        "resources": {},
    }
