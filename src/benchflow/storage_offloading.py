from __future__ import annotations

from typing import Any

from .cluster import CommandError
from .models import ResolvedRunPlan

STORAGE_OFFLOADING_TYPE_PVC = "pvc"
STORAGE_OFFLOADING_TYPE_HOST_PATH = "hostPath"


def _normalize_storage_type(value: object) -> str:
    normalized = str(value or STORAGE_OFFLOADING_TYPE_PVC).strip()
    compact = normalized.replace("_", "").replace("-", "").lower()
    if compact in {"pvc", "persistentvolumeclaim"}:
        return STORAGE_OFFLOADING_TYPE_PVC
    if compact == "hostpath":
        return STORAGE_OFFLOADING_TYPE_HOST_PATH
    raise CommandError(
        "deployment profile options.storage_offloading.type must be 'pvc' or 'hostPath'"
    )


def storage_offloading_config(plan: ResolvedRunPlan) -> dict[str, Any] | None:
    raw_config = plan.deployment.options.get("storage_offloading")
    if raw_config in (None, "", False):
        return None
    if not isinstance(raw_config, dict):
        raise CommandError(
            "deployment profile options.storage_offloading must be a mapping"
        )
    if raw_config.get("enabled") is False:
        return None

    port_raw = raw_config.get("storage_events_port", 5559)
    try:
        storage_events_port = int(str(port_raw).strip())
    except ValueError as exc:
        raise CommandError(
            "deployment profile options.storage_offloading.storage_events_port "
            "must be an integer"
        ) from exc
    if storage_events_port <= 0:
        raise CommandError(
            "deployment profile options.storage_offloading.storage_events_port "
            "must be greater than 0"
        )

    storage_type = _normalize_storage_type(raw_config.get("type"))
    mount_path = str(raw_config.get("mount_path") or "/mnt/files-storage").rstrip("/")
    if not mount_path:
        raise CommandError(
            "deployment profile options.storage_offloading.mount_path must not be empty"
        )
    if not mount_path.startswith("/"):
        raise CommandError(
            "deployment profile options.storage_offloading.mount_path must be absolute"
        )

    directory_path = str(
        raw_config.get("directory_path") or f"{mount_path}/kv-cache"
    ).rstrip("/")
    if not directory_path:
        raise CommandError(
            "deployment profile options.storage_offloading.directory_path "
            "must not be empty"
        )
    if not directory_path.startswith("/"):
        raise CommandError(
            "deployment profile options.storage_offloading.directory_path "
            "must be absolute"
        )

    config: dict[str, Any] = {
        "type": storage_type,
        "mount_path": mount_path,
        "directory_path": directory_path,
        "storage_events_port": storage_events_port,
    }

    if storage_type == STORAGE_OFFLOADING_TYPE_PVC:
        config.update(
            {
                "pvc_name": str(
                    raw_config.get("pvc_name") or "llm-d-storage-offloading-cache"
                ).strip(),
                "storage_class": str(
                    raw_config.get("storage_class") or "nfs-rwx"
                ).strip(),
                "size": str(raw_config.get("size") or "2500Gi").strip(),
                "access_mode": str(
                    raw_config.get("access_mode") or "ReadWriteMany"
                ).strip(),
            }
        )
        for key in ("pvc_name", "size", "access_mode"):
            if not config[key]:
                raise CommandError(
                    f"deployment profile options.storage_offloading.{key} "
                    "must not be empty"
                )
        return config

    host_path = str(
        raw_config.get("host_path")
        or f"/var/lib/benchflow/storage-offloading/{plan.deployment.release_name}"
    ).strip()
    if not host_path:
        raise CommandError(
            "deployment profile options.storage_offloading.host_path must not be empty"
        )
    if not host_path.startswith("/"):
        raise CommandError(
            "deployment profile options.storage_offloading.host_path must be absolute"
        )
    config.update(
        {
            "host_path": host_path,
            "host_path_type": str(
                raw_config.get("host_path_type") or "DirectoryOrCreate"
            ).strip(),
        }
    )
    return config
