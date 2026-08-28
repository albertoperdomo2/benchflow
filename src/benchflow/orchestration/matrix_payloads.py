from __future__ import annotations

import base64
import copy
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from ..cluster import (
    create_manifest,
    require_any_command,
    run_command,
    run_json_command,
)
from ..contracts import ValidationError
from ..models import sanitize_name

RUN_PLANS_PARAM = "RUN_PLANS"
RUN_PLANS_CONFIGMAP_PARAM = "RUN_PLANS_CONFIGMAP"
RUN_PLANS_CONFIGMAP_LABEL = "benchflow.io/run-plans-configmap"
RUN_PLANS_CONFIGMAP_KEY = "run-plans.json"
RUN_PLANS_COMPRESSED_CONFIGMAP_KEY = "run-plans.json.gz"
RUN_PLANS_ANNOTATION = "benchflow.io/run-plans-json"
RUN_PLANS_ENCODING_ANNOTATION = "benchflow.io/run-plans-encoding"
RUN_PLANS_CHECKSUM_ANNOTATION = "benchflow.io/run-plans-sha256"
RUN_PLANS_GZIP_ENCODING = "gzip"
MATRIX_RESULTS_CONFIGMAP_PARAM = "MATRIX_RESULTS_CONFIGMAP"
MATRIX_RESULTS_CONFIGMAP_LABEL = "benchflow.io/matrix-results-configmap"
MATRIX_PIPELINE_NAME = "benchflow-matrix"
EXECUTION_NAME_LABEL = "benchflow.io/execution-name"

# ConfigMaps are limited to 1 MiB. Keep enough room for Base64 expansion and
# object metadata instead of relying on the API server to reject an oversized
# matrix payload with an opaque RequestEntityTooLarge response.
_MAX_COMPRESSED_RUN_PLANS_BYTES = 700 * 1024


def matrix_run_plans_configmap_name(execution_name: str) -> str:
    return sanitize_name(f"{execution_name}-run-plans", max_length=63)


def matrix_results_configmap_name(execution_name: str) -> str:
    digest = hashlib.sha1(execution_name.encode("utf-8")).hexdigest()[:8]
    prefix = sanitize_name(execution_name, max_length=45)
    return sanitize_name(f"{prefix}-results-{digest}", max_length=63)


def matrix_run_plans_configmap_name_from_labels(labels: dict[str, str] | None) -> str:
    if not labels:
        return ""
    return str(labels.get(RUN_PLANS_CONFIGMAP_LABEL) or "").strip()


def is_matrix_manifest(manifest: dict[str, Any]) -> bool:
    spec = manifest.get("spec", {}) or {}
    pipeline_ref = spec.get("pipelineRef", {}) or {}
    pipeline_name = str(pipeline_ref.get("name") or "").strip()
    if pipeline_name == MATRIX_PIPELINE_NAME:
        return True
    labels = (manifest.get("metadata", {}) or {}).get("labels", {}) or {}
    return (
        str(labels.get("benchflow.io/platform") or "").strip() == "matrix"
        and str(labels.get("benchflow.io/mode") or "").strip() == "matrix"
    )


def _run_plans_json_from_manifest(manifest: dict[str, Any]) -> str:
    metadata = manifest.get("metadata", {}) or {}
    annotations = metadata.get("annotations", {}) or {}
    annotation_value = str(annotations.get(RUN_PLANS_ANNOTATION) or "").strip()
    if annotation_value:
        return annotation_value
    params = (manifest.get("spec", {}) or {}).get("params", []) or []
    for param in params:
        if str(param.get("name") or "").strip() != RUN_PLANS_PARAM:
            continue
        value = param.get("value")
        if value is None:
            raise ValidationError("matrix execution RUN_PLANS param is empty")
        return str(value)
    raise ValidationError("matrix execution manifest is missing RUN_PLANS")


def create_matrix_run_plans_configmap(
    *,
    namespace: str,
    execution_name: str,
    run_plans_json: str,
) -> str:
    configmap_name = matrix_run_plans_configmap_name(execution_name)
    raw_payload = run_plans_json.encode("utf-8")
    compressed_payload = gzip.compress(raw_payload, mtime=0)
    if len(compressed_payload) > _MAX_COMPRESSED_RUN_PLANS_BYTES:
        raise ValidationError(
            "compressed matrix RunPlans exceed the safe ConfigMap payload limit "
            f"({len(compressed_payload)} > {_MAX_COMPRESSED_RUN_PLANS_BYTES} bytes); "
            "split the experiment into smaller matrices"
        )
    payload = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": configmap_name,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/name": "benchflow",
                "benchflow.io/managed-by": "benchflow",
                EXECUTION_NAME_LABEL: execution_name,
                RUN_PLANS_CONFIGMAP_LABEL: configmap_name,
            },
            "annotations": {
                RUN_PLANS_ENCODING_ANNOTATION: RUN_PLANS_GZIP_ENCODING,
                RUN_PLANS_CHECKSUM_ANNOTATION: hashlib.sha256(raw_payload).hexdigest(),
            },
        },
        "binaryData": {
            RUN_PLANS_COMPRESSED_CONFIGMAP_KEY: base64.b64encode(
                compressed_payload
            ).decode("ascii")
        },
    }
    create_manifest(
        json.dumps(payload, separators=(",", ":"), sort_keys=True), namespace
    )
    return configmap_name


def decode_matrix_run_plans_configmap(payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata", {}) or {}
    annotations = metadata.get("annotations", {}) or {}
    encoding = str(annotations.get(RUN_PLANS_ENCODING_ANNOTATION) or "").strip()

    if not encoding:
        data = payload.get("data", {}) or {}
        raw_payload = str(data.get(RUN_PLANS_CONFIGMAP_KEY) or "")
        if not raw_payload:
            raise ValidationError(
                f"matrix RunPlans ConfigMap is missing data.{RUN_PLANS_CONFIGMAP_KEY}"
            )
    elif encoding == RUN_PLANS_GZIP_ENCODING:
        binary_data = payload.get("binaryData", {}) or {}
        encoded_payload = str(binary_data.get(RUN_PLANS_COMPRESSED_CONFIGMAP_KEY) or "")
        if not encoded_payload:
            raise ValidationError(
                "compressed matrix RunPlans ConfigMap is missing "
                f"binaryData.{RUN_PLANS_COMPRESSED_CONFIGMAP_KEY}"
            )
        try:
            compressed_payload = base64.b64decode(encoded_payload, validate=True)
            raw_bytes = gzip.decompress(compressed_payload)
            raw_payload = raw_bytes.decode("utf-8")
        except (ValueError, OSError, UnicodeDecodeError) as exc:
            raise ValidationError(
                "compressed matrix RunPlans ConfigMap payload is invalid"
            ) from exc
        expected_checksum = str(
            annotations.get(RUN_PLANS_CHECKSUM_ANNOTATION) or ""
        ).strip()
        actual_checksum = hashlib.sha256(raw_bytes).hexdigest()
        if not expected_checksum:
            raise ValidationError(
                "compressed matrix RunPlans ConfigMap is missing its SHA-256 checksum"
            )
        if actual_checksum != expected_checksum:
            raise ValidationError(
                "compressed matrix RunPlans ConfigMap checksum does not match"
            )
    else:
        raise ValidationError(
            f"unsupported matrix RunPlans ConfigMap encoding: {encoding!r}"
        )

    try:
        decoded = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise ValidationError(
            "matrix RunPlans ConfigMap contains invalid JSON"
        ) from exc
    if not isinstance(decoded, list) or not decoded:
        raise ValidationError(
            "matrix RunPlans ConfigMap must contain a non-empty JSON array"
        )
    return raw_payload


def write_matrix_run_plans_file(
    *, namespace: str, configmap_name: str, output_file: Path
) -> None:
    kubectl_cmd = require_any_command("oc", "kubectl")
    command = [kubectl_cmd, "get", "configmap", configmap_name]
    if namespace:
        command.extend(["-n", namespace])
    command.extend(["-o", "json"])
    payload = run_json_command(command)
    raw_payload = decode_matrix_run_plans_configmap(payload)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(raw_payload, encoding="utf-8")


def delete_matrix_run_plans_configmap(namespace: str, configmap_name: str) -> None:
    if not configmap_name:
        return
    kubectl_cmd = require_any_command("oc", "kubectl")
    run_command(
        [
            kubectl_cmd,
            "delete",
            "configmap",
            configmap_name,
            "-n",
            namespace,
            "--ignore-not-found",
            "--wait=false",
        ],
        check=False,
    )


def adopt_matrix_configmap(
    *,
    namespace: str,
    configmap_name: str,
    owner_payload: dict[str, Any],
) -> None:
    if not configmap_name:
        return
    metadata = owner_payload.get("metadata", {}) or {}
    owner_name = str(metadata.get("name") or "").strip()
    owner_uid = str(metadata.get("uid") or "").strip()
    if not owner_name or not owner_uid:
        return
    kubectl_cmd = require_any_command("oc", "kubectl")
    run_command(
        [
            kubectl_cmd,
            "patch",
            "configmap",
            configmap_name,
            "-n",
            namespace,
            "--type",
            "merge",
            "-p",
            json.dumps(
                {
                    "metadata": {
                        "ownerReferences": [
                            {
                                "apiVersion": "tekton.dev/v1",
                                "kind": "PipelineRun",
                                "name": owner_name,
                                "uid": owner_uid,
                                "controller": False,
                                "blockOwnerDeletion": False,
                            }
                        ]
                    }
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
        ]
    )


def adopt_matrix_run_plans_configmap(
    *,
    namespace: str,
    configmap_name: str,
    owner_payload: dict[str, Any],
) -> None:
    adopt_matrix_configmap(
        namespace=namespace,
        configmap_name=configmap_name,
        owner_payload=owner_payload,
    )


def create_matrix_results_configmap(
    *,
    namespace: str,
    execution_name: str,
    owner_payload: dict[str, Any] | None = None,
) -> str:
    configmap_name = matrix_results_configmap_name(execution_name)
    kubectl_cmd = require_any_command("oc", "kubectl")
    existing = run_command(
        [
            kubectl_cmd,
            "get",
            "configmap",
            configmap_name,
            "-n",
            namespace,
        ],
        capture_output=True,
        check=False,
    )
    if existing.returncode == 0:
        if owner_payload is not None:
            adopt_matrix_configmap(
                namespace=namespace,
                configmap_name=configmap_name,
                owner_payload=owner_payload,
            )
        return configmap_name
    payload = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": configmap_name,
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/name": "benchflow",
                "benchflow.io/managed-by": "benchflow",
                EXECUTION_NAME_LABEL: execution_name,
                MATRIX_RESULTS_CONFIGMAP_LABEL: configmap_name,
            },
        },
        "data": {},
    }
    create_manifest(
        json.dumps(payload, separators=(",", ":"), sort_keys=True),
        namespace,
    )
    if owner_payload is not None:
        adopt_matrix_configmap(
            namespace=namespace,
            configmap_name=configmap_name,
            owner_payload=owner_payload,
        )
    return configmap_name


def matrix_result_key(child_execution_name: str) -> str:
    cleaned = sanitize_name(child_execution_name, max_length=240)
    return f"{cleaned or 'child'}.json"


def patch_matrix_result(
    *,
    namespace: str,
    configmap_name: str,
    child_execution_name: str,
    record: dict[str, Any],
) -> None:
    if not configmap_name:
        raise ValidationError("matrix results ConfigMap name must not be empty")
    key = matrix_result_key(child_execution_name)
    kubectl_cmd = require_any_command("oc", "kubectl")
    run_command(
        [
            kubectl_cmd,
            "patch",
            "configmap",
            configmap_name,
            "-n",
            namespace,
            "--type",
            "merge",
            "-p",
            json.dumps(
                {
                    "data": {
                        key: json.dumps(
                            record,
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    }
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
        ]
    )


def read_matrix_results_configmap(
    *,
    namespace: str,
    configmap_name: str,
) -> list[dict[str, Any]]:
    if not configmap_name:
        return []
    kubectl_cmd = require_any_command("oc", "kubectl")
    payload = run_json_command(
        [kubectl_cmd, "get", "configmap", configmap_name, "-n", namespace, "-o", "json"]
    )
    data = payload.get("data", {}) or {}
    records: list[dict[str, Any]] = []
    for key in sorted(data):
        try:
            record = json.loads(str(data[key]))
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def materialize_matrix_run_plans_configmap(
    *,
    namespace: str,
    execution_name: str,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    if not is_matrix_manifest(manifest):
        return manifest, ""
    run_plans_json = _run_plans_json_from_manifest(manifest)
    configmap_name = create_matrix_run_plans_configmap(
        namespace=namespace,
        execution_name=execution_name,
        run_plans_json=run_plans_json,
    )
    rendered = copy.deepcopy(manifest)
    metadata = rendered.setdefault("metadata", {})
    labels = metadata.setdefault("labels", {})
    labels[RUN_PLANS_CONFIGMAP_LABEL] = configmap_name
    annotations = metadata.setdefault("annotations", {})
    annotations.pop(RUN_PLANS_ANNOTATION, None)
    spec = rendered.setdefault("spec", {})
    params = list(spec.get("params", []) or [])
    updated_params: list[dict[str, Any]] = []
    inserted = False
    for param in params:
        param_name = str(param.get("name") or "").strip()
        if param_name in {RUN_PLANS_PARAM, RUN_PLANS_CONFIGMAP_PARAM}:
            updated_params.append(
                {"name": RUN_PLANS_CONFIGMAP_PARAM, "value": configmap_name}
            )
            inserted = True
            continue
        updated_params.append(param)
    if not inserted:
        updated_params.append(
            {"name": RUN_PLANS_CONFIGMAP_PARAM, "value": configmap_name}
        )
    spec["params"] = updated_params
    return rendered, configmap_name
