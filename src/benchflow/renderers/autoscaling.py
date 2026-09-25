from __future__ import annotations

from copy import deepcopy
from typing import Any

from jinja2 import TemplateError

from ..assets import render_jinja_text
from ..models import ResolvedRunPlan, ValidationError


def render_scaled_object(plan: ResolvedRunPlan) -> dict[str, Any] | None:
    """Render the optional raw KEDA ScaledObject from a deployment profile."""
    raw = plan.deployment.options.get("scaled_object")
    if raw in (None, {}):
        return None
    if not isinstance(raw, dict):
        raise ValidationError("deployment options.scaled_object must be a mapping")

    variables = {
        "release_name": plan.deployment.release_name,
        "namespace": plan.deployment.namespace,
        "model_name": plan.model.resolved_name(),
    }

    def render_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: render_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [render_value(item) for item in value]
        if isinstance(value, str) and "{{" in value:
            try:
                return render_jinja_text(value, variables)
            except TemplateError as exc:
                raise ValidationError(
                    "invalid template in deployment options.scaled_object"
                ) from exc
        return value

    manifest = render_value(deepcopy(raw))
    if not isinstance(manifest, dict):  # pragma: no cover - guarded above
        raise ValidationError("deployment options.scaled_object must be a mapping")
    if manifest.get("apiVersion") != "keda.sh/v1alpha1":
        raise ValidationError(
            "deployment options.scaled_object.apiVersion must be keda.sh/v1alpha1"
        )
    if manifest.get("kind") != "ScaledObject":
        raise ValidationError(
            "deployment options.scaled_object.kind must be ScaledObject"
        )

    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise ValidationError(
            "deployment options.scaled_object.metadata must be a mapping"
        )
    name = str(metadata.get("name") or "").strip()
    if not name:
        raise ValidationError(
            "deployment options.scaled_object.metadata.name must not be empty"
        )
    metadata["name"] = name
    metadata["namespace"] = plan.deployment.namespace
    metadata["labels"] = {
        **plan.metadata.labels,
        **(metadata.get("labels") or {}),
        "benchflow.io/release": plan.deployment.release_name,
        "benchflow.io/managed-by": "benchflow",
        "benchflow.io/resource": "scaled-object",
    }

    spec = manifest.get("spec")
    if not isinstance(spec, dict):
        raise ValidationError("deployment options.scaled_object.spec must be a mapping")
    scale_target_ref = spec.get("scaleTargetRef")
    if not isinstance(scale_target_ref, dict):
        raise ValidationError(
            "deployment options.scaled_object.spec.scaleTargetRef must be a mapping"
        )
    if not str(scale_target_ref.get("name") or "").strip():
        raise ValidationError(
            "deployment options.scaled_object.spec.scaleTargetRef.name must not be empty"
        )
    triggers = spec.get("triggers")
    if not isinstance(triggers, list) or not triggers:
        raise ValidationError(
            "deployment options.scaled_object.spec.triggers must be a non-empty list"
        )
    return manifest


def scaled_object_name(plan: ResolvedRunPlan) -> str | None:
    manifest = render_scaled_object(plan)
    if manifest is None:
        return None
    return str(manifest["metadata"]["name"])
