from __future__ import annotations

import re
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined


_PLACEHOLDER_RE = re.compile(r"{{\s*([A-Z0-9_]+)\s*}}")


def _assets_root():
    return files("benchflow").joinpath("assets")


def asset_text(relative_path: str | Path) -> str:
    path = Path(relative_path)
    return _assets_root().joinpath(*path.parts).read_text(encoding="utf-8")


def _render_string(value: str, variables: dict[str, Any]) -> Any:
    match = _PLACEHOLDER_RE.fullmatch(value)
    if match:
        return variables.get(match.group(1))

    def _replace(found: re.Match[str]) -> str:
        replacement = variables.get(found.group(1))
        return "" if replacement is None else str(replacement)

    return _PLACEHOLDER_RE.sub(_replace, value)


def _render_value(value: Any, variables: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        rendered: dict[str, Any] = {}
        for key, item in value.items():
            rendered_item = _render_value(item, variables)
            if rendered_item is None:
                continue
            rendered[key] = rendered_item
        return rendered
    if isinstance(value, list):
        rendered_items = [_render_value(item, variables) for item in value]
        return [item for item in rendered_items if item is not None]
    if isinstance(value, str):
        return _render_string(value, variables)
    return value


def render_yaml_documents(
    relative_path: str | Path, variables: dict[str, Any]
) -> list[dict[str, Any]]:
    documents = list(yaml.safe_load_all(asset_text(relative_path)))
    rendered = [_render_value(document, variables) for document in documents]
    return [document for document in rendered if document is not None]


def _jinja_environment() -> Environment:
    return Environment(
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_jinja_text(source: str, variables: dict[str, Any]) -> str:
    template = _jinja_environment().from_string(source)
    return template.render(**variables)


def render_jinja_yaml_document(
    relative_path: str | Path, variables: dict[str, Any]
) -> dict[str, Any]:
    rendered = render_jinja_text(asset_text(relative_path), variables)
    document = yaml.safe_load(rendered)
    if not isinstance(document, dict):
        raise ValueError(f"{relative_path} did not render to a YAML mapping document")
    return document
