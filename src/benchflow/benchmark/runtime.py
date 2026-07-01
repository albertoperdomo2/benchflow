import json
import logging
import math
import subprocess
import sys
import shutil
import os
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import click
import mlflow
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

from ..mlflow_compat import configure_mlflow_tracking, create_mlflow_client
from ..ui import configure_logging, emit
from .comparison_metrics import build_comparison_metric_panels

# Disable SSL warnings if using self-signed certificates
if os.environ.get("MLFLOW_TRACKING_INSECURE_TLS", "false").lower() == "true":
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    from .processor import BenchmarkProcessor

    PROCESSOR_AVAILABLE = True
except ImportError:
    PROCESSOR_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("BenchmarkProcessor not available - reports will not be generated")


# Configure logging level from environment variable
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
configure_logging(log_level)
logger = logging.getLogger(__name__)

_MLFLOW_METRIC_SECTIONS: dict[str, set[str]] = {
    "requests": {
        "total_requests",
        "successful_requests",
        "failed_requests",
        "error_rate",
        "request_concurrency_mean",
    },
    "throughput": {
        "throughput_requests_per_sec",
        "throughput_output_tokens_per_sec",
        "total_tokens_per_second",
        "total_input_tokens",
        "total_output_tokens",
        "total_tokens",
    },
    "e2e_latency": {
        "latency_mean_sec",
        "latency_median_sec",
        "latency_p50_sec",
        "latency_p90_sec",
        "latency_p95_sec",
        "latency_p99_sec",
    },
    "ttft": {
        "ttft_mean_ms",
        "ttft_median_ms",
        "ttft_p95_ms",
        "ttft_p99_ms",
    },
    "tpot": {
        "tpot_mean_ms",
        "tpot_median_ms",
        "tpot_p95_ms",
        "tpot_p99_ms",
    },
    "itl": {
        "itl_mean_ms",
        "itl_median_ms",
        "itl_p95_ms",
        "itl_p99_ms",
    },
}

DATA_PROFILE_PARAMS = {
    "prompt_tokens",
    "prompt_tokens_stdev",
    "prompt_tokens_min",
    "prompt_tokens_max",
    "output_tokens",
    "output_tokens_stdev",
    "output_tokens_min",
    "output_tokens_max",
    "turns",
    "prefix_tokens",
    "prefix_count",
    "prefix_buckets",
}


def _is_data_profile_param(key: str, value: Any) -> bool:
    if value is None:
        return False
    # Pre-warmup is an execution phase used to populate caches; it does not
    # define the benchmark workload shape being compared.
    if key.startswith("pre_warmup_"):
        return False
    return key in DATA_PROFILE_PARAMS


class BenchmarkExecutionError(RuntimeError):
    def __init__(self, message: str, *, run_id: str = "") -> None:
        super().__init__(message)
        self.run_id = run_id


def _mlflow_metric_name(metric_name: str) -> str:
    for section, names in _MLFLOW_METRIC_SECTIONS.items():
        if metric_name in names:
            return f"{section}/{metric_name}"
    return metric_name


def _metrics_for_mlflow(metrics: dict[str, Any]) -> dict[str, Any]:
    return {_mlflow_metric_name(key): value for key, value in metrics.items()}


def _stringify_data_profile_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return value


def _rates_to_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return ",".join(cleaned) if cleaned else None
    cleaned = str(value).strip()
    return cleaned or None


def _decode_guidellm_scalar(value: Any) -> Any:
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return raw
        lowered = raw.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "null":
            return None
        try:
            return int(raw)
        except ValueError:
            pass
        try:
            return float(raw)
        except ValueError:
            pass
        if raw[0] in "[{":
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
    return value


def _decode_guidellm_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _decode_guidellm_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_guidellm_value(item) for item in value]
    return _decode_guidellm_scalar(value)


def _parse_guidellm_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return _decode_guidellm_value(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return {}
        try:
            parsed_json = json.loads(raw)
        except json.JSONDecodeError:
            parsed_json = None
        if isinstance(parsed_json, dict):
            return _decode_guidellm_value(parsed_json)
        if "=" in raw:
            return {
                key: _decode_guidellm_scalar(item)
                for key, item in _parse_data_profile_config(raw).items()
            }
        return {"kind": raw}
    raise BenchmarkExecutionError(
        f"guidellm {field_name} must be a mapping, JSON object, or key=value string"
    )


def _parse_data_profile_config(data: str | None) -> dict[str, Any]:
    if not data:
        return {}

    raw = str(data).strip()
    if not raw:
        return {}

    try:
        parsed_json = json.loads(raw)
    except json.JSONDecodeError:
        parsed_json = None

    if isinstance(parsed_json, dict):
        parsed: dict[str, Any] = {}
        for key, value in parsed_json.items():
            clean_key = str(key).strip()
            if not clean_key:
                continue
            parsed[clean_key] = _stringify_data_profile_value(value)
        return parsed

    parsed: dict[str, Any] = {}
    for part in raw.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        clean_key = key.strip()
        if not clean_key:
            continue
        parsed[clean_key] = value.strip()
    return parsed


_GUIDELLM_PLURAL_FLAGS = {
    "constraints": "constraint",
    "data_preprocessors": "data_preprocessor",
    "labels": "label",
    "outputs": "output",
    "overrides": "override",
}


def _guidellm_flag_name(key: str) -> str:
    return f"--{_GUIDELLM_PLURAL_FLAGS.get(key, key).replace('_', '-')}"


def _merge_guidellm_mapping(
    args: dict[str, Any],
    key: str,
    updates: dict[str, Any],
) -> None:
    current = _parse_guidellm_mapping(args.pop(key, None), key)
    current.update(_decode_guidellm_value(updates))
    args[key] = current


def _ensure_guidellm_mapping(
    args: dict[str, Any],
    key: str,
    defaults: dict[str, Any],
    updates: dict[str, Any] | None = None,
) -> None:
    current = _parse_guidellm_mapping(args.pop(key, None), key)
    for default_key, default_value in _decode_guidellm_value(defaults).items():
        current.setdefault(default_key, default_value)
    if updates:
        current.update(_decode_guidellm_value(updates))
    args[key] = current


def _normalize_guidellm_benchmark_args(
    *,
    target: str,
    model: str,
    benchmark_args: dict[str, Any],
    output_path: str,
) -> dict[str, Any]:
    args = _decode_guidellm_value(benchmark_args)

    backend_defaults = {"kind": "openai_http", "timeout": 600}
    if target.startswith("https://"):
        backend_defaults["verify"] = False
    _ensure_guidellm_mapping(
        args,
        "backend",
        backend_defaults,
        {"target": target, "model": model},
    )

    if "profile" not in args:
        args["profile"] = {"kind": "synchronous"}
    if "data" not in args:
        args["data"] = {
            "kind": "synthetic_text",
            "prompt_tokens": 1000,
            "output_tokens": 1000,
        }
    _ensure_guidellm_mapping(
        args,
        "tokenizer",
        {"kind": "huggingface_auto", "model": model},
    )

    outputs = [{"kind": "json", "path": str(output_path)}]
    for output_key in ("output", "outputs"):
        if output_key not in args:
            continue
        value = args.pop(output_key)
        outputs.extend(value if isinstance(value, list) else [value])
    args["output"] = _decode_guidellm_value(outputs)

    args.pop("pre_warmup", None)
    args.setdefault("disable_console_interactive", True)
    return args


def _guidellm_cli_value(value: Any) -> str:
    decoded = _decode_guidellm_value(value)
    if isinstance(decoded, (dict, list)):
        return json.dumps(decoded, separators=(",", ":"))
    return str(decoded)


def _guidellm_override_value(value: Any) -> str:
    decoded = _decode_guidellm_value(value)
    if isinstance(decoded, list):
        return ",".join(str(item) for item in decoded)
    return _guidellm_cli_value(decoded)


def _append_guidellm_cli_arg(cmd: list[str], key: str, value: Any) -> None:
    if value is None or value == "":
        return

    flag = _guidellm_flag_name(key)
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return

    if key in {"label", "labels"} and isinstance(value, dict):
        for label_key, label_value in value.items():
            cmd.extend([flag, f"{label_key}={label_value}"])
        return

    if key in {"override", "overrides"}:
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, dict):
                for override_key, override_value in item.items():
                    cmd.extend(
                        [
                            flag,
                            str(override_key),
                            _guidellm_override_value(override_value),
                        ]
                    )
            elif isinstance(item, list | tuple) and len(item) == 2:
                cmd.extend([flag, str(item[0]), _guidellm_override_value(item[1])])
            else:
                cmd.extend([flag, _guidellm_cli_value(item)])
        return

    if isinstance(value, list):
        for item in value:
            if item is None or item == "":
                continue
            cmd.extend([flag, _guidellm_cli_value(item)])
        return

    cmd.extend([flag, _guidellm_cli_value(value)])


def build_guidellm_v07_command(
    target: str,
    model: str,
    benchmark_args: dict[str, Any],
    output_path: str,
) -> list[str]:
    args = _normalize_guidellm_benchmark_args(
        target=target,
        model=model,
        benchmark_args=benchmark_args,
        output_path=output_path,
    )
    cmd = ["guidellm", "run"]
    for key, value in args.items():
        _append_guidellm_cli_arg(cmd, key, value)
    return cmd


def _list_run_artifacts_recursively(artifact_uri: str, root_path: str) -> list[str]:
    repo = get_artifact_repository(artifact_uri)
    pending = [root_path]
    discovered: list[str] = []

    while pending:
        current_path = pending.pop()
        for entry in repo.list_artifacts(current_path):
            if entry.is_dir:
                pending.append(entry.path)
                continue
            discovered.append(entry.path)

    return sorted(discovered)


def _download_run_artifact(artifact_uri: str, artifact_path: str, dst_path: str) -> str:
    repo = get_artifact_repository(artifact_uri)
    return repo.download_artifacts(artifact_path, dst_path=dst_path)


def _resolve_accelerator(
    params: dict[str, Any], tags: dict[str, Any] | None = None
) -> str:
    placeholder_values = {"unknown", "n/a", "na", "none"}
    accelerator = str(params.get("accelerator") or "").strip()
    if accelerator and accelerator.lower() not in placeholder_values:
        return accelerator
    if tags is not None:
        accelerator = str(tags.get("accelerator") or "").strip()
        if accelerator and accelerator.lower() not in placeholder_values:
            return accelerator
    return accelerator or "unknown"


def _resolve_report_output_path(
    default_filename: str,
    *,
    output_dir: str | None = None,
    output_file: str | None = None,
) -> str:
    if output_file:
        resolved = Path(output_file)
    elif output_dir:
        resolved = Path(output_dir) / default_filename
    else:
        resolved = Path("/tmp") / default_filename
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return str(resolved)


def _get_nested(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely get a nested value from a dictionary."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, default)
    return d


def _sequence_value(value: Any, index: int) -> Any:
    if isinstance(value, list):
        if index < len(value):
            return value[index]
        return value[0] if value else None
    return value


def _join_optional_ints(values: list[int] | None) -> str | None:
    if not values:
        return None
    return ",".join(str(value) for value in values)


def _mlflow_step_from_value(value: Any) -> int | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(numeric) or numeric < 0:
        return None

    if numeric.is_integer():
        return int(numeric)

    rounded = max(1, int(math.ceil(numeric)))
    logger.warning(
        "MLflow metric steps must be integers; rounded load value %s to step %s",
        value,
        rounded,
    )
    return rounded


def _extract_guidellm_load_step(
    benchmark: Dict[str, Any],
    benchmark_index: int,
    *,
    rate_type: str,
) -> tuple[int, str, Any] | None:
    config = benchmark.get("config") or {}
    args = benchmark.get("args") or {}
    strategy = (
        config.get("strategy")
        or _get_nested(benchmark, "scheduler", "strategy")
        or args.get("strategy")
        or {}
    )
    profile = config.get("profile") or args.get("profile") or {}

    if rate_type == "concurrent":
        field_order = ("streams", "rate", "max_concurrency")
    elif rate_type == "throughput":
        field_order = ("max_concurrency", "streams", "rate")
    else:
        field_order = ("rate", "streams", "max_concurrency")

    field_labels = {
        "streams": "concurrency",
        "rate": "request_rate",
        "max_concurrency": "max_concurrency",
    }

    for field_name in field_order:
        value = strategy.get(field_name) if isinstance(strategy, dict) else None
        if value is None and isinstance(profile, dict):
            value = _sequence_value(profile.get(field_name), benchmark_index)
        step = _mlflow_step_from_value(value)
        if step is not None:
            return step, field_labels[field_name], value

    return None


def parse_multiturn_expression(expression: str, concurrency: int) -> str:
    """
    Parse expression containing '*concurrency' and replace with actual value.

    Examples:
        "2*concurrency" with concurrency=32 -> "64"
        "10*concurrency" with concurrency=64 -> "640"
        "128" with concurrency=32 -> "128"

    Args:
        expression: String expression that may contain '*concurrency'
        concurrency: The concurrency value to substitute

    Returns:
        Parsed string with concurrency substituted
    """
    expression = str(expression).strip()
    if "*concurrency" in expression.lower():
        # Extract the multiplier
        parts = expression.lower().split("*concurrency")
        try:
            multiplier = int(parts[0].strip())
            return str(multiplier * concurrency)
        except ValueError:
            logger.warning(f"Could not parse multiplier in expression: {expression}")
            return expression
    return expression


def parse_multiturn_data_param(data: str, concurrency: int) -> str:
    """
    Parse data parameter and replace *concurrency expressions.

    Example:
        "prompt_tokens=128,output_tokens=128,prefix_count=2*concurrency"
        with concurrency=32 becomes
        "prompt_tokens=128,output_tokens=128,prefix_count=64"

    Args:
        data: Data parameter string with potential *concurrency expressions
        concurrency: The concurrency value to substitute

    Returns:
        Parsed data string with concurrency values substituted
    """
    if not data:
        return data

    parts = []
    for part in data.split(","):
        if "=" in part:
            key, value = part.split("=", 1)
            parsed_value = parse_multiturn_expression(value.strip(), concurrency)
            parts.append(f"{key.strip()}={parsed_value}")
        else:
            parts.append(part.strip())

    return ",".join(parts)


def substitute_multiturn_expressions(value: Any, concurrency: int) -> Any:
    if isinstance(value, dict):
        return {
            key: substitute_multiturn_expressions(item, concurrency)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [substitute_multiturn_expressions(item, concurrency) for item in value]
    if not isinstance(value, str) or not _has_multiturn_expression(value):
        return value
    if "=" in value:
        return parse_multiturn_data_param(value, concurrency)
    return parse_multiturn_expression(value, concurrency)


def _guidellm_profile_mapping(benchmark_args: dict[str, Any]) -> dict[str, Any]:
    return _parse_guidellm_mapping(benchmark_args.get("profile"), "profile")


def _guidellm_override_items(benchmark_args: dict[str, Any]) -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    for key in ("override", "overrides"):
        value = benchmark_args.get(key)
        if value is None:
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, dict):
                items.extend(
                    (str(item_key), item_value) for item_key, item_value in item.items()
                )
            elif isinstance(item, list | tuple) and len(item) == 2:
                items.append((str(item[0]), item[1]))
    return items


def guidellm_profile_mapping(benchmark_args: dict[str, Any]) -> dict[str, Any]:
    return _guidellm_profile_mapping(benchmark_args)


def guidellm_backend_mapping(benchmark_args: dict[str, Any]) -> dict[str, Any]:
    return _parse_guidellm_mapping(benchmark_args.get("backend"), "backend")


def guidellm_data_mapping(benchmark_args: dict[str, Any]) -> dict[str, Any]:
    return _parse_guidellm_mapping(benchmark_args.get("data"), "data")


def _guidellm_load_field(benchmark_args: dict[str, Any]) -> str | None:
    for key, _value in _guidellm_override_items(benchmark_args):
        if key.startswith("profile."):
            field = key.split(".", 1)[1]
            if field in {"streams", "rate", "max_concurrency", "sweep_size"}:
                return field
    profile = _guidellm_profile_mapping(benchmark_args)
    for field in ("streams", "rate", "max_concurrency", "sweep_size"):
        if field in profile:
            return field
    return None


def guidellm_load_field(benchmark_args: dict[str, Any]) -> str | None:
    return _guidellm_load_field(benchmark_args)


def _guidellm_load_values(benchmark_args: dict[str, Any]) -> list[Any]:
    field = _guidellm_load_field(benchmark_args)
    if field is None:
        return []
    override_key = f"profile.{field}"
    for key, value in _guidellm_override_items(benchmark_args):
        if key == override_key:
            if isinstance(value, list):
                return value
            if isinstance(value, str) and "," in value:
                return [part.strip() for part in value.split(",") if part.strip()]
            return [] if value is None else [value]
    value = _guidellm_profile_mapping(benchmark_args).get(field)
    if isinstance(value, list):
        return value
    return [] if value is None else [value]


def guidellm_load_values(benchmark_args: dict[str, Any]) -> list[Any]:
    return _guidellm_load_values(benchmark_args)


def guidellm_constraints(benchmark_args: dict[str, Any]) -> list[Any]:
    values: list[Any] = []
    for key in ("constraint", "constraints"):
        value = benchmark_args.get(key)
        if value is None:
            continue
        values.extend(value if isinstance(value, list) else [value])
    return _decode_guidellm_value(values)


def _guidellm_args_for_load(
    benchmark_args: dict[str, Any],
    load_value: Any,
) -> dict[str, Any]:
    iteration_args = substitute_multiturn_expressions(benchmark_args, int(load_value))
    field = _guidellm_load_field(iteration_args)
    if field is None:
        return iteration_args
    override_key = f"profile.{field}"
    override_items = _guidellm_override_items(iteration_args)
    if any(key == override_key for key, _value in override_items):
        iteration_args.pop("override", None)
        iteration_args.pop("overrides", None)
        iteration_args["override"] = [
            {
                key: load_value if key == override_key else value
                for key, value in override_items
            }
        ]
    else:
        profile = _guidellm_profile_mapping(iteration_args)
        profile[field] = load_value
        iteration_args["profile"] = profile
    return iteration_args


def _coerce_profile_value(raw: Any) -> Any:
    if isinstance(raw, str):
        cleaned = raw.strip()
        if not cleaned:
            return cleaned
        try:
            return int(cleaned)
        except ValueError:
            try:
                return float(cleaned)
            except ValueError:
                return cleaned
    return raw


def _extract_data_profile_params(params: dict[str, Any]) -> dict[str, Any]:
    preferred_order = [
        "prompt_tokens",
        "prompt_tokens_stdev",
        "prompt_tokens_min",
        "prompt_tokens_max",
        "output_tokens",
        "output_tokens_stdev",
        "output_tokens_min",
        "output_tokens_max",
        "turns",
        "prefix_tokens",
        "prefix_count",
    ]
    extracted = {
        key: _coerce_profile_value(value)
        for key, value in params.items()
        if _is_data_profile_param(key, value)
    }
    ordered: dict[str, Any] = {}
    for key in preferred_order:
        if key in extracted:
            ordered[key] = extracted.pop(key)
    for key in sorted(extracted):
        ordered[key] = extracted[key]
    return ordered


def _has_multiturn_expression(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return any(_has_multiturn_expression(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_multiturn_expression(item) for item in value)
    return "*concurrency" in str(value).lower()


def _multiturn_mode_enabled(benchmark_args: dict[str, Any]) -> bool:
    return _has_multiturn_expression(benchmark_args)


def extract_metrics_from_benchmark(benchmark: Dict[str, Any]) -> Dict[str, Any]:
    metrics = {}
    try:
        all_metrics = benchmark.get("metrics", {})
        scheduler_metrics = benchmark.get("scheduler_metrics", {})
        run_stats = benchmark.get("run_stats", {})

        # Fallback from scheduler_metrics to run_stats for older versions
        requests_made = scheduler_metrics.get("requests_made", {}) or run_stats.get(
            "requests_made", {}
        )

        metric_map = {
            "total_requests": requests_made.get("total"),
            "successful_requests": requests_made.get("successful"),
            "failed_requests": requests_made.get("errored"),
            "throughput_requests_per_sec": _get_nested(
                all_metrics, "requests_per_second", "successful", "mean"
            ),
            "total_tokens_per_second": _get_nested(
                all_metrics, "tokens_per_second", "successful", "mean"
            ),
            "throughput_output_tokens_per_sec": _get_nested(
                all_metrics, "output_tokens_per_second", "successful", "mean"
            ),
            "request_concurrency_mean": _get_nested(
                all_metrics, "request_concurrency", "successful", "mean"
            ),
            "latency_mean_sec": _get_nested(
                all_metrics, "request_latency", "successful", "mean"
            ),
            "latency_median_sec": _get_nested(
                all_metrics, "request_latency", "successful", "median"
            ),
            "latency_p50_sec": _get_nested(
                all_metrics, "request_latency", "successful", "percentiles", "p50"
            ),
            "latency_p90_sec": _get_nested(
                all_metrics, "request_latency", "successful", "percentiles", "p90"
            ),
            "latency_p95_sec": _get_nested(
                all_metrics, "request_latency", "successful", "percentiles", "p95"
            ),
            "latency_p99_sec": _get_nested(
                all_metrics, "request_latency", "successful", "percentiles", "p99"
            ),
            "ttft_mean_ms": _get_nested(
                all_metrics, "time_to_first_token_ms", "successful", "mean"
            ),
            "ttft_median_ms": _get_nested(
                all_metrics, "time_to_first_token_ms", "successful", "median"
            ),
            "ttft_p95_ms": _get_nested(
                all_metrics,
                "time_to_first_token_ms",
                "successful",
                "percentiles",
                "p95",
            ),
            "ttft_p99_ms": _get_nested(
                all_metrics,
                "time_to_first_token_ms",
                "successful",
                "percentiles",
                "p99",
            ),
            "itl_mean_ms": _get_nested(
                all_metrics, "inter_token_latency_ms", "successful", "mean"
            ),
            "itl_median_ms": _get_nested(
                all_metrics, "inter_token_latency_ms", "successful", "median"
            ),
            "itl_p95_ms": _get_nested(
                all_metrics,
                "inter_token_latency_ms",
                "successful",
                "percentiles",
                "p95",
            ),
            "itl_p99_ms": _get_nested(
                all_metrics,
                "inter_token_latency_ms",
                "successful",
                "percentiles",
                "p99",
            ),
            "tpot_mean_ms": _get_nested(
                all_metrics, "time_per_output_token_ms", "successful", "mean"
            ),
            "tpot_median_ms": _get_nested(
                all_metrics, "time_per_output_token_ms", "successful", "median"
            ),
            "tpot_p95_ms": _get_nested(
                all_metrics,
                "time_per_output_token_ms",
                "successful",
                "percentiles",
                "p95",
            ),
            "tpot_p99_ms": _get_nested(
                all_metrics,
                "time_per_output_token_ms",
                "successful",
                "percentiles",
                "p99",
            ),
            "total_input_tokens": _get_nested(
                all_metrics, "prompt_token_count", "successful", "total_sum"
            ),
            "total_output_tokens": _get_nested(
                all_metrics, "output_token_count", "successful", "total_sum"
            ),
        }

        # Add only non-None metrics
        metrics = {k: v for k, v in metric_map.items() if v is not None}

        # Calculated metrics
        if metrics.get("total_requests", 0) > 0 and "failed_requests" in metrics:
            metrics["error_rate"] = (
                metrics["failed_requests"] / metrics["total_requests"]
            )
        elif "total_requests" in metrics:
            metrics["error_rate"] = 0.0

        total_input = metrics.get("total_input_tokens", 0)
        total_output = metrics.get("total_output_tokens", 0)
        if total_input > 0 or total_output > 0:
            metrics["total_tokens"] = total_input + total_output

        logger.info(f"Extracted {len(metrics)} metrics from benchmark object")
        return metrics

    except Exception as e:
        logger.error(
            f"Error extracting metrics from benchmark object: {e}", exc_info=True
        )
        return {}


def run_guidellm_cli(
    target: str,
    model: str,
    benchmark_args: dict[str, Any],
    output_path: str = "benchmark_output.json",
) -> tuple[str, str]:
    output_path_obj = Path(output_path)
    cmd = build_guidellm_v07_command(
        target=target,
        model=model,
        benchmark_args=benchmark_args,
        output_path=str(output_path_obj),
    )

    logger.info(f"Running guidellm command: {' '.join(cmd)}")

    console_log_path = str(
        output_path_obj.with_name(f"{output_path_obj.stem}_console.log")
    )

    try:
        with open(console_log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            for line in process.stdout:
                emit(line, end="")
                log_file.write(line)
                log_file.flush()

            return_code = process.wait()

            if return_code != 0:
                logger.error(f"Guidellm command failed with return code {return_code}")
                raise RuntimeError(
                    "guidellm benchmark command failed "
                    f"(exit code {return_code}); see {console_log_path}"
                )
            else:
                logger.info("Guidellm completed successfully")

        return str(output_path_obj), console_log_path

    except Exception as e:
        logger.error(f"Guidellm command failed: {e}")
        raise


def _pre_warmup_value(pre_warmup: Any, key: str, default: Any = None) -> Any:
    if pre_warmup is None:
        return default
    if isinstance(pre_warmup, dict):
        value = pre_warmup.get(key, default)
    else:
        value = getattr(pre_warmup, key, None)
        if value is None:
            value = getattr(pre_warmup, "args", {}).get(key, default)
    return default if value is None else value


def _pre_warmup_enabled(pre_warmup: Any) -> bool:
    return bool(_pre_warmup_value(pre_warmup, "enabled", False))


def _run_guidellm_pre_warmup(
    *,
    target: str,
    model: str,
    benchmark_args: dict[str, Any],
    output_dir: str,
    pre_warmup: Any,
) -> tuple[str, str] | None:
    if not _pre_warmup_enabled(pre_warmup):
        return None

    load_value = _pre_warmup_value(pre_warmup, "rate")
    if load_value is None:
        raise BenchmarkExecutionError("GuideLLM pre-warmup requires a rate")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_json = f"{output_dir}/pre_warmup_output.json"
    warmup_args = _guidellm_args_for_load(benchmark_args, int(load_value))
    warmup_args.pop("warmup", None)
    for key, value in (
        pre_warmup.items()
        if isinstance(pre_warmup, dict)
        else getattr(pre_warmup, "args", {}).items()
    ):
        if key not in {"enabled", "rate"} and value is not None:
            warmup_args[key] = value
    warmup_constraints = warmup_args.get("constraint") or warmup_args.get("constraints")

    logger.info(
        "Running GuideLLM pre-warmup: load=%s, constraints=%s",
        load_value,
        warmup_constraints if warmup_constraints is not None else "not set",
    )
    return run_guidellm_cli(
        target=target,
        model=model,
        benchmark_args=warmup_args,
        output_path=output_json,
    )


def generate_visualization_report(
    json_path: str,
    model: str,
    accelerator: str = None,
    version: str = None,
    tp_size: int = 1,
    runtime_args: str = "",
    output_dir: str = None,
    output_file: str = None,
    replicas: int = 1,
    notes: list[str] | None = None,
    repeat_section_legends: bool = False,
    include_total_throughput: bool = False,
) -> str:
    """
    Generate HTML visualization report from benchmark JSON.
    This is failure-proof - returns None if generation fails.

    Args:
        json_path: Path to benchmark JSON file
        model: Model name
        accelerator: Accelerator type
        version: Version identifier
        tp_size: Tensor parallelism size
        runtime_args: Runtime arguments
        output_dir: Output directory for HTML report
        output_file: Explicit HTML report path
        replicas: Number of replicas
        notes: Optional subtitle note lines
        repeat_section_legends: Repeat side legends per section for screenshots
        include_total_throughput: Render dashed total-throughput overlay in throughput charts

    Returns:
        Path to HTML report, or None if generation failed
    """
    if not PROCESSOR_AVAILABLE:
        logger.info("Skipping visualization - BenchmarkProcessor not available")
        return None

    try:
        logger.info("Generating visualization report...")

        # Get S3 configuration from environment
        s3_bucket = os.environ.get("S3_BUCKET", "psap-dashboard-data")
        s3_key = os.environ.get(
            "S3_KEY", "main/llmd-dashboard/llmd-dashboard.csv"
        )  # Primary key (legacy env var, not used when downloading both)

        # Auto-generate output filename
        model_short = model.split("/")[-1].replace(" ", "_").replace("-", "_").lower()
        version_str = version.lower() if version else "unknown"
        html_filename = f"{model_short}_tp{tp_size}_{version_str}_report.html"

        html_path = _resolve_report_output_path(
            html_filename,
            output_dir=output_dir,
            output_file=output_file,
        )

        processor = BenchmarkProcessor(
            json_path=json_path,
            s3_bucket=s3_bucket,
            s3_key=s3_key,
            accelerator=accelerator or "unknown",
            model_name=model,
            version=version or "unknown",
            tp_size=tp_size,
            runtime_args=runtime_args,
            output_html=html_path,
            replicas=replicas,
            notes=notes or [],
            repeat_section_legends=repeat_section_legends,
            include_total_throughput=include_total_throughput,
        )

        processor.process()

        if Path(html_path).exists():
            logger.info(f"Visualization report generated: {html_path}")
            return html_path
        else:
            logger.warning(
                "Visualization report generation completed but file not found"
            )
            return None

    except Exception as e:
        logger.warning(
            f"Visualization report generation failed (non-fatal): {e}", exc_info=True
        )
        return None


def _run_and_process_benchmark(
    target: str,
    model: str,
    benchmark_args: dict[str, Any],
    output_dir: str,
) -> tuple:
    """Helper to run guidellm and process results."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_json = f"{output_dir}/benchmark_output.json"

    json_path, console_log_path = run_guidellm_cli(
        target=target,
        model=model,
        benchmark_args=benchmark_args,
        output_path=output_json,
    )

    benchmarks = []
    if Path(json_path).exists():
        logger.info(f"Benchmark results saved to: {json_path}")
        with open(json_path, "r") as f:
            result_json = json.load(f)
        benchmarks = result_json.get("benchmarks", [])
        logger.info(f"Found {len(benchmarks)} benchmark results")
    else:
        raise FileNotFoundError(f"Benchmark output JSON not found: {json_path}")

    if not Path(console_log_path).exists():
        logger.warning(f"Console log not found: {console_log_path}")

    return json_path, console_log_path, benchmarks


def run_benchmark_without_mlflow(
    target: str,
    model: str,
    benchmark_args: dict[str, Any],
    pre_warmup: Any = None,
    output_dir: str = "/benchmark-results",
    accelerator: str = None,
    version: str = None,
    tp_size: int = 1,
    runtime_args: str = "",
    replicas: int = 1,
) -> str:
    """Run benchmark without MLflow tracking, saving results to specified directory."""
    load_values = _guidellm_load_values(benchmark_args)
    load_values_text = _rates_to_string(load_values)
    logger.info("Running benchmark without MLflow tracking")
    logger.info(
        "Starting benchmark for load values: "
        f"{load_values_text if load_values_text is not None else 'not set'}"
    )
    logger.info(f"Results will be saved to: {output_dir}")

    multiturn_mode = _multiturn_mode_enabled(benchmark_args)
    if multiturn_mode and not load_values:
        raise BenchmarkExecutionError(
            "multiturn benchmark requires profile load values to be set"
        )
    if multiturn_mode:
        logger.info(
            "Multiturn mode enabled - running separate commands per concurrency"
        )
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        _run_guidellm_pre_warmup(
            target=target,
            model=model,
            benchmark_args=benchmark_args,
            output_dir=output_dir,
            pre_warmup=pre_warmup,
        )
        logger.info(f"Running {len(load_values)} separate benchmark commands")

        for load_value in load_values:
            concurrency = int(load_value)
            iteration_args = _guidellm_args_for_load(benchmark_args, concurrency)

            logger.info(f"Starting benchmark for concurrency={concurrency}")
            logger.info(f"  Parsed benchmark args: {iteration_args}")

            output_json = f"{output_dir}/benchmark_output_rate_{concurrency}.json"
            json_path, console_log_path = run_guidellm_cli(
                target=target,
                model=model,
                benchmark_args=iteration_args,
                output_path=output_json,
            )

            benchmarks = []
            if Path(json_path).exists():
                logger.info(f"Benchmark results saved to: {json_path}")
                with open(json_path, "r") as f:
                    result_json = json.load(f)
                benchmarks = result_json.get("benchmarks", [])
                logger.info(f"Found {len(benchmarks)} benchmark results")
            else:
                raise FileNotFoundError(f"Benchmark output JSON not found: {json_path}")

            for i, benchmark in enumerate(benchmarks):
                metrics = extract_metrics_from_benchmark(benchmark)
                if metrics:
                    logger.info(
                        f"Benchmark {i + 1} metrics for concurrency={concurrency}: "
                        f"{json.dumps(metrics, indent=2)}"
                    )

            if Path(console_log_path).exists():
                logger.info(f"Console log saved to: {console_log_path}")

        logger.info(
            "Multiturn benchmarks completed. Visualization report generation skipped."
        )
        return output_dir

    _run_guidellm_pre_warmup(
        target=target,
        model=model,
        benchmark_args=benchmark_args,
        output_dir=output_dir,
        pre_warmup=pre_warmup,
    )

    json_path, console_log_path, benchmarks = _run_and_process_benchmark(
        target=target,
        model=model,
        benchmark_args=benchmark_args,
        output_dir=output_dir,
    )

    for i, benchmark in enumerate(benchmarks):
        metrics = extract_metrics_from_benchmark(benchmark)
        if metrics:
            logger.info(f"Benchmark {i + 1} metrics: {json.dumps(metrics, indent=2)}")

    if Path(console_log_path).exists():
        logger.info(f"Console log saved to: {console_log_path}")

    return json_path


def run_benchmark_with_mlflow(
    target: str,
    model: str,
    benchmark_args: dict[str, Any],
    pre_warmup: Any = None,
    accelerator: str = None,
    experiment_name: str = "guidellm-benchmarks",
    mlflow_tracking_uri: str = None,
    mlflow_run_id: str = "",
    tags: Dict[str, str] = None,
    version: str = None,
    tp_size: int = 1,
    runtime_args: str = "",
    replicas: str = "N/A",
    prefill_replicas: str = "N/A",
    decode_replicas: str = "N/A",
    output_dir: str | None = None,
) -> str:
    configure_mlflow_tracking(mlflow_tracking_uri)

    mlflow.set_experiment(experiment_name)
    load_values = _guidellm_load_values(benchmark_args)
    load_values_text = _rates_to_string(load_values)

    multiturn_mode = _multiturn_mode_enabled(benchmark_args)

    # Run name for the whole sweep
    # Use the execution name if provided by the backend, otherwise generate one
    execution_name = os.environ.get("EXECUTION_NAME", "")
    if execution_name:
        run_name = execution_name
    else:
        mode_suffix = "multiturn" if multiturn_mode else "sweep"
        run_name = f"{model.split('/')[-1]}_{mode_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info(
        "Starting benchmark sweep: load_values=%s",
        load_values_text if load_values_text is not None else "not set",
    )
    if multiturn_mode:
        logger.info(
            "Multiturn mode enabled - running separate commands per concurrency"
        )

    start_run_kwargs = (
        {"run_id": mlflow_run_id.strip()}
        if str(mlflow_run_id or "").strip()
        else {"run_name": run_name}
    )
    with mlflow.start_run(**start_run_kwargs) as run:
        try:
            # Common params for the whole sweep
            params = {
                "target": target,
                "model": model,
                "tp": tp_size,
                "replicas": replicas,
                "prefill_replicas": prefill_replicas,
                "decode_replicas": decode_replicas,
                "multiturn_mode": multiturn_mode,
            }
            profile_args = _guidellm_profile_mapping(benchmark_args)
            backend_args = guidellm_backend_mapping(benchmark_args)
            load_field = _guidellm_load_field(benchmark_args)
            if profile_args:
                params["profile"] = _stringify_data_profile_value(profile_args)
                if profile_args.get("kind") is not None:
                    params["profile_kind"] = str(profile_args["kind"])
            if backend_args:
                params["backend"] = _stringify_data_profile_value(backend_args)
                if backend_args.get("kind") is not None:
                    params["backend_type"] = str(backend_args["kind"])
            if load_field:
                params["load_field"] = load_field
            if load_values_text:
                params["load_values"] = load_values_text
                # Existing comparison reports and historical MLflow runs use
                # "rates"; keep the logged param compatible without requiring
                # profiles to carry a legacy rates key.
                params["rates"] = load_values_text
            for key, value in benchmark_args.items():
                if value is None:
                    continue
                if key == "data" and value:
                    if isinstance(value, dict):
                        params.update(
                            {
                                str(data_key): _stringify_data_profile_value(data_value)
                                for data_key, data_value in value.items()
                                if data_value is not None
                            }
                        )
                    else:
                        params.update(_parse_data_profile_config(str(value)))
                    continue
                if key in {"backend", "profile"}:
                    continue
                params[key] = _stringify_data_profile_value(value)
            if _pre_warmup_enabled(pre_warmup):
                params["pre_warmup_rate"] = _pre_warmup_value(pre_warmup, "rate")
                for key, value in (
                    pre_warmup.items()
                    if isinstance(pre_warmup, dict)
                    else getattr(pre_warmup, "args", {}).items()
                ):
                    if key not in {"enabled", "rate"} and value is not None:
                        params[f"pre_warmup_{key}"] = _stringify_data_profile_value(
                            value
                        )
            if accelerator:
                params["accelerator"] = accelerator
            if version:
                params["version"] = version

            mlflow.log_params(params)

            guidellm_version = os.environ.get("GUIDELLM_VERSION", "unknown")
            try:
                vllm_version = requests.get(f"{target}/version", verify=False).json()[
                    "version"
                ]
            except Exception:
                vllm_version = "unknown"

            default_tags = {
                "vllm_version": vllm_version,
                "guidellm_version": guidellm_version,
            }
            if execution_name:
                default_tags["execution_name"] = execution_name
            if tags:
                default_tags.update(tags)
            mlflow.set_tags(default_tags)

            warmup_artifacts = _run_guidellm_pre_warmup(
                target=target,
                model=model,
                benchmark_args=benchmark_args,
                output_dir=output_dir or "/tmp",
                pre_warmup=pre_warmup,
            )
            if warmup_artifacts:
                warmup_json, warmup_console_log = warmup_artifacts
                if Path(warmup_json).exists():
                    mlflow.log_artifact(warmup_json, "warmup")
                    logger.info("Logged pre-warmup JSON artifact")
                if Path(warmup_console_log).exists():
                    mlflow.log_artifact(warmup_console_log, "warmup")
                    logger.info("Logged pre-warmup console output")

            # Multi-turn mode: loop over concurrencies and run separate commands
            if multiturn_mode:
                if not load_values:
                    raise BenchmarkExecutionError(
                        "multiturn benchmark requires profile load values to be set",
                        run_id=run.info.run_id,
                    )
                logger.info(f"Running {len(load_values)} separate benchmark commands")
                target_output_dir = Path(output_dir or "/tmp")
                target_output_dir.mkdir(parents=True, exist_ok=True)
                successful_concurrencies: list[int] = []
                failed_concurrencies: list[tuple[str, str]] = []

                for load_value in load_values:
                    try:
                        concurrency = int(load_value)
                        logger.info(f"Starting benchmark for concurrency={concurrency}")
                        iteration_args = _guidellm_args_for_load(
                            benchmark_args, concurrency
                        )
                        logger.info(f"  Parsed benchmark args: {iteration_args}")

                        # Generate unique output paths for this concurrency
                        output_json = str(
                            target_output_dir
                            / f"benchmark_output_rate_{concurrency}.json"
                        )
                        console_log_path = output_json.replace(".json", "_console.log")

                        # Run guidellm for this concurrency only
                        json_path, console_log = run_guidellm_cli(
                            target=target,
                            model=model,
                            benchmark_args=iteration_args,
                            output_path=output_json,
                        )

                        # Process results
                        benchmarks = []
                        if Path(json_path).exists():
                            logger.info(f"Benchmark results saved to: {json_path}")
                            with open(json_path, "r") as f:
                                result_json = json.load(f)
                            benchmarks = result_json.get("benchmarks", [])
                            logger.info(f"Found {len(benchmarks)} benchmark results")
                        else:
                            raise FileNotFoundError(
                                f"Benchmark output JSON not found: {json_path}"
                            )

                        # Extract and log metrics with step=concurrency
                        for benchmark in benchmarks:
                            metrics = extract_metrics_from_benchmark(benchmark)
                            if metrics:
                                metrics["concurrency"] = concurrency
                                for key, value in _metrics_for_mlflow(metrics).items():
                                    mlflow.log_metric(key, value, step=concurrency)
                                logger.info(
                                    f"Logged {len(metrics)} metrics for concurrency={concurrency}"
                                )

                        # Log artifacts for this concurrency
                        if Path(json_path).exists():
                            mlflow.log_artifact(json_path, "results")
                            logger.info(
                                f"Logged JSON artifact for concurrency={concurrency}"
                            )

                        if Path(console_log).exists():
                            mlflow.log_artifact(console_log, "logs")
                            logger.info(
                                f"Logged console log for concurrency={concurrency}"
                            )

                        logger.info(
                            f"Completed benchmark for concurrency={concurrency}"
                        )
                        successful_concurrencies.append(concurrency)

                    except Exception as e:
                        logger.error(
                            f"Benchmark failed for concurrency={concurrency}: {e}",
                            exc_info=True,
                        )
                        failed_concurrencies.append(
                            (str(concurrency), str(e).strip() or type(e).__name__)
                        )
                        logger.info("Continuing with remaining concurrencies...")
                        continue

                if failed_concurrencies:
                    summary = ", ".join(
                        f"{concurrency} ({reason})"
                        for concurrency, reason in failed_concurrencies
                    )
                    if not successful_concurrencies:
                        logger.error("All multiturn concurrencies failed")
                    else:
                        logger.error(
                            "Multiturn benchmark completed with failed concurrencies: "
                            f"{summary}"
                        )
                    raise BenchmarkExecutionError(
                        "multiturn benchmark failed for concurrency value(s): "
                        f"{summary}",
                        run_id=run.info.run_id,
                    )

                if not successful_concurrencies:
                    raise BenchmarkExecutionError(
                        "multiturn benchmark produced no successful concurrency runs",
                        run_id=run.info.run_id,
                    )

                # NOTE: HTML report generation is skipped for multi-turn mode
                # Report generation will be handled separately after all runs complete
                logger.info(
                    "Multi-turn benchmarks completed. HTML report generation skipped (handle separately)."
                )

            else:
                # Original single-command mode (backward compatible)
                (
                    json_path,
                    console_log_path,
                    benchmarks,
                ) = _run_and_process_benchmark(
                    target=target,
                    model=model,
                    benchmark_args=benchmark_args,
                    output_dir=output_dir or "/tmp",
                )

                if not benchmarks:
                    logger.warning("No benchmarks found in JSON output")

                for benchmark_index, benchmark in enumerate(benchmarks):
                    load_step = _extract_guidellm_load_step(
                        benchmark,
                        benchmark_index,
                        rate_type="",
                    )
                    if load_step is None:
                        step_value, load_label, load_value = 0, "load_step", 0
                        logger.warning(
                            "Could not find GuideLLM load value. Metrics will be "
                            "logged at step 0."
                        )
                    else:
                        step_value, load_label, load_value = load_step

                    metrics = extract_metrics_from_benchmark(benchmark)
                    if metrics:
                        metrics[load_label] = load_value
                        for key, value in _metrics_for_mlflow(metrics).items():
                            mlflow.log_metric(key, value, step=step_value)
                        logger.info(
                            f"Logged {len(metrics)} metrics for step {step_value} "
                            f"({load_label}={load_value})"
                        )

                if Path(json_path).exists():
                    mlflow.log_artifact(json_path, "results")
                    logger.info("Logged full JSON artifact")

                if Path(console_log_path).exists():
                    mlflow.log_artifact(console_log_path, "logs")
                    logger.info("Logged console output")

            logger.info(f"Run completed: {run.info.run_id}")
            return run.info.run_id

        except Exception as e:
            logger.error(f"Benchmark sweep failed: {e}", exc_info=True)
            mlflow.log_param("error", str(e))
            raise BenchmarkExecutionError(
                f"Benchmark sweep failed: {e}", run_id=run.info.run_id
            ) from e


def fetch_mlflow_runs(run_ids: list, mlflow_tracking_uri: str = None) -> list:
    """
    Fetch MLflow runs by their IDs and download their benchmark JSON artifacts.

    Args:
        run_ids: List of MLflow run IDs
        mlflow_tracking_uri: MLflow tracking URI (optional)

    Returns:
        List of dictionaries containing run metadata and benchmark data.
        Each dict includes a 'composed_version' field that appends either the
        'epp' tag or, if absent, the deployment profile to the base version.
    """
    configure_mlflow_tracking(mlflow_tracking_uri)
    client = create_mlflow_client(mlflow_tracking_uri)

    runs_data = []

    for run_id in run_ids:
        try:
            logger.info(f"Fetching MLflow run: {run_id}")
            run = client.get_run(run_id)

            params = run.data.params
            tags = run.data.tags

            # Compose version with epp tag first, then deployment profile.
            base_version = params.get("version", "unknown")
            version_suffix = (
                tags.get("epp")
                or tags.get("deployment_profile")
                or tags.get("deployment_type")
                or ""
            ).strip()

            if version_suffix:
                composed_version = f"{base_version}-{version_suffix}"
                logger.info(
                    "Composed version: %s + suffix=%s -> %s",
                    base_version,
                    version_suffix,
                    composed_version,
                )
            else:
                composed_version = base_version
                logger.info(
                    "No epp, deployment_profile, or deployment_type tag found, "
                    "using base version: %s",
                    composed_version,
                )

            # Check if cached version exists
            cache_dir = f"/tmp/mlflow/{run_id}/results"
            cached_files = (
                list(Path(cache_dir).glob("benchmark*.json"))
                if Path(cache_dir).exists()
                else []
            )

            artifact_paths = []
            if cached_files:
                logger.info(
                    f"Using {len(cached_files)} cached artifact(s) for run {run_id}"
                )
                artifact_paths = [str(f) for f in sorted(cached_files)]
            else:
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                benchmark_files = [
                    path
                    for path in _list_run_artifacts_recursively(
                        run.info.artifact_uri, "results"
                    )
                    if path.startswith("results/benchmark") and path.endswith(".json")
                ]

                if not benchmark_files:
                    raise ValueError(f"No benchmark JSON files found for run {run_id}")

                logger.info(
                    f"Downloading {len(benchmark_files)} benchmark file(s) for run {run_id}"
                )

                # Download and cache all files
                for benchmark_file in benchmark_files:
                    downloaded_path = _download_run_artifact(
                        run.info.artifact_uri, benchmark_file, dst_path=cache_dir
                    )
                    cached_path = Path(cache_dir) / Path(benchmark_file).name
                    cached_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(downloaded_path, cached_path)
                    artifact_paths.append(str(cached_path))

                logger.info(
                    f"Downloaded and cached {len(artifact_paths)} artifact(s) for run {run_id}"
                )

            # Load all benchmark data files
            all_benchmarks = []
            for artifact_path in artifact_paths:
                with open(artifact_path, "r") as f:
                    data = json.load(f)
                    # Extract benchmarks from this file
                    if "benchmarks" in data:
                        all_benchmarks.extend(data["benchmarks"])

            # Combine all benchmarks into a single structure
            benchmark_data = (
                {"benchmarks": all_benchmarks} if all_benchmarks else {"benchmarks": []}
            )

            # Save combined benchmark data to a temporary file for processor
            combined_json_path = f"{cache_dir}/combined_benchmarks.json"
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            with open(combined_json_path, "w") as f:
                json.dump(benchmark_data, f)

            logger.info(
                f"Combined {len(all_benchmarks)} benchmark(s) from {len(artifact_paths)} file(s)"
            )

            runs_data.append(
                {
                    "run_id": run_id,
                    "params": params,
                    "tags": tags,
                    "artifact_uri": run.info.artifact_uri,
                    "composed_version": composed_version,
                    "benchmark_data": benchmark_data,
                    "artifact_path": combined_json_path,
                }
            )

            logger.info(f"Successfully fetched run {run_id}")

        except Exception as e:
            logger.error(f"Failed to fetch run {run_id}: {e}")
            raise

    return runs_data


def validate_runs_compatibility(runs_data: list) -> tuple:
    """
    Validate that runs have compatible configurations for plotting.

    Args:
        runs_data: List of run data dictionaries

    Returns:
        Tuple of (model, data_profile) if compatible

    Raises:
        ValueError if runs are incompatible
    """
    if not runs_data:
        raise ValueError("No runs provided for validation")

    # Extract model and data profile from first run
    first_run = runs_data[0]
    model = first_run["params"].get("model")

    first_profile = _extract_data_profile_params(first_run["params"])
    rate_configs = [first_run["params"].get("rates")]

    # Validate all runs have same configuration
    for run_data in runs_data[1:]:
        params = run_data["params"]

        if params.get("model") != model:
            raise ValueError(
                f"Model mismatch: {params.get('model')} != {model}. "
                f"All runs must use the same model."
            )

        rate_configs.append(params.get("rates"))

        current_profile = _extract_data_profile_params(params)
        if current_profile != first_profile:
            all_keys = sorted(set(first_profile) | set(current_profile))
            mismatch_parts = [
                f"{key}={current_profile.get(key)} != {first_profile.get(key)}"
                for key in all_keys
                if current_profile.get(key) != first_profile.get(key)
            ]
            raise ValueError(
                "Data profile mismatch: "
                + "; ".join(mismatch_parts)
                + ". All runs must use the same data profile."
            )

    profile_parts = [
        f"{param}={value}"
        for param, value in first_profile.items()
        if value is not None
    ]

    data_profile = ",".join(profile_parts) if profile_parts else None

    logger.info("All runs validated successfully:")
    logger.info(f"  Model: {model}")
    unique_rate_configs = {str(value) for value in rate_configs}
    if len(unique_rate_configs) == 1:
        logger.info(f"  Rate: {rate_configs[0]}")
    else:
        logger.info(
            "  Rate: mixed configurations; comparison report will use each "
            "benchmark row's own load-axis value"
        )
    logger.info(f"  Data profile: {data_profile}")

    return model, data_profile


def generate_plot_only_report(
    runs_data: list,
    versions: list = None,
    mlflow_tracking_uri: str = None,
    additional_csv_files: list = None,
    versions_override: dict = None,
    output_dir: str = None,
    output_file: str = None,
    notes: list[str] | None = None,
    repeat_section_legends: bool = False,
    include_total_throughput: bool = False,
    baseline_version: str | None = None,
    metrics_yaml_path: str | None = None,
) -> str:
    """
    Generate HTML report from existing MLflow runs without running benchmarks.

    Args:
        runs_data: List of run data dictionaries
        versions: List of versions to filter/compare (optional)
        mlflow_tracking_uri: MLflow tracking URI (optional)
        additional_csv_files: List of additional CSV file paths to include (optional)
        versions_override: Dictionary mapping old version names to new names (optional)
        output_dir: Output directory for auto-generated report filename (optional)
        output_file: Explicit report path (optional)
        notes: Optional subtitle note lines
        repeat_section_legends: Repeat side legends per section for screenshots
        include_total_throughput: Render dashed total-throughput overlay in throughput charts
        baseline_version: Optional composed version name to use as the comparison-table baseline
        metrics_yaml_path: Optional report-metrics YAML path for archived Prometheus plots

    Returns:
        Path to generated HTML report
    """
    if not PROCESSOR_AVAILABLE:
        logger.error("BenchmarkProcessor not available - cannot generate report")
        return None

    # Handle default case for versions_override
    if versions_override is None:
        versions_override = {}
    effective_versions = list(versions or [])
    if baseline_version:
        baseline_version = str(baseline_version).strip() or None
    if baseline_version and baseline_version not in effective_versions:
        effective_versions.append(baseline_version)
        logger.info(
            "Including baseline composed version in version filters: %s",
            baseline_version,
        )

    # Validate runs compatibility
    model, data_profile = validate_runs_compatibility(runs_data)

    # Extract full data profile parameters from first run
    first_run_params = runs_data[0]["params"]
    data_profile_params = _extract_data_profile_params(first_run_params)

    # Filter runs by version if specified (using prefix match for MLflow runs)
    if effective_versions:
        logger.info(f"Filtering runs by base versions: {effective_versions}")
        filtered_runs = []
        for run_data in runs_data:
            composed_version = run_data["composed_version"]
            # Check if any base version matches as a prefix
            matches = any(
                composed_version.startswith(base_v) for base_v in effective_versions
            )
            if matches:
                filtered_runs.append(run_data)
                logger.info(
                    f"Including run {run_data['run_id']} with composed version {composed_version}"
                )
            else:
                logger.info(
                    f"Skipping run {run_data['run_id']} with composed version {composed_version}"
                )

        if not filtered_runs:
            raise ValueError(
                f"No runs found matching base versions: {effective_versions}"
            )

        runs_data = filtered_runs
        logger.info(f"Using {len(runs_data)} runs after version filtering")

    # Process each run's JSON individually to get CSV data, then combine
    logger.info(f"Processing {len(runs_data)} runs individually to extract CSV data")

    # Get S3 configuration from environment
    s3_bucket = os.environ.get("S3_BUCKET", "psap-dashboard-data")
    s3_key = os.environ.get(
        "S3_KEY", "main/llmd-dashboard/llmd-dashboard.csv"
    )  # Primary key (legacy env var, not used when downloading both)

    # Download and merge consolidated CSVs from S3
    logger.info(
        "Downloading consolidated CSVs from S3 (llmd-dashboard + rhaiis-dashboard)"
    )
    from .processor import BenchmarkProcessor
    import pandas as pd

    # Create a temporary processor just to download S3 CSV
    temp_processor = BenchmarkProcessor(
        json_path=runs_data[0]["artifact_path"],  # dummy, won't use it yet
        s3_bucket=s3_bucket,
        s3_key=s3_key,
        accelerator="dummy",
        model_name=model,
        version="dummy",
        tp_size=1,
        runtime_args="",
        replicas=1,  # dummy value
        data_profile=data_profile_params,
        repeat_section_legends=repeat_section_legends,
        include_total_throughput=include_total_throughput,
    )
    consolidated_df = temp_processor.download_s3_csv()
    logger.info(f"Downloaded consolidated CSV with {len(consolidated_df)} rows")

    # Mark CSV data with source column for filtering logic
    if not consolidated_df.empty:
        consolidated_df["_data_source"] = "csv"

    # Load and merge additional CSV files using processor method
    if additional_csv_files:
        temp_processor.consolidated_df = consolidated_df
        consolidated_df = temp_processor.load_additional_csvs(additional_csv_files)
        # Mark additional CSV data as well
        if not consolidated_df.empty and "_data_source" not in consolidated_df.columns:
            consolidated_df["_data_source"] = "csv"

    # Process each run to get its CSV data
    all_run_dataframes = []
    ttft_distribution_dfs = []

    for run_data in runs_data:
        run_id = run_data["run_id"]
        params = run_data["params"]
        artifact_path = run_data["artifact_path"]
        composed_version = run_data["composed_version"]

        accelerator = _resolve_accelerator(params, run_data.get("tags"))
        tp_size = int(params.get("tp", 1))

        # Extract replicas from MLflow params
        replicas = params.get("replicas", "N/A")
        # Convert "N/A" to 1 for consistency with default behavior
        try:
            replicas_int = int(replicas) if replicas != "N/A" else 1
        except (ValueError, TypeError):
            replicas_int = 1

        logger.info(
            f"Processing run {run_id} (composed_version={composed_version}, TP={tp_size}, replicas={replicas_int})"
        )

        # Create processor for this run using composed version
        processor = BenchmarkProcessor(
            json_path=artifact_path,
            s3_bucket=s3_bucket,
            s3_key=s3_key,
            accelerator=accelerator,
            model_name=model,
            version=composed_version,
            tp_size=tp_size,
            runtime_args="",
            replicas=replicas_int,
            data_profile=data_profile_params,
            repeat_section_legends=repeat_section_legends,
            include_total_throughput=include_total_throughput,
        )

        # Parse this run's JSON to DataFrame (replicas will be included via processor)
        run_df = processor.parse_guidellm_json()
        ttft_distribution_df = processor.parse_ttft_distribution_json()

        # Mark MLflow data with source column for filtering logic
        run_df["_data_source"] = "mlflow"
        if not ttft_distribution_df.empty:
            ttft_distribution_dfs.append(ttft_distribution_df)

        logger.info(f"Extracted {len(run_df)} rows from run {run_id}")

        all_run_dataframes.append(run_df)

    # Combine all run DataFrames using BenchmarkProcessor's merge logic
    logger.info(f"Combining {len(all_run_dataframes)} DataFrames")
    combined_runs_df = pd.concat(all_run_dataframes, ignore_index=True)
    logger.info(f"Combined runs DataFrame has {len(combined_runs_df)} rows")
    combined_ttft_distribution_df = (
        pd.concat(ttft_distribution_dfs, ignore_index=True)
        if ttft_distribution_dfs
        else pd.DataFrame()
    )

    # Use BenchmarkProcessor's merge_data logic to properly combine
    logger.info("Merging with consolidated CSV using processor's merge logic")
    temp_processor.consolidated_df = consolidated_df
    temp_processor.new_data_df = combined_runs_df
    final_df = temp_processor.merge_data()
    logger.info(f"Final merged DataFrame has {len(final_df)} rows")

    # Re-add _data_source column after merge (it gets dropped by merge_data fieldnames filter)
    # Identify which rows came from MLflow vs CSV by checking if version exists in our MLflow runs
    mlflow_versions = set(run_data["composed_version"] for run_data in runs_data)
    final_df["_data_source"] = final_df["version"].apply(
        lambda v: "mlflow" if v in mlflow_versions else "csv"
    )
    # Keep the immutable run identity separate from display labels. Version
    # overrides are presentation-only; --baseline must match composed versions.
    final_df["_composed_version"] = final_df["version"].fillna("unknown").astype(str)
    logger.info(
        f"Restored _data_source column: "
        f"{(final_df['_data_source'] == 'mlflow').sum()} MLflow rows, "
        f"{(final_df['_data_source'] == 'csv').sum()} CSV rows"
    )

    # Filter by versions if specified (different logic for CSV vs MLflow data)
    if effective_versions:
        logger.info(f"Filtering combined data by versions: {effective_versions}")
        initial_rows = len(final_df)

        # Apply different filtering logic based on data source
        def should_keep_row(row):
            data_source = row.get("_data_source", "csv")
            version = row["version"]

            if data_source == "csv":
                # CSV data: exact match only
                return version in effective_versions
            else:  # mlflow
                # MLflow data: prefix match (base version matches)
                return any(version.startswith(base_v) for base_v in effective_versions)

        mask = final_df.apply(should_keep_row, axis=1)
        final_df = final_df[mask]

        logger.info(
            f"After version filtering: {len(final_df)} rows (removed {initial_rows - len(final_df)} rows)"
        )
        logger.info("  CSV data filtered with exact match")
        logger.info("  MLflow data filtered with prefix match")
        if not combined_ttft_distribution_df.empty:
            distribution_mask = combined_ttft_distribution_df["version"].apply(
                lambda value: any(
                    str(value).startswith(base_v) for base_v in effective_versions
                )
            )
            combined_ttft_distribution_df = combined_ttft_distribution_df[
                distribution_mask
            ].copy()

    # Apply version overrides after filtering, before plotting
    if versions_override:
        logger.info(f"Applying {len(versions_override)} version override(s)")
        for old_ver, new_ver in versions_override.items():
            matching_rows = final_df["version"] == old_ver
            count = matching_rows.sum()
            if count > 0:
                final_df.loc[matching_rows, "version"] = new_ver
                logger.info(f"  Renamed {count} rows: {old_ver} → {new_ver}")
            else:
                logger.warning(f"  No rows found with version '{old_ver}' to rename")
            if not combined_ttft_distribution_df.empty:
                combined_ttft_distribution_df.loc[
                    combined_ttft_distribution_df["version"] == old_ver, "version"
                ] = new_ver

    comparison_metric_panels = []
    if metrics_yaml_path:
        comparison_metric_panels = build_comparison_metric_panels(
            metrics_yaml_path=Path(metrics_yaml_path),
            runs_data=runs_data,
            version_overrides=versions_override,
        )

    # Remove the temporary source column before generating report
    if "_data_source" in final_df.columns:
        final_df = final_df.drop(columns=["_data_source"])

    # Determine compare_versions from the data
    compare_versions = sorted(final_df["version"].unique().tolist())
    logger.info(f"Versions in final data: {compare_versions}")

    # Extract metadata from first run for filename
    first_run = runs_data[0]
    params = first_run["params"]

    # Auto-generate output filename
    model_short = model.split("/")[-1].replace(" ", "_").replace("-", "_").lower()
    version_str = "_".join(compare_versions).lower().replace(".", "").replace("-", "")
    html_filename = f"{model_short}_comparison_{version_str}_report.html"
    html_path = _resolve_report_output_path(
        html_filename,
        output_dir=output_dir,
        output_file=output_file,
    )

    # Generate report using the combined DataFrame
    final_processor = BenchmarkProcessor(
        json_path=first_run["artifact_path"],
        s3_bucket=s3_bucket,
        s3_key=s3_key,
        accelerator=_resolve_accelerator(params, first_run.get("tags")),
        model_name=model,
        version=(
            compare_versions[0]
            if compare_versions
            else params.get("version", "unknown")
        ),
        tp_size=int(params.get("tp", 1)),
        runtime_args="",
        compare_versions=compare_versions,
        output_html=html_path,
        data_profile=data_profile_params,
        notes=notes or [],
        repeat_section_legends=repeat_section_legends,
        include_total_throughput=include_total_throughput,
        baseline_version=baseline_version,
        comparison_metric_panels=comparison_metric_panels,
    )

    # Override with our merged and filtered data
    final_processor.combined_df = final_df
    final_processor.ttft_distribution_df = combined_ttft_distribution_df
    final_processor.config = final_processor.load_config()
    final_processor.generate_report()

    if Path(html_path).exists():
        logger.info(f"Comparison report generated: {html_path}")
        return html_path
    else:
        logger.error("Report generation failed - file not found")
        return None


def _parse_tag_mappings(tags: tuple[str, ...]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for tag in tags:
        if "=" not in tag:
            raise click.BadParameter(
                f"invalid tag format: {tag}. Expected format: key=value",
                param_hint="--tag",
            )
        key, value = tag.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _parse_version_overrides(values: tuple[str, ...]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for mapping in values:
        if "=" not in mapping:
            raise click.BadParameter(
                f"invalid version override format: {mapping}. Expected format: old_name=new_name",
                param_hint="--versions-override",
            )
        old_version, new_version = mapping.split("=", 1)
        old_version = old_version.strip()
        new_version = new_version.strip()
        overrides[old_version] = new_version
        logger.info(f"  Will rename: {old_version} → {new_version}")
    return overrides


def _authenticate_huggingface_if_needed() -> None:
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return
    try:
        from huggingface_hub import login as hf_login

        hf_login(
            token=hf_token,
            add_to_git_credential=False,
            skip_if_logged_in=True,
        )
        logger.info("Successfully authenticated with HuggingFace")
    except Exception as exc:
        logger.warning(
            "Python Hugging Face login failed (%s); trying CLI fallback",
            exc,
        )
        if shutil.which("hf"):
            hf_cmd = ["hf", "auth", "login", "--token", hf_token]
        elif shutil.which("huggingface-cli"):
            hf_cmd = ["huggingface-cli", "login", "--token", hf_token]
        else:
            raise RuntimeError(
                "HF_TOKEN is set but no Hugging Face login method is available"
            ) from exc

        subprocess.run(
            hf_cmd,
            check=True,
            capture_output=True,
            timeout=30,
        )
        logger.info("Successfully authenticated with HuggingFace")


def _run_plot_only_mode(
    *,
    mlflow_run_ids: str,
    versions: str | None,
    mlflow_tracking_uri: str | None,
    additional_csv_files: tuple[str, ...],
    versions_override_values: tuple[str, ...],
    baseline_version: str | None,
    metrics_yaml_path: str | None,
) -> int:
    logger.info("Plot-only mode enabled")

    if additional_csv_files:
        logger.info(f"Will include {len(additional_csv_files)} additional CSV file(s)")
        for csv_file in additional_csv_files:
            if not Path(csv_file).exists():
                raise click.BadParameter(
                    f"additional CSV file not found: {csv_file}",
                    param_hint="--additional-csv",
                )

    run_ids = [rid.strip() for rid in mlflow_run_ids.split(",") if rid.strip()]
    versions_list = (
        [v.strip() for v in versions.split(",") if v.strip()] if versions else None
    )

    versions_override = {}
    if versions_override_values:
        logger.info(f"Parsing {len(versions_override_values)} version override(s)")
        versions_override = _parse_version_overrides(versions_override_values)

    logger.info(f"Fetching {len(run_ids)} MLflow runs...")

    try:
        runs_data = fetch_mlflow_runs(run_ids, mlflow_tracking_uri)

        if not runs_data:
            logger.error("No runs fetched successfully")
            return 1

        html_report = generate_plot_only_report(
            runs_data=runs_data,
            versions=versions_list,
            mlflow_tracking_uri=mlflow_tracking_uri,
            additional_csv_files=list(additional_csv_files) or None,
            versions_override=versions_override,
            baseline_version=baseline_version,
            metrics_yaml_path=metrics_yaml_path,
        )

        if html_report:
            logger.info("\nPlot generation completed successfully.")
            logger.info(f"  Report saved to: {html_report}")
            return 0
        logger.error("Plot generation failed")
        return 1
    except Exception as exc:
        logger.error(f"Plot generation failed: {exc}", exc_info=True)
        return 1


def _run_benchmark_mode(
    *,
    target: str,
    model: str,
    backend_type: str,
    rate_type: str | None,
    data_samples: int | None,
    warmup: str | None,
    rate: str | None,
    data: str | None,
    max_seconds: str | None,
    max_requests: str | None,
    processor: str | None,
    accelerator: str | None,
    profile: str | None,
    version: str | None,
    tp: int,
    runtime_args: str,
    experiment_name: str,
    mlflow_tracking_uri: str | None,
    tags: tuple[str, ...],
    replicas: str,
    prefill_replicas: str,
    decode_replicas: str,
) -> int:
    parsed_tags = _parse_tag_mappings(tags)
    profile_kind = profile or rate_type or "concurrent"
    profile_args: dict[str, Any] = {"kind": profile_kind}
    override_args: dict[str, Any] = {}
    if rate:
        load_values = [item.strip() for item in rate.split(",") if item.strip()]
        load_field = "rate" if profile_kind == "poisson" else "streams"
        if profile_kind == "throughput":
            load_field = "max_concurrency"
        override_args[f"profile.{load_field}"] = load_values

    benchmark_args: dict[str, Any] = {
        "backend": {"kind": backend_type},
        "profile": profile_args,
        "override": override_args or None,
        "data": {"kind": "synthetic_text", **_parse_guidellm_mapping(data, "data")},
    }
    if data_samples is not None:
        benchmark_args["data_loader"] = {"kind": "pytorch", "samples": data_samples}
    constraints: list[dict[str, Any]] = []
    if max_seconds is not None:
        constraints.append({"kind": "max_duration", "seconds": max_seconds})
    if max_requests is not None:
        constraints.append({"kind": "max_requests", "count": max_requests})
    if constraints:
        benchmark_args["constraint"] = constraints
    if warmup is not None:
        benchmark_args["warmup"] = warmup
    if processor:
        benchmark_args["tokenizer"] = {
            "kind": "huggingface_auto",
            "model": processor,
        }
    benchmark_args = {
        key: value for key, value in benchmark_args.items() if value is not None
    }
    logger.info(
        f"Starting benchmark sweep for rates: {rate if rate is not None else 'not set'}"
    )
    _authenticate_huggingface_if_needed()

    mlflow_enabled = os.environ.get("MLFLOW_ENABLED", "false").lower() == "true"

    if not mlflow_enabled:
        logger.info("MLflow tracking disabled - running benchmark without MLflow")
        try:
            json_path = run_benchmark_without_mlflow(
                target=target,
                model=model,
                benchmark_args=benchmark_args,
                output_dir="/benchmark-results",
                accelerator=accelerator,
                version=version,
                tp_size=tp,
                runtime_args=runtime_args,
            )
            logger.info("\nBenchmark completed successfully.")
            logger.info(f"  Results saved to: {json_path}")
            return 0
        except Exception as exc:
            logger.error(f"Benchmark failed: {exc}")
            return 1

    logger.info("MLflow tracking enabled")
    try:
        run_id = run_benchmark_with_mlflow(
            target=target,
            model=model,
            benchmark_args=benchmark_args,
            accelerator=accelerator,
            experiment_name=experiment_name,
            mlflow_tracking_uri=mlflow_tracking_uri,
            tags=parsed_tags,
            version=version,
            tp_size=tp,
            runtime_args=runtime_args,
            replicas=replicas,
            prefill_replicas=prefill_replicas,
            decode_replicas=decode_replicas,
        )
        logger.info("\nBenchmark sweep completed successfully.")
        logger.info(f"  MLflow Run ID: {run_id}")
        return 0
    except Exception as exc:
        logger.error(f"Benchmark sweep failed: {exc}")
        return 1


@click.command(
    help=(
        "Run a GuideLLM benchmark with optional MLflow logging, or generate "
        "comparison reports from existing MLflow runs."
    )
)
@click.option("--target", help="Target URL. Required for benchmark mode.")
@click.option("--model", help="Model name. Required for benchmark mode.")
@click.option(
    "--backend-type",
    default="openai_http",
    show_default=True,
    help="Backend type.",
)
@click.option(
    "--rate-type",
    help="Rate type.",
)
@click.option(
    "--rate",
    help="Rate value(s), comma-separated. Required for benchmark mode.",
)
@click.option(
    "--data-samples",
    type=int,
    help="Limit the number of data samples used by GuideLLM.",
)
@click.option(
    "--warmup",
    type=str,
    help=('GuideLLM warmup value, for example 10 or {"value":30,"mode":"duration"}.'),
)
@click.option(
    "--data",
    help=(
        "Data config, for example prompt_tokens=1000,output_tokens=1000. "
        "Expressions like prefix_count=2*concurrency automatically enable one run per concurrency."
    ),
)
@click.option(
    "--max-seconds",
    help=(
        "Max duration in seconds. Expressions like 2*concurrency automatically "
        "enable one run per concurrency."
    ),
)
@click.option(
    "--max-requests",
    help=(
        "Max requests. Expressions like 10*concurrency automatically enable "
        "one run per concurrency."
    ),
)
@click.option("--processor", help="Processor or tokenizer name.")
@click.option("--accelerator", help="Accelerator type, for example H200 or A100.")
@click.option("--version", help="Version identifier for visualization reports.")
@click.option(
    "--tp",
    type=int,
    default=1,
    show_default=True,
    help="Tensor parallelism size for visualization reports.",
)
@click.option(
    "--runtime-args",
    default="",
    show_default=True,
    help="Runtime arguments for visualization reports.",
)
@click.option(
    "--replicas",
    default="N/A",
    show_default=True,
    help="Replica count for standard deployment mode.",
)
@click.option(
    "--prefill-replicas",
    default="N/A",
    show_default=True,
    help="Prefill worker replica count for P/D disaggregation.",
)
@click.option(
    "--decode-replicas",
    default="N/A",
    show_default=True,
    help="Decode worker replica count for P/D disaggregation.",
)
@click.option(
    "--experiment-name",
    default="guidellm-benchmarks",
    show_default=True,
    help="MLflow experiment name.",
)
@click.option(
    "--mlflow-tracking-uri",
    default=lambda: os.environ.get("MLFLOW_TRACKING_URI"),
    show_default="env MLFLOW_TRACKING_URI",
    help="MLflow tracking URI.",
)
@click.option(
    "--tag",
    "tags",
    multiple=True,
    metavar="KEY=VALUE",
    help="Additional MLflow tag. Repeat to set multiple tags.",
)
@click.option(
    "--plot-only",
    is_flag=True,
    help="Generate plots from existing MLflow runs without running benchmarks.",
)
@click.option(
    "--mlflow-run-ids",
    help="Comma-separated list of MLflow run IDs to plot. Required with --plot-only.",
)
@click.option(
    "--versions",
    help="Comma-separated versions to compare. Filters runs and sets compare_versions.",
)
@click.option(
    "--baseline",
    "baseline_version",
    help="Composed version name to use as the baseline for inline table deltas.",
)
@click.option(
    "--versions-override",
    "versions_override_values",
    multiple=True,
    metavar="OLD=NEW",
    help=(
        "Version rename mapping applied after filtering but before plotting. "
        "Repeat to set multiple mappings. Only for --plot-only mode."
    ),
)
@click.option(
    "--additional-csv",
    "additional_csv_files",
    multiple=True,
    metavar="PATH",
    help=(
        "Additional CSV file to include in comparison plots. "
        "Repeat to set multiple files. Only for --plot-only mode."
    ),
)
@click.option(
    "--metrics-yaml",
    "metrics_yaml_path",
    type=click.Path(dir_okay=False, path_type=str),
    help=(
        "Report-only YAML selecting archived Prometheus metrics to append to "
        "comparison reports. Only for --plot-only mode."
    ),
)
def cli(
    target: str | None,
    model: str | None,
    backend_type: str,
    rate_type: str | None,
    data_samples: int | None,
    warmup: str | None,
    rate: str | None,
    data: str | None,
    max_seconds: str | None,
    max_requests: str | None,
    processor: str | None,
    accelerator: str | None,
    profile: str | None,
    version: str | None,
    tp: int,
    runtime_args: str,
    replicas: str,
    prefill_replicas: str,
    decode_replicas: str,
    experiment_name: str,
    mlflow_tracking_uri: str | None,
    tags: tuple[str, ...],
    plot_only: bool,
    mlflow_run_ids: str | None,
    versions: str | None,
    baseline_version: str | None,
    versions_override_values: tuple[str, ...],
    additional_csv_files: tuple[str, ...],
    metrics_yaml_path: str | None,
) -> int:
    if plot_only:
        if not mlflow_run_ids:
            raise click.UsageError(
                "--mlflow-run-ids is required when using --plot-only"
            )
        return _run_plot_only_mode(
            mlflow_run_ids=mlflow_run_ids,
            versions=versions,
            mlflow_tracking_uri=mlflow_tracking_uri,
            additional_csv_files=additional_csv_files,
            versions_override_values=versions_override_values,
            baseline_version=baseline_version,
            metrics_yaml_path=metrics_yaml_path,
        )

    missing = []
    if not target:
        missing.append("--target")
    if not model:
        missing.append("--model")
    if not rate:
        missing.append("--rate")
    if missing:
        raise click.UsageError(
            f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} required for benchmark mode"
        )

    return _run_benchmark_mode(
        target=target,
        model=model,
        backend_type=backend_type,
        rate_type=rate_type,
        data_samples=data_samples,
        warmup=warmup,
        rate=rate,
        data=data,
        max_seconds=max_seconds,
        max_requests=max_requests,
        processor=processor,
        accelerator=accelerator,
        profile=profile,
        version=version,
        tp=tp,
        runtime_args=runtime_args,
        experiment_name=experiment_name,
        mlflow_tracking_uri=mlflow_tracking_uri,
        tags=tags,
        replicas=replicas,
        prefill_replicas=prefill_replicas,
        decode_replicas=decode_replicas,
    )


def main(argv: list[str] | None = None) -> int:
    try:
        result = cli.main(
            args=argv, prog_name="guidellm-runtime", standalone_mode=False
        )
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    return int(result or 0)


if __name__ == "__main__":
    sys.exit(main())
