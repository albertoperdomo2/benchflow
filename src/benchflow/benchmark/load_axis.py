from __future__ import annotations

from typing import Any

BENCHMARK_INDEX_KEY = "_benchflow_benchmark_index"


def _get_nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if not isinstance(data, dict):
            return default
        data = data.get(key, default)
    return data


def _axis_value(values: Any, benchmark_index: int) -> Any:
    if not isinstance(values, list):
        return values
    if len(values) == 1:
        return values[0]
    if benchmark_index < len(values):
        return values[benchmark_index]
    return values[0] if values else None


def _benchmark_index(benchmark_run: dict[str, Any], fallback: int) -> int:
    raw_index = benchmark_run.get(BENCHMARK_INDEX_KEY, fallback)
    try:
        return int(raw_index)
    except (TypeError, ValueError):
        return fallback


def extract_intended_load(benchmark_run: dict[str, Any], benchmark_index: int) -> Any:
    """Extract the GuideLLM sweep axis across old and new payload schemas."""

    axis_index = _benchmark_index(benchmark_run, benchmark_index)

    scheduler_streams = _get_nested(benchmark_run, "scheduler", "strategy", "streams")
    if scheduler_streams is not None:
        return _axis_value(scheduler_streams, axis_index)

    strategy_streams = _get_nested(benchmark_run, "config", "strategy", "streams")
    if strategy_streams is not None:
        return _axis_value(strategy_streams, axis_index)

    strategy_rate = _get_nested(benchmark_run, "config", "strategy", "rate")
    if strategy_rate is not None:
        return _axis_value(strategy_rate, axis_index)

    profile_args = _get_nested(benchmark_run, "config", "profile") or _get_nested(
        benchmark_run, "args", "profile", default={}
    )
    if not isinstance(profile_args, dict):
        profile_args = {}

    profile_streams = profile_args.get("streams", [])
    if profile_streams:
        return _axis_value(profile_streams, axis_index)

    profile_rate = profile_args.get("rate", [])
    if profile_rate:
        return _axis_value(profile_rate, axis_index)

    args_rate = _get_nested(benchmark_run, "args", "rate")
    if args_rate:
        return _axis_value(args_rate, axis_index)

    max_concurrency = _get_nested(
        benchmark_run, "config", "strategy", "max_concurrency"
    )
    if max_concurrency is not None:
        return _axis_value(max_concurrency, axis_index)

    profile_max_concurrency = profile_args.get("max_concurrency")
    if profile_max_concurrency is not None:
        return _axis_value(profile_max_concurrency, axis_index)

    return axis_index + 1


def coerce_load_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def benchmark_load_sort_key(
    benchmark_run: dict[str, Any], benchmark_index: int
) -> tuple[int, float, str, int]:
    load_value = extract_intended_load(benchmark_run, benchmark_index)
    numeric_load = coerce_load_number(load_value)
    if numeric_load is not None:
        return (0, numeric_load, "", benchmark_index)
    return (1, float("inf"), str(load_value), benchmark_index)


def is_rate_based_benchmark(benchmark_run: dict[str, Any]) -> bool:
    """Return true when the GuideLLM sweep axis is request rate, not concurrency."""

    explicit_axis = benchmark_run.get("load axis") or benchmark_run.get("load_axis")
    if explicit_axis is not None:
        normalized = str(explicit_axis).strip().lower()
        return normalized in {"rps", "rate", "request rate", "requests_per_second"}

    strategy = _get_nested(benchmark_run, "config", "strategy", default={})
    profile = _get_nested(benchmark_run, "config", "profile", default={})
    args_profile = _get_nested(benchmark_run, "args", "profile")
    if args_profile is None:
        args_profile = benchmark_run.get("profile")

    if isinstance(strategy, dict) and strategy.get("rate") is not None:
        return True
    if isinstance(profile, dict) and profile.get("rate"):
        return True

    rate_strategy_types = {
        "poisson",
        "fixed",
        "fixed_rate",
        "fixed-rate",
        "fixed_schedule",
        "fixed-schedule",
    }
    observed_types: set[str] = set()
    for raw_value in (
        strategy.get("type_") if isinstance(strategy, dict) else None,
        profile.get("type_") if isinstance(profile, dict) else None,
        profile.get("strategy_type") if isinstance(profile, dict) else None,
        args_profile,
    ):
        if raw_value is not None:
            observed_types.add(str(raw_value).strip().lower())

    if observed_types & rate_strategy_types:
        return True

    strategy_types = profile.get("strategy_types") if isinstance(profile, dict) else []
    if isinstance(strategy_types, list) and strategy_types:
        normalized = {str(item).strip().lower() for item in strategy_types}
        if normalized and normalized <= rate_strategy_types:
            return True

    return False


def benchmark_axis_label(benchmark_run: dict[str, Any], benchmark_index: int) -> str:
    return "RPS" if is_rate_based_benchmark(benchmark_run) else "Concurrency"


def format_load_value(value: Any) -> str:
    numeric_value = coerce_load_number(value)
    if numeric_value is None:
        return str(value)
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return f"{numeric_value:g}"
