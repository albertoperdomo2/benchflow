"""BenchFlow vLLM worker profiler for RHOAI runtime pods.

This module is loaded automatically by Python via PYTHONPATH. It installs an
import hook for vLLM's GPU worker implementation and wraps
Worker.execute_model() with a torch profiler window defined by
VLLM_PROFILER_RANGES.

The narrow-path contract is fixed:
- CPU and CUDA activities are always enabled
- Chrome traces are always exported
- output files are written under VLLM_PROFILER_OUTPUT_DIR
"""

from __future__ import annotations

import functools
import importlib
import importlib.abc
import importlib.util
import os
import socket
import sys
import time
from typing import Any

os.environ.setdefault("VLLM_RPC_TIMEOUT", "1800000")

TARGET_MODULE = "vllm.v1.worker.gpu_worker"
TARGET_CLASS = "Worker"
TARGET_METHOD = "execute_model"
DEFAULT_CALL_RANGES = "100-150"
DEFAULT_CAPTURE_CALLS = "100"
DEFAULT_OUTPUT_DIR = "/tmp/benchflow-profiler"


def _parse_call_ranges(value: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for item in str(value or "").split(","):
        candidate = item.strip()
        if not candidate:
            continue
        if "-" not in candidate:
            raise ValueError(f"invalid profiler range: {candidate!r}")
        start_raw, end_raw = candidate.split("-", 1)
        start = int(start_raw.strip())
        end = int(end_raw.strip())
        if end < start:
            raise ValueError(f"invalid profiler range: {candidate!r}")
        ranges.append((start, end))
    if not ranges:
        raise ValueError("no profiler call ranges configured")
    return ranges


def _parse_capture_calls(value: str) -> int:
    raw = str(value or "").strip()
    if not raw.isdigit():
        raise ValueError("idle-triggered profiling requires a positive call count")
    capture_calls = int(raw)
    if capture_calls <= 0:
        raise ValueError("idle-triggered profiling requires a positive call count")
    return capture_calls


def _parse_idle_seconds(value: str | None) -> float | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    idle_seconds = float(raw)
    if idle_seconds <= 0:
        raise ValueError("VLLM_PROFILER_IDLE_SECONDS must be greater than 0")
    return idle_seconds


def _profiler_config() -> tuple[list[tuple[int, int]], str, float | None, int | None]:
    idle_seconds = _parse_idle_seconds(os.environ.get("VLLM_PROFILER_IDLE_SECONDS"))
    capture_calls = (
        _parse_capture_calls(
            os.environ.get("VLLM_PROFILER_RANGES", DEFAULT_CAPTURE_CALLS)
        )
        if idle_seconds is not None
        else None
    )
    ranges = (
        []
        if idle_seconds is not None
        else _parse_call_ranges(
            os.environ.get("VLLM_PROFILER_RANGES", DEFAULT_CALL_RANGES)
        )
    )
    output_dir = os.environ.get("VLLM_PROFILER_OUTPUT_DIR", DEFAULT_OUTPUT_DIR).strip()
    if not output_dir:
        output_dir = DEFAULT_OUTPUT_DIR
    return ranges, output_dir, idle_seconds, capture_calls


CALL_RANGES, OUTPUT_DIR, IDLE_SECONDS, CAPTURE_CALLS = _profiler_config()


def _pod_identity() -> str:
    return (
        os.environ.get("POD_NAME")
        or os.environ.get("HOSTNAME")
        or socket.gethostname()
        or "unknown-pod"
    )


def _trace_path(output_dir: str, start: int, end: int) -> str:
    return os.path.join(
        output_dir,
        f"trace_{_pod_identity()}_pid{os.getpid()}_range{start}-{end}.json",
    )


def _summary_path(output_dir: str, start: int, end: int) -> str:
    return os.path.join(
        output_dir,
        f"summary_{_pod_identity()}_pid{os.getpid()}_range{start}-{end}.txt",
    )


class _PostImportLoader(importlib.abc.Loader):
    def __init__(self, loader: importlib.abc.Loader):
        self.loader = loader

    def create_module(self, spec):
        if hasattr(self.loader, "create_module"):
            return self.loader.create_module(spec)
        return None

    def exec_module(self, module):
        self.loader.exec_module(module)
        _safe_wrap_worker(module)


class _PostImportFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != TARGET_MODULE:
            return None

        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)

        if spec is not None and spec.loader is not None:
            spec.loader = _PostImportLoader(spec.loader)
            return spec
        return None


def _new_profiler():
    import torch
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    return profile(
        activities=activities,
        record_shapes=True,
        with_stack=True,
        profile_memory=False,
        with_modules=False,
    )


def _write_summary_table(profiler: Any, path: str) -> None:
    table = profiler.key_averages().table(sort_by="cuda_time_total", row_limit=50)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(table)
        handle.write("\n")


def _wrap_execute_model(original_func):
    if IDLE_SECONDS is not None:
        return _wrap_execute_model_idle_triggered(original_func)
    return _wrap_execute_model_ranges(original_func)


def _wrap_execute_model_ranges(original_func):
    try:
        profiler = _new_profiler()
    except Exception as exc:  # pragma: no cover - runtime-only fallback
        print(
            f"[benchflow-profiler] unable to initialize torch profiler: {exc}",
            file=sys.stderr,
        )
        return original_func

    call_count = 0
    range_index = 0
    profiling_active = False

    @functools.wraps(original_func)
    def _wrapped(*args, **kwargs):
        nonlocal profiler, call_count, range_index, profiling_active

        call_count += 1
        if not profiling_active and range_index < len(CALL_RANGES):
            start, end = CALL_RANGES[range_index]
            if call_count == start:
                print(
                    f"[benchflow-profiler] starting range {start}-{end} "
                    f"on call {call_count}",
                    file=sys.stderr,
                )
                profiler.start()
                profiling_active = True

        result = original_func(*args, **kwargs)

        if profiling_active and range_index < len(CALL_RANGES):
            start, end = CALL_RANGES[range_index]
            if call_count == end:
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                try:
                    profiler.stop()
                    _write_summary_table(
                        profiler, _summary_path(OUTPUT_DIR, start, end)
                    )
                    profiler.export_chrome_trace(_trace_path(OUTPUT_DIR, start, end))
                    print(
                        f"[benchflow-profiler] exported range {start}-{end} to "
                        f"{OUTPUT_DIR}",
                        file=sys.stderr,
                    )
                except Exception as exc:  # pragma: no cover - runtime-only fallback
                    print(
                        f"[benchflow-profiler] failed to export range "
                        f"{start}-{end}: {exc}",
                        file=sys.stderr,
                    )
                finally:
                    profiling_active = False
                    range_index += 1
                    if range_index < len(CALL_RANGES):
                        profiler = _new_profiler()

        return result

    return _wrapped


def _wrap_execute_model_idle_triggered(original_func):
    call_count = 0
    last_execute_model_time: float | None = None
    profiling_active = False
    profiled_current_burst = False
    profiler = None
    active_start = 0
    active_end = 0

    @functools.wraps(original_func)
    def _wrapped(*args, **kwargs):
        nonlocal active_end, active_start, call_count, last_execute_model_time
        nonlocal profiled_current_burst, profiler, profiling_active

        now = time.monotonic()
        idle_gap = (
            None if last_execute_model_time is None else now - last_execute_model_time
        )
        if idle_gap is None or idle_gap >= IDLE_SECONDS:
            profiled_current_burst = False

        call_count += 1
        if not profiling_active and not profiled_current_burst:
            assert CAPTURE_CALLS is not None
            active_start = call_count
            active_end = call_count + CAPTURE_CALLS - 1
            try:
                profiler = _new_profiler()
                profiler.start()
                profiling_active = True
                print(
                    f"[benchflow-profiler] starting idle-triggered range "
                    f"{active_start}-{active_end} on call {call_count}",
                    file=sys.stderr,
                )
            except Exception as exc:  # pragma: no cover - runtime-only fallback
                profiler = None
                profiled_current_burst = True
                print(
                    f"[benchflow-profiler] unable to initialize torch profiler: {exc}",
                    file=sys.stderr,
                )

        result = original_func(*args, **kwargs)
        last_execute_model_time = time.monotonic()

        if profiling_active and call_count >= active_end:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            try:
                assert profiler is not None
                profiler.stop()
                _write_summary_table(
                    profiler, _summary_path(OUTPUT_DIR, active_start, active_end)
                )
                profiler.export_chrome_trace(
                    _trace_path(OUTPUT_DIR, active_start, active_end)
                )
                print(
                    f"[benchflow-profiler] exported idle-triggered range "
                    f"{active_start}-{active_end} to {OUTPUT_DIR}",
                    file=sys.stderr,
                )
            except Exception as exc:  # pragma: no cover - runtime-only fallback
                print(
                    f"[benchflow-profiler] failed to export idle-triggered range "
                    f"{active_start}-{active_end}: {exc}",
                    file=sys.stderr,
                )
            finally:
                profiler = None
                profiling_active = False
                profiled_current_burst = True

        return result

    return _wrapped


def _safe_wrap_worker(module=None) -> None:
    try:
        resolved_module = module or sys.modules.get(TARGET_MODULE)
        if resolved_module is None:
            return
        worker_class = getattr(resolved_module, TARGET_CLASS, None)
        if worker_class is None:
            print(
                f"[benchflow-profiler] target class {TARGET_CLASS!r} not found",
                file=sys.stderr,
            )
            return
        original_method = getattr(worker_class, TARGET_METHOD, None)
        if original_method is None:
            print(
                f"[benchflow-profiler] target method {TARGET_METHOD!r} not found",
                file=sys.stderr,
            )
            return
        setattr(worker_class, TARGET_METHOD, _wrap_execute_model(original_method))
    except Exception as exc:  # pragma: no cover - runtime-only fallback
        print(
            f"[benchflow-profiler] failed to install profiler: {exc}", file=sys.stderr
        )


sys.meta_path.insert(0, _PostImportFinder())
MODE_DETAIL = (
    f"idle-triggered capture_calls={CAPTURE_CALLS} idle_seconds={IDLE_SECONDS}"
    if IDLE_SECONDS is not None
    else f"call ranges {CALL_RANGES}"
)
print(
    f"[benchflow-profiler] installed for {TARGET_MODULE}.{TARGET_CLASS}.{TARGET_METHOD} "
    f"with {MODE_DETAIL}",
    file=sys.stderr,
)
