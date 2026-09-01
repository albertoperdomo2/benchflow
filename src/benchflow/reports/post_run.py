from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..benchmark import generate_run_report
from ..ui import detail, warning
from .tracing import generate_trace_distribution_report


@dataclass(frozen=True, slots=True)
class PostRunReports:
    benchmark: Path | None = None
    tracing: Path | None = None


def generate_post_run_reports(*, artifacts_dir: Path) -> PostRunReports:
    benchmark_report: Path | None = None
    tracing_report: Path | None = None

    try:
        benchmark_report = generate_run_report(artifacts_dir=artifacts_dir)
    except Exception as exc:  # noqa: BLE001
        warning(f"Skipping benchmark post-run report generation: {exc}")
    else:
        detail(f"Generated benchmark post-run report at {benchmark_report}")

    try:
        tracing_report = generate_trace_distribution_report(artifacts_dir=artifacts_dir)
    except Exception as exc:  # noqa: BLE001
        warning(f"Skipping tracing post-run report generation: {exc}")
    else:
        if tracing_report is not None:
            detail(f"Generated tracing post-run report at {tracing_report}")

    return PostRunReports(
        benchmark=benchmark_report,
        tracing=tracing_report,
    )
