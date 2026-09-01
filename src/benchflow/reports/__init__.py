"""BenchFlow post-run report generation.

TODO: Move comparison-report orchestration and rendering into this package once
benchmark result parsing is separated cleanly from presentation. Benchmark
backends should produce normalized report inputs; this package should own both
post-run and comparison report rendering.
"""

from .post_run import PostRunReports, generate_post_run_reports
from .tracing import generate_trace_distribution_report

__all__ = [
    "PostRunReports",
    "generate_post_run_reports",
    "generate_trace_distribution_report",
]
