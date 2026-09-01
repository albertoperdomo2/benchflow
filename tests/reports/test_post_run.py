from __future__ import annotations

from pathlib import Path

from benchflow.reports import post_run


def test_generates_tracing_report_when_benchmark_report_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    generated: list[Path] = []

    def fail_benchmark_report(**_kwargs):
        raise RuntimeError("missing benchmark output")

    def generate_tracing_report(*, artifacts_dir: Path):
        generated.append(artifacts_dir)
        return artifacts_dir / "reports" / "trace_distribution_report.html"

    monkeypatch.setattr(post_run, "generate_run_report", fail_benchmark_report)
    monkeypatch.setattr(
        post_run,
        "generate_trace_distribution_report",
        generate_tracing_report,
    )

    reports = post_run.generate_post_run_reports(artifacts_dir=tmp_path)

    assert reports.benchmark is None
    assert reports.tracing == (tmp_path / "reports" / "trace_distribution_report.html")
    assert generated == [tmp_path]


def test_generates_benchmark_report_when_tracing_report_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    benchmark_path = tmp_path / "reports" / "full_run_artifacts_report.html"

    monkeypatch.setattr(
        post_run,
        "generate_run_report",
        lambda **_kwargs: benchmark_path,
    )
    monkeypatch.setattr(
        post_run,
        "generate_trace_distribution_report",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("missing traces")),
    )

    reports = post_run.generate_post_run_reports(artifacts_dir=tmp_path)

    assert reports.benchmark == benchmark_path
    assert reports.tracing is None
