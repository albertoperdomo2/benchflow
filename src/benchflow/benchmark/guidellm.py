from __future__ import annotations

import os
from importlib import metadata
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ..cluster import CommandError
from ..models import ResolvedRunPlan
from ..ui import detail, step, success
from .common import (
    BenchmarkRunFailed,
    benchmark_version_from_plan,
    resolved_accelerator,
)
from . import runtime as runtime_module


def _load_guidellm_module():
    try:
        return runtime_module
    except Exception as exc:  # noqa: BLE001
        versions: list[str] = []
        for package_name in ("guidellm", "transformers", "huggingface_hub"):
            try:
                versions.append(f"{package_name}=={metadata.version(package_name)}")
            except metadata.PackageNotFoundError:
                continue
        version_text = f" ({', '.join(versions)})" if versions else ""
        raise CommandError(
            f"failed to load GuideLLM benchmark module: {exc}{version_text}"
        ) from exc


def _iso8601_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _runtime_args(plan: ResolvedRunPlan) -> str:
    return " ".join(plan.deployment.runtime.vllm_args)


def _configure_benchmark_runtime() -> dict[str, str]:
    runtime_root = Path("/tmp/benchflow-guidellm")
    home_dir = runtime_root / "home"
    hf_home = runtime_root / "huggingface"
    xdg_cache_home = runtime_root / "xdg-cache"
    docker_config = runtime_root / "docker"

    for path in (home_dir, hf_home, xdg_cache_home, docker_config):
        path.mkdir(parents=True, exist_ok=True)

    return {
        "HOME": str(home_dir),
        "DOCKER_CONFIG": str(docker_config),
        "HF_HOME": str(hf_home),
        "XDG_CACHE_HOME": str(xdg_cache_home),
        "HF_HUB_CACHE": str(hf_home / "hub"),
        "HF_XET_CACHE": str(hf_home / "xet"),
        "TRANSFORMERS_CACHE": str(hf_home / "transformers"),
    }


@contextmanager
def _patched_environment(extra_env: dict[str, str]):
    original = {key: os.environ.get(key) for key in extra_env}
    os.environ.update(extra_env)
    try:
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_benchmark(
    *,
    plan: ResolvedRunPlan,
    target: str | None = None,
    output_dir: Path | None = None,
    mlflow_tracking_uri: str | None = None,
    enable_mlflow: bool = True,
    extra_tags: dict[str, str] | None = None,
) -> tuple[str, str, str]:
    module = _load_guidellm_module()
    benchmark_target = target or plan.deployment.target.base_url
    tags = dict(plan.mlflow.tags)
    if extra_tags:
        tags.update(extra_tags)

    start_time = _iso8601_now()
    run_id = ""
    benchmark_env = _configure_benchmark_runtime()
    benchmark_env.update(plan.benchmark.env)
    accelerator = resolved_accelerator(plan)
    if output_dir is not None:
        benchmark_env["GUIDELLM_OUTPUT_DIR"] = str(output_dir)
    step(f"Preparing benchmark run for {plan.model.name}")
    detail(f"Target: {benchmark_target}")
    detail(
        f"Rates: {','.join(str(rate) for rate in plan.benchmark.rates)}, "
        f"duration: {plan.benchmark.max_seconds}s, max requests: "
        f"{plan.benchmark.max_requests if plan.benchmark.max_requests is not None else 'unbounded'}"
    )
    detail(f"Benchmark output mode: {'MLflow' if enable_mlflow else 'local artifacts'}")
    detail(f"Runtime HOME: {benchmark_env['HOME']}")
    detail(f"Hugging Face cache: {benchmark_env['HF_HUB_CACHE']}")
    if output_dir is not None:
        detail(f"Output directory: {output_dir}")

    with _patched_environment(benchmark_env):
        try:
            if enable_mlflow:
                step("Executing GuideLLM benchmark with MLflow tracking")
                run_id = module.run_benchmark_with_mlflow(
                    target=benchmark_target,
                    model=plan.model.name,
                    rate=",".join(str(rate) for rate in plan.benchmark.rates),
                    backend_type=plan.benchmark.backend_type,
                    request_type=plan.benchmark.request_type or None,
                    rate_type=plan.benchmark.rate_type,
                    data=plan.benchmark.data,
                    max_seconds=plan.benchmark.max_seconds,
                    max_requests=plan.benchmark.max_requests,
                    accelerator=accelerator,
                    experiment_name=plan.mlflow.experiment,
                    mlflow_tracking_uri=mlflow_tracking_uri
                    or os.environ.get("MLFLOW_TRACKING_URI"),
                    tags=tags,
                    version=benchmark_version_from_plan(plan),
                    tp_size=plan.deployment.runtime.tensor_parallelism,
                    runtime_args=_runtime_args(plan),
                    replicas=str(plan.deployment.runtime.replicas),
                    output_dir=str(output_dir) if output_dir is not None else None,
                )
            else:
                if output_dir is None:
                    raise CommandError(
                        "--output-dir is required when MLflow is disabled"
                    )
                step("Executing GuideLLM benchmark without MLflow tracking")
                module.run_benchmark_without_mlflow(
                    target=benchmark_target,
                    model=plan.model.name,
                    rate=",".join(str(rate) for rate in plan.benchmark.rates),
                    backend_type=plan.benchmark.backend_type,
                    request_type=plan.benchmark.request_type or None,
                    rate_type=plan.benchmark.rate_type,
                    data=plan.benchmark.data,
                    max_seconds=plan.benchmark.max_seconds,
                    max_requests=plan.benchmark.max_requests,
                    output_dir=str(output_dir),
                    accelerator=accelerator,
                    version=benchmark_version_from_plan(plan),
                    tp_size=plan.deployment.runtime.tensor_parallelism,
                    runtime_args=_runtime_args(plan),
                    replicas=plan.deployment.runtime.replicas,
                )
        except Exception as exc:  # noqa: BLE001
            end_time = _iso8601_now()
            failed_run_id = str(getattr(exc, "run_id", "") or "")
            if failed_run_id:
                warning(
                    "GuideLLM failed after creating MLflow run "
                    f"{failed_run_id}; preserving that run for later uploads"
                )
            raise BenchmarkRunFailed(
                str(exc),
                run_id=failed_run_id,
                start_time=start_time,
                end_time=end_time,
            ) from exc

    end_time = _iso8601_now()
    success(
        f"GuideLLM benchmark completed for {plan.model.name} "
        f"({'MLflow run ' + run_id if run_id else 'local output'})"
    )
    return run_id, start_time, end_time


def generate_report(
    *,
    json_path: Path | None = None,
    model: str | None = None,
    accelerator: str | None = None,
    version: str | None = None,
    tp_size: int = 1,
    runtime_args: str = "",
    output_dir: Path | None = None,
    output_file: Path | None = None,
    replicas: int = 1,
    mlflow_run_ids: list[str] | None = None,
    mlflow_tracking_uri: str | None = None,
    versions: list[str] | None = None,
    version_overrides: dict[str, str] | None = None,
    additional_csv_files: list[str] | None = None,
    notes: list[str] | None = None,
    repeat_section_legends: bool = False,
) -> Path:
    module = _load_guidellm_module()

    if mlflow_run_ids:
        runs_data = module.fetch_mlflow_runs(mlflow_run_ids, mlflow_tracking_uri)
        html_path = module.generate_plot_only_report(
            runs_data=runs_data,
            versions=versions,
            mlflow_tracking_uri=mlflow_tracking_uri,
            additional_csv_files=additional_csv_files,
            versions_override=version_overrides or {},
            output_dir=str(output_dir) if output_dir else None,
            output_file=str(output_file) if output_file else None,
            notes=notes or [],
            repeat_section_legends=repeat_section_legends,
        )
        if not html_path:
            raise CommandError("GuideLLM report generation returned no output path")
        return Path(html_path)

    if json_path is None or model is None or version is None:
        raise CommandError(
            "single-run report generation requires --json-path, --model, and --version"
        )

    html_path = module.generate_visualization_report(
        json_path=str(json_path),
        model=model,
        accelerator=accelerator,
        version=version,
        tp_size=tp_size,
        runtime_args=runtime_args,
        output_dir=str(output_dir) if output_dir else None,
        output_file=str(output_file) if output_file else None,
        replicas=replicas,
        notes=notes or [],
        repeat_section_legends=repeat_section_legends,
    )
    if not html_path:
        raise CommandError("GuideLLM report generation returned no output path")
    return Path(html_path)
