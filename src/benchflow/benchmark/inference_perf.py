from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import yaml

from ..cluster import CommandError, require_command
from ..mlflow_compat import configure_mlflow_tracking
from ..models import InferencePerfBenchmarkSpec, ResolvedRunPlan, ValidationError
from ..ui import detail, step, success
from .common import (
    BenchmarkRunFailed,
    benchmark_version_from_plan,
    resolved_accelerator,
)

_ARTIFACT_ROOT = "benchmark"
_SUMMARY_FILENAME = "summary_lifecycle_metrics.json"


def is_inference_perf_artifact_paths(paths: set[str]) -> bool:
    """Recognize native Inference Perf lifecycle report artifacts."""
    return any(path.endswith(f"reports/{_SUMMARY_FILENAME}") for path in paths)


def _iso8601_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _spec(plan: ResolvedRunPlan) -> InferencePerfBenchmarkSpec:
    if plan.benchmark.tool != "inference-perf":
        raise ValidationError("Inference Perf runner requires tool: inference-perf")
    return plan.benchmark.inference_perf


def _artifact_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    return Path(tempfile.mkdtemp(prefix="benchflow-inference-perf-"))


def _runtime_environment() -> dict[str, str]:
    root = Path("/tmp/benchflow-inference-perf")
    home = root / "home"
    hf_home = root / "huggingface"
    xdg_cache = root / "xdg-cache"
    for path in (home, hf_home, xdg_cache):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "HOME": str(home),
        "HF_HOME": str(hf_home),
        "XDG_CACHE_HOME": str(xdg_cache),
        "HF_HUB_CACHE": str(hf_home / "hub"),
        "TRANSFORMERS_CACHE": str(hf_home / "transformers"),
    }


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValidationError(f"{field_name} must be a mapping")
    return copy.deepcopy(value)


def render_config(
    *, plan: ResolvedRunPlan, target: str, artifact_dir: Path
) -> dict[str, Any]:
    """Resolve a native Inference Perf config with BenchFlow-owned wiring."""
    config = copy.deepcopy(_spec(plan).config)
    model_name = plan.model.resolved_name()

    server = _mapping(config.get("server"), "benchmark.inference_perf.config.server")
    server.setdefault("type", "vllm")
    server["model_name"] = model_name
    server["base_url"] = target
    config["server"] = server

    tokenizer = _mapping(
        config.get("tokenizer"), "benchmark.inference_perf.config.tokenizer"
    )
    tokenizer.setdefault("pretrained_model_name_or_path", model_name)
    config["tokenizer"] = tokenizer

    storage = _mapping(config.get("storage"), "benchmark.inference_perf.config.storage")
    local_storage = _mapping(
        storage.get("local_storage"),
        "benchmark.inference_perf.config.storage.local_storage",
    )
    local_storage["path"] = str(artifact_dir / "reports")
    storage["local_storage"] = local_storage
    config["storage"] = storage
    return config


def _write_config(config: dict[str, Any], artifact_dir: Path) -> Path:
    path = artifact_dir / "inference-perf-config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _run_command(command: list[str], *, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                print(line, end="")
            exit_code = process.wait()
    except OSError as exc:
        raise CommandError(f"failed to execute {' '.join(command)}: {exc}") from exc
    if exit_code:
        raise CommandError(f"{' '.join(command)} exited with status {exit_code}")


def _load_summary(artifact_dir: Path) -> dict[str, Any] | None:
    path = artifact_dir / "reports" / _SUMMARY_FILENAME
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BenchmarkRunFailed(f"Inference Perf summary is not a JSON object: {path}")
    return payload


def _nested_number(payload: dict[str, Any], *path: str) -> float | None:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    try:
        return float(current)
    except (TypeError, ValueError):
        return None


def _log_summary_metrics(summary: dict[str, Any]) -> None:
    metric_paths = {
        "inference_perf.successful_requests": ("successes", "count"),
        "inference_perf.failed_requests": ("failures", "count"),
        "inference_perf.requests_per_second": (
            "successes",
            "throughput",
            "requests_per_sec",
        ),
        "inference_perf.input_tokens_per_second": (
            "successes",
            "throughput",
            "input_tokens_per_sec",
        ),
        "inference_perf.output_tokens_per_second": (
            "successes",
            "throughput",
            "output_tokens_per_sec",
        ),
        "inference_perf.total_tokens_per_second": (
            "successes",
            "throughput",
            "total_tokens_per_sec",
        ),
        "inference_perf.request_latency_seconds_p95": (
            "successes",
            "latency",
            "request_latency",
            "p95",
        ),
        "inference_perf.ttft_seconds_p95": (
            "successes",
            "latency",
            "time_to_first_token",
            "p95",
        ),
        "inference_perf.itl_seconds_p95": (
            "successes",
            "latency",
            "inter_token_latency",
            "p95",
        ),
    }
    for name, path in metric_paths.items():
        value = _nested_number(summary, *path)
        if value is not None:
            mlflow.log_metric(name, value)


def _log_artifacts(artifact_dir: Path) -> None:
    for child in sorted(artifact_dir.iterdir()):
        if child.is_dir():
            mlflow.log_artifacts(
                str(child), artifact_path=f"{_ARTIFACT_ROOT}/{child.name}"
            )
        elif child.is_file():
            mlflow.log_artifact(str(child), artifact_path=_ARTIFACT_ROOT)


def run_benchmark(
    *,
    plan: ResolvedRunPlan,
    target: str | None = None,
    output_dir: Path | None = None,
    mlflow_tracking_uri: str | None = None,
    enable_mlflow: bool = True,
    mlflow_run_id: str = "",
    extra_tags: dict[str, str] | None = None,
) -> tuple[str, str, str]:
    require_command("inference-perf")
    artifact_dir = _artifact_dir(output_dir)
    remove_artifact_dir = output_dir is None
    benchmark_target = target or plan.deployment.target.base_url
    start_time = _iso8601_now()
    run_id = ""
    config = render_config(
        plan=plan, target=benchmark_target, artifact_dir=artifact_dir
    )
    config_path = _write_config(config, artifact_dir)
    log_path = artifact_dir / "logs" / "inference-perf.log"
    command = ["inference-perf", "--config_file", str(config_path)]
    environment = dict(os.environ)
    environment.update(_runtime_environment())
    environment.update(plan.benchmark.env)

    tags = dict(plan.mlflow.tags)
    if extra_tags:
        tags.update(extra_tags)
    tags.setdefault("accelerator", resolved_accelerator(plan))
    tags.setdefault("version", benchmark_version_from_plan(plan))
    tags.setdefault("benchmark_tool", "inference-perf")

    model_name = plan.model.resolved_name()
    step(f"Preparing Inference Perf benchmark run for {model_name}")
    detail(f"Target: {benchmark_target}")
    detail(f"Artifact directory: {artifact_dir}")
    detail(f"MLflow: {'enabled' if enable_mlflow else 'disabled'}")
    try:
        if enable_mlflow:
            tracking_uri = mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
            if not tracking_uri:
                raise BenchmarkRunFailed(
                    "MLFLOW_TRACKING_URI is required when MLflow is enabled"
                )
            configure_mlflow_tracking(tracking_uri)
            mlflow.set_experiment(plan.mlflow.experiment or "Default")
            start_run_kwargs = (
                {"run_id": mlflow_run_id.strip()}
                if mlflow_run_id.strip()
                else {"tags": tags}
            )
            with mlflow.start_run(**start_run_kwargs) as run:
                run_id = run.info.run_id
                if mlflow_run_id:
                    mlflow.set_tags(tags)
                mlflow.log_params(
                    {
                        "benchmark_tool": "inference-perf",
                        "backend_type": str(
                            (config.get("server") or {}).get("type", "")
                        ),
                        "target": benchmark_target,
                        "model": model_name,
                        "tp": plan.deployment.runtime.tensor_parallelism,
                        "replicas": plan.deployment.runtime.replicas,
                        "version": benchmark_version_from_plan(plan),
                        "inference_perf_data_type": str(
                            (config.get("data") or {}).get("type", "")
                        ),
                        "inference_perf_load_type": str(
                            (config.get("load") or {}).get("type", "")
                        ),
                    }
                )
                try:
                    _run_command(command, env=environment, log_path=log_path)
                    summary = _load_summary(artifact_dir)
                    if summary is not None:
                        _log_summary_metrics(summary)
                finally:
                    # Preserve the rendered config and combined process log even
                    # when the native runner exits unsuccessfully.
                    _log_artifacts(artifact_dir)
        else:
            _run_command(command, env=environment, log_path=log_path)
            _load_summary(artifact_dir)
    except Exception as exc:  # noqa: BLE001
        raise BenchmarkRunFailed(
            str(exc),
            run_id=run_id,
            start_time=start_time,
            end_time=_iso8601_now(),
        ) from exc
    finally:
        if remove_artifact_dir:
            shutil.rmtree(artifact_dir, ignore_errors=True)

    end_time = _iso8601_now()
    success(
        f"Inference Perf benchmark completed for {model_name} "
        f"({'MLflow run ' + run_id if run_id else 'local output'})"
    )
    return run_id, start_time, end_time
