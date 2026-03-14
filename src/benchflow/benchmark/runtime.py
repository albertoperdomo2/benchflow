import argparse
import json
import logging
import subprocess
import sys
import shutil
import os
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import mlflow

from ..ui import configure_logging, emit

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


def _get_nested(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely get a nested value from a dictionary."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, default)
    return d


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
    rate: str,
    backend_type: str = "openai_http",
    rate_type: str = "concurrent",
    data: str = None,
    max_seconds=None,
    max_requests=None,
    processor: str = None,
    output_path: str = "benchmark_output.json",
) -> tuple[str, str]:
    cmd = [
        "guidellm",
        "benchmark",
        "run",
        "--target",
        target,
        "--model",
        model,
        "--backend-type",
        backend_type,
        "--rate-type",
        rate_type,
        "--rate",
        str(rate),
        "--output-path",
        output_path,
    ]

    cmd.extend(["--backend-args", '{"timeout": 600}'])
    if target.startswith("https://"):
        # cmd.extend(["--backend-kwargs", '{"verify": false}'])
        cmd.extend(["--backend-args", '{"verify": false, "timeout": 600}'])
    if data:
        cmd.extend(["--data", data])
    if max_seconds:
        cmd.extend(["--max-seconds", str(max_seconds)])
    if max_requests:
        cmd.extend(["--max-requests", str(max_requests)])
    if processor:
        cmd.extend(["--processor", processor])

    logger.info(f"Running guidellm command: {' '.join(cmd)}")

    console_log_path = output_path.replace(".json", "_console.log")

    try:
        with open(console_log_path, "w") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
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

        return output_path, console_log_path

    except Exception as e:
        logger.error(f"Guidellm command failed: {e}")
        raise


def generate_visualization_report(
    json_path: str,
    model: str,
    accelerator: str = None,
    version: str = None,
    tp_size: int = 1,
    runtime_args: str = "",
    output_dir: str = None,
    replicas: int = 1,
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
        replicas: Number of replicas

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

        if output_dir:
            html_path = str(Path(output_dir) / html_filename)
        else:
            html_path = f"/tmp/{html_filename}"

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
    rate: str,
    backend_type: str,
    rate_type: str,
    data: str,
    max_seconds,
    max_requests,
    processor: str,
    output_dir: str,
    accelerator: str,
    version: str,
    tp_size: int,
    runtime_args: str,
    replicas: int = 1,
) -> tuple:
    """Helper to run guidellm and process results."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_json = f"{output_dir}/benchmark_output.json"

    json_path, console_log_path = run_guidellm_cli(
        target=target,
        model=model,
        rate=rate,
        backend_type=backend_type,
        rate_type=rate_type,
        data=data,
        max_seconds=max_seconds,
        max_requests=max_requests,
        processor=processor,
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

    html_report = generate_visualization_report(
        json_path=json_path,
        model=model,
        accelerator=accelerator,
        version=version,
        tp_size=tp_size,
        runtime_args=runtime_args,
        output_dir=output_dir,
        replicas=replicas,
    )

    return json_path, console_log_path, benchmarks, html_report


def run_benchmark_without_mlflow(
    target: str,
    model: str,
    rate: str,
    backend_type: str = "openai_http",
    rate_type: str = "concurrent",
    data: str = None,
    max_seconds=None,
    max_requests=None,
    processor: str = None,
    output_dir: str = "/benchmark-results",
    accelerator: str = None,
    version: str = None,
    tp_size: int = 1,
    runtime_args: str = "",
    replicas: int = 1,
) -> str:
    """Run benchmark without MLflow tracking, saving results to specified directory."""
    logger.info("Running benchmark without MLflow tracking")
    logger.info(f"Starting benchmark for rates: {rate}")
    logger.info(f"Results will be saved to: {output_dir}")

    json_path, console_log_path, benchmarks, html_report = _run_and_process_benchmark(
        target=target,
        model=model,
        rate=rate,
        backend_type=backend_type,
        rate_type=rate_type,
        data=data,
        max_seconds=max_seconds,
        max_requests=max_requests,
        processor=processor,
        output_dir=output_dir,
        accelerator=accelerator,
        version=version,
        tp_size=tp_size,
        runtime_args=runtime_args,
        replicas=replicas,
    )

    for i, benchmark in enumerate(benchmarks):
        metrics = extract_metrics_from_benchmark(benchmark)
        if metrics:
            logger.info(f"Benchmark {i + 1} metrics: {json.dumps(metrics, indent=2)}")

    if Path(console_log_path).exists():
        logger.info(f"Console log saved to: {console_log_path}")

    if html_report and Path(html_report).exists():
        logger.info(f"Visualization report saved to: {html_report}")
    else:
        logger.info("Visualization report not generated (continuing without it)")

    return json_path


def run_benchmark_with_mlflow(
    target: str,
    model: str,
    rate: str,
    backend_type: str = "openai_http",
    rate_type: str = "concurrent",
    data: str = None,
    max_seconds=None,
    max_requests=None,
    processor: str = None,
    accelerator: str = None,
    experiment_name: str = "guidellm-benchmarks",
    mlflow_tracking_uri: str = None,
    tags: Dict[str, str] = None,
    version: str = None,
    tp_size: int = 1,
    runtime_args: str = "",
    replicas: str = "N/A",
    prefill_replicas: str = "N/A",
    decode_replicas: str = "N/A",
) -> str:
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    mlflow.set_experiment(experiment_name)

    # Check if multi-turn mode is enabled
    multiturn_mode = os.environ.get("MULTITURN", "false").lower() == "true"

    # Run name for the whole sweep
    # Use PipelineRun name if provided (Tekton integration), otherwise generate one
    pipeline_run_name = os.environ.get("PIPELINE_RUN_NAME", "")
    if pipeline_run_name:
        run_name = pipeline_run_name
    else:
        mode_suffix = "multiturn" if multiturn_mode else "sweep"
        run_name = f"{model.split('/')[-1]}_{mode_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info(f"Starting benchmark sweep: rates={rate}")
    if multiturn_mode:
        logger.info(
            "MULTITURN mode enabled - running separate commands per concurrency"
        )

    with mlflow.start_run(run_name=run_name) as run:
        try:
            # Common params for the whole sweep
            params = {
                "target": target,
                "model": model,
                "backend_type": backend_type,
                "rate_type": rate_type,
                "rates": rate,
                "tp": tp_size,
                "replicas": replicas,
                "prefill_replicas": prefill_replicas,
                "decode_replicas": decode_replicas,
                "multiturn_mode": multiturn_mode,
            }
            if data:
                params.update(
                    {
                        d.split("=")[0].strip(): d.split("=")[1].strip()
                        for d in data.split(",")
                    }
                )
            if max_seconds:
                params["max_seconds"] = max_seconds
            if max_requests:
                params["max_requests"] = max_requests
            if processor:
                params["processor"] = processor
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
                "model": model,
                "rate_type": rate_type,
                "vllm": vllm_version,
                "guidellm": guidellm_version,
            }
            if pipeline_run_name:
                default_tags["pipeline_run"] = pipeline_run_name
            if accelerator:
                default_tags["accelerator"] = accelerator
            if tags:
                default_tags.update(tags)
            mlflow.set_tags(default_tags)

            # Multi-turn mode: loop over concurrencies and run separate commands
            if multiturn_mode:
                concurrencies = [r.strip() for r in rate.split(",")]
                logger.info(f"Running {len(concurrencies)} separate benchmark commands")

                for concurrency_str in concurrencies:
                    try:
                        concurrency = int(concurrency_str)
                        logger.info(f"Starting benchmark for concurrency={concurrency}")

                        # Parse data and max_requests with concurrency substitution
                        parsed_data = (
                            parse_multiturn_data_param(data, concurrency)
                            if data
                            else None
                        )
                        parsed_max_requests = None
                        if max_requests:
                            parsed_max_requests = int(
                                parse_multiturn_expression(
                                    str(max_requests), concurrency
                                )
                            )

                        parsed_max_seconds = None
                        if max_seconds:
                            parsed_max_seconds = int(
                                parse_multiturn_expression(
                                    str(max_seconds), concurrency
                                )
                            )

                        logger.info(f"  Original data: {data}")
                        logger.info(f"  Parsed data: {parsed_data}")
                        logger.info(f"  Original max_requests: {max_requests}")
                        logger.info(f"  Parsed max_requests: {parsed_max_requests}")
                        logger.info(f"  Original max_seconds: {max_seconds}")
                        logger.info(f"  Parsed max_seconds: {parsed_max_seconds}")

                        # Generate unique output paths for this concurrency
                        output_json = f"/tmp/benchmark_output_rate_{concurrency}.json"
                        console_log_path = output_json.replace(".json", "_console.log")

                        # Run guidellm for this concurrency only
                        json_path, console_log = run_guidellm_cli(
                            target=target,
                            model=model,
                            rate=concurrency_str,
                            backend_type=backend_type,
                            rate_type=rate_type,
                            data=parsed_data,
                            max_seconds=parsed_max_seconds,
                            max_requests=parsed_max_requests,
                            processor=processor,
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
                            logger.warning(f"Output JSON not found: {json_path}")

                        # Extract and log metrics with step=concurrency
                        for benchmark in benchmarks:
                            metrics = extract_metrics_from_benchmark(benchmark)
                            if metrics:
                                metrics["concurrency"] = concurrency
                                for key, value in metrics.items():
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

                    except Exception as e:
                        logger.error(
                            f"Benchmark failed for concurrency={concurrency_str}: {e}",
                            exc_info=True,
                        )
                        logger.info("Continuing with remaining concurrencies...")
                        continue

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
                    html_report,
                ) = _run_and_process_benchmark(
                    target=target,
                    model=model,
                    rate=rate,
                    backend_type=backend_type,
                    rate_type=rate_type,
                    data=data,
                    max_seconds=max_seconds,
                    max_requests=max_requests,
                    processor=processor,
                    output_dir="/tmp",
                    accelerator=accelerator,
                    version=version,
                    tp_size=tp_size,
                    runtime_args=runtime_args,
                    replicas=int(replicas) if replicas != "N/A" else 1,
                )

                if not benchmarks:
                    logger.warning("No benchmarks found in JSON output")

                for benchmark in benchmarks:
                    concurrency_step = 0
                    config_or_args = benchmark.get("config") or benchmark.get(
                        "args", {}
                    )
                    try:
                        concurrency_step = int(config_or_args["strategy"]["streams"])
                    except (KeyError, TypeError, IndexError):
                        try:
                            concurrency_step = int(
                                config_or_args["profile"]["streams"][0]
                            )
                        except (KeyError, TypeError, IndexError):
                            logger.warning(
                                "Could not find concurrency 'streams'. "
                                "Metrics will be logged without a step."
                            )

                    metrics = extract_metrics_from_benchmark(benchmark)
                    if metrics:
                        metrics["concurrency"] = concurrency_step
                        for key, value in metrics.items():
                            mlflow.log_metric(key, value, step=concurrency_step)
                        logger.info(
                            f"Logged {len(metrics)} metrics for step "
                            f"(concurrency={concurrency_step})"
                        )

                if Path(json_path).exists():
                    mlflow.log_artifact(json_path, "results")
                    logger.info("Logged full JSON artifact")

                if Path(console_log_path).exists():
                    mlflow.log_artifact(console_log_path, "logs")
                    logger.info("Logged console output")

                if html_report and Path(html_report).exists():
                    mlflow.log_artifact(html_report, "reports")
                    logger.info(f"Logged visualization report to MLflow: {html_report}")
                else:
                    logger.info(
                        "Visualization report not generated (continuing without it)"
                    )

            logger.info(f"Run completed: {run.info.run_id}")
            return run.info.run_id

        except Exception as e:
            logger.error(f"Benchmark sweep failed: {e}", exc_info=True)
            mlflow.log_param("error", str(e))
            raise


def fetch_mlflow_runs(run_ids: list, mlflow_tracking_uri: str = None) -> list:
    """
    Fetch MLflow runs by their IDs and download their benchmark JSON artifacts.

    Args:
        run_ids: List of MLflow run IDs
        mlflow_tracking_uri: MLflow tracking URI (optional)

    Returns:
        List of dictionaries containing run metadata and benchmark data
        Each dict includes a 'composed_version' field that appends the 'epp' tag
        to the base version if present (e.g., llm-d-0.4-precise-prefix-caching)
    """
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    runs_data = []

    for run_id in run_ids:
        try:
            logger.info(f"Fetching MLflow run: {run_id}")
            run = mlflow.get_run(run_id)

            params = run.data.params
            tags = run.data.tags

            # Compose version with epp tag if present
            base_version = params.get("version", "unknown")
            epp_tag = tags.get("epp")

            if epp_tag:
                composed_version = f"{base_version}-{epp_tag}"
                logger.info(
                    f"Composed version: {base_version} + epp={epp_tag} -> {composed_version}"
                )
            else:
                composed_version = base_version
                logger.info(f"No epp tag found, using base version: {composed_version}")

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
                # Download ALL benchmark*.json files from MLflow
                client = mlflow.tracking.MlflowClient()
                artifacts = client.list_artifacts(run_id, "results")
                benchmark_files = [
                    a.path
                    for a in artifacts
                    if a.path.startswith("results/benchmark")
                    and a.path.endswith(".json")
                ]

                if not benchmark_files:
                    raise ValueError(f"No benchmark JSON files found for run {run_id}")

                logger.info(
                    f"Downloading {len(benchmark_files)} benchmark file(s) for run {run_id}"
                )

                # Download and cache all files
                for benchmark_file in benchmark_files:
                    downloaded_path = client.download_artifacts(run_id, benchmark_file)
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
        Tuple of (model, rate, data_profile) if compatible

    Raises:
        ValueError if runs are incompatible
    """
    if not runs_data:
        raise ValueError("No runs provided for validation")

    # Extract model, rate, and data profile from first run
    first_run = runs_data[0]
    model = first_run["params"].get("model")
    rate = first_run["params"].get("rates")

    # Data profile parameters (may or may not be present)
    data_profile_params = [
        "prompt_tokens",
        "output_tokens",
        "turns",
        "prefix_tokens",
        "prefix_count",
    ]

    first_profile = {
        param: first_run["params"].get(param) for param in data_profile_params
    }

    # Validate all runs have same configuration
    for run_data in runs_data[1:]:
        params = run_data["params"]

        if params.get("model") != model:
            raise ValueError(
                f"Model mismatch: {params.get('model')} != {model}. "
                f"All runs must use the same model."
            )

        if params.get("rates") != rate:
            raise ValueError(
                f"Rate mismatch: {params.get('rates')} != {rate}. "
                f"All runs must use the same rate configuration."
            )

        # Check each data profile parameter
        for param in data_profile_params:
            if params.get(param) != first_profile[param]:
                raise ValueError(
                    f"Data profile mismatch: {param}={params.get(param)} != {first_profile[param]}. "
                    f"All runs must use the same data profile."
                )

    # Build data profile string from present parameters only
    profile_parts = []
    for param in data_profile_params:
        value = first_profile[param]
        if value is not None:
            profile_parts.append(f"{param}={value}")

    data_profile = ",".join(profile_parts) if profile_parts else None

    logger.info(f"All runs validated successfully:")
    logger.info(f"  Model: {model}")
    logger.info(f"  Rate: {rate}")
    logger.info(f"  Data profile: {data_profile}")

    return model, rate, data_profile


def generate_plot_only_report(
    runs_data: list,
    versions: list = None,
    mlflow_tracking_uri: str = None,
    additional_csv_files: list = None,
    versions_override: dict = None,
) -> str:
    """
    Generate HTML report from existing MLflow runs without running benchmarks.

    Args:
        runs_data: List of run data dictionaries
        versions: List of versions to filter/compare (optional)
        mlflow_tracking_uri: MLflow tracking URI (optional)
        additional_csv_files: List of additional CSV file paths to include (optional)
        versions_override: Dictionary mapping old version names to new names (optional)

    Returns:
        Path to generated HTML report
    """
    if not PROCESSOR_AVAILABLE:
        logger.error("BenchmarkProcessor not available - cannot generate report")
        return None

    # Handle default case for versions_override
    if versions_override is None:
        versions_override = {}

    # Validate runs compatibility
    model, rate, data_profile = validate_runs_compatibility(runs_data)

    # Extract data profile parameters from first run
    first_run_params = runs_data[0]["params"]
    data_profile_params = {
        "prompt_tokens": first_run_params.get("prompt_tokens"),
        "output_tokens": first_run_params.get("output_tokens"),
        "turns": first_run_params.get("turns"),
        "prefix_tokens": first_run_params.get("prefix_tokens"),
        "prefix_count": first_run_params.get("prefix_count"),
    }
    # Convert string values to int if present
    for key, value in data_profile_params.items():
        if value is not None:
            try:
                data_profile_params[key] = int(value)
            except (ValueError, TypeError):
                pass  # Keep as is if conversion fails

    # Filter runs by version if specified (using prefix match for MLflow runs)
    if versions:
        logger.info(f"Filtering runs by base versions: {versions}")
        filtered_runs = []
        for run_data in runs_data:
            composed_version = run_data["composed_version"]
            # Check if any base version matches as a prefix
            matches = any(composed_version.startswith(base_v) for base_v in versions)
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
            raise ValueError(f"No runs found matching base versions: {versions}")

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
        **data_profile_params,  # Pass data profile parameters
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

    for run_data in runs_data:
        run_id = run_data["run_id"]
        params = run_data["params"]
        artifact_path = run_data["artifact_path"]
        composed_version = run_data["composed_version"]

        accelerator = params.get("accelerator", "unknown")
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
            version=composed_version,  # Use composed version with epp tag
            tp_size=tp_size,
            runtime_args="",
            replicas=replicas_int,
            **data_profile_params,  # Pass data profile parameters
        )

        # Parse this run's JSON to DataFrame (replicas will be included via processor)
        run_df = processor.parse_guidellm_json()

        # Mark MLflow data with source column for filtering logic
        run_df["_data_source"] = "mlflow"

        logger.info(f"Extracted {len(run_df)} rows from run {run_id}")

        all_run_dataframes.append(run_df)

    # Combine all run DataFrames using BenchmarkProcessor's merge logic
    logger.info(f"Combining {len(all_run_dataframes)} DataFrames")
    combined_runs_df = pd.concat(all_run_dataframes, ignore_index=True)
    logger.info(f"Combined runs DataFrame has {len(combined_runs_df)} rows")

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
    logger.info(
        f"Restored _data_source column: "
        f"{(final_df['_data_source'] == 'mlflow').sum()} MLflow rows, "
        f"{(final_df['_data_source'] == 'csv').sum()} CSV rows"
    )

    # Filter by versions if specified (different logic for CSV vs MLflow data)
    if versions:
        logger.info(f"Filtering combined data by versions: {versions}")
        initial_rows = len(final_df)

        # Apply different filtering logic based on data source
        def should_keep_row(row):
            data_source = row.get("_data_source", "csv")
            version = row["version"]

            if data_source == "csv":
                # CSV data: exact match only
                return version in versions
            else:  # mlflow
                # MLflow data: prefix match (base version matches)
                return any(version.startswith(base_v) for base_v in versions)

        mask = final_df.apply(should_keep_row, axis=1)
        final_df = final_df[mask]

        logger.info(
            f"After version filtering: {len(final_df)} rows (removed {initial_rows - len(final_df)} rows)"
        )
        logger.info(f"  CSV data filtered with exact match")
        logger.info(f"  MLflow data filtered with prefix match")

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
    html_path = f"/tmp/{html_filename}"

    # Generate report using the combined DataFrame
    final_processor = BenchmarkProcessor(
        json_path=first_run["artifact_path"],
        s3_bucket=s3_bucket,
        s3_key=s3_key,
        accelerator=params.get("accelerator", "unknown"),
        model_name=model,
        version=params.get("version", "unknown"),
        tp_size=int(params.get("tp", 1)),
        runtime_args="",
        compare_versions=compare_versions,
        output_html=html_path,
        **data_profile_params,  # Pass data profile parameters
    )

    # Override with our merged and filtered data
    final_processor.combined_df = final_df
    final_processor.config = final_processor.load_config()
    final_processor.generate_report()

    if Path(html_path).exists():
        logger.info(f"Comparison report generated: {html_path}")
        return html_path
    else:
        logger.error("Report generation failed - file not found")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="GuideLLM Benchmark with MLflow Logging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--target", help="Target URL (required for benchmark mode)")
    parser.add_argument("--model", help="Model name (required for benchmark mode)")
    parser.add_argument("--backend-type", default="openai_http", help="Backend type")
    parser.add_argument("--rate-type", default="concurrent", help="Rate type")
    parser.add_argument(
        "--rate", help="Rate value(s), comma-separated (required for benchmark mode)"
    )
    parser.add_argument(
        "--data", help="Data config (e.g., 'prompt_tokens=1000,output_tokens=1000')"
    )
    parser.add_argument(
        "--max-seconds",
        help="Max duration in seconds (supports expressions in MULTITURN mode)",
    )
    parser.add_argument(
        "--max-requests",
        help="Max number of requests (supports expressions like '10*concurrency' in MULTITURN mode)",
    )
    parser.add_argument("--processor", help="Processor/tokenizer name")

    parser.add_argument("--accelerator", help="Accelerator type (e.g., H200, A100)")
    parser.add_argument(
        "--version", help="Version identifier for visualization reports"
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallelism size for visualization reports",
    )
    parser.add_argument(
        "--runtime-args", default="", help="Runtime arguments for visualization reports"
    )

    # Replica configuration parameters
    parser.add_argument(
        "--replicas",
        default="N/A",
        help="Number of replicas for standard deployment mode",
    )
    parser.add_argument(
        "--prefill-replicas",
        default="N/A",
        help="Number of prefill worker replicas for P/D disaggregation",
    )
    parser.add_argument(
        "--decode-replicas",
        default="N/A",
        help="Number of decode worker replicas for P/D disaggregation",
    )

    parser.add_argument(
        "--experiment-name",
        default="guidellm-benchmarks",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI"),
        help="MLflow tracking URI (defaults to MLFLOW_TRACKING_URI environment variable)",
    )
    parser.add_argument(
        "--tag", action="append", dest="tags", help="Additional tags (key=value)"
    )

    # Plot-only mode arguments
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Generate plots from existing MLflow runs without running benchmarks",
    )
    parser.add_argument(
        "--mlflow-run-ids",
        help="Comma-separated list of MLflow run IDs to plot (required with --plot-only)",
    )
    parser.add_argument(
        "--versions",
        help="Comma-separated list of versions to compare (filters runs and sets compare_versions)",
    )
    parser.add_argument(
        "--versions-override",
        action="append",
        dest="versions_override",
        help="Version rename mappings (old_name=new_name). Renames version labels after filtering but before plotting. Can be specified multiple times. Only applies to --plot-only mode.",
    )
    parser.add_argument(
        "--additional-csv",
        action="append",
        dest="additional_csv_files",
        help="Additional CSV file(s) to include in comparison plots (only for --plot-only mode). Can be specified multiple times.",
    )

    args = parser.parse_args()

    # Handle plot-only mode
    if args.plot_only:
        logger.info("Plot-only mode enabled")

        # Validate required arguments for plot-only mode
        if not args.mlflow_run_ids:
            parser.error("--mlflow-run-ids is required when using --plot-only")

        # Validate additional CSV files if provided
        if args.additional_csv_files:
            logger.info(
                f"Will include {len(args.additional_csv_files)} additional CSV file(s)"
            )
            for csv_file in args.additional_csv_files:
                if not Path(csv_file).exists():
                    parser.error(f"Additional CSV file not found: {csv_file}")

        # Parse run IDs and versions
        run_ids = [rid.strip() for rid in args.mlflow_run_ids.split(",") if rid.strip()]
        versions = (
            [v.strip() for v in args.versions.split(",") if v.strip()]
            if args.versions
            else None
        )

        # Parse version override mappings
        versions_override = {}
        if args.versions_override:
            logger.info(f"Parsing {len(args.versions_override)} version override(s)")
            for mapping in args.versions_override:
                if "=" not in mapping:
                    parser.error(
                        f"Invalid version override format: {mapping}. Expected format: old_name=new_name"
                    )
                old_version, new_version = mapping.split("=", 1)
                old_version = old_version.strip()
                new_version = new_version.strip()
                versions_override[old_version] = new_version
                logger.info(f"  Will rename: {old_version} → {new_version}")

        logger.info(f"Fetching {len(run_ids)} MLflow runs...")

        try:
            # Fetch runs from MLflow
            runs_data = fetch_mlflow_runs(run_ids, args.mlflow_tracking_uri)

            if not runs_data:
                logger.error("No runs fetched successfully")
                return 1

            # Generate plot-only report
            html_report = generate_plot_only_report(
                runs_data=runs_data,
                versions=versions,
                mlflow_tracking_uri=args.mlflow_tracking_uri,
                additional_csv_files=args.additional_csv_files,
                versions_override=versions_override,
            )

            if html_report:
                logger.info("\nPlot generation completed successfully.")
                logger.info(f"  Report saved to: {html_report}")
                return 0
            else:
                logger.error("Plot generation failed")
                return 1

        except Exception as e:
            logger.error(f"Plot generation failed: {e}", exc_info=True)
            return 1

    # Validate required arguments for benchmark mode
    if not args.target:
        parser.error("--target is required for benchmark mode")
    if not args.model:
        parser.error("--model is required for benchmark mode")
    if not args.rate:
        parser.error("--rate is required for benchmark mode")

    tags = {}
    if args.tags:
        for tag in args.tags:
            key, value = tag.split("=", 1)
            tags[key.strip()] = value.strip()

    logger.info(f"Starting benchmark sweep for rates: {args.rate}")

    # Log in to HF
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
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

    # Check if MLflow is enabled via environment variable
    mlflow_enabled = os.environ.get("MLFLOW_ENABLED", "false").lower() == "true"

    if not mlflow_enabled:
        logger.info("MLflow tracking disabled - running benchmark without MLflow")
        try:
            json_path = run_benchmark_without_mlflow(
                target=args.target,
                model=args.model,
                rate=args.rate,
                backend_type=args.backend_type,
                rate_type=args.rate_type,
                data=args.data,
                max_seconds=args.max_seconds,
                max_requests=args.max_requests,
                processor=args.processor,
                output_dir="/benchmark-results",
                accelerator=args.accelerator,
                version=args.version,
                tp_size=args.tp,
                runtime_args=args.runtime_args,
            )
            logger.info("\nBenchmark completed successfully.")
            logger.info(f"  Results saved to: {json_path}")
            return 0
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return 1

    logger.info("MLflow tracking enabled")
    try:
        run_id = run_benchmark_with_mlflow(
            target=args.target,
            model=args.model,
            rate=args.rate,
            backend_type=args.backend_type,
            rate_type=args.rate_type,
            data=args.data,
            max_seconds=args.max_seconds,
            max_requests=args.max_requests,
            processor=args.processor,
            accelerator=args.accelerator,
            experiment_name=args.experiment_name,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            tags=tags,
            version=args.version,
            tp_size=args.tp,
            runtime_args=args.runtime_args,
            replicas=args.replicas,
            prefill_replicas=args.prefill_replicas,
            decode_replicas=args.decode_replicas,
        )
        logger.info("\nBenchmark sweep completed successfully.")
        logger.info(f"  MLflow Run ID: {run_id}")
        return 0
    except Exception as e:
        logger.error(f"Benchmark sweep failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
