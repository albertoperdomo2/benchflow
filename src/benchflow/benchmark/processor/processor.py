import ast
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List

import boto3
import pandas as pd
import plotly.graph_objects as go
import yaml
from botocore.exceptions import ClientError
from plotly.subplots import make_subplots

from ...ui import configure_logging

configure_logging("INFO")
logger = logging.getLogger("benchmark_processor")


def _get_nested(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely get a nested value from a dictionary."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, default)
    return d


def _parse_request_data(requests_data: Any) -> Dict[str, int]:
    """Parse request data to extract prompt and output tokens."""
    prompt_tokens = 0
    output_tokens = 0
    data_str = ""

    if isinstance(requests_data, str):
        # Try to evaluate as a Python literal first (handles "['...']" format)
        try:
            evaluated_data = ast.literal_eval(requests_data)
            if isinstance(evaluated_data, list) and evaluated_data:
                data_str = evaluated_data[0]
            elif isinstance(evaluated_data, str):
                data_str = evaluated_data
        except (ValueError, SyntaxError):
            # If evaluation fails, use the string directly
            data_str = requests_data
    elif isinstance(requests_data, list) and requests_data:
        data_str = requests_data[0]

    if data_str:
        for part in data_str.split(","):
            if "=" in part:
                key, value = part.strip().split("=", 1)
                key = key.strip()
                value = value.strip()
                try:
                    if key == "prompt_tokens":
                        prompt_tokens = int(value)
                    elif key == "output_tokens":
                        output_tokens = int(value)
                except ValueError:
                    pass  # Ignore non-integer values

    return {"prompt_tokens": prompt_tokens, "output_tokens": output_tokens}


class BenchmarkProcessor:
    """
    Main class for processing benchmark JSON files and generating reports.

    Workflow:
    1. Download consolidated CSVs from S3 (llmd-dashboard + rhaiis-dashboard)
    2. Merge historical CSVs together
    3. Process JSON benchmark file to CSV
    4. Merge with consolidated historical data
    5. Generate HTML report based on config
    """

    def __init__(
        self,
        json_path: str,
        s3_bucket: str,
        s3_key: str,
        accelerator: str,
        model_name: str,
        version: str,
        tp_size: int,
        runtime_args: str,
        compare_versions: Optional[List[str]] = None,
        config_path: Optional[str] = None,
        output_html: Optional[str] = None,
        aws_profile: Optional[str] = None,
        replicas: int = 1,
        prompt_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        turns: Optional[int] = None,
        prefix_tokens: Optional[int] = None,
        prefix_count: Optional[int] = None,
    ):
        """
        Initialize the benchmark processor.

        Args:
            json_path: Path to guidellm JSON benchmark file
            s3_bucket: S3 bucket name containing consolidated CSVs (default: psap-dashboard-data)
            s3_key: S3 key (path) - legacy parameter, not used (downloads both llmd-dashboard and rhaiis-dashboard)
            accelerator: Accelerator type (e.g., H200, MI300X)
            model_name: Model name
            version: Version/framework identifier
            tp_size: Tensor parallelism size
            runtime_args: Runtime configuration arguments
            compare_versions: List of versions to compare against (includes current version)
            config_path: Optional path to YAML config file (auto-generated if not provided)
            output_html: Output HTML report filename (optional)
            aws_profile: AWS profile name (optional)
            replicas: Number of replicas (default: 1)
            prompt_tokens: Prompt tokens for data profile (optional)
            output_tokens: Output tokens for data profile (optional)
            turns: Number of turns for multi-turn benchmarks (optional)
            prefix_tokens: Prefix tokens for prefix caching benchmarks (optional)
            prefix_count: Prefix count for prefix caching benchmarks (optional)
        """
        self.json_path = json_path
        self.config_path = config_path
        self.s3_bucket = s3_bucket
        self.s3_key = s3_key
        self.accelerator = accelerator
        self.model_name = model_name
        self.version = version
        self.tp_size = tp_size
        self.runtime_args = runtime_args
        self.replicas = replicas
        self.output_html = output_html or "benchmark_report.html"

        # Data profile parameters
        self.prompt_tokens = prompt_tokens
        self.output_tokens = output_tokens
        self.turns = turns
        self.prefix_tokens = prefix_tokens
        self.prefix_count = prefix_count

        # Versions to comare (always include the current version)
        if compare_versions is None:
            # XXX: Default versions to compare against
            compare_versions = ["llm-d-0.3", "RHOAI-3.0", "RHAIIS-3.2.3"]

        if version not in compare_versions:
            compare_versions.append(version)

        self.compare_versions = compare_versions

        session = (
            boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
        )
        self.s3_client = session.client("s3")

        self.consolidated_df: Optional[pd.DataFrame] = None
        self.new_data_df: Optional[pd.DataFrame] = None
        self.combined_df: Optional[pd.DataFrame] = None
        self.config: Optional[Dict[str, Any]] = None

    def download_s3_csv(self) -> pd.DataFrame:
        """
        Download consolidated CSV files from S3 and merge them.

        Downloads multiple CSV files from S3 (llmd-dashboard and rhaiis-dashboard)
        and merges them together for comprehensive historical comparison.

        Returns:
            DataFrame containing consolidated benchmark data from all sources
        """
        # Define the two CSV sources
        csv_keys = [
            "main/llmd-dashboard/llmd-dashboard.csv",
            "main/rhaiis-dashboard/consolidated_dashboard.csv",
        ]

        all_dataframes = []

        for key in csv_keys:
            logger.info(f"Downloading s3://{self.s3_bucket}/{key}")

            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb", delete=False, suffix=".csv"
                ) as tmp_file:
                    self.s3_client.download_fileobj(self.s3_bucket, key, tmp_file)
                    tmp_path = tmp_file.name

                df = pd.read_csv(tmp_path)
                os.unlink(tmp_path)

                logger.info(f"Downloaded {len(df)} rows from {key}")
                all_dataframes.append(df)

            except ClientError as e:
                if e.response["Error"]["Code"] == "NoSuchKey":
                    logger.warning(f"S3 file not found: s3://{self.s3_bucket}/{key}")
                    logger.info(f"Skipping {key} (not found)")
                else:
                    logger.error(f"Error downloading {key}: {e}")
                    raise

        # Merge all downloaded CSVs
        if all_dataframes:
            merged_df = pd.concat(all_dataframes, ignore_index=True)
            logger.info(
                f"Merged {len(all_dataframes)} CSV files into {len(merged_df)} total rows"
            )
            return merged_df
        else:
            logger.warning("No CSV files were downloaded from S3")
            logger.info("Starting with empty consolidated data")
            return pd.DataFrame()

    def load_additional_csvs(self, csv_file_paths: List[str]) -> pd.DataFrame:
        """
        Load additional CSV files and merge them with the consolidated data.

        Args:
            csv_file_paths: List of paths to additional CSV files

        Returns:
            DataFrame containing all data merged (S3 CSV + additional CSVs)
        """
        if not csv_file_paths:
            logger.info("No additional CSV files to load")
            return self.consolidated_df

        logger.info(f"Loading {len(csv_file_paths)} additional CSV file(s)")
        additional_dfs = []

        for csv_file in csv_file_paths:
            logger.info(f"Loading additional CSV: {csv_file}")
            try:
                additional_df = pd.read_csv(csv_file)
                logger.info(f"Loaded {len(additional_df)} rows from {csv_file}")
                additional_dfs.append(additional_df)
            except Exception as e:
                logger.error(f"Failed to load {csv_file}: {e}")
                raise ValueError(f"Could not load additional CSV file {csv_file}: {e}")

        # Merge additional CSVs with consolidated CSV
        if additional_dfs:
            logger.info(
                f"Merging {len(additional_dfs)} additional CSV(s) with consolidated data"
            )
            all_csvs = [self.consolidated_df] + additional_dfs
            merged_df = pd.concat(all_csvs, ignore_index=True)
            logger.info(f"After merging: {len(merged_df)} total rows")
            return merged_df

        return self.consolidated_df

    def process_benchmark_section(
        self, benchmark_run: Dict[str, Any], benchmark_index: int
    ) -> Dict[str, Any]:
        """
        Process a single benchmark section and extract performance metrics.

        Args:
            benchmark_run: Benchmark run data from JSON
            benchmark_index: Index of the benchmark run

        Returns:
            Dictionary containing processed benchmark metrics
        """
        full_model_name = f"{self.accelerator}-{self.model_name}-{self.tp_size}"

        uuid = _get_nested(benchmark_run, "config", "run_id") or benchmark_run.get(
            "run_id"
        )

        requests_data = _get_nested(
            benchmark_run, "request_loader", "data"
        ) or _get_nested(benchmark_run, "config", "requests", "data")

        # Use provided data profile parameters if available, otherwise parse from JSON
        if self.prompt_tokens is not None and self.output_tokens is not None:
            config_prompt_tokens = self.prompt_tokens
            config_output_tokens = self.output_tokens
        else:
            token_info = _parse_request_data(requests_data)
            config_prompt_tokens = token_info["prompt_tokens"]
            config_output_tokens = token_info["output_tokens"]

        # Try multiple locations for concurrency/streams info
        # 1. New format: scheduler.strategy.streams (single value for multiturn mode)
        streams_value = _get_nested(benchmark_run, "scheduler", "strategy", "streams")

        if streams_value is not None:
            # Single value (multiturn mode - each file has one benchmark at one concurrency)
            intended_concurrency = streams_value
        else:
            # 2. Old format: config.profile.streams or args.profile.streams (array)
            profile_args = _get_nested(
                benchmark_run, "config", "profile"
            ) or _get_nested(benchmark_run, "args", "profile", default={})
            streams = profile_args.get("streams", [])

            if benchmark_index < len(streams):
                intended_concurrency = streams[benchmark_index]
            else:
                intended_concurrency = streams[0] if streams else None

        metrics = benchmark_run.get("metrics", {})
        successful_metrics = lambda *keys: _get_nested(
            metrics, *keys, "successful", default={}
        )

        measured_concurrency = successful_metrics("request_concurrency").get("mean")
        measured_rps = successful_metrics("requests_per_second").get("mean")
        output_tok_per_sec = successful_metrics("output_tokens_per_second").get(
            "mean", 0
        )
        total_tok_per_sec = successful_metrics("tokens_per_second").get("mean", 0)

        requests_made = _get_nested(
            benchmark_run, "scheduler_metrics", "requests_made", default={}
        )
        successful_reqs = requests_made.get("successful", 0)
        errored_reqs = requests_made.get("errored", 0)

        prompt_tok_metrics = successful_metrics("prompt_token_count")
        output_tok_metrics = successful_metrics("output_token_count")
        ttft_metrics = successful_metrics("time_to_first_token_ms")
        tpot_metrics = successful_metrics("time_per_output_token_ms")
        itl_metrics = successful_metrics("inter_token_latency_ms")
        request_latency_metrics = successful_metrics("request_latency")

        row = {
            "run": full_model_name,
            "accelerator": self.accelerator,
            "model": self.model_name,
            "version": self.version,
            "prompt toks": config_prompt_tokens,
            "output toks": config_output_tokens,
            "TP": self.tp_size,
            "measured concurrency": measured_concurrency,
            "intended concurrency": intended_concurrency,
            "measured rps": measured_rps,
            "output_tok/sec": output_tok_per_sec,
            "total_tok/sec": total_tok_per_sec,
            "prompt_token_count_mean": prompt_tok_metrics.get("mean"),
            "prompt_token_count_p99": _get_nested(
                prompt_tok_metrics, "percentiles", "p99"
            ),
            "output_token_count_mean": output_tok_metrics.get("mean"),
            "output_token_count_p99": _get_nested(
                output_tok_metrics, "percentiles", "p99"
            ),
            "ttft_median": ttft_metrics.get("median"),
            "ttft_p95": _get_nested(ttft_metrics, "percentiles", "p95"),
            "ttft_p1": _get_nested(ttft_metrics, "percentiles", "p01"),
            "ttft_p999": _get_nested(ttft_metrics, "percentiles", "p999"),
            "tpot_median": tpot_metrics.get("median"),
            "tpot_p95": _get_nested(tpot_metrics, "percentiles", "p95"),
            "tpot_p99": _get_nested(tpot_metrics, "percentiles", "p99"),
            "tpot_p999": _get_nested(tpot_metrics, "percentiles", "p999"),
            "tpot_p1": _get_nested(tpot_metrics, "percentiles", "p01"),
            "itl_median": itl_metrics.get("median"),
            "itl_p95": _get_nested(itl_metrics, "percentiles", "p95"),
            "itl_p999": _get_nested(itl_metrics, "percentiles", "p999"),
            "itl_p1": _get_nested(itl_metrics, "percentiles", "p01"),
            "request_latency_median": request_latency_metrics.get("median"),
            "request_latency_min": request_latency_metrics.get("min"),
            "request_latency_max": request_latency_metrics.get("max"),
            "successful_requests": successful_reqs,
            "errored_requests": errored_reqs,
            "uuid": uuid,
            "ttft_mean": ttft_metrics.get("mean"),
            "ttft_p99": _get_nested(ttft_metrics, "percentiles", "p99"),
            "ttft_p90": _get_nested(ttft_metrics, "percentiles", "p90"),
            "itl_mean": itl_metrics.get("mean"),
            "itl_p99": _get_nested(itl_metrics, "percentiles", "p99"),
            "itl_p90": _get_nested(itl_metrics, "percentiles", "p90"),
            "tpot_p90": _get_nested(tpot_metrics, "percentiles", "p90"),
            "request_latency_p90": _get_nested(
                request_latency_metrics, "percentiles", "p90"
            ),
            "request_latency_p95": _get_nested(
                request_latency_metrics, "percentiles", "p95"
            ),
            "request_latency_p99": _get_nested(
                request_latency_metrics, "percentiles", "p99"
            ),
            "runtime_args": self.runtime_args,
            "replicas": self.replicas,
        }

        # Calculate TTFT P99/P50 ratio (latency spread indicator)
        ttft_p99_val = _get_nested(ttft_metrics, "percentiles", "p99")
        ttft_p50_val = ttft_metrics.get("median")
        if ttft_p99_val is not None and ttft_p50_val is not None and ttft_p50_val > 0:
            row["ttft_p99_p50_ratio"] = ttft_p99_val / ttft_p50_val
        else:
            row["ttft_p99_p50_ratio"] = None

        return row

    def parse_guidellm_json(self) -> pd.DataFrame:
        """
        Parse GuideLL JSON benchmark results.

        Returns:
            DataFrame containing processed benchmark data
        """
        logger.info(f"Processing JSON file: {self.json_path}")

        try:
            with open(self.json_path) as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found at {self.json_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Could not decode JSON from {self.json_path}")

        if not data.get("benchmarks"):
            raise ValueError("JSON file does not contain a 'benchmarks' key")

        benchmarks = data["benchmarks"]

        if len(benchmarks) > 1:
            logger.info(f"Processing {len(benchmarks)} separate benchmark sections")
        else:
            logger.info("Processing single benchmark")

        all_run_data = []
        for i, benchmark_run in enumerate(benchmarks):
            row_data = self.process_benchmark_section(benchmark_run, i)
            if row_data:
                all_run_data.append(row_data)

        if not all_run_data:
            raise ValueError("No valid data extracted from benchmark sections")

        df = pd.DataFrame(all_run_data)
        logger.info(f"Extracted {len(df)} rows from JSON")
        return df

    def merge_data(self) -> pd.DataFrame:
        """
        Merge new benchmark data with consolidated CSV.

        Returns:
            Combined DataFrame
        """
        logger.info("Merging new data with consolidated data")

        if self.consolidated_df.empty:
            combined = self.new_data_df
        else:
            combined = pd.concat(
                [self.consolidated_df, self.new_data_df], ignore_index=True
            )

        # Ensure all required columns are present
        fieldnames = [
            "run",
            "accelerator",
            "model",
            "version",
            "prompt toks",
            "output toks",
            "TP",
            "measured concurrency",
            "intended concurrency",
            "measured rps",
            "output_tok/sec",
            "total_tok/sec",
            "prompt_token_count_mean",
            "prompt_token_count_p99",
            "output_token_count_mean",
            "output_token_count_p99",
            "ttft_median",
            "ttft_p95",
            "ttft_p1",
            "ttft_p999",
            "tpot_median",
            "tpot_p95",
            "tpot_p99",
            "tpot_p999",
            "tpot_p1",
            "itl_median",
            "itl_p95",
            "itl_p999",
            "itl_p1",
            "request_latency_median",
            "request_latency_min",
            "request_latency_max",
            "successful_requests",
            "errored_requests",
            "uuid",
            "ttft_mean",
            "ttft_p99",
            "ttft_p90",
            "ttft_p99_p50_ratio",
            "itl_mean",
            "itl_p99",
            "itl_p90",
            "tpot_p90",
            "request_latency_p90",
            "request_latency_p95",
            "request_latency_p99",
            "runtime_args",
            "replicas",
        ]

        for col in fieldnames:
            if col not in combined.columns:
                combined[col] = None

        combined = combined[fieldnames]
        logger.info(f"Combined data has {len(combined)} total rows")

        return combined

    def generate_auto_config(self) -> Dict[str, Any]:
        """
        Auto-generate configuration based on command-line arguments.

        Returns:
            Auto-generated configuration dictionary
        """
        logger.info("Auto-generating configuration from command-line arguments")

        # Use provided data profile parameters if available
        prompt_toks = self.prompt_tokens
        output_toks = self.output_tokens

        # If not provided, try to extract from JSON
        if prompt_toks is None or output_toks is None:
            with open(self.json_path) as f:
                data = json.load(f)

            # Use 1000 as absolute fallback
            if prompt_toks is None:
                prompt_toks = 1000
            if output_toks is None:
                output_toks = 1000

            if data.get("benchmarks"):
                benchmark = data["benchmarks"][0]
                requests_data = _get_nested(
                    benchmark, "request_loader", "data"
                ) or _get_nested(benchmark, "config", "requests", "data")
                token_info = _parse_request_data(requests_data)
                if self.prompt_tokens is None:
                    prompt_toks = token_info["prompt_tokens"] or prompt_toks
                if self.output_tokens is None:
                    output_toks = token_info["output_tokens"] or output_toks

        config = {
            "models": [
                {
                    "model": self.model_name,
                    "prompt_toks": prompt_toks,
                    "output_toks": output_toks,
                }
            ],
            "metric_groups": [
                # Row 1: Throughput
                {
                    "name": "Throughput",
                    "plots": [
                        {
                            "x_metric": "intended concurrency",
                            "y_metric": "output_tok/sec",
                            "x_label": "Concurrency",
                            "y_label": "Output tok/s",
                            "title": "Throughput",
                            "higher_is_better": True,
                        },
                    ],
                },
                # Row 2: TTFT percentiles
                {
                    "name": "TTFT",
                    "plots": [
                        {
                            "x_metric": "intended concurrency",
                            "y_metric": "ttft_p1",
                            "x_label": "Concurrency",
                            "y_label": "P1 (ms)",
                            "title": "TTFT P1",
                            "higher_is_better": False,
                        },
                        {
                            "x_metric": "intended concurrency",
                            "y_metric": "ttft_median",
                            "x_label": "Concurrency",
                            "y_label": "P50 (ms)",
                            "title": "TTFT P50",
                            "higher_is_better": False,
                        },
                        {
                            "x_metric": "intended concurrency",
                            "y_metric": "ttft_p90",
                            "x_label": "Concurrency",
                            "y_label": "P90 (ms)",
                            "title": "TTFT P90",
                            "higher_is_better": False,
                        },
                        {
                            "x_metric": "intended concurrency",
                            "y_metric": "ttft_p99",
                            "x_label": "Concurrency",
                            "y_label": "P99 (ms)",
                            "title": "TTFT P99",
                            "higher_is_better": False,
                        },
                        {
                            "x_metric": "intended concurrency",
                            "y_metric": "ttft_p99_p50_ratio",
                            "x_label": "Concurrency",
                            "y_label": "P99/P50 Ratio",
                            "title": "TTFT Spread",
                            "higher_is_better": False,
                        },
                    ],
                },
                # Row 3: TPOT percentiles
                {
                    "name": "TPOT",
                    "plots": [
                        {
                            "x_metric": "intended concurrency",
                            "y_metric": "tpot_median",
                            "x_label": "Concurrency",
                            "y_label": "P50 (ms)",
                            "title": "TPOT P50",
                            "higher_is_better": False,
                        },
                        {
                            "x_metric": "intended concurrency",
                            "y_metric": "tpot_p90",
                            "x_label": "Concurrency",
                            "y_label": "P90 (ms)",
                            "title": "TPOT P90",
                            "higher_is_better": False,
                        },
                        {
                            "x_metric": "intended concurrency",
                            "y_metric": "tpot_p99",
                            "x_label": "Concurrency",
                            "y_label": "P99 (ms)",
                            "title": "TPOT P99",
                            "higher_is_better": False,
                        },
                    ],
                },
                # Row 4: ITL percentiles
                {
                    "name": "ITL",
                    "plots": [
                        {
                            "x_metric": "intended concurrency",
                            "y_metric": "itl_median",
                            "x_label": "Concurrency",
                            "y_label": "P50 (ms)",
                            "title": "ITL P50",
                            "higher_is_better": False,
                        },
                        {
                            "x_metric": "intended concurrency",
                            "y_metric": "itl_p90",
                            "x_label": "Concurrency",
                            "y_label": "P90 (ms)",
                            "title": "ITL P90",
                            "higher_is_better": False,
                        },
                        {
                            "x_metric": "intended concurrency",
                            "y_metric": "itl_p99",
                            "x_label": "Concurrency",
                            "y_label": "P99 (ms)",
                            "title": "ITL P99",
                            "higher_is_better": False,
                        },
                    ],
                },
                # Row 5: E2E Latency percentiles
                {
                    "name": "E2E Latency",
                    "plots": [
                        {
                            "x_metric": "intended concurrency",
                            "y_metric": "request_latency_median",
                            "x_label": "Concurrency",
                            "y_label": "P50 (s)",
                            "title": "E2E Latency P50",
                            "higher_is_better": False,
                        },
                        {
                            "x_metric": "intended concurrency",
                            "y_metric": "request_latency_p90",
                            "x_label": "Concurrency",
                            "y_label": "P90 (s)",
                            "title": "E2E Latency P90",
                            "higher_is_better": False,
                        },
                        {
                            "x_metric": "intended concurrency",
                            "y_metric": "request_latency_p99",
                            "x_label": "Concurrency",
                            "y_label": "P99 (s)",
                            "title": "E2E Latency P99",
                            "higher_is_better": False,
                        },
                    ],
                },
            ],
            "filters": {
                "accelerators": [self.accelerator],
                "versions": self.compare_versions,
            },
            "styling": {
                "colors": [
                    "#3274A1",
                    "#E1812C",
                    "#7CB57C",
                    "#D95F5F",
                    "#9E67AB",
                    "#B8704F",
                    "#E89CAE",
                    "#7C7C7C",
                    "#1F77B4",
                    "#FF7F0E",
                    "#2CA02C",
                    "#D62728",
                ],
                "markers": [
                    "circle",
                    "square",
                    "diamond",
                    "triangle-up",
                    "triangle-down",
                    "cross",
                    "x",
                    "star",
                    "pentagon",
                    "hexagon",
                ],
            },
        }

        logger.info(f"Auto-generated config for model: {self.model_name}")
        logger.info(f"Comparing versions: {self.compare_versions}")
        logger.info(f"Token configuration: {prompt_toks}in/{output_toks}out")

        return config

    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file or auto-generate.

        Returns:
            Configuration dictionary
        """
        if self.config_path:
            config_file = Path(self.config_path)

            if not config_file.exists():
                raise FileNotFoundError(
                    f"Configuration file not found: {self.config_path}"
                )

            logger.info(f"Loading configuration from: {self.config_path}")

            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            required_sections = ["models", "plots", "filters", "styling"]
            for section in required_sections:
                if section not in config:
                    raise ValueError(
                        f"Missing required configuration section: {section}"
                    )

            logger.info("Configuration loaded successfully")
            return config
        else:
            return self.generate_auto_config()

    def filter_data_for_config(
        self, df: pd.DataFrame, model_config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Filter dataframe for a specific model configuration.

        Args:
            df: Input dataframe
            model_config: Model configuration from config file

        Returns:
            Filtered dataframe
        """
        filtered = df.copy()

        if "model" in model_config:
            filtered = filtered[filtered["model"] == model_config["model"]]

        if "prompt_toks" in model_config:
            filtered = filtered[filtered["prompt toks"] == model_config["prompt_toks"]]
        if "output_toks" in model_config:
            filtered = filtered[filtered["output toks"] == model_config["output_toks"]]

        accelerator_filter = self.config["filters"]["accelerators"]
        if accelerator_filter:
            filtered = filtered[filtered["accelerator"].isin(accelerator_filter)]

        version_filter = self.config["filters"]["versions"]
        if version_filter:
            filtered = filtered[filtered["version"].isin(version_filter)]

        return filtered

    def generate_report(self) -> None:
        """
        Generate HTML report based on configuration with metrics grouped by type.
        Max 3 columns, with throughput spanning full width.
        """
        logger.info("Generating HTML report")

        all_data = self.combined_df

        if all_data.empty:
            logger.error("No data available to plot")
            return

        if "version" not in all_data.columns:
            all_data["version"] = "N/A"
        all_data["version"] = all_data["version"].fillna("N/A").astype(str)

        if "replicas" not in all_data.columns:
            all_data["replicas"] = 1
        all_data["replicas"] = all_data["replicas"].fillna(1).astype(int)

        model_config = self.config["models"][0]
        metric_groups = self.config["metric_groups"]
        colors = self.config["styling"]["colors"]
        markers = self.config["styling"]["markers"]

        # Define grid layout: 6 rows × 3 columns
        # Row 1: Throughput (spans 3 columns)
        # Row 2-3: TTFT (5 plots split: 3 + 2)
        # Row 4: TPOT (3 plots)
        # Row 5: ITL (3 plots)
        # Row 6: E2E Latency (3 plots)

        specs = [
            [{"colspan": 3}, None, None],  # Row 1: Throughput spans 3 columns
            [{}, {}, {}],  # Row 2: TTFT P1, P50, P90
            [{}, {}, None],  # Row 3: TTFT P99, Spread, empty
            [{}, {}, {}],  # Row 4: TPOT P50, P90, P99
            [{}, {}, {}],  # Row 5: ITL P50, P90, P99
            [{}, {}, {}],  # Row 6: E2E Latency P50, P90, P99
        ]

        # Build subplot titles - one per actual subplot (15 total, skipping None specs)
        # Titles are assigned in order to non-None spec cells
        subplot_titles = [
            # Row 1: Throughput (1 subplot)
            "<b>Throughput</b><br><sub>Higher is better</sub>",
            # Row 2: TTFT P1, P50, P90 (3 subplots)
            "<b>TTFT P1</b><br><sub>Lower is better</sub>",
            "<b>TTFT P50</b><br><sub>Lower is better</sub>",
            "<b>TTFT P90</b><br><sub>Lower is better</sub>",
            # Row 3: TTFT P99, Spread (2 subplots, 3rd cell is None)
            "<b>TTFT P99</b><br><sub>Lower is better</sub>",
            "<b>TTFT Spread</b><br><sub>Lower is better</sub>",
            # Row 4: TPOT P50, P90, P99 (3 subplots)
            "<b>TPOT P50</b><br><sub>Lower is better</sub>",
            "<b>TPOT P90</b><br><sub>Lower is better</sub>",
            "<b>TPOT P99</b><br><sub>Lower is better</sub>",
            # Row 5: ITL P50, P90, P99 (3 subplots)
            "<b>ITL P50</b><br><sub>Lower is better</sub>",
            "<b>ITL P90</b><br><sub>Lower is better</sub>",
            "<b>ITL P99</b><br><sub>Lower is better</sub>",
            # Row 6: E2E Latency P50, P90, P99 (3 subplots)
            "<b>E2E Latency P50</b><br><sub>Lower is better</sub>",
            "<b>E2E Latency P90</b><br><sub>Lower is better</sub>",
            "<b>E2E Latency P99</b><br><sub>Lower is better</sub>",
        ]  # Total: 15 titles for 15 actual subplots

        fig = make_subplots(
            rows=6,
            cols=3,
            specs=specs,
            subplot_titles=subplot_titles,
            vertical_spacing=0.10,  # Slightly more spacing for section titles
            horizontal_spacing=0.08,
        )

        # Filter data for the model
        filtered_data = self.filter_data_for_config(all_data, model_config)

        # Collect all configurations for consistent coloring
        all_configs = set()
        if not filtered_data.empty:
            for cfg in filtered_data.groupby(
                ["accelerator", "version", "TP", "replicas"]
            ).groups.keys():
                all_configs.add(cfg)

        all_configs = sorted(list(all_configs))

        config_to_color = {}
        config_to_marker = {}
        for idx, cfg in enumerate(all_configs):
            accelerator, version, tp, replicas = cfg
            label = f"{accelerator} | {version} | TP={int(tp)} | R={int(replicas)}"
            config_to_color[label] = colors[idx % len(colors)]
            config_to_marker[label] = markers[idx % len(markers)]

        legend_entries = set()

        # Define plot positions: (row, col, metric_key)
        plot_positions = [
            # Row 1: Throughput
            (1, 1, "output_tok/sec", "Concurrency", "Output tok/s"),
            # Row 2: TTFT P1, P50, P90
            (2, 1, "ttft_p1", "Concurrency", "P1 (ms)"),
            (2, 2, "ttft_median", "Concurrency", "P50 (ms)"),
            (2, 3, "ttft_p90", "Concurrency", "P90 (ms)"),
            # Row 3: TTFT P99, Spread
            (3, 1, "ttft_p99", "Concurrency", "P99 (ms)"),
            (3, 2, "ttft_p99_p50_ratio", "Concurrency", "P99/P50 Ratio"),
            # Row 4: TPOT P50, P90, P99
            (4, 1, "tpot_median", "Concurrency", "P50 (ms)"),
            (4, 2, "tpot_p90", "Concurrency", "P90 (ms)"),
            (4, 3, "tpot_p99", "Concurrency", "P99 (ms)"),
            # Row 5: ITL P50, P90, P99
            (5, 1, "itl_median", "Concurrency", "P50 (ms)"),
            (5, 2, "itl_p90", "Concurrency", "P90 (ms)"),
            (5, 3, "itl_p99", "Concurrency", "P99 (ms)"),
            # Row 6: E2E Latency P50, P90, P99
            (6, 1, "request_latency_median", "Concurrency", "P50 (s)"),
            (6, 2, "request_latency_p90", "Concurrency", "P90 (s)"),
            (6, 3, "request_latency_p99", "Concurrency", "P99 (s)"),
        ]

        # Plot each metric
        for row, col, metric_key, x_label, y_label in plot_positions:
            if filtered_data.empty:
                continue

            plot_data = filtered_data.sort_values(by="intended concurrency")

            for group_key, group_data in plot_data.groupby(
                ["accelerator", "version", "TP", "replicas"]
            ):
                accelerator, version, tp, replicas = group_key
                label = f"{accelerator} | {version} | TP={int(tp)} | R={int(replicas)}"

                color = config_to_color[label]
                marker = config_to_marker[label]

                show_legend = label not in legend_entries
                if show_legend:
                    legend_entries.add(label)

                fig.add_trace(
                    go.Scatter(
                        x=group_data["intended concurrency"],
                        y=group_data[metric_key],
                        mode="lines+markers",
                        name=label,
                        line=dict(color=color, width=2),
                        marker=dict(
                            size=8, symbol=marker, line=dict(width=1, color="white")
                        ),
                        showlegend=show_legend,
                        legendgroup=label,
                    ),
                    row=row,
                    col=col,
                )

        # Update axis labels for each subplot
        axis_labels = [
            (1, 1, "Concurrency", "Output tok/s"),  # Throughput
            (2, 1, "Concurrency", "P1 (ms)"),  # TTFT P1
            (2, 2, "Concurrency", "P50 (ms)"),  # TTFT P50
            (2, 3, "Concurrency", "P90 (ms)"),  # TTFT P90
            (3, 1, "Concurrency", "P99 (ms)"),  # TTFT P99
            (3, 2, "Concurrency", "P99/P50 Ratio"),  # TTFT Spread
            (4, 1, "Concurrency", "P50 (ms)"),  # TPOT P50
            (4, 2, "Concurrency", "P90 (ms)"),  # TPOT P90
            (4, 3, "Concurrency", "P99 (ms)"),  # TPOT P99
            (5, 1, "Concurrency", "P50 (ms)"),  # ITL P50
            (5, 2, "Concurrency", "P90 (ms)"),  # ITL P90
            (5, 3, "Concurrency", "P99 (ms)"),  # ITL P99
            (6, 1, "Concurrency", "P50 (s)"),  # E2E P50
            (6, 2, "Concurrency", "P90 (s)"),  # E2E P90
            (6, 3, "Concurrency", "P99 (s)"),  # E2E P99
        ]

        for row, col, x_label, y_label in axis_labels:
            fig.update_xaxes(title_text=x_label, row=row, col=col)
            fig.update_yaxes(title_text=y_label, row=row, col=col)

        # Calculate dimensions
        plot_width = 450 * 3  # 3 columns × 450px each = 1350px
        plot_height = 400 * 6  # 6 rows × 400px each = 2400px

        model_short_name = model_config["model"].split("/")[-1]
        versions_str = ", ".join(self.compare_versions)

        # Build data profile subtitle
        data_profile_parts = []
        if self.prompt_tokens is not None:
            data_profile_parts.append(f"prompt_tokens: {self.prompt_tokens}")
        if self.output_tokens is not None:
            data_profile_parts.append(f"output_tokens: {self.output_tokens}")
        if self.turns is not None:
            data_profile_parts.append(f"turns: {self.turns}")
        if self.prefix_tokens is not None:
            data_profile_parts.append(f"prefix_tokens: {self.prefix_tokens}")
        if self.prefix_count is not None:
            data_profile_parts.append(f"prefix_count: {self.prefix_count}")

        data_profile_str = (
            " | ".join(data_profile_parts)
            if data_profile_parts
            else f"Input Tokens: {model_config['prompt_toks']} | Output Tokens: {model_config['output_toks']}"
        )

        fig.update_layout(
            title={
                "text": (
                    f"<b>{model_short_name} Performance Report</b><br>"
                    f"<sub>Comparing versions: {versions_str} | "
                    f"Accelerator: {self.accelerator} | TP: {self.tp_size}</sub><br>"
                    f"<sub>{data_profile_str}</sub>"
                ),
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 18},
                "y": 0.99,
                "yanchor": "top",
            },
            height=plot_height,
            width=plot_width,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font={"family": "Arial, sans-serif", "size": 11},
            margin=dict(t=150, l=80, r=250, b=50),
            legend={
                "title": {"text": "<b>Configuration</b>"},
                "orientation": "v",
                "yanchor": "top",
                "y": 1,
                "xanchor": "left",
                "x": 1.01,
                "bordercolor": "black",
                "borderwidth": 1,
            },
            showlegend=True,
        )

        # Add centered section titles as separators between metric groups
        annotations = [
            # TTFT section title (in gap between row 1 and row 2)
            dict(
                text="<b>— Time To First Token (TTFT) —</b>",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.865,
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=12, color="#666"),
                bgcolor="rgba(255,255,255,0.9)",
            ),
            # TPOT section title (in gap between row 3 and row 4)
            dict(
                text="<b>— Time Per Output Token (TPOT) —</b>",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.500,
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=12, color="#666"),
                bgcolor="rgba(255,255,255,0.9)",
            ),
            # ITL section title (in gap between row 4 and row 5)
            dict(
                text="<b>— Inter-Token Latency (ITL) —</b>",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.315,
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=12, color="#666"),
                bgcolor="rgba(255,255,255,0.9)",
            ),
            # E2E Latency section title (in gap between row 5 and row 6)
            dict(
                text="<b>— End-to-End Request Latency —</b>",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.130,
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=12, color="#666"),
                bgcolor="rgba(255,255,255,0.9)",
            ),
        ]

        fig.update_layout(annotations=list(fig.layout.annotations) + annotations)

        # Ensure all axes have consistent borders and grids (apply to all at once)
        fig.update_xaxes(
            showgrid=True,
            gridwidth=0.5,
            gridcolor="lightgray",
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
        )

        fig.update_yaxes(
            showgrid=True,
            gridwidth=0.5,
            gridcolor="lightgray",
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
        )

        fig.write_html(self.output_html)
        logger.info(f"Report saved to {self.output_html}")
        logger.info(f"Report contains 6 rows × 3 columns with 15 total plots")

    def process(self) -> None:
        """
        Execute the full benchmark processing workflow.

        Steps:
        1. Download consolidated CSV from S3
        2. Parse JSON benchmark file
        3. Merge data
        4. Load report configuration
        5. Generate HTML report
        """
        logger.info("Starting benchmark processing workflow")

        self.consolidated_df = self.download_s3_csv()
        self.new_data_df = self.parse_guidellm_json()
        self.combined_df = self.merge_data()
        self.config = self.load_config()
        self.generate_report()

        logger.info("Benchmark processing complete")
