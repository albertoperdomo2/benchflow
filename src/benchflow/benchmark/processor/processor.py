import ast
import html
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

from ...plotting import REPORT_COLOR_PALETTE
from ...ui import configure_logging

configure_logging("INFO")
logger = logging.getLogger("benchmark_processor")

DATA_PROFILE_PREFERRED_ORDER = [
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


def _get_nested(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely get a nested value from a dictionary."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, default)
    return d


def _coerce_data_profile_value(raw: Any) -> Any:
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


def _parse_request_profile(requests_data: Any) -> Dict[str, Any]:
    """Parse request data into a generic data-profile dictionary."""
    data_str = ""
    parsed_dict: Dict[str, Any] = {}

    if isinstance(requests_data, str):
        # Try to evaluate as a Python literal first (handles "['...']" format)
        try:
            evaluated_data = ast.literal_eval(requests_data)
            if isinstance(evaluated_data, list) and evaluated_data:
                data_str = evaluated_data[0]
            elif isinstance(evaluated_data, str):
                data_str = evaluated_data
            elif isinstance(evaluated_data, dict):
                parsed_dict = {
                    str(key).strip(): _coerce_data_profile_value(value)
                    for key, value in evaluated_data.items()
                    if str(key).strip()
                }
        except (ValueError, SyntaxError):
            try:
                decoded_json = json.loads(requests_data)
            except json.JSONDecodeError:
                data_str = requests_data
            else:
                if isinstance(decoded_json, dict):
                    parsed_dict = {
                        str(key).strip(): _coerce_data_profile_value(value)
                        for key, value in decoded_json.items()
                        if str(key).strip()
                    }
                elif isinstance(decoded_json, list) and decoded_json:
                    data_str = str(decoded_json[0])
                elif isinstance(decoded_json, str):
                    data_str = decoded_json
    elif isinstance(requests_data, list) and requests_data:
        data_str = requests_data[0]
    elif isinstance(requests_data, dict):
        parsed_dict = {
            str(key).strip(): _coerce_data_profile_value(value)
            for key, value in requests_data.items()
            if str(key).strip()
        }

    if parsed_dict:
        return parsed_dict

    parsed: Dict[str, Any] = {}
    if not data_str:
        return parsed

    for part in data_str.split(","):
        if "=" not in part:
            continue
        key, value = part.strip().split("=", 1)
        key = key.strip()
        if not key:
            continue
        parsed[key] = _coerce_data_profile_value(value)

    return parsed


def _parse_request_data(requests_data: Any) -> Dict[str, int]:
    """Parse request data to extract prompt and output tokens."""
    profile = _parse_request_profile(requests_data)
    prompt_tokens = profile.get("prompt_tokens", 0)
    output_tokens = profile.get("output_tokens", 0)
    try:
        prompt_tokens = int(prompt_tokens)
    except (TypeError, ValueError):
        prompt_tokens = 0
    try:
        output_tokens = int(output_tokens)
    except (TypeError, ValueError):
        output_tokens = 0
    return {"prompt_tokens": prompt_tokens, "output_tokens": output_tokens}


def _ordered_data_profile_items(data_profile: Dict[str, Any]) -> list[tuple[str, Any]]:
    preferred = [
        (key, data_profile[key])
        for key in DATA_PROFILE_PREFERRED_ORDER
        if key in data_profile and data_profile[key] is not None
    ]
    remaining = sorted(
        (
            (key, value)
            for key, value in data_profile.items()
            if key not in DATA_PROFILE_PREFERRED_ORDER and value is not None
        ),
        key=lambda item: item[0],
    )
    return [*preferred, *remaining]


def _extract_intended_concurrency(
    benchmark_run: Dict[str, Any], benchmark_index: int
) -> Any:
    """Extract intended concurrency across old and new GuideLLM schemas."""
    streams_value = _get_nested(benchmark_run, "scheduler", "strategy", "streams")
    if streams_value is not None:
        return streams_value

    profile_args = _get_nested(benchmark_run, "config", "profile") or _get_nested(
        benchmark_run, "args", "profile", default={}
    )
    streams = profile_args.get("streams", [])
    if benchmark_index < len(streams):
        return streams[benchmark_index]
    return streams[0] if streams else None


def _extract_ttft_sample(request_stats: Dict[str, Any]) -> float | None:
    value = request_stats.get("time_to_first_token_ms")
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    request_start = _get_nested(request_stats, "info", "timings", "request_start")
    if request_start is None:
        request_start = _get_nested(request_stats, "info", "timings", "resolve_start")
    first_token = request_stats.get("first_token_iteration")
    if first_token is None or request_start is None:
        return None

    try:
        return 1000.0 * (float(first_token) - float(request_start))
    except (TypeError, ValueError):
        return None


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    color = hex_color.lstrip("#")
    if len(color) != 6:
        return hex_color
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def _wrap_header_items(items: list[str], *, max_chars: int = 110) -> list[str]:
    lines: list[str] = []
    current: list[str] = []

    for raw_item in items:
        item = str(raw_item).strip()
        if not item:
            continue
        candidate_parts = [*current, item]
        candidate = " | ".join(candidate_parts)
        if current and len(candidate) > max_chars:
            lines.append(" | ".join(current))
            current = [item]
        else:
            current = candidate_parts

    if current:
        lines.append(" | ".join(current))

    return lines or ["unknown"]


def _format_summary_values(values: list[Any]) -> str:
    normalized = []
    for value in values:
        if isinstance(value, float) and value.is_integer():
            normalized.append(str(int(value)))
        else:
            normalized.append(str(value))
    cleaned = sorted({item.strip() for item in normalized if item and item.strip()})
    if not cleaned:
        return "unknown"
    return cleaned[0] if len(cleaned) == 1 else ", ".join(cleaned)


def _format_table_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))

    if number.is_integer():
        return f"{int(number):,}"
    if abs(number) >= 100:
        return f"{number:,.1f}"
    if abs(number) >= 10:
        return f"{number:,.2f}"
    return f"{number:,.3f}"


def _format_configuration_label(row: pd.Series) -> str:
    accelerator = str(row.get("accelerator") or "unknown")
    version = str(row.get("version") or "unknown")
    tp = _format_table_number(row.get("TP"))
    replicas = _format_table_number(row.get("replicas"))
    return (
        f"{html.escape(accelerator)} | "
        f"{html.escape(version)} | "
        f"TP={html.escape(tp)} | "
        f"R={html.escape(replicas)}"
    )


def _render_comparison_table(filtered_data: pd.DataFrame) -> str:
    if filtered_data.empty:
        return ""

    columns = [
        ("Throughput", "output_tok/sec"),
        ("TTFT P50 (ms)", "ttft_median"),
        ("TTFT P90 (ms)", "ttft_p90"),
        ("TTFT P99 (ms)", "ttft_p99"),
        ("TPOT P50 (ms)", "tpot_median"),
        ("TPOT P90 (ms)", "tpot_p90"),
        ("ITL P50 (ms)", "itl_median"),
        ("E2E P50 (s)", "request_latency_median"),
        ("E2E P90 (s)", "request_latency_p90"),
    ]

    table_df = filtered_data.copy()
    table_df["version"] = table_df["version"].fillna("unknown").astype(str)
    table_df["accelerator"] = table_df["accelerator"].fillna("unknown").astype(str)
    table_df["replicas"] = table_df["replicas"].fillna(1)
    table_df["TP"] = table_df["TP"].fillna(1)
    table_df = table_df.sort_values(
        by=["intended concurrency", "accelerator", "version", "TP", "replicas"]
    )

    header_cells = "".join(
        f"<th>{html.escape(label)}</th>"
        for label, _ in [("Configuration", "")] + columns
    )
    table_rows: list[str] = [f"<tr>{header_cells}</tr>"]
    metric_column_width = (100 - 30) / len(columns)
    colgroup = (
        "<colgroup>"
        "<col style='width: 30%;'>"
        + "".join(f"<col style='width: {metric_column_width:.3f}%;'>" for _ in columns)
        + "</colgroup>"
    )

    for concurrency, group_data in table_df.groupby("intended concurrency", sort=True):
        table_rows.append(
            "<tr class='benchflow-report-table-group'>"
            f"<th colspan='{len(columns) + 1}'>Concurrency {html.escape(_format_table_number(concurrency))}</th>"
            "</tr>"
        )
        for _, row in group_data.iterrows():
            value_cells = "".join(
                f"<td>{html.escape(_format_table_number(row.get(metric_key)))}</td>"
                for _, metric_key in columns
            )
            table_rows.append(
                f"<tr><td>{_format_configuration_label(row)}</td>{value_cells}</tr>"
            )

    return f"""
<section class="benchflow-report-table-section">
  <details class="benchflow-report-table-details">
    <summary>Raw Comparison Table</summary>
    <p>Exact benchmark metrics grouped by intended concurrency.</p>
    <div class="benchflow-report-table-shell">
      <table class="benchflow-report-table">
        {colgroup}
        <thead>
          {table_rows[0]}
        </thead>
        <tbody>
          {"".join(table_rows[1:])}
        </tbody>
      </table>
    </div>
  </details>
</section>
"""


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
        data_profile: Optional[Dict[str, Any]] = None,
        prompt_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        turns: Optional[int] = None,
        prefix_tokens: Optional[int] = None,
        prefix_count: Optional[int] = None,
        repeat_section_legends: bool = False,
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
            data_profile: Full data profile dictionary (optional)
            prompt_tokens: Prompt tokens for data profile (optional)
            output_tokens: Output tokens for data profile (optional)
            turns: Number of turns for multi-turn benchmarks (optional)
            prefix_tokens: Prefix tokens for prefix caching benchmarks (optional)
            prefix_count: Prefix count for prefix caching benchmarks (optional)
            repeat_section_legends: Render repeated side legends per section (optional)
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
        self.repeat_section_legends = repeat_section_legends

        # Data profile parameters
        normalized_profile = {
            str(key): _coerce_data_profile_value(value)
            for key, value in (data_profile or {}).items()
            if value is not None
        }
        if prompt_tokens is not None:
            normalized_profile["prompt_tokens"] = prompt_tokens
        if output_tokens is not None:
            normalized_profile["output_tokens"] = output_tokens
        if turns is not None:
            normalized_profile["turns"] = turns
        if prefix_tokens is not None:
            normalized_profile["prefix_tokens"] = prefix_tokens
        if prefix_count is not None:
            normalized_profile["prefix_count"] = prefix_count

        self.data_profile = normalized_profile
        self.prompt_tokens = normalized_profile.get("prompt_tokens")
        self.output_tokens = normalized_profile.get("output_tokens")
        self.turns = normalized_profile.get("turns")
        self.prefix_tokens = normalized_profile.get("prefix_tokens")
        self.prefix_count = normalized_profile.get("prefix_count")

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
        self.ttft_distribution_df: Optional[pd.DataFrame] = None

    def _resolved_data_profile(self) -> Dict[str, Any]:
        if self.data_profile:
            return dict(self.data_profile)

        try:
            with open(self.json_path) as f:
                payload = json.load(f)
        except Exception:
            return {}

        benchmarks = payload.get("benchmarks") or []
        if not benchmarks:
            return {}

        requests_data = _get_nested(
            benchmarks[0], "request_loader", "data"
        ) or _get_nested(benchmarks[0], "config", "requests", "data")
        return _parse_request_profile(requests_data)

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

        intended_concurrency = _extract_intended_concurrency(
            benchmark_run, benchmark_index
        )

        metrics = benchmark_run.get("metrics", {})

        def successful_metrics(*keys: str) -> Dict[str, Any]:
            return _get_nested(metrics, *keys, "successful", default={})

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

    def parse_ttft_distribution_json(self) -> pd.DataFrame:
        """
        Parse request-level TTFT samples from GuideLLM JSON.

        Returns:
            DataFrame containing TTFT samples grouped by intended concurrency
        """
        logger.info(f"Extracting TTFT distribution samples from {self.json_path}")

        try:
            with open(self.json_path) as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found at {self.json_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Could not decode JSON from {self.json_path}")

        benchmarks = data.get("benchmarks") or []
        if not benchmarks:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for benchmark_index, benchmark_run in enumerate(benchmarks):
            intended_concurrency = _extract_intended_concurrency(
                benchmark_run, benchmark_index
            )
            successful_requests = _get_nested(
                benchmark_run, "requests", "successful", default=[]
            )
            if not isinstance(successful_requests, list):
                continue

            for request_stats in successful_requests:
                if not isinstance(request_stats, dict):
                    continue
                ttft_ms = _extract_ttft_sample(request_stats)
                if ttft_ms is None:
                    continue
                rows.append(
                    {
                        "accelerator": self.accelerator,
                        "model": self.model_name,
                        "version": self.version,
                        "TP": self.tp_size,
                        "replicas": self.replicas,
                        "intended concurrency": intended_concurrency,
                        "ttft_ms": ttft_ms,
                    }
                )

        distribution_df = pd.DataFrame(rows)
        logger.info(f"Extracted {len(distribution_df)} TTFT request samples from JSON")
        return distribution_df

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
                "colors": list(REPORT_COLOR_PALETTE),
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
        colors = self.config["styling"]["colors"]
        markers = self.config["styling"]["markers"]

        has_ttft_distribution = (
            self.ttft_distribution_df is not None
            and not self.ttft_distribution_df.empty
        )

        if has_ttft_distribution:
            specs = [
                [{"colspan": 3}, None, None],
                [{"colspan": 3}, None, None],
                [{"colspan": 3}, None, None],
                [{"colspan": 3}, None, None],
                [{}, {}, {}],
                [{}, {}, None],
                [{}, {}, {}],
                [{}, {}, {}],
                [{}, {}, {}],
            ]
            subplot_titles = [
                "<b>Throughput</b><br><sub>Higher is better</sub>",
                "<b>Throughput Efficiency by GPU</b><br><sub>Higher is better</sub>",
                "<b>Token Throughput per GPU vs End-to-End Latency</b><br><sub>Higher throughput, lower latency</sub>",
                "<b>TTFT Distribution by Concurrency</b><br><sub>Lower is better</sub>",
                "<b>TTFT P1</b><br><sub>Lower is better</sub>",
                "<b>TTFT P50</b><br><sub>Lower is better</sub>",
                "<b>TTFT P90</b><br><sub>Lower is better</sub>",
                "<b>TTFT P99</b><br><sub>Lower is better</sub>",
                "<b>TTFT Spread</b><br><sub>Lower is better</sub>",
                "<b>TPOT P50</b><br><sub>Lower is better</sub>",
                "<b>TPOT P90</b><br><sub>Lower is better</sub>",
                "<b>TPOT P99</b><br><sub>Lower is better</sub>",
                "<b>ITL P50</b><br><sub>Lower is better</sub>",
                "<b>ITL P90</b><br><sub>Lower is better</sub>",
                "<b>ITL P99</b><br><sub>Lower is better</sub>",
                "<b>E2E Latency P50</b><br><sub>Lower is better</sub>",
                "<b>E2E Latency P90</b><br><sub>Lower is better</sub>",
                "<b>E2E Latency P99</b><br><sub>Lower is better</sub>",
            ]
            total_rows = 9
            row_heights = [1.8, 1.5, 1.5, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0]
        else:
            specs = [
                [{"colspan": 3}, None, None],
                [{"colspan": 3}, None, None],
                [{"colspan": 3}, None, None],
                [{}, {}, {}],
                [{}, {}, None],
                [{}, {}, {}],
                [{}, {}, {}],
                [{}, {}, {}],
            ]
            subplot_titles = [
                "<b>Throughput</b><br><sub>Higher is better</sub>",
                "<b>Throughput Efficiency by GPU</b><br><sub>Higher is better</sub>",
                "<b>Token Throughput per GPU vs End-to-End Latency</b><br><sub>Higher throughput, lower latency</sub>",
                "<b>TTFT P1</b><br><sub>Lower is better</sub>",
                "<b>TTFT P50</b><br><sub>Lower is better</sub>",
                "<b>TTFT P90</b><br><sub>Lower is better</sub>",
                "<b>TTFT P99</b><br><sub>Lower is better</sub>",
                "<b>TTFT Spread</b><br><sub>Lower is better</sub>",
                "<b>TPOT P50</b><br><sub>Lower is better</sub>",
                "<b>TPOT P90</b><br><sub>Lower is better</sub>",
                "<b>TPOT P99</b><br><sub>Lower is better</sub>",
                "<b>ITL P50</b><br><sub>Lower is better</sub>",
                "<b>ITL P90</b><br><sub>Lower is better</sub>",
                "<b>ITL P99</b><br><sub>Lower is better</sub>",
                "<b>E2E Latency P50</b><br><sub>Lower is better</sub>",
                "<b>E2E Latency P90</b><br><sub>Lower is better</sub>",
                "<b>E2E Latency P99</b><br><sub>Lower is better</sub>",
            ]
            total_rows = 8
            row_heights = [1.8, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0]

        fig = make_subplots(
            rows=total_rows,
            cols=3,
            specs=specs,
            subplot_titles=subplot_titles,
            vertical_spacing=0.055,
            horizontal_spacing=0.08,
            row_heights=row_heights,
        )

        # Filter data for the model
        filtered_data = self.filter_data_for_config(all_data, model_config).copy()
        if not filtered_data.empty:
            for column in [
                "output_tok/sec",
                "intended concurrency",
                "TP",
                "replicas",
                "request_latency_median",
            ]:
                filtered_data[column] = pd.to_numeric(
                    filtered_data[column], errors="coerce"
                )
            gpu_count = filtered_data["TP"] * filtered_data["replicas"]
            filtered_data["throughput_per_gpu"] = filtered_data["output_tok/sec"] / (
                gpu_count.replace(0, pd.NA)
            )
            filtered_data["throughput_per_concurrency"] = filtered_data[
                "output_tok/sec"
            ] / filtered_data["intended concurrency"].replace(0, pd.NA)

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

        legend_labels = list(config_to_color.keys())
        legend_entries = set()

        if has_ttft_distribution:
            plot_positions = [
                (1, 1, "output_tok/sec", "Concurrency", "Output tok/s"),
                (5, 1, "ttft_p1", "Concurrency", "P1 (ms)"),
                (5, 2, "ttft_median", "Concurrency", "P50 (ms)"),
                (5, 3, "ttft_p90", "Concurrency", "P90 (ms)"),
                (6, 1, "ttft_p99", "Concurrency", "P99 (ms)"),
                (6, 2, "ttft_p99_p50_ratio", "Concurrency", "P99/P50 Ratio"),
                (7, 1, "tpot_median", "Concurrency", "P50 (ms)"),
                (7, 2, "tpot_p90", "Concurrency", "P90 (ms)"),
                (7, 3, "tpot_p99", "Concurrency", "P99 (ms)"),
                (8, 1, "itl_median", "Concurrency", "P50 (ms)"),
                (8, 2, "itl_p90", "Concurrency", "P90 (ms)"),
                (8, 3, "itl_p99", "Concurrency", "P99 (ms)"),
                (9, 1, "request_latency_median", "Concurrency", "P50 (s)"),
                (9, 2, "request_latency_p90", "Concurrency", "P90 (s)"),
                (9, 3, "request_latency_p99", "Concurrency", "P99 (s)"),
            ]
        else:
            plot_positions = [
                (1, 1, "output_tok/sec", "Concurrency", "Output tok/s"),
                (4, 1, "ttft_p1", "Concurrency", "P1 (ms)"),
                (4, 2, "ttft_median", "Concurrency", "P50 (ms)"),
                (4, 3, "ttft_p90", "Concurrency", "P90 (ms)"),
                (5, 1, "ttft_p99", "Concurrency", "P99 (ms)"),
                (5, 2, "ttft_p99_p50_ratio", "Concurrency", "P99/P50 Ratio"),
                (6, 1, "tpot_median", "Concurrency", "P50 (ms)"),
                (6, 2, "tpot_p90", "Concurrency", "P90 (ms)"),
                (6, 3, "tpot_p99", "Concurrency", "P99 (ms)"),
                (7, 1, "itl_median", "Concurrency", "P50 (ms)"),
                (7, 2, "itl_p90", "Concurrency", "P90 (ms)"),
                (7, 3, "itl_p99", "Concurrency", "P99 (ms)"),
                (8, 1, "request_latency_median", "Concurrency", "P50 (s)"),
                (8, 2, "request_latency_p90", "Concurrency", "P90 (s)"),
                (8, 3, "request_latency_p99", "Concurrency", "P99 (s)"),
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

                show_legend = (
                    False
                    if self.repeat_section_legends
                    else label not in legend_entries
                )
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

        if not filtered_data.empty:
            efficiency_data = filtered_data.dropna(
                subset=["throughput_per_gpu", "throughput_per_concurrency"]
            ).sort_values(by="intended concurrency")
            for group_key, group_data in efficiency_data.groupby(
                ["accelerator", "version", "TP", "replicas"]
            ):
                accelerator, version, tp, replicas = group_key
                label = f"{accelerator} | {version} | TP={int(tp)} | R={int(replicas)}"
                color = config_to_color[label]
                marker = config_to_marker[label]
                fig.add_trace(
                    go.Scatter(
                        x=group_data["throughput_per_concurrency"],
                        y=group_data["throughput_per_gpu"],
                        mode="lines+markers",
                        name=label,
                        line=dict(color=color, width=2),
                        marker=dict(
                            size=8, symbol=marker, line=dict(width=1, color="white")
                        ),
                        showlegend=False,
                        legendgroup=label,
                    ),
                    row=2,
                    col=1,
                )

            latency_efficiency_data = filtered_data.dropna(
                subset=["throughput_per_gpu", "request_latency_median"]
            ).sort_values(by="request_latency_median")
            for group_key, group_data in latency_efficiency_data.groupby(
                ["accelerator", "version", "TP", "replicas"]
            ):
                accelerator, version, tp, replicas = group_key
                label = f"{accelerator} | {version} | TP={int(tp)} | R={int(replicas)}"
                color = config_to_color[label]
                marker = config_to_marker[label]
                fig.add_trace(
                    go.Scatter(
                        x=group_data["request_latency_median"],
                        y=group_data["throughput_per_gpu"],
                        mode="lines+markers",
                        name=label,
                        line=dict(color=color, width=2),
                        marker=dict(
                            size=8, symbol=marker, line=dict(width=1, color="white")
                        ),
                        showlegend=False,
                        legendgroup=label,
                    ),
                    row=3,
                    col=1,
                )

        if has_ttft_distribution:
            distribution_data = self.ttft_distribution_df.copy()
            numeric_concurrency = pd.to_numeric(
                distribution_data["intended concurrency"], errors="coerce"
            )
            distribution_data["intended concurrency"] = [
                (
                    str(int(value))
                    if pd.notna(value) and float(value).is_integer()
                    else str(value)
                )
                if pd.notna(value)
                else str(raw_value)
                for value, raw_value in zip(
                    numeric_concurrency,
                    distribution_data["intended concurrency"],
                    strict=False,
                )
            ]
            distribution_categories = [
                str(int(value)) if float(value).is_integer() else str(value)
                for value in sorted(numeric_concurrency.dropna().unique().tolist())
            ]
            for group_key, group_data in distribution_data.groupby(
                ["accelerator", "version", "TP", "replicas"]
            ):
                accelerator, version, tp, replicas = group_key
                label = f"{accelerator} | {version} | TP={int(tp)} | R={int(replicas)}"
                color = config_to_color.get(label, colors[0])
                show_legend = (
                    False
                    if self.repeat_section_legends
                    else label not in legend_entries
                )
                if show_legend:
                    legend_entries.add(label)
                fig.add_trace(
                    go.Box(
                        x=group_data["intended concurrency"],
                        y=group_data["ttft_ms"],
                        name=label,
                        legendgroup=label,
                        showlegend=show_legend,
                        line=dict(color=color, width=1.5),
                        fillcolor=_hex_to_rgba(color, 0.12),
                        marker=dict(
                            color=_hex_to_rgba(color, 0.45),
                            size=4,
                            line=dict(width=0),
                        ),
                        boxpoints="outliers",
                        pointpos=0,
                        jitter=0.28,
                        whiskerwidth=0.6,
                    ),
                    row=4,
                    col=1,
                )
            fig.update_xaxes(
                categoryorder="array",
                categoryarray=distribution_categories,
                row=4,
                col=1,
            )

        if has_ttft_distribution:
            axis_labels = [
                (1, 1, "Concurrency", "Output tok/s"),
                (
                    2,
                    1,
                    "Interactivity (toks/s/concurrency)",
                    "Output throughput per GPU (toks/s/gpu)",
                ),
                (
                    3,
                    1,
                    "End-to-end latency (s)",
                    "Token throughput per GPU (toks/s/gpu)",
                ),
                (4, 1, "Concurrency", "TTFT (ms)"),
                (5, 1, "Concurrency", "P1 (ms)"),
                (5, 2, "Concurrency", "P50 (ms)"),
                (5, 3, "Concurrency", "P90 (ms)"),
                (6, 1, "Concurrency", "P99 (ms)"),
                (6, 2, "Concurrency", "P99/P50 Ratio"),
                (7, 1, "Concurrency", "P50 (ms)"),
                (7, 2, "Concurrency", "P90 (ms)"),
                (7, 3, "Concurrency", "P99 (ms)"),
                (8, 1, "Concurrency", "P50 (ms)"),
                (8, 2, "Concurrency", "P90 (ms)"),
                (8, 3, "Concurrency", "P99 (ms)"),
                (9, 1, "Concurrency", "P50 (s)"),
                (9, 2, "Concurrency", "P90 (s)"),
                (9, 3, "Concurrency", "P99 (s)"),
            ]
        else:
            axis_labels = [
                (1, 1, "Concurrency", "Output tok/s"),
                (
                    2,
                    1,
                    "Interactivity (toks/s/concurrency)",
                    "Output throughput per GPU (toks/s/gpu)",
                ),
                (
                    3,
                    1,
                    "End-to-end latency (s)",
                    "Token throughput per GPU (toks/s/gpu)",
                ),
                (4, 1, "Concurrency", "P1 (ms)"),
                (4, 2, "Concurrency", "P50 (ms)"),
                (4, 3, "Concurrency", "P90 (ms)"),
                (5, 1, "Concurrency", "P99 (ms)"),
                (5, 2, "Concurrency", "P99/P50 Ratio"),
                (6, 1, "Concurrency", "P50 (ms)"),
                (6, 2, "Concurrency", "P90 (ms)"),
                (6, 3, "Concurrency", "P99 (ms)"),
                (7, 1, "Concurrency", "P50 (ms)"),
                (7, 2, "Concurrency", "P90 (ms)"),
                (7, 3, "Concurrency", "P99 (ms)"),
                (8, 1, "Concurrency", "P50 (s)"),
                (8, 2, "Concurrency", "P90 (s)"),
                (8, 3, "Concurrency", "P99 (s)"),
            ]

        for row, col, x_label, y_label in axis_labels:
            fig.update_xaxes(title_text=x_label, row=row, col=col)
            fig.update_yaxes(title_text=y_label, row=row, col=col)

        def _row_top(row: int) -> float:
            domain = fig.get_subplot(row, 1).yaxis.domain
            return domain[1]

        def _build_repeated_legend(legend_name: str, y_top: float) -> None:
            for legend_index, label in enumerate(legend_labels):
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="lines+markers",
                        name=label,
                        legend=legend_name,
                        legendgroup=label,
                        legendrank=legend_index,
                        showlegend=True,
                        hoverinfo="skip",
                        line=dict(color=config_to_color[label], width=2),
                        marker=dict(
                            size=8,
                            symbol=config_to_marker[label],
                            line=dict(width=1, color="white"),
                            color=config_to_color[label],
                        ),
                    )
                )

            fig.update_layout(
                **{
                    legend_name: dict(
                        title={"text": "<b>Configuration</b>"},
                        orientation="v",
                        yanchor="top",
                        y=y_top,
                        xanchor="left",
                        x=1.01,
                        bordercolor="#cbd5e1",
                        borderwidth=1,
                        bgcolor="rgba(255,255,255,0.94)",
                        font={"size": 10},
                        groupclick="togglegroup",
                    )
                }
            )

        legend_positions = [
            ("legend", _row_top(1)),
            ("legend2", _row_top(2)),
            ("legend3", _row_top(3)),
        ]
        if has_ttft_distribution:
            legend_positions.extend(
                [
                    ("legend4", _row_top(4)),
                    ("legend5", _row_top(5)),
                    ("legend6", _row_top(7)),
                    ("legend7", _row_top(8)),
                    ("legend8", _row_top(9)),
                ]
            )
        else:
            legend_positions.extend(
                [
                    ("legend4", _row_top(4)),
                    ("legend5", _row_top(6)),
                    ("legend6", _row_top(7)),
                    ("legend7", _row_top(8)),
                ]
            )

        if self.repeat_section_legends and legend_labels:
            for legend_name, y_top in legend_positions:
                _build_repeated_legend(legend_name, y_top)

        # Calculate dimensions
        plot_width = 1540 if self.repeat_section_legends else 1440
        plot_height = int(
            (420 * sum(row_heights)) + (140 if has_ttft_distribution else 0)
        )

        model_short_name = model_config["model"].split("/")[-1]

        if not filtered_data.empty:
            header_versions = (
                filtered_data["version"]
                .fillna("unknown")
                .astype(str)
                .drop_duplicates()
                .tolist()
            )
            header_accelerator = _format_summary_values(
                filtered_data["accelerator"].fillna("unknown").astype(str).tolist()
            )
            header_tp = _format_summary_values(filtered_data["TP"].tolist())
            header_replicas = _format_summary_values(filtered_data["replicas"].tolist())
        else:
            header_versions = list(self.compare_versions)
            header_accelerator = str(self.accelerator)
            header_tp = str(self.tp_size)
            header_replicas = str(self.replicas)

        version_lines = _wrap_header_items(header_versions)

        # Build data profile subtitle
        data_profile_parts = [
            f"{key}: {value}"
            for key, value in _ordered_data_profile_items(self._resolved_data_profile())
        ]

        data_profile_str = (
            " | ".join(data_profile_parts)
            if data_profile_parts
            else f"Input Tokens: {model_config['prompt_toks']} | Output Tokens: {model_config['output_toks']}"
        )

        title_lines = [
            f"<b>{html.escape(model_short_name)} Performance Report</b>",
            (
                "<span style='font-size:13px;'>"
                f"Accelerator: {html.escape(header_accelerator)} | "
                f"TP: {html.escape(header_tp)} | "
                f"R: {html.escape(header_replicas)}"
                "</span>"
            ),
        ]
        for index, line in enumerate(version_lines):
            label = "<b>Configurations:</b> " if index == 0 else ""
            title_lines.append(
                f"<span style='font-size:12px;'>{label}{html.escape(line)}</span>"
            )
        title_lines.append(
            f"<span style='font-size:12px;'>{html.escape(data_profile_str)}</span>"
        )
        top_margin = 90 + (len(title_lines) * 24)

        fig.update_layout(
            title={
                "text": "<br>".join(title_lines),
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
            margin=dict(
                t=top_margin,
                l=80,
                r=360 if self.repeat_section_legends else 250,
                b=50,
            ),
            showlegend=bool(legend_labels),
            boxmode="group",
        )

        def _section_divider_y(upper_row: int, lower_row: int) -> float:
            upper_domain = fig.get_subplot(upper_row, 1).yaxis.domain
            lower_domain = fig.get_subplot(lower_row, 1).yaxis.domain
            return (upper_domain[0] + lower_domain[1]) / 2

        def _section_divider(text: str, upper_row: int, lower_row: int) -> dict:
            return dict(
                text=text,
                xref="paper",
                yref="paper",
                x=0.5,
                y=_section_divider_y(upper_row, lower_row),
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(size=12, color="#666"),
                bgcolor="rgba(255,255,255,0.9)",
            )

        # Add centered section titles as separators between metric groups.
        if has_ttft_distribution:
            annotations = [
                _section_divider("<b>— Time To First Token (TTFT) —</b>", 3, 4),
                _section_divider("<b>— Time Per Output Token (TPOT) —</b>", 6, 7),
                _section_divider("<b>— Inter-Token Latency (ITL) —</b>", 7, 8),
                _section_divider("<b>— End-to-End Request Latency —</b>", 8, 9),
            ]
        else:
            annotations = [
                _section_divider("<b>— Time To First Token (TTFT) —</b>", 3, 4),
                _section_divider("<b>— Time Per Output Token (TPOT) —</b>", 5, 6),
                _section_divider("<b>— Inter-Token Latency (ITL) —</b>", 6, 7),
                _section_divider("<b>— End-to-End Request Latency —</b>", 7, 8),
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

        chart_html = fig.to_html(
            full_html=False,
            include_plotlyjs=True,
            config={"responsive": False},
        )
        comparison_table_html = _render_comparison_table(filtered_data)
        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(model_short_name)} Performance Report</title>
  <style>
    body {{
      margin: 0;
      background: white;
      color: #1f2a44;
      font-family: Arial, sans-serif;
    }}
    .benchflow-report-shell {{
      width: {plot_width}px;
      margin: 0 auto;
    }}
    .benchflow-report-shell > div:first-child {{
      width: {plot_width}px;
      margin: 0;
    }}
    .benchflow-report-shell .plotly-graph-div {{
      margin: 0;
    }}
    .benchflow-report-table-section {{
      width: 100%;
      margin: 24px 0 48px;
    }}
    .benchflow-report-table-details {{
      background: white;
    }}
    .benchflow-report-table-details summary {{
      padding: 10px 12px;
      font-size: 20px;
      font-weight: 700;
      cursor: pointer;
      list-style-position: inside;
    }}
    .benchflow-report-table-details[open] summary {{
      border-bottom: none;
    }}
    .benchflow-report-table-section p {{
      margin: 12px 0 14px;
      font-size: 12px;
      text-align: center;
    }}
    .benchflow-report-table-shell {{
      overflow-x: auto;
      padding: 0 10px 10px;
    }}
    .benchflow-report-table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 11px;
      background: white;
    }}
    .benchflow-report-table th,
    .benchflow-report-table td {{
      border: 1px solid #1f2a44;
      padding: 6px 7px;
      vertical-align: top;
    }}
    .benchflow-report-table thead th {{
      background: #f4f6f8;
      font-weight: 700;
      text-align: left;
    }}
    .benchflow-report-table th:first-child,
    .benchflow-report-table td:first-child {{
      word-break: break-word;
      white-space: normal;
    }}
    .benchflow-report-table tbody td {{
      text-align: right;
    }}
    .benchflow-report-table tbody td:first-child,
    .benchflow-report-table tbody th {{
      text-align: left;
    }}
    .benchflow-report-table-group th {{
      background: #eef2f5;
      font-size: 12px;
      font-weight: 700;
    }}
    .benchflow-report-table tbody tr:nth-child(even) td {{
      background: #fafbfc;
    }}
  </style>
</head>
<body>
<div class="benchflow-report-shell">
{chart_html}
{comparison_table_html}
</div>
</body>
</html>
"""
        Path(self.output_html).write_text(full_html, encoding="utf-8")
        logger.info(f"Report saved to {self.output_html}")
        total_plots = 18 if has_ttft_distribution else 17
        logger.info(
            f"Report contains {total_rows} rows × 3 columns with {total_plots} total plots"
        )

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
        self.ttft_distribution_df = self.parse_ttft_distribution_json()
        self.combined_df = self.merge_data()
        self.config = self.load_config()
        self.generate_report()

        logger.info("Benchmark processing complete")
