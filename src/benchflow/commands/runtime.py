from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import click

from ..artifacts import collect_artifacts
from ..benchmark import (
    BenchmarkRunFailed,
    benchmark_version_from_plan,
    generate_report,
    run_benchmark,
)
from ..cleanup import cleanup_llmd
from ..cluster import create_manifest, follow_pipelinerun, get_current_namespace
from ..deploy import deploy_llmd
from ..execution import load_run_plan_from_sources, require_platform
from ..install import BootstrapOptions, run_bootstrap
from ..loaders import load_run_plan_data
from ..metrics import collect_metrics
from ..mlflow_upload import upload_to_mlflow
from ..model import download_model
from ..models import ValidationError
from ..repository import clone_repo
from ..renderers.tekton import render_pipelinerun
from ..tasking import assert_task_status, write_stage_results
from ..ui import detail, emit, step, success, warning
from ..waiting import wait_for_endpoint
from .shared import (
    dump_yaml,
    experiment_input_options,
    invoke_handler,
    load_runtime_plan,
    parse_mapping,
    parse_version_overrides,
    repo_root_from,
    runtime_plan_source_options,
)


def cmd_bootstrap(args: argparse.Namespace) -> int:
    repo_root = repo_root_from(args)
    return run_bootstrap(
        repo_root,
        BootstrapOptions(
            namespace=args.namespace or "benchflow",
            install_tekton=not args.skip_tekton_install,
            install_grafana=not args.skip_grafana_install,
            tekton_channel=args.tekton_channel or "latest",
            models_storage_class=args.models_storage_class,
            models_storage_size=args.models_size or "250Gi",
            models_storage_access_mode=args.models_access_mode or "ReadWriteOnce",
            results_storage_class=args.results_storage_class,
            results_storage_size=args.results_size or "20Gi",
        ),
    )


def cmd_watch(args: argparse.Namespace) -> int:
    namespace = args.namespace or get_current_namespace()
    return 0 if follow_pipelinerun(namespace, args.pipelinerun_name) else 1


def cmd_repo_clone(args: argparse.Namespace) -> int:
    commit = clone_repo(
        url=args.url,
        revision=args.revision,
        output_dir=Path(args.output_dir).resolve(),
        delete_existing=not args.no_delete_existing,
    )
    if args.commit_output:
        Path(args.commit_output).resolve().write_text(commit, encoding="utf-8")
    if args.url_output:
        Path(args.url_output).resolve().write_text(args.url, encoding="utf-8")
    print(commit)
    return 0


def cmd_model_download(args: argparse.Namespace) -> int:
    plan = load_runtime_plan(args)
    require_platform(plan, "llm-d")
    target_dir = download_model(
        plan,
        models_storage_path=Path(args.models_storage_path).resolve(),
        skip_if_exists=not args.no_skip_if_exists,
    )
    print(target_dir)
    return 0


def cmd_deploy_llmd(args: argparse.Namespace) -> int:
    plan = load_runtime_plan(args)
    require_platform(plan, "llm-d")
    checkout_dir = deploy_llmd(
        plan,
        workspace_dir=Path(args.workspace_dir).resolve()
        if args.workspace_dir
        else None,
        manifests_dir=Path(args.manifests_dir).resolve()
        if args.manifests_dir
        else None,
        pipeline_run_name=args.pipeline_run_name or "",
        skip_if_exists=not args.no_skip_if_exists,
        verify=not args.no_verify,
        verify_timeout_seconds=args.verify_timeout_seconds,
    )
    print(checkout_dir)
    return 0


def cmd_undeploy_llmd(args: argparse.Namespace) -> int:
    plan = load_runtime_plan(args)
    require_platform(plan, "llm-d")
    cleanup_llmd(
        plan,
        wait_for_deletion=not args.no_wait,
        timeout_seconds=args.timeout_seconds,
        skip_if_not_exists=not args.no_skip_if_not_exists,
    )
    print(plan.deployment.release_name)
    return 0


def cmd_wait_endpoint(args: argparse.Namespace) -> int:
    target_url = args.target_url
    endpoint_path = args.endpoint_path
    if not target_url:
        plan = load_runtime_plan(args)
        require_platform(plan, "llm-d")
        target_url = plan.deployment.target.base_url
        endpoint_path = args.endpoint_path or plan.deployment.target.path
    wait_for_endpoint(
        target_url=target_url,
        endpoint_path=endpoint_path or "/v1/models",
        timeout_seconds=args.timeout_seconds,
        retry_interval_seconds=args.retry_interval,
        verify_tls=args.verify_tls,
    )
    print("ready")
    return 0


def cmd_benchmark_run(args: argparse.Namespace) -> int:
    plan = load_runtime_plan(args)
    require_platform(plan, "llm-d")
    if plan.benchmark.tool != "guidellm":
        raise ValidationError(
            f"unsupported benchmark tool: {plan.benchmark.tool}; only guidellm is implemented"
        )
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    benchmark_target = args.target_url or plan.deployment.target.base_url
    step(f"Running {plan.benchmark.tool} benchmark against {benchmark_target}")
    detail(
        "Rates: "
        + ",".join(str(rate) for rate in plan.benchmark.rates)
        + f", rate type: {plan.benchmark.rate_type}, backend: {plan.benchmark.backend_type}"
    )
    detail(f"Benchmark data: {plan.benchmark.data}")
    detail(
        f"MLflow: {'disabled' if args.no_mlflow else 'enabled'}, "
        f"output dir: {str(output_dir) if output_dir is not None else 'not requested'}"
    )
    previous_pipeline_run_name = os.environ.get("PIPELINE_RUN_NAME")
    run_id = ""
    start_time = ""
    end_time = ""
    try:
        if args.pipeline_run_name:
            os.environ["PIPELINE_RUN_NAME"] = args.pipeline_run_name
        try:
            run_id, start_time, end_time = run_benchmark(
                plan=plan,
                target=args.target_url,
                output_dir=output_dir,
                mlflow_tracking_uri=args.mlflow_tracking_uri,
                enable_mlflow=not args.no_mlflow,
                extra_tags=parse_mapping(args.tag, "--tag"),
            )
        except BenchmarkRunFailed as exc:
            run_id = exc.run_id
            start_time = exc.start_time
            end_time = exc.end_time
            if run_id:
                detail(
                    "Preserving MLflow run information after benchmark failure: "
                    f"run_id={run_id}, start={start_time}, end={end_time}"
                )
            if args.mlflow_run_id_output and run_id:
                Path(args.mlflow_run_id_output).resolve().write_text(
                    run_id, encoding="utf-8"
                )
            if args.benchmark_start_time_output and start_time:
                Path(args.benchmark_start_time_output).resolve().write_text(
                    start_time, encoding="utf-8"
                )
            if args.benchmark_end_time_output and end_time:
                Path(args.benchmark_end_time_output).resolve().write_text(
                    end_time, encoding="utf-8"
                )
            raise
    finally:
        if args.pipeline_run_name:
            if previous_pipeline_run_name is None:
                os.environ.pop("PIPELINE_RUN_NAME", None)
            else:
                os.environ["PIPELINE_RUN_NAME"] = previous_pipeline_run_name
    if args.mlflow_run_id_output:
        Path(args.mlflow_run_id_output).resolve().write_text(run_id, encoding="utf-8")
    if args.benchmark_start_time_output:
        Path(args.benchmark_start_time_output).resolve().write_text(
            start_time, encoding="utf-8"
        )
    if args.benchmark_end_time_output:
        Path(args.benchmark_end_time_output).resolve().write_text(
            end_time, encoding="utf-8"
        )

    success(
        f"Benchmark finished. Start: {start_time}, end: {end_time}, "
        f"MLflow run: {run_id or 'not created'}"
    )
    if run_id:
        print(run_id)
    elif output_dir is not None:
        print(output_dir)
    else:
        print("completed")
    return 0


def cmd_benchmark_report(args: argparse.Namespace) -> int:
    plan = None
    if (
        args.run_plan_file
        or args.run_plan_json
        or args.experiment
        or args.model
        or args.deployment_profile
    ):
        plan = load_runtime_plan(args)
        require_platform(plan, "llm-d")

    json_path = Path(args.json_path).resolve() if args.json_path else None
    model = args.model_name or (plan.model.name if plan is not None else None)
    version = args.version or (
        benchmark_version_from_plan(plan) if plan is not None else None
    )
    tp_size = (
        args.tp
        if args.tp is not None
        else (plan.deployment.runtime.tensor_parallelism if plan is not None else 1)
    )
    runtime_args = args.runtime_args or (
        " ".join(plan.deployment.runtime.vllm_args) if plan is not None else ""
    )
    replicas = (
        args.replicas
        if args.replicas is not None
        else (plan.deployment.runtime.replicas if plan is not None else 1)
    )

    report_path = generate_report(
        json_path=json_path,
        model=model,
        accelerator=args.accelerator,
        version=version,
        tp_size=tp_size,
        runtime_args=runtime_args,
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
        replicas=replicas,
        mlflow_run_ids=[
            item.strip() for item in args.mlflow_run_ids.split(",") if item.strip()
        ]
        if args.mlflow_run_ids
        else None,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        versions=[item.strip() for item in args.versions.split(",") if item.strip()]
        if args.versions
        else None,
        version_overrides=parse_version_overrides(args.version_override),
        additional_csv_files=args.additional_csv or None,
    )
    print(report_path)
    return 0


def cmd_artifacts_collect(args: argparse.Namespace) -> int:
    plan = load_runtime_plan(args)
    require_platform(plan, "llm-d")
    artifact_dir = collect_artifacts(
        plan,
        artifacts_dir=Path(args.artifacts_dir).resolve(),
        pipeline_run_name=args.pipeline_run_name or "",
    )
    print(artifact_dir)
    return 0


def cmd_metrics_collect(args: argparse.Namespace) -> int:
    plan = load_runtime_plan(args)
    require_platform(plan, "llm-d")
    metrics_dir = collect_metrics(
        plan,
        benchmark_start_time=args.benchmark_start_time,
        benchmark_end_time=args.benchmark_end_time,
        artifacts_dir=Path(args.artifacts_dir).resolve(),
    )
    print(metrics_dir)
    return 0


def cmd_mlflow_upload(args: argparse.Namespace) -> int:
    plan = load_runtime_plan(args)
    require_platform(plan, "llm-d")
    upload_to_mlflow(
        plan,
        mlflow_run_id=args.mlflow_run_id,
        benchmark_start_time=args.benchmark_start_time,
        benchmark_end_time=args.benchmark_end_time,
        artifacts_dir=Path(args.artifacts_dir).resolve(),
        grafana_url=args.grafana_url or "",
    )
    print(args.mlflow_run_id)
    return 0


def cmd_task_resolve_run_plan(args: argparse.Namespace) -> int:
    plan = load_run_plan_from_sources(run_plan_json=args.run_plan_json)
    require_platform(plan, "llm-d")
    if plan.benchmark.tool != "guidellm":
        raise ValidationError(
            f"unsupported benchmark tool: {plan.benchmark.tool}; only guidellm is implemented"
        )
    step(
        f"Resolved RunPlan for {plan.metadata.name} "
        f"({plan.deployment.platform}/{plan.deployment.mode})"
    )
    detail(
        "Stages: "
        f"download={plan.stages.download}, deploy={plan.stages.deploy}, "
        f"benchmark={plan.stages.benchmark}, collect={plan.stages.collect}, "
        f"cleanup={plan.stages.cleanup}"
    )
    emit(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    write_stage_results(
        plan,
        stage_download_path=Path(args.stage_download_path).resolve(),
        stage_deploy_path=Path(args.stage_deploy_path).resolve(),
        stage_benchmark_path=Path(args.stage_benchmark_path).resolve(),
        stage_collect_path=Path(args.stage_collect_path).resolve(),
        stage_cleanup_path=Path(args.stage_cleanup_path).resolve(),
    )
    success("RunPlan resolved and stage outputs written")
    print("resolved")
    return 0


def cmd_task_assert_status(args: argparse.Namespace) -> int:
    allowed_statuses = list(args.allowed_status)
    if args.allowed_statuses_text:
        allowed_statuses.extend(
            [
                item.strip()
                for item in args.allowed_statuses_text.replace("\n", ",").split(",")
                if item.strip()
            ]
        )
    assert_task_status(args.task_name, args.task_status, allowed_statuses)
    print(args.task_status)
    return 0


def cmd_task_run_experiment_matrix(args: argparse.Namespace) -> int:
    try:
        raw_run_plans = json.loads(args.run_plans_json)
    except json.JSONDecodeError as exc:
        raise ValidationError("invalid JSON passed to --run-plans-json") from exc

    if not isinstance(raw_run_plans, list) or not raw_run_plans:
        raise ValidationError("--run-plans-json must contain a non-empty JSON array")

    plans = [load_run_plan_data(item) for item in raw_run_plans]
    namespaces = {plan.deployment.namespace for plan in plans}
    if len(namespaces) != 1:
        raise ValidationError(
            "matrix execution requires all child RunPlans to target the same namespace"
        )

    step(
        f"Running matrix supervisor for {plans[0].metadata.name} "
        f"with {len(plans)} profile combination(s)"
    )
    failures: list[str] = []
    total = len(plans)

    for index, plan in enumerate(plans, start=1):
        descriptor = (
            f"deployment={plan.profiles.deployment}, "
            f"benchmark={plan.profiles.benchmark}, "
            f"metrics={plan.profiles.metrics}"
        )
        step(f"[{index}/{total}] Submitting child PipelineRun")
        detail(descriptor)
        manifest = render_pipelinerun(plan, pipeline_name=args.child_pipeline_name)
        submitted = create_manifest(dump_yaml(manifest), plan.deployment.namespace)
        name = str(submitted.get("metadata", {}).get("name") or "")
        if not name:
            raise ValidationError("child PipelineRun submission returned no name")
        detail(f"Created PipelineRun {name} in namespace {plan.deployment.namespace}")
        succeeded = follow_pipelinerun(plan.deployment.namespace, name)
        if succeeded:
            success(f"[{index}/{total}] {name} succeeded")
            continue
        warning(f"[{index}/{total}] {name} failed")
        failures.append(name)

    if failures:
        raise ValidationError(
            f"{len(failures)} matrix child run(s) failed: {', '.join(failures)}"
        )

    success(f"Matrix supervisor completed {total} child PipelineRun(s)")
    print("completed")
    return 0


@click.command(
    "bootstrap",
    help=(
        "Bootstrap BenchFlow into a namespace and install NFD, the NVIDIA GPU "
        "Operator, Tekton, Grafana, RBAC, and PVCs."
    ),
    short_help="Bootstrap BenchFlow and cluster dependencies",
)
@click.option(
    "--repo-root",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="BenchFlow repository root. Defaults to the current checkout.",
)
@click.option(
    "--namespace",
    help="Target namespace. Defaults to benchflow.",
)
@click.option(
    "--skip-tekton-install",
    is_flag=True,
    help="Do not install Tekton if it is missing.",
)
@click.option(
    "--skip-grafana-install",
    is_flag=True,
    help="Do not install Grafana in the dedicated Grafana namespace.",
)
@click.option(
    "--tekton-channel",
    default="latest",
    show_default=True,
    help="OpenShift Pipelines operator channel.",
)
@click.option(
    "--models-storage-class",
    help="StorageClass for the shared model cache PVC.",
)
@click.option(
    "--models-size",
    help="Requested size for the model cache PVC.",
)
@click.option(
    "--models-access-mode",
    help="Access mode for the model cache PVC.",
)
@click.option(
    "--results-storage-class",
    help="StorageClass for the benchmark results PVC.",
)
@click.option(
    "--results-size",
    help="Requested size for the benchmark results PVC.",
)
def bootstrap_command(**kwargs: object) -> int:
    return invoke_handler(cmd_bootstrap, **kwargs)


@click.group(
    "repo",
    help="Repository helpers used by deployment workflows.",
    short_help="Repository utilities",
)
def repo_group() -> None:
    pass


@repo_group.command(
    "clone",
    help="Clone a repository into a local directory for deployment work.",
    short_help="Clone a source repository",
)
@click.option("--url", required=True, help="Repository URL.")
@click.option(
    "--revision",
    default="main",
    show_default=True,
    help="Revision, branch, or tag to check out.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory where the repository should be cloned.",
)
@click.option(
    "--no-delete-existing",
    is_flag=True,
    help="Keep an existing output directory instead of replacing it.",
)
@click.option(
    "--commit-output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write the resolved commit SHA to this file.",
)
@click.option(
    "--url-output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write the source repository URL to this file.",
)
def repo_clone(**kwargs: object) -> int:
    return invoke_handler(cmd_repo_clone, **kwargs)


@click.group(
    "model",
    help="Manage cached models used by BenchFlow runs.",
    short_help="Model cache operations",
)
def model_group() -> None:
    pass


@model_group.command(
    "download",
    help="Download a model referenced by the RunPlan into the shared model cache.",
    short_help="Download a model into the cache",
)
@runtime_plan_source_options
@click.option(
    "--models-storage-path",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Mounted path of the model cache PVC.",
)
@click.option(
    "--no-skip-if-exists",
    is_flag=True,
    help="Force a download even when the model is already cached.",
)
def model_download(**kwargs: object) -> int:
    return invoke_handler(cmd_model_download, **kwargs)


@click.group(
    "deploy",
    help="Deploy a scenario from a resolved BenchFlow RunPlan.",
    short_help="Deployment operations",
)
def deploy_group() -> None:
    pass


@deploy_group.command(
    "llm-d",
    help="Deploy an llm-d scenario from a resolved RunPlan.",
    short_help="Deploy an llm-d scenario",
)
@runtime_plan_source_options
@click.option(
    "--workspace-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory where llm-d will be cloned and patched for deployment.",
)
@click.option(
    "--manifests-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory where rendered manifests should be written.",
)
@click.option(
    "--pipeline-run-name",
    help="Owning PipelineRun name for log and label propagation.",
)
@click.option(
    "--no-skip-if-exists",
    is_flag=True,
    help="Redeploy even if the target release already exists.",
)
@click.option(
    "--no-verify",
    is_flag=True,
    help="Skip post-deploy readiness verification.",
)
@click.option(
    "--verify-timeout-seconds",
    type=int,
    default=900,
    show_default=True,
    help="Maximum time to wait for deployment verification.",
)
def deploy_llmd_command(**kwargs: object) -> int:
    return invoke_handler(cmd_deploy_llmd, **kwargs)


@click.group(
    "undeploy",
    help="Remove a deployment created from a BenchFlow RunPlan.",
    short_help="Cleanup deployment resources",
)
def undeploy_group() -> None:
    pass


@undeploy_group.command(
    "llm-d",
    help="Tear down an llm-d deployment from a resolved RunPlan.",
    short_help="Remove an llm-d deployment",
)
@runtime_plan_source_options
@click.option(
    "--no-wait",
    is_flag=True,
    help="Do not wait for deployment resources to disappear.",
)
@click.option(
    "--timeout-seconds",
    type=int,
    default=600,
    show_default=True,
    help="Maximum time to wait for cleanup.",
)
@click.option(
    "--no-skip-if-not-exists",
    is_flag=True,
    help="Fail instead of skipping when the release is already absent.",
)
def undeploy_llmd_command(**kwargs: object) -> int:
    return invoke_handler(cmd_undeploy_llmd, **kwargs)


@click.group(
    "wait",
    help="Wait for endpoints or other runtime conditions to become ready.",
    short_help="Wait for runtime conditions",
)
def wait_group() -> None:
    pass


@wait_group.command(
    "endpoint",
    help="Poll the resolved target endpoint until it becomes reachable.",
    short_help="Wait for the deployment endpoint",
)
@runtime_plan_source_options
@click.option("--target-url", help="Override the target base URL to probe.")
@click.option("--endpoint-path", help="Endpoint path to probe.")
@click.option(
    "--timeout-seconds",
    type=int,
    default=3600,
    show_default=True,
    help="Maximum time to wait for readiness.",
)
@click.option(
    "--retry-interval",
    type=int,
    default=10,
    show_default=True,
    help="Seconds between readiness probes.",
)
@click.option(
    "--verify-tls",
    is_flag=True,
    help="Verify TLS certificates when probing the endpoint.",
)
def wait_endpoint_command(**kwargs: object) -> int:
    return invoke_handler(cmd_wait_endpoint, **kwargs)


@click.group(
    "benchmark",
    help="Run benchmarks and generate reports for BenchFlow scenarios.",
    short_help="Benchmark execution and reporting",
)
def benchmark_group() -> None:
    pass


@benchmark_group.command(
    "run",
    help="Execute the configured benchmark and optionally upload results to MLflow.",
    short_help="Run a GuideLLM benchmark",
)
@runtime_plan_source_options
@click.option("--target-url", help="Override the benchmark target URL.")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory where benchmark outputs should be written.",
)
@click.option(
    "--mlflow-tracking-uri",
    help="Override the MLflow tracking URI for the benchmark run.",
)
@click.option(
    "--no-mlflow",
    is_flag=True,
    help="Disable MLflow tracking for this benchmark run.",
)
@click.option(
    "--tag",
    multiple=True,
    metavar="KEY=VALUE",
    help="Extra MLflow tags for the benchmark run.",
)
@click.option(
    "--pipeline-run-name",
    help="Owning PipelineRun name for MLflow tagging.",
)
@click.option(
    "--mlflow-run-id-output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write the MLflow run ID to this file.",
)
@click.option(
    "--benchmark-start-time-output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write the benchmark start timestamp to this file.",
)
@click.option(
    "--benchmark-end-time-output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write the benchmark end timestamp to this file.",
)
def benchmark_run_command(**kwargs: object) -> int:
    return invoke_handler(cmd_benchmark_run, **kwargs)


@benchmark_group.command(
    "report",
    help="Generate a report from benchmark JSON and optional MLflow metadata.",
    short_help="Generate a benchmark report",
)
@runtime_plan_source_options
@click.option(
    "--json-path",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to the benchmark JSON input.",
)
@click.option("--model-name", help="Model name to display in the report.")
@click.option("--accelerator", help="Accelerator label to include in the report.")
@click.option("--version", help="Version string for the report.")
@click.option("--tp", type=int, help="Tensor parallelism to show in the report.")
@click.option("--runtime-args", help="Runtime arguments string to show in the report.")
@click.option("--replicas", type=int, help="Replica count to show in the report.")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory where the report should be written.",
)
@click.option(
    "--mlflow-run-ids",
    help="Comma-separated MLflow run IDs to include in the report.",
)
@click.option(
    "--mlflow-tracking-uri",
    help="MLflow tracking URI for report enrichment.",
)
@click.option(
    "--versions",
    help="Comma-separated version list for multi-run report generation.",
)
@click.option(
    "--version-override",
    multiple=True,
    metavar="OLD=NEW",
    help="Version label override. Repeat to set multiple mappings.",
)
@click.option(
    "--additional-csv",
    multiple=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Additional CSV inputs to include in the report.",
)
def benchmark_report_command(**kwargs: object) -> int:
    return invoke_handler(cmd_benchmark_report, **kwargs)


@click.group(
    "artifacts",
    help="Collect benchmark and run artifacts into a local directory.",
    short_help="Artifact collection",
)
def artifacts_group() -> None:
    pass


@artifacts_group.command(
    "collect",
    help="Collect the artifacts BenchFlow expects from a finished run.",
    short_help="Collect run artifacts",
)
@runtime_plan_source_options
@click.option(
    "--artifacts-dir",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory where collected artifacts should be written.",
)
@click.option(
    "--pipeline-run-name",
    help="PipelineRun name to collect artifacts from.",
)
def artifacts_collect_command(**kwargs: object) -> int:
    return invoke_handler(cmd_artifacts_collect, **kwargs)


@click.group(
    "metrics",
    help="Collect Prometheus metrics for BenchFlow benchmark windows.",
    short_help="Metrics collection",
)
def metrics_group() -> None:
    pass


@metrics_group.command(
    "collect",
    help="Collect benchmark metrics from Prometheus or Thanos for a resolved RunPlan.",
    short_help="Collect Prometheus metrics",
)
@runtime_plan_source_options
@click.option(
    "--benchmark-start-time",
    required=True,
    help="Benchmark start time in ISO-8601 format.",
)
@click.option(
    "--benchmark-end-time",
    required=True,
    help="Benchmark end time in ISO-8601 format.",
)
@click.option(
    "--artifacts-dir",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory where collected metrics should be written.",
)
def metrics_collect_command(**kwargs: object) -> int:
    return invoke_handler(cmd_metrics_collect, **kwargs)


@click.group(
    "mlflow",
    help="Upload and organize BenchFlow benchmark outputs in MLflow.",
    short_help="MLflow integration",
)
def mlflow_group() -> None:
    pass


@mlflow_group.command(
    "upload",
    help="Upload benchmark artifacts, metrics, and metadata to MLflow.",
    short_help="Upload artifacts and metrics to MLflow",
)
@runtime_plan_source_options
@click.option("--mlflow-run-id", required=True, help="MLflow run ID to update.")
@click.option(
    "--benchmark-start-time",
    required=True,
    help="Benchmark start time in ISO-8601 format.",
)
@click.option(
    "--benchmark-end-time",
    required=True,
    help="Benchmark end time in ISO-8601 format.",
)
@click.option(
    "--artifacts-dir",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory that contains the artifacts to upload.",
)
@click.option(
    "--grafana-url",
    help="Grafana URL tag to attach to the MLflow run.",
)
def mlflow_upload_command(**kwargs: object) -> int:
    return invoke_handler(cmd_mlflow_upload, **kwargs)


@click.group(
    "task",
    help="Internal commands invoked by Tekton tasks inside the BenchFlow image.",
    short_help="Internal Tekton task entrypoints",
    hidden=True,
)
def task_group() -> None:
    pass


@task_group.command(
    "resolve-run-plan",
    help="Internal command used by Tekton tasks to resolve a RunPlan into stage files.",
    short_help="Resolve a RunPlan into stage files",
)
@click.option("--run-plan-json", required=True, help="Inline RunPlan JSON payload.")
@click.option(
    "--stage-download-path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="File that receives the download stage flag.",
)
@click.option(
    "--stage-deploy-path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="File that receives the deploy stage flag.",
)
@click.option(
    "--stage-benchmark-path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="File that receives the benchmark stage flag.",
)
@click.option(
    "--stage-collect-path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="File that receives the collect stage flag.",
)
@click.option(
    "--stage-cleanup-path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="File that receives the cleanup stage flag.",
)
def task_resolve_run_plan_command(**kwargs: object) -> int:
    return invoke_handler(cmd_task_resolve_run_plan, **kwargs)


@task_group.command(
    "assert-status",
    help="Internal command used by Tekton tasks to assert task status transitions.",
    short_help="Assert a task status transition",
)
@click.option("--task-name", required=True, help="Task name to report in the error.")
@click.option("--task-status", required=True, help="Observed task status.")
@click.option(
    "--allowed-status",
    multiple=True,
    default=("Succeeded", "None"),
    show_default=True,
    help="Allowed status value. Repeat to allow more than one.",
)
@click.option(
    "--allowed-statuses-text",
    default="",
    help="Comma-separated or newline-separated allowed statuses.",
)
def task_assert_status_command(**kwargs: object) -> int:
    return invoke_handler(cmd_task_assert_status, **kwargs)


@task_group.command(
    "run-experiment-matrix",
    help=(
        "Internal command used by Tekton to run a cartesian product of resolved "
        "RunPlans sequentially in the cluster."
    ),
    short_help="Run a matrix of child PipelineRuns",
)
@click.option(
    "--run-plans-json",
    required=True,
    help="JSON array of resolved RunPlan objects.",
)
@click.option(
    "--child-pipeline-name",
    default="benchflow-e2e",
    show_default=True,
    help="Pipeline name to use for the child PipelineRuns.",
)
def task_run_experiment_matrix_command(**kwargs: object) -> int:
    return invoke_handler(cmd_task_run_experiment_matrix, **kwargs)


@click.command(
    "watch",
    help="Stream a PipelineRun and report its terminal state.",
    short_help="Follow a PipelineRun until completion",
)
@click.argument("pipelinerun_name")
@click.option(
    "--namespace",
    help="Namespace that contains the PipelineRun. Defaults to the current oc project.",
)
def watch_command(**kwargs: object) -> int:
    return invoke_handler(cmd_watch, **kwargs)
