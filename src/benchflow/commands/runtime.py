from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..artifacts import collect_artifacts
from ..benchmark import generate_report, run_benchmark
from ..cleanup import cleanup_llmd
from ..cluster import follow_pipelinerun, get_current_namespace
from ..deploy import deploy_llmd
from ..execution import load_run_plan_from_sources, require_platform
from ..install import InstallOptions, run_install
from ..metrics import collect_metrics
from ..mlflow_upload import upload_to_mlflow
from ..model import download_model
from ..models import ValidationError
from ..repository import clone_repo
from ..tasking import assert_task_status, write_stage_results
from ..ui import detail, emit, step, success
from ..waiting import wait_for_endpoint
from .shared import (
    add_experiment_input_arguments,
    add_parser,
    add_run_plan_arguments,
    load_runtime_plan,
    parse_mapping,
    parse_version_overrides,
    repo_root_from,
)


DEPLOY_SUBCOMMANDS = ("llm-d",)
UNDEPLOY_SUBCOMMANDS = ("llm-d",)
WAIT_SUBCOMMANDS = ("endpoint",)
BENCHMARK_SUBCOMMANDS = ("run", "report")
REPO_SUBCOMMANDS = ("clone",)
MODEL_SUBCOMMANDS = ("download",)
ARTIFACTS_SUBCOMMANDS = ("collect",)
METRICS_SUBCOMMANDS = ("collect",)
MLFLOW_SUBCOMMANDS = ("upload",)
TASK_SUBCOMMANDS = ("resolve-run-plan", "assert-status")


def register_runtime_plan_source_arguments(parser: argparse.ArgumentParser) -> None:
    add_run_plan_arguments(parser)
    add_experiment_input_arguments(parser)


def cmd_install(args: argparse.Namespace) -> int:
    repo_root = repo_root_from(args)
    return run_install(
        repo_root,
        InstallOptions(
            namespace=args.namespace or "benchflow",
            install_tekton=not args.skip_tekton_install,
            install_grafana=not args.skip_grafana_install,
            tekton_channel=args.tekton_channel or "latest",
            grafana_channel=args.grafana_channel or "v5",
            grafana_starting_csv="grafana-operator.v5.21.2",
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
        source_dir=Path(args.source_dir).resolve() if args.source_dir else None,
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
    try:
        if args.pipeline_run_name:
            os.environ["PIPELINE_RUN_NAME"] = args.pipeline_run_name
        run_id, start_time, end_time = run_benchmark(
            plan=plan,
            target=args.target_url,
            output_dir=output_dir,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            enable_mlflow=not args.no_mlflow,
            extra_tags=parse_mapping(args.tag, "--tag"),
        )
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
        f"{plan.deployment.platform}-{plan.deployment.mode}"
        if plan is not None
        else None
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


def configure_install_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repo-root",
        help="BenchFlow repository root. Defaults to the current checkout.",
    )
    parser.add_argument(
        "--namespace",
        help="Target namespace. Defaults to benchflow.",
    )
    parser.add_argument(
        "--skip-tekton-install",
        action="store_true",
        help="Do not install Tekton if it is missing.",
    )
    parser.add_argument(
        "--skip-grafana-install",
        action="store_true",
        help="Do not install the Grafana operator if it is missing.",
    )
    parser.add_argument(
        "--tekton-channel",
        help="OpenShift Pipelines operator channel.",
    )
    parser.add_argument(
        "--grafana-channel",
        help="Grafana operator channel.",
    )
    parser.add_argument(
        "--models-storage-class",
        help="StorageClass for the shared model cache PVC.",
    )
    parser.add_argument(
        "--models-size",
        help="Requested size for the model cache PVC.",
    )
    parser.add_argument(
        "--models-access-mode",
        help="Access mode for the model cache PVC.",
    )
    parser.add_argument(
        "--results-storage-class",
        help="StorageClass for the benchmark results PVC.",
    )
    parser.add_argument(
        "--results-size",
        help="Requested size for the benchmark results PVC.",
    )
    parser.set_defaults(func=cmd_install)


def register_repo_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    clone = add_parser(
        subparsers,
        "clone",
        help_text="Clone a source repository",
        description="Clone a repository into a local directory for deployment work.",
    )
    clone.add_argument("--url", required=True)
    clone.add_argument("--revision", default="main")
    clone.add_argument("--output-dir", required=True)
    clone.add_argument("--no-delete-existing", action="store_true")
    clone.add_argument("--commit-output")
    clone.add_argument("--url-output")
    clone.set_defaults(func=cmd_repo_clone)


def register_model_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    download = add_parser(
        subparsers,
        "download",
        help_text="Download a model into the model cache",
        description="Download a model referenced by the RunPlan into the shared model cache.",
    )
    register_runtime_plan_source_arguments(download)
    download.add_argument("--models-storage-path", required=True)
    download.add_argument("--no-skip-if-exists", action="store_true")
    download.set_defaults(func=cmd_model_download)


def register_deploy_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    llmd = add_parser(
        subparsers,
        "llm-d",
        help_text="Deploy an llm-d scenario",
        description="Deploy an llm-d scenario from a resolved RunPlan.",
    )
    register_runtime_plan_source_arguments(llmd)
    llmd.add_argument("--source-dir")
    llmd.add_argument("--manifests-dir")
    llmd.add_argument("--pipeline-run-name")
    llmd.add_argument("--no-skip-if-exists", action="store_true")
    llmd.add_argument("--no-verify", action="store_true")
    llmd.add_argument("--verify-timeout-seconds", type=int, default=900)
    llmd.set_defaults(func=cmd_deploy_llmd)


def register_undeploy_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    llmd = add_parser(
        subparsers,
        "llm-d",
        help_text="Remove an llm-d deployment",
        description="Tear down an llm-d deployment from a resolved RunPlan.",
    )
    register_runtime_plan_source_arguments(llmd)
    llmd.add_argument("--no-wait", action="store_true")
    llmd.add_argument("--timeout-seconds", type=int, default=600)
    llmd.add_argument("--no-skip-if-not-exists", action="store_true")
    llmd.set_defaults(func=cmd_undeploy_llmd)


def register_wait_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    endpoint = add_parser(
        subparsers,
        "endpoint",
        help_text="Wait for the deployment endpoint to become ready",
        description="Poll the resolved target endpoint until it becomes reachable.",
    )
    register_runtime_plan_source_arguments(endpoint)
    endpoint.add_argument("--target-url")
    endpoint.add_argument("--endpoint-path")
    endpoint.add_argument("--timeout-seconds", type=int, default=3600)
    endpoint.add_argument("--retry-interval", type=int, default=10)
    endpoint.add_argument("--verify-tls", action="store_true")
    endpoint.set_defaults(func=cmd_wait_endpoint)


def register_benchmark_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    run = add_parser(
        subparsers,
        "run",
        help_text="Run a GuideLLM benchmark",
        description="Execute the configured benchmark and optionally upload results to MLflow.",
    )
    register_runtime_plan_source_arguments(run)
    run.add_argument("--target-url")
    run.add_argument("--output-dir")
    run.add_argument("--mlflow-tracking-uri")
    run.add_argument("--no-mlflow", action="store_true")
    run.add_argument("--tag", action="append", default=[], metavar="KEY=VALUE")
    run.add_argument("--pipeline-run-name")
    run.add_argument("--mlflow-run-id-output")
    run.add_argument("--benchmark-start-time-output")
    run.add_argument("--benchmark-end-time-output")
    run.set_defaults(func=cmd_benchmark_run)

    report = add_parser(
        subparsers,
        "report",
        help_text="Generate a benchmark report",
        description="Generate a report from benchmark JSON and optional MLflow metadata.",
    )
    register_runtime_plan_source_arguments(report)
    report.add_argument("--json-path")
    report.add_argument("--model-name")
    report.add_argument("--accelerator")
    report.add_argument("--version")
    report.add_argument("--tp", type=int)
    report.add_argument("--runtime-args")
    report.add_argument("--replicas", type=int)
    report.add_argument("--output-dir")
    report.add_argument("--mlflow-run-ids")
    report.add_argument("--mlflow-tracking-uri")
    report.add_argument("--versions")
    report.add_argument(
        "--version-override", action="append", default=[], metavar="OLD=NEW"
    )
    report.add_argument("--additional-csv", action="append", default=[])
    report.set_defaults(func=cmd_benchmark_report)


def register_metrics_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    collect = add_parser(
        subparsers,
        "collect",
        help_text="Collect Prometheus metrics for a benchmark window",
        description="Collect benchmark metrics from Prometheus or Thanos for a resolved RunPlan.",
    )
    register_runtime_plan_source_arguments(collect)
    collect.add_argument("--benchmark-start-time", required=True)
    collect.add_argument("--benchmark-end-time", required=True)
    collect.add_argument("--artifacts-dir", required=True)
    collect.set_defaults(func=cmd_metrics_collect)


def register_artifacts_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    collect = add_parser(
        subparsers,
        "collect",
        help_text="Collect run artifacts into a local directory",
        description="Collect the artifacts BenchFlow expects from a finished run.",
    )
    register_runtime_plan_source_arguments(collect)
    collect.add_argument("--artifacts-dir", required=True)
    collect.add_argument("--pipeline-run-name")
    collect.set_defaults(func=cmd_artifacts_collect)


def register_mlflow_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    upload = add_parser(
        subparsers,
        "upload",
        help_text="Upload artifacts and metrics to MLflow",
        description="Upload benchmark artifacts, metrics, and metadata to MLflow.",
    )
    register_runtime_plan_source_arguments(upload)
    upload.add_argument("--mlflow-run-id", required=True)
    upload.add_argument("--benchmark-start-time", required=True)
    upload.add_argument("--benchmark-end-time", required=True)
    upload.add_argument("--artifacts-dir", required=True)
    upload.add_argument("--grafana-url")
    upload.set_defaults(func=cmd_mlflow_upload)


def register_task_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    resolve = add_parser(
        subparsers,
        "resolve-run-plan",
        help_text="Internal task entrypoint for RunPlan stage outputs",
        description="Internal command used by Tekton tasks to resolve a RunPlan into stage files.",
    )
    resolve.add_argument("--run-plan-json", required=True)
    resolve.add_argument("--stage-download-path", required=True)
    resolve.add_argument("--stage-deploy-path", required=True)
    resolve.add_argument("--stage-benchmark-path", required=True)
    resolve.add_argument("--stage-collect-path", required=True)
    resolve.add_argument("--stage-cleanup-path", required=True)
    resolve.set_defaults(func=cmd_task_resolve_run_plan)

    assert_status_cmd = add_parser(
        subparsers,
        "assert-status",
        help_text="Internal task entrypoint for status assertions",
        description="Internal command used by Tekton tasks to assert task status transitions.",
    )
    assert_status_cmd.add_argument("--task-name", required=True)
    assert_status_cmd.add_argument("--task-status", required=True)
    assert_status_cmd.add_argument(
        "--allowed-status", action="append", default=["Succeeded", "None"]
    )
    assert_status_cmd.add_argument("--allowed-statuses-text", default="")
    assert_status_cmd.set_defaults(func=cmd_task_assert_status)


def configure_watch_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("pipelinerun_name", help="PipelineRun name to follow.")
    parser.add_argument(
        "--namespace",
        help="Namespace that contains the PipelineRun. Defaults to the current oc project.",
    )
    parser.set_defaults(func=cmd_watch)


def runtime_command_options(
    command: str, experiment_options: tuple[str, ...], subcommand: str | None = None
) -> tuple[str, ...]:
    if command == "repo" and subcommand == "clone":
        return (
            "--url",
            "--revision",
            "--output-dir",
            "--no-delete-existing",
            "--commit-output",
            "--url-output",
        )
    if command == "model" and subcommand == "download":
        return (
            *experiment_options,
            "--run-plan-file",
            "--run-plan-json",
            "--models-storage-path",
            "--no-skip-if-exists",
        )
    if command == "deploy" and subcommand == "llm-d":
        return (
            *experiment_options,
            "--run-plan-file",
            "--run-plan-json",
            "--source-dir",
            "--manifests-dir",
            "--pipeline-run-name",
            "--no-skip-if-exists",
            "--no-verify",
            "--verify-timeout-seconds",
        )
    if command == "undeploy" and subcommand == "llm-d":
        return (
            *experiment_options,
            "--run-plan-file",
            "--run-plan-json",
            "--no-wait",
            "--timeout-seconds",
            "--no-skip-if-not-exists",
        )
    if command == "wait" and subcommand == "endpoint":
        return (
            *experiment_options,
            "--run-plan-file",
            "--run-plan-json",
            "--target-url",
            "--endpoint-path",
            "--timeout-seconds",
            "--retry-interval",
            "--verify-tls",
        )
    if command == "benchmark":
        if subcommand == "run":
            return (
                *experiment_options,
                "--run-plan-file",
                "--run-plan-json",
                "--target-url",
                "--output-dir",
                "--mlflow-tracking-uri",
                "--no-mlflow",
                "--tag",
                "--pipeline-run-name",
                "--mlflow-run-id-output",
                "--benchmark-start-time-output",
                "--benchmark-end-time-output",
            )
        if subcommand == "report":
            return (
                *experiment_options,
                "--run-plan-file",
                "--run-plan-json",
                "--json-path",
                "--model-name",
                "--accelerator",
                "--version",
                "--tp",
                "--runtime-args",
                "--replicas",
                "--output-dir",
                "--mlflow-run-ids",
                "--mlflow-tracking-uri",
                "--versions",
                "--version-override",
                "--additional-csv",
            )
    if command == "artifacts" and subcommand == "collect":
        return (
            *experiment_options,
            "--run-plan-file",
            "--run-plan-json",
            "--artifacts-dir",
            "--pipeline-run-name",
        )
    if command == "metrics" and subcommand == "collect":
        return (
            *experiment_options,
            "--run-plan-file",
            "--run-plan-json",
            "--benchmark-start-time",
            "--benchmark-end-time",
            "--artifacts-dir",
        )
    if command == "mlflow" and subcommand == "upload":
        return (
            *experiment_options,
            "--run-plan-file",
            "--run-plan-json",
            "--mlflow-run-id",
            "--benchmark-start-time",
            "--benchmark-end-time",
            "--artifacts-dir",
            "--grafana-url",
        )
    if command == "task":
        if subcommand == "resolve-run-plan":
            return (
                "--run-plan-json",
                "--stage-download-path",
                "--stage-deploy-path",
                "--stage-benchmark-path",
                "--stage-collect-path",
                "--stage-cleanup-path",
            )
        if subcommand == "assert-status":
            return (
                "--task-name",
                "--task-status",
                "--allowed-status",
                "--allowed-statuses-text",
            )
    if command == "install":
        return (
            "--repo-root",
            "--namespace",
            "--skip-tekton-install",
            "--skip-grafana-install",
            "--tekton-channel",
            "--grafana-channel",
            "--models-storage-class",
            "--models-size",
            "--models-access-mode",
            "--results-storage-class",
            "--results-size",
        )
    if command == "watch":
        return ("--namespace",)
    if command == "completion":
        return ()
    return ()
