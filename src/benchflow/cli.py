from __future__ import annotations

import argparse
import sys

from .cluster import CommandError
from .commands.completion import cmd_complete_internal, cmd_completion
from .commands.experiment import register_experiment_commands
from .commands.profiles import register_profiles_commands
from .commands.runtime import (
    configure_install_parser,
    configure_watch_parser,
    register_artifacts_commands,
    register_benchmark_commands,
    register_deploy_commands,
    register_metrics_commands,
    register_mlflow_commands,
    register_model_commands,
    register_repo_commands,
    register_task_commands,
    register_undeploy_commands,
    register_wait_commands,
)
from .commands.shared import add_parser
from .models import ValidationError
from .ui import HelpFormatter, error as ui_error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bflow",
        description=(
            "BenchFlow installs the cluster prerequisites and runs LLM inference "
            "benchmarks from experiment files or direct CLI flags."
        ),
        epilog=(
            "Examples:\n"
            "  bflow install\n"
            "  bflow experiment run experiments/examples/qwen3-06b-llm-d-smoke.yaml\n"
            "  bflow experiment run --model Qwen/Qwen3-0.6B "
            "--deployment-profile llm-d-inference-scheduling "
            "--benchmark-profile concurrent-1k-1k"
        ),
        formatter_class=HelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        title="Commands",
        metavar="COMMAND",
    )

    install = add_parser(
        subparsers,
        "install",
        help_text="Install BenchFlow and cluster dependencies",
        description="Install BenchFlow into a namespace and bootstrap Tekton, Grafana, RBAC, and PVCs.",
    )
    configure_install_parser(install)

    experiment = add_parser(
        subparsers,
        "experiment",
        help_text="Work with experiment definitions",
        description="Validate, resolve, render, run, or clean up BenchFlow experiments.",
    )
    experiment_subparsers = experiment.add_subparsers(
        dest="experiment_command",
        required=True,
        title="Experiment commands",
        metavar="EXPERIMENT_COMMAND",
    )
    register_experiment_commands(experiment_subparsers)

    repo = add_parser(
        subparsers,
        "repo",
        help_text="Repository utilities",
        description="Repository helpers used by deployment workflows.",
    )
    repo_subparsers = repo.add_subparsers(
        dest="repo_command",
        required=True,
        title="Repo commands",
        metavar="REPO_COMMAND",
    )
    register_repo_commands(repo_subparsers)

    model = add_parser(
        subparsers,
        "model",
        help_text="Model cache operations",
        description="Manage cached models used by BenchFlow runs.",
    )
    model_subparsers = model.add_subparsers(
        dest="model_command",
        required=True,
        title="Model commands",
        metavar="MODEL_COMMAND",
    )
    register_model_commands(model_subparsers)

    deploy = add_parser(
        subparsers,
        "deploy",
        help_text="Deployment operations",
        description="Deploy a scenario from a resolved BenchFlow RunPlan.",
    )
    deploy_subparsers = deploy.add_subparsers(
        dest="deploy_command",
        required=True,
        title="Deployment commands",
        metavar="DEPLOYMENT_COMMAND",
    )
    register_deploy_commands(deploy_subparsers)

    undeploy = add_parser(
        subparsers,
        "undeploy",
        help_text="Cleanup deployment resources",
        description="Remove a deployment created from a BenchFlow RunPlan.",
    )
    undeploy_subparsers = undeploy.add_subparsers(
        dest="undeploy_command",
        required=True,
        title="Cleanup commands",
        metavar="CLEANUP_COMMAND",
    )
    register_undeploy_commands(undeploy_subparsers)

    wait = add_parser(
        subparsers,
        "wait",
        help_text="Wait for runtime conditions",
        description="Wait for endpoints or other runtime conditions to become ready.",
    )
    wait_subparsers = wait.add_subparsers(
        dest="wait_command",
        required=True,
        title="Wait commands",
        metavar="WAIT_COMMAND",
    )
    register_wait_commands(wait_subparsers)

    benchmark = add_parser(
        subparsers,
        "benchmark",
        help_text="Benchmark execution and reporting",
        description="Run benchmarks and generate reports for BenchFlow scenarios.",
    )
    benchmark_subparsers = benchmark.add_subparsers(
        dest="benchmark_command",
        required=True,
        title="Benchmark commands",
        metavar="BENCHMARK_COMMAND",
    )
    register_benchmark_commands(benchmark_subparsers)

    artifacts = add_parser(
        subparsers,
        "artifacts",
        help_text="Artifact collection",
        description="Collect benchmark and run artifacts into a local directory.",
    )
    artifacts_subparsers = artifacts.add_subparsers(
        dest="artifacts_command",
        required=True,
        title="Artifact commands",
        metavar="ARTIFACT_COMMAND",
    )
    register_artifacts_commands(artifacts_subparsers)

    metrics = add_parser(
        subparsers,
        "metrics",
        help_text="Metrics collection",
        description="Collect Prometheus metrics for BenchFlow benchmark windows.",
    )
    metrics_subparsers = metrics.add_subparsers(
        dest="metrics_command",
        required=True,
        title="Metrics commands",
        metavar="METRICS_COMMAND",
    )
    register_metrics_commands(metrics_subparsers)

    mlflow_cmd = add_parser(
        subparsers,
        "mlflow",
        help_text="MLflow integration",
        description="Upload and organize BenchFlow benchmark outputs in MLflow.",
    )
    mlflow_subparsers = mlflow_cmd.add_subparsers(
        dest="mlflow_command",
        required=True,
        title="MLflow commands",
        metavar="MLFLOW_COMMAND",
    )
    register_mlflow_commands(mlflow_subparsers)

    task = add_parser(
        subparsers,
        "task",
        help_text="Internal Tekton task entrypoints",
        description="Internal commands invoked by Tekton tasks inside the BenchFlow image.",
    )
    task_subparsers = task.add_subparsers(
        dest="task_command",
        required=True,
        title="Task commands",
        metavar="TASK_COMMAND",
    )
    register_task_commands(task_subparsers)

    watch = add_parser(
        subparsers,
        "watch",
        help_text="Follow a PipelineRun until completion",
        description="Stream a PipelineRun and report its terminal state.",
    )
    configure_watch_parser(watch)

    register_experiment_commands(subparsers, hidden=True)

    profiles = add_parser(
        subparsers,
        "profiles",
        help_text="Inspect available profiles",
        description="List and inspect deployment, benchmark, and metrics profiles.",
    )
    profile_subparsers = profiles.add_subparsers(
        dest="profiles_command",
        required=True,
        title="Profile commands",
        metavar="PROFILE_COMMAND",
    )
    register_profiles_commands(profile_subparsers)

    completion = add_parser(
        subparsers,
        "completion",
        help_text="Generate shell completion",
        description="Emit shell completion setup for bash or zsh.",
    )
    completion.add_argument("shell", choices=("bash", "zsh"), help="Target shell.")
    completion.set_defaults(func=cmd_completion)

    complete_internal = add_parser(subparsers, "_complete", add_help=False, hidden=True)
    complete_internal.add_argument("shell", choices=("bash", "zsh"))
    complete_internal.add_argument("words", nargs=argparse.REMAINDER)
    complete_internal.set_defaults(func=cmd_complete_internal)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (CommandError, ValidationError) as exc:
        ui_error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
