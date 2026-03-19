from __future__ import annotations

import sys

import click
from click.shell_completion import get_completion_class

from .cluster import CommandError
from .commands.experiment import experiment_group
from .commands.profiles import profiles_group
from .commands.run_plan import run_plan_group
from .commands.runtime import (
    artifacts_group,
    benchmark_group,
    bootstrap_command,
    deploy_group,
    metrics_group,
    mlflow_group,
    model_group,
    logs_command,
    repo_group,
    setup_group,
    task_group,
    target_group,
    teardown_group,
    undeploy_group,
    wait_group,
    watch_command,
)
from .models import ValidationError
from .ui import error as ui_error


CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
    "show_default": True,
}


@click.group(
    context_settings=CONTEXT_SETTINGS,
    help=(
        "BenchFlow bootstraps cluster prerequisites and runs LLM inference "
        "benchmarks from experiment files or direct CLI flags."
    ),
    epilog=(
        "\b\n"
        "Examples:\n"
        "  bflow bootstrap\n"
        "  bflow experiment run experiments/smoke/qwen3-06b-llm-d-smoke.yaml\n"
        "  bflow experiment resolve experiments/smoke/qwen3-06b-llm-d-smoke.yaml "
        "--format json > runplan.json\n"
        "  bflow run-plan run runplan.json\n"
        "  bflow experiment run --model Qwen/Qwen3-0.6B "
        "--deployment-profile llm-d-inference-scheduling "
        "--benchmark-profile guidellm-concurrent-1k-1k"
    ),
)
def cli() -> None:
    pass


def _completion_var(prog_name: str) -> str:
    return f"_{prog_name.replace('-', '_').upper()}_COMPLETE"


@cli.command(
    "completion",
    short_help="Generate shell completion",
    help="Emit the shell completion script for bash or zsh.",
)
@click.argument("shell", type=click.Choice(("bash", "zsh")))
def completion_command(shell: str) -> int:
    completion_class = get_completion_class(shell)
    if completion_class is None:
        raise click.ClickException(f"unsupported shell: {shell}")
    script = completion_class(cli, {}, "bflow", _completion_var("bflow")).source()
    click.echo(script, nl=False)
    return 0


cli.add_command(bootstrap_command)
cli.add_command(target_group)
cli.add_command(experiment_group)
cli.add_command(run_plan_group)
cli.add_command(repo_group)
cli.add_command(model_group)
cli.add_command(logs_command)
cli.add_command(setup_group)
cli.add_command(teardown_group)
cli.add_command(deploy_group)
cli.add_command(undeploy_group)
cli.add_command(wait_group)
cli.add_command(benchmark_group)
cli.add_command(artifacts_group)
cli.add_command(metrics_group)
cli.add_command(mlflow_group)
cli.add_command(task_group)
cli.add_command(watch_command)
cli.add_command(profiles_group)


def main(argv: list[str] | None = None) -> int:
    try:
        result = cli.main(args=argv, prog_name="bflow", standalone_mode=False)
    except KeyboardInterrupt:
        return 130
    except click.Abort:
        return 130
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except FileNotFoundError as exc:
        ui_error(str(exc))
        return 1
    except (CommandError, ValidationError) as exc:
        ui_error(str(exc))
        return 1
    return int(result or 0)


if __name__ == "__main__":
    sys.exit(main())
