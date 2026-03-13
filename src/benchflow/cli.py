from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from .artifacts import collect_artifacts
from .benchmark import generate_report, run_benchmark
from .cluster import (
    CommandError,
    create_manifest,
    discover_repo_root,
    follow_pipelinerun,
    get_current_namespace,
)
from .cleanup import cleanup_llmd
from .deploy import deploy_llmd
from .execution import load_run_plan_from_sources, require_platform
from .install import InstallOptions, run_install
from .loaders import (
    ProfileCatalog,
    list_profile_entries,
    load_experiment,
    load_yaml_file,
)
from .metrics import collect_metrics
from .mlflow_upload import upload_to_mlflow
from .model import download_model
from .models import (
    Experiment,
    ExperimentSpec,
    Metadata,
    MlflowSpec,
    ModelSpec,
    StageSpec,
    ValidationError,
    sanitize_name,
)
from .plans import resolve_run_plan
from .repository import clone_repo
from .renderers.deployment import write_deployment_assets
from .renderers.tekton import render_pipelinerun
from .tasking import assert_task_status, write_stage_results
from .ui import HelpFormatter, error as ui_error
from .waiting import wait_for_endpoint


TOP_LEVEL_COMMANDS = (
    "install",
    "experiment",
    "repo",
    "model",
    "deploy",
    "undeploy",
    "wait",
    "benchmark",
    "artifacts",
    "metrics",
    "mlflow",
    "task",
    "validate",
    "resolve",
    "render-pipelinerun",
    "render-deployment",
    "run",
    "watch",
    "cleanup",
    "profiles",
    "completion",
)

EXPERIMENT_SUBCOMMANDS = (
    "validate",
    "resolve",
    "render-pipelinerun",
    "render-deployment",
    "run",
    "cleanup",
)

EXPERIMENT_OPTIONS = (
    "--repo-root",
    "--profiles-dir",
    "--namespace",
    "--name",
    "--label",
    "--model",
    "--model-revision",
    "--deployment-profile",
    "--benchmark-profile",
    "--metrics-profile",
    "--service-account",
    "--ttl-seconds-after-finished",
    "--mlflow-experiment",
    "--mlflow-tag",
    "--download",
    "--no-download",
    "--deploy",
    "--no-deploy",
    "--benchmark",
    "--no-benchmark",
    "--collect",
    "--no-collect",
    "--cleanup",
    "--no-cleanup",
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

EXPERIMENT_COMMAND_OPTIONS = {
    "validate": EXPERIMENT_OPTIONS,
    "resolve": (*EXPERIMENT_OPTIONS, "--format"),
    "render-pipelinerun": (*EXPERIMENT_OPTIONS, "--pipeline-name"),
    "render-deployment": (*EXPERIMENT_OPTIONS, "--output-dir"),
    "run": (*EXPERIMENT_OPTIONS, "--pipeline-name", "--output", "--follow"),
    "cleanup": (*EXPERIMENT_OPTIONS, "--pipeline-name", "--output", "--no-follow"),
}

OPTION_VALUE_CHOICES = {
    "--kind": ("deployment", "benchmark", "metrics"),
    "--format": ("yaml", "json", "table"),
    "--tekton-channel": ("latest",),
    "--grafana-channel": ("v5",),
}

PATH_VALUE_OPTIONS = {
    "--repo-root",
    "--profiles-dir",
    "--output",
    "--output-dir",
    "--run-plan-file",
    "--source-dir",
    "--manifests-dir",
    "--json-path",
    "--artifacts-dir",
    "--models-storage-path",
    "--commit-output",
    "--url-output",
    "--mlflow-run-id-output",
    "--benchmark-start-time-output",
    "--benchmark-end-time-output",
    "--stage-download-path",
    "--stage-deploy-path",
    "--stage-benchmark-path",
    "--stage-collect-path",
    "--stage-cleanup-path",
}


def _dump_yaml(data) -> str:
    return yaml.safe_dump(data, sort_keys=False, width=1_000_000)


def _repo_root_from(args: argparse.Namespace) -> Path:
    if getattr(args, "repo_root", None):
        return Path(args.repo_root).resolve()
    return discover_repo_root(Path.cwd())


def _profiles_dir_from(args: argparse.Namespace) -> Path:
    if args.profiles_dir:
        return Path(args.profiles_dir).resolve()
    return _repo_root_from(args) / "profiles"


def _parse_mapping(values: list[str] | None, option_name: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValidationError(
                f"{option_name} entries must be KEY=VALUE, got: {value!r}"
            )
        key, mapped_value = value.split("=", 1)
        key = key.strip()
        if not key:
            raise ValidationError(f"{option_name} entries must include a non-empty key")
        parsed[key] = mapped_value
    return parsed


def _experiment_from_args(args: argparse.Namespace) -> Experiment:
    base_experiment: Experiment | None = None
    if getattr(args, "experiment", None):
        base_experiment = load_experiment(Path(args.experiment).resolve())

    if base_experiment is None:
        metadata = Metadata(name="", labels={})
        stages = StageSpec()
        mlflow = MlflowSpec()
        spec = ExperimentSpec(
            model=ModelSpec(name=""),
            deployment_profile="",
            benchmark_profile="",
            metrics_profile="default",
            namespace="benchflow",
            service_account="benchflow-runner",
            ttl_seconds_after_finished=3600,
            stages=stages,
            mlflow=mlflow,
        )
        base_experiment = Experiment(
            api_version="benchflow.io/v1alpha1",
            kind="Experiment",
            metadata=metadata,
            spec=spec,
        )

    labels = dict(base_experiment.metadata.labels)
    labels.update(_parse_mapping(getattr(args, "label", None), "--label"))

    mlflow_tags = dict(base_experiment.spec.mlflow.tags)
    mlflow_tags.update(
        _parse_mapping(getattr(args, "mlflow_tag", None), "--mlflow-tag")
    )

    model_name = getattr(args, "model", None) or base_experiment.spec.model.name
    if not model_name:
        raise ValidationError(
            "missing required input: provide an experiment file or --model"
        )

    deployment_profile = (
        getattr(args, "deployment_profile", None)
        or base_experiment.spec.deployment_profile
    )
    if not deployment_profile:
        raise ValidationError(
            "missing required input: provide an experiment file or --deployment-profile"
        )

    benchmark_profile = (
        getattr(args, "benchmark_profile", None)
        or base_experiment.spec.benchmark_profile
    )
    if not benchmark_profile:
        raise ValidationError(
            "missing required input: provide an experiment file or --benchmark-profile"
        )

    name = (
        getattr(args, "name", None)
        or base_experiment.metadata.name
        or sanitize_name(model_name)
    )

    stages = StageSpec(
        download=base_experiment.spec.stages.download,
        deploy=base_experiment.spec.stages.deploy,
        benchmark=base_experiment.spec.stages.benchmark,
        collect=base_experiment.spec.stages.collect,
        cleanup=base_experiment.spec.stages.cleanup,
    )
    for stage_name in ("download", "deploy", "benchmark", "collect", "cleanup"):
        override = getattr(args, f"stage_{stage_name}", None)
        if override is not None:
            setattr(stages, stage_name, override)

    return Experiment(
        api_version=base_experiment.api_version,
        kind="Experiment",
        metadata=Metadata(name=name, labels=labels),
        spec=ExperimentSpec(
            model=ModelSpec(
                name=model_name,
                revision=getattr(args, "model_revision", None)
                or base_experiment.spec.model.revision,
            ),
            deployment_profile=deployment_profile,
            benchmark_profile=benchmark_profile,
            metrics_profile=getattr(args, "metrics_profile", None)
            or base_experiment.spec.metrics_profile,
            namespace=getattr(args, "namespace", None)
            or base_experiment.spec.namespace,
            service_account=getattr(args, "service_account", None)
            or base_experiment.spec.service_account,
            ttl_seconds_after_finished=(
                args.ttl_seconds_after_finished
                if getattr(args, "ttl_seconds_after_finished", None) is not None
                else base_experiment.spec.ttl_seconds_after_finished
            ),
            stages=stages,
            mlflow=MlflowSpec(
                experiment=getattr(args, "mlflow_experiment", None)
                or base_experiment.spec.mlflow.experiment,
                tags=mlflow_tags,
            ),
        ),
    )


def _load_plan(args: argparse.Namespace):
    experiment = _experiment_from_args(args)
    catalog = ProfileCatalog.load(_profiles_dir_from(args))
    return resolve_run_plan(experiment, catalog)


def _load_runtime_plan(args: argparse.Namespace):
    run_plan_file = getattr(args, "run_plan_file", None)
    run_plan_json = getattr(args, "run_plan_json", None)
    if run_plan_file or run_plan_json:
        return load_run_plan_from_sources(
            run_plan_file=run_plan_file, run_plan_json=run_plan_json
        )
    return _load_plan(args)


def _parse_version_overrides(values: list[str] | None) -> dict[str, str]:
    return _parse_mapping(values, "--version-override")


def _dump(data, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(data, indent=2, sort_keys=True)
    return _dump_yaml(data)


def _add_profile_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repo-root",
        help="BenchFlow repository root. Defaults to the current checkout.",
    )
    parser.add_argument(
        "--profiles-dir",
        help="Profiles directory. Defaults to <repo-root>/profiles.",
    )


def _format_profile_list(entries: list[dict[str, object]]) -> str:
    if not entries:
        return ""

    kind_width = max(len("KIND"), max(len(str(entry["kind"])) for entry in entries))
    name_width = max(len("NAME"), max(len(str(entry["name"])) for entry in entries))
    path_width = max(len("PATH"), max(len(str(entry["path"])) for entry in entries))

    lines = [
        f"{'KIND':<{kind_width}}  {'NAME':<{name_width}}  {'PATH':<{path_width}}  DETAILS",
    ]
    for entry in entries:
        details = ", ".join(f"{key}={value}" for key, value in entry["details"].items())
        lines.append(
            f"{entry['kind']:<{kind_width}}  {entry['name']:<{name_width}}  "
            f"{entry['path']:<{path_width}}  {details}"
        )
    return "\n".join(lines)


def _load_profile_document(profiles_dir: Path, name: str, kind: str | None) -> dict:
    entries = list_profile_entries(profiles_dir)
    matches = [
        entry
        for entry in entries
        if entry.name == name and (kind is None or entry.kind == kind)
    ]

    if not matches:
        if kind is None:
            raise ValidationError(f"unknown profile: {name}")
        raise ValidationError(f"unknown {kind} profile: {name}")

    if len(matches) > 1:
        matched_kinds = ", ".join(sorted(entry.kind for entry in matches))
        raise ValidationError(
            f"profile name {name!r} is ambiguous across multiple kinds: {matched_kinds}; use --kind"
        )

    return load_yaml_file(profiles_dir / matches[0].path)


def _find_last_option_value(
    words: list[str], option: str, upto_index: int | None = None
) -> str | None:
    limit = len(words) if upto_index is None else max(upto_index, 0)
    for index in range(limit - 1):
        if words[index] == option:
            return words[index + 1]
    return None


def _resolve_completion_repo_root(words: list[str]) -> Path:
    repo_root = _find_last_option_value(words, "--repo-root")
    if repo_root:
        return Path(repo_root).resolve()
    return discover_repo_root(Path.cwd())


def _resolve_completion_profiles_dir(words: list[str]) -> Path:
    profiles_dir = _find_last_option_value(words, "--profiles-dir")
    if profiles_dir:
        return Path(profiles_dir).resolve()
    return _resolve_completion_repo_root(words) / "profiles"


def _matching_prefix(values: list[str], prefix: str) -> list[str]:
    return [value for value in values if value.startswith(prefix)]


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _complete_paths(
    prefix: str, *, root: Path | None = None, suffix: str | None = None
) -> list[str]:
    base_root = root or Path.cwd()
    prefix_path = Path(prefix) if prefix else Path(".")
    if prefix.endswith("/"):
        parent = prefix_path
        needle = ""
    else:
        parent = prefix_path.parent if prefix_path.parent != Path("") else Path(".")
        needle = prefix_path.name

    search_root = (
        (base_root / parent).resolve()
        if not prefix_path.is_absolute()
        else parent.resolve()
    )
    if not search_root.exists() or not search_root.is_dir():
        return []

    matches: list[str] = []
    for candidate in sorted(search_root.iterdir()):
        if not candidate.name.startswith(needle):
            continue
        if suffix and candidate.is_file() and not candidate.name.endswith(suffix):
            continue
        display = _display_path(
            candidate if prefix_path.is_absolute() else parent / candidate.name
        )
        if candidate.is_dir():
            display = f"{display}/"
        matches.append(display)
    return matches


def _complete_experiment_files(words: list[str], prefix: str) -> list[str]:
    repo_root = _resolve_completion_repo_root(words)
    experiments_root = repo_root / "experiments"
    if prefix:
        return _complete_paths(prefix, root=Path.cwd(), suffix=".yaml")

    if not experiments_root.exists():
        return []
    return [_display_path(path) for path in sorted(experiments_root.rglob("*.yaml"))]


def _profile_name_candidates(
    words: list[str], kind: str | None, prefix: str
) -> list[str]:
    profiles_dir = _resolve_completion_profiles_dir(words)
    entries = list_profile_entries(profiles_dir)
    names = sorted(entry.name for entry in entries if kind in (None, entry.kind))
    return _matching_prefix(names, prefix)


def _complete_for_option(
    words: list[str], prev_word: str, current_word: str
) -> list[str]:
    if prev_word == "--deployment-profile":
        return _profile_name_candidates(words, "deployment", current_word)
    if prev_word == "--benchmark-profile":
        return _profile_name_candidates(words, "benchmark", current_word)
    if prev_word == "--metrics-profile":
        return _profile_name_candidates(words, "metrics", current_word)
    if prev_word == "--kind":
        return _matching_prefix(list(OPTION_VALUE_CHOICES["--kind"]), current_word)
    if prev_word == "--format":
        if words and words[0] == "profiles" and len(words) > 1 and words[1] == "list":
            return _matching_prefix(["table", "yaml", "json"], current_word)
        return _matching_prefix(["yaml", "json"], current_word)
    if prev_word == "--tekton-channel":
        return _matching_prefix(
            list(OPTION_VALUE_CHOICES["--tekton-channel"]), current_word
        )
    if prev_word == "--grafana-channel":
        return _matching_prefix(
            list(OPTION_VALUE_CHOICES["--grafana-channel"]), current_word
        )
    if prev_word in PATH_VALUE_OPTIONS:
        return _complete_paths(current_word, root=Path.cwd())
    return []


def _command_options(command: str, subcommand: str | None = None) -> tuple[str, ...]:
    if command == "experiment" and subcommand in EXPERIMENT_COMMAND_OPTIONS:
        return EXPERIMENT_COMMAND_OPTIONS[subcommand]
    if command in EXPERIMENT_COMMAND_OPTIONS:
        return EXPERIMENT_COMMAND_OPTIONS[command]
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
            *EXPERIMENT_OPTIONS,
            "--run-plan-file",
            "--run-plan-json",
            "--models-storage-path",
            "--no-skip-if-exists",
        )
    if command == "deploy" and subcommand == "llm-d":
        return (
            *EXPERIMENT_OPTIONS,
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
            *EXPERIMENT_OPTIONS,
            "--run-plan-file",
            "--run-plan-json",
            "--no-wait",
            "--timeout-seconds",
            "--no-skip-if-not-exists",
        )
    if command == "wait" and subcommand == "endpoint":
        return (
            *EXPERIMENT_OPTIONS,
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
                *EXPERIMENT_OPTIONS,
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
                *EXPERIMENT_OPTIONS,
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
            *EXPERIMENT_OPTIONS,
            "--run-plan-file",
            "--run-plan-json",
            "--artifacts-dir",
            "--pipeline-run-name",
        )
    if command == "metrics" and subcommand == "collect":
        return (
            *EXPERIMENT_OPTIONS,
            "--run-plan-file",
            "--run-plan-json",
            "--benchmark-start-time",
            "--benchmark-end-time",
            "--artifacts-dir",
        )
    if command == "mlflow" and subcommand == "upload":
        return (
            *EXPERIMENT_OPTIONS,
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
    if command == "profiles":
        if subcommand == "list":
            return ("--repo-root", "--profiles-dir", "--kind", "--format")
        if subcommand == "show":
            return ("--repo-root", "--profiles-dir", "--kind", "--format")
    if command == "completion":
        return ()
    return ()


def _completion_candidates(words: list[str], cword: int) -> list[str]:
    current_word = words[cword] if 0 <= cword < len(words) else ""
    prev_word = words[cword - 1] if cword > 0 else ""

    option_value_matches = _complete_for_option(words, prev_word, current_word)
    if option_value_matches:
        return option_value_matches

    if cword <= 0:
        return _matching_prefix(list(TOP_LEVEL_COMMANDS), current_word)

    command = words[0] if words else ""
    if command not in TOP_LEVEL_COMMANDS:
        return _matching_prefix(list(TOP_LEVEL_COMMANDS), current_word)

    if command == "profiles":
        if cword == 1:
            return _matching_prefix(["list", "show"], current_word)
        subcommand = words[1] if len(words) > 1 else None
        if subcommand == "show" and not current_word.startswith("-"):
            kind = _find_last_option_value(words, "--kind", upto_index=cword)
            return _profile_name_candidates(words, kind, current_word)
        if current_word.startswith("-"):
            return _matching_prefix(
                list(_command_options(command, subcommand)), current_word
            )
        return []

    if command == "repo":
        if cword == 1:
            return _matching_prefix(list(REPO_SUBCOMMANDS), current_word)
        subcommand = words[1] if len(words) > 1 else None
        if current_word.startswith("-"):
            return _matching_prefix(
                list(_command_options(command, subcommand)), current_word
            )
        return []

    if command == "model":
        if cword == 1:
            return _matching_prefix(list(MODEL_SUBCOMMANDS), current_word)
        subcommand = words[1] if len(words) > 1 else None
        if current_word.startswith("-"):
            return _matching_prefix(
                list(_command_options(command, subcommand)), current_word
            )
        return _complete_experiment_files(words, current_word)

    if command == "deploy":
        if cword == 1:
            return _matching_prefix(list(DEPLOY_SUBCOMMANDS), current_word)
        subcommand = words[1] if len(words) > 1 else None
        if current_word.startswith("-"):
            return _matching_prefix(
                list(_command_options(command, subcommand)), current_word
            )
        return _complete_experiment_files(words, current_word)

    if command == "undeploy":
        if cword == 1:
            return _matching_prefix(list(UNDEPLOY_SUBCOMMANDS), current_word)
        subcommand = words[1] if len(words) > 1 else None
        if current_word.startswith("-"):
            return _matching_prefix(
                list(_command_options(command, subcommand)), current_word
            )
        return _complete_experiment_files(words, current_word)

    if command == "wait":
        if cword == 1:
            return _matching_prefix(list(WAIT_SUBCOMMANDS), current_word)
        subcommand = words[1] if len(words) > 1 else None
        if current_word.startswith("-"):
            return _matching_prefix(
                list(_command_options(command, subcommand)), current_word
            )
        return _complete_experiment_files(words, current_word)

    if command == "benchmark":
        if cword == 1:
            return _matching_prefix(list(BENCHMARK_SUBCOMMANDS), current_word)
        subcommand = words[1] if len(words) > 1 else None
        if current_word.startswith("-"):
            return _matching_prefix(
                list(_command_options(command, subcommand)), current_word
            )
        return _complete_experiment_files(words, current_word)

    if command == "artifacts":
        if cword == 1:
            return _matching_prefix(list(ARTIFACTS_SUBCOMMANDS), current_word)
        subcommand = words[1] if len(words) > 1 else None
        if current_word.startswith("-"):
            return _matching_prefix(
                list(_command_options(command, subcommand)), current_word
            )
        return _complete_experiment_files(words, current_word)

    if command == "metrics":
        if cword == 1:
            return _matching_prefix(list(METRICS_SUBCOMMANDS), current_word)
        subcommand = words[1] if len(words) > 1 else None
        if current_word.startswith("-"):
            return _matching_prefix(
                list(_command_options(command, subcommand)), current_word
            )
        return _complete_experiment_files(words, current_word)

    if command == "mlflow":
        if cword == 1:
            return _matching_prefix(list(MLFLOW_SUBCOMMANDS), current_word)
        subcommand = words[1] if len(words) > 1 else None
        if current_word.startswith("-"):
            return _matching_prefix(
                list(_command_options(command, subcommand)), current_word
            )
        return _complete_experiment_files(words, current_word)

    if command == "task":
        if cword == 1:
            return _matching_prefix(list(TASK_SUBCOMMANDS), current_word)
        subcommand = words[1] if len(words) > 1 else None
        if current_word.startswith("-"):
            return _matching_prefix(
                list(_command_options(command, subcommand)), current_word
            )
        return []

    if command == "completion":
        if cword == 1:
            return _matching_prefix(["bash", "zsh"], current_word)
        return []

    if command == "experiment":
        if cword == 1:
            return _matching_prefix(list(EXPERIMENT_SUBCOMMANDS), current_word)
        subcommand = words[1] if len(words) > 1 else None
        if subcommand not in EXPERIMENT_SUBCOMMANDS:
            return _matching_prefix(list(EXPERIMENT_SUBCOMMANDS), current_word)
        if current_word.startswith("-"):
            return _matching_prefix(
                list(_command_options(command, subcommand)), current_word
            )
        return _complete_experiment_files(words, current_word)

    if command == "watch":
        if current_word.startswith("-"):
            return _matching_prefix(list(_command_options(command)), current_word)
        return []

    if current_word.startswith("-"):
        return _matching_prefix(list(_command_options(command)), current_word)

    if command in EXPERIMENT_COMMAND_OPTIONS:
        return _complete_experiment_files(words, current_word)

    return []


def _bash_completion_script() -> str:
    return """_bflow_completion() {
  local cword=$((COMP_CWORD - 1))
  if (( cword < 0 )); then
    cword=0
  fi

  local -a completions=()
  while IFS= read -r line; do
    completions+=("$line")
  done < <(BFLOW_CWORD="$cword" bflow _complete bash "${COMP_WORDS[@]:1}")

  COMPREPLY=("${completions[@]}")
}

complete -F _bflow_completion bflow
"""


def _zsh_completion_script() -> str:
    return """#compdef bflow

_bflow_completion() {
  local cword=$((CURRENT - 2))
  if (( cword < 0 )); then
    cword=0
  fi

  local -a completions
  completions=("${(@f)$(BFLOW_CWORD="$cword" bflow _complete zsh "${words[@]:2}")}")
  compadd -Q -- "${completions[@]}"
}

compdef _bflow_completion bflow
"""


def cmd_validate(args: argparse.Namespace) -> int:
    _load_plan(args)
    print("valid")
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    plan = _load_plan(args)
    print(_dump(plan.to_dict(), args.format))
    return 0


def cmd_render_pipelinerun(args: argparse.Namespace) -> int:
    plan = _load_plan(args)
    manifest = render_pipelinerun(plan, pipeline_name=args.pipeline_name)
    print(_dump_yaml(manifest))
    return 0


def cmd_render_deployment(args: argparse.Namespace) -> int:
    plan = _load_plan(args)
    output_dir = Path(args.output_dir).resolve()
    written = write_deployment_assets(plan, output_dir)
    for path in written:
        print(path)
    return 0


def _render_manifest_yaml(
    args: argparse.Namespace, *, cleanup_only: bool = False
) -> tuple[object, str, str]:
    plan = _load_plan(args)
    if cleanup_only:
        plan.stages = StageSpec(
            download=False,
            deploy=False,
            benchmark=False,
            collect=False,
            cleanup=True,
        )

    manifest = render_pipelinerun(plan, pipeline_name=args.pipeline_name)
    manifest_yaml = _dump_yaml(manifest)
    namespace = plan.deployment.namespace
    return plan, manifest_yaml, namespace


def _submit_manifest(manifest_yaml: str, namespace: str) -> str:
    submitted = create_manifest(manifest_yaml, namespace)
    name = submitted.get("metadata", {}).get("name")
    if not name:
        raise CommandError("oc create returned no PipelineRun name")
    return str(name)


def cmd_install(args: argparse.Namespace) -> int:
    repo_root = _repo_root_from(args)
    return run_install(
        repo_root,
        InstallOptions(
            namespace=args.namespace or "benchflow",
            install_tekton=not args.skip_tekton_install,
            install_grafana=not args.skip_grafana_install,
            tekton_channel=args.tekton_channel or "latest",
            grafana_channel=args.grafana_channel or "v5",
            models_storage_class=args.models_storage_class,
            models_storage_size=args.models_size or "250Gi",
            models_storage_access_mode=args.models_access_mode or "ReadWriteOnce",
            results_storage_class=args.results_storage_class,
            results_storage_size=args.results_size or "20Gi",
        ),
    )


def cmd_run(args: argparse.Namespace) -> int:
    _, manifest_yaml, namespace = _render_manifest_yaml(args)

    if args.output:
        Path(args.output).resolve().write_text(manifest_yaml, encoding="utf-8")

    name = _submit_manifest(manifest_yaml, namespace)
    print(name)

    if args.follow:
        return 0 if follow_pipelinerun(namespace, name) else 1
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    namespace = args.namespace or get_current_namespace()
    return 0 if follow_pipelinerun(namespace, args.pipelinerun_name) else 1


def cmd_cleanup(args: argparse.Namespace) -> int:
    _, manifest_yaml, namespace = _render_manifest_yaml(args, cleanup_only=True)

    if args.output:
        Path(args.output).resolve().write_text(manifest_yaml, encoding="utf-8")

    name = _submit_manifest(manifest_yaml, namespace)
    print(name)

    if args.follow:
        return 0 if follow_pipelinerun(namespace, name) else 1
    return 0


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
    plan = _load_runtime_plan(args)
    require_platform(plan, "llm-d")
    target_dir = download_model(
        plan,
        models_storage_path=Path(args.models_storage_path).resolve(),
        skip_if_exists=not args.no_skip_if_exists,
    )
    print(target_dir)
    return 0


def cmd_deploy_llmd(args: argparse.Namespace) -> int:
    plan = _load_runtime_plan(args)
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
    plan = _load_runtime_plan(args)
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
        plan = _load_runtime_plan(args)
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
    plan = _load_runtime_plan(args)
    require_platform(plan, "llm-d")
    if plan.benchmark.tool != "guidellm":
        raise ValidationError(
            f"unsupported benchmark tool: {plan.benchmark.tool}; only guidellm is implemented"
        )
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
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
            extra_tags=_parse_mapping(args.tag, "--tag"),
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
        plan = _load_runtime_plan(args)
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
        version_overrides=_parse_version_overrides(args.version_override),
        additional_csv_files=args.additional_csv or None,
    )
    print(report_path)
    return 0


def cmd_artifacts_collect(args: argparse.Namespace) -> int:
    plan = _load_runtime_plan(args)
    require_platform(plan, "llm-d")
    artifact_dir = collect_artifacts(
        plan,
        artifacts_dir=Path(args.artifacts_dir).resolve(),
        pipeline_run_name=args.pipeline_run_name or "",
    )
    print(artifact_dir)
    return 0


def cmd_metrics_collect(args: argparse.Namespace) -> int:
    plan = _load_runtime_plan(args)
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
    plan = _load_runtime_plan(args)
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
    write_stage_results(
        plan,
        stage_download_path=Path(args.stage_download_path).resolve(),
        stage_deploy_path=Path(args.stage_deploy_path).resolve(),
        stage_benchmark_path=Path(args.stage_benchmark_path).resolve(),
        stage_collect_path=Path(args.stage_collect_path).resolve(),
        stage_cleanup_path=Path(args.stage_cleanup_path).resolve(),
    )
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


def cmd_profiles_list(args: argparse.Namespace) -> int:
    profiles_dir = _profiles_dir_from(args)
    entries = [
        entry
        for entry in list_profile_entries(profiles_dir)
        if args.kind in (None, entry.kind)
    ]
    payload = [entry.to_dict() for entry in entries]

    if args.format == "table":
        output = _format_profile_list(payload)
        if output:
            print(output)
        return 0

    print(_dump(payload, args.format))
    return 0


def cmd_profiles_show(args: argparse.Namespace) -> int:
    profiles_dir = _profiles_dir_from(args)
    document = _load_profile_document(profiles_dir, args.name, args.kind)
    print(_dump(document, args.format))
    return 0


def cmd_completion(args: argparse.Namespace) -> int:
    if args.shell == "bash":
        print(_bash_completion_script(), end="")
        return 0
    if args.shell == "zsh":
        print(_zsh_completion_script(), end="")
        return 0
    raise ValidationError(f"unsupported shell: {args.shell}")


def cmd_complete_internal(args: argparse.Namespace) -> int:
    cword_raw = os.environ.get("BFLOW_CWORD")
    try:
        cword = int(cword_raw) if cword_raw is not None else max(len(args.words) - 1, 0)
    except ValueError as exc:
        raise ValidationError(f"invalid BFLOW_CWORD value: {cword_raw!r}") from exc

    suggestions = _completion_candidates(args.words, cword)
    if suggestions:
        print("\n".join(suggestions))
    return 0


def _add_experiment_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "experiment",
        nargs="?",
        help="Experiment file to load. If omitted, define the experiment entirely with flags.",
    )
    parser.add_argument(
        "--repo-root",
        help="BenchFlow repository root. Defaults to the current checkout.",
    )
    parser.add_argument(
        "--profiles-dir",
        help="Profiles directory. Defaults to <repo-root>/profiles.",
    )
    parser.add_argument("--namespace", help="Target namespace for the run.")
    parser.add_argument("--name", help="Experiment name override.")
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Experiment label override. Repeat to set multiple labels.",
    )
    parser.add_argument(
        "--model", help="Model identifier, for example Qwen/Qwen3-0.6B."
    )
    parser.add_argument("--model-revision", help="Model revision or tag.")
    parser.add_argument(
        "--deployment-profile",
        help="Deployment profile name.",
    )
    parser.add_argument(
        "--benchmark-profile",
        help="Benchmark profile name.",
    )
    parser.add_argument(
        "--metrics-profile",
        help="Metrics profile name.",
    )
    parser.add_argument(
        "--service-account", help="Service account used by the PipelineRun."
    )
    parser.add_argument(
        "--ttl-seconds-after-finished",
        type=int,
        help="TTL for finished PipelineRuns.",
    )
    parser.add_argument("--mlflow-experiment", help="MLflow experiment name override.")
    parser.add_argument(
        "--mlflow-tag",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="MLflow tag override. Repeat to set multiple tags.",
    )
    for stage_name in ("download", "deploy", "benchmark", "collect", "cleanup"):
        parser.add_argument(
            f"--{stage_name}",
            dest=f"stage_{stage_name}",
            action=argparse.BooleanOptionalAction,
            default=None,
            help=f"Enable or disable the {stage_name} stage.",
        )


def _add_run_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-plan-file",
        help="Path to a pre-resolved RunPlan file.",
    )
    parser.add_argument(
        "--run-plan-json",
        help="Inline RunPlan JSON payload.",
    )


def _add_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    *,
    help_text: str | None = None,
    description: str | None = None,
    add_help: bool = True,
    hidden: bool = False,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        name,
        help=argparse.SUPPRESS if hidden else help_text,
        description=description,
        formatter_class=HelpFormatter,
        add_help=add_help,
    )
    if hidden:
        subparsers._choices_actions = [  # type: ignore[attr-defined]
            action
            for action in subparsers._choices_actions  # type: ignore[attr-defined]
            if getattr(action, "dest", None) != name
        ]
    return parser


def _register_experiment_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    hidden: bool = False,
) -> None:
    validate = _add_parser(
        subparsers,
        "validate",
        help_text="Validate an experiment definition",
        description="Validate an experiment file or CLI-defined experiment.",
        hidden=hidden,
    )
    _add_experiment_input_arguments(validate)
    validate.set_defaults(func=cmd_validate)

    resolve = _add_parser(
        subparsers,
        "resolve",
        help_text="Resolve profiles into a complete RunPlan",
        description="Resolve an experiment into the fully expanded RunPlan used by BenchFlow.",
        hidden=hidden,
    )
    _add_experiment_input_arguments(resolve)
    resolve.add_argument("--format", choices=("yaml", "json"), default="yaml")
    resolve.set_defaults(func=cmd_resolve)

    render_pr = _add_parser(
        subparsers,
        "render-pipelinerun",
        help_text="Render the Tekton PipelineRun manifest",
        description="Render the Tekton PipelineRun that would be submitted for an experiment.",
        hidden=hidden,
    )
    _add_experiment_input_arguments(render_pr)
    render_pr.add_argument(
        "--pipeline-name",
        default="benchflow-e2e",
        help="Pipeline name to reference in the rendered PipelineRun.",
    )
    render_pr.set_defaults(func=cmd_render_pipelinerun)

    render_deployment = _add_parser(
        subparsers,
        "render-deployment",
        help_text="Render deployment manifests to disk",
        description="Render deployment assets for an experiment without submitting a run.",
        hidden=hidden,
    )
    _add_experiment_input_arguments(render_deployment)
    render_deployment.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the rendered deployment assets should be written.",
    )
    render_deployment.set_defaults(func=cmd_render_deployment)

    run = _add_parser(
        subparsers,
        "run",
        help_text="Submit an experiment as a PipelineRun",
        description="Submit an experiment to the cluster and optionally follow it.",
        hidden=hidden,
    )
    _add_experiment_input_arguments(run)
    run.add_argument(
        "--pipeline-name",
        default="benchflow-e2e",
        help="Pipeline name to reference when rendering the PipelineRun.",
    )
    run.add_argument(
        "--output",
        help="Write the rendered PipelineRun manifest to this file before submitting.",
    )
    run.add_argument(
        "--follow",
        action="store_true",
        help="Follow the PipelineRun after submission.",
    )
    run.set_defaults(func=cmd_run, follow=False)

    cleanup = _add_parser(
        subparsers,
        "cleanup",
        help_text="Submit a cleanup-only PipelineRun",
        description="Submit a cleanup-only run for an experiment.",
        hidden=hidden,
    )
    _add_experiment_input_arguments(cleanup)
    cleanup.add_argument(
        "--pipeline-name",
        default="benchflow-e2e",
        help="Pipeline name to reference when rendering the cleanup PipelineRun.",
    )
    cleanup.add_argument(
        "--output",
        help="Write the rendered cleanup PipelineRun manifest to this file before submitting.",
    )
    cleanup.add_argument(
        "--no-follow",
        dest="follow",
        action="store_false",
        help="Submit the cleanup PipelineRun without following it.",
    )
    cleanup.set_defaults(func=cmd_cleanup, follow=True)


def _register_runtime_plan_source_arguments(parser: argparse.ArgumentParser) -> None:
    _add_run_plan_arguments(parser)
    _add_experiment_input_arguments(parser)


def _register_repo_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    clone = _add_parser(
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


def _register_model_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    download = _add_parser(
        subparsers,
        "download",
        help_text="Download a model into the model cache",
        description="Download a model referenced by the RunPlan into the shared model cache.",
    )
    _register_runtime_plan_source_arguments(download)
    download.add_argument("--models-storage-path", required=True)
    download.add_argument("--no-skip-if-exists", action="store_true")
    download.set_defaults(func=cmd_model_download)


def _register_deploy_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    llmd = _add_parser(
        subparsers,
        "llm-d",
        help_text="Deploy an llm-d scenario",
        description="Deploy an llm-d scenario from a resolved RunPlan.",
    )
    _register_runtime_plan_source_arguments(llmd)
    llmd.add_argument("--source-dir")
    llmd.add_argument("--manifests-dir")
    llmd.add_argument("--pipeline-run-name")
    llmd.add_argument("--no-skip-if-exists", action="store_true")
    llmd.add_argument("--no-verify", action="store_true")
    llmd.add_argument("--verify-timeout-seconds", type=int, default=900)
    llmd.set_defaults(func=cmd_deploy_llmd)


def _register_undeploy_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    llmd = _add_parser(
        subparsers,
        "llm-d",
        help_text="Remove an llm-d deployment",
        description="Tear down an llm-d deployment from a resolved RunPlan.",
    )
    _register_runtime_plan_source_arguments(llmd)
    llmd.add_argument("--no-wait", action="store_true")
    llmd.add_argument("--timeout-seconds", type=int, default=600)
    llmd.add_argument("--no-skip-if-not-exists", action="store_true")
    llmd.set_defaults(func=cmd_undeploy_llmd)


def _register_wait_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    endpoint = _add_parser(
        subparsers,
        "endpoint",
        help_text="Wait for the deployment endpoint to become ready",
        description="Poll the resolved target endpoint until it becomes reachable.",
    )
    _register_runtime_plan_source_arguments(endpoint)
    endpoint.add_argument("--target-url")
    endpoint.add_argument("--endpoint-path")
    endpoint.add_argument("--timeout-seconds", type=int, default=3600)
    endpoint.add_argument("--retry-interval", type=int, default=10)
    endpoint.add_argument("--verify-tls", action="store_true")
    endpoint.set_defaults(func=cmd_wait_endpoint)


def _register_benchmark_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    run = _add_parser(
        subparsers,
        "run",
        help_text="Run a GuideLLM benchmark",
        description="Execute the configured benchmark and optionally upload results to MLflow.",
    )
    _register_runtime_plan_source_arguments(run)
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

    report = _add_parser(
        subparsers,
        "report",
        help_text="Generate a benchmark report",
        description="Generate a report from benchmark JSON and optional MLflow metadata.",
    )
    _register_runtime_plan_source_arguments(report)
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


def _register_metrics_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    collect = _add_parser(
        subparsers,
        "collect",
        help_text="Collect Prometheus metrics for a benchmark window",
        description="Collect benchmark metrics from Prometheus or Thanos for a resolved RunPlan.",
    )
    _register_runtime_plan_source_arguments(collect)
    collect.add_argument("--benchmark-start-time", required=True)
    collect.add_argument("--benchmark-end-time", required=True)
    collect.add_argument("--artifacts-dir", required=True)
    collect.set_defaults(func=cmd_metrics_collect)


def _register_artifacts_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    collect = _add_parser(
        subparsers,
        "collect",
        help_text="Collect run artifacts into a local directory",
        description="Collect the artifacts BenchFlow expects from a finished run.",
    )
    _register_runtime_plan_source_arguments(collect)
    collect.add_argument("--artifacts-dir", required=True)
    collect.add_argument("--pipeline-run-name")
    collect.set_defaults(func=cmd_artifacts_collect)


def _register_mlflow_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    upload = _add_parser(
        subparsers,
        "upload",
        help_text="Upload artifacts and metrics to MLflow",
        description="Upload benchmark artifacts, metrics, and metadata to MLflow.",
    )
    _register_runtime_plan_source_arguments(upload)
    upload.add_argument("--mlflow-run-id", required=True)
    upload.add_argument("--benchmark-start-time", required=True)
    upload.add_argument("--benchmark-end-time", required=True)
    upload.add_argument("--artifacts-dir", required=True)
    upload.add_argument("--grafana-url")
    upload.set_defaults(func=cmd_mlflow_upload)


def _register_task_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    resolve = _add_parser(
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

    assert_status_cmd = _add_parser(
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

    install = _add_parser(
        subparsers,
        "install",
        help_text="Install BenchFlow and cluster dependencies",
        description="Install BenchFlow into a namespace and bootstrap Tekton, Grafana, RBAC, and PVCs.",
    )
    install.add_argument(
        "--repo-root",
        help="BenchFlow repository root. Defaults to the current checkout.",
    )
    install.add_argument(
        "--namespace",
        help="Target namespace. Defaults to benchflow.",
    )
    install.add_argument(
        "--skip-tekton-install",
        action="store_true",
        help="Do not install Tekton if it is missing.",
    )
    install.add_argument(
        "--skip-grafana-install",
        action="store_true",
        help="Do not install the Grafana operator if it is missing.",
    )
    install.add_argument(
        "--tekton-channel",
        help="OpenShift Pipelines operator channel.",
    )
    install.add_argument(
        "--grafana-channel",
        help="Grafana operator channel.",
    )
    install.add_argument(
        "--models-storage-class",
        help="StorageClass for the shared model cache PVC.",
    )
    install.add_argument(
        "--models-size",
        help="Requested size for the model cache PVC.",
    )
    install.add_argument(
        "--models-access-mode",
        help="Access mode for the model cache PVC.",
    )
    install.add_argument(
        "--results-storage-class",
        help="StorageClass for the benchmark results PVC.",
    )
    install.add_argument(
        "--results-size",
        help="Requested size for the benchmark results PVC.",
    )
    install.set_defaults(func=cmd_install)

    experiment = _add_parser(
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
    _register_experiment_commands(experiment_subparsers)

    repo = _add_parser(
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
    _register_repo_commands(repo_subparsers)

    model = _add_parser(
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
    _register_model_commands(model_subparsers)

    deploy = _add_parser(
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
    _register_deploy_commands(deploy_subparsers)

    undeploy = _add_parser(
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
    _register_undeploy_commands(undeploy_subparsers)

    wait = _add_parser(
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
    _register_wait_commands(wait_subparsers)

    benchmark = _add_parser(
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
    _register_benchmark_commands(benchmark_subparsers)

    artifacts = _add_parser(
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
    _register_artifacts_commands(artifacts_subparsers)

    metrics = _add_parser(
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
    _register_metrics_commands(metrics_subparsers)

    mlflow_cmd = _add_parser(
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
    _register_mlflow_commands(mlflow_subparsers)

    task = _add_parser(
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
    _register_task_commands(task_subparsers)

    watch = _add_parser(
        subparsers,
        "watch",
        help_text="Follow a PipelineRun until completion",
        description="Stream a PipelineRun and report its terminal state.",
    )
    watch.add_argument("pipelinerun_name", help="PipelineRun name to follow.")
    watch.add_argument(
        "--namespace",
        help="Namespace that contains the PipelineRun. Defaults to the current oc project.",
    )
    watch.set_defaults(func=cmd_watch)

    _register_experiment_commands(subparsers, hidden=True)

    profiles = _add_parser(
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

    profiles_list = _add_parser(
        profile_subparsers,
        "list",
        help_text="List available profiles",
        description="List the profiles available in the current repository.",
    )
    _add_profile_source_arguments(profiles_list)
    profiles_list.add_argument(
        "--kind",
        choices=("deployment", "benchmark", "metrics"),
        help="Restrict the list to one profile kind.",
    )
    profiles_list.add_argument(
        "--format",
        choices=("table", "yaml", "json"),
        default="table",
        help="Output format.",
    )
    profiles_list.set_defaults(func=cmd_profiles_list)

    profiles_show = _add_parser(
        profile_subparsers,
        "show",
        help_text="Show a single profile",
        description="Print a single profile document in YAML or JSON.",
    )
    _add_profile_source_arguments(profiles_show)
    profiles_show.add_argument("name", help="Profile name.")
    profiles_show.add_argument(
        "--kind",
        choices=("deployment", "benchmark", "metrics"),
        help="Profile kind when the name is ambiguous.",
    )
    profiles_show.add_argument(
        "--format",
        choices=("yaml", "json"),
        default="yaml",
        help="Output format.",
    )
    profiles_show.set_defaults(func=cmd_profiles_show)

    completion = _add_parser(
        subparsers,
        "completion",
        help_text="Generate shell completion",
        description="Emit shell completion setup for bash or zsh.",
    )
    completion.add_argument("shell", choices=("bash", "zsh"), help="Target shell.")
    completion.set_defaults(func=cmd_completion)

    complete_internal = _add_parser(
        subparsers, "_complete", add_help=False, hidden=True
    )
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
