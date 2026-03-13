from __future__ import annotations

import argparse
import os
from pathlib import Path

from ..cluster import discover_repo_root
from ..loaders import list_profile_entries
from ..models import ValidationError
from .experiment import (
    EXPERIMENT_COMMAND_OPTIONS,
    EXPERIMENT_OPTIONS,
    EXPERIMENT_SUBCOMMANDS,
)
from .profiles import PROFILE_SUBCOMMANDS, profile_command_options
from .runtime import (
    ARTIFACTS_SUBCOMMANDS,
    BENCHMARK_SUBCOMMANDS,
    DEPLOY_SUBCOMMANDS,
    METRICS_SUBCOMMANDS,
    MLFLOW_SUBCOMMANDS,
    MODEL_SUBCOMMANDS,
    REPO_SUBCOMMANDS,
    TASK_SUBCOMMANDS,
    UNDEPLOY_SUBCOMMANDS,
    WAIT_SUBCOMMANDS,
    runtime_command_options,
)


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
    if command == "profiles":
        return profile_command_options(subcommand)
    return runtime_command_options(command, EXPERIMENT_OPTIONS, subcommand)


def completion_candidates(words: list[str], cword: int) -> list[str]:
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
            return _matching_prefix(list(PROFILE_SUBCOMMANDS), current_word)
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


def bash_completion_script() -> str:
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


def zsh_completion_script() -> str:
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


def cmd_completion(args: argparse.Namespace) -> int:
    if args.shell == "bash":
        print(bash_completion_script(), end="")
        return 0
    if args.shell == "zsh":
        print(zsh_completion_script(), end="")
        return 0
    raise ValidationError(f"unsupported shell: {args.shell}")


def cmd_complete_internal(args: argparse.Namespace) -> int:
    cword_raw = os.environ.get("BFLOW_CWORD")
    try:
        cword = int(cword_raw) if cword_raw is not None else max(len(args.words) - 1, 0)
    except ValueError as exc:
        raise ValidationError(f"invalid BFLOW_CWORD value: {cword_raw!r}") from exc

    suggestions = completion_candidates(args.words, cword)
    if suggestions:
        print("\n".join(suggestions))
    return 0
