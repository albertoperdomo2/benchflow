from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from ..cluster import discover_repo_root
from ..execution import load_run_plan_from_sources
from ..loaders import (
    ProfileCatalog,
    list_profile_entries,
    load_experiment,
    load_yaml_file,
)
from ..models import (
    Experiment,
    ExperimentSpec,
    Metadata,
    MlflowSpec,
    ModelSpec,
    StageSpec,
    ValidationError,
    sanitize_name,
)
from ..plans import resolve_run_plan
from ..ui import HelpFormatter


def dump_yaml(data) -> str:
    return yaml.safe_dump(data, sort_keys=False, width=1_000_000)


def dump(data, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(data, indent=2, sort_keys=True)
    return dump_yaml(data)


def repo_root_from(args: argparse.Namespace) -> Path:
    if getattr(args, "repo_root", None):
        return Path(args.repo_root).resolve()
    return discover_repo_root(Path.cwd())


def profiles_dir_from(args: argparse.Namespace) -> Path:
    if getattr(args, "profiles_dir", None):
        return Path(args.profiles_dir).resolve()
    return repo_root_from(args) / "profiles"


def parse_mapping(values: list[str] | None, option_name: str) -> dict[str, str]:
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


def parse_version_overrides(values: list[str] | None) -> dict[str, str]:
    return parse_mapping(values, "--version-override")


def experiment_from_args(args: argparse.Namespace) -> Experiment:
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
    labels.update(parse_mapping(getattr(args, "label", None), "--label"))

    mlflow_tags = dict(base_experiment.spec.mlflow.tags)
    mlflow_tags.update(parse_mapping(getattr(args, "mlflow_tag", None), "--mlflow-tag"))

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


def load_plan(args: argparse.Namespace):
    experiment = experiment_from_args(args)
    catalog = ProfileCatalog.load(profiles_dir_from(args))
    return resolve_run_plan(experiment, catalog)


def load_runtime_plan(args: argparse.Namespace):
    run_plan_file = getattr(args, "run_plan_file", None)
    run_plan_json = getattr(args, "run_plan_json", None)
    if run_plan_file or run_plan_json:
        return load_run_plan_from_sources(
            run_plan_file=run_plan_file, run_plan_json=run_plan_json
        )
    return load_plan(args)


def add_profile_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repo-root",
        help="BenchFlow repository root. Defaults to the current checkout.",
    )
    parser.add_argument(
        "--profiles-dir",
        help="Profiles directory. Defaults to <repo-root>/profiles.",
    )


def add_experiment_input_arguments(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument("--deployment-profile", help="Deployment profile name.")
    parser.add_argument("--benchmark-profile", help="Benchmark profile name.")
    parser.add_argument("--metrics-profile", help="Metrics profile name.")
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


def add_run_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-plan-file",
        help="Path to a pre-resolved RunPlan file.",
    )
    parser.add_argument(
        "--run-plan-json",
        help="Inline RunPlan JSON payload.",
    )


def add_parser(
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


def format_profile_list(entries: list[dict[str, object]]) -> str:
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


def load_profile_document(profiles_dir: Path, name: str, kind: str | None) -> dict:
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
