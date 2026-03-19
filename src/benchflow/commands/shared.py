from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import click
import yaml
from click.shell_completion import CompletionItem

from ..cluster import discover_repo_root
from ..orchestration import load_run_plan_from_sources
from ..loaders import (
    ProfileCatalog,
    list_profile_entries,
    load_experiment,
    load_yaml_file,
)
from ..matrix import require_single_experiment_plan, resolve_experiment_matrix
from ..models import (
    ExecutionSpec,
    Experiment,
    ExperimentSpec,
    Metadata,
    MlflowSpec,
    ModelSpec,
    OverrideImagesSpec,
    OverrideLlmdSpec,
    OverrideRhoaiSpec,
    OverrideRuntimeSpec,
    OverrideScaleSpec,
    OverrideSpec,
    StageSpec,
    ValidationError,
    normalize_model_names,
    sanitize_name,
)
from ..plans import resolve_run_plan


def dump_yaml(data) -> str:
    return yaml.safe_dump(data, sort_keys=False, width=1_000_000)


def dump(data, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(data, indent=2, sort_keys=True)
    return dump_yaml(data)


def invoke_handler(
    handler: Callable[[argparse.Namespace], int], **kwargs: object
) -> int:
    return handler(argparse.Namespace(**kwargs))


def repo_root_from(args: argparse.Namespace) -> Path:
    if getattr(args, "repo_root", None):
        return Path(args.repo_root).resolve()
    return discover_repo_root(Path.cwd())


def profiles_dir_from(args: argparse.Namespace) -> Path:
    if getattr(args, "profiles_dir", None):
        return Path(args.profiles_dir).resolve()
    return repo_root_from(args) / "profiles"


def parse_mapping(
    values: list[str] | tuple[str, ...] | None, option_name: str
) -> dict[str, str]:
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


def parse_version_overrides(
    values: list[str] | tuple[str, ...] | None,
) -> dict[str, str]:
    return parse_mapping(values, "--version-override")


def parse_axis_strings(
    values: tuple[str, ...] | list[str] | None, option_name: str
) -> str | list[str] | None:
    cleaned = [str(value).strip() for value in (values or []) if str(value).strip()]
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned[0]
    return cleaned


def _first_model_name(value: str | list[str]) -> str:
    return normalize_model_names(value, "spec.model.name")[0]


def parse_axis_ints(
    values: tuple[int, ...] | list[int] | None, option_name: str
) -> int | list[int] | None:
    if not values:
        return None
    parsed = [int(value) for value in values]
    if len(parsed) == 1:
        return parsed[0]
    return parsed


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
            deployment_profile=[],
            benchmark_profile=[],
            metrics_profile=["detailed"],
            namespace="benchflow",
            service_account="benchflow-runner",
            ttl_seconds_after_finished=3600,
            stages=stages,
            mlflow=mlflow,
            execution=ExecutionSpec(),
            overrides=OverrideSpec(),
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

    cli_model = parse_axis_strings(getattr(args, "model", None), "--model")
    model_name = cli_model if cli_model is not None else base_experiment.spec.model.name
    if not model_name:
        raise ValidationError(
            "missing required input: provide an experiment file or --model"
        )

    deployment_profile = (
        [getattr(args, "deployment_profile", None)]
        if getattr(args, "deployment_profile", None)
        else list(base_experiment.spec.deployment_profile)
    )
    if not deployment_profile:
        raise ValidationError(
            "missing required input: provide an experiment file or --deployment-profile"
        )

    benchmark_profile = (
        [getattr(args, "benchmark_profile", None)]
        if getattr(args, "benchmark_profile", None)
        else list(base_experiment.spec.benchmark_profile)
    )
    if not benchmark_profile:
        raise ValidationError(
            "missing required input: provide an experiment file or --benchmark-profile"
        )

    metrics_profile = (
        [getattr(args, "metrics_profile", None)]
        if getattr(args, "metrics_profile", None)
        else list(base_experiment.spec.metrics_profile)
    )

    name = (
        getattr(args, "name", None)
        or base_experiment.metadata.name
        or sanitize_name(_first_model_name(model_name))
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

    runtime_image = parse_axis_strings(
        getattr(args, "runtime_image", None), "--runtime-image"
    )
    scheduler_image = parse_axis_strings(
        getattr(args, "scheduler_image", None), "--scheduler-image"
    )
    replicas = parse_axis_ints(getattr(args, "replicas", None), "--replicas")
    tensor_parallelism = parse_axis_ints(getattr(args, "tp", None), "--tp")
    llmd_repo_ref = parse_axis_strings(
        getattr(args, "llmd_repo_ref", None), "--llmd-repo-ref"
    )
    cli_env = parse_mapping(getattr(args, "env", None), "--env")
    cli_vllm_args = [str(item) for item in (getattr(args, "vllm_arg", None) or [])]

    overrides = OverrideSpec(
        images=OverrideImagesSpec(
            runtime=(
                runtime_image
                if runtime_image is not None
                else base_experiment.spec.overrides.images.runtime
            ),
            scheduler=(
                scheduler_image
                if scheduler_image is not None
                else base_experiment.spec.overrides.images.scheduler
            ),
        ),
        scale=OverrideScaleSpec(
            replicas=(
                replicas
                if replicas is not None
                else base_experiment.spec.overrides.scale.replicas
            ),
            tensor_parallelism=(
                tensor_parallelism
                if tensor_parallelism is not None
                else base_experiment.spec.overrides.scale.tensor_parallelism
            ),
        ),
        runtime=OverrideRuntimeSpec(
            vllm_args=[
                *base_experiment.spec.overrides.runtime.vllm_args,
                *cli_vllm_args,
            ],
            env={
                **base_experiment.spec.overrides.runtime.env,
                **cli_env,
            },
        ),
        llm_d=OverrideLlmdSpec(
            repo_ref=(
                llmd_repo_ref
                if llmd_repo_ref is not None
                else base_experiment.spec.overrides.llm_d.repo_ref
            )
        ),
        rhoai=OverrideRhoaiSpec(
            enable_auth=(
                getattr(args, "rhoai_auth", None)
                if getattr(args, "rhoai_auth", None) is not None
                else base_experiment.spec.overrides.rhoai.enable_auth
            )
        ),
    )

    return Experiment(
        api_version=base_experiment.api_version,
        kind="Experiment",
        metadata=Metadata(name=name, labels=labels),
        spec=ExperimentSpec(
            model=ModelSpec(name=model_name),
            deployment_profile=deployment_profile,
            benchmark_profile=benchmark_profile,
            metrics_profile=metrics_profile,
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
            execution=ExecutionSpec(backend=base_experiment.spec.execution.backend),
            overrides=overrides,
        ),
    )


def load_plan(args: argparse.Namespace):
    experiment = require_single_experiment_plan(experiment_from_args(args))
    catalog = ProfileCatalog.load(profiles_dir_from(args))
    return resolve_run_plan(experiment, catalog)


def load_plans(args: argparse.Namespace):
    experiment = experiment_from_args(args)
    catalog = ProfileCatalog.load(profiles_dir_from(args))
    return resolve_experiment_matrix(experiment, catalog)


def load_runtime_plan(args: argparse.Namespace):
    run_plan_file = getattr(args, "run_plan_file", None)
    run_plan_json = getattr(args, "run_plan_json", None)
    if run_plan_file or run_plan_json:
        return load_run_plan_from_sources(
            run_plan_file=run_plan_file, run_plan_json=run_plan_json
        )
    return load_plan(args)


def _completion_repo_root(ctx: click.Context) -> Path:
    repo_root = ctx.params.get("repo_root")
    if repo_root:
        return Path(repo_root).resolve()
    return discover_repo_root(Path.cwd())


def _completion_profiles_dir(ctx: click.Context) -> Path:
    profiles_dir = ctx.params.get("profiles_dir")
    if profiles_dir:
        return Path(profiles_dir).resolve()
    return _completion_repo_root(ctx) / "profiles"


def complete_profile_names(
    kind: str | None = None,
) -> Callable[[click.Context, click.Parameter, str], list[CompletionItem]]:
    def _complete(
        ctx: click.Context, param: click.Parameter, incomplete: str
    ) -> list[CompletionItem]:
        profiles_dir = _completion_profiles_dir(ctx)
        names = sorted(
            {
                entry.name
                for entry in list_profile_entries(profiles_dir)
                if kind in (None, entry.kind)
            }
        )
        return [
            CompletionItem(name)
            for name in names
            if not incomplete or name.startswith(incomplete)
        ]

    return _complete


def apply_click_options(
    decorators: list[Callable[[Callable[..., object]], Callable[..., object]]],
) -> Callable[[Callable[..., object]], Callable[..., object]]:
    def _decorate(func: Callable[..., object]) -> Callable[..., object]:
        for decorator in reversed(decorators):
            func = decorator(func)
        return func

    return _decorate


def profile_source_options(func: Callable[..., object]) -> Callable[..., object]:
    return apply_click_options(
        [
            click.option(
                "--repo-root",
                type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
                help="BenchFlow repository root. Defaults to the current checkout.",
            ),
            click.option(
                "--profiles-dir",
                type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
                help="Profiles directory. Defaults to <repo-root>/profiles.",
            ),
        ]
    )(func)


def experiment_input_options(func: Callable[..., object]) -> Callable[..., object]:
    decorators: list[Callable[[Callable[..., object]], Callable[..., object]]] = [
        click.argument(
            "experiment",
            required=False,
            type=click.Path(dir_okay=False, path_type=Path),
        ),
        click.option(
            "--repo-root",
            type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
            help="BenchFlow repository root. Defaults to the current checkout.",
        ),
        click.option(
            "--profiles-dir",
            type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
            help="Profiles directory. Defaults to <repo-root>/profiles.",
        ),
        click.option("--namespace", help="Target namespace for the run."),
        click.option("--name", help="Experiment name override."),
        click.option(
            "--label",
            multiple=True,
            metavar="KEY=VALUE",
            help="Experiment label override. Repeat to set multiple labels.",
        ),
        click.option(
            "--model",
            multiple=True,
            help="Model identifier. Repeat to build a matrix axis.",
        ),
        click.option(
            "--deployment-profile",
            shell_complete=complete_profile_names("deployment"),
            help="Deployment profile name.",
        ),
        click.option(
            "--benchmark-profile",
            shell_complete=complete_profile_names("benchmark"),
            help="Benchmark profile name.",
        ),
        click.option(
            "--metrics-profile",
            shell_complete=complete_profile_names("metrics"),
            help="Metrics profile name.",
        ),
        click.option(
            "--service-account",
            help="Service account used by the execution.",
        ),
        click.option(
            "--ttl-seconds-after-finished",
            type=int,
            help="TTL for finished executions.",
        ),
        click.option(
            "--mlflow-experiment",
            help="MLflow experiment name override.",
        ),
        click.option(
            "--mlflow-tag",
            multiple=True,
            metavar="KEY=VALUE",
            help="MLflow tag override. Repeat to set multiple tags.",
        ),
        click.option(
            "--runtime-image",
            "runtime_image",
            multiple=True,
            help="Override the runtime image. Repeat to build a matrix axis.",
        ),
        click.option(
            "--scheduler-image",
            "scheduler_image",
            multiple=True,
            help="Override the scheduler image. Repeat to build a matrix axis.",
        ),
        click.option(
            "--replicas",
            type=int,
            multiple=True,
            help="Override the replica count. Repeat to build a matrix axis.",
        ),
        click.option(
            "--tp",
            type=int,
            multiple=True,
            help="Override tensor parallelism. Repeat to build a matrix axis.",
        ),
        click.option(
            "--vllm-arg",
            multiple=True,
            help="Append one vLLM argument to the profile defaults. Repeat as needed.",
        ),
        click.option(
            "--env",
            multiple=True,
            metavar="KEY=VALUE",
            help="Runtime environment override. Repeat to set multiple variables.",
        ),
        click.option(
            "--llmd-repo-ref",
            multiple=True,
            help="Override the llm-d repository ref. Repeat to build a matrix axis.",
        ),
        click.option(
            "--rhoai-auth/--no-rhoai-auth",
            "rhoai_auth",
            default=None,
            show_default=False,
            help="Override RHOAI auth handling.",
        ),
    ]
    for stage_name in ("download", "deploy", "benchmark", "collect", "cleanup"):
        decorators.append(
            click.option(
                f"--{stage_name}/--no-{stage_name}",
                f"stage_{stage_name}",
                default=None,
                show_default=False,
                help=f"Enable or disable the {stage_name} stage.",
            )
        )
    return apply_click_options(decorators)(func)


def runtime_plan_source_options(func: Callable[..., object]) -> Callable[..., object]:
    return apply_click_options(
        [
            click.option(
                "--run-plan-file",
                type=click.Path(dir_okay=False, path_type=Path),
                help="Path to a pre-resolved RunPlan file.",
            ),
            click.option(
                "--run-plan-json",
                help="Inline RunPlan JSON payload.",
            ),
        ]
    )(experiment_input_options(func))


def run_plan_source_options(func: Callable[..., object]) -> Callable[..., object]:
    return apply_click_options(
        [
            click.option(
                "--run-plan-file",
                type=click.Path(dir_okay=False, path_type=Path),
                help="Path to a resolved RunPlan file.",
            ),
            click.option(
                "--run-plan-json",
                help="Inline RunPlan JSON payload.",
            ),
        ]
    )(func)


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


def format_experiment_list(entries: list[dict[str, object]]) -> str:
    if not entries:
        return ""

    status_width = max(
        len("STATUS"), max(len(str(entry["status"])) for entry in entries)
    )
    name_width = max(len("RUN"), max(len(str(entry["name"])) for entry in entries))
    experiment_width = max(
        len("EXPERIMENT"), max(len(str(entry["experiment"])) for entry in entries)
    )
    platform_width = max(
        len("PLATFORM"), max(len(str(entry["platform"])) for entry in entries)
    )
    mode_width = max(len("MODE"), max(len(str(entry["mode"])) for entry in entries))
    started_width = max(
        len("STARTED"), max(len(str(entry["start_time"])) for entry in entries)
    )

    lines = [
        f"{'STATUS':<{status_width}}  {'RUN':<{name_width}}  "
        f"{'EXPERIMENT':<{experiment_width}}  {'PLATFORM':<{platform_width}}  "
        f"{'MODE':<{mode_width}}  {'STARTED':<{started_width}}",
    ]
    for entry in entries:
        lines.append(
            f"{entry['status']:<{status_width}}  {entry['name']:<{name_width}}  "
            f"{entry['experiment']:<{experiment_width}}  {entry['platform']:<{platform_width}}  "
            f"{entry['mode']:<{mode_width}}  "
            f"{entry['start_time']:<{started_width}}"
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
        kind_label = kind or "any"
        raise ValidationError(f"no {kind_label} profile named {name!r}")
    if len(matches) > 1:
        kinds = ", ".join(sorted(entry.kind for entry in matches))
        raise ValidationError(
            f"profile name {name!r} is ambiguous; specify --kind (matches: {kinds})"
        )
    return load_yaml_file(Path(matches[0].path))
