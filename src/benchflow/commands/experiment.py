from __future__ import annotations

import argparse
import json
from pathlib import Path

import click

from ..cluster import CommandError, get_current_namespace
from ..contracts import StageSpec
from ..loaders import load_run_plan_data
from ..orchestration import (
    MATRIX_PARENT_EXECUTION_LABEL,
    cancel_execution,
    follow_execution,
    list_benchflow_executions,
    list_execution_payloads,
    render_execution_manifest,
    render_matrix_execution_manifest,
    submit_execution_manifest,
    summarize_execution,
    get_execution,
)
from ..renderers.deployment import write_deployment_assets
from ..ui import detail, step, success
from .shared import (
    dump,
    dump_yaml,
    experiment_input_options,
    format_experiment_list,
    invoke_handler,
    load_plan,
    load_plans,
)


def _namespace_from_args(args: argparse.Namespace) -> str:
    return args.namespace or get_current_namespace()


def cmd_validate(args: argparse.Namespace) -> int:
    load_plans(args)
    print("valid")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    namespace = _namespace_from_args(args)
    entries = list_benchflow_executions(namespace, include_completed=True)
    if args.format == "json":
        print(dump(entries, "json"))
        return 0
    if args.format == "yaml":
        print(dump(entries, "yaml"))
        return 0
    if not entries:
        print("No BenchFlow experiments found.")
        return 0
    print(format_experiment_list(entries))
    return 0


def _resolve_cancel_targets(
    namespace: str, identifier: str, *, cancel_all: bool
) -> list[dict[str, object]]:
    try:
        exact = summarize_execution(namespace, identifier)
    except CommandError:
        exact = None
    else:
        if not exact.get("experiment"):
            raise CommandError(
                f"execution {identifier} exists in {namespace} but is not a BenchFlow experiment"
            )
        if exact.get("finished"):
            raise CommandError(f"execution {identifier} is already finished")
        return [exact]

    entries = list_benchflow_executions(namespace, include_completed=False)
    matches = [entry for entry in entries if entry.get("experiment") == identifier]
    if not matches:
        raise CommandError(
            f"no active BenchFlow experiment found for {identifier!r} in namespace {namespace}"
        )
    if len(matches) > 1 and not cancel_all:
        names = ", ".join(str(entry["name"]) for entry in matches)
        raise CommandError(
            f"multiple active executions match experiment {identifier!r}: {names}; "
            "use the execution name or pass --all"
        )
    return matches if cancel_all else [matches[0]]


def cmd_cancel(args: argparse.Namespace) -> int:
    namespace = _namespace_from_args(args)
    targets = _resolve_cancel_targets(
        namespace,
        args.identifier,
        cancel_all=args.all_matches,
    )
    step(
        f"Cancelling {len(targets)} BenchFlow execution"
        f"{'' if len(targets) == 1 else 's'} in namespace {namespace}"
    )
    for target in targets:
        name = str(target["name"])
        detail(f"{name} (experiment={target['experiment']}, status={target['status']})")
        cancel_execution(namespace, name)
        success(f"Cancellation requested for {name}")
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    plans = load_plans(args)
    payload: object
    if len(plans) == 1:
        payload = plans[0].to_dict()
    else:
        payload = [plan.to_dict() for plan in plans]
    print(dump(payload, args.format))
    return 0


def cmd_render_pipelinerun(args: argparse.Namespace) -> int:
    plans = load_plans(args)
    if len(plans) == 1:
        manifest = render_execution_manifest(
            plans[0],
            execution_name=args.pipeline_name,
            benchflow_image=args.benchflow_image,
        )
    else:
        manifest = render_matrix_execution_manifest(
            plans,
            child_execution_name=args.pipeline_name,
            benchflow_image=args.benchflow_image,
        )
    print(dump_yaml(manifest))
    return 0


def cmd_render_deployment(args: argparse.Namespace) -> int:
    plan = load_plan(args)
    output_dir = Path(args.output_dir).resolve()
    written = write_deployment_assets(plan, output_dir)
    for path in written:
        print(path)
    return 0


def _pipeline_param_value(payload: dict[str, object], param_name: str) -> str:
    params = (payload.get("spec", {}) or {}).get("params", []) or []
    for param in params:
        if str((param or {}).get("name") or "").strip() != param_name:
            continue
        return str((param or {}).get("value") or "").strip()
    return ""


def _pipeline_ref_name(payload: dict[str, object]) -> str:
    spec = payload.get("spec", {}) or {}
    pipeline_ref = spec.get("pipelineRef", {}) or {}
    return str(pipeline_ref.get("name") or "").strip()


def _execution_status_matches(summary: dict[str, object], status: str) -> bool:
    if not bool(summary.get("finished")):
        return False
    if status == "all":
        return True
    succeeded = bool(summary.get("succeeded"))
    if status == "failed":
        return not succeeded
    if status == "succeeded":
        return succeeded
    raise CommandError(f"unsupported status selector: {status}")


def _execution_run_plan(namespace: str, execution_name: str):
    payload = get_execution(namespace, execution_name)
    run_plan_json = _pipeline_param_value(payload, "RUN_PLAN")
    if not run_plan_json:
        raise CommandError(
            f"execution {execution_name} in {namespace} does not expose a RUN_PLAN param"
        )
    try:
        run_plan_payload = json.loads(run_plan_json)
    except json.JSONDecodeError as exc:
        raise CommandError(
            f"execution {execution_name} in {namespace} has an invalid RUN_PLAN payload"
        ) from exc
    return load_run_plan_data(run_plan_payload)


def _rerun_execution(args: argparse.Namespace) -> int:
    namespace = _namespace_from_args(args)
    execution_ref = str(args.experiment or "").strip()
    if not execution_ref:
        raise CommandError("execution name is required when using --status")

    summary = summarize_execution(namespace, execution_ref)
    if not summary.get("finished"):
        raise CommandError(f"execution {execution_ref} is not finished yet")

    payload = get_execution(namespace, execution_ref)
    labels = (payload.get("metadata", {}) or {}).get("labels", {}) or {}
    is_matrix = (
        str(labels.get("benchflow.io/platform") or "").strip() == "matrix"
        and str(labels.get("benchflow.io/mode") or "").strip() == "matrix"
    )

    if not is_matrix:
        if not _execution_status_matches(summary, args.status):
            raise CommandError(
                f"execution {execution_ref} does not match --status {args.status}"
            )
        plan = _execution_run_plan(namespace, execution_ref)
        manifest = render_execution_manifest(
            plan,
            execution_name=_pipeline_ref_name(payload) or args.pipeline_name,
            benchflow_image=args.benchflow_image,
        )
        if args.output:
            Path(args.output).resolve().write_text(
                dump_yaml(manifest), encoding="utf-8"
            )
        name = submit_execution_manifest(manifest, namespace)
        print(name)
        if args.follow:
            return 0 if follow_execution(namespace, name) else 1
        return 0

    child_payloads = list_execution_payloads(
        namespace,
        label_selector=f"{MATRIX_PARENT_EXECUTION_LABEL}={execution_ref}",
    )
    if not child_payloads:
        raise CommandError(
            f"matrix execution {execution_ref} has no linked child executions; "
            "only matrix runs submitted after parent-child linkage support can be rerun this way"
        )

    matching_children: list[tuple[str, object]] = []
    for child_payload in child_payloads:
        child_name = str(
            ((child_payload.get("metadata", {}) or {}).get("name") or "")
        ).strip()
        if not child_name:
            continue
        child_summary = summarize_execution(namespace, child_name)
        if _execution_status_matches(child_summary, args.status):
            matching_children.append(
                (child_name, _execution_run_plan(namespace, child_name))
            )

    if not matching_children:
        raise CommandError(
            f"matrix execution {execution_ref} has no child executions matching --status {args.status}"
        )

    child_pipeline_name = (
        _pipeline_param_value(payload, "CHILD_PIPELINE_NAME") or "benchflow-e2e"
    )
    plans = [plan for _, plan in matching_children]
    if len(plans) == 1:
        manifest = render_execution_manifest(
            plans[0],
            execution_name=child_pipeline_name,
            benchflow_image=args.benchflow_image,
        )
    else:
        manifest = render_matrix_execution_manifest(
            plans,
            child_execution_name=child_pipeline_name,
            benchflow_image=args.benchflow_image,
        )
    if args.output:
        Path(args.output).resolve().write_text(dump_yaml(manifest), encoding="utf-8")
    name = submit_execution_manifest(manifest, namespace)
    print(name)
    if args.follow:
        return 0 if follow_execution(namespace, name) else 1
    return 0


def _render_manifest_yaml(
    args: argparse.Namespace, *, cleanup_only: bool = False
) -> tuple[object, str, str]:
    plans = load_plans(args)
    if cleanup_only and len(plans) != 1:
        raise CommandError(
            "cleanup only supports a single profile combination; "
            "use a single-profile experiment or override the profiles on the CLI"
        )
    plan = plans[0]
    if cleanup_only:
        plan.stages = StageSpec(
            download=False,
            deploy=False,
            benchmark=False,
            collect=False,
            cleanup=True,
        )

    if len(plans) == 1:
        manifest = render_execution_manifest(
            plan,
            execution_name=args.pipeline_name,
            benchflow_image=args.benchflow_image,
        )
        namespace = plan.deployment.namespace
    else:
        manifest = render_matrix_execution_manifest(
            plans,
            child_execution_name=args.pipeline_name,
            benchflow_image=args.benchflow_image,
        )
        namespace = plan.deployment.namespace
    manifest_yaml = dump_yaml(manifest)
    return plans if len(plans) > 1 else plan, manifest_yaml, namespace


def cmd_run(args: argparse.Namespace) -> int:
    if args.status:
        experiment_arg = getattr(args, "experiment", None)
        if experiment_arg is not None and Path(experiment_arg).exists():
            raise CommandError(
                "--status only applies when the positional argument is an execution name, not an experiment file"
            )
        return _rerun_execution(args)

    plan_or_plans, manifest_yaml, namespace = _render_manifest_yaml(args)

    if args.output:
        Path(args.output).resolve().write_text(manifest_yaml, encoding="utf-8")

    name = submit_execution_manifest(
        render_execution_manifest(
            plan_or_plans,
            execution_name=args.pipeline_name,
            benchflow_image=args.benchflow_image,
        )
        if hasattr(plan_or_plans, "execution")
        else render_matrix_execution_manifest(
            plan_or_plans,
            child_execution_name=args.pipeline_name,
            benchflow_image=args.benchflow_image,
        ),
        namespace,
    )
    print(f"workflow/{name}")

    if args.follow:
        return 0 if follow_execution(namespace, name) else 1
    return 0


def cmd_cleanup(args: argparse.Namespace) -> int:
    plan_or_plans, manifest_yaml, namespace = _render_manifest_yaml(
        args, cleanup_only=True
    )

    if args.output:
        Path(args.output).resolve().write_text(manifest_yaml, encoding="utf-8")

    name = submit_execution_manifest(
        render_execution_manifest(
            plan_or_plans,
            execution_name=args.pipeline_name,
            benchflow_image=args.benchflow_image,
        ),
        namespace,
    )
    print(name)

    if args.follow:
        return 0 if follow_execution(namespace, name) else 1
    return 0


@click.group(
    "experiment",
    help="Validate, resolve, render, submit, or cancel BenchFlow experiments.",
    short_help="Work with experiment definitions",
)
def experiment_group() -> None:
    pass


@experiment_group.command(
    "list",
    help="List BenchFlow executions in a namespace. Finished runs are shown by default.",
    short_help="List experiments in the cluster",
)
@click.option(
    "--namespace",
    help="Namespace to inspect. Defaults to the current oc project.",
)
@click.option(
    "--format",
    "format",
    type=click.Choice(("table", "yaml", "json")),
    default="table",
    show_default=True,
    help="Output format.",
)
def experiment_list(**kwargs: object) -> int:
    return invoke_handler(cmd_list, **kwargs)


@experiment_group.command(
    "cancel",
    help=(
        "Cancel a running BenchFlow execution by execution name or by "
        "experiment name label."
    ),
    short_help="Cancel a running experiment",
)
@click.argument("identifier")
@click.option(
    "--namespace",
    help="Namespace that contains the execution. Defaults to the current oc project.",
)
@click.option(
    "--all",
    "all_matches",
    is_flag=True,
    help="Cancel all active executions that match the experiment name.",
)
def experiment_cancel(**kwargs: object) -> int:
    return invoke_handler(cmd_cancel, **kwargs)


@experiment_group.command(
    "validate",
    help="Validate an experiment file or CLI-defined experiment.",
    short_help="Validate an experiment definition",
)
@experiment_input_options
def experiment_validate(**kwargs: object) -> int:
    return invoke_handler(cmd_validate, **kwargs)


@experiment_group.command(
    "resolve",
    help=(
        "Resolve an experiment into the fully expanded RunPlan used by BenchFlow. "
        "Matrix experiments resolve to a list of RunPlans."
    ),
    short_help="Resolve profiles into a RunPlan",
)
@experiment_input_options
@click.option(
    "--format",
    "format",
    type=click.Choice(("yaml", "json")),
    default="yaml",
    show_default=True,
    help="Output format.",
)
def experiment_resolve(**kwargs: object) -> int:
    return invoke_handler(cmd_resolve, **kwargs)


@experiment_group.command(
    "render-pipelinerun",
    help=(
        "Render the PipelineRun manifest that would be submitted for an experiment. "
        "Matrix experiments render the supervisor PipelineRun."
    ),
    short_help="Render a PipelineRun manifest",
)
@experiment_input_options
@click.option(
    "--pipeline-name",
    default="benchflow-e2e",
    show_default=True,
    help="Pipeline name to reference in the rendered PipelineRun.",
)
@click.option(
    "--benchflow-image",
    help="BenchFlow control image to use for all Pipeline tasks.",
)
def experiment_render_pipelinerun(**kwargs: object) -> int:
    return invoke_handler(cmd_render_pipelinerun, **kwargs)


@experiment_group.command(
    "render-deployment",
    help="Render deployment assets for an experiment without submitting a run.",
    short_help="Render deployment manifests to disk",
)
@experiment_input_options
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory where the rendered deployment assets should be written.",
)
def experiment_render_deployment(**kwargs: object) -> int:
    return invoke_handler(cmd_render_deployment, **kwargs)


@experiment_group.command(
    "run",
    help=(
        "Submit an experiment to the cluster and optionally follow it. "
        "Matrix experiments submit a supervisor execution that submits child "
        "combinations and lets Kueue admit them when capacity is available."
    ),
    short_help="Submit an experiment as an execution",
)
@experiment_input_options
@click.option(
    "--pipeline-name",
    default="benchflow-e2e",
    show_default=True,
    help="Pipeline name to reference when rendering the PipelineRun.",
)
@click.option(
    "--benchflow-image",
    help="BenchFlow control image to use for all Pipeline tasks.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write the rendered PipelineRun manifest to this file before submitting.",
)
@click.option(
    "--follow",
    is_flag=True,
    help="Follow the execution after submission.",
)
@click.option(
    "--status",
    type=click.Choice(("failed", "succeeded", "all")),
    help=(
        "Rerun a previous execution by status when the positional argument is "
        "a BenchFlow execution name instead of an experiment file."
    ),
)
def experiment_run(**kwargs: object) -> int:
    return invoke_handler(cmd_run, **kwargs)


@experiment_group.command(
    "cleanup",
    help="Submit a cleanup-only run for an experiment.",
    short_help="Submit a cleanup-only execution",
)
@experiment_input_options
@click.option(
    "--pipeline-name",
    default="benchflow-e2e",
    show_default=True,
    help="Pipeline name to reference when rendering the cleanup PipelineRun.",
)
@click.option(
    "--benchflow-image",
    help="BenchFlow control image to use for all Pipeline tasks.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write the rendered cleanup PipelineRun manifest to this file before submitting.",
)
@click.option(
    "--follow/--no-follow",
    default=True,
    show_default=True,
    help="Follow the cleanup execution after submission.",
)
def experiment_cleanup(**kwargs: object) -> int:
    return invoke_handler(cmd_cleanup, **kwargs)
