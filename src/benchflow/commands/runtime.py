from __future__ import annotations

import argparse
import base64
import ipaddress
import json
import os
import re
from pathlib import Path

import click

from ..benchmark import (
    BenchmarkRunFailed,
    benchmark_version_from_plan,
)
from ..cluster import (
    TARGET_KUBECONFIG_HOST_ALIASES_ANNOTATION,
    get_current_namespace,
    require_any_command,
    run_command,
)
from ..contracts import ExecutionContext, ValidationError
from ..orchestration import (
    follow_execution,
    list_execution_steps,
    load_run_plan_from_sources,
    require_platform,
    run_matrix_supervisor,
    stream_execution_logs,
)
from ..remote_jobs import remote_run_plan_json, run_remote_job
from ..install import (
    BootstrapOptions,
    reconcile_management_cluster_queue,
    run_bootstrap,
)
from ..kueue import (
    DEFAULT_CONTROLLER_IMAGE,
    LOCAL_CLUSTER_QUEUE,
    discover_cluster_gpu_capacity,
    run_remote_capacity_controller,
)
from ..loaders import load_run_plan_data
from ..repository import clone_repo
from ..tasking import assert_task_status
from ..toolbox import (
    cleanup_deployment,
    cleanup_run_plan,
    collect_plan_artifacts,
    collect_plan_metrics,
    deploy_platform,
    download_cached_model,
    generate_plan_report,
    resolve_run_plan_stages,
    resolve_target_url,
    run_plan_benchmark,
    setup_platform,
    teardown_platform,
    upload_artifact_directory,
    upload_plan_results,
)
from ..ui import detail, warning
from ..waiting import wait_for_endpoint
from .shared import (
    invoke_handler,
    load_runtime_plan,
    parse_mapping,
    parse_version_overrides,
    repo_root_from,
    runtime_plan_source_options,
)


def _execution_context(
    *,
    execution_name: str = "",
    workspace_dir: str | Path | None = None,
    manifests_dir: str | Path | None = None,
    models_storage_path: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    state_path: str | Path | None = None,
) -> ExecutionContext:
    def _resolve(value: str | Path | None) -> Path | None:
        if value is None:
            return None
        return Path(value).resolve()

    return ExecutionContext(
        execution_name=execution_name,
        workspace_dir=_resolve(workspace_dir),
        manifests_dir=_resolve(manifests_dir),
        models_storage_path=_resolve(models_storage_path),
        artifacts_dir=_resolve(artifacts_dir),
        state_path=_resolve(state_path),
    )


def _teardown_requested(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise ValidationError(f"invalid teardown value: {value!r}")


_DNS_SUBDOMAIN_RE = re.compile(r"^[a-z0-9]([-.a-z0-9]*[a-z0-9])?$")


def _normalize_cluster_name(cluster_name: str) -> str:
    normalized = str(cluster_name).strip()
    if not normalized:
        raise ValidationError("cluster name must not be empty")
    if len(normalized) > 253 or not _DNS_SUBDOMAIN_RE.fullmatch(normalized):
        raise ValidationError(
            "cluster name must be a valid lowercase DNS subdomain, for example psap-llmd-h200"
        )
    return normalized


def _parse_host_aliases(values: tuple[str, ...] | list[str] | None) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for raw in values or ():
        item = str(raw).strip()
        if not item:
            continue
        hostname, separator, ip_address = item.partition("=")
        hostname = hostname.strip()
        ip_address = ip_address.strip()
        if not separator or not hostname or not ip_address:
            raise ValidationError("host alias must be in the form hostname=ip-address")
        try:
            ipaddress.ip_address(ip_address)
        except ValueError as exc:
            raise ValidationError(
                f"invalid IP address for host alias {hostname!r}: {ip_address!r}"
            ) from exc
        aliases[hostname] = ip_address
    return aliases


def _apply_target_kubeconfig_secret(
    *,
    namespace: str,
    secret_name: str,
    kubeconfig_path: Path,
    cluster_name: str | None = None,
    host_aliases: dict[str, str] | None = None,
) -> None:
    kubectl_cmd = require_any_command("oc", "kubectl")
    labels = {
        "app.kubernetes.io/name": "benchflow",
        "benchflow.io/secret-kind": "target-kubeconfig",
    }
    if cluster_name:
        labels["benchflow.io/cluster-name"] = cluster_name

    namespace_payload = {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {"name": namespace},
    }
    run_command(
        [kubectl_cmd, "apply", "-f", "-"],
        input_text=json.dumps(namespace_payload),
    )

    payload = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {
            "name": secret_name,
            "namespace": namespace,
            "labels": labels,
            "annotations": (
                {
                    TARGET_KUBECONFIG_HOST_ALIASES_ANNOTATION: json.dumps(
                        host_aliases or {}, separators=(",", ":"), sort_keys=True
                    )
                }
                if host_aliases
                else {}
            ),
        },
        "type": "Opaque",
        "data": {
            "kubeconfig": base64.b64encode(kubeconfig_path.read_bytes()).decode("ascii")
        },
    }
    run_command(
        [kubectl_cmd, "apply", "-n", namespace, "-f", "-"],
        input_text=json.dumps(payload),
    )


def cmd_bootstrap(args: argparse.Namespace) -> int:
    repo_root = repo_root_from(args)
    namespace = args.namespace or "benchflow"
    benchflow_image = str(args.benchflow_image or DEFAULT_CONTROLLER_IMAGE).strip()
    if not benchflow_image:
        raise ValidationError("benchflow image must not be empty")
    target_kubeconfig = (
        str(Path(args.target_kubeconfig).resolve()) if args.target_kubeconfig else None
    )
    host_aliases = _parse_host_aliases(getattr(args, "host_alias", None))
    single_cluster = bool(args.single_cluster)
    cluster_name = (
        _normalize_cluster_name(args.cluster_name) if args.cluster_name else None
    )
    if single_cluster and target_kubeconfig:
        raise ValidationError(
            "--single-cluster cannot be used together with --target-kubeconfig"
        )
    if single_cluster and cluster_name:
        raise ValidationError(
            "--single-cluster cannot be used together with --cluster-name"
        )
    if cluster_name and not target_kubeconfig:
        raise ValidationError(
            "--cluster-name requires --target-kubeconfig during bootstrap"
        )
    if host_aliases and not (target_kubeconfig and cluster_name):
        raise ValidationError(
            "--host-alias requires --target-kubeconfig together with --cluster-name"
        )
    if target_kubeconfig and cluster_name:
        _apply_target_kubeconfig_secret(
            namespace=namespace,
            secret_name=cluster_name,
            kubeconfig_path=Path(target_kubeconfig),
            cluster_name=cluster_name,
            host_aliases=host_aliases,
        )
    remote_target = bool(target_kubeconfig)
    install_tekton = (
        args.install_tekton
        if args.install_tekton is not None
        else (single_cluster or not remote_target)
    )
    install_grafana = (
        args.install_grafana
        if args.install_grafana is not None
        else (single_cluster or not remote_target)
    )
    result = run_bootstrap(
        repo_root,
        BootstrapOptions(
            namespace=namespace,
            single_cluster=single_cluster,
            install_grafana=install_grafana,
            install_tekton=install_tekton,
            install_kueue=(single_cluster or not remote_target),
            install_accelerator_prerequisites=(single_cluster or remote_target),
            install_models_storage=(single_cluster or remote_target),
            install_results_storage=True,
            target_kubeconfig=target_kubeconfig,
            benchflow_image=benchflow_image,
            models_storage_class=args.models_storage_class,
            models_storage_size=args.models_size or "250Gi",
            models_storage_access_mode=args.models_access_mode or "ReadWriteOnce",
            results_storage_class=args.results_storage_class,
            results_storage_size=args.results_size or "20Gi",
        ),
    )
    if result != 0:
        return result

    if single_cluster:
        gpu_capacity = discover_cluster_gpu_capacity()
        if gpu_capacity == 0:
            warning(
                "Discovered 0 GPUs while registering the local Kueue queue; rerun bootstrap once GPU resources are visible if this cluster should execute GPU workloads."
            )
        reconcile_management_cluster_queue(
            repo_root,
            namespace=namespace,
            cluster_name=LOCAL_CLUSTER_QUEUE,
            gpu_capacity=gpu_capacity,
            benchflow_image=benchflow_image,
        )
    elif target_kubeconfig and cluster_name:
        gpu_capacity = discover_cluster_gpu_capacity(target_kubeconfig)
        if gpu_capacity == 0:
            warning(
                f"Discovered 0 GPUs in target cluster {cluster_name!r}; rerun bootstrap once GPU resources are visible if this target cluster should execute GPU workloads."
            )
        reconcile_management_cluster_queue(
            repo_root,
            namespace=namespace,
            cluster_name=cluster_name,
            gpu_capacity=gpu_capacity,
            benchflow_image=benchflow_image,
        )
    return result


def cmd_target_kubeconfig_secret_create(args: argparse.Namespace) -> int:
    namespace = args.namespace or get_current_namespace()
    kubeconfig_path = Path(args.kubeconfig).expanduser().resolve()
    if not kubeconfig_path.is_file():
        raise ValidationError(f"kubeconfig file not found: {kubeconfig_path}")

    secret_name = str(args.name).strip()
    key_name = str(args.key or "kubeconfig").strip()
    host_aliases = _parse_host_aliases(getattr(args, "host_alias", None))
    if not secret_name:
        raise ValidationError("secret name must not be empty")
    if not key_name:
        raise ValidationError("secret key must not be empty")
    if key_name != "kubeconfig":
        kubectl_cmd = require_any_command("oc", "kubectl")
        payload = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": secret_name,
                "namespace": namespace,
                "labels": {
                    "app.kubernetes.io/name": "benchflow",
                    "benchflow.io/secret-kind": "target-kubeconfig",
                },
                "annotations": (
                    {
                        TARGET_KUBECONFIG_HOST_ALIASES_ANNOTATION: json.dumps(
                            host_aliases, separators=(",", ":"), sort_keys=True
                        )
                    }
                    if host_aliases
                    else {}
                ),
            },
            "type": "Opaque",
            "data": {
                key_name: base64.b64encode(kubeconfig_path.read_bytes()).decode("ascii")
            },
        }
        run_command(
            [kubectl_cmd, "apply", "-n", namespace, "-f", "-"],
            input_text=json.dumps(payload),
        )
    else:
        _apply_target_kubeconfig_secret(
            namespace=namespace,
            secret_name=secret_name,
            kubeconfig_path=kubeconfig_path,
            host_aliases=host_aliases,
        )
    print(secret_name)
    return 0


def _write_output_file(path_value: str | Path | None, content: str) -> None:
    if path_value is None:
        return
    path = Path(path_value).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _has_runtime_plan_source(args: argparse.Namespace) -> bool:
    return any(
        (
            getattr(args, "run_plan_file", None),
            getattr(args, "run_plan_json", None),
            getattr(args, "experiment", None),
            getattr(args, "model", None),
            getattr(args, "deployment_profile", None),
        )
    )


def cmd_watch(args: argparse.Namespace) -> int:
    namespace = args.namespace or get_current_namespace()
    return 0 if follow_execution(namespace, args.execution_name) else 1


def cmd_logs(args: argparse.Namespace) -> int:
    namespace = args.namespace or get_current_namespace()
    if args.all_logs and args.step:
        raise ValidationError("use either --step or --all, not both")

    if not args.all_logs and not args.step:
        steps = list_execution_steps(namespace, args.execution_name)
        if not steps:
            detail("No selectable pipeline tasks were found yet")
            return 0
        for step_name in steps:
            print(step_name)
        return 0

    stream_execution_logs(
        namespace,
        args.execution_name,
        step_name=args.step,
        all_logs=args.all_logs,
        all_containers=args.all_containers,
    )
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
    plan = load_runtime_plan(args)
    target_dir = download_cached_model(
        plan,
        context=_execution_context(models_storage_path=args.models_storage_path),
        skip_if_exists=not args.no_skip_if_exists,
    )
    print(target_dir)
    return 0


def cmd_setup_llmd(args: argparse.Namespace) -> int:
    plan = load_runtime_plan(args)
    require_platform(plan, "llm-d")
    state = setup_platform(
        plan,
        context=_execution_context(
            workspace_dir=args.workspace_dir,
            state_path=args.state_path,
        ),
    )
    if args.state_path:
        print(Path(args.state_path).resolve())
    else:
        print(json.dumps(state, sort_keys=True))
    return 0


def cmd_setup_rhoai(args: argparse.Namespace) -> int:
    plan = load_runtime_plan(args)
    require_platform(plan, "rhoai")
    state = setup_platform(
        plan,
        context=_execution_context(state_path=args.state_path),
    )
    if args.state_path:
        print(Path(args.state_path).resolve())
    else:
        print(json.dumps(state, sort_keys=True))
    return 0


def cmd_teardown_llmd(args: argparse.Namespace) -> int:
    plan = load_runtime_plan(args)
    require_platform(plan, "llm-d")
    teardown_platform(
        plan,
        {},
        context=_execution_context(
            workspace_dir=args.workspace_dir,
            state_path=args.state_path,
        ),
    )
    print(plan.deployment.release_name)
    return 0


def cmd_teardown_rhoai(args: argparse.Namespace) -> int:
    plan = load_runtime_plan(args)
    require_platform(plan, "rhoai")
    teardown_platform(plan, {}, context=_execution_context(state_path=args.state_path))
    print(plan.deployment.release_name)
    return 0


def cmd_deploy_llmd(args: argparse.Namespace) -> int:
    plan = load_runtime_plan(args)
    require_platform(plan, "llm-d")
    checkout_dir = deploy_platform(
        plan,
        context=_execution_context(
            execution_name=args.execution_name or "",
            workspace_dir=args.workspace_dir,
            manifests_dir=args.manifests_dir,
        ),
        skip_if_exists=not args.no_skip_if_exists,
        verify=not args.no_verify,
        verify_timeout_seconds=args.verify_timeout_seconds,
    )
    print(checkout_dir)
    return 0


def cmd_undeploy_llmd(args: argparse.Namespace) -> int:
    plan = load_runtime_plan(args)
    require_platform(plan, "llm-d")
    cleanup_deployment(
        plan,
        wait_for_deletion=not args.no_wait,
        timeout_seconds=args.timeout_seconds,
        skip_if_not_exists=not args.no_skip_if_not_exists,
    )
    print(plan.deployment.release_name)
    return 0


def cmd_deploy_rhoai(args: argparse.Namespace) -> int:
    plan = load_runtime_plan(args)
    require_platform(plan, "rhoai")
    output_dir = deploy_platform(
        plan,
        context=_execution_context(manifests_dir=args.manifests_dir),
        skip_if_exists=not args.no_skip_if_exists,
        verify=not args.no_verify,
        verify_timeout_seconds=args.verify_timeout_seconds,
    )
    print(output_dir)
    return 0


def cmd_undeploy_rhoai(args: argparse.Namespace) -> int:
    plan = load_runtime_plan(args)
    require_platform(plan, "rhoai")
    cleanup_deployment(
        plan,
        wait_for_deletion=not args.no_wait,
        timeout_seconds=args.timeout_seconds,
        skip_if_not_exists=not args.no_skip_if_not_exists,
    )
    print(plan.deployment.release_name)
    return 0


def cmd_wait_endpoint(args: argparse.Namespace) -> int:
    plan = load_runtime_plan(args) if _has_runtime_plan_source(args) else None
    if plan is not None and plan.target_cluster.enabled():
        remote_args = [
            "wait",
            "endpoint",
            "--run-plan-json",
            remote_run_plan_json(plan),
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--retry-interval",
            str(args.retry_interval),
        ]
        if args.target_url:
            remote_args.extend(["--target-url", args.target_url])
        if args.endpoint_path:
            remote_args.extend(["--endpoint-path", args.endpoint_path])
        if args.verify_tls:
            remote_args.append("--verify-tls")
        run_remote_job(
            plan,
            job_kind="wait-endpoint",
            args=remote_args,
            timeout_seconds=max(args.timeout_seconds + 300, 900),
        )
        print("ready")
        return 0

    target_url = args.target_url
    endpoint_path = args.endpoint_path
    if not target_url:
        if plan is None:
            plan = load_runtime_plan(args)
        target_url, endpoint_path = resolve_target_url(
            plan,
            target_url=args.target_url,
            endpoint_path=args.endpoint_path,
        )
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
    if plan.benchmark.tool != "guidellm":
        raise ValidationError(
            f"unsupported benchmark tool: {plan.benchmark.tool}; only guidellm is implemented"
        )
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    benchmark_target, _ = resolve_target_url(plan, target_url=args.target_url)
    run_id = ""
    start_time = ""
    end_time = ""
    try:
        outcome = run_plan_benchmark(
            plan=plan,
            target_url=benchmark_target,
            output_dir=output_dir,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            enable_mlflow=not args.no_mlflow,
            extra_tags=parse_mapping(args.tag, "--tag"),
            execution_name=args.execution_name or "",
        )
        run_id = outcome.run_id
        start_time = outcome.start_time
        end_time = outcome.end_time
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
            _write_output_file(args.mlflow_run_id_output, run_id)
        if args.benchmark_start_time_output and start_time:
            _write_output_file(args.benchmark_start_time_output, start_time)
        if args.benchmark_end_time_output and end_time:
            _write_output_file(args.benchmark_end_time_output, end_time)
        raise
    if args.mlflow_run_id_output:
        _write_output_file(args.mlflow_run_id_output, run_id)
    if args.benchmark_start_time_output:
        _write_output_file(args.benchmark_start_time_output, start_time)
    if args.benchmark_end_time_output:
        _write_output_file(args.benchmark_end_time_output, end_time)

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

    report_path = generate_plan_report(
        plan=plan,
        json_path=json_path,
        model_name=model,
        accelerator=args.accelerator,
        version=version,
        tp=tp_size,
        runtime_args=runtime_args,
        replicas=replicas,
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
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
    artifact_dir = collect_plan_artifacts(
        plan,
        context=_execution_context(
            execution_name=args.execution_name or "",
            artifacts_dir=args.artifacts_dir,
        ),
        mlflow_run_id=args.mlflow_run_id or "",
    )
    if args.upload_direct_to_mlflow and args.mlflow_run_id:
        upload_artifact_directory(
            mlflow_run_id=args.mlflow_run_id,
            artifacts_dir=artifact_dir,
            artifact_path_prefix=str(args.artifact_path_prefix or ""),
            cleanup_after_upload=bool(args.cleanup_after_upload),
            exclude_names=set(getattr(args, "exclude_name", []) or []),
        )
    print(artifact_dir)
    return 0


def cmd_metrics_collect(args: argparse.Namespace) -> int:
    plan = load_runtime_plan(args)
    metrics_dir = collect_plan_metrics(
        plan,
        benchmark_start_time=args.benchmark_start_time,
        benchmark_end_time=args.benchmark_end_time,
        context=_execution_context(artifacts_dir=args.artifacts_dir),
        mlflow_run_id=args.mlflow_run_id or "",
    )
    if args.upload_direct_to_mlflow and args.mlflow_run_id:
        upload_artifact_directory(
            mlflow_run_id=args.mlflow_run_id,
            artifacts_dir=metrics_dir,
            artifact_path_prefix=str(args.artifact_path_prefix or ""),
            cleanup_after_upload=bool(args.cleanup_after_upload),
        )
    print(metrics_dir)
    return 0


def cmd_mlflow_upload(args: argparse.Namespace) -> int:
    plan = load_runtime_plan(args)
    upload_plan_results(
        plan,
        mlflow_run_id=args.mlflow_run_id,
        benchmark_start_time=args.benchmark_start_time,
        benchmark_end_time=args.benchmark_end_time,
        context=_execution_context(artifacts_dir=args.artifacts_dir),
        grafana_url=args.grafana_url or "",
    )
    print(args.mlflow_run_id)
    return 0


def cmd_task_resolve_run_plan(args: argparse.Namespace) -> int:
    plan = load_run_plan_from_sources(run_plan_json=args.run_plan_json)
    resolve_run_plan_stages(
        plan,
        stage_download_path=Path(args.stage_download_path).resolve(),
        stage_deploy_path=Path(args.stage_deploy_path).resolve(),
        stage_benchmark_path=Path(args.stage_benchmark_path).resolve(),
        stage_collect_path=Path(args.stage_collect_path).resolve(),
        stage_cleanup_path=Path(args.stage_cleanup_path).resolve(),
    )
    print("resolved")
    return 0


def cmd_task_setup_run_plan(args: argparse.Namespace) -> int:
    plan = load_run_plan_from_sources(run_plan_json=args.run_plan_json)
    state_path = Path(args.state_path).resolve() if args.state_path else None
    setup_mode = str(args.setup_mode or "auto").strip().lower()

    if setup_mode == "skip":
        detail("Skipping platform setup because SETUP_MODE=skip")
        if state_path is not None:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text("{}", encoding="utf-8")
            print(state_path)
        else:
            print("{}")
        return 0

    if setup_mode != "auto":
        raise ValidationError(
            f"unsupported setup mode: {args.setup_mode!r}; expected auto or skip"
        )

    state = setup_platform(plan, context=_execution_context(state_path=state_path))
    if not state and state_path is not None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("{}", encoding="utf-8")

    if state_path is not None:
        print(state_path)
    else:
        print(json.dumps(state, sort_keys=True))
    return 0


def cmd_task_deploy_run_plan(args: argparse.Namespace) -> int:
    plan = load_run_plan_from_sources(run_plan_json=args.run_plan_json)
    output_dir = deploy_platform(
        plan,
        context=_execution_context(
            execution_name=args.execution_name or "",
            workspace_dir=args.workspace_dir,
            manifests_dir=args.manifests_dir,
        ),
        skip_if_exists=not args.no_skip_if_exists,
        verify=not args.no_verify,
        verify_timeout_seconds=args.verify_timeout_seconds,
    )
    print(output_dir)
    return 0


def cmd_task_cleanup_run_plan(args: argparse.Namespace) -> int:
    plan = load_run_plan_from_sources(run_plan_json=args.run_plan_json)
    cleanup_run_plan(
        plan,
        context=_execution_context(state_path=args.setup_state_path),
        teardown=_teardown_requested(args.teardown_text, default=True),
        wait_for_deletion=not args.no_wait,
        timeout_seconds=args.timeout_seconds,
        skip_if_not_exists=not args.no_skip_if_not_exists,
    )
    print(plan.deployment.release_name)
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
    run_matrix_supervisor(
        plans,
        child_execution_name=args.child_pipeline_name,
        benchflow_image=os.environ.get("BENCHFLOW_IMAGE"),
    )
    print("completed")
    return 0


def cmd_task_remote_capacity_controller(args: argparse.Namespace) -> int:
    run_remote_capacity_controller(
        namespace=args.namespace,
        poll_interval_seconds=args.poll_interval_seconds,
    )
    return 0


@click.command(
    "bootstrap",
    help=(
        "Bootstrap BenchFlow for one of three modes: management cluster only, "
        "remote target cluster, or a single cluster that does both."
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
    "--install-grafana/--no-install-grafana",
    default=None,
    help=(
        "Install Grafana in the dedicated Grafana namespace. Defaults to enabled "
        "for management-cluster and --single-cluster bootstrap, and disabled "
        "when --target-kubeconfig is set."
    ),
)
@click.option(
    "--install-tekton/--no-install-tekton",
    default=None,
    help=(
        "Install and reconcile Tekton resources on the target cluster. Defaults "
        "to enabled for management-cluster and --single-cluster bootstrap, and "
        "disabled when --target-kubeconfig is set."
    ),
)
@click.option(
    "--target-kubeconfig",
    type=click.Path(dir_okay=False, path_type=Path),
    help=(
        "Kubeconfig used to bootstrap a remote target cluster instead of the current one. "
        "This implies target-cluster bootstrap unless --install-tekton or "
        "--install-grafana is set."
    ),
)
@click.option(
    "--single-cluster",
    is_flag=True,
    help=(
        "Bootstrap one cluster as both the management and target cluster. "
        "This installs Tekton, Grafana, the shared PVCs, NFD, and the GPU operator together."
    ),
)
@click.option(
    "--cluster-name",
    help=(
        "Name of the remote cluster. When used with --target-kubeconfig, bootstrap also "
        "creates a management-cluster kubeconfig Secret with this name."
    ),
)
@click.option(
    "--host-alias",
    multiple=True,
    help=(
        "Optional management-cluster host alias for this target cluster, in the form "
        "hostname=ip-address. Repeat for multiple entries."
    ),
)
@click.option(
    "--benchflow-image",
    default=DEFAULT_CONTROLLER_IMAGE,
    show_default=True,
    help=(
        "BenchFlow image used for management-cluster support components such as the "
        "remote-capacity controller."
    ),
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
    "target",
    help="Helpers for management-cluster to target-cluster workflows.",
    short_help="Target-cluster helpers",
)
def target_group() -> None:
    pass


@click.group(
    "kubeconfig-secret",
    help="Manage target-cluster kubeconfig Secrets for Tekton runs.",
    short_help="Target kubeconfig Secrets",
)
def target_kubeconfig_secret_group() -> None:
    pass


@target_kubeconfig_secret_group.command(
    "create",
    help="Create or update a Secret that stores a target-cluster kubeconfig for Tekton runs.",
    short_help="Create a target kubeconfig Secret",
)
@click.option(
    "--name",
    required=True,
    help="Secret name to create or update.",
)
@click.option(
    "--kubeconfig",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to the target-cluster kubeconfig file.",
)
@click.option(
    "--namespace",
    help="Namespace in the management cluster. Defaults to the current project.",
)
@click.option(
    "--key",
    default="kubeconfig",
    show_default=True,
    help="Secret data key that stores the kubeconfig content.",
)
@click.option(
    "--host-alias",
    multiple=True,
    help=(
        "Optional management-cluster host alias for this target cluster, in the form "
        "hostname=ip-address. Repeat for multiple entries."
    ),
)
def target_kubeconfig_secret_create_command(**kwargs: object) -> int:
    return invoke_handler(cmd_target_kubeconfig_secret_create, **kwargs)


target_group.add_command(target_kubeconfig_secret_group)


@click.group(
    "repo",
    help="Repository helpers used by deployment pipelines.",
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
    "setup",
    help="Setup platform prerequisites for a resolved BenchFlow RunPlan.",
    short_help="Platform setup",
)
def setup_group() -> None:
    pass


@setup_group.command(
    "llm-d",
    help="Setup llm-d gateway prerequisites from a resolved RunPlan.",
    short_help="Setup llm-d prerequisites",
)
@runtime_plan_source_options
@click.option(
    "--workspace-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory where the llm-d repository should be cloned for setup.",
)
@click.option(
    "--state-path",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write reversible setup state to this JSON file.",
)
def setup_llmd_command(**kwargs: object) -> int:
    return invoke_handler(cmd_setup_llmd, **kwargs)


@setup_group.command(
    "rhoai",
    help="Setup RHOAI operator, DataScienceCluster, and Gateway prerequisites from a resolved RunPlan.",
    short_help="Setup RHOAI prerequisites",
)
@runtime_plan_source_options
@click.option(
    "--state-path",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write reversible setup state to this JSON file.",
)
def setup_rhoai_command(**kwargs: object) -> int:
    return invoke_handler(cmd_setup_rhoai, **kwargs)


@click.group(
    "teardown",
    help="Tear down platform setup using a previously recorded setup state.",
    short_help="Platform teardown",
)
def teardown_group() -> None:
    pass


@teardown_group.command(
    "llm-d",
    help="Tear down llm-d gateway setup from a resolved RunPlan and setup state file.",
    short_help="Tear down llm-d setup",
)
@runtime_plan_source_options
@click.option(
    "--workspace-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory where the llm-d repository should be cloned for teardown.",
)
@click.option(
    "--state-path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="JSON setup state file produced by the setup step.",
)
def teardown_llmd_command(**kwargs: object) -> int:
    return invoke_handler(cmd_teardown_llmd, **kwargs)


@teardown_group.command(
    "rhoai",
    help="Tear down RHOAI setup from a resolved RunPlan and setup state file.",
    short_help="Tear down RHOAI setup",
)
@runtime_plan_source_options
@click.option(
    "--state-path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="JSON setup state file produced by the setup step.",
)
def teardown_rhoai_command(**kwargs: object) -> int:
    return invoke_handler(cmd_teardown_rhoai, **kwargs)


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
    "--execution-name",
    help="Owning execution name for log and label propagation.",
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


@deploy_group.command(
    "rhoai",
    help="Deploy a RHOAI LLMInferenceService scenario from a resolved RunPlan.",
    short_help="Deploy a RHOAI scenario",
)
@runtime_plan_source_options
@click.option(
    "--manifests-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory where rendered manifests should be written.",
)
@click.option(
    "--no-skip-if-exists",
    is_flag=True,
    help="Redeploy even if the target service already exists.",
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
def deploy_rhoai_command(**kwargs: object) -> int:
    return invoke_handler(cmd_deploy_rhoai, **kwargs)


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


@undeploy_group.command(
    "rhoai",
    help="Tear down a RHOAI deployment from a resolved RunPlan.",
    short_help="Remove a RHOAI deployment",
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
    default=300,
    show_default=True,
    help="Maximum time to wait for cleanup.",
)
@click.option(
    "--no-skip-if-not-exists",
    is_flag=True,
    help="Fail instead of skipping when the service is already absent.",
)
def undeploy_rhoai_command(**kwargs: object) -> int:
    return invoke_handler(cmd_undeploy_rhoai, **kwargs)


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
    default=lambda: os.environ.get("MLFLOW_TRACKING_URI"),
    show_default="env MLFLOW_TRACKING_URI",
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
    "--execution-name",
    help="Owning execution name for MLflow tagging.",
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
    default=lambda: os.environ.get("MLFLOW_TRACKING_URI"),
    show_default="env MLFLOW_TRACKING_URI",
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
    "--execution-name",
    help="Execution name to collect artifacts from.",
)
@click.option(
    "--mlflow-run-id",
    help="If set, upload the collected artifacts directly to this MLflow run.",
)
@click.option(
    "--artifact-path-prefix",
    default="",
    help="MLflow artifact path prefix used when --mlflow-run-id is set.",
)
@click.option(
    "--cleanup-after-upload",
    is_flag=True,
    help="Delete the local artifact directory contents after a direct MLflow upload.",
)
@click.option(
    "--upload-direct-to-mlflow",
    is_flag=True,
    help="Upload the collected artifacts directly to MLflow from this command.",
)
@click.option(
    "--exclude-name",
    multiple=True,
    help="Artifact file or directory name to skip during direct MLflow upload.",
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
@click.option(
    "--mlflow-run-id",
    help="If set, upload the collected metrics directly to this MLflow run.",
)
@click.option(
    "--artifact-path-prefix",
    default="",
    help="MLflow artifact path prefix used when --mlflow-run-id is set.",
)
@click.option(
    "--cleanup-after-upload",
    is_flag=True,
    help="Delete the local metrics directory contents after a direct MLflow upload.",
)
@click.option(
    "--upload-direct-to-mlflow",
    is_flag=True,
    help="Upload the collected metrics directly to MLflow from this command.",
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
    help="Internal commands invoked by execution backend tasks inside the BenchFlow image.",
    short_help="Internal backend task entrypoints",
    hidden=True,
)
def task_group() -> None:
    pass


@task_group.command(
    "resolve-run-plan",
    help="Internal command used by backend tasks to resolve a RunPlan into stage files.",
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
    "setup-run-plan",
    help="Internal command used by the execution backend to setup platform prerequisites for a RunPlan.",
    short_help="Setup a RunPlan",
)
@click.option("--run-plan-json", required=True, help="Inline RunPlan JSON payload.")
@click.option(
    "--setup-mode",
    default="auto",
    show_default=True,
    help="Setup mode. Supported values: auto, skip.",
)
@click.option(
    "--state-path",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path where the reversible setup state JSON should be written.",
)
def task_setup_run_plan_command(**kwargs: object) -> int:
    return invoke_handler(cmd_task_setup_run_plan, **kwargs)


@task_group.command(
    "deploy-run-plan",
    help="Internal command used by the execution backend to deploy the platform described by a RunPlan.",
    short_help="Deploy a RunPlan",
)
@click.option("--run-plan-json", required=True, help="Inline RunPlan JSON payload.")
@click.option(
    "--workspace-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Workspace directory used by platforms that require source checkout.",
)
@click.option(
    "--manifests-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory where rendered manifests should be written.",
)
@click.option(
    "--execution-name",
    default="",
    help="Owning execution name for label and log propagation.",
)
@click.option(
    "--no-skip-if-exists",
    is_flag=True,
    help="Redeploy even if the target workload already exists.",
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
def task_deploy_run_plan_command(**kwargs: object) -> int:
    return invoke_handler(cmd_task_deploy_run_plan, **kwargs)


@task_group.command(
    "cleanup-run-plan",
    help="Internal command used by the execution backend to clean up the platform described by a RunPlan.",
    short_help="Clean up a RunPlan",
)
@click.option("--run-plan-json", required=True, help="Inline RunPlan JSON payload.")
@click.option(
    "--no-wait",
    is_flag=True,
    help="Do not wait for resource deletion.",
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
    help="Fail instead of skipping when the workload is already absent.",
)
@click.option(
    "--setup-state-path",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to the reversible setup state JSON file.",
)
@click.option(
    "--teardown-text",
    default="true",
    show_default=True,
    help="Whether platform setup should be torn down after scenario cleanup.",
)
def task_cleanup_run_plan_command(**kwargs: object) -> int:
    return invoke_handler(cmd_task_cleanup_run_plan, **kwargs)


@task_group.command(
    "assert-status",
    help="Internal command used by backend tasks to assert task status transitions.",
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
        "Internal command used by the execution backend to run a cartesian product of resolved "
        "RunPlans sequentially in the cluster."
    ),
    short_help="Run a matrix of child executions",
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
    help="Pipeline name to use for the child executions.",
)
def task_run_experiment_matrix_command(**kwargs: object) -> int:
    return invoke_handler(cmd_task_run_experiment_matrix, **kwargs)


@task_group.command(
    "remote-capacity-controller",
    help="Internal command used by the management cluster to reconcile remote Kueue capacity.",
    short_help="Run the remote-capacity controller",
)
@click.option(
    "--namespace",
    required=True,
    help="BenchFlow namespace watched by the controller.",
)
@click.option(
    "--poll-interval-seconds",
    default=10,
    show_default=True,
    type=int,
    help="Polling interval used by the controller loop.",
)
def task_remote_capacity_controller_command(**kwargs: object) -> int:
    return invoke_handler(cmd_task_remote_capacity_controller, **kwargs)


@click.command(
    "watch",
    help="Watch PipelineRun and task status progress until a BenchFlow execution finishes.",
    short_help="Watch execution progress",
)
@click.argument("execution_name")
@click.option(
    "--namespace",
    help="Namespace that contains the execution. Defaults to the current oc project.",
)
def watch_command(**kwargs: object) -> int:
    return invoke_handler(cmd_watch, **kwargs)


@click.command(
    "logs",
    help=(
        "List selectable pipeline tasks or stream logs for one task or the full "
        "execution."
    ),
    short_help="Inspect execution logs",
)
@click.argument("execution_name")
@click.option(
    "--namespace",
    help="Namespace that contains the execution. Defaults to the current oc project.",
)
@click.option(
    "--step",
    help="Logical pipeline task name to stream logs from.",
)
@click.option(
    "--all",
    "all_logs",
    is_flag=True,
    help="Stream logs for the full execution instead of one step.",
)
@click.option(
    "--all-containers",
    is_flag=True,
    help="Include non-main task containers when streaming one pipeline task.",
)
def logs_command(**kwargs: object) -> int:
    return invoke_handler(cmd_logs, **kwargs)
