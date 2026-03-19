from __future__ import annotations

from itertools import product

from .loaders import ProfileCatalog
from .models import (
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
    ResolvedRunPlan,
    StageSpec,
    ValidationError,
    normalize_model_names,
    normalize_profile_refs,
)
from .plans import resolve_run_plan


def profile_matrix_axes(
    experiment: Experiment,
) -> tuple[list[str], list[str], list[str]]:
    return (
        normalize_profile_refs(
            experiment.spec.deployment_profile, "spec.deployment_profile"
        ),
        normalize_profile_refs(
            experiment.spec.benchmark_profile, "spec.benchmark_profile"
        ),
        normalize_profile_refs(experiment.spec.metrics_profile, "spec.metrics_profile"),
    )


def _axis_values(value) -> list[object]:
    if isinstance(value, list):
        return list(value)
    return [value]


def _override_axes(
    experiment: Experiment,
) -> tuple[list[object], list[object], list[object], list[object], list[object]]:
    overrides = experiment.spec.overrides
    return (
        _axis_values(overrides.images.runtime),
        _axis_values(overrides.images.scheduler),
        _axis_values(overrides.scale.replicas),
        _axis_values(overrides.scale.tensor_parallelism),
        _axis_values(overrides.llm_d.repo_ref),
    )


def _model_axis(experiment: Experiment) -> list[str]:
    return normalize_model_names(experiment.spec.model.name, "spec.model.name")


def is_matrix_experiment(experiment: Experiment) -> bool:
    model_names = _model_axis(experiment)
    deployment_profiles, benchmark_profiles, metrics_profiles = profile_matrix_axes(
        experiment
    )
    (
        runtime_images,
        scheduler_images,
        replicas_values,
        tensor_parallelism_values,
        repo_refs,
    ) = _override_axes(experiment)
    return any(
        len(values) > 1
        for values in (
            model_names,
            deployment_profiles,
            benchmark_profiles,
            metrics_profiles,
            runtime_images,
            scheduler_images,
            replicas_values,
            tensor_parallelism_values,
            repo_refs,
        )
    )


def experiment_matrix_size(experiment: Experiment) -> int:
    model_names = _model_axis(experiment)
    deployment_profiles, benchmark_profiles, metrics_profiles = profile_matrix_axes(
        experiment
    )
    (
        runtime_images,
        scheduler_images,
        replicas_values,
        tensor_parallelism_values,
        repo_refs,
    ) = _override_axes(experiment)
    return (
        len(model_names)
        * len(deployment_profiles)
        * len(benchmark_profiles)
        * len(metrics_profiles)
        * len(runtime_images)
        * len(scheduler_images)
        * len(replicas_values)
        * len(tensor_parallelism_values)
        * len(repo_refs)
    )


def expand_experiment_matrix(experiment: Experiment) -> list[Experiment]:
    model_names = _model_axis(experiment)
    deployment_profiles, benchmark_profiles, metrics_profiles = profile_matrix_axes(
        experiment
    )
    (
        runtime_images,
        scheduler_images,
        replicas_values,
        tensor_parallelism_values,
        repo_refs,
    ) = _override_axes(experiment)
    expanded: list[Experiment] = []

    for (
        model_name,
        deployment_profile,
        benchmark_profile,
        metrics_profile,
        runtime_image,
        scheduler_image,
        replicas,
        tensor_parallelism,
        repo_ref,
    ) in product(
        model_names,
        deployment_profiles,
        benchmark_profiles,
        metrics_profiles,
        runtime_images,
        scheduler_images,
        replicas_values,
        tensor_parallelism_values,
        repo_refs,
    ):
        expanded.append(
            Experiment(
                api_version=experiment.api_version,
                kind=experiment.kind,
                metadata=Metadata(
                    name=experiment.metadata.name,
                    labels=dict(experiment.metadata.labels),
                ),
                spec=ExperimentSpec(
                    model=ModelSpec(name=model_name),
                    deployment_profile=[deployment_profile],
                    benchmark_profile=[benchmark_profile],
                    metrics_profile=[metrics_profile],
                    namespace=experiment.spec.namespace,
                    service_account=experiment.spec.service_account,
                    ttl_seconds_after_finished=experiment.spec.ttl_seconds_after_finished,
                    stages=StageSpec(
                        download=experiment.spec.stages.download,
                        deploy=experiment.spec.stages.deploy,
                        benchmark=experiment.spec.stages.benchmark,
                        collect=experiment.spec.stages.collect,
                        cleanup=experiment.spec.stages.cleanup,
                    ),
                    mlflow=MlflowSpec(
                        experiment=experiment.spec.mlflow.experiment,
                        tags=dict(experiment.spec.mlflow.tags),
                    ),
                    execution=ExecutionSpec(backend=experiment.spec.execution.backend),
                    overrides=OverrideSpec(
                        images=OverrideImagesSpec(
                            runtime=runtime_image,
                            scheduler=scheduler_image,
                        ),
                        scale=OverrideScaleSpec(
                            replicas=replicas,
                            tensor_parallelism=tensor_parallelism,
                        ),
                        runtime=OverrideRuntimeSpec(
                            vllm_args=list(experiment.spec.overrides.runtime.vllm_args),
                            env=dict(experiment.spec.overrides.runtime.env),
                        ),
                        llm_d=OverrideLlmdSpec(repo_ref=repo_ref),
                        rhoai=OverrideRhoaiSpec(
                            enable_auth=experiment.spec.overrides.rhoai.enable_auth
                        ),
                    ),
                ),
            )
        )

    return expanded


def resolve_experiment_matrix(
    experiment: Experiment, catalog: ProfileCatalog
) -> list[ResolvedRunPlan]:
    return [
        resolve_run_plan(item, catalog) for item in expand_experiment_matrix(experiment)
    ]


def require_single_experiment_plan(experiment: Experiment) -> Experiment:
    matrix_size = experiment_matrix_size(experiment)
    if matrix_size != 1:
        raise ValidationError(
            f"experiment expands to {matrix_size} profile combinations; "
            "this command only supports a single combination"
        )
    return expand_experiment_matrix(experiment)[0]
