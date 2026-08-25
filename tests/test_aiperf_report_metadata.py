from types import SimpleNamespace

from benchflow.benchmark import aiperf


def _run(*, params=None, tags=None):
    return SimpleNamespace(data=SimpleNamespace(params=params or {}, tags=tags or {}))


def test_reads_aiperf_012_model_and_dataset_schema() -> None:
    runs_data = [
        {
            "summary": {
                "input_config": {
                    "models": {"items": [{"name": "thinkingmachines/Inkling-NVFP4"}]},
                    "datasets": [
                        {
                            "dataset": "weka_hf",
                            "hf_weka_dataset": ("semianalysisai/cc-traces-weka-062126"),
                        }
                    ],
                }
            }
        }
    ]

    assert aiperf._comparison_model_name(runs_data) == (
        "thinkingmachines/Inkling-NVFP4"
    )
    assert aiperf._comparison_dataset_label(runs_data) == (
        "semianalysisai/cc-traces-weka-062126"
    )


def test_keeps_legacy_aiperf_model_and_dataset_schema() -> None:
    runs_data = [
        {
            "summary": {
                "input_config": {
                    "endpoint": {"model_names": ["legacy/model"]},
                    "input": {"public_dataset": "legacy-dataset"},
                }
            }
        }
    ]

    assert aiperf._comparison_model_name(runs_data) == "legacy/model"
    assert aiperf._comparison_dataset_label(runs_data) == "legacy-dataset"


def test_falls_back_to_mlflow_model_and_dataset_metadata() -> None:
    runs_data = [
        {
            "summary": {"input_config": {}},
            "model": "fallback/model",
            "public_dataset": "weka_hf",
            "hf_weka_dataset": "org/exact-weka-corpus",
        }
    ]

    assert aiperf._comparison_model_name(runs_data) == "fallback/model"
    assert aiperf._comparison_dataset_label(runs_data) == "org/exact-weka-corpus"


def test_mlflow_parallelism_prefers_params_and_accepts_tags() -> None:
    parameter_run = _run(params={"pp": "4"}, tags={"pp": "2"})
    tag_run = _run(tags={"pipeline_parallelism": "2"})

    assert (
        aiperf._mlflow_value(parameter_run, "pp", "pipeline_parallelism", default="1")
        == "4"
    )
    assert (
        aiperf._mlflow_value(tag_run, "pp", "pipeline_parallelism", default="1") == "2"
    )


def test_comparison_shape_includes_pipeline_parallelism() -> None:
    line = aiperf._comparison_shape_line(
        [{"accelerator": "H100", "tp": "4", "pp": "2", "replicas": "1"}]
    )

    assert line == "Accelerator: H100 | TP: 4 | PP: 2 | R: 1"


def test_aiperf_version_filter_matches_base_or_composed_version() -> None:
    run = _run(
        params={"version": "rhaiis-raw-vllm"},
        tags={"deployment_profile": "inkling-raw-vllm-tp4-pp2"},
    )

    assert aiperf._matches_requested_version(run, {"rhaiis-raw-vllm"})
    assert aiperf._matches_requested_version(
        run, {"rhaiis-raw-vllm-inkling-raw-vllm-tp4-pp2"}
    )
    assert not aiperf._matches_requested_version(run, {"other"})
