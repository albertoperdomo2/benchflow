import json
from pathlib import Path

import yaml

from benchflow.loaders import ProfileCatalog, load_experiment
from benchflow.matrix import resolve_experiment_matrix
from benchflow.renderers.deployment import render_rhoai_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_rhoai_selective_loading_renders_distributed_default_epp(
    tmp_path: Path,
) -> None:
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        """apiVersion: benchflow.io/v1alpha1
kind: Experiment
metadata:
  name: rhoai-selective-loading-test
spec:
  model:
    name: nvidia/Llama-3_1-Nemotron-Ultra-253B-v1-FP8
  deployment_profile: rhoai-distributed-default-selective-loading
  benchmark_profile: aiperf-smoke
  metrics_profile: detailed
  namespace: benchflow
  overrides:
    images:
      runtime: quay.io/example/vllm:selective-loading
      scheduler: quay.io/example/epp:selective-loading
""",
        encoding="utf-8",
    )
    plan = resolve_experiment_matrix(
        load_experiment(experiment_path),
        ProfileCatalog.load(REPO_ROOT / "profiles"),
    )[0]

    assert plan.deployment.platform == "rhoai"
    assert plan.deployment.mode == "distributed-default"
    assert plan.deployment.platform_version == "RHOAI-3.5.0"
    assert plan.deployment.runtime.replicas == 1
    assert plan.deployment.runtime.tensor_parallelism == 2
    assert plan.deployment.runtime.shared_memory_size == "300Gi"
    assert plan.deployment.runtime.service_account_name == (
        "benchflow-hostpath-runtime"
    )
    assert len(plan.deployment.runtime.host_paths) == 1
    assert plan.deployment.runtime.host_paths[0].mount_path == "/mnt/nvme-kv-cache"

    manifest = render_rhoai_manifest(plan)
    scheduler = manifest["spec"]["router"]["scheduler"]["template"]["containers"][0]
    assert scheduler["image"] == "quay.io/example/epp:selective-loading"
    assert "--allow-experimental-plugins" in scheduler["args"]
    config_index = scheduler["args"].index("--config-text")
    config = yaml.safe_load(scheduler["args"][config_index + 1])
    assert config["kind"] == "EndpointPickerConfig"
    selective_policy = next(
        plugin
        for plugin in config["plugins"]
        if plugin["type"] == "selective-kv-policy"
    )
    assert selective_policy["parameters"] == {
        "loadPolicy": "threshold",
        "minExternalReusableTokens": 1024,
    }
    precise_producer = next(
        plugin
        for plugin in config["plugins"]
        if plugin["type"] == "precise-prefix-cache-producer"
    )
    assert precise_producer["parameters"]["tokenProcessorConfig"] == {
        "blockSizeTokens": 64
    }
    assert precise_producer["parameters"]["kvEventsConfig"]["podDiscoveryConfig"] == {
        "socketPort": 5556,
        "replaySocketPort": 5559,
    }
    token_producer = next(
        plugin for plugin in config["plugins"] if plugin["type"] == "token-producer"
    )
    assert token_producer["parameters"]["modelName"] == plan.model.name
    assert token_producer["parameters"]["vllm"]["url"].endswith(
        "-kserve-workload-svc.benchflow.svc:8000"
    )
    assert token_producer["parameters"]["vllm"]["caCertPath"] == (
        "/var/run/kserve/tls/ca.crt"
    )
    affinity_filter = next(
        plugin
        for plugin in config["plugins"]
        if plugin["type"] == "prefix-cache-affinity-filter"
    )
    assert affinity_filter["parameters"] == {
        "prefixMatchInfoProducerName": "precise-prefix-cache-producer",
        "peakPrefillThroughput": 15926,
    }
    assert config["schedulingProfiles"][0]["plugins"] == [
        {"pluginRef": "prefix-cache-affinity-filter"},
        {"pluginRef": "token-load-scorer"},
    ]
    model_server = manifest["spec"]["template"]["containers"][0]
    assert model_server["image"] == "quay.io/example/vllm:selective-loading"
    assert model_server["resources"]["requests"]["nvidia.com/gpu"] == "2"
    assert model_server["resources"]["limits"]["nvidia.com/gpu"] == "2"
    kv_transfer_arg = next(
        arg for arg in model_server["args"] if arg.startswith("--kv-transfer-config=")
    )
    assert '"spec_name":"TieringOffloadingSpec"' in kv_transfer_arg
    assert '"root_dir":"/mnt/nvme-kv-cache"' in kv_transfer_arg
    assert "--block-size=64" in model_server["args"]
    kv_events_arg = next(
        arg for arg in model_server["args"] if arg.startswith("--kv-events-config=")
    )
    kv_events = json.loads(kv_events_arg.split("=", 1)[1])
    assert kv_events == {
        "enable_kv_cache_events": True,
        "publisher": "zmq",
        "endpoint": "$(KV_EVENTS_ENDPOINT)",
        "replay_endpoint": "$(KV_EVENTS_REPLAY_ENDPOINT)",
        "topic": f"kv@$(POD_IP):$(POD_PORT)@{plan.model.name}",
    }
    assert model_server["ports"] == [
        {"containerPort": 5556, "name": "kv-events", "protocol": "TCP"},
        {"containerPort": 5559, "name": "kv-replay", "protocol": "TCP"},
    ]
    env = {entry["name"]: entry for entry in model_server["env"]}
    assert env["POD_IP"]["valueFrom"]["fieldRef"]["fieldPath"] == "status.podIP"
    assert env["POD_PORT"]["value"] == "8000"
    assert env["KV_EVENTS_ENDPOINT"]["value"] == "tcp://*:5556"
    assert env["KV_EVENTS_REPLAY_ENDPOINT"]["value"] == "tcp://*:5559"


def test_rhoai_nvme_offloading_renders_precise_prefix_routing_without_selective_kv(
    tmp_path: Path,
) -> None:
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        """apiVersion: benchflow.io/v1alpha1
kind: Experiment
metadata:
  name: rhoai-nvme-precise-test
spec:
  model:
    name: Qwen/Qwen3-32B
  deployment_profile: multi-tier-offloading-nvme
  benchmark_profile: aiperf-smoke
  metrics_profile: detailed
  namespace: benchflow
  overrides:
    images:
      runtime: quay.io/example/vllm:precise
      scheduler: quay.io/example/epp:precise
""",
        encoding="utf-8",
    )
    plan = resolve_experiment_matrix(
        load_experiment(experiment_path),
        ProfileCatalog.load(REPO_ROOT / "profiles"),
    )[0]

    manifest = render_rhoai_manifest(plan)
    scheduler = manifest["spec"]["router"]["scheduler"]["template"]["containers"][0]
    assert "--allow-experimental-plugins" in scheduler["args"]
    config_index = scheduler["args"].index("--config-text")
    config = yaml.safe_load(scheduler["args"][config_index + 1])
    plugin_types = [plugin["type"] for plugin in config["plugins"]]
    assert "precise-prefix-cache-producer" in plugin_types
    assert "inflight-load-producer" in plugin_types
    assert "prefix-cache-affinity-filter" in plugin_types
    assert "token-load-scorer" in plugin_types
    assert "selective-kv-policy" not in plugin_types
    assert config["dataLayer"] == {
        "sources": [
            {
                "pluginRef": "endpoint-notification-source",
                "extractors": [{"pluginRef": "precise-prefix-cache-producer"}],
            }
        ]
    }
    model_server = manifest["spec"]["template"]["containers"][0]
    assert "--enable-prefix-caching" in model_server["args"]
    assert "--block-size=64" in model_server["args"]
    assert any(arg.startswith("--kv-events-config=") for arg in model_server["args"])


def test_rhoai_selective_loading_disabled_renders_always_recompute_policy(
    tmp_path: Path,
) -> None:
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        """apiVersion: benchflow.io/v1alpha1
kind: Experiment
metadata:
  name: rhoai-selective-loading-disabled-test
spec:
  model:
    name: Qwen/Qwen3-32B
  deployment_profile: rhoai-distributed-default-selective-loading-disabled
  benchmark_profile: aiperf-smoke
  metrics_profile: detailed
  namespace: benchflow
  overrides:
    images:
      runtime: quay.io/example/vllm:selective-loading
      scheduler: quay.io/example/epp:selective-loading
""",
        encoding="utf-8",
    )
    plan = resolve_experiment_matrix(
        load_experiment(experiment_path),
        ProfileCatalog.load(REPO_ROOT / "profiles"),
    )[0]

    manifest = render_rhoai_manifest(plan)
    scheduler = manifest["spec"]["router"]["scheduler"]["template"]["containers"][0]
    assert "--allow-experimental-plugins" in scheduler["args"]
    config_index = scheduler["args"].index("--config-text")
    config = yaml.safe_load(scheduler["args"][config_index + 1])
    selective_policy = next(
        plugin
        for plugin in config["plugins"]
        if plugin["type"] == "selective-kv-policy"
    )
    assert selective_policy["parameters"] == {"loadPolicy": "disable"}

    model_server = manifest["spec"]["template"]["containers"][0]
    kv_transfer_arg = next(
        arg for arg in model_server["args"] if arg.startswith("--kv-transfer-config=")
    )
    kv_transfer = json.loads(kv_transfer_arg.split("=", 1)[1])
    secondary_tier = kv_transfer["kv_connector_extra_config"]["secondary_tiers"][0]
    assert secondary_tier == {
        "type": "fs",
        "root_dir": "/mnt/nvme-kv-cache",
        "n_read_threads": 64,
        "n_write_threads": 64,
    }


def test_selective_loading_focused_crossover_resolves_independent_cells() -> None:
    catalog = ProfileCatalog.load(REPO_ROOT / "profiles")
    experiment = load_experiment(
        REPO_ROOT / "experiments" / "rhoai" / "selective-loading-focused-crossover.yaml"
    )
    plans = resolve_experiment_matrix(experiment, catalog)

    assert len(plans) == 24
    assert {plan.profiles.deployment for plan in plans} == {
        "multi-tier-offloading-nvme",
        "rhoai-distributed-default-selective-loading-disabled",
    }
    assert {plan.profiles.benchmark for plan in plans} == {
        f"guidellm-selective-loading-calibration-p{prefix}-c{concurrency}"
        for prefix in (1024, 2048, 4096, 8192)
        for concurrency in (16, 32, 64)
    }
    assert all(plan.deployment.runtime.placement.mode == "sequential" for plan in plans)
    expected_prefix_counts = {1024: 2048, 2048: 1024, 4096: 512, 8192: 256}
    for plan in plans:
        node_affinity = plan.deployment.runtime.affinity["nodeAffinity"]
        required = node_affinity["requiredDuringSchedulingIgnoredDuringExecution"]
        node_values = required["nodeSelectorTerms"][0]["matchExpressions"][0]["values"]
        assert node_values == ["diadochos-hqxzk-gpu-h100-mt46x"]

        guidellm = plan.benchmark.guidellm
        data = guidellm.args["data"]
        bucket = data["prefix_buckets"][0]
        assert bucket["prefix_count"] == expected_prefix_counts[bucket["prefix_tokens"]]
        assert data["turns"] == 1
        assert guidellm.pre_warmup.args["rate"] == 32
