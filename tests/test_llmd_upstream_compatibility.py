from __future__ import annotations

import json
from pathlib import Path

import yaml

from benchflow.deploy.llmd import (
    _llmd_recipe_render_dir,
    _patch_recipe_modelserver_overlay,
    _patch_scheduler_values,
)
from benchflow.loaders import ProfileCatalog, load_experiment
from benchflow.matrix import resolve_experiment_matrix


REPO_ROOT = Path(__file__).resolve().parents[1]


def _precise_plan(tmp_path: Path):
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        """apiVersion: benchflow.io/v1alpha1
kind: Experiment
metadata:
  name: llmd-upstream-compatibility
spec:
  model:
    name: Qwen/Qwen3-32B
  deployment_profile: llm-d-precise-prefix-cache-routing-standalone
  benchmark_profile: aiperf-smoke
  metrics_profile: detailed
  namespace: benchflow
""",
        encoding="utf-8",
    )
    return resolve_experiment_matrix(
        load_experiment(experiment_path),
        ProfileCatalog.load(REPO_ROOT / "profiles"),
    )[0]


def test_current_upstream_nested_render_overlays_are_not_deployed(
    tmp_path: Path,
) -> None:
    plan = _precise_plan(tmp_path)
    render_dir = tmp_path / "guides" / "precise-prefix-cache-routing" / "render"
    nested_render_dir = render_dir / "standalone"
    nested_render_dir.mkdir(parents=True)
    for filename in ("deployment.yaml", "service.yaml", "kustomization.yaml"):
        (nested_render_dir / filename).touch()

    assert _llmd_recipe_render_dir(tmp_path, plan, router_chart=True) is None


def test_legacy_root_render_overlay_remains_supported(tmp_path: Path) -> None:
    plan = _precise_plan(tmp_path)
    render_dir = tmp_path / "guides" / "precise-prefix-cache-routing" / "render"
    render_dir.mkdir(parents=True)
    for filename in ("deployment.yaml", "service.yaml", "kustomization.yaml"):
        (render_dir / filename).touch()

    assert _llmd_recipe_render_dir(tmp_path, plan, router_chart=True) == render_dir


def test_current_upstream_endpoint_discovery_is_not_rewritten_to_render_service(
    tmp_path: Path,
) -> None:
    plan = _precise_plan(tmp_path)
    values_path = tmp_path / "precise.values.yaml"
    values_path.write_text(
        """router:
  epp:
    pluginsConfigFile: precise.yaml
    pluginsCustomConfig:
      precise.yaml: |
        apiVersion: llm-d.ai/v1
        kind: EndpointPickerConfig
        plugins:
          - type: token-producer
            parameters:
              modelName: Qwen/Qwen3-32B
              vllm:
                endpointDiscovery:
                  attemptTimeout: 2s
  modelServers:
    matchLabels: {}
""",
        encoding="utf-8",
    )

    _patch_scheduler_values(
        plan,
        values_path,
        recipe_layout=True,
        router_chart=True,
        render_service_name=None,
    )

    values = yaml.safe_load(values_path.read_text(encoding="utf-8"))
    epp = values["router"]["epp"]
    config = yaml.safe_load(epp["pluginsCustomConfig"][epp["pluginsConfigFile"]])
    token_producer = next(
        plugin for plugin in config["plugins"] if plugin["type"] == "token-producer"
    )
    vllm = token_producer["parameters"]["vllm"]
    assert vllm["endpointDiscovery"] == {"attemptTimeout": "2s"}
    assert "url" not in vllm


def test_current_upstream_precise_vllm_contract_is_preserved(tmp_path: Path) -> None:
    plan = _precise_plan(tmp_path)
    overlay_dir = tmp_path / "modelserver"
    overlay_dir.mkdir()
    (overlay_dir / "kustomization.yaml").write_text(
        "resources: []\nlabels: []\n",
        encoding="utf-8",
    )
    (overlay_dir / "patch-vllm.yaml").write_text(
        """apiVersion: apps/v1
kind: Deployment
metadata:
  name: decode
spec:
  template:
    spec:
      containers:
        - name: modelserver
          args:
            - Qwen/Qwen3-32B
            - --enable-scale-out
            - --disable-access-log-for-endpoints=/health,/metrics,/v1/models,/v1/completions/render,/v1/chat/completions/render
            - --tensor-parallel-size=2
            - --block-size=64
            - --kv-events-config
            - '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"$(KV_EVENTS_ENDPOINT)","replay_endpoint":"$(KV_EVENTS_REPLAY_ENDPOINT)","topic":"kv@$(POD_IP):$(POD_PORT)@Qwen/Qwen3-32B"}'
            - --shutdown-timeout=45
          env:
            - name: KV_EVENTS_REPLAY_ENDPOINT
              value: tcp://*:5559
          ports:
            - name: kv-events
              containerPort: 5556
            - name: kv-replay
              containerPort: 5559
""",
        encoding="utf-8",
    )

    _patch_recipe_modelserver_overlay(plan, overlay_dir, router_chart=True)

    patch = yaml.safe_load(
        (overlay_dir / "patch-vllm.yaml").read_text(encoding="utf-8")
    )
    container = patch["spec"]["template"]["spec"]["containers"][0]
    args = container["args"]
    assert "--enable-scale-out" in args
    assert "--shutdown-timeout=45" in args
    assert (
        "--disable-access-log-for-endpoints=/health,/metrics,/v1/models,"
        "/v1/completions/render,/v1/chat/completions/render"
    ) in args

    kv_index = args.index("--kv-events-config")
    kv_events = json.loads(args[kv_index + 1])
    assert kv_events["endpoint"] == "$(KV_EVENTS_ENDPOINT)"
    assert kv_events["replay_endpoint"] == "$(KV_EVENTS_REPLAY_ENDPOINT)"
    assert kv_events["topic"] == "kv@$(POD_IP):$(POD_PORT)@Qwen/Qwen3-32B"
