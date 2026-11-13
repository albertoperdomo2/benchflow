from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timezone
from io import StringIO
import json
import math
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
import yaml

from benchflow import lmcache
from benchflow.deploy.llmd import _patch_recipe_modelserver_overlay
from benchflow.contracts import ExecutionContext
from benchflow.loaders import ProfileCatalog, load_experiment, load_run_plan_data
from benchflow.matrix import resolve_experiment_matrix
from benchflow.models import ValidationError
from benchflow.orchestration.service import _materialize_execution_name
from benchflow.orchestration.tekton import render_pipelinerun
from benchflow.platform_state import base_setup_key_for_plan, setup_key_for_plan
from benchflow.plans import scope_execution_release
from benchflow.renderers.deployment import render_rhoai_manifest, render_llmd_values
from benchflow.renderers.deployment import write_deployment_assets
from benchflow.toolbox import platform
from benchflow.toolbox import run_plan as run_plan_toolbox

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "quay.io/benchflow-test/vllm-lmcache:provided"


@pytest.fixture
def plan():
    base = resolve_experiment_matrix(
        load_experiment(
            ROOT / "experiments/smoke/qwen3-06b-rhoai-distributed-default-smoke.yaml"
        ),
        ProfileCatalog.load(ROOT / "profiles"),
    )[0]
    deployment = replace(
        base.deployment,
        runtime=replace(
            base.deployment.runtime, image=IMAGE, replicas=1, tensor_parallelism=1
        ),
        options={"enable_auth": False, "lmcache": {"cpu_gib": 8}},
    )
    return replace(base, deployment=lmcache.configure_lmcache(deployment))


def test_resolved_connector_survives_serialization_and_both_renderers(plan):
    restored = load_run_plan_data(plan.to_dict())
    assert restored.deployment.runtime.vllm_args == plan.deployment.runtime.vllm_args
    manifest = render_rhoai_manifest(restored)
    args = manifest["spec"]["template"]["containers"][0]["args"]
    transfer = [arg for arg in args if arg.startswith("--kv-transfer-config=")]
    assert len(transfer) == 1
    config = json.loads(transfer[0].split("=", 1)[1])
    assert config["kv_connector"] == "LMCacheMPConnector"
    assert (
        config["kv_connector_module_path"]
        == "lmcache.integration.vllm.lmcache_mp_connector"
    )
    assert config["kv_connector_extra_config"] == {
        "lmcache.mp.host": "tcp://benchflow-lmcache.benchflow.svc.cluster.local",
        "lmcache.mp.port": 5555,
        "lmcache.mp.isolated_ipc": True,
    }
    llmd = replace(
        restored,
        deployment=replace(
            restored.deployment, platform="llm-d", mode="inference-scheduling"
        ),
    )
    assert transfer[0] in render_llmd_values(llmd)["runtime"]["vllmArgs"]
    lmcache.lmcache_spec(llmd.deployment)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"cpu_gib": True}, "positive integer"),
        ({"cpu_gib": 0}, "positive integer"),
        ({"chunk_size": "256"}, "positive integer"),
        ({"verify_readback": "true"}, "boolean"),
        ({"local_disk": True}, "unsupported"),
    ],
)
def test_reject_invalid_options(plan, options, match):
    deployment = replace(plan.deployment, options={"lmcache": options})
    with pytest.raises(ValidationError, match=match):
        lmcache.configure_lmcache(deployment)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"replicas": 2}, "one replica"),
        ({"pipeline_parallelism": 2}, "pipeline_parallelism"),
        ({"image": ""}, "--runtime-image"),
        (
            {
                "vllm_args": [
                    '--kv-transfer-config={"kv_connector":"OffloadingConnector"}'
                ]
            },
            "conflicting",
        ),
        ({"vllm_args": ["--enable-sleep-mode"]}, "sleep-mode"),
    ],
)
def test_reject_unsupported_runtime(plan, change, match):
    deployment = replace(
        plan.deployment, runtime=replace(plan.deployment.runtime, **change)
    )
    with pytest.raises(ValidationError, match=match):
        lmcache.configure_lmcache(deployment)


def test_manifests_keep_cuda_local_and_do_not_reserve_gpus(plan):
    documents = {d["kind"]: d for d in lmcache.lmcache_documents(plan)}
    service = documents["Service"]["spec"]
    assert service["internalTrafficPolicy"] == "Local"
    assert "type" not in service  # ClusterIP; never a public endpoint.
    pod = documents["DaemonSet"]["spec"]["template"]["spec"]
    assert pod["runtimeClassName"] == "nvidia"
    assert pod["nodeSelector"]["nvidia.com/gpu.present"] == "true"
    assert not pod.get("hostIPC") and not pod.get("hostNetwork")
    server = pod["containers"][0]
    assert server["image"] == plan.deployment.runtime.image
    assert "nvidia.com/gpu" not in json.dumps(server["resources"])
    assert server["resources"]["requests"]["memory"] == "16Gi"
    assert "--isolated-ipc" in server["args"]
    assert server["readinessProbe"]["httpGet"]["path"] == "/healthcheck"
    assert documents["ServiceMonitor"]["spec"]["endpoints"][0]["port"] == "http"
    assert documents["NetworkPolicy"]["spec"]["policyTypes"] == ["Ingress"]
    compile(server["command"][2], "<lmcache-preflight>", "exec")


def test_supplied_image_is_preserved_for_model_and_cache_server(plan):
    image = "registry.example/team/vllm-lmcache:latest"
    deployment = lmcache.configure_lmcache(
        replace(plan.deployment, runtime=replace(plan.deployment.runtime, image=image))
    )
    supplied = replace(plan, deployment=deployment)
    server = next(
        doc for doc in lmcache.lmcache_documents(supplied) if doc["kind"] == "DaemonSet"
    )
    assert deployment.runtime.image == image
    assert server["spec"]["template"]["spec"]["containers"][0]["image"] == image


def test_setup_key_isolates_executions_and_configuration(plan):
    first = scope_execution_release(plan, execution_name="test-one")
    second = scope_execution_release(plan, execution_name="test-two")
    assert setup_key_for_plan(first) != setup_key_for_plan(second)
    changed = replace(
        first,
        deployment=replace(first.deployment, options={"lmcache": {"cpu_gib": 16}}),
    )
    assert setup_key_for_plan(first) != setup_key_for_plan(changed)
    assert base_setup_key_for_plan(first) == base_setup_key_for_plan(changed)
    manifest, _ = _materialize_execution_name(
        render_pipelinerun(plan, pipeline_name="benchflow-e2e", setup_mode="auto")
    )
    serialized = next(
        p["value"] for p in manifest["spec"]["params"] if p["name"] == "RUN_PLAN"
    )
    scoped = load_run_plan_data(json.loads(serialized))
    assert manifest["metadata"]["annotations"][
        "benchflow.io/setup-key"
    ] == setup_key_for_plan(scoped)


@pytest.fixture
def lifecycle(monkeypatch, plan):
    state = {
        "installed_key": base_setup_key_for_plan(plan),
        "setup_state": {"platform": "rhoai"},
    }
    calls = []
    monkeypatch.setattr(platform, "require_any_command", lambda *_: "oc")
    monkeypatch.setattr(
        platform, "platform_prepare_lock", lambda *_args, **_kw: nullcontext()
    )
    monkeypatch.setattr(platform, "load_cluster_platform_state", lambda *_: dict(state))

    def persist(_cmd, _ns, payload):
        state.update(payload)
        calls.append(("persist", payload["installed_key"]))

    monkeypatch.setattr(platform, "persist_cluster_platform_state", persist)
    monkeypatch.setattr(platform, "ensure_tracing_plane", lambda *_: None)
    monkeypatch.setattr(
        platform, "ensure_lmcache", lambda *_: calls.append(("ensure",))
    )
    monkeypatch.setattr(
        platform, "remove_lmcache", lambda *_: calls.append(("remove",))
    )
    monkeypatch.setattr(
        platform, "release_lmcache_gpu_access", lambda *_: calls.append(("gpu-access",))
    )
    monkeypatch.setattr(
        platform, "setup_rhoai", lambda *_a, **_kw: pytest.fail("must reuse RHOAI")
    )
    monkeypatch.setattr(
        platform,
        "reset_rhoai_platform",
        lambda: pytest.fail("must not uninstall RHOAI"),
    )
    return state, calls


def test_switch_cache_reuses_platform_and_records_setup_state(
    plan, lifecycle, tmp_path
):
    state, calls = lifecycle
    state_path = tmp_path / "setup.json"
    result = platform.setup_platform(
        plan, context=ExecutionContext(state_path=state_path)
    )
    assert calls == [
        ("persist", ""),
        ("remove",),
        ("ensure",),
        ("persist", setup_key_for_plan(plan)),
    ]
    assert state["installed_key"] == setup_key_for_plan(plan)
    assert result["lmcache"]["setup_key"] == setup_key_for_plan(plan)
    assert json.loads(state_path.read_text()) == result
    calls.clear()
    platform.setup_platform(plan)
    assert calls == [("ensure",)]  # Same release/config keeps its warm cache.


def test_failed_setup_is_recoverable_without_reusing_native_key(
    plan, lifecycle, monkeypatch
):
    state, calls = lifecycle

    def fail(*_):
        raise lmcache.CommandError("CUDA unavailable")

    monkeypatch.setattr(platform, "ensure_lmcache", fail)
    with pytest.raises(lmcache.CommandError, match="CUDA"):
        platform.setup_platform(plan)
    assert state["installed_key"] == ""
    assert state["setup_state"]["base_setup_key"] == base_setup_key_for_plan(plan)
    with pytest.raises(ValidationError, match="platform state"):
        monkeypatch.setattr(
            "benchflow.platform_state.load_cluster_platform_state", lambda *_: state
        )
        lmcache.require_lmcache(plan, "oc")
    monkeypatch.setattr(platform, "ensure_lmcache", lambda *_: None)
    platform.setup_platform(plan)
    assert state["installed_key"] == setup_key_for_plan(plan)


def test_workflow_cleanup_removes_cache_and_retains_base_platform(
    plan, lifecycle, monkeypatch, tmp_path
):
    state, calls = lifecycle
    context = ExecutionContext(state_path=tmp_path / "setup.json")
    platform.setup_platform(plan, context=context)
    calls.clear()
    monkeypatch.setattr(
        run_plan_toolbox,
        "cleanup_deployment",
        lambda *_a, **_kw: calls.append(("model",)),
    )
    run_plan_toolbox.cleanup_run_plan(plan, context=context)
    assert calls == [
        ("model",),
        ("persist", ""),
        ("remove",),
        ("gpu-access",),
        ("persist", base_setup_key_for_plan(plan)),
    ]
    assert state["setup_state"]["platform"] == "rhoai"
    assert "lmcache" not in state["setup_state"]
    assert not context.state_path.exists()


def test_cleanup_still_removes_cache_after_model_cleanup_failure(
    plan, lifecycle, monkeypatch
):
    state, calls = lifecycle
    platform.setup_platform(plan)
    calls.clear()

    def fail(*_a, **_kw):
        raise lmcache.CommandError("model cleanup failed")

    monkeypatch.setattr(run_plan_toolbox, "cleanup_deployment", fail)
    with pytest.raises(lmcache.CommandError, match="model cleanup failed"):
        run_plan_toolbox.cleanup_run_plan(plan)
    assert ("remove",) in calls and ("gpu-access",) in calls
    assert state["installed_key"] == base_setup_key_for_plan(plan)


def test_failed_setup_cache_is_removed_by_final_cleanup(plan, lifecycle, monkeypatch):
    state, calls = lifecycle

    def fail(*_):
        raise lmcache.CommandError("cache startup failed")

    monkeypatch.setattr(platform, "ensure_lmcache", fail)
    with pytest.raises(lmcache.CommandError, match="cache startup failed"):
        platform.setup_platform(plan)
    calls.clear()
    monkeypatch.setattr(run_plan_toolbox, "cleanup_deployment", lambda *_a, **_kw: None)
    run_plan_toolbox.cleanup_run_plan(plan)
    assert ("remove",) in calls and ("gpu-access",) in calls
    assert state["installed_key"] == base_setup_key_for_plan(plan)


def test_stale_cleanup_cannot_remove_another_executions_cache(plan, lifecycle):
    state, calls = lifecycle
    later = scope_execution_release(plan, execution_name="later-execution")
    platform.setup_platform(later)
    calls.clear()
    platform.cleanup_lmcache(plan)
    assert calls == []
    assert state["installed_key"] == setup_key_for_plan(later)


@pytest.mark.parametrize(
    "failure_step", ["remove_lmcache", "release_lmcache_gpu_access"]
)
def test_cache_cleanup_failure_keeps_owner_for_retry(
    plan, lifecycle, monkeypatch, failure_step
):
    state, calls = lifecycle
    platform.setup_platform(plan)
    original = getattr(platform, failure_step)

    def fail(*_):
        raise lmcache.CommandError("cache cleanup failed")

    monkeypatch.setattr(platform, failure_step, fail)
    with pytest.raises(lmcache.CommandError, match="cache cleanup failed"):
        platform.cleanup_lmcache(plan)
    assert state["installed_key"] == ""
    assert state["setup_state"]["lmcache"]["setup_key"] == setup_key_for_plan(plan)
    monkeypatch.setattr(platform, failure_step, original)
    platform.cleanup_lmcache(plan)
    assert state["installed_key"] == base_setup_key_for_plan(plan)
    assert "lmcache" not in state["setup_state"]


def test_ensure_rejects_zero_servers_and_applies_policy_first(plan, monkeypatch):
    monkeypatch.setattr(
        lmcache, "ensure_lmcache_gpu_access", lambda *_: "runtime-class"
    )
    calls = []

    def command(args, **kwargs):
        calls.append((args, kwargs))
        if "nodes" in args:
            return SimpleNamespace(
                stdout=json.dumps({"items": [{"metadata": {"labels": {}}}]})
            )
        if "daemonset" in args:
            return SimpleNamespace(
                stdout=json.dumps(
                    {"status": {"desiredNumberScheduled": 0, "numberReady": 0}}
                )
            )
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(lmcache, "run_command", command)
    with pytest.raises(lmcache.CommandError, match="no ready server"):
        lmcache.ensure_lmcache(plan, "oc")
    applied = [
        json.loads(kw["input_text"])["kind"] for _, kw in calls if "input_text" in kw
    ]
    assert applied[0] == "NetworkPolicy"


def test_native_setup_removes_leftover_cache(plan, monkeypatch):
    monkeypatch.setattr(lmcache, "release_lmcache_gpu_access", lambda *_: None)
    calls = []
    native = replace(plan, deployment=replace(plan.deployment, options={}))
    monkeypatch.setattr(lmcache, "run_command", lambda args, **_: calls.append(args))
    lmcache.ensure_lmcache(native, "oc")
    assert len(calls) == 1
    assert calls[0][1] == "delete"
    assert "--cascade=foreground" in calls[0]


def test_llmd_recipe_overlay_and_review_assets_include_connector(plan, tmp_path):
    plan = replace(
        plan,
        deployment=replace(
            plan.deployment,
            platform="llm-d",
            mode="inference-scheduling",
            repo_ref="v0.9.0",
        ),
    )
    overlay = tmp_path / "modelserver"
    overlay.mkdir()
    (overlay / "kustomization.yaml").write_text("resources: []\nlabels: []\n")
    (overlay / "patch-vllm.yaml").write_text(
        yaml.safe_dump(
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {"name": "decode"},
                "spec": {
                    "template": {"spec": {"containers": [{"name": "modelserver"}]}}
                },
            }
        )
    )
    _patch_recipe_modelserver_overlay(plan, overlay, router_chart=True)
    patch = yaml.safe_load((overlay / "patch-vllm.yaml").read_text())
    container = patch["spec"]["template"]["spec"]["containers"][0]
    assert any("LMCacheMPConnector" in arg for arg in container["args"])
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    outputs = write_deployment_assets(plan, tmp_path / "review")
    shared = next(path for path in outputs if path.name == "platform-lmcache.yaml")
    assert "DaemonSet" in {
        doc["kind"] for doc in yaml.safe_load_all(shared.read_text())
    }


@pytest.mark.parametrize("model", ["Qwen/unconfigured-model", "unknown/hybrid"])
def test_unqualified_model_is_rejected_before_setup(plan, model):
    raw = plan.to_dict()
    raw["model"]["name"] = model
    with pytest.raises(ValidationError, match="model layouts need validation"):
        load_run_plan_data(raw)


def test_same_release_different_tensor_parallelism_changes_cache_key(plan):
    changed = replace(
        plan,
        deployment=replace(
            plan.deployment,
            runtime=replace(plan.deployment.runtime, tensor_parallelism=2),
        ),
    )
    assert setup_key_for_plan(changed) != setup_key_for_plan(plan)


def test_collect_includes_shared_cache_pods_metrics_and_manifests(
    plan, monkeypatch, tmp_path
):
    from benchflow import artifacts

    pod = {
        "metadata": {"name": "benchflow-lmcache-test", "labels": lmcache.LABELS},
        "spec": {"containers": [{"name": "lmcache-server"}]},
    }
    monkeypatch.setattr(artifacts, "require_any_command", lambda *_: "oc")
    monkeypatch.setattr(artifacts, "_collect_platform_state", lambda *_: 0)
    monkeypatch.setattr(artifacts, "_collect_cephfs_diagnostics", lambda *_a, **_kw: {})
    monkeypatch.setattr(artifacts, "collect_traces", lambda *_a, **_kw: {})
    monkeypatch.setattr(artifacts, "run_json_command", lambda *_: {"items": [pod]})
    logs = []
    monkeypatch.setattr(
        artifacts, "_collect_pod_logs", lambda _cmd, _ns, name, *_: logs.append(name)
    )

    def command(args, **_kwargs):
        stdout = ""
        if args[1] == "exec":
            stdout = "lmcache_mp_l1_read_chunks_total 4\n"
        elif args[1:3] == ["get", "daemonsets"]:
            stdout = json.dumps(
                {
                    "items": [
                        {"metadata": {"name": lmcache.NAME, "labels": lmcache.LABELS}}
                    ]
                }
            )
        elif args[-1] == "yaml":
            stdout = "kind: DaemonSet\n"
        else:
            stdout = '{"items":[]}'
        return SimpleNamespace(returncode=0, stdout=stdout)

    monkeypatch.setattr(artifacts, "run_command", command)
    artifacts.collect_artifacts(
        plan, artifacts_dir=tmp_path, include_execution_logs=False
    )
    assert logs == ["benchflow-lmcache-test"]
    assert (
        (tmp_path / "metrics/lmcache/benchflow-lmcache-test.prom")
        .read_text()
        .endswith("4\n")
    )
    assert list(tmp_path.rglob("benchflow-lmcache.yaml"))


def test_smoke_experiment_requires_and_uses_supplied_image(tmp_path):
    source = ROOT / "experiments/smoke/qwen3-06b-rhoai-lmcache-smoke.yaml"
    catalog = ProfileCatalog.load(ROOT / "profiles")
    with pytest.raises(ValidationError, match="--runtime-image"):
        resolve_experiment_matrix(load_experiment(source), catalog)
    raw = yaml.safe_load(source.read_text())
    raw["spec"]["overrides"] = {"images": {"runtime": IMAGE}}
    configured = tmp_path / "experiment.yaml"
    configured.write_text(yaml.safe_dump(raw))
    resolved = resolve_experiment_matrix(
        load_experiment(configured),
        catalog,
    )[0]
    assert resolved.deployment.runtime.image == IMAGE
    assert resolved.deployment.platform_version == "RHOAI-3.6.0-ea.2"
    assert resolved.deployment.platform_channel == "beta"
    assert resolved.benchmark.guidellm.args["constraint"][0] == {
        "kind": "max_requests",
        "count": 24,
    }
    assert lmcache.lmcache_spec(resolved.deployment).verify_readback
    assert resolved.deployment.runtime.replicas == 1
    assert resolved.deployment.runtime.tensor_parallelism == 1
    assert resolved.stages.cleanup
    assert resolved.profiles.metrics == "lmcache-smoke"
    assert "lmcache_chunks_loaded_total" in resolved.metrics.queries


@pytest.mark.parametrize(
    "cache_enabled,query_step,collection_delay",
    [(True, 15, 1), (True, 15, 120), (True, 60, 1), (False, 15, 1)],
)
def test_metrics_capture_final_cache_scrape_without_extending_benchmark(
    plan, monkeypatch, tmp_path, cache_enabled, query_step, collection_delay
):
    from benchflow.metrics import prometheus

    start = "2026-11-08T20:02:59Z"
    end = "2026-11-08T20:03:30Z"
    start_ts = datetime.fromisoformat(start).timestamp()
    end_ts = datetime.fromisoformat(end).timestamp()
    # Reproduce the real smoke: the first scrape arrived three seconds late.
    first_scrape_ts = end_ts + 3
    cache_query = 'lmcache_mp_num_chunks_loaded_total{namespace="$namespace"}'
    queries = {"model_cpu": "container_cpu_usage_seconds_total"}
    if cache_enabled:
        queries["cache_loads"] = cache_query
    plan = replace(
        plan,
        deployment=replace(
            plan.deployment,
            options=plan.deployment.options if cache_enabled else {},
        ),
        metrics=replace(
            plan.metrics,
            queries=queries,
            query_step=f"{query_step}s",
            verify_tls=False,
        ),
    )
    token = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
    original_exists, original_read_text = Path.exists, Path.read_text
    monkeypatch.setattr(
        Path, "exists", lambda p: True if p == token else original_exists(p)
    )
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda p, **kw: "test-token" if p == token else original_read_text(p, **kw),
    )
    monkeypatch.setattr(prometheus, "require_any_command", lambda *_: "oc")
    now = [end_ts + collection_delay]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(prometheus.time, "time", lambda: now[0])
    monkeypatch.setattr(prometheus.time, "sleep", sleep)
    requested = {}

    def urlopen(request, **_):
        params = parse_qs(urlsplit(request.full_url).query)
        query = params["query"][0]
        query_end = int(params["end"][0])
        assert now[0] >= query_end  # Never query a still-future scrape window.
        requested[query] = params
        result = []
        if "lmcache_mp_" in query:
            evaluation = start_ts + math.ceil(
                (first_scrape_ts - start_ts) / query_step
            ) * query_step
            if query_end >= evaluation:
                result = [{"metric": {}, "values": [[evaluation, "24"]]}]
        return StringIO(json.dumps({"status": "success", "data": {"result": result}}))

    monkeypatch.setattr(prometheus.urllib.request, "urlopen", urlopen)
    prometheus.collect_metrics(
        plan,
        benchmark_start_time=start,
        benchmark_end_time=end,
        artifacts_dir=tmp_path,
    )
    assert int(requested[queries["model_cpu"]]["end"][0]) == end_ts
    expected_end = end_ts
    if cache_enabled:
        expected_end = start_ts + math.ceil(
            (end_ts + 30 - start_ts) / query_step
        ) * query_step
        cache_request = requested[cache_query.replace("$namespace", "benchflow")]
        assert int(cache_request["end"][0]) == expected_end
        assert int(cache_request["step"][0][:-1]) == query_step
    assert sleeps == (
        [expected_end - (end_ts + collection_delay)]
        if expected_end > end_ts + collection_delay
        else []
    )
    for name in ("metrics_summary.json", "archive_index.json"):
        saved = json.loads((tmp_path / "metrics" / name).read_text())
        assert saved["benchmark_start_time"] == start
        assert saved["benchmark_end_time"] == end
        assert saved["collection_end_time"] == datetime.fromtimestamp(
            expected_end, timezone.utc
        ).isoformat().replace("+00:00", "Z")
        if cache_enabled:
            entry = saved.get("queries", saved.get("metrics"))["cache_loads"]
            assert entry["query_end_time"] == saved["collection_end_time"]
            assert entry["series_count"] == 1
    if cache_enabled:
        payload = json.loads(
            (tmp_path / "metrics/prometheus/cache_loads.json").read_text()
        )
        assert payload["data"]["result"][0]["values"][0][1] == "24"


@pytest.mark.parametrize(
    "after_counts,restarts,passes",
    [((11, 6), 0, True), ((11, 5), 0, False), ((10, 6), 0, False), ((11, 6), 1, False)],
)
def test_readback_gate_requires_new_stores_and_successful_loads(
    plan, monkeypatch, tmp_path, after_counts, restarts, passes
):
    before = {"server": {"uid": "same", "stores": 10, "loads": 5, "restarts": 0}}
    after = {
        "server": {
            "uid": "same",
            "stores": after_counts[0],
            "loads": after_counts[1],
            "restarts": restarts,
        }
    }
    monkeypatch.setattr(lmcache, "lmcache_readback_snapshot", lambda *_: after)
    if passes:
        lmcache.verify_lmcache_readback(plan, "oc", before, tmp_path)
    else:
        with pytest.raises(lmcache.CommandError, match="smoke test failed"):
            lmcache.verify_lmcache_readback(plan, "oc", before, tmp_path)
    evidence = json.loads((tmp_path / "lmcache-readback.json").read_text())
    assert evidence["passed"] is passes


def test_readback_gate_failure_preserves_benchmark_identity(
    plan, monkeypatch, tmp_path
):
    from benchflow.toolbox import benchmark

    runtime_plan = replace(
        plan,
        deployment=replace(
            plan.deployment, options={"lmcache": {"verify_readback": True}}
        ),
    )
    monkeypatch.setattr(benchmark, "require_any_command", lambda *_: "oc")
    monkeypatch.setattr(benchmark, "lmcache_readback_snapshot", lambda *_: {})
    monkeypatch.setattr(
        benchmark, "run_benchmark", lambda **_: ("run-1", "start", "end")
    )

    def fail(*_):
        raise lmcache.CommandError("no successful readback")

    monkeypatch.setattr(benchmark, "verify_lmcache_readback", fail)
    with pytest.raises(
        benchmark.BenchmarkRunFailed, match="no successful readback"
    ) as caught:
        benchmark.run_plan_benchmark(
            runtime_plan, target_url="http://example.test", output_dir=tmp_path
        )
    assert caught.value.run_id == "run-1"
    assert caught.value.start_time == "start"
    assert caught.value.end_time == "end"
