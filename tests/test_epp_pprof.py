from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
import yaml

from benchflow.deploy.llmd import (
    _ensure_epp_pprof_rbac,
    _patch_scheduler_values,
)
from benchflow.benchmark import BenchmarkRunFailed
from benchflow.llmd_epp import LlmdEppIdentity
from benchflow.loaders import (
    ProfileCatalog,
    load_experiment,
    load_metrics_profile,
    load_run_plan_data,
)
from benchflow.matrix import resolve_experiment_matrix
from benchflow.mlflow_upload import _benchmark_workspace_artifact_root
from benchflow.models import EppPprofSpec, ValidationError
from benchflow.profiling.epp_pprof import EppPprofCaptureError, EppPprofSession
from benchflow.toolbox import benchmark as benchmark_toolbox


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tracing_plan(tmp_path: Path):
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        """apiVersion: benchflow.io/v1alpha1
kind: Experiment
metadata:
  name: pprof-test
spec:
  model:
    name: Qwen/Qwen3.6-35B-A3B
  deployment_profile: llm-d-optimized-baseline-scalability
  benchmark_profile: aiperf-smoke
  metrics_profile: detailed-tracing
  namespace: benchflow
""",
        encoding="utf-8",
    )
    catalog = ProfileCatalog.load(REPO_ROOT / "profiles")
    return resolve_experiment_matrix(load_experiment(experiment), catalog)[0]


def _pod(name: str, uid: str, ip: str) -> dict:
    return {
        "metadata": {"name": name, "uid": uid},
        "spec": {
            "containers": [
                {"name": "epp", "image": "ghcr.io/llm-d/llm-d-router:v0.10.0"}
            ]
        },
        "status": {
            "podIP": ip,
            "conditions": [{"type": "Ready", "status": "True"}],
            "containerStatuses": [
                {
                    "name": "epp",
                    "image": "ghcr.io/llm-d/llm-d-router:v0.10.0",
                    "imageID": "sha256:abc",
                }
            ],
        },
    }


def test_metrics_profile_loads_epp_pprof_defaults(tmp_path: Path) -> None:
    profile_path = tmp_path / "metrics.yaml"
    profile_path.write_text(
        """apiVersion: benchflow.io/v1alpha1
kind: MetricsProfile
metadata:
  name: pprof
spec:
  prometheus_url: https://prometheus.example
  epp_pprof: {}
""",
        encoding="utf-8",
    )

    profile = load_metrics_profile(profile_path)

    assert profile.spec.epp_pprof == EppPprofSpec(
        start_delay_seconds=60,
        cpu_duration_seconds=30,
        collect_heap=True,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("start_delay_seconds", -1), ("cpu_duration_seconds", 0)],
)
def test_metrics_profile_rejects_invalid_epp_pprof_timing(
    tmp_path: Path, field: str, value: int
) -> None:
    profile_path = tmp_path / "metrics.yaml"
    profile_path.write_text(
        f"""apiVersion: benchflow.io/v1alpha1
kind: MetricsProfile
metadata:
  name: pprof
spec:
  prometheus_url: https://prometheus.example
  epp_pprof:
    {field}: {value}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match=field):
        load_metrics_profile(profile_path)


def test_epp_pprof_survives_run_plan_round_trip(tracing_plan) -> None:
    tracing_plan.metrics.epp_pprof = EppPprofSpec(
        start_delay_seconds=12,
        cpu_duration_seconds=17,
        collect_heap=False,
    )

    restored = load_run_plan_data(tracing_plan.to_dict())

    assert restored.metrics.epp_pprof == tracing_plan.metrics.epp_pprof


def test_epp_pprof_rejects_non_llmd_deployment(tmp_path: Path) -> None:
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        """apiVersion: benchflow.io/v1alpha1
kind: Experiment
metadata:
  name: unsupported-pprof
spec:
  model:
    name: Qwen/Qwen3-0.6B
  deployment_profile: rhoai-distributed-default
  benchmark_profile: aiperf-smoke
  metrics_profile: epp-tracing-full-pprof
""",
        encoding="utf-8",
    )
    catalog = ProfileCatalog.load(REPO_ROOT / "profiles")

    with pytest.raises(ValidationError, match="only for llm-d"):
        resolve_experiment_matrix(load_experiment(experiment_path), catalog)


def test_router_values_make_pprof_explicitly_opt_in(
    tracing_plan, tmp_path: Path
) -> None:
    values_path = tmp_path / "values.yaml"
    values_path.write_text("router: {}\n", encoding="utf-8")

    _patch_scheduler_values(
        tracing_plan, values_path, recipe_layout=True, router_chart=True
    )
    values = yaml.safe_load(values_path.read_text(encoding="utf-8"))
    assert values["router"]["epp"]["flags"]["enable-pprof"] is False

    tracing_plan.metrics.epp_pprof = EppPprofSpec()
    _patch_scheduler_values(
        tracing_plan, values_path, recipe_layout=True, router_chart=True
    )
    values = yaml.safe_load(values_path.read_text(encoding="utf-8"))
    assert values["router"]["epp"]["flags"]["enable-pprof"] is True


def test_epp_pprof_rbac_is_scoped_to_debug_endpoint(monkeypatch, tracing_plan) -> None:
    tracing_plan.metrics.epp_pprof = EppPprofSpec()
    calls: list[tuple[list[str], str]] = []

    def fake_run_command(args, *, input_text=None, **_kwargs):
        calls.append((args, input_text or ""))
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr("benchflow.deploy.llmd.run_command", fake_run_command)

    _ensure_epp_pprof_rbac(tracing_plan, "oc")

    documents = list(yaml.safe_load_all(calls[0][1]))
    role, binding = documents
    assert role["kind"] == "ClusterRole"
    assert role["rules"] == [
        {
            "nonResourceURLs": ["/debug/pprof", "/debug/pprof/*"],
            "verbs": ["get"],
        }
    ]
    assert binding["subjects"] == [
        {
            "kind": "ServiceAccount",
            "name": tracing_plan.service_account,
            "namespace": tracing_plan.deployment.namespace,
        }
    ]


def test_captures_all_ready_epp_pods_in_parallel(
    monkeypatch, tmp_path: Path, tracing_plan
) -> None:
    tracing_plan.metrics.epp_pprof = EppPprofSpec(
        start_delay_seconds=0, cpu_duration_seconds=30, collect_heap=True
    )
    pods = [_pod("epp-a", "uid-a", "10.0.0.1"), _pod("epp-b", "uid-b", "10.0.0.2")]
    barrier = threading.Barrier(2)

    monkeypatch.setattr(
        "benchflow.profiling.epp_pprof.require_any_command", lambda *_args: "oc"
    )
    monkeypatch.setattr(
        "benchflow.profiling.epp_pprof.resolve_llmd_epp_identity",
        lambda *_args: LlmdEppIdentity(
            helm_release_name="gaie-release",
            deployment_name="gaie-release-epp",
            selectors=("app=epp",),
        ),
    )

    def fake_run_json_command(args):
        if "deployment" in args:
            return {"spec": {"replicas": 2}}
        return {"items": pods}

    def fake_request(url: str, *, timeout_seconds: int) -> bytes:
        assert timeout_seconds > 0
        if "/profile?" in url:
            barrier.wait(timeout=1)
            return f"cpu:{url}".encode()
        return f"heap:{url}".encode()

    monkeypatch.setattr(
        "benchflow.profiling.epp_pprof.run_json_command", fake_run_json_command
    )
    monkeypatch.setattr("benchflow.profiling.epp_pprof._request_profile", fake_request)

    session = EppPprofSession(tracing_plan, tmp_path, tracing_plan.metrics.epp_pprof)
    session.start()
    assert session._thread is not None
    session._thread.join(timeout=2)
    session.finish("2026-09-24T12:00:00Z")

    summary = json.loads(
        (tmp_path / "pprof/epp/capture-summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "complete"
    assert summary["expected_pods"] == 2
    assert summary["complete_pods"] == ["epp-a", "epp-b"]
    for pod in pods:
        pod_dir = tmp_path / "pprof/epp" / pod["metadata"]["name"]
        assert (pod_dir / "cpu.pprof").stat().st_size > 0
        assert (pod_dir / "heap.pprof").stat().st_size > 0
        capture = json.loads((pod_dir / "capture.json").read_text(encoding="utf-8"))
        assert capture["benchmark_started_at"]
        assert capture["benchmark_ended_at"] == "2026-09-24T12:00:00Z"
        assert capture["capture_started_at"]
        assert capture["capture_ended_at"]
        assert capture["status"] == "complete"


def test_partial_capture_is_preserved_and_fails_requirement(
    monkeypatch, tmp_path: Path, tracing_plan
) -> None:
    tracing_plan.metrics.epp_pprof = EppPprofSpec(
        start_delay_seconds=0, cpu_duration_seconds=30, collect_heap=True
    )
    pods = [_pod("epp-a", "uid-a", "10.0.0.1"), _pod("epp-b", "uid-b", "10.0.0.2")]
    monkeypatch.setattr(
        "benchflow.profiling.epp_pprof.require_any_command", lambda *_args: "oc"
    )
    monkeypatch.setattr(
        "benchflow.profiling.epp_pprof.resolve_llmd_epp_identity",
        lambda *_args: LlmdEppIdentity(
            "gaie-release", "gaie-release-epp", ("app=epp",)
        ),
    )
    monkeypatch.setattr(
        "benchflow.profiling.epp_pprof.run_json_command",
        lambda args: (
            {"spec": {"replicas": 2}} if "deployment" in args else {"items": pods}
        ),
    )

    def fake_request(url: str, *, timeout_seconds: int) -> bytes:
        del timeout_seconds
        if "10.0.0.2" in url and "/profile?" in url:
            raise OSError("connection reset")
        return b"profile"

    monkeypatch.setattr("benchflow.profiling.epp_pprof._request_profile", fake_request)

    session = EppPprofSession(tracing_plan, tmp_path, tracing_plan.metrics.epp_pprof)
    session.start()
    assert session._thread is not None
    session._thread.join(timeout=2)
    with pytest.raises(EppPprofCaptureError, match="incomplete"):
        session.finish("2026-09-24T12:00:00Z")

    failed_capture = json.loads(
        (tmp_path / "pprof/epp/epp-b/capture.json").read_text(encoding="utf-8")
    )
    assert failed_capture["status"] == "partial"
    assert failed_capture["cpu_profile_bytes"] == 0
    assert failed_capture["heap_profile_bytes"] > 0
    assert "connection reset" in failed_capture["error"]
    assert (tmp_path / "pprof/epp/epp-a/cpu.pprof").exists()


def test_unready_pod_gets_failed_capture_metadata(
    monkeypatch, tmp_path: Path, tracing_plan
) -> None:
    tracing_plan.metrics.epp_pprof = EppPprofSpec(start_delay_seconds=0)
    pod = _pod("epp-a", "uid-a", "10.0.0.1")
    pod["status"]["conditions"] = [{"type": "Ready", "status": "False"}]
    monkeypatch.setattr(
        "benchflow.profiling.epp_pprof.require_any_command", lambda *_args: "oc"
    )
    monkeypatch.setattr(
        "benchflow.profiling.epp_pprof.resolve_llmd_epp_identity",
        lambda *_args: LlmdEppIdentity(
            "gaie-release", "gaie-release-epp", ("app=epp",)
        ),
    )
    monkeypatch.setattr(
        "benchflow.profiling.epp_pprof.run_json_command",
        lambda args: (
            {"spec": {"replicas": 1}} if "deployment" in args else {"items": [pod]}
        ),
    )

    session = EppPprofSession(tracing_plan, tmp_path, tracing_plan.metrics.epp_pprof)
    session.start()
    assert session._thread is not None
    session._thread.join(timeout=2)
    with pytest.raises(EppPprofCaptureError, match="were ready"):
        session.finish("2026-09-24T12:00:00Z")

    capture = json.loads(
        (tmp_path / "pprof/epp/epp-a/capture.json").read_text(encoding="utf-8")
    )
    assert capture["status"] == "failed"
    assert capture["capture_started_at"] is None
    assert capture["capture_ended_at"] is None
    assert "not ready" in capture["error"]


def test_mlflow_fallback_preserves_pprof_subtree() -> None:
    assert (
        _benchmark_workspace_artifact_root(Path("pprof/epp/pod-a/capture.json"))
        == "benchmark"
    )


def test_benchmark_starts_pprof_at_load_generator_launch(
    monkeypatch, tmp_path: Path, tracing_plan
) -> None:
    tracing_plan.metrics.epp_pprof = EppPprofSpec(
        start_delay_seconds=7, cpu_duration_seconds=11, collect_heap=False
    )
    events: list[object] = []

    class FakeSession:
        def __init__(self, plan, output_dir, spec):
            events.append((plan, output_dir, spec))

        def start(self):
            events.append("start")

        def finish(self, ended_at):
            events.append(("finish", ended_at))

    def fake_run_benchmark(**kwargs):
        events.append(kwargs["extra_tags"])
        kwargs["on_load_generator_launch"]()
        return "run-id", "start", "end"

    monkeypatch.setattr(benchmark_toolbox, "EppPprofSession", FakeSession)
    monkeypatch.setattr(benchmark_toolbox, "run_benchmark", fake_run_benchmark)

    outcome = benchmark_toolbox.run_plan_benchmark(
        tracing_plan,
        target_url="http://example.test",
        output_dir=tmp_path,
        enable_mlflow=False,
    )

    assert outcome.run_id == "run-id"
    assert "start" in events
    assert ("finish", "end") in events
    tags = next(event for event in events if isinstance(event, dict))
    assert tags["epp_pprof"] == "true"
    assert tags["epp_pprof_start_delay_seconds"] == "7"
    assert tags["epp_pprof_cpu_duration_seconds"] == "11"


def test_successful_benchmark_fails_when_required_pprof_is_incomplete(
    monkeypatch, tmp_path: Path, tracing_plan
) -> None:
    tracing_plan.metrics.epp_pprof = EppPprofSpec(start_delay_seconds=0)

    class FakeSession:
        def __init__(self, *_args):
            pass

        def start(self):
            pass

        def finish(self, _ended_at):
            raise EppPprofCaptureError("expected 2 EPP pods, discovered 1")

    monkeypatch.setattr(benchmark_toolbox, "EppPprofSession", FakeSession)
    monkeypatch.setattr(
        benchmark_toolbox,
        "run_benchmark",
        lambda **_kwargs: ("run-id", "start", "end"),
    )

    with pytest.raises(BenchmarkRunFailed, match="capture requirement failed") as exc:
        benchmark_toolbox.run_plan_benchmark(
            tracing_plan,
            target_url="http://example.test",
            output_dir=tmp_path,
            enable_mlflow=False,
        )

    assert exc.value.run_id == "run-id"
    assert exc.value.start_time == "start"
    assert exc.value.end_time == "end"
    assert (
        _benchmark_workspace_artifact_root(Path("pprof/epp/pod-a/cpu.pprof"))
        == "benchmark"
    )
