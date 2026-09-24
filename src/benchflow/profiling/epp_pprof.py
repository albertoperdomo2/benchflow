from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import ProxyHandler, Request, build_opener

from ..cluster import require_any_command, run_json_command
from ..llmd_epp import resolve_llmd_epp_identity
from ..models import EppPprofSpec, ResolvedRunPlan
from ..ui import detail, step

_METRICS_PORT = 9090
_SERVICE_ACCOUNT_TOKEN = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")


class EppPprofCaptureError(RuntimeError):
    """Raised after preserving artifacts when required EPP capture is incomplete."""


def _iso8601_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _ready(item: dict[str, Any]) -> bool:
    if item.get("metadata", {}).get("deletionTimestamp"):
        return False
    return any(
        condition.get("type") == "Ready" and condition.get("status") == "True"
        for condition in item.get("status", {}).get("conditions", []) or []
        if isinstance(condition, dict)
    )


def _epp_container(item: dict[str, Any]) -> dict[str, str]:
    spec_containers = item.get("spec", {}).get("containers", []) or []
    status_containers = item.get("status", {}).get("containerStatuses", []) or []
    statuses = {
        str(status.get("name") or ""): status
        for status in status_containers
        if isinstance(status, dict)
    }
    selected: dict[str, Any] = {}
    for container in spec_containers:
        if str(container.get("name") or "") == "epp":
            selected = container
            break
    if not selected and spec_containers:
        selected = spec_containers[0]
    name = str(selected.get("name") or "")
    status = statuses.get(name, {})
    return {
        "container": name,
        "epp_image": str(selected.get("image") or status.get("image") or ""),
        "epp_image_id": str(status.get("imageID") or ""),
    }


def _service_account_token() -> str:
    try:
        return _SERVICE_ACCOUNT_TOKEN.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _request_profile(url: str, *, timeout_seconds: int) -> bytes:
    headers = {"Accept": "application/octet-stream"}
    token = _service_account_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)  # noqa: S310
    opener = build_opener(ProxyHandler({}))
    with opener.open(request, timeout=timeout_seconds) as response:  # noqa: S310
        return response.read()


class EppPprofSession:
    """Capture all EPP replicas over one concurrent benchmark window."""

    def __init__(
        self, plan: ResolvedRunPlan, output_dir: Path, spec: EppPprofSpec
    ) -> None:
        self.plan = plan
        self.output_dir = output_dir / "pprof" / "epp"
        self.spec = spec
        self._benchmark_started_at = ""
        self._benchmark_ended_at = ""
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._summary: dict[str, Any] = {}

    def start(self) -> None:
        if self._thread is not None:
            return
        self._benchmark_started_at = _iso8601_now()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run,
            name="benchflow-epp-pprof",
            daemon=True,
        )
        self._thread.start()
        detail(
            "Scheduled concurrent EPP pprof capture "
            f"{self.spec.start_delay_seconds}s after load-generator launch"
        )

    def finish(self, benchmark_ended_at: str) -> None:
        self._benchmark_ended_at = benchmark_ended_at or _iso8601_now()
        self._cancel.set()
        if self._thread is None:
            self._summary = self._base_summary(
                status="failed",
                error="load generator was never launched; EPP pprof capture did not start",
            )
        else:
            self._thread.join()
        self._finalize_metadata()
        if self._summary.get("status") != "complete":
            raise EppPprofCaptureError(
                str(self._summary.get("error") or "EPP pprof capture was incomplete")
            )

    def _base_summary(self, *, status: str, error: str | None) -> dict[str, Any]:
        return {
            "benchmark_started_at": self._benchmark_started_at or None,
            "benchmark_ended_at": self._benchmark_ended_at or None,
            "capture_delay_seconds": self.spec.start_delay_seconds,
            "cpu_duration_seconds": self.spec.cpu_duration_seconds,
            "collect_heap": self.spec.collect_heap,
            "expected_pods": 0,
            "discovered_pods": [],
            "complete_pods": [],
            "partial_pods": [],
            "failed_pods": [],
            "status": status,
            "error": error,
        }

    def _run(self) -> None:
        if self._cancel.wait(self.spec.start_delay_seconds):
            self._summary = self._base_summary(
                status="failed",
                error="benchmark ended before the scheduled EPP pprof capture began",
            )
            return

        step("Capturing CPU and heap profiles from all EPP replicas in parallel")
        try:
            kubectl_cmd = require_any_command("oc", "kubectl")
            identity = resolve_llmd_epp_identity(
                self.plan.deployment.namespace,
                self.plan.deployment.release_name,
                self.plan.deployment.gateway,
                kubectl_cmd,
            )
            deployment = run_json_command(
                [
                    kubectl_cmd,
                    "get",
                    "deployment",
                    identity.deployment_name,
                    "-n",
                    self.plan.deployment.namespace,
                    "-o",
                    "json",
                ]
            )
            expected = int(deployment.get("spec", {}).get("replicas", 1) or 1)
            pods_by_uid: dict[str, dict[str, Any]] = {}
            for selector in identity.selectors:
                payload = run_json_command(
                    [
                        kubectl_cmd,
                        "get",
                        "pods",
                        "-n",
                        self.plan.deployment.namespace,
                        "-l",
                        selector,
                        "-o",
                        "json",
                    ]
                )
                for pod in payload.get("items", []) or []:
                    uid = str(pod.get("metadata", {}).get("uid") or "")
                    if uid:
                        pods_by_uid[uid] = pod
                if pods_by_uid:
                    break
            discovered = list(pods_by_uid.values())
            ready = [pod for pod in discovered if _ready(pod)]
            unready = [pod for pod in discovered if not _ready(pod)]
            results = [self._record_uncaptured_pod(pod) for pod in unready]
            results.extend(self._capture_ready_pods(ready))
            self._summary = self._summarize(expected, discovered, results)
        except Exception as exc:  # noqa: BLE001
            self._summary = self._base_summary(status="failed", error=str(exc))

    def _capture_ready_pods(self, pods: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not pods:
            return []
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=len(pods)) as executor:
            futures = [executor.submit(self._capture_pod, pod) for pod in pods]
            for future in as_completed(futures):
                results.append(future.result())
        return results

    def _capture_pod(self, pod: dict[str, Any]) -> dict[str, Any]:
        metadata = pod.get("metadata", {}) or {}
        status = pod.get("status", {}) or {}
        pod_name = str(metadata.get("name") or "unknown-pod")
        pod_ip = str(status.get("podIP") or "")
        pod_host = f"[{pod_ip}]" if ":" in pod_ip else pod_ip
        container = _epp_container(pod)
        pod_dir = self.output_dir / pod_name
        pod_dir.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {
            "pod_name": pod_name,
            "pod_uid": str(metadata.get("uid") or ""),
            "pod_ip": pod_ip,
            **container,
            "metrics_port": _METRICS_PORT,
            "capture_started_at": _iso8601_now(),
            "capture_ended_at": None,
            "cpu_duration_seconds": self.spec.cpu_duration_seconds,
            "benchmark_started_at": self._benchmark_started_at,
            "benchmark_ended_at": None,
            "tracing_mode": self.plan.metrics.tracing.mode,
            "sampling_ratio": self.plan.metrics.tracing.sample_ratio,
            "status": "failed",
            "cpu_profile_bytes": 0,
            "heap_profile_bytes": 0,
            "error": None,
        }
        errors: list[str] = []
        try:
            cpu = _request_profile(
                f"http://{pod_host}:{_METRICS_PORT}/debug/pprof/profile"
                f"?seconds={self.spec.cpu_duration_seconds}",
                timeout_seconds=self.spec.cpu_duration_seconds + 30,
            )
            if not cpu:
                raise ValueError("CPU profile response was empty")
            (pod_dir / "cpu.pprof").write_bytes(cpu)
            record["cpu_profile_bytes"] = len(cpu)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"cpu: {exc}")

        if self.spec.collect_heap:
            try:
                heap = _request_profile(
                    f"http://{pod_host}:{_METRICS_PORT}/debug/pprof/heap",
                    timeout_seconds=30,
                )
                if not heap:
                    raise ValueError("heap profile response was empty")
                (pod_dir / "heap.pprof").write_bytes(heap)
                record["heap_profile_bytes"] = len(heap)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"heap: {exc}")

        record["capture_ended_at"] = _iso8601_now()
        successful = int(record["cpu_profile_bytes"] > 0) + int(
            not self.spec.collect_heap or record["heap_profile_bytes"] > 0
        )
        required = 1 + int(self.spec.collect_heap)
        record["status"] = (
            "complete"
            if successful == required
            else "partial"
            if successful
            else "failed"
        )
        record["error"] = "; ".join(errors) or None
        _write_json(pod_dir / "capture.json", record)
        return record

    def _record_uncaptured_pod(self, pod: dict[str, Any]) -> dict[str, Any]:
        metadata = pod.get("metadata", {}) or {}
        status = pod.get("status", {}) or {}
        pod_name = str(metadata.get("name") or "unknown-pod")
        record: dict[str, Any] = {
            "pod_name": pod_name,
            "pod_uid": str(metadata.get("uid") or ""),
            "pod_ip": str(status.get("podIP") or ""),
            **_epp_container(pod),
            "metrics_port": _METRICS_PORT,
            "capture_started_at": None,
            "capture_ended_at": None,
            "cpu_duration_seconds": self.spec.cpu_duration_seconds,
            "benchmark_started_at": self._benchmark_started_at,
            "benchmark_ended_at": None,
            "tracing_mode": self.plan.metrics.tracing.mode,
            "sampling_ratio": self.plan.metrics.tracing.sample_ratio,
            "status": "failed",
            "cpu_profile_bytes": 0,
            "heap_profile_bytes": 0,
            "error": "EPP pod was not ready at the scheduled capture time",
        }
        _write_json(self.output_dir / pod_name / "capture.json", record)
        return record

    def _summarize(
        self,
        expected: int,
        discovered: list[dict[str, Any]],
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        summary = self._base_summary(status="failed", error=None)
        summary["expected_pods"] = expected
        summary["discovered_pods"] = sorted(
            str(pod.get("metadata", {}).get("name") or "") for pod in discovered
        )
        for state, key in (
            ("complete", "complete_pods"),
            ("partial", "partial_pods"),
            ("failed", "failed_pods"),
        ):
            summary[key] = sorted(
                str(result["pod_name"])
                for result in results
                if result.get("status") == state
            )
        errors: list[str] = []
        if len(discovered) != expected:
            errors.append(
                f"expected {expected} EPP pod(s), discovered {len(discovered)}"
            )
        ready_count = sum(bool(result.get("capture_started_at")) for result in results)
        if ready_count != len(discovered):
            errors.append(
                f"only {ready_count}/{len(discovered)} discovered EPP pod(s) were ready"
            )
        incomplete = len(summary["partial_pods"]) + len(summary["failed_pods"])
        if incomplete:
            errors.append(f"{incomplete} EPP pod capture(s) were incomplete")
        if not discovered:
            errors.append("no EPP pods were discovered")
        summary["status"] = "complete" if not errors else "failed"
        summary["error"] = "; ".join(errors) or None
        return summary

    def _finalize_metadata(self) -> None:
        self._summary["benchmark_ended_at"] = self._benchmark_ended_at or None
        for capture_path in self.output_dir.glob("*/capture.json"):
            payload = json.loads(capture_path.read_text(encoding="utf-8"))
            payload["benchmark_ended_at"] = self._benchmark_ended_at or None
            _write_json(capture_path, payload)
        _write_json(self.output_dir / "capture-summary.json", self._summary)
