from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    execution_name: str = ""
    workspace_dir: Path | None = None
    manifests_dir: Path | None = None
    models_storage_path: Path | None = None
    artifacts_dir: Path | None = None
    state_path: Path | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkOutcome:
    run_id: str
    start_time: str
    end_time: str


@dataclass(frozen=True, slots=True)
class ExecutionSummary:
    name: str
    namespace: str
    experiment: str
    platform: str
    mode: str
    backend: str
    status: str
    finished: bool
    succeeded: bool
    start_time: str
    completion_time: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "namespace": self.namespace,
            "experiment": self.experiment,
            "platform": self.platform,
            "mode": self.mode,
            "backend": self.backend,
            "status": self.status,
            "finished": self.finished,
            "succeeded": self.succeeded,
            "start_time": self.start_time,
            "completion_time": self.completion_time,
            "message": self.message,
        }
