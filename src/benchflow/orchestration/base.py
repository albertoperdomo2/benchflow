from __future__ import annotations

from typing import Any, Protocol

from ..contracts import ExecutionSummary, ResolvedRunPlan


class ExecutionOrchestrator(Protocol):
    name: str

    def render_run(
        self,
        plan: ResolvedRunPlan,
        *,
        execution_name: str,
        setup_mode: str,
        teardown: bool,
    ) -> dict[str, Any]: ...

    def render_matrix(
        self,
        plans: list[ResolvedRunPlan],
        *,
        execution_name: str,
        child_execution_name: str,
    ) -> dict[str, Any]: ...

    def get(self, namespace: str, name: str) -> dict[str, Any] | None: ...

    def list(
        self, namespace: str, *, label_selector: str = ""
    ) -> list[dict[str, Any]]: ...

    def summarize(self, resource: dict[str, Any]) -> ExecutionSummary: ...

    def cancel(self, namespace: str, name: str) -> None: ...

    def follow(self, namespace: str, name: str, *, poll_interval: int = 5) -> bool: ...
