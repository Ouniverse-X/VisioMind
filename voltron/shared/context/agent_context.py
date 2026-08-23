"""Shared execution context for orchestrated task lifecycles."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from voltron.shared.context.task import TaskRequest
from voltron.shared.results.agent_result import AgentResult


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp string."""

    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExecutionContext:
    """Context shared for the lifetime of one orchestrated task."""

    trace_id: str
    task_request: TaskRequest
    started_at: str = field(default_factory=utc_now_iso)
    runtime_state: dict[str, Any] = field(default_factory=dict)
    results: list[AgentResult] = field(default_factory=list)
