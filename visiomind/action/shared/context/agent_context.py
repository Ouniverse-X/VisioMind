from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from visiomind.action.shared.context.task import TaskRequest
from visiomind.action.shared.results.agent_result import AgentResult


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExecutionContext:
    trace_id: str
    task_request: TaskRequest
    started_at: str = field(default_factory=utc_now_iso)
    runtime_state: dict[str, Any] = field(default_factory=dict)
    results: list[AgentResult] = field(default_factory=list)
