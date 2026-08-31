from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from voltron.shared.enums import AgentStatus


@dataclass
class AgentResult:
    subtask_id: str
    status: AgentStatus
    result: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    state_changes: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: int = 0

    runtime_artifacts: dict[str, Any] = field(default_factory=dict)
