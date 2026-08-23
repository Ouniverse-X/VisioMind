"""Shared result models emitted by agents and runtime flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from voltron.shared.enums import AgentStatus


@dataclass
class AgentResult:
    """Standardized result envelope returned by any agent."""

    subtask_id: str
    status: AgentStatus
    result: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    state_changes: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: int = 0
    # Internal-only execution artifacts for runtime adapters (e.g. projected action).
    # This field is intentionally excluded from public response serialization.
    runtime_artifacts: dict[str, Any] = field(default_factory=dict)
