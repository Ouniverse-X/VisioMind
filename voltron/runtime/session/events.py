"""Runtime event models for user-facing Voltron sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VoltronEvent:
    """A normalized event emitted by Voltron session and orchestration layers."""

    event_type: str
    source: str
    message: str
    payload: dict[str, Any] = field(default_factory=dict)
    task_id: str | None = None


__all__ = ["VoltronEvent"]
