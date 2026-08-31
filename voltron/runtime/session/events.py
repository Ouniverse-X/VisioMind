from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VoltronEvent:
    event_type: str
    source: str
    message: str
    payload: dict[str, Any] = field(default_factory=dict)
    task_id: str | None = None


__all__ = ["VoltronEvent"]
