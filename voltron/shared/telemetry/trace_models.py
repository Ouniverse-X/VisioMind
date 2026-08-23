"""Shared trace telemetry models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .event_models import EventRecord


@dataclass
class TraceRecord:
    """Structured trace/span record for runtime execution flows."""

    trace_id: str
    span_id: str
    name: str
    status: str = "ok"
    parent_span_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[EventRecord] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "name": self.name,
            "status": self.status,
            "parent_span_id": self.parent_span_id,
            "attributes": dict(self.attributes),
            "events": [event.to_payload() for event in self.events],
        }
