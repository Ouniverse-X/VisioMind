from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable


@dataclass
class EventRecord:
    ts: str
    event: str
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        event: str,
        payload: dict[str, Any],
        now: Callable[[], datetime] | None = None,
    ) -> "EventRecord":
        timestamp = (now or datetime.now)().isoformat(timespec="seconds")
        return cls(ts=timestamp, event=event, payload=dict(payload))

    def to_payload(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "event": self.event,
            "payload": dict(self.payload),
        }
