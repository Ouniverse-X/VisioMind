from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExecutionControlSignal:
    action: str
    reason: str | None = None
    requested_by: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "requested_by": self.requested_by,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ExecutionControlSignal":
        return cls(
            action=str(payload.get("action", "")),
            reason=payload.get("reason"),
            requested_by=payload.get("requested_by"),
            metadata=dict(payload.get("metadata") or {}),
        )


def request_stop(
    *,
    reason: str | None = None,
    requested_by: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ExecutionControlSignal:
    return ExecutionControlSignal(
        action="stop",
        reason=reason,
        requested_by=requested_by,
        metadata=dict(metadata or {}),
    )


def request_resume(
    *,
    reason: str | None = None,
    requested_by: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ExecutionControlSignal:
    return ExecutionControlSignal(
        action="resume",
        reason=reason,
        requested_by=requested_by,
        metadata=dict(metadata or {}),
    )
