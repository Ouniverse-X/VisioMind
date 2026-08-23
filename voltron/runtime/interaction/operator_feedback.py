"""Structured operator feedback models for runtime interaction surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OperatorFeedback:
    """Serializable operator-facing feedback payload."""

    message: str
    severity: str = "info"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "OperatorFeedback":
        return cls(
            message=str(payload.get("message", "")),
            severity=str(payload.get("severity", "info")),
            metadata=dict(payload.get("metadata") or {}),
        )


def build_operator_feedback(
    *,
    message: str,
    severity: str = "info",
    metadata: dict[str, Any] | None = None,
) -> OperatorFeedback:
    return OperatorFeedback(
        message=message,
        severity=severity,
        metadata=dict(metadata or {}),
    )
