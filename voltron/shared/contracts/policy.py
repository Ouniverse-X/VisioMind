"""Policy adapter interface for GR00T or alternative action backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class PolicyRequest:
    """Canonical request envelope for policy backends."""

    observation: dict[str, Any]
    options: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyResult:
    """Canonical action/result envelope returned by policy backends."""

    action: dict[str, Any]
    info: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class PolicyAdapter(Protocol):
    """Minimal policy contract consumed by VLN/VLA agents."""

    def ping(self) -> bool:
        """Check backend health."""

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Infer one action chunk from an observation."""

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset backend policy state."""

    def get_modality_config(self) -> dict[str, Any]:
        """Return modality/action schema for runtime validation."""
