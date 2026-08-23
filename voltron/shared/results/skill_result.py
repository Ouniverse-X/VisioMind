"""Shared skill-level result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from voltron.shared.enums import AgentStatus


@dataclass
class SkillResult:
    """Normalized result envelope produced by shared skill surfaces."""

    skill_id: str
    status: AgentStatus
    payload: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
