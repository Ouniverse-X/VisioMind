from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from visiomind.action.shared.enums import AgentStatus


@dataclass
class SkillResult:
    skill_id: str
    status: AgentStatus
    payload: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
