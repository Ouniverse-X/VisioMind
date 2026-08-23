"""Shared skill-execution contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from voltron.shared.context import ExecutionContext, LocalSkillSelection, Subtask
from voltron.shared.results import SkillResult


@dataclass
class SkillRequest:
    """Canonical request envelope for skill-level execution."""

    subtask: Subtask
    context: ExecutionContext
    selection: LocalSkillSelection | None = None
    runtime_inputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class SkillExecutor(Protocol):
    """Protocol for shared skill-level execution surfaces."""

    def execute(self, request: SkillRequest) -> SkillResult:
        """Execute one skill request and return a normalized skill result."""
