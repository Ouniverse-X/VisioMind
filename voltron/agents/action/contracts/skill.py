"""Skill contract used by Action local execution."""

from __future__ import annotations

from typing import Protocol

from voltron.shared.context import ExecutionContext, LocalSkillSelection, Subtask
from voltron.shared.results import AgentResult


class VLASkill(Protocol):
    """Minimal contract for an Action skill implementation."""

    skill_id: str
    supported_actions: tuple[str, ...]

    def can_handle(self, subtask: Subtask, context: ExecutionContext) -> bool:
        """Return whether this skill can handle the given subtask."""

    def execute(
        self,
        subtask: Subtask,
        context: ExecutionContext,
        selection: LocalSkillSelection,
    ) -> AgentResult:
        """Execute the subtask and return a standard agent result."""
