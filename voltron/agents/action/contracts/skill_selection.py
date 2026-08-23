"""Skill-selection contract used by the Action agent."""

from __future__ import annotations

from typing import Protocol

from voltron.shared.context import ExecutionContext, LocalSkillSelection, Subtask


class LocalSkillSelector(Protocol):
    """Local model contract for choosing the primary Action skill."""

    def select_skill(
        self,
        subtask: Subtask,
        context: ExecutionContext,
        available_skill_ids: list[str],
    ) -> LocalSkillSelection:
        """Choose the best primary skill for the current Action subtask."""
