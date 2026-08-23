"""Skill-selection contract for Navigation local routing."""

from __future__ import annotations

from typing import Protocol

from voltron.shared.context import ExecutionContext, LocalSkillSelection, Subtask


class VLNSkillSelector(Protocol):
    """Choose a Navigation-local skill for the current subtask."""

    def select_skill(
        self,
        subtask: Subtask,
        context: ExecutionContext,
        available_skill_ids: list[str],
    ) -> LocalSkillSelection:
        """Choose one Navigation skill for the current subtask."""
