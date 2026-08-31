from __future__ import annotations

from typing import Protocol

from visiomind.action.shared.context import ExecutionContext, LocalSkillSelection, Subtask


class LocalSkillSelector(Protocol):
    def select_skill(
        self,
        subtask: Subtask,
        context: ExecutionContext,
        available_skill_ids: list[str],
    ) -> LocalSkillSelection:
        pass
