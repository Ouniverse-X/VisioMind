from __future__ import annotations

from typing import Protocol

from visiomind.action.shared.context import ExecutionContext, LocalSkillSelection, Subtask
from visiomind.action.shared.results import AgentResult


class VLASkill(Protocol):
    skill_id: str
    supported_actions: tuple[str, ...]

    def can_handle(self, subtask: Subtask, context: ExecutionContext) -> bool:
        pass

    def execute(
        self,
        subtask: Subtask,
        context: ExecutionContext,
        selection: LocalSkillSelection,
    ) -> AgentResult:
        pass
