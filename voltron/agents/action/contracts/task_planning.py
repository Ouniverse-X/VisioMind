from __future__ import annotations

from typing import Protocol

from voltron.agents.action.models import ActionExecutionPlan, ActionReplanDecision
from voltron.shared.context import ExecutionContext, Subtask


class ActionTaskPlanningSkill(Protocol):
    def build_plan_prompt(self, subtask: Subtask, context: ExecutionContext) -> str:
        pass

    def parse_plan_response(
        self,
        content: str,
        subtask: Subtask,
        context: ExecutionContext,
    ) -> ActionExecutionPlan:
        pass

    def build_fallback_plan(
        self,
        subtask: Subtask,
        context: ExecutionContext,
        *,
        reason: str,
    ) -> ActionExecutionPlan:
        pass

    def replan(
        self,
        subtask: Subtask,
        context: ExecutionContext,
        *,
        active_step_id: str,
        reason: str,
    ) -> ActionReplanDecision:
        pass


VLATaskPlanningSkill = ActionTaskPlanningSkill
