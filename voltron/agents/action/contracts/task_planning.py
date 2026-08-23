"""Task-planning contract used inside the Action agent."""

from __future__ import annotations

from typing import Protocol

from voltron.agents.action.models import ActionExecutionPlan, ActionReplanDecision
from voltron.shared.context import ExecutionContext, Subtask


class ActionTaskPlanningSkill(Protocol):
    """Contract for Action planning skills that define planning workflow and parsing."""

    def build_plan_prompt(self, subtask: Subtask, context: ExecutionContext) -> str:
        """Return the user prompt used by the Action agent's planning model."""

    def parse_plan_response(
        self,
        content: str,
        subtask: Subtask,
        context: ExecutionContext,
    ) -> ActionExecutionPlan:
        """Parse model output into an internal Action execution plan."""

    def build_fallback_plan(
        self,
        subtask: Subtask,
        context: ExecutionContext,
        *,
        reason: str,
    ) -> ActionExecutionPlan:
        """Return a safe fallback plan when model planning is unavailable."""

    def replan(
        self,
        subtask: Subtask,
        context: ExecutionContext,
        *,
        active_step_id: str,
        reason: str,
    ) -> ActionReplanDecision:
        """Decide whether to replan the active steps in a subtask."""


VLATaskPlanningSkill = ActionTaskPlanningSkill
