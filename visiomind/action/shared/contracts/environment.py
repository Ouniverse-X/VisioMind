from __future__ import annotations

from typing import Any, Protocol

from visiomind.action.shared.context import ExecutionContext, Plan, Subtask, TaskRequest
from visiomind.action.shared.results import AgentResult
from visiomind.action.shared.models import SubtaskStepOutcome


class RuntimeEnvironment(Protocol):
    def reset(self, request: TaskRequest, plan: Plan, context: ExecutionContext) -> dict[str, Any]:
        pass

    def update_plan(self, plan: Plan, context: ExecutionContext) -> None:
        pass

    def build_runtime_inputs(self, subtask: Subtask, context: ExecutionContext) -> dict[str, Any]:
        pass

    def on_agent_result(
        self,
        subtask: Subtask,
        result: AgentResult,
        context: ExecutionContext,
    ) -> SubtaskStepOutcome:
        pass

    def on_subtask_completion_decision(
        self,
        subtask: Subtask,
        decision: dict[str, Any],
        context: ExecutionContext,
    ) -> None:
        pass

    def task_succeeded(self, context: ExecutionContext) -> bool:
        pass

    def summary(self) -> dict[str, Any]:
        pass

    def close(self) -> None:
        pass
