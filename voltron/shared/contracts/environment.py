"""Runtime environment protocol for closed-loop task execution."""

from __future__ import annotations

from typing import Any, Protocol

from voltron.shared.context import ExecutionContext, Plan, Subtask, TaskRequest
from voltron.shared.results import AgentResult
from voltron.shared.models import SubtaskStepOutcome


class RuntimeEnvironment(Protocol):
    """Protocol for simulator/robot adapters used by the closed-loop orchestrator."""

    def reset(self, request: TaskRequest, plan: Plan, context: ExecutionContext) -> dict[str, Any]:
        """Reset environment for a new task episode."""

    def update_plan(self, plan: Plan, context: ExecutionContext) -> None:
        """Append or refresh runtime plan information after dynamic planning."""

    def build_runtime_inputs(self, subtask: Subtask, context: ExecutionContext) -> dict[str, Any]:
        """Collect freshest runtime payload for the current subtask."""

    def on_agent_result(
        self,
        subtask: Subtask,
        result: AgentResult,
        context: ExecutionContext,
    ) -> SubtaskStepOutcome:
        """Apply one agent result to the environment and return step outcome."""

    def on_subtask_completion_decision(
        self,
        subtask: Subtask,
        decision: dict[str, Any],
        context: ExecutionContext,
    ) -> None:
        """Consume a terminal completion verdict when the adapter needs it."""

    def task_succeeded(self, context: ExecutionContext) -> bool:
        """Return whether the full task is complete and successful."""

    def summary(self) -> dict[str, Any]:
        """Return environment-level diagnostics/summary."""

    def close(self) -> None:
        """Release environment resources."""
