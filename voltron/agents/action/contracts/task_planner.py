"""Model-calling contract used by the Action agent for internal plan generation."""

from __future__ import annotations

from typing import Protocol

from voltron.shared.context import ExecutionContext, Subtask


class ActionTaskPlanner(Protocol):
    """Contract for Action planning models invoked by the Action agent itself."""

    def generate_plan(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        prompt: str,
    ) -> str:
        """Return raw model output for internal Action task decomposition."""
