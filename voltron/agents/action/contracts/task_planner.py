from __future__ import annotations

from typing import Protocol

from voltron.shared.context import ExecutionContext, Subtask


class ActionTaskPlanner(Protocol):
    def generate_plan(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        prompt: str,
    ) -> str:
        pass
