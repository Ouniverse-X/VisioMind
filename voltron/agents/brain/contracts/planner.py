from __future__ import annotations

from typing import Any, Protocol

from voltron.shared.context import Plan, Subtask


class TaskPlanner(Protocol):
    def plan(self, task_description: str, context: dict[str, Any]) -> Plan:
        pass

    def plan_next(
        self,
        task_description: str,
        context: dict[str, Any],
        execution_state: dict[str, Any],
    ) -> Plan:
        pass

    def replan(
        self,
        task_description: str,
        context: dict[str, Any],
        failed_subtask: Subtask,
        failure_reason: str,
        execution_state: dict[str, Any],
    ) -> Plan:
        pass
