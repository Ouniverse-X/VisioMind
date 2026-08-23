"""Planner interface used by the Brain agent."""

from __future__ import annotations

from typing import Any, Protocol

from voltron.shared.context import Plan, Subtask


class TaskPlanner(Protocol):
    """Planner protocol for initial planning and replanning."""

    def plan(self, task_description: str, context: dict[str, Any]) -> Plan:
        """Generate an initial plan from task and memory context."""

    def plan_next(
        self,
        task_description: str,
        context: dict[str, Any],
        execution_state: dict[str, Any],
    ) -> Plan:
        """Generate the next executable subtask chunk from runtime feedback."""

    def replan(
        self,
        task_description: str,
        context: dict[str, Any],
        failed_subtask: Subtask,
        failure_reason: str,
        execution_state: dict[str, Any],
    ) -> Plan:
        """Generate a replacement plan after a subtask failure."""
