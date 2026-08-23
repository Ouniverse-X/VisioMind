"""Skill contract for Navigation local preparation steps."""

from __future__ import annotations

from typing import Any, Protocol

from voltron.shared.context import ExecutionContext, Subtask
from voltron.shared.contracts import NavigatorBackend


class VLNSkill(Protocol):
    """Contract for Navigation-local preparation skills."""

    skill_id: str

    def can_handle(self, subtask: Subtask, context: ExecutionContext) -> bool:
        """Return whether this skill can handle the given subtask."""

    def prepare(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        navigator: NavigatorBackend,
        start: dict[str, Any],
        goal: dict[str, Any],
        navigation_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Prepare structured data consumed by the Navigation agent before execution."""
