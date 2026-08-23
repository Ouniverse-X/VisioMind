"""Object-approach anchor-selection contract for Navigation."""

from __future__ import annotations

from typing import Any, Protocol

from voltron.shared.context import ExecutionContext, Subtask


class VLNApproachPointSelector(Protocol):
    """Choose one prepared object-approach candidate for execution."""

    def select_candidate(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        goal: dict[str, Any],
        prepared_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Return one prepared object-approach candidate."""
