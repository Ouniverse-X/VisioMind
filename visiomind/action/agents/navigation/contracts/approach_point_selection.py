from __future__ import annotations

from typing import Any, Protocol

from visiomind.action.shared.context import ExecutionContext, Subtask


class VLNApproachPointSelector(Protocol):
    def select_candidate(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        goal: dict[str, Any],
        prepared_payload: dict[str, Any],
    ) -> dict[str, Any]:
        pass
