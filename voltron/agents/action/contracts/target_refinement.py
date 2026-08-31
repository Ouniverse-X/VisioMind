from __future__ import annotations

from typing import Protocol

from voltron.agents.action.models import VLATargetRefinement
from voltron.shared.context import ExecutionContext, Subtask


class VLATargetRefiner(Protocol):
    def refine_target(
        self,
        subtask: Subtask,
        context: ExecutionContext,
    ) -> VLATargetRefinement:
        pass
