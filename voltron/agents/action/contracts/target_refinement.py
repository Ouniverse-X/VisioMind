"""Target-refinement contract used inside the Action agent."""

from __future__ import annotations

from typing import Protocol

from voltron.agents.action.models import VLATargetRefinement
from voltron.shared.context import ExecutionContext, Subtask


class VLATargetRefiner(Protocol):
    """Tool contract for producing a more precise local execution target."""

    def refine_target(
        self,
        subtask: Subtask,
        context: ExecutionContext,
    ) -> VLATargetRefinement:
        """Return structured target refinements derived from task context and observations."""
