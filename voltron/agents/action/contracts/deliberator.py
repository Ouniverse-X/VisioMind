"""Internal deliberator contract used by the Action agent."""

from __future__ import annotations

from typing import Protocol

from voltron.agents.action.models import VLADeliberation
from voltron.shared.context import ExecutionContext, Subtask


class VLADeliberator(Protocol):
    """Agent-internal policy for deciding whether to use VLA tools before execution."""

    def deliberate(
        self,
        subtask: Subtask,
        context: ExecutionContext,
    ) -> VLADeliberation:
        """Return the next internal decision for the Action agent."""
