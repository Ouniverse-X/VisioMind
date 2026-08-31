from __future__ import annotations

from typing import Protocol

from voltron.agents.action.models import VLADeliberation
from voltron.shared.context import ExecutionContext, Subtask


class VLADeliberator(Protocol):
    def deliberate(
        self,
        subtask: Subtask,
        context: ExecutionContext,
    ) -> VLADeliberation:
        pass
