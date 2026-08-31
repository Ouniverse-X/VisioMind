from __future__ import annotations

from typing import Protocol

from visiomind.action.agents.action.models import VLADeliberation
from visiomind.action.shared.context import ExecutionContext, Subtask


class VLADeliberator(Protocol):
    def deliberate(
        self,
        subtask: Subtask,
        context: ExecutionContext,
    ) -> VLADeliberation:
        pass
