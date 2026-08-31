from __future__ import annotations

from typing import Protocol

from voltron.agents.action.models import ActionInternalStep, ActionStepVerification
from voltron.shared.context import ExecutionContext, Subtask


class ActionStepVerifier(Protocol):
    def verify_step(
        self,
        *,
        parent_subtask: Subtask,
        internal_subtask: Subtask,
        step_payload: ActionInternalStep,
        context: ExecutionContext,
        executed_control_steps: int,
        verification_index: int,
    ) -> ActionStepVerification:
        pass
