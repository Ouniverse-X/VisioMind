from __future__ import annotations

from typing import Protocol

from visiomind.action.agents.action.models import ActionInternalStep, ActionStepVerification
from visiomind.action.shared.context import ExecutionContext, Subtask


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
