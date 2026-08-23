"""Verification contract for Action internal step completion."""

from __future__ import annotations

from typing import Protocol

from voltron.agents.action.models import ActionInternalStep, ActionStepVerification
from voltron.shared.context import ExecutionContext, Subtask


class ActionStepVerifier(Protocol):
    """Contract for Action-owned internal-step verifiers."""

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
        """Return whether the active internal step has satisfied its completion conditions."""
