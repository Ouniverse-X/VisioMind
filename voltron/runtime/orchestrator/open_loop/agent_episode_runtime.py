"""Runtime tool surface for stateful agents in open-loop execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from voltron.shared.context import ExecutionContext, Subtask
from voltron.shared.enums import AgentStatus
from voltron.shared.models import RuntimeFeedback, SubtaskStepOutcome
from voltron.shared.results import AgentResult


@dataclass
class OpenLoopAgentEpisodeRuntime:
    """Static runtime tools for agents that own an open-loop subtask episode."""

    runtime_inputs: dict[str, Any] = field(default_factory=dict)
    attempt: int = 1
    max_control_steps: int = 1

    def prepare_control_step(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        static_parameters: dict[str, Any],
        control_step: int,
    ) -> dict[str, Any]:
        del context, control_step
        subtask.parameters = {**static_parameters, **self.runtime_inputs}
        return dict(self.runtime_inputs)

    def publish_agent_result(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        result: AgentResult,
        control_step: int,
    ) -> AgentResult:
        del context
        result.result.setdefault("agent", subtask.agent.value)
        result.result.setdefault("attempt", self.attempt)
        result.result["control_step"] = control_step
        return result

    def apply_agent_result(
        self,
        *,
        subtask: Subtask,
        result: AgentResult,
        context: ExecutionContext,
    ) -> SubtaskStepOutcome:
        del subtask, context
        if result.status == AgentStatus.FAILURE:
            return SubtaskStepOutcome(
                done=True,
                success=False,
                failure_reason=result.error_code or "AGENT_FAILURE",
                feedback={},
            )
        return SubtaskStepOutcome(
            done=True,
            success=True,
            feedback=RuntimeFeedback(extras={"mode": "open_loop"}),
        )

    def update_feedback(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        result: AgentResult,
        control_step: int,
        feedback: Any,
    ) -> dict[str, Any]:
        del subtask, context, control_step
        serialized = _serialize_feedback(feedback)
        if serialized:
            result.result["env_feedback"] = serialized
        return serialized

    def record_agent_failure(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        result: AgentResult,
        failure_reason: str,
    ) -> None:
        del subtask, context, result, failure_reason

    def record_agent_success(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        result: AgentResult,
    ) -> None:
        del subtask, context, result

    def environment_failure_result(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        result: AgentResult,
        control_step: int,
        feedback: Any,
        failure_reason: str | None,
    ) -> AgentResult:
        del context
        return AgentResult(
            subtask_id=subtask.subtask_id,
            status=AgentStatus.FAILURE,
            error_code=failure_reason or "SUBTASK_FAILED",
            result={
                "message": "open-loop episode marked subtask failure",
                "attempt": self.attempt,
                "control_step": control_step,
                "env_feedback": _serialize_feedback(feedback),
            },
            latency_ms=result.latency_ms,
        )

    def timeout_result(self, *, subtask: Subtask) -> AgentResult:
        return AgentResult(
            subtask_id=subtask.subtask_id,
            status=AgentStatus.FAILURE,
            error_code="SUBTASK_TIMEOUT",
            result={
                "message": f"subtask exceeded {self.max_control_steps} open-loop control steps",
                "attempt": self.attempt,
                "control_step": self.max_control_steps,
            },
        )


def _serialize_feedback(feedback: Any) -> dict[str, Any]:
    normalized = RuntimeFeedback.from_value(feedback)
    if normalized is not None:
        return normalized.to_dict()
    if isinstance(feedback, dict):
        return dict(feedback)
    return {}
