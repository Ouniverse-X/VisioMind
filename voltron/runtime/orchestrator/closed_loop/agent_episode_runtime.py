from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import completion_policy
from voltron.shared.context import ExecutionContext, Subtask
from voltron.shared.contracts import RuntimeEnvironment
from voltron.shared.enums import AgentName, AgentStatus
from voltron.shared.models import RuntimeFeedback
from voltron.shared.results import AgentResult
from voltron.runtime.telemetry.navigation_payloads import summarize_agent_result_for_event

from .navigation_events import (
    emit_nav2_path_snapshot_if_new,
    emit_navigation_candidates_snapshot_if_new,
)


@dataclass
class AgentEpisodeRuntime:
    orchestrator: Any
    environment: RuntimeEnvironment
    attempt: int
    max_control_steps: int
    _latest_runtime_inputs: dict[str, Any] = field(default_factory=dict, init=False)

    def prepare_control_step(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        static_parameters: dict[str, Any],
        control_step: int,
    ) -> dict[str, Any]:
        runtime_inputs = self.environment.build_runtime_inputs(subtask, context)
        self._latest_runtime_inputs = dict(runtime_inputs)
        subtask.parameters = {**static_parameters, **runtime_inputs}
        self.orchestrator._update_working_memory_task_context(
            context=context,
            updates={
                "execution_state": {
                    "task_phase": f"{subtask.subtask_id}:{subtask.agent.value}:{subtask.action}",
                    "parent_task_phase": None,
                    "current_subtask": {
                        "subtask_id": subtask.subtask_id,
                        "execution_id": subtask.runtime_id,
                        "plan_revision": subtask.plan_revision,
                        "agent": subtask.agent.value,
                        "action": subtask.action,
                        "target": dict(subtask.target),
                        "instruction": str(subtask.parameters.get("instruction", "")),
                    },
                    "current_internal_subtask": None,
                    "action_internal_plan": None,
                    "robot_state": _extract_robot_state(
                        runtime_inputs=runtime_inputs,
                        env_feedback=context.runtime_state.get("environment", {}),
                    ),
                    "control_step": control_step,
                }
            },
        )
        return runtime_inputs

    def publish_agent_result(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        result: AgentResult,
        control_step: int,
    ) -> AgentResult:
        result.result.setdefault("agent", subtask.agent.value)
        result.result.setdefault("attempt", self.attempt)
        result.result.setdefault("execution_id", subtask.runtime_id)
        result.result.setdefault("plan_revision", subtask.plan_revision)
        result.result["control_step"] = control_step
        emit_navigation_candidates_snapshot_if_new(
            orchestrator=self.orchestrator,
            subtask=subtask,
            context=context,
            control_step=control_step,
            result=dict(result.result),
        )
        emit_nav2_path_snapshot_if_new(
            orchestrator=self.orchestrator,
            subtask=subtask,
            context=context,
            control_step=control_step,
            result=dict(result.result),
            runtime_artifacts=(
                dict(result.runtime_artifacts)
                if isinstance(result.runtime_artifacts, dict)
                else None
            ),
        )
        self.orchestrator._emit_event(
            event_type="agent_result",
            source=subtask.agent.value,
            message=agent_result_event_message(subtask=subtask, result=result),
            payload={
                "subtask_id": result.subtask_id,
                "execution_id": subtask.runtime_id,
                "plan_revision": subtask.plan_revision,
                "agent": subtask.agent.value,
                "status": result.status.value,
                "error_code": result.error_code,
                "control_step": control_step,
                "result": summarize_agent_result_for_event(dict(result.result)),
            },
            task_id=context.task_request.task_id,
        )
        self.orchestrator._update_working_memory_for_agent_result(
            context=context,
            subtask=subtask,
            result=result,
        )
        record_working_observation = getattr(
            self.orchestrator,
            "_record_working_observation_for_agent_result",
            None,
        )
        if callable(record_working_observation):
            record_working_observation(
                context=context,
                subtask=subtask,
                result=result,
                control_step=control_step,
            )
        latest_object_approach = completion_policy.extract_object_approach_context(
            subtask=subtask,
            result=result,
        )
        if latest_object_approach is not None:
            context.runtime_state["latest_object_approach_attempt"] = latest_object_approach
        return result

    def apply_agent_result(
        self,
        *,
        subtask: Subtask,
        result: AgentResult,
        context: ExecutionContext,
    ):
        environment_outcome = self.environment.on_agent_result(subtask, result, context)
        decision = self.orchestrator._evaluate_completion_step(
            subtask=subtask,
            context=context,
            environment=self.environment,
            result=result,
            environment_outcome=environment_outcome,
            control_step=int(result.result.get("control_step") or 0),
        )
        return decision.to_step_outcome()

    def update_feedback(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        result: AgentResult,
        control_step: int,
        feedback: Any,
    ) -> dict[str, Any]:
        serialized_feedback = _serialize_runtime_feedback(feedback)
        result.result["env_feedback"] = serialized_feedback
        self.orchestrator._update_working_memory_task_context(
            context=context,
            updates={
                "execution_state": {
                    "latest_agent_result": {
                        "subtask_id": result.subtask_id,
                        "agent": subtask.agent.value,
                        "status": result.status.value,
                        "control_step": control_step,
                    },
                    "robot_state": _extract_robot_state(
                        runtime_inputs=self._latest_runtime_inputs,
                        env_feedback=serialized_feedback,
                    ),
                }
            },
        )
        record_working_observation = getattr(
            self.orchestrator,
            "_record_working_observation_for_runtime_feedback",
            None,
        )
        if callable(record_working_observation):
            record_working_observation(
                context=context,
                subtask=subtask,
                result=result,
                feedback=serialized_feedback,
                control_step=control_step,
            )
        notify_environment_step = getattr(self.orchestrator, "_on_environment_step", None)
        if callable(notify_environment_step):
            notify_environment_step(
                context=context,
                environment=self.environment,
                env_step=int(serialized_feedback.get("step_count") or control_step),
                source_subtask=subtask,
                feedback=serialized_feedback,
            )
        return serialized_feedback

    def record_agent_failure(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        result: AgentResult,
        failure_reason: str,
    ) -> None:
        completion_policy.record_failure_from_result(
            memory=self.orchestrator.brain_agent.memory,
            context=context,
            subtask=subtask,
            result=result,
            failure_reason=failure_reason,
            latest_object_approach=completion_policy.extract_object_approach_context(
                subtask=subtask,
                result=result,
            ),
        )

    def record_agent_success(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        result: AgentResult,
    ) -> None:
        completion_policy.record_success_from_result(
            memory=self.orchestrator.brain_agent.memory,
            context=context,
            subtask=subtask,
            result=result,
            latest_object_approach=completion_policy.extract_object_approach_context(
                subtask=subtask,
                result=result,
            ),
        )

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
        reason = failure_reason or "SUBTASK_FAILED"
        self.record_agent_failure(
            subtask=subtask, context=context, result=result, failure_reason=reason
        )
        return AgentResult(
            subtask_id=subtask.subtask_id,
            status=AgentStatus.FAILURE,
            error_code=reason,
            result={
                "message": "environment marked subtask failure",
                "execution_id": subtask.runtime_id,
                "plan_revision": subtask.plan_revision,
                "attempt": self.attempt,
                "control_step": control_step,
                "env_feedback": _serialize_runtime_feedback(feedback),
            },
            latency_ms=result.latency_ms,
        )

    def timeout_result(self, *, subtask: Subtask) -> AgentResult:
        return AgentResult(
            subtask_id=subtask.subtask_id,
            status=AgentStatus.FAILURE,
            error_code="SUBTASK_TIMEOUT",
            result={
                "message": f"subtask exceeded {self.max_control_steps} control steps",
                "execution_id": subtask.runtime_id,
                "plan_revision": subtask.plan_revision,
                "attempt": self.attempt,
                "control_step": self.max_control_steps,
            },
        )


def _extract_robot_state(
    *,
    runtime_inputs: dict[str, Any],
    env_feedback: dict[str, Any],
) -> dict[str, Any]:
    pose = runtime_inputs.get("pose")
    if pose in (None, {}):
        pose = env_feedback.get("pose")
    return {
        "pose": pose,
        "current_room": env_feedback.get("current_room"),
        "current_region": env_feedback.get("current_region"),
    }


def _serialize_runtime_feedback(feedback: Any) -> dict[str, Any]:
    normalized = RuntimeFeedback.from_value(feedback)
    if normalized is not None:
        return normalized.to_dict()
    if isinstance(feedback, dict):
        return dict(feedback)
    return {}


def agent_result_event_message(*, subtask: Subtask, result: AgentResult) -> str:
    if result.status != AgentStatus.SUCCESS:
        return f"{subtask.agent.value} {subtask.subtask_id} returned {result.status.value}"
    action_keys = result.result.get("action_keys")
    policy_info = result.result.get("policy_info")
    goal_reached = False
    if isinstance(policy_info, dict):
        goal_reached = bool(
            policy_info.get("goal_reached") or policy_info.get("controller_mode") == "goal_reached"
        )
    if (
        subtask.agent == AgentName.NAVIGATION
        and isinstance(action_keys, list)
        and action_keys
        and not goal_reached
    ):
        return f"{subtask.agent.value} {subtask.subtask_id} produced action"
    return f"{subtask.agent.value} {subtask.subtask_id} returned {result.status.value}"
