from __future__ import annotations

from typing import Any

from .agent_episode_runtime import AgentEpisodeRuntime, agent_result_event_message
from . import completion_policy, navigation_events, subtask_dispatch
from visiomind.action.shared.context import ExecutionContext, Subtask
from visiomind.action.shared.contracts import RuntimeEnvironment
from visiomind.action.shared.enums import AgentStatus
from visiomind.action.shared.models import RuntimeFeedback
from visiomind.action.shared.results import AgentResult
from visiomind.action.runtime.telemetry.navigation_payloads import (
    build_navigation_candidates_snapshot as _build_navigation_candidates_snapshot,
    summarize_agent_result_for_event as _summarize_agent_result_for_event,
)


def extract_robot_state(
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


def serialize_runtime_feedback(feedback: Any) -> dict[str, Any]:
    normalized = RuntimeFeedback.from_value(feedback)
    if normalized is not None:
        return normalized.to_dict()
    if isinstance(feedback, dict):
        return dict(feedback)
    return {}


def summarize_agent_result_for_event(result: dict[str, Any]) -> dict[str, Any]:
    return _summarize_agent_result_for_event(result)


def build_navigation_candidates_snapshot(
    *,
    subtask_id: str,
    control_step: int | None,
    result: dict[str, Any],
) -> dict[str, Any] | None:
    return _build_navigation_candidates_snapshot(
        subtask_id=subtask_id,
        control_step=control_step,
        result=result,
    )


def run_subtask_control_loop(
    *,
    orchestrator: object,
    subtask: Subtask,
    context: ExecutionContext,
    environment: RuntimeEnvironment,
    attempt: int,
) -> AgentResult:
    agent = subtask_dispatch.resolve_subtask_agent(orchestrator=orchestrator, subtask=subtask)
    static_parameters = dict(subtask.parameters)
    orchestrator._maybe_reset_vla_policy(subtask=subtask, context=context)
    orchestrator._emit_event(
        event_type="subtask_dispatch",
        source=subtask.agent.value,
        message=f"dispatch {subtask.agent.value} {subtask.subtask_id} {subtask.action}",
        payload={
            "subtask_id": subtask.subtask_id,
            "execution_id": subtask.runtime_id,
            "plan_revision": subtask.plan_revision,
            "agent": subtask.agent.value,
            "action": subtask.action,
            "target": dict(subtask.target),
            "attempt": attempt,
        },
        task_id=context.task_request.task_id,
    )

    episode_runner = getattr(agent, "run_episode", None)
    if callable(episode_runner):
        runtime = AgentEpisodeRuntime(
            orchestrator=orchestrator,
            environment=environment,
            attempt=attempt,
            max_control_steps=orchestrator.max_control_steps_per_subtask,
        )
        return episode_runner(subtask=subtask, context=context, runtime=runtime)

    for control_step in range(1, orchestrator.max_control_steps_per_subtask + 1):
        runtime_inputs = environment.build_runtime_inputs(subtask, context)
        subtask.parameters = {**static_parameters, **runtime_inputs}
        orchestrator._update_working_memory_task_context(
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
                    "robot_state": extract_robot_state(
                        runtime_inputs=runtime_inputs,
                        env_feedback=context.runtime_state.get("environment", {}),
                    ),
                }
            },
        )

        result = agent.execute(subtask, context)
        result.result.setdefault("agent", subtask.agent.value)
        result.result.setdefault("attempt", attempt)
        result.result.setdefault("execution_id", subtask.runtime_id)
        result.result.setdefault("plan_revision", subtask.plan_revision)
        result.result["control_step"] = control_step
        navigation_events.emit_navigation_candidates_snapshot_if_new(
            orchestrator=orchestrator,
            subtask=subtask,
            context=context,
            control_step=control_step,
            result=dict(result.result),
        )
        navigation_events.emit_nav2_path_snapshot_if_new(
            orchestrator=orchestrator,
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
        orchestrator._emit_event(
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
        orchestrator._update_working_memory_for_agent_result(
            context=context,
            subtask=subtask,
            result=result,
        )
        record_working_observation = getattr(
            orchestrator,
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

        if result.status == AgentStatus.FAILURE:
            completion_policy.record_failure_from_result(
                memory=orchestrator.brain_agent.memory,
                context=context,
                subtask=subtask,
                result=result,
                failure_reason=result.error_code or "AGENT_FAILURE",
                latest_object_approach=latest_object_approach,
            )
            return result

        step_outcome = environment.on_agent_result(subtask, result, context)
        if step_outcome.feedback:
            serialized_feedback = serialize_runtime_feedback(step_outcome.feedback)
            result.result["env_feedback"] = serialized_feedback
            orchestrator._update_working_memory_task_context(
                context=context,
                updates={
                    "execution_state": {
                        "latest_agent_result": {
                            "subtask_id": result.subtask_id,
                            "agent": subtask.agent.value,
                            "status": result.status.value,
                            "control_step": control_step,
                        },
                        "robot_state": extract_robot_state(
                            runtime_inputs=runtime_inputs,
                            env_feedback=serialized_feedback,
                        ),
                    }
                },
            )
            record_working_observation = getattr(
                orchestrator,
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
            notify_environment_step = getattr(orchestrator, "_on_environment_step", None)
            if callable(notify_environment_step):
                notify_environment_step(
                    context=context,
                    environment=environment,
                    env_step=int(serialized_feedback.get("step_count") or control_step),
                    source_subtask=subtask,
                    feedback=serialized_feedback,
                )

        completion_decision = orchestrator._evaluate_completion_step(
            subtask=subtask,
            context=context,
            environment=environment,
            result=result,
            environment_outcome=step_outcome,
            control_step=control_step,
        )
        if not completion_decision.done:
            continue

        if completion_decision.success is False:
            completion_policy.record_failure_from_result(
                memory=orchestrator.brain_agent.memory,
                context=context,
                subtask=subtask,
                result=result,
                failure_reason=completion_decision.failure_reason or "SUBTASK_FAILED",
                latest_object_approach=latest_object_approach,
            )
            return AgentResult(
                subtask_id=subtask.subtask_id,
                status=AgentStatus.FAILURE,
                error_code=completion_decision.failure_reason or "SUBTASK_FAILED",
                result={
                    "message": "completion monitor marked subtask failure",
                    "execution_id": subtask.runtime_id,
                    "plan_revision": subtask.plan_revision,
                    "attempt": attempt,
                    "control_step": control_step,
                    "env_feedback": serialize_runtime_feedback(completion_decision.feedback),
                    "completion_verdict": completion_decision.verdict.to_dict(),
                },
                latency_ms=result.latency_ms,
            )

        completion_policy.record_success_from_result(
            memory=orchestrator.brain_agent.memory,
            context=context,
            subtask=subtask,
            result=result,
            latest_object_approach=latest_object_approach,
        )
        return result

    latest_object_approach = context.runtime_state.get("latest_object_approach_attempt")
    if isinstance(latest_object_approach, dict):
        completion_policy.record_failure_from_result(
            memory=orchestrator.brain_agent.memory,
            context=context,
            subtask=subtask,
            result=None,
            failure_reason="SUBTASK_TIMEOUT",
            latest_object_approach=latest_object_approach,
        )
    return AgentResult(
        subtask_id=subtask.subtask_id,
        status=AgentStatus.FAILURE,
        error_code="SUBTASK_TIMEOUT",
        result={
            "message": f"subtask exceeded {orchestrator.max_control_steps_per_subtask} control steps",
            "execution_id": subtask.runtime_id,
            "plan_revision": subtask.plan_revision,
            "attempt": attempt,
            "control_step": orchestrator.max_control_steps_per_subtask,
        },
    )
