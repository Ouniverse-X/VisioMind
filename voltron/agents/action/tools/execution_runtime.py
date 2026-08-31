from __future__ import annotations

from dataclasses import asdict
from typing import Any

from voltron.agents.action.models import ActionExecutionPlan
from voltron.shared.action_semantics import (
    action_instruction,
    is_state_change_action,
    normalize_action_name,
)
from voltron.shared.context import ExecutionContext, Subtask
from voltron.shared.results import AgentResult


def execution_sessions(context: ExecutionContext) -> dict[str, dict[str, Any]]:
    sessions = context.runtime_state.get("action_execution_sessions")
    if isinstance(sessions, dict):
        return sessions
    sessions = {}
    context.runtime_state["action_execution_sessions"] = sessions
    return sessions


def create_execution_session(execution_plan: ActionExecutionPlan) -> dict[str, Any]:
    active_step_id = execution_plan.steps[0].internal_step_id if execution_plan.steps else None
    return {
        "execution_plan": execution_plan,
        "next_step_index": 0,
        "completed_steps": [],
        "replan_history": [],
        "active_step_id": active_step_id,
        "active_step_control_steps": 0,
        "active_step_verification_checks": 0,
        "active_step_positive_streak": 0,
        "active_step_negative_streak": 0,
        "active_step_last_success_record": None,
        "last_step_verification": None,
    }


def serialize_internal_step(
    step_payload: Any, *, selected_skill_id: str | None = None
) -> dict[str, Any]:
    payload = asdict(step_payload)
    payload["parent_subtask_id"] = str(step_payload.internal_step_id).split(".")[0]
    if selected_skill_id:
        payload["selected_skill_id"] = selected_skill_id
    return payload


def decorate_execution_result(
    *,
    parent_subtask: Subtask,
    base_result: AgentResult,
    execution_mode: str,
    execution_plan: ActionExecutionPlan,
    internal_steps: list[dict[str, Any]],
    active_internal_step: dict[str, Any] | None,
    replan_history: list[dict[str, Any]],
    execution_progress: dict[str, Any],
    step_verification: dict[str, Any] | None = None,
    latency_ms: int | None = None,
) -> AgentResult:
    runtime_artifacts = dict(base_result.runtime_artifacts)
    runtime_artifacts["action_execution_mode"] = execution_mode
    runtime_artifacts["action_execution_plan"] = asdict(execution_plan)
    runtime_artifacts["action_internal_steps"] = internal_steps
    runtime_artifacts["action_active_internal_step"] = active_internal_step
    runtime_artifacts["action_execution_progress"] = execution_progress
    runtime_artifacts["action_replan_history"] = list(replan_history)
    runtime_artifacts["action_step_verification"] = step_verification
    return AgentResult(
        subtask_id=parent_subtask.subtask_id,
        status=base_result.status,
        result=dict(base_result.result),
        error_code=base_result.error_code,
        state_changes=list(base_result.state_changes),
        latency_ms=base_result.latency_ms if latency_ms is None else latency_ms,
        runtime_artifacts=runtime_artifacts,
    )


def _parent_policy_instruction(parent_subtask: Subtask) -> str:
    explicit = str(
        parent_subtask.parameters.get("instruction")
        or parent_subtask.context.get("instruction")
        or ""
    ).strip()
    if explicit:
        return explicit
    return action_instruction(action=parent_subtask.action, target=dict(parent_subtask.target))


def build_internal_subtask(*, parent_subtask: Subtask, step_payload: Any) -> Subtask:
    parameters = dict(parent_subtask.parameters)
    parameters["instruction"] = step_payload.instruction
    parameters["parent_action"] = parent_subtask.action
    parameters["parent_instruction"] = _parent_policy_instruction(parent_subtask)
    parent_action = normalize_action_name(parent_subtask.action)
    if is_state_change_action(parent_action):
        policy_options = dict(parameters.get("policy_options") or {})
        policy_options.setdefault("action", parent_action)
        policy_options.setdefault("action_type", parent_action)
        policy_options.setdefault("raw_action", parent_subtask.action)
        policy_options.setdefault("instruction", parameters["parent_instruction"])
        parameters["policy_options"] = policy_options
    if step_payload.success_cues:
        parameters["success_cues"] = list(step_payload.success_cues)
    observation = parameters.get("observation")
    if isinstance(observation, dict):
        observation_payload = dict(observation)
        vla_prompt = parameters.get("vla_prompt")
        observation_payload["annotation.human.coarse_action"] = (
            vla_prompt.strip()
            if isinstance(vla_prompt, str) and vla_prompt.strip()
            else step_payload.instruction
        )
        parameters["observation"] = observation_payload
    selector_hints = dict(parameters.get("selector_hints", {}))
    if step_payload.preferred_skill_id:
        selector_hints["preferred_skill_id"] = step_payload.preferred_skill_id
    if selector_hints:
        parameters["selector_hints"] = selector_hints

    context = dict(parent_subtask.context)
    context["action_parent_subtask_id"] = parent_subtask.subtask_id
    context["action_internal_step"] = asdict(step_payload)
    context["instruction"] = step_payload.instruction
    return Subtask(
        subtask_id=step_payload.internal_step_id,
        agent=parent_subtask.agent,
        action=step_payload.action,
        target=dict(step_payload.target),
        parameters=parameters,
        context=context,
    )


def replace_pending_steps(
    *,
    execution_plan: ActionExecutionPlan,
    start_index: int,
    replacement_steps: list[Any],
) -> ActionExecutionPlan:
    return ActionExecutionPlan(
        parent_subtask_id=execution_plan.parent_subtask_id,
        goal_summary=execution_plan.goal_summary,
        steps=[*execution_plan.steps[:start_index], *replacement_steps],
        source=execution_plan.source,
        metadata=dict(execution_plan.metadata),
    )


def reset_active_step_tracking(session: dict[str, Any], active_step_id: str | None) -> None:
    session["active_step_id"] = active_step_id
    session["active_step_control_steps"] = 0
    session["active_step_verification_checks"] = 0
    session["active_step_positive_streak"] = 0
    session["active_step_negative_streak"] = 0
    session["active_step_last_success_record"] = None
    session["last_step_verification"] = None
