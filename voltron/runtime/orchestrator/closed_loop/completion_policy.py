from __future__ import annotations

from typing import Any

from voltron.shared.context import ExecutionContext, Subtask
from voltron.shared.enums import AgentName
from voltron.shared.results import AgentResult


def record_object_approach_outcome(
    *,
    memory: Any,
    approach: dict[str, Any],
    outcome: str,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    try:
        memory.record_object_approach_outcome(
            scene_id=str(approach.get("scene_id") or ""),
            target=dict(approach.get("target") or {}),
            candidate=dict(approach.get("candidate") or {}),
            outcome=outcome,
            reason=reason,
            metadata=metadata,
        )
    except Exception:
        pass


def flush_pending_object_approach_outcome(
    *,
    memory: Any,
    context: ExecutionContext,
    success: bool,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    pending = context.runtime_state.pop("pending_object_approach", None)
    if not isinstance(pending, dict):
        return
    record_object_approach_outcome(
        memory=memory,
        approach=pending,
        outcome="success" if success else "failure",
        reason=reason,
        metadata=metadata,
    )
    context.runtime_state.pop("latest_object_approach_attempt", None)


def record_success_from_result(
    *,
    memory: Any,
    context: ExecutionContext,
    subtask: Subtask,
    result: AgentResult,
    latest_object_approach: dict[str, Any] | None,
) -> None:
    if subtask.agent == AgentName.NAVIGATION and latest_object_approach is not None:
        context.runtime_state["pending_object_approach"] = latest_object_approach
        return
    if subtask.agent == AgentName.ACTION:
        update_carried_object_state(context=context, subtask=subtask)
        record_action_subtask_outcome(
            memory=memory,
            subtask=subtask,
            result=result,
            success=True,
            failure_reason=None,
        )
        flush_pending_object_approach_outcome(
            memory=memory,
            context=context,
            success=True,
            reason="post_approach_vla_succeeded",
            metadata={"subtask_id": subtask.subtask_id},
        )
        context.runtime_state.pop("latest_object_approach_attempt", None)


def update_carried_object_state(*, context: ExecutionContext, subtask: Subtask) -> None:
    action = _normalized_action(subtask.action)
    if action in {"pick", "pick up", "pickup", "grasp", "grab"}:
        context.runtime_state["carried_object"] = {
            "subtask_id": subtask.subtask_id,
            "action": subtask.action,
            "target": dict(subtask.target),
            "object": _target_object_name(subtask),
        }
        return
    if action in {"place", "put", "put down", "putdown", "drop", "release"}:
        context.runtime_state.pop("carried_object", None)


def _normalized_action(value: Any) -> str:
    return " ".join(str(value or "").replace("_", " ").replace("-", " ").lower().split())


def _target_object_name(subtask: Subtask) -> str | None:
    for key in ("object", "object_name", "name"):
        value = subtask.target.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def record_failure_from_result(
    *,
    memory: Any,
    context: ExecutionContext,
    subtask: Subtask,
    result: AgentResult | None = None,
    failure_reason: str,
    latest_object_approach: dict[str, Any] | None,
) -> None:
    if subtask.agent == AgentName.NAVIGATION:
        approach = latest_object_approach
        if approach is None:
            cached = context.runtime_state.get("latest_object_approach_attempt")
            approach = cached if isinstance(cached, dict) else None
        if approach is not None:
            record_object_approach_outcome(
                memory=memory,
                approach=approach,
                outcome="failure",
                reason=failure_reason,
                metadata={"subtask_id": subtask.subtask_id},
            )
            context.runtime_state.pop("latest_object_approach_attempt", None)
        return
    if subtask.agent == AgentName.ACTION:
        record_action_subtask_outcome(
            memory=memory,
            subtask=subtask,
            result=result,
            success=False,
            failure_reason=failure_reason,
        )
        flush_pending_object_approach_outcome(
            memory=memory,
            context=context,
            success=False,
            reason="post_approach_vla_failed",
            metadata={"subtask_id": subtask.subtask_id, "failure_reason": failure_reason},
        )
        context.runtime_state.pop("latest_object_approach_attempt", None)


def record_action_subtask_outcome(
    *,
    memory: Any,
    subtask: Subtask,
    result: AgentResult | None,
    success: bool,
    failure_reason: str | None,
) -> None:
    record_action = getattr(memory, "record_action", None)
    if not callable(record_action):
        return
    try:
        payload = build_action_subtask_record_payload(
            subtask=subtask,
            result=result,
            success=success,
            failure_reason=failure_reason,
        )
        record_action(payload)
    except Exception:
        pass


def build_action_subtask_record_payload(
    *,
    subtask: Subtask,
    result: AgentResult | None,
    success: bool,
    failure_reason: str | None,
) -> dict[str, Any]:
    result_payload = dict(result.result) if result is not None else {}
    runtime_artifacts = dict(result.runtime_artifacts) if result is not None else {}
    env_feedback = result_payload.get("env_feedback")
    if not isinstance(env_feedback, dict):
        env_feedback = {}

    execution_plan = runtime_artifacts.get("action_execution_plan")
    execution_progress = runtime_artifacts.get("action_execution_progress")
    active_internal_step = runtime_artifacts.get("action_active_internal_step")
    collaborative_step_id = subtask.parameters.get("collaborative_step_id")
    completion_condition_source = subtask.parameters.get("completion_condition_source")
    completion_criteria = [
        dict(item)
        for item in subtask.parameters.get("completion_criteria") or []
        if isinstance(item, dict)
    ]

    parameters = {
        "subtask_id": subtask.subtask_id,
        "agent": subtask.agent.value,
        "action": subtask.action,
        "target": dict(subtask.target),
        "instruction": str(
            subtask.parameters.get("instruction") or subtask.context.get("instruction") or ""
        ),
        "collaborative_step_id": collaborative_step_id,
        "completion_condition_source": completion_condition_source,
        "completion_criteria": completion_criteria,
        "attempt": result_payload.get("attempt"),
        "control_step": result_payload.get("control_step"),
        "env_step": env_feedback.get("step_count"),
        "reward": env_feedback.get("reward"),
        "task_success": env_feedback.get("task_success"),
        "execution_mode": runtime_artifacts.get("action_execution_mode"),
        "execution_plan": _compact_action_execution_plan(execution_plan),
        "execution_progress": dict(execution_progress)
        if isinstance(execution_progress, dict)
        else {},
        "active_internal_step": _compact_internal_step(active_internal_step),
        "internal_steps": _compact_action_internal_steps(
            runtime_artifacts.get("action_internal_steps"),
            execution_plan=execution_plan if isinstance(execution_plan, dict) else {},
            active_internal_step=active_internal_step
            if isinstance(active_internal_step, dict)
            else {},
            execution_progress=execution_progress if isinstance(execution_progress, dict) else {},
            failure_reason=failure_reason,
        ),
        "replan_history": list(runtime_artifacts.get("action_replan_history", []))
        if isinstance(runtime_artifacts.get("action_replan_history"), list)
        else [],
        "step_verification": dict(runtime_artifacts.get("action_step_verification"))
        if isinstance(runtime_artifacts.get("action_step_verification"), dict)
        else None,
    }

    return {
        "action_type": "action.subtask_completed" if success else "action.subtask_failed",
        "target": _action_record_target(subtask),
        "parameters": parameters,
        "collaborative_step_id": collaborative_step_id,
        "completion_condition_source": completion_condition_source,
        "completion_criteria": completion_criteria,
        "control_step": parameters.get("control_step"),
        "env_step": parameters.get("env_step"),
        "task_success": parameters.get("task_success"),
        "pre_state": {},
        "post_state": {},
        "success": success,
        "failure_reason": failure_reason,
        "duration": 0.0,
        "episodic_record": True,
    }


def _action_record_target(subtask: Subtask) -> str:
    for key in ("object", "object_id", "room", "region"):
        value = subtask.target.get(key)
        if value not in (None, ""):
            return str(value)
    return str(subtask.target or subtask.action)


def _compact_action_execution_plan(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    steps = value.get("steps")
    return {
        "parent_subtask_id": value.get("parent_subtask_id"),
        "goal_summary": value.get("goal_summary"),
        "source": value.get("source"),
        "steps": [_compact_internal_step(step) for step in steps]
        if isinstance(steps, list)
        else [],
    }


def _compact_action_internal_steps(
    value: Any,
    *,
    execution_plan: dict[str, Any],
    active_internal_step: dict[str, Any],
    execution_progress: dict[str, Any],
    failure_reason: str | None,
) -> list[dict[str, Any]]:
    steps_by_id: dict[str, dict[str, Any]] = {}
    ordered_ids: list[str] = []
    plan_steps = execution_plan.get("steps")
    if isinstance(plan_steps, list):
        for item in plan_steps:
            step = _compact_internal_step(item)
            step_id = step.get("internal_step_id")
            if not step_id:
                continue
            step_id = str(step_id)
            steps_by_id[step_id] = step
            ordered_ids.append(step_id)

    completed_step_ids = set(execution_progress.get("completed_step_ids") or [])
    pending_step_ids = set(execution_progress.get("pending_step_ids") or [])
    if isinstance(value, list):
        for item in value:
            step = _compact_internal_step(item)
            step_id = step.get("internal_step_id")
            if not step_id:
                continue
            step_id = str(step_id)
            steps_by_id[step_id] = {**steps_by_id.get(step_id, {}), **step}
            if step_id not in ordered_ids:
                ordered_ids.append(step_id)

    active_step = _compact_internal_step(active_internal_step)
    active_step_id = active_step.get("internal_step_id")
    if active_step_id:
        active_step_id = str(active_step_id)
        steps_by_id[active_step_id] = {**steps_by_id.get(active_step_id, {}), **active_step}
        if active_step_id not in ordered_ids:
            ordered_ids.append(active_step_id)

    compact_steps = []
    for step_id in ordered_ids:
        item = steps_by_id.get(step_id, {})
        step = _compact_internal_step(item)
        if not step:
            continue
        if not step.get("status"):
            if step_id in completed_step_ids:
                step["status"] = "success"
            elif failure_reason and step_id in pending_step_ids:
                step["status"] = "failure"
            else:
                step["status"] = "pending"
        if (
            failure_reason
            and step["status"] not in {"success", "completed"}
            and not step.get("failure_reason")
        ):
            step["failure_reason"] = failure_reason
        compact_steps.append(step)
    return compact_steps


def _compact_internal_step(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result = value.get("result") if isinstance(value.get("result"), dict) else {}
    return {
        "internal_step_id": value.get("internal_step_id"),
        "name": value.get("name"),
        "instruction": value.get("instruction"),
        "action": value.get("action"),
        "target": dict(value.get("target") or {}),
        "status": value.get("status"),
        "error_code": value.get("error_code"),
        "failure_reason": value.get("failure_reason") or result.get("failure_reason"),
        "selected_skill_id": value.get("selected_skill_id") or result.get("skill_id"),
        "step_index": value.get("step_index"),
        "total_steps": value.get("total_steps"),
        "executed_control_steps": value.get("executed_control_steps"),
        "verification_checks": value.get("verification_checks"),
        "last_step_verification": value.get("last_step_verification"),
    }


def extract_object_approach_context(
    *,
    subtask: Subtask,
    result: AgentResult,
) -> dict[str, Any] | None:
    if subtask.agent != AgentName.NAVIGATION:
        return None
    candidate = result.runtime_artifacts.get("selected_object_approach")
    if not isinstance(candidate, dict) or not candidate:
        return None
    grounded_goal = (
        result.result.get("grounded_goal") or result.runtime_artifacts.get("grounded_goal") or {}
    )
    if not isinstance(grounded_goal, dict):
        grounded_goal = {}
    scene_id = (
        result.result.get("scene_id")
        or grounded_goal.get("scene_id")
        or subtask.parameters.get("scene_id")
    )
    return {
        "scene_id": scene_id,
        "target": {
            "object": grounded_goal.get("object_name") or subtask.target.get("object"),
            "object_id": grounded_goal.get("object_id") or subtask.target.get("object_id"),
            "room_id": grounded_goal.get("room_id") or subtask.target.get("room_id"),
            "room_name": grounded_goal.get("room_name") or subtask.target.get("room"),
            "floor_id": grounded_goal.get("floor_id") or subtask.target.get("floor_id"),
        },
        "candidate": dict(candidate),
    }
