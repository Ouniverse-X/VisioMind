from __future__ import annotations

from typing import Any

from voltron.integrations.simulator.behavior.artifacts import (
    process_logger as behavior_process_logger,
)
from voltron.integrations.simulator.behavior.tools import runtime_inputs as behavior_runtime_inputs
from voltron.integrations.simulator.behavior.tools import runtime_vla as behavior_runtime_vla
from voltron.integrations.simulator.behavior.tools import subtasks as behavior_subtasks


def prepare_agent_result_runtime_state(
    *,
    context: Any,
    subtask: Any,
    result: Any,
    navigation_runtime_state: dict[str, dict[str, Any]],
    logged_subtask_attempts: set[tuple[str, int]],
    logged_action_internal_attempts: set[tuple[str, int]],
    logged_action_internal_replans: set[tuple[str, int]],
    call_env_method: Any,
    instruction_for_subtask: Any,
    subtask_max_steps: Any,
    record_event: Any,
    emit_progress: Any,
) -> dict[str, Any]:
    task_request = getattr(context, "task_request", None)
    task_type = getattr(task_request, "task_type", None)
    navigation_runtime_state = behavior_runtime_inputs.capture_navigation_runtime_state(
        subtask=subtask,
        result=result,
        navigation_runtime_state=navigation_runtime_state,
        call_env_method=call_env_method,
    )

    result_payload = result.result if isinstance(getattr(result, "result", None), dict) else {}
    control_step = _coerce_positive_int(result_payload.get("control_step"))
    attempt = _coerce_positive_int(result_payload.get("attempt")) or 1

    instruction = instruction_for_subtask(subtask)
    max_steps = subtask_max_steps(subtask)
    logged_subtask_attempts = behavior_process_logger.log_subtask_attempt_start(
        subtask=subtask,
        attempt=attempt,
        instruction=instruction,
        max_steps=max_steps,
        logged_subtask_attempts=logged_subtask_attempts,
        record_event=record_event,
        emit_progress=emit_progress,
    )

    applied = behavior_runtime_vla.apply_vla_internal_runtime_state(
        subtask=subtask,
        result=result,
        attempt=attempt,
        control_step=control_step,
        logged_action_internal_attempts=logged_action_internal_attempts,
        logged_action_internal_replans=logged_action_internal_replans,
        display_name_builder=behavior_subtasks.action_internal_display_name,
        build_plan_created_payload=behavior_process_logger.build_vla_internal_plan_created_payload,
        build_replan_payload=behavior_process_logger.build_vla_internal_replan_payload,
        build_step_start_payload=behavior_process_logger.build_vla_internal_step_start_payload,
        record_event=record_event,
    )

    return {
        "task_type": task_type,
        "navigation_runtime_state": navigation_runtime_state,
        "control_step": control_step,
        "attempt": attempt,
        "logged_subtask_attempts": logged_subtask_attempts,
        "active_action_internal_step": applied["active_internal_step"],
        "logged_action_internal_attempts": applied["logged_action_internal_attempts"],
        "logged_action_internal_replans": applied["logged_action_internal_replans"],
    }


def _coerce_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
