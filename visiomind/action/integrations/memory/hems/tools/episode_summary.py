from __future__ import annotations

from copy import deepcopy
from typing import Any

from visiomind.action.shared.telemetry.payload_sanitizer import strip_image_payloads


def annotate_episode(
    *, episode: Any, task_context: dict[str, Any], working_state: dict[str, Any]
) -> None:
    execution_state = task_context.get("execution_state", {})
    namespace = task_context.get("runtime_namespace", {})

    if namespace:
        initial_state = dict(getattr(episode, "initial_state", {}) or {})
        initial_state["runtime_namespace"] = deepcopy(namespace)
        episode.initial_state = initial_state

    final_state = dict(getattr(episode, "final_state", {}) or {})
    final_state["working_memory_summary"] = strip_image_payloads(
        {
            "task_phase": execution_state.get("task_phase"),
            "parent_task_phase": execution_state.get("parent_task_phase"),
            "current_plan": deepcopy(execution_state.get("current_plan", [])),
            "current_subtask": deepcopy(execution_state.get("current_subtask", {})),
            "current_internal_subtask": deepcopy(execution_state.get("current_internal_subtask")),
            "action_internal_plan": deepcopy(execution_state.get("action_internal_plan")),
            "robot_state": deepcopy(execution_state.get("robot_state", {})),
            "recent_decisions": deepcopy(execution_state.get("recent_decisions", [])),
            "latest_scene_report": deepcopy(execution_state.get("latest_scene_report", {})),
            "latest_navigation_report": deepcopy(
                execution_state.get("latest_navigation_report", {})
            ),
            "latest_environment_feedback": _compact_environment_feedback(execution_state),
            "working_state": deepcopy(working_state),
        }
    )
    episode.final_state = final_state


def _compact_environment_feedback(execution_state: dict[str, Any]) -> dict[str, Any]:
    feedback = execution_state.get("environment_feedback")
    if not isinstance(feedback, dict) or not feedback:
        latest_result = execution_state.get("latest_result")
        if isinstance(latest_result, dict):
            feedback = latest_result.get("env_feedback")
    if not isinstance(feedback, dict) or not feedback:
        return {}

    compact: dict[str, Any] = {}
    for key in (
        "step_count",
        "control_step",
        "task_progress",
        "task_success",
        "reward",
        "truncated",
        "terminated",
        "goal_status",
        "subtask_completed",
        "subtask_succeeded",
        "subtask_completion_reason",
    ):
        if key in feedback:
            compact[key] = deepcopy(feedback[key])

    heartbeat = feedback.get("environment_vlm_heartbeat")
    if isinstance(heartbeat, dict) and heartbeat:
        compact["environment_vlm_heartbeat"] = _compact_vlm_heartbeat(heartbeat)
    return {key: value for key, value in compact.items() if value not in (None, "", {})}


def _compact_vlm_heartbeat(heartbeat: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in (
        "available",
        "enabled",
        "source",
        "subtask_completed",
        "subtask_succeeded",
        "subtask_completion_reason",
        "request_in_flight",
        "last_result",
        "success_confirmation_count",
        "success_confirmation_threshold",
    ):
        if key not in heartbeat:
            continue
        value = deepcopy(heartbeat[key])
        if isinstance(value, str):
            value = value.strip()[:240]
        compact[key] = value
    return {key: value for key, value in compact.items() if value not in (None, "", {})}
