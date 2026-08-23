"""State write-back helpers for the BEHAVIOR runtime adapter facade."""

from __future__ import annotations

from typing import Any


def apply_reset_result(adapter: Any, reset: dict[str, Any]) -> dict[str, Any]:
    resolved_metadata = reset["resolved_metadata"]
    adapter._scene_id = resolved_metadata["scene_id"]
    adapter._hovsg_graph_root = resolved_metadata["hovsg_graph_root"]
    adapter._hovsg_graph_path = resolved_metadata["hovsg_graph_path"]
    adapter._hovsg_nav_graph_type = resolved_metadata["hovsg_nav_graph_type"]
    adapter._hovsg_localizer = None
    adapter._runtime_subtasks = reset["runtime_subtasks"]
    adapter._runtime_subtasks_by_id = reset["runtime_subtasks_by_id"]

    reset_state = reset["reset_state"]
    adapter._last_obs = reset_state["last_obs"]
    adapter._last_info = reset_state["last_info"]
    adapter._last_reward = reset_state["last_reward"]
    adapter._terminated = reset_state["terminated"]
    adapter._truncated = reset_state["truncated"]
    adapter._task_success = reset_state["task_success"]
    adapter._step_count = reset_state["step_count"]
    adapter._closed = reset_state["closed"]
    adapter._navigation_runtime_state = reset_state["navigation_runtime_state"]
    adapter._active_subtask_name = reset_state["active_subtask_name"]
    adapter._active_subtask_instruction = reset_state["active_subtask_instruction"]
    adapter._active_action_internal_step = reset_state["active_action_internal_step"]
    adapter._logged_subtask_attempts = reset_state["logged_subtask_attempts"]
    adapter._logged_action_internal_attempts = reset_state["logged_action_internal_attempts"]
    adapter._logged_action_internal_replans = reset_state.get("logged_action_internal_replans", set())
    return reset["result_payload"]


def apply_plan_update(adapter: Any, updated: dict[str, Any]) -> None:
    adapter._runtime_subtasks = updated["runtime_subtasks"]
    adapter._runtime_subtasks_by_id = updated["runtime_subtasks_by_id"]
    adapter.env_kwargs = updated["env_kwargs"]


def apply_configured_runtime_subtasks(adapter: Any, configured: dict[str, Any]) -> dict[str, Any]:
    adapter._runtime_subtasks = configured["runtime_subtasks"]
    adapter._runtime_subtasks_by_id = configured["runtime_subtasks_by_id"]
    return configured


def apply_synced_runtime_subtask(adapter: Any, synced: dict[str, Any]) -> dict[str, Any] | None:
    adapter._active_subtask_name = synced["active_subtask_name"]
    adapter._active_subtask_instruction = synced["active_subtask_instruction"]
    adapter._last_info = synced["last_info"]
    return synced["runtime_subtask"]


def apply_prepared_step_state(adapter: Any, prepared: dict[str, Any]) -> dict[str, int | None]:
    adapter._task_type = prepared["task_type"]
    adapter._navigation_runtime_state = prepared["navigation_runtime_state"]
    adapter._logged_subtask_attempts = prepared["logged_subtask_attempts"]
    adapter._active_action_internal_step = prepared["active_action_internal_step"]
    adapter._logged_action_internal_attempts = prepared["logged_action_internal_attempts"]
    adapter._logged_action_internal_replans = prepared.get("logged_action_internal_replans", set())
    return {
        "control_step": prepared["control_step"],
        "attempt": prepared["attempt"],
    }


def apply_step_state(adapter: Any, step_state: dict[str, Any]) -> dict[str, Any]:
    adapter._step_count = step_state["step_count"]
    adapter._last_obs = step_state["last_obs"]
    adapter._last_info = step_state["last_info"]
    adapter._last_reward = step_state["last_reward"]
    adapter._terminated = step_state["terminated"]
    adapter._truncated = step_state["truncated"]
    adapter._task_success = bool(step_state["task_success"])
    return {
        "success_flag": bool(step_state["success_flag"]),
        "subtask_completed": bool(step_state["subtask_completed"]),
        "subtask_succeeded": bool(step_state["subtask_succeeded"]),
        "subtask_completion_reason": step_state["subtask_completion_reason"],
    }


def apply_close_result(adapter: Any, result: dict[str, Any]) -> None:
    adapter._env = result["env"]
    adapter._closed = bool(result["closed"])
