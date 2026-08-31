from __future__ import annotations

from typing import Any, Callable

from visiomind.action.shared.context import Plan, TaskRequest


def resolve_request_runtime_metadata(
    *,
    metadata: dict[str, Any] | None,
    scene_id: str | None,
    hovsg_graph_root: str | None,
    hovsg_graph_path: str | None,
    hovsg_nav_graph_type: str | None,
    normalize_runtime_str: Callable[[Any], str | None],
) -> dict[str, str | None]:
    metadata = dict(metadata or {})
    return {
        "scene_id": normalize_runtime_str(metadata.get("scene_id")) or scene_id,
        "hovsg_graph_root": normalize_runtime_str(metadata.get("hovsg_graph_root"))
        or hovsg_graph_root,
        "hovsg_graph_path": normalize_runtime_str(metadata.get("hovsg_graph_path"))
        or hovsg_graph_path,
        "hovsg_nav_graph_type": normalize_runtime_str(metadata.get("hovsg_nav_graph_type"))
        or hovsg_nav_graph_type,
    }


def build_run_start_payload(*, request: TaskRequest, env_id: str, plan: Plan) -> dict[str, Any]:
    return {
        "task_id": request.task_id,
        "task_description": request.description,
        "task_type": request.task_type.value,
        "env_id": env_id,
        "subtask_count": len(plan.subtasks),
        "plan": serialize_plan(plan),
    }


def serialize_subtask(subtask: Any) -> dict[str, Any]:
    agent = getattr(subtask, "agent", None)
    payload = {
        "subtask_id": getattr(subtask, "subtask_id", None),
        "agent": getattr(agent, "value", agent),
        "action": getattr(subtask, "action", None),
        "target": dict(getattr(subtask, "target", {}) or {}),
        "parameters": dict(getattr(subtask, "parameters", {}) or {}),
        "context": dict(getattr(subtask, "context", {}) or {}),
    }
    execution_id = getattr(subtask, "execution_id", None)
    if execution_id:
        payload["execution_id"] = execution_id
        payload["plan_revision"] = getattr(subtask, "plan_revision", 0)
    replaces_execution_id = getattr(subtask, "replaces_execution_id", None)
    if replaces_execution_id:
        payload["replaces_execution_id"] = replaces_execution_id
    return payload


def serialize_plan(plan: Plan) -> dict[str, Any]:
    return {
        "metadata": dict(plan.metadata),
        "subtask_count": len(plan.subtasks),
        "subtasks": [serialize_subtask(subtask) for subtask in plan.subtasks],
    }


def build_brain_plan_payload(
    *,
    plan: Plan,
    reason: str,
    request: TaskRequest | None = None,
    failure_reason: str | None = None,
    attempt: int | None = None,
) -> dict[str, Any]:
    payload = {
        "reason": reason,
        **serialize_plan(plan),
    }
    if request is not None:
        payload["task_id"] = request.task_id
        payload["task_description"] = request.description
        payload["task_type"] = request.task_type.value
    if failure_reason:
        payload["failure_reason"] = failure_reason
    if attempt is not None:
        payload["attempt"] = attempt
    return payload


def build_reset_ok_payload(*, info: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_progress": info.get("task_progress"),
        "success": info.get("success"),
        "valid": info.get("valid"),
        "current_room": info.get("current_room"),
        "current_region": info.get("current_region"),
    }


def build_reset_result_payload(
    *,
    env_id: str,
    request: TaskRequest,
    plan: Plan,
    last_info: dict[str, Any],
    pose: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "mode": "behavior_runtime_environment",
        "env_id": env_id,
        "task_id": request.task_id,
        "subtask_count": len(plan.subtasks),
        "current_room": last_info.get("current_room"),
        "current_region": last_info.get("current_region"),
        "room_id": last_info.get("room_id"),
        "floor_id": last_info.get("floor_id"),
        "pose": pose,
    }
