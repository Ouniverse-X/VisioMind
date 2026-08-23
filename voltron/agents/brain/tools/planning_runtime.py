"""Planning-runtime helpers for the Brain agent."""

from __future__ import annotations

import math
import re
from typing import Any

from . import navigation_runtime
from voltron.shared.context import ExecutionContext, Plan, Subtask
from voltron.shared.models import RuntimeFeedback


def record_plan(*, context: ExecutionContext, plan: Plan, reason: str) -> None:
    history = context.runtime_state.setdefault("plan_history", [])
    history.append(
        {
            "reason": reason,
            "metadata": dict(plan.metadata),
            "subtask_ids": [subtask.subtask_id for subtask in plan.subtasks],
            "execution_ids": [subtask.runtime_id for subtask in plan.subtasks],
            "subtasks": [serialize_subtask_summary(subtask) for subtask in plan.subtasks],
        }
    )
    planned_ids = context.runtime_state.setdefault("planned_subtask_ids", [])
    planned_ids.extend(subtask.subtask_id for subtask in plan.subtasks)
    planned_execution_ids = context.runtime_state.setdefault("planned_execution_ids", [])
    planned_execution_ids.extend(subtask.runtime_id for subtask in plan.subtasks)
    context.runtime_state["dynamic_execution"] = bool(plan.metadata.get("dynamic_execution", False))


def serialize_subtask_summary(subtask: Subtask) -> dict[str, Any]:
    summary = {
        "subtask_id": subtask.subtask_id,
        "agent": subtask.agent.value,
        "action": subtask.action,
        "target": dict(subtask.target),
        "instruction": str(subtask.parameters.get("instruction", "")),
    }
    if subtask.execution_id:
        summary["execution_id"] = subtask.execution_id
        summary["plan_revision"] = subtask.plan_revision
    if subtask.replaces_execution_id:
        summary["replaces_execution_id"] = subtask.replaces_execution_id
    return summary


def format_task_phase(subtask_summary: dict[str, Any]) -> str | None:
    if not subtask_summary:
        return None
    subtask_id = str(subtask_summary.get("subtask_id") or "").strip()
    agent = str(subtask_summary.get("agent") or "").strip()
    action = str(subtask_summary.get("action") or "").strip()
    if subtask_id and agent and action:
        return f"{subtask_id}:{agent}:{action}"
    return None


def recent_plan_decisions(context: ExecutionContext) -> list[dict[str, Any]]:
    history = list(context.runtime_state.get("plan_history", []))
    return [
        {
            "reason": str(item.get("reason") or ""),
            "metadata": dict(item.get("metadata") or {}),
            "subtasks": list(item.get("subtasks") or []),
        }
        for item in history[-5:]
    ]


def build_runtime_namespace(context: ExecutionContext) -> dict[str, Any]:
    metadata = context.task_request.metadata if isinstance(context.task_request.metadata, dict) else {}
    return {
        "trace_id": context.trace_id,
        "task_id": context.task_request.task_id,
        "task_type": context.task_request.task_type.value,
        "scene_id": metadata.get("scene_id"),
        "task_backend": metadata.get("backend"),
    }


def serialize_result(result: Any, *, environment_state: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = getattr(result, "result", {})
    scene_report = payload.get("scene_report")
    env_feedback = navigation_runtime.runtime_feedback_dict(payload.get("env_feedback"))
    serialized = {
        "subtask_id": getattr(result, "subtask_id", None),
        "execution_id": payload.get("execution_id"),
        "plan_revision": payload.get("plan_revision"),
        "agent": payload.get("agent"),
        "status": getattr(getattr(result, "status", None), "value", None),
        "error_code": getattr(result, "error_code", None),
        "task_complete": bool(payload.get("task_complete", False)),
        "scene_report": dict(scene_report) if isinstance(scene_report, dict) else {},
        "env_feedback": env_feedback,
        "action_keys": list(payload.get("action_keys", [])),
    }
    navigation_failure_context = build_navigation_failure_context(
        result=result,
        environment_state=environment_state,
    )
    if navigation_failure_context:
        serialized["navigation_failure_context"] = navigation_failure_context
    return serialized


def build_navigation_failure_context(
    *,
    result: Any,
    environment_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if getattr(result, "error_code", None) != "NAV_PATH_UNAVAILABLE":
        return {}
    payload = getattr(result, "result", {})
    if not isinstance(payload, dict):
        return {}
    path_plan = payload.get("path_plan")
    if not isinstance(path_plan, dict) or not path_plan:
        return {}

    local_goal = _compact_runtime_goal(path_plan.get("local_goal"))
    transition_anchor = _compact_runtime_goal(path_plan.get("transition_anchor"))
    execution_goal = _compact_runtime_goal(path_plan.get("execution_goal"))
    portal_block_reason = _portal_block_reason(path_plan)
    blocked_transition = _blocked_transition(path_plan=path_plan)
    transition_points = _transition_points(path_plan=path_plan)
    door_candidates = _door_candidates(
        path_plan=path_plan,
        environment_state=environment_state,
        blocked_transition=blocked_transition,
        transition_points=transition_points,
    )

    context: dict[str, Any] = {
        "failure_type": (
            "portal_path_unavailable"
            if blocked_transition or portal_block_reason
            else "path_unavailable"
        ),
        "path_backend": path_plan.get("path_backend"),
        "nav2_error": path_plan.get("nav2_error"),
        "portal_block_reason": portal_block_reason,
    }
    if local_goal:
        context["local_goal"] = local_goal
    if transition_anchor:
        context["transition_anchor"] = transition_anchor
    if execution_goal:
        context["execution_goal"] = execution_goal
    if blocked_transition:
        context["blocked_transition"] = blocked_transition
    if door_candidates:
        context["door_candidates"] = door_candidates
    return {key: value for key, value in context.items() if value not in (None, "", [], {})}


def _compact_runtime_goal(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    keys = (
        "goal_type",
        "waypoint_type",
        "object_id",
        "object_name",
        "room_id",
        "room_name",
        "source_room_name",
        "target_room_name",
        "floor_id",
        "x",
        "y",
        "z",
        "portal_gap",
        "portal_source_point",
        "portal_target_point",
    )
    return {key: value.get(key) for key in keys if value.get(key) not in (None, "", [], {})}


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", {}, []):
            return value
    return None


def _blocked_transition(*, path_plan: dict[str, Any]) -> dict[str, str]:
    explicit = path_plan.get("blocked_transition")
    if isinstance(explicit, dict):
        source_room = _first_non_empty(
            explicit.get("source_room_name"),
            explicit.get("source_room"),
        )
        target_room = _first_non_empty(
            explicit.get("target_room_name"),
            explicit.get("target_room"),
        )
        if source_room and target_room:
            return {
                "source_room_name": str(source_room),
                "target_room_name": str(target_room),
            }
    for key in ("local_goal", "transition_anchor", "nav2_compute_goal"):
        goal = path_plan.get(key)
        if not isinstance(goal, dict):
            continue
        source_room = _first_non_empty(goal.get("source_room_name"), goal.get("source_room"))
        target_room = _first_non_empty(goal.get("target_room_name"), goal.get("target_room"))
        room_name = _first_non_empty(goal.get("room_name"), goal.get("region_name"))
        if target_room is None and room_name != source_room:
            target_room = room_name
        if source_room and target_room:
            return {
                "source_room_name": str(source_room),
                "target_room_name": str(target_room),
            }
    if _portal_block_reason(path_plan):
        start = path_plan.get("start")
        goal = path_plan.get("goal")
        start = start if isinstance(start, dict) else {}
        goal = goal if isinstance(goal, dict) else {}
        source_room = _first_non_empty(
            start.get("current_room"),
            start.get("room_name"),
            start.get("current_region"),
            start.get("region"),
        )
        target_room = _first_non_empty(
            goal.get("room_name"),
            goal.get("target_room_name"),
            goal.get("region_name"),
        )
        if source_room and target_room and str(source_room) != str(target_room):
            return {
                "source_room_name": str(source_room),
                "target_room_name": str(target_room),
            }
    return {}


def _transition_points(*, path_plan: dict[str, Any]) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for goal_key in ("local_goal", "transition_anchor", "nav2_compute_goal"):
        goal = path_plan.get(goal_key)
        if not isinstance(goal, dict):
            continue
        for point_key in ("portal_source_point", "portal_target_point", "midpoint", "position"):
            point = _point_xy(goal.get(point_key))
            if point is not None:
                points.append(point)
        point = _point_xy(goal)
        if point is not None:
            points.append(point)
    return points


def _door_candidates(
    *,
    path_plan: dict[str, Any],
    environment_state: dict[str, Any] | None,
    blocked_transition: dict[str, str],
    transition_points: list[dict[str, float]],
) -> list[dict[str, Any]]:
    room_names = {
        value
        for value in (
            blocked_transition.get("source_room_name"),
            blocked_transition.get("target_room_name"),
        )
        if value
    }
    explicit_candidates = path_plan.get("door_candidates")
    explicit_candidates = (
        explicit_candidates if isinstance(explicit_candidates, list) else []
    )
    objects = _scene_objects(environment_state) + _scene_objects(path_plan)
    ranked: list[tuple[float, dict[str, Any]]] = []
    seen: set[str] = set()
    for item in explicit_candidates:
        if not isinstance(item, dict):
            continue
        candidate = {
            key: item[key]
            for key in (
                "id",
                "name",
                "room",
                "floor_id",
                "position",
                "is_open",
                "in_rooms",
                "source_room_id",
                "source_room_name",
                "target_room_id",
                "target_room_name",
            )
            if item.get(key) not in (None, "", [], {})
        }
        object_id = str(candidate.get("id") or candidate.get("name") or "")
        if not candidate or (object_id and object_id in seen):
            continue
        if object_id:
            seen.add(object_id)
        ranked.append((-1.0, candidate))
    for obj in objects:
        if not isinstance(obj, dict) or not _is_door_like(obj):
            continue
        object_id = str(_first_non_empty(obj.get("id"), obj.get("object_id"), obj.get("name")) or "")
        if object_id and object_id in seen:
            continue
        room_name = _first_non_empty(obj.get("room"), obj.get("room_name"), obj.get("region"))
        if room_names and room_name and str(room_name) not in room_names:
            continue
        position = _point_xy(obj.get("position"))
        distance = _nearest_distance(position, transition_points)
        candidate = {
            "id": obj.get("id") or obj.get("object_id"),
            "name": obj.get("name") or obj.get("object_name") or obj.get("category"),
            "room": room_name,
            "floor_id": obj.get("floor_id"),
            "position": obj.get("position"),
        }
        if distance is not None:
            candidate["distance_to_transition_m"] = round(distance, 3)
        clean = {key: value for key, value in candidate.items() if value not in (None, "", [], {})}
        if clean:
            seen.add(object_id)
            ranked.append((distance if distance is not None else 999.0, clean))
    ranked.sort(key=lambda item: item[0])
    return [candidate for _, candidate in ranked[:5]]


def _portal_block_reason(path_plan: dict[str, Any]) -> str | None:
    reason = str(path_plan.get("reason") or "").strip().lower()
    diagnostics = path_plan.get("object_approach_diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    selection_reason = str(
        diagnostics.get("selection_failure_reason") or ""
    ).strip().lower()
    if "blocked_by_closed_door" in {reason, selection_reason}:
        return "blocked_by_closed_door"
    if str(path_plan.get("path_backend") or "").strip().lower() == "portal_path_unavailable":
        return str(path_plan.get("nav2_error") or "empty_path")
    return None


def _scene_objects(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    objects = value.get("objects")
    if isinstance(objects, list):
        return [item for item in objects if isinstance(item, dict)]
    scene = value.get("scene")
    if isinstance(scene, dict) and isinstance(scene.get("objects"), list):
        return [item for item in scene["objects"] if isinstance(item, dict)]
    global_plan = value.get("global_plan")
    if isinstance(global_plan, dict):
        return _scene_objects(global_plan)
    return []


def _is_door_like(obj: dict[str, Any]) -> bool:
    text = " ".join(
        str(obj.get(key) or "").lower().replace("_", " ")
        for key in ("name", "object_name", "category", "model")
    )
    return "door" in text


def _point_xy(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    x_coord = value.get("x")
    y_coord = value.get("y")
    if not isinstance(x_coord, (int, float)) or not isinstance(y_coord, (int, float)):
        return None
    point = {"x": float(x_coord), "y": float(y_coord)}
    z_coord = value.get("z")
    if isinstance(z_coord, (int, float)):
        point["z"] = float(z_coord)
    return point


def _nearest_distance(
    position: dict[str, float] | None,
    transition_points: list[dict[str, float]],
) -> float | None:
    if position is None or not transition_points:
        return None
    return min(
        math.hypot(position["x"] - point["x"], position["y"] - point["y"])
        for point in transition_points
    )


def last_scene_report(results: list[Any]) -> dict[str, Any]:
    for item in reversed(results):
        payload = getattr(item, "result", {})
        scene_report = payload.get("scene_report")
        if isinstance(scene_report, dict) and scene_report:
            return dict(scene_report)
    return {}


def extract_task_progress(result: Any) -> float | None:
    feedback = RuntimeFeedback.from_value(getattr(result, "result", {}).get("env_feedback"))
    value = feedback.task_progress if feedback is not None else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def next_subtask_index(results: list[Any], latest_result: Any) -> int:
    max_index = 0
    for item in results:
        max_index = max(max_index, subtask_numeric_suffix(getattr(item, "subtask_id", None)))
    max_index = max(max_index, subtask_numeric_suffix(getattr(latest_result, "subtask_id", None)))
    return max_index + 1 if max_index > 0 else 1


def subtask_numeric_suffix(value: Any) -> int:
    text = str(value or "").strip()
    match = re.fullmatch(r"st_(\d+)", text)
    if not match:
        return 0
    try:
        return int(match.group(1))
    except ValueError:
        return 0
