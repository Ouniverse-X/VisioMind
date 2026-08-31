from __future__ import annotations

from typing import Any

from visiomind.action.shared.enums import AgentName, TaskType

from visiomind.action.integrations.simulator.behavior.observation import robot_state as behavior_robot_state


def evaluate_navigation_goal_completion(
    *,
    localizer: Any | None,
    pose: dict[str, Any] | None,
    orientation: dict[str, Any] | None,
    scene_id: str | None,
    last_info: dict[str, Any],
    nav_state: dict[str, Any],
    target: dict[str, Any],
    task_type: TaskType,
    object_goal_distance_tolerance_m: float,
    object_goal_heading_tolerance_rad: float,
) -> dict[str, Any]:
    updated_last_info = dict(last_info)
    if localizer is None or pose is None or scene_id is None:
        return {"goal_reached": False, "task_success": False, "last_info": updated_last_info}

    try:
        localized = localizer.update({"scene_id": scene_id}, pose=pose)
    except Exception:
        return {"goal_reached": False, "task_success": False, "last_info": updated_last_info}

    if not isinstance(localized, dict) or not localized:
        return {"goal_reached": False, "task_success": False, "last_info": updated_last_info}

    updated_last_info.update(localized)
    updated_last_info["scene_id"] = scene_id

    goal = behavior_robot_state.navigation_completion_goal(nav_state)
    goal_type = str(goal.get("goal_type") or "").strip().lower()
    object_goal_reached = False
    object_view_goal_reached = False

    if goal_type == "object":
        object_status = _evaluate_object_goal_completion(
            pose=pose,
            orientation=orientation,
            goal=goal,
            localized=localized,
            nav_state=nav_state,
            distance_tolerance_m=object_goal_distance_tolerance_m,
            heading_tolerance_rad=object_goal_heading_tolerance_rad,
        )
        updated_last_info.update(object_status["last_info_updates"])
        object_goal_reached = bool(object_status["reached"])
        if not object_goal_reached:
            view_status = _evaluate_object_view_goal_completion(
                pose=pose,
                goal=goal,
                localized=localized,
                nav_state=nav_state,
                last_info=updated_last_info,
                distance_tolerance_m=object_goal_distance_tolerance_m,
            )
            updated_last_info.update(view_status["last_info_updates"])
            object_view_goal_reached = bool(view_status["reached"])
            object_goal_reached = object_view_goal_reached

    controller_mode = str(nav_state.get("controller_mode") or "").strip().lower()
    path_backend = str(nav_state.get("path_backend") or "").strip().lower()
    policy_goal_reached = (
        bool(nav_state.get("goal_reached"))
        or controller_mode == "goal_reached"
        or path_backend == "global_goal_reached"
    )
    goal_reached = bool(
        behavior_robot_state.navigation_goal_match_status(
            goal=goal,
            target=target,
            localized=localized,
            object_goal_reached=object_goal_reached,
            policy_goal_reached=policy_goal_reached,
        ).get("reached")
    )
    if not goal_reached:
        return {"goal_reached": False, "task_success": False, "last_info": updated_last_info}

    updated_last_info.update(
        {
            "subtask_completed": True,
            "subtask_succeeded": True,
            "subtask_completion_reason": "view_goal_reached"
            if object_view_goal_reached
            else "goal_reached",
        }
    )
    task_success = task_type == TaskType.NAVIGATION
    if task_success:
        updated_last_info["success"] = True
        updated_last_info["task_progress"] = 1.0
    return {"goal_reached": True, "task_success": task_success, "last_info": updated_last_info}


def apply_navigation_success_override(
    *,
    agent: Any,
    last_info: dict[str, Any],
    task_success: bool,
    nav_state: dict[str, Any],
    target: dict[str, Any],
    task_type: TaskType,
    localizer: Any | None,
    pose: dict[str, Any] | None,
    orientation: dict[str, Any] | None,
    scene_id: str | None,
    object_goal_distance_tolerance_m: float,
    object_goal_heading_tolerance_rad: float,
) -> dict[str, Any]:
    updated_last_info = dict(last_info)
    updated_task_success = bool(task_success)
    if agent != AgentName.NAVIGATION:
        return {
            "last_info": updated_last_info,
            "task_success": updated_task_success,
        }

    outcome = evaluate_navigation_goal_completion(
        localizer=localizer,
        pose=pose,
        orientation=orientation,
        scene_id=scene_id,
        last_info=updated_last_info,
        nav_state=nav_state,
        target=target,
        task_type=task_type,
        object_goal_distance_tolerance_m=object_goal_distance_tolerance_m,
        object_goal_heading_tolerance_rad=object_goal_heading_tolerance_rad,
    )
    return {
        "last_info": outcome["last_info"],
        "task_success": updated_task_success or bool(outcome["task_success"]),
    }


def _evaluate_object_goal_completion(
    *,
    pose: dict[str, Any],
    orientation: dict[str, Any] | None,
    goal: dict[str, Any],
    localized: dict[str, Any],
    nav_state: dict[str, Any],
    distance_tolerance_m: float,
    heading_tolerance_rad: float,
) -> dict[str, Any]:
    last_info_updates: dict[str, Any] = {}
    geometry_status = behavior_robot_state.object_goal_geometry_status(
        pose=pose,
        orientation=orientation,
        goal=goal,
        localized=localized,
        distance_tolerance_m=distance_tolerance_m,
        heading_tolerance_rad=heading_tolerance_rad,
    )
    if geometry_status is not None:
        last_info_updates["object_goal_distance_m"] = geometry_status.get("planar_distance")
        last_info_updates["object_goal_heading_error_rad"] = geometry_status.get("heading_error")

    status = behavior_robot_state.object_navigation_goal_status(
        geometry_status=geometry_status,
        nav_state=nav_state,
    )
    controller_mode = status.get("controller_mode")
    if controller_mode:
        last_info_updates["controller_mode"] = controller_mode
    return {"reached": bool(status.get("reached")), "last_info_updates": last_info_updates}


def _evaluate_object_view_goal_completion(
    *,
    pose: dict[str, Any],
    goal: dict[str, Any],
    localized: dict[str, Any],
    nav_state: dict[str, Any],
    last_info: dict[str, Any],
    distance_tolerance_m: float,
) -> dict[str, Any]:
    last_info_updates: dict[str, Any] = {}
    nav_goal = nav_state.get("nav_goal")
    view_goal = nav_goal if isinstance(nav_goal, dict) and nav_goal else goal
    if not _is_view_goal(view_goal):
        return {"reached": False, "last_info_updates": last_info_updates}

    heartbeat = last_info.get("environment_vlm_heartbeat")
    if not isinstance(heartbeat, dict) or not bool(heartbeat.get("subtask_succeeded")):
        return {"reached": False, "last_info_updates": last_info_updates}

    target_position = behavior_robot_state.goal_position(view_goal)
    if target_position is None:
        target_position = behavior_robot_state.goal_position(goal)
    if target_position is None:
        return {"reached": False, "last_info_updates": last_info_updates}

    axis_x, axis_y = behavior_robot_state.planar_axes(localized.get("vertical_axis"))
    current_x = behavior_robot_state.to_float(pose.get(axis_x))
    current_y = behavior_robot_state.to_float(pose.get(axis_y))
    goal_x = behavior_robot_state.to_float(target_position.get(axis_x))
    goal_y = behavior_robot_state.to_float(target_position.get(axis_y))
    if None in {current_x, current_y, goal_x, goal_y}:
        return {"reached": False, "last_info_updates": last_info_updates}

    distance = (
        (float(goal_x) - float(current_x)) ** 2 + (float(goal_y) - float(current_y)) ** 2
    ) ** 0.5
    tolerance = _object_view_goal_distance_tolerance(
        nav_state=nav_state,
        fallback_tolerance_m=distance_tolerance_m,
    )
    last_info_updates["object_view_goal_distance_m"] = distance
    last_info_updates["object_view_goal_distance_tolerance_m"] = tolerance
    return {"reached": distance <= tolerance, "last_info_updates": last_info_updates}


def _is_view_goal(goal: dict[str, Any]) -> bool:
    stop_conditions = (
        {str(item).strip().lower() for item in goal.get("stop_condition", []) if str(item).strip()}
        if isinstance(goal.get("stop_condition"), list)
        else set()
    )
    if stop_conditions & {"interaction_ready", "object_reachable", "same_side"}:
        return False
    return bool(stop_conditions & {"object_visible", "view_ready"})


def _object_view_goal_distance_tolerance(
    *,
    nav_state: dict[str, Any],
    fallback_tolerance_m: float,
) -> float:
    tolerance = max(0.05, float(abs(fallback_tolerance_m)))
    selected = nav_state.get("selected_object_approach")
    if isinstance(selected, dict):
        approach_distance = behavior_robot_state.to_float(selected.get("approach_distance_m"))
        if approach_distance is not None:
            tolerance = max(
                tolerance, float(approach_distance) + max(0.05, float(abs(fallback_tolerance_m)))
            )
    return tolerance
