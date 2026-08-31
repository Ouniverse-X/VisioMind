from __future__ import annotations

import math
from typing import Any

from voltron.integrations.simulator.behavior.artifacts import (
    process_logger as behavior_process_logger,
)
from voltron.shared.geometry_frames import (
    is_identity_transform,
    resolve_frame_contract,
    transform_orientation,
    transform_position,
)


def goal_position(goal: dict[str, Any]) -> dict[str, Any] | None:
    position = goal.get("position")
    if isinstance(position, dict):
        return dict(position)
    coords: dict[str, Any] = {}
    for axis in ("x", "y", "z"):
        if axis not in goal:
            return None
        coords[axis] = goal.get(axis)
    return coords


def navigation_completion_goal(nav_state: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(nav_state, dict):
        return {}
    base = dict(nav_state.get("nav_goal") or {})
    goal_type = str(base.get("goal_type") or "").strip().lower()
    selected_object_approach = nav_state.get("selected_object_approach")
    if (
        goal_type == "object"
        and isinstance(selected_object_approach, dict)
        and selected_object_approach
    ):
        merged = {**base, **selected_object_approach}
        position = goal_position(selected_object_approach)
        if position is not None:
            merged["position"] = position
        return merged
    for key in ("execution_goal", "target_waypoint", "local_goal"):
        candidate = nav_state.get(key)
        if not isinstance(candidate, dict) or not candidate:
            continue
        merged = {**base, **candidate}
        position = goal_position(candidate)
        if position is not None:
            merged["position"] = position
        return merged
    return base


def normalize_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.replace("_", " ").lower().split()).strip()
    return normalized or None


def extract_simulator_pose(
    *,
    last_info: dict[str, Any],
    last_obs: dict[str, Any],
) -> dict[str, Any] | None:
    for candidate in (
        last_info.get("simulator_pose"),
        last_obs.get("simulator_pose"),
        last_info.get("simulator_robot_pose"),
        last_obs.get("simulator_robot_pose"),
    ):
        if isinstance(candidate, dict):
            return dict(candidate)

    for key in (
        "state.robot.base_pose",
        "state.base_pose",
        "state.robot.pose",
        "state.pose",
        "state.robot_pos",
    ):
        pose = array_to_pose(last_obs.get(key))
        if pose is not None:
            return pose

    for container, keys in (
        (last_obs, ("pose", "robot_pose")),
        (last_info, ("pose", "robot_pose")),
    ):
        if _field_is_scene_frame(container, field="pose"):
            continue
        scene_pose = container.get("scene_pose")
        for key in keys:
            candidate = container.get(key)
            if isinstance(candidate, dict) and candidate != scene_pose:
                return dict(candidate)
    return None


def extract_scene_pose(
    *,
    last_info: dict[str, Any],
    last_obs: dict[str, Any],
) -> dict[str, Any] | None:
    for candidate in (
        last_info.get("scene_pose"),
        last_obs.get("scene_pose"),
    ):
        if isinstance(candidate, dict):
            return dict(candidate)
    for container in (last_info, last_obs):
        if not _field_is_scene_frame(container, field="pose"):
            continue
        for key in ("pose", "robot_pose"):
            candidate = container.get(key)
            if isinstance(candidate, dict):
                return dict(candidate)
    return None


def extract_runtime_pose(
    *,
    last_info: dict[str, Any],
    last_obs: dict[str, Any],
    frame_config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    pose = extract_simulator_pose(last_info=last_info, last_obs=last_obs)
    if pose is None:
        return extract_scene_pose(last_info=last_info, last_obs=last_obs)
    contract = resolve_frame_contract(frame_config)
    transform = contract["scene_from_simulator_transform"]
    if is_identity_transform(transform):
        return pose
    return transform_position(pose, transform) or pose


def extract_simulator_orientation(
    *,
    last_info: dict[str, Any],
    last_obs: dict[str, Any],
) -> dict[str, Any] | None:
    for candidate in (
        last_info.get("simulator_orientation"),
        last_obs.get("simulator_orientation"),
        last_info.get("simulator_robot_orientation"),
        last_obs.get("simulator_robot_orientation"),
    ):
        if isinstance(candidate, dict):
            return dict(candidate)

    yaw = array_to_yaw(last_obs.get("state.robot_2d_ori"))
    if yaw is not None:
        return {"yaw": yaw}

    for key in ("state.robot.base_orientation", "state.base_orientation", "state.orientation"):
        arr = behavior_process_logger.to_numpy(last_obs.get(key))
        if arr is None or arr.size == 0:
            continue
        flat = arr.reshape(-1)
        if flat.size >= 4:
            return {
                "x": float(flat[0]),
                "y": float(flat[1]),
                "z": float(flat[2]),
                "w": float(flat[3]),
            }
        if flat.size >= 3:
            return {"roll": float(flat[0]), "pitch": float(flat[1]), "yaw": float(flat[2])}

    for container, keys in (
        (last_obs, ("orientation", "robot_orientation")),
        (last_info, ("orientation", "robot_orientation")),
    ):
        if _field_is_scene_frame(container, field="orientation"):
            continue
        scene_orientation = container.get("scene_orientation")
        for key in keys:
            candidate = container.get(key)
            if isinstance(candidate, dict) and candidate != scene_orientation:
                return dict(candidate)
    return None


def extract_scene_orientation(
    *,
    last_info: dict[str, Any],
    last_obs: dict[str, Any],
) -> dict[str, Any] | None:
    for candidate in (
        last_info.get("scene_orientation"),
        last_obs.get("scene_orientation"),
    ):
        if isinstance(candidate, dict):
            return dict(candidate)
    for container in (last_info, last_obs):
        if not _field_is_scene_frame(container, field="orientation"):
            continue
        for key in ("orientation", "robot_orientation"):
            candidate = container.get(key)
            if isinstance(candidate, dict):
                return dict(candidate)
    return None


def extract_runtime_orientation(
    *,
    last_info: dict[str, Any],
    last_obs: dict[str, Any],
    frame_config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    orientation = extract_simulator_orientation(last_info=last_info, last_obs=last_obs)
    if orientation is None:
        return extract_scene_orientation(last_info=last_info, last_obs=last_obs)
    contract = resolve_frame_contract(frame_config)
    return (
        transform_orientation(
            orientation,
            contract["scene_from_simulator_transform"],
            source_vertical_axis=contract["simulator_vertical_axis"],
            target_vertical_axis=contract["scene_vertical_axis"],
        )
        or orientation
    )


def extract_runtime_robot_state(
    *,
    last_info: dict[str, Any],
    last_obs: dict[str, Any],
    frame_config: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any] | None]:
    simulator_pose = extract_simulator_pose(last_info=last_info, last_obs=last_obs)
    simulator_orientation = extract_simulator_orientation(
        last_info=last_info,
        last_obs=last_obs,
    )
    contract = resolve_frame_contract(frame_config)
    transform = contract["scene_from_simulator_transform"]
    pose = extract_scene_pose(last_info=last_info, last_obs=last_obs)
    if simulator_pose is not None:
        pose = simulator_pose
    if simulator_pose is not None and not is_identity_transform(transform):
        pose = transform_position(simulator_pose, transform) or simulator_pose
    orientation = extract_scene_orientation(last_info=last_info, last_obs=last_obs)
    if simulator_orientation is not None:
        orientation = (
            transform_orientation(
                simulator_orientation,
                transform,
                source_vertical_axis=contract["simulator_vertical_axis"],
                target_vertical_axis=contract["scene_vertical_axis"],
            )
            or simulator_orientation
        )
    return {
        "pose": pose,
        "orientation": orientation,
        "simulator_pose": simulator_pose,
        "simulator_orientation": simulator_orientation,
    }


def _field_is_scene_frame(container: dict[str, Any], *, field: str) -> bool:
    if not isinstance(container, dict):
        return False
    markers = (
        container.get(f"{field}_frame"),
        container.get(f"robot_{field}_frame"),
        container.get("frame_id"),
        container.get("coordinate_frame"),
    )
    return any(str(marker or "").strip().lower() in {"scene", "map", "hovsg"} for marker in markers)


def extract_runtime_region(*, last_info: dict[str, Any], last_obs: dict[str, Any]) -> str | None:
    for candidate in (
        last_info.get("region"),
        last_info.get("room"),
        last_info.get("current_region"),
        last_info.get("current_room"),
        last_obs.get("region"),
        last_obs.get("room"),
        last_obs.get("current_region"),
        last_obs.get("current_room"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def extract_runtime_nav_feedback(*, last_info: dict[str, Any]) -> dict[str, Any]:
    feedback = dict(last_info.get("nav_feedback", {}))
    for key in ("collision", "stuck", "blocked", "reachable", "local_progress"):
        if key in last_info:
            feedback[key] = last_info.get(key)
    return feedback


def array_to_pose(value: Any) -> dict[str, Any] | None:
    arr = behavior_process_logger.to_numpy(value)
    if arr is None or arr.size < 3:
        return None
    flat = arr.reshape(-1)
    return {"x": float(flat[0]), "y": float(flat[1]), "z": float(flat[2])}


def array_to_yaw(value: Any) -> float | None:
    arr = behavior_process_logger.to_numpy(value)
    if arr is None or arr.size == 0:
        return None
    flat = arr.reshape(-1)
    try:
        return float(flat[0])
    except (TypeError, ValueError):
        return None


def orientation_to_yaw(value: Any) -> float | None:
    if isinstance(value, dict):
        if "yaw" in value:
            try:
                return float(value["yaw"])
            except (TypeError, ValueError):
                return None
        quaternion = [value.get("x"), value.get("y"), value.get("z"), value.get("w")]
        if all(component is not None for component in quaternion):
            try:
                return quat_to_yaw(
                    float(quaternion[0]),
                    float(quaternion[1]),
                    float(quaternion[2]),
                    float(quaternion[3]),
                )
            except (TypeError, ValueError):
                return None

    arr = behavior_process_logger.to_numpy(value)
    if arr is None or arr.size == 0:
        return None
    flat = arr.reshape(-1)
    if flat.size >= 4:
        return quat_to_yaw(
            float(flat[0]),
            float(flat[1]),
            float(flat[2]),
            float(flat[3]),
        )
    if flat.size >= 3:
        return float(flat[2])
    return None


def quat_to_yaw(x_coord: float, y_coord: float, z_coord: float, w_coord: float) -> float:
    siny_cosp = 2.0 * (w_coord * z_coord + x_coord * y_coord)
    cosy_cosp = 1.0 - 2.0 * (y_coord * y_coord + z_coord * z_coord)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def planar_axes(vertical_axis: Any) -> tuple[str, str]:
    if vertical_axis == "x":
        return "y", "z"
    if vertical_axis == "z":
        return "x", "y"
    return "x", "z"


def to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def object_goal_geometry_status(
    *,
    pose: dict[str, Any],
    orientation: dict[str, Any] | None,
    goal: dict[str, Any],
    localized: dict[str, Any],
    distance_tolerance_m: float,
    heading_tolerance_rad: float,
) -> dict[str, Any] | None:
    target_position = goal_position(goal)
    if target_position is None:
        return None

    axis_x, axis_y = planar_axes(localized.get("vertical_axis"))
    current_x = to_float(pose.get(axis_x))
    current_y = to_float(pose.get(axis_y))
    goal_x = to_float(target_position.get(axis_x))
    goal_y = to_float(target_position.get(axis_y))
    if None in {current_x, current_y, goal_x, goal_y}:
        return None

    delta_x = float(goal_x) - float(current_x)
    delta_y = float(goal_y) - float(current_y)
    planar_distance = math.hypot(delta_x, delta_y)
    if planar_distance > distance_tolerance_m:
        return {"reached": False, "planar_distance": planar_distance, "heading_error": None}

    yaw = orientation_to_yaw(orientation)
    if yaw is None:
        return {"reached": False, "planar_distance": planar_distance, "heading_error": None}

    desired_heading = to_float(goal.get("desired_heading"))
    if desired_heading is None:
        desired_heading = math.atan2(delta_y, delta_x)
    heading_error = wrap_angle(desired_heading - yaw)
    return {
        "reached": abs(heading_error) <= heading_tolerance_rad,
        "planar_distance": planar_distance,
        "heading_error": heading_error,
    }


def object_navigation_goal_status(
    *,
    geometry_status: dict[str, Any] | None,
    nav_state: dict[str, Any],
) -> dict[str, Any]:
    path_backend = str(nav_state.get("path_backend") or "").strip().lower()
    if path_backend == "global_goal_reached":
        controller_mode = str(nav_state.get("controller_mode") or "").strip().lower() or None
        return {"reached": True, "controller_mode": controller_mode}

    if not isinstance(geometry_status, dict) or not geometry_status.get("reached"):
        return {"reached": False, "controller_mode": None}

    has_policy_completion_signal = any(
        key in nav_state for key in ("goal_reached", "controller_mode")
    )
    if not has_policy_completion_signal:
        return {"reached": True, "controller_mode": None}

    controller_mode = str(nav_state.get("controller_mode") or "").strip().lower() or None
    policy_goal_reached = (
        bool(nav_state.get("goal_reached"))
        or controller_mode == "goal_reached"
        or path_backend == "global_goal_reached"
    )
    return {"reached": policy_goal_reached, "controller_mode": controller_mode}


def navigation_goal_match_status(
    *,
    goal: dict[str, Any],
    target: dict[str, Any],
    localized: dict[str, Any],
    object_goal_reached: bool,
    policy_goal_reached: bool = False,
) -> dict[str, Any]:
    goal_type = str(goal.get("goal_type") or "").strip().lower()
    target_room_id = str(target.get("room_id") or goal.get("room_id") or "").strip()
    target_room_name = normalize_label(target.get("room_name") or goal.get("room_name"))
    room_id = str(localized.get("room_id") or "").strip()
    floor_id = str(localized.get("floor_id") or "").strip()
    current_room = normalize_label(localized.get("current_room"))
    current_region = normalize_label(localized.get("current_region"))

    if goal_type == "object":
        return {
            "reached": bool(object_goal_reached),
            "match_reason": "object_goal_reached"
            if object_goal_reached
            else "object_goal_incomplete",
        }
    if target_room_id and room_id and target_room_id == room_id:
        return {"reached": True, "match_reason": "target_room_id"}
    if goal_type == "room" and room_id and str(goal.get("room_id") or "").strip() == room_id:
        return {"reached": True, "match_reason": "goal_room_id"}
    if goal_type == "floor" and floor_id and str(goal.get("floor_id") or "").strip() == floor_id:
        return {"reached": True, "match_reason": "goal_floor_id"}

    for candidate in (
        target_room_name,
        target.get("room"),
        target.get("region"),
        target.get("location"),
        goal.get("room_name"),
        goal.get("instruction"),
    ):
        normalized = normalize_label(candidate)
        if normalized and normalized in {current_room, current_region}:
            return {"reached": True, "match_reason": "semantic_region_match"}
    if policy_goal_reached and goal_type not in {"room", "floor"}:
        return {"reached": True, "match_reason": "policy_goal_reached"}
    if policy_goal_reached:
        return {"reached": False, "match_reason": "policy_goal_semantic_mismatch"}
    return {"reached": False, "match_reason": "no_match"}
