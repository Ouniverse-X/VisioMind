"""Runtime observation helpers for the Navigation agent."""

from __future__ import annotations

from typing import Any

import numpy as np

from voltron.shared.context import ExecutionContext, Subtask


def extract_pose(subtask: Subtask, observation: dict[str, Any]) -> dict[str, Any] | None:
    for candidate in (
        subtask.parameters.get("pose"),
        subtask.context.get("pose"),
        observation.get("pose"),
    ):
        if isinstance(candidate, dict):
            return dict(candidate)
    for key in ("state.robot.base_pose", "state.base_pose", "state.robot.pose", "state.pose", "state.robot_pos"):
        pose = array_to_pose(observation.get(key))
        if pose is not None:
            return pose
    return None


def build_policy_observation(
    *,
    observation: dict[str, Any],
    scene_id: str | None,
    pose: dict[str, Any] | None,
    orientation: dict[str, Any] | None,
    current_region: str | None,
) -> dict[str, Any]:
    policy_observation = dict(observation)
    if scene_id:
        policy_observation.setdefault("scene_id", scene_id)
    if pose is not None:
        policy_observation.setdefault("pose", dict(pose))
    if orientation is not None:
        policy_observation.setdefault("orientation", dict(orientation))
    if current_region:
        policy_observation.setdefault("current_region", current_region)
        policy_observation.setdefault("current_room", current_region)
    return policy_observation


def merge_runtime_navigation_options(
    *,
    subtask: Subtask,
    existing_options: dict[str, Any] | None,
) -> dict[str, Any] | None:
    options = dict(existing_options or {})
    dynamic_local_waypoints = (
        options.get("waypoint_tracking_mode") == "global_local_hybrid"
        or options.get("waypoint_scope") == "dynamic_local_segment"
    )
    for key in (
        "active_waypoint_index",
        "recovery_mode",
        "exploration_target",
        "pose",
        "orientation",
        "vertical_axis",
        "nav_vertical_axis",
    ):
        value = subtask.parameters.get(key)
        if value is not None:
            if key == "active_waypoint_index" and dynamic_local_waypoints:
                continue
            options[key] = value
    return options or None


def extract_policy_runtime_artifacts(info: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(info, dict):
        return {}
    artifacts: dict[str, Any] = {}
    for key in (
        "active_waypoint_index",
        "recovery_mode",
        "recovery_profile",
        "recovery_cycles_on_waypoint",
        "exploration_target",
        "controller_mode",
        "follow_status",
        "goal_reached",
        "local_segment_complete",
        "requires_replan",
        "replan_reason",
        "distance_to_waypoint",
        "heading_error",
        "yaw_source",
        "path_backend",
        "path_tracking_mode",
        "tracking_target",
        "target_waypoint",
        "vertical_axis",
        "loop_detected",
        "oscillation_detected",
        "steps_since_progress",
        "best_distance_to_waypoint",
        "path_cross_track_error",
        "path_signed_cross_track_error",
        "path_segment_index",
        "path_tangent_heading",
        "nav2_input_path_point_count",
        "policy_path_transform",
    ):
        if key in info:
            artifacts[key] = info[key]
    return artifacts


def extract_region(subtask: Subtask, observation: dict[str, Any]) -> str | None:
    for candidate in (
        subtask.parameters.get("region"),
        subtask.context.get("region"),
        observation.get("region"),
        observation.get("room"),
        observation.get("current_region"),
        observation.get("current_room"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def extract_orientation(subtask: Subtask, observation: dict[str, Any]) -> dict[str, Any] | None:
    yaw = array_to_yaw(observation.get("state.robot_2d_ori"))
    if yaw is not None:
        return {"yaw": yaw}

    for key in ("state.robot.base_orientation", "state.base_orientation", "state.orientation"):
        yaw = orientation_to_yaw(observation.get(key))
        if yaw is not None:
            return {"yaw": yaw}

    for candidate in (
        subtask.parameters.get("orientation"),
        subtask.context.get("orientation"),
        observation.get("orientation"),
    ):
        if isinstance(candidate, dict):
            return dict(candidate)
    return None


def extract_nav_feedback(subtask: Subtask, observation: dict[str, Any]) -> dict[str, Any] | None:
    for candidate in (
        subtask.parameters.get("nav_feedback"),
        subtask.context.get("nav_feedback"),
        observation.get("nav_feedback"),
    ):
        if isinstance(candidate, dict):
            return dict(candidate)
    return None


def resolve_scene_id(subtask: Subtask, context: ExecutionContext, observation: dict[str, Any]) -> str | None:
    for candidate in (
        subtask.parameters.get("scene_id"),
        subtask.context.get("scene_id"),
        observation.get("scene_id"),
        context.task_request.metadata.get("scene_id"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def extract_start_metadata(backend_state: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(backend_state, dict):
        return {}

    start: dict[str, Any] = {}
    for key in (
        "scene_id",
        "current_room",
        "current_region",
        "room_id",
        "floor_id",
        "region",
        "nav_node",
        "object_id",
        "vertical_axis",
        "localization_guard",
    ):
        value = backend_state.get(key)
        if value is not None:
            start[key] = value
    return start


def extract_runtime_region(backend_state: dict[str, Any] | None) -> str | None:
    if not isinstance(backend_state, dict):
        return None
    for key in ("current_region", "current_room"):
        value = backend_state.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def array_to_pose(value: Any) -> dict[str, Any] | None:
    array = to_numpy(value)
    if array is None or array.size < 3:
        return None
    flattened = array.reshape(-1)
    return {
        "x": float(flattened[0]),
        "y": float(flattened[1]),
        "z": float(flattened[2]),
    }


def array_to_yaw(value: Any) -> float | None:
    array = to_numpy(value)
    if array is None or array.size == 0:
        return None
    flattened = array.reshape(-1)
    try:
        return float(flattened[0])
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

    array = to_numpy(value)
    if array is None or array.size == 0:
        return None
    flattened = array.reshape(-1)
    if flattened.size >= 4:
        return quat_to_yaw(
            float(flattened[0]),
            float(flattened[1]),
            float(flattened[2]),
            float(flattened[3]),
        )
    if flattened.size >= 3:
        return float(flattened[2])
    return None


def quat_to_yaw(x_coord: float, y_coord: float, z_coord: float, w_coord: float) -> float:
    siny_cosp = 2.0 * (w_coord * z_coord + x_coord * y_coord)
    cosy_cosp = 1.0 - 2.0 * (y_coord * y_coord + z_coord * z_coord)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def to_numpy(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.asarray(value)
    except Exception:
        return None
