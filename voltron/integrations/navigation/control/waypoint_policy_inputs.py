"""Input parsing helpers for the waypoint policy adapter."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

_WAYPOINT_METADATA_KEYS = (
    "floor_id",
    "nav_node",
    "room_id",
    "room_name",
    "waypoint_type",
    "source_room_id",
    "source_room_name",
    "portal_gap",
    "portal_span",
    "portal_source_point",
    "portal_target_point",
    "portal_normal_axis",
    "portal_boundary_value",
    "portal_normal_sign",
    "portal_span_axis",
    "portal_span_min",
    "portal_span_max",
    "portal_refined_from_traversability",
    "portal_desired_heading",
    "portal_alignment_stage",
    "portal_egress_guard",
    "portal_egress_source",
    "portal_egress_depth_m",
    "portal_required_egress_depth_m",
    "portal_egress_guard_persisted",
    "transition_anchor",
    "desired_heading",
    "object_id",
    "object_name",
    "object_position",
    "approach_distance_m",
    "approach_boundary_distance_m",
)


def extract_waypoints(*, options: dict[str, Any]) -> list[dict[str, float]]:
    candidates = options.get("nav_waypoints")
    if not isinstance(candidates, list):
        nav_plan = options.get("nav_plan")
        if isinstance(nav_plan, dict):
            candidates = nav_plan.get("waypoints")
    if not isinstance(candidates, list):
        return []

    waypoints: list[dict[str, float]] = []
    for candidate in candidates:
        waypoint = _coerce_waypoint_candidate(candidate)
        if waypoint is not None:
            waypoints.append(waypoint)
    return waypoints


def pending_local_path_transition_goal(*, options: dict[str, Any]) -> dict[str, float] | None:
    if bool(options.get("suppress_pending_local_path_transition_goal")):
        return None

    nav_plan = options.get("nav_plan")
    if not isinstance(nav_plan, dict):
        return None

    candidate = None
    candidate_is_refined_transition = False
    refined_transition_anchor = nav_plan.get("transition_anchor")
    if isinstance(refined_transition_anchor, dict) and refined_transition_anchor.get(
        "portal_refined_from_traversability"
    ):
        candidate = refined_transition_anchor
        candidate_is_refined_transition = True
    else:
        execution_goal = nav_plan.get("execution_goal")
        if (
            isinstance(execution_goal, dict)
            and str(execution_goal.get("waypoint_type", "")).strip().lower() == "post_portal_goal"
        ):
            candidate = execution_goal
        else:
            candidate = nav_plan.get("local_goal")
        if not isinstance(candidate, dict):
            candidate = refined_transition_anchor
    if not isinstance(candidate, dict):
        return None
    if bool(nav_plan.get("nav2_path_clipped_for_clearance")) and not candidate_is_refined_transition:
        return None

    return _coerce_waypoint_candidate(candidate)


def extract_pose(
    *,
    observation: dict[str, Any],
    options: dict[str, Any],
) -> dict[str, float] | None:
    for candidate in (observation.get("pose"), options.get("pose")):
        if isinstance(candidate, dict):
            pose = normalize_pose_dict(candidate)
            if pose is not None:
                return pose

    for key in ("state.robot.base_pose", "state.base_pose", "state.robot.pose", "state.pose"):
        pose = array_to_pose(observation.get(key))
        if pose is not None:
            return pose
    pose = array_to_pose(observation.get("state.robot_pos"))
    if pose is not None:
        return pose
    return None


def extract_yaw(
    *,
    observation: dict[str, Any],
    options: dict[str, Any],
) -> float:
    yaw, _ = extract_yaw_with_source(observation=observation, options=options)
    return yaw


def extract_yaw_with_source(
    *,
    observation: dict[str, Any],
    options: dict[str, Any],
) -> tuple[float, str]:
    yaw = array_to_yaw(observation.get("state.robot_2d_ori"))
    if yaw is not None:
        return yaw, "state.robot_2d_ori"

    for key in ("state.robot.base_orientation", "state.base_orientation", "state.orientation"):
        yaw = orientation_to_yaw(observation.get(key))
        if yaw is not None:
            return yaw, key

    for source, candidate in (
        ("observation.orientation", observation.get("orientation")),
        ("options.orientation", options.get("orientation")),
    ):
        yaw = orientation_to_yaw(candidate)
        if yaw is not None:
            return yaw, source

    return 0.0, "default_zero"


def resolve_vertical_axis(
    *,
    observation: dict[str, Any],
    options: dict[str, Any],
) -> str:
    resolved: str | None = None
    for candidate in (
        options.get("nav_vertical_axis"),
        options.get("vertical_axis"),
        observation.get("vertical_axis"),
    ):
        if isinstance(candidate, str) and candidate in {"x", "y", "z"}:
            resolved = candidate
            break

    nav_plan = options.get("nav_plan")
    if resolved is None and isinstance(nav_plan, dict):
        candidate = nav_plan.get("vertical_axis")
        if isinstance(candidate, str) and candidate in {"x", "y", "z"}:
            resolved = candidate
    resolved = resolved or "z"
    if resolved == "y" and _looks_like_isaac_xy_ground_path(options=options):
        return "z"
    return resolved


def _looks_like_isaac_xy_ground_path(*, options: dict[str, Any]) -> bool:
    if str(options.get("path_tracking_mode") or "").strip().lower() != "nav2_local_path":
        return False
    waypoints = options.get("nav_waypoints")
    if not isinstance(waypoints, list) or not waypoints:
        return False

    ys: list[float] = []
    zs: list[float] = []
    for waypoint in waypoints:
        if not isinstance(waypoint, dict):
            continue
        try:
            ys.append(float(waypoint.get("y", 0.0)))
            zs.append(float(waypoint.get("z", 0.0)))
        except (TypeError, ValueError):
            continue
    if not zs:
        return False

    z_span = max(zs) - min(zs)
    y_span = max(ys) - min(ys) if ys else 0.0
    return z_span <= 0.05 and (len(zs) == 1 or y_span > 0.05)


def horizontal_axes(vertical_axis: str) -> tuple[str, str]:
    axes_by_vertical = {
        "x": ("y", "z"),
        "y": ("x", "z"),
        "z": ("x", "y"),
    }
    return axes_by_vertical.get(vertical_axis, ("x", "y"))


def extract_nav_feedback(
    *,
    observation: dict[str, Any],
    options: dict[str, Any],
) -> dict[str, Any]:
    for candidate in (observation.get("nav_feedback"), options.get("nav_feedback")):
        if isinstance(candidate, dict):
            return dict(candidate)
    return {}


def normalize_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split()).strip()
    return normalized or None


def normalize_pose_dict(candidate: dict[str, Any]) -> dict[str, float] | None:
    try:
        return {
            "x": float(candidate["x"]),
            "y": float(candidate["y"]),
            "z": float(candidate.get("z", 0.0)),
        }
    except (KeyError, TypeError, ValueError):
        return None


def array_to_pose(value: Any) -> dict[str, float] | None:
    arr = to_numpy(value)
    if arr is None or arr.size < 3:
        return None
    flat = arr.reshape(-1)
    return {
        "x": float(flat[0]),
        "y": float(flat[1]),
        "z": float(flat[2]),
    }


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

    arr = to_numpy(value)
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


def array_to_yaw(value: Any) -> float | None:
    arr = to_numpy(value)
    if arr is None or arr.size == 0:
        return None
    flat = arr.reshape(-1)
    try:
        return float(flat[0])
    except (TypeError, ValueError):
        return None


def quat_to_yaw(x_coord: float, y_coord: float, z_coord: float, w_coord: float) -> float:
    siny_cosp = 2.0 * (w_coord * z_coord + x_coord * y_coord)
    cosy_cosp = 1.0 - 2.0 * (y_coord * y_coord + z_coord * z_coord)
    return math.atan2(siny_cosp, cosy_cosp)


def coerce_index(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def to_numpy(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.asarray(value)
    except Exception:
        return None


def _coerce_waypoint_candidate(candidate: Any) -> dict[str, float] | None:
    if not isinstance(candidate, dict):
        return None
    try:
        waypoint = {
            "x": float(candidate["x"]),
            "y": float(candidate["y"]),
            "z": float(candidate.get("z", 0.0)),
        }
    except (KeyError, TypeError, ValueError):
        return None

    for key in _WAYPOINT_METADATA_KEYS:
        if key in candidate:
            waypoint[key] = candidate[key]
    return waypoint
