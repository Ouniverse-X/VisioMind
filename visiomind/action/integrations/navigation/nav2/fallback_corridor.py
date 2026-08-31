from __future__ import annotations

import math
from typing import Any, Callable


def append_transition_corridor_to_path(
    *,
    path_points: list[dict[str, float]],
    doorway_corridor: dict[str, Any],
    vertical_axis: str,
    start_from: str,
    world_pose_to_nav2_plane: Callable[..., dict[str, float] | None],
) -> list[dict[str, float]]:
    result = [dict(point) for point in path_points]
    ordered_keys = ("source_anchor", "midpoint", "target_anchor")
    try:
        start_index = ordered_keys.index(start_from)
    except ValueError:
        start_index = 0
    for key in ordered_keys[start_index:]:
        candidate = doorway_corridor.get(key)
        if not isinstance(candidate, dict):
            continue
        point = world_pose_to_nav2_plane(candidate, vertical_axis=vertical_axis)
        if point is None:
            continue
        if (
            not result
            or math.hypot(result[-1]["x"] - point["x"], result[-1]["y"] - point["y"]) > 1e-6
        ):
            result.append(point)
        else:
            result[-1] = point
    return result


def doorway_corridor_stage_key(
    *,
    waypoint: dict[str, Any] | None,
    doorway_corridor: dict[str, Any] | None,
    same_waypoint_signature: Callable[[Any, Any], bool],
) -> str:
    if not isinstance(waypoint, dict) or not isinstance(doorway_corridor, dict):
        return "source_anchor"
    for key in ("source_anchor", "midpoint", "target_anchor"):
        candidate = doorway_corridor.get(key)
        if same_waypoint_signature(candidate, waypoint):
            return key
    return "source_anchor"


def build_doorway_corridor_fallback_waypoints(
    *,
    start_pose: dict[str, Any] | None,
    execution_goal: dict[str, Any],
    doorway_corridor: dict[str, Any] | None,
    scene_id: str | None,
    vertical_axis: str,
    nav2_trav_map_filename: str | None,
    waypoint_spacing: float,
    distance: Callable[[dict[str, Any], dict[str, Any]], float],
    filter_waypoints_for_local_clearance: Callable[..., list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if not isinstance(start_pose, dict) or not isinstance(doorway_corridor, dict):
        return []

    local_waypoints: list[dict[str, Any]] = []
    for key in ("source_anchor", "midpoint", "target_anchor"):
        waypoint = doorway_corridor.get(key)
        if not isinstance(waypoint, dict):
            continue
        if distance(start_pose, waypoint) < float(waypoint_spacing) * 0.5:
            continue
        local_waypoints.append(dict(waypoint))

    if not local_waypoints:
        return []

    final_waypoint = dict(execution_goal)
    if distance(local_waypoints[-1], final_waypoint) < float(waypoint_spacing) * 0.5:
        local_waypoints[-1] = final_waypoint
    else:
        local_waypoints.append(final_waypoint)

    return filter_waypoints_for_local_clearance(
        start_pose=start_pose,
        vertical_axis=vertical_axis,
        waypoints=local_waypoints,
        scene_id=scene_id,
        nav2_trav_map_filename=nav2_trav_map_filename,
    )
