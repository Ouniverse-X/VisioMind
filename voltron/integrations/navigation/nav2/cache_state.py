from __future__ import annotations

from typing import Any, Callable


def reuse_cached_segment(*, cached_plan: dict[str, Any] | None, error_text: str) -> dict[str, Any]:
    reused = dict(cached_plan or {})
    reused["nav2_cache_reused"] = True
    if error_text:
        existing_error = str(reused.get("nav2_error") or "").strip()
        if existing_error:
            reused["nav2_error"] = existing_error if error_text in existing_error else f"{existing_error}; {error_text}"
        else:
            reused["nav2_error"] = error_text
    return reused


def build_cached_local_segment(
    *,
    scene_id: str | None,
    vertical_axis: str,
    active_global_waypoint_index: int,
    local_goal: dict[str, Any],
    execution_goal: dict[str, Any],
    nav2_compute_goal: dict[str, Any],
    waypoints: list[dict[str, Any]],
    nav2_raw_path_points: list[dict[str, float]],
    nav2_path_points: list[dict[str, float]],
    nav2_raw_path_length: int,
    dense_waypoint_index: int,
) -> dict[str, Any]:
    return {
        "scene_id": scene_id,
        "vertical_axis": vertical_axis,
        "global_waypoint_index": int(active_global_waypoint_index),
        "local_goal": dict(local_goal),
        "execution_goal": dict(execution_goal),
        "nav2_compute_goal": dict(nav2_compute_goal),
        "waypoints": [dict(waypoint) for waypoint in waypoints if isinstance(waypoint, dict)],
        "nav2_raw_path_points": [
            dict(point) for point in nav2_raw_path_points if isinstance(point, dict)
        ],
        "nav2_path_points": [dict(point) for point in nav2_path_points if isinstance(point, dict)],
        "nav2_raw_path_length": int(nav2_raw_path_length),
        "dense_waypoint_index": int(dense_waypoint_index),
    }


def reuse_matching_local_segment(
    *,
    cached_segment: dict[str, Any] | None,
    scene_id: str | None,
    vertical_axis: str,
    active_global_waypoint_index: int,
    start_pose: dict[str, Any] | None,
    local_goal: dict[str, Any],
    execution_goal: dict[str, Any],
    nav2_compute_goal: dict[str, Any],
    error_text: str,
    same_waypoint_signature: Callable[[Any, Any], bool],
    planar_distance: Callable[..., float],
    waypoint_spacing: float,
) -> dict[str, Any] | None:
    cached = cached_segment
    if not isinstance(cached, dict):
        return None
    if cached.get("scene_id") != scene_id:
        return None
    if cached.get("vertical_axis") != vertical_axis:
        return None
    if int(cached.get("global_waypoint_index", -1)) != int(active_global_waypoint_index):
        return None
    if not same_waypoint_signature(cached.get("local_goal"), local_goal):
        return None
    if not same_waypoint_signature(cached.get("execution_goal"), execution_goal):
        return None
    if not same_waypoint_signature(cached.get("nav2_compute_goal"), nav2_compute_goal):
        return None

    cached_waypoints = cached.get("waypoints")
    if not isinstance(cached_waypoints, list) or not cached_waypoints:
        return None
    reusable_waypoints = [dict(waypoint) for waypoint in cached_waypoints if isinstance(waypoint, dict)]
    if not reusable_waypoints:
        return None

    if isinstance(start_pose, dict):
        nearest_distance = min(
            planar_distance(first=start_pose, second=waypoint, vertical_axis=vertical_axis)
            for waypoint in reusable_waypoints
        )
        if nearest_distance > max(2.0, float(waypoint_spacing) * 6.0):
            return None

    return reuse_cached_segment(
        cached_plan={
            "waypoints": reusable_waypoints,
            "nav2_raw_path_points": [
                dict(point) for point in cached.get("nav2_raw_path_points", []) if isinstance(point, dict)
            ],
            "nav2_path_points": [
                dict(point) for point in cached.get("nav2_path_points", []) if isinstance(point, dict)
            ],
            "nav2_raw_path_length": int(cached.get("nav2_raw_path_length", 0)),
            "dense_waypoint_index": int(cached.get("dense_waypoint_index", 0)),
        },
        error_text=error_text,
    )
