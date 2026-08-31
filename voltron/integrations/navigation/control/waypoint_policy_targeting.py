from __future__ import annotations

import math
from typing import Any


def tracking_target(
    adapter: Any,
    *,
    waypoints: list[dict[str, float]],
    active_index: int,
    axis_x: str,
    axis_y: str,
) -> dict[str, float]:
    target = dict(waypoints[active_index])
    if adapter._uses_local_path_tracking():
        return local_path_tracking_target(
            adapter,
            waypoints=waypoints,
            active_index=active_index,
            axis_x=axis_x,
            axis_y=axis_y,
        )
    if adapter.lookahead_points <= 0:
        return target

    weighted_waypoints: list[tuple[float, dict[str, float]]] = []
    for offset in range(adapter.lookahead_points + 1):
        index = active_index + offset
        if index >= len(waypoints):
            break
        candidate = waypoints[index]
        if weighted_waypoints and not can_blend_tracking_waypoint(
            adapter,
            weighted_waypoints[-1][1],
            candidate,
        ):
            break
        weighted_waypoints.append((adapter.lookahead_decay**offset, candidate))
    weight_sum = sum(weight for weight, _ in weighted_waypoints)
    if weight_sum <= 1e-6:
        return target

    blended: dict[str, float] = {}
    for axis in ("x", "y", "z"):
        blended[axis] = (
            sum(
                weight * float(point.get(axis, target.get(axis, 0.0)))
                for weight, point in weighted_waypoints
            )
            / weight_sum
        )
    for key in ("floor_id", "room_id", "room_name", "nav_node"):
        if key in target:
            blended[key] = target[key]
    return blended


def local_path_tracking_target(
    adapter: Any,
    *,
    waypoints: list[dict[str, float]],
    active_index: int,
    axis_x: str,
    axis_y: str,
) -> dict[str, float]:
    target = dict(waypoints[active_index])
    tracking_horizon = adaptive_local_path_tracking_horizon(
        adapter,
        waypoints=waypoints,
        active_index=active_index,
        axis_x=axis_x,
        axis_y=axis_y,
    )
    if tracking_horizon <= 1e-6:
        return target

    remaining = tracking_horizon
    previous = target
    for index in range(active_index + 1, len(waypoints)):
        candidate = waypoints[index]
        segment = adapter._planar_distance(
            pose=previous,
            target=candidate,
            axis_x=axis_x,
            axis_y=axis_y,
        )
        if segment <= 1e-6:
            previous = candidate
            continue
        if segment >= remaining:
            ratio = remaining / segment
            interpolated = dict(previous)
            interpolated[axis_x] = float(previous[axis_x]) + ratio * (
                float(candidate[axis_x]) - float(previous[axis_x])
            )
            interpolated[axis_y] = float(previous[axis_y]) + ratio * (
                float(candidate[axis_y]) - float(previous[axis_y])
            )
            if "z" in previous and "z" in candidate:
                interpolated["z"] = float(previous["z"]) + ratio * (
                    float(candidate["z"]) - float(previous["z"])
                )
            return interpolated
        remaining -= segment
        previous = candidate
    return dict(previous)


def adaptive_local_path_tracking_horizon(
    adapter: Any,
    *,
    waypoints: list[dict[str, float]],
    active_index: int,
    axis_x: str,
    axis_y: str,
) -> float:
    base_horizon = adapter.local_path_tracking_horizon
    if adapter.local_path_max_tracking_horizon <= base_horizon + 1e-6:
        return base_horizon
    if active_index + 2 >= len(waypoints):
        return adapter.local_path_max_tracking_horizon

    first = waypoints[active_index]
    second = waypoints[active_index + 1]
    third = waypoints[active_index + 2]
    first_heading = math.atan2(
        float(second[axis_y]) - float(first[axis_y]),
        float(second[axis_x]) - float(first[axis_x]),
    )
    second_heading = math.atan2(
        float(third[axis_y]) - float(second[axis_y]),
        float(third[axis_x]) - float(second[axis_x]),
    )
    curvature = abs(adapter._wrap_angle(second_heading - first_heading))
    if adapter.local_path_curve_threshold_rad <= 1e-6:
        return base_horizon
    straightness = max(0.0, 1.0 - min(1.0, curvature / adapter.local_path_curve_threshold_rad))
    return base_horizon + straightness * (adapter.local_path_max_tracking_horizon - base_horizon)


def can_blend_tracking_waypoint(
    adapter: Any,
    current: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    if adapter._uses_local_path_tracking():
        return False
    if str(current.get("waypoint_type", "")).strip().lower() == "portal":
        return False
    if str(candidate.get("waypoint_type", "")).strip().lower() == "portal":
        return False
    current_room_id = current.get("room_id")
    candidate_room_id = candidate.get("room_id")
    if (
        current_room_id is not None
        and candidate_room_id is not None
        and str(current_room_id) != str(candidate_room_id)
    ):
        return False
    current_room = adapter._normalize_label(current.get("room_name"))
    candidate_room = adapter._normalize_label(candidate.get("room_name"))
    if current_room and candidate_room and current_room != candidate_room:
        return False
    return True
