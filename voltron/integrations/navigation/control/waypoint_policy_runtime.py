"""Runtime primitives for the waypoint policy adapter."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_path_tracking_mode(
    *,
    options: dict[str, Any],
    waypoints: list[dict[str, float]],
) -> str | None:
    candidate = options.get("path_tracking_mode")
    if isinstance(candidate, str):
        normalized = candidate.strip().lower()
        if normalized:
            return normalized

    if any(
        str(waypoint.get("waypoint_type", "")).strip().lower() == "local_path"
        for waypoint in waypoints
    ):
        return "nav2_local_path"
    dense_local_waypoint_count = sum(
        1
        for waypoint in waypoints
        if str(waypoint.get("waypoint_type", "")).strip().lower() == "local_dense_path"
    )
    if dense_local_waypoint_count >= 2:
        return "semantic_local_path"
    return None


def uses_local_path_tracking(adapter: Any) -> bool:
    return adapter._path_tracking_mode in {"nav2_local_path", "semantic_local_path"}


def local_path_info_value(adapter: Any, key: str) -> Any:
    if not isinstance(adapter._local_path_follow_state, dict):
        return None
    return adapter._local_path_follow_state.get(key)


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def waypoint_list_signature(
    *,
    waypoints: list[dict[str, float]],
    path_tracking_mode: str | None,
) -> tuple[tuple[float, float, float], ...]:
    if path_tracking_mode in {"nav2_local_path", "semantic_local_path"} and waypoints:
        return tuple(
            (
                round(float(waypoint["x"]), 1),
                round(float(waypoint["y"]), 1),
                0.0,
            )
            for waypoint in waypoints
        )
    signature: list[tuple[float, float, float]] = []
    for waypoint in waypoints:
        signature.append(
            (
                round(float(waypoint["x"]), 3),
                round(float(waypoint["y"]), 3),
                round(float(waypoint.get("z", 0.0)), 3),
            )
        )
    return tuple(signature)


def zero_action() -> dict[str, Any]:
    return base_action(0.0, 0.0, 0.0)


def base_action(
    local_x_velocity: float,
    local_y_velocity: float,
    angular_velocity: float,
) -> dict[str, Any]:
    return {
        "action.base": np.array(
            [[[float(local_x_velocity), float(local_y_velocity), float(angular_velocity)]]],
            dtype=np.float32,
        )
    }
