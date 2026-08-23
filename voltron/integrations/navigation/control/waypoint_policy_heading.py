"""Heading and alignment helpers for the waypoint policy adapter."""

from __future__ import annotations

import math
from typing import Any


def effective_waypoint_tolerance(
    adapter: Any,
    *,
    target: dict[str, Any],
    is_final: bool,
) -> float:
    if adapter._uses_local_path_tracking():
        base_tolerance = adapter.local_path_final_waypoint_tolerance if is_final else adapter.local_path_waypoint_tolerance
    else:
        base_tolerance = adapter.final_waypoint_tolerance if is_final else adapter.waypoint_tolerance
    if is_final and str(target.get("waypoint_type", "")).strip().lower() == "object_approach":
        return min(base_tolerance, adapter.object_approach_final_waypoint_tolerance_m)
    return base_tolerance


def portal_desired_heading(*, target: dict[str, Any]) -> float | None:
    value = target.get("portal_desired_heading")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def desired_heading(*, target: dict[str, Any]) -> float | None:
    value = target.get("desired_heading")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def waypoint_desired_heading(*, target: dict[str, Any]) -> float | None:
    resolved_desired_heading = desired_heading(target=target)
    if resolved_desired_heading is not None:
        return resolved_desired_heading
    return portal_desired_heading(target=target)


def is_portal_like_waypoint(target: dict[str, Any]) -> bool:
    waypoint_type = str(target.get("waypoint_type", "")).strip().lower()
    return waypoint_type in {"portal", "pre_portal_standoff", "portal_midpoint"}


def waypoint_requires_portal_heading_alignment(
    adapter: Any,
    target: dict[str, Any],
    *,
    distance: float | None = None,
) -> bool:
    if portal_desired_heading(target=target) is None:
        return False
    if not is_portal_like_waypoint(target):
        return False
    if distance is not None and distance > adapter.portal_alignment_distance_threshold:
        return False
    alignment_stage = str(target.get("portal_alignment_stage", "")).strip().lower()
    if alignment_stage not in {"source_anchor", "midpoint", "target_anchor"}:
        return False
    return not adapter._portal_has_wide_clearance(target=target)


def waypoint_requires_final_heading_alignment(
    adapter: Any,
    target: dict[str, Any],
    *,
    distance: float | None = None,
    is_final: bool,
) -> bool:
    if not is_final:
        return False
    if desired_heading(target=target) is None:
        return False
    tolerance = effective_waypoint_tolerance(adapter, target=target, is_final=True)
    if distance is not None and distance > tolerance:
        return False
    return True


def waypoint_heading_tolerance_rad(
    adapter: Any,
    *,
    target: dict[str, Any],
    distance: float | None,
    is_final: bool,
) -> float | None:
    if waypoint_requires_portal_heading_alignment(adapter, target, distance=distance):
        return adapter.portal_alignment_heading_threshold_rad
    if waypoint_requires_final_heading_alignment(
        adapter,
        target,
        distance=distance,
        is_final=is_final,
    ):
        return adapter.final_waypoint_heading_tolerance_rad
    return None


def waypoint_heading_error(adapter: Any, *, target: dict[str, Any], yaw: float) -> float | None:
    resolved_desired_heading = waypoint_desired_heading(target=target)
    if resolved_desired_heading is None:
        return None
    return adapter._wrap_angle(resolved_desired_heading - yaw)


def should_face_final_object_target(
    adapter: Any,
    *,
    target: dict[str, Any],
    distance: float,
    heading_error: float,
    is_final_waypoint: bool,
) -> bool:
    if not is_final_waypoint:
        return False
    if str(target.get("waypoint_type", "")).strip().lower() != "object_approach":
        return False
    heading_tolerance = waypoint_heading_tolerance_rad(
        adapter,
        target=target,
        distance=distance,
        is_final=is_final_waypoint,
    )
    if heading_tolerance is None:
        return False
    waypoint_tolerance = effective_waypoint_tolerance(adapter, target=target, is_final=True)
    if distance > waypoint_tolerance:
        return False
    return abs(float(heading_error)) > heading_tolerance


def final_waypoint_alignment_heading(
    adapter: Any,
    *,
    target: dict[str, Any],
    distance: float,
    is_final: bool,
) -> float | None:
    if not waypoint_requires_final_heading_alignment(
        adapter,
        target,
        distance=distance,
        is_final=is_final,
    ):
        return None
    return desired_heading(target=target)


def heading_progress_is_improving(
    adapter: Any,
    *,
    target: dict[str, Any],
    heading_error: float,
) -> bool:
    current_heading_error = abs(float(heading_error))
    heading_margin: float | None = None
    if waypoint_requires_portal_heading_alignment(adapter, target):
        heading_margin = max(0.01, adapter.portal_alignment_heading_threshold_rad * 0.1)
    elif waypoint_requires_final_heading_alignment(adapter, target, is_final=True):
        heading_margin = max(0.02, adapter.final_waypoint_heading_tolerance_rad * 0.1)
    elif adapter.prefer_forward_facing_motion:
        heading_margin = max(0.02, adapter.forward_facing_heading_threshold_rad * 0.1)
    if heading_margin is None:
        return False

    if adapter._best_heading_error_to_waypoint is None:
        adapter._best_heading_error_to_waypoint = current_heading_error
        return True

    if current_heading_error < adapter._best_heading_error_to_waypoint - heading_margin:
        adapter._best_heading_error_to_waypoint = current_heading_error
        return True
    return False


def align_local_path_heading_to_tracking_target(
    adapter: Any,
    *,
    desired_heading: float,
    tracking_distance: float,
    dx: float,
    dy: float,
    cross_track_error: float,
    guidance_world_x: float,
    guidance_world_y: float,
) -> tuple[float, float, float]:
    if tracking_distance <= 1e-6:
        return desired_heading, guidance_world_x, guidance_world_y
    if cross_track_error > adapter.local_path_rejoin_distance_threshold:
        return desired_heading, guidance_world_x, guidance_world_y

    bearing_heading = math.atan2(dy, dx)
    heading_delta = abs(adapter._wrap_angle(bearing_heading - desired_heading))
    threshold = max(0.35, adapter.local_path_curve_threshold_rad)
    if heading_delta <= threshold:
        return desired_heading, guidance_world_x, guidance_world_y

    return bearing_heading, dx / tracking_distance, dy / tracking_distance
