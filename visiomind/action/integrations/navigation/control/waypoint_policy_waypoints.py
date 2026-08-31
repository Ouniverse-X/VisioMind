from __future__ import annotations

import math
from typing import Any


def advance_completed_waypoints(
    adapter: Any,
    *,
    pose: dict[str, float],
    yaw: float,
    waypoints: list[dict[str, float]],
    start_index: int,
    horizontal_axes: tuple[str, str],
    current_region: str | None,
) -> int:
    index = start_index
    axis_x, axis_y = horizontal_axes
    index = advance_stale_local_path_index_from_projection(
        adapter,
        pose=pose,
        waypoints=waypoints,
        index=index,
        axis_x=axis_x,
        axis_y=axis_y,
    )
    while index < len(waypoints):
        target = waypoints[index]
        if waypoint_reached(
            adapter,
            pose=pose,
            yaw=yaw,
            target=target,
            axis_x=axis_x,
            axis_y=axis_y,
            is_final=index == len(waypoints) - 1,
            current_region=current_region,
        ):
            index += 1
            continue

        next_index = index + 1
        if next_index < len(waypoints):
            current_distance = planar_distance(
                pose=pose,
                target=target,
                axis_x=axis_x,
                axis_y=axis_y,
            )
            next_distance = planar_distance(
                pose=pose,
                target=waypoints[next_index],
                axis_x=axis_x,
                axis_y=axis_y,
            )
            if can_skip_waypoint(adapter, target, waypoints[next_index]) and (
                next_distance + adapter.waypoint_progress_margin < current_distance
            ):
                index = next_index
                continue
            break
        break
    return index


def advance_stale_local_path_index_from_projection(
    adapter: Any,
    *,
    pose: dict[str, float],
    waypoints: list[dict[str, float]],
    index: int,
    axis_x: str,
    axis_y: str,
) -> int:
    if not adapter._uses_local_path_tracking() or len(waypoints) < 2 or index >= len(waypoints) - 1:
        return index
    try:
        projection_state = adapter._local_path_projection_state(
            pose=pose,
            waypoints=waypoints,
            active_index=index,
            axis_x=axis_x,
            axis_y=axis_y,
        )
        projected_segment = int(projection_state.get("segment_index", index))
        cross_track_error = float(projection_state.get("cross_track_error", 0.0))
    except (TypeError, ValueError, KeyError):
        return index

    projected_index = min(max(projected_segment + 1, 0), len(waypoints) - 1)
    projection_lead = projected_index - index
    if projection_lead <= 0:
        return index
    min_projection_lead = max(2, int(getattr(adapter, "lookahead_points", 2) or 2))
    if projection_lead <= min_projection_lead:
        short_path = len(waypoints) <= min_projection_lead + 1
        skippable_local_points = all(
            str(waypoint.get("waypoint_type", "")).strip().lower()
            in {"local_path", "local_dense_path"}
            for waypoint in waypoints[index : projected_index + 1]
        )
        rejoin_threshold = max(
            float(getattr(adapter, "local_path_rejoin_distance_threshold", 0.0) or 0.0),
            float(getattr(adapter, "local_path_waypoint_tolerance", 0.0) or 0.0) * 2.0,
        )
        if not short_path or not skippable_local_points or cross_track_error > rejoin_threshold:
            return index
    return projected_index


def apply_locked_portal_stage_index(
    adapter: Any,
    *,
    waypoints: list[dict[str, float]],
    active_index: int,
) -> int:
    if (
        active_index >= len(waypoints)
        or adapter._portal_stage_lock_floor <= 0
        or adapter._portal_stage_lock_signature is None
    ):
        return active_index
    target = waypoints[active_index]
    if adapter._portal_stage_signature(target) != adapter._portal_stage_lock_signature:
        return active_index
    if adapter._portal_stage_order(target) >= adapter._portal_stage_lock_floor:
        return active_index
    for index in range(active_index + 1, len(waypoints)):
        candidate = waypoints[index]
        if adapter._portal_stage_signature(candidate) != adapter._portal_stage_lock_signature:
            continue
        if adapter._portal_stage_order(candidate) >= adapter._portal_stage_lock_floor:
            return index
    return active_index


def waypoint_reached(
    adapter: Any,
    *,
    pose: dict[str, float],
    yaw: float,
    target: dict[str, float],
    axis_x: str,
    axis_y: str,
    is_final: bool,
    current_region: str | None,
) -> bool:
    waypoint_type = str(target.get("waypoint_type", "")).strip().lower()
    waypoint_region = adapter._normalize_label(target.get("room_name"))
    tolerance = adapter._effective_waypoint_tolerance(target=target, is_final=is_final)
    planar_gap = planar_distance(pose=pose, target=target, axis_x=axis_x, axis_y=axis_y)
    heading_aligned = True
    heading_tolerance = adapter._waypoint_heading_tolerance_rad(
        target=target,
        distance=planar_gap,
        is_final=is_final,
    )
    if heading_tolerance is not None:
        heading_error = adapter._waypoint_heading_error(target=target, yaw=yaw)
        heading_aligned = heading_error is not None and abs(heading_error) <= heading_tolerance

    if bool(target.get("portal_egress_guard")) and not portal_egress_depth_reached(
        pose=pose,
        target=target,
    ):
        return False

    if adapter._uses_local_path_tracking():
        if portal_midpoint_near_aperture_reached(
            adapter,
            pose=pose,
            target=target,
            waypoint_type=waypoint_type,
            planar_gap=planar_gap,
            tolerance=tolerance,
            heading_aligned=heading_aligned,
            is_final=is_final,
        ):
            return True
        if (
            waypoint_type == "portal_midpoint"
            and str(target.get("portal_alignment_stage", "")).strip().lower() == "midpoint"
        ):
            return False
        if waypoint_type == "portal":
            if portal_target_anchor_position_reached(
                pose=pose,
                target=target,
                planar_gap=planar_gap,
                tolerance=tolerance,
            ):
                return True
            if portal_source_exit_reached(
                adapter,
                pose=pose,
                target=target,
                current_region=current_region,
                planar_gap=planar_gap,
                tolerance=tolerance,
                heading_aligned=heading_aligned,
            ):
                return True
            if waypoint_region:
                return bool(
                    current_region
                    and current_region == waypoint_region
                    and planar_gap <= tolerance
                    and heading_aligned
                )
            return planar_gap <= tolerance and heading_aligned
        return planar_gap <= tolerance and heading_aligned

    if waypoint_type == "portal":
        if portal_target_anchor_position_reached(
            pose=pose,
            target=target,
            planar_gap=planar_gap,
            tolerance=tolerance,
        ):
            return True
        if portal_source_exit_reached(
            adapter,
            pose=pose,
            target=target,
            current_region=current_region,
            planar_gap=planar_gap,
            tolerance=tolerance,
            heading_aligned=heading_aligned,
        ):
            return True
        if waypoint_region:
            return bool(
                current_region
                and current_region == waypoint_region
                and planar_gap <= tolerance
                and heading_aligned
            )
        return planar_gap <= tolerance and heading_aligned

    if planar_gap <= tolerance and heading_aligned:
        return True

    if is_final:
        return False

    return False


def portal_egress_depth_reached(
    *,
    pose: dict[str, float],
    target: dict[str, Any],
) -> bool:
    normal_axis = str(target.get("portal_normal_axis") or "")
    if normal_axis not in {"x", "y", "z"}:
        return False
    try:
        boundary = float(target["portal_boundary_value"])
        normal_sign = 1.0 if float(target.get("portal_normal_sign", 1.0)) >= 0.0 else -1.0
        required_depth = max(
            0.0,
            float(
                target.get(
                    "portal_required_egress_depth_m",
                    target.get("portal_egress_depth_m", 0.0),
                )
            ),
        )
        pose_depth = (float(pose[normal_axis]) - boundary) * normal_sign
    except (KeyError, TypeError, ValueError):
        return False
    return pose_depth >= required_depth


def portal_midpoint_near_aperture_reached(
    adapter: Any,
    *,
    pose: dict[str, float],
    target: dict[str, Any],
    waypoint_type: str,
    planar_gap: float,
    tolerance: float,
    heading_aligned: bool,
    is_final: bool,
) -> bool:
    if waypoint_type != "portal_midpoint":
        return False
    if str(target.get("portal_alignment_stage", "")).strip().lower() != "midpoint":
        return False
    if not heading_aligned:
        return False

    try:
        normal_axis = str(target["portal_normal_axis"])
        span_axis = str(target["portal_span_axis"])
        boundary_value = float(target["portal_boundary_value"])
        normal_sign = float(target.get("portal_normal_sign", 1.0))
        normal_value = float(pose[normal_axis])
        span_value = float(pose[span_axis])
        target_span_value = float(target[span_axis])
        span_min = float(target["portal_span_min"])
        span_max = float(target["portal_span_max"])
    except (KeyError, TypeError, ValueError):
        return planar_gap <= tolerance
    if normal_axis not in {"x", "y"} or span_axis not in {"x", "y"}:
        return planar_gap <= tolerance
    if normal_sign == 0.0:
        normal_sign = 1.0

    effective_deadband = adapter._effective_portal_alignment_lateral_deadband(target=target)
    aperture_acceptance = max(float(tolerance), float(tolerance) + max(0.0, effective_deadband))
    normal_direction = 1.0 if normal_sign >= 0.0 else -1.0
    signed_distance_to_aperture = (boundary_value - normal_value) * normal_direction
    if is_final:
        if signed_distance_to_aperture > 0.0:
            return False
        if span_value < min(span_min, span_max) - effective_deadband:
            return False
        if span_value > max(span_min, span_max) + effective_deadband:
            return False
        return planar_gap <= max(float(tolerance), effective_deadband) and abs(
            span_value - target_span_value
        ) <= max(effective_deadband, float(tolerance))

    if planar_gap <= tolerance:
        return True
    if signed_distance_to_aperture < -float(tolerance):
        return True
    if signed_distance_to_aperture > aperture_acceptance:
        return False

    if span_value < min(span_min, span_max) - effective_deadband:
        return False
    if span_value > max(span_min, span_max) + effective_deadband:
        return False
    return abs(span_value - target_span_value) <= effective_deadband


def can_skip_waypoint(
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
    return adapter._can_blend_tracking_waypoint(current, candidate)


def planar_distance(
    *,
    pose: dict[str, float],
    target: dict[str, float],
    axis_x: str,
    axis_y: str,
) -> float:
    dx = float(target[axis_x]) - float(pose[axis_x])
    dy = float(target[axis_y]) - float(pose[axis_y])
    return math.hypot(dx, dy)


def portal_midpoint_from_metadata(
    *,
    target: dict[str, Any],
) -> dict[str, float] | None:
    try:
        span_axis = str(target["portal_span_axis"])
        normal_axis = str(target["portal_normal_axis"])
        boundary_value = float(target["portal_boundary_value"])
        span_min = float(target["portal_span_min"])
        span_max = float(target["portal_span_max"])
    except (KeyError, TypeError, ValueError):
        return None
    if span_axis not in {"x", "y"} or normal_axis not in {"x", "y"}:
        return None

    midpoint = {"x": 0.0, "y": 0.0}
    midpoint[span_axis] = 0.5 * (span_min + span_max)
    midpoint[normal_axis] = boundary_value
    return midpoint


def portal_source_exit_reached(
    adapter: Any,
    *,
    pose: dict[str, float],
    target: dict[str, Any],
    current_region: str | None,
    planar_gap: float,
    tolerance: float,
    heading_aligned: bool,
) -> bool:
    if not heading_aligned:
        return False

    source_region = adapter._normalize_label(
        target.get("source_room_name") or target.get("source_room_id")
    )
    if not source_region or not current_region or current_region == source_region:
        return False

    acceptance = portal_source_exit_acceptance_radius(adapter, tolerance=tolerance)
    if portal_aperture_distance_reached(pose=pose, target=target, acceptance=acceptance):
        return True
    return planar_gap <= acceptance


def portal_source_exit_acceptance_radius(adapter: Any, *, tolerance: float) -> float:
    footprint_width = float(getattr(adapter, "portal_alignment_footprint_width_m", 0.0) or 0.0)
    return max(float(tolerance), float(tolerance) + 0.5 * max(0.0, footprint_width))


def portal_target_anchor_position_reached(
    *,
    pose: dict[str, float],
    target: dict[str, Any],
    planar_gap: float,
    tolerance: float,
) -> bool:
    if str(target.get("portal_alignment_stage", "")).strip().lower() != "target_anchor":
        return False
    try:
        normal_axis = str(target["portal_normal_axis"])
        span_axis = str(target["portal_span_axis"])
        boundary_value = float(target["portal_boundary_value"])
        normal_sign = float(target.get("portal_normal_sign", 1.0))
        normal_value = float(pose[normal_axis])
        target_normal_value = float(target[normal_axis])
        span_value = float(pose[span_axis])
        span_min = float(target["portal_span_min"])
        span_max = float(target["portal_span_max"])
    except (KeyError, TypeError, ValueError):
        return planar_gap <= tolerance
    if normal_axis not in {"x", "y"} or span_axis not in {"x", "y"}:
        return planar_gap <= tolerance
    if normal_sign == 0.0:
        normal_sign = 1.0
    target_offset = max(0.0, (target_normal_value - boundary_value) * normal_sign)
    target_side_depth = (normal_value - boundary_value) * normal_sign
    required_depth = min(0.22, max(0.16, target_offset * 0.55))
    span_margin = max(tolerance, 0.0)
    span_in_range = (
        min(span_min, span_max) - span_margin <= span_value <= max(span_min, span_max) + span_margin
    )
    target_span_value = float(target[span_axis])
    span_centered = abs(span_value - target_span_value) <= max(float(tolerance), 0.18)
    if target_side_depth >= required_depth and span_in_range and span_centered:
        return True
    if planar_gap > tolerance:
        return False
    if (normal_value - boundary_value) * normal_sign < -tolerance:
        return False
    normal_anchor_tolerance = min(float(tolerance), max(0.08, float(tolerance) * 0.5))
    if abs(normal_value - target_normal_value) > normal_anchor_tolerance:
        return False
    if span_value < min(span_min, span_max) - span_margin:
        return False
    if span_value > max(span_min, span_max) + span_margin:
        return False
    return True


def portal_aperture_distance_reached(
    *,
    pose: dict[str, float],
    target: dict[str, Any],
    acceptance: float,
) -> bool:
    try:
        normal_axis = str(target["portal_normal_axis"])
        span_axis = str(target["portal_span_axis"])
        boundary_value = float(target["portal_boundary_value"])
        span_min = float(target["portal_span_min"])
        span_max = float(target["portal_span_max"])
        normal_value = float(pose[normal_axis])
        span_value = float(pose[span_axis])
    except (KeyError, TypeError, ValueError):
        return False
    if normal_axis not in {"x", "y"} or span_axis not in {"x", "y"}:
        return False

    span_margin = max(acceptance, 0.0)
    if span_value < min(span_min, span_max) - span_margin:
        return False
    if span_value > max(span_min, span_max) + span_margin:
        return False
    return abs(normal_value - boundary_value) <= acceptance
