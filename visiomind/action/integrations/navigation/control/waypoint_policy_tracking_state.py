from __future__ import annotations

import math
from typing import Any

import numpy as np


def portal_prealign_signature(target: dict[str, Any]) -> tuple[Any, ...] | None:
    try:
        return (
            str(target.get("source_room_name", "")),
            str(target.get("room_name", "")),
            str(target["portal_normal_axis"]),
            round(float(target["portal_boundary_value"]), 4),
            str(target["portal_span_axis"]),
            round(float(target["portal_span_min"]), 4),
            round(float(target["portal_span_max"]), 4),
        )
    except (KeyError, TypeError, ValueError):
        return None


def local_path_portal_prealign_state(
    adapter: Any,
    *,
    pose: dict[str, float],
    target: dict[str, Any],
    axis_x: str,
    axis_y: str,
    tangent_heading: float,
) -> dict[str, Any] | None:
    if not adapter.prefer_forward_facing_motion:
        return None
    if adapter.portal_prealign_distance_threshold_m <= 1e-6:
        return None
    if str(target.get("waypoint_type", "")).strip().lower() != "local_path":
        adapter._portal_prealign_lock_signature = None
        return None

    portal_heading = adapter._portal_desired_heading(target=target)
    if portal_heading is None:
        adapter._portal_prealign_lock_signature = None
        return None
    midpoint = adapter._portal_midpoint_from_metadata(target=target)
    if midpoint is None:
        adapter._portal_prealign_lock_signature = None
        return None
    signature = portal_prealign_signature(target)
    if signature is None:
        adapter._portal_prealign_lock_signature = None
        return None

    dx = float(midpoint[axis_x]) - float(pose[axis_x])
    dy = float(midpoint[axis_y]) - float(pose[axis_y])
    midpoint_distance = math.hypot(dx, dy)

    portal_forward = math.cos(portal_heading) * dx + math.sin(portal_heading) * dy
    if portal_forward <= 0.0:
        if adapter._portal_prealign_lock_signature == signature:
            adapter._portal_prealign_lock_signature = None
        return None

    lock_active = adapter._portal_prealign_lock_signature == signature
    if midpoint_distance <= adapter.portal_prealign_distance_threshold_m:
        adapter._portal_prealign_lock_signature = signature
        lock_active = True
    if not lock_active:
        return None

    blend = 1.0 - (midpoint_distance / max(adapter.portal_prealign_distance_threshold_m, 1e-6))
    blend = float(np.clip(blend, 0.0, 1.0))
    if lock_active:
        blend = max(blend, 0.35)
    if blend <= 1e-6:
        return None

    heading_delta = adapter._wrap_angle(portal_heading - tangent_heading)
    return {
        "desired_heading": adapter._wrap_angle(tangent_heading + blend * heading_delta),
        "blend": blend,
        "midpoint_distance": midpoint_distance,
        "midpoint": midpoint,
        "lock_active": lock_active,
    }


def apply_portal_prealign_guidance(
    adapter: Any,
    *,
    pose: dict[str, float],
    tracking_target: dict[str, float],
    guidance_world_x: float,
    guidance_world_y: float,
    portal_prealign: dict[str, Any],
    axis_x: str,
    axis_y: str,
) -> tuple[dict[str, float], float, float, float]:
    del adapter
    midpoint = portal_prealign.get("midpoint")
    if not isinstance(midpoint, dict):
        return (
            tracking_target,
            guidance_world_x,
            guidance_world_y,
            math.atan2(guidance_world_y, guidance_world_x),
        )

    blend = float(np.clip(float(portal_prealign.get("blend", 0.0)), 0.0, 1.0))
    if blend <= 1e-6:
        return (
            tracking_target,
            guidance_world_x,
            guidance_world_y,
            math.atan2(guidance_world_y, guidance_world_x),
        )

    adjusted_target = dict(tracking_target)
    adjusted_target[axis_x] = (1.0 - blend) * float(tracking_target[axis_x]) + blend * float(
        midpoint[axis_x]
    )
    adjusted_target[axis_y] = (1.0 - blend) * float(tracking_target[axis_y]) + blend * float(
        midpoint[axis_y]
    )

    approach_dx = float(adjusted_target[axis_x]) - float(pose[axis_x])
    approach_dy = float(adjusted_target[axis_y]) - float(pose[axis_y])
    approach_norm = math.hypot(approach_dx, approach_dy)
    if approach_norm <= 1e-6:
        desired_heading = math.atan2(guidance_world_y, guidance_world_x)
        return adjusted_target, guidance_world_x, guidance_world_y, desired_heading

    approach_x = approach_dx / approach_norm
    approach_y = approach_dy / approach_norm
    blended_guidance_x = (1.0 - blend) * guidance_world_x + blend * approach_x
    blended_guidance_y = (1.0 - blend) * guidance_world_y + blend * approach_y
    blended_guidance_norm = math.hypot(blended_guidance_x, blended_guidance_y)
    if blended_guidance_norm > 1e-6:
        guidance_world_x = blended_guidance_x / blended_guidance_norm
        guidance_world_y = blended_guidance_y / blended_guidance_norm
    desired_heading = math.atan2(guidance_world_y, guidance_world_x)
    return adjusted_target, guidance_world_x, guidance_world_y, desired_heading


def tracking_state(
    adapter: Any,
    *,
    pose: dict[str, float],
    yaw: float,
    waypoints: list[dict[str, float]],
    active_index: int,
    axis_x: str,
    axis_y: str,
) -> tuple[dict[str, float], dict[str, float], float, float, float, float, float]:
    if adapter._uses_local_path_tracking():
        return local_path_tracking_state(
            adapter,
            pose=pose,
            yaw=yaw,
            waypoints=waypoints,
            active_index=active_index,
            axis_x=axis_x,
            axis_y=axis_y,
        )
    adapter._local_path_follow_state = None
    target = waypoints[active_index]
    tracking_target = adapter._tracking_target(
        waypoints=waypoints,
        active_index=active_index,
        axis_x=axis_x,
        axis_y=axis_y,
    )
    dx = float(tracking_target[axis_x]) - float(pose[axis_x])
    dy = float(tracking_target[axis_y]) - float(pose[axis_y])
    tracking_distance = math.hypot(dx, dy)
    distance = adapter._planar_distance(
        pose=pose,
        target=target,
        axis_x=axis_x,
        axis_y=axis_y,
    )
    desired_heading = math.atan2(dy, dx)
    final_heading = adapter._final_waypoint_alignment_heading(
        target=target,
        distance=distance,
        is_final=active_index == len(waypoints) - 1,
    )
    if final_heading is not None:
        return (
            target,
            dict(target),
            0.0,
            distance,
            adapter._wrap_angle(final_heading - yaw),
            0.0,
            0.0,
        )
    heading_error = adapter._wrap_angle(desired_heading - yaw)
    local_forward = math.cos(yaw) * dx + math.sin(yaw) * dy
    local_lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy
    return (
        target,
        tracking_target,
        tracking_distance,
        distance,
        heading_error,
        local_forward,
        local_lateral,
    )


def local_path_tracking_state(
    adapter: Any,
    *,
    pose: dict[str, float],
    yaw: float,
    waypoints: list[dict[str, float]],
    active_index: int,
    axis_x: str,
    axis_y: str,
) -> tuple[dict[str, float], dict[str, float], float, float, float, float, float]:
    target = waypoints[active_index]
    target_distance = adapter._planar_distance(
        pose=pose,
        target=target,
        axis_x=axis_x,
        axis_y=axis_y,
    )
    final_heading = adapter._final_waypoint_alignment_heading(
        target=target,
        distance=target_distance,
        is_final=active_index == len(waypoints) - 1,
    )
    if final_heading is not None:
        return (
            target,
            dict(target),
            0.0,
            target_distance,
            adapter._wrap_angle(final_heading - yaw),
            0.0,
            0.0,
        )
    if adapter._waypoint_requires_portal_heading_alignment(target, distance=target_distance):
        return portal_stage_tracking_state(
            adapter,
            pose=pose,
            yaw=yaw,
            target=target,
            axis_x=axis_x,
            axis_y=axis_y,
        )
    projection_state = local_path_projection_state(
        adapter,
        pose=pose,
        waypoints=waypoints,
        active_index=active_index,
        axis_x=axis_x,
        axis_y=axis_y,
    )
    closest_point = dict(projection_state["closest_point"])
    closest_segment_index = int(projection_state["segment_index"])
    cross_track_error = float(projection_state["cross_track_error"])
    tracking_horizon = adapter._adaptive_local_path_tracking_horizon(
        waypoints=waypoints,
        active_index=max(active_index, closest_segment_index),
        axis_x=axis_x,
        axis_y=axis_y,
    )
    if (
        cross_track_error > adapter.local_path_rejoin_distance_threshold
        or should_rejoin_before_curve(
            adapter,
            waypoints=waypoints,
            segment_index=closest_segment_index,
            axis_x=axis_x,
            axis_y=axis_y,
            cross_track_error=cross_track_error,
        )
    ):
        tracking_horizon = min(tracking_horizon, adapter.local_path_rejoin_horizon)
    tracking_target = advance_local_path_point(
        adapter,
        waypoints=waypoints,
        start_segment_index=closest_segment_index,
        start_point=closest_point,
        remaining=max(0.0, tracking_horizon),
        axis_x=axis_x,
        axis_y=axis_y,
    )
    dx = float(tracking_target[axis_x]) - float(pose[axis_x])
    dy = float(tracking_target[axis_y]) - float(pose[axis_y])
    tracking_distance = math.hypot(dx, dy)
    distance = adapter._planar_distance(
        pose=pose,
        target=target,
        axis_x=axis_x,
        axis_y=axis_y,
    )
    tangent_x = float(projection_state["tangent_x"])
    tangent_y = float(projection_state["tangent_y"])
    normal_x = -tangent_y
    normal_y = tangent_x
    signed_cross_track_error = float(projection_state["signed_cross_track_error"])
    progress_scale = adapter._local_path_progress_scale(cross_track_error=cross_track_error)
    correction_ratio = adapter._local_path_correction_ratio(
        signed_cross_track_error=signed_cross_track_error
    )
    guidance_world_x = tangent_x * progress_scale + normal_x * correction_ratio
    guidance_world_y = tangent_y * progress_scale + normal_y * correction_ratio
    guidance_norm = math.hypot(guidance_world_x, guidance_world_y)
    if guidance_norm <= 1e-6:
        guidance_world_x = tangent_x
        guidance_world_y = tangent_y
        guidance_norm = 1.0
    desired_heading = float(projection_state["tangent_heading"])
    portal_prealign = local_path_portal_prealign_state(
        adapter,
        pose=pose,
        target=target,
        axis_x=axis_x,
        axis_y=axis_y,
        tangent_heading=desired_heading,
    )
    if portal_prealign is not None:
        tracking_target, guidance_world_x, guidance_world_y, desired_heading = (
            apply_portal_prealign_guidance(
                adapter,
                pose=pose,
                tracking_target=tracking_target,
                guidance_world_x=guidance_world_x,
                guidance_world_y=guidance_world_y,
                portal_prealign=portal_prealign,
                axis_x=axis_x,
                axis_y=axis_y,
            )
        )
        dx = float(tracking_target[axis_x]) - float(pose[axis_x])
        dy = float(tracking_target[axis_y]) - float(pose[axis_y])
        tracking_distance = math.hypot(dx, dy)
    elif adapter.prefer_forward_facing_motion:
        desired_heading, guidance_world_x, guidance_world_y = (
            adapter._align_local_path_heading_to_tracking_target(
                desired_heading=desired_heading,
                tracking_distance=tracking_distance,
                dx=dx,
                dy=dy,
                cross_track_error=cross_track_error,
                guidance_world_x=guidance_world_x,
                guidance_world_y=guidance_world_y,
            )
        )
    heading_error = adapter._wrap_angle(desired_heading - yaw)
    local_forward = math.cos(yaw) * guidance_world_x + math.sin(yaw) * guidance_world_y
    local_lateral = -math.sin(yaw) * guidance_world_x + math.cos(yaw) * guidance_world_y
    adapter._local_path_follow_state = {
        "closest_point": closest_point,
        "segment_index": closest_segment_index,
        "cross_track_error": cross_track_error,
        "signed_cross_track_error": signed_cross_track_error,
        "tangent_heading": desired_heading,
        "tangent_x": tangent_x,
        "tangent_y": tangent_y,
        "progress_scale": progress_scale,
        "guidance_norm": guidance_norm,
        "tracking_target": dict(tracking_target),
        "portal_prealign_active": bool(portal_prealign),
    }
    if portal_prealign is not None:
        adapter._local_path_follow_state["portal_prealign_blend"] = float(portal_prealign["blend"])
        adapter._local_path_follow_state["portal_midpoint_distance"] = float(
            portal_prealign["midpoint_distance"]
        )
        adapter._local_path_follow_state["portal_prealign_locked"] = bool(
            portal_prealign.get("lock_active", False)
        )
    return (
        target,
        tracking_target,
        tracking_distance,
        distance,
        heading_error,
        local_forward,
        local_lateral,
    )


def portal_stage_tracking_state(
    adapter: Any,
    *,
    pose: dict[str, float],
    yaw: float,
    target: dict[str, float],
    axis_x: str,
    axis_y: str,
) -> tuple[dict[str, float], dict[str, float], float, float, float, float, float]:
    tracking_target = dict(target)
    dx = float(tracking_target[axis_x]) - float(pose[axis_x])
    dy = float(tracking_target[axis_y]) - float(pose[axis_y])
    tracking_distance = math.hypot(dx, dy)
    bearing_heading = math.atan2(dy, dx)
    desired_heading = portal_tracking_heading(
        adapter,
        target=target,
        tracking_distance=tracking_distance,
        dx=dx,
        dy=dy,
    )
    if desired_heading is None:
        desired_heading = bearing_heading
    heading_error = adapter._wrap_angle(desired_heading - yaw)
    local_forward = math.cos(yaw) * dx + math.sin(yaw) * dy
    local_lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy
    adapter._local_path_follow_state = {
        "closest_point": dict(target),
        "segment_index": 0,
        "cross_track_error": 0.0,
        "signed_cross_track_error": local_lateral,
        "tangent_heading": desired_heading,
        "tangent_x": math.cos(desired_heading),
        "tangent_y": math.sin(desired_heading),
        "progress_scale": 1.0,
        "guidance_norm": tracking_distance,
        "tracking_target": dict(tracking_target),
        "tracking_mode": "portal_alignment_stage",
        "portal_alignment_stage": target.get("portal_alignment_stage"),
    }
    return (
        target,
        tracking_target,
        tracking_distance,
        tracking_distance,
        heading_error,
        local_forward,
        local_lateral,
    )


def portal_tracking_heading(
    adapter: Any,
    *,
    target: dict[str, Any],
    tracking_distance: float,
    dx: float,
    dy: float,
) -> float | None:
    portal_heading = adapter._portal_desired_heading(target=target)
    if portal_heading is None:
        return None

    alignment_stage = str(target.get("portal_alignment_stage", "")).strip().lower()
    if alignment_stage not in {"source_anchor", "midpoint"}:
        return portal_heading
    if not adapter.prefer_forward_facing_motion:
        return portal_heading
    if tracking_distance <= 1e-6:
        return portal_heading

    portal_forward = math.cos(portal_heading) * dx + math.sin(portal_heading) * dy
    portal_lateral = -math.sin(portal_heading) * dx + math.cos(portal_heading) * dy
    effective_deadband = effective_portal_alignment_lateral_deadband(adapter, target=target)
    centered_in_doorway_frame = abs(portal_lateral) <= effective_deadband
    if centered_in_doorway_frame:
        return portal_heading
    if adapter.prefer_forward_facing_motion and adapter.holonomic:
        return portal_heading

    bearing_heading = math.atan2(dy, dx)
    heading_delta = adapter._wrap_angle(bearing_heading - portal_heading)
    correction_limit = min(
        math.pi / 3.0,
        max(0.25, math.atan2(abs(portal_lateral), max(0.05, abs(portal_forward)))),
    )
    heading_delta = float(np.clip(heading_delta, -correction_limit, correction_limit))
    return adapter._wrap_angle(portal_heading + heading_delta)


def portal_width_from_target(adapter: Any, *, target: dict[str, Any]) -> float | None:
    span_min = adapter._coerce_float(target.get("portal_span_min"))
    span_max = adapter._coerce_float(target.get("portal_span_max"))
    if span_min is not None and span_max is not None:
        return abs(span_max - span_min)
    return adapter._coerce_float(target.get("portal_span"))


def effective_portal_alignment_lateral_deadband(adapter: Any, *, target: dict[str, Any]) -> float:
    effective_deadband = adapter.portal_alignment_lateral_deadband
    alignment_stage = str(target.get("portal_alignment_stage", "")).strip().lower()
    if alignment_stage == "source_anchor":
        return effective_deadband
    portal_width = portal_width_from_target(adapter, target=target)
    if portal_width is None or adapter.portal_alignment_footprint_width_m <= 1e-6:
        return effective_deadband
    half_clearance_budget = max(
        0.0, 0.5 * (portal_width - adapter.portal_alignment_footprint_width_m)
    )
    return float(
        min(
            effective_deadband,
            max(adapter.portal_alignment_min_lateral_deadband_m, half_clearance_budget),
        )
    )


def portal_has_wide_clearance(adapter: Any, *, target: dict[str, Any]) -> bool:
    portal_width = portal_width_from_target(adapter, target=target)
    if portal_width is None or adapter.portal_alignment_footprint_width_m <= 1e-6:
        return False
    extra_width = portal_width - adapter.portal_alignment_footprint_width_m
    return extra_width >= adapter.portal_alignment_wide_clearance_margin_m


def portal_stage_order(target: dict[str, Any]) -> int:
    alignment_stage = str(target.get("portal_alignment_stage", "")).strip().lower()
    return {
        "source_anchor": 1,
        "midpoint": 2,
        "target_anchor": 3,
    }.get(alignment_stage, 0)


def portal_stage_signature(adapter: Any, target: dict[str, Any]) -> tuple[Any, ...] | None:
    if not adapter._is_portal_like_waypoint(target):
        return None
    return portal_prealign_signature(target)


def update_portal_stage_lock(adapter: Any, *, target: dict[str, Any]) -> None:
    signature = portal_stage_signature(adapter, target)
    if signature is None:
        return
    stage_order_value = portal_stage_order(target)
    if stage_order_value < 2:
        return
    if signature != adapter._portal_stage_lock_signature:
        adapter._portal_stage_lock_signature = signature
        adapter._portal_stage_lock_floor = stage_order_value
        return
    adapter._portal_stage_lock_floor = max(adapter._portal_stage_lock_floor, stage_order_value)


def local_path_projection_state(
    adapter: Any,
    *,
    pose: dict[str, float],
    waypoints: list[dict[str, float]],
    active_index: int,
    axis_x: str,
    axis_y: str,
) -> dict[str, Any]:
    if not waypoints:
        return {
            "closest_point": {},
            "segment_index": 0,
            "projection": 0.0,
            "cross_track_error": 0.0,
            "signed_cross_track_error": 0.0,
            "tangent_x": 1.0,
            "tangent_y": 0.0,
            "tangent_heading": 0.0,
        }
    if len(waypoints) == 1:
        waypoint = dict(waypoints[0])
        return {
            "closest_point": waypoint,
            "segment_index": 0,
            "projection": 0.0,
            "cross_track_error": adapter._planar_distance(
                pose=pose,
                target=waypoint,
                axis_x=axis_x,
                axis_y=axis_y,
            ),
            "signed_cross_track_error": 0.0,
            "tangent_x": 1.0,
            "tangent_y": 0.0,
            "tangent_heading": 0.0,
        }

    start_index = max(0, min(active_index, len(waypoints) - 1) - 1)
    best_state: dict[str, Any] | None = None
    pose_x = float(pose[axis_x])
    pose_y = float(pose[axis_y])

    for index in range(start_index, len(waypoints) - 1):
        current = waypoints[index]
        nxt = waypoints[index + 1]
        segment_dx = float(nxt[axis_x]) - float(current[axis_x])
        segment_dy = float(nxt[axis_y]) - float(current[axis_y])
        segment_length_sq = segment_dx * segment_dx + segment_dy * segment_dy
        if segment_length_sq <= 1e-8:
            candidate = dict(current)
            projection = 0.0
        else:
            projection = (
                (pose_x - float(current[axis_x])) * segment_dx
                + (pose_y - float(current[axis_y])) * segment_dy
            ) / segment_length_sq
            projection = float(np.clip(projection, 0.0, 1.0))
            candidate = dict(current)
            candidate[axis_x] = float(current[axis_x]) + projection * segment_dx
            candidate[axis_y] = float(current[axis_y]) + projection * segment_dy
            if "z" in current and "z" in nxt:
                candidate["z"] = float(current["z"]) + projection * (
                    float(nxt["z"]) - float(current["z"])
                )

        tangent_x, tangent_y = local_path_segment_tangent(
            adapter,
            waypoints=waypoints,
            segment_index=index,
            axis_x=axis_x,
            axis_y=axis_y,
        )
        if projection >= 0.65 and index + 1 < len(waypoints) - 1:
            next_tangent_x, next_tangent_y = local_path_segment_tangent(
                adapter,
                waypoints=waypoints,
                segment_index=index + 1,
                axis_x=axis_x,
                axis_y=axis_y,
            )
            blend = min(1.0, max(0.0, (projection - 0.65) / 0.35))
            tangent_x = (1.0 - blend) * tangent_x + blend * next_tangent_x
            tangent_y = (1.0 - blend) * tangent_y + blend * next_tangent_y
            tangent_norm = math.hypot(tangent_x, tangent_y)
            if tangent_norm > 1e-6:
                tangent_x /= tangent_norm
                tangent_y /= tangent_norm

        delta_x = pose_x - float(candidate[axis_x])
        delta_y = pose_y - float(candidate[axis_y])
        distance_sq = delta_x * delta_x + delta_y * delta_y
        signed_cross_track_error = tangent_x * delta_y - tangent_y * delta_x
        candidate_state = {
            "closest_point": candidate,
            "segment_index": index,
            "projection": projection,
            "cross_track_error": math.sqrt(max(0.0, distance_sq)),
            "signed_cross_track_error": signed_cross_track_error,
            "tangent_x": tangent_x,
            "tangent_y": tangent_y,
            "tangent_heading": math.atan2(tangent_y, tangent_x),
            "distance_sq": distance_sq,
        }
        if best_state is None or distance_sq < float(best_state["distance_sq"]):
            best_state = candidate_state

    if best_state is None:
        waypoint = dict(waypoints[start_index])
        tangent_x, tangent_y = local_path_segment_tangent(
            adapter,
            waypoints=waypoints,
            segment_index=start_index,
            axis_x=axis_x,
            axis_y=axis_y,
        )
        best_state = {
            "closest_point": waypoint,
            "segment_index": start_index,
            "projection": 0.0,
            "cross_track_error": adapter._planar_distance(
                pose=pose,
                target=waypoint,
                axis_x=axis_x,
                axis_y=axis_y,
            ),
            "signed_cross_track_error": 0.0,
            "tangent_x": tangent_x,
            "tangent_y": tangent_y,
            "tangent_heading": math.atan2(tangent_y, tangent_x),
            "distance_sq": 0.0,
        }
    best_state.pop("distance_sq", None)
    return best_state


def closest_point_on_local_path(
    adapter: Any,
    *,
    pose: dict[str, float],
    waypoints: list[dict[str, float]],
    active_index: int,
    axis_x: str,
    axis_y: str,
) -> tuple[dict[str, float], int, float]:
    state = local_path_projection_state(
        adapter,
        pose=pose,
        waypoints=waypoints,
        active_index=active_index,
        axis_x=axis_x,
        axis_y=axis_y,
    )
    return (
        dict(state["closest_point"]),
        int(state["segment_index"]),
        float(state["cross_track_error"]),
    )


def local_path_segment_tangent(
    adapter: Any,
    *,
    waypoints: list[dict[str, float]],
    segment_index: int,
    axis_x: str,
    axis_y: str,
) -> tuple[float, float]:
    del adapter
    if len(waypoints) < 2:
        return 1.0, 0.0
    search_order = list(range(segment_index, len(waypoints) - 1)) + list(
        range(segment_index - 1, -1, -1)
    )
    for index in search_order:
        current = waypoints[index]
        nxt = waypoints[index + 1]
        dx = float(nxt[axis_x]) - float(current[axis_x])
        dy = float(nxt[axis_y]) - float(current[axis_y])
        norm = math.hypot(dx, dy)
        if norm > 1e-6:
            return dx / norm, dy / norm
    return 1.0, 0.0


def advance_local_path_point(
    adapter: Any,
    *,
    waypoints: list[dict[str, float]],
    start_segment_index: int,
    start_point: dict[str, float],
    remaining: float,
    axis_x: str,
    axis_y: str,
) -> dict[str, float]:
    if not waypoints:
        return dict(start_point)
    if start_segment_index >= len(waypoints) - 1 or remaining <= 1e-6:
        return dict(start_point)

    previous = dict(start_point)
    for index in range(start_segment_index + 1, len(waypoints)):
        candidate = waypoints[index]
        segment = adapter._planar_distance(
            pose=previous,
            target=candidate,
            axis_x=axis_x,
            axis_y=axis_y,
        )
        if segment <= 1e-6:
            previous = dict(candidate)
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
        previous = dict(candidate)
    return dict(previous)


def should_rejoin_before_curve(
    adapter: Any,
    *,
    waypoints: list[dict[str, float]],
    segment_index: int,
    axis_x: str,
    axis_y: str,
    cross_track_error: float,
) -> bool:
    if cross_track_error <= adapter.local_path_cross_track_deadband:
        return False
    if segment_index < 0 or segment_index + 2 >= len(waypoints):
        return False

    tangent_x, tangent_y = local_path_segment_tangent(
        adapter,
        waypoints=waypoints,
        segment_index=segment_index,
        axis_x=axis_x,
        axis_y=axis_y,
    )
    next_tangent_x, next_tangent_y = local_path_segment_tangent(
        adapter,
        waypoints=waypoints,
        segment_index=segment_index + 1,
        axis_x=axis_x,
        axis_y=axis_y,
    )
    curvature = abs(
        adapter._wrap_angle(
            math.atan2(next_tangent_y, next_tangent_x) - math.atan2(tangent_y, tangent_x)
        )
    )
    return curvature >= adapter.local_path_curve_threshold_rad
