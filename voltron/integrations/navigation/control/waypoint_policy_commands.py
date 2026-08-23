"""Command-generation helpers for the waypoint policy adapter."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def tracking_command(
    adapter: Any,
    *,
    target: dict[str, float],
    current_region: str | None,
    is_final_waypoint: bool,
    tracking_distance: float,
    target_distance: float,
    heading_error: float,
    local_forward: float,
    local_lateral: float,
) -> tuple[tuple[float, float, float], str]:
    control_distance = max(tracking_distance, target_distance)
    if adapter._uses_local_path_tracking():
        return local_path_tracking_command(
            adapter,
            target=target,
            current_region=current_region,
            target_distance=target_distance,
            is_final_waypoint=is_final_waypoint,
            control_distance=control_distance,
            heading_error=heading_error,
            local_forward=local_forward,
            local_lateral=local_lateral,
        )
    angular_velocity = angular_velocity_for_tracking(
        adapter,
        heading_error=heading_error,
        is_final_waypoint=is_final_waypoint,
    )
    if adapter._should_face_final_object_target(
        target=target,
        distance=target_distance,
        heading_error=heading_error,
        is_final_waypoint=is_final_waypoint,
    ):
        return (0.0, 0.0, angular_velocity), "face_target"
    if not adapter.holonomic and abs(heading_error) >= adapter.rotate_in_place_threshold_rad:
        return (0.0, 0.0, angular_velocity), "rotate_in_place"
    if should_use_portal_alignment(
        adapter,
        target=target,
        current_region=current_region,
        distance=target_distance,
    ):
        return portal_alignment_command(
            adapter,
            target=target,
            distance=target_distance,
            heading_error=heading_error,
            local_forward=local_forward,
            local_lateral=local_lateral,
            angular_velocity=angular_velocity,
        )

    if not adapter.holonomic:
        heading_scale = max(0.0, math.cos(heading_error))
        linear_velocity = float(
            np.clip(adapter.linear_gain * control_distance * heading_scale, 0.0, adapter.max_linear_velocity)
        )
        return (linear_velocity, 0.0, angular_velocity), "track_waypoint"
    if adapter.prefer_forward_facing_motion:
        return forward_facing_tracking_command(
            adapter,
            control_distance=control_distance,
            heading_error=heading_error,
            local_forward=local_forward,
            angular_velocity=angular_velocity,
            max_linear_velocity=adapter.max_linear_velocity,
            linear_gain=adapter.linear_gain,
        ), "track_waypoint"

    local_target_norm = math.hypot(local_forward, local_lateral)
    if local_target_norm <= 1e-6:
        return (0.0, 0.0, angular_velocity), "track_waypoint"

    target_speed = holonomic_tracking_speed(
        adapter,
        tracking_distance=control_distance,
        is_final_waypoint=is_final_waypoint,
    )
    velocity_scale = target_speed / local_target_norm
    return (
        float(local_forward * velocity_scale),
        float(local_lateral * velocity_scale),
        angular_velocity,
    ), "track_waypoint"


def local_path_tracking_command(
    adapter: Any,
    *,
    target: dict[str, float],
    current_region: str | None,
    target_distance: float,
    is_final_waypoint: bool,
    control_distance: float,
    heading_error: float,
    local_forward: float,
    local_lateral: float,
) -> tuple[tuple[float, float, float], str]:
    angular_velocity = angular_velocity_for_tracking(
        adapter,
        heading_error=heading_error,
        is_final_waypoint=is_final_waypoint,
    )
    if adapter._should_face_final_object_target(
        target=target,
        distance=target_distance,
        heading_error=heading_error,
        is_final_waypoint=is_final_waypoint,
    ):
        return (0.0, 0.0, angular_velocity), "face_target"
    if adapter._waypoint_requires_portal_heading_alignment(target, distance=target_distance):
        return portal_alignment_command(
            adapter,
            target=target,
            distance=target_distance,
            heading_error=heading_error,
            local_forward=local_forward,
            local_lateral=local_lateral,
            angular_velocity=angular_velocity,
        )
    guidance_norm = math.hypot(local_forward, local_lateral)
    if guidance_norm <= 1e-6:
        return (0.0, 0.0, angular_velocity), "track_waypoint"
    cross_track_error = float(adapter._local_path_info_value("cross_track_error") or 0.0)
    if adapter.prefer_forward_facing_motion:
        heading_scale = max(
            adapter.local_path_min_progress_scale,
            math.cos(np.clip(heading_error, -math.pi / 2.0, math.pi / 2.0)),
        )
        speed_scale = min(
            heading_scale,
            local_path_progress_scale(
                adapter,
                cross_track_error=cross_track_error,
            ),
        )
        if adapter._should_use_transition_recovery(
            target=target,
            current_region=current_region,
            distance=target_distance,
        ):
            speed_scale = max(adapter.local_path_transition_min_progress_scale, speed_scale)
        forward_command = forward_facing_tracking_command(
            adapter,
            control_distance=control_distance,
            heading_error=heading_error,
            local_forward=local_forward,
            angular_velocity=angular_velocity,
            max_linear_velocity=adapter.local_path_max_linear_velocity,
            linear_gain=adapter.local_path_linear_gain * speed_scale,
        )
        lateral_velocity = 0.0
        if adapter.holonomic and cross_track_error > adapter.local_path_cross_track_deadband:
            target_speed = holonomic_tracking_speed(
                adapter,
                tracking_distance=control_distance,
                is_final_waypoint=is_final_waypoint,
            )
            correction_speed = target_speed * speed_scale
            lateral_velocity = float(local_lateral * correction_speed / guidance_norm)
            lateral_velocity = float(
                np.clip(
                    lateral_velocity,
                    -adapter.local_path_max_linear_velocity,
                    adapter.local_path_max_linear_velocity,
                )
            )
        return (forward_command[0], lateral_velocity, forward_command[2]), "track_waypoint"

    target_speed = holonomic_tracking_speed(
        adapter,
        tracking_distance=control_distance,
        is_final_waypoint=is_final_waypoint,
    )
    heading_scale = max(
        adapter.local_path_min_progress_scale,
        math.cos(np.clip(heading_error, -math.pi / 2.0, math.pi / 2.0)),
    )
    speed_scale = min(
        heading_scale,
        local_path_progress_scale(adapter, cross_track_error=cross_track_error),
    )
    if adapter._should_use_transition_recovery(
        target=target,
        current_region=current_region,
        distance=target_distance,
    ):
        speed_scale = max(adapter.local_path_transition_min_progress_scale, speed_scale)
    target_speed *= speed_scale
    velocity_scale = target_speed / guidance_norm
    return (
        float(local_forward * velocity_scale),
        float(local_lateral * velocity_scale),
        angular_velocity,
    ), "track_waypoint"


def angular_velocity_for_tracking(
    adapter: Any,
    *,
    heading_error: float,
    is_final_waypoint: bool,
) -> float:
    angular_gain = adapter.angular_gain
    max_angular_velocity = adapter.max_angular_velocity
    if adapter._uses_local_path_tracking():
        gain_scale = adapter.local_path_angular_gain_scale
        if is_final_waypoint:
            gain_scale = min(1.0, max(gain_scale, 0.4))
        angular_gain *= gain_scale
        max_angular_velocity *= gain_scale
    return float(
        np.clip(
            angular_gain * heading_error,
            -max_angular_velocity,
            max_angular_velocity,
        )
    )


def holonomic_tracking_speed(
    adapter: Any,
    *,
    tracking_distance: float,
    is_final_waypoint: bool,
) -> float:
    if not adapter._uses_local_path_tracking():
        return float(np.clip(adapter.linear_gain * tracking_distance, 0.0, adapter.max_linear_velocity))
    if is_final_waypoint:
        return float(
            np.clip(
                adapter.local_path_linear_gain * tracking_distance,
                0.0,
                adapter.local_path_max_linear_velocity,
            )
        )
    cruise_speed = adapter.local_path_min_cruise_velocity + (adapter.local_path_linear_gain * tracking_distance)
    return float(np.clip(cruise_speed, 0.0, adapter.local_path_max_linear_velocity))


def local_path_progress_scale(adapter: Any, *, cross_track_error: float) -> float:
    if cross_track_error <= adapter.local_path_cross_track_deadband:
        return 1.0
    rejoin_threshold = max(adapter.local_path_rejoin_distance_threshold, 1e-6)
    ratio = min(1.0, cross_track_error / rejoin_threshold)
    return float(1.0 - (1.0 - adapter.local_path_min_progress_scale) * ratio)


def local_path_correction_ratio(adapter: Any, *, signed_cross_track_error: float) -> float:
    correction = -signed_cross_track_error * adapter.local_path_cross_track_gain
    if abs(signed_cross_track_error) <= adapter.local_path_cross_track_deadband:
        return 0.0
    return float(
        np.clip(
            correction,
            -adapter.local_path_max_correction_ratio,
            adapter.local_path_max_correction_ratio,
        )
    )


def should_use_portal_alignment(
    adapter: Any,
    *,
    target: dict[str, float],
    current_region: str | None,
    distance: float,
) -> bool:
    if not adapter._is_portal_like_waypoint(target):
        return False
    if distance > adapter.portal_alignment_distance_threshold:
        return False
    if adapter._portal_has_wide_clearance(target=target):
        return False
    waypoint_region = adapter._normalize_label(target.get("room_name"))
    if waypoint_region is None:
        return True
    return current_region != waypoint_region


def portal_alignment_command(
    adapter: Any,
    *,
    target: dict[str, Any],
    distance: float,
    heading_error: float,
    local_forward: float,
    local_lateral: float,
    angular_velocity: float,
) -> tuple[tuple[float, float, float], str]:
    effective_deadband = adapter._effective_portal_alignment_lateral_deadband(target=target)
    lateral_velocity = 0.0
    if adapter.holonomic:
        lateral_velocity = float(
            np.clip(
                local_lateral * adapter.linear_gain,
                -adapter.portal_alignment_max_lateral_velocity,
                adapter.portal_alignment_max_lateral_velocity,
            )
        )

    centered = abs(local_lateral) <= effective_deadband
    aligned = abs(heading_error) <= adapter.portal_alignment_heading_threshold_rad
    forward_bias = max(0.0, local_forward)
    if not aligned:
        return (0.0, 0.0, angular_velocity), "align_portal"
    if centered and aligned:
        linear_velocity = float(np.clip(adapter.linear_gain * distance, 0.0, adapter.max_linear_velocity))
        return (linear_velocity, lateral_velocity, angular_velocity), "track_waypoint"

    if adapter.holonomic and adapter.prefer_forward_facing_motion and not centered:
        heading_scale = max(0.25, math.cos(np.clip(heading_error, -math.pi / 2.0, math.pi / 2.0)))
        return (0.0, lateral_velocity * heading_scale, angular_velocity), "align_portal"

    lateral_ratio = min(1.0, abs(local_lateral) / max(distance, 1e-6))
    heading_scale = max(0.2, math.cos(np.clip(heading_error, -math.pi / 2.0, math.pi / 2.0)))
    center_scale = max(0.2, 1.0 - lateral_ratio)
    linear_velocity = float(
        np.clip(
            adapter.linear_gain * forward_bias * heading_scale * center_scale,
            0.0,
            adapter.portal_alignment_max_linear_velocity,
        )
    )
    return (linear_velocity, lateral_velocity, angular_velocity), "align_portal"


def forward_facing_tracking_command(
    adapter: Any,
    *,
    control_distance: float,
    heading_error: float,
    local_forward: float,
    angular_velocity: float,
    max_linear_velocity: float,
    linear_gain: float,
) -> tuple[float, float, float]:
    if abs(heading_error) >= adapter.rotate_in_place_threshold_rad:
        return (0.0, 0.0, angular_velocity)
    heading_scale = max(0.0, math.cos(np.clip(heading_error, -math.pi / 2.0, math.pi / 2.0)))
    if abs(heading_error) > adapter.forward_facing_heading_threshold_rad:
        heading_scale = min(heading_scale, 0.35)
    forward_bias = max(0.0, local_forward)
    linear_velocity = float(
        np.clip(
            linear_gain * min(control_distance, forward_bias) * heading_scale,
            0.0,
            max_linear_velocity,
        )
    )
    return (linear_velocity, 0.0, angular_velocity)
