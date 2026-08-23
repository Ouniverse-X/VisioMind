"""Recovery helpers for the waypoint policy adapter."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def should_start_recovery(
    adapter: Any,
    *,
    nav_feedback: dict[str, Any],
    target: dict[str, Any],
    is_final_waypoint: bool,
    distance: float,
    heading_error: float,
) -> bool:
    if not adapter.recovery_enabled or adapter._recovery_steps_remaining > 0:
        return False
    if adapter._recovery_cycles_on_waypoint >= adapter.max_recovery_attempts_per_waypoint:
        return False
    if distance <= adapter._effective_waypoint_tolerance(
        target=target,
        is_final=is_final_waypoint,
    ):
        return False
    if final_object_approach_near_handoff(
        adapter,
        target=target,
        is_final_waypoint=is_final_waypoint,
        distance=distance,
    ):
        return False
    if final_object_approach_is_closing(
        adapter,
        target=target,
        is_final_waypoint=is_final_waypoint,
    ):
        return False
    if should_delay_forward_facing_local_path_recovery(
        adapter,
        target=target,
        heading_error=heading_error,
    ):
        return False
    if feedback_reports_blocked(nav_feedback) and feedback_has_no_progress(adapter, nav_feedback):
        return True
    if adapter._waypoint_requires_portal_heading_alignment(target, distance=distance):
        if adapter._steps_since_waypoint_progress < portal_alignment_patience(adapter):
            return False
    if adapter._heading_progress_this_step:
        return False
    if waypoint_progress_stalled(adapter) and pose_stalled(adapter):
        return True
    if oscillation_detected(adapter) and pose_stalled(adapter):
        return True
    if len(adapter._distance_history) < adapter.progress_window:
        return False
    improvement = max(adapter._distance_history) - min(adapter._distance_history)
    return improvement < adapter.progress_epsilon and pose_stalled(adapter)


def final_object_approach_near_handoff(
    adapter: Any,
    *,
    target: dict[str, Any],
    is_final_waypoint: bool,
    distance: float,
) -> bool:
    if not is_final_waypoint:
        return False
    if str(target.get("waypoint_type", "")).strip().lower() != "object_approach":
        return False
    tolerance = adapter._effective_waypoint_tolerance(target=target, is_final=True)
    settle_band = max(
        adapter.dead_loop_pose_epsilon,
        adapter.progress_epsilon * 3.0,
        0.05,
    )
    return float(distance) <= tolerance + settle_band


def final_object_approach_is_closing(
    adapter: Any,
    *,
    target: dict[str, Any],
    is_final_waypoint: bool,
) -> bool:
    if not is_final_waypoint:
        return False
    if str(target.get("waypoint_type", "")).strip().lower() != "object_approach":
        return False
    if len(adapter._distance_history) < 2:
        return False
    closing_distance = float(adapter._distance_history[0]) - float(adapter._distance_history[-1])
    closing_epsilon = max(0.003, min(0.01, adapter.progress_epsilon * 0.1))
    return closing_distance >= closing_epsilon


def should_delay_forward_facing_local_path_recovery(
    adapter: Any,
    *,
    target: dict[str, Any],
    heading_error: float,
) -> bool:
    if not adapter.prefer_forward_facing_motion or not adapter._uses_local_path_tracking():
        return False
    if adapter._waypoint_requires_portal_heading_alignment(target):
        return False
    if waypoint_progress_stalled(adapter) and pose_stalled(adapter):
        return False
    if abs(float(heading_error)) > max(adapter.forward_facing_heading_threshold_rad, 0.45):
        return True
    return adapter._steps_since_waypoint_progress < forward_facing_local_path_recovery_patience(adapter)


def forward_facing_local_path_recovery_patience(adapter: Any) -> int:
    return max(12, blocked_progress_patience(adapter) * 3)


def enter_recovery(
    adapter: Any,
    *,
    active_index: int,
    target: dict[str, float],
    current_region: str | None,
    distance: float,
    heading_error: float,
    local_forward: float,
    local_lateral: float,
    nav_feedback: dict[str, Any],
) -> None:
    adapter._recovery_steps_remaining = adapter.recovery_steps
    adapter._recovery_direction = -1.0 if local_lateral < 0.0 else 1.0
    adapter._recovery_profile = "reverse_escape"
    adapter._recovery_command_override = None
    if feedback_reports_blocked(nav_feedback) and feedback_has_no_progress(adapter, nav_feedback):
        adapter._recovery_reason = "feedback_blocked"
    elif oscillation_detected(adapter):
        adapter._recovery_reason = "oscillation_detected"
    else:
        adapter._recovery_reason = "progress_stalled"
    if adapter._uses_local_path_tracking():
        adapter._recovery_profile = "corridor_rejoin"
        adapter._recovery_command_override = local_path_recovery_command(
            adapter,
            heading_error=heading_error,
            local_forward=local_forward,
            local_lateral=local_lateral,
            target=target,
            current_region=current_region,
            distance=distance,
        )
    if should_use_transition_recovery(
        adapter,
        target=target,
        current_region=current_region,
        distance=distance,
    ):
        adapter._recovery_profile = "forward_probe"
        adapter._recovery_command_override = transition_recovery_command(
            adapter,
            heading_error=heading_error,
            local_forward=local_forward,
            local_lateral=local_lateral,
        )
    adapter._distance_history = []
    adapter._pose_history = []
    if adapter._recovery_waypoint_index == active_index:
        adapter._recovery_cycles_on_waypoint += 1
    else:
        adapter._recovery_waypoint_index = active_index
        adapter._recovery_cycles_on_waypoint = 1


def step_recovery_mode(adapter: Any) -> str | None:
    if adapter._recovery_steps_remaining <= 0:
        adapter._recovery_reason = None
        adapter._recovery_profile = None
        adapter._recovery_command_override = None
        return None
    adapter._recovery_steps_remaining -= 1
    return adapter._recovery_reason


def recovery_command(adapter: Any) -> tuple[float, float, float]:
    if adapter._recovery_command_override is not None:
        return adapter._recovery_command_override
    return (
        adapter.recovery_linear_velocity,
        adapter._recovery_direction * adapter.recovery_lateral_velocity if adapter.holonomic else 0.0,
        adapter._recovery_direction * adapter.recovery_angular_velocity,
    )


def should_request_local_path_replan(adapter: Any) -> bool:
    """Stop repeating recovery against one stale local path segment."""

    if not adapter._uses_local_path_tracking():
        return False
    if adapter._recovery_steps_remaining > 0:
        return False
    if adapter._recovery_cycles_on_waypoint < adapter.max_recovery_attempts_per_waypoint:
        return False
    return bool(waypoint_progress_stalled(adapter) or oscillation_detected(adapter))


def should_use_transition_recovery(
    adapter: Any,
    *,
    target: dict[str, float],
    current_region: str | None,
    distance: float,
) -> bool:
    if distance > adapter.transition_recovery_distance_threshold:
        return False
    waypoint_type = str(target.get("waypoint_type", "")).strip().lower()
    waypoint_region = adapter._normalize_label(target.get("room_name"))
    if waypoint_type == "portal":
        return True
    if waypoint_region is None:
        return False
    return current_region != waypoint_region


def transition_recovery_command(
    adapter: Any,
    *,
    heading_error: float,
    local_forward: float,
    local_lateral: float,
) -> tuple[float, float, float]:
    heading_scale = max(0.4, math.cos(np.clip(heading_error, -math.pi / 2.0, math.pi / 2.0)))
    if local_forward < 0.0:
        heading_scale = min(heading_scale, 0.6)
    linear_velocity = adapter.transition_recovery_linear_velocity * heading_scale
    lateral_velocity = 0.0
    if adapter.holonomic and not adapter.prefer_forward_facing_motion and abs(local_lateral) >= 0.05:
        lateral_scale = max(0.5, min(1.0, abs(local_lateral)))
        lateral_velocity = math.copysign(
            adapter.transition_recovery_lateral_velocity * lateral_scale,
            local_lateral,
        )
    angular_velocity = float(
        np.clip(
            adapter.angular_gain * heading_error,
            -adapter.transition_recovery_angular_velocity,
            adapter.transition_recovery_angular_velocity,
        )
    )
    return (float(linear_velocity), float(lateral_velocity), angular_velocity)


def local_path_recovery_command(
    adapter: Any,
    *,
    heading_error: float,
    local_forward: float,
    local_lateral: float,
    target: dict[str, float],
    current_region: str | None,
    distance: float,
) -> tuple[float, float, float]:
    guidance_norm = math.hypot(local_forward, local_lateral)
    if guidance_norm <= 1e-6:
        angular_velocity = float(
            np.clip(
                adapter.angular_gain * heading_error,
                -adapter.transition_recovery_angular_velocity,
                adapter.transition_recovery_angular_velocity,
            )
        )
        return (0.0, 0.0, angular_velocity)

    transition_recovery = should_use_transition_recovery(
        adapter,
        target=target,
        current_region=current_region,
        distance=distance,
    )
    target_speed = adapter.transition_recovery_linear_velocity
    if not transition_recovery:
        target_speed = min(target_speed, max(0.05, 0.75 * target_speed))
    velocity_scale = target_speed / guidance_norm
    forward_velocity = max(0.0, float(local_forward * velocity_scale))
    lateral_velocity = 0.0
    if adapter.holonomic:
        lateral_velocity = float(local_lateral * velocity_scale)
        lateral_velocity = float(
            np.clip(
                lateral_velocity,
                -adapter.transition_recovery_lateral_velocity,
                adapter.transition_recovery_lateral_velocity,
            )
        )
    angular_velocity = float(
        np.clip(
            adapter.angular_gain * heading_error,
            -adapter.transition_recovery_angular_velocity,
            adapter.transition_recovery_angular_velocity,
        )
    )
    return (forward_velocity, lateral_velocity, angular_velocity)


def pose_stalled(adapter: Any) -> bool:
    if len(adapter._pose_history) < adapter.progress_window:
        return False
    xs = [point[0] for point in adapter._pose_history]
    ys = [point[1] for point in adapter._pose_history]
    displacement = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    pose_epsilon = adapter.dead_loop_pose_epsilon
    if adapter._uses_local_path_tracking():
        pose_epsilon = min(
            pose_epsilon,
            max(0.01, adapter.local_path_waypoint_tolerance * 0.25),
        )
    return displacement < pose_epsilon


def should_skip_stuck_waypoint(
    adapter: Any,
    *,
    active_index: int,
    waypoints: list[dict[str, float]],
    current_region: str | None,
    distance: float,
) -> bool:
    if adapter._uses_local_path_tracking():
        return should_skip_stuck_local_path_waypoint(
            adapter,
            active_index=active_index,
            waypoints=waypoints,
            distance=distance,
        )
    if active_index >= len(waypoints) - 1:
        return False
    if str(waypoints[active_index].get("waypoint_type", "")).strip().lower() == "portal":
        return False
    if adapter._recovery_cycles_on_waypoint < adapter.max_recovery_attempts_per_waypoint:
        return False
    if not (pose_stalled(adapter) or waypoint_progress_stalled(adapter) or oscillation_detected(adapter)):
        return False
    if distance <= max(adapter.final_waypoint_tolerance, adapter.waypoint_tolerance):
        return False
    waypoint_region = adapter._normalize_label(waypoints[active_index].get("room_name"))
    if current_region and waypoint_region and current_region == waypoint_region:
        return False
    return True


def should_skip_stuck_local_path_waypoint(
    adapter: Any,
    *,
    active_index: int,
    waypoints: list[dict[str, float]],
    distance: float,
) -> bool:
    if active_index >= len(waypoints) - 1:
        return False
    waypoint_type = str(waypoints[active_index].get("waypoint_type", "")).strip().lower()
    if waypoint_type not in {"local_path", "local_dense_path"}:
        return False
    if adapter._recovery_cycles_on_waypoint < adapter.max_recovery_attempts_per_waypoint:
        return False
    if not (pose_stalled(adapter) or waypoint_progress_stalled(adapter) or oscillation_detected(adapter)):
        return False
    follow_state = adapter._local_path_follow_state if isinstance(adapter._local_path_follow_state, dict) else {}
    try:
        cross_track_error = abs(float(follow_state.get("cross_track_error", 0.0)))
    except (TypeError, ValueError):
        return False
    near_path_threshold = max(
        adapter.local_path_rejoin_distance_threshold,
        adapter.local_path_waypoint_tolerance * 2.0,
    )
    if cross_track_error > near_path_threshold:
        return False
    if distance > near_path_threshold:
        return False
    return True


def waypoint_progress_stalled(adapter: Any) -> bool:
    if adapter._best_distance_to_waypoint is None:
        return False
    return adapter._steps_since_waypoint_progress >= waypoint_patience(adapter)


def waypoint_patience(adapter: Any) -> int:
    if adapter.waypoint_patience_steps is not None:
        return adapter.waypoint_patience_steps
    return max(adapter.progress_window * 2, 12)


def blocked_progress_patience(adapter: Any) -> int:
    if adapter.blocked_progress_patience_steps is not None:
        return adapter.blocked_progress_patience_steps
    return max(2, adapter.progress_window // 2)


def portal_alignment_patience(adapter: Any) -> int:
    if adapter.portal_alignment_patience_steps is not None:
        return adapter.portal_alignment_patience_steps
    return max(120, waypoint_patience(adapter) * 6)


def feedback_reports_blocked(nav_feedback: dict[str, Any]) -> bool:
    if bool(nav_feedback.get("collision")) or bool(nav_feedback.get("blocked")):
        return True
    reachable = nav_feedback.get("reachable")
    return reachable is False


def feedback_has_no_progress(adapter: Any, nav_feedback: dict[str, Any]) -> bool:
    if adapter._steps_since_waypoint_progress < blocked_progress_patience(adapter):
        return False
    local_progress = adapter._coerce_float(nav_feedback.get("local_progress"))
    if local_progress is not None and local_progress <= adapter.blocked_local_progress_epsilon:
        return True
    if waypoint_progress_stalled(adapter) or pose_stalled(adapter):
        return True
    if len(adapter._distance_history) < adapter.progress_window:
        return False
    improvement = max(adapter._distance_history) - min(adapter._distance_history)
    return improvement < adapter.progress_epsilon


def oscillation_detected(adapter: Any) -> bool:
    if adapter._steps_since_waypoint_progress < max(2, waypoint_patience(adapter) // 2):
        return False
    if len(adapter._heading_error_history) < 4:
        return False
    signs = [
        1 if error > adapter.oscillation_heading_epsilon else -1
        for error in adapter._heading_error_history
        if abs(error) >= adapter.oscillation_heading_epsilon
    ]
    if len(signs) < 4:
        return False
    flips = sum(1 for previous, current in zip(signs, signs[1:]) if previous != current)
    return flips >= adapter.oscillation_flip_threshold
