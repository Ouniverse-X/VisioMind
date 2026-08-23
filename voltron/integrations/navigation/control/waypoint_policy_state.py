"""Configuration and runtime state helpers for the waypoint policy adapter."""

from __future__ import annotations

from typing import Any

import numpy as np


def configure_adapter(
    adapter: Any,
    *,
    max_linear_velocity: float,
    max_angular_velocity: float,
    linear_gain: float,
    angular_gain: float,
    waypoint_tolerance: float,
    final_waypoint_tolerance: float,
    object_approach_final_waypoint_tolerance_m: float,
    rotate_in_place_threshold_rad: float,
    waypoint_progress_margin: float,
    holonomic: bool,
    lookahead_points: int,
    lookahead_decay: float,
    command_smoothing: float,
    local_path_waypoint_tolerance: float,
    local_path_final_waypoint_tolerance: float,
    local_path_tracking_horizon: float,
    local_path_max_tracking_horizon: float,
    local_path_curve_threshold_rad: float,
    local_path_rejoin_distance_threshold: float,
    local_path_rejoin_horizon: float,
    local_path_linear_gain: float,
    local_path_min_cruise_velocity: float,
    local_path_max_linear_velocity: float,
    local_path_cross_track_deadband: float,
    local_path_cross_track_gain: float,
    local_path_max_correction_ratio: float,
    local_path_min_progress_scale: float,
    local_path_transition_min_progress_scale: float,
    local_path_angular_gain_scale: float,
    recovery_enabled: bool,
    recovery_steps: int,
    recovery_linear_velocity: float,
    recovery_lateral_velocity: float,
    recovery_angular_velocity: float,
    portal_alignment_distance_threshold: float,
    portal_alignment_heading_threshold_rad: float,
    final_waypoint_heading_tolerance_rad: float,
    portal_alignment_lateral_deadband: float,
    portal_alignment_footprint_width_m: float,
    portal_alignment_min_lateral_deadband_m: float,
    portal_alignment_wide_clearance_margin_m: float,
    portal_alignment_max_linear_velocity: float,
    portal_alignment_max_lateral_velocity: float,
    portal_prealign_distance_threshold_m: float,
    transition_recovery_distance_threshold: float,
    transition_recovery_linear_velocity: float,
    transition_recovery_lateral_velocity: float,
    transition_recovery_angular_velocity: float,
    portal_alignment_patience_steps: int | None,
    progress_window: int,
    progress_epsilon: float,
    blocked_local_progress_epsilon: float,
    blocked_progress_patience_steps: int | None,
    dead_loop_pose_epsilon: float,
    prefer_forward_facing_motion: bool,
    forward_facing_heading_threshold_rad: float,
    max_recovery_attempts_per_waypoint: int,
    waypoint_patience_steps: int | None,
    oscillation_window: int,
    oscillation_heading_epsilon: float,
    oscillation_flip_threshold: int,
) -> None:
    adapter.max_linear_velocity = float(max_linear_velocity)
    adapter.max_angular_velocity = float(max_angular_velocity)
    adapter.linear_gain = float(linear_gain)
    adapter.angular_gain = float(angular_gain)
    adapter.waypoint_tolerance = float(waypoint_tolerance)
    adapter.final_waypoint_tolerance = float(final_waypoint_tolerance)
    adapter.object_approach_final_waypoint_tolerance_m = float(abs(object_approach_final_waypoint_tolerance_m))
    adapter.rotate_in_place_threshold_rad = float(rotate_in_place_threshold_rad)
    adapter.waypoint_progress_margin = float(waypoint_progress_margin)
    adapter.holonomic = bool(holonomic)
    adapter.lookahead_points = max(0, int(lookahead_points))
    adapter.lookahead_decay = float(np.clip(lookahead_decay, 0.0, 1.0))
    adapter.command_smoothing = float(np.clip(command_smoothing, 0.0, 0.95))
    adapter.local_path_waypoint_tolerance = float(abs(local_path_waypoint_tolerance))
    adapter.local_path_final_waypoint_tolerance = float(abs(local_path_final_waypoint_tolerance))
    adapter.local_path_tracking_horizon = float(abs(local_path_tracking_horizon))
    adapter.local_path_max_tracking_horizon = float(
        max(abs(local_path_tracking_horizon), abs(local_path_max_tracking_horizon))
    )
    adapter.local_path_curve_threshold_rad = float(abs(local_path_curve_threshold_rad))
    adapter.local_path_rejoin_distance_threshold = float(abs(local_path_rejoin_distance_threshold))
    adapter.local_path_rejoin_horizon = float(abs(local_path_rejoin_horizon))
    adapter.local_path_linear_gain = float(abs(local_path_linear_gain))
    adapter.local_path_min_cruise_velocity = float(abs(local_path_min_cruise_velocity))
    adapter.local_path_max_linear_velocity = float(abs(local_path_max_linear_velocity))
    adapter.local_path_cross_track_deadband = float(abs(local_path_cross_track_deadband))
    adapter.local_path_cross_track_gain = float(abs(local_path_cross_track_gain))
    adapter.local_path_max_correction_ratio = float(abs(local_path_max_correction_ratio))
    adapter.local_path_min_progress_scale = float(np.clip(abs(local_path_min_progress_scale), 0.05, 1.0))
    adapter.local_path_transition_min_progress_scale = float(
        np.clip(abs(local_path_transition_min_progress_scale), 0.05, 1.0)
    )
    adapter.local_path_angular_gain_scale = float(np.clip(abs(local_path_angular_gain_scale), 0.0, 1.0))
    adapter.recovery_enabled = bool(recovery_enabled)
    adapter.recovery_steps = max(1, int(recovery_steps))
    adapter.recovery_linear_velocity = float(recovery_linear_velocity)
    adapter.recovery_lateral_velocity = float(recovery_lateral_velocity)
    adapter.recovery_angular_velocity = float(recovery_angular_velocity)
    adapter.portal_alignment_distance_threshold = float(abs(portal_alignment_distance_threshold))
    adapter.portal_alignment_heading_threshold_rad = float(abs(portal_alignment_heading_threshold_rad))
    adapter.final_waypoint_heading_tolerance_rad = float(abs(final_waypoint_heading_tolerance_rad))
    adapter.portal_alignment_lateral_deadband = float(abs(portal_alignment_lateral_deadband))
    adapter.portal_alignment_footprint_width_m = float(abs(portal_alignment_footprint_width_m))
    adapter.portal_alignment_min_lateral_deadband_m = float(abs(portal_alignment_min_lateral_deadband_m))
    adapter.portal_alignment_wide_clearance_margin_m = float(abs(portal_alignment_wide_clearance_margin_m))
    adapter.portal_alignment_max_linear_velocity = float(abs(portal_alignment_max_linear_velocity))
    adapter.portal_alignment_max_lateral_velocity = float(abs(portal_alignment_max_lateral_velocity))
    adapter.portal_prealign_distance_threshold_m = float(abs(portal_prealign_distance_threshold_m))
    adapter.transition_recovery_distance_threshold = float(abs(transition_recovery_distance_threshold))
    adapter.transition_recovery_linear_velocity = float(abs(transition_recovery_linear_velocity))
    adapter.transition_recovery_lateral_velocity = float(abs(transition_recovery_lateral_velocity))
    adapter.transition_recovery_angular_velocity = float(abs(transition_recovery_angular_velocity))
    adapter.portal_alignment_patience_steps = (
        max(2, int(portal_alignment_patience_steps))
        if portal_alignment_patience_steps is not None
        else None
    )
    adapter.progress_window = max(2, int(progress_window))
    adapter.progress_epsilon = float(progress_epsilon)
    adapter.blocked_local_progress_epsilon = float(abs(blocked_local_progress_epsilon))
    adapter.blocked_progress_patience_steps = (
        max(1, int(blocked_progress_patience_steps))
        if blocked_progress_patience_steps is not None
        else None
    )
    adapter.dead_loop_pose_epsilon = float(dead_loop_pose_epsilon)
    adapter.prefer_forward_facing_motion = bool(prefer_forward_facing_motion)
    adapter.forward_facing_heading_threshold_rad = float(abs(forward_facing_heading_threshold_rad))
    adapter.max_recovery_attempts_per_waypoint = max(1, int(max_recovery_attempts_per_waypoint))
    adapter.waypoint_patience_steps = (
        max(2, int(waypoint_patience_steps)) if waypoint_patience_steps is not None else None
    )
    adapter.oscillation_window = max(4, int(oscillation_window))
    adapter.oscillation_heading_epsilon = float(abs(oscillation_heading_epsilon))
    adapter.oscillation_flip_threshold = max(2, int(oscillation_flip_threshold))


def initialize_runtime_state(adapter: Any) -> None:
    adapter._active_waypoint_index = 0
    adapter._waypoint_signature = None
    adapter._distance_history = []
    adapter._pose_history = []
    adapter._heading_error_history = []
    adapter._last_base_command = (0.0, 0.0, 0.0)
    adapter._recovery_steps_remaining = 0
    adapter._recovery_reason = None
    adapter._recovery_profile = None
    adapter._recovery_direction = 1.0
    adapter._recovery_command_override = None
    adapter._recovery_waypoint_index = None
    adapter._recovery_cycles_on_waypoint = 0
    adapter._tracking_waypoint_index = None
    adapter._tracking_waypoint_signature = None
    adapter._best_distance_to_waypoint = None
    adapter._best_heading_error_to_waypoint = None
    adapter._heading_progress_this_step = False
    adapter._steps_since_waypoint_progress = 0
    adapter._path_tracking_mode = None
    adapter._local_path_follow_state = None
    adapter._portal_prealign_lock_signature = None
    adapter._portal_stage_lock_signature = None
    adapter._portal_stage_lock_floor = 0


def reset(adapter: Any, options: dict[str, Any] | None = None) -> dict[str, Any]:
    del options
    initialize_runtime_state(adapter)
    return {"status": "reset"}
