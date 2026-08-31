from __future__ import annotations

from typing import Any

import numpy as np

from . import waypoint_policy_commands
from . import waypoint_policy_execution
from . import waypoint_policy_heading
from . import waypoint_policy_inputs
from . import waypoint_policy_progress
from . import waypoint_policy_recovery
from . import waypoint_policy_runtime
from . import waypoint_policy_state
from . import waypoint_policy_tracking_state
from . import waypoint_policy_targeting
from . import waypoint_policy_waypoints


class WaypointPolicyAdapter:
    def __init__(
        self,
        *,
        max_linear_velocity: float = 0.45,
        max_angular_velocity: float = 0.45,
        linear_gain: float = 0.45,
        angular_gain: float = 0.7,
        waypoint_tolerance: float = 0.9,
        final_waypoint_tolerance: float = 0.9,
        object_approach_final_waypoint_tolerance_m: float = 0.45,
        rotate_in_place_threshold_rad: float = 1.05,
        waypoint_progress_margin: float = 0.35,
        holonomic: bool = True,
        lookahead_points: int = 2,
        lookahead_decay: float = 0.65,
        command_smoothing: float = 0.35,
        local_path_waypoint_tolerance: float = 0.12,
        local_path_final_waypoint_tolerance: float = 0.2,
        local_path_tracking_horizon: float = 0.35,
        local_path_max_tracking_horizon: float = 0.8,
        local_path_curve_threshold_rad: float = 0.35,
        local_path_rejoin_distance_threshold: float = 0.45,
        local_path_rejoin_horizon: float = 0.12,
        local_path_linear_gain: float = 0.75,
        local_path_min_cruise_velocity: float = 0.22,
        local_path_max_linear_velocity: float = 0.65,
        local_path_cross_track_deadband: float = 0.10,
        local_path_cross_track_gain: float = 3.0,
        local_path_max_correction_ratio: float = 1.1,
        local_path_min_progress_scale: float = 0.65,
        local_path_transition_min_progress_scale: float = 0.75,
        local_path_angular_gain_scale: float = 0.4,
        recovery_enabled: bool = True,
        recovery_steps: int = 6,
        recovery_linear_velocity: float = -0.08,
        recovery_lateral_velocity: float = 0.18,
        recovery_angular_velocity: float = 0.45,
        portal_alignment_distance_threshold: float = 1.2,
        portal_alignment_heading_threshold_rad: float = 0.35,
        final_waypoint_heading_tolerance_rad: float = 0.65,
        portal_alignment_lateral_deadband: float = 0.18,
        portal_alignment_footprint_width_m: float = 0.72,
        portal_alignment_min_lateral_deadband_m: float = 0.01,
        portal_alignment_wide_clearance_margin_m: float = 0.4,
        portal_alignment_max_linear_velocity: float = 0.18,
        portal_alignment_max_lateral_velocity: float = 0.2,
        portal_prealign_distance_threshold_m: float = 1.2,
        transition_recovery_distance_threshold: float = 4.0,
        transition_recovery_linear_velocity: float = 0.08,
        transition_recovery_lateral_velocity: float = 0.08,
        transition_recovery_angular_velocity: float = 0.3,
        portal_alignment_patience_steps: int | None = None,
        progress_window: int = 8,
        progress_epsilon: float = 0.05,
        blocked_local_progress_epsilon: float = 0.01,
        blocked_progress_patience_steps: int | None = None,
        dead_loop_pose_epsilon: float = 0.12,
        prefer_forward_facing_motion: bool = False,
        forward_facing_heading_threshold_rad: float = 0.35,
        max_recovery_attempts_per_waypoint: int = 2,
        waypoint_patience_steps: int | None = None,
        oscillation_window: int = 8,
        oscillation_heading_epsilon: float = 0.2,
        oscillation_flip_threshold: int = 4,
    ) -> None:
        waypoint_policy_state.configure_adapter(
            self,
            max_linear_velocity=max_linear_velocity,
            max_angular_velocity=max_angular_velocity,
            linear_gain=linear_gain,
            angular_gain=angular_gain,
            waypoint_tolerance=waypoint_tolerance,
            final_waypoint_tolerance=final_waypoint_tolerance,
            object_approach_final_waypoint_tolerance_m=object_approach_final_waypoint_tolerance_m,
            rotate_in_place_threshold_rad=rotate_in_place_threshold_rad,
            waypoint_progress_margin=waypoint_progress_margin,
            holonomic=holonomic,
            lookahead_points=lookahead_points,
            lookahead_decay=lookahead_decay,
            command_smoothing=command_smoothing,
            local_path_waypoint_tolerance=local_path_waypoint_tolerance,
            local_path_final_waypoint_tolerance=local_path_final_waypoint_tolerance,
            local_path_tracking_horizon=local_path_tracking_horizon,
            local_path_max_tracking_horizon=local_path_max_tracking_horizon,
            local_path_curve_threshold_rad=local_path_curve_threshold_rad,
            local_path_rejoin_distance_threshold=local_path_rejoin_distance_threshold,
            local_path_rejoin_horizon=local_path_rejoin_horizon,
            local_path_linear_gain=local_path_linear_gain,
            local_path_min_cruise_velocity=local_path_min_cruise_velocity,
            local_path_max_linear_velocity=local_path_max_linear_velocity,
            local_path_cross_track_deadband=local_path_cross_track_deadband,
            local_path_cross_track_gain=local_path_cross_track_gain,
            local_path_max_correction_ratio=local_path_max_correction_ratio,
            local_path_min_progress_scale=local_path_min_progress_scale,
            local_path_transition_min_progress_scale=local_path_transition_min_progress_scale,
            local_path_angular_gain_scale=local_path_angular_gain_scale,
            recovery_enabled=recovery_enabled,
            recovery_steps=recovery_steps,
            recovery_linear_velocity=recovery_linear_velocity,
            recovery_lateral_velocity=recovery_lateral_velocity,
            recovery_angular_velocity=recovery_angular_velocity,
            portal_alignment_distance_threshold=portal_alignment_distance_threshold,
            portal_alignment_heading_threshold_rad=portal_alignment_heading_threshold_rad,
            final_waypoint_heading_tolerance_rad=final_waypoint_heading_tolerance_rad,
            portal_alignment_lateral_deadband=portal_alignment_lateral_deadband,
            portal_alignment_footprint_width_m=portal_alignment_footprint_width_m,
            portal_alignment_min_lateral_deadband_m=portal_alignment_min_lateral_deadband_m,
            portal_alignment_wide_clearance_margin_m=portal_alignment_wide_clearance_margin_m,
            portal_alignment_max_linear_velocity=portal_alignment_max_linear_velocity,
            portal_alignment_max_lateral_velocity=portal_alignment_max_lateral_velocity,
            portal_prealign_distance_threshold_m=portal_prealign_distance_threshold_m,
            transition_recovery_distance_threshold=transition_recovery_distance_threshold,
            transition_recovery_linear_velocity=transition_recovery_linear_velocity,
            transition_recovery_lateral_velocity=transition_recovery_lateral_velocity,
            transition_recovery_angular_velocity=transition_recovery_angular_velocity,
            portal_alignment_patience_steps=portal_alignment_patience_steps,
            progress_window=progress_window,
            progress_epsilon=progress_epsilon,
            blocked_local_progress_epsilon=blocked_local_progress_epsilon,
            blocked_progress_patience_steps=blocked_progress_patience_steps,
            dead_loop_pose_epsilon=dead_loop_pose_epsilon,
            prefer_forward_facing_motion=prefer_forward_facing_motion,
            forward_facing_heading_threshold_rad=forward_facing_heading_threshold_rad,
            max_recovery_attempts_per_waypoint=max_recovery_attempts_per_waypoint,
            waypoint_patience_steps=waypoint_patience_steps,
            oscillation_window=oscillation_window,
            oscillation_heading_epsilon=oscillation_heading_epsilon,
            oscillation_flip_threshold=oscillation_flip_threshold,
        )
        waypoint_policy_state.initialize_runtime_state(self)

    def ping(self) -> bool:
        return True

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return waypoint_policy_state.reset(self, options=options)

    def get_modality_config(self) -> dict[str, Any]:
        return {"action": {"modality_keys": ["base"]}}

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return waypoint_policy_execution.get_action(
            self,
            observation=observation,
            options=options,
        )

    def _advance_completed_waypoints(
        self,
        *,
        pose: dict[str, float],
        yaw: float,
        waypoints: list[dict[str, float]],
        start_index: int,
        horizontal_axes: tuple[str, str],
        current_region: str | None,
    ) -> int:
        return waypoint_policy_waypoints.advance_completed_waypoints(
            self,
            pose=pose,
            yaw=yaw,
            waypoints=waypoints,
            start_index=start_index,
            horizontal_axes=horizontal_axes,
            current_region=current_region,
        )

    def _apply_locked_portal_stage_index(
        self,
        *,
        waypoints: list[dict[str, float]],
        active_index: int,
    ) -> int:
        return waypoint_policy_waypoints.apply_locked_portal_stage_index(
            self,
            waypoints=waypoints,
            active_index=active_index,
        )

    @staticmethod
    def _extract_waypoints(*, options: dict[str, Any]) -> list[dict[str, float]]:
        return waypoint_policy_inputs.extract_waypoints(options=options)

    @staticmethod
    def _pending_local_path_transition_goal(*, options: dict[str, Any]) -> dict[str, float] | None:
        return waypoint_policy_inputs.pending_local_path_transition_goal(options=options)

    def _tracking_target(
        self,
        *,
        waypoints: list[dict[str, float]],
        active_index: int,
        axis_x: str,
        axis_y: str,
    ) -> dict[str, float]:
        return waypoint_policy_targeting.tracking_target(
            self,
            waypoints=waypoints,
            active_index=active_index,
            axis_x=axis_x,
            axis_y=axis_y,
        )

    def _local_path_tracking_target(
        self,
        *,
        waypoints: list[dict[str, float]],
        active_index: int,
        axis_x: str,
        axis_y: str,
    ) -> dict[str, float]:
        return waypoint_policy_targeting.local_path_tracking_target(
            self,
            waypoints=waypoints,
            active_index=active_index,
            axis_x=axis_x,
            axis_y=axis_y,
        )

    def _adaptive_local_path_tracking_horizon(
        self,
        *,
        waypoints: list[dict[str, float]],
        active_index: int,
        axis_x: str,
        axis_y: str,
    ) -> float:
        return waypoint_policy_targeting.adaptive_local_path_tracking_horizon(
            self,
            waypoints=waypoints,
            active_index=active_index,
            axis_x=axis_x,
            axis_y=axis_y,
        )

    def _can_blend_tracking_waypoint(
        self,
        current: dict[str, Any],
        candidate: dict[str, Any],
    ) -> bool:
        return waypoint_policy_targeting.can_blend_tracking_waypoint(self, current, candidate)

    @staticmethod
    def _extract_pose(
        *,
        observation: dict[str, Any],
        options: dict[str, Any],
    ) -> dict[str, float] | None:
        return waypoint_policy_inputs.extract_pose(
            observation=observation,
            options=options,
        )

    @staticmethod
    def _extract_yaw(
        *,
        observation: dict[str, Any],
        options: dict[str, Any],
    ) -> float:
        return waypoint_policy_inputs.extract_yaw(
            observation=observation,
            options=options,
        )

    @staticmethod
    def _extract_yaw_with_source(
        *,
        observation: dict[str, Any],
        options: dict[str, Any],
    ) -> tuple[float, str]:
        return waypoint_policy_inputs.extract_yaw_with_source(
            observation=observation,
            options=options,
        )

    @staticmethod
    def _resolve_vertical_axis(
        *,
        observation: dict[str, Any],
        options: dict[str, Any],
    ) -> str:
        return waypoint_policy_inputs.resolve_vertical_axis(
            observation=observation,
            options=options,
        )

    @staticmethod
    def _horizontal_axes(vertical_axis: str) -> tuple[str, str]:
        return waypoint_policy_inputs.horizontal_axes(vertical_axis)

    @staticmethod
    def _extract_nav_feedback(
        *,
        observation: dict[str, Any],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        return waypoint_policy_inputs.extract_nav_feedback(
            observation=observation,
            options=options,
        )

    def _waypoint_reached(
        self,
        *,
        pose: dict[str, float],
        yaw: float,
        target: dict[str, float],
        axis_x: str,
        axis_y: str,
        is_final: bool,
        current_region: str | None,
    ) -> bool:
        return waypoint_policy_waypoints.waypoint_reached(
            self,
            pose=pose,
            yaw=yaw,
            target=target,
            axis_x=axis_x,
            axis_y=axis_y,
            is_final=is_final,
            current_region=current_region,
        )

    def _can_skip_waypoint(
        self,
        current: dict[str, Any],
        candidate: dict[str, Any],
    ) -> bool:
        return waypoint_policy_waypoints.can_skip_waypoint(self, current, candidate)

    @staticmethod
    def _planar_distance(
        *,
        pose: dict[str, float],
        target: dict[str, float],
        axis_x: str,
        axis_y: str,
    ) -> float:
        return waypoint_policy_waypoints.planar_distance(
            pose=pose,
            target=target,
            axis_x=axis_x,
            axis_y=axis_y,
        )

    @staticmethod
    def _portal_midpoint_from_metadata(
        *,
        target: dict[str, Any],
    ) -> dict[str, float] | None:
        return waypoint_policy_waypoints.portal_midpoint_from_metadata(target=target)

    @staticmethod
    def _portal_prealign_signature(target: dict[str, Any]) -> tuple[Any, ...] | None:
        return waypoint_policy_tracking_state.portal_prealign_signature(target)

    def _local_path_portal_prealign_state(
        self,
        *,
        pose: dict[str, float],
        target: dict[str, Any],
        axis_x: str,
        axis_y: str,
        tangent_heading: float,
    ) -> dict[str, Any] | None:
        return waypoint_policy_tracking_state.local_path_portal_prealign_state(
            self,
            pose=pose,
            target=target,
            axis_x=axis_x,
            axis_y=axis_y,
            tangent_heading=tangent_heading,
        )

    def _apply_portal_prealign_guidance(
        self,
        *,
        pose: dict[str, float],
        tracking_target: dict[str, float],
        guidance_world_x: float,
        guidance_world_y: float,
        portal_prealign: dict[str, Any],
        axis_x: str,
        axis_y: str,
    ) -> tuple[dict[str, float], float, float, float]:
        return waypoint_policy_tracking_state.apply_portal_prealign_guidance(
            self,
            pose=pose,
            tracking_target=tracking_target,
            guidance_world_x=guidance_world_x,
            guidance_world_y=guidance_world_y,
            portal_prealign=portal_prealign,
            axis_x=axis_x,
            axis_y=axis_y,
        )

    @staticmethod
    def _normalize_label(value: Any) -> str | None:
        return waypoint_policy_inputs.normalize_label(value)

    def _tracking_state(
        self,
        *,
        pose: dict[str, float],
        yaw: float,
        waypoints: list[dict[str, float]],
        active_index: int,
        axis_x: str,
        axis_y: str,
    ) -> tuple[dict[str, float], dict[str, float], float, float, float, float, float]:
        return waypoint_policy_tracking_state.tracking_state(
            self,
            pose=pose,
            yaw=yaw,
            waypoints=waypoints,
            active_index=active_index,
            axis_x=axis_x,
            axis_y=axis_y,
        )

    def _local_path_tracking_state(
        self,
        *,
        pose: dict[str, float],
        yaw: float,
        waypoints: list[dict[str, float]],
        active_index: int,
        axis_x: str,
        axis_y: str,
    ) -> tuple[dict[str, float], dict[str, float], float, float, float, float, float]:
        return waypoint_policy_tracking_state.local_path_tracking_state(
            self,
            pose=pose,
            yaw=yaw,
            waypoints=waypoints,
            active_index=active_index,
            axis_x=axis_x,
            axis_y=axis_y,
        )

    def _portal_stage_tracking_state(
        self,
        *,
        pose: dict[str, float],
        yaw: float,
        target: dict[str, float],
        axis_x: str,
        axis_y: str,
    ) -> tuple[dict[str, float], dict[str, float], float, float, float, float, float]:
        return waypoint_policy_tracking_state.portal_stage_tracking_state(
            self,
            pose=pose,
            yaw=yaw,
            target=target,
            axis_x=axis_x,
            axis_y=axis_y,
        )

    def _portal_tracking_heading(
        self,
        *,
        target: dict[str, Any],
        tracking_distance: float,
        dx: float,
        dy: float,
    ) -> float | None:
        return waypoint_policy_tracking_state.portal_tracking_heading(
            self,
            target=target,
            tracking_distance=tracking_distance,
            dx=dx,
            dy=dy,
        )

    def _portal_width_from_target(self, *, target: dict[str, Any]) -> float | None:
        return waypoint_policy_tracking_state.portal_width_from_target(self, target=target)

    def _effective_portal_alignment_lateral_deadband(self, *, target: dict[str, Any]) -> float:
        return waypoint_policy_tracking_state.effective_portal_alignment_lateral_deadband(
            self,
            target=target,
        )

    def _portal_has_wide_clearance(self, *, target: dict[str, Any]) -> bool:
        return waypoint_policy_tracking_state.portal_has_wide_clearance(self, target=target)

    @staticmethod
    def _portal_stage_order(target: dict[str, Any]) -> int:
        return waypoint_policy_tracking_state.portal_stage_order(target)

    def _portal_stage_signature(self, target: dict[str, Any]) -> tuple[Any, ...] | None:
        return waypoint_policy_tracking_state.portal_stage_signature(self, target)

    def _update_portal_stage_lock(self, *, target: dict[str, Any]) -> None:
        waypoint_policy_tracking_state.update_portal_stage_lock(self, target=target)

    def _local_path_projection_state(
        self,
        *,
        pose: dict[str, float],
        waypoints: list[dict[str, float]],
        active_index: int,
        axis_x: str,
        axis_y: str,
    ) -> dict[str, Any]:
        return waypoint_policy_tracking_state.local_path_projection_state(
            self,
            pose=pose,
            waypoints=waypoints,
            active_index=active_index,
            axis_x=axis_x,
            axis_y=axis_y,
        )

    def _closest_point_on_local_path(
        self,
        *,
        pose: dict[str, float],
        waypoints: list[dict[str, float]],
        active_index: int,
        axis_x: str,
        axis_y: str,
    ) -> tuple[dict[str, float], int, float]:
        return waypoint_policy_tracking_state.closest_point_on_local_path(
            self,
            pose=pose,
            waypoints=waypoints,
            active_index=active_index,
            axis_x=axis_x,
            axis_y=axis_y,
        )

    def _local_path_segment_tangent(
        self,
        *,
        waypoints: list[dict[str, float]],
        segment_index: int,
        axis_x: str,
        axis_y: str,
    ) -> tuple[float, float]:
        return waypoint_policy_tracking_state.local_path_segment_tangent(
            self,
            waypoints=waypoints,
            segment_index=segment_index,
            axis_x=axis_x,
            axis_y=axis_y,
        )

    def _advance_local_path_point(
        self,
        *,
        waypoints: list[dict[str, float]],
        start_segment_index: int,
        start_point: dict[str, float],
        remaining: float,
        axis_x: str,
        axis_y: str,
    ) -> dict[str, float]:
        return waypoint_policy_tracking_state.advance_local_path_point(
            self,
            waypoints=waypoints,
            start_segment_index=start_segment_index,
            start_point=start_point,
            remaining=remaining,
            axis_x=axis_x,
            axis_y=axis_y,
        )

    def _should_rejoin_before_curve(
        self,
        *,
        waypoints: list[dict[str, float]],
        segment_index: int,
        axis_x: str,
        axis_y: str,
        cross_track_error: float,
    ) -> bool:
        return waypoint_policy_tracking_state.should_rejoin_before_curve(
            self,
            waypoints=waypoints,
            segment_index=segment_index,
            axis_x=axis_x,
            axis_y=axis_y,
            cross_track_error=cross_track_error,
        )

    def _tracking_command(
        self,
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
        return waypoint_policy_commands.tracking_command(
            self,
            target=target,
            current_region=current_region,
            is_final_waypoint=is_final_waypoint,
            tracking_distance=tracking_distance,
            target_distance=target_distance,
            heading_error=heading_error,
            local_forward=local_forward,
            local_lateral=local_lateral,
        )

    def _local_path_tracking_command(
        self,
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
        return waypoint_policy_commands.local_path_tracking_command(
            self,
            target=target,
            current_region=current_region,
            target_distance=target_distance,
            is_final_waypoint=is_final_waypoint,
            control_distance=control_distance,
            heading_error=heading_error,
            local_forward=local_forward,
            local_lateral=local_lateral,
        )

    def _angular_velocity_for_tracking(
        self,
        *,
        heading_error: float,
        is_final_waypoint: bool,
    ) -> float:
        return waypoint_policy_commands.angular_velocity_for_tracking(
            self,
            heading_error=heading_error,
            is_final_waypoint=is_final_waypoint,
        )

    def _holonomic_tracking_speed(
        self,
        *,
        tracking_distance: float,
        is_final_waypoint: bool,
    ) -> float:
        return waypoint_policy_commands.holonomic_tracking_speed(
            self,
            tracking_distance=tracking_distance,
            is_final_waypoint=is_final_waypoint,
        )

    def _local_path_progress_scale(self, *, cross_track_error: float) -> float:
        return waypoint_policy_commands.local_path_progress_scale(
            self,
            cross_track_error=cross_track_error,
        )

    def _local_path_correction_ratio(self, *, signed_cross_track_error: float) -> float:
        return waypoint_policy_commands.local_path_correction_ratio(
            self,
            signed_cross_track_error=signed_cross_track_error,
        )

    def _should_use_portal_alignment(
        self,
        *,
        target: dict[str, float],
        current_region: str | None,
        distance: float,
    ) -> bool:
        return waypoint_policy_commands.should_use_portal_alignment(
            self,
            target=target,
            current_region=current_region,
            distance=distance,
        )

    def _portal_alignment_command(
        self,
        *,
        target: dict[str, Any],
        distance: float,
        heading_error: float,
        local_forward: float,
        local_lateral: float,
        angular_velocity: float,
    ) -> tuple[tuple[float, float, float], str]:
        return waypoint_policy_commands.portal_alignment_command(
            self,
            target=target,
            distance=distance,
            heading_error=heading_error,
            local_forward=local_forward,
            local_lateral=local_lateral,
            angular_velocity=angular_velocity,
        )

    def _forward_facing_tracking_command(
        self,
        *,
        control_distance: float,
        heading_error: float,
        local_forward: float,
        angular_velocity: float,
        max_linear_velocity: float,
        linear_gain: float,
    ) -> tuple[float, float, float]:
        return waypoint_policy_commands.forward_facing_tracking_command(
            self,
            control_distance=control_distance,
            heading_error=heading_error,
            local_forward=local_forward,
            angular_velocity=angular_velocity,
            max_linear_velocity=max_linear_velocity,
            linear_gain=linear_gain,
        )

    def _push_distance(self, distance: float) -> None:
        waypoint_policy_progress.push_distance(self, distance)

    def _push_pose(self, *, pose: dict[str, float], axis_x: str, axis_y: str) -> None:
        waypoint_policy_progress.push_pose(
            self,
            pose=pose,
            axis_x=axis_x,
            axis_y=axis_y,
        )

    def _update_waypoint_progress(
        self,
        *,
        active_index: int,
        target: dict[str, Any],
        distance: float,
        progress_distance: float,
        heading_error: float,
    ) -> None:
        waypoint_policy_progress.update_waypoint_progress(
            self,
            active_index=active_index,
            target=target,
            distance=distance,
            progress_distance=progress_distance,
            heading_error=heading_error,
        )

    @staticmethod
    def _progress_tracking_signature(target: dict[str, Any]) -> tuple[Any, ...]:
        return waypoint_policy_progress.progress_tracking_signature(target)

    def _progress_distance(
        self,
        *,
        target: dict[str, Any],
        pose: dict[str, float],
        waypoints: list[dict[str, float]],
        active_index: int,
        axis_x: str,
        axis_y: str,
        distance: float,
    ) -> float:
        return waypoint_policy_progress.progress_distance(
            self,
            target=target,
            pose=pose,
            waypoints=waypoints,
            active_index=active_index,
            axis_x=axis_x,
            axis_y=axis_y,
            distance=distance,
        )

    def _remaining_local_path_distance(
        self,
        *,
        pose: dict[str, float],
        waypoints: list[dict[str, float]],
        active_index: int,
        axis_x: str,
        axis_y: str,
    ) -> float:
        return waypoint_policy_progress.remaining_local_path_distance(
            self,
            pose=pose,
            waypoints=waypoints,
            active_index=active_index,
            axis_x=axis_x,
            axis_y=axis_y,
        )

    def _progress_reference_label(self, *, target: dict[str, Any]) -> str:
        return waypoint_policy_progress.progress_reference_label(self, target=target)

    def _should_start_recovery(
        self,
        *,
        nav_feedback: dict[str, Any],
        target: dict[str, Any],
        is_final_waypoint: bool,
        distance: float,
        heading_error: float,
    ) -> bool:
        return waypoint_policy_recovery.should_start_recovery(
            self,
            nav_feedback=nav_feedback,
            target=target,
            is_final_waypoint=is_final_waypoint,
            distance=distance,
            heading_error=heading_error,
        )

    def _should_delay_forward_facing_local_path_recovery(
        self,
        *,
        target: dict[str, Any],
        heading_error: float,
    ) -> bool:
        return waypoint_policy_recovery.should_delay_forward_facing_local_path_recovery(
            self,
            target=target,
            heading_error=heading_error,
        )

    def _forward_facing_local_path_recovery_patience(self) -> int:
        return waypoint_policy_recovery.forward_facing_local_path_recovery_patience(self)

    def _enter_recovery(
        self,
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
        waypoint_policy_recovery.enter_recovery(
            self,
            active_index=active_index,
            target=target,
            current_region=current_region,
            distance=distance,
            heading_error=heading_error,
            local_forward=local_forward,
            local_lateral=local_lateral,
            nav_feedback=nav_feedback,
        )

    def _step_recovery_mode(self) -> str | None:
        return waypoint_policy_recovery.step_recovery_mode(self)

    def _recovery_command(self) -> tuple[float, float, float]:
        return waypoint_policy_recovery.recovery_command(self)

    def _should_request_local_path_replan(self) -> bool:
        return waypoint_policy_recovery.should_request_local_path_replan(self)

    def _should_use_transition_recovery(
        self,
        *,
        target: dict[str, float],
        current_region: str | None,
        distance: float,
    ) -> bool:
        return waypoint_policy_recovery.should_use_transition_recovery(
            self,
            target=target,
            current_region=current_region,
            distance=distance,
        )

    def _transition_recovery_command(
        self,
        *,
        heading_error: float,
        local_forward: float,
        local_lateral: float,
    ) -> tuple[float, float, float]:
        return waypoint_policy_recovery.transition_recovery_command(
            self,
            heading_error=heading_error,
            local_forward=local_forward,
            local_lateral=local_lateral,
        )

    def _local_path_recovery_command(
        self,
        *,
        heading_error: float,
        local_forward: float,
        local_lateral: float,
        target: dict[str, float],
        current_region: str | None,
        distance: float,
    ) -> tuple[float, float, float]:
        return waypoint_policy_recovery.local_path_recovery_command(
            self,
            heading_error=heading_error,
            local_forward=local_forward,
            local_lateral=local_lateral,
            target=target,
            current_region=current_region,
            distance=distance,
        )

    def _pose_stalled(self) -> bool:
        return waypoint_policy_recovery.pose_stalled(self)

    def _should_skip_stuck_waypoint(
        self,
        *,
        active_index: int,
        waypoints: list[dict[str, float]],
        current_region: str | None,
        distance: float,
    ) -> bool:
        return waypoint_policy_recovery.should_skip_stuck_waypoint(
            self,
            active_index=active_index,
            waypoints=waypoints,
            current_region=current_region,
            distance=distance,
        )

    def _waypoint_progress_stalled(self) -> bool:
        return waypoint_policy_recovery.waypoint_progress_stalled(self)

    def _waypoint_patience(self) -> int:
        return waypoint_policy_recovery.waypoint_patience(self)

    def _blocked_progress_patience(self) -> int:
        return waypoint_policy_recovery.blocked_progress_patience(self)

    def _portal_alignment_patience(self) -> int:
        return waypoint_policy_recovery.portal_alignment_patience(self)

    def _feedback_reports_blocked(self, nav_feedback: dict[str, Any]) -> bool:
        return waypoint_policy_recovery.feedback_reports_blocked(nav_feedback)

    def _feedback_has_no_progress(self, nav_feedback: dict[str, Any]) -> bool:
        return waypoint_policy_recovery.feedback_has_no_progress(self, nav_feedback)

    def _oscillation_detected(self) -> bool:
        return waypoint_policy_recovery.oscillation_detected(self)

    def _smooth_command(self, command: tuple[float, float, float]) -> tuple[float, float, float]:
        return waypoint_policy_progress.smooth_command(self, command)

    def _clear_progress_state(self) -> None:
        waypoint_policy_progress.clear_progress_state(self)

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        return waypoint_policy_runtime.coerce_float(value)

    def _reset_waypoint_tracking_state(self) -> None:
        waypoint_policy_progress.reset_waypoint_tracking_state(self)

    @staticmethod
    def _resolve_path_tracking_mode(
        *,
        options: dict[str, Any],
        waypoints: list[dict[str, float]],
    ) -> str | None:
        return waypoint_policy_runtime.resolve_path_tracking_mode(
            options=options,
            waypoints=waypoints,
        )

    def _uses_local_path_tracking(self) -> bool:
        return waypoint_policy_runtime.uses_local_path_tracking(self)

    def _local_path_info_value(self, key: str) -> Any:
        return waypoint_policy_runtime.local_path_info_value(self, key)

    def _effective_waypoint_tolerance(
        self,
        *,
        target: dict[str, Any],
        is_final: bool,
    ) -> float:
        return waypoint_policy_heading.effective_waypoint_tolerance(
            self,
            target=target,
            is_final=is_final,
        )

    @staticmethod
    def _portal_desired_heading(*, target: dict[str, Any]) -> float | None:
        return waypoint_policy_heading.portal_desired_heading(target=target)

    @staticmethod
    def _desired_heading(*, target: dict[str, Any]) -> float | None:
        return waypoint_policy_heading.desired_heading(target=target)

    @classmethod
    def _waypoint_desired_heading(cls, *, target: dict[str, Any]) -> float | None:
        return waypoint_policy_heading.waypoint_desired_heading(target=target)

    @staticmethod
    def _is_portal_like_waypoint(target: dict[str, Any]) -> bool:
        return waypoint_policy_heading.is_portal_like_waypoint(target)

    def _waypoint_requires_portal_heading_alignment(
        self,
        target: dict[str, Any],
        *,
        distance: float | None = None,
    ) -> bool:
        return waypoint_policy_heading.waypoint_requires_portal_heading_alignment(
            self,
            target,
            distance=distance,
        )

    def _waypoint_requires_final_heading_alignment(
        self,
        target: dict[str, Any],
        *,
        distance: float | None = None,
        is_final: bool,
    ) -> bool:
        return waypoint_policy_heading.waypoint_requires_final_heading_alignment(
            self,
            target,
            distance=distance,
            is_final=is_final,
        )

    def _waypoint_heading_tolerance_rad(
        self,
        *,
        target: dict[str, Any],
        distance: float | None,
        is_final: bool,
    ) -> float | None:
        return waypoint_policy_heading.waypoint_heading_tolerance_rad(
            self,
            target=target,
            distance=distance,
            is_final=is_final,
        )

    def _waypoint_heading_error(self, *, target: dict[str, Any], yaw: float) -> float | None:
        return waypoint_policy_heading.waypoint_heading_error(self, target=target, yaw=yaw)

    def _should_face_final_object_target(
        self,
        *,
        target: dict[str, Any],
        distance: float,
        heading_error: float,
        is_final_waypoint: bool,
    ) -> bool:
        return waypoint_policy_heading.should_face_final_object_target(
            self,
            target=target,
            distance=distance,
            heading_error=heading_error,
            is_final_waypoint=is_final_waypoint,
        )

    def _final_waypoint_alignment_heading(
        self,
        *,
        target: dict[str, Any],
        distance: float,
        is_final: bool,
    ) -> float | None:
        return waypoint_policy_heading.final_waypoint_alignment_heading(
            self,
            target=target,
            distance=distance,
            is_final=is_final,
        )

    def _heading_progress_is_improving(
        self,
        *,
        target: dict[str, Any],
        heading_error: float,
    ) -> bool:
        return waypoint_policy_heading.heading_progress_is_improving(
            self,
            target=target,
            heading_error=heading_error,
        )

    def _align_local_path_heading_to_tracking_target(
        self,
        *,
        desired_heading: float,
        tracking_distance: float,
        dx: float,
        dy: float,
        cross_track_error: float,
        guidance_world_x: float,
        guidance_world_y: float,
    ) -> tuple[float, float, float]:
        return waypoint_policy_heading.align_local_path_heading_to_tracking_target(
            self,
            desired_heading=desired_heading,
            tracking_distance=tracking_distance,
            dx=dx,
            dy=dy,
            cross_track_error=cross_track_error,
            guidance_world_x=guidance_world_x,
            guidance_world_y=guidance_world_y,
        )

    @staticmethod
    def _normalize_pose_dict(candidate: dict[str, Any]) -> dict[str, float] | None:
        return waypoint_policy_inputs.normalize_pose_dict(candidate)

    @staticmethod
    def _array_to_pose(value: Any) -> dict[str, float] | None:
        return waypoint_policy_inputs.array_to_pose(value)

    @staticmethod
    def _orientation_to_yaw(value: Any) -> float | None:
        return waypoint_policy_inputs.orientation_to_yaw(value)

    @staticmethod
    def _array_to_yaw(value: Any) -> float | None:
        return waypoint_policy_inputs.array_to_yaw(value)

    @staticmethod
    def _quat_to_yaw(x_coord: float, y_coord: float, z_coord: float, w_coord: float) -> float:
        return waypoint_policy_inputs.quat_to_yaw(x_coord, y_coord, z_coord, w_coord)

    @staticmethod
    def _wrap_angle(value: float) -> float:
        return waypoint_policy_runtime.wrap_angle(value)

    @staticmethod
    def _coerce_index(value: Any) -> int | None:
        return waypoint_policy_inputs.coerce_index(value)

    @staticmethod
    def _waypoint_list_signature(
        *,
        waypoints: list[dict[str, float]],
        path_tracking_mode: str | None,
    ) -> tuple[tuple[float, float, float], ...]:
        return waypoint_policy_runtime.waypoint_list_signature(
            waypoints=waypoints,
            path_tracking_mode=path_tracking_mode,
        )

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray | None:
        return waypoint_policy_inputs.to_numpy(value)

    @staticmethod
    def _zero_action() -> dict[str, Any]:
        return waypoint_policy_runtime.zero_action()

    @staticmethod
    def _base_action(
        local_x_velocity: float,
        local_y_velocity: float,
        angular_velocity: float,
    ) -> dict[str, Any]:
        return waypoint_policy_runtime.base_action(
            local_x_velocity,
            local_y_velocity,
            angular_velocity,
        )
