from __future__ import annotations

from typing import Any

_TRANSITION_TARGET_SIDE_MARGIN_M = 0.05


def get_action(
    adapter: Any,
    *,
    observation: dict[str, Any],
    options: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    options = dict(options or {})
    waypoints = adapter._extract_waypoints(options=options)
    if not waypoints:
        adapter._clear_progress_state()
        return adapter._zero_action(), {
            "backend": "waypoint",
            "controller_mode": "idle",
            "active_waypoint_index": 0,
            "goal_reached": False,
            "reason": "nav_waypoints_missing",
        }

    adapter._path_tracking_mode = adapter._resolve_path_tracking_mode(
        options=options, waypoints=waypoints
    )

    signature = adapter._waypoint_list_signature(
        waypoints=waypoints,
        path_tracking_mode=adapter._path_tracking_mode,
    )
    requested_index = adapter._coerce_index(options.get("active_waypoint_index"))
    if signature != adapter._waypoint_signature:
        adapter._waypoint_signature = signature
        adapter._active_waypoint_index = requested_index if requested_index is not None else 0
    elif requested_index is not None:
        adapter._active_waypoint_index = requested_index

    pose = adapter._extract_pose(observation=observation, options=options)
    if pose is None:
        adapter._clear_progress_state()
        return adapter._zero_action(), {
            "backend": "waypoint",
            "controller_mode": "idle",
            "active_waypoint_index": adapter._active_waypoint_index,
            "goal_reached": False,
            "reason": "pose_missing",
        }

    yaw, yaw_source = adapter._extract_yaw_with_source(observation=observation, options=options)
    vertical_axis = adapter._resolve_vertical_axis(observation=observation, options=options)
    horizontal_axes = adapter._horizontal_axes(vertical_axis)
    current_region = adapter._normalize_label(
        observation.get("current_region")
        or observation.get("current_room")
        or observation.get("region")
        or options.get("current_region")
        or options.get("current_room")
        or options.get("region")
    )
    nav_feedback = adapter._extract_nav_feedback(observation=observation, options=options)
    previous_active_index = adapter._active_waypoint_index
    active_index = adapter._advance_completed_waypoints(
        pose=pose,
        yaw=yaw,
        waypoints=waypoints,
        start_index=max(0, adapter._active_waypoint_index),
        horizontal_axes=horizontal_axes,
        current_region=current_region,
    )
    active_index = adapter._apply_locked_portal_stage_index(
        waypoints=waypoints,
        active_index=active_index,
    )
    pending_transition_goal = adapter._pending_local_path_transition_goal(options=options)
    if (
        pending_transition_goal is not None
        and adapter._uses_local_path_tracking()
        and 0 <= active_index < len(waypoints)
        and _should_discard_stale_pre_transition_path(
            pose=pose,
            current_region=current_region,
            pending_transition_goal=pending_transition_goal,
            nav_plan=options.get("nav_plan"),
            normalize_label=adapter._normalize_label,
        )
    ):
        waypoints = [pending_transition_goal]
        active_index = 0
    if (
        pending_transition_goal is not None
        and adapter._uses_local_path_tracking()
        and 0 <= active_index < len(waypoints)
    ):
        target = waypoints[active_index]
        waypoint_type = str(target.get("waypoint_type", "")).strip().lower()
        alignment_stage = str(target.get("portal_alignment_stage", "")).strip()
        distance = adapter._planar_distance(
            pose=pose,
            target=target,
            axis_x=horizontal_axes[0],
            axis_y=horizontal_axes[1],
        )
        tolerance = adapter._effective_waypoint_tolerance(target=target, is_final=True)
        if waypoint_type == "portal" and not alignment_stage and distance <= tolerance:
            active_index += 1
    if active_index >= len(waypoints):
        if pending_transition_goal is not None and not adapter._waypoint_reached(
            pose=pose,
            yaw=yaw,
            target=pending_transition_goal,
            axis_x=horizontal_axes[0],
            axis_y=horizontal_axes[1],
            is_final=True,
            current_region=current_region,
        ):
            waypoints = [*waypoints, pending_transition_goal]
            active_index = len(waypoints) - 1
    adapter._active_waypoint_index = active_index
    if active_index != previous_active_index:
        adapter._reset_waypoint_tracking_state()

    if active_index >= len(waypoints):
        adapter._clear_progress_state()
        return adapter._zero_action(), {
            "backend": "waypoint",
            "controller_mode": "goal_reached",
            "active_waypoint_index": active_index,
            "goal_reached": True,
            "distance_to_waypoint": 0.0,
            "heading_error": 0.0,
            "vertical_axis": vertical_axis,
        }

    axis_x, axis_y = horizontal_axes
    (
        target,
        tracking_target,
        tracking_distance,
        distance,
        heading_error,
        local_forward,
        local_lateral,
    ) = adapter._tracking_state(
        pose=pose,
        yaw=yaw,
        waypoints=waypoints,
        active_index=active_index,
        axis_x=axis_x,
        axis_y=axis_y,
    )
    adapter._update_portal_stage_lock(target=target)

    progress_distance = adapter._progress_distance(
        target=target,
        pose=pose,
        waypoints=waypoints,
        active_index=active_index,
        axis_x=axis_x,
        axis_y=axis_y,
        distance=distance,
    )
    adapter._push_distance(progress_distance)
    adapter._push_pose(pose=pose, axis_x=axis_x, axis_y=axis_y)
    adapter._update_waypoint_progress(
        active_index=active_index,
        target=target,
        distance=distance,
        progress_distance=progress_distance,
        heading_error=heading_error,
    )
    loop_detected = adapter._should_skip_stuck_waypoint(
        active_index=active_index,
        waypoints=waypoints,
        current_region=current_region,
        distance=distance,
    )
    if loop_detected:
        active_index = min(active_index + 1, len(waypoints) - 1)
        adapter._active_waypoint_index = active_index
        adapter._reset_waypoint_tracking_state()
        (
            target,
            tracking_target,
            tracking_distance,
            distance,
            heading_error,
            local_forward,
            local_lateral,
        ) = adapter._tracking_state(
            pose=pose,
            yaw=yaw,
            waypoints=waypoints,
            active_index=active_index,
            axis_x=axis_x,
            axis_y=axis_y,
        )
    if adapter._should_start_recovery(
        nav_feedback=nav_feedback,
        target=target,
        is_final_waypoint=active_index == len(waypoints) - 1,
        distance=distance,
        heading_error=heading_error,
    ):
        adapter._enter_recovery(
            active_index=active_index,
            target=target,
            current_region=current_region,
            distance=distance,
            heading_error=heading_error,
            local_forward=local_forward,
            local_lateral=local_lateral,
            nav_feedback=nav_feedback,
        )
    recovery_mode = adapter._step_recovery_mode()
    requires_replan = recovery_mode is None and adapter._should_request_local_path_replan()
    if requires_replan:
        base_command = (0.0, 0.0, 0.0)
        controller_mode = "replan_required"
    elif recovery_mode is not None:
        base_command = adapter._recovery_command()
        controller_mode = "recover"
    else:
        base_command, controller_mode = adapter._tracking_command(
            target=target,
            current_region=current_region,
            is_final_waypoint=active_index == len(waypoints) - 1,
            tracking_distance=tracking_distance,
            target_distance=distance,
            heading_error=heading_error,
            local_forward=local_forward,
            local_lateral=local_lateral,
        )
    base_command = adapter._smooth_command(base_command)

    return adapter._base_action(*base_command), {
        "backend": "waypoint",
        "controller_mode": controller_mode,
        "active_waypoint_index": active_index,
        "goal_reached": False,
        "distance_to_waypoint": distance,
        "tracking_distance": tracking_distance,
        "heading_error": heading_error,
        "local_target": {
            "forward": local_forward,
            "lateral": local_lateral,
        },
        "tracking_target": dict(tracking_target),
        "target_waypoint": dict(target),
        "vertical_axis": vertical_axis,
        "yaw_source": yaw_source,
        "recovery_mode": recovery_mode,
        "recovery_profile": adapter._recovery_profile,
        "recovery_cycles_on_waypoint": adapter._recovery_cycles_on_waypoint,
        "requires_replan": requires_replan,
        "replan_reason": "local_path_recovery_exhausted" if requires_replan else None,
        "loop_detected": loop_detected,
        "oscillation_detected": adapter._oscillation_detected(),
        "steps_since_progress": adapter._steps_since_waypoint_progress,
        "best_distance_to_waypoint": adapter._best_distance_to_waypoint,
        "progress_distance": progress_distance,
        "progress_reference": adapter._progress_reference_label(target=target),
        "path_tracking_mode": adapter._path_tracking_mode,
        "path_cross_track_error": adapter._local_path_info_value("cross_track_error"),
        "path_signed_cross_track_error": adapter._local_path_info_value("signed_cross_track_error"),
        "path_segment_index": adapter._local_path_info_value("segment_index"),
        "path_tangent_heading": adapter._local_path_info_value("tangent_heading"),
        "path_portal_prealign_active": adapter._local_path_info_value("portal_prealign_active"),
        "path_portal_prealign_blend": adapter._local_path_info_value("portal_prealign_blend"),
        "path_portal_prealign_locked": adapter._local_path_info_value("portal_prealign_locked"),
        "path_portal_midpoint_distance": adapter._local_path_info_value("portal_midpoint_distance"),
        "portal_alignment_lateral_deadband_effective": (
            adapter._effective_portal_alignment_lateral_deadband(target=target)
            if adapter._is_portal_like_waypoint(target)
            else None
        ),
    }


def _should_discard_stale_pre_transition_path(
    *,
    pose: dict[str, float],
    current_region: str | None,
    pending_transition_goal: dict[str, Any],
    nav_plan: Any,
    normalize_label: Any,
) -> bool:
    if (
        str(pending_transition_goal.get("waypoint_type") or "").strip().lower()
        != "post_portal_goal"
    ):
        return False
    if not isinstance(nav_plan, dict):
        return False

    transition_anchor = nav_plan.get("transition_anchor")
    if _pose_is_on_transition_target_side(pose=pose, transition_anchor=transition_anchor):
        return True

    current_region_norm = normalize_label(current_region)
    if current_region_norm is None:
        return False
    target_region_norm = normalize_label(
        pending_transition_goal.get("room_name")
        or pending_transition_goal.get("room_id")
        or (transition_anchor.get("room_name") if isinstance(transition_anchor, dict) else None)
        or (transition_anchor.get("room_id") if isinstance(transition_anchor, dict) else None)
    )
    source_region_norm = normalize_label(
        transition_anchor.get("source_room_name") if isinstance(transition_anchor, dict) else None
    )
    return bool(
        target_region_norm
        and current_region_norm == target_region_norm
        and (source_region_norm is None or current_region_norm != source_region_norm)
    )


def _pose_is_on_transition_target_side(
    *,
    pose: dict[str, float],
    transition_anchor: Any,
) -> bool:
    if not isinstance(transition_anchor, dict):
        return False
    axis = transition_anchor.get("portal_normal_axis")
    if axis not in {"x", "y", "z"}:
        return False
    try:
        value = float(pose[axis])
        boundary = float(transition_anchor["portal_boundary_value"])
        normal_sign = float(transition_anchor["portal_normal_sign"])
    except (KeyError, TypeError, ValueError):
        return False
    return (value - boundary) * normal_sign >= _TRANSITION_TARGET_SIDE_MARGIN_M
