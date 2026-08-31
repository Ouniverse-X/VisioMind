from __future__ import annotations

from typing import Any


def push_distance(adapter: Any, distance: float) -> None:
    adapter._distance_history.append(float(distance))
    if len(adapter._distance_history) > adapter.progress_window:
        adapter._distance_history = adapter._distance_history[-adapter.progress_window :]


def push_pose(
    adapter: Any,
    *,
    pose: dict[str, float],
    axis_x: str,
    axis_y: str,
) -> None:
    adapter._pose_history.append((float(pose[axis_x]), float(pose[axis_y])))
    if len(adapter._pose_history) > adapter.progress_window:
        adapter._pose_history = adapter._pose_history[-adapter.progress_window :]


def update_waypoint_progress(
    adapter: Any,
    *,
    active_index: int,
    target: dict[str, Any],
    distance: float,
    progress_distance: float,
    heading_error: float,
) -> None:
    del distance
    adapter._heading_progress_this_step = False
    target_signature = progress_tracking_signature(target=target)
    if (
        adapter._tracking_waypoint_index != active_index
        or adapter._tracking_waypoint_signature != target_signature
    ):
        adapter._tracking_waypoint_index = active_index
        adapter._tracking_waypoint_signature = target_signature
        adapter._best_distance_to_waypoint = float(progress_distance)
        adapter._best_heading_error_to_waypoint = abs(float(heading_error))
        adapter._steps_since_waypoint_progress = 0
        adapter._heading_error_history = []
    elif (
        adapter._best_distance_to_waypoint is None
        or progress_distance < adapter._best_distance_to_waypoint - adapter.progress_epsilon
    ):
        adapter._best_distance_to_waypoint = float(progress_distance)
        adapter._best_heading_error_to_waypoint = abs(float(heading_error))
        adapter._steps_since_waypoint_progress = 0
    else:
        if adapter._heading_progress_is_improving(
            target=target,
            heading_error=heading_error,
        ):
            adapter._heading_progress_this_step = True
            adapter._steps_since_waypoint_progress = 0
        else:
            adapter._steps_since_waypoint_progress += 1

    adapter._heading_error_history.append(float(heading_error))
    if len(adapter._heading_error_history) > adapter.oscillation_window:
        adapter._heading_error_history = adapter._heading_error_history[
            -adapter.oscillation_window :
        ]


def progress_tracking_signature(target: dict[str, Any]) -> tuple[Any, ...]:
    waypoint_type = str(target.get("waypoint_type", "")).strip().lower()
    alignment_stage = str(target.get("portal_alignment_stage", "")).strip().lower()
    if alignment_stage in {"source_anchor", "midpoint", "target_anchor"}:
        return (
            waypoint_type,
            alignment_stage,
            round(float(target.get("x", 0.0)), 4),
            round(float(target.get("y", 0.0)), 4),
            round(float(target.get("z", 0.0)), 4),
            str(target.get("room_name", "")),
            str(target.get("source_room_name", "")),
        )
    return (waypoint_type,)


def progress_distance(
    adapter: Any,
    *,
    target: dict[str, Any],
    pose: dict[str, float],
    waypoints: list[dict[str, float]],
    active_index: int,
    axis_x: str,
    axis_y: str,
    distance: float,
) -> float:
    if (
        not adapter._uses_local_path_tracking()
        or adapter._waypoint_requires_portal_heading_alignment(
            target,
            distance=distance,
        )
    ):
        return float(distance)
    return remaining_local_path_distance(
        adapter,
        pose=pose,
        waypoints=waypoints,
        active_index=active_index,
        axis_x=axis_x,
        axis_y=axis_y,
    )


def remaining_local_path_distance(
    adapter: Any,
    *,
    pose: dict[str, float],
    waypoints: list[dict[str, float]],
    active_index: int,
    axis_x: str,
    axis_y: str,
) -> float:
    if not waypoints:
        return 0.0
    if active_index >= len(waypoints):
        return 0.0
    if len(waypoints) == 1:
        return adapter._planar_distance(
            pose=pose,
            target=waypoints[0],
            axis_x=axis_x,
            axis_y=axis_y,
        )

    closest_point, closest_segment_index, _ = adapter._closest_point_on_local_path(
        pose=pose,
        waypoints=waypoints,
        active_index=active_index,
        axis_x=axis_x,
        axis_y=axis_y,
    )
    previous = dict(closest_point)
    remaining = 0.0
    start_index = min(max(closest_segment_index + 1, active_index), len(waypoints) - 1)
    for index in range(start_index, len(waypoints)):
        candidate = waypoints[index]
        remaining += adapter._planar_distance(
            pose=previous,
            target=candidate,
            axis_x=axis_x,
            axis_y=axis_y,
        )
        previous = dict(candidate)
    return float(remaining)


def progress_reference_label(adapter: Any, *, target: dict[str, Any]) -> str:
    if (
        adapter._uses_local_path_tracking()
        and not adapter._waypoint_requires_portal_heading_alignment(target)
    ):
        return "local_path_remaining_distance"
    return "waypoint_distance"


def smooth_command(
    adapter: Any,
    command: tuple[float, float, float],
) -> tuple[float, float, float]:
    if adapter.command_smoothing <= 0.0:
        adapter._last_base_command = command
        return command
    previous_weight = adapter.command_smoothing
    current_weight = 1.0 - previous_weight
    smoothed = tuple(
        current_weight * current + previous_weight * previous
        for current, previous in zip(command, adapter._last_base_command)
    )
    adapter._last_base_command = smoothed
    return smoothed


def clear_progress_state(adapter: Any) -> None:
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


def reset_waypoint_tracking_state(adapter: Any) -> None:
    adapter._distance_history = []
    adapter._pose_history = []
    adapter._heading_error_history = []
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
    adapter._local_path_follow_state = None
    adapter._portal_prealign_lock_signature = None
    adapter._portal_stage_lock_signature = None
    adapter._portal_stage_lock_floor = 0
