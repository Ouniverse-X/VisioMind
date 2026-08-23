from __future__ import annotations

from typing import Any

from . import fallback_corridor as nav2_fallback_corridor


def plan_local_path(
    navigator: Any,
    *,
    scene_id: str | None,
    start_pose: dict[str, Any] | None,
    vertical_axis: str,
    current_region: str | None,
    execution_goal: dict[str, Any],
    transition_anchor: dict[str, Any] | None,
    nav2_compute_goal: dict[str, Any],
    doorway_corridor: dict[str, Any] | None,
    nav2_trav_map_filename: str | None,
    nav2_scene_obstacle_inflation_radius_m: float,
    pre_transition_stage: bool | None = None,
) -> dict[str, Any]:
    local_waypoints: list[dict[str, Any]] = []
    local_backend = "nav2_local_pending"
    nav2_error: str | None = None
    nav2_empty_path_reason: str | None = None
    nav2_raw_path_length = 0
    nav2_raw_path_points: list[dict[str, float]] = []
    nav2_path_points: list[dict[str, float]] = []
    nav2_path_clipped_for_clearance = False

    local_goal_position = navigator._waypoint_position(nav2_compute_goal)
    if isinstance(start_pose, dict) and local_goal_position is not None:
        start_xy = navigator._world_pose_to_nav2_plane(
            start_pose, vertical_axis=vertical_axis
        )
        goal_xy = navigator._world_pose_to_nav2_plane(
            local_goal_position, vertical_axis=vertical_axis
        )
        if start_xy is not None and goal_xy is not None:
            try:
                path_response = navigator._compute_nav2_path_response(
                    scene_id=scene_id,
                    start_xy=start_xy,
                    goal_xy=goal_xy,
                    nav2_trav_map_filename=nav2_trav_map_filename,
                    nav2_scene_obstacle_inflation_radius_m=nav2_scene_obstacle_inflation_radius_m,
                    navigation_goal=execution_goal,
                    vertical_axis=vertical_axis,
                )
                path_points = navigator._extract_path_points(path_response)
                if not path_points:
                    try:
                        from .nav2_runtime_bridge import diagnose_empty_path

                        diagnostic_map = navigator._load_stamped_traversability_grid(
                            scene_id=scene_id,
                            map_resolution=navigator.portal_analysis_map_resolution,
                            trav_map_filename=nav2_trav_map_filename,
                            navigation_goal=execution_goal,
                            vertical_axis=vertical_axis,
                        )
                        nav2_empty_path_reason = diagnose_empty_path(
                            map_spec=diagnostic_map,
                            start_xy=start_xy,
                            goal_xy=goal_xy,
                        )
                    except Exception:
                        nav2_empty_path_reason = "map_unavailable"
                    raise RuntimeError(path_response.get("error") or "empty_path")
                nav2_raw_path_length = len(path_points)
                nav2_raw_path_points = [
                    {"x": float(point["x"]), "y": float(point["y"])}
                    for point in path_points
                ]
                path_target = nav2_compute_goal
                if pre_transition_stage is None:
                    pre_transition_stage = navigator._is_pre_transition_stage(
                        current_region=current_region,
                        execution_goal=execution_goal,
                        transition_anchor=transition_anchor,
                    )
                if doorway_corridor is not None and pre_transition_stage:
                    path_points = nav2_fallback_corridor.append_transition_corridor_to_path(
                        path_points=path_points,
                        doorway_corridor=doorway_corridor,
                        vertical_axis=vertical_axis,
                        start_from=nav2_fallback_corridor.doorway_corridor_stage_key(
                            waypoint=nav2_compute_goal,
                            doorway_corridor=doorway_corridor,
                            same_waypoint_signature=navigator._same_waypoint_signature,
                        ),
                        world_pose_to_nav2_plane=navigator._world_pose_to_nav2_plane,
                    )
                    if bool(doorway_corridor.get("midpoint_only")) and isinstance(
                        doorway_corridor.get("midpoint"), dict
                    ):
                        path_target = doorway_corridor["midpoint"]
                    else:
                        path_target = transition_anchor or execution_goal
                path_points, nav2_path_clipped_for_clearance = (
                    navigator._refine_nav2_local_path_points(
                        scene_id=scene_id,
                        path_points=path_points,
                        nav2_trav_map_filename=nav2_trav_map_filename,
                        navigation_goal=execution_goal,
                        vertical_axis=vertical_axis,
                    )
                )
                if not path_points and nav2_path_clipped_for_clearance:
                    raise RuntimeError("room_exit_path")
                nav2_path_points = [
                    {"x": float(point["x"]), "y": float(point["y"])}
                    for point in path_points
                ]
                local_waypoints = navigator._world_waypoints_from_nav2_path(
                    path_points=path_points,
                    vertical_axis=vertical_axis,
                    start_pose=start_pose,
                    target=path_target,
                    append_target=not nav2_path_clipped_for_clearance,
                )
                local_backend = "nav2_local"
            except Exception as exc:
                nav2_error = str(exc)
        else:
            nav2_error = "pose_projection_failed"
    else:
        nav2_error = "pose_or_goal_missing"

    return {
        "local_waypoints": local_waypoints,
        "local_backend": local_backend,
        "nav2_error": nav2_error,
        "nav2_empty_path_reason": nav2_empty_path_reason,
        "dynamic_map_update": getattr(navigator, "_last_dynamic_map_update", None),
        "nav2_raw_path_length": nav2_raw_path_length,
        "nav2_raw_path_points": nav2_raw_path_points,
        "nav2_path_points": nav2_path_points,
        "nav2_path_clipped_for_clearance": nav2_path_clipped_for_clearance,
    }
