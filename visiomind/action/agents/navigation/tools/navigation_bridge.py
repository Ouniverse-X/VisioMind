from __future__ import annotations

from typing import Any


class GoalConditionedNavigationBridge:
    def build_policy_options(
        self,
        *,
        existing_options: dict[str, Any] | None,
        grounded_goal: dict[str, Any] | None,
        path_plan: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        options = dict(existing_options or {})
        if grounded_goal:
            options["nav_goal"] = dict(grounded_goal)
        if path_plan:
            options["nav_plan"] = dict(path_plan)
            waypoints = path_plan.get("waypoints")
            if isinstance(waypoints, list):
                options["nav_waypoints"] = list(waypoints)
            for key in (
                "waypoint_tracking_mode",
                "waypoint_scope",
                "global_waypoint_index",
                "dense_waypoint_index",
                "path_backend",
                "nav2_trav_map_filename",
                "nav2_path_points",
            ):
                if key in path_plan:
                    options[key] = path_plan[key]
            vertical_axis = path_plan.get("vertical_axis")
            if isinstance(vertical_axis, str) and vertical_axis in {"x", "y", "z"}:
                options["nav_vertical_axis"] = vertical_axis
        return options or None

    def build_runtime_artifacts(
        self,
        *,
        grounded_goal: dict[str, Any] | None,
        path_plan: dict[str, Any] | None,
        backend_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        artifacts: dict[str, Any] = {}
        if grounded_goal:
            artifacts["grounded_goal"] = dict(grounded_goal)
            artifacts["nav_goal"] = dict(grounded_goal)
            if isinstance(grounded_goal.get("grounding_candidates"), list):
                artifacts["grounding_candidates"] = list(grounded_goal["grounding_candidates"])
            if isinstance(grounded_goal.get("selected_grounding_candidate"), dict):
                artifacts["selected_grounding_candidate"] = dict(
                    grounded_goal["selected_grounding_candidate"]
                )
        if path_plan:
            artifacts["path_plan"] = dict(path_plan)
            waypoints = path_plan.get("waypoints")
            if isinstance(waypoints, list):
                artifacts["waypoints"] = list(waypoints)
            for key in (
                "path_backend",
                "requested_planner",
                "nav2_profile",
                "nav2_error",
                "nav2_trav_map_filename",
                "nav2_path_points",
                "waypoint_tracking_mode",
                "waypoint_scope",
                "global_waypoint_index",
                "dense_waypoint_index",
                "local_goal",
                "execution_goal",
                "nav2_compute_goal",
                "transition_anchor",
            ):
                if key in path_plan:
                    artifacts[key] = path_plan[key]
            vertical_axis = path_plan.get("vertical_axis")
            if isinstance(vertical_axis, str) and vertical_axis in {"x", "y", "z"}:
                artifacts["vertical_axis"] = vertical_axis
        if backend_state:
            artifacts["backend_state"] = dict(backend_state)
            for key in (
                "active_waypoint_index",
                "recovery_mode",
                "exploration_target",
                "vertical_axis",
                "localization_guard",
            ):
                if key in backend_state:
                    artifacts[key] = backend_state[key]
        return artifacts
