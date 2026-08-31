from __future__ import annotations

import re
from typing import Any

from . import portal_safety as nav2_portal_safety
from .navigator import DEFAULT_NAV2_VERSION_PROFILE
from ..control.waypoint_policy import WaypointPolicyAdapter


class Nav2PolicyAdapter:
    def __init__(
        self,
        *,
        fallback_policy: WaypointPolicyAdapter | None = None,
        default_version_profile: str = DEFAULT_NAV2_VERSION_PROFILE,
    ) -> None:
        self.fallback_policy = fallback_policy or WaypointPolicyAdapter()
        self.default_version_profile = default_version_profile

    def ping(self) -> bool:
        return True

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        self.fallback_policy.reset(options=options)
        return {"status": "reset"}

    def get_modality_config(self) -> dict[str, Any]:
        return {"action": {"modality_keys": ["base"]}}

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        options = dict(options or {})
        if not self._should_use_nav2(options=options, observation=observation):
            return self.fallback_policy.get_action(observation, options=options)

        path_points = self._nav2_path_points(options=options)
        original_path_points = [dict(point) for point in path_points]
        path_points, target, pre_transition = self._prepare_local_segment(
            observation=observation,
            options=options,
            path_points=path_points,
        )
        nav2_waypoints = self._world_waypoints_from_nav2_path(
            observation=observation,
            options=options,
            path_points=path_points,
            target=target,
        )
        if not nav2_waypoints:
            return self.fallback_policy.get_action(observation, options=options)

        tracker_options = dict(options)
        tracker_options["nav_waypoints"] = nav2_waypoints
        tracker_options["path_tracking_mode"] = "nav2_local_path"
        if pre_transition:
            tracker_options["suppress_pending_local_path_transition_goal"] = True
        tracker_options.pop("active_waypoint_index", None)

        action, info = self.fallback_policy.get_action(observation, options=tracker_options)
        wrapped_info = dict(info)
        clipped_segment_complete = bool(target is not None and target.get("replan_after_reaching"))
        local_segment_complete = bool(wrapped_info.get("goal_reached")) and (
            pre_transition or clipped_segment_complete
        )
        if local_segment_complete:
            wrapped_info["goal_reached"] = False
            wrapped_info["controller_mode"] = "segment_complete"
            wrapped_info["local_segment_complete"] = True
            wrapped_info["requires_replan"] = True
            wrapped_info["replan_reason"] = (
                "clearance_clipped_segment_complete"
                if clipped_segment_complete
                else "portal_transition_segment_complete"
            )
        wrapped_info["backend"] = "nav2_waypoint"
        wrapped_info["path_backend"] = "nav2_local"
        wrapped_info["tracking_source"] = "nav2_path_points"
        wrapped_info["path_point_count"] = len(path_points)
        wrapped_info["nav2_input_path_point_count"] = len(original_path_points)
        wrapped_info["policy_path_transform"] = self._path_transform_label(
            original=original_path_points,
            prepared=path_points,
        )
        wrapped_info["nav2_profile"] = str(
            options.get("nav2_profile") or self.default_version_profile
        )
        wrapped_info["pre_transition_stage"] = pre_transition
        wrapped_info["portal_egress_guard"] = bool(target and target.get("portal_egress_guard"))
        return action, wrapped_info

    @staticmethod
    def _should_use_nav2(*, options: dict[str, Any], observation: dict[str, Any]) -> bool:
        if str(options.get("path_backend", "")).strip().lower() != "nav2_local":
            return False
        if observation.get("pose") is None and options.get("pose") is None:
            return False
        return True

    @staticmethod
    def _extract_pose(
        *, observation: dict[str, Any], options: dict[str, Any]
    ) -> dict[str, float] | None:
        for candidate in (observation.get("pose"), options.get("pose")):
            if not isinstance(candidate, dict):
                continue
            try:
                return {
                    "x": float(candidate["x"]),
                    "y": float(candidate["y"]),
                    "z": float(candidate.get("z", 0.0)),
                }
            except (KeyError, TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _resolve_vertical_axis(*, observation: dict[str, Any], options: dict[str, Any]) -> str:
        for candidate in (
            options.get("nav_vertical_axis"),
            options.get("vertical_axis"),
            observation.get("vertical_axis"),
        ):
            if isinstance(candidate, str) and candidate in {"x", "y", "z"}:
                return candidate
        nav_plan = options.get("nav_plan")
        if isinstance(nav_plan, dict):
            candidate = nav_plan.get("vertical_axis")
            if isinstance(candidate, str) and candidate in {"x", "y", "z"}:
                return candidate
        return "z"

    @staticmethod
    def _nav2_path_points(*, options: dict[str, Any]) -> list[dict[str, float]]:
        candidates = options.get("nav2_path_points")
        if not isinstance(candidates, list):
            nav_plan = options.get("nav_plan")
            if isinstance(nav_plan, dict):
                candidates = nav_plan.get("nav2_path_points")
        if not isinstance(candidates, list):
            return []

        points: list[dict[str, float]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            try:
                points.append({"x": float(candidate["x"]), "y": float(candidate["y"])})
            except (KeyError, TypeError, ValueError):
                continue
        return points

    def _prepare_local_segment(
        self,
        *,
        observation: dict[str, Any],
        options: dict[str, Any],
        path_points: list[dict[str, float]],
    ) -> tuple[list[dict[str, float]], dict[str, Any] | None, bool]:
        target = self._resolve_target_waypoint(options=options)
        transition_anchor = self._resolve_transition_anchor(options=options)
        execution_goal = self._resolve_execution_goal(options=options)
        nav2_compute_goal = self._resolve_nav2_compute_goal(options=options)
        nav2_compute_goal_is_execution = (
            target is not None
            and execution_goal is not None
            and nav2_compute_goal is not None
            and self._same_waypoint_position(first=nav2_compute_goal, second=execution_goal)
        )
        midpoint_anchor = self._resolve_midpoint_transition_anchor(options=options)
        midpoint_transition = midpoint_anchor is not None and not nav2_compute_goal_is_execution
        if midpoint_transition:
            transition_anchor = midpoint_anchor
        if not path_points or target is None or transition_anchor is None or execution_goal is None:
            return path_points, target, False

        target_waypoint_type = str(target.get("waypoint_type", "")).strip().lower()
        execution_waypoint_type = str(execution_goal.get("waypoint_type", "")).strip().lower()
        if (
            target_waypoint_type != "post_portal_goal"
            and execution_waypoint_type != "post_portal_goal"
        ):
            return path_points, target, False

        current_region = self._normalize_label(
            observation.get("current_region")
            or observation.get("current_room")
            or options.get("current_region")
            or options.get("current_room")
        )
        target_region = self._normalize_label(
            target.get("room_name") or execution_goal.get("room_name")
        )

        vertical_axis = self._resolve_vertical_axis(observation=observation, options=options)
        anchor_xy = self._world_to_nav2_plane(
            waypoint=transition_anchor, vertical_axis=vertical_axis
        )
        if anchor_xy is None:
            return path_points, target, False
        pose = self._extract_pose(observation=observation, options=options)
        pose_xy = None
        if pose is not None:
            pose_xy = self._world_to_nav2_plane(waypoint=pose, vertical_axis=vertical_axis)

        if target_region is not None and current_region == target_region:
            if midpoint_transition and not nav2_compute_goal_is_execution:
                if pose is not None and not self._pose_reached_midpoint_exit_waypoint(
                    pose=pose,
                    options=options,
                    midpoint_anchor=transition_anchor,
                ):
                    pass
                else:
                    return path_points, target, False
            elif (
                pose is not None
                and pose_xy is not None
                and not self._pose_reached_target_side_transition_anchor(
                    pose=pose,
                    pose_xy=pose_xy,
                    anchor=transition_anchor,
                    anchor_xy=anchor_xy,
                )
            ):
                pass
            else:
                return path_points, target, False

        if nav2_compute_goal_is_execution and midpoint_anchor is not None and pose_xy is not None:
            midpoint_xy = self._world_to_nav2_plane(
                waypoint=midpoint_anchor, vertical_axis=vertical_axis
            )
            if midpoint_xy is not None and self._pose_reached_or_crossed_midpoint_anchor(
                pose=pose,
                pose_xy=pose_xy,
                anchor=midpoint_anchor,
                anchor_xy=midpoint_xy,
            ):
                return path_points, target, False

        if nav2_compute_goal_is_execution:
            if (
                pose is not None
                and pose_xy is not None
                and not self._pose_reached_target_side_transition_anchor(
                    pose=pose,
                    pose_xy=pose_xy,
                    anchor=transition_anchor,
                    anchor_xy=anchor_xy,
                )
            ):
                egress_waypoint = nav2_portal_safety.egress_waypoint(
                    anchor=transition_anchor,
                    path_points=path_points,
                )
                if egress_waypoint is not None:
                    egress_xy = self._world_to_nav2_plane(
                        waypoint=egress_waypoint,
                        vertical_axis=vertical_axis,
                    )
                    if (
                        egress_xy is not None
                        and egress_waypoint.get("portal_egress_source") == "nav2_path"
                    ):
                        egress_path = self._truncate_path_points_at_anchor(
                            path_points=path_points,
                            anchor_xy=egress_xy,
                        )
                        if egress_path:
                            return egress_path, egress_waypoint, True

                return path_points, target, False
            return path_points, target, False

        trimmed_path = self._truncate_path_points_at_anchor(
            path_points=path_points, anchor_xy=anchor_xy
        )
        if not trimmed_path:
            return path_points, target, False
        segment_target = transition_anchor
        if midpoint_transition:
            exit_waypoint = self._midpoint_transition_exit_waypoint(
                options=options, midpoint_anchor=transition_anchor
            )
            if exit_waypoint is not None:
                segment_target = exit_waypoint
            else:
                segment_target = self._clean_local_path_waypoint(transition_anchor)
        if self._nav2_path_was_clipped(options=options):
            last_point = trimmed_path[-1]
            if not self._same_planar_position(
                first=last_point,
                second=anchor_xy,
                tolerance=0.05,
            ):
                segment_target = self._clipped_segment_target(
                    point=last_point,
                    reference=transition_anchor,
                )
        return trimmed_path, segment_target, True

    @staticmethod
    def _nav2_path_was_clipped(*, options: dict[str, Any]) -> bool:
        nav_plan = options.get("nav_plan")
        return bool(isinstance(nav_plan, dict) and nav_plan.get("nav2_path_clipped_for_clearance"))

    @staticmethod
    def _same_planar_position(
        *,
        first: dict[str, Any],
        second: dict[str, Any],
        tolerance: float,
    ) -> bool:
        try:
            return (
                (float(first["x"]) - float(second["x"])) ** 2
                + (float(first["y"]) - float(second["y"])) ** 2
            ) ** 0.5 <= max(0.0, float(tolerance))
        except (KeyError, TypeError, ValueError):
            return False

    @staticmethod
    def _clipped_segment_target(
        *,
        point: dict[str, Any],
        reference: dict[str, Any],
    ) -> dict[str, Any]:
        target = {
            "x": float(point["x"]),
            "y": float(point["y"]),
            "z": float(reference.get("z", 0.0)),
            "waypoint_type": "local_path",
            "local_segment_end": True,
            "replan_after_reaching": True,
        }
        for key in ("floor_id", "room_id", "room_name"):
            if key in reference:
                target[key] = reference[key]
        return target

    @staticmethod
    def _path_transform_label(
        *,
        original: list[dict[str, float]],
        prepared: list[dict[str, float]],
    ) -> str:
        if original == prepared:
            return "nav2_path_preserved"
        if prepared and len(prepared) <= len(original) and prepared == original[: len(prepared)]:
            return "nav2_path_prefix"
        return "nav2_path_transformed"

    def _world_waypoints_from_nav2_path(
        self,
        *,
        observation: dict[str, Any],
        options: dict[str, Any],
        path_points: list[dict[str, float]],
        target: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not path_points:
            return []

        reference_pose = self._extract_pose(observation=observation, options=options)
        if reference_pose is None:
            return []

        vertical_axis = self._resolve_vertical_axis(observation=observation, options=options)
        if target is None:
            target = self._resolve_target_waypoint(options=options)
        suppress_corridor_metadata = bool(
            target is not None and target.get("_suppress_doorway_corridor_metadata")
        )
        doorway_corridor = (
            [] if suppress_corridor_metadata else self._doorway_corridor_waypoints(options=options)
        )
        waypoints: list[dict[str, Any]] = []
        for point in path_points:
            world = self._nav2_plane_to_world(
                point_xy=point,
                vertical_axis=vertical_axis,
                reference_pose=reference_pose,
            )
            if world is None:
                continue
            waypoint = dict(world)
            corridor_match = self._match_doorway_corridor_waypoint(
                waypoint=waypoint,
                doorway_corridor=doorway_corridor,
            )
            if corridor_match is not None:
                waypoint["waypoint_type"] = str(corridor_match.get("waypoint_type") or "local_path")
                waypoint = self._merge_context_metadata(waypoint=waypoint, target=corridor_match)
            else:
                waypoint["waypoint_type"] = "local_path"
                if target is not None:
                    waypoint = self._merge_local_path_metadata(waypoint=waypoint, target=target)
            if self._should_append_waypoint(waypoints=waypoints, candidate=waypoint):
                waypoints.append(waypoint)
                reference_pose = waypoint

        if not waypoints:
            return []
        if target is not None:
            target_waypoint = self._merge_target_metadata(waypoint=target, target=target)
            target_waypoint.pop("_suppress_doorway_corridor_metadata", None)
            if self._should_append_waypoint(waypoints=waypoints, candidate=target_waypoint):
                waypoints.append(target_waypoint)
            else:
                waypoints[-1] = target_waypoint
        return waypoints

    @staticmethod
    def _resolve_target_waypoint(*, options: dict[str, Any]) -> dict[str, Any] | None:
        nav_plan = options.get("nav_plan")
        if isinstance(nav_plan, dict):
            for key in ("execution_goal", "local_goal", "transition_anchor"):
                candidate = Nav2PolicyAdapter._normalize_waypoint_candidate(nav_plan.get(key))
                if candidate is not None:
                    return candidate

        nav_waypoints = options.get("nav_waypoints")
        if isinstance(nav_waypoints, list) and nav_waypoints:
            candidate = Nav2PolicyAdapter._normalize_waypoint_candidate(nav_waypoints[-1])
            if candidate is not None:
                return candidate

        return Nav2PolicyAdapter._normalize_waypoint_candidate(options.get("nav_goal"))

    @staticmethod
    def _resolve_transition_anchor(*, options: dict[str, Any]) -> dict[str, Any] | None:
        nav_plan = options.get("nav_plan")
        if not isinstance(nav_plan, dict):
            return None
        anchor = Nav2PolicyAdapter._normalize_waypoint_candidate(nav_plan.get("transition_anchor"))
        if anchor is None:
            return None
        for candidate in Nav2PolicyAdapter._transition_anchor_metadata_candidates(
            nav_plan=nav_plan
        ):
            if not Nav2PolicyAdapter._same_waypoint_position(
                first=anchor, second=candidate, tolerance=1e-3
            ):
                continue
            merged = dict(candidate)
            merged.update(anchor)
            return merged
        return anchor

    @staticmethod
    def _resolve_execution_goal(*, options: dict[str, Any]) -> dict[str, Any] | None:
        nav_plan = options.get("nav_plan")
        if not isinstance(nav_plan, dict):
            return None
        return Nav2PolicyAdapter._normalize_waypoint_candidate(nav_plan.get("execution_goal"))

    @staticmethod
    def _resolve_nav2_compute_goal(*, options: dict[str, Any]) -> dict[str, Any] | None:
        nav_plan = options.get("nav_plan")
        if not isinstance(nav_plan, dict):
            return None
        return Nav2PolicyAdapter._normalize_waypoint_candidate(nav_plan.get("nav2_compute_goal"))

    @staticmethod
    def _transition_anchor_metadata_candidates(*, nav_plan: dict[str, Any]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        execution_goal = nav_plan.get("execution_goal")
        if isinstance(execution_goal, dict):
            nested_anchor = Nav2PolicyAdapter._normalize_waypoint_candidate(
                execution_goal.get("transition_anchor")
            )
            if nested_anchor is not None:
                candidates.append(nested_anchor)
        doorway_corridor = nav_plan.get("doorway_corridor")
        if isinstance(doorway_corridor, dict):
            corridor_anchor = Nav2PolicyAdapter._normalize_waypoint_candidate(
                doorway_corridor.get("target_anchor")
            )
            if corridor_anchor is not None:
                candidates.append(corridor_anchor)
        return candidates

    @staticmethod
    def _resolve_midpoint_transition_anchor(*, options: dict[str, Any]) -> dict[str, Any] | None:
        nav_plan = options.get("nav_plan")
        if not isinstance(nav_plan, dict):
            return None
        nav2_compute_goal = Nav2PolicyAdapter._normalize_waypoint_candidate(
            nav_plan.get("nav2_compute_goal")
        )
        if nav2_compute_goal is not None:
            waypoint_type = str(nav2_compute_goal.get("waypoint_type", "")).strip().lower()
            if waypoint_type == "portal_midpoint":
                return nav2_compute_goal
        doorway_corridor = nav_plan.get("doorway_corridor")
        if not isinstance(doorway_corridor, dict) or not bool(
            doorway_corridor.get("midpoint_only")
        ):
            return None
        for candidate in (doorway_corridor.get("midpoint"),):
            waypoint = Nav2PolicyAdapter._normalize_waypoint_candidate(candidate)
            if waypoint is None:
                continue
            waypoint_type = str(waypoint.get("waypoint_type", "")).strip().lower()
            if waypoint_type == "portal_midpoint":
                return waypoint
        return None

    @staticmethod
    def _midpoint_transition_exit_waypoint(
        *,
        options: dict[str, Any],
        midpoint_anchor: dict[str, Any],
    ) -> dict[str, Any] | None:
        nav_plan = options.get("nav_plan")
        if not isinstance(nav_plan, dict):
            return None
        doorway_corridor = nav_plan.get("doorway_corridor")

        target_anchor = None
        if isinstance(doorway_corridor, dict):
            target_anchor = Nav2PolicyAdapter._normalize_waypoint_candidate(
                doorway_corridor.get("target_anchor")
            )
        if target_anchor is None:
            target_anchor = Nav2PolicyAdapter._normalize_waypoint_candidate(
                nav_plan.get("transition_anchor")
            )

        try:
            normal_axis = str(midpoint_anchor["portal_normal_axis"])
            span_axis = str(midpoint_anchor["portal_span_axis"])
            boundary_value = float(midpoint_anchor["portal_boundary_value"])
            normal_sign = float(midpoint_anchor.get("portal_normal_sign", 1.0))
            midpoint_span = float(midpoint_anchor[span_axis])
        except (KeyError, TypeError, ValueError):
            return None
        if normal_axis not in {"x", "y"} or span_axis not in {"x", "y"} or normal_axis == span_axis:
            return None
        if normal_sign == 0.0:
            normal_sign = 1.0
        normal_direction = 1.0 if normal_sign >= 0.0 else -1.0

        exit_offset = 0.90
        if target_anchor is not None:
            try:
                target_offset = (
                    float(target_anchor[normal_axis]) - boundary_value
                ) * normal_direction
            except (KeyError, TypeError, ValueError):
                target_offset = 0.0
            if target_offset > 0.0:
                exit_offset = min(1.10, max(0.90, target_offset + 0.45))

        exit_waypoint = dict(midpoint_anchor)
        if target_anchor is not None:
            for key in ("floor_id", "room_id", "room_name", "nav_node"):
                if key in target_anchor:
                    exit_waypoint[key] = target_anchor[key]
        exit_waypoint[normal_axis] = boundary_value + normal_direction * exit_offset
        exit_waypoint[span_axis] = midpoint_span
        return Nav2PolicyAdapter._clean_local_path_waypoint(
            exit_waypoint, suppress_corridor_metadata=True
        )

    @staticmethod
    def _clean_local_path_waypoint(
        waypoint: dict[str, Any],
        *,
        suppress_corridor_metadata: bool = False,
    ) -> dict[str, Any]:
        cleaned = dict(waypoint)
        cleaned["waypoint_type"] = "local_path"
        if suppress_corridor_metadata:
            cleaned["_suppress_doorway_corridor_metadata"] = True
        for key in list(cleaned):
            if key.startswith("portal_") or key.startswith("source_room"):
                cleaned.pop(key, None)
        return cleaned

    @staticmethod
    def _normalize_waypoint_candidate(candidate: Any) -> dict[str, Any] | None:
        if not isinstance(candidate, dict):
            return None
        normalized = dict(candidate)
        position = candidate.get("position")
        if isinstance(position, dict):
            normalized.update(position)
        coords: dict[str, float] = {}
        for axis in ("x", "y", "z"):
            value = normalized.get(axis)
            if not isinstance(value, (int, float)):
                return None
            coords[axis] = float(value)
        normalized.update(coords)
        return normalized

    @staticmethod
    def _same_waypoint_position(
        *, first: dict[str, Any], second: dict[str, Any], tolerance: float = 1e-4
    ) -> bool:
        for axis in ("x", "y", "z"):
            try:
                if abs(float(first[axis]) - float(second[axis])) > tolerance:
                    return False
            except (KeyError, TypeError, ValueError):
                return False
        return True

    @staticmethod
    def _nav2_plane_to_world(
        *,
        point_xy: dict[str, Any],
        vertical_axis: str,
        reference_pose: dict[str, Any],
    ) -> dict[str, float] | None:
        if not isinstance(point_xy.get("x"), (int, float)) or not isinstance(
            point_xy.get("y"), (int, float)
        ):
            return None
        plane_axes = Nav2PolicyAdapter._plane_axes(vertical_axis)
        if plane_axes is None:
            return None

        world = {"x": 0.0, "y": 0.0, "z": 0.0}
        for axis in ("x", "y", "z"):
            value = reference_pose.get(axis, 0.0)
            world[axis] = float(value) if isinstance(value, (int, float)) else 0.0
        world[plane_axes[0]] = float(point_xy["x"])
        world[plane_axes[1]] = float(point_xy["y"])
        return world

    @staticmethod
    def _world_to_nav2_plane(
        *, waypoint: dict[str, Any], vertical_axis: str
    ) -> dict[str, float] | None:
        plane_axes = Nav2PolicyAdapter._plane_axes(vertical_axis)
        if plane_axes is None:
            return None
        try:
            return {
                "x": float(waypoint[plane_axes[0]]),
                "y": float(waypoint[plane_axes[1]]),
            }
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _truncate_path_points_at_anchor(
        *,
        path_points: list[dict[str, float]],
        anchor_xy: dict[str, float],
    ) -> list[dict[str, float]]:
        if not path_points:
            return []
        nearest_index = min(
            range(len(path_points)),
            key=lambda idx: (path_points[idx]["x"] - anchor_xy["x"]) ** 2
            + (path_points[idx]["y"] - anchor_xy["y"]) ** 2,
        )
        end_index = nearest_index + 1
        if len(path_points) > 1:
            end_index = max(2, end_index)
        end_index = min(len(path_points), end_index)
        return [dict(point) for point in path_points[:end_index]]

    @staticmethod
    def _pose_reached_transition_anchor(
        *,
        pose_xy: dict[str, float],
        anchor_xy: dict[str, float],
        tolerance_m: float = 0.22,
    ) -> bool:
        dx = float(pose_xy["x"]) - float(anchor_xy["x"])
        dy = float(pose_xy["y"]) - float(anchor_xy["y"])
        return (dx * dx + dy * dy) ** 0.5 <= tolerance_m

    @staticmethod
    def _pose_reached_target_side_transition_anchor(
        *,
        pose: dict[str, float],
        pose_xy: dict[str, float],
        anchor: dict[str, Any],
        anchor_xy: dict[str, float],
        planar_tolerance_m: float = 0.22,
        normal_tolerance_m: float = 0.10,
    ) -> bool:
        target_side_anchor = (
            str(anchor.get("portal_alignment_stage", "")).strip().lower() == "target_anchor"
        )
        if target_side_anchor and nav2_portal_safety.has_portal_frame(anchor):
            return nav2_portal_safety.pose_has_sufficient_egress(
                pose=pose,
                anchor=anchor,
                span_tolerance_m=planar_tolerance_m,
            )

        if not Nav2PolicyAdapter._pose_reached_transition_anchor(
            pose_xy=pose_xy,
            anchor_xy=anchor_xy,
            tolerance_m=planar_tolerance_m,
        ):
            return False

        del pose, normal_tolerance_m
        return True

    @staticmethod
    def _pose_has_sufficient_target_side_portal_depth(
        *,
        pose: dict[str, float],
        anchor: dict[str, Any],
        planar_tolerance_m: float,
    ) -> bool:
        return nav2_portal_safety.pose_has_sufficient_egress(
            pose=pose,
            anchor=anchor,
            span_tolerance_m=planar_tolerance_m,
        )

    @staticmethod
    def _pose_reached_or_crossed_midpoint_anchor(
        *,
        pose: dict[str, float] | None,
        pose_xy: dict[str, float],
        anchor: dict[str, Any],
        anchor_xy: dict[str, float],
        tolerance_m: float = 0.22,
    ) -> bool:
        if Nav2PolicyAdapter._pose_reached_transition_anchor(
            pose_xy=pose_xy,
            anchor_xy=anchor_xy,
            tolerance_m=tolerance_m,
        ):
            return True

        if pose is None:
            return False
        if str(anchor.get("portal_alignment_stage", "")).strip().lower() != "midpoint":
            return False
        try:
            normal_axis = str(anchor["portal_normal_axis"])
            normal_sign = float(anchor.get("portal_normal_sign", 1.0))
            boundary_value = float(anchor["portal_boundary_value"])
            pose_normal = float(pose[normal_axis])
        except (KeyError, TypeError, ValueError):
            return False
        if normal_axis not in {"x", "y", "z"}:
            return False
        if normal_sign == 0.0:
            normal_sign = 1.0
        return (pose_normal - boundary_value) * normal_sign >= 0.0

    @staticmethod
    def _pose_reached_midpoint_exit_waypoint(
        *,
        pose: dict[str, float],
        options: dict[str, Any],
        midpoint_anchor: dict[str, Any],
        normal_tolerance_m: float = 0.08,
        span_tolerance_m: float = 0.18,
    ) -> bool:
        exit_waypoint = Nav2PolicyAdapter._midpoint_transition_exit_waypoint(
            options=options,
            midpoint_anchor=midpoint_anchor,
        )
        if exit_waypoint is None:
            return False
        try:
            normal_axis = str(midpoint_anchor["portal_normal_axis"])
            span_axis = str(midpoint_anchor["portal_span_axis"])
            boundary_value = float(midpoint_anchor["portal_boundary_value"])
            normal_sign = float(midpoint_anchor.get("portal_normal_sign", 1.0))
            pose_normal = float(pose[normal_axis])
            exit_normal = float(exit_waypoint[normal_axis])
            pose_span = float(pose[span_axis])
            exit_span = float(exit_waypoint[span_axis])
            span_min = float(midpoint_anchor["portal_span_min"])
            span_max = float(midpoint_anchor["portal_span_max"])
        except (KeyError, TypeError, ValueError):
            return False
        if normal_axis not in {"x", "y", "z"} or span_axis not in {"x", "y", "z"}:
            return False
        if normal_axis == span_axis:
            return False
        if normal_sign == 0.0:
            normal_sign = 1.0
        pose_depth = (pose_normal - boundary_value) * normal_sign
        exit_depth = (exit_normal - boundary_value) * normal_sign
        if pose_depth < max(0.0, exit_depth - normal_tolerance_m):
            return False
        span_low = min(span_min, span_max) - span_tolerance_m
        span_high = max(span_min, span_max) + span_tolerance_m
        if pose_span < span_low or pose_span > span_high:
            return False
        return abs(pose_span - exit_span) <= span_tolerance_m

    @staticmethod
    def _normalize_label(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split()).strip()
        return normalized or None

    @staticmethod
    def _plane_axes(vertical_axis: str) -> tuple[str, str] | None:
        mapping = {
            "x": ("y", "z"),
            "y": ("x", "z"),
            "z": ("x", "y"),
        }
        return mapping.get(vertical_axis)

    @staticmethod
    def _should_append_waypoint(
        *, waypoints: list[dict[str, Any]], candidate: dict[str, Any]
    ) -> bool:
        if not waypoints:
            return True
        previous = waypoints[-1]
        dx = float(candidate["x"]) - float(previous["x"])
        dy = float(candidate["y"]) - float(previous["y"])
        dz = float(candidate["z"]) - float(previous["z"])
        return (dx * dx + dy * dy + dz * dz) > 1e-8

    @staticmethod
    def _merge_target_metadata(
        *, waypoint: dict[str, Any], target: dict[str, Any]
    ) -> dict[str, Any]:
        merged = dict(waypoint)
        merged["waypoint_type"] = str(target.get("waypoint_type") or "goal")
        return Nav2PolicyAdapter._merge_context_metadata(waypoint=merged, target=target)

    @staticmethod
    def _merge_local_path_metadata(
        *, waypoint: dict[str, Any], target: dict[str, Any]
    ) -> dict[str, Any]:
        merged = dict(waypoint)
        if "floor_id" in target:
            merged["floor_id"] = target["floor_id"]
        return merged

    @staticmethod
    def _merge_context_metadata(
        *, waypoint: dict[str, Any], target: dict[str, Any]
    ) -> dict[str, Any]:
        merged = dict(waypoint)
        for key in (
            "room_id",
            "room_name",
            "floor_id",
            "goal_type",
            "object_id",
            "object_name",
            "nav_node",
            "source_room_id",
            "source_room_name",
            "transition_anchor",
            "portal_gap",
            "portal_span",
            "portal_source_point",
            "portal_target_point",
            "portal_normal_axis",
            "portal_boundary_value",
            "portal_normal_sign",
            "portal_span_axis",
            "portal_span_min",
            "portal_span_max",
            "portal_refined_from_traversability",
            "portal_desired_heading",
            "portal_alignment_stage",
        ):
            if key in target:
                merged[key] = target[key]
        return merged

    @staticmethod
    def _doorway_corridor_waypoints(*, options: dict[str, Any]) -> list[dict[str, Any]]:
        nav_plan = options.get("nav_plan")
        if not isinstance(nav_plan, dict):
            return []
        nav2_compute_goal = Nav2PolicyAdapter._normalize_waypoint_candidate(
            nav_plan.get("nav2_compute_goal")
        )
        execution_goal = Nav2PolicyAdapter._normalize_waypoint_candidate(
            nav_plan.get("execution_goal")
        )
        if (
            nav2_compute_goal is not None
            and execution_goal is not None
            and Nav2PolicyAdapter._same_waypoint_position(
                first=nav2_compute_goal, second=execution_goal
            )
        ):
            return []
        doorway_corridor = nav_plan.get("doorway_corridor")
        if not isinstance(doorway_corridor, dict):
            return []

        candidates: list[dict[str, Any]] = []
        for key in ("source_anchor", "midpoint", "target_anchor"):
            candidate = Nav2PolicyAdapter._normalize_waypoint_candidate(doorway_corridor.get(key))
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    @staticmethod
    def _match_doorway_corridor_waypoint(
        *,
        waypoint: dict[str, Any],
        doorway_corridor: list[dict[str, Any]],
        distance_tolerance: float = 0.2,
    ) -> dict[str, Any] | None:
        if not doorway_corridor:
            return None

        best_match: dict[str, Any] | None = None
        best_distance_sq = distance_tolerance * distance_tolerance
        for candidate in doorway_corridor:
            dx = float(candidate["x"]) - float(waypoint["x"])
            dy = float(candidate["y"]) - float(waypoint["y"])
            dz = float(candidate.get("z", 0.0)) - float(waypoint.get("z", 0.0))
            distance_sq = dx * dx + dy * dy + dz * dz
            if distance_sq > best_distance_sq:
                continue
            best_match = candidate
            best_distance_sq = distance_sq
        return best_match
