from __future__ import annotations

import json
import math
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from . import cache_state as nav2_cache_state
from . import candidate_validation as nav2_candidate_validation
from . import fallback_corridor as nav2_fallback_corridor
from . import local_path_planning as nav2_local_path_planning
from . import portal_safety as nav2_portal_safety
from . import profile_runtime as nav2_profile_runtime
from . import semantic_plan as nav2_semantic_plan


@dataclass(frozen=True)
class Nav2VersionProfile:
    profile_id: str
    ros_distro: str
    isaac_sim_version: str
    setup_script: str
    setup_script_candidates: tuple[str, ...] = ()
    python_bin: str = "python3"
    notes: tuple[str, ...] = ()

    def resolved_setup_script(self) -> str:
        return nav2_profile_runtime.resolve_profile_setup_script(self)


NAV2_VERSION_PROFILES: dict[str, Nav2VersionProfile] = {
    "humble_isaacsim_4_5": Nav2VersionProfile(
        profile_id="humble_isaacsim_4_5",
        ros_distro="humble",
        isaac_sim_version="4.5",
        setup_script="/opt/ros/humble/setup.bash",
        setup_script_candidates=(
            "/mnt/data/opt/ros/humble/setup.bash",
            "/opt/ros/humble/setup.bash",
        ),
        notes=(
            "Pinned for the current Voltron + Isaac Sim 4.5 deployment.",
            "Voltron stays in its conda env; ROS2/Nav2 are sourced in a subprocess.",
        ),
    )
}

DEFAULT_NAV2_VERSION_PROFILE = "humble_isaacsim_4_5"
DEFAULT_R1PRO_NAV_FOOTPRINT: tuple[tuple[float, float], ...] = (
    (0.24, 0.34),
    (0.24, -0.34),
    (-0.40, -0.34),
    (-0.40, 0.34),
)
DEFAULT_R1PRO_NAV_FOOTPRINT_PADDING_M = 0.02


NAV2_RUNTIME_GUARANTEED_CLEARANCE_RADIUS_M = 0.25


def _uses_objectless_base_map(filename: str | None) -> bool:
    normalized = str(filename or "floor_trav_no_obj_0.png").strip().lower()
    return "no_obj" in normalized or "no_object" in normalized


class SemanticNavigator(Protocol):
    def update(
        self,
        observation: dict[str, Any],
        *,
        pose: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def ground_goal(
        self,
        instruction: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def generate_object_approach_candidates(
        self,
        *,
        start: dict[str, Any],
        goal: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...

    def plan_path(
        self,
        *,
        start: dict[str, Any],
        goal: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


class Nav2PathClient(Protocol):
    def inspect_environment(self) -> dict[str, Any]: ...

    def compute_path(
        self,
        *,
        start_xy: dict[str, float],
        goal_xy: dict[str, float],
        frame_id: str,
        planner_id: str | None,
        timeout_s: float,
    ) -> dict[str, Any]: ...


class SubprocessNav2ComputePathClient:
    def __init__(
        self,
        *,
        profile: Nav2VersionProfile,
        action_name: str = "compute_path_to_pose",
        worker_script: str | Path | None = None,
    ) -> None:
        self.profile = profile
        self.action_name = str(action_name).strip() or "compute_path_to_pose"
        default_script = Path(__file__).resolve().parent / "compute_path_worker.py"
        self.worker_script = (
            Path(worker_script).expanduser() if worker_script is not None else default_script
        )
        self._environment_summary: dict[str, Any] | None = None

    def _build_overlay_env(self) -> dict[str, str]:
        return nav2_profile_runtime.build_overlay_env()

    @staticmethod
    def _prepend_path(value: str, existing: str | None) -> str:
        return nav2_profile_runtime.prepend_path(value, existing)

    def inspect_environment(self) -> dict[str, Any]:
        if self._environment_summary is not None:
            return dict(self._environment_summary)

        summary = nav2_profile_runtime.inspect_runtime_environment(
            profile=self.profile,
            worker_script=self.worker_script,
            base_env=self._build_overlay_env(),
        )
        self._environment_summary = summary
        return dict(summary)

    def compute_path(
        self,
        *,
        start_xy: dict[str, float],
        goal_xy: dict[str, float],
        frame_id: str,
        planner_id: str | None,
        timeout_s: float,
    ) -> dict[str, Any]:
        summary = self.inspect_environment()
        if not summary.get("worker_exists"):
            raise RuntimeError(f"Nav2 worker script not found: {self.worker_script}")
        if not summary.get("ros2_cli"):
            raise RuntimeError("ROS2 CLI unavailable for the selected Nav2 profile")
        if not summary.get("rclpy"):
            raise RuntimeError("rclpy is unavailable in the selected Nav2 profile")
        if not summary.get("nav2_msgs"):
            raise RuntimeError("nav2_msgs is unavailable in the selected Nav2 profile")

        request = {
            "action_name": self.action_name,
            "frame_id": frame_id,
            "planner_id": planner_id or "",
            "timeout_s": float(timeout_s),
            "start": {
                "x": float(start_xy["x"]),
                "y": float(start_xy["y"]),
                "yaw": float(start_xy.get("yaw", 0.0)),
            },
            "goal": {
                "x": float(goal_xy["x"]),
                "y": float(goal_xy["y"]),
                "yaw": float(goal_xy.get("yaw", 0.0)),
            },
        }
        resolved_setup_script = self.profile.resolved_setup_script()
        command = (
            f"source {shlex.quote(resolved_setup_script)} && "
            f"{self.profile.python_bin} {shlex.quote(str(self.worker_script))}"
        )
        result = subprocess.run(
            ["bash", "-lc", command],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            timeout=max(2.0, float(timeout_s) + 5.0),
            check=False,
            env=self._build_overlay_env(),
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            message = stderr or stdout or "nav2 worker failed"
            raise RuntimeError(message)
        try:
            payload = json.loads(result.stdout.strip() or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid Nav2 worker response: {result.stdout.strip()}") from exc
        return payload


class Nav2NavigatorAdapter:
    def __init__(
        self,
        *,
        semantic_backend: SemanticNavigator,
        path_client: Nav2PathClient | None = None,
        version_profile: str = DEFAULT_NAV2_VERSION_PROFILE,
        action_name: str = "compute_path_to_pose",
        frame_id: str = "map",
        planner_id: str | None = None,
        timeout_s: float = 8.0,
        strict: bool = False,
        trav_map_filename: str | None = None,
        waypoint_spacing: float = 0.35,
        global_waypoint_tolerance: float = 0.9,
        final_global_waypoint_tolerance: float = 0.9,
        final_global_waypoint_heading_tolerance_rad: float = 0.65,
        portal_analysis_map_resolution: float = 0.05,
        portal_clearance_radius_m: float = 0.35,
        portal_corridor_standoff_m: float = 0.18,
        portal_sampling_step_m: float = 0.05,
        local_path_clearance_radius_m: float = 0.0,
        local_path_waypoint_spacing_m: float = 0.35,
        portal_footprint: tuple[tuple[float, float], ...] = DEFAULT_R1PRO_NAV_FOOTPRINT,
        portal_footprint_padding_m: float = DEFAULT_R1PRO_NAV_FOOTPRINT_PADDING_M,
    ) -> None:
        if version_profile not in NAV2_VERSION_PROFILES:
            raise ValueError(
                f"Unknown Nav2 version profile '{version_profile}'. "
                f"Available: {sorted(NAV2_VERSION_PROFILES)}"
            )
        self.semantic_backend = semantic_backend
        self.version_profile = version_profile
        self.profile = NAV2_VERSION_PROFILES[version_profile]
        self.action_name = str(action_name).strip() or "compute_path_to_pose"
        self.frame_id = str(frame_id).strip() or "map"
        self.planner_id = None
        if isinstance(planner_id, str):
            normalized_planner_id = planner_id.strip()
            self.planner_id = normalized_planner_id or None
        self.timeout_s = max(1.0, float(timeout_s))
        self.strict = bool(strict)
        self.trav_map_filename = (
            str(trav_map_filename).strip()
            if isinstance(trav_map_filename, str) and trav_map_filename.strip()
            else None
        )
        self.waypoint_spacing = max(0.05, float(waypoint_spacing))
        self.global_waypoint_tolerance = max(0.05, float(global_waypoint_tolerance))
        self.final_global_waypoint_tolerance = max(0.05, float(final_global_waypoint_tolerance))
        self.final_global_waypoint_heading_tolerance_rad = max(
            0.05, float(abs(final_global_waypoint_heading_tolerance_rad))
        )
        self.portal_analysis_map_resolution = max(0.02, float(portal_analysis_map_resolution))
        self.portal_clearance_radius_m = max(0.0, float(portal_clearance_radius_m))
        self.portal_corridor_standoff_m = max(0.05, float(portal_corridor_standoff_m))
        self.portal_sampling_step_m = max(0.02, float(portal_sampling_step_m))
        self.portal_footprint = self._normalize_portal_footprint(portal_footprint)
        self.portal_footprint_padding_m = max(0.0, float(portal_footprint_padding_m))
        self._portal_half_width_m = self._portal_half_width(
            footprint=self.portal_footprint,
            padding_m=self.portal_footprint_padding_m,
        )
        self._portal_forward_extent_m = self._portal_forward_extent(
            footprint=self.portal_footprint,
            padding_m=self.portal_footprint_padding_m,
        )
        self._portal_rear_extent_m = self._portal_rear_extent(
            footprint=self.portal_footprint,
            padding_m=self.portal_footprint_padding_m,
        )
        self.portal_egress_depth_m = max(
            0.30,
            max(math.hypot(float(point[0]), float(point[1])) for point in self.portal_footprint)
            + self.portal_footprint_padding_m
            + 0.10,
        )
        self.local_path_clearance_radius_m = max(0.0, float(local_path_clearance_radius_m))
        self.local_path_waypoint_spacing_m = max(0.05, float(local_path_waypoint_spacing_m))
        self.path_client = path_client or SubprocessNav2ComputePathClient(
            profile=self.profile,
            action_name=self.action_name,
        )
        self._last_successful_local_segment: dict[str, Any] | None = None
        self._last_dynamic_map_update: dict[str, Any] | None = None
        self._last_runtime_overlay_signature = ""
        self._last_runtime_overlay_geometry: list[dict[str, Any]] = []
        self._last_object_approach_generation_diagnostics: dict[str, Any] = {}
        self._last_object_approach_generation_key: tuple[str, str] | None = None
        self._pending_portal_egress_anchor: dict[str, Any] | None = None
        self._pending_portal_egress_scene_id: str | None = None

    @staticmethod
    def _normalize_portal_footprint(
        footprint: tuple[tuple[float, float], ...],
    ) -> tuple[tuple[float, float], ...]:
        normalized: list[tuple[float, float]] = []
        for point in footprint:
            if not isinstance(point, (tuple, list)) or len(point) != 2:
                continue
            try:
                normalized.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError):
                continue
        if len(normalized) < 3:
            return DEFAULT_R1PRO_NAV_FOOTPRINT
        return tuple(normalized)

    @staticmethod
    def _portal_half_width(
        *,
        footprint: tuple[tuple[float, float], ...],
        padding_m: float,
    ) -> float:
        lateral_extent = max(abs(y_coord) for _, y_coord in footprint)
        return float(lateral_extent + padding_m)

    @staticmethod
    def _portal_forward_extent(
        *,
        footprint: tuple[tuple[float, float], ...],
        padding_m: float,
    ) -> float:
        forward_extent = max(0.0, max(x_coord for x_coord, _ in footprint))
        return float(forward_extent + padding_m)

    @staticmethod
    def _portal_rear_extent(
        *,
        footprint: tuple[tuple[float, float], ...],
        padding_m: float,
    ) -> float:
        rear_extent = max(0.0, -min(x_coord for x_coord, _ in footprint))
        return float(rear_extent + padding_m)

    def update(
        self,
        observation: dict[str, Any],
        *,
        pose: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        backend_state = dict(self.semantic_backend.update(observation, pose=pose))
        backend_state["nav_backend"] = "nav2"
        backend_state["nav2_profile"] = self.version_profile
        return backend_state

    def ground_goal(
        self,
        instruction: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        goal = dict(self.semantic_backend.ground_goal(instruction, context=context))
        goal["nav_backend"] = "nav2"
        goal["nav2_profile"] = self.version_profile
        return goal

    def generate_object_approach_candidates(
        self,
        *,
        start: dict[str, Any],
        goal: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not hasattr(self.semantic_backend, "generate_object_approach_candidates"):
            return []
        candidates = list(
            self.semantic_backend.generate_object_approach_candidates(
                start=start,
                goal=goal,
                context=context,
            )
        )
        diagnostics_store = getattr(
            self.semantic_backend,
            "_object_approach_diagnostics",
            None,
        )
        scene_id = str(
            goal.get("scene_id") or start.get("scene_id") or (context or {}).get("scene_id") or ""
        )
        object_id = str(goal.get("object_id") or "")
        diagnostics = (
            diagnostics_store.get((scene_id, object_id))
            if isinstance(diagnostics_store, dict)
            else None
        )
        self._last_object_approach_generation_diagnostics = (
            dict(diagnostics) if isinstance(diagnostics, dict) else {}
        )
        self._last_object_approach_generation_key = (scene_id, object_id)
        return candidates

    def plan_path(
        self,
        *,
        start: dict[str, Any],
        goal: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context = dict(context or {})
        goal = dict(goal)
        scene_id = (
            str(
                goal.get("scene_id") or start.get("scene_id") or context.get("scene_id") or ""
            ).strip()
            or None
        )
        nav2_environment = self.path_client.inspect_environment()
        candidate_validation: dict[str, Any] | None = None
        if str(goal.get("goal_type") or "").strip().lower() == "object":
            candidates = [
                dict(candidate)
                for candidate in goal.get("object_approach_candidates") or []
                if isinstance(candidate, dict)
            ]
            selected_candidate = goal.get("selected_object_approach")
            if not candidates and isinstance(selected_candidate, dict):
                candidates = [dict(selected_candidate)]
            start_pose = start.get("pose")
            if candidates and isinstance(start_pose, dict):
                preliminary_vertical_axis = self._resolve_vertical_axis(
                    start=start,
                    goal=goal,
                    context=context,
                )
                preliminary_trav_map_filename = self._resolve_nav2_trav_map_filename(
                    start=start,
                    goal=goal,
                    context=context,
                )
                candidate_validation = nav2_candidate_validation.validate_object_approach_candidates(
                    self,
                    start=start_pose,
                    candidates=candidates,
                    selected_candidate_id=(
                        str(selected_candidate.get("candidate_id") or "")
                        if isinstance(selected_candidate, dict)
                        else None
                    ),
                    scene_id=scene_id,
                    vertical_axis=preliminary_vertical_axis,
                    nav2_trav_map_filename=preliminary_trav_map_filename,
                    nav2_scene_obstacle_inflation_radius_m=self._nav2_scene_obstacle_inflation_radius_m(),
                    navigation_goal=goal,
                )
                generation_diagnostics = (
                    self._last_object_approach_generation_diagnostics
                    if self._last_object_approach_generation_key
                    == (scene_id, str(goal.get("object_id") or ""))
                    else {}
                )
                candidate_validation = {
                    **generation_diagnostics,
                    **candidate_validation,
                    "candidate_count_before_nav2_validation": len(candidates),
                }
                executable = candidate_validation.get("selected_candidate")
                if isinstance(executable, dict):
                    semantic_selection_source = executable.get("selection_source")
                    executable = {
                        **executable,
                        "semantic_selection_source": semantic_selection_source,
                        "selection_source": "nav2_candidate_validation",
                        "nav2_validation_status": "executable",
                        "nav2_path_length_m": (
                            candidate_validation.get("selected_candidate_result") or {}
                        ).get("path_length_m"),
                        "runtime_map_revision": candidate_validation.get("runtime_map_revision"),
                        "runtime_overlay_signature": candidate_validation.get(
                            "runtime_overlay_signature"
                        ),
                    }
                    candidate_validation["selected_candidate"] = dict(executable)
                    goal["selected_object_approach"] = executable
                    goal["object_approach_candidates"] = candidates
                    goal.pop("object_approach_selection_failed", None)
                    goal.pop("object_approach_selection_failure_reason", None)
                else:
                    return nav2_candidate_validation.candidate_validation_failure_plan(
                        start=start,
                        goal=goal,
                        scene_id=scene_id,
                        vertical_axis=preliminary_vertical_axis,
                        nav2_environment=nav2_environment,
                        validation=candidate_validation,
                        nav2_trav_map_filename=preliminary_trav_map_filename,
                        nav2_profile=self.version_profile,
                    )
        semantic_plan = nav2_semantic_plan.normalize_semantic_plan(
            self.semantic_backend.plan_path(start=start, goal=goal, context=context)
        )
        if candidate_validation is not None:
            semantic_plan["nav2_candidate_validation"] = candidate_validation
            semantic_diagnostics = dict(semantic_plan.get("object_approach_diagnostics") or {})
            semantic_diagnostics.update(
                {
                    key: candidate_validation.get(key)
                    for key in (
                        "candidate_count_before_clearance",
                        "candidate_count_after_point_clearance",
                        "candidate_count_after_graph_handoff",
                        "candidate_count_after_segment_clearance",
                        "candidate_count_after_portal_filter",
                        "candidate_count_before_nav2_validation",
                        "candidate_count_submitted_to_nav2",
                        "candidate_count_nav2_executable",
                    )
                    if candidate_validation.get(key) is not None
                }
            )
            semantic_diagnostics["nav2_candidate_results"] = list(
                candidate_validation.get("candidate_results") or []
            )
            semantic_plan["object_approach_diagnostics"] = semantic_diagnostics
            if isinstance(goal.get("selected_object_approach"), dict):
                semantic_plan["selected_object_approach"] = dict(goal["selected_object_approach"])
        vertical_axis = self._resolve_vertical_axis(
            start=start,
            goal=goal,
            context=context,
            semantic_plan=semantic_plan,
        )
        nav2_trav_map_filename = self._resolve_nav2_trav_map_filename(
            start=start,
            goal=goal,
            context=context,
        )
        global_waypoints = self._normalize_waypoints(semantic_plan.get("waypoints"))
        dense_waypoints = self._normalize_waypoints(semantic_plan.get("dense_waypoints"))
        if not semantic_plan.get("found", True) or not global_waypoints:
            fallback = dict(semantic_plan)
            fallback.setdefault("planner", "hovsg_global_nav2_local")
            fallback["path_backend"] = "semantic_global_only"
            fallback["requested_planner"] = "nav2_compute_path_to_pose"
            fallback["scene_id"] = fallback.get("scene_id") or scene_id
            fallback["vertical_axis"] = fallback.get("vertical_axis") or vertical_axis
            fallback["nav2_profile"] = self.version_profile
            fallback["nav2_environment"] = nav2_environment
            fallback["waypoint_tracking_mode"] = "global_local_hybrid"
            fallback["waypoint_scope"] = "dynamic_local_segment"
            fallback["object_approach_candidates"] = semantic_plan.get(
                "object_approach_candidates", []
            )
            fallback["selected_object_approach"] = semantic_plan.get("selected_object_approach")
            return fallback

        start_pose = start.get("pose")
        current_region = self._resolve_current_region(start=start, context=context)
        pending_portal_egress_anchor = self._active_pending_portal_egress_anchor(
            scene_id=scene_id,
            pose=start_pose,
        )
        active_global_waypoint_index = self._resolve_active_global_waypoint_index(
            start_pose=start_pose,
            start_orientation=start.get("orientation"),
            global_waypoints=global_waypoints,
            current_region=current_region,
            vertical_axis=vertical_axis,
            previous_index=self._resolve_previous_global_waypoint_index(context=context),
        )
        if (
            active_global_waypoint_index >= len(global_waypoints)
            and pending_portal_egress_anchor is None
        ):
            final_waypoint = dict(global_waypoints[-1])
            execution_goal, transition_anchor = self._resolve_local_execution_goal(
                semantic_plan=semantic_plan,
                global_waypoints=global_waypoints,
                active_index=max(0, len(global_waypoints) - 1),
                current_region=current_region,
            )
            return {
                "planner": "hovsg_global_nav2_local",
                "path_backend": "global_goal_reached",
                "scene_id": scene_id,
                "vertical_axis": vertical_axis,
                "start": start,
                "goal": goal,
                "waypoints": [],
                "path_nodes": [],
                "path_cost": 0.0,
                "found": True,
                "frame_id": self.frame_id,
                "planner_id": self.planner_id,
                "action_name": self.action_name,
                "requested_planner": "nav2_compute_path_to_pose",
                "nav2_profile": self.version_profile,
                "nav2_environment": nav2_environment,
                "global_plan": semantic_plan,
                "global_waypoints": global_waypoints,
                "dense_waypoints": dense_waypoints,
                "global_waypoint_index": active_global_waypoint_index,
                "local_goal": final_waypoint,
                "execution_goal": execution_goal,
                "transition_anchor": transition_anchor,
                "waypoint_tracking_mode": "global_local_hybrid",
                "waypoint_scope": "dynamic_local_segment",
                "object_approach_candidates": semantic_plan.get("object_approach_candidates", []),
                "selected_object_approach": semantic_plan.get("selected_object_approach"),
            }

        if active_global_waypoint_index >= len(global_waypoints):
            active_global_waypoint_index = max(0, len(global_waypoints) - 1)

        local_goal = dict(global_waypoints[active_global_waypoint_index])
        execution_goal, transition_anchor = self._resolve_local_execution_goal(
            semantic_plan=semantic_plan,
            global_waypoints=global_waypoints,
            active_index=active_global_waypoint_index,
            current_region=current_region,
        )
        if isinstance(transition_anchor, dict) and nav2_portal_safety.has_portal_frame(
            transition_anchor
        ):
            transition_anchor = dict(transition_anchor)
            transition_anchor["portal_required_egress_depth_m"] = self.portal_egress_depth_m
        transition_anchor, doorway_corridor = self._refine_transition_anchor_with_traversability(
            scene_id=scene_id,
            vertical_axis=vertical_axis,
            current_region=current_region,
            transition_anchor=transition_anchor,
            execution_goal=execution_goal,
            nav2_trav_map_filename=nav2_trav_map_filename,
        )
        if pending_portal_egress_anchor is None:
            self._remember_pending_portal_egress_anchor(
                scene_id=scene_id,
                anchor=transition_anchor,
            )
            pending_portal_egress_anchor = self._active_pending_portal_egress_anchor(
                scene_id=scene_id,
                pose=start_pose,
            )
        if pending_portal_egress_anchor is not None:
            transition_anchor = dict(pending_portal_egress_anchor)
            transition_anchor["portal_egress_guard_persisted"] = True
            doorway_corridor = None
        pre_transition_stage = self._is_pre_transition_stage(
            current_region=current_region,
            execution_goal=execution_goal,
            transition_anchor=transition_anchor,
        )
        if pre_transition_stage and self._pose_is_on_transition_target_side(
            pose=start_pose,
            transition_anchor=transition_anchor,
        ):
            pre_transition_stage = False
        nav2_compute_goal = self._resolve_nav2_compute_goal(
            semantic_plan=semantic_plan,
            current_region=current_region,
            execution_goal=execution_goal,
            transition_anchor=transition_anchor,
        )
        if not pre_transition_stage and not self._has_explicit_nav2_compute_goal(
            semantic_plan=semantic_plan
        ):
            nav2_compute_goal = dict(execution_goal)
        if doorway_corridor is not None and pre_transition_stage:
            nav2_compute_goal = dict(doorway_corridor["midpoint"])
        nav2_scene_obstacle_inflation_radius_m = self._nav2_scene_obstacle_inflation_radius_m()
        local_path_plan = nav2_local_path_planning.plan_local_path(
            self,
            scene_id=scene_id,
            start_pose=start_pose,
            vertical_axis=vertical_axis,
            current_region=current_region,
            execution_goal=execution_goal,
            transition_anchor=transition_anchor,
            nav2_compute_goal=nav2_compute_goal,
            doorway_corridor=doorway_corridor,
            nav2_trav_map_filename=nav2_trav_map_filename,
            nav2_scene_obstacle_inflation_radius_m=nav2_scene_obstacle_inflation_radius_m,
            pre_transition_stage=pre_transition_stage,
        )
        local_waypoints = list(local_path_plan["local_waypoints"])
        local_backend = str(local_path_plan["local_backend"])
        nav2_error = local_path_plan["nav2_error"]
        nav2_empty_path_reason = local_path_plan.get("nav2_empty_path_reason")
        dynamic_map_update = local_path_plan.get("dynamic_map_update")
        nav2_raw_path_length = int(local_path_plan["nav2_raw_path_length"])
        nav2_raw_path_points = list(local_path_plan["nav2_raw_path_points"])
        nav2_path_points = list(local_path_plan["nav2_path_points"])
        dense_waypoint_index = 0
        nav2_cache_reused = False
        nav2_path_clipped_for_clearance = bool(local_path_plan["nav2_path_clipped_for_clearance"])
        if (
            nav2_path_clipped_for_clearance
            and doorway_corridor is not None
            and pre_transition_stage
        ):
            local_waypoints = []

        if not local_waypoints and not nav2_path_clipped_for_clearance:
            cached_segment = self._reuse_cached_local_segment(
                scene_id=scene_id,
                vertical_axis=vertical_axis,
                active_global_waypoint_index=active_global_waypoint_index,
                start_pose=start_pose,
                local_goal=local_goal,
                execution_goal=execution_goal,
                nav2_compute_goal=nav2_compute_goal,
                error_text=nav2_error or "",
            )
            if cached_segment is not None:
                local_waypoints = cached_segment["waypoints"]
                nav2_raw_path_points = cached_segment["nav2_raw_path_points"]
                nav2_path_points = cached_segment["nav2_path_points"]
                nav2_raw_path_length = int(cached_segment["nav2_raw_path_length"])
                dense_waypoint_index = int(cached_segment["dense_waypoint_index"])
                local_backend = "nav2_local"
                nav2_error = cached_segment.get("nav2_error")
                nav2_cache_reused = bool(cached_segment.get("nav2_cache_reused"))

        if not local_waypoints:
            if self.strict and nav2_error is not None:
                return {
                    "planner": "hovsg_global_nav2_local",
                    "path_backend": "nav2_unavailable",
                    "scene_id": scene_id,
                    "vertical_axis": vertical_axis,
                    "start": start,
                    "goal": goal,
                    "waypoints": [],
                    "path_nodes": [],
                    "found": False,
                    "frame_id": self.frame_id,
                    "planner_id": self.planner_id,
                    "action_name": self.action_name,
                    "requested_planner": "nav2_compute_path_to_pose",
                    "nav2_profile": self.version_profile,
                    "nav2_environment": nav2_environment,
                    "nav2_raw_path_length": nav2_raw_path_length,
                    "nav2_raw_path_points": nav2_raw_path_points,
                    "nav2_path_points": nav2_path_points,
                    "nav2_error": nav2_error,
                    "nav2_empty_path_reason": nav2_empty_path_reason,
                    "dynamic_map_update": dynamic_map_update,
                    "nav2_cache_reused": nav2_cache_reused,
                    "nav2_path_clipped_for_clearance": nav2_path_clipped_for_clearance,
                    "nav2_scene_obstacle_inflation_radius_m": nav2_scene_obstacle_inflation_radius_m,
                    "nav2_trav_map_filename": nav2_trav_map_filename,
                    "global_plan": semantic_plan,
                    "global_waypoints": global_waypoints,
                    "dense_waypoints": dense_waypoints,
                    "global_waypoint_index": active_global_waypoint_index,
                    "waypoint_tracking_mode": "global_local_hybrid",
                    "waypoint_scope": "dynamic_local_segment",
                    "local_goal": local_goal,
                    "execution_goal": execution_goal,
                    "nav2_compute_goal": nav2_compute_goal,
                    "transition_anchor": transition_anchor,
                    "object_approach_candidates": semantic_plan.get(
                        "object_approach_candidates", []
                    ),
                    "selected_object_approach": semantic_plan.get("selected_object_approach"),
                }
            if nav2_error == "room_exit_path" and not (
                doorway_corridor is not None and pre_transition_stage
            ):
                return {
                    "planner": "hovsg_global_nav2_local",
                    "path_backend": "room_exit_path_unavailable",
                    "scene_id": scene_id,
                    "vertical_axis": vertical_axis,
                    "start": start,
                    "goal": goal,
                    "waypoints": [],
                    "path_nodes": [],
                    "found": False,
                    "frame_id": self.frame_id,
                    "planner_id": self.planner_id,
                    "action_name": self.action_name,
                    "requested_planner": "nav2_compute_path_to_pose",
                    "nav2_profile": self.version_profile,
                    "nav2_environment": nav2_environment,
                    "nav2_raw_path_length": nav2_raw_path_length,
                    "nav2_raw_path_points": nav2_raw_path_points,
                    "nav2_path_points": nav2_path_points,
                    "nav2_error": nav2_error,
                    "nav2_empty_path_reason": nav2_empty_path_reason,
                    "dynamic_map_update": dynamic_map_update,
                    "nav2_cache_reused": nav2_cache_reused,
                    "nav2_path_clipped_for_clearance": nav2_path_clipped_for_clearance,
                    "nav2_scene_obstacle_inflation_radius_m": nav2_scene_obstacle_inflation_radius_m,
                    "nav2_trav_map_filename": nav2_trav_map_filename,
                    "global_plan": semantic_plan,
                    "global_waypoints": global_waypoints,
                    "dense_waypoints": dense_waypoints,
                    "global_waypoint_index": active_global_waypoint_index,
                    "waypoint_tracking_mode": "global_local_hybrid",
                    "waypoint_scope": "dynamic_local_segment",
                    "local_goal": local_goal,
                    "execution_goal": execution_goal,
                    "nav2_compute_goal": nav2_compute_goal,
                    "transition_anchor": transition_anchor,
                    "object_approach_candidates": semantic_plan.get(
                        "object_approach_candidates", []
                    ),
                    "selected_object_approach": semantic_plan.get("selected_object_approach"),
                }
            if (
                nav2_error is not None
                and self._is_empty_nav2_path_error(nav2_error)
                and pre_transition_stage
                and transition_anchor is not None
                and doorway_corridor is None
            ):
                return self._portal_path_unavailable_plan(
                    scene_id=scene_id,
                    vertical_axis=vertical_axis,
                    start=start,
                    goal=goal,
                    nav2_environment=nav2_environment,
                    nav2_error=nav2_error,
                    nav2_raw_path_length=nav2_raw_path_length,
                    nav2_raw_path_points=nav2_raw_path_points,
                    nav2_path_points=nav2_path_points,
                    nav2_cache_reused=nav2_cache_reused,
                    nav2_path_clipped_for_clearance=nav2_path_clipped_for_clearance,
                    nav2_scene_obstacle_inflation_radius_m=nav2_scene_obstacle_inflation_radius_m,
                    nav2_trav_map_filename=nav2_trav_map_filename,
                    semantic_plan=semantic_plan,
                    global_waypoints=global_waypoints,
                    dense_waypoints=dense_waypoints,
                    active_global_waypoint_index=active_global_waypoint_index,
                    local_goal=local_goal,
                    execution_goal=execution_goal,
                    nav2_compute_goal=nav2_compute_goal,
                    transition_anchor=transition_anchor,
                    dense_waypoint_index=dense_waypoint_index,
                )
            local_waypoints = self._doorway_corridor_fallback_waypoints(
                start_pose=start_pose,
                execution_goal=execution_goal,
                doorway_corridor=doorway_corridor,
                scene_id=scene_id,
                vertical_axis=vertical_axis,
                nav2_trav_map_filename=nav2_trav_map_filename,
            )
            if local_waypoints:
                local_backend = "doorway_corridor_checked_fallback"
            if not local_waypoints:
                local_waypoints = self._semantic_dense_local_waypoints(
                    start_pose=start_pose,
                    target=execution_goal,
                    dense_waypoints=dense_waypoints,
                    vertical_axis=vertical_axis,
                    previous_dense_index=self._resolve_previous_dense_waypoint_index(
                        context=context
                    ),
                )
                local_waypoints = self._filter_waypoints_for_local_clearance(
                    start_pose=start_pose,
                    vertical_axis=vertical_axis,
                    waypoints=local_waypoints,
                    scene_id=scene_id,
                    nav2_trav_map_filename=nav2_trav_map_filename,
                    navigation_goal=execution_goal,
                )
                if local_waypoints:
                    local_backend = "semantic_dense_local_fallback"
            dense_waypoint_index = self._resolve_dense_waypoint_index(
                local_waypoints=local_waypoints, dense_waypoints=dense_waypoints
            )

        if local_backend == "nav2_local" and local_waypoints:
            self._cache_local_segment(
                scene_id=scene_id,
                vertical_axis=vertical_axis,
                active_global_waypoint_index=active_global_waypoint_index,
                local_goal=local_goal,
                execution_goal=execution_goal,
                nav2_compute_goal=nav2_compute_goal,
                waypoints=local_waypoints,
                nav2_raw_path_points=nav2_raw_path_points,
                nav2_path_points=nav2_path_points,
                nav2_raw_path_length=nav2_raw_path_length,
                dense_waypoint_index=dense_waypoint_index,
            )

        if not local_waypoints:
            target_only_waypoints = self._filter_waypoints_for_local_clearance(
                start_pose=start_pose,
                vertical_axis=vertical_axis,
                waypoints=[dict(local_goal)],
                scene_id=scene_id,
                nav2_trav_map_filename=nav2_trav_map_filename,
                navigation_goal=execution_goal,
            )
            if target_only_waypoints:
                local_backend = "target_only_fallback"
                local_waypoints = target_only_waypoints
                dense_waypoint_index = 0

        if not local_waypoints:
            return {
                "planner": "hovsg_global_nav2_local",
                "path_backend": "fallback_blocked_for_clearance",
                "scene_id": scene_id,
                "vertical_axis": vertical_axis,
                "start": start,
                "goal": goal,
                "waypoints": [],
                "path_nodes": [],
                "path_cost": 0.0,
                "found": False,
                "frame_id": self.frame_id,
                "planner_id": self.planner_id,
                "action_name": self.action_name,
                "requested_planner": "nav2_compute_path_to_pose",
                "nav2_profile": self.version_profile,
                "nav2_environment": nav2_environment,
                "nav2_raw_path_length": nav2_raw_path_length,
                "nav2_raw_path_points": nav2_raw_path_points,
                "nav2_path_points": nav2_path_points,
                "nav2_error": nav2_error,
                "nav2_empty_path_reason": nav2_empty_path_reason,
                "dynamic_map_update": dynamic_map_update,
                "nav2_cache_reused": nav2_cache_reused,
                "nav2_path_clipped_for_clearance": nav2_path_clipped_for_clearance,
                "nav2_scene_obstacle_inflation_radius_m": nav2_scene_obstacle_inflation_radius_m,
                "nav2_trav_map_filename": nav2_trav_map_filename,
                "global_plan": semantic_plan,
                "global_waypoints": global_waypoints,
                "dense_waypoints": dense_waypoints,
                "global_waypoint_index": active_global_waypoint_index,
                "local_goal": local_goal,
                "execution_goal": execution_goal,
                "nav2_compute_goal": nav2_compute_goal,
                "transition_anchor": transition_anchor,
                "doorway_corridor": doorway_corridor,
                "dense_waypoint_index": dense_waypoint_index,
                "waypoint_tracking_mode": "global_local_hybrid",
                "waypoint_scope": "dynamic_local_segment",
                "object_approach_candidates": semantic_plan.get("object_approach_candidates", []),
                "selected_object_approach": semantic_plan.get("selected_object_approach"),
            }

        return {
            "planner": "hovsg_global_nav2_local",
            "path_backend": local_backend,
            "scene_id": scene_id,
            "vertical_axis": vertical_axis,
            "start": start,
            "goal": goal,
            "waypoints": local_waypoints,
            "path_nodes": [],
            "path_cost": self._path_cost(local_waypoints, vertical_axis=vertical_axis),
            "found": True,
            "frame_id": self.frame_id,
            "planner_id": self.planner_id,
            "action_name": self.action_name,
            "requested_planner": "nav2_compute_path_to_pose",
            "nav2_profile": self.version_profile,
            "nav2_environment": nav2_environment,
            "nav2_raw_path_length": nav2_raw_path_length,
            "nav2_raw_path_points": nav2_raw_path_points,
            "nav2_path_points": nav2_path_points,
            "nav2_error": nav2_error,
            "nav2_empty_path_reason": nav2_empty_path_reason,
            "dynamic_map_update": dynamic_map_update,
            "nav2_cache_reused": nav2_cache_reused,
            "nav2_path_clipped_for_clearance": nav2_path_clipped_for_clearance,
            "nav2_scene_obstacle_inflation_radius_m": nav2_scene_obstacle_inflation_radius_m,
            "nav2_trav_map_filename": nav2_trav_map_filename,
            "global_plan": semantic_plan,
            "global_waypoints": global_waypoints,
            "dense_waypoints": dense_waypoints,
            "global_waypoint_index": active_global_waypoint_index,
            "local_goal": local_goal,
            "execution_goal": execution_goal,
            "nav2_compute_goal": nav2_compute_goal,
            "transition_anchor": transition_anchor,
            "doorway_corridor": doorway_corridor,
            "dense_waypoint_index": dense_waypoint_index,
            "waypoint_tracking_mode": "global_local_hybrid",
            "waypoint_scope": "dynamic_local_segment",
            "object_approach_candidates": semantic_plan.get("object_approach_candidates", []),
            "selected_object_approach": semantic_plan.get("selected_object_approach"),
        }

    def _nav2_scene_obstacle_inflation_radius_m(self) -> float:
        if self.local_path_clearance_radius_m <= 0.0:
            return 0.0
        return max(
            0.0,
            self.local_path_clearance_radius_m - NAV2_RUNTIME_GUARANTEED_CLEARANCE_RADIUS_M,
        )

    def _local_path_post_clearance_radius_m(self) -> float:
        return self._nav2_scene_obstacle_inflation_radius_m()

    @staticmethod
    def _is_empty_nav2_path_error(nav2_error: str) -> bool:
        return str(nav2_error or "").strip().lower() == "empty_path"

    def _portal_path_unavailable_plan(
        self,
        *,
        scene_id: str | None,
        vertical_axis: str,
        start: dict[str, Any],
        goal: dict[str, Any],
        nav2_environment: dict[str, Any],
        nav2_error: str,
        nav2_raw_path_length: int,
        nav2_raw_path_points: list[dict[str, float]],
        nav2_path_points: list[dict[str, float]],
        nav2_cache_reused: bool,
        nav2_path_clipped_for_clearance: bool,
        nav2_scene_obstacle_inflation_radius_m: float,
        nav2_trav_map_filename: str | None,
        semantic_plan: dict[str, Any],
        global_waypoints: list[dict[str, Any]],
        dense_waypoints: list[dict[str, Any]],
        active_global_waypoint_index: int,
        local_goal: dict[str, Any],
        execution_goal: dict[str, Any],
        nav2_compute_goal: dict[str, Any],
        transition_anchor: dict[str, Any],
        dense_waypoint_index: int,
    ) -> dict[str, Any]:
        return {
            "planner": "hovsg_global_nav2_local",
            "path_backend": "portal_path_unavailable",
            "scene_id": scene_id,
            "vertical_axis": vertical_axis,
            "start": start,
            "goal": goal,
            "waypoints": [],
            "path_nodes": [],
            "path_cost": 0.0,
            "found": False,
            "frame_id": self.frame_id,
            "planner_id": self.planner_id,
            "action_name": self.action_name,
            "requested_planner": "nav2_compute_path_to_pose",
            "nav2_profile": self.version_profile,
            "nav2_environment": nav2_environment,
            "nav2_raw_path_length": nav2_raw_path_length,
            "nav2_raw_path_points": nav2_raw_path_points,
            "nav2_path_points": nav2_path_points,
            "nav2_error": nav2_error,
            "nav2_cache_reused": nav2_cache_reused,
            "nav2_path_clipped_for_clearance": nav2_path_clipped_for_clearance,
            "nav2_scene_obstacle_inflation_radius_m": nav2_scene_obstacle_inflation_radius_m,
            "nav2_trav_map_filename": nav2_trav_map_filename,
            "global_plan": semantic_plan,
            "global_waypoints": global_waypoints,
            "dense_waypoints": dense_waypoints,
            "global_waypoint_index": active_global_waypoint_index,
            "local_goal": local_goal,
            "execution_goal": execution_goal,
            "nav2_compute_goal": nav2_compute_goal,
            "transition_anchor": transition_anchor,
            "doorway_corridor": None,
            "dense_waypoint_index": dense_waypoint_index,
            "waypoint_tracking_mode": "global_local_hybrid",
            "waypoint_scope": "dynamic_local_segment",
            "failure_type": "portal_path_unavailable",
            "object_approach_candidates": semantic_plan.get("object_approach_candidates", []),
            "selected_object_approach": semantic_plan.get("selected_object_approach"),
        }

    def _runtime_door_map_overlays(
        self, scene_id: str | None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
        backend = getattr(self, "semantic_backend", None)
        if backend is None or not scene_id:
            return [], [], ""
        try:
            from ..hovsg import door_gating as hovsg_door_gating
            from ..hovsg import runtime_state as hovsg_runtime_state

            door_signature = hovsg_runtime_state.door_signature(backend, scene_id)
            if not door_signature:
                return [], [], ""
            return (
                hovsg_door_gating.runtime_door_obstacles(backend, scene_id),
                hovsg_door_gating.open_door_clear_regions(backend, scene_id),
                door_signature,
            )
        except Exception:
            return [], [], ""

    def _runtime_navigation_obstacles(
        self,
        scene_id: str | None,
        *,
        navigation_goal: dict[str, Any] | None = None,
        vertical_axis: str = "z",
        trav_map_filename: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
        del vertical_axis
        obstacles, door_clear_regions, object_clear_regions, signature, _ = (
            self._runtime_navigation_overlay_details(
                scene_id,
                navigation_goal=navigation_goal,
                trav_map_filename=trav_map_filename,
            )
        )
        return obstacles, [*door_clear_regions, *object_clear_regions], signature

    def _runtime_navigation_overlay_details(
        self,
        scene_id: str | None,
        *,
        navigation_goal: dict[str, Any] | None = None,
        trav_map_filename: str | None = None,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        str,
        str,
    ]:
        door_obstacles, door_clear_regions, door_signature = self._runtime_door_map_overlays(
            scene_id
        )
        object_obstacles: list[dict[str, Any]] = []
        object_clear_regions: list[dict[str, Any]] = []
        object_signature = ""
        sensor_obstacles: list[dict[str, Any]] = []
        sensor_signature = ""
        try:
            from ..hovsg import object_gating as hovsg_object_gating

            object_overlay = hovsg_object_gating.runtime_object_map_overlays(
                getattr(self, "semantic_backend", None),
                scene_id,
                navigation_goal=navigation_goal,
                include_unchanged=_uses_objectless_base_map(
                    trav_map_filename or self.trav_map_filename
                ),
            )
            object_obstacles = list(object_overlay.get("obstacles") or [])
            object_clear_regions = list(object_overlay.get("clear_regions") or [])
            object_signature = str(object_overlay.get("signature") or "")
        except Exception:
            pass

        try:
            from ..hovsg import sensor_gating as hovsg_sensor_gating

            sensor_overlay = hovsg_sensor_gating.runtime_sensor_map_overlays(
                getattr(self, "semantic_backend", None),
                scene_id,
            )
            sensor_obstacles = list(sensor_overlay.get("obstacles") or [])
            sensor_signature = str(sensor_overlay.get("signature") or "")
        except Exception:
            pass

        clear_regions = list(door_clear_regions)
        try:
            from ..hovsg import door_gating as hovsg_door_gating

            portal_region = hovsg_door_gating.open_portal_clear_region(
                navigation_goal,
                normal_padding_m=self.local_path_clearance_radius_m,
            )
            clear_regions = hovsg_door_gating.prefer_canonical_portal_clear_region(
                clear_regions,
                portal_region,
            )
        except Exception:
            pass
        signature_parts = [
            part for part in (door_signature, object_signature, sensor_signature) if part
        ]
        signature = "|".join(signature_parts)
        return (
            [*door_obstacles, *object_obstacles, *sensor_obstacles],
            clear_regions,
            object_clear_regions,
            signature,
            door_signature,
        )

    @staticmethod
    def _normalize_polygon_points(polygon: Any) -> list[tuple[float, float]]:
        normalized: list[tuple[float, float]] = []
        if not isinstance(polygon, list):
            return normalized
        for point in polygon:
            if not isinstance(point, (tuple, list)) or len(point) < 2:
                continue
            try:
                normalized.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError):
                continue
        return normalized

    def _load_stamped_traversability_grid(
        self,
        *,
        scene_id: str,
        map_resolution: float,
        trav_map_filename: str | None,
        navigation_goal: dict[str, Any] | None = None,
        vertical_axis: str = "z",
    ) -> dict[str, Any]:
        del vertical_axis
        from .nav2_runtime_bridge import (
            clear_exported_door_artifacts_from_map_spec,
            clear_regions_from_map_spec,
            load_scene_traversability_grid,
            stamp_obstacles_into_map_spec,
        )

        map_spec = load_scene_traversability_grid(
            scene_id=scene_id,
            map_resolution=map_resolution,
            trav_map_filename=trav_map_filename,
        )
        obstacles, door_clear_regions, object_clear_regions, _, door_signature = (
            self._runtime_navigation_overlay_details(
                scene_id,
                navigation_goal=navigation_goal,
                trav_map_filename=trav_map_filename,
            )
        )
        if door_signature:
            map_spec = clear_exported_door_artifacts_from_map_spec(
                map_spec,
                scene_id=scene_id,
                map_resolution=map_resolution,
            )
        if object_clear_regions:
            from .nav2_runtime_bridge import (
                clear_exported_object_artifacts_from_map_spec,
            )

            map_spec = clear_exported_object_artifacts_from_map_spec(
                map_spec,
                scene_id=scene_id,
                map_resolution=map_resolution,
                regions=object_clear_regions,
            )
        if door_clear_regions:
            map_spec = clear_regions_from_map_spec(map_spec, door_clear_regions)
        if obstacles:
            map_spec = stamp_obstacles_into_map_spec(map_spec, obstacles)
        return map_spec

    def _compute_nav2_path_response(
        self,
        *,
        scene_id: str | None,
        start_xy: dict[str, float],
        goal_xy: dict[str, float],
        nav2_trav_map_filename: str | None,
        nav2_scene_obstacle_inflation_radius_m: float,
        navigation_goal: dict[str, Any] | None = None,
        vertical_axis: str = "z",
    ) -> dict[str, Any]:
        del vertical_axis
        if scene_id and isinstance(self.path_client, SubprocessNav2ComputePathClient):
            from .nav2_runtime_bridge import get_nav2_runtime_bridge_client

            runtime_bridge = get_nav2_runtime_bridge_client(
                version_profile=self.version_profile,
                frame_id=self.frame_id,
                action_name=self.action_name,
            )
            (
                obstacles,
                door_clear_regions,
                object_clear_regions,
                obstacles_signature,
                door_signature,
            ) = self._runtime_navigation_overlay_details(
                scene_id,
                navigation_goal=navigation_goal,
                trav_map_filename=nav2_trav_map_filename,
            )
            self._last_runtime_overlay_signature = str(obstacles_signature or "")
            self._last_runtime_overlay_geometry = [
                dict(obstacle) for obstacle in obstacles if isinstance(obstacle, dict)
            ]
            map_response = runtime_bridge.ensure_scene(
                scene_id=scene_id,
                trav_map_filename=nav2_trav_map_filename,
                obstacle_inflation_radius_m=nav2_scene_obstacle_inflation_radius_m,
                obstacles=obstacles,
                clear_regions=door_clear_regions,
                clear_exported_door_artifacts=bool(door_signature),
                clear_exported_object_artifacts=bool(object_clear_regions),
                object_clear_regions=object_clear_regions,
                obstacles_signature=obstacles_signature,
            )
            if isinstance(map_response, dict) and str(map_response.get("status") or "ok") != "ok":
                raise RuntimeError(str(map_response.get("error") or "nav2_map_update_failed"))
            if isinstance(map_response, dict):
                dynamic_map_update = map_response.get("dynamic_map_update")
                map_revision = map_response.get("map_revision")
                if isinstance(dynamic_map_update, dict):
                    self._last_dynamic_map_update = dict(dynamic_map_update)
                elif map_revision is not None:
                    self._last_dynamic_map_update = {
                        "map_revision": map_revision,
                        "reused": bool(map_response.get("reused")),
                    }
            runtime_bridge.set_pose(
                pose_xy={"x": start_xy["x"], "y": start_xy["y"]},
                yaw=float(start_xy.get("yaw", 0.0)),
            )
            return runtime_bridge.compute_path(
                start_xy=start_xy,
                goal_xy=goal_xy,
                planner_id=self.planner_id,
                timeout_s=self.timeout_s,
            )
        return self.path_client.compute_path(
            start_xy=start_xy,
            goal_xy=goal_xy,
            frame_id=self.frame_id,
            planner_id=self.planner_id,
            timeout_s=self.timeout_s,
        )

    def _finalize_fallback(
        self,
        *,
        start: dict[str, Any],
        goal: dict[str, Any],
        context: dict[str, Any],
        scene_id: str | None,
        vertical_axis: str,
        fallback_reason: str,
    ) -> dict[str, Any]:
        fallback_plan = self._semantic_fallback(
            start=start,
            goal=goal,
            context=context,
            scene_id=scene_id,
            vertical_axis=vertical_axis,
            reason=fallback_reason,
        )
        return fallback_plan

    def _cache_local_segment(
        self,
        *,
        scene_id: str | None,
        vertical_axis: str,
        active_global_waypoint_index: int,
        local_goal: dict[str, Any],
        execution_goal: dict[str, Any],
        nav2_compute_goal: dict[str, Any],
        waypoints: list[dict[str, Any]],
        nav2_raw_path_points: list[dict[str, float]],
        nav2_path_points: list[dict[str, float]],
        nav2_raw_path_length: int,
        dense_waypoint_index: int,
    ) -> None:
        cached_segment = nav2_cache_state.build_cached_local_segment(
            scene_id=scene_id,
            vertical_axis=vertical_axis,
            active_global_waypoint_index=active_global_waypoint_index,
            local_goal=local_goal,
            execution_goal=execution_goal,
            nav2_compute_goal=nav2_compute_goal,
            waypoints=waypoints,
            nav2_raw_path_points=nav2_raw_path_points,
            nav2_path_points=nav2_path_points,
            nav2_raw_path_length=nav2_raw_path_length,
            dense_waypoint_index=dense_waypoint_index,
        )
        _, _, obstacle_signature = self._runtime_navigation_obstacles(
            scene_id,
            navigation_goal=execution_goal,
            vertical_axis=vertical_axis,
        )
        cached_segment["navigation_obstacles_signature"] = obstacle_signature
        self._last_successful_local_segment = cached_segment

    def _reuse_cached_local_segment(
        self,
        *,
        scene_id: str | None,
        vertical_axis: str,
        active_global_waypoint_index: int,
        start_pose: dict[str, Any] | None,
        local_goal: dict[str, Any],
        execution_goal: dict[str, Any],
        nav2_compute_goal: dict[str, Any],
        error_text: str,
    ) -> dict[str, Any] | None:
        cached = self._last_successful_local_segment
        if isinstance(cached, dict):
            _, _, current_obstacle_signature = self._runtime_navigation_obstacles(
                scene_id,
                navigation_goal=execution_goal,
                vertical_axis=vertical_axis,
            )
            if str(cached.get("navigation_obstacles_signature", "")) != str(
                current_obstacle_signature
            ):
                return None
        return nav2_cache_state.reuse_matching_local_segment(
            cached_segment=self._last_successful_local_segment,
            scene_id=scene_id,
            vertical_axis=vertical_axis,
            active_global_waypoint_index=active_global_waypoint_index,
            start_pose=start_pose,
            local_goal=local_goal,
            execution_goal=execution_goal,
            nav2_compute_goal=nav2_compute_goal,
            error_text=error_text,
            same_waypoint_signature=self._same_waypoint_signature,
            planar_distance=self._planar_distance,
            waypoint_spacing=self.waypoint_spacing,
        )

    @staticmethod
    def _same_waypoint_signature(first: Any, second: Any, *, tolerance: float = 1e-4) -> bool:
        if not isinstance(first, dict) or not isinstance(second, dict):
            return False
        for axis in ("x", "y", "z"):
            try:
                if abs(float(first[axis]) - float(second[axis])) > tolerance:
                    return False
            except (KeyError, TypeError, ValueError):
                return False
        for key in ("room_id", "room_name", "waypoint_type"):
            first_value = first.get(key)
            second_value = second.get(key)
            if first_value is None or second_value is None:
                continue
            if str(first_value) != str(second_value):
                return False
        for key in ("desired_heading", "portal_desired_heading"):
            first_value = first.get(key)
            second_value = second.get(key)
            if first_value is None or second_value is None:
                continue
            try:
                if abs(float(first_value) - float(second_value)) > tolerance:
                    return False
            except (TypeError, ValueError):
                return False
        return True

    def _semantic_fallback(
        self,
        *,
        start: dict[str, Any],
        goal: dict[str, Any],
        context: dict[str, Any],
        scene_id: str | None,
        vertical_axis: str,
        reason: str,
    ) -> dict[str, Any]:
        if self.strict:
            return {
                "planner": "nav2_compute_path_to_pose",
                "path_backend": "nav2_unavailable",
                "scene_id": scene_id,
                "vertical_axis": vertical_axis,
                "start": start,
                "goal": goal,
                "waypoints": [],
                "path_nodes": [],
                "found": False,
                "frame_id": self.frame_id,
                "planner_id": self.planner_id,
                "action_name": self.action_name,
                "nav2_profile": self.version_profile,
                "nav2_environment": self.path_client.inspect_environment(),
                "reason": reason,
            }

        fallback = dict(self.semantic_backend.plan_path(start=start, goal=goal, context=context))
        fallback["path_backend"] = "semantic_fallback"
        fallback["requested_planner"] = "nav2_compute_path_to_pose"
        fallback["nav2_profile"] = self.version_profile
        fallback["nav2_environment"] = self.path_client.inspect_environment()
        fallback["nav2_error"] = reason
        fallback["scene_id"] = fallback.get("scene_id") or scene_id
        fallback["vertical_axis"] = fallback.get("vertical_axis") or vertical_axis
        return fallback

    @staticmethod
    def _normalize_waypoints(raw_waypoints: Any) -> list[dict[str, Any]]:
        return nav2_semantic_plan.normalize_waypoints(raw_waypoints)

    @staticmethod
    def _waypoint_position(waypoint: dict[str, Any]) -> dict[str, float] | None:
        return nav2_semantic_plan.waypoint_position(waypoint)

    @staticmethod
    def _resolve_current_region(*, start: dict[str, Any], context: dict[str, Any]) -> str | None:
        parameters = context.get("parameters", {})
        map_state = context.get("map_state", {})
        for candidate in (
            start.get("current_region"),
            start.get("current_room"),
            start.get("region"),
            parameters.get("current_region") if isinstance(parameters, dict) else None,
            parameters.get("current_room") if isinstance(parameters, dict) else None,
            parameters.get("region") if isinstance(parameters, dict) else None,
            map_state.get("current_region") if isinstance(map_state, dict) else None,
            map_state.get("current_room") if isinstance(map_state, dict) else None,
            map_state.get("region") if isinstance(map_state, dict) else None,
        ):
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return None

    @staticmethod
    def _resolve_previous_global_waypoint_index(*, context: dict[str, Any]) -> int:
        parameters = context.get("parameters", {})
        map_state = context.get("map_state", {})
        for candidate in (
            parameters.get("global_waypoint_index") if isinstance(parameters, dict) else None,
            map_state.get("global_waypoint_index") if isinstance(map_state, dict) else None,
            parameters.get("active_waypoint_index") if isinstance(parameters, dict) else None,
            map_state.get("active_waypoint_index") if isinstance(map_state, dict) else None,
        ):
            index = Nav2NavigatorAdapter._coerce_index(candidate)
            if index is not None:
                return index
        return 0

    @staticmethod
    def _resolve_previous_dense_waypoint_index(*, context: dict[str, Any]) -> int:
        parameters = context.get("parameters", {})
        map_state = context.get("map_state", {})
        for candidate in (
            parameters.get("dense_waypoint_index") if isinstance(parameters, dict) else None,
            map_state.get("dense_waypoint_index") if isinstance(map_state, dict) else None,
        ):
            index = Nav2NavigatorAdapter._coerce_index(candidate)
            if index is not None:
                return index
        return 0

    def _resolve_active_global_waypoint_index(
        self,
        *,
        start_pose: dict[str, Any] | None,
        start_orientation: dict[str, Any] | None,
        global_waypoints: list[dict[str, Any]],
        current_region: str | None,
        vertical_axis: str,
        previous_index: int,
    ) -> int:
        if not isinstance(start_pose, dict):
            return max(0, min(previous_index, len(global_waypoints)))
        index = max(0, min(previous_index, len(global_waypoints)))
        while index < len(global_waypoints):
            if not self._global_waypoint_reached(
                start_pose=start_pose,
                start_orientation=start_orientation,
                target=global_waypoints[index],
                current_region=current_region,
                vertical_axis=vertical_axis,
                is_final=index == len(global_waypoints) - 1,
            ):
                break
            index += 1
        return index

    def _global_waypoint_reached(
        self,
        *,
        start_pose: dict[str, Any],
        start_orientation: dict[str, Any] | None,
        target: dict[str, Any],
        current_region: str | None,
        vertical_axis: str,
        is_final: bool,
    ) -> bool:
        distance = self._planar_distance(
            first=start_pose,
            second=target,
            vertical_axis=vertical_axis,
        )
        tolerance = (
            self.final_global_waypoint_tolerance if is_final else self.global_waypoint_tolerance
        )
        waypoint_type = str(target.get("waypoint_type", "")).strip().lower()
        waypoint_region = self._normalize_label(target.get("room_name"))
        current_region_norm = self._normalize_label(current_region)

        if waypoint_type == "portal":
            if waypoint_region:
                reached = bool(
                    current_region_norm
                    and current_region_norm == waypoint_region
                    and distance <= tolerance
                )
            else:
                reached = distance <= tolerance
            if reached and nav2_portal_safety.has_portal_frame(target):
                reached = nav2_portal_safety.pose_has_sufficient_egress(
                    pose=start_pose,
                    anchor=target,
                    required_depth_m=self.portal_egress_depth_m,
                )
        else:
            reached = distance <= tolerance
        if not reached:
            return False
        if not is_final:
            return True

        desired_heading = self._waypoint_desired_heading(target)
        if desired_heading is None:
            return True
        yaw = self._orientation_to_yaw(start_orientation)
        if yaw is None:
            return False
        heading_error = self._wrap_angle(desired_heading - yaw)
        return abs(heading_error) <= self.final_global_waypoint_heading_tolerance_rad

    @staticmethod
    def _waypoint_desired_heading(target: dict[str, Any]) -> float | None:
        for key in ("desired_heading", "portal_desired_heading"):
            value = target.get(key)
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _orientation_to_yaw(orientation: dict[str, Any] | None) -> float | None:
        if not isinstance(orientation, dict):
            return None
        try:
            return float(orientation.get("yaw"))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _resolve_local_execution_goal(
        *,
        semantic_plan: dict[str, Any] | None,
        global_waypoints: list[dict[str, Any]],
        active_index: int,
        current_region: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        return nav2_semantic_plan.resolve_local_execution_goal(
            semantic_plan=semantic_plan,
            global_waypoints=global_waypoints,
            active_index=active_index,
            current_region=current_region,
        )

    @staticmethod
    def _resolve_nav2_compute_goal(
        *,
        semantic_plan: dict[str, Any] | None,
        current_region: str | None,
        execution_goal: dict[str, Any],
        transition_anchor: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return nav2_semantic_plan.resolve_nav2_compute_goal(
            semantic_plan=semantic_plan,
            current_region=current_region,
            execution_goal=execution_goal,
            transition_anchor=transition_anchor,
        )

    @classmethod
    def _is_pre_transition_stage(
        cls,
        *,
        current_region: str | None,
        execution_goal: dict[str, Any],
        transition_anchor: dict[str, Any] | None,
    ) -> bool:
        del cls
        return nav2_semantic_plan.is_pre_transition_stage(
            current_region=current_region,
            execution_goal=execution_goal,
            transition_anchor=transition_anchor,
        )

    @staticmethod
    def _has_explicit_nav2_compute_goal(*, semantic_plan: dict[str, Any] | None) -> bool:
        return (
            isinstance(semantic_plan, dict)
            and nav2_semantic_plan.valid_waypoint_override(semantic_plan.get("nav2_compute_goal"))
            is not None
        )

    def _pose_is_on_transition_target_side(
        self,
        *,
        pose: dict[str, Any] | None,
        transition_anchor: dict[str, Any] | None,
    ) -> bool:
        if not isinstance(pose, dict) or not isinstance(transition_anchor, dict):
            return False
        return nav2_portal_safety.pose_has_sufficient_egress(
            pose=pose,
            anchor=transition_anchor,
            required_depth_m=self.portal_egress_depth_m,
        )

    def _active_pending_portal_egress_anchor(
        self,
        *,
        scene_id: str | None,
        pose: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        anchor = self._pending_portal_egress_anchor
        normalized_scene_id = str(scene_id or "")
        if (
            not isinstance(anchor, dict)
            or self._pending_portal_egress_scene_id != normalized_scene_id
        ):
            self._clear_pending_portal_egress_anchor()
            return None
        if not isinstance(pose, dict):
            return None
        depth = nav2_portal_safety.target_side_depth_m(
            pose=pose,
            anchor=anchor,
        )
        if depth is None:
            self._clear_pending_portal_egress_anchor()
            return None
        required_depth = nav2_portal_safety.required_egress_depth_m(
            anchor,
            default_depth_m=self.portal_egress_depth_m,
        )
        if depth >= required_depth:
            self._clear_pending_portal_egress_anchor()
            return None
        if depth < 0.0:
            return None
        return dict(anchor)

    def _remember_pending_portal_egress_anchor(
        self,
        *,
        scene_id: str | None,
        anchor: dict[str, Any] | None,
    ) -> None:
        if not nav2_portal_safety.has_portal_frame(anchor):
            return
        normalized = dict(anchor)
        normalized["portal_required_egress_depth_m"] = self.portal_egress_depth_m
        self._pending_portal_egress_anchor = normalized
        self._pending_portal_egress_scene_id = str(scene_id or "")

    def _clear_pending_portal_egress_anchor(self) -> None:
        self._pending_portal_egress_anchor = None
        self._pending_portal_egress_scene_id = None

    def _refine_transition_anchor_with_traversability(
        self,
        *,
        scene_id: str | None,
        vertical_axis: str,
        current_region: str | None,
        transition_anchor: dict[str, Any] | None,
        execution_goal: dict[str, Any],
        nav2_trav_map_filename: str | None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if not isinstance(transition_anchor, dict):
            return transition_anchor, None
        if not scene_id:
            return transition_anchor, None
        if not self._is_pre_transition_stage(
            current_region=current_region,
            execution_goal=execution_goal,
            transition_anchor=transition_anchor,
        ):
            return transition_anchor, None

        required_keys = (
            "portal_span_axis",
            "portal_span_min",
            "portal_span_max",
            "portal_normal_axis",
            "portal_boundary_value",
            "portal_normal_sign",
        )
        if any(key not in transition_anchor for key in required_keys):
            return transition_anchor, None

        try:
            from .nav2_runtime_bridge import (
                point_has_clearance,
                segment_has_clearance,
            )
        except Exception:
            return transition_anchor, None

        try:
            map_spec = self._load_stamped_traversability_grid(
                scene_id=scene_id,
                map_resolution=self.portal_analysis_map_resolution,
                trav_map_filename=nav2_trav_map_filename,
            )
        except Exception:
            return transition_anchor, None

        plane_axes = self._plane_axes(vertical_axis)
        if plane_axes is None:
            return transition_anchor, None

        span_axis = str(transition_anchor.get("portal_span_axis") or "").strip()
        normal_axis = str(transition_anchor.get("portal_normal_axis") or "").strip()
        if span_axis not in plane_axes or normal_axis not in plane_axes or span_axis == normal_axis:
            return transition_anchor, None

        span_min = float(transition_anchor["portal_span_min"])
        span_max = float(transition_anchor["portal_span_max"])
        if span_max < span_min:
            span_min, span_max = span_max, span_min
        if (span_max - span_min) < self.portal_sampling_step_m:
            return transition_anchor, None

        boundary_value = float(transition_anchor["portal_boundary_value"])
        normal_sign = 1.0 if float(transition_anchor["portal_normal_sign"]) >= 0.0 else -1.0
        portal_clearance = max(self.portal_clearance_radius_m, self._portal_half_width_m)
        source_standoff = max(self.portal_corridor_standoff_m, self._portal_forward_extent_m + 0.05)
        target_standoff = max(self.portal_corridor_standoff_m, self._portal_rear_extent_m + 0.05)
        target_offset = abs(float(transition_anchor[normal_axis]) - boundary_value)
        target_offset = max(target_offset, target_standoff)
        source_offset = source_standoff
        preferred_span_value = float(transition_anchor[span_axis])

        valid_samples: list[float] = []
        midpoint_valid_samples: list[float] = []
        sample_value = span_min
        while sample_value <= span_max + 1e-6:
            source_xy = self._portal_point_on_plane(
                plane_axes=plane_axes,
                span_axis=span_axis,
                normal_axis=normal_axis,
                span_value=sample_value,
                normal_value=boundary_value - normal_sign * source_offset,
            )
            midpoint_xy = self._portal_point_on_plane(
                plane_axes=plane_axes,
                span_axis=span_axis,
                normal_axis=normal_axis,
                span_value=sample_value,
                normal_value=boundary_value,
            )
            target_xy = self._portal_point_on_plane(
                plane_axes=plane_axes,
                span_axis=span_axis,
                normal_axis=normal_axis,
                span_value=sample_value,
                normal_value=boundary_value + normal_sign * target_offset,
            )
            midpoint_has_clearance = point_has_clearance(
                map_spec=map_spec,
                point_xy=midpoint_xy,
                clearance_radius_m=portal_clearance,
            )
            if midpoint_has_clearance:
                midpoint_valid_samples.append(sample_value)
            if (
                point_has_clearance(
                    map_spec=map_spec,
                    point_xy=source_xy,
                    clearance_radius_m=portal_clearance,
                )
                and point_has_clearance(
                    map_spec=map_spec,
                    point_xy=target_xy,
                    clearance_radius_m=portal_clearance,
                )
                and midpoint_has_clearance
                and segment_has_clearance(
                    map_spec=map_spec,
                    start_xy=source_xy,
                    end_xy=target_xy,
                    clearance_radius_m=max(0.0, portal_clearance - 0.05),
                )
                and segment_has_clearance(
                    map_spec=map_spec,
                    start_xy=source_xy,
                    end_xy=midpoint_xy,
                    clearance_radius_m=max(0.0, portal_clearance - 0.05),
                )
                and segment_has_clearance(
                    map_spec=map_spec,
                    start_xy=midpoint_xy,
                    end_xy=target_xy,
                    clearance_radius_m=max(0.0, portal_clearance - 0.05),
                )
            ):
                valid_samples.append(sample_value)
            sample_value += self.portal_sampling_step_m

        if not valid_samples:
            if not midpoint_valid_samples:
                return transition_anchor, None
            refined_span_value = self._choose_midpoint_only_portal_span(
                valid_samples=midpoint_valid_samples,
                preferred_span_value=preferred_span_value,
                sampling_step_m=self.portal_sampling_step_m,
            )
            doorway_midpoint = dict(transition_anchor)
            doorway_midpoint[span_axis] = refined_span_value
            doorway_midpoint[normal_axis] = boundary_value
            doorway_midpoint["waypoint_type"] = "portal_midpoint"
            doorway_midpoint["portal_refined_from_traversability"] = True
            doorway_midpoint["portal_alignment_stage"] = "midpoint"
            doorway_midpoint["midpoint_only"] = True
            return transition_anchor, {
                "midpoint": doorway_midpoint,
                "span_axis": span_axis,
                "refined_span_value": refined_span_value,
                "midpoint_only": True,
            }

        sampled_runs: list[tuple[float, float]] = []
        run_start = valid_samples[0]
        previous_value = valid_samples[0]
        for current_value in valid_samples[1:]:
            if (current_value - previous_value) <= self.portal_sampling_step_m * 1.5:
                previous_value = current_value
                continue
            sampled_runs.append((run_start, previous_value))
            run_start = current_value
            previous_value = current_value
        sampled_runs.append((run_start, previous_value))
        widest_run_width = max(abs(item[1] - item[0]) for item in sampled_runs)
        comparable_width_threshold = widest_run_width * 0.8
        comparable_runs = [
            item for item in sampled_runs if abs(item[1] - item[0]) >= comparable_width_threshold
        ]
        comparable_runs.sort(
            key=lambda item: (
                abs(((item[0] + item[1]) * 0.5) - preferred_span_value),
                -abs(item[1] - item[0]),
            )
        )
        chosen_run = comparable_runs[0]
        refined_span_value = (chosen_run[0] + chosen_run[1]) * 0.5

        refined_anchor = dict(transition_anchor)
        refined_anchor[span_axis] = refined_span_value
        refined_anchor[normal_axis] = boundary_value + normal_sign * target_offset
        refined_anchor["portal_refined_from_traversability"] = True
        refined_anchor["portal_desired_heading"] = self._portal_heading_on_plane(
            plane_axes=plane_axes,
            normal_axis=normal_axis,
            normal_sign=normal_sign,
        )
        refined_anchor["portal_alignment_stage"] = "target_anchor"

        source_anchor = dict(refined_anchor)
        source_anchor[normal_axis] = boundary_value - normal_sign * source_offset
        source_anchor["waypoint_type"] = "pre_portal_standoff"
        source_anchor["room_id"] = transition_anchor.get("source_room_id")
        source_anchor["room_name"] = transition_anchor.get("source_room_name")
        source_anchor["portal_alignment_stage"] = "source_anchor"

        doorway_midpoint = dict(refined_anchor)
        doorway_midpoint[normal_axis] = boundary_value
        doorway_midpoint["waypoint_type"] = "portal_midpoint"
        doorway_midpoint["portal_alignment_stage"] = "midpoint"

        return refined_anchor, {
            "source_anchor": source_anchor,
            "midpoint": doorway_midpoint,
            "target_anchor": dict(refined_anchor),
            "span_axis": span_axis,
            "refined_span_value": refined_span_value,
        }

    @staticmethod
    def _choose_midpoint_only_portal_span(
        *,
        valid_samples: list[float],
        preferred_span_value: float,
        sampling_step_m: float,
    ) -> float:
        sampled_runs: list[tuple[float, float]] = []
        run_start = valid_samples[0]
        previous_value = valid_samples[0]
        for current_value in valid_samples[1:]:
            if (current_value - previous_value) <= sampling_step_m * 1.5:
                previous_value = current_value
                continue
            sampled_runs.append((run_start, previous_value))
            run_start = current_value
            previous_value = current_value
        sampled_runs.append((run_start, previous_value))

        def _distance_to_run(item: tuple[float, float]) -> float:
            run_min, run_max = item
            if run_min <= preferred_span_value <= run_max:
                return 0.0
            return min(abs(preferred_span_value - run_min), abs(preferred_span_value - run_max))

        chosen_run = min(
            sampled_runs,
            key=lambda item: (
                _distance_to_run(item),
                -abs(item[1] - item[0]),
            ),
        )
        if chosen_run[0] <= preferred_span_value <= chosen_run[1]:
            return float(preferred_span_value)
        return float((chosen_run[0] + chosen_run[1]) * 0.5)

    @staticmethod
    def _portal_point_on_plane(
        *,
        plane_axes: tuple[str, str],
        span_axis: str,
        normal_axis: str,
        span_value: float,
        normal_value: float,
    ) -> dict[str, float]:
        point = {plane_axes[0]: 0.0, plane_axes[1]: 0.0}
        point[span_axis] = float(span_value)
        point[normal_axis] = float(normal_value)
        return {"x": float(point[plane_axes[0]]), "y": float(point[plane_axes[1]])}

    @staticmethod
    def _portal_heading_on_plane(
        *,
        plane_axes: tuple[str, str],
        normal_axis: str,
        normal_sign: float,
    ) -> float:
        direction = {plane_axes[0]: 0.0, plane_axes[1]: 0.0}
        direction[normal_axis] = 1.0 if normal_sign >= 0.0 else -1.0
        return float(math.atan2(direction[plane_axes[1]], direction[plane_axes[0]]))

    def _append_transition_corridor_to_path(
        self,
        *,
        path_points: list[dict[str, float]],
        doorway_corridor: dict[str, Any],
        vertical_axis: str,
        start_from: str = "source_anchor",
    ) -> list[dict[str, float]]:
        return nav2_fallback_corridor.append_transition_corridor_to_path(
            path_points=path_points,
            doorway_corridor=doorway_corridor,
            vertical_axis=vertical_axis,
            start_from=start_from,
            world_pose_to_nav2_plane=self._world_pose_to_nav2_plane,
        )

    @staticmethod
    def _doorway_corridor_stage_key(
        waypoint: dict[str, Any] | None,
        doorway_corridor: dict[str, Any] | None,
    ) -> str:
        return nav2_fallback_corridor.doorway_corridor_stage_key(
            waypoint=waypoint,
            doorway_corridor=doorway_corridor,
            same_waypoint_signature=Nav2NavigatorAdapter._same_waypoint_signature,
        )

    def _semantic_dense_local_waypoints(
        self,
        *,
        start_pose: dict[str, Any] | None,
        target: dict[str, Any],
        dense_waypoints: list[dict[str, Any]],
        vertical_axis: str,
        previous_dense_index: int,
    ) -> list[dict[str, Any]]:
        if not isinstance(start_pose, dict):
            return [dict(target)]
        if not dense_waypoints:
            return [dict(target)]

        nearest_index = self._nearest_dense_waypoint_index(
            start_pose=start_pose,
            dense_waypoints=dense_waypoints,
            vertical_axis=vertical_axis,
        )
        start_index = max(nearest_index, min(previous_dense_index, len(dense_waypoints) - 1))
        target_index = self._dense_target_waypoint_index(
            target=target,
            dense_waypoints=dense_waypoints,
            start_index=start_index,
            vertical_axis=vertical_axis,
        )
        if target_index < start_index:
            target_index = start_index

        segment = dense_waypoints[start_index : target_index + 1]
        local_waypoints: list[dict[str, Any]] = []
        for waypoint in segment:
            if self._distance(start_pose, waypoint) < self.waypoint_spacing * 0.5:
                continue
            local_waypoint = dict(waypoint)
            local_waypoint["waypoint_type"] = "local_dense_path"
            self._append_if_far_enough(local_waypoints, local_waypoint)

        if local_waypoints:
            final_waypoint = dict(target)
            if self._distance(local_waypoints[-1], final_waypoint) < self.waypoint_spacing * 0.5:
                local_waypoints[-1] = final_waypoint
            else:
                local_waypoints.append(final_waypoint)
            return local_waypoints
        return [dict(target)]

    def _doorway_corridor_fallback_waypoints(
        self,
        *,
        start_pose: dict[str, Any] | None,
        execution_goal: dict[str, Any],
        doorway_corridor: dict[str, Any] | None,
        scene_id: str | None,
        vertical_axis: str,
        nav2_trav_map_filename: str | None,
    ) -> list[dict[str, Any]]:
        return nav2_fallback_corridor.build_doorway_corridor_fallback_waypoints(
            start_pose=start_pose,
            execution_goal=execution_goal,
            doorway_corridor=doorway_corridor,
            scene_id=scene_id,
            vertical_axis=vertical_axis,
            nav2_trav_map_filename=nav2_trav_map_filename,
            waypoint_spacing=self.waypoint_spacing,
            distance=self._distance,
            filter_waypoints_for_local_clearance=self._filter_waypoints_for_local_clearance,
        )

    def _filter_waypoints_for_local_clearance(
        self,
        *,
        start_pose: dict[str, Any] | None,
        vertical_axis: str,
        waypoints: list[dict[str, Any]],
        scene_id: str | None,
        nav2_trav_map_filename: str | None,
        navigation_goal: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(start_pose, dict):
            return [dict(waypoint) for waypoint in waypoints if isinstance(waypoint, dict)]
        clearance_radius_m = self._local_path_post_clearance_radius_m()
        if not scene_id or clearance_radius_m <= 0.0:
            return [dict(waypoint) for waypoint in waypoints if isinstance(waypoint, dict)]

        try:
            from .nav2_runtime_bridge import (
                DEFAULT_NAV2_MAP_RESOLUTION,
                point_has_clearance,
                segment_has_clearance,
            )
        except Exception:
            return [dict(waypoint) for waypoint in waypoints if isinstance(waypoint, dict)]

        try:
            map_spec = self._load_stamped_traversability_grid(
                scene_id=scene_id,
                map_resolution=DEFAULT_NAV2_MAP_RESOLUTION,
                trav_map_filename=nav2_trav_map_filename,
                navigation_goal=navigation_goal,
                vertical_axis=vertical_axis,
            )
        except Exception:
            return [dict(waypoint) for waypoint in waypoints if isinstance(waypoint, dict)]

        previous_xy = self._world_pose_to_nav2_plane(start_pose, vertical_axis=vertical_axis)
        if previous_xy is None:
            return [dict(waypoint) for waypoint in waypoints if isinstance(waypoint, dict)]

        filtered_waypoints: list[dict[str, Any]] = []
        step_m = max(
            float(map_spec.get("resolution", self.portal_analysis_map_resolution)) * 0.5,
            0.025,
        )
        for waypoint in waypoints:
            if not isinstance(waypoint, dict):
                continue
            current_xy = self._world_pose_to_nav2_plane(waypoint, vertical_axis=vertical_axis)
            if current_xy is None:
                continue
            if not point_has_clearance(
                map_spec=map_spec,
                point_xy=current_xy,
                clearance_radius_m=clearance_radius_m,
            ):
                continue
            if not segment_has_clearance(
                map_spec=map_spec,
                start_xy=previous_xy,
                end_xy=current_xy,
                clearance_radius_m=clearance_radius_m,
                step_m=step_m,
            ):
                continue
            filtered_waypoints.append(dict(waypoint))
            previous_xy = current_xy
        return filtered_waypoints

    @staticmethod
    def _resolve_dense_waypoint_index(
        *,
        local_waypoints: list[dict[str, Any]],
        dense_waypoints: list[dict[str, Any]],
    ) -> int:
        if not local_waypoints or not dense_waypoints:
            return 0
        first_waypoint = local_waypoints[0]
        for index, dense_waypoint in enumerate(dense_waypoints):
            if dense_waypoint.get("nav_node") == first_waypoint.get("nav_node"):
                return index
        return 0

    def _nearest_dense_waypoint_index(
        self,
        *,
        start_pose: dict[str, Any],
        dense_waypoints: list[dict[str, Any]],
        vertical_axis: str,
    ) -> int:
        best_index = 0
        best_distance = None
        for index, waypoint in enumerate(dense_waypoints):
            distance = self._planar_distance(
                first=start_pose,
                second=waypoint,
                vertical_axis=vertical_axis,
            )
            if best_distance is None or distance < best_distance:
                best_index = index
                best_distance = distance
        return best_index

    def _dense_target_waypoint_index(
        self,
        *,
        target: dict[str, Any],
        dense_waypoints: list[dict[str, Any]],
        start_index: int,
        vertical_axis: str,
    ) -> int:
        target_room_id = str(target.get("room_id") or "").strip() or None
        waypoint_type = str(target.get("waypoint_type", "")).strip().lower()
        if target_room_id:
            for index in range(start_index, len(dense_waypoints)):
                candidate_room_id = str(dense_waypoints[index].get("room_id") or "").strip() or None
                if candidate_room_id == target_room_id:
                    if waypoint_type == "portal":
                        return index
        best_index = len(dense_waypoints) - 1
        best_distance = None
        for index in range(start_index, len(dense_waypoints)):
            distance = self._planar_distance(
                first=dense_waypoints[index],
                second=target,
                vertical_axis=vertical_axis,
            )
            if best_distance is None or distance < best_distance:
                best_index = index
                best_distance = distance
        return best_index

    @staticmethod
    def _coerce_index(value: Any) -> int | None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return max(0, parsed)

    @staticmethod
    def _normalize_label(value: Any) -> str | None:
        return nav2_semantic_plan.normalize_label(value)

    @staticmethod
    def _planar_distance(
        *,
        first: dict[str, Any],
        second: dict[str, Any],
        vertical_axis: str,
    ) -> float:
        plane_axes = Nav2NavigatorAdapter._plane_axes(vertical_axis)
        if plane_axes is None:
            return float("inf")
        dx = float(second.get(plane_axes[0], 0.0)) - float(first.get(plane_axes[0], 0.0))
        dy = float(second.get(plane_axes[1], 0.0)) - float(first.get(plane_axes[1], 0.0))
        return math.hypot(dx, dy)

    @staticmethod
    def _resolve_vertical_axis(
        *,
        start: dict[str, Any],
        goal: dict[str, Any],
        context: dict[str, Any],
        semantic_plan: dict[str, Any] | None = None,
    ) -> str:
        parameters = context.get("parameters", {})
        map_state = context.get("map_state", {})
        for candidate in (
            semantic_plan.get("vertical_axis") if isinstance(semantic_plan, dict) else None,
            goal.get("vertical_axis"),
            start.get("vertical_axis"),
            context.get("vertical_axis"),
            parameters.get("vertical_axis") if isinstance(parameters, dict) else None,
            map_state.get("vertical_axis") if isinstance(map_state, dict) else None,
        ):
            if isinstance(candidate, str) and candidate in {"x", "y", "z"}:
                return candidate
        return "z"

    @staticmethod
    def _goal_position(goal: dict[str, Any]) -> dict[str, float] | None:
        position = goal.get("position")
        if not isinstance(position, dict):
            return None
        for axis in ("x", "y", "z"):
            if not isinstance(position.get(axis), (int, float)):
                return None
        return {
            "x": float(position["x"]),
            "y": float(position["y"]),
            "z": float(position["z"]),
        }

    def _resolve_nav2_trav_map_filename(
        self,
        *,
        start: dict[str, Any],
        goal: dict[str, Any],
        context: dict[str, Any],
    ) -> str | None:
        if self.trav_map_filename:
            return self.trav_map_filename
        return self._resolve_nav2_trav_map_filename_from_context(
            start=start,
            goal=goal,
            context=context,
        )

    @staticmethod
    def _resolve_nav2_trav_map_filename_from_context(
        *,
        start: dict[str, Any],
        goal: dict[str, Any],
        context: dict[str, Any],
    ) -> str | None:
        parameters = context.get("parameters", {})
        map_state = context.get("map_state", {})
        for candidate in (
            goal.get("nav2_trav_map_filename"),
            start.get("nav2_trav_map_filename"),
            context.get("nav2_trav_map_filename"),
            parameters.get("nav2_trav_map_filename") if isinstance(parameters, dict) else None,
            map_state.get("nav2_trav_map_filename") if isinstance(map_state, dict) else None,
        ):
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

        scene_file = None
        for candidate in (
            goal.get("scene_file"),
            start.get("scene_file"),
            context.get("scene_file"),
            parameters.get("scene_file") if isinstance(parameters, dict) else None,
        ):
            if isinstance(candidate, str) and candidate.strip():
                scene_file = candidate.strip().lower()
                break

        if scene_file is None:
            return None
        if any(
            token in scene_file
            for token in (
                "open_door",
                "door_open",
                "doors_open",
                "all_doors_open",
                "sliding_full",
            )
        ):
            return "floor_trav_open_door_0.png"
        if "no_door" in scene_file or "doorless" in scene_file:
            return "floor_trav_no_door_0.png"
        if "no_obj" in scene_file or "no_object" in scene_file:
            return "floor_trav_no_obj_0.png"
        return None

    @staticmethod
    def _world_pose_to_nav2_plane(
        pose: dict[str, Any],
        *,
        vertical_axis: str,
    ) -> dict[str, float] | None:
        plane_axes = Nav2NavigatorAdapter._plane_axes(vertical_axis)
        if plane_axes is None:
            return None
        first = pose.get(plane_axes[0])
        second = pose.get(plane_axes[1])
        if not isinstance(first, (int, float)) or not isinstance(second, (int, float)):
            return None
        yaw = pose.get("yaw", 0.0)
        if not isinstance(yaw, (int, float)):
            yaw = 0.0
        return {"x": float(first), "y": float(second), "yaw": float(yaw)}

    @staticmethod
    def _nav2_plane_to_world(
        point_xy: dict[str, Any],
        *,
        vertical_axis: str,
        reference_pose: dict[str, Any],
    ) -> dict[str, float] | None:
        if not isinstance(point_xy.get("x"), (int, float)) or not isinstance(
            point_xy.get("y"), (int, float)
        ):
            return None
        plane_axes = Nav2NavigatorAdapter._plane_axes(vertical_axis)
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
    def _plane_axes(vertical_axis: str) -> tuple[str, str] | None:
        mapping = {
            "x": ("y", "z"),
            "y": ("x", "z"),
            "z": ("x", "y"),
        }
        return mapping.get(vertical_axis)

    @staticmethod
    def _extract_path_points(path_response: dict[str, Any]) -> list[dict[str, float]]:
        raw_points = path_response.get("points")
        if not isinstance(raw_points, list):
            return []
        points: list[dict[str, float]] = []
        for point in raw_points:
            if not isinstance(point, dict):
                continue
            x_coord = point.get("x")
            y_coord = point.get("y")
            if not isinstance(x_coord, (int, float)) or not isinstance(y_coord, (int, float)):
                continue
            points.append({"x": float(x_coord), "y": float(y_coord)})
        return points

    def _refine_nav2_local_path_points(
        self,
        *,
        scene_id: str | None,
        path_points: list[dict[str, float]],
        nav2_trav_map_filename: str | None,
        navigation_goal: dict[str, Any] | None = None,
        vertical_axis: str = "z",
    ) -> tuple[list[dict[str, float]], bool]:
        room_clipped = False
        if self.local_path_waypoint_spacing_m < (self.waypoint_spacing - 1e-6):
            densified_points = self._densify_nav2_path_points(
                path_points=path_points,
                spacing_m=self.local_path_waypoint_spacing_m,
            )
        else:
            densified_points = [dict(point) for point in path_points]
        clearance_radius_m = self._local_path_post_clearance_radius_m()
        if len(densified_points) <= 1 or not scene_id or clearance_radius_m <= 0.0:
            return densified_points, room_clipped
        try:
            from .nav2_runtime_bridge import (
                DEFAULT_NAV2_MAP_RESOLUTION,
                point_has_clearance,
                segment_has_clearance,
            )
        except Exception:
            return densified_points, room_clipped

        try:
            map_spec = self._load_stamped_traversability_grid(
                scene_id=scene_id,
                map_resolution=DEFAULT_NAV2_MAP_RESOLUTION,
                trav_map_filename=nav2_trav_map_filename,
                navigation_goal=navigation_goal,
                vertical_axis=vertical_axis,
            )
        except Exception:
            return densified_points, room_clipped

        safe_points = [dict(densified_points[0])]
        step_m = max(
            float(map_spec.get("resolution", self.portal_analysis_map_resolution)) * 0.5,
            0.025,
        )
        for point in densified_points[1:]:
            if not point_has_clearance(
                map_spec=map_spec,
                point_xy=point,
                clearance_radius_m=clearance_radius_m,
            ):
                break
            if not segment_has_clearance(
                map_spec=map_spec,
                start_xy=safe_points[-1],
                end_xy=point,
                clearance_radius_m=clearance_radius_m,
                step_m=step_m,
            ):
                break
            safe_points.append(dict(point))

        if len(safe_points) < 2:
            return [], True
        return safe_points, room_clipped or len(safe_points) < len(densified_points)

    @staticmethod
    def _densify_nav2_path_points(
        *,
        path_points: list[dict[str, float]],
        spacing_m: float,
    ) -> list[dict[str, float]]:
        if len(path_points) <= 1:
            return [dict(point) for point in path_points]
        spacing = max(0.05, float(spacing_m))
        densified = [dict(path_points[0])]
        for start_point, end_point in zip(path_points, path_points[1:]):
            start_x = float(start_point["x"])
            start_y = float(start_point["y"])
            end_x = float(end_point["x"])
            end_y = float(end_point["y"])
            segment_length = math.hypot(end_x - start_x, end_y - start_y)
            sample_count = max(1, int(math.ceil(segment_length / spacing)))
            for sample_index in range(1, sample_count + 1):
                ratio = sample_index / sample_count
                densified.append(
                    {
                        "x": start_x + (end_x - start_x) * ratio,
                        "y": start_y + (end_y - start_y) * ratio,
                    }
                )
        return densified

    def _world_waypoints_from_nav2_path(
        self,
        *,
        path_points: list[dict[str, float]],
        vertical_axis: str,
        start_pose: dict[str, Any],
        target: dict[str, Any],
        append_target: bool = True,
    ) -> list[dict[str, Any]]:
        reference_pose = dict(start_pose)
        waypoints: list[dict[str, Any]] = []
        for index, point in enumerate(path_points):
            world = self._nav2_plane_to_world(
                point, vertical_axis=vertical_axis, reference_pose=reference_pose
            )
            if world is None:
                continue
            waypoint = dict(world)
            waypoint["waypoint_type"] = "local_path"
            if self._append_if_far_enough(
                waypoints, waypoint, spacing=self.local_path_waypoint_spacing_m
            ):
                reference_pose = waypoint
            if append_target and index == len(path_points) - 1:
                target_waypoint = dict(target)
                if self._append_if_far_enough(
                    waypoints,
                    target_waypoint,
                    spacing=self.local_path_waypoint_spacing_m,
                ):
                    reference_pose = target_waypoint
                else:
                    if waypoints:
                        waypoints[-1] = target_waypoint
                    else:
                        waypoints.append(target_waypoint)
        return waypoints

    def _append_if_far_enough(
        self,
        waypoints: list[dict[str, Any]],
        waypoint: dict[str, Any],
        *,
        spacing: float | None = None,
    ) -> bool:
        min_spacing = self.waypoint_spacing if spacing is None else max(0.0, float(spacing))
        if not waypoints:
            waypoints.append(dict(waypoint))
            return True
        if self._distance(waypoints[-1], waypoint) >= min_spacing:
            waypoints.append(dict(waypoint))
            return True
        return False

    @staticmethod
    def _distance(first: dict[str, Any], second: dict[str, Any]) -> float:
        dx = float(second.get("x", 0.0)) - float(first.get("x", 0.0))
        dy = float(second.get("y", 0.0)) - float(first.get("y", 0.0))
        dz = float(second.get("z", 0.0)) - float(first.get("z", 0.0))
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def _path_cost(waypoints: list[dict[str, Any]], *, vertical_axis: str) -> float:
        if len(waypoints) < 2:
            return 0.0
        cost = 0.0
        for previous, current in zip(waypoints, waypoints[1:]):
            cost += Nav2NavigatorAdapter._distance(previous, current)
        return cost
