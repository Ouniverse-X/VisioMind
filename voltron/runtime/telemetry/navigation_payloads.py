"""Compact navigation payloads for high-frequency telemetry."""

from __future__ import annotations

import hashlib
import json
from typing import Any


_GOAL_KEYS = (
    "scene_id",
    "goal_type",
    "object_id",
    "object_name",
    "room_id",
    "room_name",
    "floor_id",
    "position",
    "grounding_query",
    "spatial_relation",
    "stop_condition",
    "constraints",
    "followup_context",
    "nav_backend",
    "nav2_profile",
)
_CANDIDATE_KEYS = (
    "candidate_id",
    "x",
    "y",
    "z",
    "floor_id",
    "room_id",
    "room_name",
    "nav_node",
    "waypoint_type",
    "object_id",
    "object_name",
    "approach_distance_m",
    "approach_boundary_distance_m",
    "desired_heading",
    "candidate_geometry_score",
    "selection_source",
    "handoff_distance_m",
    "approach_room_relation",
    "path_cost",
    "nav2_validation_status",
    "nav2_path_length_m",
    "runtime_map_revision",
    "runtime_overlay_signature",
)
_WAYPOINT_KEYS = (
    "x",
    "y",
    "z",
    "floor_id",
    "room_id",
    "room_name",
    "waypoint_type",
    "nav_node",
    "desired_heading",
    "portal_normal_axis",
    "portal_boundary_value",
    "portal_span_axis",
    "portal_span_min",
    "portal_span_max",
    "source_room_name",
    "transition_anchor",
)


def summarize_agent_result_for_event(result: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in (
        "agent",
        "attempt",
        "control_step",
        "action_keys",
        "policy_info",
        "memory_update",
        "navigator_backend",
        "scene_id",
        "waypoint_count",
        "message",
        "error_type",
        "error_stage",
    ):
        if key in result:
            value = result.get(key)
            summary[key] = (
                value[:1000] if key == "message" and isinstance(value, str) else value
            )

    for key in ("grounded_goal", "nav_goal"):
        value = result.get(key)
        if isinstance(value, dict):
            summary[key] = summarize_goal(value)

    selected = _resolve_selected_object_approach(result)
    if isinstance(selected, dict):
        summary["selected_object_approach"] = summarize_candidate(selected)

    candidates = _resolve_object_approach_candidates(result)
    if candidates:
        summary["object_approach_candidate_count"] = len(candidates)
        summary["object_approach_candidate_ids"] = [
            str(candidate.get("candidate_id"))
            for candidate in candidates
            if isinstance(candidate, dict) and candidate.get("candidate_id") is not None
        ]

    path_plan = result.get("path_plan")
    if isinstance(path_plan, dict):
        path_summary = summarize_path_plan(path_plan)
        if path_summary:
            summary["path_plan"] = path_summary

    candidate_audit = result.get("candidate_detection_audit")
    if isinstance(candidate_audit, list) and all(
        isinstance(entry, dict) for entry in candidate_audit
    ):
        try:
            summary["candidate_detection_audit"] = json.loads(
                json.dumps(candidate_audit)
            )
        except (TypeError, ValueError):
            pass

    return summary


def build_navigation_candidates_snapshot(
    *,
    subtask_id: str,
    control_step: int | None,
    result: dict[str, Any],
) -> dict[str, Any] | None:
    candidates = [
        summarize_candidate(candidate)
        for candidate in _resolve_object_approach_candidates(result)
    ]
    candidates = [candidate for candidate in candidates if candidate]
    selected = _resolve_selected_object_approach(result)
    selected_summary = (
        summarize_candidate(selected) if isinstance(selected, dict) else {}
    )
    if not candidates and not selected_summary:
        return None

    goal = result.get("grounded_goal") or result.get("nav_goal")
    target = summarize_goal(goal) if isinstance(goal, dict) else {}
    if (
        target
        and isinstance(selected, dict)
        and isinstance(selected.get("object_position"), dict)
    ):
        target.setdefault("position", dict(selected["object_position"]))

    return {
        "subtask_id": subtask_id,
        "control_step": control_step,
        "candidate_count": len(candidates),
        "selected_candidate_id": selected_summary.get("candidate_id"),
        "selected_object_approach": selected_summary,
        "target": target,
        "candidates": candidates,
    }


def navigation_candidates_snapshot_signature(
    snapshot: dict[str, Any] | None,
) -> str | None:
    if not isinstance(snapshot, dict):
        return None
    candidate_parts = []
    for candidate in snapshot.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        candidate_parts.append(
            ":".join(
                [
                    str(candidate.get("candidate_id") or ""),
                    _fmt_float(candidate.get("x")),
                    _fmt_float(candidate.get("y")),
                    _fmt_float(candidate.get("z")),
                ]
            )
        )
    return "|".join(
        [
            str(snapshot.get("subtask_id") or ""),
            str(snapshot.get("selected_candidate_id") or ""),
            ";".join(candidate_parts),
        ]
    )


def build_nav2_path_snapshot(
    *,
    subtask_id: str,
    control_step: int | None,
    result: dict[str, Any],
    runtime_artifacts: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Build a path-debug snapshot from the full Navigation runtime artifacts."""

    path_plan = _resolve_path_plan(result=result, runtime_artifacts=runtime_artifacts)
    if path_plan is None:
        return None
    requested_planner = str(path_plan.get("requested_planner") or "").strip().lower()
    has_nav2_diagnostics = any(
        key in path_plan
        for key in (
            "nav2_raw_path_points",
            "nav2_path_points",
            "nav2_raw_path_length",
            "nav2_error",
            "nav2_trav_map_filename",
        )
    )
    if "nav2" not in requested_planner and not has_nav2_diagnostics:
        return None

    snapshot: dict[str, Any] = {
        "subtask_id": subtask_id,
        "control_step": control_step,
    }
    for key in (
        "planner",
        "path_backend",
        "requested_planner",
        "scene_id",
        "vertical_axis",
        "frame_id",
        "planner_id",
        "action_name",
        "nav2_profile",
        "nav2_error",
        "nav2_empty_path_reason",
        "nav2_trav_map_filename",
        "nav2_raw_path_length",
        "nav2_cache_reused",
        "nav2_path_clipped_for_clearance",
        "nav2_scene_obstacle_inflation_radius_m",
        "reason",
        "global_waypoint_index",
        "dense_waypoint_index",
        "waypoint_tracking_mode",
        "waypoint_scope",
    ):
        if key in path_plan:
            snapshot[key] = path_plan.get(key)

    for key in (
        "start",
        "goal",
        "local_goal",
        "execution_goal",
        "nav2_compute_goal",
        "transition_anchor",
    ):
        value = path_plan.get(key)
        if isinstance(value, dict):
            snapshot[key] = dict(value)

    for key in ("blocked_transition",):
        value = path_plan.get(key)
        if isinstance(value, dict):
            snapshot[key] = dict(value)
    for key in ("closed_doors", "door_candidates"):
        value = path_plan.get(key)
        if isinstance(value, list):
            snapshot[key] = [dict(item) if isinstance(item, dict) else item for item in value]

    for key in ("nav2_raw_path_points", "nav2_path_points"):
        points = _clean_nav2_path_points(path_plan.get(key))
        if points or key in path_plan:
            snapshot[key] = points

    waypoints = path_plan.get("waypoints")
    if isinstance(waypoints, list):
        snapshot["waypoints"] = [
            dict(point) for point in waypoints if isinstance(point, dict)
        ]

    nav2_environment = path_plan.get("nav2_environment")
    if isinstance(nav2_environment, dict):
        snapshot["nav2_environment"] = dict(nav2_environment)
    doorway_corridor = path_plan.get("doorway_corridor")
    if isinstance(doorway_corridor, dict):
        snapshot["doorway_corridor"] = dict(doorway_corridor)
    global_plan = path_plan.get("global_plan")
    diagnostics = path_plan.get("object_approach_diagnostics")
    if not isinstance(diagnostics, dict) and isinstance(global_plan, dict):
        diagnostics = global_plan.get("object_approach_diagnostics")
    if isinstance(diagnostics, dict):
        snapshot["object_approach_diagnostics"] = dict(diagnostics)
    dynamic_map_update = path_plan.get("dynamic_map_update")
    if isinstance(dynamic_map_update, dict):
        snapshot["dynamic_map_update"] = dict(dynamic_map_update)
    candidate_validation = path_plan.get("nav2_candidate_validation")
    if not isinstance(candidate_validation, dict) and isinstance(global_plan, dict):
        candidate_validation = global_plan.get("nav2_candidate_validation")
    if isinstance(candidate_validation, dict):
        snapshot["nav2_candidate_validation"] = dict(candidate_validation)
        for key in (
            "candidate_count_before_clearance",
            "candidate_count_after_point_clearance",
            "candidate_count_after_graph_handoff",
            "candidate_count_after_segment_clearance",
            "candidate_count_after_portal_filter",
            "candidate_count_before_nav2_validation",
            "candidate_count_submitted_to_nav2",
            "candidate_count_nav2_executable",
            "runtime_map_revision",
            "runtime_overlay_signature",
        ):
            if key in candidate_validation:
                snapshot[key] = candidate_validation.get(key)
        snapshot["nav2_called"] = bool(
            candidate_validation.get("candidate_count_submitted_to_nav2")
        )
    selected_approach = path_plan.get("selected_object_approach")
    if not isinstance(selected_approach, dict) and isinstance(global_plan, dict):
        selected_approach = global_plan.get("selected_object_approach")
    if not isinstance(selected_approach, dict):
        goal = path_plan.get("goal")
        if isinstance(goal, dict):
            selected_approach = goal.get("selected_object_approach")
    if not isinstance(selected_approach, dict) and isinstance(
        candidate_validation, dict
    ):
        selected_approach = candidate_validation.get("selected_candidate")
    if isinstance(selected_approach, dict):
        selected_summary = summarize_candidate(selected_approach)
        snapshot["selected_object_approach"] = selected_summary
        snapshot["selected_candidate_id"] = selected_summary.get("candidate_id")
    if "nav2_trav_map_filename" in snapshot:
        snapshot["static_base_trav_map"] = snapshot["nav2_trav_map_filename"]
    goal = path_plan.get("goal")
    if isinstance(goal, dict) and goal.get("map_revision"):
        snapshot["semantic_graph_map_revision"] = goal.get("map_revision")
    return snapshot


def nav2_path_snapshot_signature(snapshot: dict[str, Any] | None) -> str | None:
    if not isinstance(snapshot, dict):
        return None
    signature_payload = {
        key: value for key, value in snapshot.items() if key != "control_step"
    }
    encoded = json.dumps(
        signature_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def summarize_navigation_progress_entry(key: str, value: Any) -> tuple[str, Any] | None:
    if key in {"prepared_navigation_payload", "grounding_candidates"}:
        if key == "grounding_candidates" and isinstance(value, list):
            return "grounding_candidate_count", len(value)
        return None
    if key in {"nav_goal", "grounded_goal"} and isinstance(value, dict):
        return key, summarize_goal(value)
    if key == "selected_grounding_candidate" and isinstance(value, dict):
        return key, summarize_grounding_selection(value)
    if key == "selected_object_approach" and isinstance(value, dict):
        return key, summarize_candidate(value)
    if key == "object_approach_selection" and isinstance(value, dict):
        return key, summarize_object_approach_selection(value)
    if key in {
        "tracking_target",
        "target_waypoint",
        "local_goal",
        "execution_goal",
        "nav2_compute_goal",
        "transition_anchor",
    }:
        if isinstance(value, dict):
            return key, summarize_waypoint(value)
    return key, value


def summarize_goal(goal: dict[str, Any]) -> dict[str, Any]:
    summary = {key: goal.get(key) for key in _GOAL_KEYS if key in goal}
    grounding_candidates = goal.get("grounding_candidates")
    if isinstance(grounding_candidates, list):
        summary["grounding_candidate_count"] = len(grounding_candidates)
    selected = goal.get("selected_grounding_candidate")
    if isinstance(selected, dict) and selected:
        summary["selected_grounding_candidate"] = summarize_grounding_selection(
            selected
        )
    return summary


def summarize_candidate(candidate: Any) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        return {}
    summary = {key: candidate.get(key) for key in _CANDIDATE_KEYS if key in candidate}
    object_position = candidate.get("object_position")
    if isinstance(object_position, dict):
        summary["object_position"] = {
            axis: object_position.get(axis)
            for axis in ("x", "y", "z")
            if axis in object_position
        }
    evidence = candidate.get("nearby_object_evidence")
    if isinstance(evidence, dict):
        for key in (
            "nearest_object_id",
            "nearest_object_name",
            "nearest_object_distance_m",
            "path_nearest_object_id",
            "path_nearest_object_name",
            "path_nearest_object_distance_m",
        ):
            if key in evidence:
                summary[key] = evidence.get(key)
    return summary


def summarize_grounding_selection(selection: dict[str, Any]) -> dict[str, Any]:
    candidate = selection.get("candidate")
    summary = {
        key: selection.get(key)
        for key in ("object_id", "object_name", "room_id", "room_name", "reason")
        if key in selection
    }
    if isinstance(candidate, dict):
        summary["candidate"] = {
            key: candidate.get(key)
            for key in (
                "object_id",
                "object_name",
                "room_id",
                "room_name",
                "floor_id",
                "position",
            )
            if key in candidate
        }
    return summary


def summarize_object_approach_selection(selection: dict[str, Any]) -> dict[str, Any]:
    summary = {
        key: selection.get(key)
        for key in ("selected_candidate_id", "reason", "source", "failure_reason")
        if key in selection
    }
    candidate = selection.get("candidate")
    if isinstance(candidate, dict):
        summary["candidate"] = summarize_candidate(candidate)
    return summary


def summarize_waypoint(waypoint: dict[str, Any]) -> dict[str, Any]:
    return {key: waypoint.get(key) for key in _WAYPOINT_KEYS if key in waypoint}


def summarize_path_plan(path_plan: dict[str, Any]) -> dict[str, Any]:
    summary = {
        key: path_plan.get(key)
        for key in (
            "path_backend",
            "found",
            "scene_id",
            "vertical_axis",
            "global_waypoint_index",
            "dense_waypoint_index",
            "waypoint_tracking_mode",
            "waypoint_scope",
            "nav2_error",
            "nav2_trav_map_filename",
            "reason",
        )
        if key in path_plan
    }
    waypoints = path_plan.get("waypoints")
    if isinstance(waypoints, list):
        summary["waypoint_count"] = len(waypoints)
    selected = path_plan.get("selected_object_approach")
    if isinstance(selected, dict):
        summary["selected_object_approach"] = summarize_candidate(selected)
    candidates = path_plan.get("object_approach_candidates")
    if isinstance(candidates, list):
        summary["object_approach_candidate_count"] = len(candidates)
    return summary


def _resolve_object_approach_candidates(result: dict[str, Any]) -> list[dict[str, Any]]:
    for container in _navigation_candidate_containers(result):
        if not isinstance(container, dict):
            continue
        candidates = container.get("object_approach_candidates")
        if isinstance(candidates, list):
            return [
                candidate for candidate in candidates if isinstance(candidate, dict)
            ]
    return []


def _resolve_selected_object_approach(result: dict[str, Any]) -> dict[str, Any] | None:
    for container in _navigation_candidate_containers(result):
        if not isinstance(container, dict):
            continue
        selected = container.get("selected_object_approach")
        if isinstance(selected, dict):
            return selected
    return None


def _navigation_candidate_containers(result: dict[str, Any]) -> tuple[Any, ...]:
    return (
        result,
        result.get("grounded_goal")
        if isinstance(result.get("grounded_goal"), dict)
        else None,
        result.get("nav_goal") if isinstance(result.get("nav_goal"), dict) else None,
        result.get("path_plan") if isinstance(result.get("path_plan"), dict) else None,
        result.get("prepared_navigation_payload")
        if isinstance(result.get("prepared_navigation_payload"), dict)
        else None,
    )


def _resolve_path_plan(
    *,
    result: dict[str, Any],
    runtime_artifacts: dict[str, Any] | None,
) -> dict[str, Any] | None:
    for container in (runtime_artifacts, result):
        if not isinstance(container, dict):
            continue
        path_plan = container.get("path_plan")
        if isinstance(path_plan, dict):
            return path_plan
    return None


def _clean_nav2_path_points(value: Any) -> list[dict[str, float]]:
    if not isinstance(value, list):
        return []
    points: list[dict[str, float]] = []
    for point in value:
        if not isinstance(point, dict):
            continue
        try:
            points.append({"x": float(point["x"]), "y": float(point["y"])})
        except (KeyError, TypeError, ValueError):
            continue
    return points


def _fmt_float(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""
