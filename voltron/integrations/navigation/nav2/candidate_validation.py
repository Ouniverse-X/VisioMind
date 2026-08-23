"""Runtime Nav2 validation for semantic object-approach candidates."""

from __future__ import annotations

import math
from typing import Any


def validate_object_approach_candidates(
    navigator: Any,
    *,
    start: dict[str, Any],
    candidates: list[dict[str, Any]],
    selected_candidate_id: str | None,
    scene_id: str | None,
    vertical_axis: str,
    nav2_trav_map_filename: str | None,
    nav2_scene_obstacle_inflation_radius_m: float,
    navigation_goal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Try candidates against the current runtime-overlay-backed Nav2 map.

    HOV-SG candidate geometry and portal checks remain upstream hard constraints.
    Ordinary-object proximity along a semantic graph path is intentionally not
    inspected here; Nav2's costmap is the authority for that dynamic segment.
    """
    ordered = _ordered_candidates(candidates, selected_candidate_id)
    navigator._last_dynamic_map_update = None
    navigator._last_runtime_overlay_signature = ""
    navigator._last_runtime_overlay_geometry = []
    results: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_result: dict[str, Any] | None = None
    for candidate in ordered:
        candidate_id = str(candidate.get("candidate_id") or "")
        hard_reasons = _hard_rejection_reasons(candidate)
        if hard_reasons:
            results.append(
                {
                    "candidate_id": candidate_id,
                    "status": "blocked",
                    "submitted_to_nav2": False,
                    "hard_rejection_reasons": hard_reasons,
                    "rejection_reasons": hard_reasons,
                    "soft_penalties": _soft_penalties(candidate),
                }
            )
            continue

        candidate_position = _candidate_position(candidate)
        start_xy = _world_pose_to_nav2_plane(
            navigator,
            start,
            vertical_axis=vertical_axis,
        )
        goal_xy = _world_pose_to_nav2_plane(
            navigator,
            candidate_position,
            vertical_axis=vertical_axis,
        )
        if start_xy is None or goal_xy is None:
            result = {
                "candidate_id": candidate_id,
                "status": "planner_error",
                "submitted_to_nav2": False,
                "reason": "candidate_pose_projection_failed",
                "hard_rejection_reasons": ["candidate_pose_projection_failed"],
                "rejection_reasons": ["candidate_pose_projection_failed"],
                "soft_penalties": _soft_penalties(candidate),
            }
            results.append(result)
            continue

        try:
            response = navigator._compute_nav2_path_response(
                scene_id=scene_id,
                start_xy=start_xy,
                goal_xy=goal_xy,
                nav2_trav_map_filename=nav2_trav_map_filename,
                nav2_scene_obstacle_inflation_radius_m=nav2_scene_obstacle_inflation_radius_m,
                navigation_goal={
                    **dict(navigation_goal or {}),
                    **candidate,
                    "waypoint_type": "object_approach",
                },
                vertical_axis=vertical_axis,
            )
        except Exception as exc:
            error = str(exc) or "nav2_candidate_validation_error"
            status = classify_nav2_validation_error(error)
            result = {
                "candidate_id": candidate_id,
                "status": status,
                "submitted_to_nav2": True,
                "error": error,
                "hard_rejection_reasons": [],
                "rejection_reasons": [f"nav2_{status}"],
                "soft_penalties": _soft_penalties(candidate),
            }
            results.append(result)
            continue

        path_points = navigator._extract_path_points(response)
        response_error = str(response.get("error") or "") if isinstance(response, dict) else ""
        if not path_points:
            status = classify_nav2_validation_error(response_error or "empty_path")
            result = {
                "candidate_id": candidate_id,
                "status": status,
                "submitted_to_nav2": True,
                "error": response_error or "empty_path",
                "hard_rejection_reasons": [],
                "rejection_reasons": [
                    "nav2_blocked" if status == "blocked" else f"nav2_{status}"
                ],
                "soft_penalties": _soft_penalties(candidate),
            }
            results.append(result)
            continue

        path_length = _path_length(path_points)
        result = {
            "candidate_id": candidate_id,
            "status": "executable",
            "submitted_to_nav2": True,
            "path_length_m": path_length,
            "path_point_count": len(path_points),
            "hard_rejection_reasons": [],
            "rejection_reasons": [],
            "soft_penalties": _soft_penalties(candidate),
        }
        results.append(result)
        selected = dict(candidate)
        selected_result = result
        break

    evaluated_ids = {str(result.get("candidate_id") or "") for result in results}
    for candidate in ordered:
        candidate_id = str(candidate.get("candidate_id") or "")
        if candidate_id in evaluated_ids:
            continue
        results.append(
            {
                "candidate_id": candidate_id,
                "status": "not_evaluated",
                "submitted_to_nav2": False,
                "reason": "early_stop_after_executable_candidate",
                "hard_rejection_reasons": _hard_rejection_reasons(candidate),
                "rejection_reasons": [],
                "soft_penalties": _soft_penalties(candidate),
            }
        )

    update = getattr(navigator, "_last_dynamic_map_update", None)
    map_revision = update.get("map_revision") if isinstance(update, dict) else None
    overlay_signature = str(
        getattr(navigator, "_last_runtime_overlay_signature", "") or ""
    )
    executable_count = sum(
        result.get("status") == "executable" for result in results
    )
    return {
        "status": "executable" if selected is not None else _aggregate_failure_status(results),
        "selected_candidate": selected,
        "selected_candidate_result": selected_result,
        "candidate_results": results,
        "candidate_count_submitted_to_nav2": sum(
            bool(result.get("submitted_to_nav2")) for result in results
        ),
        "candidate_count_nav2_executable": int(executable_count),
        "candidate_count_not_evaluated": sum(
            result.get("status") == "not_evaluated" for result in results
        ),
        "runtime_map_revision": map_revision,
        "runtime_overlay_signature": overlay_signature,
        "runtime_overlay_geometry": _overlay_geometry(
            getattr(navigator, "_last_runtime_overlay_geometry", [])
        ),
        "selection_source": "nav2_candidate_validation" if selected is not None else None,
    }


def classify_nav2_validation_error(error: str) -> str:
    normalized = str(error or "").strip().lower()
    if any(token in normalized for token in ("timeout", "timed out", "deadline")):
        return "timeout"
    if any(
        token in normalized
        for token in (
            "map_update",
            "overlay",
            "costmap",
            "static_layer",
            "trav_map",
            "map_unavailable",
        )
    ):
        return "overlay_unavailable"
    if normalized in {"empty_path", "no_path", "path_not_found"} or "empty path" in normalized:
        return "blocked"
    return "planner_error"


def _ordered_candidates(
    candidates: list[dict[str, Any]],
    selected_candidate_id: str | None,
) -> list[dict[str, Any]]:
    normalized_selected = str(selected_candidate_id or "").strip()
    ordered = [dict(candidate) for candidate in candidates if isinstance(candidate, dict)]
    if not normalized_selected:
        return ordered
    return sorted(
        ordered,
        key=lambda candidate: (
            0 if str(candidate.get("candidate_id") or "") == normalized_selected else 1,
            float(candidate.get("path_cost", float("inf"))),
            float(candidate.get("candidate_geometry_score", float("inf"))),
            str(candidate.get("candidate_id") or ""),
        ),
    )


def _candidate_position(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "x": candidate.get("x"),
        "y": candidate.get("y"),
        "z": candidate.get("z"),
        "yaw": candidate.get("desired_heading", 0.0),
    }


def _world_pose_to_nav2_plane(
    navigator: Any,
    pose: dict[str, Any],
    *,
    vertical_axis: str,
) -> dict[str, float] | None:
    if not isinstance(pose, dict):
        return None
    return navigator._world_pose_to_nav2_plane(pose, vertical_axis=vertical_axis)


def _hard_rejection_reasons(candidate: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if candidate.get("path_found") is False or candidate.get("found") is False:
        reasons.append("unreachable_semantic_path")
    if candidate.get("blocked_by_closed_door"):
        reasons.append("blocked_by_closed_door")
    handoff_distance = candidate.get("handoff_distance_m")
    if isinstance(handoff_distance, (int, float)) and float(handoff_distance) > 1.0:
        reasons.append("unstable_navigation_handoff")
    evidence = candidate.get("nearby_object_evidence")
    if isinstance(evidence, dict):
        nearest = _candidate_clearance_m(evidence)
        if isinstance(nearest, (int, float)) and float(nearest) < 0.5:
            reasons.append("insufficient_candidate_clearance")
    return reasons


def _candidate_clearance_m(evidence: dict[str, Any]) -> float | None:
    nearby_objects = evidence.get("nearby_objects")
    if isinstance(nearby_objects, list):
        distances = [
            float(item["distance_to_candidate_m"])
            for item in nearby_objects
            if isinstance(item, dict)
            and isinstance(item.get("distance_to_candidate_m"), (int, float))
        ]
        if distances:
            return min(distances)
    nearest = evidence.get("nearest_object_distance_m")
    return float(nearest) if isinstance(nearest, (int, float)) else None


def _soft_penalties(candidate: dict[str, Any]) -> dict[str, float]:
    penalties: dict[str, float] = {}
    history_penalty = candidate.get("history_penalty")
    if isinstance(history_penalty, (int, float)):
        penalties["history"] = max(0.0, float(history_penalty))
    elif candidate.get("blocked_by_history"):
        penalties["history"] = 1000.0
    evidence = candidate.get("nearby_object_evidence")
    if not isinstance(evidence, dict):
        return penalties
    path_distance = evidence.get("path_nearest_object_distance_m")
    if not isinstance(path_distance, (int, float)):
        return penalties
    penalties["path_nearby_object"] = max(0.0, 0.5 - float(path_distance))
    return penalties


def _aggregate_failure_status(results: list[dict[str, Any]]) -> str:
    statuses = {str(result.get("status") or "") for result in results}
    if not statuses:
        return "blocked"
    if "overlay_unavailable" in statuses:
        return "overlay_unavailable"
    if "timeout" in statuses:
        return "timeout"
    if statuses <= {"blocked"}:
        return "blocked"
    return "planner_error"


def _overlay_geometry(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    geometry: list[dict[str, Any]] = []
    for item in value[:100]:
        if not isinstance(item, dict):
            continue
        entry = {
            key: item[key]
            for key in (
                "name",
                "object_id",
                "overlay_kind",
                "geometry_source",
                "geometry_id",
                "geometry_hash",
                "min",
                "max",
                "aabb",
                "position",
                "half_extent_m",
                "polygons",
                "world_polygons",
            )
            if key in item
        }
        if entry:
            geometry.append(entry)
    return geometry


def _path_length(points: list[dict[str, float]]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(
        math.hypot(
            float(current["x"]) - float(previous["x"]),
            float(current["y"]) - float(previous["y"]),
        )
        for previous, current in zip(points, points[1:])
    )


def candidate_validation_failure_plan(
    *,
    start: dict[str, Any],
    goal: dict[str, Any],
    scene_id: str | None,
    vertical_axis: str,
    nav2_environment: dict[str, Any],
    validation: dict[str, Any],
    nav2_trav_map_filename: str | None = None,
    nav2_profile: str | None = None,
) -> dict[str, Any]:
    status = str(validation.get("status") or "blocked")
    return {
        "planner": "hovsg_global_nav2_local",
        "path_backend": "nav2_candidate_validation_unavailable",
        "scene_id": scene_id,
        "vertical_axis": vertical_axis,
        "start": start,
        "goal": goal,
        "waypoints": [],
        "path_nodes": [],
        "found": False,
        "reason": "NAV_PATH_UNAVAILABLE",
        "nav2_candidate_validation_status": status,
        "nav2_candidate_validation": validation,
        "nav2_profile": nav2_profile
        or (
            nav2_environment.get("profile_id")
            if isinstance(nav2_environment, dict)
            else None
        ),
        "nav2_environment": nav2_environment,
        "nav2_trav_map_filename": nav2_trav_map_filename,
        "requested_planner": "nav2_compute_path_to_pose",
        "object_approach_candidates": list(goal.get("object_approach_candidates") or []),
        "selected_object_approach": None,
    }


__all__ = [
    "classify_nav2_validation_error",
    "candidate_validation_failure_plan",
    "validate_object_approach_candidates",
]
