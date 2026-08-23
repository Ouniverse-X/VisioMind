"""Object-approach runtime helpers for the Navigation agent."""

from __future__ import annotations

import re
from typing import Any

from voltron.shared.context import ExecutionContext, LocalSkillSelection, Subtask
from voltron.shared.contracts import MemoryAdapter


def should_use_object_approach_flow(
    *,
    subtask: Subtask,
    grounded_goal: dict[str, Any] | None,
) -> bool:
    if not isinstance(grounded_goal, dict):
        return False
    if str(grounded_goal.get("goal_type") or "").strip().lower() != "object":
        return False
    return bool(
        str(
            grounded_goal.get("object_id")
            or grounded_goal.get("object_name")
            or subtask.target.get("object_id")
            or subtask.target.get("object")
            or ""
        ).strip()
    )


def clone_optional_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return dict(value)
    return None


def object_approach_cache_bucket(context: ExecutionContext) -> dict[str, Any]:
    return context.runtime_state.setdefault("vln_object_approach_cache", {})


def object_approach_cache_key(subtask: Subtask) -> str:
    """Scope cached grounding and paths to one versioned subtask execution."""

    return subtask.runtime_id


def load_cached_object_approach_state(
    *,
    context: ExecutionContext,
    subtask: Subtask,
) -> dict[str, Any] | None:
    bucket = object_approach_cache_bucket(context)
    cached = bucket.get(object_approach_cache_key(subtask))
    if not isinstance(cached, dict):
        return None
    grounded_goal = cached.get("grounded_goal")
    if not isinstance(grounded_goal, dict):
        return None
    if not should_use_object_approach_flow(subtask=subtask, grounded_goal=grounded_goal):
        return None
    return cached


def restore_cached_object_approach_state(cached: dict[str, Any]) -> dict[str, Any]:
    return {
        "grounded_goal": dict(cached["grounded_goal"]),
        "navigation_skill_selection": clone_optional_dict(cached.get("navigation_skill_selection")),
        "prepared_navigation_payload": clone_optional_dict(cached.get("prepared_navigation_payload")),
        "object_approach_selection": clone_optional_dict(cached.get("object_approach_selection")),
        "selected_object_approach": clone_optional_dict(cached.get("selected_object_approach")),
        "path_plan": clone_optional_dict(cached.get("path_plan")),
        "interpreted_goal": clone_optional_dict(cached.get("interpreted_goal")),
        "navigation_grounding_context": clone_optional_dict(cached.get("navigation_grounding_context")),
    }


def store_cached_object_approach_state(
    *,
    context: ExecutionContext,
    subtask: Subtask,
    grounded_goal: dict[str, Any],
    navigation_skill_selection: dict[str, Any] | None,
    prepared_navigation_payload: dict[str, Any] | None,
    object_approach_selection: dict[str, Any] | None,
    selected_object_approach: dict[str, Any] | None,
    path_plan: dict[str, Any] | None = None,
    interpreted_goal: dict[str, Any] | None = None,
    navigation_grounding_context: dict[str, Any] | None = None,
) -> None:
    bucket = object_approach_cache_bucket(context)
    bucket[object_approach_cache_key(subtask)] = {
        "grounded_goal": dict(grounded_goal),
        "navigation_skill_selection": clone_optional_dict(navigation_skill_selection),
        "prepared_navigation_payload": clone_optional_dict(prepared_navigation_payload),
        "object_approach_selection": clone_optional_dict(object_approach_selection),
        "selected_object_approach": clone_optional_dict(selected_object_approach),
        "path_plan": clone_optional_dict(path_plan),
        "interpreted_goal": clone_optional_dict(interpreted_goal),
        "navigation_grounding_context": clone_optional_dict(navigation_grounding_context),
        "scene_revision": _scene_revision_from_goal(
            grounded_goal,
            selected_object_approach,
        ),
    }


def clear_cached_object_approach_state(
    *,
    context: ExecutionContext,
    subtask: Subtask,
) -> None:
    bucket = object_approach_cache_bucket(context)
    bucket.pop(object_approach_cache_key(subtask), None)


def should_replan_cached_object_approach(
    *,
    subtask: Subtask,
    observation: dict[str, Any],
) -> bool:
    recovery_mode = str(subtask.parameters.get("recovery_mode") or "").strip().lower()
    if recovery_mode and recovery_mode not in {"progress_stalled", "feedback_blocked", "oscillation_detected"}:
        return True
    nav_feedback = observation.get("nav_feedback")
    if not isinstance(nav_feedback, dict):
        return False
    return bool(nav_feedback.get("stuck") or nav_feedback.get("collision"))


def should_reuse_cached_path_plan(
    *,
    cached_object_approach: dict[str, Any] | None,
    path_plan: dict[str, Any] | None,
    subtask: Subtask,
    observation: dict[str, Any],
) -> bool:
    if cached_dynamic_local_segment_completed(subtask=subtask, path_plan=path_plan):
        return False
    cached_revision = (
        cached_object_approach.get("scene_revision")
        if isinstance(cached_object_approach, dict)
        else None
    )
    current_revision = _scene_revision_from_observation(observation)
    if (
        isinstance(cached_revision, dict)
        and cached_revision
        and isinstance(current_revision, dict)
        and current_revision
        and any(
            cached_revision.get(key)
            and current_revision.get(key)
            and cached_revision[key] != current_revision[key]
            for key in ("scene_state_signature", "relation_signature", "map_revision")
        )
    ):
        return False
    return bool(
        cached_object_approach is not None
        and path_plan is not None
        and not should_replan_cached_object_approach(subtask=subtask, observation=observation)
    )


def _scene_revision_from_goal(
    goal: dict[str, Any],
    selected_candidate: dict[str, Any] | None,
) -> dict[str, Any]:
    revision = {
        key: goal.get(key)
        for key in (
            "scene_state_signature",
            "relation_signature",
            "map_revision",
        )
        if goal.get(key)
    }
    if isinstance(selected_candidate, dict) and selected_candidate.get(
        "candidate_cache_revision"
    ):
        revision["candidate_cache_revision"] = selected_candidate[
            "candidate_cache_revision"
        ]
    return revision


def _scene_revision_from_observation(
    observation: dict[str, Any],
) -> dict[str, Any]:
    scene_state = observation.get("scene_state")
    if not isinstance(scene_state, dict):
        return {}
    return {
        "scene_state_signature": scene_state.get("signature"),
        "relation_signature": scene_state.get("relation_signature"),
        "map_revision": scene_state.get("map_revision"),
    }


def filter_candidates_for_goal_room(
    *,
    candidates: list[dict[str, Any]],
    goal: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_room_id = str(goal.get("room_id") or "").strip()
    target_room_name = normalize_label(goal.get("room_name") or goal.get("room") or goal.get("region"))
    if not target_room_id and target_room_name is None:
        return [dict(candidate) for candidate in candidates], []

    matching: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        candidate_room_id = str(item.get("room_id") or "").strip()
        candidate_room_name = normalize_label(item.get("room_name") or item.get("room") or item.get("region"))
        room_matches = bool(target_room_id and candidate_room_id == target_room_id) or bool(
            target_room_name and candidate_room_name == target_room_name
        )
        if room_matches:
            matching.append(item)
        else:
            rejected.append(item)

    if matching:
        return matching, rejected
    return [dict(candidate) for candidate in candidates], []


def normalize_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split()).strip()
    return normalized or None


def cached_dynamic_local_segment_completed(
    *,
    subtask: Subtask,
    path_plan: dict[str, Any] | None,
) -> bool:
    if not isinstance(path_plan, dict):
        return False
    if not is_dynamic_local_segment_path(path_plan):
        return False

    parameters = subtask.parameters if isinstance(subtask.parameters, dict) else {}
    if bool(
        parameters.get("requires_replan")
        or parameters.get("local_segment_complete")
    ):
        return True
    controller_mode = str(parameters.get("controller_mode") or "").strip().lower()
    if bool(parameters.get("goal_reached")) or controller_mode == "goal_reached":
        return True

    active_index = coerce_nonnegative_int(parameters.get("active_waypoint_index"))
    if active_index is None:
        return False
    nav2_path_points = path_plan.get("nav2_path_points")
    if isinstance(nav2_path_points, list) and nav2_path_points:
        return active_index >= len(nav2_path_points)
    waypoints = path_plan.get("waypoints")
    if isinstance(waypoints, list) and active_index >= len(waypoints):
        return True
    return False


def is_dynamic_local_segment_path(path_plan: dict[str, Any]) -> bool:
    return any(
        str(path_plan.get(key) or "").strip().lower() == expected
        for key, expected in (
            ("waypoint_scope", "dynamic_local_segment"),
            ("waypoint_tracking_mode", "global_local_hybrid"),
            ("path_backend", "nav2_local"),
        )
    )


def coerce_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    if index < 0:
        return None
    return index


def prime_object_approach_history(
    *,
    memory: MemoryAdapter,
    subtask: Subtask,
    scene_id: str | None,
    goal: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not scene_id or not isinstance(goal, dict):
        return None
    if str(goal.get("goal_type") or "").strip().lower() != "object":
        return None
    target = {
        "object": goal.get("object_name") or subtask.target.get("object"),
        "object_id": goal.get("object_id") or subtask.target.get("object_id"),
        "room_id": goal.get("room_id") or subtask.target.get("room_id"),
        "room_name": goal.get("room_name") or subtask.target.get("room") or subtask.target.get("region"),
        "floor_id": goal.get("floor_id") or subtask.target.get("floor_id"),
    }
    return memory.get_object_approach_history(scene_id=scene_id, target=target, top_k=10)


def apply_prepared_payload_candidates(
    *,
    context: ExecutionContext,
    subtask: Subtask,
    grounded_goal: dict[str, Any],
    prepared_navigation_payload: dict[str, Any] | None,
    approach_point_selector: Any,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    next_goal = dict(grounded_goal)
    object_approach_selection: dict[str, Any] | None = None
    selected_object_approach: dict[str, Any] | None = None

    if not isinstance(prepared_navigation_payload, dict):
        return next_goal, object_approach_selection, selected_object_approach

    prepared_history = prepared_navigation_payload.get("history")
    if isinstance(prepared_history, dict):
        context.runtime_state["object_approach_history"] = prepared_history

    candidates = list(prepared_navigation_payload.get("candidates") or [])
    if not candidates:
        return next_goal, object_approach_selection, selected_object_approach

    candidates, room_rejected_candidates = filter_candidates_for_goal_room(
        candidates=[dict(candidate) for candidate in candidates],
        goal=next_goal,
    )
    filtered_navigation_payload = dict(prepared_navigation_payload)
    filtered_navigation_payload["candidates"] = candidates
    next_goal["object_approach_candidates"] = candidates
    object_approach_selection = dict(
        approach_point_selector.select_candidate(
            subtask=subtask,
            context=context,
            goal=next_goal,
            prepared_payload=filtered_navigation_payload,
        )
    )
    if room_rejected_candidates:
        object_approach_selection["room_rejected_candidate_ids"] = [
            candidate.get("candidate_id") for candidate in room_rejected_candidates
        ]
    selected_candidate = object_approach_selection.get("candidate")
    if isinstance(selected_candidate, dict) and selected_candidate:
        selected_object_approach = dict(selected_candidate)
        next_goal["selected_object_approach"] = selected_object_approach
        context.runtime_state["selected_object_approach"] = dict(selected_object_approach)
    elif selected_candidate is None:
        next_goal["object_approach_selection_failed"] = True
        next_goal["object_approach_selection_failure_reason"] = str(
            object_approach_selection.get("reason") or "object-approach candidate selection failed"
        )
        context.runtime_state.pop("selected_object_approach", None)
    return next_goal, object_approach_selection, selected_object_approach


def serialize_skill_selection(selection: LocalSkillSelection) -> dict[str, Any]:
    return {
        "skill_id": selection.skill_id,
        "confidence": selection.confidence,
        "reason": selection.reason,
        "source": selection.source,
        "fallback_skill_candidates": list(selection.fallback_skill_candidates),
        "metadata": dict(selection.metadata),
    }
