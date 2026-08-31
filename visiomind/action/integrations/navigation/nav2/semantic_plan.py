from __future__ import annotations

from typing import Any


def valid_waypoint_override(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    waypoint = dict(value)
    position = waypoint_position(waypoint)
    if position is None:
        return None
    waypoint.update(position)
    return waypoint


def normalize_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.lower().replace("_", " ").split()).strip()
    return normalized or None


def waypoint_position(waypoint: dict[str, Any]) -> dict[str, float] | None:
    coords: dict[str, float] = {}
    for axis in ("x", "y", "z"):
        value = waypoint.get(axis)
        if not isinstance(value, (int, float)):
            return None
        coords[axis] = float(value)
    return coords


def normalize_waypoints(raw_waypoints: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_waypoints, list):
        return []
    waypoints: list[dict[str, Any]] = []
    for raw_waypoint in raw_waypoints:
        if not isinstance(raw_waypoint, dict):
            continue
        waypoint = dict(raw_waypoint)
        position = waypoint_position(waypoint)
        if position is None:
            continue
        waypoint.update(position)
        waypoints.append(waypoint)
    return waypoints


def normalize_semantic_plan(plan: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(plan, dict):
        return {}

    normalized = dict(plan)
    normalized["waypoints"] = normalize_waypoints(plan.get("waypoints"))
    normalized["dense_waypoints"] = normalize_waypoints(plan.get("dense_waypoints"))
    for key in ("execution_goal", "transition_anchor", "nav2_compute_goal"):
        normalized_waypoint = valid_waypoint_override(plan.get(key))
        if normalized_waypoint is None:
            normalized.pop(key, None)
            continue
        normalized[key] = normalized_waypoint
    return normalized


def resolve_local_execution_goal(
    *,
    semantic_plan: dict[str, Any] | None,
    global_waypoints: list[dict[str, Any]],
    active_index: int,
    current_region: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    explicit_transition_anchor = None
    object_approach_goal = None
    if isinstance(semantic_plan, dict):
        explicit_execution_goal = valid_waypoint_override(semantic_plan.get("execution_goal"))
        explicit_transition_anchor = valid_waypoint_override(semantic_plan.get("transition_anchor"))
        if explicit_execution_goal is not None:
            return dict(explicit_execution_goal), (
                dict(explicit_transition_anchor) if explicit_transition_anchor is not None else None
            )
        object_approach_goal = _selected_object_approach_goal(semantic_plan)
        if object_approach_goal is not None and _goal_matches_current_region(
            goal=object_approach_goal,
            current_region=current_region,
        ):
            return dict(object_approach_goal), None

    local_goal = dict(global_waypoints[active_index])
    waypoint_type = str(local_goal.get("waypoint_type", "")).strip().lower()
    if waypoint_type != "portal":
        if object_approach_goal is not None:
            return dict(object_approach_goal), dict(
                explicit_transition_anchor
            ) if explicit_transition_anchor is not None else None
        return local_goal, dict(
            explicit_transition_anchor
        ) if explicit_transition_anchor is not None else None

    if explicit_transition_anchor is not None:
        post_transition = _final_execution_goal(
            global_waypoints=global_waypoints,
            active_index=active_index,
            object_approach_goal=object_approach_goal,
        )
        post_transition.setdefault("transition_anchor", dict(explicit_transition_anchor))
        post_transition["waypoint_type"] = "post_portal_goal"
        return post_transition, dict(explicit_transition_anchor)

    transition_index = active_index
    transition_anchor = local_goal
    if not portal_requires_segmented_navigation(local_goal):
        transition_index = _next_segmented_portal_index(
            global_waypoints=global_waypoints,
            start_index=active_index + 1,
        )
        if transition_index is None:
            final_goal = _final_execution_goal(
                global_waypoints=global_waypoints,
                active_index=active_index,
                object_approach_goal=object_approach_goal,
            )
            return final_goal, (
                dict(explicit_transition_anchor) if explicit_transition_anchor is not None else None
            )
        transition_anchor = dict(global_waypoints[transition_index])

    next_index = transition_index + 1
    if next_index >= len(global_waypoints):
        return transition_anchor, (
            dict(explicit_transition_anchor) if explicit_transition_anchor is not None else None
        )

    post_transition = (
        dict(object_approach_goal)
        if object_approach_goal is not None
        else dict(global_waypoints[next_index])
    )
    post_transition.setdefault("transition_anchor", dict(transition_anchor))
    post_transition["waypoint_type"] = "post_portal_goal"
    if explicit_transition_anchor is not None:
        return post_transition, dict(explicit_transition_anchor)
    return post_transition, dict(transition_anchor)


def portal_requires_segmented_navigation(waypoint: dict[str, Any]) -> bool:
    for key in (
        "requires_segmented_navigation",
        "portal_requires_segmented_navigation",
        "requires_state_transition",
        "state_gated",
        "interaction_required",
    ):
        if waypoint.get(key) is True:
            return True

    if isinstance(waypoint.get("portal_door_open"), bool):
        return True

    for key in (
        "portal_source",
        "portal_type",
        "transition_type",
        "transport_type",
        "portal_object_category",
        "portal_object_name",
    ):
        if _contains_segmented_portal_keyword(waypoint.get(key)):
            return True
    return False


def _next_segmented_portal_index(
    *,
    global_waypoints: list[dict[str, Any]],
    start_index: int,
) -> int | None:
    for index in range(max(0, start_index), len(global_waypoints)):
        waypoint = global_waypoints[index]
        if str(waypoint.get("waypoint_type") or "").strip().lower() != "portal":
            continue
        if portal_requires_segmented_navigation(waypoint):
            return index
    return None


def _final_execution_goal(
    *,
    global_waypoints: list[dict[str, Any]],
    active_index: int,
    object_approach_goal: dict[str, Any] | None,
) -> dict[str, Any]:
    if object_approach_goal is not None:
        return dict(object_approach_goal)
    for waypoint in reversed(global_waypoints[active_index + 1 :]):
        if str(waypoint.get("waypoint_type") or "").strip().lower() != "portal":
            return dict(waypoint)
    return dict(global_waypoints[-1])


def _contains_segmented_portal_keyword(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.lower()
    for separator in ("_", "-", "/", ":", "."):
        normalized = normalized.replace(separator, " ")
    tokens = set(normalized.split())
    return bool(
        tokens
        & {
            "airlock",
            "door",
            "elevator",
            "escalator",
            "gate",
            "lift",
            "stair",
            "staircase",
            "stairs",
            "turnstile",
        }
    )


def _selected_object_approach_goal(semantic_plan: dict[str, Any]) -> dict[str, Any] | None:
    selected = valid_waypoint_override(semantic_plan.get("selected_object_approach"))
    if selected is None:
        return None
    if str(selected.get("waypoint_type") or "").strip().lower() not in {
        "object_approach",
        "goal",
        "post_portal_goal",
    }:
        return None
    return selected


def _goal_matches_current_region(*, goal: dict[str, Any], current_region: str | None) -> bool:
    current_region_norm = normalize_label(current_region)
    if current_region_norm is None:
        return False
    for key in ("room_name", "room", "region", "current_region"):
        if normalize_label(goal.get(key)) == current_region_norm:
            return True
    return False


def resolve_nav2_compute_goal(
    *,
    semantic_plan: dict[str, Any] | None,
    current_region: str | None,
    execution_goal: dict[str, Any],
    transition_anchor: dict[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(semantic_plan, dict):
        explicit_compute_goal = valid_waypoint_override(semantic_plan.get("nav2_compute_goal"))
        if explicit_compute_goal is not None:
            return dict(explicit_compute_goal)

    if not isinstance(transition_anchor, dict):
        return dict(execution_goal)
    current_region_norm = normalize_label(current_region)
    target_region_norm = normalize_label(
        transition_anchor.get("room_name") or execution_goal.get("room_name")
    )
    if target_region_norm is None or current_region_norm != target_region_norm:
        return dict(transition_anchor)
    return dict(execution_goal)


def is_pre_transition_stage(
    *,
    current_region: str | None,
    execution_goal: dict[str, Any],
    transition_anchor: dict[str, Any] | None,
) -> bool:
    if not isinstance(transition_anchor, dict):
        return False
    current_region_norm = normalize_label(current_region)
    target_region_norm = normalize_label(
        transition_anchor.get("room_name") or execution_goal.get("room_name")
    )
    return bool(target_region_norm and current_region_norm != target_region_norm)
