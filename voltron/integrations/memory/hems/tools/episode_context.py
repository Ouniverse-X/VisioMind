"""Completed-episode context helpers for MemoryAgent extraction."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable


def build_scene_memory_context(
    *,
    maps: dict[str, dict[str, Any]],
    serializer: Callable[[Any], Any],
    max_scenes: int = 5,
    max_items: int = 8,
    max_entries_per_target: int = 3,
) -> dict[str, Any]:
    """Build a bounded scene-memory summary suitable for LLM extraction input."""

    scenes = []
    for scene_id, entry in list(maps.items())[:max(0, int(max_scenes))]:
        if not isinstance(entry, dict):
            continue
        scenes.append(
            summarize_scene_entry(
                scene_id=scene_id,
                entry=entry,
                serializer=serializer,
                max_items=max_items,
                max_entries_per_target=max_entries_per_target,
            )
        )
    return {
        "scene_count": len(maps),
        "scenes": scenes,
    }


def summarize_scene_entry(
    *,
    scene_id: str,
    entry: dict[str, Any],
    serializer: Callable[[Any], Any],
    max_items: int,
    max_entries_per_target: int,
) -> dict[str, Any]:
    map_payload = entry.get("map_payload", {})
    if not isinstance(map_payload, dict):
        map_payload = {}

    summary: dict[str, Any] = {
        "scene_id": str(scene_id),
        "metadata": serializer(entry.get("metadata", {})),
        "map_keys": sorted(map_payload),
    }

    navigation = summarize_navigation(
        map_payload.get("navigation", {}),
        serializer=serializer,
        max_items=max_items,
    )
    if navigation:
        summary["navigation"] = navigation

    object_approach = summarize_object_approach_memory(
        map_payload.get("object_approach_memory", {}),
        serializer=serializer,
        max_items=max_items,
        max_entries_per_target=max_entries_per_target,
    )
    if object_approach:
        summary["object_approach_memory"] = object_approach

    obstacles = _limited_list(map_payload.get("obstacles", []), max_items=max_items, serializer=serializer)
    if obstacles:
        summary["obstacles"] = obstacles

    exploration = summarize_exploration(
        map_payload.get("exploration", {}),
        serializer=serializer,
        max_items=max_items,
    )
    if exploration:
        summary["exploration"] = exploration

    return summary


def summarize_navigation(
    navigation: Any,
    *,
    serializer: Callable[[Any], Any],
    max_items: int,
) -> dict[str, Any]:
    if not isinstance(navigation, dict):
        return {}

    summary: dict[str, Any] = {}
    for key in ("last_region", "visited_regions", "last_pose"):
        if key in navigation:
            value = navigation[key]
            if key == "visited_regions" and isinstance(value, list):
                value = value[:max_items]
            summary[key] = serializer(value)

    backend_state = _select_keys(
        navigation.get("backend_state"),
        (
            "scene_id",
            "pose",
            "vertical_axis",
            "current_room",
            "current_region",
            "room_id",
            "floor_id",
            "nav_backend",
            "nav2_profile",
            "active_waypoint_index",
            "localization_guard",
        ),
        serializer=serializer,
    )
    if backend_state:
        summary["backend_state"] = backend_state

    grounded_goal = summarize_grounded_goal(
        navigation.get("last_grounded_goal"),
        serializer=serializer,
    )
    if grounded_goal:
        summary["last_grounded_goal"] = grounded_goal

    path_plan = summarize_path_plan(
        navigation.get("last_path_plan"),
        serializer=serializer,
    )
    if path_plan:
        summary["last_path_plan"] = path_plan

    selected = _first_mapping_value(
        path_plan.get("selected_object_approach") if isinstance(path_plan, dict) else None,
        grounded_goal.get("selected_object_approach") if isinstance(grounded_goal, dict) else None,
    )
    if selected:
        summary["selected_object_approach"] = serializer(selected)

    return summary


def summarize_grounded_goal(
    grounded_goal: Any,
    *,
    serializer: Callable[[Any], Any],
) -> dict[str, Any]:
    if not isinstance(grounded_goal, dict):
        return {}

    summary = _select_keys(
        grounded_goal,
        (
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
            "selected_grounding_candidate",
            "selected_object_approach",
            "nav_backend",
            "nav2_profile",
        ),
        serializer=serializer,
    )
    _add_count(
        summary,
        grounded_goal,
        source_key="grounding_candidates",
        count_key="grounding_candidate_count",
    )
    _add_count(
        summary,
        grounded_goal,
        source_key="object_approach_candidates",
        count_key="object_approach_candidate_count",
    )
    return summary


def summarize_path_plan(
    path_plan: Any,
    *,
    serializer: Callable[[Any], Any],
) -> dict[str, Any]:
    if not isinstance(path_plan, dict):
        return {}

    summary = _select_keys(
        path_plan,
        (
            "planner",
            "path_backend",
            "scene_id",
            "vertical_axis",
            "found",
            "path_cost",
            "frame_id",
            "planner_id",
            "action_name",
            "requested_planner",
            "nav2_profile",
            "nav2_environment",
            "nav2_error",
            "nav2_cache_reused",
            "nav2_path_clipped_for_clearance",
            "waypoint_tracking_mode",
            "waypoint_scope",
            "local_goal",
            "execution_goal",
            "transition_anchor",
            "doorway_corridor",
            "selected_object_approach",
        ),
        serializer=serializer,
    )
    for source_key, count_key in (
        ("waypoints", "waypoint_count"),
        ("path_nodes", "path_node_count"),
        ("global_waypoints", "global_waypoint_count"),
        ("dense_waypoints", "dense_waypoint_count"),
        ("object_approach_candidates", "object_approach_candidate_count"),
    ):
        _add_count(summary, path_plan, source_key=source_key, count_key=count_key)
    return summary


def summarize_object_approach_memory(
    object_memory: Any,
    *,
    serializer: Callable[[Any], Any],
    max_items: int,
    max_entries_per_target: int,
) -> dict[str, Any]:
    if not isinstance(object_memory, dict) or not object_memory:
        return {}

    targets = []
    for target_key, bucket in list(object_memory.items())[:max_items]:
        if not isinstance(bucket, dict):
            continue
        entries = bucket.get("entries", [])
        if not isinstance(entries, list):
            entries = []
        targets.append(
            {
                "target_key": str(target_key),
                "target": serializer(bucket.get("target", {})),
                "entry_count": len(entries),
                "entries": [
                    summarize_object_approach_entry(entry, serializer=serializer)
                    for entry in entries[-max_entries_per_target:]
                    if isinstance(entry, dict)
                ],
            }
        )
    return {
        "target_count": len(object_memory),
        "targets": targets,
    }


def summarize_object_approach_entry(
    entry: dict[str, Any],
    *,
    serializer: Callable[[Any], Any],
) -> dict[str, Any]:
    summary = _select_keys(
        entry,
        ("timestamp", "outcome", "reason", "candidate_signature", "metadata"),
        serializer=serializer,
    )
    candidate = entry.get("candidate")
    if isinstance(candidate, dict):
        summary["candidate"] = _select_keys(
            candidate,
            (
                "candidate_id",
                "nav_node",
                "x",
                "y",
                "z",
                "floor_id",
                "room_id",
                "room_name",
                "waypoint_type",
                "object_id",
                "object_name",
                "approach_distance_m",
                "path_cost",
                "selection_source",
            ),
            serializer=serializer,
        )
    return summary


def summarize_exploration(
    exploration: Any,
    *,
    serializer: Callable[[Any], Any],
    max_items: int,
) -> dict[str, Any]:
    if not isinstance(exploration, dict):
        return {}
    summary: dict[str, Any] = {}
    if "frontiers" in exploration:
        summary["frontiers"] = _limited_list(exploration.get("frontiers"), max_items=max_items, serializer=serializer)
    if "explored_regions" in exploration:
        summary["explored_regions"] = _limited_list(
            exploration.get("explored_regions"),
            max_items=max_items,
            serializer=serializer,
        )
    if "evidence" in exploration:
        evidence = exploration.get("evidence")
        if isinstance(evidence, list):
            summary["evidence"] = [serializer(item) for item in evidence[-max_items:]]
    return {key: value for key, value in summary.items() if value}


def _select_keys(value: Any, keys: tuple[str, ...], *, serializer: Callable[[Any], Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    selected = {}
    for key in keys:
        if key not in value:
            continue
        current = value[key]
        if current is None or current == {} or current == []:
            continue
        selected[key] = serializer(deepcopy(current))
    return selected


def _limited_list(value: Any, *, max_items: int, serializer: Callable[[Any], Any]) -> list[Any]:
    if not isinstance(value, list):
        return []
    return [serializer(item) for item in value[:max_items]]


def _add_count(summary: dict[str, Any], source: dict[str, Any], *, source_key: str, count_key: str) -> None:
    value = source.get(source_key)
    if isinstance(value, list):
        summary[count_key] = len(value)


def _first_mapping_value(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict) and value:
            return value
    return {}


__all__ = [
    "build_scene_memory_context",
    "summarize_scene_entry",
]
