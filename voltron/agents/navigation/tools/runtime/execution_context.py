"""Execution-context helpers for the Navigation agent."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from . import observation as runtime_observation
from voltron.shared.context import ExecutionContext, MemorySnapshot, ObservationContext, Subtask


def resolve_instruction(subtask: Subtask) -> str:
    for candidate in (
        subtask.parameters.get("instruction"),
        subtask.context.get("instruction"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    region = str(subtask.target.get("region", "")).strip()
    room = str(subtask.target.get("room", "")).strip()
    target = room or region
    if target:
        return f"{subtask.action} to {target}"
    return subtask.action


def build_navigation_context(
    *,
    memory: Any,
    subtask: Subtask,
    context: ExecutionContext,
    observation: dict[str, Any],
    scene_id: str | None,
    map_state: dict[str, Any],
) -> dict[str, Any]:
    scene_map = memory.load_map(scene_id) if scene_id else None
    memory_snapshot = MemorySnapshot.from_memory(memory, recent_observation_limit=5)
    observation_context = ObservationContext.from_observation(
        observation,
        metadata={
            "trace_id": context.trace_id,
            "task_id": context.task_request.task_id,
            "subtask_id": subtask.subtask_id,
        },
    )
    nav_context = {
        "trace_id": context.trace_id,
        "task_id": context.task_request.task_id,
        "task_description": context.task_request.description,
        "scene_id": scene_id,
        "subtask_id": subtask.subtask_id,
        "action": subtask.action,
        "target": dict(subtask.target),
        "parameters": dict(subtask.parameters),
        "context": dict(subtask.context),
        "map_state": dict(map_state),
        "scene_map": scene_map,
        **memory_snapshot.to_planning_context(),
        "observation_context": observation_context.to_payload(),
        "observation_keys": observation_context.observation_keys,
    }
    _copy_runtime_navigation_metadata(
        nav_context,
        sources=(subtask.parameters, subtask.context, observation, map_state),
    )
    return nav_context


def _copy_runtime_navigation_metadata(target: dict[str, Any], *, sources: tuple[dict[str, Any], ...]) -> None:
    for key in (
        "behavior_scene_file",
        "scene_file",
        "nav2_trav_map_filename",
        "portal_annotations",
        "transition_portals",
        "portals",
        "scene_state",
    ):
        for source in sources:
            value = source.get(key)
            if _has_runtime_metadata_value(value):
                target[key] = deepcopy(value)
                break


def _has_runtime_metadata_value(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (dict, list)):
        return bool(value)
    return value is not None


def build_start_state(
    *,
    subtask: Subtask,
    context: ExecutionContext,
    observation: dict[str, Any],
    backend_state: dict[str, Any] | None,
) -> dict[str, Any]:
    start = runtime_observation.extract_start_metadata(backend_state)
    pose = runtime_observation.extract_pose(subtask=subtask, observation=observation)
    orientation = runtime_observation.extract_orientation(subtask=subtask, observation=observation)
    if pose is not None:
        start["pose"] = pose
        if orientation is not None:
            start["orientation"] = dict(orientation)
        start["source"] = "runtime"
        region = runtime_observation.extract_region(subtask=subtask, observation=observation)
        if region:
            start["region"] = region
        return start
    if backend_state and isinstance(backend_state.get("pose"), dict):
        start["pose"] = dict(backend_state["pose"])
        if orientation is not None:
            start["orientation"] = dict(orientation)
        start["source"] = "navigator"
        return start
    return {
        **start,
        "trace_id": context.trace_id,
        "subtask_id": subtask.subtask_id,
        "source": "fallback",
    }


def merge_map_state(map_state: dict[str, Any], backend_state: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(backend_state, dict):
        map_state.update(backend_state)
    return map_state


def refresh_navigation_context(
    nav_context: dict[str, Any],
    *,
    map_state: dict[str, Any],
    backend_state: dict[str, Any] | None,
) -> None:
    nav_context["map_state"] = dict(map_state)
    if isinstance(backend_state, dict):
        nav_context["backend_state"] = dict(backend_state)


def sync_scene_map(
    *,
    memory: Any,
    scene_id: str | None,
    backend_state: dict[str, Any] | None,
    grounded_goal: dict[str, Any] | None,
    path_plan: dict[str, Any] | None,
) -> None:
    if not scene_id:
        return
    backend_state_payload = dict(backend_state or {})
    scene_map_seed = backend_state_payload.pop("scene_map_seed", None)
    scene_map_seed_metadata = backend_state_payload.pop("scene_map_seed_metadata", None)
    if isinstance(scene_map_seed, dict):
        seed_scene_id = str(scene_map_seed.get("scene_id") or scene_id)
        memory.save_map(
            seed_scene_id,
            scene_map_seed,
            metadata=(
                dict(scene_map_seed_metadata)
                if isinstance(scene_map_seed_metadata, dict)
                else {}
            ),
        )
    source_action_ids = _record_navigation_lineage_events(
        memory=memory,
        scene_id=scene_id,
        backend_state=backend_state_payload,
        grounded_goal=grounded_goal,
        path_plan=path_plan,
    )
    metadata: dict[str, Any] = {"source_agent": "NAVIGATION"}
    if source_action_ids:
        metadata["source_action_id"] = source_action_ids[-1]
    memory.update_map(
        scene_id,
        {
            "navigation": {
                "backend_state": backend_state_payload,
                "last_grounded_goal": dict(grounded_goal or {}),
                "last_path_plan": dict(path_plan or {}),
                "source_action_ids": list(source_action_ids),
            }
        },
        metadata=metadata,
    )


def _record_navigation_lineage_events(
    *,
    memory: Any,
    scene_id: str,
    backend_state: dict[str, Any],
    grounded_goal: dict[str, Any] | None,
    path_plan: dict[str, Any] | None,
) -> list[str]:
    record_event = getattr(memory, "record_navigation_event", None)
    if not callable(record_event):
        return []

    events = _navigation_lineage_events(
        scene_id=scene_id,
        backend_state=backend_state,
        grounded_goal=grounded_goal,
        path_plan=path_plan,
    )
    action_ids: list[str] = []
    for event in events:
        result = record_event(event)
        if isinstance(result, dict) and result.get("action_id"):
            action_ids.append(str(result["action_id"]))
    return action_ids


def _navigation_lineage_events(
    *,
    scene_id: str,
    backend_state: dict[str, Any],
    grounded_goal: dict[str, Any] | None,
    path_plan: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    selected_object_approach = _first_mapping(
        _mapping_value(path_plan, "selected_object_approach"),
        _mapping_value(grounded_goal, "selected_object_approach"),
    )
    if selected_object_approach:
        events.append(
            {
                "event_type": "object_approach_candidate_selected",
                "scene_id": scene_id,
                "target": dict(grounded_goal or {}),
                "backend_state": dict(backend_state),
                "selected_object_approach": selected_object_approach,
                "success": True,
            }
        )

    if isinstance(path_plan, dict):
        found = path_plan.get("found", True)
        event_type = "route_planned" if found is not False else "path_unavailable"
        events.append(
            {
                "event_type": event_type,
                "scene_id": scene_id,
                "target": dict(grounded_goal or {}),
                "backend_state": dict(backend_state),
                "path_plan": dict(path_plan),
                "success": found is not False,
                "failure_reason": path_plan.get("reason") or path_plan.get("error"),
            }
        )
    return events


def _mapping_value(payload: dict[str, Any] | None, key: str) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    return dict(value) if isinstance(value, dict) else None


def _first_mapping(*values: dict[str, Any] | None) -> dict[str, Any] | None:
    for value in values:
        if value:
            return value
    return None


def waypoint_count(path_plan: dict[str, Any] | None) -> int:
    if not isinstance(path_plan, dict):
        return 0
    waypoints = path_plan.get("waypoints")
    if not isinstance(waypoints, list):
        return 0
    return len(waypoints)
