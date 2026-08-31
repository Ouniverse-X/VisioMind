from __future__ import annotations

from typing import Any

from . import door_gating as hovsg_door_gating
from .models import HOVSGSceneAsset


def build_geometric_waypoints(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    path_nodes: list[Any],
    goal: dict[str, Any],
    goal_position: dict[str, float],
    start: dict[str, Any],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    node_waypoints = [adapter._node_to_waypoint(scene.nav_graph, node_id) for node_id in path_nodes]
    if not node_waypoints:
        return []

    room_steps = room_steps_from_node_waypoints(node_waypoints)
    if goal.get("goal_type") == "object" and len(room_steps) <= 1:
        return [
            adapter._goal_waypoint(
                scene=scene,
                goal=goal,
                goal_position=goal_position,
                fallback=node_waypoints[-1],
            )
        ]
    if len(room_steps) >= 2:
        room_waypoints = build_room_transition_waypoints(
            adapter,
            scene=scene,
            room_steps=room_steps,
            goal=goal,
            goal_position=goal_position,
            start=start,
            context=context,
        )
        if room_waypoints:
            return room_waypoints

    waypoints: list[dict[str, Any]] = []
    for previous, current in zip(node_waypoints, node_waypoints[1:]):
        previous_room = previous.get("room_id")
        current_room = current.get("room_id")
        if previous_room and current_room and previous_room != current_room:
            transition = adapter._transition_waypoint(
                scene=scene,
                source_room_id=str(previous_room),
                target_room_id=str(current_room),
                fallback_from=previous,
                fallback_to=current,
                start=start,
                goal=goal,
                context=context,
            )
            if transition is not None:
                adapter._append_waypoint_if_distinct(waypoints, transition)
                continue
        adapter._append_waypoint_if_distinct(waypoints, dict(current))

    final_waypoint = adapter._goal_waypoint(
        scene=scene,
        goal=goal,
        goal_position=goal_position,
        fallback=node_waypoints[-1],
    )
    adapter._append_waypoint_if_distinct(waypoints, final_waypoint)

    if not waypoints:
        adapter._append_waypoint_if_distinct(
            waypoints,
            adapter._goal_waypoint(
                scene=scene,
                goal=goal,
                goal_position=goal_position,
                fallback=node_waypoints[-1],
            ),
        )
    return waypoints


def build_dense_waypoints(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    node_waypoints: list[dict[str, Any]],
    goal: dict[str, Any],
    goal_position: dict[str, float],
) -> list[dict[str, Any]]:
    if not node_waypoints:
        return []

    dense_waypoints: list[dict[str, Any]] = []
    for waypoint in node_waypoints:
        adapter._append_waypoint_if_distinct(dense_waypoints, dict(waypoint))
    final_waypoint = adapter._goal_waypoint(
        scene=scene,
        goal=goal,
        goal_position=goal_position,
        fallback=node_waypoints[-1],
    )
    adapter._append_waypoint_if_distinct(dense_waypoints, final_waypoint)
    return dense_waypoints


def build_room_transition_waypoints(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    room_steps: list[dict[str, Any]],
    goal: dict[str, Any],
    goal_position: dict[str, float],
    start: dict[str, Any],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    effective_steps = collapse_room_steps(
        adapter,
        scene=scene,
        room_steps=room_steps,
        goal_room_id=str(goal["room_id"]) if goal.get("room_id") is not None else None,
        start=start,
        goal=goal,
        context=context,
    )
    if len(effective_steps) < 2:
        return []

    waypoints: list[dict[str, Any]] = []
    for current_step, next_step in zip(effective_steps, effective_steps[1:]):
        current_room = current_step.get("room_id")
        next_room = next_step.get("room_id")
        if current_room and next_room and current_room != next_room:
            transition = adapter._transition_waypoint(
                scene=scene,
                source_room_id=str(current_room),
                target_room_id=str(next_room),
                fallback_from=current_step["exit_anchor"],
                fallback_to=next_step["entry_anchor"],
                start=start,
                goal=goal,
                context=context,
            )
            if transition is not None:
                adapter._append_waypoint_if_distinct(waypoints, transition)
                continue
        adapter._append_waypoint_if_distinct(waypoints, dict(next_step["entry_anchor"]))

    final_waypoint = adapter._goal_waypoint(
        scene=scene,
        goal=goal,
        goal_position=goal_position,
        fallback=effective_steps[-1]["exit_anchor"],
    )
    adapter._append_waypoint_if_distinct(waypoints, final_waypoint)
    return waypoints


def room_steps_from_node_waypoints(node_waypoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for waypoint in node_waypoints:
        room_id = waypoint.get("room_id")
        if steps and room_id is not None and steps[-1].get("room_id") == room_id:
            steps[-1]["exit_anchor"] = dict(waypoint)
            continue
        steps.append(
            {
                "room_id": room_id,
                "room_name": waypoint.get("room_name"),
                "floor_id": waypoint.get("floor_id"),
                "entry_anchor": dict(waypoint),
                "exit_anchor": dict(waypoint),
            }
        )
    return steps


def collapse_room_steps(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    room_steps: list[dict[str, Any]],
    goal_room_id: str | None,
    start: dict[str, Any],
    goal: dict[str, Any],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    if len(room_steps) < 3:
        return room_steps

    collapsed = [room_steps[0]]
    current_index = 0
    while current_index < len(room_steps) - 1:
        next_index = preferred_direct_room_step_index(
            adapter,
            scene=scene,
            room_steps=room_steps,
            current_index=current_index,
            goal_room_id=goal_room_id,
            start=start,
            goal=goal,
            context=context,
        )
        if next_index is None:
            next_index = current_index + 1
        collapsed.append(room_steps[next_index])
        current_index = next_index
    return collapsed


def preferred_direct_room_step_index(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    room_steps: list[dict[str, Any]],
    current_index: int,
    goal_room_id: str | None,
    start: dict[str, Any],
    goal: dict[str, Any],
    context: dict[str, Any],
) -> int | None:
    if current_index >= len(room_steps) - 2:
        return None

    current_room_id = room_steps[current_index].get("room_id")
    if not isinstance(current_room_id, str) or not current_room_id:
        return None

    preferred_index = None
    preferred_metrics = None
    for candidate_index in range(len(room_steps) - 1, current_index + 1, -1):
        candidate_room_id = room_steps[candidate_index].get("room_id")
        if (
            not isinstance(candidate_room_id, str)
            or not candidate_room_id
            or candidate_room_id == current_room_id
        ):
            continue
        metrics = strong_room_transition_metrics(
            adapter,
            scene=scene,
            source_room_id=current_room_id,
            target_room_id=candidate_room_id,
            start=start,
            goal=goal,
            context=context,
        )
        if metrics is None:
            continue
        if goal_room_id and candidate_room_id == goal_room_id:
            return candidate_index
        if preferred_metrics is None or metrics["span"] > preferred_metrics["span"]:
            preferred_index = candidate_index
            preferred_metrics = metrics
    return preferred_index


def strong_room_transition_metrics(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    source_room_id: str,
    target_room_id: str,
    start: dict[str, Any],
    goal: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any] | None:
    source_room = scene.rooms.get(source_room_id)
    target_room = scene.rooms.get(target_room_id)
    explicit_portal = adapter._explicit_transition_portal(
        scene=scene,
        source_room=source_room,
        target_room=target_room,
        context=context,
    )
    if explicit_portal is not None:
        if explicit_portal.get("portal_door_open") is False:
            return None
        return explicit_portal_transition_metrics(adapter, scene=scene, portal=explicit_portal)
    if hovsg_door_gating.room_pair_blocked(adapter, scene, source_room_id, target_room_id):
        return None
    if not rooms_are_directly_adjacent(
        scene,
        source_room_id=source_room_id,
        target_room_id=target_room_id,
        adapter=adapter,
    ):
        return None
    metrics = adapter._room_transition_metrics(
        scene,
        source_room,
        target_room,
        start=start,
        goal=goal,
        context=context,
    )
    if metrics is None:
        return None
    if metrics["gap"] > adapter.direct_room_transition_max_gap:
        return None
    if metrics["span"] < adapter.direct_room_transition_min_span:
        return None
    return metrics


def explicit_portal_transition_metrics(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    portal: dict[str, Any],
) -> dict[str, Any]:
    source_point = adapter._project_horizontal(scene, portal.get("portal_source_point") or {})
    target_point = adapter._project_horizontal(scene, portal.get("portal_target_point") or {})
    if source_point is None or target_point is None:
        source_point = target_point = adapter._project_horizontal(scene, portal)
    gap = float(portal.get("portal_gap", 0.0) or 0.0)
    if source_point is not None and target_point is not None and gap <= 0.0:
        gap = (
            (float(target_point[0]) - float(source_point[0])) ** 2
            + (float(target_point[1]) - float(source_point[1])) ** 2
        ) ** 0.5
    return {
        "source_point": tuple(source_point) if source_point is not None else None,
        "target_point": tuple(target_point) if target_point is not None else None,
        "gap": gap,
        "span": max(0.0, float(portal.get("portal_span", 0.0) or 0.0)),
        "explicit_portal": True,
    }


def rooms_are_directly_adjacent(
    scene: HOVSGSceneAsset,
    *,
    source_room_id: str,
    target_room_id: str,
    adapter: Any | None = None,
) -> bool:
    adjacency = (
        hovsg_door_gating.effective_room_adjacency(adapter, scene)
        if adapter is not None
        else scene.room_adjacency
    )
    if not adjacency:
        return True
    source_neighbors = adjacency.get(source_room_id)
    target_neighbors = adjacency.get(target_room_id)
    if source_neighbors is None and target_neighbors is None:
        return True
    if isinstance(source_neighbors, set) and target_room_id in source_neighbors:
        return True
    if isinstance(target_neighbors, set) and source_room_id in target_neighbors:
        return True
    return False
