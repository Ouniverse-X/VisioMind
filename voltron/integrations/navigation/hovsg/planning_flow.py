from __future__ import annotations

from typing import Any

import networkx as nx

from . import door_gating as hovsg_door_gating
from . import object_approach as hovsg_object_approach
from . import scene_loading as hovsg_scene_loading
from .models import HOVSGSceneAsset


def generate_object_approach_candidates(
    adapter: Any,
    *,
    start: dict[str, Any],
    goal: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    context = dict(context or {})
    scene_id = str(
        goal.get("scene_id") or start.get("scene_id") or context.get("scene_id") or ""
    ).strip()
    scene = adapter._ensure_scene(scene_id) if scene_id else None
    if scene is None:
        return []

    start_pose = start.get("pose")
    if not isinstance(start_pose, dict):
        return []

    start_floor = adapter._infer_floor_id(scene, start) or adapter._infer_floor_from_height(
        scene, start_pose
    )
    start_room_id = adapter._infer_room_id(scene, start)
    start_node = adapter._nearest_nav_node(
        scene.nav_graph,
        start_pose,
        floor_id=start_floor,
        room_id=start_room_id,
    )
    candidates = adapter._build_object_approach_candidates(
        scene=scene,
        goal=goal,
        start=start,
        context=context,
    )
    if start_node is None:
        return candidates
    return adapter._score_object_approach_candidates(
        scene=scene,
        start_node=start_node,
        candidates=candidates,
    )


def plan_path(
    adapter: Any,
    *,
    start: dict[str, Any],
    goal: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = dict(context or {})
    scene_id = str(
        goal.get("scene_id") or start.get("scene_id") or context.get("scene_id") or ""
    ).strip()
    scene = adapter._ensure_scene(scene_id) if scene_id else None
    if scene is None:
        return _planner_result(
            scene_id=scene_id or None,
            vertical_axis=scene.vertical_axis if scene is not None else None,
            nav_graph_type=None,
            nav_graph_asset_path=None,
            start=start,
            goal=goal,
            found=False,
            reason="scene_unavailable",
        )

    start_pose = start.get("pose", {})
    start = adapter._refine_start_anchor(scene=scene, start=start)
    start_pose = start.get("pose", {})
    raw_goal_position = adapter._goal_position(scene, goal)
    if not isinstance(start_pose, dict) or raw_goal_position is None:
        return _planner_result(
            scene_id=scene_id,
            vertical_axis=scene.vertical_axis,
            nav_graph_type=scene.nav_graph_type,
            nav_graph_asset_path=scene.nav_graph_asset_path,
            start=start,
            goal=goal,
            found=False,
            reason="pose_or_goal_missing",
        )
    if goal.get("goal_type") == "object" and goal.get("object_approach_selection_failed"):
        result = _planner_result(
            scene_id=scene_id,
            vertical_axis=scene.vertical_axis,
            nav_graph_type=scene.nav_graph_type,
            nav_graph_asset_path=scene.nav_graph_asset_path,
            start=start,
            goal=goal,
            found=False,
            reason="object_approach_selection_failed",
            object_approach_candidates=list(goal.get("object_approach_candidates") or []),
        )
        result["selected_object_approach"] = None
        return result

    start_floor = adapter._infer_floor_id(scene, start) or adapter._infer_floor_from_height(
        scene, start_pose
    )
    start_room_id = adapter._infer_room_id(scene, start)
    goal_room_id = adapter._infer_room_id(scene, goal)
    goal_floor = adapter._infer_floor_id(scene, goal) or adapter._infer_floor_from_height(
        scene, raw_goal_position
    )
    start_node = adapter._nearest_nav_node(
        scene.nav_graph,
        start_pose,
        floor_id=start_floor,
        room_id=start_room_id,
    )
    object_approach_candidates: list[dict[str, Any]] = []
    selected_object_approach: dict[str, Any] | None = None
    goal_position = raw_goal_position
    goal_node = None
    if start_node is not None:
        (
            goal_position,
            goal_node,
            object_approach_candidates,
            selected_object_approach,
        ) = adapter._resolve_goal_position_and_node(
            scene=scene,
            goal=goal,
            start_node=start_node,
            start=start,
            context=context,
            fallback_goal_position=raw_goal_position,
        )
    object_approach_diagnostics = (
        hovsg_object_approach.object_approach_diagnostics(
            adapter,
            scene_id=scene_id,
            object_id=str(goal.get("object_id") or ""),
        )
        if goal.get("goal_type") == "object"
        else None
    )
    if goal.get("goal_type") == "object" and selected_object_approach is None:
        selection_failure_reason = str(
            (object_approach_diagnostics or {}).get("selection_failure_reason") or ""
        )
        blocked_by_closed_door = selection_failure_reason == "blocked_by_closed_door"
        result = _planner_result(
            scene_id=scene_id,
            vertical_axis=scene.vertical_axis,
            nav_graph_type=scene.nav_graph_type,
            nav_graph_asset_path=scene.nav_graph_asset_path,
            start=start,
            goal=goal,
            found=False,
            reason=(
                "blocked_by_closed_door"
                if blocked_by_closed_door
                else "OBJECT_APPROACH_UNAVAILABLE"
            ),
            object_approach_candidates=object_approach_candidates,
            object_approach_diagnostics=object_approach_diagnostics,
        )
        result["selected_object_approach"] = None
        result["object_approach_selection_failed"] = True
        if blocked_by_closed_door:
            door_candidates = hovsg_door_gating.blocking_door_candidates(
                adapter,
                scene,
                source_room_id=start_room_id,
                target_room_id=goal_room_id,
            )
            result["closed_doors"] = hovsg_door_gating.closed_door_names(
                adapter,
                scene_id,
            )
            if door_candidates:
                result["door_candidates"] = door_candidates
                result["blocked_transition"] = _blocked_transition_from_door(door_candidates[0])
        return result
    goal_node = (
        adapter._nearest_nav_node(
            scene.nav_graph,
            goal_position,
            floor_id=goal_floor,
            room_id=adapter._infer_room_id(scene, goal),
        )
        if goal_node is None
        else goal_node
    )
    if start_node is None or goal_node is None:
        return _planner_result(
            scene_id=scene_id,
            vertical_axis=scene.vertical_axis,
            nav_graph_type=scene.nav_graph_type,
            nav_graph_asset_path=scene.nav_graph_asset_path,
            start=start,
            goal=goal,
            found=False,
            reason="nav_nodes_missing",
            object_approach_candidates=object_approach_candidates,
            selected_object_approach=selected_object_approach,
        )

    active_graph, door_gated = hovsg_door_gating.filtered_nav_graph(adapter, scene)
    active_graph_type = scene.nav_graph_type
    active_graph_path = scene.nav_graph_asset_path
    try:
        path_nodes = nx.shortest_path(active_graph, start_node, goal_node, weight="dist")
    except nx.NetworkXNoPath:
        if door_gated and nx.has_path(scene.nav_graph, start_node, goal_node):
            door_candidates = hovsg_door_gating.blocking_door_candidates(
                adapter,
                scene,
                source_room_id=start_room_id,
                target_room_id=goal_room_id,
            )
            return _planner_result(
                scene_id=scene_id,
                vertical_axis=scene.vertical_axis,
                nav_graph_type=scene.nav_graph_type,
                nav_graph_asset_path=scene.nav_graph_asset_path,
                start=start,
                goal=goal,
                found=False,
                reason="blocked_by_closed_door",
                object_approach_candidates=object_approach_candidates,
                selected_object_approach=selected_object_approach,
                closed_doors=hovsg_door_gating.closed_door_names(adapter, scene_id),
                blocked_transition=(
                    _blocked_transition_from_door(door_candidates[0]) if door_candidates else None
                ),
                door_candidates=door_candidates,
            )
        fallback_graph_asset = hovsg_scene_loading.load_nav_graph(
            scene.graph_path / "nav_graph",
            metadata=scene.metadata,
            config={"nav_graph_type": "global_room_graph"},
        )
        fallback_graph = fallback_graph_asset.graph
        if scene.nav_graph_type == "voronoi_graph" and fallback_graph.number_of_nodes() > 0:
            fallback_start_node = adapter._nearest_nav_node(
                fallback_graph,
                start_pose,
                floor_id=start_floor,
                room_id=adapter._infer_room_id(scene, start),
            )
            fallback_goal_node = adapter._nearest_nav_node(
                fallback_graph,
                goal_position,
                floor_id=goal_floor,
                room_id=adapter._infer_room_id(scene, goal),
            )
            if fallback_start_node is not None and fallback_goal_node is not None:
                fallback_active_graph, _ = hovsg_door_gating.filtered_nav_graph(
                    adapter,
                    scene,
                    graph=fallback_graph,
                )
                path_nodes = nx.shortest_path(
                    fallback_active_graph,
                    fallback_start_node,
                    fallback_goal_node,
                    weight="dist",
                )
                active_graph = fallback_graph
                active_graph_type = fallback_graph_asset.graph_type
                active_graph_path = fallback_graph_asset.graph_path
            else:
                raise
        else:
            raise
    node_waypoints = [adapter._node_to_waypoint(active_graph, node_id) for node_id in path_nodes]
    waypoints = adapter._build_geometric_waypoints(
        scene=scene,
        path_nodes=path_nodes,
        goal=goal,
        goal_position=goal_position,
        start=start,
        context=context,
    )
    dense_waypoints = adapter._build_dense_waypoints(
        scene=scene,
        node_waypoints=node_waypoints,
        goal=goal,
        goal_position=goal_position,
    )
    return {
        "planner": "hovsg_navigator",
        "scene_id": scene_id,
        "vertical_axis": scene.vertical_axis,
        "nav_graph_type": active_graph_type,
        "nav_graph_asset_path": str(active_graph_path) if active_graph_path else None,
        "requested_nav_graph_type": scene.nav_graph_type,
        "start": start,
        "goal": goal,
        "path_nodes": [adapter._serialize_node_id(node_id) for node_id in path_nodes],
        "graph_waypoints": node_waypoints,
        "waypoints": waypoints,
        "dense_waypoints": dense_waypoints,
        "path_cost": path_cost(active_graph, path_nodes),
        "room_sequence": room_sequence(waypoints),
        "graph_room_sequence": graph_room_sequence(adapter, scene=scene, waypoints=node_waypoints),
        "found": True,
        "object_approach_candidates": object_approach_candidates,
        "selected_object_approach": selected_object_approach,
        "object_approach_diagnostics": object_approach_diagnostics,
    }


def path_cost(graph: nx.Graph, path_nodes: list[Any]) -> float:
    if len(path_nodes) < 2:
        return 0.0
    cost = 0.0
    for source, target in zip(path_nodes, path_nodes[1:]):
        edge = graph.get_edge_data(source, target, default={})
        try:
            cost += float(edge.get("dist", 1.0))
        except (TypeError, ValueError):
            cost += 1.0
    return cost


def room_sequence(waypoints: list[dict[str, Any]]) -> list[str]:
    sequence: list[str] = []
    for waypoint in waypoints:
        room_name = waypoint.get("room_name")
        if not isinstance(room_name, str) or not room_name.strip():
            continue
        if not sequence or sequence[-1] != room_name:
            sequence.append(room_name)
    return sequence


def graph_room_sequence(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    waypoints: list[dict[str, Any]],
) -> list[str]:
    sequence: list[str] = []
    for waypoint in waypoints:
        room_name = waypoint.get("room_name")
        if not isinstance(room_name, str) or not room_name.strip():
            room_id = waypoint.get("room_id")
            if isinstance(room_id, (str, int)):
                room = scene.rooms.get(str(room_id))
                room_name = (
                    room.name if room is not None and isinstance(room.name, str) else str(room_id)
                )
        if not isinstance(room_name, str) or not room_name.strip():
            room = adapter._containing_room(scene, waypoint)
            if room is not None and isinstance(room.name, str) and room.name.strip():
                room_name = room.name
        if not isinstance(room_name, str) or not room_name.strip():
            continue
        if not sequence or sequence[-1] != room_name:
            sequence.append(room_name)
    return sequence


def _planner_result(
    *,
    scene_id: str | None,
    vertical_axis: str | None,
    nav_graph_type: str | None,
    nav_graph_asset_path: Any,
    start: dict[str, Any],
    goal: dict[str, Any],
    found: bool,
    reason: str | None = None,
    object_approach_candidates: list[dict[str, Any]] | None = None,
    selected_object_approach: dict[str, Any] | None = None,
    object_approach_diagnostics: dict[str, Any] | None = None,
    closed_doors: list[str] | None = None,
    blocked_transition: dict[str, Any] | None = None,
    door_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result = {
        "planner": "hovsg_navigator",
        "scene_id": scene_id,
        "vertical_axis": vertical_axis,
        "nav_graph_type": nav_graph_type,
        "nav_graph_asset_path": str(nav_graph_asset_path) if nav_graph_asset_path else None,
        "start": start,
        "goal": goal,
        "waypoints": [],
        "dense_waypoints": [],
        "graph_waypoints": [],
        "graph_room_sequence": [],
        "path_nodes": [],
        "found": found,
    }
    if reason is not None:
        result["reason"] = reason
    if object_approach_candidates is not None:
        result["object_approach_candidates"] = object_approach_candidates
    if selected_object_approach is not None:
        result["selected_object_approach"] = selected_object_approach
    if object_approach_diagnostics is not None:
        result["object_approach_diagnostics"] = object_approach_diagnostics
    if closed_doors:
        result["closed_doors"] = list(closed_doors)
    if blocked_transition:
        result["blocked_transition"] = dict(blocked_transition)
    if door_candidates:
        result["door_candidates"] = [dict(candidate) for candidate in door_candidates]
    return result


def _blocked_transition_from_door(door: dict[str, Any]) -> dict[str, Any]:
    return {
        key: door[key]
        for key in (
            "source_room_id",
            "source_room_name",
            "target_room_id",
            "target_room_name",
        )
        if door.get(key) not in (None, "")
    }
