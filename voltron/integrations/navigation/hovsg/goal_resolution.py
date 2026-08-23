"""Goal and anchor resolution helpers for the HOV-SG navigator."""

from __future__ import annotations

from typing import Any

import networkx as nx

from . import object_approach as hovsg_object_approach
from . import effective_scene as hovsg_effective_scene
from .models import HOVSGFloorAsset, HOVSGObjectAsset, HOVSGRoomAsset, HOVSGSceneAsset


def match_room(
    adapter: Any, scene: HOVSGSceneAsset | None, text: str | None
) -> HOVSGRoomAsset | None:
    if scene is None or not isinstance(text, str) or not text.strip():
        return None
    stripped = text.strip()
    if stripped in scene.rooms:
        return scene.rooms[stripped]
    valid = [
        (
            max(
                adapter._score_text_match(str(room.name or room.room_id), text),
                adapter._score_text_match(str(room.room_id), text),
            ),
            room,
        )
        for room in scene.rooms.values()
    ]
    ranked = [(score, room) for score, room in valid if score > 0.0]
    if not ranked:
        return None
    ranked.sort(
        key=lambda item: (
            -item[0],
            adapter._room_polygon_area(scene, item[1]) is None,
            adapter._room_polygon_area(scene, item[1]) or float("inf"),
        )
    )
    return ranked[0][1]


def match_object(
    adapter: Any,
    scene: HOVSGSceneAsset | None,
    text: str | None,
    room_id: str | None = None,
) -> HOVSGObjectAsset | None:
    if scene is None or not isinstance(text, str) or not text.strip():
        return None
    objects = list(scene.objects.values())
    if isinstance(room_id, str) and room_id.strip():
        objects = [obj for obj in objects if obj.room_id == room_id]
    valid = [
        (adapter._score_text_match(str(obj.name or obj.object_id), text), obj)
        for obj in objects
    ]
    ranked = [(score, obj) for score, obj in valid if score > 0.0]
    if not ranked:
        return None
    ranked.sort(key=lambda item: -item[0])
    return ranked[0][1]


def match_floor(
    adapter: Any, scene: HOVSGSceneAsset | None, text: str | None
) -> HOVSGFloorAsset | None:
    if scene is None or not isinstance(text, str) or not text.strip():
        return None
    stripped = text.strip().lower()
    if stripped.isdigit() and stripped in scene.floors:
        return scene.floors[stripped]
    valid = [
        (adapter._score_text_match(str(floor.name or floor.floor_id), text), floor)
        for floor in scene.floors.values()
    ]
    ranked = [(score, floor) for score, floor in valid if score > 0.0]
    if not ranked:
        return None
    ranked.sort(key=lambda item: -item[0])
    return ranked[0][1]


def goal_position(
    adapter: Any, scene: HOVSGSceneAsset, goal: dict[str, Any]
) -> dict[str, float] | None:
    if isinstance(goal.get("position"), dict):
        return goal["position"]
    if all(key in goal for key in ("x", "y", "z")):
        try:
            return {
                **goal,
                "x": float(goal["x"]),
                "y": float(goal["y"]),
                "z": float(goal["z"]),
            }
        except (TypeError, ValueError):
            return None

    if goal.get("goal_type") == "object":
        effective = hovsg_effective_scene.effective_scene_view(
            adapter,
            scene,
        ).resolve_object(goal.get("object_id"))
        if effective is not None:
            return dict(effective.position) if effective.position is not None else None
    if goal.get("goal_type") == "room" and goal.get("room_id") in scene.rooms:
        return scene.rooms[str(goal["room_id"])].centroid
    if goal.get("goal_type") == "floor" and goal.get("floor_id") in scene.floors:
        return scene.floors[str(goal["floor_id"])].centroid
    return None


def goal_waypoint(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    goal: dict[str, Any],
    goal_position: dict[str, float],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    del adapter, scene
    waypoint = {
        "x": float(goal_position["x"]),
        "y": float(goal_position["y"]),
        "z": float(goal_position["z"]),
        "floor_id": goal.get("floor_id") or fallback.get("floor_id"),
        "room_id": goal.get("room_id") or fallback.get("room_id"),
        "room_name": goal.get("room_name") or fallback.get("room_name"),
        "waypoint_type": "goal",
    }
    for key in (
        "candidate_id",
        "nav_node",
        "desired_heading",
        "object_id",
        "object_name",
        "object_position",
        "approach_distance_m",
        "approach_boundary_distance_m",
        "path_cost",
    ):
        if key in goal_position:
            waypoint[key] = goal_position[key]
    if "nav_node" in fallback:
        waypoint["nav_node"] = fallback["nav_node"]
    return waypoint


def resolve_goal_position_and_node(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    goal: dict[str, Any],
    start_node: Any,
    start: dict[str, Any],
    context: dict[str, Any],
    fallback_goal_position: dict[str, float],
) -> tuple[dict[str, Any], Any | None, list[dict[str, Any]], dict[str, Any] | None]:
    if goal.get("goal_type") != "object":
        return fallback_goal_position, None, [], None

    selected = goal.get("selected_object_approach")
    if isinstance(selected, dict) and selected:
        selected_candidate = dict(selected)
        selected_node = adapter._normalize_node_id(selected_candidate.get("nav_node"))
        candidates = list(goal.get("object_approach_candidates") or [])
        if not candidates:
            candidates = adapter._score_object_approach_candidates(
                scene=scene,
                start_node=start_node,
                candidates=adapter._build_object_approach_candidates(
                    scene=scene,
                    goal=goal,
                    start=start,
                    context=context,
                ),
            )
        diagnostics = hovsg_object_approach.object_approach_diagnostics(
            adapter,
            scene_id=str(scene.scene_id or ""),
            object_id=str(goal.get("object_id") or ""),
        )
        diagnostics["selected_candidate_id"] = selected_candidate.get("candidate_id")
        diagnostics["selection_failure_reason"] = None
        hovsg_object_approach._store_candidate_diagnostics(
            adapter,
            scene_id=str(scene.scene_id or ""),
            object_id=str(goal.get("object_id") or ""),
            diagnostics=diagnostics,
        )
        return selected_candidate, selected_node, candidates, selected_candidate

    candidates = adapter._score_object_approach_candidates(
        scene=scene,
        start_node=start_node,
        candidates=adapter._build_object_approach_candidates(
            scene=scene,
            goal=goal,
            start=start,
            context=context,
        ),
    )
    if not candidates:
        return fallback_goal_position, None, [], None

    selected = adapter._select_object_approach_candidate(
        scene=scene,
        start_node=start_node,
        candidates=candidates,
    )
    if selected is None:
        return fallback_goal_position, None, candidates, None
    diagnostics = hovsg_object_approach.object_approach_diagnostics(
        adapter,
        scene_id=str(scene.scene_id or ""),
        object_id=str(goal.get("object_id") or ""),
    )
    diagnostics["selected_candidate_id"] = selected.get("candidate_id")
    diagnostics["selection_failure_reason"] = None
    hovsg_object_approach._store_candidate_diagnostics(
        adapter,
        scene_id=str(scene.scene_id or ""),
        object_id=str(goal.get("object_id") or ""),
        diagnostics=diagnostics,
    )
    selected_node = adapter._normalize_node_id(selected.get("nav_node"))
    return selected, selected_node, candidates, selected


def infer_floor_id(
    adapter: Any, scene: HOVSGSceneAsset, anchor: dict[str, Any]
) -> str | None:
    if isinstance(anchor.get("floor_id"), (str, int)):
        return str(anchor["floor_id"])
    room_id = anchor.get("room_id")
    if room_id is not None and str(room_id) in scene.rooms:
        return scene.rooms[str(room_id)].floor_id
    object_id = anchor.get("object_id")
    if object_id is not None:
        effective = hovsg_effective_scene.effective_scene_view(
            adapter,
            scene,
        ).resolve_object(object_id)
        room = scene.rooms.get(effective.room_id) if effective is not None else None
        return room.floor_id if room is not None else None
    return None


def refine_start_anchor(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    start: dict[str, Any],
) -> dict[str, Any]:
    pose = start.get("pose")
    if not isinstance(pose, dict):
        return start
    localized = adapter._localize_pose(
        scene,
        pose,
        previous_room_id=adapter._last_localized_room_ids.get(scene.scene_id),
        persist=False,
    )
    if not localized:
        return start
    refined = dict(start)
    guard = start.get("localization_guard")
    rejected_room = guard.get("rejected_room") if isinstance(guard, dict) else None
    localized_room = localized.get("current_room") or localized.get("current_region")
    if (
        isinstance(rejected_room, str)
        and isinstance(localized_room, str)
        and adapter._normalize_text(localized_room)
        == adapter._normalize_text(rejected_room)
    ):
        refined.setdefault("scene_id", scene.scene_id)
        return refined
    refined.update(localized)
    refined.setdefault("scene_id", scene.scene_id)
    return refined


def infer_room_id(
    adapter: Any, scene: HOVSGSceneAsset, anchor: dict[str, Any]
) -> str | None:
    if isinstance(anchor.get("room_id"), (str, int)):
        room_id = str(anchor["room_id"])
        if room_id in scene.rooms:
            return room_id
    object_id = anchor.get("object_id")
    if object_id is not None:
        effective = hovsg_effective_scene.effective_scene_view(
            adapter,
            scene,
        ).resolve_object(object_id)
        if effective is not None:
            return effective.room_id
    normalized_label_candidates = (
        anchor.get("current_room"),
        anchor.get("current_region"),
        anchor.get("room_name"),
    )
    for candidate in normalized_label_candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        normalized = adapter._normalize_text(candidate)
        for room in scene.rooms.values():
            if (
                isinstance(room.name, str)
                and adapter._normalize_text(room.name) == normalized
            ):
                return room.room_id
    return None


def nearest_nav_node(
    adapter: Any,
    graph: nx.Graph,
    position: dict[str, Any],
    floor_id: str | None = None,
    room_id: str | None = None,
) -> Any | None:
    px = adapter._to_float(position.get("x"))
    py = adapter._to_float(position.get("y"))
    pz = adapter._to_float(position.get("z"))
    if px is None or py is None or pz is None or graph.number_of_nodes() == 0:
        return None

    search_scopes = (
        (floor_id, room_id),
        (None, room_id),
        (floor_id, None),
        (None, None),
    )
    for candidate_floor, candidate_room in search_scopes:
        best_node = None
        best_dist = None
        for node_id, attrs in graph.nodes(data=True):
            node_floor = attrs.get("floor_id")
            if (
                candidate_floor is not None
                and node_floor is not None
                and str(node_floor) != str(candidate_floor)
            ):
                continue
            node_room = attrs.get("room_id")
            if (
                candidate_room is not None
                and node_room is not None
                and str(node_room) != str(candidate_room)
            ):
                continue
            if candidate_room is not None and node_room is None:
                continue
            pos = attrs.get("pos")
            if not isinstance(pos, (list, tuple)) or len(pos) < 3:
                continue
            dist = (
                (float(pos[0]) - px) ** 2
                + (float(pos[1]) - py) ** 2
                + (float(pos[2]) - pz) ** 2
            )
            if best_dist is None or dist < best_dist:
                best_node = node_id
                best_dist = dist
        if best_node is not None:
            return best_node
    return None


def node_to_waypoint(adapter: Any, graph: nx.Graph, node_id: Any) -> dict[str, Any]:
    attrs = graph.nodes[node_id]
    pos = attrs.get("pos", [0.0, 0.0, 0.0])
    return {
        "x": float(pos[0]),
        "y": float(pos[1]),
        "z": float(pos[2]),
        "floor_id": attrs.get("floor_id"),
        "room_id": attrs.get("room_id"),
        "room_name": attrs.get("room_name"),
        "nav_node": adapter._serialize_node_id(node_id),
        "waypoint_type": "nav_node",
    }
