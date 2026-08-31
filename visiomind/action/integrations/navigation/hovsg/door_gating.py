from __future__ import annotations

from typing import Any

import networkx as nx

from visiomind.action.shared.models.scene_state import (
    RuntimeDoorState,
    door_is_navigation_passable,
)
from visiomind.action.shared.geometry_frames import (
    horizontal_axis_indices,
    vertical_axis_index,
)

from . import runtime_state as hovsg_runtime_state
from . import scene_loading as hovsg_scene_loading
from .models import HOVSGSceneAsset

DEFAULT_DOOR_HALF_EXTENT_M = 0.5


def door_room_links(adapter: Any, scene: HOVSGSceneAsset) -> list[dict[str, Any]]:
    state = hovsg_runtime_state.current_scene_state(adapter, scene.scene_id)
    if state is None:
        return []
    room_tokens = {
        room_id: _room_tokens(room_id, room.name) for room_id, room in scene.rooms.items()
    }
    links: list[dict[str, Any]] = []
    for door in state.doors.values():
        matched_room_ids = sorted(
            {
                room_id
                for room_id, tokens in room_tokens.items()
                if any(_normalize(in_room) in tokens for in_room in door.in_rooms)
            }
        )
        if len(matched_room_ids) != 2:
            continue
        links.append(
            {
                "door": door,
                "rooms": frozenset(matched_room_ids),
            }
        )
    return links


def blocked_room_pairs(adapter: Any, scene: HOVSGSceneAsset) -> set[frozenset[str]]:
    door_states_by_pair: dict[frozenset[str], list[bool | None]] = {}
    for link in door_room_links(adapter, scene):
        door_states_by_pair.setdefault(link["rooms"], []).append(
            door_is_navigation_passable(link["door"])
        )
    return {
        pair
        for pair, states in door_states_by_pair.items()
        if states and all(state is False for state in states)
    }


def opened_room_pairs(adapter: Any, scene: HOVSGSceneAsset) -> set[frozenset[str]]:
    return {
        link["rooms"]
        for link in door_room_links(adapter, scene)
        if door_is_navigation_passable(link["door"]) is True
    }


def effective_room_adjacency(adapter: Any, scene: HOVSGSceneAsset) -> dict[str, set[str]] | None:
    static = scene.room_adjacency
    blocked = blocked_room_pairs(adapter, scene)
    opened = opened_room_pairs(adapter, scene)
    if not blocked and not opened:
        return static
    adjacency: dict[str, set[str]] = (
        {room_id: set(neighbors) for room_id, neighbors in static.items()}
        if static
        else {room_id: set() for room_id in scene.rooms}
    )
    for pair in opened:
        left, right = sorted(pair)
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    for pair in blocked:
        left, right = sorted(pair)
        adjacency.get(left, set()).discard(right)
        adjacency.get(right, set()).discard(left)
    return adjacency


def room_pair_blocked(
    adapter: Any, scene: HOVSGSceneAsset, source_room_id: str, target_room_id: str
) -> bool:
    return frozenset((str(source_room_id), str(target_room_id))) in blocked_room_pairs(
        adapter, scene
    )


def filtered_nav_graph(
    adapter: Any, scene: HOVSGSceneAsset, graph: nx.Graph | None = None
) -> tuple[nx.Graph, bool]:
    active_graph = graph if graph is not None else scene.nav_graph
    blocked = blocked_room_pairs(adapter, scene)
    if not blocked:
        return active_graph, False

    room_id_by_name = {
        str(room.name): room_id
        for room_id, room in scene.rooms.items()
        if isinstance(room.name, str) and room.name
    }

    def _edge_allowed(left_node: Any, right_node: Any) -> bool:
        left_room = hovsg_scene_loading.resolve_room_id_from_nav_node_attrs(
            active_graph.nodes[left_node], room_id_by_name
        )
        right_room = hovsg_scene_loading.resolve_room_id_from_nav_node_attrs(
            active_graph.nodes[right_node], room_id_by_name
        )
        if left_room is None or right_room is None or left_room == right_room:
            return True
        return frozenset((left_room, right_room)) not in blocked

    return nx.subgraph_view(active_graph, filter_edge=_edge_allowed), True


def closed_door_obstacles(adapter: Any, scene_id: str | None) -> list[dict[str, Any]]:
    obstacles: list[dict[str, Any]] = []
    for door in hovsg_runtime_state.door_states(adapter, scene_id).values():
        if door_is_navigation_passable(door) is not False:
            continue
        obstacles.extend(
            _door_obstacles(
                door,
                navigation_floor_height=_door_navigation_floor_height(adapter, scene_id, door),
                vertical_axis=_door_vertical_axis(adapter, scene_id),
            )
        )
    return obstacles


def runtime_door_obstacles(adapter: Any, scene_id: str | None) -> list[dict[str, Any]]:
    obstacles: list[dict[str, Any]] = []
    for door in hovsg_runtime_state.door_states(adapter, scene_id).values():
        if door.navigation_passable is True:
            continue
        obstacles.extend(
            _door_obstacles(
                door,
                navigation_floor_height=_door_navigation_floor_height(adapter, scene_id, door),
                vertical_axis=_door_vertical_axis(adapter, scene_id),
            )
        )
    return obstacles


def open_door_clear_regions(adapter: Any, scene_id: str | None) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for door in hovsg_runtime_state.door_states(adapter, scene_id).values():
        if door_is_navigation_passable(door) is not True:
            continue
        if any(
            isinstance(part, dict) and part.get("world_polygons") for part in door.collision_parts
        ):
            continue
        regions.extend(
            _door_obstacles(
                door,
                navigation_floor_height=_door_navigation_floor_height(adapter, scene_id, door),
                vertical_axis=_door_vertical_axis(adapter, scene_id),
            )
        )
    return regions


def open_portal_clear_region(
    navigation_goal: Any,
    *,
    normal_padding_m: float = 0.0,
) -> dict[str, Any] | None:
    if not isinstance(navigation_goal, dict):
        return None
    candidates = [navigation_goal]
    transition_anchor = navigation_goal.get("transition_anchor")
    if isinstance(transition_anchor, dict):
        candidates.append(transition_anchor)
    for portal in candidates:
        if portal.get("portal_door_open") is not True:
            continue
        normal_axis = portal.get("portal_normal_axis")
        span_axis = portal.get("portal_span_axis")
        if {normal_axis, span_axis} != {"x", "y"}:
            continue
        try:
            span_min = float(portal["portal_span_min"])
            span_max = float(portal["portal_span_max"])
            source = portal["portal_source_point"]
            target = portal["portal_target_point"]
            padding = max(0.0, float(normal_padding_m))
            normal_min = min(float(source[normal_axis]), float(target[normal_axis])) - padding
            normal_max = max(float(source[normal_axis]), float(target[normal_axis])) + padding
        except (KeyError, TypeError, ValueError):
            continue
        bounds = {
            normal_axis: (normal_min, normal_max),
            span_axis: (min(span_min, span_max), max(span_min, span_max)),
        }
        return {
            "name": portal.get("portal_object_name") or "open_portal",
            "min": {"x": bounds["x"][0], "y": bounds["y"][0]},
            "max": {"x": bounds["x"][1], "y": bounds["y"][1]},
        }
    return None


def prefer_canonical_portal_clear_region(
    regions: list[dict[str, Any]],
    portal_region: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if portal_region is None:
        return list(regions)
    portal_name = _normalize(portal_region.get("name"))
    if portal_name:
        regions = [region for region in regions if _normalize(region.get("name")) != portal_name]
    return [*regions, portal_region]


def closed_door_names(adapter: Any, scene_id: str | None) -> list[str]:
    return sorted(
        name
        for name, door in hovsg_runtime_state.door_states(adapter, scene_id).items()
        if door_is_navigation_passable(door) is False
    )


def blocking_door_candidates(
    adapter: Any,
    scene: HOVSGSceneAsset,
    *,
    source_room_id: str | None,
    target_room_id: str | None,
) -> list[dict[str, Any]]:
    source_id = str(source_room_id or "").strip()
    target_id = str(target_room_id or "").strip()
    if not source_id or not target_id or source_id == target_id:
        return []
    room_graph = nx.Graph()
    room_graph.add_nodes_from(scene.rooms)
    adjacency = scene.room_adjacency or {
        room_id: set(room.connected_room_ids) for room_id, room in scene.rooms.items()
    }
    for room_id, neighbors in adjacency.items():
        for neighbor in neighbors:
            room_graph.add_edge(str(room_id), str(neighbor))
    blocked_pairs = blocked_room_pairs(adapter, scene)
    try:
        room_route = nx.shortest_path(room_graph, source_id, target_id)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        room_route = []
    oriented_pairs = (
        [
            (left, right)
            for left, right in zip(room_route, room_route[1:])
            if frozenset((left, right)) in blocked_pairs
        ]
        if room_route
        else []
    )
    direct_pair = frozenset((source_id, target_id))
    if not oriented_pairs and direct_pair in blocked_pairs:
        oriented_pairs = [(source_id, target_id)]
    if not oriented_pairs:
        return []

    candidates: list[dict[str, Any]] = []
    for source_id, target_id in oriented_pairs:
        pair = frozenset((source_id, target_id))
        source_room = scene.rooms.get(source_id)
        target_room = scene.rooms.get(target_id)
        source_name = str(source_room.name or source_id) if source_room else source_id
        target_name = str(target_room.name or target_id) if target_room else target_id
        for link in door_room_links(adapter, scene):
            door = link["door"]
            if link["rooms"] != pair or door_is_navigation_passable(door) is not False:
                continue
            candidate: dict[str, Any] = {
                "id": door.name,
                "name": door.name,
                "is_open": False,
                "source_room_id": source_id,
                "source_room_name": source_name,
                "target_room_id": target_id,
                "target_room_name": target_name,
                "room": source_name,
                "in_rooms": list(door.in_rooms),
            }
            if source_room is not None:
                candidate["floor_id"] = source_room.floor_id
            if door.position is not None:
                candidate["position"] = dict(door.position)
            candidates.append(candidate)
    return candidates


def _door_obstacles(
    door: RuntimeDoorState,
    *,
    navigation_floor_height: float | None = None,
    vertical_axis: str = "z",
) -> list[dict[str, Any]]:
    obstacles = []
    compatible_part_count = 0
    for part in door.collision_parts:
        part_axis = str(part.get("vertical_axis") or "").lower()
        if part_axis and part_axis != vertical_axis:
            continue
        compatible_part_count += 1
        if not _part_overlaps_navigation_height(
            part,
            navigation_floor_height,
            vertical_axis=vertical_axis,
        ):
            continue
        obstacle = _polygon_obstacle(door.name, part, link=part.get("link"))
        if obstacle is not None:
            obstacles.append(obstacle)
            continue
        obstacle = _aabb_obstacle(
            door.name,
            part,
            link=part.get("link"),
            vertical_axis=vertical_axis,
        )
        if obstacle is not None:
            obstacles.append(obstacle)
    if obstacles:
        return obstacles
    if door.collision_parts and compatible_part_count > 0:
        return []
    fallback = (
        _aabb_obstacle(door.name, door.aabb, vertical_axis=vertical_axis)
        if _part_overlaps_navigation_height(
            door.aabb,
            navigation_floor_height,
            vertical_axis=vertical_axis,
        )
        else None
    )
    if fallback is not None:
        return [fallback]
    if isinstance(door.position, dict):
        try:
            horizontal_axes = _horizontal_axis_names(vertical_axis)
            return [
                {
                    "name": door.name,
                    "position": {
                        "x": float(door.position[horizontal_axes[0]]),
                        "y": float(door.position[horizontal_axes[1]]),
                    },
                    "half_extent_m": DEFAULT_DOOR_HALF_EXTENT_M,
                }
            ]
        except (KeyError, TypeError, ValueError):
            pass
    return []


def _door_navigation_floor_height(
    adapter: Any,
    scene_id: str | None,
    door: RuntimeDoorState,
) -> float | None:
    scenes = getattr(adapter, "_scenes", None)
    scene = scenes.get(scene_id) if isinstance(scenes, dict) and scene_id else None
    if scene is None:
        return None
    vertical_axis = str(getattr(scene, "vertical_axis", "z") or "z")
    room_tokens = {_normalize(value) for value in door.in_rooms}
    rooms = getattr(scene, "rooms", {})
    floors = getattr(scene, "floors", {})
    for room in rooms.values():
        if room_tokens and not ({_normalize(room.room_id), _normalize(room.name)} & room_tokens):
            continue
        floor = floors.get(str(room.floor_id))
        try:
            return float(floor.floor_zero_level) if floor is not None else None
        except (TypeError, ValueError):
            continue
    if isinstance(door.position, dict):
        try:
            door_height = float(door.position[vertical_axis])
            levels = [
                float(floor.floor_zero_level)
                for floor in floors.values()
                if floor.floor_zero_level is not None
            ]
            return min(levels, key=lambda level: abs(level - door_height)) if levels else None
        except (KeyError, TypeError, ValueError):
            pass
    return None


def _door_vertical_axis(adapter: Any, scene_id: str | None) -> str:
    scenes = getattr(adapter, "_scenes", None)
    scene = scenes.get(scene_id) if isinstance(scenes, dict) and scene_id else None
    return str(getattr(scene, "vertical_axis", "z") or "z")


def _part_overlaps_navigation_height(
    part: Any,
    floor_height: float | None,
    *,
    vertical_axis: str = "z",
    robot_clearance_height_m: float = 1.6,
    floor_slop_m: float = 0.15,
) -> bool:
    if floor_height is None or not isinstance(part, dict):
        return True
    height_min = part.get("height_min")
    height_max = part.get("height_max")
    if not isinstance(height_min, (int, float)) or not isinstance(height_max, (int, float)):
        corner_min = part.get("min")
        corner_max = part.get("max")
        if not isinstance(corner_min, (list, tuple)) or not isinstance(corner_max, (list, tuple)):
            return True
        try:
            height_index = vertical_axis_index(vertical_axis)
            height_min = float(corner_min[height_index])
            height_max = float(corner_max[height_index])
        except (IndexError, TypeError, ValueError):
            return True
    return (
        float(height_max) >= float(floor_height) - floor_slop_m
        and float(height_min) <= float(floor_height) + robot_clearance_height_m
    )


def _polygon_obstacle(
    name: str,
    part: Any,
    *,
    link: Any = None,
) -> dict[str, Any] | None:
    if not isinstance(part, dict):
        return None
    raw_polygons = part.get("world_polygons")
    if not isinstance(raw_polygons, (list, tuple)):
        return None
    polygons: list[list[list[float]]] = []
    for raw_polygon in raw_polygons:
        if not isinstance(raw_polygon, (list, tuple)):
            continue
        polygon: list[list[float]] = []
        for point in raw_polygon:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                polygon.append([float(point[0]), float(point[1])])
            except (TypeError, ValueError):
                continue
        if len(polygon) >= 3:
            polygons.append(polygon)
    if not polygons:
        return None
    obstacle: dict[str, Any] = {
        "name": name,
        "overlay_kind": "articulated_door",
        "geometry_source": str(part.get("geometry_source") or "collision_mesh"),
        "polygons": polygons,
    }
    for key in (
        "geometry_id",
        "geometry_hash",
        "geometry_revision",
        "pose_revision",
        "joint_type",
        "joint_position",
    ):
        if key in part:
            obstacle[key] = part[key]
    if link is not None:
        obstacle["link"] = str(link)
    return obstacle


def _aabb_obstacle(
    name: str,
    aabb: Any,
    *,
    link: Any = None,
    vertical_axis: str = "z",
) -> dict[str, Any] | None:
    if not isinstance(aabb, dict):
        return None
    corner_min = aabb.get("min")
    corner_max = aabb.get("max")
    if not isinstance(corner_min, (list, tuple)) or not isinstance(corner_max, (list, tuple)):
        return None
    try:
        horizontal_indices = horizontal_axis_indices(vertical_axis)
        obstacle = {
            "name": name,
            "min": {
                "x": float(corner_min[horizontal_indices[0]]),
                "y": float(corner_min[horizontal_indices[1]]),
            },
            "max": {
                "x": float(corner_max[horizontal_indices[0]]),
                "y": float(corner_max[horizontal_indices[1]]),
            },
        }
    except (IndexError, TypeError, ValueError):
        return None
    if link is not None:
        obstacle["link"] = str(link)
    return obstacle


def _horizontal_axis_names(vertical_axis: str) -> tuple[str, str]:
    return {
        "x": ("y", "z"),
        "y": ("x", "z"),
        "z": ("x", "y"),
    }.get(vertical_axis, ("x", "y"))


def _room_tokens(room_id: str, room_name: Any) -> set[str]:
    return {_normalize(room_id), _normalize(room_name)} - {""}


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower()


__all__ = [
    "DEFAULT_DOOR_HALF_EXTENT_M",
    "blocked_room_pairs",
    "closed_door_names",
    "closed_door_obstacles",
    "door_room_links",
    "effective_room_adjacency",
    "filtered_nav_graph",
    "opened_room_pairs",
    "open_door_clear_regions",
    "open_portal_clear_region",
    "prefer_canonical_portal_clear_region",
    "runtime_door_obstacles",
    "room_pair_blocked",
]
