"""Export BEHAVIOR scenes into lightweight scene maps and HOV-SG-style assets.

This module builds a minimal navigation graph directly from BEHAVIOR scene assets:

- ``scenes/<scene_model>/layout/floor_{ins,sem,trav}_0.png``
- ``scenes/<scene_model>/json/<scene_model>_best.json``

The output mirrors the subset of HOV-SG artifacts consumed by
``HOVSGNavigatorAdapter`` so Voltron can exercise room / object grounding and
cross-room path planning without importing the full HOV-SG mapping stack.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import cv2
import networkx as nx
from PIL import Image
from skimage.morphology import medial_axis

from voltron.shared.models.scene_state import door_is_open_from_joints

DEFAULT_MAP_RESOLUTION = 0.1
DEFAULT_LAYOUT_PIXEL_RESOLUTION = 0.01
DEFAULT_FLOOR_ID = "0"
DEFAULT_VERTICAL_AXIS = "z"

STRUCTURAL_OBJECT_PREFIXES = (
    "baseboard_",
    "ceilings_",
    "downlight_",
    "electric_switch_",
    "floors_",
    "roof_",
    "stairs_",
    "walls_",
    "window_",
)


@dataclass(frozen=True)
class RoomRegion:
    room_id: str
    floor_id: str
    instance_name: str
    room_type: str
    ins_id: int
    sem_id: int
    centroid: dict[str, float]
    vertices: list[list[float]]


def export_behavior_scene_graph(
    *,
    scene_model: str,
    behavior_assets_root: str | Path,
    output_root: str | Path,
    scene_instance: str | None = None,
    scene_file: str | Path | None = None,
    include_structural_objects: bool = False,
    include_object_categories: list[str] | tuple[str, ...] | None = None,
    map_resolution: float = DEFAULT_MAP_RESOLUTION,
    trav_map_filename: str = "floor_trav_0.png",
) -> dict[str, Any]:
    """Export a BEHAVIOR scene into lightweight graph assets and a scene map."""

    assets_root = Path(behavior_assets_root).expanduser().resolve()
    output_dir = Path(output_root).expanduser().resolve() / scene_model
    graph_dir = output_dir / "graph"
    floors_dir = graph_dir / "floors"
    rooms_dir = graph_dir / "rooms"
    objects_dir = graph_dir / "objects"
    nav_graph_dir = graph_dir / "nav_graph"

    for directory in (floors_dir, rooms_dir, objects_dir, nav_graph_dir):
        directory.mkdir(parents=True, exist_ok=True)

    scene_dir = assets_root / "scenes" / scene_model
    layout_dir = scene_dir / "layout"
    json_dir = scene_dir / "json"
    if scene_file is not None:
        scene_json_path = Path(scene_file).expanduser().resolve()
    else:
        scene_json_path = json_dir / f"{scene_instance or scene_model + '_best'}.json"
    if not scene_json_path.exists():
        raise FileNotFoundError(f"Scene JSON not found: {scene_json_path}")

    room_categories_path = assets_root / "metadata" / "room_categories.txt"
    if not room_categories_path.exists():
        raise FileNotFoundError(f"Room categories file not found: {room_categories_path}")

    scene_payload = json.loads(scene_json_path.read_text(encoding="utf-8"))
    room_categories = [
        line.strip()
        for line in room_categories_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    room_ins_map = _load_grayscale(layout_dir / "floor_insseg_0.png")
    room_sem_map = _load_grayscale(layout_dir / "floor_semseg_0.png")
    trav_map_name = str(trav_map_filename or "floor_trav_0.png").strip() or "floor_trav_0.png"
    trav_map = _load_grayscale(layout_dir / trav_map_name)
    room_ins_map, room_sem_map, trav_map = _normalize_maps(
        room_ins_map=room_ins_map,
        room_sem_map=room_sem_map,
        trav_map=trav_map,
        map_resolution=map_resolution,
    )
    map_size = int(room_ins_map.shape[0])

    room_regions = _extract_room_regions(
        room_ins_map=room_ins_map,
        room_sem_map=room_sem_map,
        room_categories=room_categories,
        map_size=map_size,
        map_resolution=map_resolution,
    )
    room_by_instance = {room.instance_name: room for room in room_regions}

    object_registry = (
        scene_payload.get("state", {})
        .get("registry", {})
        .get("object_registry", {})
    )
    included_categories = _normalize_category_set(include_object_categories)
    object_entries = _extract_object_entries(
        scene_payload=scene_payload,
        object_registry=object_registry,
        room_by_instance=room_by_instance,
        include_structural_objects=include_structural_objects,
        include_object_categories=included_categories,
    )

    adjacency_by_room = _build_room_adjacency(
        room_ins_map=room_ins_map,
        trav_map=trav_map,
        room_regions=room_regions,
        scene_payload=scene_payload,
    )
    routes = _build_room_routes(adjacency_by_room)
    nav_graph = _build_nav_graph(room_regions=room_regions, adjacency_by_room=adjacency_by_room)
    voronoi_graph = _build_sparse_voronoi_graph(
        room_ins_map=room_ins_map,
        trav_map=trav_map,
        room_regions=room_regions,
        map_size=map_size,
        map_resolution=map_resolution,
    )

    floor_vertices = _floor_vertices(room_regions)
    floor_payload = {
        "floor_id": DEFAULT_FLOOR_ID,
        "name": scene_model,
        "rooms": [room.room_id for room in room_regions],
        "vertices": floor_vertices,
        "floor_zero_level": 0.0,
        "position": _floor_position(room_regions),
    }
    _write_json(floors_dir / f"{DEFAULT_FLOOR_ID}.json", floor_payload)

    for room in room_regions:
        room_payload = {
            "room_id": room.room_id,
            "name": room.instance_name,
            "floor_id": room.floor_id,
            "objects": [entry["object_id"] for entry in object_entries if entry["room_id"] == room.room_id],
            "vertices": room.vertices,
            "position": dict(room.centroid),
        }
        _write_json(rooms_dir / f"{room.room_id}.json", room_payload)

    for entry in object_entries:
        _write_json(objects_dir / f"{entry['object_id']}.json", entry)

    _write_json(nav_graph_dir / "global_nav_graph_graph.json", nav_graph)
    _write_json(nav_graph_dir / "sparse_voronoi_graph.json", voronoi_graph)
    _write_json(
        graph_dir / "metadata.json",
        {
            "scene_id": scene_model,
            "source": "behavior_scene_graph",
            "scene_map_source": "gt",
            "scene_model": scene_model,
            "scene_json": str(scene_json_path),
            "map_resolution": map_resolution,
            "trav_map_filename": trav_map_name,
            "default_nav_graph_type": "voronoi_graph",
            "nav_graphs": [
                {
                    "type": "global_room_graph",
                    "filename": "global_nav_graph_graph.json",
                },
                {
                    "type": "voronoi_graph",
                    "filename": "sparse_voronoi_graph.json",
                }
            ],
            "vertical_axis": DEFAULT_VERTICAL_AXIS,
        },
    )

    scene_map_payload = {
        "scene_id": scene_model,
        "source": "behavior_scene_graph",
        "scene_map_source": "gt",
        "graph_path": str(graph_dir),
        "coord_system": {"vertical_axis": DEFAULT_VERTICAL_AXIS},
        "floors": [
            {
                "id": DEFAULT_FLOOR_ID,
                "name": scene_model,
                "zero_level": 0.0,
                "rooms": [room.instance_name for room in room_regions],
            }
        ],
        "regions": [
            {
                "id": room.room_id,
                "name": room.instance_name,
                "room_type": room.room_type,
                "floor_id": room.floor_id,
                "centroid": dict(room.centroid),
                "connected_regions": sorted(adjacency_by_room.get(room.instance_name, [])),
            }
            for room in room_regions
        ],
        "anchors": [
            {
                "id": entry["object_id"],
                "name": entry["name"],
                "room": _room_name_from_id(entry["room_id"], room_regions),
                "floor_id": DEFAULT_FLOOR_ID,
                "position": dict(entry["position"]),
            }
            for entry in object_entries
        ],
        "topology": {
            "adjacency": {key: sorted(value) for key, value in sorted(adjacency_by_room.items())},
            "routes": routes,
        },
        "exploration": {
            "frontiers": [
                {
                    "region": room.instance_name,
                    "floor_id": room.floor_id,
                    "centroid": dict(room.centroid),
                }
                for room in room_regions
            ]
        },
    }
    scene_map_path = output_dir / "scene_map.json"
    _write_json(scene_map_path, scene_map_payload)

    return {
        "scene_id": scene_model,
        "scene_json": str(scene_json_path),
        "graph_path": str(graph_dir),
        "scene_map_path": str(scene_map_path),
        "trav_map_filename": trav_map_name,
        "room_count": len(room_regions),
        "object_count": len(object_entries),
        "nav_node_count": len(nav_graph["nodes"]),
        "nav_edge_count": len(nav_graph["links"]),
        "voronoi_node_count": len(voronoi_graph["nodes"]),
        "voronoi_edge_count": len(voronoi_graph["links"]),
    }


def _extract_room_regions(
    *,
    room_ins_map: np.ndarray,
    room_sem_map: np.ndarray,
    room_categories: list[str],
    map_size: int,
    map_resolution: float,
) -> list[RoomRegion]:
    unique_ins_ids = sorted(int(value) for value in np.unique(room_ins_map) if int(value) != 0)
    sem_to_ins: dict[int, list[int]] = {}
    for ins_id in unique_ins_ids:
        rows, cols = np.where(room_ins_map == ins_id)
        if rows.size == 0:
            continue
        sem_id = int(room_sem_map[rows[0], cols[0]])
        sem_to_ins.setdefault(sem_id, []).append(ins_id)

    regions: list[RoomRegion] = []
    room_index = 0
    for sem_id, ins_ids in sem_to_ins.items():
        room_type = room_categories[sem_id - 1]
        for local_index, ins_id in enumerate(ins_ids):
            rows, cols = np.where(room_ins_map == ins_id)
            centroid_xy = _interior_point(rows, cols, map_size=map_size, map_resolution=map_resolution)
            regions.append(
                RoomRegion(
                    room_id=f"{DEFAULT_FLOOR_ID}_{room_index}",
                    floor_id=DEFAULT_FLOOR_ID,
                    instance_name=f"{room_type}_{local_index}",
                    room_type=room_type,
                    ins_id=ins_id,
                    sem_id=sem_id,
                    centroid={"x": centroid_xy[0], "y": centroid_xy[1], "z": 0.0},
                    vertices=_room_vertices(
                        mask=(room_ins_map == ins_id),
                        rows=rows,
                        cols=cols,
                        map_size=map_size,
                        map_resolution=map_resolution,
                    ),
                )
            )
            room_index += 1
    return regions


def _extract_object_entries(
    *,
    scene_payload: dict[str, Any],
    object_registry: dict[str, Any],
    room_by_instance: dict[str, RoomRegion],
    include_structural_objects: bool,
    include_object_categories: set[str],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    next_index_by_room: dict[str, int] = defaultdict(int)
    object_infos = scene_payload.get("objects_info", {}).get("init_info", {})

    for object_name, object_info in object_infos.items():
        object_args = object_info.get("args", {})
        object_category = _normalize_category(object_args.get("category"))
        keep_selected_category = bool(object_category and object_category in include_object_categories)
        if not include_structural_objects and not keep_selected_category and _is_structural_object(object_name):
            continue

        room_names = object_args.get("in_rooms", [])
        if isinstance(room_names, str):
            room_names = [room_names]
        room_name = next((name for name in room_names if name in room_by_instance), None)
        if room_name is None:
            continue

        root_link = object_registry.get(object_name, {}).get("root_link", {})
        position = root_link.get("pos")
        if not isinstance(position, list) or len(position) < 3:
            continue

        room = room_by_instance[room_name]
        object_index = next_index_by_room[room.room_id]
        next_index_by_room[room.room_id] += 1
        entries.append(
            {
                "object_id": f"{room.room_id}_{object_index}",
                "room_id": room.room_id,
                "name": _normalize_object_name(object_name),
                "category": object_category,
                "source_object_name": object_name,
                "vertices": _object_vertices(position),
                "position": {
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "z": float(position[2]),
                },
            }
        )

    return entries


def _build_room_adjacency(
    *,
    room_ins_map: np.ndarray,
    trav_map: np.ndarray,
    room_regions: list[RoomRegion],
    scene_payload: dict[str, Any],
) -> dict[str, set[str]]:
    adjacency_by_ins_id = _adjacency_from_traversable_boundaries(room_ins_map=room_ins_map, trav_map=trav_map)
    room_name_by_ins_id = {room.ins_id: room.instance_name for room in room_regions}
    adjacency: dict[str, set[str]] = {room.instance_name: set() for room in room_regions}

    for ins_id, neighbors in adjacency_by_ins_id.items():
        room_name = room_name_by_ins_id.get(ins_id)
        if room_name is None:
            continue
        for neighbor_ins_id in neighbors:
            neighbor_name = room_name_by_ins_id.get(neighbor_ins_id)
            if neighbor_name is None or neighbor_name == room_name:
                continue
            adjacency[room_name].add(neighbor_name)

    _add_explicit_door_adjacency(adjacency=adjacency, scene_payload=scene_payload)

    if sum(len(neighbors) for neighbors in adjacency.values()) == 0:
        for previous, current in zip(room_regions, room_regions[1:]):
            adjacency[previous.instance_name].add(current.instance_name)
            adjacency[current.instance_name].add(previous.instance_name)

    return adjacency


def _add_explicit_door_adjacency(
    *,
    adjacency: dict[str, set[str]],
    scene_payload: dict[str, Any],
) -> None:
    object_infos = scene_payload.get("objects_info", {}).get("init_info", {})
    if not isinstance(object_infos, dict):
        return
    object_registry = scene_payload.get("state", {}).get("registry", {}).get("object_registry", {})
    if not isinstance(object_registry, dict):
        object_registry = {}

    for object_key, object_info in object_infos.items():
        if not isinstance(object_info, dict):
            continue
        object_args = object_info.get("args")
        if not isinstance(object_args, dict):
            continue
        if not _is_door_category(object_args.get("category")):
            continue
        object_name = object_args.get("name") if isinstance(object_args.get("name"), str) else str(object_key)
        if not _door_state_is_open(object_registry.get(object_name) or object_registry.get(str(object_key))):
            continue
        raw_room_names = object_args.get("in_rooms", [])
        if isinstance(raw_room_names, str):
            raw_room_names = [raw_room_names]
        if not isinstance(raw_room_names, list):
            continue
        room_names = sorted({str(name) for name in raw_room_names if str(name) in adjacency})
        for left_index, left_room in enumerate(room_names):
            for right_room in room_names[left_index + 1 :]:
                if left_room == right_room:
                    continue
                adjacency[left_room].add(right_room)
                adjacency[right_room].add(left_room)


def _door_state_is_open(state: Any) -> bool:
    if not isinstance(state, dict):
        return True
    joint_pos = state.get("joint_pos")
    is_open = door_is_open_from_joints([value for value in joint_pos if _is_number(value)] if isinstance(joint_pos, list) else None)
    return True if is_open is None else is_open


def _ensure_connected_adjacency(
    *,
    adjacency: dict[str, set[str]],
    room_regions: list[RoomRegion],
) -> dict[str, set[str]]:
    graph = nx.Graph()
    graph.add_nodes_from(adjacency.keys())
    for room_name, neighbors in adjacency.items():
        for neighbor_name in neighbors:
            graph.add_edge(room_name, neighbor_name)

    if graph.number_of_nodes() <= 1:
        return adjacency

    room_by_name = {room.instance_name: room for room in room_regions}
    while not nx.is_connected(graph):
        components = [sorted(component) for component in nx.connected_components(graph)]
        best_pair: tuple[str, str] | None = None
        best_distance: float | None = None
        for left_index, left_component in enumerate(components):
            for right_component in components[left_index + 1 :]:
                for left_room in left_component:
                    for right_room in right_component:
                        distance = _euclidean_distance(
                            room_by_name[left_room].centroid,
                            room_by_name[right_room].centroid,
                        )
                        if best_distance is None or distance < best_distance:
                            best_distance = distance
                            best_pair = (left_room, right_room)
        if best_pair is None:
            break
        left_room, right_room = best_pair
        adjacency[left_room].add(right_room)
        adjacency[right_room].add(left_room)
        graph.add_edge(left_room, right_room)
    return adjacency


def _adjacency_from_traversable_boundaries(
    *,
    room_ins_map: np.ndarray,
    trav_map: np.ndarray,
) -> dict[int, set[int]]:
    traversable = trav_map == 255
    room_ids = room_ins_map.astype(np.int64)
    adjacency: dict[int, set[int]] = defaultdict(set)

    for axis in (0, 1):
        current_rooms = np.take(room_ids, indices=range(room_ids.shape[axis] - 1), axis=axis)
        next_rooms = np.take(room_ids, indices=range(1, room_ids.shape[axis]), axis=axis)
        current_trav = np.take(traversable, indices=range(traversable.shape[axis] - 1), axis=axis)
        next_trav = np.take(traversable, indices=range(1, traversable.shape[axis]), axis=axis)
        mask = (
            current_trav
            & next_trav
            & (current_rooms != 0)
            & (next_rooms != 0)
            & (current_rooms != next_rooms)
        )
        if not np.any(mask):
            continue
        pairs = np.stack((current_rooms[mask], next_rooms[mask]), axis=1)
        for left, right in pairs.tolist():
            adjacency[int(left)].add(int(right))
            adjacency[int(right)].add(int(left))

    return adjacency


def _build_room_routes(adjacency_by_room: dict[str, set[str]]) -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    room_names = sorted(adjacency_by_room)
    for start in room_names:
        for goal in room_names:
            if start == goal:
                continue
            path = _shortest_room_path(adjacency_by_room, start, goal)
            if not path:
                continue
            routes.append(
                {
                    "start": {"region": start},
                    "goal": {"region": goal},
                    "path": path,
                }
            )
    return routes


def _shortest_room_path(adjacency_by_room: dict[str, set[str]], start: str, goal: str) -> list[str]:
    if start == goal:
        return [start]
    queue: deque[list[str]] = deque([[start]])
    visited = {start}

    while queue:
        path = queue.popleft()
        room_name = path[-1]
        for neighbor in sorted(adjacency_by_room.get(room_name, [])):
            if neighbor in visited:
                continue
            next_path = path + [neighbor]
            if neighbor == goal:
                return next_path
            visited.add(neighbor)
            queue.append(next_path)
    return []


def _build_nav_graph(
    *,
    room_regions: list[RoomRegion],
    adjacency_by_room: dict[str, set[str]],
) -> dict[str, Any]:
    node_id_by_room = {room.instance_name: [index, index, room.floor_id] for index, room in enumerate(room_regions)}
    nodes = [
        {
            "id": node_id_by_room[room.instance_name],
            "pos": [room.centroid["x"], room.centroid["y"], room.centroid["z"]],
            "floor_id": room.floor_id,
            "room_id": room.room_id,
            "room_name": room.instance_name,
        }
        for room in room_regions
    ]

    links: list[dict[str, Any]] = []
    room_by_name = {room.instance_name: room for room in room_regions}
    for room_name, neighbors in sorted(adjacency_by_room.items()):
        for neighbor_name in sorted(neighbors):
            if room_name >= neighbor_name:
                continue
            room = room_by_name[room_name]
            neighbor = room_by_name[neighbor_name]
            links.append(
                {
                    "source": node_id_by_room[room_name],
                    "target": node_id_by_room[neighbor_name],
                    "dist": _euclidean_distance(room.centroid, neighbor.centroid),
                }
            )

    return {"nodes": nodes, "links": links}


def _build_sparse_voronoi_graph(
    *,
    room_ins_map: np.ndarray,
    trav_map: np.ndarray,
    room_regions: list[RoomRegion],
    map_size: int,
    map_resolution: float,
) -> dict[str, Any]:
    traversable = trav_map == 255
    if not np.any(traversable):
        return {"nodes": [], "links": []}

    skeleton, distance_map = medial_axis(traversable, return_distance=True)
    if not np.any(skeleton):
        return {"nodes": [], "links": []}

    skeleton_coords = [tuple(int(value) for value in coord) for coord in np.argwhere(skeleton)]
    skeleton_set = set(skeleton_coords)
    if not skeleton_set:
        return {"nodes": [], "links": []}

    neighbors_by_coord: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for row, col in skeleton_coords:
        neighbors: list[tuple[int, int]] = []
        for delta_row in (-1, 0, 1):
            for delta_col in (-1, 0, 1):
                if delta_row == 0 and delta_col == 0:
                    continue
                candidate = (row + delta_row, col + delta_col)
                if candidate in skeleton_set:
                    neighbors.append(candidate)
        neighbors_by_coord[(row, col)] = sorted(neighbors)

    interest_coords = [
        coord for coord, neighbors in neighbors_by_coord.items() if len(neighbors) != 2
    ]
    if not interest_coords:
        max_clearance_coord = max(
            skeleton_coords,
            key=lambda coord: float(distance_map[coord[0], coord[1]]),
        )
        interest_coords = [max_clearance_coord]
        farthest_coord = max(
            skeleton_coords,
            key=lambda coord: (coord[0] - max_clearance_coord[0]) ** 2 + (coord[1] - max_clearance_coord[1]) ** 2,
        )
        if farthest_coord != max_clearance_coord:
            interest_coords.append(farthest_coord)
    elif len(interest_coords) == 1 and len(skeleton_coords) > 1:
        start_coord = interest_coords[0]
        farthest_coord = max(
            skeleton_coords,
            key=lambda coord: (coord[0] - start_coord[0]) ** 2 + (coord[1] - start_coord[1]) ** 2,
        )
        if farthest_coord != start_coord:
            interest_coords.append(farthest_coord)

    room_by_ins_id = {room.ins_id: room for room in room_regions}
    room_by_name = {room.instance_name: room for room in room_regions}
    node_id_by_coord = {
        coord: [index, index, DEFAULT_FLOOR_ID]
        for index, coord in enumerate(sorted(set(interest_coords)))
    }

    nodes = [
        _voronoi_node_payload(
            coord=coord,
            node_id=node_id_by_coord[coord],
            room_ins_map=room_ins_map,
            room_by_ins_id=room_by_ins_id,
            room_by_name=room_by_name,
            distance_map=distance_map,
            map_size=map_size,
            map_resolution=map_resolution,
        )
        for coord in sorted(node_id_by_coord)
    ]

    links: list[dict[str, Any]] = []
    seen_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    interest_set = set(node_id_by_coord)
    for start_coord in sorted(interest_set):
        for next_coord in neighbors_by_coord[start_coord]:
            edge_key = tuple(sorted((start_coord, next_coord)))
            if edge_key in seen_edges:
                continue

            path = [start_coord, next_coord]
            previous_coord = start_coord
            current_coord = next_coord
            while current_coord not in interest_set:
                candidates = [coord for coord in neighbors_by_coord[current_coord] if coord != previous_coord]
                if not candidates:
                    break
                previous_coord = current_coord
                current_coord = candidates[0]
                path.append(current_coord)

            if current_coord == start_coord or current_coord not in interest_set:
                continue

            seen_edges.add(tuple(sorted((start_coord, current_coord))))
            links.append(
                _voronoi_link_payload(
                    source_id=node_id_by_coord[start_coord],
                    target_id=node_id_by_coord[current_coord],
                    path=path,
                    distance_map=distance_map,
                    map_size=map_size,
                    map_resolution=map_resolution,
                )
            )

    return {"nodes": nodes, "links": links}


def _voronoi_node_payload(
    *,
    coord: tuple[int, int],
    node_id: list[Any],
    room_ins_map: np.ndarray,
    room_by_ins_id: dict[int, RoomRegion],
    room_by_name: dict[str, RoomRegion],
    distance_map: np.ndarray,
    map_size: int,
    map_resolution: float,
) -> dict[str, Any]:
    row, col = coord
    x_coord, y_coord = _map_to_world(
        row=float(row),
        col=float(col),
        map_size=map_size,
        map_resolution=map_resolution,
    )
    room = room_by_ins_id.get(int(room_ins_map[row, col]))
    if room is None:
        room = min(
            room_by_name.values(),
            key=lambda candidate: (candidate.centroid["x"] - x_coord) ** 2 + (candidate.centroid["y"] - y_coord) ** 2,
        )
    clearance_m = float(distance_map[row, col]) * map_resolution
    return {
        "id": node_id,
        "pos": [x_coord, y_coord, 0.0],
        "floor_id": room.floor_id,
        "room_id": room.room_id,
        "room_name": room.instance_name,
        "clearance_m": clearance_m,
        "map_row": row,
        "map_col": col,
    }


def _voronoi_link_payload(
    *,
    source_id: list[Any],
    target_id: list[Any],
    path: list[tuple[int, int]],
    distance_map: np.ndarray,
    map_size: int,
    map_resolution: float,
) -> dict[str, Any]:
    polyline: list[list[float]] = []
    path_length = 0.0
    min_clearance_m = None
    previous_xy: tuple[float, float] | None = None
    for row, col in path:
        x_coord, y_coord = _map_to_world(
            row=float(row),
            col=float(col),
            map_size=map_size,
            map_resolution=map_resolution,
        )
        polyline.append([x_coord, y_coord, 0.0])
        if previous_xy is not None:
            path_length += float(np.hypot(x_coord - previous_xy[0], y_coord - previous_xy[1]))
        previous_xy = (x_coord, y_coord)
        clearance_m = float(distance_map[row, col]) * map_resolution
        if min_clearance_m is None or clearance_m < min_clearance_m:
            min_clearance_m = clearance_m

    return {
        "source": source_id,
        "target": target_id,
        "dist": path_length,
        "min_clearance_m": float(min_clearance_m or 0.0),
        "polyline": polyline,
    }


def _floor_vertices(room_regions: list[RoomRegion]) -> list[list[float]]:
    all_vertices = [vertex for room in room_regions for vertex in room.vertices]
    if not all_vertices:
        return []
    min_x = min(vertex[0] for vertex in all_vertices)
    max_x = max(vertex[0] for vertex in all_vertices)
    min_y = min(vertex[1] for vertex in all_vertices)
    max_y = max(vertex[1] for vertex in all_vertices)
    return [
        [min_x, min_y, 0.0],
        [max_x, min_y, 0.0],
        [max_x, max_y, 0.0],
        [min_x, max_y, 0.0],
    ]


def _room_name_from_id(room_id: str, room_regions: list[RoomRegion]) -> str:
    for room in room_regions:
        if room.room_id == room_id:
            return room.instance_name
    return room_id


def _floor_position(room_regions: list[RoomRegion]) -> dict[str, float]:
    if not room_regions:
        return {"x": 0.0, "y": 0.0, "z": 0.0}
    x_coord = sum(room.centroid["x"] for room in room_regions) / float(len(room_regions))
    y_coord = sum(room.centroid["y"] for room in room_regions) / float(len(room_regions))
    return {"x": x_coord, "y": y_coord, "z": 0.0}


def _pixel_centroid(
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    map_size: int,
    map_resolution: float,
) -> tuple[float, float]:
    row = float(rows.mean())
    col = float(cols.mean())
    return _map_to_world(row=row, col=col, map_size=map_size, map_resolution=map_resolution)


def _interior_point(
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    map_size: int,
    map_resolution: float,
) -> tuple[float, float]:
    mean_row = float(rows.mean())
    mean_col = float(cols.mean())
    distances = (rows.astype(np.float64) - mean_row) ** 2 + (cols.astype(np.float64) - mean_col) ** 2
    index = int(np.argmin(distances))
    return _map_to_world(
        row=float(rows[index]),
        col=float(cols[index]),
        map_size=map_size,
        map_resolution=map_resolution,
    )


def _room_vertices(
    *,
    mask: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    map_size: int,
    map_resolution: float,
) -> list[list[float]]:
    contour_vertices = _contour_vertices(mask=mask, map_size=map_size, map_resolution=map_resolution)
    if contour_vertices:
        return contour_vertices
    return _bbox_vertices(rows=rows, cols=cols, map_size=map_size, map_resolution=map_resolution)


def _contour_vertices(
    *,
    mask: np.ndarray,
    map_size: int,
    map_resolution: float,
) -> list[list[float]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    if contour.shape[0] < 3:
        return []
    epsilon = 0.01 * cv2.arcLength(contour, closed=True)
    simplified = cv2.approxPolyDP(contour, epsilon=epsilon, closed=True)
    vertices: list[list[float]] = []
    for point in simplified.reshape(-1, 2):
        col = float(point[0])
        row = float(point[1])
        x, y = _map_to_world(row=row, col=col, map_size=map_size, map_resolution=map_resolution)
        vertices.append([x, y, 0.0])
    return vertices


def _bbox_vertices(
    *,
    rows: np.ndarray,
    cols: np.ndarray,
    map_size: int,
    map_resolution: float,
) -> list[list[float]]:
    min_row = float(rows.min())
    max_row = float(rows.max())
    min_col = float(cols.min())
    max_col = float(cols.max())
    min_x, min_y = _map_to_world(row=min_row, col=min_col, map_size=map_size, map_resolution=map_resolution)
    max_x, max_y = _map_to_world(row=max_row, col=max_col, map_size=map_size, map_resolution=map_resolution)
    low_x, high_x = sorted((min_x, max_x))
    low_y, high_y = sorted((min_y, max_y))
    return [
        [low_x, low_y, 0.0],
        [high_x, low_y, 0.0],
        [high_x, high_y, 0.0],
        [low_x, high_y, 0.0],
    ]


def _map_to_world(*, row: float, col: float, map_size: int, map_resolution: float) -> tuple[float, float]:
    x = (col - map_size / 2.0) * map_resolution
    y = (row - map_size / 2.0) * map_resolution
    return x, y


def _object_vertices(position: list[float], radius: float = 0.15) -> list[list[float]]:
    x = float(position[0])
    y = float(position[1])
    z = float(position[2])
    return [
        [x - radius, y - radius, z],
        [x + radius, y - radius, z],
        [x + radius, y + radius, z],
        [x - radius, y + radius, z],
    ]


def _centroid_from_vertices(vertices: list[list[float]]) -> dict[str, float]:
    if not vertices:
        return {"x": 0.0, "y": 0.0, "z": 0.0}
    values = np.asarray(vertices, dtype=float)
    centroid = values.mean(axis=0)
    return {"x": float(centroid[0]), "y": float(centroid[1]), "z": float(centroid[2])}


def _normalize_object_name(object_name: str) -> str:
    name = re.sub(r"_\d+$", "", object_name)
    parts = name.split("_")
    if len(parts) >= 2 and re.fullmatch(r"[a-z]{6}", parts[-1] or ""):
        parts = parts[:-1]
    return " ".join(parts)


def _is_structural_object(object_name: str) -> bool:
    return object_name.startswith(STRUCTURAL_OBJECT_PREFIXES)


def _is_door_category(value: Any) -> bool:
    category = _normalize_category(value).replace("-", "_").replace(" ", "_")
    return category == "door" or category.endswith("_door")


def _normalize_category(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def _normalize_category_set(values: list[str] | tuple[str, ...] | None) -> set[str]:
    if values is None:
        return set()
    return {normalized for value in values if (normalized := _normalize_category(value))}


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _euclidean_distance(left: dict[str, float], right: dict[str, float]) -> float:
    return float(
        np.linalg.norm(
            np.asarray([left["x"], left["y"], left["z"]], dtype=float)
            - np.asarray([right["x"], right["y"], right["z"]], dtype=float)
        )
    )


def _load_grayscale(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Required layout asset not found: {path}")
    return np.asarray(Image.open(path).convert("L"))


def _normalize_maps(
    *,
    room_ins_map: np.ndarray,
    room_sem_map: np.ndarray,
    trav_map: np.ndarray,
    map_resolution: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if room_ins_map.shape[0] < 1000:
        return room_ins_map, room_sem_map, trav_map

    target_size = int(room_ins_map.shape[0] * DEFAULT_LAYOUT_PIXEL_RESOLUTION / map_resolution)
    target_size = max(target_size, 1)
    room_ins_map = cv2.resize(room_ins_map, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    room_sem_map = cv2.resize(room_sem_map, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    trav_map = cv2.resize(trav_map, (target_size, target_size))
    trav_map[trav_map < 255] = 0
    return room_ins_map, room_sem_map, trav_map


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def _default_behavior_assets_root() -> str:
    env_path = os.getenv("OMNIGIBSON_DATA_PATH")
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.name == "datasets":
            assets_root = candidate / "behavior-1k-assets"
            if assets_root.exists():
                return str(assets_root)
        if (candidate / "behavior-1k-assets").exists():
            return str(candidate / "behavior-1k-assets")
    return "/mnt/data/huangyixuan/isaac/BEHAVIOR-1K/datasets/behavior-1k-assets"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-model", required=True, help="BEHAVIOR scene model, e.g. house_single_floor")
    parser.add_argument(
        "--behavior-assets-root",
        default=_default_behavior_assets_root(),
        help="Root directory containing behavior-1k-assets",
    )
    parser.add_argument(
        "--output-root",
        default="/mnt/data/huangyixuan/hovsg_exports",
        help="Directory for generated scene exports",
    )
    parser.add_argument("--scene-instance", default=None, help="Optional scene instance JSON name without .json")
    parser.add_argument(
        "--scene-file",
        default=None,
        help="Optional absolute path to a BEHAVIOR scene JSON file. Overrides --scene-instance.",
    )
    parser.add_argument(
        "--include-structural-objects",
        action="store_true",
        help="Include floors / walls / ceilings and similar structural assets in the object graph",
    )
    parser.add_argument(
        "--include-object-category",
        action="append",
        default=[],
        help="Object category to keep even when its instance name matches the structural-object filter. Repeatable.",
    )
    parser.add_argument(
        "--trav-map-filename",
        default="floor_trav_0.png",
        help="Traversability layout filename under scenes/<scene_model>/layout.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = export_behavior_scene_graph(
        scene_model=args.scene_model,
        behavior_assets_root=args.behavior_assets_root,
        output_root=args.output_root,
        scene_instance=args.scene_instance,
        scene_file=args.scene_file,
        include_structural_objects=args.include_structural_objects,
        include_object_categories=args.include_object_category,
        trav_map_filename=args.trav_map_filename,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
