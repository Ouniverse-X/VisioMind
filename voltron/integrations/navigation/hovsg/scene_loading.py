"""Scene and graph loading helpers for the HOV-SG navigation integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx

from voltron.shared.models.scene_state import navigation_role_from_category

from .models import (
    HOVSGFloorAsset,
    HOVSGNavGraphAsset,
    HOVSGObjectAsset,
    HOVSGRoomAsset,
    HOVSGSceneAsset,
)


def resolve_graph_path(
    *,
    scene_id: str,
    graph_root: Path | None,
    scene_roots: dict[str, Path],
    config: dict[str, Any] | None = None,
) -> Path:
    config = dict(config or {})
    explicit = config.get("graph_path")
    if explicit:
        path = Path(explicit).expanduser()
        return path if (path / "floors").exists() else path / "graph"

    if scene_id in scene_roots:
        root = scene_roots[scene_id]
        return root if (root / "floors").exists() else root / "graph"

    if graph_root is not None:
        candidates = [
            graph_root / scene_id / "graph",
            graph_root / scene_id,
            graph_root,
        ]
        for candidate in candidates:
            if (candidate / "floors").exists() and (candidate / "rooms").exists():
                return candidate

    raise FileNotFoundError(f"Cannot resolve HOV-SG graph path for scene '{scene_id}'")


def load_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path / "metadata.json"
    if metadata_path.exists():
        return read_json(metadata_path)
    return {}


def resolve_vertical_axis(
    metadata: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
    default_vertical_axis: str = "y",
) -> str:
    config = dict(config or {})
    for candidate in (
        config.get("vertical_axis"),
        metadata.get("vertical_axis"),
        default_vertical_axis,
    ):
        if isinstance(candidate, str) and candidate in {"x", "y", "z"}:
            return candidate
    return "y"


def resolve_scene_map_source(metadata: dict[str, Any], *, config: dict[str, Any] | None = None) -> str | None:
    config = dict(config or {})
    for candidate in (
        config.get("scene_map_source"),
        metadata.get("scene_map_source"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def load_floors(path: Path) -> dict[str, HOVSGFloorAsset]:
    floors: dict[str, HOVSGFloorAsset] = {}
    for json_file in sorted(path.glob("*.json")):
        metadata = read_json(json_file)
        floor_id = str(metadata["floor_id"])
        floors[floor_id] = HOVSGFloorAsset(
            floor_id=floor_id,
            name=metadata.get("name"),
            room_ids=[str(room_id) for room_id in metadata.get("rooms", [])],
            floor_zero_level=to_float(metadata.get("floor_zero_level")),
            centroid=position_from_metadata(metadata),
        )
    return floors


def load_rooms(path: Path) -> dict[str, HOVSGRoomAsset]:
    rooms: dict[str, HOVSGRoomAsset] = {}
    for json_file in sorted(path.glob("*.json")):
        metadata = read_json(json_file)
        room_id = str(metadata["room_id"])
        rooms[room_id] = HOVSGRoomAsset(
            room_id=room_id,
            floor_id=str(metadata["floor_id"]),
            name=metadata.get("name"),
            object_ids=[str(object_id) for object_id in metadata.get("objects", [])],
            centroid=position_from_metadata(metadata),
            vertices=[point for point in metadata.get("vertices", []) if isinstance(point, list) and len(point) >= 3],
            connected_room_ids=[],
        )
    return rooms


def load_room_adjacency(
    *,
    graph_path: Path,
    rooms: dict[str, HOVSGRoomAsset],
) -> dict[str, set[str]] | None:
    payload = load_scene_map_payload(graph_path)
    if payload is None:
        return None

    regions = payload.get("regions")
    if not isinstance(regions, list):
        return None

    room_id_by_name = {
        str(room.name): room_id
        for room_id, room in rooms.items()
        if isinstance(room.name, str) and room.name
    }
    adjacency: dict[str, set[str]] = {room_id: set() for room_id in rooms}
    found_explicit_topology = False
    for region in regions:
        if not isinstance(region, dict):
            continue
        region_id = str(region.get("id") or "").strip()
        region_name = str(region.get("name") or "").strip()
        room_id = region_id if region_id in rooms else room_id_by_name.get(region_name)
        if room_id is None:
            continue

        connected_regions = region.get("connected_regions")
        if not isinstance(connected_regions, list):
            continue

        found_explicit_topology = True
        connected_room_ids: set[str] = set()
        for candidate in connected_regions:
            candidate_text = str(candidate or "").strip()
            if not candidate_text:
                continue
            candidate_room_id = candidate_text if candidate_text in rooms else room_id_by_name.get(candidate_text)
            if candidate_room_id is not None and candidate_room_id != room_id:
                connected_room_ids.add(candidate_room_id)
        adjacency[room_id].update(connected_room_ids)

    if not found_explicit_topology:
        return None

    for room_id, neighbors in adjacency.items():
        room = rooms.get(room_id)
        if room is not None:
            room.connected_room_ids = sorted(neighbors)
    return adjacency


def load_scene_map_payload(graph_path: Path) -> dict[str, Any] | None:
    scene_map_path = graph_path.parent / "scene_map.json"
    if not scene_map_path.exists():
        return None
    return read_json(scene_map_path)


def validate_scene_graph_consistency(scene: HOVSGSceneAsset) -> None:
    payload = load_scene_map_payload(scene.graph_path)
    if payload is None:
        return
    validate_scene_map_graph_path(scene=scene, payload=payload)
    validate_scene_map_nav_graph_topology(scene=scene)


def validate_scene_map_graph_path(*, scene: HOVSGSceneAsset, payload: dict[str, Any]) -> None:
    configured_graph_path = payload.get("graph_path")
    if not isinstance(configured_graph_path, str) or not configured_graph_path.strip():
        return
    scene_map_graph_path = Path(configured_graph_path).expanduser().resolve()
    actual_graph_path = scene.graph_path.expanduser().resolve()
    if scene_map_graph_path != actual_graph_path:
        raise ValueError(
            "Scene map graph_path mismatch for "
            f"scene '{scene.scene_id}': scene_map={scene_map_graph_path}, loaded_graph={actual_graph_path}"
        )


def validate_scene_map_nav_graph_topology(*, scene: HOVSGSceneAsset) -> None:
    if not scene.room_adjacency:
        return
    if scene.nav_graph_type != "global_room_graph":
        return

    scene_map_edges = {
        tuple(sorted((room_id, neighbor_room_id)))
        for room_id, neighbors in scene.room_adjacency.items()
        for neighbor_room_id in neighbors
        if neighbor_room_id in scene.rooms and neighbor_room_id != room_id
    }

    nav_graph_edges: set[tuple[str, str]] = set()
    room_id_by_name = {
        str(room.name): room_id
        for room_id, room in scene.rooms.items()
        if isinstance(room.name, str) and room.name
    }
    for left_node, right_node in scene.nav_graph.edges():
        left_attrs = scene.nav_graph.nodes[left_node]
        right_attrs = scene.nav_graph.nodes[right_node]
        left_room_id = resolve_room_id_from_nav_node_attrs(left_attrs, room_id_by_name)
        right_room_id = resolve_room_id_from_nav_node_attrs(right_attrs, room_id_by_name)
        if left_room_id is None or right_room_id is None or left_room_id == right_room_id:
            continue
        nav_graph_edges.add(tuple(sorted((left_room_id, right_room_id))))

    if scene_map_edges == nav_graph_edges:
        return

    missing_in_nav = scene_map_edges - nav_graph_edges
    unexpected_in_nav = nav_graph_edges - scene_map_edges
    details: list[str] = []
    if missing_in_nav:
        details.append(
            "missing_in_nav_graph=" + ", ".join(format_room_edge(scene, edge) for edge in sorted(missing_in_nav))
        )
    if unexpected_in_nav:
        details.append(
            "unexpected_in_nav_graph="
            + ", ".join(format_room_edge(scene, edge) for edge in sorted(unexpected_in_nav))
        )
    raise ValueError(f"Scene map / nav graph topology mismatch for scene '{scene.scene_id}': " + "; ".join(details))


def resolve_room_id_from_nav_node_attrs(attrs: dict[str, Any], room_id_by_name: dict[str, str]) -> str | None:
    room_id = attrs.get("room_id")
    if isinstance(room_id, (str, int)):
        return str(room_id)
    room_name = attrs.get("room_name")
    if isinstance(room_name, str) and room_name.strip():
        return room_id_by_name.get(room_name.strip())
    return None


def format_room_edge(scene: HOVSGSceneAsset, edge: tuple[str, str]) -> str:
    left_room = scene.rooms.get(edge[0])
    right_room = scene.rooms.get(edge[1])
    left_name = left_room.name if left_room is not None and isinstance(left_room.name, str) else edge[0]
    right_name = right_room.name if right_room is not None and isinstance(right_room.name, str) else edge[1]
    return f"{left_name}<->{right_name}"


def load_objects(path: Path) -> dict[str, HOVSGObjectAsset]:
    objects: dict[str, HOVSGObjectAsset] = {}
    for json_file in sorted(path.glob("*.json")):
        metadata = read_json(json_file)
        object_id = str(metadata["object_id"])
        objects[object_id] = HOVSGObjectAsset(
            object_id=object_id,
            room_id=str(metadata["room_id"]),
            name=metadata.get("name"),
            centroid=position_from_metadata(metadata),
            vertices=[point for point in metadata.get("vertices", []) if isinstance(point, list) and len(point) >= 3],
            navigation_role=(
                str(metadata["navigation_role"])
                if metadata.get("navigation_role")
                else navigation_role_from_category(
                    metadata.get("category") or metadata.get("name")
                )
            ),
        )
    return objects


def load_nav_graph(
    path: Path,
    *,
    metadata: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> HOVSGNavGraphAsset:
    metadata = dict(metadata or {})
    config = dict(config or {})
    graph_entries = nav_graph_entries(path=path, metadata=metadata)
    preferred_type = preferred_nav_graph_type(metadata=metadata, config=config)

    ordered_entries: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()

    def append_entry(entry: dict[str, Any]) -> None:
        graph_path = entry["path"]
        if graph_path in seen_paths:
            return
        seen_paths.add(graph_path)
        ordered_entries.append(entry)

    if preferred_type is not None:
        for entry in graph_entries:
            if entry["type"] == preferred_type:
                append_entry(entry)
    for entry in graph_entries:
        append_entry(entry)

    for entry in ordered_entries:
        json_file = entry["path"]
        if not json_file.exists():
            continue
        data = read_json(json_file)
        graph = nx.Graph()
        for node in data.get("nodes", []):
            node_id = normalize_node_id(node.get("id"))
            attrs = {key: value for key, value in node.items() if key != "id"}
            graph.add_node(node_id, **attrs)
        for edge in data.get("links", []):
            source = normalize_node_id(edge.get("source"))
            target = normalize_node_id(edge.get("target"))
            attrs = {key: value for key, value in edge.items() if key not in {"source", "target"}}
            graph.add_edge(source, target, **attrs)
        return HOVSGNavGraphAsset(
            graph=graph,
            graph_type=str(entry["type"]),
            graph_path=json_file,
        )
    return HOVSGNavGraphAsset(
        graph=nx.Graph(),
        graph_type=preferred_type or "unknown",
        graph_path=None,
    )


def nav_graph_entries(path: Path, *, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    configured_entries = metadata.get("nav_graphs")
    if isinstance(configured_entries, list):
        for item in configured_entries:
            if not isinstance(item, dict):
                continue
            filename = item.get("filename")
            if not isinstance(filename, str) or not filename.strip():
                continue
            graph_type = normalize_nav_graph_type(item.get("type"))
            entries.append(
                {
                    "type": graph_type,
                    "path": path / filename.strip(),
                }
            )
    if entries:
        return entries

    legacy_candidates = [
        ("global_room_graph", path / "global_nav_graph_graph.json"),
        ("voronoi_graph", path / "sparse_voronoi_graph.json"),
    ]
    for graph_type, graph_path in legacy_candidates:
        entries.append({"type": graph_type, "path": graph_path})
    for graph_path in sorted(path.glob("*_graph.json")):
        entries.append(
            {
                "type": infer_nav_graph_type_from_filename(graph_path.name),
                "path": graph_path,
            }
        )
    return entries


def preferred_nav_graph_type(metadata: dict[str, Any], config: dict[str, Any]) -> str | None:
    for candidate in (
        config.get("nav_graph_type"),
        config.get("preferred_nav_graph_type"),
        metadata.get("default_nav_graph_type"),
        metadata.get("nav_graph_type"),
    ):
        normalized = normalize_nav_graph_type(candidate)
        if normalized is not None:
            return normalized
    return None


def normalize_nav_graph_type(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    aliases = {
        "global_nav_graph": "global_room_graph",
        "global_room_graph": "global_room_graph",
        "room_graph": "global_room_graph",
        "voronoi": "voronoi_graph",
        "sparse_voronoi_graph": "voronoi_graph",
        "voronoi_graph": "voronoi_graph",
    }
    return aliases.get(normalized, normalized)


def infer_nav_graph_type_from_filename(filename: str) -> str:
    normalized = filename.strip().lower()
    if normalized == "global_nav_graph_graph.json":
        return "global_room_graph"
    if normalized == "sparse_voronoi_graph.json":
        return "voronoi_graph"
    return Path(filename).stem


def normalize_node_id(node_id: Any) -> Any:
    if isinstance(node_id, list):
        return tuple(node_id)
    return node_id


def centroid_from_vertices(vertices: Any) -> dict[str, float] | None:
    if not isinstance(vertices, list) or not vertices:
        return None
    points = [point for point in vertices if isinstance(point, list) and len(point) >= 3]
    if not points:
        return None
    count = float(len(points))
    return {
        "x": sum(float(point[0]) for point in points) / count,
        "y": sum(float(point[1]) for point in points) / count,
        "z": sum(float(point[2]) for point in points) / count,
    }


def position_from_metadata(metadata: dict[str, Any]) -> dict[str, float] | None:
    position = metadata.get("position")
    if isinstance(position, dict):
        x_coord = to_float(position.get("x"))
        y_coord = to_float(position.get("y"))
        z_coord = to_float(position.get("z"))
        if x_coord is not None and y_coord is not None and z_coord is not None:
            return {"x": x_coord, "y": y_coord, "z": z_coord}
    return centroid_from_vertices(metadata.get("vertices"))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "centroid_from_vertices",
    "format_room_edge",
    "infer_nav_graph_type_from_filename",
    "load_floors",
    "load_metadata",
    "load_nav_graph",
    "load_objects",
    "load_room_adjacency",
    "load_rooms",
    "load_scene_map_payload",
    "nav_graph_entries",
    "normalize_nav_graph_type",
    "normalize_node_id",
    "position_from_metadata",
    "preferred_nav_graph_type",
    "read_json",
    "resolve_graph_path",
    "resolve_room_id_from_nav_node_attrs",
    "resolve_scene_map_source",
    "resolve_vertical_axis",
    "to_float",
    "validate_scene_graph_consistency",
    "validate_scene_map_graph_path",
    "validate_scene_map_nav_graph_topology",
]
