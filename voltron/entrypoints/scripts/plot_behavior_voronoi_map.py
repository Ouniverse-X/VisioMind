#!/usr/bin/env python3
"""Visualize a BEHAVIOR sparse Voronoi graph on top of a layout map."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from PIL import Image

DEFAULT_LAYOUT_PIXEL_RESOLUTION = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-map", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-room", type=str, default=None)
    parser.add_argument("--goal-room", type=str, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def world_to_pixel(
    x: float,
    y: float,
    *,
    image_size: tuple[int, int],
    pixel_resolution: float = DEFAULT_LAYOUT_PIXEL_RESOLUTION,
) -> tuple[float, float]:
    width_px, height_px = image_size
    center_x = width_px / 2.0
    center_y = height_px / 2.0
    px = center_x + (x / pixel_resolution)
    py = center_y + (y / pixel_resolution)
    return px, py


def load_room_polygons(graph_root: Path) -> dict[str, np.ndarray]:
    room_dir = graph_root / "rooms"
    polygons: dict[str, np.ndarray] = {}
    for room_file in sorted(room_dir.glob("*.json")):
        room = load_json(room_file)
        polygons[str(room["name"])] = np.asarray([[float(v[0]), float(v[1])] for v in room["vertices"]], dtype=float)
    return polygons


def shortest_room_path(adjacency: dict[str, list[str]], start: str, goal: str) -> list[str]:
    if start == goal:
        return [start]
    queue: deque[list[str]] = deque([[start]])
    visited = {start}
    while queue:
        path = queue.popleft()
        current = path[-1]
        for neighbor in adjacency.get(current, []):
            if neighbor in visited:
                continue
            next_path = path + [neighbor]
            if neighbor == goal:
                return next_path
            visited.add(neighbor)
            queue.append(next_path)
    return []


def main() -> None:
    args = parse_args()

    scene_map = load_json(args.scene_map)
    voronoi_graph = load_json(args.graph_root / "nav_graph" / "sparse_voronoi_graph.json")
    image = np.asarray(Image.open(args.layout).convert("L"))
    image_size = (image.shape[1], image.shape[0])
    room_polygons = load_room_polygons(args.graph_root)

    adjacency = {
        str(room): [str(item) for item in neighbors]
        for room, neighbors in (scene_map.get("topology", {}).get("adjacency", {}) or {}).items()
    }
    highlighted_rooms: set[str] = set()
    highlighted_route: list[str] = []
    if args.start_room and args.goal_room:
        highlighted_route = shortest_room_path(adjacency, args.start_room, args.goal_room)
        highlighted_rooms = set(highlighted_route)

    node_pos_by_id = {
        tuple(node["id"]): world_to_pixel(
            float(node["pos"][0]),
            float(node["pos"][1]),
            image_size=image_size,
        )
        for node in voronoi_graph.get("nodes", [])
    }

    base_segments: list[list[tuple[float, float]]] = []
    highlighted_segments: list[list[tuple[float, float]]] = []
    highlighted_nodes_x: list[float] = []
    highlighted_nodes_y: list[float] = []
    node_x: list[float] = []
    node_y: list[float] = []

    for node in voronoi_graph.get("nodes", []):
        node_id = tuple(node["id"])
        px, py = node_pos_by_id[node_id]
        node_x.append(px)
        node_y.append(py)
        if str(node.get("room_name")) in highlighted_rooms:
            highlighted_nodes_x.append(px)
            highlighted_nodes_y.append(py)

    for link in voronoi_graph.get("links", []):
        polyline = [
            world_to_pixel(float(point[0]), float(point[1]), image_size=image_size)
            for point in (link.get("polyline") or [])
        ]
        if len(polyline) < 2:
            continue
        source_room = None
        target_room = None
        source = tuple(link.get("source", []))
        target = tuple(link.get("target", []))
        for node in voronoi_graph.get("nodes", []):
            node_id = tuple(node["id"])
            if node_id == source:
                source_room = str(node.get("room_name"))
            elif node_id == target:
                target_room = str(node.get("room_name"))
        if source_room in highlighted_rooms or target_room in highlighted_rooms:
            highlighted_segments.append(polyline)
        else:
            base_segments.append(polyline)

    fig, ax = plt.subplots(figsize=(12, 12), dpi=180)
    ax.imshow(image, cmap="gray", origin="upper")

    for room_name, vertices_w in room_polygons.items():
        vertices_px = np.asarray(
            [world_to_pixel(x, y, image_size=image_size) for x, y in vertices_w],
            dtype=float,
        )
        is_highlight = room_name in highlighted_rooms
        ax.add_patch(
            Polygon(
                vertices_px,
                closed=True,
                fill=False,
                linewidth=1.8 if is_highlight else 0.7,
                edgecolor="#ffb000" if is_highlight else "#7a7a7a",
                alpha=0.95 if is_highlight else 0.35,
            )
        )
        if is_highlight:
            center = vertices_px.mean(axis=0)
            ax.text(center[0], center[1], room_name, fontsize=8, color="#ffb000", ha="center", va="center")

    if base_segments:
        ax.add_collection(LineCollection(base_segments, colors="#1f77ff", linewidths=0.4, alpha=0.28))
    if highlighted_segments:
        ax.add_collection(LineCollection(highlighted_segments, colors="#00d084", linewidths=0.9, alpha=0.9))

    ax.scatter(node_x, node_y, s=1.0, c="#4fc3ff", alpha=0.35, linewidths=0)
    if highlighted_nodes_x:
        ax.scatter(highlighted_nodes_x, highlighted_nodes_y, s=3.0, c="#00d084", alpha=0.85, linewidths=0)

    title = "Sparse Voronoi Graph Overlay"
    if highlighted_route:
        title += "\nRoute rooms: " + " -> ".join(highlighted_route)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
