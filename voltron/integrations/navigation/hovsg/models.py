"""Shared HOV-SG asset models for navigation integrations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx


@dataclass
class HOVSGFloorAsset:
    floor_id: str
    name: str | None
    room_ids: list[str]
    floor_zero_level: float | None
    centroid: dict[str, float] | None


@dataclass
class HOVSGRoomAsset:
    room_id: str
    floor_id: str
    name: str | None
    object_ids: list[str]
    centroid: dict[str, float] | None
    vertices: list[list[float]]
    connected_room_ids: list[str]


@dataclass
class HOVSGObjectAsset:
    object_id: str
    room_id: str
    name: str | None
    centroid: dict[str, float] | None
    vertices: list[list[float]]
    navigation_role: str | None = None


@dataclass
class HOVSGSceneAsset:
    scene_id: str
    graph_path: Path
    floors: dict[str, HOVSGFloorAsset]
    rooms: dict[str, HOVSGRoomAsset]
    objects: dict[str, HOVSGObjectAsset]
    nav_graph: nx.Graph
    nav_graph_type: str
    nav_graph_asset_path: Path | None
    scene_map_source: str | None = None
    vertical_axis: str = "y"
    room_adjacency: dict[str, set[str]] | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class HOVSGNavGraphAsset:
    graph: nx.Graph
    graph_type: str
    graph_path: Path | None


@dataclass(frozen=True)
class HOVSGRoomLocalization:
    room: HOVSGRoomAsset
    contains: bool
    distance_to_boundary: float
    area: float | None
    centroid_distance_sq: float


__all__ = [
    "HOVSGFloorAsset",
    "HOVSGNavGraphAsset",
    "HOVSGObjectAsset",
    "HOVSGRoomAsset",
    "HOVSGRoomLocalization",
    "HOVSGSceneAsset",
]
