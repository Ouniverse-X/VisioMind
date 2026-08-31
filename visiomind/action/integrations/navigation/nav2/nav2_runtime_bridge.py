from __future__ import annotations

import atexit
from collections import deque
import json
import math
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .navigator import NAV2_VERSION_PROFILES, Nav2VersionProfile

DEFAULT_NAV2_MAP_RESOLUTION = 0.1
DEFAULT_PORTAL_ANALYSIS_MAP_RESOLUTION = 0.05
R1PRO_NAV_FOOTPRINT: tuple[tuple[float, float], ...] = (
    (0.24, 0.34),
    (0.24, -0.34),
    (-0.40, -0.34),
    (-0.40, 0.34),
)
R1PRO_NAV_FOOTPRINT_PADDING_M = 0.02
R1PRO_NAV_CLEARANCE_RADIUS_M = 0.35
DEFAULT_NAV2_TRAV_MAP_FILENAME = "floor_trav_no_obj_0.png"


def nav_footprint_string(
    footprint: tuple[tuple[float, float], ...] = R1PRO_NAV_FOOTPRINT,
) -> str:
    return json.dumps([[float(x_coord), float(y_coord)] for x_coord, y_coord in footprint])


def _prepend_path(value: str, existing: str | None) -> str:
    existing = (existing or "").strip()
    if not existing:
        return value
    parts = existing.split(":")
    if value in parts:
        return existing
    return f"{value}:{existing}"


def _build_overlay_env() -> dict[str, str]:
    env = dict(os.environ)
    overlay_prefix = env.get("VISIOMIND_ACTION_NAV2_PREFIX", "").strip()
    overlay_python = env.get("VISIOMIND_ACTION_NAV2_PYTHONPATH", "").strip()
    if not overlay_prefix:
        return env

    env["AMENT_PREFIX_PATH"] = _prepend_path(overlay_prefix, env.get("AMENT_PREFIX_PATH"))
    env["CMAKE_PREFIX_PATH"] = _prepend_path(overlay_prefix, env.get("CMAKE_PREFIX_PATH"))
    env["COLCON_PREFIX_PATH"] = _prepend_path(overlay_prefix, env.get("COLCON_PREFIX_PATH"))
    env["LD_LIBRARY_PATH"] = _prepend_path(f"{overlay_prefix}/lib", env.get("LD_LIBRARY_PATH"))
    if overlay_python:
        env["PYTHONPATH"] = _prepend_path(overlay_python, env.get("PYTHONPATH"))
    return env


def _resolve_trav_map_path(*, layout_dir: Path, trav_map_filename: str | None = None) -> Path:
    candidates: list[str] = []
    if isinstance(trav_map_filename, str) and trav_map_filename.strip():
        candidates.append(trav_map_filename.strip())
    candidates.extend((DEFAULT_NAV2_TRAV_MAP_FILENAME, "floor_trav_0.png"))

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        path = layout_dir / candidate
        if path.is_file():
            return path

    requested = candidates[0] if candidates else DEFAULT_NAV2_TRAV_MAP_FILENAME
    raise FileNotFoundError(f"Traversability map not found: {layout_dir / requested}")


def _scene_layout_root(scene_id: str) -> Path:
    candidates: list[Path] = []
    data_root = os.environ.get("OMNIGIBSON_DATA_PATH", "").strip()
    scene_id_candidates = [scene_id]
    scene_parts = [part for part in str(scene_id).split("_") if part]
    for end in range(len(scene_parts) - 1, 2, -1):
        candidate_scene_id = "_".join(scene_parts[:end])
        if candidate_scene_id and candidate_scene_id not in scene_id_candidates:
            scene_id_candidates.append(candidate_scene_id)
    if data_root:
        base = Path(data_root).expanduser().resolve()
        for candidate_scene_id in scene_id_candidates:
            candidates.extend(
                [
                    base / "scenes" / candidate_scene_id / "layout",
                    base / candidate_scene_id / "layout",
                    base / "behavior-1k-assets" / "scenes" / candidate_scene_id / "layout",
                    base
                    / "2025-challenge-task-instances"
                    / "scenes"
                    / candidate_scene_id
                    / "layout",
                ]
            )

    workspace_root = Path(__file__).resolve().parents[4]
    fallback_base = workspace_root / "BEHAVIOR-1K" / "datasets"
    for candidate_scene_id in scene_id_candidates:
        candidates.extend(
            [
                fallback_base / "scenes" / candidate_scene_id / "layout",
                fallback_base / candidate_scene_id / "layout",
                fallback_base / "behavior-1k-assets" / "scenes" / candidate_scene_id / "layout",
                fallback_base
                / "2025-challenge-task-instances"
                / "scenes"
                / candidate_scene_id
                / "layout",
            ]
        )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not locate layout assets for scene '{scene_id}'")


def load_scene_map_spec(
    *,
    scene_id: str,
    map_resolution: float = DEFAULT_NAV2_MAP_RESOLUTION,
    trav_map_filename: str | None = None,
    obstacle_inflation_radius_m: float = 0.0,
) -> dict[str, Any]:
    layout_dir = _scene_layout_root(scene_id)
    trav_map_path = _resolve_trav_map_path(
        layout_dir=layout_dir, trav_map_filename=trav_map_filename
    )

    img = Image.open(trav_map_path).convert("L")
    img_np = np.asarray(img, dtype=np.uint8)
    if img_np.ndim != 2 or img_np.shape[0] != img_np.shape[1]:
        raise RuntimeError(f"Expected square traversability map, got shape={img_np.shape}")

    map_size = int(img_np.shape[0] * 0.01 / float(map_resolution))
    if map_size <= 0:
        raise RuntimeError(f"Invalid Nav2 map size derived from {trav_map_path}")

    if map_size != img_np.shape[0]:
        resized = Image.fromarray(img_np).resize((map_size, map_size), resample=Image.NEAREST)
        img_np = np.asarray(resized, dtype=np.uint8)

    free_mask = img_np == 255
    obstacle_mask = ~free_mask
    obstacle_mask = _inflate_obstacle_mask(
        obstacle_mask,
        inflation_radius_m=obstacle_inflation_radius_m,
        map_resolution=map_resolution,
    )
    img_np = np.where(obstacle_mask, 100, 0).astype(np.int8)
    origin_xy = -(map_size / 2.0) * float(map_resolution)
    return {
        "scene_id": scene_id,
        "width": map_size,
        "height": map_size,
        "resolution": float(map_resolution),
        "origin": {"x": float(origin_xy), "y": float(origin_xy), "yaw": 0.0},
        "data": img_np.reshape(-1).astype(int).tolist(),
        "source": str(trav_map_path),
    }


def _inflate_obstacle_mask(
    obstacle_mask: np.ndarray,
    *,
    inflation_radius_m: float,
    map_resolution: float,
) -> np.ndarray:
    radius_m = max(0.0, float(inflation_radius_m))
    resolution = float(map_resolution)
    if radius_m <= 0.0 or resolution <= 0.0 or not bool(np.any(obstacle_mask)):
        return obstacle_mask

    cell_radius = max(1, int(math.ceil(radius_m / resolution)))
    offsets = np.arange(-cell_radius, cell_radius + 1, dtype=np.float64)
    offset_y, offset_x = np.meshgrid(offsets, offsets, indexing="ij")
    disk = np.hypot(offset_x, offset_y) * resolution <= radius_m + 1e-9
    padded = np.pad(obstacle_mask, cell_radius, mode="constant", constant_values=False)
    window_size = cell_radius * 2 + 1
    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        (window_size, window_size),
    )
    return np.any(windows[..., disk], axis=-1)


def load_scene_traversability_grid(
    *,
    scene_id: str,
    map_resolution: float = DEFAULT_PORTAL_ANALYSIS_MAP_RESOLUTION,
    trav_map_filename: str | None = None,
    obstacle_inflation_radius_m: float = 0.0,
) -> dict[str, Any]:
    map_spec = load_scene_map_spec(
        scene_id=scene_id,
        map_resolution=map_resolution,
        trav_map_filename=trav_map_filename,
        obstacle_inflation_radius_m=obstacle_inflation_radius_m,
    )
    width = int(map_spec["width"])
    height = int(map_spec["height"])
    data = np.asarray(map_spec["data"], dtype=np.int16).reshape(height, width)
    return {
        **map_spec,
        "grid": data,
        "free_mask": data == 0,
    }


def clear_exported_door_artifacts_from_map_spec(
    map_spec: dict[str, Any],
    *,
    scene_id: str,
    map_resolution: float,
    obstacle_inflation_radius_m: float = 0.0,
    doorless_trav_map_filename: str = "floor_trav_no_door_0.png",
) -> dict[str, Any]:
    try:
        doorless_spec = load_scene_map_spec(
            scene_id=scene_id,
            map_resolution=map_resolution,
            trav_map_filename=doorless_trav_map_filename,
            obstacle_inflation_radius_m=obstacle_inflation_radius_m,
        )
        width = int(map_spec["width"])
        height = int(map_spec["height"])
        if width != int(doorless_spec["width"]) or height != int(doorless_spec["height"]):
            return map_spec
        if map_spec.get("origin") != doorless_spec.get("origin"):
            return map_spec
        grid = np.asarray(map_spec["data"], dtype=np.int16).reshape(height, width).copy()
        doorless_grid = np.asarray(doorless_spec["data"], dtype=np.int16).reshape(height, width)
    except (KeyError, TypeError, ValueError, OSError, RuntimeError):
        return map_spec

    artifact_mask = (grid != 0) & (doorless_grid == 0)
    if not bool(np.any(artifact_mask)):
        return map_spec
    grid[artifact_mask] = 0
    cleared = {
        **map_spec,
        "data": grid.reshape(-1).astype(int).tolist(),
        "cleared_exported_door_artifact_cells": int(np.count_nonzero(artifact_mask)),
        "doorless_reference": str(doorless_spec.get("source") or doorless_trav_map_filename),
    }
    if "grid" in map_spec:
        cleared["grid"] = grid
        cleared["free_mask"] = grid == 0
    return cleared


def clear_exported_object_artifacts_from_map_spec(
    map_spec: dict[str, Any],
    *,
    scene_id: str,
    map_resolution: float,
    regions: list[dict[str, Any]],
    obstacle_inflation_radius_m: float = 0.0,
    objectless_trav_map_filename: str = "floor_trav_no_obj_0.png",
) -> dict[str, Any]:
    if not regions:
        return map_spec
    try:
        objectless_spec = load_scene_map_spec(
            scene_id=scene_id,
            map_resolution=map_resolution,
            trav_map_filename=objectless_trav_map_filename,
            obstacle_inflation_radius_m=obstacle_inflation_radius_m,
        )
        width = int(map_spec["width"])
        height = int(map_spec["height"])
        if width != int(objectless_spec["width"]) or height != int(objectless_spec["height"]):
            return map_spec
        if map_spec.get("origin") != objectless_spec.get("origin"):
            return map_spec
        grid = np.asarray(map_spec["data"], dtype=np.int16).reshape(height, width).copy()
        objectless_grid = np.asarray(objectless_spec["data"], dtype=np.int16).reshape(height, width)
        resolution = float(map_spec["resolution"])
        origin = map_spec["origin"]
        origin_x = float(origin["x"])
        origin_y = float(origin["y"])
    except (KeyError, TypeError, ValueError, OSError, RuntimeError):
        return map_spec

    artifact_mask = (grid != 0) & (objectless_grid == 0)
    if not bool(np.any(artifact_mask)):
        return map_spec
    region_mask = np.zeros_like(artifact_mask, dtype=bool)
    clear_polygon_count = 0
    dirty_bounds: tuple[float, float, float, float] | None = None
    for region in regions:
        for polygon in _obstacle_polygons(region, default_half_extent_m=0.5):
            patch = _polygon_cell_patch(
                polygon,
                width=width,
                height=height,
                resolution=resolution,
                origin_x=origin_x,
                origin_y=origin_y,
            )
            if patch is None:
                continue
            row_slice, col_slice, polygon_mask = patch
            region_mask[row_slice, col_slice] |= polygon_mask
            dirty_bounds = _merge_world_bounds(dirty_bounds, _polygon_bounds(polygon))
            clear_polygon_count += 1
    clear_mask = artifact_mask & region_mask
    if not bool(np.any(clear_mask)):
        return map_spec
    grid[clear_mask] = 0
    cleared = {
        **map_spec,
        "data": grid.reshape(-1).astype(int).tolist(),
        "cleared_exported_object_artifact_cells": int(np.count_nonzero(clear_mask)),
        "objectless_reference": str(objectless_spec.get("source") or objectless_trav_map_filename),
    }
    cleared["dynamic_map_update"] = _dynamic_update_metadata(
        map_spec,
        dirty_bounds=dirty_bounds,
        clear_polygon_count=clear_polygon_count,
    )
    if "grid" in map_spec:
        cleared["grid"] = grid
        cleared["free_mask"] = grid == 0
    return cleared


def stamp_obstacles_into_map_spec(
    map_spec: dict[str, Any],
    obstacles: list[dict[str, Any]],
    *,
    default_half_extent_m: float = 0.5,
) -> dict[str, Any]:
    if not obstacles:
        return map_spec
    try:
        width = int(map_spec["width"])
        height = int(map_spec["height"])
        resolution = float(map_spec["resolution"])
        origin = map_spec["origin"]
        origin_x = float(origin["x"])
        origin_y = float(origin["y"])
    except (KeyError, TypeError, ValueError):
        return map_spec

    grid = np.asarray(map_spec["data"], dtype=np.int16).reshape(height, width).copy()
    stamped_polygon_count = 0
    dirty_bounds: tuple[float, float, float, float] | None = None
    for obstacle in obstacles:
        for polygon in _obstacle_polygons(
            obstacle,
            default_half_extent_m=default_half_extent_m,
        ):
            patch = _polygon_cell_patch(
                polygon,
                width=width,
                height=height,
                resolution=resolution,
                origin_x=origin_x,
                origin_y=origin_y,
            )
            if patch is None:
                continue
            row_slice, col_slice, polygon_mask = patch
            grid_patch = grid[row_slice, col_slice]
            grid_patch[polygon_mask] = 100
            dirty_bounds = _merge_world_bounds(dirty_bounds, _polygon_bounds(polygon))
            stamped_polygon_count += 1
    if not stamped_polygon_count:
        return map_spec

    stamped = {
        **map_spec,
        "data": grid.reshape(-1).astype(int).tolist(),
        "stamped_obstacle_count": len(obstacles),
        "stamped_polygon_count": stamped_polygon_count,
    }
    stamped["dynamic_map_update"] = _dynamic_update_metadata(
        map_spec,
        dirty_bounds=dirty_bounds,
        stamp_polygon_count=stamped_polygon_count,
    )
    if "grid" in map_spec:
        stamped["grid"] = grid
        stamped["free_mask"] = grid == 0
    return stamped


def clear_regions_from_map_spec(
    map_spec: dict[str, Any],
    regions: list[dict[str, Any]],
    *,
    default_half_extent_m: float = 0.5,
) -> dict[str, Any]:
    if not regions:
        return map_spec
    try:
        width = int(map_spec["width"])
        height = int(map_spec["height"])
        resolution = float(map_spec["resolution"])
        origin = map_spec["origin"]
        origin_x = float(origin["x"])
        origin_y = float(origin["y"])
    except (KeyError, TypeError, ValueError):
        return map_spec

    grid = np.asarray(map_spec["data"], dtype=np.int16).reshape(height, width).copy()
    cleared_polygon_count = 0
    dirty_bounds: tuple[float, float, float, float] | None = None
    for region in regions:
        for polygon in _obstacle_polygons(
            region,
            default_half_extent_m=default_half_extent_m,
        ):
            patch = _polygon_cell_patch(
                polygon,
                width=width,
                height=height,
                resolution=resolution,
                origin_x=origin_x,
                origin_y=origin_y,
            )
            if patch is None:
                continue
            row_slice, col_slice, polygon_mask = patch
            grid_patch = grid[row_slice, col_slice]
            grid_patch[polygon_mask] = 0
            dirty_bounds = _merge_world_bounds(dirty_bounds, _polygon_bounds(polygon))
            cleared_polygon_count += 1
    if not cleared_polygon_count:
        return map_spec

    cleared = {
        **map_spec,
        "data": grid.reshape(-1).astype(int).tolist(),
        "cleared_region_count": len(regions),
        "cleared_polygon_count": cleared_polygon_count,
    }
    cleared["dynamic_map_update"] = _dynamic_update_metadata(
        map_spec,
        dirty_bounds=dirty_bounds,
        clear_polygon_count=cleared_polygon_count,
    )
    if "grid" in map_spec:
        cleared["grid"] = grid
        cleared["free_mask"] = grid == 0
    return cleared


def _obstacle_bounds(
    obstacle: dict[str, Any],
    *,
    default_half_extent_m: float,
) -> tuple[float, float, float, float] | None:
    polygons = _explicit_obstacle_polygons(obstacle)
    polygon_bounds = [_polygon_bounds(polygon) for polygon in polygons]
    polygon_bounds = [bounds for bounds in polygon_bounds if bounds is not None]
    if polygon_bounds:
        merged = polygon_bounds[0]
        for bounds in polygon_bounds[1:]:
            merged = _merge_world_bounds(merged, bounds)
        return merged

    return _legacy_obstacle_bounds(
        obstacle,
        default_half_extent_m=default_half_extent_m,
    )


def _legacy_obstacle_bounds(
    obstacle: dict[str, Any],
    *,
    default_half_extent_m: float,
) -> tuple[float, float, float, float] | None:
    corner_min = obstacle.get("min")
    corner_max = obstacle.get("max")
    if isinstance(corner_min, dict) and isinstance(corner_max, dict):
        try:
            return (
                float(corner_min["x"]),
                float(corner_min["y"]),
                float(corner_max["x"]),
                float(corner_max["y"]),
            )
        except (KeyError, TypeError, ValueError):
            return None
    position = obstacle.get("position")
    if isinstance(position, dict):
        try:
            half_extent = float(obstacle.get("half_extent_m", default_half_extent_m))
            x_coord = float(position["x"])
            y_coord = float(position["y"])
        except (KeyError, TypeError, ValueError):
            return None
        return (
            x_coord - half_extent,
            y_coord - half_extent,
            x_coord + half_extent,
            y_coord + half_extent,
        )
    return None


def _obstacle_polygons(
    obstacle: dict[str, Any],
    *,
    default_half_extent_m: float,
) -> list[list[tuple[float, float]]]:
    polygons = _explicit_obstacle_polygons(obstacle)
    if polygons:
        return polygons
    bounds = _legacy_obstacle_bounds(
        obstacle,
        default_half_extent_m=default_half_extent_m,
    )
    if bounds is None:
        return []
    min_x, min_y, max_x, max_y = bounds
    return [
        [
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
        ]
    ]


def _explicit_obstacle_polygons(
    obstacle: dict[str, Any],
) -> list[list[tuple[float, float]]]:
    raw_polygons = obstacle.get("polygons")
    if not isinstance(raw_polygons, list):
        raw_polygon = obstacle.get("polygon")
        raw_polygons = [raw_polygon] if isinstance(raw_polygon, list) else []
    polygons: list[list[tuple[float, float]]] = []
    for raw_polygon in raw_polygons:
        if not isinstance(raw_polygon, list):
            continue
        polygon: list[tuple[float, float]] = []
        for point in raw_polygon:
            if isinstance(point, dict):
                try:
                    polygon.append((float(point["x"]), float(point["y"])))
                except (KeyError, TypeError, ValueError):
                    continue
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                try:
                    polygon.append((float(point[0]), float(point[1])))
                except (TypeError, ValueError):
                    continue
        if len(polygon) >= 3:
            polygons.append(polygon)
    return polygons


def _polygon_bounds(
    polygon: list[tuple[float, float]],
) -> tuple[float, float, float, float] | None:
    if len(polygon) < 3:
        return None
    return (
        min(point[0] for point in polygon),
        min(point[1] for point in polygon),
        max(point[0] for point in polygon),
        max(point[1] for point in polygon),
    )


def _polygon_cell_patch(
    polygon: list[tuple[float, float]],
    *,
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
) -> tuple[slice, slice, np.ndarray] | None:
    bounds = _polygon_bounds(polygon)
    if bounds is None or resolution <= 0.0:
        return None
    min_x, min_y, max_x, max_y = bounds
    col_min = max(0, int(math.floor((min_x - origin_x) / resolution)))
    col_max = min(
        width - 1,
        max(
            col_min,
            int(math.ceil((max_x - origin_x) / resolution - 1e-9)) - 1,
        ),
    )
    row_min = max(0, int(math.floor((min_y - origin_y) / resolution)))
    row_max = min(
        height - 1,
        max(
            row_min,
            int(math.ceil((max_y - origin_y) / resolution - 1e-9)) - 1,
        ),
    )
    if row_min > row_max or col_min > col_max:
        return None

    image = Image.new("1", (col_max - col_min + 1, row_max - row_min + 1), 0)
    draw = ImageDraw.Draw(image)
    pixel_polygon = [
        (
            (point[0] - origin_x) / resolution - col_min - 0.5,
            (point[1] - origin_y) / resolution - row_min - 0.5,
        )
        for point in polygon
    ]
    draw.polygon(pixel_polygon, fill=1)
    mask = np.asarray(image, dtype=bool)
    if not bool(np.any(mask)):
        return None
    return (
        slice(row_min, row_max + 1),
        slice(col_min, col_max + 1),
        mask,
    )


def _merge_world_bounds(
    first: tuple[float, float, float, float] | None,
    second: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    if first is None:
        return second
    if second is None:
        return first
    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    )


def _dynamic_update_metadata(
    map_spec: dict[str, Any],
    *,
    dirty_bounds: tuple[float, float, float, float] | None,
    clear_polygon_count: int = 0,
    stamp_polygon_count: int = 0,
) -> dict[str, Any]:
    previous = map_spec.get("dynamic_map_update")
    previous = dict(previous) if isinstance(previous, dict) else {}
    previous_bounds = previous.get("dirty_bounds")
    normalized_previous_bounds = None
    if isinstance(previous_bounds, dict):
        try:
            normalized_previous_bounds = (
                float(previous_bounds["min_x"]),
                float(previous_bounds["min_y"]),
                float(previous_bounds["max_x"]),
                float(previous_bounds["max_y"]),
            )
        except (KeyError, TypeError, ValueError):
            normalized_previous_bounds = None
    merged_bounds = _merge_world_bounds(normalized_previous_bounds, dirty_bounds)
    metadata = {
        "clear_polygon_count": int(previous.get("clear_polygon_count", 0))
        + int(clear_polygon_count),
        "stamp_polygon_count": int(previous.get("stamp_polygon_count", 0))
        + int(stamp_polygon_count),
    }
    if merged_bounds is not None:
        metadata["dirty_bounds"] = {
            "min_x": merged_bounds[0],
            "min_y": merged_bounds[1],
            "max_x": merged_bounds[2],
            "max_y": merged_bounds[3],
        }
    return metadata


def occupancy_grid_update_from_specs(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any] | None:
    try:
        width = int(current["width"])
        height = int(current["height"])
        if width != int(previous["width"]) or height != int(previous["height"]):
            return None
        if current.get("origin") != previous.get("origin"):
            return None
        if float(current["resolution"]) != float(previous["resolution"]):
            return None
        previous_grid = np.asarray(previous["data"], dtype=np.int16).reshape(height, width)
        current_grid = np.asarray(current["data"], dtype=np.int16).reshape(height, width)
    except (KeyError, TypeError, ValueError):
        return None
    changed = previous_grid != current_grid
    if not bool(np.any(changed)):
        return {
            "x": 0,
            "y": 0,
            "width": 0,
            "height": 0,
            "data": [],
            "changed_cell_count": 0,
            "dirty_bounds": None,
        }
    rows, cols = np.nonzero(changed)
    row_min = int(rows.min())
    row_max = int(rows.max())
    col_min = int(cols.min())
    col_max = int(cols.max())
    patch = current_grid[row_min : row_max + 1, col_min : col_max + 1]
    resolution = float(current["resolution"])
    origin = current["origin"]
    origin_x = float(origin["x"])
    origin_y = float(origin["y"])
    return {
        "x": col_min,
        "y": row_min,
        "width": col_max - col_min + 1,
        "height": row_max - row_min + 1,
        "data": patch.reshape(-1).astype(int).tolist(),
        "changed_cell_count": int(np.count_nonzero(changed)),
        "dirty_bounds": {
            "min_x": origin_x + col_min * resolution,
            "min_y": origin_y + row_min * resolution,
            "max_x": origin_x + (col_max + 1) * resolution,
            "max_y": origin_y + (row_max + 1) * resolution,
        },
    }


def diagnose_empty_path(
    *,
    map_spec: dict[str, Any],
    start_xy: dict[str, float],
    goal_xy: dict[str, float],
) -> str:
    start_cell = world_to_map_cell(map_spec=map_spec, point_xy=start_xy)
    goal_cell = world_to_map_cell(map_spec=map_spec, point_xy=goal_xy)
    if start_cell is None:
        return "start_outside_map"
    if goal_cell is None:
        return "goal_outside_map"
    try:
        width = int(map_spec["width"])
        height = int(map_spec["height"])
        grid = np.asarray(map_spec["data"], dtype=np.int16).reshape(height, width)
    except (KeyError, TypeError, ValueError):
        return "map_unavailable"
    if grid[start_cell] != 0:
        return "start_blocked"
    if grid[goal_cell] != 0:
        return "goal_blocked"
    if start_cell == goal_cell:
        return "planner_error"

    queue = deque([start_cell])
    visited = {start_cell}
    while queue:
        row, col = queue.popleft()
        for next_row, next_col in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            cell = (next_row, next_col)
            if (
                next_row < 0
                or next_col < 0
                or next_row >= height
                or next_col >= width
                or cell in visited
                or grid[cell] != 0
            ):
                continue
            if cell == goal_cell:
                return "planner_error"
            visited.add(cell)
            queue.append(cell)
    return "map_disconnected"


def world_to_map_cell(
    *, map_spec: dict[str, Any], point_xy: dict[str, float]
) -> tuple[int, int] | None:
    try:
        resolution = float(map_spec["resolution"])
        origin = map_spec["origin"]
        origin_x = float(origin["x"])
        origin_y = float(origin["y"])
        width = int(map_spec["width"])
        height = int(map_spec["height"])
        x_coord = float(point_xy["x"])
        y_coord = float(point_xy["y"])
    except (KeyError, TypeError, ValueError):
        return None

    col = int(round((x_coord - origin_x) / resolution))
    row = int(round((y_coord - origin_y) / resolution))
    if row < 0 or col < 0 or row >= height or col >= width:
        return None
    return row, col


def map_cell_to_world(*, map_spec: dict[str, Any], row: int, col: int) -> dict[str, float]:
    resolution = float(map_spec["resolution"])
    origin = map_spec["origin"]
    return {
        "x": float(origin["x"]) + float(col) * resolution,
        "y": float(origin["y"]) + float(row) * resolution,
    }


def point_has_clearance(
    *,
    map_spec: dict[str, Any],
    point_xy: dict[str, float],
    clearance_radius_m: float,
) -> bool:
    cell = world_to_map_cell(map_spec=map_spec, point_xy=point_xy)
    if cell is None:
        return False

    free_mask = map_spec.get("free_mask")
    if not isinstance(free_mask, np.ndarray):
        return False

    row, col = cell
    radius_cells = max(
        0,
        int(math.ceil(max(0.0, float(clearance_radius_m)) / float(map_spec["resolution"]))),
    )
    row_min = max(0, row - radius_cells)
    row_max = min(free_mask.shape[0], row + radius_cells + 1)
    col_min = max(0, col - radius_cells)
    col_max = min(free_mask.shape[1], col + radius_cells + 1)
    patch = free_mask[row_min:row_max, col_min:col_max]
    if patch.size == 0:
        return False
    radius_m = max(0.0, float(clearance_radius_m))
    row_offsets = (np.arange(row_min, row_max, dtype=np.float64) - float(row)) * float(
        map_spec["resolution"]
    )
    col_offsets = (np.arange(col_min, col_max, dtype=np.float64) - float(col)) * float(
        map_spec["resolution"]
    )
    offset_y, offset_x = np.meshgrid(row_offsets, col_offsets, indexing="ij")
    disk = np.hypot(offset_x, offset_y) <= radius_m + 1e-9
    return bool(np.all(patch[disk]))


def segment_has_clearance(
    *,
    map_spec: dict[str, Any],
    start_xy: dict[str, float],
    end_xy: dict[str, float],
    clearance_radius_m: float,
    step_m: float | None = None,
) -> bool:
    start_x = float(start_xy["x"])
    start_y = float(start_xy["y"])
    end_x = float(end_xy["x"])
    end_y = float(end_xy["y"])
    distance = math.hypot(end_x - start_x, end_y - start_y)
    if distance <= 1e-6:
        return point_has_clearance(
            map_spec=map_spec,
            point_xy=start_xy,
            clearance_radius_m=clearance_radius_m,
        )

    step = (
        float(step_m)
        if isinstance(step_m, (int, float)) and step_m > 0.0
        else max(
            float(map_spec["resolution"]) * 0.5,
            0.025,
        )
    )
    sample_count = max(2, int(math.ceil(distance / step)) + 1)
    for sample_index in range(sample_count):
        ratio = sample_index / max(1, sample_count - 1)
        point = {
            "x": start_x + (end_x - start_x) * ratio,
            "y": start_y + (end_y - start_y) * ratio,
        }
        if not point_has_clearance(
            map_spec=map_spec,
            point_xy=point,
            clearance_radius_m=clearance_radius_m,
        ):
            return False
    return True


class PersistentNav2RuntimeBridgeClient:
    def __init__(
        self,
        *,
        profile: Nav2VersionProfile,
        frame_id: str = "map",
        action_name: str = "compute_path_to_pose",
        worker_script: str | Path | None = None,
    ) -> None:
        self.profile = profile
        self.frame_id = str(frame_id).strip() or "map"
        self.action_name = str(action_name).strip() or "compute_path_to_pose"
        default_worker = Path(__file__).resolve().parent / "runtime_worker.py"
        self.worker_script = (
            Path(worker_script).expanduser() if worker_script is not None else default_worker
        )
        self._process: subprocess.Popen[str] | None = None
        self._configured_scene_signature: tuple[str, str, str, str, str, str] | None = None
        self._configured_base_signature: tuple[str, str, str, str] | None = None
        self._configured_map_spec: dict[str, Any] | None = None
        self._configured_map_revision: int | None = None
        atexit.register(self.close)

    def close(self) -> None:
        if self._process is None:
            self._configured_scene_signature = None
            self._configured_base_signature = None
            self._configured_map_spec = None
            self._configured_map_revision = None
            return
        try:
            self._send_request({"cmd": "shutdown"}, tolerate_errors=True)
        except Exception:
            pass
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None
        self._configured_scene_signature = None
        self._configured_base_signature = None
        self._configured_map_spec = None
        self._configured_map_revision = None

    def ensure_scene(
        self,
        *,
        scene_id: str,
        map_resolution: float = DEFAULT_NAV2_MAP_RESOLUTION,
        trav_map_filename: str | None = None,
        obstacle_inflation_radius_m: float = 0.0,
        obstacles: list[dict[str, Any]] | None = None,
        clear_regions: list[dict[str, Any]] | None = None,
        clear_exported_door_artifacts: bool = False,
        clear_exported_object_artifacts: bool = False,
        object_clear_regions: list[dict[str, Any]] | None = None,
        obstacles_signature: str = "",
    ) -> dict[str, Any]:
        map_name = (
            str(trav_map_filename or DEFAULT_NAV2_TRAV_MAP_FILENAME).strip()
            or DEFAULT_NAV2_TRAV_MAP_FILENAME
        )
        inflation_radius = max(0.0, float(obstacle_inflation_radius_m))
        overlay_signature = json.dumps(
            {
                "obstacles": obstacles or [],
                "clear_regions": clear_regions or [],
                "clear_exported_door_artifacts": bool(clear_exported_door_artifacts),
                "clear_exported_object_artifacts": bool(clear_exported_object_artifacts),
                "object_clear_regions": object_clear_regions or [],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        scene_signature = (
            scene_id,
            map_name,
            f"{float(map_resolution):.6f}",
            f"{inflation_radius:.4f}",
            str(obstacles_signature or ""),
            overlay_signature,
        )
        base_signature = scene_signature[:4]
        if (
            self._configured_scene_signature == scene_signature
            and self._process is not None
            and self._process.poll() is None
        ):
            return {
                "configured_scene_id": scene_id,
                "trav_map_filename": map_name,
                "reused": True,
                "map_revision": self._configured_map_revision,
            }

        map_spec = load_scene_map_spec(
            scene_id=scene_id,
            map_resolution=map_resolution,
            trav_map_filename=trav_map_filename,
            obstacle_inflation_radius_m=inflation_radius,
        )
        if clear_exported_door_artifacts:
            map_spec = clear_exported_door_artifacts_from_map_spec(
                map_spec,
                scene_id=scene_id,
                map_resolution=map_resolution,
                obstacle_inflation_radius_m=inflation_radius,
            )
        if clear_exported_object_artifacts:
            map_spec = clear_exported_object_artifacts_from_map_spec(
                map_spec,
                scene_id=scene_id,
                map_resolution=map_resolution,
                regions=object_clear_regions or [],
                obstacle_inflation_radius_m=inflation_radius,
            )
        if clear_regions:
            map_spec = clear_regions_from_map_spec(map_spec, clear_regions)
        if obstacles:
            map_spec = stamp_obstacles_into_map_spec(map_spec, obstacles)
        map_spec = {
            **map_spec,
            "overlay_signature": str(obstacles_signature or ""),
        }
        running = self._process is not None and self._process.poll() is None
        if (
            running
            and self._configured_base_signature == base_signature
            and isinstance(self._configured_map_spec, dict)
        ):
            update = occupancy_grid_update_from_specs(self._configured_map_spec, map_spec)
            if update is not None and int(update.get("changed_cell_count", 0)) > 0:
                response = self._send_request(
                    {
                        "cmd": "update_map",
                        "scene_id": scene_id,
                        "overlay_signature": str(obstacles_signature or ""),
                        "update": update,
                    }
                )
                if isinstance(response, dict):
                    update_with_revision = {
                        **update,
                        "map_revision": response.get("map_revision"),
                    }
                    response = {
                        **response,
                        "incremental": True,
                        "dynamic_map_update": update_with_revision,
                    }
            else:
                response = {
                    "status": "ok",
                    "scene_id": scene_id,
                    "incremental": True,
                    "reused": True,
                    "dynamic_map_update": update,
                }
        else:
            response = self._send_request(
                {
                    "cmd": "configure",
                    "scene_id": scene_id,
                    "frame_id": self.frame_id,
                    "action_name": self.action_name,
                    "map": map_spec,
                }
            )
        response_ok = not isinstance(response, dict) or str(response.get("status") or "ok") == "ok"
        if response_ok:
            self._configured_scene_signature = scene_signature
            self._configured_base_signature = base_signature
            self._configured_map_spec = map_spec
            if isinstance(response, dict):
                response_revision = response.get("map_revision")
                dynamic_update = response.get("dynamic_map_update")
                if response_revision is None and isinstance(dynamic_update, dict):
                    response_revision = dynamic_update.get("map_revision")
                if isinstance(response_revision, int):
                    self._configured_map_revision = response_revision
        if isinstance(response, dict) and "dynamic_map_update" not in response:
            initial_update = map_spec.get("dynamic_map_update")
            if isinstance(initial_update, dict):
                initial_update = {
                    **initial_update,
                    "map_revision": response.get("map_revision"),
                }
            response = {
                **response,
                "dynamic_map_update": initial_update,
            }
        return response

    def set_pose(
        self,
        *,
        pose_xy: dict[str, float],
        yaw: float,
        linear_velocity: dict[str, float] | None = None,
        angular_velocity_z: float = 0.0,
    ) -> dict[str, Any]:
        return self._send_request(
            {
                "cmd": "set_pose",
                "pose": {
                    "x": float(pose_xy["x"]),
                    "y": float(pose_xy["y"]),
                    "yaw": float(yaw),
                },
                "twist": {
                    "vx": float((linear_velocity or {}).get("x", 0.0)),
                    "vy": float((linear_velocity or {}).get("y", 0.0)),
                    "wz": float(angular_velocity_z),
                },
            }
        )

    def compute_path(
        self,
        *,
        start_xy: dict[str, float],
        goal_xy: dict[str, float],
        planner_id: str | None,
        timeout_s: float,
    ) -> dict[str, Any]:
        return self._send_request(
            {
                "cmd": "compute_path",
                "frame_id": self.frame_id,
                "planner_id": planner_id or "",
                "timeout_s": float(timeout_s),
                "start": {
                    "x": float(start_xy["x"]),
                    "y": float(start_xy["y"]),
                    "yaw": float(start_xy.get("yaw", 0.0)),
                },
                "goal": {
                    "x": float(goal_xy["x"]),
                    "y": float(goal_xy["y"]),
                    "yaw": float(goal_xy.get("yaw", 0.0)),
                },
            }
        )

    def follow_path(
        self,
        *,
        path_points: list[dict[str, float]],
        timeout_s: float,
    ) -> dict[str, Any]:
        return self._send_request(
            {
                "cmd": "follow_path",
                "frame_id": self.frame_id,
                "timeout_s": float(timeout_s),
                "points": [
                    {
                        "x": float(point["x"]),
                        "y": float(point["y"]),
                    }
                    for point in path_points
                ],
            }
        )

    def get_cmd_vel(self) -> dict[str, Any]:
        return self._send_request({"cmd": "get_cmd_vel"})

    def cancel_follow_path(self) -> dict[str, Any]:
        return self._send_request({"cmd": "cancel_follow_path"})

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process

        if not self.worker_script.is_file():
            raise RuntimeError(f"Nav2 runtime worker not found: {self.worker_script}")

        resolved_setup_script = self.profile.resolved_setup_script()
        command = (
            f"source {shlex.quote(resolved_setup_script)} && "
            f"{self.profile.python_bin} -u {shlex.quote(str(self.worker_script))}"
        )
        self._process = subprocess.Popen(
            ["bash", "-lc", command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=_build_overlay_env(),
        )
        return self._process

    def _send_request(
        self, payload: dict[str, Any], *, tolerate_errors: bool = False
    ) -> dict[str, Any]:
        process = self._ensure_process()
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("Nav2 runtime worker pipes are unavailable")

        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()

        response_line = process.stdout.readline()
        if not response_line:
            stderr_tail = ""
            if process.stderr is not None:
                try:
                    stderr_tail = process.stderr.read().strip()
                except Exception:
                    stderr_tail = ""
            raise RuntimeError(stderr_tail or "Nav2 runtime worker exited without a response")

        response = json.loads(response_line)
        if response.get("status") == "error" and not tolerate_errors:
            raise RuntimeError(str(response.get("error") or "nav2_runtime_bridge_error"))
        return response


_RUNTIME_CLIENTS: dict[tuple[str, str], PersistentNav2RuntimeBridgeClient] = {}


def get_nav2_runtime_bridge_client(
    *,
    version_profile: str,
    frame_id: str = "map",
    action_name: str = "compute_path_to_pose",
) -> PersistentNav2RuntimeBridgeClient:
    key = (version_profile, frame_id)
    client = _RUNTIME_CLIENTS.get(key)
    if client is None:
        client = PersistentNav2RuntimeBridgeClient(
            profile=NAV2_VERSION_PROFILES[version_profile],
            frame_id=frame_id,
            action_name=action_name,
        )
        _RUNTIME_CLIENTS[key] = client
    return client
