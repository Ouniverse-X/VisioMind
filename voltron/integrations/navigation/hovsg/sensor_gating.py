"""Temporary perception obstacle overlays with step-based decay."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from . import runtime_state as hovsg_runtime_state


def runtime_sensor_map_overlays(
    adapter: Any,
    scene_id: str | None,
) -> dict[str, Any]:
    state = hovsg_runtime_state.current_scene_state(adapter, scene_id)
    if state is None or not state.temporary_obstacles:
        return {"obstacles": [], "signature": "", "active": False}
    obstacles: list[dict[str, Any]] = []
    for obstacle in state.temporary_obstacles:
        if obstacle.expires_at_step is not None and int(state.step) > int(
            obstacle.expires_at_step
        ):
            continue
        polygons = _expand_polygons_for_covariance(
            obstacle.polygons,
            obstacle.covariance_xy,
        )
        if not polygons:
            continue
        obstacles.append(
            {
                "name": obstacle.obstacle_id,
                "overlay_kind": "sensor_temporary",
                "geometry_id": f"sensor:{obstacle.source}:{obstacle.obstacle_id}",
                "geometry_source": obstacle.source,
                "confidence": obstacle.confidence,
                "expires_at_step": obstacle.expires_at_step,
                "polygons": polygons,
            }
        )
    if not obstacles:
        return {"obstacles": [], "signature": "", "active": False}
    encoded = json.dumps(obstacles, sort_keys=True, separators=(",", ":"))
    return {
        "obstacles": obstacles,
        "signature": hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:16],
        "active": True,
    }


def _expand_polygons_for_covariance(
    polygons: list[list[list[float]]],
    covariance_xy: list[list[float]] | None,
) -> list[list[list[float]]]:
    padding = 0.0
    if covariance_xy is not None:
        try:
            padding = min(
                1.0,
                2.0
                * math.sqrt(
                    max(0.0, float(covariance_xy[0][0]), float(covariance_xy[1][1]))
                ),
            )
        except (IndexError, TypeError, ValueError):
            padding = 0.0
    expanded: list[list[list[float]]] = []
    for polygon in polygons:
        if len(polygon) < 3:
            continue
        center_x = sum(float(point[0]) for point in polygon) / len(polygon)
        center_y = sum(float(point[1]) for point in polygon) / len(polygon)
        expanded_polygon: list[list[float]] = []
        for point in polygon:
            x_coord = float(point[0])
            y_coord = float(point[1])
            dx = x_coord - center_x
            dy = y_coord - center_y
            norm = math.hypot(dx, dy)
            if padding > 0.0 and norm > 1e-9:
                x_coord += padding * dx / norm
                y_coord += padding * dy / norm
            expanded_polygon.append([x_coord, y_coord])
        expanded.append(expanded_polygon)
    return expanded


__all__ = ["runtime_sensor_map_overlays"]
