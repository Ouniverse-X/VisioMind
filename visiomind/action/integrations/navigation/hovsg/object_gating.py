from __future__ import annotations

import hashlib
import json
from typing import Any

from visiomind.action.shared.models.scene_state import (
    NON_BLOCKING_NAVIGATION_ROLES,
    RuntimeObjectState,
)
from visiomind.action.shared.geometry_frames import (
    horizontal_axis_indices,
    vertical_axis_index,
)

from . import runtime_state as hovsg_runtime_state
from . import effective_scene as hovsg_effective_scene
from . import scene_runtime as hovsg_scene_runtime
from .models import HOVSGObjectAsset

OBJECT_MOVE_THRESHOLD_M = 0.05
DEFAULT_OBJECT_HALF_EXTENT_M = 0.5
MIN_SUPPORT_SURFACE_AREA_M2 = 4.0
MAX_SUPPORT_SURFACE_THICKNESS_M = 0.2
MAX_SUPPORT_SURFACE_TOP_OFFSET_M = 0.08
SUPPORT_SURFACE_FLOOR_SLOP_M = 0.2


def _empty_overlay() -> dict[str, Any]:
    return {
        "obstacles": [],
        "clear_regions": [],
        "clear_polygons": [],
        "stamp_polygons": [],
        "dirty_bounds": None,
        "geometry_revision": "",
        "pose_revision": "",
        "map_revision": "",
        "signature": "",
        "geometry_sources": {},
        "aabb_fallback_count": 0,
        "active": False,
    }


def runtime_object_map_overlays(
    adapter: Any,
    scene_id: str | None,
    *,
    navigation_goal: dict[str, Any] | None = None,
    include_unchanged: bool = False,
) -> dict[str, Any]:
    if not scene_id:
        return _empty_overlay()
    state = hovsg_runtime_state.current_scene_state(adapter, scene_id)
    if state is None and not include_unchanged:
        return _empty_overlay()

    scene = hovsg_scene_runtime.ensure_scene(adapter, scene_id)
    if scene is None:
        return _empty_overlay()
    effective_view = hovsg_effective_scene.effective_scene_view(
        adapter,
        scene,
        scene_id=scene_id,
    )

    if state is not None and state.objects:
        _initialize_collision_baselines(adapter, scene_id, state)
    obstacles: list[dict[str, Any]] = []
    clear_regions: list[dict[str, Any]] = []
    matched_names: set[str] = set()
    for obj in scene.objects.values():
        effective_object = effective_view.resolve_object(obj.object_id)
        if (
            state is not None
            and hovsg_runtime_state.match_runtime_door(
                state,
                object_name=obj.name,
                object_id=obj.object_id,
                static_centroid=obj.centroid,
            )
            is not None
        ):
            continue
        runtime_object = (
            hovsg_runtime_state.match_runtime_object(
                state,
                object_name=obj.name,
                object_id=obj.object_id,
                static_centroid=obj.centroid,
            )
            if state is not None
            else None
        )
        if runtime_object is None:
            if effective_object is not None and not effective_object.participates_in_navigation:
                continue
            if include_unchanged:
                canonical_obstacle = _canonical_object_obstacle(
                    adapter,
                    scene_id,
                    scene,
                    obj,
                    None,
                    overlay_kind="canonical_object",
                )
                if canonical_obstacle is not None:
                    obstacles.append(canonical_obstacle)
            continue
        if runtime_object.name in matched_names:
            continue
        matched_names.add(runtime_object.name)
        object_moved = _object_moved(adapter, scene, scene_id, runtime_object, obj)
        if effective_object is not None and not effective_object.participates_in_navigation:
            if not include_unchanged:
                static_region = _static_object_region(
                    adapter,
                    scene_id,
                    scene,
                    obj,
                    runtime_object,
                )
                if static_region is not None:
                    clear_regions.append(static_region)
            continue
        if not include_unchanged and not object_moved:
            continue
        if object_moved and not include_unchanged:
            static_region = _static_object_region(
                adapter,
                scene_id,
                scene,
                obj,
                runtime_object,
            )
            if static_region is not None:
                clear_regions.append(static_region)
        if effective_object is not None and _prefer_effective_geometry(runtime_object):
            if effective_object.navigation_footprints:
                obstacles.append(_effective_object_obstacle(effective_object))
            continue
        navigation_floor_height = _object_navigation_floor_height(scene, obj)
        if include_unchanged and not object_moved:
            runtime_obstacles = _runtime_object_obstacles(
                runtime_object,
                navigation_floor_height=navigation_floor_height,
                vertical_axis=str(getattr(scene, "vertical_axis", "z") or "z"),
            )
            precise_obstacles = [
                obstacle
                for obstacle in runtime_obstacles
                if obstacle.get("geometry_source") != "aabb_fallback"
            ]
            if precise_obstacles:
                obstacles.extend(precise_obstacles)
                continue
            if runtime_object.collision_parts and not runtime_obstacles:
                continue
            canonical_obstacle = _canonical_object_obstacle(
                adapter,
                scene_id,
                scene,
                obj,
                runtime_object,
                overlay_kind="canonical_object",
            )
            if canonical_obstacle is not None:
                obstacles.append(canonical_obstacle)
                continue
        obstacles.extend(
            _runtime_object_obstacles(
                runtime_object,
                navigation_floor_height=navigation_floor_height,
                vertical_axis=str(getattr(scene, "vertical_axis", "z") or "z"),
            )
        )

    for runtime_object in state.objects.values() if state is not None else []:
        if runtime_object.name in matched_names:
            continue
        effective_object = effective_view.resolve_object(runtime_object.name)
        if effective_object is None or not effective_object.participates_in_navigation:
            continue
        if _prefer_effective_geometry(runtime_object):
            if effective_object.navigation_footprints:
                obstacles.append(_effective_object_obstacle(effective_object))
            continue
        runtime_obstacles = _runtime_object_obstacles(
            runtime_object,
            navigation_floor_height=_runtime_object_navigation_floor_height(
                adapter,
                scene,
                runtime_object,
            ),
            vertical_axis=str(getattr(scene, "vertical_axis", "z") or "z"),
        )
        if runtime_obstacles:
            obstacles.extend(runtime_obstacles)
        elif not runtime_object.collision_parts and effective_object.navigation_footprints:
            obstacles.append(_effective_object_obstacle(effective_object))

    del navigation_goal
    active = bool(obstacles or clear_regions)
    map_revision = _overlay_signature(obstacles, clear_regions) if active else ""
    geometry_revision = _geometry_revision(obstacles, clear_regions)
    return {
        "obstacles": obstacles,
        "clear_regions": clear_regions,
        "clear_polygons": _overlay_polygons(clear_regions),
        "stamp_polygons": _overlay_polygons(obstacles),
        "dirty_bounds": _dirty_bounds([*clear_regions, *obstacles]),
        "geometry_revision": geometry_revision,
        "pose_revision": str(state.signature or "") if state is not None else "",
        "map_revision": map_revision,
        "signature": map_revision,
        "geometry_sources": _geometry_source_counts(obstacles, clear_regions),
        "aabb_fallback_count": sum(
            1
            for item in [*obstacles, *clear_regions]
            if item.get("geometry_source") == "aabb_fallback"
        ),
        "active": active,
    }


def _effective_object_obstacle(
    effective_object: hovsg_effective_scene.EffectiveObjectState,
) -> dict[str, Any]:
    polygons = [
        [[float(point[0]), float(point[1])] for point in polygon]
        for polygon in effective_object.navigation_footprints
    ]
    obstacle: dict[str, Any] = {
        "name": effective_object.name,
        "object_id": effective_object.object_id,
        "overlay_kind": "runtime_object",
        "geometry_source": effective_object.geometry_source,
        "geometry_id": f"effective:{effective_object.object_id}",
        "geometry_hash": effective_object.geometry_revision,
        "polygons": polygons,
    }
    if isinstance(effective_object.position, dict):
        obstacle["position"] = dict(effective_object.position)
    points = [point for polygon in polygons for point in polygon]
    if points:
        minimum = {
            "x": min(float(point[0]) for point in points),
            "y": min(float(point[1]) for point in points),
        }
        maximum = {
            "x": max(float(point[0]) for point in points),
            "y": max(float(point[1]) for point in points),
        }
        obstacle["min"] = minimum
        obstacle["max"] = maximum
        obstacle["aabb"] = {"min": minimum, "max": maximum}
    return obstacle


def _prefer_effective_geometry(runtime_object: RuntimeObjectState) -> bool:
    return bool(
        runtime_object.covariance_xy
        or runtime_object.oriented_bbox
        or (not runtime_object.collision_parts and runtime_object.aabb is None)
    )


def _object_moved(
    adapter: Any,
    scene: Any,
    scene_id: str,
    runtime_object: RuntimeObjectState,
    static_object: HOVSGObjectAsset,
) -> bool:
    position_moved = False
    if isinstance(runtime_object.position, dict) and isinstance(static_object.centroid, dict):
        projector = getattr(adapter, "_project_horizontal", None)
        if callable(projector):
            try:
                runtime_xy = projector(scene, runtime_object.position)
                static_xy = projector(scene, static_object.centroid)
                if isinstance(runtime_xy, (list, tuple)) and isinstance(static_xy, (list, tuple)):
                    distance = (
                        (float(runtime_xy[0]) - float(static_xy[0])) ** 2
                        + (float(runtime_xy[1]) - float(static_xy[1])) ** 2
                    ) ** 0.5
                    position_moved = distance >= OBJECT_MOVE_THRESHOLD_M
            except (IndexError, TypeError, ValueError):
                position_moved = False
    baseline = _collision_baseline(adapter, scene_id, runtime_object)
    return position_moved or baseline != _collision_parts_signature(runtime_object)


def _initialize_collision_baselines(adapter: Any, scene_id: str, state: Any) -> None:
    baselines = getattr(adapter, "_runtime_object_collision_baselines", None)
    if not isinstance(baselines, dict):
        baselines = {}
        adapter._runtime_object_collision_baselines = baselines
    steps = getattr(adapter, "_runtime_object_collision_baseline_steps", None)
    if not isinstance(steps, dict):
        steps = {}
        adapter._runtime_object_collision_baseline_steps = steps
    previous_step = steps.get(scene_id)
    if isinstance(previous_step, int) and int(state.step) < previous_step:
        baselines.pop(scene_id, None)
    scene_baselines = baselines.setdefault(scene_id, {})
    steps[scene_id] = int(state.step)
    for name, runtime_object in state.objects.items():
        scene_baselines.setdefault(name, _collision_parts_signature(runtime_object))


def _collision_baseline(adapter: Any, scene_id: str, runtime_object: RuntimeObjectState) -> tuple:
    baselines = getattr(adapter, "_runtime_object_collision_baselines", {})
    scene_baselines = baselines.get(scene_id, {}) if isinstance(baselines, dict) else {}
    return scene_baselines.get(runtime_object.name, _collision_parts_signature(runtime_object))


def _collision_parts_signature(runtime_object: RuntimeObjectState) -> tuple:
    if runtime_object.collision_parts:
        return tuple(
            (
                str(part.get("link") or ""),
                tuple(round(float(value), 3) for value in part.get("min", [])[:3]),
                tuple(round(float(value), 3) for value in part.get("max", [])[:3]),
            )
            for part in sorted(
                runtime_object.collision_parts,
                key=lambda item: str(item.get("link") or ""),
            )
            if isinstance(part, dict)
        )
    if isinstance(runtime_object.aabb, dict):
        return (
            (
                "__aabb__",
                tuple(round(float(value), 3) for value in runtime_object.aabb.get("min", [])[:3]),
                tuple(round(float(value), 3) for value in runtime_object.aabb.get("max", [])[:3]),
            ),
        )
    return ()


def _static_object_region(
    adapter: Any,
    scene_id: str,
    scene: Any,
    obj: HOVSGObjectAsset,
    runtime_object: RuntimeObjectState,
) -> dict[str, Any] | None:
    canonical_region = _canonical_object_obstacle(
        adapter,
        scene_id,
        scene,
        obj,
        runtime_object,
        overlay_kind="moved_object_exported_footprint",
    )
    if canonical_region is not None:
        return canonical_region

    projector = getattr(adapter, "_project_horizontal", None)
    if isinstance(obj.centroid, dict):
        try:
            point = projector(scene, obj.centroid) if callable(projector) else None
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                return None
            half_extent = DEFAULT_OBJECT_HALF_EXTENT_M
            return {
                "name": runtime_object.name,
                "object_id": obj.object_id,
                "overlay_kind": "moved_object_exported_footprint",
                "geometry_source": "aabb_fallback",
                "position": {
                    "x": float(point[0]),
                    "y": float(point[1]),
                },
                "half_extent_m": half_extent,
            }
        except (KeyError, TypeError, ValueError):
            pass
    return None


def _canonical_object_obstacle(
    adapter: Any,
    scene_id: str,
    scene: Any,
    obj: HOVSGObjectAsset,
    runtime_object: RuntimeObjectState | None,
    *,
    overlay_kind: str,
) -> dict[str, Any] | None:
    cached = _canonical_object_footprint(adapter, scene_id, scene, obj)
    if cached is None:
        return None
    polygon, geometry_hash = cached
    return {
        "name": runtime_object.name if runtime_object is not None else obj.name,
        "object_id": obj.object_id,
        "overlay_kind": overlay_kind,
        "geometry_id": f"hovsg:{scene_id}:{obj.object_id}",
        "geometry_hash": geometry_hash,
        "geometry_source": "semantic_polygon",
        "polygons": [[[float(point[0]), float(point[1])] for point in polygon]],
    }


def _canonical_object_footprint(
    adapter: Any,
    scene_id: str,
    scene: Any,
    obj: HOVSGObjectAsset,
) -> tuple[list[tuple[float, float]], str] | None:
    normalized_vertices = []
    for vertex in obj.vertices:
        if not isinstance(vertex, (list, tuple)) or len(vertex) < 3:
            continue
        try:
            normalized_vertices.append([round(float(vertex[index]), 6) for index in range(3)])
        except (TypeError, ValueError):
            continue
    if not normalized_vertices:
        return None
    geometry_hash = hashlib.sha1(
        json.dumps(normalized_vertices, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    cache = getattr(adapter, "_canonical_object_footprint_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        adapter._canonical_object_footprint_cache = cache
    cache_key = (str(scene_id), str(obj.object_id), geometry_hash)
    cached = cache.get(cache_key)
    if isinstance(cached, list):
        return list(cached), geometry_hash

    projector = getattr(adapter, "_project_horizontal", None)
    if not callable(projector):
        return None
    points: list[tuple[float, float]] = []
    for vertex in normalized_vertices:
        try:
            point = projector(
                scene,
                {"x": vertex[0], "y": vertex[1], "z": vertex[2]},
            )
        except Exception:
            continue
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            points.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError):
            continue
    polygon = _convex_hull(points)
    if len(polygon) < 3:
        return None
    cache[cache_key] = list(polygon)
    return polygon, geometry_hash


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    if len(unique) < 3:
        return []

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (
            second[0] - origin[0]
        )

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _runtime_object_obstacles(
    runtime_object: RuntimeObjectState,
    *,
    navigation_floor_height: float | None = None,
    vertical_axis: str = "z",
) -> list[dict[str, Any]]:
    if runtime_object.navigation_role in NON_BLOCKING_NAVIGATION_ROLES:
        return []
    obstacles: list[dict[str, Any]] = []
    for part in runtime_object.collision_parts:
        part_axis = str(part.get("vertical_axis") or "").lower()
        if part_axis and part_axis != vertical_axis:
            continue
        if not _part_blocks_navigation(
            part,
            navigation_floor_height,
            vertical_axis=vertical_axis,
        ):
            continue
        obstacle = _polygon_obstacle(
            runtime_object.name,
            part,
            link=part.get("link"),
        )
        if obstacle is not None:
            obstacles.append(obstacle)
            continue
        obstacle = _aabb_obstacle(
            runtime_object.name,
            part,
            link=part.get("link"),
            vertical_axis=vertical_axis,
        )
        if obstacle is not None:
            obstacles.append(obstacle)
    if obstacles:
        return obstacles
    fallback = (
        _aabb_obstacle(
            runtime_object.name,
            runtime_object.aabb,
            vertical_axis=vertical_axis,
        )
        if _part_blocks_navigation(
            runtime_object.aabb,
            navigation_floor_height,
            vertical_axis=vertical_axis,
        )
        else None
    )
    if fallback is not None:
        return [fallback]
    if runtime_object.collision_parts:
        return []
    if isinstance(runtime_object.position, dict):
        try:
            horizontal_axes = _horizontal_axis_names(vertical_axis)
            return [
                {
                    "name": runtime_object.name,
                    "overlay_kind": "moved_object",
                    "geometry_source": "aabb_fallback",
                    "position": {
                        "x": float(runtime_object.position[horizontal_axes[0]]),
                        "y": float(runtime_object.position[horizontal_axes[1]]),
                    },
                    "half_extent_m": DEFAULT_OBJECT_HALF_EXTENT_M,
                }
            ]
        except (KeyError, TypeError, ValueError):
            pass
    return []


def _runtime_object_navigation_floor_height(
    adapter: Any,
    scene: Any,
    runtime_object: RuntimeObjectState,
) -> float | None:
    vertical_axis = str(getattr(scene, "vertical_axis", "z") or "z")
    room_id = str(runtime_object.room_hint or "").strip() or None
    rooms = getattr(scene, "rooms", {})
    if room_id not in rooms:
        try:
            room_id = hovsg_runtime_state.containing_room_id(
                adapter,
                scene,
                runtime_object.position,
            )
        except Exception:
            room_id = None
    if room_id in rooms:
        floor_id = str(getattr(rooms[room_id], "floor_id", "") or "")
        floor = getattr(scene, "floors", {}).get(floor_id)
        try:
            return float(floor.floor_zero_level) if floor is not None else None
        except (TypeError, ValueError):
            pass

    floor_heights: list[float] = []
    for floor in getattr(scene, "floors", {}).values():
        try:
            floor_heights.append(float(floor.floor_zero_level))
        except (AttributeError, TypeError, ValueError):
            continue
    if not floor_heights:
        return None
    try:
        object_height = float((runtime_object.position or {}).get(vertical_axis, 0.0))
    except (TypeError, ValueError):
        object_height = floor_heights[0]
    return min(floor_heights, key=lambda height: abs(object_height - height))


def _object_navigation_floor_height(scene: Any, obj: HOVSGObjectAsset) -> float | None:
    rooms = getattr(scene, "rooms", {})
    floors = getattr(scene, "floors", {})
    room = rooms.get(str(getattr(obj, "room_id", "")))
    floor = floors.get(str(room.floor_id)) if room is not None else None
    try:
        return float(floor.floor_zero_level) if floor is not None else None
    except (TypeError, ValueError):
        return None


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
        if isinstance(corner_min, (list, tuple)) and isinstance(corner_max, (list, tuple)):
            try:
                height_index = vertical_axis_index(vertical_axis)
                height_min = float(corner_min[height_index])
                height_max = float(corner_max[height_index])
            except (IndexError, TypeError, ValueError):
                return True
        else:
            return True
    navigation_min = float(floor_height) - floor_slop_m
    navigation_max = float(floor_height) + robot_clearance_height_m
    return float(height_max) >= navigation_min and float(height_min) <= navigation_max


def _part_blocks_navigation(
    part: Any,
    floor_height: float | None,
    *,
    vertical_axis: str = "z",
) -> bool:
    if not _part_overlaps_navigation_height(
        part,
        floor_height,
        vertical_axis=vertical_axis,
    ):
        return False
    if floor_height is None or not isinstance(part, dict):
        return True
    height_bounds = _part_height_bounds(part, vertical_axis=vertical_axis)
    if height_bounds is None:
        return True
    height_min, height_max = height_bounds
    thickness = max(0.0, height_max - height_min)
    footprint_area = _part_horizontal_footprint_area(
        part,
        vertical_axis=vertical_axis,
    )
    is_large_floor_surface = (
        footprint_area is not None
        and footprint_area >= MIN_SUPPORT_SURFACE_AREA_M2
        and thickness <= MAX_SUPPORT_SURFACE_THICKNESS_M
        and height_min >= float(floor_height) - SUPPORT_SURFACE_FLOOR_SLOP_M
        and height_max <= float(floor_height) + MAX_SUPPORT_SURFACE_TOP_OFFSET_M
    )
    return not is_large_floor_surface


def _part_height_bounds(
    part: dict[str, Any],
    *,
    vertical_axis: str,
) -> tuple[float, float] | None:
    height_min = part.get("height_min")
    height_max = part.get("height_max")
    if isinstance(height_min, (int, float)) and isinstance(height_max, (int, float)):
        return float(height_min), float(height_max)
    corner_min = part.get("min")
    corner_max = part.get("max")
    if not isinstance(corner_min, (list, tuple)) or not isinstance(corner_max, (list, tuple)):
        return None
    try:
        height_index = vertical_axis_index(vertical_axis)
        return float(corner_min[height_index]), float(corner_max[height_index])
    except (IndexError, TypeError, ValueError):
        return None


def _part_horizontal_footprint_area(
    part: dict[str, Any],
    *,
    vertical_axis: str,
) -> float | None:
    points: list[tuple[float, float]] = []
    for polygon in _normalized_polygons(part.get("world_polygons")):
        points.extend((float(point[0]), float(point[1])) for point in polygon)
    if not points:
        corner_min = part.get("min")
        corner_max = part.get("max")
        if isinstance(corner_min, (list, tuple)) and isinstance(corner_max, (list, tuple)):
            try:
                horizontal_indices = horizontal_axis_indices(vertical_axis)
                return max(
                    0.0,
                    float(corner_max[horizontal_indices[0]])
                    - float(corner_min[horizontal_indices[0]]),
                ) * max(
                    0.0,
                    float(corner_max[horizontal_indices[1]])
                    - float(corner_min[horizontal_indices[1]]),
                )
            except (IndexError, TypeError, ValueError):
                return None
        return None
    return max(0.0, max(point[0] for point in points) - min(point[0] for point in points)) * max(
        0.0,
        max(point[1] for point in points) - min(point[1] for point in points),
    )


def _polygon_obstacle(
    name: str,
    part: Any,
    *,
    link: Any = None,
) -> dict[str, Any] | None:
    if not isinstance(part, dict):
        return None
    polygons = _normalized_polygons(part.get("world_polygons"))
    if not polygons:
        return None
    obstacle: dict[str, Any] = {
        "name": name,
        "overlay_kind": "runtime_object",
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


def _normalized_polygons(value: Any) -> list[list[list[float]]]:
    if not isinstance(value, (list, tuple)):
        return []
    polygons: list[list[list[float]]] = []
    for raw_polygon in value:
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
    return polygons


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
            "overlay_kind": "moved_object",
            "geometry_source": "aabb_fallback",
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


def _overlay_polygons(items: list[dict[str, Any]]) -> list[list[list[float]]]:
    polygons: list[list[list[float]]] = []
    for item in items:
        raw_polygons = item.get("polygons")
        if not isinstance(raw_polygons, list):
            raw_polygon = item.get("polygon")
            raw_polygons = [raw_polygon] if isinstance(raw_polygon, list) else []
        for raw_polygon in raw_polygons:
            if not isinstance(raw_polygon, list):
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
    return polygons


def _dirty_bounds(items: list[dict[str, Any]]) -> dict[str, float] | None:
    points: list[tuple[float, float]] = []
    for polygon in _overlay_polygons(items):
        points.extend((float(point[0]), float(point[1])) for point in polygon)
    for item in items:
        corner_min = item.get("min")
        corner_max = item.get("max")
        if isinstance(corner_min, dict) and isinstance(corner_max, dict):
            try:
                points.extend(
                    [
                        (float(corner_min["x"]), float(corner_min["y"])),
                        (float(corner_max["x"]), float(corner_max["y"])),
                    ]
                )
            except (KeyError, TypeError, ValueError):
                pass
        position = item.get("position")
        if isinstance(position, dict):
            try:
                half_extent = float(item.get("half_extent_m", DEFAULT_OBJECT_HALF_EXTENT_M))
                x_coord = float(position["x"])
                y_coord = float(position["y"])
                points.extend(
                    [
                        (x_coord - half_extent, y_coord - half_extent),
                        (x_coord + half_extent, y_coord + half_extent),
                    ]
                )
            except (KeyError, TypeError, ValueError):
                pass
    if not points:
        return None
    return {
        "min_x": min(point[0] for point in points),
        "min_y": min(point[1] for point in points),
        "max_x": max(point[0] for point in points),
        "max_y": max(point[1] for point in points),
    }


def _overlay_signature(
    obstacles: list[dict[str, Any]],
    clear_regions: list[dict[str, Any]],
) -> str:
    encoded = json.dumps(
        {"clear": clear_regions, "stamp": obstacles},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:16]


def _geometry_revision(
    obstacles: list[dict[str, Any]],
    clear_regions: list[dict[str, Any]],
) -> str:
    parts = sorted(
        (
            str(item.get("geometry_id") or item.get("link") or item.get("name") or ""),
            str(item.get("geometry_hash") or ""),
            str(item.get("geometry_source") or ""),
        )
        for item in [*clear_regions, *obstacles]
    )
    if not parts:
        return ""
    return hashlib.sha1(json.dumps(parts, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def _geometry_source_counts(
    obstacles: list[dict[str, Any]],
    clear_regions: list[dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in [*clear_regions, *obstacles]:
        source = str(item.get("geometry_source") or "unknown")
        counts[source] = counts.get(source, 0) + 1
    return counts


__all__ = [
    "DEFAULT_OBJECT_HALF_EXTENT_M",
    "OBJECT_MOVE_THRESHOLD_M",
    "runtime_object_map_overlays",
]
