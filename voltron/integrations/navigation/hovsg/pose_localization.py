"""Pose localization and planar geometry helpers for the HOV-SG navigator."""

from __future__ import annotations

import math
from typing import Any

from .models import HOVSGRoomAsset, HOVSGRoomLocalization, HOVSGSceneAsset


def localize_pose(
    adapter: Any,
    scene: HOVSGSceneAsset,
    pose: dict[str, Any],
    *,
    previous_room_id: str | None,
    persist: bool,
) -> dict[str, Any]:
    localized: dict[str, Any] = {}
    room = localized_room(
        adapter,
        scene,
        pose,
        previous_room_id=previous_room_id,
    )
    if room is not None:
        localized["current_room"] = room.name or room.room_id
        localized["current_region"] = room.name or room.room_id
        localized["room_id"] = room.room_id
        localized["floor_id"] = room.floor_id
        if persist:
            adapter._last_localized_room_ids[scene.scene_id] = room.room_id
    elif persist:
        adapter._last_localized_room_ids.pop(scene.scene_id, None)
    return localized


def containing_room(
    adapter: Any,
    scene: HOVSGSceneAsset,
    pose: dict[str, Any],
) -> HOVSGRoomAsset | None:
    if has_complete_pose(adapter, pose):
        localization = select_room_localization(
            adapter,
            scene,
            pose,
            previous_room_id=None,
        )
        if localization is not None:
            return localization.room
    return None


def localized_room(
    adapter: Any,
    scene: HOVSGSceneAsset,
    pose: dict[str, Any],
    *,
    previous_room_id: str | None,
) -> HOVSGRoomAsset | None:
    localization = select_room_localization(
        adapter,
        scene,
        pose,
        previous_room_id=previous_room_id,
    )
    if localization is None:
        return None
    return localization.room


def select_room_localization(
    adapter: Any,
    scene: HOVSGSceneAsset,
    pose: dict[str, Any],
    *,
    previous_room_id: str | None,
) -> HOVSGRoomLocalization | None:
    if not has_complete_pose(adapter, pose):
        return None
    localizations = room_localizations(adapter, scene, pose)
    if not localizations:
        return None

    previous = None
    if previous_room_id:
        previous = next(
            (item for item in localizations if item.room.room_id == previous_room_id),
            None,
        )

    containing = [item for item in localizations if item.contains]
    containing.sort(
        key=lambda item: (
            item.area is None,
            item.area or float("inf"),
            item.centroid_distance_sq,
        )
    )
    if previous is not None:
        if previous.distance_to_boundary <= adapter.room_hysteresis_margin:
            return previous
        if containing and previous.room.room_id == containing[0].room.room_id:
            return previous
    if not containing:
        return None
    return containing[0]


def room_localizations(
    adapter: Any,
    scene: HOVSGSceneAsset,
    pose: dict[str, Any],
) -> list[HOVSGRoomLocalization]:
    point = project_horizontal(adapter, scene, pose)
    if point is None:
        return []

    localizations: list[HOVSGRoomLocalization] = []
    for room in scene.rooms.values():
        polygon = [project_horizontal(adapter, scene, {"x": v[0], "y": v[1], "z": v[2]}) for v in room.vertices]
        projected = [vertex for vertex in polygon if vertex is not None]
        if len(projected) < 3:
            continue
        distance_to_boundary = point_to_polygon_boundary_distance(point, projected)
        contains = point_in_polygon(point, projected) or distance_to_boundary <= adapter.room_boundary_tolerance
        localizations.append(
            HOVSGRoomLocalization(
                room=room,
                contains=contains,
                distance_to_boundary=distance_to_boundary,
                area=polygon_area(projected),
                centroid_distance_sq=centroid_distance_sq(adapter, room.centroid, pose),
            )
        )
    return localizations


def has_complete_pose(adapter: Any, pose: dict[str, Any]) -> bool:
    return all(adapter._to_float(pose.get(axis)) is not None for axis in ("x", "y", "z"))


def room_contains_pose(
    adapter: Any,
    scene: HOVSGSceneAsset,
    room: HOVSGRoomAsset,
    pose: dict[str, Any],
) -> bool:
    if not room.vertices:
        return False
    point = project_horizontal(adapter, scene, pose)
    polygon = [project_horizontal(adapter, scene, {"x": v[0], "y": v[1], "z": v[2]}) for v in room.vertices]
    if point is None or any(vertex is None for vertex in polygon):
        return False
    projected = [vertex for vertex in polygon if vertex is not None]
    return point_in_polygon(point, projected) or (
        point_to_polygon_boundary_distance(point, projected) <= adapter.room_boundary_tolerance
    )


def horizontal_axes(scene: HOVSGSceneAsset) -> tuple[str, str]:
    axes_by_vertical = {
        "x": ("y", "z"),
        "y": ("x", "z"),
        "z": ("x", "y"),
    }
    return axes_by_vertical.get(scene.vertical_axis, ("x", "z"))


def project_horizontal(
    adapter: Any,
    scene: HOVSGSceneAsset,
    point: dict[str, Any],
) -> tuple[float, float] | None:
    axes = horizontal_axes(scene)
    first = adapter._to_float(point.get(axes[0]))
    second = adapter._to_float(point.get(axes[1]))
    if first is None or second is None:
        return None
    return first, second


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    x_coord, y_coord = point
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        intersects = ((current_y > y_coord) != (previous_y > y_coord)) and (
            x_coord
            < (previous_x - current_x) * (y_coord - current_y) / ((previous_y - current_y) or 1e-9) + current_x
        )
        if intersects:
            inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def point_to_polygon_boundary_distance(
    point: tuple[float, float],
    polygon: list[tuple[float, float]],
) -> float:
    if len(polygon) < 2:
        return float("inf")
    best_distance = float("inf")
    previous = polygon[-1]
    for current in polygon:
        best_distance = min(best_distance, distance_point_to_segment(point, previous, current))
        previous = current
    return best_distance


def distance_point_to_segment(
    point: tuple[float, float],
    segment_start: tuple[float, float],
    segment_end: tuple[float, float],
) -> float:
    sx, sy = segment_start
    ex, ey = segment_end
    dx = ex - sx
    dy = ey - sy
    if abs(dx) <= 1e-9 and abs(dy) <= 1e-9:
        return math.hypot(point[0] - sx, point[1] - sy)
    projection = ((point[0] - sx) * dx + (point[1] - sy) * dy) / (dx * dx + dy * dy)
    projection = max(0.0, min(1.0, projection))
    nearest_x = sx + projection * dx
    nearest_y = sy + projection * dy
    return math.hypot(point[0] - nearest_x, point[1] - nearest_y)


def room_polygon_area(
    adapter: Any,
    scene: HOVSGSceneAsset,
    room: HOVSGRoomAsset,
) -> float | None:
    polygon = [project_horizontal(adapter, scene, {"x": v[0], "y": v[1], "z": v[2]}) for v in room.vertices]
    projected = [vertex for vertex in polygon if vertex is not None]
    if len(projected) < 3:
        return None
    return polygon_area(projected)


def polygon_area(polygon: list[tuple[float, float]]) -> float:
    area = 0.0
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        area += previous_x * current_y - current_x * previous_y
        previous_x, previous_y = current_x, current_y
    return abs(area) * 0.5


def centroid_distance_sq(
    adapter: Any,
    centroid: dict[str, float] | None,
    pose: dict[str, Any],
) -> float:
    if centroid is None:
        return float("inf")
    px = adapter._to_float(pose.get("x"))
    py = adapter._to_float(pose.get("y"))
    pz = adapter._to_float(pose.get("z"))
    if px is None or py is None or pz is None:
        return float("inf")
    return (float(centroid["x"]) - px) ** 2 + (float(centroid["y"]) - py) ** 2 + (float(centroid["z"]) - pz) ** 2


def infer_floor_from_height(
    adapter: Any,
    scene: HOVSGSceneAsset,
    pose: dict[str, Any],
) -> str | None:
    axis_value = vertical_axis_value(adapter, scene, pose)
    if axis_value is None:
        return None
    best_floor = None
    best_delta = None
    for floor in scene.floors.values():
        if floor.floor_zero_level is None:
            continue
        delta = abs(floor.floor_zero_level - axis_value)
        if best_delta is None or delta < best_delta:
            best_floor = floor.floor_id
            best_delta = delta
    return best_floor


def vertical_axis_value(
    adapter: Any,
    scene: HOVSGSceneAsset,
    pose: dict[str, Any],
) -> float | None:
    return adapter._to_float(pose.get(scene.vertical_axis))
