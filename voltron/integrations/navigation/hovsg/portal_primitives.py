from __future__ import annotations

import math
from typing import Any

from .models import HOVSGRoomAsset, HOVSGSceneAsset


def room_polygon_2d(
    adapter: Any, scene: HOVSGSceneAsset, room: HOVSGRoomAsset
) -> list[tuple[float, float]]:
    projected = [
        adapter._project_horizontal(scene, {"x": v[0], "y": v[1], "z": v[2]}) for v in room.vertices
    ]
    return [point for point in projected if point is not None]


def polygon_segments(
    polygon: list[tuple[float, float]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    if len(polygon) < 2:
        return []
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    previous = polygon[-1]
    for current in polygon:
        segments.append((previous, current))
        previous = current
    return segments


def segment_entry_point_into_polygon(
    adapter: Any,
    *,
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    polygon: list[tuple[float, float]],
) -> tuple[float, float] | None:
    if len(polygon) < 3 or not adapter._point_in_polygon(end_point, polygon):
        return None
    if adapter._point_in_polygon(start_point, polygon):
        return start_point

    intersections: list[tuple[float, tuple[float, float]]] = []
    for segment_start, segment_end in polygon_segments(polygon):
        intersection = segment_intersection(start_point, end_point, segment_start, segment_end)
        if intersection is None:
            continue
        point, fraction = intersection
        if fraction < -1e-6 or fraction > 1.0 + 1e-6:
            continue
        intersections.append((max(0.0, min(1.0, fraction)), point))

    if not intersections:
        return None
    intersections.sort(key=lambda item: item[0])
    return intersections[0][1]


def segment_intersection(
    a0: tuple[float, float],
    a1: tuple[float, float],
    b0: tuple[float, float],
    b1: tuple[float, float],
) -> tuple[tuple[float, float], float] | None:
    ax = a1[0] - a0[0]
    ay = a1[1] - a0[1]
    bx = b1[0] - b0[0]
    by = b1[1] - b0[1]
    cross = ax * by - ay * bx
    qpx = b0[0] - a0[0]
    qpy = b0[1] - a0[1]

    if abs(cross) <= 1e-9:
        if abs(qpx * ay - qpy * ax) > 1e-9:
            return None
        length_sq = ax * ax + ay * ay
        if length_sq <= 1e-9:
            return None
        t0 = (qpx * ax + qpy * ay) / length_sq
        t1 = ((b1[0] - a0[0]) * ax + (b1[1] - a0[1]) * ay) / length_sq
        overlap_start = max(0.0, min(t0, t1))
        overlap_end = min(1.0, max(t0, t1))
        if overlap_start > overlap_end + 1e-9:
            return None
        point = (a0[0] + ax * overlap_start, a0[1] + ay * overlap_start)
        return point, overlap_start

    t = (qpx * by - qpy * bx) / cross
    u = (qpx * ay - qpy * ax) / cross
    if -1e-9 <= t <= 1.0 + 1e-9 and -1e-9 <= u <= 1.0 + 1e-9:
        point = (a0[0] + t * ax, a0[1] + t * ay)
        return point, t
    return None


def closest_points_between_segments(
    a0: tuple[float, float],
    a1: tuple[float, float],
    b0: tuple[float, float],
    b1: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    aligned = closest_points_on_axis_aligned_segments(a0, a1, b0, b1)
    if aligned is not None:
        return aligned
    candidates = [
        (a0, closest_point_on_segment(a0, b0, b1)),
        (a1, closest_point_on_segment(a1, b0, b1)),
        (closest_point_on_segment(b0, a0, a1), b0),
        (closest_point_on_segment(b1, a0, a1), b1),
    ]
    best_pair = candidates[0]
    best_distance = None
    for point_a, point_b in candidates:
        distance = (point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_pair = (point_a, point_b)
    return best_pair


def closest_points_on_axis_aligned_segments(
    a0: tuple[float, float],
    a1: tuple[float, float],
    b0: tuple[float, float],
    b1: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    a_vertical = abs(a0[0] - a1[0]) <= 1e-6
    b_vertical = abs(b0[0] - b1[0]) <= 1e-6
    a_horizontal = abs(a0[1] - a1[1]) <= 1e-6
    b_horizontal = abs(b0[1] - b1[1]) <= 1e-6

    if a_vertical and b_vertical:
        low = max(min(a0[1], a1[1]), min(b0[1], b1[1]))
        high = min(max(a0[1], a1[1]), max(b0[1], b1[1]))
        if low <= high:
            middle = (low + high) * 0.5
            return ((a0[0], middle), (b0[0], middle))

    if a_horizontal and b_horizontal:
        low = max(min(a0[0], a1[0]), min(b0[0], b1[0]))
        high = min(max(a0[0], a1[0]), max(b0[0], b1[0]))
        if low <= high:
            middle = (low + high) * 0.5
            return ((middle, a0[1]), (middle, b0[1]))

    return None


def closest_point_on_segment(
    point: tuple[float, float],
    seg_start: tuple[float, float],
    seg_end: tuple[float, float],
) -> tuple[float, float]:
    dx = seg_end[0] - seg_start[0]
    dy = seg_end[1] - seg_start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-9:
        return seg_start
    projection = ((point[0] - seg_start[0]) * dx + (point[1] - seg_start[1]) * dy) / length_sq
    projection = max(0.0, min(1.0, projection))
    return (seg_start[0] + projection * dx, seg_start[1] + projection * dy)


def axis_aligned_segment_axes(
    seg_start: tuple[float, float],
    seg_end: tuple[float, float],
) -> tuple[int, int] | None:
    dx = float(seg_end[0]) - float(seg_start[0])
    dy = float(seg_end[1]) - float(seg_start[1])
    if abs(dx) <= 1e-6 and abs(dy) > 1e-6:
        return 1, 0
    if abs(dy) <= 1e-6 and abs(dx) > 1e-6:
        return 0, 1
    return None


def segment_point_at_axis_value(
    *,
    seg_start: tuple[float, float],
    seg_end: tuple[float, float],
    axis_index: int,
    axis_value: float,
) -> tuple[float, float] | None:
    start_value = float(seg_start[axis_index])
    end_value = float(seg_end[axis_index])
    delta = end_value - start_value
    if abs(delta) <= 1e-9:
        return None
    ratio = (float(axis_value) - start_value) / delta
    if ratio < -1e-6 or ratio > 1.0 + 1e-6:
        return None
    ratio = max(0.0, min(1.0, ratio))
    return (
        float(seg_start[0]) + (float(seg_end[0]) - float(seg_start[0])) * ratio,
        float(seg_start[1]) + (float(seg_end[1]) - float(seg_start[1])) * ratio,
    )


def portal_plane_point(
    *,
    plane_axes: tuple[str, str],
    span_axis: str,
    normal_axis: str,
    span_value: float,
    normal_value: float,
) -> dict[str, float]:
    point = {plane_axes[0]: 0.0, plane_axes[1]: 0.0}
    point[span_axis] = float(span_value)
    point[normal_axis] = float(normal_value)
    return {"x": float(point[plane_axes[0]]), "y": float(point[plane_axes[1]])}


def lift_horizontal_point(
    adapter: Any,
    scene: HOVSGSceneAsset,
    horizontal_point: tuple[float, float],
    *,
    source_room: HOVSGRoomAsset | None,
    target_room: HOVSGRoomAsset | None,
) -> dict[str, float] | None:
    centroid = midpoint_from_positions(
        adapter,
        source_room.centroid if source_room is not None else None,
        target_room.centroid if target_room is not None else None,
    )
    if centroid is None:
        return None
    vertical_value = float(centroid[scene.vertical_axis])
    if scene.vertical_axis == "x":
        return {
            "x": vertical_value,
            "y": float(horizontal_point[0]),
            "z": float(horizontal_point[1]),
        }
    if scene.vertical_axis == "y":
        return {
            "x": float(horizontal_point[0]),
            "y": vertical_value,
            "z": float(horizontal_point[1]),
        }
    return {"x": float(horizontal_point[0]), "y": float(horizontal_point[1]), "z": vertical_value}


def midpoint_from_positions(
    adapter: Any,
    first: dict[str, Any] | None,
    second: dict[str, Any] | None,
) -> dict[str, float] | None:
    if not isinstance(first, dict) and not isinstance(second, dict):
        return None
    if not isinstance(first, dict):
        first = second
    if not isinstance(second, dict):
        second = first
    if not isinstance(first, dict) or not isinstance(second, dict):
        return None
    coordinates: dict[str, float] = {}
    for axis in ("x", "y", "z"):
        first_value = adapter._to_float(first.get(axis))
        second_value = adapter._to_float(second.get(axis))
        if first_value is None or second_value is None:
            return None
        coordinates[axis] = (first_value + second_value) * 0.5
    return coordinates


def append_waypoint_if_distinct(
    waypoints: list[dict[str, Any]],
    waypoint: dict[str, Any],
    *,
    epsilon: float = 0.05,
) -> None:
    if not waypoints:
        waypoints.append(waypoint)
        return
    previous = waypoints[-1]
    distance = math.sqrt(
        (float(previous["x"]) - float(waypoint["x"])) ** 2
        + (float(previous["y"]) - float(waypoint["y"])) ** 2
        + (float(previous["z"]) - float(waypoint["z"])) ** 2
    )
    if distance > epsilon or previous.get("room_id") != waypoint.get("room_id"):
        waypoints.append(waypoint)
        return

    waypoints[-1] = {**previous, **waypoint}
