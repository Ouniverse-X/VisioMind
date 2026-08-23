"""Portal waypoint assembly for HOV-SG room transitions."""

from __future__ import annotations

import math
from typing import Any

from .models import HOVSGRoomAsset, HOVSGSceneAsset
from .portal_candidates import transition_points_from_bboxes
from .portal_primitives import (
    closest_points_between_segments,
    lift_horizontal_point,
    midpoint_from_positions,
    polygon_segments,
    room_polygon_2d,
    segment_entry_point_into_polygon,
)
from .portal_refinement import room_transition_metrics


def transition_waypoint(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    source_room_id: str,
    target_room_id: str,
    fallback_from: dict[str, Any],
    fallback_to: dict[str, Any],
    start: dict[str, Any],
    goal: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any] | None:
    source_room = scene.rooms.get(source_room_id)
    target_room = scene.rooms.get(target_room_id)
    explicit_portal = adapter._explicit_transition_portal(
        scene=scene,
        source_room=source_room,
        target_room=target_room,
        context=context,
    )
    if explicit_portal is not None:
        return explicit_portal

    direct_transition = adapter._strong_room_transition_metrics(
        scene=scene,
        source_room_id=source_room_id,
        target_room_id=target_room_id,
        start=start,
        goal=goal,
        context=context,
    )
    portal_center = None
    if direct_transition is not None:
        portal_center = room_transition_target_entry(
            adapter,
            scene,
            source_room,
            target_room,
            start=start,
            goal=goal,
            context=context,
        )
    if portal_center is None:
        portal_center = edge_transition_target_entry(
            adapter,
            scene=scene,
            source_room=source_room,
            target_room=target_room,
            fallback_from=fallback_from,
            fallback_to=fallback_to,
        )
    if portal_center is None:
        portal_center = room_transition_target_entry(
            adapter,
            scene,
            source_room,
            target_room,
            start=start,
            goal=goal,
            context=context,
        )
    if portal_center is None:
        portal_center = midpoint_from_positions(adapter, fallback_from, fallback_to)
    if portal_center is None:
        return None

    waypoint = {
        "x": float(portal_center["x"]),
        "y": float(portal_center["y"]),
        "z": float(portal_center["z"]),
        "floor_id": target_room.floor_id if target_room is not None else fallback_to.get("floor_id"),
        "room_id": target_room.room_id if target_room is not None else fallback_to.get("room_id"),
        "source_room_id": source_room.room_id if source_room is not None else fallback_from.get("room_id"),
        "source_room_name": source_room.name if source_room is not None else fallback_from.get("room_name"),
        "room_name": target_room.name if target_room is not None else fallback_to.get("room_name"),
        "waypoint_type": "portal",
    }
    waypoint.update(
        portal_waypoint_metadata(
            adapter,
            scene=scene,
            source_room=source_room,
            target_room=target_room,
            start=start,
            goal=goal,
            context=context,
        )
    )
    return waypoint


def edge_transition_target_entry(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    source_room: HOVSGRoomAsset | None,
    target_room: HOVSGRoomAsset | None,
    fallback_from: dict[str, Any],
    fallback_to: dict[str, Any],
) -> dict[str, float] | None:
    if target_room is None:
        return None
    source_point = adapter._project_horizontal(scene, fallback_from)
    target_point = adapter._project_horizontal(scene, fallback_to)
    if source_point is None or target_point is None:
        return None
    target_polygon = room_polygon_2d(adapter, scene, target_room)
    if len(target_polygon) < 3:
        return None
    boundary_point = segment_entry_point_into_polygon(
        adapter,
        start_point=source_point,
        end_point=target_point,
        polygon=target_polygon,
    )
    if boundary_point is None:
        return None
    return offset_horizontal_point_along_segment_into_room(
        adapter,
        scene,
        boundary_point=boundary_point,
        toward_point=target_point,
        room=target_room,
        fallback_room=source_room,
    )


def room_transition_target_entry(
    adapter: Any,
    scene: HOVSGSceneAsset,
    source_room: HOVSGRoomAsset | None,
    target_room: HOVSGRoomAsset | None,
    *,
    start: dict[str, Any],
    goal: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, float] | None:
    transition_points = room_transition_points(
        adapter,
        scene,
        source_room,
        target_room,
        start=start,
        goal=goal,
        context=context,
    )
    if transition_points is None:
        return None
    _, target_point = transition_points
    return offset_horizontal_point_into_room(
        adapter,
        scene,
        boundary_point=target_point,
        room=target_room,
        fallback_room=source_room,
    )


def room_transition_center(
    adapter: Any,
    scene: HOVSGSceneAsset,
    source_room: HOVSGRoomAsset | None,
    target_room: HOVSGRoomAsset | None,
    *,
    start: dict[str, Any],
    goal: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, float] | None:
    transition_points = room_transition_points(
        adapter,
        scene,
        source_room,
        target_room,
        start=start,
        goal=goal,
        context=context,
    )
    if transition_points is None:
        return None
    source_point, target_point = transition_points
    midpoint = (
        (source_point[0] + target_point[0]) * 0.5,
        (source_point[1] + target_point[1]) * 0.5,
    )
    return lift_horizontal_point(
        adapter,
        scene,
        midpoint,
        source_room=source_room,
        target_room=target_room,
    )


def room_transition_points(
    adapter: Any,
    scene: HOVSGSceneAsset,
    source_room: HOVSGRoomAsset | None,
    target_room: HOVSGRoomAsset | None,
    *,
    start: dict[str, Any],
    goal: dict[str, Any],
    context: dict[str, Any],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if source_room is None or target_room is None:
        return None
    source_polygon = room_polygon_2d(adapter, scene, source_room)
    target_polygon = room_polygon_2d(adapter, scene, target_room)
    if len(source_polygon) < 2 or len(target_polygon) < 2:
        return None
    metrics = room_transition_metrics(
        adapter,
        scene,
        source_room,
        target_room,
        start=start,
        goal=goal,
        context=context,
    )
    if isinstance(metrics, dict) and metrics.get("traversability_blocked"):
        return None
    source_point = metrics.get("source_point") if isinstance(metrics, dict) else None
    target_point = metrics.get("target_point") if isinstance(metrics, dict) else None
    if isinstance(source_point, tuple) and isinstance(target_point, tuple):
        return source_point, target_point

    best_pair: tuple[tuple[float, float], tuple[float, float]] | None = None
    best_distance = None
    for source_start, source_end in polygon_segments(source_polygon):
        for target_start, target_end in polygon_segments(target_polygon):
            source_point, target_point = closest_points_between_segments(
                source_start,
                source_end,
                target_start,
                target_end,
            )
            distance = (source_point[0] - target_point[0]) ** 2 + (source_point[1] - target_point[1]) ** 2
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_pair = (source_point, target_point)
    return best_pair


def portal_waypoint_metadata(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    source_room: HOVSGRoomAsset | None,
    target_room: HOVSGRoomAsset | None,
    start: dict[str, Any],
    goal: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    metrics = room_transition_metrics(
        adapter,
        scene,
        source_room,
        target_room,
        start=start,
        goal=goal,
        context=context,
    )
    if not isinstance(metrics, dict):
        return {}
    source_point = metrics.get("source_point")
    target_point = metrics.get("target_point")
    if not isinstance(source_point, tuple) or not isinstance(target_point, tuple):
        return {}

    source_point_world = lift_horizontal_point(adapter, scene, source_point, source_room=source_room, target_room=target_room)
    target_point_world = lift_horizontal_point(adapter, scene, target_point, source_room=source_room, target_room=target_room)
    if source_point_world is None or target_point_world is None:
        return {}

    axes = adapter._horizontal_axes(scene)
    normal_axis_index = metrics.get("normal_axis_index")
    span_axis_index = metrics.get("span_axis_index")
    metadata: dict[str, Any] = {
        "portal_gap": float(metrics.get("gap", 0.0)),
        "portal_span": float(metrics.get("span", 0.0)),
        "portal_source_point": source_point_world,
        "portal_target_point": target_point_world,
    }
    if isinstance(normal_axis_index, int) and 0 <= normal_axis_index < len(axes):
        metadata["portal_normal_axis"] = axes[normal_axis_index]
        metadata["portal_boundary_value"] = float(metrics.get("boundary_value", 0.0))
        source_axis_value = float(source_point[normal_axis_index])
        target_axis_value = float(target_point[normal_axis_index])
        metadata["portal_normal_sign"] = 1 if target_axis_value >= source_axis_value else -1
    if isinstance(span_axis_index, int) and 0 <= span_axis_index < len(axes):
        metadata["portal_span_axis"] = axes[span_axis_index]
        metadata["portal_span_min"] = float(metrics.get("span_min", 0.0))
        metadata["portal_span_max"] = float(metrics.get("span_max", 0.0))
    return metadata


def offset_horizontal_point_along_segment_into_room(
    adapter: Any,
    scene: HOVSGSceneAsset,
    *,
    boundary_point: tuple[float, float],
    toward_point: tuple[float, float],
    room: HOVSGRoomAsset | None,
    fallback_room: HOVSGRoomAsset | None,
) -> dict[str, float] | None:
    polygon = room_polygon_2d(adapter, scene, room) if room is not None else []
    dx = toward_point[0] - boundary_point[0]
    dy = toward_point[1] - boundary_point[1]
    distance = math.hypot(dx, dy)
    if distance <= 1e-6:
        return offset_horizontal_point_into_room(
            adapter,
            scene,
            boundary_point=boundary_point,
            room=room,
            fallback_room=fallback_room,
        )

    offset = min(adapter.portal_target_offset, max(0.0, distance - 1e-3))
    candidate = (
        boundary_point[0] + dx / distance * offset,
        boundary_point[1] + dy / distance * offset,
    )
    if len(polygon) >= 3 and adapter._point_in_polygon(candidate, polygon):
        return lift_horizontal_point(
            adapter,
            scene,
            candidate,
            source_room=fallback_room,
            target_room=room,
        )
    return offset_horizontal_point_into_room(
        adapter,
        scene,
        boundary_point=boundary_point,
        room=room,
        fallback_room=fallback_room,
    )


def offset_horizontal_point_into_room(
    adapter: Any,
    scene: HOVSGSceneAsset,
    *,
    boundary_point: tuple[float, float],
    room: HOVSGRoomAsset | None,
    fallback_room: HOVSGRoomAsset | None,
) -> dict[str, float] | None:
    centroid_room = room if room is not None else fallback_room
    if centroid_room is None or centroid_room.centroid is None:
        return None
    centroid_2d = adapter._project_horizontal(scene, centroid_room.centroid)
    if centroid_2d is None:
        return None

    dx = centroid_2d[0] - boundary_point[0]
    dy = centroid_2d[1] - boundary_point[1]
    distance = math.hypot(dx, dy)
    if distance <= 1e-6:
        horizontal_point = boundary_point
    else:
        offset = min(adapter.portal_target_offset, max(0.0, distance - 1e-3))
        horizontal_point = (
            boundary_point[0] + dx / distance * offset,
            boundary_point[1] + dy / distance * offset,
        )

    return lift_horizontal_point(
        adapter,
        scene,
        horizontal_point,
        source_room=fallback_room,
        target_room=room,
    )


def transition_center_from_bboxes(
    adapter: Any,
    scene: HOVSGSceneAsset,
    source_polygon: list[tuple[float, float]],
    target_polygon: list[tuple[float, float]],
    *,
    source_room: HOVSGRoomAsset,
    target_room: HOVSGRoomAsset,
) -> dict[str, float] | None:
    transition_points = transition_points_from_bboxes(source_polygon, target_polygon)
    if transition_points is None:
        return None
    source_point, target_point = transition_points
    midpoint = (
        (source_point[0] + target_point[0]) * 0.5,
        (source_point[1] + target_point[1]) * 0.5,
    )
    return lift_horizontal_point(
        adapter,
        scene,
        midpoint,
        source_room=source_room,
        target_room=target_room,
    )
