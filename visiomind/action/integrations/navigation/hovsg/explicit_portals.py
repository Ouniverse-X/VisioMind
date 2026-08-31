from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from visiomind.action.shared.models.scene_state import (
    door_is_navigation_passable,
    is_door_category as _is_door_category,
)

from .models import HOVSGRoomAsset, HOVSGSceneAsset
from .portal_primitives import axis_aligned_segment_axes, polygon_segments, room_polygon_2d


def explicit_transition_portal(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    source_room: HOVSGRoomAsset | None,
    target_room: HOVSGRoomAsset | None,
    context: dict[str, Any],
) -> dict[str, Any] | None:
    if source_room is None or target_room is None:
        return None
    portal = _portal_from_annotations(
        source_room=source_room, target_room=target_room, context=context
    )
    if portal is None:
        portal = _portal_from_runtime_state(
            adapter,
            scene=scene,
            source_room=source_room,
            target_room=target_room,
        )
    if portal is None:
        portal = _portal_from_behavior_scene(
            source_room=source_room, target_room=target_room, context=context
        )
    if portal is None:
        return None
    portal = _with_runtime_door_passability(
        adapter,
        scene=scene,
        source_room=source_room,
        target_room=target_room,
        portal=portal,
    )
    waypoint = _portal_waypoint(
        adapter, scene=scene, source_room=source_room, target_room=target_room, portal=portal
    )
    if waypoint is not None and isinstance(portal.get("door_is_open"), bool):
        waypoint["portal_door_open"] = portal["door_is_open"]
    return waypoint


def _with_runtime_door_passability(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    source_room: HOVSGRoomAsset,
    target_room: HOVSGRoomAsset,
    portal: dict[str, Any],
) -> dict[str, Any]:
    from . import runtime_state as hovsg_runtime_state

    candidates = [
        door
        for door in hovsg_runtime_state.door_states(adapter, scene.scene_id).values()
        if _room_pair_matches(door.in_rooms, source_room, target_room)
    ]
    portal_name = _normalize(portal.get("object_name"))
    if portal_name:
        named = [door for door in candidates if _normalize(door.name) == portal_name]
        if len(named) == 1:
            candidates = named
    if len(candidates) != 1:
        return portal
    door = candidates[0]
    navigation_passable = door_is_navigation_passable(door)
    if not isinstance(navigation_passable, bool):
        return portal
    return {
        **portal,
        "object_name": portal.get("object_name") or door.name,
        "door_is_open": navigation_passable,
    }


def _portal_from_runtime_state(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    source_room: HOVSGRoomAsset,
    target_room: HOVSGRoomAsset,
) -> dict[str, Any] | None:
    from . import runtime_state as hovsg_runtime_state

    candidates: list[dict[str, Any]] = []
    for door in hovsg_runtime_state.door_states(adapter, scene.scene_id).values():
        if not _room_pair_matches(door.in_rooms, source_room, target_room):
            continue
        center = _coerce_point(door.position)
        if center is None:
            continue
        candidate: dict[str, Any] = {
            "center": center,
            "source": "runtime_door",
            "object_name": door.name,
        }
        navigation_passable = door_is_navigation_passable(door)
        if isinstance(navigation_passable, bool):
            candidate["door_is_open"] = navigation_passable
        span = _door_span_from_aabb(door.aabb)
        if span is not None:
            candidate["span"] = span
        candidates.append(candidate)
    if not candidates:
        return None
    return min(
        candidates, key=lambda item: _room_midpoint_distance_sq(item, source_room, target_room)
    )


def _door_span_from_aabb(aabb: Any) -> float | None:
    if not isinstance(aabb, dict):
        return None
    corner_min = aabb.get("min")
    corner_max = aabb.get("max")
    if not isinstance(corner_min, list) or not isinstance(corner_max, list):
        return None
    try:
        extents = [float(corner_max[axis]) - float(corner_min[axis]) for axis in (0, 1)]
    except (IndexError, TypeError, ValueError):
        return None
    span = max(extents)
    return span if span > 0.0 else None


def _portal_from_annotations(
    *,
    source_room: HOVSGRoomAsset,
    target_room: HOVSGRoomAsset,
    context: dict[str, Any],
) -> dict[str, Any] | None:
    annotations = _context_portal_annotations(context)
    entries = (
        annotations
        if isinstance(annotations, list)
        else [annotations]
        if isinstance(annotations, dict)
        else []
    )
    if isinstance(annotations, dict) and all(
        isinstance(value, dict) for value in annotations.values()
    ):
        entries = list(annotations.values())
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        match_direction = _entry_room_match_direction(entry, source_room, target_room)
        if match_direction is None:
            continue
        center = _point_from_keys(entry, ("center", "position", "midpoint", "portal_center"))
        source_point = _point_from_keys(
            entry, ("source_point", "source_anchor", "portal_source_point")
        )
        target_point = _point_from_keys(
            entry, ("target_point", "target_anchor", "portal_target_point")
        )
        if match_direction == "reverse":
            source_point, target_point = target_point, source_point
        if center is None and source_point is not None and target_point is not None:
            center = {
                "x": (source_point["x"] + target_point["x"]) * 0.5,
                "y": (source_point["y"] + target_point["y"]) * 0.5,
                "z": (source_point["z"] + target_point["z"]) * 0.5,
            }
        if center is not None:
            return {
                "center": center,
                "source": "annotation",
                "object_name": entry.get("name")
                or entry.get("object_name")
                or entry.get("portal_object_name"),
                "span": entry.get("span") or entry.get("width") or entry.get("portal_span"),
                "source_point": source_point,
                "target_point": target_point,
            }
    return None


def _portal_from_behavior_scene(
    *,
    source_room: HOVSGRoomAsset,
    target_room: HOVSGRoomAsset,
    context: dict[str, Any],
) -> dict[str, Any] | None:
    scene_file = _context_scene_file(context)
    if not isinstance(scene_file, str) or not scene_file.strip():
        return None
    try:
        payload = json.loads(Path(scene_file).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    object_registry = _object_registry(payload)
    candidates: list[dict[str, Any]] = []
    for object_key, object_info in _objects_init_info(payload).items():
        if not isinstance(object_info, dict):
            continue
        args = object_info.get("args") if isinstance(object_info.get("args"), dict) else {}
        if not _is_door_category(args.get("category") or object_info.get("category")):
            continue
        if not _room_pair_matches(
            args.get("in_rooms") or object_info.get("in_rooms"), source_room, target_room
        ):
            continue
        object_name = args.get("name") if isinstance(args.get("name"), str) else str(object_key)
        state = object_registry.get(object_name)
        if not isinstance(state, dict):
            state = object_registry.get(str(object_key))
        center = _runtime_position(state) or _point_from_keys(args, ("position", "center", "pos"))
        if center is not None:
            candidates.append(
                {"center": center, "source": "explicit_door", "object_name": object_name}
            )
    if not candidates:
        return None
    return min(
        candidates, key=lambda item: _room_midpoint_distance_sq(item, source_room, target_room)
    )


def _context_scene_file(context: dict[str, Any]) -> str | None:
    for container in _context_containers(context):
        for key in ("behavior_scene_file", "scene_file"):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _context_portal_annotations(context: dict[str, Any]) -> Any:
    for container in _context_containers(context):
        for key in ("transition_portals", "portal_annotations", "portals"):
            value = container.get(key)
            if isinstance(value, (dict, list)) and value:
                return value
    return None


def _context_containers(context: dict[str, Any]) -> list[dict[str, Any]]:
    containers: list[dict[str, Any]] = []
    if isinstance(context, dict):
        containers.append(context)
        for key in ("parameters", "map_state", "context", "observation"):
            value = context.get(key)
            if isinstance(value, dict):
                containers.append(value)
    return containers


def _portal_waypoint(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    source_room: HOVSGRoomAsset,
    target_room: HOVSGRoomAsset,
    portal: dict[str, Any],
) -> dict[str, Any] | None:
    center = _coerce_point(portal.get("center"))
    if center is None:
        return None
    center_2d = adapter._project_horizontal(scene, center)
    if center_2d is None:
        return None
    _snap_vertical_to_rooms(scene, center, source_room, target_room)

    axes = adapter._horizontal_axes(scene)
    default_normal_axis, default_span_axis, default_normal_sign = _portal_axes(
        adapter,
        scene,
        source_room,
        target_room,
        center_2d=center_2d,
    )
    offset = max(0.0, float(getattr(adapter, "portal_target_offset", 0.35)))
    span = _positive_float(portal.get("span")) or 1.0

    source_point = _coerce_point(portal.get("source_point")) or _offset_point(
        scene,
        center,
        default_normal_axis,
        -default_normal_sign * offset,
    )
    target_point = _coerce_point(portal.get("target_point")) or _offset_point(
        scene,
        center,
        default_normal_axis,
        default_normal_sign * offset,
    )
    if source_point is None or target_point is None:
        return None
    normal_axis, span_axis, normal_sign = _portal_axes_from_points(
        adapter, scene, source_point, target_point
    ) or (
        default_normal_axis,
        default_span_axis,
        default_normal_sign,
    )
    normal_index = axes.index(normal_axis)
    span_index = axes.index(span_axis)

    waypoint = {
        "x": float(target_point["x"]),
        "y": float(target_point["y"]),
        "z": float(target_point["z"]),
        "floor_id": target_room.floor_id,
        "room_id": target_room.room_id,
        "source_room_id": source_room.room_id,
        "source_room_name": source_room.name,
        "room_name": target_room.name,
        "waypoint_type": "portal",
        "portal_source": portal.get("source") or "explicit",
        "portal_source_point": source_point,
        "portal_target_point": target_point,
        "portal_gap": 0.0,
        "portal_span": span,
        "portal_span_axis": span_axis,
        "portal_span_min": float(center_2d[span_index] - span * 0.5),
        "portal_span_max": float(center_2d[span_index] + span * 0.5),
        "portal_normal_axis": normal_axis,
        "portal_boundary_value": float(center_2d[normal_index]),
        "portal_normal_sign": normal_sign,
        "portal_desired_heading": _heading(
            axes=axes, normal_axis=normal_axis, normal_sign=normal_sign
        ),
        "portal_alignment_stage": "target_anchor",
    }
    if isinstance(portal.get("object_name"), str) and portal["object_name"]:
        waypoint["portal_object_name"] = portal["object_name"]
    return waypoint


def _portal_axes(
    adapter: Any,
    scene: HOVSGSceneAsset,
    source_room: HOVSGRoomAsset,
    target_room: HOVSGRoomAsset,
    *,
    center_2d: tuple[float, float],
) -> tuple[str, str, int]:
    axes = adapter._horizontal_axes(scene)
    boundary_axes = _portal_axes_from_room_boundaries(
        adapter,
        scene=scene,
        source_room=source_room,
        target_room=target_room,
        center_2d=center_2d,
    )
    if boundary_axes is not None:
        normal_index, span_index, normal_sign = boundary_axes
        return axes[normal_index], axes[span_index], normal_sign

    source_centroid = adapter._project_horizontal(scene, source_room.centroid or {})
    target_centroid = adapter._project_horizontal(scene, target_room.centroid or {})
    if source_centroid is None or target_centroid is None:
        return axes[0], axes[1], 1
    delta = (target_centroid[0] - source_centroid[0], target_centroid[1] - source_centroid[1])
    normal_index = 0 if abs(delta[0]) >= abs(delta[1]) else 1
    return axes[normal_index], axes[1 - normal_index], 1 if delta[normal_index] >= 0.0 else -1


def _portal_axes_from_room_boundaries(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    source_room: HOVSGRoomAsset,
    target_room: HOVSGRoomAsset,
    center_2d: tuple[float, float],
) -> tuple[int, int, int] | None:
    source_polygon = room_polygon_2d(adapter, scene, source_room)
    target_polygon = room_polygon_2d(adapter, scene, target_room)
    best: tuple[tuple[float, float, float, float], int, int, float, float] | None = None
    for source_start, source_end in polygon_segments(source_polygon):
        source_axes = axis_aligned_segment_axes(source_start, source_end)
        if source_axes is None:
            continue
        for target_start, target_end in polygon_segments(target_polygon):
            target_axes = axis_aligned_segment_axes(target_start, target_end)
            if target_axes != source_axes:
                continue
            span_index, normal_index = source_axes
            span_min = max(
                min(source_start[span_index], source_end[span_index]),
                min(target_start[span_index], target_end[span_index]),
            )
            span_max = min(
                max(source_start[span_index], source_end[span_index]),
                max(target_start[span_index], target_end[span_index]),
            )
            if span_max < span_min:
                continue

            center_span = float(center_2d[span_index])
            span_distance = max(span_min - center_span, 0.0, center_span - span_max)
            source_normal = (
                float(source_start[normal_index]) + float(source_end[normal_index])
            ) * 0.5
            target_normal = (
                float(target_start[normal_index]) + float(target_end[normal_index])
            ) * 0.5
            boundary_value = (source_normal + target_normal) * 0.5
            boundary_distance = abs(float(center_2d[normal_index]) - boundary_value)
            gap = abs(target_normal - source_normal)
            overlap = max(0.0, span_max - span_min)
            score = (
                math.hypot(span_distance, boundary_distance),
                span_distance,
                gap,
                -overlap,
            )
            if best is None or score < best[0]:
                best = (score, normal_index, span_index, source_normal, target_normal)

    if best is None:
        return None
    _, normal_index, span_index, source_normal, target_normal = best
    delta = target_normal - source_normal
    if abs(delta) <= 1e-6:
        source_centroid = adapter._project_horizontal(scene, source_room.centroid or {})
        target_centroid = adapter._project_horizontal(scene, target_room.centroid or {})
        if source_centroid is None or target_centroid is None:
            normal_sign = 1
        else:
            normal_sign = (
                1 if target_centroid[normal_index] >= source_centroid[normal_index] else -1
            )
    else:
        normal_sign = 1 if delta >= 0.0 else -1
    return normal_index, span_index, normal_sign


def _portal_axes_from_points(
    adapter: Any,
    scene: HOVSGSceneAsset,
    source_point: dict[str, Any],
    target_point: dict[str, Any],
) -> tuple[str, str, int] | None:
    axes = adapter._horizontal_axes(scene)
    source_2d = adapter._project_horizontal(scene, source_point)
    target_2d = adapter._project_horizontal(scene, target_point)
    if source_2d is None or target_2d is None:
        return None
    delta = (float(target_2d[0]) - float(source_2d[0]), float(target_2d[1]) - float(source_2d[1]))
    if abs(delta[0]) < 1e-9 and abs(delta[1]) < 1e-9:
        return None
    normal_index = 0 if abs(delta[0]) >= abs(delta[1]) else 1
    return axes[normal_index], axes[1 - normal_index], 1 if delta[normal_index] >= 0.0 else -1


def _entry_matches_rooms(
    entry: dict[str, Any], source_room: HOVSGRoomAsset, target_room: HOVSGRoomAsset
) -> bool:
    return _entry_room_match_direction(entry, source_room, target_room) is not None


def _entry_room_match_direction(
    entry: dict[str, Any],
    source_room: HOVSGRoomAsset,
    target_room: HOVSGRoomAsset,
) -> str | None:
    source_values = (
        entry.get("source_room"),
        entry.get("source_room_id"),
        entry.get("source_room_name"),
    )
    target_values = (
        entry.get("target_room"),
        entry.get("target_room_id"),
        entry.get("target_room_name"),
    )
    if _room_value_matches(source_values, source_room) and _room_value_matches(
        target_values, target_room
    ):
        return "forward"
    if _room_value_matches(source_values, target_room) and _room_value_matches(
        target_values, source_room
    ):
        return "reverse"
    if _room_pair_matches(
        entry.get("in_rooms") or entry.get("rooms") or entry.get("room_names"),
        source_room,
        target_room,
    ):
        return "forward"
    return None


def _room_pair_matches(
    in_rooms: Any, source_room: HOVSGRoomAsset, target_room: HOVSGRoomAsset
) -> bool:
    if not isinstance(in_rooms, list):
        return False
    tokens = {_normalize(item) for item in in_rooms} - {""}
    return bool(tokens & _room_tokens(source_room)) and bool(tokens & _room_tokens(target_room))


def _room_value_matches(values: tuple[Any, ...], room: HOVSGRoomAsset) -> bool:
    room_tokens = _room_tokens(room)
    return any(_normalize(value) in room_tokens for value in values)


def _room_tokens(room: HOVSGRoomAsset) -> set[str]:
    return {_normalize(room.room_id), _normalize(room.name)} - {""}


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower()


def _objects_init_info(payload: dict[str, Any]) -> dict[str, Any]:
    objects_info = payload.get("objects_info")
    if isinstance(objects_info, dict) and isinstance(objects_info.get("init_info"), dict):
        return objects_info["init_info"]
    init_info = payload.get("init_info")
    if not isinstance(init_info, dict):
        return {}
    return {
        key: value
        for key, value in init_info.items()
        if isinstance(value, dict) and isinstance(value.get("args"), dict)
    }


def _object_registry(payload: dict[str, Any]) -> dict[str, Any]:
    state = payload.get("state")
    registry = state.get("registry") if isinstance(state, dict) else None
    object_registry = registry.get("object_registry") if isinstance(registry, dict) else None
    return object_registry if isinstance(object_registry, dict) else {}


def _runtime_position(state: Any) -> dict[str, float] | None:
    if not isinstance(state, dict):
        return None
    root_link = state.get("root_link")
    if isinstance(root_link, dict):
        point = _coerce_point(root_link.get("pos") or root_link.get("position"))
        if point is not None:
            return point
    return _point_from_keys(state, ("position", "pos", "center"))


def _point_from_keys(mapping: Any, keys: tuple[str, ...]) -> dict[str, float] | None:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        point = _coerce_point(mapping.get(key))
        if point is not None:
            return point
    return None


def _coerce_point(value: Any) -> dict[str, float] | None:
    if isinstance(value, dict):
        x_coord = _to_float(value.get("x"))
        y_coord = _to_float(value.get("y"))
        z_coord = _to_float(value.get("z", 0.0))
    elif isinstance(value, (list, tuple)) and len(value) >= 3:
        x_coord = _to_float(value[0])
        y_coord = _to_float(value[1])
        z_coord = _to_float(value[2])
    else:
        return None
    if x_coord is None or y_coord is None or z_coord is None:
        return None
    return {"x": x_coord, "y": y_coord, "z": z_coord}


def _snap_vertical_to_rooms(
    scene: HOVSGSceneAsset,
    point: dict[str, float],
    source_room: HOVSGRoomAsset,
    target_room: HOVSGRoomAsset,
) -> None:
    values = [
        value
        for value in (
            _to_float((room.centroid or {}).get(scene.vertical_axis))
            for room in (source_room, target_room)
        )
        if value is not None
    ]
    if values:
        point[scene.vertical_axis] = sum(values) / float(len(values))


def _offset_point(
    scene: HOVSGSceneAsset, center: dict[str, float], axis: str, delta: float
) -> dict[str, float]:
    point = dict(center)
    point[axis] = float(point[axis]) + float(delta)
    point[scene.vertical_axis] = float(center[scene.vertical_axis])
    return point


def _heading(*, axes: tuple[str, str], normal_axis: str, normal_sign: int) -> float:
    if normal_axis == axes[0]:
        return 0.0 if normal_sign >= 0 else math.pi
    return math.pi * 0.5 if normal_sign >= 0 else -math.pi * 0.5


def _room_midpoint_distance_sq(
    portal: dict[str, Any], source_room: HOVSGRoomAsset, target_room: HOVSGRoomAsset
) -> float:
    center = portal.get("center")
    if not isinstance(center, dict) or source_room.centroid is None or target_room.centroid is None:
        return 0.0
    midpoint = {
        "x": (float(source_room.centroid["x"]) + float(target_room.centroid["x"])) * 0.5,
        "y": (float(source_room.centroid["y"]) + float(target_room.centroid["y"])) * 0.5,
        "z": (float(source_room.centroid["z"]) + float(target_room.centroid["z"])) * 0.5,
    }
    return sum((float(center[axis]) - midpoint[axis]) ** 2 for axis in ("x", "y", "z"))


def _positive_float(value: Any) -> float | None:
    number = _to_float(value)
    return number if number is not None and number > 0.0 else None


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
