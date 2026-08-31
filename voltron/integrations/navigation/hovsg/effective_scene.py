from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from voltron.shared.models.scene_state import (
    NAVIGATION_ROLE_OBSTACLE,
    NON_BLOCKING_NAVIGATION_ROLES,
    RuntimeObjectState,
    RuntimeRelationState,
    compute_relation_signature,
    navigation_role_from_category,
)

from . import runtime_state as hovsg_runtime_state
from .models import HOVSGObjectAsset, HOVSGSceneAsset

_EXCLUSIVE_RELATIONS = {
    "attached_to",
    "held_by",
    "in_room",
    "inside",
    "on_floor",
    "on_top",
}
_PORTAL_TOKENS = {"door", "doorway", "gate", "opening", "portal"}
_RELATION_SOURCE_PRIORITY = {
    "simulator": 500,
    "deterministic": 500,
    "task_state": 450,
    "attachment": 450,
    "geometry": 300,
    "perception": 220,
    "vlm": 200,
    "static_hovsg": 100,
}
_CATEGORY_RADIUS_M = {
    "digital_camera": 0.12,
    "camera": 0.12,
    "box": 0.35,
    "carton": 0.35,
    "robot": 0.45,
}


@dataclass
class EffectiveObjectState:
    object_id: str
    name: str
    category: str | None
    position: dict[str, float] | None
    room_id: str | None
    floor_id: str | None
    footprint: list[tuple[float, float]] = field(default_factory=list)
    footprints: list[list[tuple[float, float]]] = field(default_factory=list)
    navigation_footprints: list[list[tuple[float, float]]] = field(default_factory=list)
    collision_parts: list[dict[str, Any]] = field(default_factory=list)
    geometry_source: str = "category_circle"
    static_object_id: str | None = None
    runtime_object_name: str | None = None
    navigation_role: str | None = NAVIGATION_ROLE_OBSTACLE
    relations: list[RuntimeRelationState] = field(default_factory=list)
    covariance_xy: list[list[float]] | None = None
    uncertainty_margin_m: float = 0.0
    pose_revision: str = ""
    geometry_revision: str = ""
    relation_revision: str = ""
    map_revision: str = ""
    runtime_only: bool = False

    def relation_targets(self, relation: str) -> list[str]:
        return [
            item.object_id
            for item in self.relations
            if item.relation == relation and item.object_id is not None
        ]

    @property
    def held_by_self(self) -> bool:
        return any(
            item.relation == "held_by" and is_self_robot_id(item.object_id)
            for item in self.relations
        )

    @property
    def is_robot(self) -> bool:
        return is_robot_label(self.category) or is_robot_label(self.name)

    @property
    def participates_in_navigation(self) -> bool:
        return self.navigation_role not in NON_BLOCKING_NAVIGATION_ROLES and not self.held_by_self

    @property
    def approachable(self) -> bool:
        return not self.held_by_self and not self.is_robot


@dataclass
class EffectiveSceneView:
    scene_id: str
    objects: dict[str, EffectiveObjectState]
    relations: list[RuntimeRelationState]
    relation_signature: str
    map_revision: str
    aliases: dict[str, str] = field(default_factory=dict)

    def resolve_object(self, object_id: Any) -> EffectiveObjectState | None:
        key = str(object_id or "").strip()
        if not key:
            return None
        canonical = self.aliases.get(key, key)
        return self.objects.get(canonical)


def effective_scene_view(
    adapter: Any,
    scene: HOVSGSceneAsset,
    *,
    scene_id: str | None = None,
) -> EffectiveSceneView:
    resolved_scene_id = str(scene_id or getattr(scene, "scene_id", "") or "")
    state = hovsg_runtime_state.current_scene_state(adapter, resolved_scene_id)
    cache = getattr(adapter, "_effective_scene_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        adapter._effective_scene_cache = cache
    state_signature = state.signature if state is not None else ""
    relation_signature = (
        state.relation_signature
        or compute_relation_signature(state.relations, current_step=state.step)
        if state is not None
        else ""
    )
    cache_key = (resolved_scene_id, state_signature, relation_signature)
    cached = cache.get(cache_key)
    if isinstance(cached, EffectiveSceneView):
        return cached

    objects: dict[str, EffectiveObjectState] = {}
    aliases: dict[str, str] = {}
    room_aliases = _room_aliases(scene)
    floor_aliases = _floor_aliases(scene)
    matched_runtime_names: set[str] = set()
    runtime_objects = state.objects if state is not None else {}

    for static_object in getattr(scene, "objects", {}).values():
        runtime_object = (
            hovsg_runtime_state.match_runtime_object(
                state,
                object_name=static_object.name,
                object_id=static_object.object_id,
                static_centroid=static_object.centroid,
            )
            if state is not None
            else None
        )
        if runtime_object is not None:
            matched_runtime_names.add(runtime_object.name)
        effective = _effective_object(
            adapter,
            scene=scene,
            static_object=static_object,
            runtime_object=runtime_object,
        )
        objects[effective.object_id] = effective
        _register_aliases(aliases, effective)
        if state is not None:
            runtime_door = hovsg_runtime_state.match_runtime_door(
                state,
                object_name=static_object.name,
                object_id=static_object.object_id,
                static_centroid=static_object.centroid,
            )
            if runtime_door is not None:
                _register_alias(aliases, runtime_door.name, effective.object_id)

    for runtime_object in runtime_objects.values():
        if runtime_object.name in matched_runtime_names:
            continue
        effective = _effective_object(
            adapter,
            scene=scene,
            static_object=None,
            runtime_object=runtime_object,
        )
        objects[effective.object_id] = effective
        _register_aliases(aliases, effective)

    relation_candidates = _static_relations(scene, objects)
    relation_candidates.extend(
        _derived_location_relations(objects, state_step=state.step if state else 0)
    )
    if state is not None:
        relation_candidates.extend(
            _canonicalize_runtime_relations(
                state.relations,
                aliases=aliases,
                room_aliases=room_aliases,
                floor_aliases=floor_aliases,
                current_step=state.step,
            )
        )
    relation_candidates.extend(_connected_room_relations(scene))
    relation_candidates.extend(_portal_blocking_relations(scene, objects))
    relations = merge_relations(relation_candidates)
    relations.extend(_inverse_relations(relations))
    relations = merge_relations(relations)
    effective_relation_signature = compute_relation_signature(relations)

    relations_by_subject: dict[str, list[RuntimeRelationState]] = {}
    for relation in relations:
        relations_by_subject.setdefault(relation.subject_id, []).append(relation)
    for object_id, effective in objects.items():
        effective.relations = list(relations_by_subject.get(object_id, []))
        effective.relation_revision = _relation_revision(effective.relations)
        effective.map_revision = _object_map_revision(effective)

    map_revision = hashlib.sha1(
        "|".join(
            [
                *(objects[key].map_revision for key in sorted(objects)),
                effective_relation_signature,
                state.door_signature() if state is not None else "",
            ]
        ).encode("utf-8")
    ).hexdigest()[:16]
    view = EffectiveSceneView(
        scene_id=resolved_scene_id,
        objects=objects,
        relations=relations,
        relation_signature=effective_relation_signature,
        map_revision=map_revision,
        aliases=aliases,
    )
    cache.clear()
    cache[cache_key] = view
    return view


def approach_owner(
    view: EffectiveSceneView,
    target: EffectiveObjectState,
) -> EffectiveObjectState | None:
    if not target.approachable:
        return None
    for relation_name in ("on_top", "inside"):
        for owner_id in target.relation_targets(relation_name):
            owner = view.resolve_object(owner_id)
            if owner is not None and not owner.held_by_self:
                return owner
    return target


def merge_relations(
    relations: list[RuntimeRelationState],
) -> list[RuntimeRelationState]:
    active: dict[tuple[str, str, str], RuntimeRelationState] = {}
    tombstones: dict[tuple[str, str, str], int] = {}
    for relation in relations:
        if not relation.subject_id or not relation.relation:
            continue
        key = (relation.subject_id, relation.relation, relation.object_id or "")
        priority = relation_source_priority(relation.source)
        if relation.removed:
            tombstones[key] = max(priority, tombstones.get(key, -1))
            continue
        if tombstones.get(key, -1) >= priority:
            continue
        previous = active.get(key)
        if previous is None or _relation_sort_key(relation) > _relation_sort_key(previous):
            active[key] = relation

    for key, tombstone_priority in tombstones.items():
        existing = active.get(key)
        if existing is not None and relation_source_priority(existing.source) <= tombstone_priority:
            active.pop(key, None)

    grouped: dict[tuple[str, str], list[RuntimeRelationState]] = {}
    for relation in active.values():
        group_relation = (
            "open_state" if relation.relation in {"open", "closed"} else relation.relation
        )
        grouped.setdefault((relation.subject_id, group_relation), []).append(relation)
    merged: list[RuntimeRelationState] = []
    for (_subject, relation_name), candidates in grouped.items():
        if relation_name in _EXCLUSIVE_RELATIONS or relation_name == "open_state":
            merged.append(max(candidates, key=_relation_sort_key))
        else:
            merged.extend(candidates)
    return sorted(
        merged,
        key=lambda item: (item.subject_id, item.relation, item.object_id or ""),
    )


def relation_source_priority(source: str | None) -> int:
    normalized = str(source or "").strip().lower()
    for token, priority in _RELATION_SOURCE_PRIORITY.items():
        if token in normalized:
            return priority
    return 250


def is_self_robot_id(value: Any) -> bool:
    normalized = _normalize(value)
    return normalized in {"self", "self_robot", "controlled_robot", "robot_self"}


def is_robot_label(value: Any) -> bool:
    tokens = set(_normalize(value).split("_"))
    return "robot" in tokens or "humanoid" in tokens


def _effective_object(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    static_object: HOVSGObjectAsset | None,
    runtime_object: RuntimeObjectState | None,
) -> EffectiveObjectState:
    runtime_only = static_object is None
    object_id = (
        f"runtime:{runtime_object.name}"
        if runtime_only
        and runtime_object is not None
        and not runtime_object.name.startswith("runtime:")
        else runtime_object.name
        if runtime_only and runtime_object is not None
        else str(static_object.object_id)
    )
    name = str(
        runtime_object.name
        if runtime_only and runtime_object is not None
        else getattr(static_object, "name", None) or getattr(static_object, "object_id", object_id)
    )
    category = (
        runtime_object.category
        if runtime_object is not None and runtime_object.category
        else getattr(static_object, "name", None)
        if static_object is not None
        else None
    )
    position = (
        dict(runtime_object.position)
        if runtime_object is not None and runtime_object.position is not None
        else dict(static_object.centroid)
        if static_object is not None and getattr(static_object, "centroid", None) is not None
        else None
    )
    room_id = _effective_room_id(
        adapter,
        scene=scene,
        static_object=static_object,
        runtime_object=runtime_object,
        position=position,
    )
    rooms = getattr(scene, "rooms", {})
    room = rooms.get(room_id) if room_id is not None else None
    floor_id = (
        runtime_object.floor_hint
        if runtime_object is not None and runtime_object.floor_hint
        else room.floor_id
        if room is not None
        else None
    )
    floor_height = _floor_height(scene, floor_id)
    footprints, navigation_footprints, geometry_source = _effective_footprints(
        adapter,
        scene=scene,
        static_object=static_object,
        runtime_object=runtime_object,
        position=position,
        floor_height=floor_height,
        category=category,
    )
    covariance = runtime_object.covariance_xy if runtime_object is not None else None
    uncertainty_margin = _uncertainty_margin(covariance)
    if uncertainty_margin > 0.0:
        footprints = [_expand_polygon(item, uncertainty_margin) for item in footprints]
        navigation_footprints = [
            _expand_polygon(item, uncertainty_margin) for item in navigation_footprints
        ]
    footprint = convex_hull([point for polygon in footprints for point in polygon])
    pose_revision = _hash_payload(position or {})
    geometry_revision = _hash_payload(
        {
            "source": geometry_source,
            "footprints": footprints,
            "navigation_footprints": navigation_footprints,
        }
    )
    return EffectiveObjectState(
        object_id=str(object_id),
        name=name,
        category=category,
        position=position,
        room_id=room_id,
        floor_id=floor_id,
        footprint=footprint,
        footprints=footprints,
        navigation_footprints=navigation_footprints,
        collision_parts=(
            [dict(part) for part in runtime_object.collision_parts]
            if runtime_object is not None
            else []
        ),
        geometry_source=geometry_source,
        static_object_id=getattr(static_object, "object_id", None),
        runtime_object_name=runtime_object.name if runtime_object is not None else None,
        navigation_role=(
            runtime_object.navigation_role
            if runtime_object is not None and runtime_object.navigation_role
            else getattr(static_object, "navigation_role", None)
            or navigation_role_from_category(category or name)
        ),
        covariance_xy=covariance,
        uncertainty_margin_m=uncertainty_margin,
        pose_revision=pose_revision,
        geometry_revision=geometry_revision,
        runtime_only=runtime_only,
    )


def _effective_footprints(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    static_object: HOVSGObjectAsset | None,
    runtime_object: RuntimeObjectState | None,
    position: dict[str, float] | None,
    floor_height: float | None,
    category: str | None,
) -> tuple[list[list[tuple[float, float]]], list[list[tuple[float, float]]], str]:
    if runtime_object is not None and runtime_object.collision_parts:
        vertical_axis = str(getattr(scene, "vertical_axis", "z") or "z")
        all_polygons = _collision_part_polygons(
            runtime_object.collision_parts,
            vertical_axis=vertical_axis,
        )
        navigation_polygons = _collision_part_polygons(
            runtime_object.collision_parts,
            floor_height=floor_height,
            vertical_axis=vertical_axis,
        )
        if all_polygons:
            return all_polygons, navigation_polygons, "collision_polygon"
    if runtime_object is not None and runtime_object.aabb is not None:
        polygon = _project_aabb(adapter, scene, runtime_object.aabb)
        if polygon:
            navigation = (
                [polygon]
                if _part_overlaps_navigation_height(
                    runtime_object.aabb,
                    floor_height,
                    vertical_axis=str(getattr(scene, "vertical_axis", "z") or "z"),
                )
                else []
            )
            return [polygon], navigation, "runtime_aabb"
    if runtime_object is not None and runtime_object.oriented_bbox is not None:
        polygon = _oriented_bbox_polygon(adapter, scene, runtime_object.oriented_bbox)
        if polygon:
            navigation = (
                [polygon]
                if _oriented_bbox_overlaps_navigation_height(
                    runtime_object.oriented_bbox,
                    floor_height=floor_height,
                    vertical_axis=str(getattr(scene, "vertical_axis", "z") or "z"),
                )
                else []
            )
            return [polygon], navigation, "oriented_bbox"
    static_polygon = _static_polygon(adapter, scene, static_object)
    if static_polygon:
        if (
            runtime_object is not None
            and runtime_object.position is not None
            and static_object is not None
            and getattr(static_object, "centroid", None) is not None
        ):
            static_xy = adapter._project_horizontal(scene, static_object.centroid)
            runtime_xy = adapter._project_horizontal(scene, runtime_object.position)
            if static_xy is not None and runtime_xy is not None:
                static_polygon = [
                    (
                        point[0] + runtime_xy[0] - static_xy[0],
                        point[1] + runtime_xy[1] - static_xy[1],
                    )
                    for point in static_polygon
                ]
        navigation = (
            [static_polygon]
            if _static_overlaps_navigation_height(
                scene,
                static_object,
                floor_height,
            )
            else []
        )
        return [static_polygon], navigation, "semantic_polygon"
    circle = _category_circle(adapter, scene, position, category)
    return ([circle], [circle], "category_circle") if circle else ([], [], "none")


def _collision_part_polygons(
    parts: list[dict[str, Any]],
    *,
    floor_height: float | None = None,
    vertical_axis: str = "z",
) -> list[list[tuple[float, float]]]:
    polygons: list[list[tuple[float, float]]] = []
    for part in parts:
        part_vertical_axis = str(part.get("vertical_axis") or "").lower()
        if part_vertical_axis and part_vertical_axis != vertical_axis:
            continue
        if floor_height is not None and not _part_overlaps_navigation_height(
            part,
            floor_height,
            vertical_axis=vertical_axis,
        ):
            continue
        raw_polygons = part.get("world_polygons") if isinstance(part, dict) else None
        if not isinstance(raw_polygons, (list, tuple)):
            continue
        for raw_polygon in raw_polygons:
            polygon = _coerce_polygon(raw_polygon)
            if polygon:
                polygons.append(polygon)
    return polygons


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
        minimum = part.get("min")
        maximum = part.get("max")
        if not isinstance(minimum, (list, tuple)) or not isinstance(maximum, (list, tuple)):
            return True
        try:
            height_index = {"x": 0, "y": 1, "z": 2}.get(vertical_axis, 2)
            height_min = float(minimum[height_index])
            height_max = float(maximum[height_index])
        except (IndexError, TypeError, ValueError):
            return True
    return (
        float(height_max) >= floor_height - floor_slop_m
        and float(height_min) <= floor_height + robot_clearance_height_m
    )


def _project_aabb(
    adapter: Any,
    scene: HOVSGSceneAsset,
    aabb: dict[str, list[float]],
) -> list[tuple[float, float]]:
    minimum = aabb.get("min")
    maximum = aabb.get("max")
    if not isinstance(minimum, (list, tuple)) or not isinstance(maximum, (list, tuple)):
        return []
    points = []
    for x in (minimum[0], maximum[0]):
        for y in (minimum[1], maximum[1]):
            for z in (minimum[2], maximum[2]):
                projected = adapter._project_horizontal(
                    scene,
                    {"x": float(x), "y": float(y), "z": float(z)},
                )
                if projected is not None:
                    points.append((float(projected[0]), float(projected[1])))
    return convex_hull(points)


def _oriented_bbox_polygon(
    adapter: Any,
    scene: HOVSGSceneAsset,
    bbox: dict[str, Any],
) -> list[tuple[float, float]]:
    corners = bbox.get("world_corners")
    if isinstance(corners, list):
        projected = [adapter._project_horizontal(scene, point) for point in corners]
        return convex_hull(
            [(float(point[0]), float(point[1])) for point in projected if point is not None]
        )
    center = bbox.get("center")
    size = bbox.get("size")
    half_extents = bbox.get("half_extents")
    if not isinstance(center, dict):
        return []
    extents = half_extents if isinstance(half_extents, list) else size
    if not isinstance(extents, list) or len(extents) < 2:
        return []
    center_xy = adapter._project_horizontal(scene, center)
    if center_xy is None:
        return []
    scale = 1.0 if half_extents is not None else 0.5
    horizontal_axes = {
        "x": (1, 2),
        "y": (0, 2),
        "z": (0, 1),
    }.get(str(getattr(scene, "vertical_axis", "z") or "z"), (0, 1))
    if len(extents) >= 3:
        half_x = abs(float(extents[horizontal_axes[0]])) * scale
        half_y = abs(float(extents[horizontal_axes[1]])) * scale
    else:
        half_x = abs(float(extents[0])) * scale
        half_y = abs(float(extents[1])) * scale
    yaw = float(bbox.get("yaw") or 0.0)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    polygon = []
    for local_x, local_y in (
        (-half_x, -half_y),
        (half_x, -half_y),
        (half_x, half_y),
        (-half_x, half_y),
    ):
        polygon.append(
            (
                float(center_xy[0]) + local_x * cosine - local_y * sine,
                float(center_xy[1]) + local_x * sine + local_y * cosine,
            )
        )
    return polygon


def _oriented_bbox_overlaps_navigation_height(
    bbox: dict[str, Any],
    *,
    floor_height: float | None,
    vertical_axis: str,
    robot_clearance_height_m: float = 1.6,
    floor_slop_m: float = 0.15,
) -> bool:
    if floor_height is None:
        return True
    corners = bbox.get("world_corners")
    height_index = {"x": 0, "y": 1, "z": 2}.get(vertical_axis, 2)
    if isinstance(corners, list):
        values = []
        for corner in corners:
            if not isinstance(corner, dict):
                continue
            try:
                values.append(float(corner[vertical_axis]))
            except (KeyError, TypeError, ValueError):
                continue
        if values:
            return (
                max(values) >= floor_height - floor_slop_m
                and min(values) <= floor_height + robot_clearance_height_m
            )
    center = bbox.get("center")
    extents = bbox.get("half_extents")
    scale = 1.0
    if not isinstance(extents, list):
        extents = bbox.get("size")
        scale = 0.5
    if not isinstance(center, dict) or not isinstance(extents, list) or len(extents) < 3:
        return True
    try:
        center_height = float(center[vertical_axis])
        half_height = abs(float(extents[height_index])) * scale
    except (KeyError, TypeError, ValueError):
        return True
    return (
        center_height + half_height >= floor_height - floor_slop_m
        and center_height - half_height <= floor_height + robot_clearance_height_m
    )


def _static_polygon(
    adapter: Any,
    scene: HOVSGSceneAsset,
    static_object: HOVSGObjectAsset | None,
) -> list[tuple[float, float]]:
    if static_object is None:
        return []
    points = []
    for vertex in static_object.vertices:
        if not isinstance(vertex, (list, tuple)) or len(vertex) < 3:
            continue
        projected = adapter._project_horizontal(
            scene,
            {"x": vertex[0], "y": vertex[1], "z": vertex[2]},
        )
        if projected is not None:
            points.append((float(projected[0]), float(projected[1])))
    return convex_hull(points)


def _category_circle(
    adapter: Any,
    scene: HOVSGSceneAsset,
    position: dict[str, float] | None,
    category: str | None,
) -> list[tuple[float, float]]:
    if position is None:
        return []
    center = adapter._project_horizontal(scene, position)
    if center is None:
        return []
    normalized = _normalize(category)
    radius = next(
        (value for token, value in _CATEGORY_RADIUS_M.items() if token in normalized),
        0.3,
    )
    return [
        (
            float(center[0]) + radius * math.cos(2.0 * math.pi * index / 12.0),
            float(center[1]) + radius * math.sin(2.0 * math.pi * index / 12.0),
        )
        for index in range(12)
    ]


def _effective_room_id(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    static_object: HOVSGObjectAsset | None,
    runtime_object: RuntimeObjectState | None,
    position: dict[str, float] | None,
) -> str | None:
    rooms = getattr(scene, "rooms", {})
    if runtime_object is not None and runtime_object.room_hint in rooms:
        return runtime_object.room_hint
    if runtime_object is not None and position is not None:
        if callable(getattr(adapter, "_containing_room", None)):
            try:
                room_id = hovsg_runtime_state.containing_room_id(
                    adapter,
                    scene,
                    position,
                )
            except Exception:
                room_id = None
            if room_id is not None:
                return room_id
    return getattr(static_object, "room_id", None) if static_object is not None else None


def _floor_height(scene: HOVSGSceneAsset, floor_id: str | None) -> float | None:
    if floor_id is None:
        return None
    floor = getattr(scene, "floors", {}).get(floor_id)
    try:
        return float(floor.floor_zero_level) if floor is not None else None
    except (TypeError, ValueError):
        return None


def _static_overlaps_navigation_height(
    scene: HOVSGSceneAsset,
    static_object: HOVSGObjectAsset | None,
    floor_height: float | None,
    *,
    robot_clearance_height_m: float = 1.6,
    floor_slop_m: float = 0.15,
) -> bool:
    if static_object is None or floor_height is None:
        return True
    axis_index = {"x": 0, "y": 1, "z": 2}.get(
        str(getattr(scene, "vertical_axis", "y")),
        1,
    )
    values = []
    for vertex in getattr(static_object, "vertices", []):
        try:
            values.append(float(vertex[axis_index]))
        except (IndexError, TypeError, ValueError):
            continue
    if not values:
        return True
    return (
        max(values) >= floor_height - floor_slop_m
        and min(values) <= floor_height + robot_clearance_height_m
    )


def _static_relations(
    scene: HOVSGSceneAsset,
    objects: dict[str, EffectiveObjectState],
) -> list[RuntimeRelationState]:
    relations = []
    for object_id, effective in objects.items():
        if effective.static_object_id is None:
            continue
        static_object = getattr(scene, "objects", {}).get(effective.static_object_id)
        if static_object is None:
            continue
        static_room_id = getattr(static_object, "room_id", None)
        room = getattr(scene, "rooms", {}).get(static_room_id)
        relations.append(
            RuntimeRelationState(
                subject_id=object_id,
                relation="in_room",
                object_id=static_room_id,
                confidence=1.0,
                source="static_hovsg",
            )
        )
        if room is not None:
            relations.append(
                RuntimeRelationState(
                    subject_id=object_id,
                    relation="on_floor",
                    object_id=room.floor_id,
                    confidence=1.0,
                    source="static_hovsg",
                )
            )
    return relations


def _derived_location_relations(
    objects: dict[str, EffectiveObjectState],
    *,
    state_step: int,
) -> list[RuntimeRelationState]:
    relations = []
    for effective in objects.values():
        source = "geometry" if effective.runtime_object_name else "static_hovsg"
        if effective.room_id is not None:
            relations.append(
                RuntimeRelationState(
                    subject_id=effective.object_id,
                    relation="in_room",
                    object_id=effective.room_id,
                    confidence=1.0,
                    source=source,
                    observed_at_step=state_step,
                    revision=effective.pose_revision,
                )
            )
        if effective.floor_id is not None:
            relations.append(
                RuntimeRelationState(
                    subject_id=effective.object_id,
                    relation="on_floor",
                    object_id=effective.floor_id,
                    confidence=1.0,
                    source=source,
                    observed_at_step=state_step,
                    revision=effective.pose_revision,
                )
            )
    return relations


def _canonicalize_runtime_relations(
    relations: list[RuntimeRelationState],
    *,
    aliases: dict[str, str],
    room_aliases: dict[str, str],
    floor_aliases: dict[str, str],
    current_step: int,
) -> list[RuntimeRelationState]:
    canonical = []
    for relation in relations:
        if relation.expires_at_step is not None and current_step > relation.expires_at_step:
            continue
        subject_aliases = room_aliases if relation.relation == "connected_through" else aliases
        subject_id = _resolve_alias(
            subject_aliases,
            relation.subject_id,
        )
        endpoint_aliases = (
            room_aliases
            if relation.relation in {"in_room", "connected_through"}
            else floor_aliases
            if relation.relation == "on_floor"
            else aliases
        )
        object_id = (
            _resolve_alias(endpoint_aliases, relation.object_id) if relation.object_id else None
        )
        canonical.append(
            RuntimeRelationState(
                subject_id=subject_id,
                relation=relation.relation,
                object_id=object_id,
                confidence=relation.confidence,
                source=relation.source,
                observed_at_step=relation.observed_at_step,
                expires_at_step=relation.expires_at_step,
                revision=relation.revision,
                removed=relation.removed,
            )
        )
    return canonical


def _connected_room_relations(scene: HOVSGSceneAsset) -> list[RuntimeRelationState]:
    relations = []
    for room_id, adjacent_ids in (getattr(scene, "room_adjacency", None) or {}).items():
        for adjacent_id in adjacent_ids:
            relations.append(
                RuntimeRelationState(
                    subject_id=str(room_id),
                    relation="connected_through",
                    object_id=str(adjacent_id),
                    confidence=1.0,
                    source="static_hovsg",
                )
            )
    return relations


def _portal_blocking_relations(
    scene: HOVSGSceneAsset,
    objects: dict[str, EffectiveObjectState],
) -> list[RuntimeRelationState]:
    portals = [
        item
        for item in objects.values()
        if item.static_object_id is not None and _is_portal_label(item.name)
    ]
    relations = []
    for obstacle in objects.values():
        if not obstacle.participates_in_navigation or not obstacle.navigation_footprints:
            continue
        if _is_portal_label(obstacle.name):
            continue
        for portal in portals:
            if not portal.footprint:
                continue
            if _bounds_overlap(
                _polygon_bounds(obstacle.footprint),
                _expand_bounds(_polygon_bounds(portal.footprint), 0.35),
            ):
                relations.append(
                    RuntimeRelationState(
                        subject_id=obstacle.object_id,
                        relation="blocking",
                        object_id=portal.object_id,
                        confidence=1.0,
                        source="geometry",
                        revision=f"{obstacle.geometry_revision}:{portal.geometry_revision}",
                    )
                )
    return relations


def _inverse_relations(
    relations: list[RuntimeRelationState],
) -> list[RuntimeRelationState]:
    inverse = {"inside": "contains", "on_top": "supports"}
    generated = []
    for relation in relations:
        inverse_name = inverse.get(relation.relation)
        if inverse_name is None or relation.object_id is None:
            continue
        generated.append(
            RuntimeRelationState(
                subject_id=relation.object_id,
                relation=inverse_name,
                object_id=relation.subject_id,
                confidence=relation.confidence,
                source=relation.source,
                observed_at_step=relation.observed_at_step,
                expires_at_step=relation.expires_at_step,
                revision=relation.revision,
            )
        )
    return generated


def _register_aliases(
    aliases: dict[str, str],
    effective: EffectiveObjectState,
) -> None:
    for alias in (
        effective.object_id,
        effective.static_object_id,
        effective.runtime_object_name,
        effective.name,
    ):
        if isinstance(alias, str) and alias:
            _register_alias(aliases, alias, effective.object_id)


def _room_aliases(scene: HOVSGSceneAsset) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for key, room in getattr(scene, "rooms", {}).items():
        canonical = str(getattr(room, "room_id", None) or key)
        for alias in (key, getattr(room, "room_id", None), getattr(room, "name", None)):
            _register_alias(aliases, alias, canonical)
    return aliases


def _floor_aliases(scene: HOVSGSceneAsset) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for key, floor in getattr(scene, "floors", {}).items():
        canonical = str(getattr(floor, "floor_id", None) or key)
        for alias in (key, getattr(floor, "floor_id", None), getattr(floor, "name", None)):
            _register_alias(aliases, alias, canonical)
    return aliases


def _register_alias(
    aliases: dict[str, str],
    alias: Any,
    canonical: str,
) -> None:
    if not isinstance(alias, str) or not alias.strip():
        return
    aliases[alias] = canonical
    aliases[_normalize(alias)] = canonical


def _resolve_alias(aliases: dict[str, str], value: str) -> str:
    return aliases.get(value, aliases.get(_normalize(value), value))


def _relation_sort_key(relation: RuntimeRelationState) -> tuple[int, int, float, str]:
    return (
        relation_source_priority(relation.source),
        int(relation.observed_at_step or -1),
        float(relation.confidence or 0.0),
        str(relation.revision or ""),
    )


def _relation_revision(relations: list[RuntimeRelationState]) -> str:
    return compute_relation_signature(relations)


def _object_map_revision(effective: EffectiveObjectState) -> str:
    return _hash_payload(
        {
            "object_id": effective.object_id,
            "navigation_role": effective.navigation_role,
            "navigation_footprints": effective.navigation_footprints,
            "held_by_self": effective.held_by_self,
            "pose_revision": effective.pose_revision,
            "geometry_revision": effective.geometry_revision,
            "relation_revision": effective.relation_revision,
        }
    )


def _uncertainty_margin(covariance: list[list[float]] | None) -> float:
    if not isinstance(covariance, list) or len(covariance) < 2:
        return 0.0
    try:
        variance = max(float(covariance[0][0]), float(covariance[1][1]), 0.0)
    except (IndexError, TypeError, ValueError):
        return 0.0
    return 2.0 * math.sqrt(variance)


def _expand_polygon(
    polygon: list[tuple[float, float]],
    margin: float,
) -> list[tuple[float, float]]:
    if len(polygon) < 3 or margin <= 0.0:
        return list(polygon)
    center_x = sum(point[0] for point in polygon) / len(polygon)
    center_y = sum(point[1] for point in polygon) / len(polygon)
    expanded = []
    for x, y in polygon:
        dx = x - center_x
        dy = y - center_y
        norm = math.hypot(dx, dy)
        if norm <= 1e-9:
            expanded.append((x, y))
        else:
            expanded.append((x + margin * dx / norm, y + margin * dy / norm))
    return expanded


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
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


def _coerce_polygon(value: Any) -> list[tuple[float, float]]:
    if not isinstance(value, (list, tuple)):
        return []
    polygon = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            polygon.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError):
            continue
    return polygon if len(polygon) >= 3 else []


def _is_portal_label(value: Any) -> bool:
    return bool(set(_normalize(value).split("_")) & _PORTAL_TOKENS)


def _polygon_bounds(
    polygon: list[tuple[float, float]],
) -> tuple[float, float, float, float]:
    return (
        min(point[0] for point in polygon),
        min(point[1] for point in polygon),
        max(point[0] for point in polygon),
        max(point[1] for point in polygon),
    )


def _expand_bounds(
    bounds: tuple[float, float, float, float],
    margin: float,
) -> tuple[float, float, float, float]:
    return (
        bounds[0] - margin,
        bounds[1] - margin,
        bounds[2] + margin,
        bounds[3] + margin,
    )


def _bounds_overlap(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    return not (
        left[2] < right[0] or right[2] < left[0] or left[3] < right[1] or right[3] < left[1]
    )


def _hash_payload(value: Any) -> str:
    return hashlib.sha1(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()[:16]


def _normalize(value: Any) -> str:
    return "_".join(re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split())


__all__ = [
    "EffectiveObjectState",
    "EffectiveSceneView",
    "approach_owner",
    "effective_scene_view",
    "is_robot_label",
    "is_self_robot_id",
    "merge_relations",
    "relation_source_priority",
]
