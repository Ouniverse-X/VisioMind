from __future__ import annotations

import math
import os
import json
import hashlib
from pathlib import Path
from typing import Any

import networkx as nx

from . import door_gating as hovsg_door_gating
from . import effective_scene as hovsg_effective_scene
from . import runtime_state as hovsg_runtime_state
from .effective_scene import EffectiveObjectState
from .models import HOVSGObjectAsset, HOVSGRoomAsset, HOVSGSceneAsset


PORTAL_LIKE_OBJECT_NAME_TOKENS = {
    "door",
    "doorway",
    "gate",
    "gateway",
    "opening",
    "portal",
}
PART_APPROACH_LATERAL_OFFSET_WEIGHT = 0.75
PART_NAME_TOKEN_ALIASES = {
    "boot": "trunk",
    "cap": "lid",
    "cover": "lid",
    "grip": "handle",
    "hatch": "trunk",
    "pull": "handle",
    "tailgate": "trunk",
}
PART_NAME_IGNORED_TOKENS = {
    "assembly",
    "collision",
    "geometry",
    "joint",
    "link",
    "mesh",
    "part",
    "visual",
}


def _new_candidate_diagnostics() -> dict[str, Any]:
    return {
        "candidate_count_before_clearance": 0,
        "candidate_count_after_point_clearance": 0,
        "candidate_count_after_graph_handoff": 0,
        "candidate_count_after_segment_clearance": 0,
        "candidate_count_after_portal_filter": 0,
        "selected_candidate_id": None,
        "selection_failure_reason": None,
    }


def _store_candidate_diagnostics(
    adapter: Any,
    *,
    scene_id: str,
    object_id: str,
    diagnostics: dict[str, Any],
) -> None:
    store = getattr(adapter, "_object_approach_diagnostics", None)
    if not isinstance(store, dict):
        store = {}
        adapter._object_approach_diagnostics = store
    store[(str(scene_id), str(object_id))] = dict(diagnostics)


def object_approach_diagnostics(
    adapter: Any,
    *,
    scene_id: str,
    object_id: str,
) -> dict[str, Any]:
    store = getattr(adapter, "_object_approach_diagnostics", None)
    if not isinstance(store, dict):
        return _new_candidate_diagnostics()
    diagnostics = store.get((str(scene_id), str(object_id)))
    return dict(diagnostics) if isinstance(diagnostics, dict) else _new_candidate_diagnostics()


def build_object_approach_candidates(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    goal: dict[str, Any],
    start: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    object_id = str(goal.get("object_id") or "").strip()
    scene_id = str(goal.get("scene_id") or scene.scene_id or "").strip()
    diagnostics = _new_candidate_diagnostics()
    effective_view = hovsg_effective_scene.effective_scene_view(adapter, scene)
    target_object = effective_view.resolve_object(object_id)
    if not object_id or target_object is None:
        diagnostics["selection_failure_reason"] = "object_not_found"
        _store_candidate_diagnostics(
            adapter,
            scene_id=scene_id,
            object_id=object_id,
            diagnostics=diagnostics,
        )
        return []

    approach_owner = hovsg_effective_scene.approach_owner(
        effective_view,
        target_object,
    )
    if approach_owner is None:
        diagnostics["selection_failure_reason"] = (
            "object_held_by_self" if target_object.held_by_self else "object_not_approachable"
        )
        _store_candidate_diagnostics(
            adapter,
            scene_id=scene_id,
            object_id=object_id,
            diagnostics=diagnostics,
        )
        return []
    owner_relation = _approach_owner_relation(target_object, approach_owner)
    requires_container_open = bool(
        owner_relation == "inside"
        and any(item.relation == "closed" for item in approach_owner.relations)
    )

    object_position = target_object.position
    if object_position is None:
        diagnostics["selection_failure_reason"] = "object_position_missing"
        _store_candidate_diagnostics(
            adapter,
            scene_id=scene_id,
            object_id=object_id,
            diagnostics=diagnostics,
        )
        return []

    object_floor_id = adapter._infer_floor_id(scene, goal) or approach_owner.floor_id
    object_room_id = adapter._infer_room_id(scene, goal) or approach_owner.room_id
    interaction_xy = adapter._project_horizontal(scene, object_position)
    approach_position = approach_owner.position or object_position
    approach_xy = adapter._project_horizontal(scene, approach_position)
    if interaction_xy is None or approach_xy is None:
        diagnostics["selection_failure_reason"] = "object_projection_missing"
        _store_candidate_diagnostics(
            adapter,
            scene_id=scene_id,
            object_id=object_id,
            diagnostics=diagnostics,
        )
        return []

    start_payload = dict(start or {})
    context_payload = dict(context or {})
    clearance_radius_m = float(getattr(adapter, "object_approach_clearance_radius_m", 0.0))
    clearance_map_spec = None
    point_has_clearance_fn = None
    if scene_id and clearance_radius_m > 0.0:
        try:
            from ..nav2.nav2_runtime_bridge import point_has_clearance
        except Exception:
            point_has_clearance = None
        if point_has_clearance is not None:
            clearance_map_spec = adapter._load_portal_analysis_map(
                scene_id=scene_id,
                start=start_payload,
                goal=goal,
                context=context_payload,
            )
            point_has_clearance_fn = point_has_clearance

    polygon = list(approach_owner.footprint)
    room = scene.rooms.get(object_room_id) if object_room_id is not None else None
    room_polygon = adapter._room_polygon_2d(scene, room) if room is not None else []
    target_part = normalized_target_part(goal)
    part_context = (
        object_part_approach_context(
            adapter,
            scene=scene,
            object_id=object_id,
            object_name=target_object.name,
            object_room_id=object_room_id,
            object_xy=interaction_xy,
            target_part=target_part,
            object_state=target_object,
        )
        if target_part and approach_owner.object_id == target_object.object_id
        else None
    )
    approach_polygon = polygon
    if isinstance(part_context, dict) and isinstance(part_context.get("object_polygon"), list):
        approach_polygon = part_context["object_polygon"]
    candidate_room_ids = object_approach_candidate_room_ids(
        adapter,
        scene=scene,
        object_room_id=object_room_id,
        object_xy=approach_xy,
        room_polygon=room_polygon,
    )
    candidate_room_polygons = {
        room_id: adapter._room_polygon_2d(scene, candidate_room)
        for room_id in candidate_room_ids
        for candidate_room in [scene.rooms.get(room_id)]
        if candidate_room is not None
    }
    candidates: list[dict[str, Any]] = []
    safe_nav_node_fallbacks: list[dict[str, Any]] = []
    global_safe_nav_node_fallbacks: list[dict[str, Any]] = []
    for node_id, attrs in scene.nav_graph.nodes(data=True):
        node_room_id = adapter._resolve_room_id_from_nav_node_attrs(
            attrs,
            room_id_by_name={
                str(room.name): room_id
                for room_id, room in scene.rooms.items()
                if isinstance(room.name, str) and room.name
            },
        )
        node_in_candidate_room = (
            not candidate_room_ids or node_room_id is None or (node_room_id in candidate_room_ids)
        )
        node_floor_id = attrs.get("floor_id")
        if (
            object_floor_id is not None
            and node_floor_id is not None
            and str(node_floor_id) != str(object_floor_id)
        ):
            continue
        pos = attrs.get("pos")
        if not isinstance(pos, (list, tuple)) or len(pos) < 3:
            continue

        candidate_position = {
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2]),
        }
        candidate_xy = adapter._project_horizontal(scene, candidate_position)
        if candidate_xy is None:
            continue
        diagnostics["candidate_count_before_clearance"] += 1
        if (
            isinstance(clearance_map_spec, dict)
            and point_has_clearance_fn is not None
            and not point_has_clearance_fn(
                map_spec=clearance_map_spec,
                point_xy={"x": float(candidate_xy[0]), "y": float(candidate_xy[1])},
                clearance_radius_m=object_part_candidate_clearance_radius(
                    default_radius_m=clearance_radius_m,
                    object_xy=approach_xy,
                    candidate_xy=candidate_xy,
                    part_context=part_context,
                ),
            )
        ):
            continue
        diagnostics["candidate_count_after_point_clearance"] += 1
        approach_distance = math.hypot(
            candidate_xy[0] - interaction_xy[0],
            candidate_xy[1] - interaction_xy[1],
        )
        boundary_distance = (
            adapter._point_to_polygon_boundary_distance(candidate_xy, approach_polygon)
            if len(approach_polygon) >= 3
            else approach_distance
        )
        if boundary_distance < adapter.object_approach_min_distance_m:
            continue
        if boundary_distance > adapter.object_approach_max_distance_m:
            continue

        candidate = {
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2]),
            "floor_id": node_floor_id or object_floor_id,
            "room_id": node_room_id or object_room_id,
            "room_name": attrs.get("room_name") or goal.get("room_name"),
            "nav_node": adapter._serialize_node_id(node_id),
            "waypoint_type": "object_approach",
            "object_id": object_id,
            "object_name": target_object.name,
            "effective_object_id": target_object.object_id,
            "approach_owner_id": approach_owner.object_id,
            "approach_owner_name": approach_owner.name,
            **({"approach_owner_relation": owner_relation} if owner_relation else {}),
            **({"requires_container_open": True} if requires_container_open else {}),
            **({"target_part": target_part} if target_part else {}),
            "object_position": dict(object_position),
            "approach_owner_position": dict(approach_position),
            "approach_distance_m": approach_distance,
            "approach_boundary_distance_m": boundary_distance,
            "desired_heading": math.atan2(
                interaction_xy[1] - candidate_xy[1],
                interaction_xy[0] - candidate_xy[0],
            ),
            "candidate_geometry_score": abs(
                boundary_distance - adapter.object_approach_preferred_distance_m
            ),
            "selection_source": "nav_node",
            "handoff_distance_m": 0.0,
            "candidate_cache_revision": _candidate_cache_revision(
                effective_view,
                target_object,
                approach_owner,
            ),
        }
        if (
            object_room_id is not None
            and node_room_id is not None
            and node_room_id != object_room_id
        ):
            candidate["approach_room_relation"] = "adjacent_room"
        global_safe_nav_node_fallbacks.append(candidate)
        if node_in_candidate_room:
            safe_nav_node_fallbacks.append(candidate)
        diagnostics["candidate_count_after_graph_handoff"] += 1
        diagnostics["candidate_count_after_segment_clearance"] += 1
        if not node_in_candidate_room:
            continue
        candidates.append(candidate)

    traversability_candidates = build_continuous_object_approach_candidates(
        adapter,
        scene=scene,
        goal=goal,
        start=start,
        object_xy=approach_xy,
        interaction_xy=interaction_xy,
        object_position=object_position,
        object_floor_id=object_floor_id,
        object_room_id=object_room_id,
        object_name=target_object.name,
        object_polygon=approach_polygon,
        room=room,
        room_polygons=candidate_room_polygons,
        context=context,
        part_context=part_context,
        diagnostics=diagnostics,
    )
    merged_candidates = merge_object_approach_candidates([*traversability_candidates, *candidates])
    fallback_pool = safe_nav_node_fallbacks or global_safe_nav_node_fallbacks
    if not merged_candidates and fallback_pool:
        fallback = min(
            fallback_pool,
            key=lambda item: (
                float(item.get("approach_boundary_distance_m", float("inf"))),
                float(item.get("approach_distance_m", float("inf"))),
                str(item.get("nav_node")),
            ),
        )
        merged_candidates = [
            {
                **fallback,
                "selection_source": (
                    "safe_nav_node_fallback"
                    if safe_nav_node_fallbacks
                    else "adjacent_safe_nav_node_fallback"
                ),
                "candidate_geometry_score": float(
                    fallback.get("approach_boundary_distance_m", 0.0)
                ),
            }
        ]
    for candidate in merged_candidates:
        candidate.setdefault("effective_object_id", target_object.object_id)
        candidate.setdefault("approach_owner_id", approach_owner.object_id)
        candidate.setdefault("approach_owner_name", approach_owner.name)
        candidate.setdefault("approach_owner_position", dict(approach_position))
        if owner_relation:
            candidate.setdefault("approach_owner_relation", owner_relation)
        if requires_container_open:
            candidate.setdefault("requires_container_open", True)
        candidate.setdefault(
            "candidate_cache_revision",
            _candidate_cache_revision(
                effective_view,
                target_object,
                approach_owner,
            ),
        )
        annotate_object_part_approach_candidate(
            adapter,
            scene=scene,
            candidate=candidate,
            object_xy=approach_xy,
            part_context=part_context,
        )
    merged_candidates.sort(
        key=lambda item: (
            float(item.get("target_part_score", 0.0)),
            float(item.get("candidate_geometry_score", 0.0)),
            float(item.get("handoff_distance_m", 0.0)),
            float(item.get("approach_distance_m", 0.0)),
            str(item.get("nav_node")),
        )
    )
    trimmed = merged_candidates[: adapter.object_approach_max_candidates]
    for index, candidate in enumerate(trimmed, start=1):
        candidate.setdefault("candidate_id", f"cand_{index:02d}")
    if not trimmed:
        diagnostics["selection_failure_reason"] = "no_safe_candidates"
    _store_candidate_diagnostics(
        adapter,
        scene_id=scene_id,
        object_id=object_id,
        diagnostics=diagnostics,
    )
    return trimmed


def _candidate_cache_revision(
    view: hovsg_effective_scene.EffectiveSceneView,
    target: EffectiveObjectState,
    owner: EffectiveObjectState,
) -> str:
    payload = "|".join(
        (
            view.scene_id,
            target.object_id,
            target.pose_revision,
            target.geometry_revision,
            target.relation_revision,
            owner.object_id,
            owner.pose_revision,
            owner.geometry_revision,
            view.relation_signature,
            view.map_revision,
        )
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _approach_owner_relation(
    target: EffectiveObjectState,
    owner: EffectiveObjectState,
) -> str | None:
    for relation_name in ("on_top", "inside"):
        if owner.object_id in target.relation_targets(relation_name):
            return relation_name
    return None


def object_approach_candidate_room_ids(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    object_room_id: str | None,
    object_xy: tuple[float, float],
    room_polygon: list[tuple[float, float]],
    boundary_threshold_m: float = 0.8,
) -> set[str]:
    if object_room_id is None:
        return set()
    room_ids = {object_room_id}
    if len(room_polygon) < 3:
        return room_ids
    if adapter._point_to_polygon_boundary_distance(object_xy, room_polygon) > boundary_threshold_m:
        return room_ids
    effective_adjacency = hovsg_door_gating.effective_room_adjacency(adapter, scene)
    adjacent_room_ids = (
        effective_adjacency.get(object_room_id, set()) if effective_adjacency else set()
    )
    for room_id, candidate_room in scene.rooms.items():
        if (
            room_id == object_room_id
            or candidate_room.floor_id != scene.rooms[object_room_id].floor_id
        ):
            continue
        if adjacent_room_ids and room_id not in adjacent_room_ids:
            continue
        candidate_polygon = adapter._room_polygon_2d(scene, candidate_room)
        if len(candidate_polygon) < 3:
            continue
        if (
            adapter._point_to_polygon_boundary_distance(object_xy, candidate_polygon)
            <= boundary_threshold_m
        ):
            room_ids.add(room_id)
    return room_ids


def build_continuous_object_approach_candidates(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    goal: dict[str, Any],
    start: dict[str, Any] | None,
    object_xy: tuple[float, float],
    object_position: dict[str, float],
    object_floor_id: str | None,
    object_room_id: str | None,
    object_name: str | None,
    object_polygon: list[tuple[float, float]],
    room: HOVSGRoomAsset | None,
    room_polygons: dict[str, list[tuple[float, float]]],
    context: dict[str, Any] | None,
    part_context: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
    interaction_xy: tuple[float, float] | None = None,
) -> list[dict[str, Any]]:
    interaction_xy = interaction_xy or object_xy
    scene_id = str(goal.get("scene_id") or scene.scene_id or "").strip()
    start_payload = dict(start or {})
    context_payload = dict(context or {})
    map_spec = adapter._load_portal_analysis_map(
        scene_id=scene_id,
        start=start_payload,
        goal=goal,
        context=context_payload,
    )
    if not isinstance(map_spec, dict):
        return []

    try:
        from ..nav2.nav2_runtime_bridge import (
            point_has_clearance,
            segment_has_clearance,
        )
    except Exception:
        return []

    start_pose = start_payload.get("pose") if isinstance(start_payload.get("pose"), dict) else None
    start_xy = (
        adapter._project_horizontal(scene, start_pose) if isinstance(start_pose, dict) else None
    )
    base_angle = math.pi
    if start_xy is not None:
        base_angle = math.atan2(start_xy[1] - object_xy[1], start_xy[0] - object_xy[0])

    radii = [
        adapter.object_approach_preferred_distance_m,
        max(
            adapter.object_approach_min_distance_m,
            adapter.object_approach_preferred_distance_m - 0.2,
        ),
        min(
            adapter.object_approach_max_distance_m,
            adapter.object_approach_preferred_distance_m + 0.2,
        ),
        adapter.object_approach_min_distance_m,
        adapter.object_approach_max_distance_m,
    ]
    unique_radii: list[float] = []
    for radius in radii:
        if any(abs(radius - existing) <= 1e-6 for existing in unique_radii):
            continue
        unique_radii.append(radius)

    candidates: list[dict[str, Any]] = []
    target_part = normalized_target_part(goal)
    angle_offsets = [
        2.0 * math.pi * float(index) / float(adapter.object_approach_angle_samples)
        for index in range(adapter.object_approach_angle_samples)
    ]
    sample_angles = [base_angle + offset for offset in angle_offsets]
    part_direction = (
        part_context.get("preferred_direction") if isinstance(part_context, dict) else None
    )
    if isinstance(part_direction, tuple) and len(part_direction) == 2:
        preferred_angle = math.atan2(float(part_direction[1]), float(part_direction[0]))
        sample_angles.extend(
            [
                preferred_angle,
                preferred_angle - math.pi / float(adapter.object_approach_angle_samples),
                preferred_angle + math.pi / float(adapter.object_approach_angle_samples),
            ]
        )
    for standoff_distance in unique_radii:
        for angle in sample_angles:
            direction_xy = (math.cos(angle), math.sin(angle))
            boundary_offset = directional_boundary_offset(
                object_xy=object_xy,
                object_polygon=object_polygon,
                direction=direction_xy,
            )
            radius = boundary_offset + standoff_distance
            candidate_xy = (
                object_xy[0] + radius * direction_xy[0],
                object_xy[1] + radius * direction_xy[1],
            )
            candidate_room_id = room_id_for_approach_point(
                adapter, point=candidate_xy, room_polygons=room_polygons
            )
            if room_polygons and candidate_room_id is None:
                continue
            if object_polygon and adapter._point_in_polygon(candidate_xy, object_polygon):
                continue

            boundary_distance = (
                adapter._point_to_polygon_boundary_distance(candidate_xy, object_polygon)
                if len(object_polygon) >= 3
                else radius
            )
            if boundary_distance < adapter.object_approach_min_distance_m:
                continue
            if boundary_distance > adapter.object_approach_max_distance_m:
                continue

            candidate_point_xy = {
                "x": float(candidate_xy[0]),
                "y": float(candidate_xy[1]),
            }
            if diagnostics is not None:
                diagnostics["candidate_count_before_clearance"] += 1
            clearance_radius_m = object_part_candidate_clearance_radius(
                default_radius_m=adapter.object_approach_clearance_radius_m,
                object_xy=object_xy,
                candidate_xy=candidate_xy,
                part_context=part_context,
            )
            if not point_has_clearance(
                map_spec=map_spec,
                point_xy=candidate_point_xy,
                clearance_radius_m=clearance_radius_m,
            ):
                continue
            if diagnostics is not None:
                diagnostics["candidate_count_after_point_clearance"] += 1

            candidate_position = adapter._lift_horizontal_point(
                scene,
                candidate_xy,
                source_room=room,
                target_room=room,
            )
            if candidate_position is None:
                continue
            nearest_node = adapter._nearest_nav_node(
                scene.nav_graph,
                candidate_position,
                floor_id=object_floor_id,
                room_id=candidate_room_id or object_room_id,
            )
            if nearest_node is None:
                continue
            nearest_node_waypoint = adapter._node_to_waypoint(scene.nav_graph, nearest_node)
            nearest_node_xy = adapter._project_horizontal(scene, nearest_node_waypoint)
            if nearest_node_xy is None:
                continue
            handoff_distance = math.hypot(
                float(candidate_xy[0]) - float(nearest_node_xy[0]),
                float(candidate_xy[1]) - float(nearest_node_xy[1]),
            )
            max_handoff_distance = float(
                getattr(adapter, "object_approach_max_graph_handoff_distance_m", 1.0)
            )
            if handoff_distance > max_handoff_distance:
                continue
            if diagnostics is not None:
                diagnostics["candidate_count_after_graph_handoff"] += 1
            if not segment_has_clearance(
                map_spec=map_spec,
                start_xy={
                    "x": float(nearest_node_xy[0]),
                    "y": float(nearest_node_xy[1]),
                },
                end_xy=candidate_point_xy,
                clearance_radius_m=clearance_radius_m,
                step_m=max(adapter.portal_analysis_map_resolution * 0.5, 0.025),
            ):
                continue
            if diagnostics is not None:
                diagnostics["candidate_count_after_segment_clearance"] += 1

            candidates.append(
                {
                    **candidate_position,
                    "floor_id": object_floor_id or nearest_node_waypoint.get("floor_id"),
                    "room_id": candidate_room_id
                    or object_room_id
                    or nearest_node_waypoint.get("room_id"),
                    "room_name": room_name_for_candidate(
                        scene=scene,
                        room_id=candidate_room_id or object_room_id,
                        fallback=nearest_node_waypoint.get("room_name") or goal.get("room_name"),
                    ),
                    "nav_node": adapter._serialize_node_id(nearest_node),
                    "waypoint_type": "object_approach",
                    "object_id": goal.get("object_id"),
                    "object_name": object_name,
                    **({"target_part": target_part} if target_part else {}),
                    "object_position": dict(object_position),
                    "approach_distance_m": radius,
                    "approach_boundary_distance_m": boundary_distance,
                    "desired_heading": math.atan2(
                        interaction_xy[1] - candidate_xy[1],
                        interaction_xy[0] - candidate_xy[0],
                    ),
                    "candidate_geometry_score": abs(
                        boundary_distance - adapter.object_approach_preferred_distance_m
                    ),
                    "selection_source": "continuous_sample",
                    "handoff_distance_m": handoff_distance,
                    **(
                        {"approach_room_relation": "adjacent_room"}
                        if object_room_id is not None
                        and candidate_room_id is not None
                        and candidate_room_id != object_room_id
                        else {}
                    ),
                }
            )
    return candidates


def room_name_for_candidate(
    *,
    scene: HOVSGSceneAsset,
    room_id: str | None,
    fallback: object,
) -> object:
    if room_id is not None:
        room = scene.rooms.get(str(room_id))
        if room is not None and room.name:
            return room.name
    return fallback


def room_id_for_approach_point(
    adapter: Any,
    *,
    point: tuple[float, float],
    room_polygons: dict[str, list[tuple[float, float]]],
) -> str | None:
    for room_id, polygon in room_polygons.items():
        if len(polygon) >= 3 and adapter._point_in_polygon(point, polygon):
            return room_id
    return None


def directional_boundary_offset(
    *,
    object_xy: tuple[float, float],
    object_polygon: list[tuple[float, float]],
    direction: tuple[float, float],
) -> float:
    if len(object_polygon) < 3:
        return 0.0
    norm = math.hypot(direction[0], direction[1])
    if norm <= 1e-9:
        return 0.0
    dx = direction[0] / norm
    dy = direction[1] / norm
    return max(
        0.0,
        *(
            (float(vertex[0]) - object_xy[0]) * dx + (float(vertex[1]) - object_xy[1]) * dy
            for vertex in object_polygon
        ),
    )


def merge_object_approach_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for candidate in candidates:
        key = (
            round(float(candidate.get("x", 0.0)), 3),
            round(float(candidate.get("y", 0.0)), 3),
            round(float(candidate.get("z", 0.0)), 3),
            str(candidate.get("nav_node")),
        )
        previous = merged.get(key)
        if previous is None:
            merged[key] = dict(candidate)
            continue
        previous_source = str(previous.get("selection_source") or "")
        current_source = str(candidate.get("selection_source") or "")
        if previous_source != "continuous_sample" and current_source == "continuous_sample":
            merged[key] = {**previous, **candidate}
    return list(merged.values())


def score_object_approach_candidates(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    start_node: Any,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    active_graph, door_gated = hovsg_door_gating.filtered_nav_graph(adapter, scene)
    blocked_by_closed_door_candidate_ids: list[str] = []
    for candidate in candidates:
        candidate_node = adapter._normalize_node_id(candidate.get("nav_node"))
        if candidate_node is None or candidate_node not in active_graph:
            continue
        try:
            path_nodes = nx.shortest_path(active_graph, start_node, candidate_node, weight="dist")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            if (
                door_gated
                and start_node in scene.nav_graph
                and candidate_node in scene.nav_graph
                and nx.has_path(scene.nav_graph, start_node, candidate_node)
            ):
                blocked_by_closed_door_candidate_ids.append(
                    str(candidate.get("candidate_id") or candidate.get("nav_node"))
                )
            continue
        scored_candidate = dict(candidate)
        scored_candidate["path_cost"] = adapter._path_cost(active_graph, path_nodes)
        scored_candidate["path_nodes"] = [adapter._serialize_node_id(node) for node in path_nodes]
        scored_candidate["nearby_object_evidence"] = object_proximity_evidence(
            adapter,
            scene=scene,
            candidate=scored_candidate,
            path_nodes=path_nodes,
        )
        if object_approach_candidate_is_too_close_to_portal(
            scored_candidate,
            min_clearance_m=float(
                getattr(adapter, "object_approach_min_portal_stance_clearance_m", 0.45)
            ),
        ):
            continue
        scored.append(scored_candidate)
    object_id = str(candidates[0].get("object_id") or "") if candidates else ""
    diagnostics = object_approach_diagnostics(
        adapter,
        scene_id=str(scene.scene_id or ""),
        object_id=object_id,
    )
    diagnostics["candidate_count_after_portal_filter"] = len(scored)
    if blocked_by_closed_door_candidate_ids:
        diagnostics["blocked_by_closed_door_candidate_ids"] = blocked_by_closed_door_candidate_ids
    if candidates and not scored:
        diagnostics["selection_failure_reason"] = (
            "blocked_by_closed_door"
            if blocked_by_closed_door_candidate_ids
            else "all_candidates_portal_or_path_filtered"
        )
    _store_candidate_diagnostics(
        adapter,
        scene_id=str(scene.scene_id or ""),
        object_id=object_id,
        diagnostics=diagnostics,
    )
    return scored


def select_object_approach_candidate(
    *,
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    best_candidate: dict[str, Any] | None = None
    best_key: tuple[float, float, float, float, str] | None = None
    for candidate in candidates:
        part_score = float(candidate.get("target_part_score", 0.0))
        path_cost = float(candidate.get("path_cost", 0.0))
        geometry_score = float(candidate.get("candidate_geometry_score", 0.0))
        approach_distance = float(candidate.get("approach_distance_m", 0.0))
        sort_key = (
            part_score,
            path_cost,
            geometry_score,
            approach_distance,
            str(candidate.get("nav_node")),
        )
        if best_key is None or sort_key < best_key:
            best_key = sort_key
            best_candidate = dict(candidate)
    return best_candidate


def normalized_target_part(goal: dict[str, Any]) -> str:
    for key in ("target_part", "part"):
        value = goal.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_object_name(value)
    target = goal.get("target")
    if isinstance(target, dict):
        value = target.get("part")
        if isinstance(value, str) and value.strip():
            return normalize_object_name(value)
    return ""


def object_part_approach_context(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    object_id: str,
    object_name: str | None,
    object_room_id: str | None,
    object_xy: tuple[float, float],
    target_part: str,
    object_state: EffectiveObjectState | None = None,
) -> dict[str, Any] | None:
    runtime_context = runtime_part_approach_context(
        scene=scene,
        object_state=object_state,
        object_xy=object_xy,
        target_part=target_part,
    )
    if runtime_context is not None:
        return {
            "target_part": target_part,
            "strategy": "runtime_part_link",
            **runtime_context,
        }
    model_context = model_part_approach_context(
        adapter,
        scene=scene,
        object_name=object_name,
        object_room_id=object_room_id,
        object_xy=object_xy,
        target_part=target_part,
    )
    if model_context is not None:
        return {
            "target_part": target_part,
            "strategy": "model_part_link",
            **model_context,
        }
    if not is_car_rear_part_target(object_name=object_name, target_part=target_part):
        return None
    direction = rear_direction_from_nearest_portal(
        adapter,
        scene=scene,
        object_id=object_id,
        object_room_id=object_room_id,
        object_xy=object_xy,
    )
    if direction is None:
        return None
    return {
        "target_part": target_part,
        "preferred_direction": direction,
        "strategy": "away_from_nearest_portal",
    }


def runtime_part_approach_context(
    *,
    scene: HOVSGSceneAsset,
    object_state: EffectiveObjectState | None,
    object_xy: tuple[float, float],
    target_part: str,
) -> dict[str, Any] | None:
    if object_state is None or not object_state.collision_parts:
        return None
    parts_with_centers = [
        (part, center)
        for part in object_state.collision_parts
        for center in [collision_part_center_2d(scene=scene, part=part)]
        if center is not None
    ]
    match = matching_named_part(
        parts_with_centers,
        target_part=target_part,
        name_getter=lambda item: item[0].get("link") or item[0].get("geometry_id"),
    )
    if match is None:
        return None
    (_, part_xy), matched_name, match_score = match
    dx = part_xy[0] - object_xy[0]
    dy = part_xy[1] - object_xy[1]
    norm = math.hypot(dx, dy)
    if norm <= 1e-6:
        return None
    return {
        "preferred_direction": (dx / norm, dy / norm),
        "resolved_part_link": matched_name,
        "part_match_score": match_score,
        "part_position": part_xy,
    }


def is_car_rear_part_target(*, object_name: str | None, target_part: str) -> bool:
    object_tokens = set(normalize_object_name(object_name).split())
    if not (object_tokens & {"car", "automobile", "vehicle"}):
        return False
    part_tokens = set(normalize_object_name(target_part).split())
    rear_tokens = {"trunk", "rear", "boot", "hatch", "tailgate"}
    return bool(part_tokens & rear_tokens)


def model_part_approach_context(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    object_name: str | None,
    object_room_id: str | None,
    object_xy: tuple[float, float],
    target_part: str,
) -> tuple[float, float] | None:
    scene_json = scene.metadata.get("scene_json") if isinstance(scene.metadata, dict) else None
    if not isinstance(scene_json, str) or not scene_json.strip():
        return None
    scene_payload = read_json_or_none(Path(scene_json).expanduser())
    if not isinstance(scene_payload, dict):
        return None
    match = matching_scene_object(
        adapter,
        scene=scene,
        scene_payload=scene_payload,
        object_name=object_name,
        object_room_id=object_room_id,
        object_xy=object_xy,
    )
    if match is None:
        return None
    metadata_path = model_metadata_path(
        scene=scene,
        category=match.get("category"),
        model=match.get("model"),
    )
    if metadata_path is None:
        return None
    model_metadata = read_json_or_none(metadata_path)
    if not isinstance(model_metadata, dict):
        return None
    local_match = target_part_local_match(model_metadata=model_metadata, target_part=target_part)
    if local_match is None:
        return None
    matched_link, local_center, match_score = local_match
    rotated = rotate_vector_by_quaternion(local_center, match.get("orientation"))
    direction = project_vector_to_scene_plane(scene=scene, vector=rotated)
    if direction is None:
        return None
    norm = math.hypot(direction[0], direction[1])
    if norm <= 1e-6:
        return None
    context: dict[str, Any] = {
        "preferred_direction": (direction[0] / norm, direction[1] / norm),
        "resolved_part_link": matched_link,
        "part_match_score": match_score,
    }
    footprint = model_footprint_polygon(scene=scene, match=match, model_metadata=model_metadata)
    if footprint is not None:
        context["object_polygon"] = footprint
    return context


def matching_scene_object(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    scene_payload: dict[str, Any],
    object_name: str | None,
    object_room_id: str | None,
    object_xy: tuple[float, float],
) -> dict[str, Any] | None:
    target_name = normalize_object_name(object_name)
    room_name = None
    if object_room_id is not None and object_room_id in scene.rooms:
        room_name = normalize_object_name(scene.rooms[object_room_id].name)
    best: tuple[float, dict[str, Any]] | None = None
    for item in iter_scene_object_specs(scene_payload):
        category = normalize_object_name(item.get("category"))
        if target_name and category and category != target_name:
            continue
        in_rooms = [
            normalize_object_name(value)
            for value in item.get("in_rooms") or []
            if isinstance(value, str)
        ]
        if room_name and in_rooms and room_name not in in_rooms:
            continue
        root_position = item.get("position")
        if not isinstance(root_position, list) or len(root_position) < 3:
            continue
        projected = adapter._project_horizontal(
            scene,
            {"x": root_position[0], "y": root_position[1], "z": root_position[2]},
        )
        if projected is None:
            continue
        distance = math.hypot(projected[0] - object_xy[0], projected[1] - object_xy[1])
        if best is None or distance < best[0]:
            best = (distance, item)
    return dict(best[1]) if best is not None else None


def iter_scene_object_specs(scene_payload: dict[str, Any]) -> list[dict[str, Any]]:
    init_info = (scene_payload.get("objects_info") or {}).get("init_info") or {}
    registry = ((scene_payload.get("state") or {}).get("registry") or {}).get(
        "object_registry"
    ) or {}
    specs: list[dict[str, Any]] = []
    if not isinstance(init_info, dict) or not isinstance(registry, dict):
        return specs
    for object_key, entry in init_info.items():
        if not isinstance(entry, dict):
            continue
        args = entry.get("args")
        if not isinstance(args, dict):
            continue
        name = str(args.get("name") or object_key)
        state = registry.get(name) if isinstance(name, str) else None
        if not isinstance(state, dict):
            state = registry.get(str(object_key))
        root_link = (state or {}).get("root_link") if isinstance(state, dict) else None
        if not isinstance(root_link, dict):
            continue
        specs.append(
            {
                "name": name,
                "category": args.get("category"),
                "model": args.get("model"),
                "in_rooms": args.get("in_rooms"),
                "position": root_link.get("pos"),
                "orientation": root_link.get("ori"),
            }
        )
    return specs


def model_metadata_path(*, scene: HOVSGSceneAsset, category: object, model: object) -> Path | None:
    if (
        not isinstance(category, str)
        or not category.strip()
        or not isinstance(model, str)
        or not model.strip()
    ):
        return None
    relative = (
        Path("behavior-1k-assets")
        / "objects"
        / category.strip()
        / model.strip()
        / "misc"
        / "metadata.json"
    )
    candidates: list[Path] = []
    env_root = os.environ.get("OMNIGIBSON_DATA_PATH")
    if env_root:
        candidates.append(Path(env_root).expanduser() / relative)
    for parent in [scene.graph_path, *scene.graph_path.parents]:
        if parent.name == "voltron":
            candidates.append(parent.parent / "BEHAVIOR-1K" / "datasets" / relative)
            break
    candidates.append(Path.cwd().parent / "BEHAVIOR-1K" / "datasets" / relative)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def target_part_local_center(
    *, model_metadata: dict[str, Any], target_part: str
) -> tuple[float, float, float] | None:
    match = target_part_local_match(
        model_metadata=model_metadata,
        target_part=target_part,
    )
    return match[1] if match is not None else None


def target_part_local_match(
    *, model_metadata: dict[str, Any], target_part: str
) -> tuple[str, tuple[float, float, float], float] | None:
    boxes = model_metadata.get("link_bounding_boxes")
    if not isinstance(boxes, dict):
        return None
    links_with_centers = [
        (link_name, center)
        for link_name, link_boxes in boxes.items()
        for center in [link_box_center(link_boxes)]
        if center is not None
    ]
    match = matching_named_part(
        links_with_centers,
        target_part=target_part,
        name_getter=lambda item: item[0],
    )
    if match is None:
        return None
    (_, center), matched_name, match_score = match
    return matched_name, center, match_score


def matching_named_part(
    parts: Any,
    *,
    target_part: str,
    name_getter: Any,
) -> tuple[Any, str, float] | None:
    best: tuple[float, str, Any] | None = None
    for part in parts:
        raw_name = name_getter(part)
        if not isinstance(raw_name, str) or not raw_name.strip():
            continue
        score = part_name_match_score(target_part, raw_name)
        if score is None:
            continue
        normalized_name = normalize_object_name(raw_name)
        candidate = (score, normalized_name, part)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        return None
    return best[2], str(name_getter(best[2])), float(best[0])


def part_name_match_score(target_part: str, candidate_name: str) -> float | None:
    target_tokens = normalized_part_tokens(target_part)
    candidate_tokens = normalized_part_tokens(candidate_name)
    if not target_tokens or not candidate_tokens:
        return None
    if target_tokens == candidate_tokens:
        return 100.0
    target_set = set(target_tokens)
    candidate_set = set(candidate_tokens)
    common = target_set & candidate_set
    if not common:
        return None
    target_head = target_tokens[-1]
    candidate_head = candidate_tokens[-1]
    score = 10.0 * len(common)
    if target_head in candidate_set:
        score += 20.0
    if candidate_head in target_set:
        score += 2.0
    score += len(common) / len(target_set)
    score += len(common) / len(candidate_set)
    score -= 0.05 * len(target_set ^ candidate_set)
    return score


def normalized_part_tokens(value: Any) -> tuple[str, ...]:
    tokens: list[str] = []
    for token in normalize_object_name(value).replace("-", " ").split():
        canonical = PART_NAME_TOKEN_ALIASES.get(token, token)
        if canonical in PART_NAME_IGNORED_TOKENS:
            continue
        tokens.append(canonical)
    return tuple(tokens)


def collision_part_center_2d(
    *, scene: HOVSGSceneAsset, part: dict[str, Any]
) -> tuple[float, float] | None:
    part_axis = str(part.get("vertical_axis") or "").strip().lower()
    scene_axis = str(getattr(scene, "vertical_axis", "z") or "z").lower()
    if part_axis and part_axis != scene_axis:
        return None
    points: list[tuple[float, float]] = []
    polygons = part.get("world_polygons")
    if isinstance(polygons, (list, tuple)):
        for polygon in polygons:
            if not isinstance(polygon, (list, tuple)):
                continue
            for point in polygon:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                try:
                    points.append((float(point[0]), float(point[1])))
                except (TypeError, ValueError):
                    continue
    if points:
        return (
            0.5 * (min(point[0] for point in points) + max(point[0] for point in points)),
            0.5 * (min(point[1] for point in points) + max(point[1] for point in points)),
        )
    minimum = part.get("min")
    maximum = part.get("max")
    if not (
        isinstance(minimum, (list, tuple))
        and isinstance(maximum, (list, tuple))
        and len(minimum) >= 3
        and len(maximum) >= 3
    ):
        return None
    try:
        center = {
            "x": 0.5 * (float(minimum[0]) + float(maximum[0])),
            "y": 0.5 * (float(minimum[1]) + float(maximum[1])),
            "z": 0.5 * (float(minimum[2]) + float(maximum[2])),
        }
    except (TypeError, ValueError):
        return None
    return project_point_to_scene_plane(scene=scene, point=center)


def model_footprint_polygon(
    *,
    scene: HOVSGSceneAsset,
    match: dict[str, Any],
    model_metadata: dict[str, Any],
) -> list[tuple[float, float]] | None:
    bbox_size = model_metadata.get("bbox_size")
    position = match.get("position")
    if (
        not isinstance(bbox_size, list)
        or len(bbox_size) < 3
        or not all(
            isinstance(value, (int, float)) and float(value) > 0.0 for value in bbox_size[:3]
        )
        or not isinstance(position, list)
        or len(position) < 3
        or not all(isinstance(value, (int, float)) for value in position[:3])
    ):
        return None
    hx, hy, hz = (float(value) * 0.5 for value in bbox_size[:3])
    if scene.vertical_axis == "z":
        local_corners = [(-hx, -hy, 0.0), (hx, -hy, 0.0), (hx, hy, 0.0), (-hx, hy, 0.0)]
    elif scene.vertical_axis == "y":
        local_corners = [(-hx, 0.0, -hz), (hx, 0.0, -hz), (hx, 0.0, hz), (-hx, 0.0, hz)]
    elif scene.vertical_axis == "x":
        local_corners = [(0.0, -hy, -hz), (0.0, hy, -hz), (0.0, hy, hz), (0.0, -hy, hz)]
    else:
        return None
    polygon: list[tuple[float, float]] = []
    for corner in local_corners:
        rotated = rotate_vector_by_quaternion(corner, match.get("orientation"))
        world = {
            "x": float(position[0]) + rotated[0],
            "y": float(position[1]) + rotated[1],
            "z": float(position[2]) + rotated[2],
        }
        projected = project_point_to_scene_plane(scene=scene, point=world)
        if projected is None:
            return None
        polygon.append(projected)
    return polygon


def link_box_center(link_boxes: Any) -> tuple[float, float, float] | None:
    if not isinstance(link_boxes, dict):
        return None
    for kind in ("visual", "collision"):
        for box_type in ("axis_aligned", "oriented"):
            box = (link_boxes.get(kind) or {}).get(box_type)
            if not isinstance(box, dict):
                continue
            transform = box.get("transform")
            if (
                isinstance(transform, list)
                and len(transform) >= 3
                and all(isinstance(row, list) and len(row) >= 4 for row in transform[:3])
            ):
                values = (transform[0][3], transform[1][3], transform[2][3])
                if all(isinstance(value, (int, float)) for value in values):
                    return float(values[0]), float(values[1]), float(values[2])
    return None


def rotate_vector_by_quaternion(
    vector: tuple[float, float, float],
    orientation: object,
) -> tuple[float, float, float]:
    if (
        not isinstance(orientation, list)
        or len(orientation) != 4
        or not all(isinstance(value, (int, float)) for value in orientation)
    ):
        return vector
    qx, qy, qz, qw = (float(value) for value in orientation)
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 1e-9:
        return vector
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    vx, vy, vz = vector

    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def project_vector_to_scene_plane(
    *,
    scene: HOVSGSceneAsset,
    vector: tuple[float, float, float],
) -> tuple[float, float] | None:
    if scene.vertical_axis == "z":
        return float(vector[0]), float(vector[1])
    if scene.vertical_axis == "y":
        return float(vector[0]), float(vector[2])
    if scene.vertical_axis == "x":
        return float(vector[1]), float(vector[2])
    return None


def project_point_to_scene_plane(
    *,
    scene: HOVSGSceneAsset,
    point: dict[str, float],
) -> tuple[float, float] | None:
    if scene.vertical_axis == "z":
        return float(point["x"]), float(point["y"])
    if scene.vertical_axis == "y":
        return float(point["x"]), float(point["z"])
    if scene.vertical_axis == "x":
        return float(point["y"]), float(point["z"])
    return None


def read_json_or_none(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def rear_direction_from_nearest_portal(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    object_id: str,
    object_room_id: str | None,
    object_xy: tuple[float, float],
) -> tuple[float, float] | None:
    nearest: tuple[float, tuple[float, float]] | None = None
    for obj in scene.objects.values():
        if str(obj.object_id) == str(object_id):
            continue
        if object_room_id is not None and obj.room_id != object_room_id:
            continue
        if not is_portal_like_object_name(obj.name):
            continue
        portal_xy = adapter._project_horizontal(scene, obj.centroid or {})
        if portal_xy is None:
            continue
        distance = math.hypot(portal_xy[0] - object_xy[0], portal_xy[1] - object_xy[1])
        if nearest is None or distance < nearest[0]:
            nearest = (distance, portal_xy)
    if nearest is None:
        return None
    portal_xy = nearest[1]
    dx = object_xy[0] - portal_xy[0]
    dy = object_xy[1] - portal_xy[1]
    norm = math.hypot(dx, dy)
    if norm <= 1e-9:
        return None
    return dx / norm, dy / norm


def annotate_object_part_approach_candidate(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    candidate: dict[str, Any],
    object_xy: tuple[float, float],
    part_context: dict[str, Any] | None,
) -> None:
    if not isinstance(part_context, dict):
        return
    candidate_xy = adapter._project_horizontal(scene, candidate)
    direction = part_context.get("preferred_direction")
    if candidate_xy is None or not isinstance(direction, tuple) or len(direction) != 2:
        return
    dx = candidate_xy[0] - object_xy[0]
    dy = candidate_xy[1] - object_xy[1]
    projection = dx * direction[0] + dy * direction[1]
    lateral_offset = abs(dx * -direction[1] + dy * direction[0])
    projection_deficit = max(0.0, -float(projection))
    standoff_score = float(candidate.get("candidate_geometry_score", 0.0))
    candidate["target_part"] = part_context.get("target_part")
    candidate["target_part_score"] = (
        projection_deficit + standoff_score + PART_APPROACH_LATERAL_OFFSET_WEIGHT * lateral_offset
    )
    candidate["target_part_alignment_m"] = float(projection)
    candidate["target_part_lateral_offset_m"] = float(lateral_offset)
    candidate["target_part_projection_deficit_m"] = projection_deficit
    candidate["target_part_strategy"] = part_context.get("strategy")
    if part_context.get("resolved_part_link"):
        candidate["target_part_link"] = part_context["resolved_part_link"]
    if isinstance(part_context.get("part_match_score"), (int, float)):
        candidate["target_part_match_score"] = float(part_context["part_match_score"])


def object_part_candidate_clearance_radius(
    *,
    default_radius_m: float,
    object_xy: tuple[float, float],
    candidate_xy: tuple[float, float],
    part_context: dict[str, Any] | None,
) -> float:
    radius = max(0.0, float(default_radius_m))
    if radius <= 0.0 or not isinstance(part_context, dict):
        return radius
    direction = part_context.get("preferred_direction")
    if not isinstance(direction, tuple) or len(direction) != 2:
        return radius
    projection = (candidate_xy[0] - object_xy[0]) * direction[0] + (
        candidate_xy[1] - object_xy[1]
    ) * direction[1]
    if projection <= 0.0:
        return radius
    return 0.0


def object_polygon_2d(
    adapter: Any,
    scene: HOVSGSceneAsset,
    obj: HOVSGObjectAsset | EffectiveObjectState,
) -> list[tuple[float, float]]:
    if isinstance(obj, EffectiveObjectState):
        return list(obj.footprint)
    runtime_scene_state = hovsg_runtime_state.current_scene_state(adapter, scene.scene_id)
    runtime_object = (
        hovsg_runtime_state.match_runtime_object(
            runtime_scene_state,
            object_name=obj.name,
            object_id=obj.object_id,
            static_centroid=obj.centroid,
        )
        if runtime_scene_state is not None
        else None
    )
    runtime_points: list[tuple[float, float]] = []
    if runtime_object is not None:
        for part in runtime_object.collision_parts:
            polygons = part.get("world_polygons") if isinstance(part, dict) else None
            if not isinstance(polygons, (list, tuple)):
                continue
            for polygon in polygons:
                if not isinstance(polygon, (list, tuple)):
                    continue
                for point in polygon:
                    if not isinstance(point, (list, tuple)) or len(point) < 2:
                        continue
                    try:
                        runtime_points.append((float(point[0]), float(point[1])))
                    except (TypeError, ValueError):
                        continue
    runtime_polygon = convex_hull_2d(runtime_points)
    if runtime_polygon:
        return runtime_polygon

    projected = [
        adapter._project_horizontal(scene, {"x": v[0], "y": v[1], "z": v[2]}) for v in obj.vertices
    ]
    polygon = [vertex for vertex in projected if vertex is not None]

    overlay_position, position_source = hovsg_runtime_state.resolve_object_centroid(
        adapter, scene, obj
    )
    if (
        polygon
        and position_source == "runtime_overlay"
        and obj.centroid is not None
        and overlay_position is not None
    ):
        static_xy = adapter._project_horizontal(scene, obj.centroid)
        overlay_xy = adapter._project_horizontal(scene, overlay_position)
        if static_xy is not None and overlay_xy is not None:
            delta = (overlay_xy[0] - static_xy[0], overlay_xy[1] - static_xy[1])
            if abs(delta[0]) > 1e-9 or abs(delta[1]) > 1e-9:
                polygon = [(vertex[0] + delta[0], vertex[1] + delta[1]) for vertex in polygon]
    return polygon


def convex_hull_2d(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
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


def object_proximity_evidence(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    candidate: dict[str, Any],
    path_nodes: list[Any],
) -> dict[str, Any]:
    candidate_xy = adapter._project_horizontal(scene, candidate)
    target_object_id = str(candidate.get("object_id") or "").strip()
    target_object_name = normalize_object_name(candidate.get("object_name"))
    path_points = [
        xy
        for node in path_nodes
        for xy in [
            adapter._project_horizontal(scene, adapter._node_to_waypoint(scene.nav_graph, node))
        ]
        if xy is not None
    ]

    effective_view = hovsg_effective_scene.effective_scene_view(adapter, scene)
    nearby_objects: list[dict[str, Any]] = []
    for obj in effective_view.objects.values():
        if str(obj.object_id) == target_object_id:
            continue
        if target_object_name and normalize_object_name(obj.name) == target_object_name:
            continue
        if candidate.get("room_id") is not None and obj.room_id != candidate.get("room_id"):
            continue
        if not obj.participates_in_navigation or not obj.navigation_footprints:
            continue
        if is_nonblocking_wall_fixture(obj.name):
            continue
        polygon = object_polygon_2d(adapter, scene, obj)
        if len(polygon) < 2:
            continue
        candidate_distance = (
            point_to_object_distance(adapter, point=candidate_xy, polygon=polygon)
            if candidate_xy is not None
            else float("inf")
        )
        path_distance = path_to_object_distance(adapter, path_points=path_points, polygon=polygon)
        closest_distance = min(candidate_distance, path_distance)
        nearby_objects.append(
            {
                "object_id": obj.object_id,
                "object_name": obj.name,
                "room_id": obj.room_id,
                "distance_to_candidate_m": round(float(candidate_distance), 3),
                "distance_to_path_m": round(float(path_distance), 3),
                "closest_distance_m": round(float(closest_distance), 3),
            }
        )

    candidate_nearest = min(
        nearby_objects,
        key=lambda item: (
            float(item["distance_to_candidate_m"]),
            str(item.get("object_id")),
        ),
        default={},
    )
    nearby_objects.sort(
        key=lambda item: (float(item["closest_distance_m"]), str(item.get("object_id")))
    )
    path_nearest = min(
        nearby_objects,
        key=lambda item: (
            float(item["distance_to_path_m"]),
            str(item.get("object_id")),
        ),
        default={},
    )
    return {
        "nearest_object_id": candidate_nearest.get("object_id"),
        "nearest_object_name": candidate_nearest.get("object_name"),
        "nearest_object_distance_m": candidate_nearest.get("distance_to_candidate_m"),
        "path_nearest_object_id": path_nearest.get("object_id"),
        "path_nearest_object_name": path_nearest.get("object_name"),
        "path_nearest_object_distance_m": path_nearest.get("distance_to_path_m"),
        "nearby_objects": nearby_objects[:5],
    }


def object_approach_candidate_is_too_close_to_portal(
    candidate: dict[str, Any],
    *,
    min_clearance_m: float,
) -> bool:
    if min_clearance_m <= 0.0:
        return False
    evidence = candidate.get("nearby_object_evidence")
    if not isinstance(evidence, dict):
        return False
    nearby_objects = evidence.get("nearby_objects")
    if not isinstance(nearby_objects, list):
        return False
    for nearby_object in nearby_objects:
        if not isinstance(nearby_object, dict):
            continue
        if not is_portal_like_object_name(nearby_object.get("object_name")):
            continue
        distance_to_candidate = nearby_object.get("distance_to_candidate_m")
        if not isinstance(distance_to_candidate, (int, float)):
            continue
        if float(distance_to_candidate) < min_clearance_m:
            return True
    return False


def is_portal_like_object_name(value: Any) -> bool:
    name = normalize_object_name(value).replace("-", " ")
    if not name:
        return False
    tokens = set(name.split())
    if tokens & PORTAL_LIKE_OBJECT_NAME_TOKENS:
        return True
    return any(token in name for token in ("doorway", "gateway"))


def object_blocks_navigation_clearance(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    obj: HOVSGObjectAsset,
    candidate: dict[str, Any],
) -> bool:
    if not object_overlaps_navigation_height(adapter, scene=scene, obj=obj, candidate=candidate):
        return False
    if is_nonblocking_wall_fixture(obj.name):
        return False
    return True


def is_nonblocking_wall_fixture(value: Any) -> bool:
    name = normalize_object_name(value).replace("-", " ")
    if not name:
        return False
    tokens = set(name.split())
    if "window" in tokens:
        return True
    fixture_names = {
        "art",
        "decor",
        "decoration",
        "frame",
        "map",
        "mirror",
        "painting",
        "photo",
        "photograph",
        "picture",
        "poster",
    }
    return bool(tokens & fixture_names)


def object_overlaps_navigation_height(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    obj: HOVSGObjectAsset,
    candidate: dict[str, Any],
    robot_clearance_height_m: float = 1.6,
    floor_slop_m: float = 0.15,
) -> bool:
    values = [
        adapter._to_float(vertex[_axis_index(scene.vertical_axis)])
        for vertex in obj.vertices
        if len(vertex) >= 3
    ]
    values = [value for value in values if value is not None]
    if not values:
        return True
    object_min = min(values)
    object_max = max(values)
    floor_value = adapter._to_float(candidate.get(scene.vertical_axis))
    if floor_value is None:
        floor = scene.floors.get(str(candidate.get("floor_id") or ""))
        floor_value = floor.floor_zero_level if floor is not None else None
    if floor_value is None:
        floor_value = 0.0
    navigation_min = float(floor_value) - floor_slop_m
    navigation_max = float(floor_value) + robot_clearance_height_m
    return object_max >= navigation_min and object_min <= navigation_max


def _axis_index(axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}.get(axis, 1)


def point_to_object_distance(
    adapter: Any,
    *,
    point: tuple[float, float] | None,
    polygon: list[tuple[float, float]],
) -> float:
    if point is None:
        return float("inf")
    if len(polygon) >= 3 and adapter._point_in_polygon(point, polygon):
        return 0.0
    return adapter._point_to_polygon_boundary_distance(point, polygon)


def path_to_object_distance(
    adapter: Any,
    *,
    path_points: list[tuple[float, float]],
    polygon: list[tuple[float, float]],
) -> float:
    if not path_points:
        return float("inf")
    best_distance = min(
        point_to_object_distance(adapter, point=point, polygon=polygon) for point in path_points
    )
    if len(path_points) < 2 or len(polygon) < 2:
        return best_distance
    object_edges = list(zip(polygon[-1:] + polygon[:-1], polygon))
    for start, end in zip(path_points, path_points[1:]):
        for edge_start, edge_end in object_edges:
            best_distance = min(
                best_distance,
                segment_to_segment_distance(
                    adapter,
                    first_start=start,
                    first_end=end,
                    second_start=edge_start,
                    second_end=edge_end,
                ),
            )
            if best_distance <= 1e-9:
                return 0.0
    return best_distance


def segment_to_segment_distance(
    adapter: Any,
    *,
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
) -> float:
    if segments_intersect(first_start, first_end, second_start, second_end):
        return 0.0
    return min(
        adapter._distance_point_to_segment(first_start, second_start, second_end),
        adapter._distance_point_to_segment(first_end, second_start, second_end),
        adapter._distance_point_to_segment(second_start, first_start, first_end),
        adapter._distance_point_to_segment(second_end, first_start, first_end),
    )


def segments_intersect(
    a_start: tuple[float, float],
    a_end: tuple[float, float],
    b_start: tuple[float, float],
    b_end: tuple[float, float],
) -> bool:
    def orientation(
        p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]
    ) -> float:
        return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

    def on_segment(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> bool:
        return (
            min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9
            and min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9
        )

    o1 = orientation(a_start, a_end, b_start)
    o2 = orientation(a_start, a_end, b_end)
    o3 = orientation(b_start, b_end, a_start)
    o4 = orientation(b_start, b_end, a_end)
    if o1 * o2 < 0.0 and o3 * o4 < 0.0:
        return True
    return bool(
        abs(o1) <= 1e-9
        and on_segment(a_start, b_start, a_end)
        or abs(o2) <= 1e-9
        and on_segment(a_start, b_end, a_end)
        or abs(o3) <= 1e-9
        and on_segment(b_start, a_start, b_end)
        or abs(o4) <= 1e-9
        and on_segment(b_start, a_end, b_end)
    )


def normalize_object_name(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.lower().replace("_", " ").split()).strip()
