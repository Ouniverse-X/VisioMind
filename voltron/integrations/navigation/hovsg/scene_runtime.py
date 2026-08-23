"""Scene runtime and goal-grounding helpers for the HOV-SG navigator."""

from __future__ import annotations

import re
import math
from typing import Any

from . import effective_scene as hovsg_effective_scene
from . import runtime_state as hovsg_runtime_state
from . import scene_loading as hovsg_scene_loading
from .models import HOVSGSceneAsset


def load_scene(
    adapter: Any,
    scene_id: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    graph_path = hovsg_scene_loading.resolve_graph_path(
        scene_id=scene_id,
        graph_root=adapter.graph_root,
        scene_roots=adapter.scene_roots,
        config=config,
    )
    metadata = hovsg_scene_loading.load_metadata(graph_path)
    rooms = hovsg_scene_loading.load_rooms(graph_path / "rooms")
    loader_config = dict(config or {})
    if adapter.nav_graph_type is not None and "nav_graph_type" not in loader_config:
        loader_config["nav_graph_type"] = adapter.nav_graph_type
    nav_graph_asset = hovsg_scene_loading.load_nav_graph(
        graph_path / "nav_graph",
        metadata=metadata,
        config=loader_config,
    )
    scene_asset = HOVSGSceneAsset(
        scene_id=scene_id,
        graph_path=graph_path,
        floors=hovsg_scene_loading.load_floors(graph_path / "floors"),
        rooms=rooms,
        objects=hovsg_scene_loading.load_objects(graph_path / "objects"),
        nav_graph=nav_graph_asset.graph,
        nav_graph_type=nav_graph_asset.graph_type,
        nav_graph_asset_path=nav_graph_asset.graph_path,
        scene_map_source=hovsg_scene_loading.resolve_scene_map_source(metadata, config=config),
        vertical_axis=hovsg_scene_loading.resolve_vertical_axis(
            metadata,
            config=config,
            default_vertical_axis=adapter.vertical_axis,
        ),
        room_adjacency=hovsg_scene_loading.load_room_adjacency(graph_path=graph_path, rooms=rooms),
        metadata=metadata,
    )
    hovsg_scene_loading.validate_scene_graph_consistency(scene_asset)
    adapter._scenes[scene_id] = scene_asset
    adapter.default_scene_id = adapter.default_scene_id or scene_id
    return {
        "scene_id": scene_id,
        "graph_path": str(graph_path),
        "floors": len(scene_asset.floors),
        "rooms": len(scene_asset.rooms),
        "objects": len(scene_asset.objects),
        "nav_nodes": scene_asset.nav_graph.number_of_nodes(),
        "nav_edges": scene_asset.nav_graph.number_of_edges(),
        "nav_graph_type": scene_asset.nav_graph_type,
        "nav_graph_asset_path": str(scene_asset.nav_graph_asset_path) if scene_asset.nav_graph_asset_path else None,
        "scene_map_source": scene_asset.scene_map_source,
        "vertical_axis": scene_asset.vertical_axis,
    }


def update(
    adapter: Any,
    observation: dict[str, Any],
    *,
    pose: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scene_id = adapter._resolve_scene_id(observation=observation)
    scene = adapter._ensure_scene(scene_id) if scene_id else None
    localized = (
        adapter._localize_pose(
            scene,
            pose,
            previous_room_id=adapter._last_localized_room_ids.get(scene.scene_id),
            persist=True,
        )
        if scene is not None and pose is not None
        else {}
    )

    scene_state_summary = hovsg_runtime_state.ingest_scene_state(
        adapter,
        scene_id=scene_id,
        payload=observation.get("scene_state"),
    )

    backend_state: dict[str, Any] = {
        "scene_id": scene_id,
        "pose": dict(pose or {}),
    }
    if scene_state_summary is not None:
        backend_state["scene_state"] = scene_state_summary
    if scene is not None:
        backend_state["vertical_axis"] = scene.vertical_axis
        scene_map_seed = adapter._load_scene_map_payload(scene.graph_path)
        if isinstance(scene_map_seed, dict):
            backend_state["scene_map_seed"] = scene_map_seed
            backend_state["scene_map_seed_metadata"] = {
                "source": scene.scene_map_source
                or scene_map_seed.get("scene_map_source")
                or scene_map_seed.get("source")
                or "scene_map",
                "graph_path": str(scene.graph_path),
            }
    backend_state.update(localized)

    nav_feedback = observation.get("nav_feedback", {})
    if isinstance(nav_feedback, dict) and "active_waypoint_index" in nav_feedback:
        backend_state["active_waypoint_index"] = nav_feedback["active_waypoint_index"]
    return backend_state


def ground_goal(
    adapter: Any,
    instruction: str,
    *,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = dict(context or {})
    scene_id = adapter._resolve_scene_id(context=context)
    scene = adapter._ensure_scene(scene_id) if scene_id else None
    hovsg_runtime_state.ingest_scene_state_from_containers(
        adapter,
        scene_id=scene_id,
        containers=(
            context,
            context.get("parameters"),
            context.get("map_state"),
            context.get("observation"),
        ),
    )

    interpreted_goal = context.get("interpreted_goal")
    interpreted_position_goal = _position_goal_from_interpreted_goal(
        scene=scene,
        scene_id=scene_id,
        interpreted_goal=interpreted_goal,
        context=context,
    )
    if interpreted_position_goal is not None:
        return interpreted_position_goal

    target = context.get("target", {})
    interpreted_target = _target_from_interpreted_goal(interpreted_goal)
    if interpreted_target:
        base_target = dict(target) if isinstance(target, dict) else {}
        base_target.update(interpreted_target)
        target = base_target
    if isinstance(target, dict):
        room_query = adapter._first_non_empty(target, "room", "region", "location")
        object_query = adapter._first_non_empty(target, "object", "target", "item")
        part_query = adapter._first_non_empty(target, "part")
        floor_query = adapter._first_non_empty(target, "floor")
    else:
        room_query = None
        object_query = None
        part_query = None
        floor_query = None

    explicit_room_query = room_query if isinstance(room_query, str) and room_query.strip() else None
    explicit_object_query = object_query if isinstance(object_query, str) and object_query.strip() else None
    explicit_part_query = part_query if isinstance(part_query, str) and part_query.strip() else None
    explicit_floor_query = floor_query if isinstance(floor_query, str) and floor_query.strip() else None

    object_search_text: str | None = None
    if explicit_object_query is not None:
        object_search_text = explicit_object_query
    elif explicit_room_query is None and explicit_floor_query is None:
        object_search_text = instruction

    room_constraint = None
    if scene is not None and explicit_room_query is not None:
        room_constraint = adapter._match_room(scene, explicit_room_query)

    object_candidates = (
        _object_grounding_candidates(
            adapter=adapter,
            scene=scene,
            text=object_search_text,
            room_query=explicit_room_query,
            context=context,
        )
        if scene is not None
        else []
    )
    effective_view = (
        hovsg_effective_scene.effective_scene_view(adapter, scene)
        if scene is not None
        else None
    )
    object_match = _match_effective_object(
        adapter=adapter,
        view=effective_view,
        text=object_search_text,
        room_id=room_constraint.room_id if room_constraint is not None else None,
    )
    if object_match is not None:
        position = object_match.position
        position_source = (
            "runtime_overlay"
            if object_match.runtime_object_name is not None
            else "static_asset"
        )
        room_id = object_match.room_id
        room = scene.rooms.get(room_id)
        owner = hovsg_effective_scene.approach_owner(effective_view, object_match)
        runtime_state = hovsg_runtime_state.current_scene_state(adapter, scene_id)
        grounded = {
            "scene_id": scene_id,
            "goal_type": "object",
            "object_id": object_match.object_id,
            "object_name": object_match.name,
            "room_id": room_id,
            "room_name": room.name if room is not None else None,
            "floor_id": room.floor_id if room is not None else None,
            "position": dict(position or {}),
            "position_source": position_source,
            "grounding_query": object_search_text,
            "grounding_candidates": object_candidates,
            "relation_revision": object_match.relation_revision,
            "map_revision": effective_view.map_revision,
            "scene_state_signature": (
                runtime_state.signature if runtime_state is not None else ""
            ),
            "relation_signature": effective_view.relation_signature,
        }
        if owner is not None and owner.object_id != object_match.object_id:
            grounded["approach_owner_id"] = owner.object_id
            grounded["approach_owner_name"] = owner.name
            if (
                owner.object_id in object_match.relation_targets("inside")
                and any(item.relation == "closed" for item in owner.relations)
            ):
                grounded["interaction_prerequisites"] = ["open_container"]
        target_part_query = explicit_part_query or _infer_object_part_query(
            object_name=object_match.name,
            texts=(object_search_text, instruction),
        )
        if target_part_query is not None:
            grounded["target_part"] = target_part_query
        return _attach_interpreted_spatial_intent(grounded, interpreted_goal)

    room_search_text = explicit_room_query or instruction
    room_match = adapter._match_room(scene, room_search_text) if scene is not None else None
    if room_match is not None:
        return _attach_interpreted_spatial_intent(
            {
                "scene_id": scene_id,
                "goal_type": "room",
                "room_id": room_match.room_id,
                "room_name": room_match.name,
                "floor_id": room_match.floor_id,
                "position": dict(room_match.centroid or {}),
            },
            interpreted_goal,
        )

    floor_search_text = explicit_floor_query or instruction
    floor_match = adapter._match_floor(scene, floor_search_text) if scene is not None else None
    if floor_match is not None:
        return _attach_interpreted_spatial_intent(
            {
                "scene_id": scene_id,
                "goal_type": "floor",
                "floor_id": floor_match.floor_id,
                "floor_name": floor_match.name,
                "position": dict(floor_match.centroid or {}),
            },
            interpreted_goal,
        )

    return _attach_interpreted_spatial_intent(
        {
            "scene_id": scene_id,
            "goal_type": "unresolved",
            "instruction": instruction,
        },
        interpreted_goal,
    )


def _infer_object_part_query(*, object_name: str | None, texts: tuple[str | None, ...]) -> str | None:
    object_tokens = set(normalize_text(str(object_name or "")).split())
    if not (object_tokens & {"car", "automobile", "vehicle"}):
        return None
    rear_tokens = ("trunk", "rear", "boot", "hatch", "tailgate")
    for text in texts:
        normalized = normalize_text(str(text or ""))
        tokens = set(normalized.split())
        for token in rear_tokens:
            if token in tokens:
                return "trunk" if token in {"boot", "hatch", "tailgate"} else token
    return None


def _target_from_interpreted_goal(interpreted_goal: Any) -> dict[str, Any]:
    if not isinstance(interpreted_goal, dict):
        return {}
    goal_kind = str(interpreted_goal.get("goal_kind") or "").strip().lower()
    target_query = interpreted_goal.get("target_query")
    target: dict[str, Any] = dict(target_query) if isinstance(target_query, dict) else {}
    _apply_interaction_followup_target(target, interpreted_goal)
    landmark = str(target.get("landmark") or "").strip().lower()
    threshold_like = any(token in landmark for token in ("threshold", "doorway", "entrance", "passage", "opening"))
    if goal_kind == "doorway" and not any(target.get(key) for key in ("object", "target", "item")):
        target["object"] = "door"
    elif goal_kind == "landmark" and threshold_like and not any(
        target.get(key) for key in ("object", "target", "item", "room", "region", "location")
    ):
        target["object"] = "door"
    return {key: value for key, value in target.items() if isinstance(value, str) and value.strip()}


def _object_grounding_candidates(
    *,
    adapter: Any,
    scene: HOVSGSceneAsset | None,
    text: str | None,
    room_query: str | None = None,
    context: dict[str, Any] | None = None,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    if scene is None or not isinstance(text, str) or not text.strip():
        return []
    effective_view = hovsg_effective_scene.effective_scene_view(adapter, scene)
    objects = [item for item in effective_view.objects.values() if item.approachable]

    start_pose = _start_pose_from_context(context)
    ranked: list[tuple[float, Any, dict[str, float] | None, str]] = []
    for obj in objects:
        score = max(
            adapter._score_text_match(str(obj.name or obj.object_id), text),
            adapter._score_text_match(str(obj.category or ""), text),
        )
        if score <= 0.0:
            continue
        room_score = 0.0
        if isinstance(room_query, str) and room_query.strip():
            room = scene.rooms.get(obj.room_id)
            room_score = adapter._score_text_match(
                " ".join(
                    value
                    for value in (room.name if room is not None else None, obj.room_id, room_query)
                    if isinstance(value, str) and value.strip()
                ),
                room_query,
            )
        position = obj.position
        position_source = (
            "runtime_overlay" if obj.runtime_object_name is not None else "static_asset"
        )
        ranked.append((score + 0.1 * room_score, obj, position, position_source))
    ranked.sort(
        key=lambda item: (
            -item[0],
            _candidate_distance_to_pose(item[2], start_pose),
            str(item[1].object_id),
        )
    )

    candidates: list[dict[str, Any]] = []
    for score, obj, position, position_source in ranked[: max(1, int(top_k))]:
        room = scene.rooms.get(obj.room_id)
        candidate = {
            "object_id": obj.object_id,
            "object_name": obj.name,
            "room_id": obj.room_id,
            "room_name": room.name if room is not None else None,
            "floor_id": room.floor_id if room is not None else None,
            "position": dict(position or {}),
            "position_source": position_source,
            "text_match_score": score,
            "grounding_query": text,
            "selection_status": "candidate",
            "geometry_source": obj.geometry_source,
            "relation_revision": obj.relation_revision,
        }
        if start_pose is not None:
            distance = _candidate_distance_to_pose(position, start_pose)
            if math.isfinite(distance):
                candidate["distance_to_start_m"] = distance
        candidates.append(candidate)
    return candidates


def _match_effective_object(
    *,
    adapter: Any,
    view: hovsg_effective_scene.EffectiveSceneView | None,
    text: str | None,
    room_id: str | None,
) -> hovsg_effective_scene.EffectiveObjectState | None:
    if view is None or not isinstance(text, str) or not text.strip():
        return None
    ranked = []
    for obj in view.objects.values():
        if not obj.approachable:
            continue
        if room_id is not None and obj.room_id != room_id:
            continue
        score = max(
            adapter._score_text_match(str(obj.name or obj.object_id), text),
            adapter._score_text_match(str(obj.category or ""), text),
        )
        if score > 0.0:
            ranked.append((score, obj))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[0], item[1].object_id))
    return ranked[0][1]


def _start_pose_from_context(context: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(context, dict):
        return None
    start = context.get("start")
    if isinstance(start, dict) and isinstance(start.get("pose"), dict):
        return dict(start["pose"])
    backend_state = context.get("backend_state")
    if isinstance(backend_state, dict) and isinstance(backend_state.get("pose"), dict):
        return dict(backend_state["pose"])
    map_state = context.get("map_state")
    if isinstance(map_state, dict) and isinstance(map_state.get("pose"), dict):
        return dict(map_state["pose"])
    return None


def _candidate_distance_to_pose(centroid: dict[str, float] | None, pose: dict[str, Any] | None) -> float:
    if not isinstance(centroid, dict) or not isinstance(pose, dict):
        return float("inf")
    try:
        cx = float(centroid["x"])
        px = float(pose["x"])
    except (KeyError, TypeError, ValueError):
        return float("inf")
    for axis in ("y", "z"):
        if axis in centroid and axis in pose:
            try:
                return math.hypot(cx - px, float(centroid[axis]) - float(pose[axis]))
            except (TypeError, ValueError):
                return float("inf")
    return abs(cx - px)


def _apply_interaction_followup_target(target: dict[str, Any], interpreted_goal: dict[str, Any]) -> None:
    existing_object = next(
        (
            str(target.get(key)).strip()
            for key in ("object", "target", "item")
            if isinstance(target.get(key), str) and str(target.get(key)).strip()
        ),
        "",
    )
    if existing_object and not _is_passage_like_text(existing_object):
        return

    stop_condition = interpreted_goal.get("stop_condition")
    if isinstance(stop_condition, str):
        stop_conditions = {stop_condition.strip().lower()}
    elif isinstance(stop_condition, list):
        stop_conditions = {str(item).strip().lower() for item in stop_condition if str(item).strip()}
    else:
        stop_conditions = set()
    interaction_conditions = {
        "interaction_ready",
        "object_reachable",
        "object_visible",
        "view_ready",
    }
    if not (stop_conditions & interaction_conditions):
        return

    followup_context = interpreted_goal.get("followup_context")
    if isinstance(followup_context, dict):
        for key in ("target", "object", "item"):
            value = followup_context.get(key)
            if isinstance(value, str) and value.strip():
                target["object"] = value.strip()
                return

    followup_target = _interaction_target_from_natural_language_context(interpreted_goal)
    if followup_target:
        target["object"] = followup_target


def _is_passage_like_text(value: str) -> bool:
    normalized = normalize_text(value)
    return any(
        token in normalized
        for token in (
            "threshold",
            "doorway",
            "entrance",
            "passage",
            "opening",
            "portal",
            "door",
        )
    )


def _interaction_target_from_natural_language_context(interpreted_goal: dict[str, Any]) -> str | None:
    for text in _interaction_context_texts(interpreted_goal):
        candidate = _extract_interaction_object_phrase(text)
        if candidate:
            return candidate
    for text in _interaction_context_texts(interpreted_goal):
        normalized = normalize_text(text)
        if normalized and normalized not in _INTERACTION_ONLY_TERMS:
            return normalized
    return None


def _interaction_context_texts(interpreted_goal: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    followup_context = interpreted_goal.get("followup_context")
    if isinstance(followup_context, dict):
        for key in (
            "task",
            "instruction",
            "objective",
            "intent",
            "purpose",
            "action",
            "intended_action",
            "next_action",
            "description",
            "free-form",
            "free_form",
            "freeform",
        ):
            value = followup_context.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())

    constraints = interpreted_goal.get("constraints")
    if isinstance(constraints, dict):
        for value in constraints.values():
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())

    return texts


_INTERACTION_VERB_PATTERN = re.compile(
    r"\b(?:"
    r"interact\s+with|"
    r"turn\s+on|turn\s+off|switch\s+on|switch\s+off|pick\s+up|"
    r"press|push|pull|toggle|turn|switch|open|close|use|operate|activate|deactivate|"
    r"grasp|grab|pick|place|move|lift|lower|rotate|slide|touch|manipulate"
    r")\b\s+([^,.;]+)",
    re.IGNORECASE,
)
_VISIBLE_OBJECT_PATTERN = re.compile(
    r"\b([a-z0-9][a-z0-9 _-]{1,80}?)\s+"
    r"(?:is|are|be|becomes|become)\s+"
    r"(?:visible|reachable|accessible|in\s+view|within\s+reach)\b",
    re.IGNORECASE,
)
_INTERACTION_ONLY_TERMS = {
    "press",
    "push",
    "pull",
    "toggle",
    "turn",
    "open",
    "close",
    "use",
    "operate",
    "activate",
    "deactivate",
    "interact",
    "manipulate",
}


def _extract_interaction_object_phrase(text: str) -> str | None:
    for pattern in (_INTERACTION_VERB_PATTERN, _VISIBLE_OBJECT_PATTERN):
        match = pattern.search(text)
        if match is None:
            continue
        candidate = _clean_interaction_object_phrase(match.group(1))
        if candidate:
            return candidate
    return None


def _clean_interaction_object_phrase(value: str) -> str | None:
    normalized = normalize_text(value)
    normalized = re.sub(
        r"^(?:after|before|then|next|and|near|nearest|nearby|at|on|off|to|with|the|a|an|this|that|"
        r"must|be|such|that|so|robot|can|could|should|will|enable|enables|enabling)\s+",
        "",
        normalized,
    )
    normalized = re.sub(r"\b(?:to|so that|for|before|after|while|when|once|and then)\b.*$", "", normalized).strip()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized or normalized in _INTERACTION_ONLY_TERMS:
        return None
    return normalized


def _attach_interpreted_spatial_intent(goal: dict[str, Any], interpreted_goal: Any) -> dict[str, Any]:
    if not isinstance(interpreted_goal, dict):
        return goal

    enriched = dict(goal)
    spatial_relation = interpreted_goal.get("spatial_relation")
    if isinstance(spatial_relation, str) and spatial_relation.strip():
        enriched["spatial_relation"] = spatial_relation.strip().lower()

    stop_condition = interpreted_goal.get("stop_condition")
    if isinstance(stop_condition, str):
        enriched["stop_condition"] = [stop_condition.strip().lower()]
    elif isinstance(stop_condition, list):
        enriched["stop_condition"] = [
            str(item).strip().lower()
            for item in stop_condition
            if str(item).strip()
        ]

    followup_context = interpreted_goal.get("followup_context")
    if isinstance(followup_context, dict):
        enriched["followup_context"] = dict(followup_context)

    constraints = interpreted_goal.get("constraints")
    if isinstance(constraints, dict):
        enriched["constraints"] = dict(constraints)

    enriched["interpreted_goal"] = dict(interpreted_goal)
    return enriched


def _position_goal_from_interpreted_goal(
    *,
    scene: HOVSGSceneAsset | None,
    scene_id: str | None,
    interpreted_goal: Any,
    context: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(interpreted_goal, dict):
        return None
    goal_kind = str(interpreted_goal.get("goal_kind") or "").strip().lower()
    if goal_kind == "position" and isinstance(interpreted_goal.get("position"), dict):
        return {
            "scene_id": scene_id,
            "goal_type": "position",
            "position": dict(interpreted_goal["position"]),
            "source": "llm_interpreted_goal",
        }
    if goal_kind != "relative_motion":
        return None

    relative_motion = interpreted_goal.get("relative_motion")
    if not isinstance(relative_motion, dict):
        return None
    distance = _to_float(relative_motion.get("distance_m"))
    if distance is None:
        return None
    direction = str(relative_motion.get("direction") or "forward").strip().lower()
    pose, orientation = _pose_and_orientation_from_context(context)
    if pose is None:
        return None

    vertical_axis = scene.vertical_axis if scene is not None else str(context.get("vertical_axis") or "z")
    axes = _horizontal_axes(vertical_axis)
    yaw = _to_float((orientation or {}).get("yaw")) or 0.0
    forward = {
        axes[0]: math.cos(yaw),
        axes[1]: math.sin(yaw),
    }
    left = {
        axes[0]: -math.sin(yaw),
        axes[1]: math.cos(yaw),
    }
    if direction in {"back", "backward", "backwards"}:
        vector = {axis: -value for axis, value in forward.items()}
    elif direction == "left":
        vector = left
    elif direction == "right":
        vector = {axis: -value for axis, value in left.items()}
    else:
        vector = forward

    position = dict(pose)
    for axis in ("x", "y", "z"):
        position[axis] = float(position.get(axis, 0.0))
    for axis, unit in vector.items():
        position[axis] = float(position.get(axis, 0.0)) + float(distance) * unit

    return {
        "scene_id": scene_id,
        "goal_type": "position",
        "position": position,
        "source": "llm_interpreted_goal",
        "interpreted_goal": dict(interpreted_goal),
    }


def _pose_and_orientation_from_context(context: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    for container_key in ("start", "backend_state", "map_state"):
        container = context.get(container_key)
        if not isinstance(container, dict):
            continue
        pose = container.get("pose")
        if isinstance(pose, dict):
            orientation = container.get("orientation")
            return dict(pose), dict(orientation) if isinstance(orientation, dict) else None
    pose = context.get("pose")
    if isinstance(pose, dict):
        orientation = context.get("orientation")
        return dict(pose), dict(orientation) if isinstance(orientation, dict) else None
    return None, None


def _horizontal_axes(vertical_axis: str) -> tuple[str, str]:
    return {
        "x": ("y", "z"),
        "y": ("x", "z"),
        "z": ("x", "y"),
    }.get(vertical_axis, ("x", "y"))


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ensure_scene(adapter: Any, scene_id: str | None) -> HOVSGSceneAsset | None:
    if not scene_id:
        return None
    if scene_id not in adapter._scenes:
        adapter.load_scene(scene_id)
    return adapter._scenes.get(scene_id)


def resolve_scene_id(
    adapter: Any,
    *,
    observation: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> str | None:
    observation = dict(observation or {})
    context = dict(context or {})
    for candidate in (
        observation.get("scene_id"),
        context.get("scene_id"),
        adapter.default_scene_id,
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def first_non_empty(mapping: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def normalize_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split()).strip()


def compact_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def score_text_match(adapter: Any, candidate: str, query: str) -> float:
    normalized_candidate = adapter._normalize_text(candidate)
    normalized_query = adapter._normalize_text(query)
    compact_candidate = adapter._compact_text(candidate)
    compact_query = adapter._compact_text(query)
    if not normalized_candidate or not normalized_query:
        return 0.0
    if normalized_candidate == normalized_query:
        return 10.0
    if compact_candidate and compact_candidate == compact_query:
        return 9.5
    if normalized_query in normalized_candidate or normalized_candidate in normalized_query:
        return 8.0
    if compact_query and (compact_query in compact_candidate or compact_candidate in compact_query):
        return 7.5

    candidate_tokens = set(normalized_candidate.split())
    query_tokens = set(normalized_query.split())
    if not candidate_tokens or not query_tokens:
        return 0.0
    overlap = candidate_tokens & query_tokens
    if not overlap:
        return 0.0
    precision = len(overlap) / len(candidate_tokens)
    recall = len(overlap) / len(query_tokens)
    return 5.0 + precision + recall
