"""Runtime scene-state overlay for the HOV-SG navigator.

Static HOV-SG assets are loaded once and treated as immutable; live object
poses and door states sampled from the simulator are kept here, per scene, and
consulted before any static centroid/adjacency lookup.
"""

from __future__ import annotations

import re
from typing import Any

from voltron.shared.models.scene_state import (
    RuntimeDoorState,
    RuntimeObjectState,
    SceneRuntimeState,
    door_is_navigation_passable,
    scene_runtime_state_from_payload,
)

from .models import HOVSGObjectAsset, HOVSGSceneAsset

_OVERLAY_ROOM_REASSIGN_DISTANCE_M = 0.5


def ingest_scene_state(
    adapter: Any,
    *,
    scene_id: str | None,
    payload: Any,
) -> dict[str, Any] | None:
    """Parse and store a scene-state payload; returns a telemetry summary."""
    if not scene_id:
        return None
    state = scene_runtime_state_from_payload(payload)
    if state is None:
        return None
    store = _store(adapter)
    previous = store.get(scene_id)
    if (
        previous is not None
        and previous.signature == state.signature
        and previous.relation_signature == state.relation_signature
        and previous.step == state.step
    ):
        return _summary(previous)
    store[scene_id] = state
    return _summary(state)


def ingest_scene_state_from_containers(
    adapter: Any,
    *,
    scene_id: str | None,
    containers: tuple[dict[str, Any] | None, ...],
) -> dict[str, Any] | None:
    for container in containers:
        if not isinstance(container, dict):
            continue
        payload = container.get("scene_state")
        if isinstance(payload, dict) and payload:
            return ingest_scene_state(adapter, scene_id=scene_id, payload=payload)
    return None


def current_scene_state(adapter: Any, scene_id: str | None) -> SceneRuntimeState | None:
    if not scene_id:
        return None
    return _store(adapter).get(scene_id)


def scene_state_signature(adapter: Any, scene_id: str | None) -> str:
    state = current_scene_state(adapter, scene_id)
    return state.signature if state is not None else ""


def door_signature(adapter: Any, scene_id: str | None) -> str:
    state = current_scene_state(adapter, scene_id)
    return state.door_signature() if state is not None else ""


def door_states(adapter: Any, scene_id: str | None) -> dict[str, RuntimeDoorState]:
    state = current_scene_state(adapter, scene_id)
    return dict(state.doors) if state is not None else {}


def relation_signature(adapter: Any, scene_id: str | None) -> str:
    state = current_scene_state(adapter, scene_id)
    return state.relation_signature if state is not None else ""


def resolve_object_centroid(
    adapter: Any,
    scene: HOVSGSceneAsset,
    obj: HOVSGObjectAsset,
) -> tuple[dict[str, float] | None, str]:
    """Return the freshest known centroid for a static object and its source."""
    state = current_scene_state(adapter, scene.scene_id)
    if state is not None:
        runtime_object = match_runtime_object(
            state,
            object_name=obj.name,
            object_id=obj.object_id,
            static_centroid=obj.centroid,
        )
        if runtime_object is not None and runtime_object.position is not None:
            return dict(runtime_object.position), "runtime_overlay"
    return (dict(obj.centroid) if obj.centroid is not None else None), "static_asset"


def match_runtime_object(
    state: SceneRuntimeState,
    *,
    object_name: str | None,
    object_id: str | None,
    static_centroid: dict[str, float] | None,
) -> RuntimeObjectState | None:
    candidates = state.objects
    if not candidates:
        return None
    for key in (object_name, object_id):
        if isinstance(key, str) and key in candidates:
            return candidates[key]

    # Runtime names carry extra junk tokens (model code, instance index), so
    # only the conservative direction is safe: every meaningful static-name
    # token must appear in the runtime name. No match → static fallback.
    target_tokens = _meaningful_tokens(object_name) or _meaningful_tokens(object_id)
    if not target_tokens:
        return None
    matched: list[RuntimeObjectState] = []
    for runtime_object in candidates.values():
        runtime_tokens = _meaningful_tokens(runtime_object.name)
        if runtime_tokens and target_tokens <= runtime_tokens:
            matched.append(runtime_object)
    if not matched:
        return None
    if len(matched) == 1:
        return matched[0]
    return min(
        matched,
        key=lambda item: _distance_sq(item.position, static_centroid),
    )


def match_runtime_door(
    state: SceneRuntimeState,
    *,
    object_name: str | None,
    object_id: str | None,
    static_centroid: dict[str, float] | None,
) -> RuntimeDoorState | None:
    """Match a static HOV-SG door object to the dedicated runtime door layer."""
    candidates = state.doors
    if not candidates:
        return None
    for key in (object_name, object_id):
        if isinstance(key, str) and key in candidates:
            return candidates[key]
    target_tokens = _meaningful_tokens(object_name) or _meaningful_tokens(object_id)
    if not target_tokens:
        return None
    matched = [
        door
        for door in candidates.values()
        if target_tokens <= _meaningful_tokens(door.name)
    ]
    if not matched:
        return None
    if len(matched) == 1:
        return matched[0]
    return min(matched, key=lambda item: _distance_sq(item.position, static_centroid))


def match_runtime_object_by_text(
    adapter: Any,
    state: SceneRuntimeState,
    text: str | None,
) -> RuntimeObjectState | None:
    """Fallback grounding for objects the static export does not know about."""
    if not isinstance(text, str) or not text.strip():
        return None
    ranked: list[tuple[float, RuntimeObjectState]] = []
    for runtime_object in state.objects.values():
        haystack = " ".join(
            token
            for token in (runtime_object.name, runtime_object.category)
            if isinstance(token, str) and token
        )
        score = adapter._score_text_match(haystack, text)
        if score > 0.0:
            ranked.append((score, runtime_object))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[0], item[1].name))
    return ranked[0][1]


def containing_room_id(adapter: Any, scene: HOVSGSceneAsset, position: dict[str, float] | None) -> str | None:
    if not isinstance(position, dict):
        return None
    room = adapter._containing_room(scene, position)
    return room.room_id if room is not None else None


def overlay_room_id(
    adapter: Any,
    scene: HOVSGSceneAsset,
    *,
    obj: HOVSGObjectAsset,
    overlay_position: dict[str, float] | None,
) -> str | None:
    """Re-localize an object's room when its overlay position drifted from the
    static centroid; returns None when the static room assignment still holds."""
    if overlay_position is None or obj.centroid is None:
        return None
    if _distance_sq(overlay_position, obj.centroid) < _OVERLAY_ROOM_REASSIGN_DISTANCE_M**2:
        return None
    room_id = containing_room_id(adapter, scene, overlay_position)
    if room_id is None or room_id == obj.room_id:
        return None
    return room_id


def _summary(state: SceneRuntimeState) -> dict[str, Any]:
    return {
        "signature": state.signature,
        "door_signature": state.door_signature(),
        "relation_signature": state.relation_signature,
        "step": state.step,
        "object_count": len(state.objects),
        "door_count": len(state.doors),
        "relation_count": len(state.relations),
        "closed_doors": sorted(
            name
            for name, door in state.doors.items()
            if door_is_navigation_passable(door) is False
        ),
        "navigation_passable_doors": sorted(
            name
            for name, door in state.doors.items()
            if door.navigation_passable is True
        ),
    }


def _store(adapter: Any) -> dict[str, SceneRuntimeState]:
    store = getattr(adapter, "_runtime_scene_states", None)
    if not isinstance(store, dict):
        store = {}
        adapter._runtime_scene_states = store
    return store


def _meaningful_tokens(value: Any) -> set[str]:
    if not isinstance(value, str):
        return set()
    tokens = re.sub(r"[^a-z0-9]+", " ", value.lower()).split()
    return {token for token in tokens if token and not token.isdigit()}


def _distance_sq(left: dict[str, float] | None, right: dict[str, float] | None) -> float:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return float("inf")
    total = 0.0
    for axis in ("x", "y", "z"):
        try:
            total += (float(left.get(axis, 0.0)) - float(right.get(axis, 0.0))) ** 2
        except (TypeError, ValueError):
            return float("inf")
    return total


__all__ = [
    "containing_room_id",
    "current_scene_state",
    "door_signature",
    "door_states",
    "ingest_scene_state",
    "ingest_scene_state_from_containers",
    "match_runtime_object",
    "match_runtime_object_by_text",
    "match_runtime_door",
    "overlay_room_id",
    "relation_signature",
    "resolve_object_centroid",
    "scene_state_signature",
]
