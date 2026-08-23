"""Object-approach memory helpers for the HEMS backend."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from . import spatial_memory


def get_object_approach_history(
    *,
    maps: dict[str, dict[str, Any]],
    scene_id: str,
    target: dict[str, Any],
    top_k: int,
    get_task_context: Callable[[], dict[str, Any]],
    target_key_builder: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    normalized_scene_id = str(scene_id or "").strip()
    target_key = target_key_builder(target)
    if not normalized_scene_id or not target_key:
        return {
            "scene_id": normalized_scene_id or None,
            "target_key": target_key or None,
            "entries": [],
        }

    entry = maps.get(normalized_scene_id)
    stored_entries: list[dict[str, Any]] = []
    if entry is not None:
        object_memory = entry.get("map_payload", {}).get("object_approach_memory", {})
        bucket = object_memory.get(target_key, {})
        raw_entries = bucket.get("entries", [])
        if isinstance(raw_entries, list):
            stored_entries = [deepcopy(item) for item in raw_entries if isinstance(item, dict)]

    if not stored_entries:
        task_context = get_task_context()
        runtime_history = task_context.get("object_approach_history", {})
        if (
            isinstance(runtime_history, dict)
            and str(runtime_history.get("scene_id") or "").strip() == normalized_scene_id
            and str(runtime_history.get("target_key") or "").strip() == target_key
        ):
            raw_entries = runtime_history.get("entries", [])
            if isinstance(raw_entries, list):
                stored_entries = [deepcopy(item) for item in raw_entries if isinstance(item, dict)]

    if top_k > 0:
        stored_entries = stored_entries[-int(top_k) :]
    return {
        "scene_id": normalized_scene_id,
        "target_key": target_key,
        "entries": stored_entries,
    }


def record_object_approach_outcome(
    *,
    maps: dict[str, dict[str, Any]],
    scene_id: str,
    target: dict[str, Any],
    candidate: dict[str, Any],
    outcome: str,
    reason: str | None,
    metadata: dict[str, Any] | None,
    now_string: str | None,
    ensure_map: Callable[[dict[str, dict[str, Any]], str], dict[str, Any]],
    merge_dicts: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
    target_key_builder: Callable[[dict[str, Any]], str],
    candidate_signature_builder: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    normalized_scene_id = str(scene_id or "").strip()
    target_key = target_key_builder(target)
    if not normalized_scene_id:
        return {"status": "skipped", "reason": "scene_id_missing"}
    if not target_key:
        return {"status": "skipped", "reason": "target_key_missing", "scene_id": normalized_scene_id}

    scene_entry = ensure_map(maps, normalized_scene_id)
    object_memory = scene_entry["map_payload"].setdefault("object_approach_memory", {})
    target_bucket = object_memory.setdefault(
        target_key,
        {
            "target": deepcopy(target),
            "entries": [],
        },
    )
    target_bucket["target"] = merge_dicts(dict(target_bucket.get("target", {})), deepcopy(target))
    entry_metadata = spatial_memory.build_provenance(
        metadata,
        now_string=now_string,
        default_source_agent="NAVIGATION",
    )

    entry = {
        "timestamp": now_string,
        "outcome": str(outcome or "").strip().lower() or "unknown",
        "reason": reason,
        "candidate": deepcopy(candidate),
        "candidate_signature": candidate_signature_builder(candidate),
        "metadata": entry_metadata,
    }
    target_bucket.setdefault("entries", []).append(entry)
    return {
        "status": "recorded",
        "scene_id": normalized_scene_id,
        "target_key": target_key,
        "entry": deepcopy(entry),
    }


__all__ = ["get_object_approach_history", "record_object_approach_outcome"]
