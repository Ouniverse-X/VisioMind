"""Scene-map ingestion helpers for semantic memory seeding."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable


def ingest_scene_map(
    *,
    scene_id: str,
    map_payload: dict[str, Any],
    metadata: dict[str, Any] | None,
    deps: dict[str, Any],
    store_memory: Callable[[Any], str],
    get_node: Callable[[str], Any | None] | None = None,
    update_node: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Seed HEMS semantic memory from a bounded scene-map payload."""
    metadata = dict(metadata or {})
    source = _metadata_value(metadata, map_payload, "source") or _source_from_payload(
        map_payload
    )
    confidence = _confidence(metadata, map_payload, source=source)
    export_hash = _metadata_value(metadata, map_payload, "export_hash")
    now = datetime.now().isoformat()
    stats = {
        "scene_id": scene_id,
        "source": source,
        "confidence": confidence,
        "regions": 0,
        "objects": 0,
        "edges": 0,
        "ingested_at": now,
    }
    if export_hash:
        stats["export_hash"] = export_hash

    region_node_ids = _ingest_regions(
        scene_id=scene_id,
        map_payload=map_payload,
        metadata=metadata,
        deps=deps,
        store_memory=store_memory,
        get_node=get_node,
        update_node=update_node,
        source=source,
        confidence=confidence,
        export_hash=export_hash,
        now=now,
    )
    stats["regions"] = len(region_node_ids["all"])

    object_node_ids, object_edge_count = _ingest_objects(
        scene_id=scene_id,
        map_payload=map_payload,
        metadata=metadata,
        deps=deps,
        store_memory=store_memory,
        get_node=get_node,
        update_node=update_node,
        source=source,
        confidence=confidence,
        export_hash=export_hash,
        now=now,
        region_node_ids=region_node_ids,
    )
    stats["objects"] = len(object_node_ids)
    stats["edges"] += object_edge_count
    stats["edges"] += _ingest_region_adjacency_edges(
        scene_id=scene_id,
        map_payload=map_payload,
        deps=deps,
        store_memory=store_memory,
        source=source,
        confidence=confidence,
        region_node_ids=region_node_ids,
    )
    return stats


def _ingest_regions(
    *,
    scene_id: str,
    map_payload: dict[str, Any],
    metadata: dict[str, Any],
    deps: dict[str, Any],
    store_memory: Callable[[Any], str],
    get_node: Callable[[str], Any | None] | None,
    update_node: Callable[[str, dict[str, Any]], None] | None,
    source: str,
    confidence: float,
    export_hash: str | None,
    now: str,
) -> dict[str, Any]:
    node_ids_by_key: dict[str, str] = {}
    all_node_ids: list[str] = []
    for region in _as_dicts(map_payload.get("regions")):
        name = _first_text(region, "name", "room", "room_name", "id")
        if not name:
            continue
        region_id = _first_text(region, "id", "region_id", "room_id") or name
        node_id = _node_id("region", scene_id, region_id)
        attributes = {
            "scene_id": scene_id,
            "region_id": region_id,
            "room_id": region.get("room_id") or region_id,
            "room_type": region.get("room_type"),
            "floor_id": region.get("floor_id"),
            "connected_regions": list(region.get("connected_regions") or []),
            "source": source,
            "provenance": "scene_map_seed",
            "graph_path": map_payload.get("graph_path"),
            "scene_map_source": map_payload.get("scene_map_source"),
            "updated_at": now,
        }
        if export_hash:
            attributes["export_hash"] = export_hash
        attributes.update(_prefixed_metadata(metadata))
        _upsert_node(
            node_id=node_id,
            name=name,
            node_type=deps["NodeType"].REGION,
            attributes=_compact_dict(attributes),
            position=_position(region.get("centroid") or region.get("position"), deps),
            confidence=confidence,
            deps=deps,
            store_memory=store_memory,
            get_node=get_node,
            update_node=update_node,
        )
        all_node_ids.append(node_id)
        for key in (region_id, name, region.get("room_id"), region.get("room_name")):
            if isinstance(key, str) and key.strip():
                node_ids_by_key[key.strip()] = node_id
    return {"by_key": node_ids_by_key, "all": all_node_ids}


def _ingest_objects(
    *,
    scene_id: str,
    map_payload: dict[str, Any],
    metadata: dict[str, Any],
    deps: dict[str, Any],
    store_memory: Callable[[Any], str],
    get_node: Callable[[str], Any | None] | None,
    update_node: Callable[[str, dict[str, Any]], None] | None,
    source: str,
    confidence: float,
    export_hash: str | None,
    now: str,
    region_node_ids: dict[str, Any],
) -> tuple[list[str], int]:
    node_ids: list[str] = []
    edge_count = 0
    for anchor in _as_dicts(map_payload.get("anchors")) + _as_dicts(
        map_payload.get("objects")
    ):
        name = _first_text(anchor, "name", "object", "category", "id")
        if not name:
            continue
        object_id = _first_text(anchor, "id", "object_id") or name
        room_key = _first_text(anchor, "room", "room_name", "room_id")
        node_id = _node_id("obj", scene_id, object_id)
        attributes = {
            "scene_id": scene_id,
            "object_id": object_id,
            "category": anchor.get("category"),
            "room": anchor.get("room") or anchor.get("room_name"),
            "room_id": anchor.get("room_id"),
            "floor_id": anchor.get("floor_id"),
            "source": source,
            "provenance": "scene_map_seed",
            "graph_path": map_payload.get("graph_path"),
            "scene_map_source": map_payload.get("scene_map_source"),
            "updated_at": now,
        }
        if export_hash:
            attributes["export_hash"] = export_hash
        attributes.update(_prefixed_metadata(metadata))
        _upsert_node(
            node_id=node_id,
            name=name,
            node_type=deps["NodeType"].OBJECT,
            attributes=_compact_dict(attributes),
            position=_position(anchor.get("position") or anchor.get("centroid"), deps),
            confidence=confidence,
            deps=deps,
            store_memory=store_memory,
            get_node=get_node,
            update_node=update_node,
        )
        node_ids.append(node_id)
        region_node_id = region_node_ids["by_key"].get(room_key or "")
        if region_node_id:
            _store_edge(
                deps=deps,
                store_memory=store_memory,
                edge_id=f"edge_{node_id}_inside_{region_node_id}",
                source_id=node_id,
                target_id=region_node_id,
                relation_type=deps["RelationType"].INSIDE,
                attributes={
                    "scene_id": scene_id,
                    "source": source,
                    "provenance": "scene_map_seed",
                },
                confidence=confidence,
            )
            edge_count += 1
    return node_ids, edge_count


def _ingest_region_adjacency_edges(
    *,
    scene_id: str,
    map_payload: dict[str, Any],
    deps: dict[str, Any],
    store_memory: Callable[[Any], str],
    source: str,
    confidence: float,
    region_node_ids: dict[str, Any],
) -> int:
    count = 0
    for region in _as_dicts(map_payload.get("regions")):
        source_key = _first_text(region, "id", "region_id", "room_id", "name")
        source_id = region_node_ids["by_key"].get(source_key or "")
        if not source_id:
            continue
        for target_key in region.get("connected_regions") or []:
            target_id = region_node_ids["by_key"].get(str(target_key))
            if not target_id:
                continue
            _store_edge(
                deps=deps,
                store_memory=store_memory,
                edge_id=f"edge_{source_id}_near_{target_id}",
                source_id=source_id,
                target_id=target_id,
                relation_type=deps["RelationType"].NEAR,
                attributes={
                    "scene_id": scene_id,
                    "source": source,
                    "provenance": "scene_map_seed",
                },
                confidence=confidence,
            )
            count += 1
    return count


def _upsert_node(
    *,
    node_id: str,
    name: str,
    node_type: Any,
    attributes: dict[str, Any],
    position: Any,
    confidence: float,
    deps: dict[str, Any],
    store_memory: Callable[[Any], str],
    get_node: Callable[[str], Any | None] | None,
    update_node: Callable[[str, dict[str, Any]], None] | None,
) -> None:
    if callable(get_node) and callable(update_node) and get_node(node_id) is not None:
        update_node(
            node_id,
            {
                "name": name,
                "position": position,
                "attributes": attributes,
                "confidence": confidence,
            },
        )
        return
    store_memory(
        deps["KGNode"](
            node_id=node_id,
            node_type=node_type,
            name=name,
            attributes=attributes,
            position=position,
            confidence=confidence,
        )
    )


def _store_edge(
    *,
    deps: dict[str, Any],
    store_memory: Callable[[Any], str],
    edge_id: str,
    source_id: str,
    target_id: str,
    relation_type: Any,
    attributes: dict[str, Any],
    confidence: float,
) -> None:
    store_memory(
        deps["KGEdge"](
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            attributes=attributes,
            confidence=confidence,
        )
    )


def _position(value: Any, deps: dict[str, Any]) -> Any | None:
    if not isinstance(value, dict):
        return None
    try:
        return deps["Position"].from_tuple(
            (
                float(value.get("x", 0.0)),
                float(value.get("y", 0.0)),
                float(value.get("z", 0.0)),
            )
        )
    except (TypeError, ValueError):
        return None


def _confidence(metadata: dict[str, Any], map_payload: dict[str, Any], *, source: str) -> float:
    for candidate in (metadata.get("confidence"), map_payload.get("confidence")):
        try:
            return max(0.0, min(1.0, float(candidate)))
        except (TypeError, ValueError):
            continue
    scene_map_source = str(map_payload.get("scene_map_source") or "").lower()
    normalized_source = source.lower()
    if scene_map_source == "gt" or "gt" in normalized_source:
        return 0.95
    if "hovsg" in normalized_source:
        return 0.85
    return 0.8


def _source_from_payload(map_payload: dict[str, Any]) -> str:
    for key in ("scene_map_source", "source"):
        value = map_payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "scene_map_seed"


def _metadata_value(
    metadata: dict[str, Any],
    map_payload: dict[str, Any],
    key: str,
) -> str | None:
    for source in (metadata, map_payload):
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _prefixed_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    copied = {}
    for key in ("run_id", "source_episode_id", "source_action_id"):
        if metadata.get(key) is not None:
            copied[key] = metadata[key]
    return copied


def _as_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _first_text(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _node_id(prefix: str, scene_id: str, key: str) -> str:
    return f"{prefix}_{_slug(scene_id)}_{_slug(key)}"


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    slug = "_".join(part for part in text.replace("/", "_").split() if part)
    slug = "".join(char if char.isalnum() or char == "_" else "_" for char in slug)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "unknown"


def _compact_dict(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


__all__ = ["ingest_scene_map"]
