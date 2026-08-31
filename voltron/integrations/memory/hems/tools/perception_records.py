from __future__ import annotations

from typing import Any, Callable

from voltron.shared.models import PerceptionReport


def record_perception(
    *,
    report: PerceptionReport,
    kg_node_cls: Any,
    node_type_enum: Any,
    kg_edge_cls: Any,
    position_cls: Any,
    relation_type_enum: Any,
    resolve_node: Callable[[str | None, str | None], Any | None],
    resolve_node_id_by_name: Callable[[str], str | None],
    parse_relation_type: Callable[[str, Any], Any],
    new_node_id: Callable[[str], str],
    store_memory: Callable[[Any], None],
    update_node: Callable[[str, dict[str, Any]], None],
    get_edge: Callable[[str], Any | None],
    verify_edge: Callable[[str, bool], None],
    add_observation: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    stats = {
        "created_nodes": 0,
        "updated_nodes": 0,
        "created_edges": 0,
        "verified_edges": 0,
        "skipped_relations": 0,
    }
    resolved_ids: dict[str, str] = {}

    for obj in report.objects:
        node = resolve_node(obj.node_id, obj.name)
        position = None
        if obj.position and len(obj.position) >= 3:
            position = position_cls.from_tuple((obj.position[0], obj.position[1], obj.position[2]))

        if node is None:
            node_id = obj.node_id or new_node_id(obj.name)
            node = kg_node_cls(
                node_id=node_id,
                node_type=node_type_enum.OBJECT,
                name=obj.name,
                attributes=dict(obj.attributes),
                position=position,
                confidence=float(max(0.0, min(1.0, obj.confidence))),
            )
            store_memory(node)
            stats["created_nodes"] += 1
        else:
            updates: dict[str, Any] = {"attributes": dict(obj.attributes)}
            if position is not None:
                updates["position"] = position
            if obj.confidence:
                updates["confidence"] = float(max(0.0, min(1.0, obj.confidence)))
            update_node(node.node_id, updates)
            stats["updated_nodes"] += 1

        resolved_ids[obj.name] = node.node_id

    for rel in report.relations:
        src = resolved_ids.get(rel.source) or resolve_node_id_by_name(rel.source)
        tgt = resolved_ids.get(rel.target) or resolve_node_id_by_name(rel.target)

        if not src or not tgt:
            stats["skipped_relations"] += 1
            continue

        rel_type = parse_relation_type(rel.relation, relation_type_enum)
        edge_id = f"edge_{src}_{rel_type.value}_{tgt}"
        existing_edge = get_edge(edge_id)

        if existing_edge is None:
            edge = kg_edge_cls(
                edge_id=edge_id,
                source_id=src,
                target_id=tgt,
                relation_type=rel_type,
                confidence=float(max(0.0, min(1.0, rel.confidence))),
            )
            store_memory(edge)
            stats["created_edges"] += 1
        else:
            verify_edge(edge_id, True)
            stats["verified_edges"] += 1

    observation = {
        "source": "vlm",
        "object_count": len(report.objects),
        "objects": [
            {
                "id": resolved_ids.get(obj.name) or obj.node_id or obj.name,
                "name": obj.name,
                "confidence": float(max(0.0, min(1.0, obj.confidence))),
                "position": list(obj.position) if obj.position is not None else None,
            }
            for obj in report.objects
        ],
        "relations": len(report.relations),
        "task_complete": report.task_complete,
        "raw_text": report.raw_text,
    }
    for key in ("region_id", "room_id", "location_id"):
        value = report.metadata.get(key)
        if value not in (None, ""):
            observation[key] = value
    add_observation(observation)
    return stats


__all__ = ["record_perception"]
