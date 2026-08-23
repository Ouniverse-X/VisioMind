"""Query and semantic lookup helpers for the HEMS backend."""

from __future__ import annotations

from typing import Any, Callable


def resolve_node(*, semantic_memory: Any, node_id: str | None, name: str | None) -> Any | None:
    if node_id:
        node = semantic_memory.get_node(node_id)
        if node is not None:
            return node
    if name:
        matches = semantic_memory.retrieve_by_name(name, top_k=1)
        if matches:
            return matches[0]
    return None


def resolve_node_id_by_name(*, semantic_memory: Any, name: str) -> str | None:
    node = resolve_node(semantic_memory=semantic_memory, node_id=None, name=name)
    return getattr(node, "node_id", None)


def parse_relation_type(relation: str, relation_enum: Any) -> Any:
    key = relation.strip().upper()
    if key in relation_enum.__members__:
        return relation_enum[key]
    return relation_enum.NEAR


def query_semantic_region(
    *,
    semantic_memory: Any,
    name: str,
    attributes: dict[str, Any] | None,
    top_k: int,
    serializer: Callable[[Any], Any],
) -> dict[str, Any]:
    nodes = semantic_memory.retrieve_by_name(name, top_k=top_k)
    region_nodes = [node for node in nodes if _is_region_node(node)]
    if attributes:
        region_nodes = [node for node in region_nodes if _attributes_match(node, attributes)]
    region_nodes = region_nodes[:top_k]

    return {
        "query_type": "region",
        "query": {"name": name, "attributes": attributes},
        "results": [serializer(node) for node in region_nodes],
        "scores": [float(getattr(node, "confidence", 1.0)) for node in region_nodes],
        "explanation": f"Found {len(region_nodes)} region nodes matching '{name}'",
        "metadata": {"top_k": top_k},
    }


def _is_region_node(node: Any) -> bool:
    node_type = getattr(node, "node_type", None)
    node_type_value = getattr(node_type, "value", node_type)
    node_type_name = getattr(node_type, "name", node_type_value)
    normalized = str(node_type_name).lower()
    return normalized in {"region", "node_type.region"}


def _attributes_match(node: Any, attributes: dict[str, Any]) -> bool:
    node_attributes = getattr(node, "attributes", {}) or {}
    return all(node_attributes.get(key) == value for key, value in attributes.items())


__all__ = [
    "parse_relation_type",
    "query_semantic_region",
    "resolve_node",
    "resolve_node_id_by_name",
]
