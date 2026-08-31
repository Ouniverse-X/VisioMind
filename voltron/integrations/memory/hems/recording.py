from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from .tools import execution_records, object_approach, perception_records, spatial_memory


def load_map(*, maps: dict[str, dict[str, Any]], scene_id: str) -> dict[str, Any]:
    entry = maps.get(scene_id)
    if entry is None:
        return {
            "scene_id": scene_id,
            "status": "missing",
            "map_payload": None,
            "metadata": {},
        }
    return spatial_memory.clone_map_entry(scene_id, entry, status="loaded")


def save_map(
    *,
    maps: dict[str, dict[str, Any]],
    scene_id: str,
    map_payload: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    persist_state: Callable[[], None],
    clone_map_entry: Callable[[str, dict[str, Any], str], dict[str, Any]],
) -> dict[str, Any]:
    provenance = spatial_memory.build_provenance(metadata)
    payload = deepcopy(map_payload)
    existing_payload = maps.get(scene_id, {}).get("map_payload", {})
    if isinstance(existing_payload, dict) and "object_approach_memory" not in payload:
        object_approach_memory = existing_payload.get("object_approach_memory")
        if object_approach_memory is not None:
            payload["object_approach_memory"] = deepcopy(object_approach_memory)
    entry = {
        "map_payload": payload,
        "metadata": deepcopy(metadata or {}),
    }
    spatial_memory.record_map_metadata_update(
        entry,
        key="last_update",
        provenance=provenance,
    )
    maps[scene_id] = entry
    persist_state()
    return clone_map_entry(scene_id, entry, status="saved")


def update_map(
    *,
    maps: dict[str, dict[str, Any]],
    scene_id: str,
    delta: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    persist_state: Callable[[], None],
    ensure_map: Callable[[dict[str, dict[str, Any]], str], dict[str, Any]],
    merge_dicts: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
    clone_map_entry: Callable[[str, dict[str, Any], str], dict[str, Any]],
) -> dict[str, Any]:
    entry = ensure_map(maps, scene_id)
    entry["map_payload"] = merge_dicts(entry["map_payload"], deepcopy(delta))
    provenance = spatial_memory.build_provenance(delta, metadata)
    spatial_memory.record_map_metadata_update(
        entry,
        key="last_update",
        provenance=provenance,
    )
    persist_state()
    return clone_map_entry(scene_id, entry, status="updated")


def mark_explored(
    *,
    maps: dict[str, dict[str, Any]],
    scene_id: str,
    evidence: dict[str, Any],
    persist_state: Callable[[], None],
    ensure_map: Callable[[dict[str, dict[str, Any]], str], dict[str, Any]],
) -> dict[str, Any]:
    entry = ensure_map(maps, scene_id)
    provenance = spatial_memory.build_provenance(evidence, default_source_agent="MEMORY")
    exploration = entry["map_payload"].setdefault("exploration", {})
    evidence_log = exploration.setdefault("evidence", [])
    evidence_payload = deepcopy(evidence)
    evidence_payload.setdefault("provenance", deepcopy(provenance))
    evidence_log.append(evidence_payload)
    spatial_memory.record_map_metadata_update(
        entry,
        key="last_exploration_update",
        provenance=provenance,
    )

    region_name = str(evidence.get("region", "")).strip()
    explored_regions = exploration.setdefault("explored_regions", [])
    if region_name and region_name not in explored_regions:
        explored_regions.append(region_name)
    persist_state()

    return {
        "scene_id": scene_id,
        "status": "marked",
        "explored_regions": list(explored_regions),
        "evidence_count": len(evidence_log),
    }


def get_exploration_frontiers(
    *, maps: dict[str, dict[str, Any]], scene_id: str
) -> list[dict[str, Any]]:
    entry = maps.get(scene_id)
    if entry is None:
        return []

    map_payload = entry["map_payload"]
    frontiers = map_payload.get("exploration", {}).get("frontiers")
    if frontiers is None:
        frontiers = map_payload.get("frontiers", [])
    return [deepcopy(frontier) for frontier in frontiers]


def get_object_approach_history(
    *,
    maps: dict[str, dict[str, Any]],
    scene_id: str,
    target: dict[str, Any],
    top_k: int,
    get_task_context: Callable[[], dict[str, Any]],
    target_key_builder: Callable[[dict[str, Any]], str],
) -> dict[str, Any]:
    return object_approach.get_object_approach_history(
        maps=maps,
        scene_id=scene_id,
        target=target,
        top_k=top_k,
        get_task_context=get_task_context,
        target_key_builder=target_key_builder,
    )


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
    persist_state: Callable[[], None],
    ensure_map: Callable[[dict[str, dict[str, Any]], str], dict[str, Any]],
    merge_dicts: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
    target_key_builder: Callable[[dict[str, Any]], str],
    candidate_signature_builder: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    recorded = object_approach.record_object_approach_outcome(
        maps=maps,
        scene_id=scene_id,
        target=target,
        candidate=candidate,
        outcome=outcome,
        reason=reason,
        metadata=metadata,
        now_string=now_string,
        ensure_map=ensure_map,
        merge_dicts=merge_dicts,
        target_key_builder=target_key_builder,
        candidate_signature_builder=candidate_signature_builder,
    )
    entry = maps.get(scene_id)
    if entry is not None and isinstance(recorded.get("entry"), dict):
        entry_metadata = recorded["entry"].get("metadata", {})
        if isinstance(entry_metadata, dict):
            spatial_memory.record_map_metadata_update(
                entry,
                key="last_object_approach_update",
                provenance=entry_metadata,
            )
    persist_state()
    return recorded


def record_perception(
    *,
    report: Any,
    deps: dict[str, Any],
    resolve_node: Callable[[str | None, str | None], Any | None],
    resolve_node_id_by_name: Callable[[str], str | None],
    parse_relation_type: Callable[[str, Any], Any],
    new_node_id: Callable[[str], str],
    store_memory: Callable[[Any], None],
    update_node: Callable[[str, dict[str, Any]], None],
    get_edge: Callable[[str], Any],
    verify_edge: Callable[[str, bool], None],
    add_observation: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    return perception_records.record_perception(
        report=report,
        kg_node_cls=deps["KGNode"],
        node_type_enum=deps["NodeType"],
        kg_edge_cls=deps["KGEdge"],
        position_cls=deps["Position"],
        relation_type_enum=deps["RelationType"],
        resolve_node=resolve_node,
        resolve_node_id_by_name=resolve_node_id_by_name,
        parse_relation_type=parse_relation_type,
        new_node_id=new_node_id,
        store_memory=store_memory,
        update_node=update_node,
        get_edge=get_edge,
        verify_edge=verify_edge,
        add_observation=add_observation,
    )


def record_navigation_update(
    *,
    payload: dict[str, Any],
    deps: dict[str, Any],
    resolve_node: Callable[[str | None, str | None], Any | None],
    new_node_id: Callable[[str, str], str],
    store_node: Callable[[Any], None],
    activate_region: Callable[[str], None],
    update_node: Callable[[str, dict[str, Any]], None],
    add_observation: Callable[[dict[str, Any]], None],
    maps: dict[str, dict[str, Any]],
    update_navigation_map: Callable[[dict[str, dict[str, Any]], str, dict[str, Any]], None],
    persist_state: Callable[[], None],
) -> dict[str, Any]:
    stats = execution_records.record_navigation_update(
        payload=payload,
        kg_node_cls=deps["KGNode"],
        node_type_enum=deps["NodeType"],
        position_cls=deps["Position"],
        resolve_node=resolve_node,
        new_node_id=new_node_id,
        store_node=store_node,
        activate_region=activate_region,
        update_node=update_node,
        add_observation=add_observation,
        maps=maps,
        update_navigation_map=update_navigation_map,
    )

    scene_id = str(payload.get("scene_id", "")).strip()
    if scene_id:
        persist_state()
    return stats


def record_action(
    *,
    payload: dict[str, Any],
    action_record_cls: Any,
    get_current_episode: Callable[[], Any],
    update_causal: Callable[..., None],
    add_observation: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    return execution_records.record_action(
        payload=payload,
        action_record_cls=action_record_cls,
        get_current_episode=get_current_episode,
        update_causal=update_causal,
        add_observation=add_observation,
    )


def record_monitor_summary(
    *,
    payload: dict[str, Any],
    get_current_episode: Callable[[], Any | None],
    add_observation: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    return execution_records.record_monitor_summary(
        payload=payload,
        get_current_episode=get_current_episode,
        add_observation=add_observation,
    )
