"""Execution-write helpers for the HEMS backend."""

from __future__ import annotations

from typing import Any, Callable


_MONITOR_FEEDBACK_KEYS = (
    "step_count",
    "reward",
    "task_progress",
    "task_success",
    "current_room",
    "current_region",
    "room_id",
    "floor_id",
    "path_backend",
    "path_tracking_mode",
    "controller_mode",
    "goal_reached",
    "loop_detected",
    "oscillation_detected",
    "steps_since_progress",
    "best_distance_to_waypoint",
)

_VLM_HEARTBEAT_KEYS = (
    "available",
    "enabled",
    "source",
    "subtask_completed",
    "subtask_succeeded",
    "subtask_completion_reason",
    "request_in_flight",
    "last_result",
    "success_confirmation_count",
    "success_confirmation_threshold",
)


def record_navigation_update(
    *,
    payload: dict[str, Any],
    kg_node_cls: Any,
    node_type_enum: Any,
    position_cls: Any,
    resolve_node: Callable[[str | None, str | None], Any | None],
    new_node_id: Callable[[str, str], str],
    store_node: Callable[[Any], None],
    activate_region: Callable[[str], None],
    update_node: Callable[[str, dict[str, Any]], None],
    add_observation: Callable[[dict[str, Any]], None],
    maps: dict[str, dict[str, Any]],
    update_navigation_map: Callable[[dict[str, dict[str, Any]], str, dict[str, Any]], None],
) -> dict[str, Any]:
    stats = {"regions_updated": 0, "obstacles_updated": 0}

    region_name = str(payload.get("region", "")).strip()
    if region_name:
        node = resolve_node(None, region_name)
        if node is None:
            node = kg_node_cls(
                node_id=new_node_id(region_name, "region"),
                node_type=node_type_enum.REGION,
                name=region_name,
                attributes={"from": "vln"},
            )
            store_node(node)
        activate_region(node.node_id)
        stats["regions_updated"] += 1

    for obstacle in payload.get("obstacles", []):
        name = str(obstacle.get("name", obstacle.get("id", "obstacle")))
        node = resolve_node(None, name)
        position = obstacle.get("position")
        updates = {"attributes": {"obstacle": True, "from": "vln"}}
        if position and len(position) >= 3:
            updates["position"] = position_cls.from_tuple((position[0], position[1], position[2]))

        if node is None:
            new_node = kg_node_cls(
                node_id=new_node_id(name, "obj"),
                node_type=node_type_enum.OBJECT,
                name=name,
                attributes=updates["attributes"],
                position=updates.get("position"),
            )
            store_node(new_node)
        else:
            update_node(node.node_id, updates)

        stats["obstacles_updated"] += 1

    add_observation({"source": "vln", "payload": payload})

    scene_id = str(payload.get("scene_id", "")).strip()
    if scene_id:
        update_navigation_map(maps=maps, scene_id=scene_id, payload=payload)
    return stats


def record_action(
    *,
    payload: dict[str, Any],
    action_record_cls: Any,
    get_current_episode: Callable[[], Any | None],
    update_causal: Callable[..., None],
    add_observation: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    action_type = str(payload.get("action_type", ""))
    target = str(payload.get("target", ""))
    success = bool(payload.get("success", True))
    pre_state = dict(payload.get("pre_state", {}))
    post_state = dict(payload.get("post_state", {}))

    record = action_record_cls(
        action_type=action_type,
        target=target,
        parameters=dict(payload.get("parameters", {})),
        pre_state=pre_state,
        post_state=post_state,
        success=success,
        duration=float(payload.get("duration", 0.0)),
        failure_reason=payload.get("failure_reason"),
    )
    for key in ("control_step", "env_step", "task_success", "task_progress"):
        if key in payload:
            setattr(record, key, payload[key])

    episode = get_current_episode()
    if episode is None:
        return {"recorded": False, "warning": "no_active_episode"}

    episodic_record = bool(payload.get("episodic_record", False))
    if episodic_record:
        episode.action_sequence.append(record)
    if target and target not in episode.objects_involved:
        episode.objects_involved.append(target)

    state_changes = []
    action_parameters = dict(payload.get("parameters", {}))
    for key in sorted(set(pre_state.keys()) | set(post_state.keys())):
        old_value = pre_state.get(key)
        new_value = post_state.get(key)
        if old_value == new_value:
            continue
        state_changes.append({"attribute": key, "old": old_value, "new": new_value})
        update_causal(
            action=action_type,
            target=target,
            expected_effect=key,
            effect_occurred=True,
            effect_value=new_value,
            parameters=action_parameters,
            conditions=pre_state,
        )

    negative_evidence = []
    for expected in _normalize_expected_effects(payload.get("expected_effects")):
        if bool(expected.get("observed", expected.get("effect_occurred", False))):
            continue
        attribute = str(
            expected.get("attribute")
            or expected.get("effect_attribute")
            or expected.get("expected_effect")
            or ""
        ).strip()
        if not attribute:
            continue
        expected_value = expected.get("expected", expected.get("effect_value"))
        edge_id = update_causal(
            action=action_type,
            target=target,
            expected_effect=attribute,
            effect_occurred=False,
            effect_value=expected_value,
            parameters=dict(expected.get("parameters") or action_parameters),
            conditions=dict(expected.get("conditions") or pre_state),
        )
        negative_evidence.append(
            {
                "attribute": attribute,
                "expected": expected_value,
                "observed": False,
                "reason": expected.get("reason") or payload.get("failure_reason"),
                "edge_id": edge_id,
            }
        )

    observation = {
        "source": "vla",
        "action_type": action_type,
        "target": target,
        "success": success,
        "state_changes": state_changes,
    }
    if negative_evidence:
        observation["negative_evidence"] = negative_evidence
    add_observation(observation)

    result = {
        "recorded": True,
        "episodic_recorded": episodic_record,
        "action_id": record.action_id if episodic_record else None,
        "state_changes": state_changes,
    }
    if negative_evidence:
        result["negative_evidence"] = negative_evidence
    return result


def _normalize_expected_effects(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [dict(value)]
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    return []


def record_monitor_summary(
    *,
    payload: dict[str, Any],
    get_current_episode: Callable[[], Any | None],
    add_observation: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    episode = get_current_episode()
    if episode is None:
        return {"recorded": False, "warning": "no_active_episode"}

    summary_record = _compact_monitor_summary_payload(payload)
    monitor_summaries = getattr(episode, "monitor_summaries", None)
    if not isinstance(monitor_summaries, list):
        monitor_summaries = []
        setattr(episode, "monitor_summaries", monitor_summaries)
    monitor_summaries.append(summary_record)

    target = str(summary_record.get("target") or "").strip()
    if target:
        objects_involved = getattr(episode, "objects_involved", None)
        if isinstance(objects_involved, list) and target not in objects_involved:
            objects_involved.append(target)

    add_observation(
        {
            "source": "vision_monitor",
            "subtask_id": summary_record.get("subtask_id"),
            "control_step": summary_record.get("control_step"),
            "env_step": summary_record.get("env_step"),
            "summary": summary_record.get("summary"),
            "target": summary_record.get("target"),
            "result": summary_record.get("result"),
        }
    )

    return {"recorded": True, "summary_index": len(monitor_summaries) - 1}


def _compact_monitor_summary_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary = dict(payload)
    feedback = payload.get("environment_feedback")
    if isinstance(feedback, dict):
        compact_feedback = _compact_monitor_environment_feedback(feedback)
        if compact_feedback:
            summary["environment_feedback"] = compact_feedback
        else:
            summary.pop("environment_feedback", None)
    return summary


def _compact_monitor_environment_feedback(feedback: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in _MONITOR_FEEDBACK_KEYS:
        if key in feedback:
            compact[key] = _compact_scalar(feedback[key])

    heartbeat = feedback.get("environment_vlm_heartbeat")
    if isinstance(heartbeat, dict):
        heartbeat_summary = {
            key: _compact_scalar(heartbeat[key])
            for key in _VLM_HEARTBEAT_KEYS
            if key in heartbeat
        }
        if heartbeat_summary:
            compact["environment_vlm_heartbeat"] = heartbeat_summary

    return {key: value for key, value in compact.items() if value is not None}


def _compact_scalar(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()[:240]
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return None


__all__ = ["record_action", "record_monitor_summary", "record_navigation_update"]
