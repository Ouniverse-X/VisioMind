from __future__ import annotations

from typing import Any


def build_reflection_evidence(
    episode_context: dict[str, Any],
    *,
    similar_episodes: Any = None,
) -> dict[str, Any]:
    episode_id = str(episode_context.get("episode_id") or "")
    actions = _action_records(episode_context)
    state_deltas = _state_deltas(episode_context, actions=actions)
    success_factors = _success_factors(actions)
    failure_factors = _failure_factors(actions)
    critical_actions = _critical_actions(episode_context, actions)
    rule_derived_evidence = _rule_derived_evidence(
        episode_context,
        actions=actions,
        critical_actions=critical_actions,
    )
    evidence = {
        "status": "completed" if _has_completed_episode(episode_context) else "skipped",
        "episode_id": episode_id,
        "task_description": str(episode_context.get("task_description") or ""),
        "task_type": _optional_str(episode_context.get("task_type")),
        "outcome": _optional_str(episode_context.get("outcome")),
        "failure_reason": episode_context.get("failure_reason"),
        "success_factors": success_factors,
        "failure_factors": failure_factors,
        "critical_actions": critical_actions,
        "rule_derived_evidence": rule_derived_evidence,
        "state_deltas": state_deltas,
        "causal_observations": _causal_observations(episode_context, state_deltas=state_deltas),
        "similar_episode_contrasts": _similar_episode_contrasts(
            similar_episodes, current_episode_id=episode_id
        ),
        "recent_observation_count": len(_list(episode_context.get("recent_observations"))),
        "source_integrity": dict(episode_context.get("source_integrity", {}))
        if isinstance(episode_context.get("source_integrity"), dict)
        else {},
        "generated_by": "memory_agent_native_reflection",
    }
    return {key: value for key, value in evidence.items() if value not in (None, [], {})}


def reflection_annotation(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "annotation_type": "reflection_evidence",
        "generated_by": "memory_agent",
        "source_episode_id": str(evidence.get("episode_id") or ""),
        "reflection_evidence": dict(evidence),
    }


def _has_completed_episode(episode_context: dict[str, Any]) -> bool:
    source_integrity = episode_context.get("source_integrity")
    if (
        isinstance(source_integrity, dict)
        and source_integrity.get("from_completed_episode") is False
    ):
        return False
    return bool(episode_context.get("episode_id"))


def _action_records(episode_context: dict[str, Any]) -> list[dict[str, Any]]:
    actions = _list(episode_context.get("actions"))
    if not actions:
        actions = _list(episode_context.get("action_sequence"))
    return [dict(action) for action in actions if isinstance(action, dict)]


def _success_factors(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    factors = []
    for index, action in enumerate(actions):
        if _action_success(action) is not True:
            continue
        factors.append(_action_summary(action, index=index))
    return factors


def _failure_factors(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    factors = []
    for index, action in enumerate(actions):
        if _action_success(action) is not False:
            continue
        summary = _action_summary(action, index=index)
        summary["failure_reason"] = _failure_reason(action)
        factors.append(summary)
    return factors


def _critical_actions(
    episode_context: dict[str, Any], actions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    critical = _failure_factors(actions)
    failure_idx = episode_context.get("failure_action_idx")
    if isinstance(failure_idx, int) and 0 <= failure_idx < len(actions):
        summary = _action_summary(actions[failure_idx], index=failure_idx)
        summary["failure_reason"] = _failure_reason(actions[failure_idx])
        if not any(item.get("action_id") == summary.get("action_id") for item in critical):
            critical.append(summary)
    return critical


def _state_deltas(
    episode_context: dict[str, Any],
    *,
    actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    transitions = _list(episode_context.get("state_transitions"))
    deltas = [_transition_delta(item) for item in transitions if isinstance(item, dict)]
    deltas = [item for item in deltas if item]
    if deltas:
        return deltas

    derived = []
    for index, action in enumerate(actions):
        action_id = _action_id(action, index=index)
        pre = action.get("pre_state")
        post = action.get("post_state")
        if not isinstance(pre, dict) or not isinstance(post, dict):
            continue
        for key in sorted(set(pre) | set(post)):
            old_value = pre.get(key)
            new_value = post.get(key)
            if old_value == new_value:
                continue
            derived.append(
                {
                    "attribute": str(key),
                    "old_value": old_value,
                    "new_value": new_value,
                    "source_action_id": action_id,
                }
            )
    return derived


def _transition_delta(value: dict[str, Any]) -> dict[str, Any]:
    attribute = value.get("attribute") or value.get("key") or value.get("field")
    old_value = _first_present(value, "old_value", "before", "from")
    new_value = _first_present(value, "new_value", "after", "to", "value")
    source_action_id = (
        value.get("source_action_id")
        or value.get("caused_by")
        or value.get("action_id")
        or value.get("cause_action_id")
    )
    if attribute is None and source_action_id is None:
        return {}
    delta = {
        "attribute": str(attribute) if attribute is not None else "",
        "old_value": old_value,
        "new_value": new_value,
    }
    if source_action_id:
        delta["source_action_id"] = str(source_action_id)
    return delta


def _causal_observations(
    episode_context: dict[str, Any],
    *,
    state_deltas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    annotations = [
        dict(item)
        for item in _list(episode_context.get("causal_annotations"))
        if isinstance(item, dict)
    ]
    if annotations:
        return annotations

    observations = []
    for delta in state_deltas:
        source_action_id = delta.get("source_action_id")
        if not source_action_id:
            continue
        observations.append(
            {
                "source_action_id": source_action_id,
                "effect_attribute": delta.get("attribute"),
                "effect_value": delta.get("new_value"),
            }
        )
    return observations


def _rule_derived_evidence(
    episode_context: dict[str, Any],
    *,
    actions: list[dict[str, Any]],
    critical_actions: list[dict[str, Any]],
) -> dict[str, Any]:
    critical_failure_action = critical_actions[0] if critical_actions else {}
    failure_classification = _failure_classification(episode_context, actions=actions)
    return _drop_empty(
        {
            "critical_failure_action": critical_failure_action,
            "failure_classification": failure_classification,
            "object_approach_selection": _object_approach_selection(
                episode_context, actions=actions
            ),
            "visual_state": _visual_state(episode_context),
            "action_backend_status": _action_backend_status(episode_context, actions=actions),
            "action_interaction_context": _action_interaction_context(
                episode_context, actions=actions
            ),
        }
    )


def _failure_classification(
    episode_context: dict[str, Any],
    *,
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    source_integrity = episode_context.get("source_integrity")
    source_integrity = source_integrity if isinstance(source_integrity, dict) else {}
    reason_text = _normalize_text(
        " ".join(
            str(value)
            for value in (
                episode_context.get("failure_reason"),
                source_integrity.get("failure_reason"),
                source_integrity.get("termination_reason"),
            )
            if value is not None
        )
    )
    action_failure = any(_action_success(action) is False for action in actions)
    environment_truncated = bool(
        source_integrity.get("environment_truncated")
        or source_integrity.get("truncated")
        or episode_context.get("environment_truncated")
        or "truncat" in reason_text
        or "max step" in reason_text
        or "time limit" in reason_text
    )
    vla_backend_failure = _is_vla_backend_failure(reason_text) or any(
        _is_vla_backend_failure(_normalize_text(_failure_reason(action))) for action in actions
    )
    simulator_task_failure = bool(
        _normalize_text(episode_context.get("outcome")) in {"failure", "failed"}
        and not environment_truncated
        and not vla_backend_failure
    )
    kind = (
        "environment_truncation"
        if environment_truncated
        else "action_failure"
        if action_failure
        else "unknown"
    )
    if vla_backend_failure and not environment_truncated:
        kind = "vla_backend_failure"
    elif simulator_task_failure:
        kind = "simulator_task_failure"
    return {
        "kind": kind,
        "action_failure": action_failure,
        "vla_backend_failure": vla_backend_failure,
        "simulator_task_failure": simulator_task_failure,
    }


def _object_approach_selection(
    episode_context: dict[str, Any],
    *,
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate = _first_mapping(
        episode_context.get("selected_object_approach"),
        episode_context.get("object_approach_selection"),
    )
    source_action_id = None
    if not candidate:
        for index, action in enumerate(actions):
            candidate = _first_mapping(
                action.get("selected_object_approach"),
                action.get("object_approach_selection"),
                _nested_mapping(action, ("parameters", "selected_object_approach")),
                _nested_mapping(action, ("parameters", "object_approach_selection")),
                _nested_mapping(action, ("result", "selected_object_approach")),
                _nested_mapping(action, ("result", "object_approach_selection")),
            )
            if candidate:
                source_action_id = _action_id(action, index=index)
                break
    if not candidate:
        for observation in _list(episode_context.get("recent_observations")):
            if not isinstance(observation, dict):
                continue
            candidate = _first_mapping(
                observation.get("selected_object_approach"),
                observation.get("object_approach_selection"),
            )
            if candidate:
                source_action_id = observation.get("action_id") or observation.get(
                    "source_action_id"
                )
                break
    if not candidate:
        return {}
    history_penalty = _drop_empty(
        {
            "blocked_by_history": candidate.get("blocked_by_history"),
            "history_penalty": candidate.get("history_penalty") or candidate.get("memory_penalty"),
        }
    )
    return _drop_empty(
        {
            "source_action_id": source_action_id,
            "candidate": dict(candidate),
            "history_penalty": history_penalty,
        }
    )


def _visual_state(episode_context: dict[str, Any]) -> dict[str, Any]:
    visual = _visual_source(episode_context)
    return _drop_empty(
        {
            "target_visible": visual.get("target_visible") if visual else None,
            "target_part_visible": visual.get("target_part_visible") if visual else None,
            "task_complete": visual.get("task_complete") if visual else None,
        }
    )


def _action_interaction_context(
    episode_context: dict[str, Any],
    *,
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    approach = _object_approach_selection(episode_context, actions=actions)
    candidate = approach.get("candidate") if isinstance(approach.get("candidate"), dict) else {}
    visual = _visual_source(episode_context)
    final_state = episode_context.get("final_state")
    final_state = final_state if isinstance(final_state, dict) else {}
    working_summary = final_state.get("working_memory_summary")
    working_summary = working_summary if isinstance(working_summary, dict) else {}
    latest_feedback = working_summary.get("latest_environment_feedback")
    latest_feedback = latest_feedback if isinstance(latest_feedback, dict) else {}
    monitor_feedback = _latest_monitor_environment_feedback(episode_context)
    environment = final_state.get("environment")
    environment = environment if isinstance(environment, dict) else {}
    failure_action = _first_failed_action(actions)
    source_integrity = episode_context.get("source_integrity")
    source_integrity = source_integrity if isinstance(source_integrity, dict) else {}
    terminal_action = _terminal_action(actions)
    terminal_feedback = _action_env_feedback(terminal_action)
    terminal_parameters = terminal_action.get("parameters")
    terminal_parameters = terminal_parameters if isinstance(terminal_parameters, dict) else {}
    heartbeat = _first_mapping(
        final_state.get("environment_vlm_heartbeat"),
        episode_context.get("environment_vlm_heartbeat"),
        environment.get("environment_vlm_heartbeat"),
        latest_feedback.get("environment_vlm_heartbeat"),
        monitor_feedback.get("environment_vlm_heartbeat"),
        _heartbeat_from_actions(actions),
    )
    failure_feedback = _action_env_feedback(failure_action)
    failure_parameters = failure_action.get("parameters")
    failure_parameters = failure_parameters if isinstance(failure_parameters, dict) else {}

    environment_truncated = bool(
        source_integrity.get("environment_truncated")
        or source_integrity.get("truncated")
        or environment.get("truncated")
        or terminal_feedback.get("truncated")
        or terminal_parameters.get("truncated")
        or latest_feedback.get("truncated")
        or monitor_feedback.get("truncated")
        or failure_feedback.get("truncated")
        or episode_context.get("environment_truncated")
        or "truncat" in _normalize_text(episode_context.get("failure_reason"))
    )
    task_success = _first_present_in_mappings(
        "task_success",
        final_state,
        environment,
        terminal_parameters,
        terminal_feedback,
        terminal_action,
        latest_feedback,
        monitor_feedback,
        failure_feedback,
        failure_parameters,
        failure_action,
        episode_context,
        source_integrity,
    )
    if task_success is None and _normalize_text(episode_context.get("outcome")) in {
        "failure",
        "failed",
    }:
        task_success = False
    task_progress = _first_present_in_mappings(
        "task_progress",
        final_state,
        environment,
        terminal_parameters,
        terminal_feedback,
        terminal_action,
        latest_feedback,
        monitor_feedback,
        failure_feedback,
        failure_parameters,
        failure_action,
        episode_context,
        source_integrity,
    )

    return _drop_empty(
        {
            "selected_candidate": _selected_candidate_context(candidate),
            "distance_context": _drop_empty(
                {
                    "approach_distance_m": candidate.get("approach_distance_m"),
                    "handoff_distance_m": candidate.get("handoff_distance_m"),
                    "estimated_visual_distance_m": visual.get("estimated_distance_m")
                    if visual
                    else None,
                }
            ),
            "visual_affordance": _drop_empty(
                {
                    "target_visible": visual.get("target_visible") if visual else None,
                    "target_part_visible": visual.get("target_part_visible") if visual else None,
                    "target_part_name": visual.get("target_part_name") if visual else None,
                    "switch_visible": visual.get("switch_visible") if visual else None,
                    "view_quality": visual.get("view_quality") if visual else None,
                }
            ),
            "contact_context": _contact_context(visual=visual, actions=actions),
            "environment_outcome": _drop_empty(
                {
                    "task_success": task_success,
                    "task_progress": task_progress,
                    "environment_truncated": environment_truncated,
                    "env_step_count": environment.get("step_count")
                    or final_state.get("step_count")
                    or terminal_parameters.get("env_step")
                    or terminal_feedback.get("step_count")
                    or terminal_action.get("env_step")
                    or latest_feedback.get("step_count")
                    or failure_feedback.get("step_count")
                    or failure_parameters.get("env_step")
                    or failure_action.get("env_step")
                    or monitor_feedback.get("step_count")
                    or episode_context.get("env_step_count"),
                    "control_step": terminal_parameters.get("control_step")
                    or terminal_action.get("control_step")
                    or terminal_feedback.get("control_step")
                    or failure_action.get("control_step")
                    or failure_feedback.get("control_step")
                    or failure_parameters.get("control_step"),
                    "goal_status": environment.get("goal_status")
                    or final_state.get("goal_status")
                    or terminal_parameters.get("goal_status")
                    or terminal_feedback.get("goal_status")
                    or latest_feedback.get("goal_status")
                    or monitor_feedback.get("goal_status")
                    or failure_feedback.get("goal_status")
                    or episode_context.get("goal_status"),
                }
            ),
            "vlm_predicate_mismatch": _vlm_predicate_mismatch(
                heartbeat=heartbeat,
                environment_task_success=task_success,
            ),
        }
    )


def _visual_source(episode_context: dict[str, Any]) -> dict[str, Any]:
    visual = _first_mapping(
        episode_context.get("last_scene_report"),
        episode_context.get("scene_report"),
    )
    if not visual:
        for action in reversed(_action_records(episode_context)):
            candidate = _first_mapping(
                _nested_mapping(action, ("result", "scene_report")),
                _nested_mapping(action, ("parameters", "scene_report")),
                _nested_mapping(action, ("scene_report",)),
                _nested_mapping(action, ("result",)),
            )
            if _looks_like_visual_report(candidate):
                visual = candidate
                break
    if not visual:
        for observation in reversed(_list(episode_context.get("recent_observations"))):
            if isinstance(observation, dict) and any(
                key in observation
                for key in (
                    "target_visible",
                    "target_part_visible",
                    "target_part_name",
                    "task_complete",
                    "switch_visible",
                    "estimated_distance_m",
                    "contact_observed",
                )
            ):
                visual = observation
                break
    return visual


def _selected_candidate_context(candidate: dict[str, Any]) -> dict[str, Any]:
    if not candidate:
        return {}
    return _drop_empty(
        {
            "candidate_id": candidate.get("candidate_id"),
            "target": _drop_empty(
                {
                    "object_id": candidate.get("object_id"),
                    "object_name": candidate.get("object_name"),
                    "room_id": candidate.get("room_id"),
                    "floor_id": candidate.get("floor_id"),
                }
            ),
            "candidate_signature": _candidate_signature(candidate),
            "history_penalty": candidate.get("history_penalty") or candidate.get("memory_penalty"),
            "blocked_by_history": candidate.get("blocked_by_history"),
        }
    )


def _candidate_signature(candidate: dict[str, Any]) -> dict[str, Any]:
    explicit = candidate.get("candidate_signature")
    if isinstance(explicit, dict) and explicit:
        return dict(explicit)
    return _drop_empty(
        {
            "nav_node": candidate.get("nav_node"),
            "floor_id": candidate.get("floor_id"),
            "room_id": candidate.get("room_id"),
            "distance_bucket_m": candidate.get("distance_bucket_m"),
            "approach_heading_sector": candidate.get("approach_heading_sector"),
        }
    )


def _contact_context(*, visual: dict[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for key in (
        "contact_observed",
        "contact_detected",
        "had_contact",
        "gripper_contact",
    ):
        if visual and key in visual:
            context[key] = visual[key]
    for action in reversed(actions):
        for key in (
            "contact_observed",
            "contact_detected",
            "had_contact",
            "gripper_contact",
        ):
            if key in action and key not in context:
                context[key] = action[key]
    return _drop_empty(context)


def _vlm_predicate_mismatch(
    *,
    heartbeat: dict[str, Any],
    environment_task_success: Any,
) -> dict[str, Any]:
    success_count = heartbeat.get("success_confirmation_count")
    success_threshold = heartbeat.get("success_confirmation_threshold")
    completion_reason = _normalize_text(heartbeat.get("subtask_completion_reason"))
    reported_success = bool(
        heartbeat.get("last_success")
        or heartbeat.get("reported_success")
        or heartbeat.get("subtask_succeeded")
        or (heartbeat.get("subtask_completed") and completion_reason in {"vlm_success", "success"})
        or (isinstance(success_count, (int, float)) and success_count > 0)
    )
    if reported_success is not True or environment_task_success is not False:
        return {}
    return _drop_empty(
        {
            "vlm_reported_success": True,
            "success_confirmation_count": success_count,
            "success_confirmation_threshold": success_threshold,
            "subtask_completion_reason": heartbeat.get("subtask_completion_reason"),
            "last_result": heartbeat.get("last_result"),
            "environment_task_success": environment_task_success,
        }
    )


def _first_failed_action(actions: list[dict[str, Any]]) -> dict[str, Any]:
    for action in reversed(actions):
        if _action_success(action) is False:
            return action
    return {}


def _terminal_action(actions: list[dict[str, Any]]) -> dict[str, Any]:
    for action in reversed(actions):
        if not isinstance(action, dict):
            continue
        if _action_env_feedback(action):
            return action
        parameters = action.get("parameters")
        if isinstance(parameters, dict) and any(
            key in parameters
            for key in ("task_success", "task_progress", "env_step", "control_step")
        ):
            return action
        if any(
            key in action for key in ("task_success", "task_progress", "env_step", "control_step")
        ):
            return action
    return {}


def _heartbeat_from_actions(actions: list[dict[str, Any]]) -> dict[str, Any]:
    for action in reversed(actions):
        heartbeat = _first_mapping(
            _nested_mapping(action, ("environment_vlm_heartbeat",)),
            _nested_mapping(action, ("env_feedback", "environment_vlm_heartbeat")),
            _nested_mapping(action, ("result", "environment_vlm_heartbeat")),
            _nested_mapping(action, ("result", "env_feedback", "environment_vlm_heartbeat")),
            _nested_mapping(action, ("parameters", "environment_vlm_heartbeat")),
            _nested_mapping(action, ("parameters", "env_feedback", "environment_vlm_heartbeat")),
        )
        if heartbeat:
            return heartbeat
    return {}


def _action_env_feedback(action: dict[str, Any]) -> dict[str, Any]:
    return _first_mapping(
        action.get("env_feedback"),
        _nested_mapping(action, ("result", "env_feedback")),
        _nested_mapping(action, ("parameters", "env_feedback")),
    )


def _latest_monitor_environment_feedback(
    episode_context: dict[str, Any],
) -> dict[str, Any]:
    for summary in reversed(_list(episode_context.get("monitor_summaries"))):
        if not isinstance(summary, dict):
            continue
        feedback = _first_mapping(
            summary.get("environment_feedback"),
            summary.get("env_feedback"),
            _nested_mapping(summary, ("payload", "environment_feedback")),
            _nested_mapping(summary, ("payload", "env_feedback")),
        )
        if feedback.get("environment_vlm_heartbeat"):
            return feedback
    return {}


def _looks_like_visual_report(value: dict[str, Any]) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    return any(
        key in value
        for key in (
            "target_visible",
            "target_part_visible",
            "target_part_name",
            "task_complete",
            "switch_visible",
            "estimated_distance_m",
            "contact_observed",
        )
    )


def _action_backend_status(
    episode_context: dict[str, Any],
    *,
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    reason_text = _normalize_text(episode_context.get("failure_reason"))
    backend = _backend_from_text(reason_text)
    status = "backend_failure" if _is_vla_backend_failure(reason_text) else None
    for action in actions:
        action_text = _normalize_text(
            " ".join(
                str(value)
                for value in (
                    action.get("backend"),
                    action.get("policy_backend"),
                    action.get("failure_reason"),
                    action.get("error"),
                )
                if value is not None
            )
        )
        backend = backend or _backend_from_text(action_text)
        if _is_vla_backend_failure(action_text):
            status = "backend_failure"
    return _drop_empty({"backend": backend, "status": status})


def _similar_episode_contrasts(
    similar_episodes: Any, *, current_episode_id: str
) -> list[dict[str, Any]]:
    results = (
        similar_episodes.get("results") if isinstance(similar_episodes, dict) else similar_episodes
    )
    contrasts = []
    for item in _list(results):
        if not isinstance(item, dict):
            continue
        episode_id = str(item.get("episode_id") or "")
        if episode_id and episode_id == current_episode_id:
            continue
        contrast = _select_non_empty(
            item,
            (
                "episode_id",
                "task_description",
                "outcome",
                "failure_reason",
                "lessons_learned",
                "improvement_suggestions",
            ),
        )
        if contrast:
            contrasts.append(contrast)
    return contrasts


def _action_summary(action: dict[str, Any], *, index: int) -> dict[str, Any]:
    return {
        "action_id": _action_id(action, index=index),
        "action_type": str(
            action.get("action_type") or action.get("type") or action.get("action") or ""
        ),
        "target": _target(action),
        "index": index,
    }


def _action_id(action: dict[str, Any], *, index: int) -> str:
    return str(action.get("action_id") or action.get("id") or f"action_{index}")


def _action_success(action: dict[str, Any]) -> bool | None:
    success = action.get("success")
    if isinstance(success, bool):
        return success
    status = str(action.get("status") or action.get("outcome") or "").strip().lower()
    if status in {"success", "succeeded", "ok", "completed"}:
        return True
    if status in {"failure", "failed", "error"}:
        return False
    if action.get("failure_reason"):
        return False
    return None


def _failure_reason(action: dict[str, Any]) -> str | None:
    value = action.get("failure_reason") or action.get("error") or action.get("reason")
    return str(value) if value is not None else None


def _target(action: dict[str, Any]) -> Any:
    target = action.get("target")
    if target is not None:
        return target
    for key in ("object", "object_name", "region", "room"):
        if action.get(key) is not None:
            return action[key]
    return ""


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _first_present(value: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in value:
            return value[key]
    return None


def _first_mapping(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict) and value:
            return value
    return {}


def _nested_mapping(value: Any, path: tuple[str, ...]) -> dict[str, Any]:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return current if isinstance(current, dict) and current else {}


def _first_present_in_mappings(key: str, *values: dict[str, Any]) -> Any:
    for value in values:
        if isinstance(value, dict) and key in value:
            return value[key]
    return None


def _select_non_empty(value: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    selected = {}
    for key in keys:
        current = value.get(key)
        if current not in (None, [], {}):
            selected[key] = current
    return selected


def _drop_empty(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item not in (None, "", [], {})}


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _is_vla_backend_failure(text: str) -> bool:
    if not text:
        return False
    backend_terms = ("vla", "openpi", "groot", "pi0", "policy backend")
    failure_terms = (
        "backend",
        "timeout",
        "connection",
        "http",
        "server",
        "unavailable",
        "oom",
    )
    return any(term in text for term in backend_terms) and any(
        term in text for term in failure_terms
    )


def _backend_from_text(text: str) -> str | None:
    if "openpi_comet" in text or "openpi comet" in text:
        return "openpi_comet"
    if "openpi" in text:
        return "openpi"
    if "groot" in text:
        return "groot"
    if "pi0" in text or "pi05" in text:
        return "pi0"
    if "vla" in text:
        return "vla"
    return None


__all__ = ["build_reflection_evidence", "reflection_annotation"]
