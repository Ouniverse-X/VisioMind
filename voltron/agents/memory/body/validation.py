from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any

from voltron.agents.memory.contracts.experience import (
    ExperienceExtractionResult,
    normalize_retrieval_hint,
)

_VALID_AGENT_NAMES = {"BRAIN", "VISION", "NAVIGATION", "ACTION", "MEMORY"}


def validate_extraction_result(
    extraction: ExperienceExtractionResult,
    episode_context: dict[str, Any],
    *,
    backend: Any | None = None,
) -> ExperienceExtractionResult:
    warnings = list(extraction.validation_warnings)
    episode_id = str(episode_context.get("episode_id") or "")
    if episode_id and extraction.source_episode_id and extraction.source_episode_id != episode_id:
        warnings.append(f"source_episode_id_mismatch:{extraction.source_episode_id}->{episode_id}")
        extraction.source_episode_id = episode_id
    elif episode_id and not extraction.source_episode_id:
        extraction.source_episode_id = episode_id

    evidence = _build_evidence_index(episode_context)

    episode_succeeded = _episode_succeeded(episode_context)

    extraction.retrieval_hints = _validate_retrieval_hints(
        extraction.retrieval_hints,
        evidence=evidence,
        warnings=warnings,
        episode_succeeded=episode_succeeded,
        existing_keys=_existing_retrieval_hint_keys(
            backend,
            episode_context=episode_context,
            hints=extraction.retrieval_hints,
        ),
    )
    extraction.procedural_skills = _validate_procedural_skills(
        extraction.procedural_skills,
        evidence=evidence,
        warnings=warnings,
    )
    extraction.causal_hypotheses = _validate_causal_hypotheses(
        extraction.causal_hypotheses,
        evidence=evidence,
        warnings=warnings,
    )
    extraction.failure_patterns = _validate_failure_patterns(
        extraction.failure_patterns,
        evidence=evidence,
        warnings=warnings,
        episode_succeeded=episode_succeeded,
    )
    extraction.semantic_updates = _validate_semantic_updates(
        extraction.semantic_updates,
        evidence=evidence,
        warnings=warnings,
        backend=backend,
    )
    extraction.object_approach_priors = _validate_object_approach_priors(
        extraction.object_approach_priors,
        evidence=evidence,
        warnings=warnings,
    )
    extraction.validation_warnings = warnings
    return extraction


def _validate_retrieval_hints(
    hints: list[Any],
    *,
    evidence: dict[str, Any],
    warnings: list[str],
    episode_succeeded: bool,
    existing_keys: set[str] | None = None,
) -> list[Any]:
    validated = []
    seen: set[str] = set()
    existing_keys = existing_keys or set()
    for index, hint in enumerate(hints):
        normalized = normalize_retrieval_hint(hint)
        if (
            episode_succeeded
            and _normalize_text(normalized.hint_type) == "failure_avoidance"
            and not normalized.source_action_ids
        ):
            warnings.append(f"success_episode_failure_avoidance_hint:{index}")
            continue
        _filter_source_action_ids(
            normalized,
            item_label=f"retrieval_hints[{index}]",
            evidence=evidence,
            warnings=warnings,
        )
        key = _dedup_key(
            "retrieval_hint",
            normalized.hint_type,
            _normalize_text(normalized.summary),
            normalized.content,
        )
        if key in seen:
            warnings.append(f"duplicate_retrieval_hint:{index}")
            continue
        if key in existing_keys:
            warnings.append(f"existing_retrieval_hint_duplicate:{index}")
            continue
        seen.add(key)
        validated.append(normalized)
    return validated


def _validate_failure_patterns(
    patterns: list[dict[str, Any]],
    *,
    evidence: dict[str, Any],
    warnings: list[str],
    episode_succeeded: bool,
) -> list[dict[str, Any]]:
    validated = []
    seen: set[str] = set()
    for index, pattern in enumerate(patterns):
        if not isinstance(pattern, dict):
            warnings.append(f"invalid_failure_pattern:{index}:not_dict")
            continue
        pattern = dict(pattern)
        if episode_succeeded and not _has_failed_source_action(pattern, evidence):
            warnings.append(f"success_episode_failure_pattern:{index}")
            continue
        _filter_source_action_ids(
            pattern,
            item_label=f"failure_patterns[{index}]",
            evidence=evidence,
            warnings=warnings,
        )
        key = _dedup_key(
            "failure_pattern",
            _normalize_text(pattern.get("pattern_type")),
            _normalize_text(pattern.get("summary")),
            pattern.get("conditions", {}),
        )
        if key in seen:
            warnings.append(f"duplicate_failure_pattern:{index}")
            continue
        seen.add(key)
        validated.append(pattern)
    return validated


def _validate_procedural_skills(
    candidates: list[dict[str, Any]],
    *,
    evidence: dict[str, Any],
    warnings: list[str],
) -> list[dict[str, Any]]:
    validated = []
    seen: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            warnings.append(f"invalid_procedural_skill:{index}:not_dict")
            continue
        candidate = dict(candidate)
        if not _filter_source_action_ids(
            candidate,
            item_label=f"procedural_skills[{index}]",
            evidence=evidence,
            warnings=warnings,
            require_supported=True,
        ):
            continue
        if _is_failed_only_skill(candidate, evidence):
            warnings.append(f"failed_only_procedural_skill:{index}")
            continue
        if not _validate_skill_steps(candidate, index=index, warnings=warnings):
            continue
        key = _dedup_key(
            "procedural_skill",
            _normalize_text(candidate.get("name")),
            candidate.get("trigger", {}),
            candidate.get("steps", []),
        )
        if key in seen:
            warnings.append(f"duplicate_procedural_skill:{index}")
            continue
        seen.add(key)
        validated.append(candidate)
    return validated


def _validate_causal_hypotheses(
    hypotheses: list[dict[str, Any]],
    *,
    evidence: dict[str, Any],
    warnings: list[str],
) -> list[dict[str, Any]]:
    validated = []
    seen: set[str] = set()
    for index, hypothesis in enumerate(hypotheses):
        if not isinstance(hypothesis, dict):
            warnings.append(f"invalid_causal_hypothesis:{index}:not_dict")
            continue
        hypothesis = dict(hypothesis)
        if not _filter_source_action_ids(
            hypothesis,
            item_label=f"causal_hypotheses[{index}]",
            evidence=evidence,
            warnings=warnings,
            require_supported=True,
        ):
            continue
        if not _causal_claim_has_support(hypothesis, evidence):
            warnings.append(f"unsupported_causal_hypothesis:{index}")
            continue
        key = _dedup_key(
            "causal_hypothesis",
            _normalize_text(hypothesis.get("action")),
            _normalize_text(hypothesis.get("target")),
            _normalize_text(hypothesis.get("expected_effect")),
            hypothesis.get("effect_value"),
            hypothesis.get("conditions", {}),
        )
        if key in seen:
            warnings.append(f"duplicate_causal_hypothesis:{index}")
            continue
        seen.add(key)
        validated.append(hypothesis)
    return validated


def _validate_semantic_updates(
    updates: list[dict[str, Any]],
    *,
    evidence: dict[str, Any],
    warnings: list[str],
    backend: Any | None,
) -> list[dict[str, Any]]:
    validated = []
    seen: set[str] = set()
    for index, update in enumerate(updates):
        if not isinstance(update, dict):
            warnings.append(f"invalid_semantic_update:{index}:not_dict")
            continue
        update = dict(update)
        _filter_source_action_ids(
            update,
            item_label=f"semantic_updates[{index}]",
            evidence=evidence,
            warnings=warnings,
        )
        if _semantic_update_conflicts(update, backend):
            warnings.append(f"conflicting_semantic_update:{index}")
            continue
        if _is_transient_semantic_update(update):
            content = (
                dict(update.get("content", {})) if isinstance(update.get("content"), dict) else {}
            )
            content.setdefault("memory_scope", "episode_state")
            update["content"] = content
            warnings.append(f"episode_scoped_semantic_update:{index}")
        key = _dedup_key(
            "semantic_update",
            _normalize_text(update.get("update_type")),
            _normalize_text(update.get("subject")),
            _normalize_text(update.get("relation")),
            _normalize_text(update.get("object")),
            update.get("content", {}),
        )
        if key in seen:
            warnings.append(f"duplicate_semantic_update:{index}")
            continue
        seen.add(key)
        validated.append(update)
    return validated


def _validate_object_approach_priors(
    priors: list[dict[str, Any]],
    *,
    evidence: dict[str, Any],
    warnings: list[str],
) -> list[dict[str, Any]]:
    validated = []
    seen: set[str] = set()
    for index, prior in enumerate(priors):
        if not isinstance(prior, dict):
            warnings.append(f"invalid_object_approach_prior:{index}:not_dict")
            continue
        prior = dict(prior)
        _filter_source_action_ids(
            prior,
            item_label=f"object_approach_priors[{index}]",
            evidence=evidence,
            warnings=warnings,
        )
        key = _dedup_key(
            "object_approach_prior",
            prior.get("scene_id"),
            prior.get("target", {}),
            _candidate_signature(prior.get("candidate", {})),
            _normalize_text(prior.get("outcome") or "prior"),
        )
        if key in seen:
            warnings.append(f"duplicate_object_approach_prior:{index}")
            continue
        seen.add(key)
        validated.append(prior)
    return validated


def _validate_skill_steps(candidate: dict[str, Any], *, index: int, warnings: list[str]) -> bool:
    steps = candidate.get("steps")
    if not isinstance(steps, list) or not steps:
        warnings.append(f"invalid_skill_steps:procedural_skills[{index}]")
        return False
    for step in steps:
        if not isinstance(step, dict):
            warnings.append(f"invalid_skill_step:procedural_skills[{index}]:not_dict")
            return False
        agent = str(step.get("agent", "")).strip().upper()
        if agent not in _VALID_AGENT_NAMES:
            warnings.append(f"invalid_step_agent:procedural_skills[{index}]:{agent or '<missing>'}")
            return False
    return True


def _filter_source_action_ids(
    item: Any,
    *,
    item_label: str,
    evidence: dict[str, Any],
    warnings: list[str],
    require_supported: bool = False,
) -> bool:
    action_ids = evidence["action_ids"]
    if not action_ids:
        return True

    original = _get_field(item, "source_action_ids", [])
    if original is None:
        original = []
    original_ids = [str(action_id) for action_id in original if str(action_id)]
    supported_ids = []
    for action_id in original_ids:
        if action_id in action_ids:
            supported_ids.append(action_id)
        else:
            warnings.append(f"unsupported_source_action_id:{item_label}:{action_id}")
    _set_field(item, "source_action_ids", supported_ids)
    if require_supported and not supported_ids:
        warnings.append(f"missing_supported_source_action_id:{item_label}")
        return False
    return True


def _causal_claim_has_support(hypothesis: dict[str, Any], evidence: dict[str, Any]) -> bool:
    source_ids = {str(action_id) for action_id in hypothesis.get("source_action_ids", [])}
    expected_effect = _normalize_text(hypothesis.get("expected_effect"))
    if expected_effect and expected_effect in evidence["transition_attributes"]:
        return True
    if expected_effect and expected_effect in evidence["causal_effect_attributes"]:
        return True
    if source_ids and source_ids & evidence["state_delta_source_action_ids"]:
        return True
    if source_ids and source_ids & evidence["causal_observation_source_action_ids"]:
        return True
    return False


def _build_evidence_index(episode_context: dict[str, Any]) -> dict[str, Any]:
    actions = []
    for key in ("actions", "action_sequence"):
        value = episode_context.get(key)
        if isinstance(value, list):
            actions.extend(item for item in value if isinstance(item, dict))
    episode = episode_context.get("episode")
    if isinstance(episode, dict) and isinstance(episode.get("action_sequence"), list):
        actions.extend(item for item in episode["action_sequence"] if isinstance(item, dict))

    action_ids = set()
    action_types = set()
    action_success: dict[str, bool | None] = {}
    for action in actions:
        action_id = action.get("action_id") or action.get("id")
        if action_id:
            action_id = str(action_id)
            action_ids.add(action_id)
            action_success[action_id] = _action_success(action)
        action_type = action.get("action_type") or action.get("action")
        if action_type:
            action_types.add(_normalize_text(action_type))

    transition_attributes = set()
    state_delta_source_action_ids = set()
    for transition in _transition_items(episode_context):
        caused_by = transition.get("caused_by")
        if caused_by:
            action_ids.add(str(caused_by))
            state_delta_source_action_ids.add(str(caused_by))
        source_action_id = transition.get("source_action_id")
        if source_action_id:
            action_ids.add(str(source_action_id))
            state_delta_source_action_ids.add(str(source_action_id))
        attribute = transition.get("attribute")
        if attribute:
            transition_attributes.add(_normalize_text(attribute))

    causal_effect_attributes = set()
    causal_observation_source_action_ids = set()
    for observation in _causal_observation_items(episode_context):
        source_action_id = (
            observation.get("source_action_id")
            or observation.get("action_id")
            or observation.get("cause_action_id")
        )
        if source_action_id:
            action_ids.add(str(source_action_id))
            causal_observation_source_action_ids.add(str(source_action_id))
        attribute = observation.get("effect_attribute") or observation.get("attribute")
        if attribute:
            causal_effect_attributes.add(_normalize_text(attribute))

    return {
        "action_ids": action_ids,
        "action_types": action_types,
        "action_success": action_success,
        "transition_attributes": transition_attributes,
        "state_delta_source_action_ids": state_delta_source_action_ids,
        "causal_effect_attributes": causal_effect_attributes,
        "causal_observation_source_action_ids": causal_observation_source_action_ids,
        "episode_outcome": _normalize_text(episode_context.get("outcome")),
    }


def _transition_items(episode_context: dict[str, Any]) -> list[dict[str, Any]]:
    transitions = []
    for key in ("state_transitions",):
        value = episode_context.get(key)
        if isinstance(value, list):
            transitions.extend(item for item in value if isinstance(item, dict))
    reflection = episode_context.get("reflection_evidence")
    if isinstance(reflection, dict) and isinstance(reflection.get("state_deltas"), list):
        transitions.extend(item for item in reflection["state_deltas"] if isinstance(item, dict))
    episode = episode_context.get("episode")
    if isinstance(episode, dict) and isinstance(episode.get("state_transitions"), list):
        transitions.extend(item for item in episode["state_transitions"] if isinstance(item, dict))
    return transitions


def _causal_observation_items(episode_context: dict[str, Any]) -> list[dict[str, Any]]:
    observations = []
    value = episode_context.get("causal_annotations")
    if isinstance(value, list):
        observations.extend(item for item in value if isinstance(item, dict))
    reflection = episode_context.get("reflection_evidence")
    if isinstance(reflection, dict) and isinstance(reflection.get("causal_observations"), list):
        observations.extend(
            item for item in reflection["causal_observations"] if isinstance(item, dict)
        )
    return observations


def _is_failed_only_skill(candidate: dict[str, Any], evidence: dict[str, Any]) -> bool:
    source_ids = [
        str(action_id) for action_id in candidate.get("source_action_ids", []) if str(action_id)
    ]
    action_success = evidence["action_success"]
    if source_ids:
        statuses = [
            action_success.get(action_id) for action_id in source_ids if action_id in action_success
        ]
        if statuses and all(status is False for status in statuses):
            return True
        if any(status is True for status in statuses):
            return False
    return evidence.get("episode_outcome") in {"failure", "failed"} and not any(
        status is True for status in action_success.values()
    )


def _existing_retrieval_hint_keys(
    backend: Any | None,
    *,
    episode_context: dict[str, Any],
    hints: list[Any],
) -> set[str]:
    if backend is None or not hints:
        return set()
    finder = getattr(backend, "find_experience_hints", None)
    if not callable(finder):
        return set()
    task_description = str(episode_context.get("task_description") or "").strip()
    if not task_description:
        return set()
    try:
        result = finder(
            task_description,
            task_type=episode_context.get("task_type"),
            top_k=max(10, len(hints) * 2),
        )
    except Exception:
        return set()
    results = result.get("results") if isinstance(result, dict) else result
    keys = set()
    for item in results if isinstance(results, list) else []:
        if not isinstance(item, dict):
            continue
        normalized = normalize_retrieval_hint(item)
        keys.add(
            _dedup_key(
                "retrieval_hint",
                normalized.hint_type,
                _normalize_text(normalized.summary),
                normalized.content,
            )
        )
    return keys


def _semantic_update_conflicts(update: dict[str, Any], backend: Any | None) -> bool:
    if backend is None:
        return False
    finder = getattr(backend, "find_object", None)
    if not callable(finder):
        return False
    subject = str(update.get("subject") or "").strip()
    relation = _normalize_text(update.get("relation"))
    new_value = update.get("object")
    if not subject or not relation or new_value in (None, ""):
        return False
    try:
        result = finder(subject, top_k=3)
    except Exception:
        return False
    results = result.get("results") if isinstance(result, dict) else result
    for item in results if isinstance(results, list) else []:
        if not isinstance(item, dict):
            continue
        existing_value = _existing_semantic_value(item, relation)
        if existing_value is None:
            continue
        if _normalize_text(existing_value) != _normalize_text(new_value):
            return True
    return False


def _existing_semantic_value(item: dict[str, Any], relation: str) -> Any:
    relations = item.get("relations")
    if isinstance(relations, dict) and relation in {_normalize_text(key) for key in relations}:
        for key, value in relations.items():
            if _normalize_text(key) == relation:
                return value
    attributes = item.get("attributes")
    if isinstance(attributes, dict):
        for key, value in attributes.items():
            if _normalize_text(key) == relation:
                return value
    for key in (
        relation,
        relation.replace("_", ""),
        "room",
        "room_name",
        "region",
        "location",
    ):
        if key in item:
            return item[key]
    if relation in {"in_room", "room", "located_in"}:
        for key in ("room", "room_name", "region", "location"):
            if key in item:
                return item[key]
    return None


def _candidate_signature(candidate: Any) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        return {}
    explicit = candidate.get("candidate_signature")
    if isinstance(explicit, dict) and explicit:
        return dict(explicit)
    if candidate.get("candidate_id"):
        return {"candidate_id": candidate.get("candidate_id")}
    if candidate.get("nav_node") is not None:
        return {"nav_node": candidate.get("nav_node")}
    signature = {}
    for axis in ("x", "y", "z"):
        value = candidate.get(axis)
        if value is None:
            continue
        try:
            signature[axis] = round(float(value), 2)
        except (TypeError, ValueError):
            continue
    for key in ("floor_id", "room_id"):
        if candidate.get(key) is not None:
            signature[key] = candidate[key]
    return signature


def _episode_succeeded(episode_context: dict[str, Any]) -> bool:
    if _normalize_text(episode_context.get("outcome")) in {
        "success",
        "succeeded",
        "completed",
    }:
        return True
    final_state = episode_context.get("final_state")
    final_state = final_state if isinstance(final_state, dict) else {}
    environment = final_state.get("environment")
    environment = environment if isinstance(environment, dict) else {}
    for source in (final_state, environment, episode_context):
        if source.get("task_success") is True:
            return True
    reflection = episode_context.get("reflection_evidence")
    if isinstance(reflection, dict) and _normalize_text(reflection.get("outcome")) in {
        "success",
        "succeeded",
        "completed",
    }:
        return True
    return False


def _has_failed_source_action(item: dict[str, Any], evidence: dict[str, Any]) -> bool:
    action_success = evidence.get("action_success", {})
    source_ids = [
        str(action_id) for action_id in item.get("source_action_ids", []) if str(action_id)
    ]
    return any(action_success.get(action_id) is False for action_id in source_ids)


def _is_transient_semantic_update(update: dict[str, Any]) -> bool:
    relation = _normalize_text(update.get("relation"))
    update_type = _normalize_text(update.get("update_type"))
    transient_relations = {
        "power_state",
        "powered_on",
        "is_on",
        "state",
        "task_state",
        "door_open",
        "open",
        "closed",
    }
    if relation in transient_relations:
        return True
    return update_type in {"attribute", "state", "state_update"} and relation.endswith("_state")


def _dedup_key(*parts: Any) -> str:
    return json.dumps([_to_plain(part) for part in parts], sort_keys=True, default=str)


def _to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return _to_plain(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _to_plain(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple, set)):
        return [_to_plain(item) for item in value]
    return value


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _get_field(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _set_field(item: Any, key: str, value: Any) -> None:
    if isinstance(item, dict):
        item[key] = value
    else:
        setattr(item, key, value)


def _action_success(action: dict[str, Any]) -> bool | None:
    success = action.get("success")
    if isinstance(success, bool):
        return success
    status = _normalize_text(action.get("status") or action.get("outcome"))
    if status in {"success", "succeeded", "ok", "completed"}:
        return True
    if status in {"failure", "failed", "error"}:
        return False
    if action.get("failure_reason") or action.get("error"):
        return False
    return None


__all__ = ["validate_extraction_result"]
