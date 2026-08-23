"""Write extracted experience back to the memory backend."""

from __future__ import annotations

from typing import Any

from voltron.agents.memory.contracts.experience import (
    ExperienceExtractionResult,
    causal_hypothesis_to_payload,
    failure_pattern_to_payload,
    hint_to_payload,
    semantic_update_to_payload,
    skill_candidate_to_payload,
)


def write_experience(
    *,
    backend: Any,
    episode_context: dict[str, Any],
    extraction: ExperienceExtractionResult,
    min_confidence_to_write: float,
    min_confidence_to_promote: float,
) -> dict[str, Any]:
    episode_id = extraction.source_episode_id or str(episode_context.get("episode_id") or "")
    task_description = str(episode_context.get("task_description") or "")
    task_type = episode_context.get("task_type")
    task_type_value = str(task_type) if task_type is not None else None

    written = {
        "episode_annotation": False,
        "experience_hints": 0,
        "failure_patterns": 0,
        "semantic_updates": 0,
        "skill_candidates": 0,
        "causal_hypotheses": 0,
        "object_approach_priors": 0,
        "task_context_mirror": False,
    }
    written_ids = {
        "experience_hint_ids": [],
        "failure_pattern_ids": [],
        "semantic_update_ids": [],
        "skill_candidate_ids": [],
        "causal_hypothesis_ids": [],
        "object_approach_prior_signatures": [],
    }
    annotation_result = None
    if episode_id:
        annotation_result = backend.annotate_completed_episode(
            episode_id,
            extraction.to_annotation(),
        )
        written["episode_annotation"] = True

    for hint in extraction.retrieval_hints:
        payload = hint_to_payload(
            hint,
            task_description=task_description,
            task_type=task_type_value,
            source_episode_id=episode_id,
        )
        if payload["confidence"] < min_confidence_to_write:
            continue
        store_result = backend.store_experience_hint(payload)
        hint_id = _extract_written_id(store_result, "hint_id", fallback=payload.get("hint_id"))
        if hint_id:
            written_ids["experience_hint_ids"].append(hint_id)
            written["experience_hints"] += 1

    store_failure_pattern = getattr(backend, "store_failure_pattern_candidate", None)
    if callable(store_failure_pattern):
        for pattern in extraction.failure_patterns:
            payload = failure_pattern_to_payload(
                pattern,
                task_description=task_description,
                task_type=task_type_value,
                source_episode_id=episode_id,
            )
            if payload["confidence"] < min_confidence_to_write:
                continue
            store_result = store_failure_pattern(payload)
            pattern_id = _extract_written_id(
                store_result,
                "pattern_id",
                fallback=payload.get("pattern_id"),
            )
            if pattern_id:
                written_ids["failure_pattern_ids"].append(pattern_id)
                written["failure_patterns"] += 1

    store_semantic_update = getattr(backend, "store_semantic_update_candidate", None)
    if callable(store_semantic_update):
        for update in extraction.semantic_updates:
            payload = semantic_update_to_payload(update, source_episode_id=episode_id)
            if payload["confidence"] < min_confidence_to_write:
                continue
            store_result = store_semantic_update(payload)
            update_id = _extract_written_id(
                store_result,
                "update_id",
                fallback=payload.get("update_id"),
            )
            if update_id:
                written_ids["semantic_update_ids"].append(update_id)
                written["semantic_updates"] += 1

    store_skill_candidate = getattr(backend, "store_skill_candidate", None)
    if callable(store_skill_candidate):
        for candidate in extraction.procedural_skills:
            payload = skill_candidate_to_payload(candidate, source_episode_id=episode_id)
            if payload["confidence"] < min_confidence_to_promote:
                continue
            store_result = store_skill_candidate(payload)
            candidate_id = _extract_written_id(
                store_result,
                "candidate_id",
                fallback=payload.get("candidate_id"),
            )
            if candidate_id:
                written_ids["skill_candidate_ids"].append(candidate_id)
                written["skill_candidates"] += 1

    store_causal_hypothesis = getattr(backend, "store_causal_hypothesis", None)
    if callable(store_causal_hypothesis):
        for hypothesis in extraction.causal_hypotheses:
            payload = causal_hypothesis_to_payload(hypothesis, source_episode_id=episode_id)
            if payload["confidence"] < min_confidence_to_promote:
                continue
            store_result = store_causal_hypothesis(payload)
            hypothesis_id = _extract_written_id(
                store_result,
                "hypothesis_id",
                fallback=payload.get("hypothesis_id"),
            )
            if hypothesis_id:
                written_ids["causal_hypothesis_ids"].append(hypothesis_id)
                written["causal_hypotheses"] += 1

    record_object_approach_outcome = getattr(backend, "record_object_approach_outcome", None)
    if callable(record_object_approach_outcome):
        for prior in extraction.object_approach_priors:
            payload = _object_approach_prior_to_payload(
                prior,
                episode_context=episode_context,
                source_episode_id=episode_id,
            )
            if not payload or payload["metadata"]["confidence"] < min_confidence_to_write:
                continue
            record_result = record_object_approach_outcome(**payload)
            signature = _object_approach_prior_signature(record_result)
            if signature:
                written_ids["object_approach_prior_signatures"].append(signature)
                written["object_approach_priors"] += 1

    verification = _verify_written_ids(backend=backend, written_ids=written_ids)
    readback = backend.find_experience_hints(
        task_description,
        task_type=task_type_value,
        top_k=max(1, written["experience_hints"]),
    )
    update_task_context = getattr(backend, "update_task_context", None)
    if callable(update_task_context):
        update_task_context(
            {
                "memory_consolidation": {
                    "latest": {
                        "episode_id": episode_id,
                        "episode_summary": extraction.episode_summary,
                        "confidence": extraction.confidence,
                        "written": dict(written),
                        "written_ids": _clone_written_ids(written_ids),
                        "verification": dict(verification),
                        "readback": readback,
                    }
                }
            }
        )
        written["task_context_mirror"] = True
    return {
        "episode_id": episode_id,
        "written": written,
        "written_ids": written_ids,
        "annotation": annotation_result,
        "verification": verification,
        "readback": readback,
    }


def _object_approach_prior_to_payload(
    value: dict[str, Any],
    *,
    episode_context: dict[str, Any],
    source_episode_id: str,
    generated_by: str = "memory_agent",
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    scene_id = str(value.get("scene_id") or episode_context.get("scene_id") or "").strip()
    target = value.get("target")
    candidate = value.get("candidate")
    if not isinstance(target, dict) or not isinstance(candidate, dict) or not scene_id:
        return None

    confidence = _clamp_confidence(value.get("confidence", 0.0))
    metadata = dict(value.get("metadata", {})) if isinstance(value.get("metadata"), dict) else {}
    metadata.update(
        {
            "confidence": confidence,
            "source_episode_id": value.get("source_episode_id") or source_episode_id,
            "source_action_ids": list(value.get("source_action_ids", [])),
            "generated_by": value.get("generated_by") or generated_by,
        }
    )
    if value.get("summary"):
        metadata["summary"] = str(value["summary"])

    return {
        "scene_id": scene_id,
        "target": dict(target),
        "candidate": dict(candidate),
        "outcome": str(value.get("outcome") or "prior").strip().lower() or "prior",
        "reason": value.get("reason"),
        "metadata": metadata,
    }


def _clamp_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def _extract_written_id(
    result: Any,
    id_key: str,
    *,
    fallback: Any = None,
) -> str | None:
    if isinstance(result, str):
        return result or None
    if isinstance(result, dict):
        if result.get("stored") is False:
            return None
        value = result.get(id_key)
        if value:
            return str(value)
    if fallback:
        return str(fallback)
    return None


def _verify_written_ids(*, backend: Any, written_ids: dict[str, list[Any]]) -> dict[str, Any]:
    experience_readback = _read_back_ids(
        backend=backend,
        getter_name="get_experience_hint",
        id_key="hint_id",
        ids=written_ids["experience_hint_ids"],
    )
    skill_readback = _read_back_ids(
        backend=backend,
        getter_name="get_skill_candidate",
        id_key="candidate_id",
        ids=written_ids["skill_candidate_ids"],
    )
    causal_readback = _read_back_ids(
        backend=backend,
        getter_name="get_causal_hypothesis",
        id_key="hypothesis_id",
        ids=written_ids["causal_hypothesis_ids"],
    )
    failure_readback = _read_back_ids(
        backend=backend,
        getter_name="get_failure_pattern_candidate",
        id_key="pattern_id",
        ids=written_ids["failure_pattern_ids"],
    )
    semantic_readback = _read_back_ids(
        backend=backend,
        getter_name="get_semantic_update_candidate",
        id_key="update_id",
        ids=written_ids["semantic_update_ids"],
    )
    object_signatures = list(written_ids["object_approach_prior_signatures"])
    readback_ids = {
        "experience_hint_ids": experience_readback["readback_ids"],
        "failure_pattern_ids": failure_readback["readback_ids"],
        "semantic_update_ids": semantic_readback["readback_ids"],
        "skill_candidate_ids": skill_readback["readback_ids"],
        "causal_hypothesis_ids": causal_readback["readback_ids"],
        "object_approach_prior_signatures": object_signatures,
    }
    missing_ids = {
        "experience_hint_ids": experience_readback["missing_ids"],
        "failure_pattern_ids": failure_readback["missing_ids"],
        "semantic_update_ids": semantic_readback["missing_ids"],
        "skill_candidate_ids": skill_readback["missing_ids"],
        "causal_hypothesis_ids": causal_readback["missing_ids"],
    }
    return {
        "written_ids": _clone_written_ids(written_ids),
        "experience_hints_verified": not missing_ids["experience_hint_ids"],
        "failure_patterns_verified": not missing_ids["failure_pattern_ids"],
        "semantic_updates_verified": not missing_ids["semantic_update_ids"],
        "skill_candidates_verified": not missing_ids["skill_candidate_ids"],
        "causal_hypotheses_verified": not missing_ids["causal_hypothesis_ids"],
        "object_approach_priors_recorded": (
            len(object_signatures) == len(written_ids["object_approach_prior_signatures"])
        ),
        "readback_ids": readback_ids,
        "missing_ids": missing_ids,
        "reload_verified": None,
    }


def _read_back_ids(
    *,
    backend: Any,
    getter_name: str,
    id_key: str,
    ids: list[str],
) -> dict[str, list[str]]:
    getter = getattr(backend, getter_name, None)
    if not ids:
        return {"readback_ids": [], "missing_ids": []}
    if not callable(getter):
        return {"readback_ids": [], "missing_ids": list(ids)}

    readback_ids: list[str] = []
    missing_ids: list[str] = []
    for item_id in ids:
        item = getter(item_id)
        if isinstance(item, dict) and str(item.get(id_key) or "") == item_id:
            readback_ids.append(item_id)
        elif item is not None and getattr(item, id_key, None) == item_id:
            readback_ids.append(item_id)
        else:
            missing_ids.append(item_id)
    return {"readback_ids": readback_ids, "missing_ids": missing_ids}


def _object_approach_prior_signature(record_result: Any) -> dict[str, Any] | None:
    if not isinstance(record_result, dict) or record_result.get("status") != "recorded":
        return None
    entry = record_result.get("entry")
    if not isinstance(entry, dict):
        return None
    candidate_signature = entry.get("candidate_signature")
    if not isinstance(candidate_signature, dict):
        return None
    return {
        "scene_id": record_result.get("scene_id"),
        "target_key": record_result.get("target_key"),
        "candidate_signature": dict(candidate_signature),
    }


def _clone_written_ids(written_ids: dict[str, list[Any]]) -> dict[str, list[Any]]:
    return {key: list(value) for key, value in written_ids.items()}
