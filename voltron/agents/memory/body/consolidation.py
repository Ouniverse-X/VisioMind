from __future__ import annotations

from typing import Any

from voltron.agents.memory.contracts.experience import (
    RetrievalHint,
    normalize_extraction_result,
)

from .validation import validate_extraction_result
from .writeback import write_experience


def consolidate_completed_episode(
    *,
    backend: Any,
    extractor: Any,
    episode_id: str | None,
    episode_context: dict[str, Any] | None = None,
    reflection_evidence: dict[str, Any] | None = None,
    min_confidence_to_write: float,
    min_confidence_to_promote: float,
) -> dict[str, Any]:
    if extractor is None:
        return {
            "ok": False,
            "episode_id": episode_id,
            "error": "extractor_not_configured",
        }

    try:
        if episode_context is None:
            episode_context = backend.get_completed_episode_context(episode_id=episode_id)
        else:
            episode_context = dict(episode_context)
        if reflection_evidence is not None:
            episode_context["reflection_evidence"] = dict(reflection_evidence)
        extraction = normalize_extraction_result(extractor.extract(episode_context))
        if not extraction.source_episode_id:
            extraction.source_episode_id = str(
                episode_context.get("episode_id") or episode_id or ""
            )
        _augment_with_rule_derived_experience(extraction, episode_context)
        extraction = validate_extraction_result(extraction, episode_context, backend=backend)
        writeback = write_experience(
            backend=backend,
            episode_context=episode_context,
            extraction=extraction,
            min_confidence_to_write=min_confidence_to_write,
            min_confidence_to_promote=min_confidence_to_promote,
        )
        return {
            "ok": True,
            "episode_id": writeback["episode_id"],
            "extraction": {
                "episode_summary": extraction.episode_summary,
                "confidence": extraction.confidence,
                "validation_warnings": list(extraction.validation_warnings),
            },
            "written": writeback["written"],
            "written_ids": writeback["written_ids"],
            "verification": writeback["verification"],
            "readback": writeback["readback"],
        }
    except Exception as exc:
        return {"ok": False, "episode_id": episode_id, "error": str(exc)}


def _augment_with_rule_derived_experience(
    extraction: Any,
    episode_context: dict[str, Any],
) -> None:
    completion_hint = _completion_criteria_hint(episode_context)
    if completion_hint is not None:
        extraction.retrieval_hints.append(completion_hint)

    rule_evidence = _rule_derived_evidence(episode_context)
    interaction = rule_evidence.get("action_interaction_context")
    if not isinstance(interaction, dict) or not interaction:
        return
    selected = interaction.get("selected_candidate")
    selected = selected if isinstance(selected, dict) else {}
    environment_outcome = interaction.get("environment_outcome")
    environment_outcome = environment_outcome if isinstance(environment_outcome, dict) else {}
    environment_outcome = _merged_final_environment_outcome(
        environment_outcome,
        episode_context=episode_context,
    )
    if selected and environment_outcome.get("task_success") is True:
        prior = _successful_object_approach_prior(
            rule_evidence,
            episode_context=episode_context,
        )
        if prior:
            extraction.object_approach_priors.append(prior)
        return
    if not selected or environment_outcome.get("task_success") is not False:
        return

    critical_action = rule_evidence.get("critical_failure_action")
    critical_action = critical_action if isinstance(critical_action, dict) else {}
    source_action_ids = (
        [str(critical_action["action_id"])] if critical_action.get("action_id") else []
    )
    candidate_id = str(selected.get("candidate_id") or "selected_candidate")
    candidate_signature = selected.get("candidate_signature")
    candidate_signature = dict(candidate_signature) if isinstance(candidate_signature, dict) else {}
    visual = interaction.get("visual_affordance")
    visual = visual if isinstance(visual, dict) else {}
    distance = interaction.get("distance_context")
    distance = distance if isinstance(distance, dict) else {}
    contact = interaction.get("contact_context")
    contact = contact if isinstance(contact, dict) else {}
    mismatch = interaction.get("vlm_predicate_mismatch")
    mismatch = mismatch if isinstance(mismatch, dict) else {}

    summary = (
        f"Candidate {candidate_id} reached a visible target, but the action stage did not satisfy "
        "the environment predicate; use predicate/task_success instead of VLM success as completion."
    )
    content = {
        "candidate_id": candidate_id,
        "candidate_signature": candidate_signature,
        "distance_context": dict(distance),
        "visual_affordance": dict(visual),
        "contact_context": dict(contact),
        "environment_outcome": dict(environment_outcome),
        "vlm_predicate_mismatch": dict(mismatch),
    }
    extraction.retrieval_hints.append(
        RetrievalHint(
            hint_type="failure_avoidance",
            summary=summary,
            confidence=0.78,
            source_action_ids=source_action_ids,
            generated_by="memory_agent_rule_derived",
            content=content,
        )
    )
    extraction.failure_patterns.append(
        {
            "pattern_type": "action_failure",
            "summary": summary,
            "conditions": _drop_empty(
                {
                    "candidate_id": candidate_id,
                    "candidate_signature": candidate_signature,
                    "target_visible": visual.get("target_visible"),
                    "target_part_visible": visual.get("target_part_visible"),
                    "target_part_name": visual.get("target_part_name"),
                    "switch_visible": visual.get("switch_visible"),
                    "approach_distance_m": distance.get("approach_distance_m"),
                    "handoff_distance_m": distance.get("handoff_distance_m"),
                    "contact_observed": contact.get("contact_observed"),
                    "task_success": environment_outcome.get("task_success"),
                    "vlm_predicate_mismatch": _mismatch_conditions(mismatch),
                }
            ),
            "recommended_response": (
                "For this candidate signature, do not treat VLM success as completion; "
                "verify the simulator predicate and prefer a candidate/action setup that creates contact "
                "with the visible target control."
            ),
            "confidence": 0.8,
            "source_action_ids": source_action_ids,
            "generated_by": "memory_agent_rule_derived",
        }
    )


def _successful_object_approach_prior(
    rule_evidence: dict[str, Any],
    *,
    episode_context: dict[str, Any],
) -> dict[str, Any] | None:
    selection = rule_evidence.get("object_approach_selection")
    selection = selection if isinstance(selection, dict) else {}
    candidate = selection.get("candidate")
    candidate = dict(candidate) if isinstance(candidate, dict) else {}
    if not candidate:
        return None

    signature = candidate.get("candidate_signature")
    if isinstance(signature, dict):
        for key, value in signature.items():
            candidate.setdefault(key, value)
        candidate["candidate_signature"] = dict(signature)

    target_name = (
        candidate.get("object_name")
        or candidate.get("object")
        or candidate.get("target")
        or _episode_target_name(episode_context)
    )
    target = _drop_empty(
        {
            "object": target_name,
            "object_id": candidate.get("object_id"),
            "room_id": candidate.get("room_id"),
            "floor_id": candidate.get("floor_id"),
        }
    )
    if not target:
        return None

    source_action_id = selection.get("source_action_id")
    source_action_ids = [str(source_action_id)] if source_action_id else []
    return {
        "scene_id": episode_context.get("scene_id"),
        "target": target,
        "candidate": candidate,
        "outcome": "success",
        "reason": "post_approach_vla_succeeded",
        "confidence": 0.82,
        "source_action_ids": source_action_ids,
        "generated_by": "memory_agent_rule_derived",
        "summary": "Successful object approach followed by environment predicate success.",
    }


def _completion_criteria_hint(episode_context: dict[str, Any]) -> RetrievalHint | None:
    if not _episode_succeeded(episode_context):
        return None
    final_state = episode_context.get("final_state")
    final_state = final_state if isinstance(final_state, dict) else {}
    interactive = _first_dict(
        final_state.get("interactive_planning"),
        episode_context.get("interactive_planning"),
    )
    if not interactive or interactive.get("status") != "confirmed":
        return None
    confirmation = interactive.get("confirmation")
    if isinstance(confirmation, dict) and confirmation.get("confirmed") is not True:
        return None

    text_plan = interactive.get("text_plan")
    text_plan = text_plan if isinstance(text_plan, dict) else {}
    criteria = [
        _compact_completion_criterion(item)
        for item in text_plan.get("success_criteria") or []
        if isinstance(item, dict) and item.get("user_confirmed") is True
    ]
    criteria = [item for item in criteria if item.get("description")]
    if not criteria:
        return None

    completion_monitor = _first_dict(
        final_state.get("completion_monitor"),
        episode_context.get("completion_monitor"),
    )
    verdict = completion_monitor.get("latest_verdict") if completion_monitor else None
    verdict = verdict if isinstance(verdict, dict) else {}
    if verdict.get("completed") is not True:
        return None

    steps = [
        _drop_empty(
            {
                "step_id": item.get("step_id"),
                "description": item.get("description"),
                "source": item.get("source") or "confirmed_text_plan",
            }
        )
        for item in text_plan.get("steps") or []
        if isinstance(item, dict)
    ]
    return RetrievalHint(
        hint_type="completion_criteria",
        summary=_completion_hint_summary(episode_context, criteria),
        confidence=max(0.75, min(0.95, float(verdict.get("confidence", 0.85) or 0.85))),
        source_action_ids=_successful_action_ids(episode_context),
        generated_by="memory_agent_completion_criteria",
        status="verified",
        content={
            "criteria": criteria,
            "task_steps": steps,
            "completion_verdict": _drop_empty(
                {
                    "completed": verdict.get("completed"),
                    "confidence": verdict.get("confidence"),
                    "source": verdict.get("source"),
                    "reason": verdict.get("reason"),
                }
            ),
            "episode_outcome": episode_context.get("outcome"),
        },
    )


def _episode_succeeded(episode_context: dict[str, Any]) -> bool:
    if str(episode_context.get("outcome") or "").lower() in {"success", "completed"}:
        return True
    final_state = episode_context.get("final_state")
    final_state = final_state if isinstance(final_state, dict) else {}
    environment = final_state.get("environment")
    environment = environment if isinstance(environment, dict) else {}
    return final_state.get("task_success") is True or environment.get("task_success") is True


def _compact_completion_criterion(item: dict[str, Any]) -> dict[str, Any]:
    return _drop_empty(
        {
            "criterion_id": item.get("criterion_id"),
            "scope": item.get("scope") or "task",
            "collaborative_step_id": item.get("collaborative_step_id"),
            "intent": item.get("intent"),
            "description": item.get("description"),
            "positive_evidence": item.get("positive_evidence"),
            "required_observations": item.get("required_observations"),
            "source": item.get("source"),
            "user_confirmed": item.get("user_confirmed"),
            "semantic_anchors": item.get("semantic_anchors"),
            "metadata": item.get("metadata"),
        }
    )


def _completion_hint_summary(
    episode_context: dict[str, Any], criteria: list[dict[str, Any]]
) -> str:
    task = str(episode_context.get("task_description") or "similar tasks").strip()
    description = str(criteria[0].get("description") or "").strip()
    if description:
        return f"For {task}, verified completion criterion: {description}"
    return f"For {task}, reuse the verified user-confirmed completion criteria."


def _successful_action_ids(episode_context: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for key in ("actions", "action_sequence"):
        for item in episode_context.get(key) or []:
            if not isinstance(item, dict) or item.get("success") is False:
                continue
            action_id = item.get("action_id")
            if action_id and str(action_id) not in ids:
                ids.append(str(action_id))
    return ids[:5]


def _first_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def _episode_target_name(episode_context: dict[str, Any]) -> str | None:
    for action in (
        episode_context.get("actions", []),
        episode_context.get("action_sequence", []),
    ):
        if not isinstance(action, list):
            continue
        for item in action:
            if not isinstance(item, dict):
                continue
            target = item.get("target")
            if isinstance(target, str) and target.strip():
                return target.strip()
            if isinstance(target, dict):
                for key in ("object", "object_name", "name", "target"):
                    value = target.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
    target = episode_context.get("target")
    if isinstance(target, str) and target.strip():
        return target.strip()
    if isinstance(target, dict):
        for key in ("object", "object_name", "name", "target"):
            value = target.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _merged_final_environment_outcome(
    environment_outcome: dict[str, Any],
    *,
    episode_context: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(environment_outcome)
    final_state = episode_context.get("final_state")
    final_state = final_state if isinstance(final_state, dict) else {}
    environment = final_state.get("environment")
    environment = environment if isinstance(environment, dict) else {}

    for key in ("task_success", "task_progress", "step_count", "goal_status"):
        value = final_state.get(key)
        if value is None:
            value = environment.get(key)
        if value is None:
            continue
        merged["env_step_count" if key == "step_count" else key] = value
    if environment.get("truncated") is not None:
        merged["environment_truncated"] = bool(environment.get("truncated"))
    return merged


def _rule_derived_evidence(episode_context: dict[str, Any]) -> dict[str, Any]:
    reflection = episode_context.get("reflection_evidence")
    if isinstance(reflection, dict):
        rule_evidence = reflection.get("rule_derived_evidence")
        if isinstance(rule_evidence, dict):
            return rule_evidence
    rule_evidence = episode_context.get("rule_derived_evidence")
    return rule_evidence if isinstance(rule_evidence, dict) else {}


def _drop_empty(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item not in (None, "", [], {})}


def _mismatch_conditions(mismatch: dict[str, Any]) -> dict[str, Any]:
    return _drop_empty(
        {
            "vlm_reported_success": mismatch.get("vlm_reported_success"),
            "success_confirmation_count": mismatch.get("success_confirmation_count"),
            "success_confirmation_threshold": mismatch.get("success_confirmation_threshold"),
            "subtask_completion_reason": mismatch.get("subtask_completion_reason"),
            "environment_task_success": mismatch.get("environment_task_success"),
        }
    )
