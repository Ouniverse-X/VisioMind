from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any

from visiomind.action.agents.action.contracts import VLASkill
from visiomind.action.agents.action.models import VLADeliberation, VLATargetRefinement
from visiomind.action.shared.context import ExecutionContext, LocalSkillSelection, Subtask
from visiomind.action.shared.results import AgentResult


def normalize_selection(selection: LocalSkillSelection, skill: VLASkill) -> LocalSkillSelection:
    if selection.skill_id == skill.skill_id:
        return selection
    return LocalSkillSelection(
        skill_id=skill.skill_id,
        confidence=selection.confidence,
        reason=selection.reason,
        source=selection.source,
        fallback_skill_candidates=selection.fallback_skill_candidates,
        metadata=selection.metadata,
    )


def build_selection_cache_key(
    *,
    subtask: Subtask,
    context: ExecutionContext,
    available_skill_ids: list[str],
) -> str:
    selector_parameters = {
        "instruction": subtask.parameters.get("instruction"),
        "control_mode": subtask.parameters.get("control_mode"),
        "policy_options": subtask.parameters.get("policy_options"),
    }
    payload: dict[str, Any] = {
        "task_id": context.task_request.task_id,
        "task_description": context.task_request.description,
        "task_type": context.task_request.task_type.value,
        "subtask_id": subtask.subtask_id,
        "action": subtask.action,
        "target": subtask.target,
        "parameters": selector_parameters,
        "context": subtask.context,
        "available_skill_ids": available_skill_ids,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def deliberate_with_cache(
    *,
    deliberator: Any,
    cache: dict[str, VLADeliberation],
    subtask: Subtask,
    context: ExecutionContext,
) -> VLADeliberation:
    cache_key = build_selection_cache_key(
        subtask=subtask,
        context=context,
        available_skill_ids=["refine_target"],
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    try:
        deliberation = deliberator.deliberate(subtask=subtask, context=context)
    except Exception as exc:
        deliberation = VLADeliberation(
            use_tool=False,
            tool_name=None,
            reason=f"fallback after vla deliberator error: {exc}",
            source="fallback",
            metadata={"deliberator_error": str(exc)},
        )
    cache[cache_key] = deliberation
    return deliberation


def refine_target_with_cache(
    *,
    target_refiner: Any,
    cache: dict[str, VLATargetRefinement],
    subtask: Subtask,
    context: ExecutionContext,
) -> VLATargetRefinement:
    cache_key = build_selection_cache_key(
        subtask=subtask,
        context=context,
        available_skill_ids=["refine_target"],
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    refinement = target_refiner.refine_target(subtask=subtask, context=context)
    cache[cache_key] = refinement
    return refinement


def serialize_target_refinement(target_refinement: VLATargetRefinement) -> dict[str, Any]:
    return {
        "refined_instruction": target_refinement.refined_instruction,
        "refined_target": dict(target_refinement.refined_target),
        "selector_hints": dict(target_refinement.selector_hints),
        "policy_hints": dict(target_refinement.policy_hints),
        "success_cues": list(target_refinement.success_cues),
        "metadata": dict(target_refinement.metadata),
    }


def apply_target_refinement(
    *,
    subtask: Subtask,
    deliberation: VLADeliberation,
    target_refinement: VLATargetRefinement,
) -> Subtask:
    refined_target = dict(subtask.target)
    refined_target.update(target_refinement.refined_target)

    refined_parameters = dict(subtask.parameters)
    if target_refinement.refined_instruction:
        refined_parameters["instruction"] = target_refinement.refined_instruction

    selector_hints = {}
    selector_hints.update(deliberation.selector_hints)
    selector_hints.update(target_refinement.selector_hints)
    if selector_hints:
        refined_parameters["selector_hints"] = selector_hints

    control_mode = target_refinement.policy_hints.get(
        "control_mode"
    ) or deliberation.policy_hints.get("control_mode")
    if isinstance(control_mode, str) and control_mode.strip():
        refined_parameters["control_mode"] = control_mode.strip()

    refined_context = dict(subtask.context)
    refined_context["vla_deliberation"] = asdict(deliberation)
    refined_context["target_refinement"] = serialize_target_refinement(target_refinement)
    return Subtask(
        subtask_id=subtask.subtask_id,
        agent=subtask.agent,
        action=subtask.action,
        target=refined_target,
        parameters=refined_parameters,
        context=refined_context,
    )


def decorate_skill_result(
    *,
    result: AgentResult,
    deliberation: VLADeliberation,
    target_refinement: VLATargetRefinement,
) -> AgentResult:
    result.result["deliberation_source"] = deliberation.source
    result.runtime_artifacts["vla_deliberation"] = asdict(deliberation)
    if target_refinement != VLATargetRefinement():
        result.runtime_artifacts["target_refinement"] = serialize_target_refinement(
            target_refinement
        )
    return result
