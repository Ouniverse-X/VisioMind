from __future__ import annotations

import re
from typing import Any

from voltron.agents.brain.contracts import (
    CollaborativePlanStep,
    PlanSuccessCondition,
    TextPlanDraft,
)
from voltron.shared.context import Plan, Subtask
from voltron.shared.enums import AgentName

_ACTION_ALIASES = {
    "pickup": "pick_up",
    "pick_up": "pick_up",
    "pick": "pick_up",
    "put": "place",
    "place_inside": "place",
    "put_inside": "place",
    "shut": "close",
    "turn_on": "turn_on",
    "turn_off": "turn_off",
}


_SUPPORTING_ACTIONS = {
    "approach",
    "align",
    "reposition",
    "reach",
    "pregrasp",
    "grasp",
    "lift",
    "release",
    "withdraw",
}


_TERMINAL_REFINEMENTS = {"pick_up": {"lift"}, "place": {"release"}}


def confirmed_action_steps(
    text_plan: TextPlanDraft | dict[str, Any],
) -> list[CollaborativePlanStep]:
    return [
        step
        for step in action_contract_steps(text_plan)
        if _step_role(step) == "milestone" and step.required
    ]


def action_contract_steps(
    text_plan: TextPlanDraft | dict[str, Any],
) -> list[CollaborativePlanStep]:
    steps: list[CollaborativePlanStep] = []
    for raw_step in _collaborative_steps(text_plan):
        step = (
            raw_step if isinstance(raw_step, CollaborativePlanStep) else _step_from_dict(raw_step)
        )
        if str(step.semantic_anchors.get("agent") or "").strip().upper() == "ACTION":
            steps.append(step)
    return steps


def align_refined_plan(confirmed_steps: list[CollaborativePlanStep], refined_plan: Plan) -> Plan:
    subtasks = [_copy_subtask_without_step_id(subtask) for subtask in refined_plan.subtasks]
    action_contract = list(confirmed_steps)
    milestone_steps = [
        step for step in action_contract if _step_role(step) == "milestone" and step.required
    ]
    optional_steps = [step for step in action_contract if step not in milestone_steps]
    explicit_groups = _explicit_groups(refined_plan.subtasks, action_contract)
    matched_indexes: set[int] = set()
    final_indexes: dict[str, int] = {}
    previous_index = -1
    previous_step: CollaborativePlanStep | None = None

    for step in milestone_steps:
        group = explicit_groups.get(step.step_id)
        if group:
            if min(group) <= previous_index and previous_step is not None:
                raise ValueError(
                    "Explicit Action descendants are interleaved: "
                    f"{previous_step.step_id} must complete before {step.step_id} starts"
                )
            _validate_explicit_group(step, group, refined_plan.subtasks)
            final_index = _state_establishing_descendant_index(
                step,
                group,
                refined_plan.subtasks,
            )
            matched_indexes.update(group)
        else:
            final_index = _find_greedy_match(
                step=step,
                subtasks=refined_plan.subtasks,
                start_index=previous_index + 1,
                excluded_indexes=matched_indexes,
            )
            if final_index is None:
                earlier_index = _find_greedy_match(
                    step=step,
                    subtasks=refined_plan.subtasks,
                    start_index=0,
                    excluded_indexes=matched_indexes,
                    end_index=previous_index + 1,
                )
                if earlier_index is not None and previous_step is not None:
                    raise ValueError(
                        "Confirmed Action stages are reordered: "
                        f"{previous_step.step_id} must precede {step.step_id}"
                    )
                raise ValueError(
                    f"Missing confirmed Action stage {step.step_id}: {step.description}"
                )
            matched_indexes.add(final_index)

        if final_index <= previous_index and previous_step is not None:
            raise ValueError(
                "Confirmed Action stages are reordered: "
                f"{previous_step.step_id} must precede {step.step_id}"
            )
        final_indexes[step.step_id] = final_index
        previous_index = final_index
        previous_step = step

    for step in optional_steps:
        group = explicit_groups.get(step.step_id)
        if group:
            _validate_explicit_group(step, group, refined_plan.subtasks)
            final_index = _state_establishing_descendant_index(
                step,
                group,
                refined_plan.subtasks,
            )
            matched_indexes.update(group)
        else:
            final_index = _find_greedy_match(
                step=step,
                subtasks=refined_plan.subtasks,
                start_index=0,
                excluded_indexes=matched_indexes,
            )
            if final_index is None:
                continue
            matched_indexes.add(final_index)
        final_indexes[step.step_id] = final_index

    for index, subtask in enumerate(refined_plan.subtasks):
        if subtask.agent != AgentName.ACTION or index in matched_indexes:
            continue
        if _is_authorized_supporting_action(
            subtask=subtask,
            action_contract=action_contract,
        ):
            continue
        raise ValueError(
            f"Refined plan has unconfirmed ACTION {_canonical_action(subtask.action)!r}"
        )

    for step_id, index in final_indexes.items():
        subtasks[index].parameters["collaborative_step_id"] = step_id
    return Plan(subtasks=subtasks, metadata=dict(refined_plan.metadata))


def _collaborative_steps(
    text_plan: TextPlanDraft | dict[str, Any],
) -> list[CollaborativePlanStep | dict[str, Any]]:
    if isinstance(text_plan, TextPlanDraft):
        return list(text_plan.collaborative_steps)
    if not isinstance(text_plan, dict):
        return []
    collaborative_plan = text_plan.get("collaborative_plan")
    if isinstance(collaborative_plan, dict) and isinstance(collaborative_plan.get("steps"), list):
        return list(collaborative_plan["steps"])
    steps = text_plan.get("collaborative_steps") or text_plan.get("steps") or []
    return list(steps) if isinstance(steps, list) else []


def _step_from_dict(raw_step: dict[str, Any]) -> CollaborativePlanStep:
    anchors = raw_step.get("semantic_anchors")
    return CollaborativePlanStep(
        step_id=str(raw_step.get("step_id") or ""),
        intent=str(raw_step.get("intent") or ""),
        description=str(raw_step.get("description") or ""),
        target=dict(raw_step.get("target") or {}),
        semantic_anchors=dict(anchors) if isinstance(anchors, dict) else {},
        source_subtask_ids=list(raw_step.get("source_subtask_ids") or []),
        role=str(raw_step.get("role") or "milestone"),
        required=bool(raw_step.get("required", True)),
        condition=(
            str(raw_step.get("condition")).strip()
            if raw_step.get("condition") not in (None, "")
            else None
        ),
        known_success_conditions=[
            PlanSuccessCondition(
                description=str(condition.get("description") or ""),
                source=str(condition.get("source") or ""),
                confidence=float(condition.get("confidence") or 0.0),
                evidence=dict(condition.get("evidence") or {}),
            )
            for condition in raw_step.get("known_success_conditions") or []
            if isinstance(condition, dict)
        ],
    )


def _explicit_groups(
    subtasks: list[Subtask],
    confirmed_steps: list[CollaborativePlanStep],
) -> dict[str, list[int]]:
    confirmed_ids = {step.step_id for step in confirmed_steps}
    groups: dict[str, list[int]] = {}
    for index, subtask in enumerate(subtasks):
        if subtask.agent != AgentName.ACTION:
            continue
        step_id = str(subtask.parameters.get("collaborative_step_id") or "").strip()
        if step_id in confirmed_ids:
            groups.setdefault(step_id, []).append(index)
    return groups


def _find_greedy_match(
    *,
    step: CollaborativePlanStep,
    subtasks: list[Subtask],
    start_index: int,
    excluded_indexes: set[int],
    end_index: int | None = None,
) -> int | None:
    upper_bound = len(subtasks) if end_index is None else min(end_index, len(subtasks))
    for index in range(max(0, start_index), upper_bound):
        if index in excluded_indexes:
            continue
        subtask = subtasks[index]
        if subtask.agent != AgentName.ACTION:
            continue
        if _matches_confirmed_step(step, subtask):
            return index
    return None


def _matches_confirmed_step(step: CollaborativePlanStep, subtask: Subtask) -> bool:
    return _confirmed_action(step) == _canonical_action(subtask.action) and _target_anchors_match(
        step, subtask
    )


def _validate_explicit_group(
    step: CollaborativePlanStep,
    group: list[int],
    subtasks: list[Subtask],
) -> None:
    confirmed_action = _confirmed_action(step)
    descendant_actions: list[str] = []
    for index in group:
        subtask = subtasks[index]
        mismatch = _target_anchor_mismatch(step, subtask)
        if mismatch is not None:
            anchor, expected, actual = mismatch
            raise ValueError(
                f"Explicit descendant for {step.step_id} has mismatched {anchor}: "
                f"expected {expected!r}, got {actual!r}"
            )
        action = _canonical_action(subtask.action)
        if action not in _SUPPORTING_ACTIONS and action != confirmed_action:
            raise ValueError(
                f"Explicit descendant for {step.step_id} must use {confirmed_action!r}, got {action!r}"
            )
        descendant_actions.append(action)

    terminal_actions = _TERMINAL_REFINEMENTS.get(confirmed_action, set())
    if confirmed_action not in descendant_actions and not terminal_actions.intersection(
        descendant_actions
    ):
        raise ValueError(
            f"Explicit descendants for {step.step_id} do not establish {confirmed_action!r}"
        )


def _state_establishing_descendant_index(
    step: CollaborativePlanStep,
    group: list[int],
    subtasks: list[Subtask],
) -> int:
    confirmed_action = _confirmed_action(step)
    exact_matches = [
        index for index in group if _canonical_action(subtasks[index].action) == confirmed_action
    ]
    terminal_actions = _TERMINAL_REFINEMENTS.get(confirmed_action, set())
    terminal_matches = [
        index for index in group if _canonical_action(subtasks[index].action) in terminal_actions
    ]
    state_establishing_matches = [*exact_matches, *terminal_matches]
    if state_establishing_matches:
        return max(state_establishing_matches)
    raise ValueError(
        f"Explicit descendants for {step.step_id} do not establish {confirmed_action!r}"
    )


def _target_anchors_match(step: CollaborativePlanStep, subtask: Subtask) -> bool:
    return _target_anchor_mismatch(step, subtask) is None


def _target_anchor_mismatch(
    step: CollaborativePlanStep,
    subtask: Subtask,
) -> tuple[str, str, str] | None:
    for anchor, keys in _TARGET_ANCHORS:
        expected = _step_anchor(step, anchor, keys)
        if not expected:
            continue
        actual = _target_value(subtask.target, keys)
        if _normalized_target(expected) != _normalized_target(actual):
            return anchor, expected, actual
    return None


_TARGET_ANCHORS = (
    ("object", ("object", "target")),
    ("destination", ("destination", "receptacle", "container")),
    ("part", ("part",)),
    ("control", ("control",)),
    ("room", ("room", "region", "room_name")),
)


def _step_anchor(step: CollaborativePlanStep, anchor: str, keys: tuple[str, ...]) -> str:
    for key in (anchor, *keys):
        value = step.semantic_anchors.get(key)
        if str(value or "").strip():
            return str(value).strip()
    return _target_value(step.target, keys)


def _target_value(target: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = target.get(key)
        if str(value or "").strip():
            return str(value).strip()
    return ""


def _confirmed_action(step: CollaborativePlanStep) -> str:
    return _canonical_action(step.semantic_anchors.get("action") or step.intent)


def _step_role(step: CollaborativePlanStep) -> str:
    normalized = str(step.role or "milestone").strip().lower()
    return normalized if normalized in {"milestone", "support", "contingency"} else "milestone"


def _is_authorized_supporting_action(
    *,
    subtask: Subtask,
    action_contract: list[CollaborativePlanStep],
) -> bool:
    if _canonical_action(subtask.action) not in _SUPPORTING_ACTIONS:
        return False
    return any(_target_anchors_match(step, subtask) for step in action_contract)


def _copy_subtask_without_step_id(subtask: Subtask) -> Subtask:
    parameters = dict(subtask.parameters)
    parameters.pop("collaborative_step_id", None)
    return Subtask(
        subtask_id=subtask.subtask_id,
        agent=subtask.agent,
        action=subtask.action,
        target=dict(subtask.target),
        parameters=parameters,
        context=dict(subtask.context),
    )


def _canonical_action(action: Any) -> str:
    normalized = re.sub(r"[\s-]+", "_", str(action or "").strip().lower())
    return _ACTION_ALIASES.get(normalized, normalized)


def _normalized_target(value: Any) -> str:
    return re.sub(r"[\s_-]+", " ", str(value or "").strip().lower())
