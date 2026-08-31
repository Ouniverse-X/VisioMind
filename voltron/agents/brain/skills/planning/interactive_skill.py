from __future__ import annotations

import re
from typing import Any

from voltron.agents.brain.contracts import (
    CollaborativePlanStep,
    PlanSuccessCondition,
    TextPlanDraft,
)
from voltron.agents.brain.tools import interaction_targeting
from voltron.shared.context import Plan, Subtask
from voltron.shared.enums import AgentName

_VAGUE_TARGET_TEXTS = {
    "it",
    "this",
    "that",
    "something",
    "object",
    "thing",
    "target object",
    "the object",
    "the thing",
    "the target object",
}

_VAGUE_REFERENCE_TOKENS = ("it", "this", "something", "thing")

_RECOGNIZED_ACTION_PHRASES = (
    "pick up",
    "pickup",
    "place",
    "put",
    "open",
    "close",
    "shut",
    "turn on",
    "turn off",
)

_RECOGNIZED_ACTIONS = {"pick_up", "place", "open", "close", "turn_on", "turn_off"}

_IMPLEMENTATION_ACTIONS = {
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

_ANCHOR_HEAD_FOLLOWERS = {
    "and",
    "are",
    "at",
    "be",
    "been",
    "being",
    "closed",
    "for",
    "from",
    "has",
    "have",
    "held",
    "in",
    "inside",
    "is",
    "latched",
    "near",
    "of",
    "on",
    "open",
    "or",
    "outside",
    "placed",
    "present",
    "ready",
    "secure",
    "supported",
    "then",
    "to",
    "under",
    "visible",
    "was",
    "were",
    "with",
    "without",
}

_ANCHOR_PART_SUFFIXES = {
    "button",
    "cover",
    "door",
    "drawer",
    "handle",
    "knob",
    "lever",
    "lid",
    "panel",
    "switch",
}


class BrainInteractivePlanningSkill:
    def __init__(
        self,
        *,
        ask_when_uncertain: bool = True,
        max_questions: int = 5,
        reuse_memory_criteria_min_confidence: float = 0.8,
    ) -> None:
        self.ask_when_uncertain = ask_when_uncertain
        self.max_questions = max(0, int(max_questions))
        self.reuse_memory_criteria_min_confidence = float(reuse_memory_criteria_min_confidence)

    def draft_text_plan(
        self,
        task_description: str,
        planning_context: dict[str, Any],
        *,
        provisional_plan: Plan | None = None,
    ) -> TextPlanDraft:
        if provisional_plan is not None:
            if not provisional_plan.subtasks:
                raise ValueError("Interactive planning requires a non-empty provisional plan")
            memory_hints = self._completion_memories(planning_context)
            selected_memory_hints = [
                self._best_matching_completion_memory(memory_hints, subtask)
                for subtask in provisional_plan.subtasks
            ]
            collaborative_steps = [
                self._collaborative_step_from_subtask(
                    subtask=subtask,
                    index=index,
                    memory_hint=memory_hint,
                    task_description=task_description,
                )
                for index, (subtask, memory_hint) in enumerate(
                    zip(provisional_plan.subtasks, selected_memory_hints),
                    start=1,
                )
            ]
            self._mark_navigation_for_contingencies(collaborative_steps)
            success_criteria = self._criteria_from_steps(collaborative_steps)
            memory_hint = max(
                (hint for hint in selected_memory_hints if hint),
                key=self._confidence,
                default=None,
            )
        else:
            memory_hint = self._best_completion_memory(planning_context)
            success_criteria = self._criteria_from_hint(memory_hint)
            success_conditions = self._success_conditions_from_hint(memory_hint)
            memory_sources = self._memory_sources(memory_hint)
            collaborative_steps = self._collaborative_steps_from_task(
                task_description=task_description,
                success_conditions=success_conditions,
                memory_sources=memory_sources,
            )

        uncertainties: list[dict[str, Any]] = []
        if provisional_plan is not None or (self.ask_when_uncertain and self.max_questions > 0):
            uncertainties.extend(self._execution_success_uncertainties(collaborative_steps))

        return TextPlanDraft(
            task_summary=task_description,
            steps=self._human_outline_steps(collaborative_steps),
            collaborative_steps=collaborative_steps,
            success_criteria=success_criteria,
            uncertainties=uncertainties,
            assumptions=[],
            memory_evidence=self._memory_evidence(memory_hint),
        )

    def _collaborative_step_from_subtask(
        self,
        *,
        subtask: Subtask,
        index: int,
        memory_hint: dict[str, Any] | None,
        task_description: str,
    ) -> CollaborativePlanStep:
        instruction = str(subtask.parameters.get("instruction") or "").strip()
        if not self._instruction_is_grounded(instruction, subtask):
            instruction = ""
        action = self._normalized_action(subtask.action)
        anchors = {
            "agent": subtask.agent.value,
            "action": action,
            "object": self._target_text(subtask.target, "object"),
            "destination": self._target_text(
                subtask.target,
                "destination",
                "receptacle",
                "container",
            ),
            "room": self._target_text(subtask.target, "room", "region", "room_name"),
            "order": index,
        }
        role, required, condition = self._step_contract(
            subtask=subtask,
            action=action,
            task_description=task_description,
        )
        return CollaborativePlanStep(
            step_id=f"step_{index:02d}",
            intent=self._collaborative_intent(subtask),
            description=instruction or self._explicit_subtask_description(subtask),
            target=dict(subtask.target),
            known_success_conditions=self._conditions_for_subtask(subtask, memory_hint),
            memory_sources=self._memory_sources_for_subtask(subtask, memory_hint),
            semantic_anchors={
                key: value for key, value in anchors.items() if value not in (None, "")
            },
            source_subtask_ids=[subtask.subtask_id],
            role=role,
            required=required,
            condition=condition,
        )

    @classmethod
    def _step_contract(
        cls,
        *,
        subtask: Subtask,
        action: str,
        task_description: str,
    ) -> tuple[str, bool, str | None]:
        parameters = subtask.parameters if isinstance(subtask.parameters, dict) else {}
        explicit_role = (
            str(
                parameters.get("outline_role")
                or parameters.get("collaborative_role")
                or parameters.get("step_role")
                or ""
            )
            .strip()
            .lower()
        )
        explicit_condition = str(
            parameters.get("condition")
            or parameters.get("when")
            or parameters.get("precondition")
            or ""
        ).strip()
        if explicit_role in {"milestone", "support", "contingency"}:
            required = bool(parameters.get("required", explicit_role == "milestone"))
            return explicit_role, required, explicit_condition or None
        if bool(parameters.get("conditional")) or explicit_condition:
            return "contingency", False, explicit_condition or "if needed at runtime"
        if subtask.agent == AgentName.VISION:
            return "support", False, None
        if subtask.agent == AgentName.ACTION and action in _IMPLEMENTATION_ACTIONS:
            return "support", False, None
        if (
            subtask.agent == AgentName.ACTION
            and action == "open"
            and not cls._task_explicitly_requests_open(task_description)
        ):
            target = cls._target_text(subtask.target, "object", "target") or "passage"
            return (
                "contingency",
                False,
                f"if the {target} is closed and blocks the planned route",
            )
        return "milestone", True, None

    @classmethod
    def _task_explicitly_requests_open(cls, task_description: str) -> bool:
        normalized = cls._normalized_match_text(task_description)
        return cls._contains_phrase(normalized, "open") or any(
            phrase in task_description for phrase in ("打开", "开启", "开门")
        )

    @staticmethod
    def _mark_navigation_for_contingencies(
        steps: list[CollaborativePlanStep],
    ) -> None:
        for index, step in enumerate(steps):
            if step.role != "contingency" or index <= 0:
                continue
            previous = steps[index - 1]
            if previous.intent != "navigate" or previous.role != "milestone":
                continue
            previous.role = "contingency"
            previous.required = False
            previous.condition = step.condition

    @staticmethod
    def _human_outline_steps(
        steps: list[CollaborativePlanStep],
    ) -> list[dict[str, Any]]:
        outline: list[dict[str, Any]] = []
        for index, step in enumerate(steps):
            if step.role == "support":
                continue
            if (
                step.role == "contingency"
                and step.intent == "navigate"
                and index + 1 < len(steps)
                and steps[index + 1].role == "contingency"
            ):
                continue
            outline.append(step.to_dict())
        return outline

    @staticmethod
    def _normalized_action(action: Any) -> str:
        normalized = str(action or "").strip().lower().replace("-", "_").replace(" ", "_")
        return {
            "pickup": "pick_up",
            "put": "place",
            "shut": "close",
        }.get(normalized, normalized)

    @classmethod
    def _collaborative_intent(cls, subtask: Subtask) -> str:
        action = cls._normalized_action(subtask.action)
        if subtask.agent == AgentName.ACTION:
            return action or "execute"
        if subtask.agent == AgentName.NAVIGATION:
            return "navigate"
        if subtask.agent == AgentName.VISION:
            return "verify" if action in {"verify", "confirm", "check"} else "observe"
        return action or "execute"

    @classmethod
    def _explicit_subtask_description(cls, subtask: Subtask) -> str:
        action = cls._normalized_action(subtask.action).replace("_", " ") or "perform"
        object_name = cls._concrete_target_text(subtask.target, "object", "target")
        part = cls._concrete_target_text(subtask.target, "part")
        control = cls._concrete_target_text(subtask.target, "control")
        destination = cls._concrete_target_text(
            subtask.target,
            "destination",
            "receptacle",
            "container",
        )
        room = cls._concrete_target_text(subtask.target, "room", "region", "room_name")

        description = f"{action.capitalize()}"
        primary_target = object_name or part or control
        if primary_target:
            description += f" the {primary_target}"
        if part and part != primary_target:
            description += f" at the {part}"
        if control and control != primary_target:
            description += f" using the {control}"
        if destination:
            description += f" at the {destination}"
        if room:
            description += f" in the {room}"
        if primary_target or destination or room:
            return f"{description}."
        raise ValueError(
            f"Interactive subtask {subtask.subtask_id} has no concrete instruction or target"
        )

    @staticmethod
    def _target_text(target: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = str(target.get(key) or "").strip()
            if value:
                return value
        return ""

    @classmethod
    def _concrete_target_text(cls, target: dict[str, Any], *keys: str) -> str:
        value = cls._target_text(target, *keys)
        return value if cls._is_concrete_target_text(value) else ""

    @classmethod
    def _is_concrete_target_text(cls, value: Any) -> bool:
        normalized = cls._normalized_match_text(str(value or ""))
        return bool(normalized) and normalized not in _VAGUE_TARGET_TEXTS

    @classmethod
    def _instruction_is_grounded(cls, instruction: str, subtask: Subtask) -> bool:
        normalized_instruction = cls._normalized_match_text(instruction)
        if not normalized_instruction:
            return False
        if any(
            cls._contains_phrase(normalized_instruction, token) for token in _VAGUE_REFERENCE_TOKENS
        ):
            return False
        anchors = cls._structured_target_anchors(subtask.target)
        if not anchors:
            return False
        matches = [
            cls._instruction_mentions_anchor(normalized_instruction, anchor) for anchor in anchors
        ]
        if subtask.agent == AgentName.ACTION:
            return all(matches)
        return any(matches)

    @classmethod
    def _instruction_mentions_anchor(cls, instruction: str, anchor: str) -> bool:
        normalized_anchor = cls._normalized_match_text(anchor)
        anchor_tokens = [token for token in normalized_anchor.split() if not token.isdigit()]
        if not anchor_tokens:
            return False
        anchor_phrase = " ".join(anchor_tokens)
        if cls._contains_phrase(instruction, anchor_phrase):
            return True
        head = anchor_tokens[-1]
        match = re.search(rf"(?<!\w){re.escape(head)}(?!\w)(?:\s+(\w+))?", instruction)
        if match is None:
            return False
        following_word = str(match.group(1) or "").lower()
        return (
            not following_word
            or following_word in _ANCHOR_HEAD_FOLLOWERS
            or following_word in _ANCHOR_PART_SUFFIXES
            or following_word.endswith("ly")
        )

    @classmethod
    def _structured_target_anchors(cls, target: dict[str, Any]) -> list[str]:
        anchors: list[str] = []
        for keys in (
            ("object", "target"),
            ("destination", "receptacle", "container"),
            ("room", "region", "room_name"),
            ("part",),
            ("control",),
        ):
            value = cls._concrete_target_text(target, *keys)
            if value and value not in anchors:
                anchors.append(value)
        return anchors

    def _conditions_for_subtask(
        self,
        subtask: Subtask,
        memory_hint: dict[str, Any] | None,
    ) -> list[PlanSuccessCondition]:
        planner_conditions: list[PlanSuccessCondition] = []
        for item in subtask.parameters.get("completion_criteria") or []:
            if not isinstance(item, dict):
                continue
            description = str(item.get("description") or item.get("summary") or "").strip()
            if not description:
                continue
            evidence = {
                key: value
                for key, value in item.items()
                if key not in {"description", "summary", "source", "confidence"}
            }
            planner_conditions.append(
                PlanSuccessCondition(
                    description=description,
                    source=str(item.get("source") or "planner"),
                    confidence=self._confidence_or_default(item, default=1.0),
                    evidence=evidence,
                )
            )
        if planner_conditions:
            return planner_conditions
        if memory_hint and self._memory_hint_matches_subtask(memory_hint, subtask):
            return self._success_conditions_for_subtask(memory_hint, subtask)
        return []

    def _success_conditions_for_subtask(
        self,
        hint: dict[str, Any],
        subtask: Subtask,
    ) -> list[PlanSuccessCondition]:
        criteria = self._criteria_from_hint(hint)
        if len(criteria) <= 1:
            return self._success_conditions_from_hint(hint)

        conditions: list[PlanSuccessCondition] = []
        for item in criteria:
            criterion_hint = {
                **dict(item),
                "hint_type": "completion_criteria",
                "summary": str(item.get("description") or item.get("summary") or "").strip(),
                "confidence": self._confidence_or_default(
                    item,
                    default=self._confidence(hint),
                ),
            }
            if not self._memory_hint_matches_subtask(criterion_hint, subtask):
                continue
            conditions.extend(self._success_conditions_from_hint(criterion_hint))
        return conditions

    @classmethod
    def _memory_hint_matches_subtask(cls, hint: dict[str, Any], subtask: Subtask) -> bool:
        if subtask.agent not in {AgentName.ACTION, AgentName.VISION}:
            return False
        object_name = cls._target_text(subtask.target, "object")
        destination = cls._target_text(
            subtask.target,
            "destination",
            "receptacle",
            "container",
        )
        if not object_name and not destination:
            return False
        hint_anchors = cls._memory_hint_anchors(hint)
        hint_text = cls._normalized_match_text(cls._completion_hint_text(hint))
        hint_actions = cls._explicit_hint_actions(hint, hint_text)
        if hint_actions and cls._normalized_action(subtask.action) not in hint_actions:
            return False
        return cls._memory_term_matches(
            term=object_name,
            structured_value=cls._target_text(hint_anchors, "object"),
            hint_text=hint_text,
        ) and cls._memory_term_matches(
            term=destination,
            structured_value=cls._target_text(
                hint_anchors,
                "destination",
                "receptacle",
                "container",
            ),
            hint_text=hint_text,
        )

    @classmethod
    def _explicit_hint_actions(cls, hint: dict[str, Any], hint_text: str) -> set[str]:
        content = hint.get("content") if isinstance(hint.get("content"), dict) else {}
        for source in (content, hint):
            for key in ("action", "intent"):
                action = cls._normalized_action(source.get(key))
                if action:
                    return {action}
        return cls._recognized_actions_in_text(hint_text)

    @classmethod
    def _recognized_actions_in_text(cls, text: str) -> set[str]:
        return {
            cls._normalized_action(phrase)
            for phrase in _RECOGNIZED_ACTION_PHRASES
            if cls._contains_phrase(text, phrase)
        }

    @staticmethod
    def _memory_hint_anchors(hint: dict[str, Any]) -> dict[str, Any]:
        content = hint.get("content") if isinstance(hint.get("content"), dict) else {}
        for value in (
            content.get("semantic_anchors"),
            content.get("anchors"),
            hint.get("semantic_anchors"),
            hint.get("anchors"),
        ):
            if isinstance(value, dict):
                return value
        return {}

    @classmethod
    def _memory_term_matches(
        cls,
        *,
        term: str,
        structured_value: str,
        hint_text: str,
    ) -> bool:
        if not term:
            return True
        normalized_term = cls._normalized_match_text(term)
        if structured_value:
            return normalized_term == cls._normalized_match_text(structured_value)
        return cls._contains_phrase(hint_text, normalized_term)

    @staticmethod
    def _contains_phrase(text: str, phrase: str) -> bool:
        return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None

    @classmethod
    def _completion_hint_text(cls, hint: dict[str, Any]) -> str:
        parts = [str(hint.get("summary") or "").strip()]
        content = hint.get("content") if isinstance(hint.get("content"), dict) else {}
        raw_criteria = cls._first_list(
            content.get("criteria"),
            content.get("completion_criteria"),
            hint.get("criteria"),
            hint.get("completion_criteria"),
        )
        for item in raw_criteria:
            if isinstance(item, dict):
                parts.append(str(item.get("description") or item.get("summary") or "").strip())
            elif isinstance(item, str):
                parts.append(item.strip())
        return " ".join(part for part in parts if part)

    @staticmethod
    def _normalized_match_text(value: str) -> str:
        return " ".join(value.lower().replace("_", " ").replace("-", " ").split())

    def _memory_sources_for_subtask(
        self,
        subtask: Subtask,
        memory_hint: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if not memory_hint or not self._memory_hint_matches_subtask(memory_hint, subtask):
            return []
        return self._memory_sources(memory_hint)

    @staticmethod
    def _criteria_from_steps(
        steps: list[CollaborativePlanStep],
    ) -> list[dict[str, Any]]:
        criteria: list[dict[str, Any]] = []
        for step in steps:
            for index, condition in enumerate(step.known_success_conditions, start=1):
                criteria.append(
                    {
                        "criterion_id": f"crit_{step.step_id}_{index:02d}",
                        "scope": "collaborative_step",
                        "collaborative_step_id": step.step_id,
                        "intent": step.intent,
                        **condition.to_dict(),
                    }
                )
        return criteria

    @staticmethod
    def _confidence_or_default(item: dict[str, Any], *, default: float) -> float:
        if item.get("confidence") in (None, ""):
            return default
        try:
            return float(item["confidence"])
        except (TypeError, ValueError):
            return default

    def _completion_memories(self, planning_context: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            item
            for item in self._iter_dicts(planning_context)
            if item.get("hint_type") == "completion_criteria"
            and self._confidence(item) >= self.reuse_memory_criteria_min_confidence
        ]

    def _best_matching_completion_memory(
        self,
        memory_hints: list[dict[str, Any]],
        subtask: Subtask,
    ) -> dict[str, Any] | None:
        candidates = [
            hint for hint in memory_hints if self._memory_hint_matches_subtask(hint, subtask)
        ]
        if not candidates:
            return None
        return max(candidates, key=self._confidence)

    def _best_completion_memory(self, planning_context: dict[str, Any]) -> dict[str, Any] | None:
        candidates = self._completion_memories(planning_context)
        if not candidates:
            return None
        return max(candidates, key=self._confidence)

    def _criteria_from_hint(self, hint: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not hint:
            return []
        content = hint.get("content") if isinstance(hint.get("content"), dict) else {}
        raw_criteria = self._first_list(
            content.get("criteria"),
            content.get("completion_criteria"),
            hint.get("criteria"),
            hint.get("completion_criteria"),
        )
        criteria = [dict(item) for item in raw_criteria if isinstance(item, dict)]
        if criteria:
            for index, item in enumerate(criteria, start=1):
                item.setdefault("criterion_id", f"crit_memory_{index:02d}")
                item.setdefault("source", "memory")
            return criteria

        summary = str(hint.get("summary") or "").strip()
        if not summary:
            return []
        return [
            {
                "criterion_id": "crit_memory_01",
                "description": summary,
                "source": "memory",
                "confidence": self._confidence(hint),
            }
        ]

    def _success_conditions_from_hint(
        self, hint: dict[str, Any] | None
    ) -> list[PlanSuccessCondition]:
        conditions: list[PlanSuccessCondition] = []
        for item in self._criteria_from_hint(hint):
            description = str(item.get("description") or item.get("summary") or "").strip()
            if not description:
                continue
            evidence = {
                key: value
                for key, value in item.items()
                if key not in {"description", "summary", "source", "confidence"}
            }
            conditions.append(
                PlanSuccessCondition(
                    description=description,
                    source=str(item.get("source") or "memory"),
                    confidence=self._confidence(item) or self._confidence(hint or {}),
                    evidence=evidence,
                )
            )
        return conditions

    def _steps_from_hint(self, hint: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not hint:
            return []
        content = hint.get("content") if isinstance(hint.get("content"), dict) else {}
        raw_steps = self._first_list(
            content.get("steps"),
            content.get("task_steps"),
            content.get("plan_steps"),
            hint.get("steps"),
        )
        steps = [dict(item) for item in raw_steps if isinstance(item, dict)]
        for index, item in enumerate(steps, start=1):
            item.setdefault("step_id", f"step_memory_{index:02d}")
            item.setdefault("source", "memory")
        return steps

    def _collaborative_steps_from_task(
        self,
        *,
        task_description: str,
        success_conditions: list[PlanSuccessCondition],
        memory_sources: list[dict[str, Any]],
    ) -> list[CollaborativePlanStep]:
        room = interaction_targeting.infer_room_from_text(task_description)
        target_object = interaction_targeting.infer_object_from_text(task_description)
        steps: list[CollaborativePlanStep] = []
        if room:
            target: dict[str, Any] = {"room": room}
            if target_object:
                target["object"] = target_object
            steps.append(
                CollaborativePlanStep(
                    step_id="step_01",
                    intent="navigate",
                    description=self._navigation_step_description(
                        room=room,
                        target_object=target_object,
                    ),
                    target=target,
                )
            )
        if target_object:
            steps.append(
                CollaborativePlanStep(
                    step_id=f"step_{len(steps) + 1:02d}",
                    intent="execute",
                    description=self._action_step_description(
                        task_description=task_description,
                        target_object=target_object,
                    ),
                    target={"object": target_object},
                    known_success_conditions=list(success_conditions),
                    memory_sources=[dict(item) for item in memory_sources],
                )
            )
            steps.append(
                CollaborativePlanStep(
                    step_id=f"step_{len(steps) + 1:02d}",
                    intent="verify",
                    description=f"Observe the {target_object} state and confirm the action result.",
                    target={"object": target_object},
                    known_success_conditions=list(success_conditions),
                    memory_sources=[dict(item) for item in memory_sources],
                )
            )
        if steps:
            return steps
        return [
            CollaborativePlanStep(
                step_id="step_01",
                intent="execute",
                description=task_description,
                known_success_conditions=list(success_conditions),
                memory_sources=[dict(item) for item in memory_sources],
            )
        ]

    def _execution_success_uncertainties(
        self,
        steps: list[CollaborativePlanStep],
    ) -> list[dict[str, Any]]:
        uncertainties: list[dict[str, Any]] = []
        for step in steps:
            agent = str(step.semantic_anchors.get("agent") or "").upper()
            requires_condition = (
                step.role == "milestone"
                and step.required
                and (agent == AgentName.ACTION.value or (not agent and step.intent == "execute"))
            )
            if not requires_condition or step.known_success_conditions:
                continue
            uncertainties.append(
                {
                    "uncertainty_id": f"success_condition_{step.step_id}",
                    "question": (
                        f"Step '{step.description}' changes the world state. "
                        "What observable condition should confirm it is complete?"
                    ),
                    "reason": "No high-confidence completion criteria were found for this execution step.",
                    "applies_to": "collaborative_step",
                    "step_id": step.step_id,
                    "intent": step.intent,
                    "options": [],
                    "required": True,
                }
            )
        return uncertainties

    @staticmethod
    def _navigation_step_description(*, room: str, target_object: str | None) -> str:
        if target_object:
            return f"Navigate to the {room} and stop near the {target_object} for interaction."
        return f"Navigate to the {room}."

    @staticmethod
    def _action_step_description(*, task_description: str, target_object: str) -> str:
        lowered = task_description.lower()
        if any(phrase in lowered for phrase in ("turn on", "switch on", "打开", "开启")):
            return f"Turn on the {target_object} using its visible control."
        if any(phrase in lowered for phrase in ("turn off", "switch off", "关闭")):
            return f"Turn off the {target_object} using its visible control."
        return f"Manipulate the {target_object} to satisfy the user instruction."

    def _memory_evidence(self, hint: dict[str, Any] | None) -> dict[str, Any]:
        if not hint:
            return {"completion_criteria_reused": False}
        return {
            "completion_criteria_reused": True,
            "hint_type": hint.get("hint_type"),
            "summary": hint.get("summary"),
            "confidence": self._confidence(hint),
        }

    def _memory_sources(self, hint: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not hint:
            return []
        source = {
            "hint_type": hint.get("hint_type"),
            "summary": hint.get("summary"),
            "confidence": self._confidence(hint),
        }
        return [{key: value for key, value in source.items() if value not in (None, "", [], {})}]

    @classmethod
    def _iter_dicts(cls, value: Any):
        if isinstance(value, dict):
            yield value
            for child in value.values():
                yield from cls._iter_dicts(child)
        elif isinstance(value, list):
            for child in value:
                yield from cls._iter_dicts(child)

    @staticmethod
    def _first_list(*values: Any) -> list[Any]:
        for value in values:
            if isinstance(value, list):
                return value
        return []

    @staticmethod
    def _confidence(item: dict[str, Any]) -> float:
        try:
            return float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            return 0.0


__all__ = ["BrainInteractivePlanningSkill"]
