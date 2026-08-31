from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlanSuccessCondition:
    description: str
    source: str
    confidence: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "description": self.description,
            "source": self.source,
            "confidence": self.confidence,
        }
        if self.evidence:
            payload["evidence"] = dict(self.evidence)
        return payload


@dataclass
class CollaborativePlanStep:
    step_id: str
    intent: str
    description: str
    target: dict[str, Any] = field(default_factory=dict)
    known_success_conditions: list[PlanSuccessCondition] = field(default_factory=list)
    uncertainties: list[dict[str, Any]] = field(default_factory=list)
    memory_sources: list[dict[str, Any]] = field(default_factory=list)
    semantic_anchors: dict[str, Any] = field(default_factory=dict)
    source_subtask_ids: list[str] = field(default_factory=list)
    role: str = "milestone"
    required: bool = True
    condition: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "step_id": self.step_id,
            "intent": self.intent,
            "description": self.description,
            "role": self.role,
            "required": bool(self.required),
        }
        if self.condition:
            payload["condition"] = self.condition
        if self.target:
            payload["target"] = dict(self.target)
        if self.semantic_anchors:
            payload["semantic_anchors"] = dict(self.semantic_anchors)
        if self.source_subtask_ids:
            payload["source_subtask_ids"] = list(self.source_subtask_ids)
        if self.known_success_conditions:
            payload["known_success_conditions"] = [
                condition.to_dict() for condition in self.known_success_conditions
            ]
        if self.uncertainties:
            payload["uncertainties"] = [dict(item) for item in self.uncertainties]
        if self.memory_sources:
            payload["memory_sources"] = [dict(item) for item in self.memory_sources]
        return payload


@dataclass
class TextPlanDraft:
    task_summary: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    collaborative_steps: list[CollaborativePlanStep] = field(default_factory=list)
    success_criteria: list[dict[str, Any]] = field(default_factory=list)
    uncertainties: list[dict[str, Any]] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    memory_evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        collaborative_steps = [step.to_dict() for step in self.collaborative_steps]
        legacy_steps = [dict(step) for step in self.steps]
        if not legacy_steps:
            legacy_steps = [step.to_dict() for step in self.collaborative_steps]
        return {
            "task_summary": self.task_summary,
            "assumptions": list(self.assumptions),
            "steps": legacy_steps,
            "collaborative_plan": {"steps": collaborative_steps},
            "success_criteria": [dict(item) for item in self.success_criteria],
            "uncertainties": [dict(item) for item in self.uncertainties],
            "memory_evidence": dict(self.memory_evidence),
        }


@dataclass
class ClarificationQuestion:
    question_id: str
    question: str
    reason: str
    applies_to: str
    options: list[str] = field(default_factory=list)
    required: bool = True
    subtask_id: str | None = None
    step_id: str | None = None
    agent: str | None = None
    intent: str | None = None

    def to_dialogue_item(self) -> dict[str, Any]:
        payload = {
            "role": "brain",
            "type": "question",
            "question_id": self.question_id,
            "text": self.question,
            "reason": self.reason,
            "applies_to": self.applies_to,
            "options": list(self.options),
            "required": bool(self.required),
        }
        if self.subtask_id:
            payload["subtask_id"] = self.subtask_id
        if self.step_id:
            payload["step_id"] = self.step_id
        if self.agent:
            payload["agent"] = self.agent
        if self.intent:
            payload["intent"] = self.intent
        return payload


@dataclass
class UserAnswer:
    question_id: str
    answer: str

    def to_dialogue_item(self) -> dict[str, Any]:
        return {
            "role": "user",
            "type": "answer",
            "question_id": self.question_id,
            "text": self.answer,
        }


@dataclass
class PlanConfirmation:
    confirmed: bool
    user_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {"confirmed": bool(self.confirmed)}
        if self.user_message:
            payload["user_message"] = self.user_message
        return payload


@dataclass
class BrainPlanningSession:
    session_id: str
    task_id: str
    user_instruction: str
    draft: TextPlanDraft
    status: str = "drafting"
    dialogue: list[dict[str, Any]] = field(default_factory=list)
    confirmation: PlanConfirmation | None = None

    def add_question(self, question: ClarificationQuestion) -> None:
        self.dialogue.append(question.to_dialogue_item())
        self.status = "clarifying"

    def add_answer(self, answer: UserAnswer) -> None:
        self.dialogue.append(answer.to_dialogue_item())
        self._apply_answer_to_draft(answer)
        self.status = "awaiting_confirmation"

    def unanswered_required_questions(self) -> list[dict[str, Any]]:
        answered = {
            str(item.get("question_id"))
            for item in self.dialogue
            if item.get("type") == "answer" and item.get("question_id")
        }
        return [
            dict(item)
            for item in self.dialogue
            if item.get("type") == "question"
            and item.get("required", True)
            and str(item.get("question_id")) not in answered
        ]

    def set_confirmation(self, confirmation: PlanConfirmation) -> None:
        self.confirmation = confirmation
        self.status = "confirmed" if confirmation.confirmed else "rejected"

    def _apply_answer_to_draft(self, answer: UserAnswer) -> None:
        question = next(
            (
                item
                for item in self.dialogue
                if item.get("type") == "question" and item.get("question_id") == answer.question_id
            ),
            {},
        )
        applies_to = str(question.get("applies_to") or "task")
        subtask_id = question.get("subtask_id")
        collaborative_step_id = question.get("step_id")
        intent = str(question.get("intent") or "").strip()
        if applies_to.startswith("subtask:"):
            _, _, parsed_subtask_id = applies_to.partition(":")
            applies_to = "subtask"
            subtask_id = subtask_id or parsed_subtask_id
        if subtask_id:
            applies_to = "subtask"
        if applies_to == "collaborative_step" and collaborative_step_id:
            applies_to = "collaborative_step"
        criterion = {
            "criterion_id": f"crit_user_{answer.question_id}",
            "scope": applies_to,
            "description": answer.answer,
            "source": "user_clarification",
            "question_id": answer.question_id,
            "user_confirmed": True,
        }
        if subtask_id:
            criterion["subtask_id"] = str(subtask_id)
        if applies_to == "collaborative_step" and collaborative_step_id:
            criterion["collaborative_step_id"] = str(collaborative_step_id)
        if applies_to == "collaborative_step" and intent:
            criterion["intent"] = intent
        collaborative_step = next(
            (
                step
                for step in self.draft.collaborative_steps
                if step.step_id == str(collaborative_step_id or "")
            ),
            None,
        )
        if collaborative_step is not None and collaborative_step.semantic_anchors:
            criterion["semantic_anchors"] = dict(collaborative_step.semantic_anchors)
        metadata = {
            key: question.get(key)
            for key in ("step_id", "agent", "intent")
            if question.get(key) not in (None, "")
        }
        if metadata:
            criterion["metadata"] = metadata
        self._attach_answer_to_collaborative_step(
            collaborative_step_id=str(collaborative_step_id or ""),
            answer=answer,
        )
        self.draft.success_criteria = [
            item
            for item in self.draft.success_criteria
            if item.get("question_id") != answer.question_id
        ]
        self.draft.success_criteria.append(criterion)
        self.draft.uncertainties = [
            item
            for item in self.draft.uncertainties
            if str(item.get("uncertainty_id") or "") != answer.question_id
        ]

    def _attach_answer_to_collaborative_step(
        self,
        *,
        collaborative_step_id: str,
        answer: UserAnswer,
    ) -> None:
        if not collaborative_step_id:
            return
        condition = PlanSuccessCondition(
            description=answer.answer,
            source="user_clarification",
            confidence=1.0,
            evidence={"question_id": answer.question_id},
        )
        for step in self.draft.collaborative_steps:
            if step.step_id != collaborative_step_id:
                continue
            step.known_success_conditions = [
                item
                for item in step.known_success_conditions
                if item.description != condition.description
            ]
            step.known_success_conditions.append(condition)
            break

    def to_task_context(self) -> dict[str, Any]:
        payload = {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "status": self.status,
            "user_instruction": self.user_instruction,
            "dialogue": [dict(item) for item in self.dialogue],
            "text_plan": self.draft.to_dict(),
        }
        if self.confirmation is not None:
            payload["confirmation"] = self.confirmation.to_dict()
        return payload


__all__ = [
    "BrainPlanningSession",
    "ClarificationQuestion",
    "CollaborativePlanStep",
    "PlanConfirmation",
    "PlanSuccessCondition",
    "TextPlanDraft",
    "UserAnswer",
]
