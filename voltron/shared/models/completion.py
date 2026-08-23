"""Shared completion-evaluation models for task, subtask, and internal-step gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _drop_empty(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value not in (None, "", [], {})}


@dataclass
class CompletionCriterion:
    """A structured condition that can prove one task scope is complete."""

    criterion_id: str
    scope: str
    description: str
    positive_evidence: list[str] = field(default_factory=list)
    negative_evidence: list[str] = field(default_factory=list)
    required_observations: list[str] = field(default_factory=list)
    subtask_id: str | None = None
    internal_step_id: str | None = None
    success_verifier: str = "vision"
    confidence_threshold: float = 0.75
    positive_streak: int = 3
    stability_steps: int = 5
    allow_environment_signal: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _drop_empty(
            {
                "criterion_id": self.criterion_id,
                "scope": self.scope,
                "description": self.description,
                "positive_evidence": list(self.positive_evidence),
                "negative_evidence": list(self.negative_evidence),
                "required_observations": list(self.required_observations),
                "subtask_id": self.subtask_id,
                "internal_step_id": self.internal_step_id,
                "success_verifier": self.success_verifier,
                "confidence_threshold": float(self.confidence_threshold),
                "positive_streak": int(self.positive_streak),
                "stability_steps": int(self.stability_steps),
                "allow_environment_signal": bool(self.allow_environment_signal),
                "metadata": dict(self.metadata),
            }
        )

    @classmethod
    def from_value(cls, value: Any) -> "CompletionCriterion":
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError("CompletionCriterion requires a mapping")
        return cls(
            criterion_id=str(value.get("criterion_id") or ""),
            scope=str(value.get("scope") or "subtask"),
            description=str(value.get("description") or ""),
            positive_evidence=[str(item) for item in value.get("positive_evidence") or []],
            negative_evidence=[str(item) for item in value.get("negative_evidence") or []],
            required_observations=[str(item) for item in value.get("required_observations") or []],
            subtask_id=value.get("subtask_id"),
            internal_step_id=value.get("internal_step_id"),
            success_verifier=str(value.get("success_verifier") or "vision"),
            confidence_threshold=float(value.get("confidence_threshold", 0.75)),
            positive_streak=int(value.get("positive_streak", 3)),
            stability_steps=int(value.get("stability_steps", 5)),
            allow_environment_signal=bool(value.get("allow_environment_signal", True)),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass
class CompletionVerdict:
    """Result of evaluating whether a completion scope can advance."""

    scope: str
    scope_id: str
    completed: bool
    confidence: float
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)
    missing_evidence: list[str] = field(default_factory=list)
    should_continue: bool = True
    should_replan: bool = False
    source: str = "vision_completion_evaluator"

    @property
    def scope_key(self) -> str:
        return f"{self.scope}:{self.scope_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "scope_id": self.scope_id,
            "scope_key": self.scope_key,
            "completed": bool(self.completed),
            "confidence": float(self.confidence),
            "reason": self.reason,
            "evidence": dict(self.evidence),
            "missing_evidence": list(self.missing_evidence),
            "should_continue": bool(self.should_continue),
            "should_replan": bool(self.should_replan),
            "source": self.source,
        }


@dataclass
class CompletionEvaluationContext:
    """Bounded prompt payload for Vision-backed completion decisions."""

    task_description: str
    scope: str
    scope_id: str
    completion_criteria: list[CompletionCriterion] = field(default_factory=list)
    confirmed_text_plan: dict[str, Any] = field(default_factory=dict)
    current_subtask: dict[str, Any] = field(default_factory=dict)
    current_internal_step: dict[str, Any] = field(default_factory=dict)
    runtime_feedback: dict[str, Any] = field(default_factory=dict)
    task_context: dict[str, Any] = field(default_factory=dict)
    recent_observations: list[dict[str, Any]] = field(default_factory=list)
    memory_evidence: dict[str, Any] = field(default_factory=dict)
    action_stability: dict[str, Any] = field(default_factory=dict)

    @property
    def scope_key(self) -> str:
        return f"{self.scope}:{self.scope_id}"

    def to_prompt_payload(self) -> dict[str, Any]:
        return _drop_empty(
            {
                "task_description": self.task_description,
                "scope": self.scope,
                "scope_id": self.scope_id,
                "scope_key": self.scope_key,
                "completion_criteria": [criterion.to_dict() for criterion in self.completion_criteria],
                "confirmed_text_plan": dict(self.confirmed_text_plan),
                "current_subtask": dict(self.current_subtask),
                "current_internal_step": dict(self.current_internal_step),
                "runtime_feedback": dict(self.runtime_feedback),
                "task_context": dict(self.task_context),
                "recent_observations": [dict(item) for item in self.recent_observations],
                "memory_evidence": dict(self.memory_evidence),
                "action_stability": dict(self.action_stability),
            }
        )


__all__ = ["CompletionCriterion", "CompletionEvaluationContext", "CompletionVerdict"]
