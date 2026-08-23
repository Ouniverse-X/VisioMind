"""Structured contracts for MemoryAgent experience extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


@dataclass
class RetrievalHint:
    """A lightweight hint that can be written to retrievable memory."""

    hint_type: str = "retrieval"
    summary: str = ""
    confidence: float = 0.0
    source_episode_id: str = ""
    source_action_ids: list[str] = field(default_factory=list)
    generated_by: str = "memory_agent"
    status: str = "candidate"
    content: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperienceExtractionResult:
    """Extractor output after schema parsing and field-level validation."""

    episode_summary: str = ""
    task_outcome: str = ""
    procedural_skills: list[dict[str, Any]] = field(default_factory=list)
    failure_patterns: list[dict[str, Any]] = field(default_factory=list)
    causal_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    semantic_updates: list[dict[str, Any]] = field(default_factory=list)
    retrieval_hints: list[RetrievalHint | dict[str, Any]] = field(default_factory=list)
    object_approach_priors: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    source_episode_id: str = ""
    validation_warnings: list[str] = field(default_factory=list)

    def to_annotation(self) -> dict[str, Any]:
        payload = _to_plain_dict(self)
        payload["confidence"] = _clamp_confidence(payload.get("confidence", 0.0))
        return payload


def normalize_extraction_result(value: ExperienceExtractionResult | dict[str, Any]) -> ExperienceExtractionResult:
    if isinstance(value, ExperienceExtractionResult):
        return value
    if not isinstance(value, dict):
        raise TypeError(f"Unsupported extraction result type: {type(value)!r}")
    hints = [normalize_retrieval_hint(item) for item in value.get("retrieval_hints", [])]
    return ExperienceExtractionResult(
        episode_summary=str(value.get("episode_summary", "")),
        task_outcome=str(value.get("task_outcome", "")),
        procedural_skills=list(value.get("procedural_skills", [])),
        failure_patterns=list(value.get("failure_patterns", [])),
        causal_hypotheses=list(value.get("causal_hypotheses", [])),
        semantic_updates=list(value.get("semantic_updates", [])),
        retrieval_hints=hints,
        object_approach_priors=list(value.get("object_approach_priors", [])),
        confidence=_clamp_confidence(value.get("confidence", 0.0)),
        source_episode_id=str(value.get("source_episode_id", "")),
        validation_warnings=list(value.get("validation_warnings", [])),
    )


def normalize_retrieval_hint(value: RetrievalHint | dict[str, Any]) -> RetrievalHint:
    if isinstance(value, RetrievalHint):
        value.confidence = _clamp_confidence(value.confidence)
        return value
    return RetrievalHint(
        hint_type=str(value.get("hint_type", "retrieval")),
        summary=str(value.get("summary", "")),
        confidence=_clamp_confidence(value.get("confidence", 0.0)),
        source_episode_id=str(value.get("source_episode_id", "")),
        source_action_ids=list(value.get("source_action_ids", [])),
        generated_by=str(value.get("generated_by", "memory_agent")),
        status=str(value.get("status", "candidate")),
        content=dict(value.get("content", {})),
    )


def hint_to_payload(
    hint: RetrievalHint | dict[str, Any],
    *,
    task_description: str,
    task_type: str | None,
    source_episode_id: str,
    generated_by: str = "memory_agent",
) -> dict[str, Any]:
    normalized = normalize_retrieval_hint(hint)
    payload = _to_plain_dict(normalized)
    payload["confidence"] = _clamp_confidence(payload.get("confidence", 0.0))
    payload.setdefault("task_description", task_description)
    payload.setdefault("task_type", task_type)
    payload["task_description"] = payload.get("task_description") or task_description
    payload["task_type"] = payload.get("task_type") or task_type
    payload["source_episode_id"] = payload.get("source_episode_id") or source_episode_id
    payload["generated_by"] = payload.get("generated_by") or generated_by
    return payload


def failure_pattern_to_payload(
    value: dict[str, Any],
    *,
    task_description: str,
    task_type: str | None,
    source_episode_id: str,
    generated_by: str = "memory_agent",
) -> dict[str, Any]:
    payload = dict(value)
    payload["confidence"] = _clamp_confidence(payload.get("confidence", 0.0))
    payload["task_description"] = payload.get("task_description") or task_description
    payload["task_type"] = payload.get("task_type") or task_type
    payload["source_episode_id"] = payload.get("source_episode_id") or source_episode_id
    payload["source_action_ids"] = list(payload.get("source_action_ids", []))
    payload["generated_by"] = payload.get("generated_by") or generated_by
    payload["status"] = payload.get("status") or "candidate"
    payload["conditions"] = dict(payload.get("conditions", {}))
    return payload


def semantic_update_to_payload(
    value: dict[str, Any],
    *,
    source_episode_id: str,
    generated_by: str = "memory_agent",
) -> dict[str, Any]:
    payload = dict(value)
    payload["confidence"] = _clamp_confidence(payload.get("confidence", 0.0))
    payload["source_episode_id"] = payload.get("source_episode_id") or source_episode_id
    payload["source_action_ids"] = list(payload.get("source_action_ids", []))
    payload["generated_by"] = payload.get("generated_by") or generated_by
    payload["status"] = payload.get("status") or "candidate"
    payload["content"] = dict(payload.get("content", {}))
    return payload


def skill_candidate_to_payload(
    value: dict[str, Any],
    *,
    source_episode_id: str,
    generated_by: str = "memory_agent",
) -> dict[str, Any]:
    payload = dict(value)
    payload["confidence"] = _clamp_confidence(payload.get("confidence", 0.0))
    payload["source_episode_id"] = payload.get("source_episode_id") or source_episode_id
    payload["source_action_ids"] = list(payload.get("source_action_ids", []))
    payload["generated_by"] = payload.get("generated_by") or generated_by
    payload["status"] = payload.get("status") or "candidate"
    return payload


def causal_hypothesis_to_payload(
    value: dict[str, Any],
    *,
    source_episode_id: str,
    generated_by: str = "memory_agent",
) -> dict[str, Any]:
    payload = dict(value)
    payload["confidence"] = _clamp_confidence(payload.get("confidence", 0.0))
    payload["source_episode_id"] = payload.get("source_episode_id") or source_episode_id
    payload["source_action_ids"] = list(payload.get("source_action_ids", []))
    payload["generated_by"] = payload.get("generated_by") or generated_by
    payload["status"] = payload.get("status") or "candidate"
    payload["conditions"] = dict(payload.get("conditions", {}))
    return payload


def _to_plain_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _to_plain_dict(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: _to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_plain_dict(v) for v in value]
    return value


def _clamp_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))
