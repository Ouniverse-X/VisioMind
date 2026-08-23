"""Skill that prepares discrete approach anchors for object-level Navigation tasks."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from voltron.shared.context import ExecutionContext, Subtask
from voltron.shared.object_approach_signature import candidate_signature, signature_values_match


class ObjectApproachSelectionSkill:
    skill_id = "object_approach_selection_skill"

    def __init__(self, memory) -> None:
        self.memory = memory

    def can_handle(self, subtask: Subtask, context: ExecutionContext) -> bool:
        grounded_goal = context.runtime_state.get("navigation_grounded_goal_for_skill")
        grounded_target = ""
        if isinstance(grounded_goal, dict) and str(grounded_goal.get("goal_type") or "").strip().lower() == "object":
            grounded_target = str(grounded_goal.get("object_id") or grounded_goal.get("object_name") or "").strip()
        return bool(
            grounded_target
            or str(subtask.target.get("object") or subtask.target.get("object_id") or "").strip()
        )

    def prepare(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        navigator,
        start: dict[str, Any],
        goal: dict[str, Any],
        navigation_context: dict[str, Any],
    ) -> dict[str, Any]:
        scene_id = str(goal.get("scene_id") or start.get("scene_id") or navigation_context.get("scene_id") or "").strip()
        history = self.memory.get_object_approach_history(
            scene_id=scene_id,
            target=self._history_target(goal=goal, subtask=subtask),
            top_k=10,
        ) if scene_id else {"scene_id": None, "target_key": None, "entries": []}
        history = self._merge_memory_navigation_guidance(
            history=history,
            guidance=subtask.context.get("memory_navigation_guidance"),
        )
        raw_candidates = []
        if hasattr(navigator, "generate_object_approach_candidates"):
            raw_candidates = navigator.generate_object_approach_candidates(
                start=start,
                goal=goal,
                context=navigation_context,
            )
        candidates = self._annotate_candidates(raw_candidates, history)
        return {
            "skill_id": self.skill_id,
            "mode": "object_approach_selection",
            "scene_id": scene_id,
            "target_key": history.get("target_key"),
            "history": history,
            "candidates": candidates,
            "selection_context": {
                "target_object": goal.get("object_name") or subtask.target.get("object"),
                "target_room": goal.get("room_name") or subtask.target.get("room") or subtask.target.get("region"),
                "current_room": start.get("current_room") or start.get("current_region"),
            },
        }

    @staticmethod
    def _history_target(*, goal: dict[str, Any], subtask: Subtask) -> dict[str, Any]:
        return {
            "object": goal.get("object_name") or subtask.target.get("object"),
            "object_id": goal.get("object_id") or subtask.target.get("object_id"),
            "room_id": goal.get("room_id") or subtask.target.get("room_id"),
            "room_name": goal.get("room_name") or subtask.target.get("room") or subtask.target.get("region"),
            "floor_id": goal.get("floor_id") or subtask.target.get("floor_id"),
        }

    def _annotate_candidates(
        self,
        candidates: list[dict[str, Any]],
        history: dict[str, Any],
    ) -> list[dict[str, Any]]:
        history_entries = list(history.get("entries") or [])
        annotated: list[dict[str, Any]] = []
        for index, item in enumerate(candidates, start=1):
            candidate = deepcopy(item)
            candidate.setdefault("candidate_id", f"cand_{index:02d}")
            summary = self._history_summary(candidate, history_entries)
            candidate["history_summary"] = summary
            candidate["history_penalty"] = float(summary["failure_count"]) * 3.0 - float(summary["success_count"]) * 1.5
            if summary["evidence_sources"]:
                candidate["memory_guidance_sources"] = list(summary["evidence_sources"])
            candidate["blocked_by_history"] = bool(
                summary["failure_count"] > 0 and summary["success_count"] == 0 and summary["recent_failure"]
            )
            annotated.append(candidate)
        return annotated

    @classmethod
    def _merge_memory_navigation_guidance(
        cls,
        *,
        history: dict[str, Any],
        guidance: Any,
    ) -> dict[str, Any]:
        merged = deepcopy(history) if isinstance(history, dict) else {"entries": []}
        entries = list(merged.get("entries") or [])
        if not isinstance(guidance, dict):
            merged["entries"] = entries
            return merged

        for item in guidance.get("avoid_object_approach_candidates", []):
            if not isinstance(item, dict):
                continue
            entries.extend(cls._guidance_entries(item, outcome="failure", count_key="failure_count"))
        for item in guidance.get("prefer_object_approach_candidates", []):
            if not isinstance(item, dict):
                continue
            entries.extend(cls._guidance_entries(item, outcome="success", count_key="success_count"))
        merged["entries"] = entries
        return merged

    @staticmethod
    def _guidance_entries(item: dict[str, Any], *, outcome: str, count_key: str) -> list[dict[str, Any]]:
        signature = item.get("candidate_signature")
        if not isinstance(signature, dict) or not signature:
            return []
        count = _positive_int(item.get(count_key), default=1)
        return [
            {
                "candidate_signature": deepcopy(signature),
                "outcome": outcome,
                "reason": item.get("reason"),
                "metadata": {"evidence_source": "planning_guidance"},
            }
            for _ in range(count)
        ]

    @staticmethod
    def _history_summary(candidate: dict[str, Any], history_entries: list[dict[str, Any]]) -> dict[str, Any]:
        failure_count = 0
        success_count = 0
        recent_failure = False
        last_reason = None
        evidence_sources: set[str] = set()
        signature = ObjectApproachSelectionSkill._candidate_signature(candidate)
        for entry in history_entries:
            entry_signature = dict(entry.get("candidate_signature") or {})
            if not signature_values_match(signature, entry_signature):
                continue
            evidence_sources.add(_history_entry_source(entry))
            outcome = str(entry.get("outcome") or "").strip().lower()
            if outcome == "success":
                success_count += 1
            elif outcome:
                failure_count += 1
                recent_failure = True
                last_reason = entry.get("reason") or last_reason
        return {
            "candidate_signature": signature,
            "failure_count": failure_count,
            "success_count": success_count,
            "recent_failure": recent_failure,
            "last_reason": last_reason,
            "evidence_sources": _ordered_evidence_sources(evidence_sources),
        }

    @staticmethod
    def _candidate_signature(candidate: dict[str, Any]) -> dict[str, Any]:
        return candidate_signature(candidate)


def _positive_int(value: Any, *, default: int) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return default
    return count if count > 0 else default


def _history_entry_source(entry: dict[str, Any]) -> str:
    metadata = entry.get("metadata")
    if isinstance(metadata, dict):
        source = metadata.get("evidence_source") or metadata.get("source")
        if isinstance(source, str) and source.strip():
            return source.strip()
    source = entry.get("evidence_source") or entry.get("source")
    if isinstance(source, str) and source.strip():
        return source.strip()
    return "persistent_history"


def _ordered_evidence_sources(sources: set[str]) -> list[str]:
    preferred = ["persistent_history", "planning_guidance"]
    ordered = [source for source in preferred if source in sources]
    ordered.extend(sorted(source for source in sources if source not in preferred))
    return ordered


class NavigationObjectApproachSelectionSkill(ObjectApproachSelectionSkill):
    """Canonical object-approach skill for the Navigation agent."""


__all__ = ["NavigationObjectApproachSelectionSkill", "ObjectApproachSelectionSkill"]
