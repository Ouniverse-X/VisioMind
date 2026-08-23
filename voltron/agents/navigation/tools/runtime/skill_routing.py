"""Skill-routing helpers for the Navigation agent runtime."""

from __future__ import annotations

from typing import Any

from voltron.shared.context import ExecutionContext, LocalSkillSelection, Subtask


def available_navigation_skill_ids(registry: Any) -> list[str]:
    if hasattr(registry, "available_skill_ids"):
        return list(registry.available_skill_ids())
    if isinstance(registry, dict):
        return list(registry.keys())
    return []


def select_navigation_skill(
    *,
    selector: Any,
    registry: Any,
    subtask: Subtask,
    context: ExecutionContext,
) -> LocalSkillSelection:
    return selector.select_skill(subtask, context, available_navigation_skill_ids(registry))


def resolve_navigation_skill(
    *,
    registry: Any,
    subtask: Subtask,
    context: ExecutionContext,
    selection: LocalSkillSelection,
) -> Any | None:
    if hasattr(registry, "resolve"):
        return registry.resolve(subtask=subtask, context=context, selection=selection)
    if isinstance(registry, dict):
        skill = registry.get(selection.skill_id)
        if skill is not None and skill.can_handle(subtask, context):
            return skill
        for candidate_id in selection.fallback_skill_candidates:
            candidate = registry.get(candidate_id)
            if candidate is not None and candidate.can_handle(subtask, context):
                return candidate
        for candidate in registry.values():
            if candidate.can_handle(subtask, context):
                return candidate
    return None
