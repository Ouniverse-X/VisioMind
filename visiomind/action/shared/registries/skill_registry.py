from __future__ import annotations

from typing import Generic, Iterable, Protocol, TypeVar

from visiomind.action.shared.context import ExecutionContext, LocalSkillSelection, Subtask


class ResolvableSkill(Protocol):
    skill_id: str

    def can_handle(self, subtask: Subtask, context: ExecutionContext) -> bool:
        pass


TSkill = TypeVar("TSkill", bound=ResolvableSkill)


class SkillRegistryBase(Generic[TSkill]):
    def __init__(self, skills: Iterable[TSkill], *, default_skill_id: str | None = None) -> None:
        self._skills = {skill.skill_id: skill for skill in skills}
        self._default_skill_id = default_skill_id

    def available_skill_ids(self) -> list[str]:
        return list(self._skills.keys())

    def resolve_selected_skill(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        selection: LocalSkillSelection,
    ) -> TSkill:
        requested = self._skills.get(selection.skill_id)
        if requested is not None and requested.can_handle(subtask, context):
            return requested

        for candidate in selection.fallback_skill_candidates:
            skill = self._skills.get(candidate)
            if skill is not None and skill.can_handle(subtask, context):
                return skill

        for skill in self._skills.values():
            if skill.can_handle(subtask, context):
                return skill

        if self._default_skill_id is not None and self._default_skill_id in self._skills:
            return self._skills[self._default_skill_id]

        raise KeyError(
            f"No compatible skill found for selection {selection.skill_id!r} "
            f"and no default skill is registered."
        )
