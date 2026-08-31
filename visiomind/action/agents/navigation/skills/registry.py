from __future__ import annotations

from visiomind.action.shared.context import ExecutionContext, LocalSkillSelection, Subtask
from visiomind.action.shared.registries import SkillRegistryBase

from .direct_navigation import DirectNavigationSkill
from .object_approach.skill import ObjectApproachSelectionSkill


class VLNSkillRegistry(SkillRegistryBase):
    def __init__(self, skills: list) -> None:
        super().__init__(skills, default_skill_id="direct_navigation_skill")

    @classmethod
    def build_default(cls, *, memory) -> "VLNSkillRegistry":
        return cls(
            skills=[
                ObjectApproachSelectionSkill(memory=memory),
                DirectNavigationSkill(),
            ]
        )

    def resolve(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        selection: LocalSkillSelection,
    ):
        return self.resolve_selected_skill(
            subtask=subtask,
            context=context,
            selection=selection,
        )


class NavigationSkillRegistry(VLNSkillRegistry):
    pass


__all__ = ["NavigationSkillRegistry", "VLNSkillRegistry"]
