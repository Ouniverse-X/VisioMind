"""Registry for VLA skill implementations."""

from __future__ import annotations

import logging
from typing import Any

from voltron.agents.action.contracts import VLASkill
from voltron.agents.action.tools.action_projection import ActionProjection
from voltron.shared.context import ExecutionContext, LocalSkillSelection, Subtask
from voltron.shared.contracts import MemoryAdapter, PolicyAdapter
from voltron.shared.registries import SkillRegistryBase

from .execution.defaults import (
    ButtonInteractionSkill,
    DefaultManipulationSkill,
    GraspManipulationSkill,
    HandleOperationSkill,
    LocalRepositionSkill,
    PlacementSkill,
)

logger = logging.getLogger(__name__)


class VLASkillRegistry(SkillRegistryBase[VLASkill]):
    """Resolve VLA skills by semantic skill id with safe fallback."""

    def __init__(self, skills: list[VLASkill]) -> None:
        super().__init__(skills, default_skill_id="default_manipulation_skill")

    @classmethod
    def build_default(
        cls,
        memory: MemoryAdapter,
        policy: PolicyAdapter,
        projector: ActionProjection,
        anygrasp_config: dict[str, Any] | None = None,
    ) -> "VLASkillRegistry":
        skills: list[VLASkill] = [
            ButtonInteractionSkill(memory=memory, policy=policy, projector=projector),
            PlacementSkill(memory=memory, policy=policy, projector=projector),
            HandleOperationSkill(memory=memory, policy=policy, projector=projector),
            LocalRepositionSkill(memory=memory, policy=policy, projector=projector),
            DefaultManipulationSkill(memory=memory, policy=policy, projector=projector),
        ]

        if anygrasp_config:
            from .execution.anygrasp_skill import AnyGraspSkill

            skills.insert(
                0,
                AnyGraspSkill(
                    memory=memory,
                    policy=policy,
                    projector=projector,
                    anygrasp_config=anygrasp_config,
                ),
            )
            logger.info("AnyGraspSkill registered (replaces GraspManipulationSkill for grasp actions)")
        else:
            skills.insert(
                0,
                GraspManipulationSkill(memory=memory, policy=policy, projector=projector),
            )

        return cls(skills=skills)

    def resolve(
        self,
        subtask: Subtask,
        context: ExecutionContext,
        selection: LocalSkillSelection,
    ) -> VLASkill:
        return self.resolve_selected_skill(
            subtask=subtask,
            context=context,
            selection=selection,
        )


class ActionSkillRegistry(VLASkillRegistry):
    """Canonical registry type for Action agent skills."""


__all__ = ["ActionSkillRegistry", "VLASkillRegistry"]
