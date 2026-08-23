"""Default direct-navigation skill used for ordinary Navigation execution."""

from __future__ import annotations

from voltron.shared.context import ExecutionContext, Subtask


class DirectNavigationSkill:
    skill_id = "direct_navigation_skill"

    def can_handle(self, subtask: Subtask, context: ExecutionContext) -> bool:
        return True

    def prepare(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        navigator,
        start: dict,
        goal: dict,
        navigation_context: dict,
    ) -> dict:
        return {
            "skill_id": self.skill_id,
            "mode": "direct_navigation",
            "selection_context": {
                "goal_type": goal.get("goal_type"),
                "room_name": goal.get("room_name"),
                "object_name": goal.get("object_name"),
            },
        }


__all__ = ["DirectNavigationSkill"]
