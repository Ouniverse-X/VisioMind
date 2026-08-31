from __future__ import annotations

from typing import Any

from voltron.agents.action.models import VLATargetRefinement
from voltron.shared.action_semantics import action_instruction, normalize_action_name
from voltron.shared.context import ExecutionContext, MemorySnapshot, Subtask
from voltron.shared.contracts import MemoryAdapter


class StructuredTargetRefiner:
    def __init__(self, memory: MemoryAdapter) -> None:
        self.memory = memory

    def refine_target(
        self,
        subtask: Subtask,
        context: ExecutionContext,
    ) -> VLATargetRefinement:
        latest_scene_report = self._latest_scene_report(context)
        memory_snapshot = MemorySnapshot.from_memory(self.memory, recent_observation_limit=3)
        target = dict(subtask.target)

        part_name = str(target.get("part") or "").strip()
        if not part_name and isinstance(latest_scene_report, dict):
            candidate = str(latest_scene_report.get("target_part_name") or "").strip()
            if candidate:
                part_name = candidate
                target["part"] = candidate

        refined_instruction = self._build_instruction(
            subtask=subtask, target=target, part_name=part_name
        )
        action_name = normalize_action_name(subtask.action)
        selector_hints: dict[str, Any] = {}
        if action_name in {
            "toggle_on",
            "toggle_off",
            "turn_on",
            "turn_off",
            "press",
            "push_button",
        }:
            selector_hints["preferred_skill_id"] = "button_interaction_skill"
        elif action_name in {"open", "close", "pull", "push", "turn", "rotate"}:
            selector_hints["preferred_skill_id"] = "handle_operation_skill"
        elif action_name in {"pick_up", "grasp", "lift", "take"}:
            selector_hints["preferred_skill_id"] = "grasp_manipulation_skill"

        metadata = {
            "task_context": memory_snapshot.task_context,
            "latest_scene_report": latest_scene_report,
            "recent_observation_count": len(memory_snapshot.recent_observations),
        }
        return VLATargetRefinement(
            refined_instruction=refined_instruction,
            refined_target=target,
            selector_hints=selector_hints,
            policy_hints={},
            success_cues=self._build_success_cues(target=target, part_name=part_name),
            metadata=metadata,
        )

    @staticmethod
    def _latest_scene_report(context: ExecutionContext) -> dict[str, Any]:
        for result in reversed(context.results):
            scene_report = result.result.get("scene_report")
            if isinstance(scene_report, dict):
                return dict(scene_report)
        return {}

    @staticmethod
    def _build_instruction(subtask: Subtask, target: dict[str, Any], part_name: str) -> str:
        existing = str(subtask.parameters.get("instruction") or "").strip()
        if existing:
            return existing
        return action_instruction(action=subtask.action, target=target, part_name=part_name)

    @staticmethod
    def _build_success_cues(target: dict[str, Any], part_name: str) -> list[str]:
        object_name = str(
            target.get("object") or target.get("object_id") or "target object"
        ).strip()
        if part_name:
            return [f"{part_name} on {object_name} has been manipulated"]
        return [f"{object_name} manipulation completed"]


__all__ = ["StructuredTargetRefiner"]
