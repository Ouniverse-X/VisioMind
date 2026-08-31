from __future__ import annotations

import json
from json import JSONDecoder
from typing import Any

from visiomind.action.agents.action.models import (
    ActionExecutionPlan,
    ActionInternalStep,
    ActionReplanDecision,
)
from visiomind.action.shared.action_semantics import (
    action_instruction,
    is_open_state_action,
    normalize_action_name,
)
from visiomind.action.shared.context import ExecutionContext, Subtask


class DefaultActionTaskPlanningSkill:
    def build_plan_prompt(self, subtask: Subtask, context: ExecutionContext) -> str:
        payload = {
            "task_description": context.task_request.description,
            "task_type": context.task_request.task_type.value,
            "subtask": {
                "subtask_id": subtask.subtask_id,
                "action": subtask.action,
                "target": dict(subtask.target),
                "instruction": subtask.parameters.get("instruction"),
                "success_cues": list(subtask.parameters.get("success_cues", [])),
                "conditions": dict(subtask.parameters.get("conditions", {})),
                "context": dict(subtask.context),
            },
        }
        return (
            "Decompose the current Action subtask into semantically useful coarse subtasks for local embodied execution.\n"
            "Return valid JSON only. Do not include markdown fences or extra text.\n"
            "Use 1 to 4 steps. Each step must contribute directly to the task goal and be useful for execution.\n"
            "ACTION subtasks run after navigation; do not create room-scale navigation, object-search, or go-to-room steps.\n"
            "When the current target is already local to the robot, you may create bounded local reposition steps "
            "such as `approach`, `align`, or `move_to_interaction_pose` if they are necessary to make the "
            "current interaction reachable. Keep those steps local to the current target and do not use them for "
            "global navigation.\n"
            "For open/close articulated-object tasks, prefer one direct manipulation step with the parent instruction.\n"
            "Prefer meaningful interaction phases such as reaching a control, actuating it, stabilizing an object, or moving to a clear placement pose.\n"
            "Do not mechanically expand into controller micro-steps or filler actions such as generic align/retract phases unless they are necessary for completing the task safely.\n"
            "Do not restate the parent instruction in the form 'goal - step name'. Each instruction should be a standalone coarse action sentence.\n"
            "Use `local_reposition_skill` for bounded local approach/align/reposition steps, not for room-scale navigation.\n"
            "Use `applicable_skills` from subtask context as procedural-memory hints when present; map them to known local skill ids only when the mapping is clear.\n"
            "You may optionally provide `preferred_skill_id` when a known local skill is clearly appropriate.\n"
            "Return a top-level JSON object with this schema:\n"
            "{\n"
            '  "goal_summary": "short summary",\n'
            '  "steps": [\n'
            "    {\n"
            '      "name": "snake_case_step_name",\n'
            '      "instruction": "coarse action instruction",\n'
            '      "action": "verb_like_action",\n'
            '      "target": {"optional": "target overrides"},\n'
            '      "preferred_skill_id": "optional known skill id",\n'
            '      "success_cues": ["optional success cue"]\n'
            "    }\n"
            "  ]\n"
            "}\n"
            f"Planning context JSON: {json.dumps(payload, ensure_ascii=False, default=str)}\n"
            "Return JSON only."
        )

    def parse_plan_response(
        self,
        content: str,
        subtask: Subtask,
        context: ExecutionContext,
    ) -> ActionExecutionPlan:
        del context
        payload = self._extract_json(content)
        if is_open_state_action(subtask.action):
            return self._direct_state_change_plan(subtask=subtask, payload=payload)
        raw_steps = payload.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError("Action task planner response must include a non-empty steps list")

        steps: list[ActionInternalStep] = []
        for index, item in enumerate(raw_steps, start=1):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or f"step_{index:02d}").strip() or f"step_{index:02d}"
            instruction = str(item.get("instruction") or "").strip()
            action = str(item.get("action") or "").strip()
            if not instruction or not action:
                continue
            raw_target = item.get("target")
            target = dict(subtask.target)
            if isinstance(raw_target, dict):
                target.update(raw_target)
            raw_success_cues = item.get("success_cues")
            success_cues = []
            if isinstance(raw_success_cues, list):
                success_cues = [str(cue).strip() for cue in raw_success_cues if str(cue).strip()]
            steps.append(
                ActionInternalStep(
                    internal_step_id=f"{subtask.subtask_id}.act_{index:02d}",
                    name=name,
                    instruction=instruction,
                    action=action,
                    target=target,
                    preferred_skill_id=str(item.get("preferred_skill_id") or "").strip() or None,
                    success_cues=success_cues,
                    metadata={"planner_response": dict(item)},
                )
            )
        if not steps:
            raise ValueError("Action task planner response did not yield any valid steps")
        return ActionExecutionPlan(
            parent_subtask_id=subtask.subtask_id,
            goal_summary=str(
                payload.get("goal_summary")
                or subtask.parameters.get("instruction")
                or subtask.action
            ).strip(),
            steps=steps,
            source="action_model_planner",
            metadata={"raw_response": payload},
        )

    def _direct_state_change_plan(
        self, *, subtask: Subtask, payload: dict[str, Any]
    ) -> ActionExecutionPlan:
        canonical_action = normalize_action_name(subtask.action)
        instruction = str(subtask.parameters.get("instruction") or "").strip()
        if not instruction:
            instruction = action_instruction(action=canonical_action, target=dict(subtask.target))
        return ActionExecutionPlan(
            parent_subtask_id=subtask.subtask_id,
            goal_summary=instruction,
            steps=[
                ActionInternalStep(
                    internal_step_id=f"{subtask.subtask_id}.act_01",
                    name=f"perform_{canonical_action}",
                    instruction=instruction,
                    action=canonical_action,
                    target=dict(subtask.target),
                    metadata={"planner_response": payload, "direct_state_change": True},
                )
            ],
            source="action_state_change_direct",
            metadata={"raw_response": payload},
        )

    def build_fallback_plan(
        self,
        subtask: Subtask,
        context: ExecutionContext,
        *,
        reason: str,
    ) -> ActionExecutionPlan:
        del context
        instruction = (
            str(subtask.parameters.get("instruction") or subtask.action).strip() or subtask.action
        )
        return ActionExecutionPlan(
            parent_subtask_id=subtask.subtask_id,
            goal_summary=instruction,
            source="action_planner_fallback",
            metadata={"reason": reason},
            steps=[
                ActionInternalStep(
                    internal_step_id=f"{subtask.subtask_id}.act_01",
                    name=f"perform_{subtask.action}",
                    instruction=instruction,
                    action=subtask.action,
                    target=dict(subtask.target),
                )
            ],
        )

    def replan(
        self,
        subtask: Subtask,
        context: ExecutionContext,
        *,
        active_step_id: str,
        reason: str,
    ) -> ActionReplanDecision:
        del subtask, context, active_step_id
        return ActionReplanDecision(should_replan=False, reason=reason)

    @staticmethod
    def _extract_json(content: str) -> dict[str, Any]:
        stripped = content.strip()
        candidates = [stripped]
        if "```" in stripped:
            for block in stripped.split("```"):
                candidate = block.strip()
                if not candidate:
                    continue
                if candidate.startswith("json"):
                    candidate = candidate[4:].strip()
                candidates.append(candidate)

        decoder = JSONDecoder()
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                for start_index, char in enumerate(candidate):
                    if char not in "{[":
                        continue
                    try:
                        payload, _ = decoder.raw_decode(candidate[start_index:])
                    except json.JSONDecodeError:
                        continue
                    break
                else:
                    continue
            if isinstance(payload, dict):
                return payload
        raise ValueError("Failed to parse JSON from Action task planner response")


DefaultVLATaskPlanningSkill = DefaultActionTaskPlanningSkill

__all__ = ["DefaultActionTaskPlanningSkill", "DefaultVLATaskPlanningSkill"]
