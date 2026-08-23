"""Replanning skill for the Brain agent."""

from __future__ import annotations

import json
from typing import Any

from voltron.agents.brain.skills.planning.skill import DefaultBrainPlanningSkill, NAVIGATION_INSTRUCTION_GUIDANCE
from voltron.shared.context import Subtask


class DefaultBrainReplanningSkill(DefaultBrainPlanningSkill):
    """Prompt/schema/parser skill for Brain replanning."""

    _FAILED_PARAMETER_KEYS = (
        "instruction",
        "control_mode",
        "allow_base_motion",
        "completion_criteria",
        "allow_task_complete",
        "stop_condition",
        "constraints",
        "collaborative_step_id",
    )
    _PLAN_SUMMARY_KEYS = (
        "subtask_id",
        "execution_id",
        "plan_revision",
        "replaces_execution_id",
        "agent",
        "action",
        "target",
        "instruction",
    )

    def build_prompt(
        self,
        task_description: str,
        context: dict[str, Any],
        failed_subtask: Subtask,
        failure_reason: str,
        execution_state: dict[str, Any],
    ) -> str:
        failed_payload = self.compact_failed_subtask_for_replan(failed_subtask)
        decision_summary = self.planning_decision_summary(context=context, execution_state=execution_state)
        return (
            "Replan after a failed Voltron subtask.\n"
            f"Task description: {task_description}\n"
            f"Failure reason: {failure_reason}\n"
            f"Failed subtask JSON: {json.dumps(failed_payload, ensure_ascii=False, default=str)}\n"
            f"Planning context JSON: {self.serialize_replanning_context(context)}\n"
            f"Execution state JSON: {self.serialize_replanning_execution_state(execution_state)}\n"
            f"Planner decision summary JSON: {json.dumps(decision_summary, ensure_ascii=False, default=str)}\n"
            "Return the complete ordered active plan revision from the failed point onward, as JSON matching the "
            "required schema. Include any recovery steps, the retry of the original objective, and every still-required "
            "future subtask from the current plan; exclude already completed work. Runtime replaces the pending plan "
            "with this revision, so do not omit an unchanged future step merely because it did not fail. "
            "Do not blindly repeat the same failed subtask when the execution state shows no progress, timeout, "
            "or unchanged scene evidence. Distance, approach readiness, and reachability come from navigation_report. "
            "If the target is visible and navigation_report.approach_ready=true, do not return NAVIGATION navigate/approach; "
            "prefer ACTION local interaction with `parameters.control_mode = \"whole_body_local\"`. Use NAVIGATION only when "
            "this is still clearly a room-level relocation or object-level approach problem. "
            "If the target is visible but navigation_report.approach_ready=false and "
            "navigation_report.approach_reachable=true, use `NAVIGATION approach_target` with an object-centered instruction. "
            f"{NAVIGATION_INSTRUCTION_GUIDANCE} "
            "Navigation door/portal recovery: when Execution state JSON contains "
            "`latest_result.navigation_failure_context.failure_type = \"portal_path_unavailable\"` with nearby "
            "`door_candidates` and either `nav2_error = \"empty_path\"` or "
            "`portal_block_reason = \"blocked_by_closed_door\"`, Brain should decide whether to insert an "
            "approach-and-open recovery before retrying the failed NAVIGATION subtask. Unless execution evidence "
            "already proves the robot is within interaction range of the selected door, return exactly this ordered "
            "recovery prefix: NAVIGATION to approach the candidate door and stop within handle reach, then ACTION "
            "open that same door, then NAVIGATION retrying the original destination. Only omit the approach step when "
            "the execution state explicitly reports that the door is interaction-ready. The runtime will not "
            "hard-code this recovery. Name the transition rooms and candidate door in each recovery instruction, "
            "and keep the original navigation destination in the retry NAVIGATION. "
            "Do not blindly repeat the same door-opening recovery when current_plan and completed_execution_ids show "
            "that the same candidate door was opened and the navigation retry still failed. "
            "Do not repeat room-only navigation for an object-level approach "
            "subtask, especially after timeout or no-progress feedback. "
            "If you choose a VISION observation subtask such as observe/find/inspect, return exactly one subtask in "
            "this response and wait for its observation result before planning any NAVIGATION/ACTION follow-up. A VISION "
            "verification subtask may appear only as the final subtask after an execution step. "
            "This response defines a new plan revision. Number replacement subtasks locally from st_01 in execution "
            "order; runtime assigns revision-scoped execution IDs and preserves the replaced plan in history."
        )

    @classmethod
    def compact_failed_subtask_for_replan(cls, subtask: Subtask) -> dict[str, Any]:
        """Serialize only planner-authored fields from a runtime-mutated subtask."""

        return {
            "subtask_id": subtask.subtask_id,
            "execution_id": subtask.runtime_id,
            "plan_revision": subtask.plan_revision,
            "agent": subtask.agent.value,
            "action": subtask.action,
            "target": cls._strip_heavy_fields(dict(subtask.target)),
            "parameters": cls._select_fields(
                subtask.parameters,
                cls._FAILED_PARAMETER_KEYS,
            ),
        }

    @classmethod
    def serialize_replanning_context(cls, context: dict[str, Any]) -> str:
        """Keep only stable planner contract fields needed to choose recovery agents."""

        compact = {
            "task_type": context.get("task_type"),
            "task_type_hint": context.get("task_type_hint"),
            "planner_mode": context.get("planner_mode", "auto"),
            "agent_capabilities": cls._strip_heavy_fields(
                context.get("agent_capabilities", [])
            ),
            "interaction_target_hints": cls._strip_heavy_fields(
                context.get("interaction_target_hints", {})
            ),
        }
        return json.dumps(compact, ensure_ascii=False, default=str)

    @classmethod
    def serialize_replanning_execution_state(
        cls,
        execution_state: dict[str, Any],
    ) -> str:
        """Build a bounded replan payload from active-plan and failure evidence only."""

        current_plan = execution_state.get("current_plan")
        compact_plan = (
            [
                cls._select_fields(item, cls._PLAN_SUMMARY_KEYS)
                for item in current_plan
                if isinstance(item, dict)
            ]
            if isinstance(current_plan, list)
            else []
        )
        compact = {
            "task_type": execution_state.get("task_type"),
            "planner_mode": execution_state.get("planner_mode"),
            "failure_reason": execution_state.get("failure_reason"),
            "latest_result": cls.compact_latest_result(
                execution_state.get("latest_result")
            ),
            "last_scene_report": cls.compact_scene_report(
                execution_state.get("last_scene_report")
            ),
            "navigation_state": cls.compact_navigation_state(
                execution_state.get("navigation_state")
            ),
            "navigation_report": cls.compact_navigation_report(
                execution_state.get("navigation_report")
            ),
            "current_plan_revision": execution_state.get("current_plan_revision"),
            "current_plan": compact_plan,
            "current_plan_execution_ids": list(
                execution_state.get("current_plan_execution_ids") or []
            )[:100],
            "completed_execution_ids": list(
                execution_state.get("completed_execution_ids") or []
            )[-100:],
        }
        return json.dumps(compact, ensure_ascii=False, default=str)


__all__ = ["DefaultBrainReplanningSkill"]
