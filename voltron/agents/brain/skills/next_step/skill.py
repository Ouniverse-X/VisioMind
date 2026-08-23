"""Next-step planning skill for the Brain agent."""

from __future__ import annotations

import json
from typing import Any

from voltron.agents.brain.skills.planning.skill import DefaultBrainPlanningSkill, NAVIGATION_INSTRUCTION_GUIDANCE


class DefaultBrainNextStepSkill(DefaultBrainPlanningSkill):
    """Prompt/schema/parser skill for Brain next-step planning."""

    def build_prompt(
        self,
        task_description: str,
        context: dict[str, Any],
        execution_state: dict[str, Any],
    ) -> str:
        decision_summary = self.planning_decision_summary(context=context, execution_state=execution_state)
        return (
            "Decide the next executable Voltron subtask chunk from runtime feedback.\n"
            f"Task description: {task_description}\n"
            f"Planning context JSON: {self.serialize_context(context)}\n"
            f"Execution state JSON: {self.serialize_execution_state(execution_state)}\n"
            f"Planner decision summary JSON: {json.dumps(decision_summary, ensure_ascii=False, default=str)}\n"
            "Distance, approach readiness, and reachability come from navigation_report; do not ask VISION inspect to infer them.\n"
            "If the target is visible and navigation_report.approach_ready=true, treat this as a local execution problem. "
            "Prefer ACTION with `parameters.control_mode = \"whole_body_local\"`.\n"
            "Decision cases:\n"
            "- If room-level localization says the robot is outside the target room/region, use `NAVIGATION navigate` first.\n"
            "- If the target is visible but navigation_report.approach_ready=false and navigation_report.approach_reachable=true, use `NAVIGATION` `approach_target` toward the object, not the room.\n"
            "- The instruction must name the target object or part, not the room instance name.\n"
            f"- {NAVIGATION_INSTRUCTION_GUIDANCE}\n"
            "- If the target is visible and navigation_report.approach_ready=true, use `ACTION` with `parameters.control_mode = \"whole_body_local\"`.\n"
            "- If the target is not yet visible in the current room, use exactly one `VISION` inspect/find step.\n"
            "If you choose a VISION observation subtask such as observe/find/inspect, return exactly one subtask in "
            "this response and wait for its observation result before planning any NAVIGATION/ACTION follow-up. A VISION "
            "verification subtask may appear only as the final subtask after an execution step.\n"
            "This response defines a new plan revision. Number subtasks locally from st_01 in execution order; "
            "runtime assigns revision-scoped execution IDs.\n"
            "Return JSON only. The `subtasks` array may be empty if the task is already complete. "
            "Prefer 1 subtask. Use at most 2 subtasks when you need an execution step plus a final VISION verification "
            "step; if the response contains a VISION observation step, it must be the only subtask."
        )


__all__ = ["DefaultBrainNextStepSkill"]
