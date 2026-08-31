from __future__ import annotations

from typing import Any

from voltron.agents.brain.tools import interaction_targeting
from voltron.shared.enums import AgentName
from voltron.shared.context import Subtask, TaskRequest


def select_interaction_seed_target(subtasks: list[Subtask]) -> dict[str, Any]:
    preferred_agents = (AgentName.VISION, AgentName.ACTION, AgentName.NAVIGATION)
    for agent in preferred_agents:
        for subtask in subtasks:
            if subtask.agent == agent and subtask.target:
                return dict(subtask.target)
    for subtask in subtasks:
        if subtask.target:
            return dict(subtask.target)
    return {}


def build_seed_interaction_target(
    *, request: TaskRequest, subtasks: list[Subtask]
) -> dict[str, Any]:
    target = select_interaction_seed_target(subtasks)
    hints = interaction_targeting.interaction_target_hints(request=request, subtasks=subtasks)

    if "object" not in target and isinstance(hints.get("object"), str):
        target["object"] = hints["object"]
    if "part" not in target and isinstance(hints.get("part"), str):
        target["part"] = hints["part"]
    if "room" not in target and isinstance(hints.get("room"), str):
        target["room"] = hints["room"]
    if "region" not in target and isinstance(hints.get("region"), str):
        target["region"] = hints["region"]
    return target


def build_seed_interaction_plan(*, request: TaskRequest, subtasks: list[Subtask]) -> list[Subtask]:
    target = build_seed_interaction_target(request=request, subtasks=subtasks)
    return [
        Subtask(
            subtask_id="st_01",
            agent=AgentName.VISION,
            action="inspect_scene",
            target=target,
            parameters={
                "instruction": interaction_targeting.interaction_seed_instruction(target),
                "allow_task_complete": False,
            },
            context={"task_description": request.description, "seed_plan": True},
        )
    ]
