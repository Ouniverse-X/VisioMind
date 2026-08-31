from __future__ import annotations

from visiomind.action.shared.context import Subtask
from visiomind.action.shared.contracts import SubtaskAgent


def resolve_subtask_agent(*, orchestrator: object, subtask: Subtask) -> SubtaskAgent:
    return orchestrator._agents[subtask.agent]
