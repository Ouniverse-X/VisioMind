"""Subtask-dispatch helpers for the closed-loop orchestrator."""

from __future__ import annotations

from voltron.shared.context import Subtask
from voltron.shared.contracts import SubtaskAgent


def resolve_subtask_agent(*, orchestrator: object, subtask: Subtask) -> SubtaskAgent:
    return orchestrator._agents[subtask.agent]
