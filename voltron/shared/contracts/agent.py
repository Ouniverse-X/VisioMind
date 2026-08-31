from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from voltron.shared.context import ExecutionContext, Subtask
from voltron.shared.results import AgentResult


@dataclass
class AgentRequest:
    subtask: Subtask
    context: ExecutionContext
    runtime_inputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class SubtaskAgent(Protocol):
    def execute(self, subtask: Subtask, context: ExecutionContext) -> AgentResult:
        pass


class EpisodeSubtaskAgent(SubtaskAgent, Protocol):
    def run_episode(
        self, *, subtask: Subtask, context: ExecutionContext, runtime: Any
    ) -> AgentResult:
        pass
