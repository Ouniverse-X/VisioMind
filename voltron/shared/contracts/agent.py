"""Agent execution interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from voltron.shared.context import ExecutionContext, Subtask
from voltron.shared.results import AgentResult


@dataclass
class AgentRequest:
    """Canonical agent-execution request envelope."""

    subtask: Subtask
    context: ExecutionContext
    runtime_inputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class SubtaskAgent(Protocol):
    """Protocol implemented by task-executing agents."""

    def execute(self, subtask: Subtask, context: ExecutionContext) -> AgentResult:
        """Execute one subtask and return a normalized result."""


class EpisodeSubtaskAgent(SubtaskAgent, Protocol):
    """Protocol for agents that own their internal subtask episode loop."""

    def run_episode(self, *, subtask: Subtask, context: ExecutionContext, runtime: Any) -> AgentResult:
        """Run a complete subtask episode using runtime tools supplied by the orchestrator."""
