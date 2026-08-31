from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from voltron.shared.context import ExecutionContext, LocalSkillSelection, Subtask
from voltron.shared.results import SkillResult


@dataclass
class SkillRequest:
    subtask: Subtask
    context: ExecutionContext
    selection: LocalSkillSelection | None = None
    runtime_inputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class SkillExecutor(Protocol):
    def execute(self, request: SkillRequest) -> SkillResult:
        pass
