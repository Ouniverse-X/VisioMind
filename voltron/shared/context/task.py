from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from voltron.shared.enums import AgentName, TaskType


@dataclass
class TaskRequest:
    task_id: str
    description: str
    task_type: TaskType = TaskType.MANIPULATION
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Subtask:
    subtask_id: str
    agent: AgentName
    action: str
    target: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    plan_revision: int = 0
    execution_id: str | None = None
    replaces_execution_id: str | None = None

    @property
    def runtime_id(self) -> str:
        return self.execution_id or self.subtask_id


@dataclass
class Plan:
    subtasks: list[Subtask]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalSkillSelection:
    skill_id: str
    confidence: float = 0.0
    reason: str = ""
    source: str = "local_selector"
    fallback_skill_candidates: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
