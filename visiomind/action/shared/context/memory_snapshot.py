from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemorySnapshot:
    working_state: dict[str, Any] = field(default_factory=dict)
    active_regions: list[str] = field(default_factory=list)
    task_context: dict[str, Any] = field(default_factory=dict)
    recent_observations: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_memory(cls, memory: Any, *, recent_observation_limit: int = 10) -> "MemorySnapshot":
        return cls(
            working_state=dict(memory.get_working_state()),
            active_regions=list(memory.get_active_regions()),
            task_context=dict(memory.get_task_context()),
            recent_observations=list(memory.get_recent_observations(n=recent_observation_limit)),
        )

    def to_planning_context(self) -> dict[str, Any]:
        return {
            "working_state": dict(self.working_state),
            "active_regions": list(self.active_regions),
            "task_context": dict(self.task_context),
            "recent_observations": [dict(item) for item in self.recent_observations],
        }
