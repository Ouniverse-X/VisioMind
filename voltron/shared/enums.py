"""Shared enums used across agents, runtime, and integrations."""

from enum import Enum


class AgentName(str, Enum):
    """Known agent roles in Voltron."""

    BRAIN = "BRAIN"
    VISION = "VISION"
    NAVIGATION = "NAVIGATION"
    ACTION = "ACTION"
    MEMORY = "MEMORY"

    @classmethod
    def parse(cls, value: object) -> "AgentName":
        normalized = str(value or "").strip().upper()
        return cls(normalized)


class AgentStatus(str, Enum):
    """Unified execution status for agent calls."""

    SUCCESS = "success"
    FAILURE = "failure"


class TaskType(str, Enum):
    """Task type used by planner and memory adapter."""

    MANIPULATION = "manipulation"
    NAVIGATION = "navigation"
    INTERACTION = "interaction"
    OBSERVATION = "observation"
