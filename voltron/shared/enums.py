from enum import Enum


class AgentName(str, Enum):
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
    SUCCESS = "success"
    FAILURE = "failure"


class TaskType(str, Enum):
    MANIPULATION = "manipulation"
    NAVIGATION = "navigation"
    INTERACTION = "interaction"
    OBSERVATION = "observation"
