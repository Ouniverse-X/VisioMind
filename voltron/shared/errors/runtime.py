"""Runtime-layer error types."""

from .base import VoltronError


class PlanningError(VoltronError):
    """Raised when plan generation/replanning fails."""


class ExecutionError(VoltronError):
    """Raised when agent execution fails irrecoverably."""
