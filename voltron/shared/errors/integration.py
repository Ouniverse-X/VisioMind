"""Integration-layer error types."""

from .base import VoltronError


class AdapterError(VoltronError):
    """Raised when an external adapter call fails."""
