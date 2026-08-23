"""Shared tool-invocation contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from voltron.shared.results import ToolResult


@dataclass
class ToolInvocation:
    """Canonical invocation envelope for agent/runtime tools."""

    tool_name: str
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolExecutor(Protocol):
    """Protocol for shared tool-execution surfaces."""

    def invoke(self, invocation: ToolInvocation) -> ToolResult:
        """Execute one tool invocation and return a normalized tool result."""
