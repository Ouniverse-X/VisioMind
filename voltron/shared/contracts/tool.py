from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from voltron.shared.results import ToolResult


@dataclass
class ToolInvocation:
    tool_name: str
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolExecutor(Protocol):
    def invoke(self, invocation: ToolInvocation) -> ToolResult:
        pass
