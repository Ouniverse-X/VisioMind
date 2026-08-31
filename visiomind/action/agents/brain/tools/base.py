from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from visiomind.action.shared.contracts import ToolInvocation
from visiomind.action.shared.results import ToolResult


@dataclass(frozen=True)
class ContextPatch:
    planning_context_updates: dict[str, Any] = field(default_factory=dict)
    task_context_updates: dict[str, Any] = field(default_factory=dict)
    tool_trace_entry: dict[str, Any] | None = None


class BrainTool(Protocol):
    def invoke(self, invocation: ToolInvocation) -> ToolResult:
        pass

    def describe_tool(self, tool_name: str) -> dict[str, Any]:
        pass

    def build_context_patch(self, invocation: ToolInvocation, result: ToolResult) -> ContextPatch:
        pass


def default_tool_trace_entry(tool_name: str, result: ToolResult) -> dict[str, Any]:
    return {
        "tool_name": tool_name,
        "ok": result.ok,
        "error_code": result.error_code,
        "payload": dict(result.payload),
        "metadata": dict(result.metadata),
    }


def default_context_patch(invocation: ToolInvocation, result: ToolResult) -> ContextPatch:
    return ContextPatch(
        planning_context_updates={"tool_outputs": {invocation.tool_name: dict(result.payload)}},
        tool_trace_entry=default_tool_trace_entry(invocation.tool_name, result),
    )


__all__ = ["BrainTool", "ContextPatch", "default_context_patch", "default_tool_trace_entry"]
