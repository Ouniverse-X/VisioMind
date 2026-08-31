from __future__ import annotations

from typing import Any


class ToolCatalog:
    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}

    def register(self, tool_name: str, tool: Any) -> None:
        self._tools[tool_name] = tool

    def get(self, tool_name: str) -> Any:
        return self._tools[tool_name]

    def names(self) -> list[str]:
        return sorted(self._tools.keys())
