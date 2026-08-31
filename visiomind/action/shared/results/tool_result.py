from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    tool_name: str
    ok: bool = True
    payload: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
