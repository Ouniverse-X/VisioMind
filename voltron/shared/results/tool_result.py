"""Shared tool-level result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Normalized result envelope produced by shared tool surfaces."""

    tool_name: str
    ok: bool = True
    payload: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
