"""Shared runtime-update contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeUpdate:
    """Canonical runtime/control-plane update envelope."""

    trace_id: str
    update_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    source: str = "runtime"
    subtask_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
