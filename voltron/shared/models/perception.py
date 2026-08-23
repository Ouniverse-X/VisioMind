"""Shared perception result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PerceptionObject:
    """Structured object detection output from the Vision agent."""

    name: str
    confidence: float
    attributes: dict[str, Any] = field(default_factory=dict)
    position: list[float] | None = None
    node_id: str | None = None


@dataclass
class PerceptionRelation:
    """Structured relation output from the Vision agent."""

    source: str
    target: str
    relation: str
    confidence: float = 1.0


@dataclass
class PerceptionReport:
    """Full structured perception report written to semantic memory."""

    objects: list[PerceptionObject] = field(default_factory=list)
    relations: list[PerceptionRelation] = field(default_factory=list)
    task_complete: bool = False
    raw_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
