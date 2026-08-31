from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PerceptionObject:
    name: str
    confidence: float
    attributes: dict[str, Any] = field(default_factory=dict)
    position: list[float] | None = None
    node_id: str | None = None


@dataclass
class PerceptionRelation:
    source: str
    target: str
    relation: str
    confidence: float = 1.0


@dataclass
class PerceptionReport:
    objects: list[PerceptionObject] = field(default_factory=list)
    relations: list[PerceptionRelation] = field(default_factory=list)
    task_complete: bool = False
    raw_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
