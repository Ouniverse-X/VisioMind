"""Shared observation-context helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ObservationContext:
    """Normalized runtime observation envelope shared across agent/runtime helpers."""

    observation: dict[str, Any] = field(default_factory=dict)
    raw_observation: dict[str, Any] = field(default_factory=dict)
    scene_report: dict[str, Any] = field(default_factory=dict)
    navigation_state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_observation(
        cls,
        observation: dict[str, Any] | None,
        *,
        scene_report: dict[str, Any] | None = None,
        navigation_state: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ObservationContext":
        normalized = dict(observation or {})
        raw_observation = normalized.get("raw_observation")
        return cls(
            observation=normalized,
            raw_observation=dict(raw_observation) if isinstance(raw_observation, dict) else {},
            scene_report=dict(scene_report or {}),
            navigation_state=dict(navigation_state or {}),
            metadata=dict(metadata or {}),
        )

    @property
    def observation_keys(self) -> list[str]:
        return sorted(self.observation.keys())

    def to_payload(self) -> dict[str, Any]:
        return {
            "observation": dict(self.observation),
            "raw_observation": dict(self.raw_observation),
            "scene_report": dict(self.scene_report),
            "navigation_state": dict(self.navigation_state),
            "metadata": dict(self.metadata),
            "observation_keys": self.observation_keys,
        }
