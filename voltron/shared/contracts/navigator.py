"""Navigation backend interface used by VLN."""

from __future__ import annotations

from typing import Any, Protocol


class NavigatorBackend(Protocol):
    """Scene-aware navigation backend owned by VLN."""

    def update(
        self,
        observation: dict[str, Any],
        *,
        pose: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update backend state from the latest runtime observation."""

    def ground_goal(
        self,
        instruction: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Ground a navigation instruction into a structured goal."""

    def generate_object_approach_candidates(
        self,
        *,
        start: dict[str, Any],
        goal: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return discrete reachable anchors for object-approach navigation tasks."""

    def plan_path(
        self,
        *,
        start: dict[str, Any],
        goal: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a structured path plan for the grounded goal."""
