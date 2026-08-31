from __future__ import annotations

from typing import Any, Protocol


class NavigatorBackend(Protocol):
    def update(
        self,
        observation: dict[str, Any],
        *,
        pose: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        pass

    def ground_goal(
        self,
        instruction: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        pass

    def generate_object_approach_candidates(
        self,
        *,
        start: dict[str, Any],
        goal: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        pass

    def plan_path(
        self,
        *,
        start: dict[str, Any],
        goal: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        pass
