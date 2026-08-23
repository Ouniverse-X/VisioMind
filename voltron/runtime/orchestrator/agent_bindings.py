"""Shared agent binding helpers for orchestrator constructors."""

from __future__ import annotations

from typing import Any


def resolve_orchestrator_agents(
    *,
    brain_agent: Any | None = None,
    vision_agent: Any | None = None,
    navigation_agent: Any | None = None,
    action_agent: Any | None = None,
) -> tuple[Any, Any, Any, Any]:
    return (
        _require_agent_binding(brain_agent, canonical_name="brain_agent"),
        _require_agent_binding(vision_agent, canonical_name="vision_agent"),
        _require_agent_binding(navigation_agent, canonical_name="navigation_agent"),
        _require_agent_binding(action_agent, canonical_name="action_agent"),
    )


def _require_agent_binding(value: Any | None, *, canonical_name: str) -> Any:
    if value is None:
        raise TypeError(f"missing required agent binding: {canonical_name}")
    return value
