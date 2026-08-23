"""Canonical public entry boundary for the HOV-SG navigator facade."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from . import backend_state as hovsg_backend_state
from . import planning_flow as hovsg_planning_flow
from . import scene_runtime as hovsg_scene_runtime


def _normalize_text(value: str) -> str:
    return value.strip()


def _normalize_mapping(
    mapping: dict[str, Any] | None,
    *,
    strip_scene_id: bool = False,
) -> dict[str, Any] | None:
    if mapping is None:
        return None

    normalized = deepcopy(mapping)
    if strip_scene_id:
        scene_id = hovsg_backend_state.normalize_scene_id(normalized.get("scene_id"))
        if scene_id is None:
            normalized.pop("scene_id", None)
        else:
            normalized["scene_id"] = scene_id
    return normalized


def _normalize_update_observation(observation: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}

    scene_id = hovsg_backend_state.normalize_scene_id(observation.get("scene_id"))
    if scene_id is not None:
        normalized["scene_id"] = scene_id

    nav_feedback = observation.get("nav_feedback")
    if isinstance(nav_feedback, dict):
        normalized["nav_feedback"] = dict(nav_feedback)
    scene_state = observation.get("scene_state")
    if isinstance(scene_state, dict) and scene_state:
        normalized["scene_state"] = deepcopy(scene_state)

    return normalized


def load_scene(
    adapter: Any,
    scene_id: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return hovsg_scene_runtime.load_scene(
        adapter,
        hovsg_backend_state.normalize_scene_id(scene_id) or "",
        config=_normalize_mapping(config),
    )


def update(
    adapter: Any,
    observation: dict[str, Any],
    *,
    pose: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return hovsg_scene_runtime.update(
        adapter,
        _normalize_update_observation(observation),
        pose=_normalize_mapping(pose),
    )


def ground_goal(
    adapter: Any,
    instruction: str,
    *,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return hovsg_scene_runtime.ground_goal(
        adapter,
        _normalize_text(instruction),
        context=_normalize_mapping(context, strip_scene_id=True),
    )


def generate_object_approach_candidates(
    adapter: Any,
    *,
    start: dict[str, Any],
    goal: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return hovsg_planning_flow.generate_object_approach_candidates(
        adapter,
        start=_normalize_mapping(start, strip_scene_id=True) or {},
        goal=_normalize_mapping(goal, strip_scene_id=True) or {},
        context=_normalize_mapping(context, strip_scene_id=True),
    )


def plan_path(
    adapter: Any,
    *,
    start: dict[str, Any],
    goal: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return hovsg_planning_flow.plan_path(
        adapter,
        start=_normalize_mapping(start, strip_scene_id=True) or {},
        goal=_normalize_mapping(goal, strip_scene_id=True) or {},
        context=_normalize_mapping(context, strip_scene_id=True),
    )
