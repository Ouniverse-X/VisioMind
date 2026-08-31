from __future__ import annotations

import re
from typing import Any

from visiomind.action.integrations.simulator.behavior.tools.bridge_environment import (
    _collect_scene_objects,
)
from visiomind.action.shared.action_semantics import (
    is_open_state_action,
    normalize_action_name,
)
from visiomind.action.shared.models.scene_state import RuntimeDoorState, is_door_category

VISION_COMPLETION_SOURCE = "vision_completion_evaluator"
DEFAULT_CLOSED_OBSERVATION_STREAK = 3


def apply_completion_decision(
    runtime: Any,
    *,
    subtask: Any,
    decision: dict[str, Any],
) -> dict[str, Any] | None:
    verdict = decision.get("verdict")
    if not isinstance(verdict, dict):
        return None
    if not (
        decision.get("done") is True
        and decision.get("success") is not False
        and verdict.get("completed") is True
        and str(verdict.get("source") or "") == VISION_COMPLETION_SOURCE
    ):
        return None

    action = normalize_action_name(getattr(subtask, "action", ""))
    if not is_open_state_action(action):
        return None
    feedback = _feedback_payload(decision.get("feedback"), verdict=verdict)
    door_name = _resolve_door_name(
        runtime,
        subtask=subtask,
        feedback=feedback,
    )
    if door_name is None:
        return None

    overrides = _override_store(runtime)
    if action == "open":
        observed_at_step = _observed_at_step(feedback, verdict)
        diagnostics = _action_completion_diagnostics(runtime, feedback)
        physical_open_state = (
            diagnostics.get("open_state") if isinstance(diagnostics, dict) else None
        )
        overrides[door_name] = {
            "navigation_passable": True,
            "source": VISION_COMPLETION_SOURCE,
            "confidence": _optional_float(verdict.get("confidence")),
            "observed_at_step": observed_at_step,
            "revision": verdict.get("scope_key")
            or f"{verdict.get('scope', 'subtask')}:{verdict.get('scope_id', '')}",
            "physical_open_observed": (
                physical_open_state if isinstance(physical_open_state, bool) else False
            ),
            "closed_observation_streak": 0,
        }
        update = "latched"
    elif action == "close":
        override_key = _matching_override_key(overrides, door_name)
        if override_key is None:
            return None
        overrides.pop(override_key, None)
        door_name = override_key
        update = "cleared"
    else:
        return None

    _invalidate_scene_state_cache(runtime)
    return {
        "door_name": door_name,
        "update": update,
        "navigation_passable": update == "latched",
        "source": VISION_COMPLETION_SOURCE,
        "active_overrides": active_override_payload(runtime),
    }


def apply_runtime_override(
    runtime: Any,
    *,
    door: RuntimeDoorState,
    current_step: int,
    closed_observation_streak: int = DEFAULT_CLOSED_OBSERVATION_STREAK,
) -> RuntimeDoorState:
    overrides = _override_store(runtime, create=False)
    override_key = _matching_override_key(overrides, door.name)
    if override_key is None:
        return door
    override = overrides.get(override_key)
    if not isinstance(override, dict) or override.get("navigation_passable") is not True:
        return door

    if door.is_open is True:
        override["physical_open_observed"] = True
        override["closed_observation_streak"] = 0
    elif door.is_open is False and override.get("physical_open_observed") is True:
        streak = int(override.get("closed_observation_streak", 0)) + 1
        override["closed_observation_streak"] = streak
        if streak >= max(1, int(closed_observation_streak)):
            overrides.pop(override_key, None)
            return door
    else:
        override["closed_observation_streak"] = 0

    door.navigation_passable = True
    door.navigation_passable_source = str(override.get("source") or VISION_COMPLETION_SOURCE)
    observed_at_step = override.get("observed_at_step")
    if isinstance(observed_at_step, (int, float)):
        door.navigation_passable_observed_at_step = int(observed_at_step)
    else:
        door.navigation_passable_observed_at_step = int(current_step)
    revision = override.get("revision")
    if isinstance(revision, (str, int)):
        door.navigation_passable_revision = revision
    return door


def clear_navigation_passability_overrides(runtime: Any) -> None:
    runtime._navigation_passable_door_overrides = {}
    _invalidate_scene_state_cache(runtime)


def active_override_payload(runtime: Any) -> dict[str, dict[str, Any]]:
    overrides = _override_store(runtime, create=False)
    payload: dict[str, dict[str, Any]] = {}
    for name, value in overrides.items():
        if not isinstance(value, dict) or value.get("navigation_passable") is not True:
            continue
        payload[name] = {
            key: item
            for key, item in value.items()
            if key
            in {
                "navigation_passable",
                "source",
                "confidence",
                "observed_at_step",
                "revision",
                "physical_open_observed",
                "closed_observation_streak",
            }
            and item is not None
        }
    return payload


def _resolve_door_name(
    runtime: Any,
    *,
    subtask: Any,
    feedback: dict[str, Any],
) -> str | None:
    diagnostics = _action_completion_diagnostics(runtime, feedback)
    selected = diagnostics.get("selected_candidate") if isinstance(diagnostics, dict) else None
    if isinstance(selected, dict):
        selected_name = _optional_text(selected.get("name"))
        selected_category = selected.get("category")
        if selected_name and (
            is_door_category(selected_category)
            or _is_portal_text(f"{selected_name} {selected_category or ''}")
        ):
            return selected_name

    target = getattr(subtask, "target", None)
    target_payload = target if isinstance(target, dict) else {}
    parameters = getattr(subtask, "parameters", None)
    parameter_payload = parameters if isinstance(parameters, dict) else {}
    target_text = " ".join(
        str(value)
        for value in (
            target_payload.get("object_id"),
            target_payload.get("object_name"),
            target_payload.get("object"),
            target_payload.get("category"),
            target_payload.get("part"),
            parameter_payload.get("instruction"),
        )
        if value not in (None, "")
    )
    if not _is_portal_text(target_text):
        return None

    live_door_names = _live_door_names(runtime)
    for value in (
        target_payload.get("object_id"),
        target_payload.get("object_name"),
        target_payload.get("object"),
    ):
        candidate = _optional_text(value)
        if candidate is None:
            continue
        exact = next(
            (name for name in live_door_names if _normalize(name) == _normalize(candidate)),
            None,
        )
        if exact is not None:
            return exact
    meaningful_target = _meaningful_tokens(target_text) - {
        "door",
        "doorway",
        "gate",
        "gateway",
        "opening",
        "portal",
        "open",
        "close",
    }
    matched = [
        name
        for name in live_door_names
        if meaningful_target and meaningful_target <= _meaningful_tokens(name)
    ]
    if len(matched) == 1:
        return matched[0]
    if len(live_door_names) == 1:
        return live_door_names[0]
    return None


def _live_door_names(runtime: Any) -> list[str]:
    env = getattr(runtime, "_env", None)
    if env is None:
        return []
    names = []
    for obj in _collect_scene_objects(env):
        category = getattr(obj, "category", None) or getattr(obj, "class_name", None)
        if not is_door_category(category):
            continue
        name = _optional_text(getattr(obj, "name", None))
        if name:
            names.append(name)
    return sorted(set(names))


def _feedback_payload(value: Any, *, verdict: dict[str, Any]) -> dict[str, Any]:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            value = value.to_dict()
        except Exception:
            value = None
    if isinstance(value, dict):
        return dict(value)
    evidence = verdict.get("evidence")
    feedback = evidence.get("feedback") if isinstance(evidence, dict) else None
    return dict(feedback) if isinstance(feedback, dict) else {}


def _action_completion_diagnostics(
    runtime: Any,
    feedback: dict[str, Any],
) -> dict[str, Any]:
    diagnostics = feedback.get("action_completion_diagnostics")
    if isinstance(diagnostics, dict):
        return diagnostics
    last_info = getattr(runtime, "_last_info", None)
    diagnostics = (
        last_info.get("action_completion_diagnostics") if isinstance(last_info, dict) else None
    )
    return diagnostics if isinstance(diagnostics, dict) else {}


def _observed_at_step(feedback: dict[str, Any], verdict: dict[str, Any]) -> int | None:
    for value in (
        feedback.get("step_count"),
        (verdict.get("evidence") or {}).get("control_step")
        if isinstance(verdict.get("evidence"), dict)
        else None,
    ):
        if isinstance(value, (int, float)):
            return int(value)
    return None


def _override_store(runtime: Any, *, create: bool = True) -> dict[str, dict[str, Any]]:
    store = getattr(runtime, "_navigation_passable_door_overrides", None)
    if isinstance(store, dict):
        return store
    if not create:
        return {}
    store = {}
    runtime._navigation_passable_door_overrides = store
    return store


def _matching_override_key(
    overrides: dict[str, dict[str, Any]],
    door_name: str,
) -> str | None:
    if door_name in overrides:
        return door_name
    normalized = _normalize(door_name)
    matched = [name for name in overrides if _normalize(name) == normalized]
    return matched[0] if len(matched) == 1 else None


def _invalidate_scene_state_cache(runtime: Any) -> None:
    cache = getattr(runtime, "_scene_runtime_state_cache", None)
    if isinstance(cache, dict):
        cache["signature"] = ""


def _is_portal_text(value: Any) -> bool:
    tokens = _meaningful_tokens(value)
    return bool(tokens & {"door", "doorway", "gate", "gateway", "opening", "portal"})


def _meaningful_tokens(value: Any) -> set[str]:
    return {
        token
        for token in re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split()
        if token and not token.isdigit()
    }


def _normalize(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split())


def _optional_text(value: Any) -> str | None:
    return str(value).strip() if value not in (None, "") and str(value).strip() else None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "DEFAULT_CLOSED_OBSERVATION_STREAK",
    "active_override_payload",
    "apply_completion_decision",
    "apply_runtime_override",
    "clear_navigation_passability_overrides",
]
