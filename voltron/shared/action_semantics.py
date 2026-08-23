"""Shared semantic helpers for high-level interaction actions."""

from __future__ import annotations

from typing import Any

_ACTION_ALIASES = {
    "open_door": "open",
    "close_door": "close",
    "switch_on": "turn_on",
    "switch_off": "turn_off",
    "toggle": "toggle_on",
    "go_to": "move_to_interaction_pose",
    "move_to": "move_to_interaction_pose",
    "local_move": "move_to_interaction_pose",
    "local_reposition": "move_to_interaction_pose",
    "reposition": "move_to_interaction_pose",
    "position": "move_to_interaction_pose",
    "move_to_reachable_pose": "move_to_interaction_pose",
    "move_to_interaction_ready_pose": "move_to_interaction_pose",
}

_OPEN_STATE_ACTIONS = {"open", "close"}
_TOGGLE_STATE_ACTIONS = {"toggle_on", "toggle_off", "turn_on", "turn_off"}
_STATE_CHANGE_ACTIONS = _OPEN_STATE_ACTIONS | _TOGGLE_STATE_ACTIONS

_ACTION_VERBS = {
    "open": "Open",
    "close": "Close",
    "toggle_on": "Turn on",
    "toggle_off": "Turn off",
    "turn_on": "Turn on",
    "turn_off": "Turn off",
    "press": "Press",
    "push_button": "Press",
    "pick_up": "Pick up",
    "grasp": "Grasp",
    "lift": "Lift",
    "take": "Take",
    "place": "Place",
    "put_down": "Put down",
    "release": "Release",
    "move_to_interaction_pose": "Move to the interaction pose for",
    "align": "Align with",
    "approach": "Approach",
    "adjust_pose": "Adjust pose for",
    "step_back": "Step back from",
}


def normalize_action_name(action: Any) -> str:
    normalized = "_".join(str(action or "").strip().lower().replace("-", "_").split())
    return _ACTION_ALIASES.get(normalized, normalized)


def is_open_state_action(action: Any) -> bool:
    return normalize_action_name(action) in _OPEN_STATE_ACTIONS


def is_toggle_state_action(action: Any) -> bool:
    return normalize_action_name(action) in _TOGGLE_STATE_ACTIONS


def is_state_change_action(action: Any) -> bool:
    return normalize_action_name(action) in _STATE_CHANGE_ACTIONS


def action_instruction(
    *,
    action: Any,
    target: dict[str, Any] | None = None,
    part_name: str | None = None,
) -> str:
    canonical = normalize_action_name(action)
    target = target or {}
    object_name = str(target.get("object") or target.get("object_id") or "target object").strip()
    part = str(part_name or target.get("part") or "").strip()
    verb = _ACTION_VERBS.get(canonical, canonical.replace("_", " ").capitalize())

    if part:
        if canonical in _OPEN_STATE_ACTIONS:
            return f"{verb} the {object_name} {part}."
        if canonical in _TOGGLE_STATE_ACTIONS:
            return f"{verb} the {object_name} using the {part}."
        return f"{verb} the {part} on the {object_name}."
    return f"{verb} the {object_name}."


__all__ = [
    "action_instruction",
    "is_open_state_action",
    "is_state_change_action",
    "is_toggle_state_action",
    "normalize_action_name",
]
