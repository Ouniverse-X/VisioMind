"""Subtask naming and instruction helpers for the BEHAVIOR runtime bridge."""

from __future__ import annotations

from typing import Any

from voltron.shared.enums import AgentName


def instruction_for_subtask(subtask: Any) -> str:
    for value in (subtask.parameters.get("instruction"), subtask.context.get("instruction")):
        if isinstance(value, str) and value.strip():
            return value.strip()

    action = " ".join(str(subtask.action).replace("_", " ").split()).strip() or "act"
    target_phrase = render_target_phrase(subtask.target)

    if subtask.agent == AgentName.NAVIGATION:
        return f"navigate to {target_phrase}" if target_phrase else action
    if subtask.agent == AgentName.VISION:
        if action.startswith(("verify", "check", "inspect", "observe")):
            return f"{action} {target_phrase}".strip()
        return f"observe {target_phrase}".strip() if target_phrase else action
    return f"{action} {target_phrase}".strip() if target_phrase else action


def render_target_phrase(target: dict[str, Any]) -> str:
    if not isinstance(target, dict) or not target:
        return ""

    room_instance = first_target_value_raw(target, ("room_name", "room_id"))
    if room_instance:
        return f"the {room_instance}"

    part = first_target_value(target, ("part", "button", "switch", "control"))
    base = first_target_value(
        target,
        (
            "object",
            "device",
            "appliance",
            "item",
            "target",
            "region",
            "location",
            "room",
            "surface",
            "container",
            "receptacle",
        ),
    )
    if part and base:
        return f"the {part} on the {base}"
    if part:
        return f"the {part}"
    if base:
        return f"the {base}"

    values: list[str] = []
    for raw in target.values():
        if raw is None:
            continue
        text = " ".join(str(raw).replace("_", " ").split()).strip()
        if text:
            values.append(text)
    if not values:
        return ""
    return f"the {' '.join(values)}"


def first_target_value(target: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = target.get(key)
        if value is None:
            continue
        text = " ".join(str(value).replace("_", " ").split()).strip()
        if text:
            return text
    return None


def first_target_value_raw(target: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = target.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def planned_subtask_name(subtask: Any, *, slugify: Any) -> str:
    action_slug = slugify(subtask.action)
    return f"{subtask.subtask_id}:{subtask.agent.value}:{action_slug}"


def env_subtask_name(last_info: dict[str, Any]) -> str | None:
    value = last_info.get("subtask_name")
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def recording_subtask_name(
    *,
    active_internal_step: dict[str, Any] | None,
    env_subtask_name: str | None,
    active_subtask_name: str | None,
) -> str | None:
    if isinstance(active_internal_step, dict):
        value = str(active_internal_step.get("display_name") or "").strip()
        if value:
            return value
    return env_subtask_name or active_subtask_name


def recording_subtask_instruction(
    *,
    active_internal_step: dict[str, Any] | None,
    active_subtask_instruction: str | None,
) -> str | None:
    if isinstance(active_internal_step, dict):
        value = str(active_internal_step.get("instruction") or "").strip()
        if value:
            return value
    if not isinstance(active_subtask_instruction, str):
        return None
    value = " ".join(active_subtask_instruction.split()).strip()
    return value or None


def resolved_subtask_name(
    *,
    subtask: Any,
    active_internal_step: dict[str, Any] | None,
    env_subtask_name: str | None,
    planned_subtask_name: str | None = None,
) -> str:
    if isinstance(active_internal_step, dict):
        value = str(active_internal_step.get("display_name") or "").strip()
        if value:
            return value
    return env_subtask_name or planned_subtask_name or f"{subtask.subtask_id}:{subtask.agent.value}:{subtask.action}"


def action_internal_display_name(payload: dict[str, Any]) -> str:
    internal_step_id = str(payload.get("internal_step_id") or "").strip()
    parent_id = str(payload.get("parent_subtask_id") or "").strip()
    if not parent_id and "." in internal_step_id:
        parent_id = internal_step_id.split(".", 1)[0]
    child_token = internal_step_id.split(".", 1)[1] if "." in internal_step_id else internal_step_id
    skill_id = str(payload.get("selected_skill_id") or payload.get("preferred_skill_id") or "").strip()
    parts = [token for token in (parent_id, child_token, skill_id) if token]
    return " | ".join(parts)


vla_internal_display_name = action_internal_display_name
