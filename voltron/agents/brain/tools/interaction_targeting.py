from __future__ import annotations

import re
from typing import Any

from voltron.agents.brain.tools import navigation_runtime
from voltron.shared.context import Subtask, TaskRequest
from voltron.shared.enums import AgentName


def interaction_target_hints(*, request: TaskRequest, subtasks: list[Subtask]) -> dict[str, str]:
    hints: dict[str, str] = {}
    metadata = request.metadata if isinstance(request.metadata, dict) else {}

    for key in (
        "object",
        "part",
        "room",
        "region",
        "room_name",
        "room_id",
        "room_label",
        "canonical_room_name",
    ):
        value = metadata.get(f"target_{key}") or metadata.get(key)
        if isinstance(value, str) and value.strip():
            hints[key] = value.strip()

    for agent in (AgentName.ACTION, AgentName.VISION, AgentName.NAVIGATION):
        for subtask in subtasks:
            if subtask.agent != agent:
                continue
            for key in (
                "object",
                "part",
                "room",
                "region",
                "room_name",
                "room_id",
                "room_label",
                "canonical_room_name",
            ):
                value = subtask.target.get(key)
                if key not in hints and isinstance(value, str) and value.strip():
                    hints[key] = value.strip()

    room_hint = infer_room_from_text(request.description)
    if room_hint and "room" not in hints and "region" not in hints:
        hints["room"] = room_hint

    object_hint = infer_object_from_text(request.description)
    if object_hint and "object" not in hints:
        hints["object"] = object_hint

    canonical_room_name = navigation_runtime.first_non_empty(
        hints.get("canonical_room_name"),
        navigation_runtime.canonical_room_name(hints.get("room_name")),
    )
    if isinstance(canonical_room_name, str) and canonical_room_name.strip():
        hints["canonical_room_name"] = canonical_room_name.strip()

    room_label = navigation_runtime.room_display_label(
        room=hints.get("room"),
        region=hints.get("region"),
        canonical_room_name_value=hints.get("canonical_room_name"),
        room_name=hints.get("room_name"),
        room_label=hints.get("room_label"),
    )
    if isinstance(room_label, str) and room_label.strip():
        hints["room_label"] = room_label.strip()

    return hints


def infer_room_from_text(task_description: str) -> str | None:
    text = task_description.strip()
    patterns = (
        r"(?:导航到|前往|去到|去往|去)(?P<room>.+?)(?:并|然后|再|后|，|,|。|$)",
        r"(?:go to|head to|navigate to|move to)\s+(?:the\s+)?(?P<room>.+?)(?:\s+and\s+|\s+then\s+|[,.]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            room = match.group("room").strip()
            if room:
                return room
    return None


def infer_object_from_text(task_description: str) -> str | None:
    text = task_description.strip()
    lowered = text.lower()
    cn_match = re.search(r"(打开|开启|关闭)(?P<object>.+)$", text)
    if cn_match:
        return cn_match.group("object").strip() or None
    en_match = re.search(
        r"(turn on|turn off|switch on|switch off|open|close)\s+(?:the\s+)?(?P<object>.+)$",
        lowered,
    )
    if en_match:
        value = en_match.group("object").strip()
        for suffix in (
            " in the living room",
            " in the kitchen",
            " in the bedroom",
            " in the bathroom",
        ):
            if value.endswith(suffix):
                value = value[: -len(suffix)].strip()
        return value or None
    return None


def interaction_seed_instruction(target: dict[str, Any]) -> str:
    target_name = ""
    for key in ("object", "device", "target", "item", "region", "room"):
        value = target.get(key)
        if isinstance(value, str) and value.strip():
            target_name = value.strip()
            break

    if target_name:
        return (
            f"Inspect the {target_name} and determine whether the object or target part is visible."
        )
    return "Inspect the target scene and determine whether the object or target part is visible."
