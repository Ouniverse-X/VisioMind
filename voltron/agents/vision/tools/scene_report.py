"""Scene report helpers for the Vision agent."""

from __future__ import annotations

import re
from typing import Any

from voltron.shared.context import Subtask
from voltron.shared.enums import AgentName


def build_scene_report(*, report: Any, subtask: Subtask, task_complete: bool) -> dict[str, Any]:
    raw_response = report.metadata.get("raw_response", {}) if isinstance(report.metadata, dict) else {}
    raw_scene_candidate = raw_response.get("scene_report") if isinstance(raw_response, dict) else {}
    raw_scene = raw_scene_candidate if isinstance(raw_scene_candidate, dict) else {}

    target_labels = _target_labels(subtask)
    object_names = [str(obj.name).strip().lower() for obj in report.objects if getattr(obj, "name", None)]
    raw_text = str(report.raw_text or "").strip().lower()
    target_part_name = _target_part_name(subtask, raw_scene)

    target_visible = _coerce_bool(raw_scene.get("target_visible"))
    if target_visible is None:
        target_visible = _contains_target(target_labels, object_names, raw_text)

    target_part_visible = _coerce_bool(raw_scene.get("target_part_visible"))
    if target_part_visible is None:
        target_part_visible = _infer_target_part_visible(
            target_part_name=target_part_name,
            object_names=object_names,
            raw_text=raw_text,
        )

    return {
        "target_visible": bool(target_visible),
        "target_part_visible": bool(target_part_visible),
        "target_part_name": target_part_name,
        "task_complete": task_complete,
    }


def allow_task_complete(subtask: Subtask) -> bool:
    if subtask.agent != AgentName.VISION:
        return False
    return subtask.parameters.get("allow_task_complete") is True


def classify_error_code(exc: Exception) -> str:
    normalized = str(exc).strip().lower()
    if "timeout" in normalized or "timed out" in normalized:
        return "VLM_TIMEOUT"
    if "http 5" in normalized or "server error" in normalized:
        return "VLM_HTTP_ERROR"
    if "connection error" in normalized or "connection aborted" in normalized:
        return "VLM_CONNECTION_ERROR"
    return "VLM_PARSE_ERROR"


def _target_labels(subtask: Subtask) -> list[str]:
    labels: list[str] = []
    for key in ("object", "device", "item", "target", "part", "button", "switch", "control"):
        value = subtask.target.get(key)
        if not value:
            continue
        text = str(value).strip().lower()
        if text:
            labels.append(text)
            labels.extend(re.split(r"[\s_\-]+", text))
    return [label for label in dict.fromkeys(labels) if label]


def _target_part_name(subtask: Subtask, raw_scene: dict[str, Any]) -> str:
    value = raw_scene.get("target_part_name")
    if isinstance(value, str) and value.strip():
        return value.strip().lower()
    for key in ("part", "button", "switch", "control"):
        part = subtask.target.get(key)
        if isinstance(part, str) and part.strip():
            return part.strip().lower()
    return ""


def _contains_target(target_labels: list[str], object_names: list[str], raw_text: str) -> bool:
    if not target_labels:
        return bool(object_names)
    flattened_names = " ".join(object_names)
    return any(label in flattened_names or label in raw_text for label in target_labels)


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "visible", "reachable"}:
            return True
        if normalized in {"false", "no", "hidden", "unreachable"}:
            return False
    return None


def _infer_target_part_visible(*, target_part_name: str, object_names: list[str], raw_text: str) -> bool:
    if target_part_name:
        if target_part_name in raw_text:
            return True
        flattened_names = " ".join(object_names)
        if target_part_name in flattened_names:
            return True
    if any(token in raw_text for token in ("button", "switch", "knob", "dial", "handle")):
        return True
    return False
