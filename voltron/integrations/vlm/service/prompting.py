from __future__ import annotations

from typing import Any

from .models import VLMProcessRequest

_SYSTEM_PROMPT = (
    "You are the visual verification module for Voltron, a mobile humanoid robot with a wheeled "
    "base, two arms with grippers, and a head camera. Given multi-view image sequences and a "
    "subtask instruction, return JSON only with this schema: "
    '{"task_complete": bool, "summary": str, "scene_report": {"target_visible": bool, '
    '"target_part_visible": bool, "target_part_name": str}, "objects": [{"name": str, '
    '"confidence": float, "attributes": dict}], "relations": [{"source": str, '
    '"target": str, "relation": str, "confidence": float}]}. The summary must be short. '
    "If the subtask is complete, start summary "
    "with 'SUCCESS:'. Do not include markdown fences or extra commentary."
)


def build_system_prompt() -> str:
    return _SYSTEM_PROMPT


def build_prompt(request: VLMProcessRequest) -> str:
    view_lines = _describe_view_order(request)
    return (
        f"The current subtask is: {request.instruction}.\n"
        "The robot has a wheeled mobile base, two arms with grippers, and a head camera.\n"
        f"{view_lines}\n"
        f"Use all provided images to determine whether the robot has completed the subtask "
        f"'{request.instruction}'. Always return the required JSON object only.\n"
        "Field guidance:\n"
        "- target_visible: whether the target object is visible in any view.\n"
        "- target_part_visible: whether the relevant button / knob / switch / handle is visible.\n"
        "- target_part_name: the visible actionable part if known, else ''.\n"
        "- objects: include up to 5 important visible objects.\n"
        "- relations: include up to 5 useful spatial or state relations.\n"
        "- summary: short natural language justification."
    )


def _describe_view_order(request: VLMProcessRequest) -> str:
    if request.image_view_order:
        descriptions = []
        for index, view_name in enumerate(request.image_view_order, start=1):
            descriptions.append(f"Image {index} comes from the {_humanize_view_name(view_name)}")
        return "; ".join(descriptions) + "."
    return (
        "Images are provided in runtime order. Do not assume a fixed wrist/head layout unless it is "
        "explicitly stated in the image_view_order metadata."
    )


def _humanize_view_name(view_name: str) -> str:
    normalized = str(view_name).strip().lower()
    if normalized.startswith("composite:"):
        layout = str(view_name).split(":", 1)[1].strip()
        return f"composite multi-view image arranged as {layout}"
    mapping = {
        "head": "head camera",
        "left_wrist": "left wrist camera",
        "right_wrist": "right wrist camera",
        "third_person": "third-person camera",
    }
    return mapping.get(normalized, normalized.replace("_", " "))


def build_openai_messages(request: VLMProcessRequest) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    image_detail = _normalize_image_detail(request.image_detail)
    for image_b64 in request.images:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": image_detail},
            }
        )
    content.append({"type": "text", "text": build_prompt(request)})
    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": content},
    ]


def build_dashscope_messages(request: VLMProcessRequest) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image_b64 in request.images:
        content.append({"image": f"data:image/jpeg;base64,{image_b64}"})
    content.append({"text": build_prompt(request)})
    return [{"role": "user", "content": content}]


def _normalize_image_detail(value: str) -> str:
    normalized = str(value or "low").strip().lower()
    return normalized if normalized in {"low", "high", "auto"} else "low"
