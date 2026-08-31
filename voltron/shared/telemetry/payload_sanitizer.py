from __future__ import annotations

from collections.abc import Mapping
from typing import Any

IMAGE_PAYLOAD_KEYS = {
    "image",
    "image_b64",
    "images_b64",
    "images",
    "image_views",
    "raw_image",
    "raw_images",
    "rgb",
}


def strip_image_payloads(value: Any) -> Any:
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text in IMAGE_PAYLOAD_KEYS:
                _add_image_summary(sanitized, key_text, item)
                continue
            sanitized[key_text] = strip_image_payloads(item)
        return sanitized
    if isinstance(value, list):
        return [strip_image_payloads(item) for item in value]
    if isinstance(value, tuple):
        return [strip_image_payloads(item) for item in value]
    return value


def _add_image_summary(payload: dict[str, Any], key: str, value: Any) -> None:
    if key in {"images_b64", "images", "image_views"} and isinstance(value, list):
        payload["image_count"] = len(value)
        return
    if key == "raw_images" and isinstance(value, list):
        payload["raw_images_count"] = len(value)
        return
    if key == "rgb" and isinstance(value, Mapping):
        payload["rgb_keys"] = sorted(str(item) for item in value.keys())
        return
    payload[f"{key}_omitted"] = True
