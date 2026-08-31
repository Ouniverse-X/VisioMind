from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np

from voltron.integrations.simulator.behavior.artifacts import (
    process_logger as behavior_process_logger,
)


def to_policy_observation(obs: dict[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {}

    for key, value in obs.items():
        if key.startswith("video."):
            arr = behavior_process_logger.to_numpy(value)
            if arr is None:
                continue
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            while arr.ndim < 5:
                arr = np.expand_dims(arr, axis=0)
            converted[key] = arr
            continue

        if key.startswith("state."):
            arr = behavior_process_logger.to_numpy(value)
            if arr is None:
                continue
            arr = arr.astype(np.float32, copy=False)
            while arr.ndim < 3:
                arr = np.expand_dims(arr, axis=0)
            converted[key] = arr
            continue

        if key == "annotation.human.coarse_action":
            if isinstance(value, str):
                converted[key] = (value,)
            elif isinstance(value, (list, tuple)):
                converted[key] = tuple(str(item) for item in value)
            else:
                converted[key] = (str(value),)

    return converted


def extract_modal_frames(obs: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    frames: dict[str, Any] = {}
    for key, value in obs.items():
        if not key.startswith(prefix):
            continue
        frames[key.rsplit(".", 1)[-1]] = value
    return frames


def extract_images_b64(obs: dict[str, Any]) -> tuple[list[str], list[str]]:
    image_keys = sorted(key for key in obs.keys() if key.startswith("video.observation.images.rgb"))
    images: list[str] = []
    image_view_order: list[str] = []
    for key in image_keys:
        encoded = encode_image_b64(obs.get(key))
        if encoded:
            images.append(encoded)
            image_view_order.append(image_view_name_from_obs_key(key))

    if images:
        return images, image_view_order
    return [base64.b64encode(b"placeholder_image").decode("ascii")], []


def image_view_name_from_obs_key(key: str) -> str:
    normalized = str(key).strip().lower()
    if normalized.endswith("head_256_256"):
        return "head"
    if normalized.endswith("left_wrist_256_256"):
        return "left_wrist"
    if normalized.endswith("right_wrist_256_256"):
        return "right_wrist"
    if normalized.endswith("third_person_256_256"):
        return "third_person"
    tail = normalized.split(".")[-1]
    return tail.replace("_256_256", "")


def encode_image_b64(image: Any) -> str | None:
    arr = behavior_process_logger.to_numpy(image)
    if arr is None or arr.size == 0:
        return None

    if arr.dtype != "uint8":
        arr = arr.clip(0, 255).astype("uint8")

    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr.squeeze(axis=0)
    if arr.ndim == 4 and arr.shape[0] > 1:
        arr = arr[-1]
    if arr.ndim != 3:
        return base64.b64encode(arr.tobytes()).decode("ascii")

    try:
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(arr).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        return base64.b64encode(arr.tobytes()).decode("ascii")
