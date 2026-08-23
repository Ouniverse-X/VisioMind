"""Debug helpers for the Voltron VLM service integration."""

from __future__ import annotations

import base64
import hashlib
import re
import shutil
from pathlib import Path

MAX_DEBUG_PATH_COMPONENT_BYTES = 120


def sanitize_path_component(value: str) -> str:
    cleaned = re.sub(r"[\\/]+", "_", value.strip())
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^\w.-]+", "_", cleaned)
    cleaned = cleaned.strip("._")
    return cleaned or "unknown"


def _bounded_path_component(value: str, max_bytes: int = MAX_DEBUG_PATH_COMPONENT_BYTES) -> str:
    cleaned = sanitize_path_component(value)
    if len(cleaned.encode("utf-8")) <= max_bytes:
        return cleaned

    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:10]
    suffix = f"_{digest}"
    prefix_budget = max(1, max_bytes - len(suffix.encode("utf-8")))
    prefix_chars: list[str] = []
    used = 0
    for char in cleaned:
        char_size = len(char.encode("utf-8"))
        if used + char_size > prefix_budget:
            break
        prefix_chars.append(char)
        used += char_size
    prefix = "".join(prefix_chars).strip("._") or cleaned[:1]
    return f"{prefix}{suffix}"


def save_debug_images(
    images_b64: list[str],
    root_dir: str | Path,
    task_name: str,
    instruction: str,
    image_view_order: list[str] | None = None,
) -> Path:
    target_dir = (
        Path(root_dir).expanduser().resolve()
        / _bounded_path_component(task_name)
        / _bounded_path_component(instruction)
    )
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    views = [str(item).strip() for item in image_view_order or []]
    for index, image_b64 in enumerate(images_b64):
        image_data = base64.b64decode(image_b64)
        view_suffix = ""
        if index < len(views) and views[index]:
            view_suffix = f"_{sanitize_path_component(views[index])}"
        file_path = target_dir / f"frame_{index:03d}{view_suffix}.jpg"
        file_path.write_bytes(image_data)

    return target_dir
