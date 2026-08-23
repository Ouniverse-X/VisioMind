"""Working-memory helpers for the HEMS integration."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable


def get_working_state(*, working_memory: Any, serializer: Callable[[Any], Any]) -> dict[str, Any]:
    return serializer(working_memory.get_state())


def get_active_regions(*, working_memory: Any) -> list[str]:
    return sorted(str(region) for region in working_memory.get_active_regions())


def get_recent_observations(*, working_memory: Any, n: int, serializer: Callable[[Any], Any]) -> list[dict[str, Any]]:
    observations = working_memory.get_recent_observations(n=n)
    return [serializer(item) for item in observations]


def get_task_context(*, working_memory: Any, serializer: Callable[[Any], Any]) -> dict[str, Any]:
    return serializer(working_memory.get_task_context())


def update_task_context(
    *,
    working_memory: Any,
    updates: dict[str, Any],
    merge_dicts: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
    serializer: Callable[[Any], Any],
) -> dict[str, Any]:
    if hasattr(working_memory, "update_task_context"):
        updated = working_memory.update_task_context(deepcopy(updates))
    else:
        current = working_memory.get_task_context()
        updated = merge_dicts(current, deepcopy(updates))
        working_memory._task_context = updated
    return serializer(updated)


def record_working_observation(
    *,
    working_memory: Any,
    observation: dict[str, Any],
    serializer: Callable[[Any], Any],
) -> dict[str, Any]:
    payload = _compact_observation(deepcopy(observation))
    add_observation = getattr(working_memory, "add_observation", None)
    if not callable(add_observation):
        return {"recorded": False, "reason": "backend_missing_add_observation"}
    add_observation(payload)
    return {
        "recorded": True,
        "source": str(payload.get("source") or "runtime"),
        "observation": serializer(payload),
    }


def _compact_observation(value: dict[str, Any]) -> dict[str, Any]:
    heavy_keys = {
        "image",
        "images",
        "raw_image",
        "raw_images",
        "image_b64",
        "images_b64",
        "embedding",
        "embeddings",
        "vector",
        "path",
        "paths",
        "waypoints",
        "dense_waypoints",
        "global_waypoints",
        "action_sequence",
    }
    return {
        str(key): _compact_value(item)
        for key, item in value.items()
        if str(key) not in heavy_keys
    }


def _compact_value(value: Any) -> Any:
    if isinstance(value, str):
        return value[:240]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {
            str(key): _compact_value(item)
            for key, item in list(value.items())[:20]
            if str(key)
            not in {
                "image",
                "images",
                "raw_image",
                "raw_images",
                "embedding",
                "embeddings",
                "vector",
                "waypoints",
                "dense_waypoints",
                "global_waypoints",
                "action_sequence",
            }
        }
    if isinstance(value, (list, tuple)):
        return [_compact_value(item) for item in list(value)[:20]]
    return str(value)[:240]


def annotate_current_episode(
    *,
    working_memory: Any,
    get_task_context: Callable[[], dict[str, Any]],
    get_working_state: Callable[[], dict[str, Any]],
    annotate_episode: Callable[..., None],
) -> bool:
    episode = working_memory.get_current_episode() if hasattr(working_memory, "get_current_episode") else None
    if episode is None:
        return False

    annotate_episode(
        episode=episode,
        task_context=get_task_context(),
        working_state=get_working_state(),
    )
    return True
