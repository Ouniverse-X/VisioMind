from __future__ import annotations

from pathlib import Path
from typing import Any

from visiomind.action.shared.models import NavigationRuntimeState, RuntimeFeedback


def build_runtime_feedback(
    *,
    step_count: int,
    reward: float,
    last_info: dict[str, Any],
    navigation_runtime_state: dict[str, dict[str, Any]],
    subtask: Any | None = None,
    extras: dict[str, Any] | None = None,
) -> RuntimeFeedback:
    navigation: NavigationRuntimeState | None = None
    if subtask is not None:
        navigation = NavigationRuntimeState.from_value(
            navigation_runtime_state.get(subtask.runtime_id)
            or navigation_runtime_state.get(subtask.subtask_id)
        )
    feedback_extras = dict(extras or {})
    heartbeat = last_info.get("environment_vlm_heartbeat")
    if isinstance(heartbeat, dict) and "environment_vlm_heartbeat" not in feedback_extras:
        feedback_extras["environment_vlm_heartbeat"] = dict(heartbeat)

    return RuntimeFeedback(
        step_count=step_count,
        reward=reward,
        task_progress=last_info.get("task_progress", 0.0),
        current_room=last_info.get("current_room"),
        current_region=last_info.get("current_region"),
        room_id=last_info.get("room_id"),
        floor_id=last_info.get("floor_id"),
        pose=last_info.get("pose"),
        navigation=navigation,
        extras=feedback_extras,
    )


def task_succeeded(*, task_success: bool, last_info: dict[str, Any]) -> bool:
    return task_success or bool(last_info.get("success", False))


def build_runtime_summary(
    *,
    env_id: str,
    step_count: int,
    task_success: bool,
    last_info: dict[str, Any],
    terminated: bool,
    truncated: bool,
    closed: bool,
    record_dir: Path | None,
    record_file_path: Path | None,
    video_path: Path | None,
    video_raw_path: Path | None,
) -> dict[str, Any]:
    chosen_video_path: str | None = None
    if video_path and video_path.exists():
        chosen_video_path = str(video_path)
    elif video_raw_path and video_raw_path.exists():
        chosen_video_path = str(video_raw_path)

    return {
        "env_id": env_id,
        "step_count": step_count,
        "task_success": task_success,
        "terminated": terminated,
        "truncated": truncated,
        "task_progress": last_info.get("task_progress"),
        "last_info": last_info,
        "closed": closed,
        "record_dir": str(record_dir) if record_dir else None,
        "process_log": str(record_file_path) if record_file_path else None,
        "video_path": chosen_video_path,
    }
