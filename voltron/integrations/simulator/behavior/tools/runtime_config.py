from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from voltron.shared.enums import TaskType


def configure_tempdir() -> None:
    preferred_tmpdir = (
        os.environ.get("VOLTRON_TMPDIR") or "/mnt/data/huangyixuan/.cache/omnigibson/tmp"
    )
    tmpdir_path = Path(preferred_tmpdir).expanduser()
    tmpdir_path.mkdir(parents=True, exist_ok=True)
    for key in ("TMPDIR", "TMP", "TEMP"):
        os.environ.setdefault(key, str(tmpdir_path))


def configure_adapter(
    adapter: Any,
    *,
    env_id: str,
    env_kwargs: dict[str, Any] | None,
    env_factory: Callable[[], Any] | None,
    auto_register: bool,
    default_subtask_max_steps: int | None,
    progress_log_every: int | None,
    recording_video_scale: float,
    logging_verbose: bool,
    logging_memory_diagnostics: bool,
    enable_transcode_watchdog: bool | None,
    runtime_termination_use_environment_success_signal: bool,
    runtime_termination_environment_signal_policy: str,
    object_goal_distance_tolerance_m: float,
    object_goal_heading_tolerance_rad: float,
    extract_runtime_kwarg: Callable[[dict[str, Any], str], str | None],
    normalize_progress_log_every: Callable[[Any], int | None],
) -> None:
    runtime_kwargs = dict(env_kwargs or {})

    adapter.env_id = env_id
    adapter._scene_id = extract_runtime_kwarg(runtime_kwargs, "scene_id")
    adapter._hovsg_graph_root = extract_runtime_kwarg(runtime_kwargs, "hovsg_graph_root")
    adapter._hovsg_graph_path = extract_runtime_kwarg(runtime_kwargs, "hovsg_graph_path")
    adapter._hovsg_nav_graph_type = extract_runtime_kwarg(runtime_kwargs, "hovsg_nav_graph_type")
    adapter.env_kwargs = runtime_kwargs
    adapter.env_factory = env_factory
    adapter.auto_register = auto_register
    adapter.default_subtask_max_steps = default_subtask_max_steps
    adapter.progress_log_every = normalize_progress_log_every(progress_log_every)
    adapter.recording_video_scale = max(0.05, min(1.0, float(recording_video_scale)))
    adapter.logging_verbose = bool(logging_verbose)
    adapter.logging_memory_diagnostics = bool(logging_memory_diagnostics)
    adapter.runtime_termination_use_environment_success_signal = bool(
        runtime_termination_use_environment_success_signal
    )
    adapter.runtime_termination_environment_signal_policy = str(
        runtime_termination_environment_signal_policy or "allow_early_success"
    )
    adapter.enable_transcode_watchdog = (
        "PYTEST_CURRENT_TEST" not in os.environ
        if enable_transcode_watchdog is None
        else bool(enable_transcode_watchdog)
    )
    adapter.object_goal_distance_tolerance_m = max(
        0.05, float(abs(object_goal_distance_tolerance_m))
    )
    adapter.object_goal_heading_tolerance_rad = max(
        0.05, float(abs(object_goal_heading_tolerance_rad))
    )


def initialize_runtime_state(adapter: Any) -> None:
    adapter._env = None
    adapter._last_obs = {}
    adapter._last_info = {}
    adapter._last_reward = 0.0
    adapter._terminated = False
    adapter._truncated = False
    adapter._task_success = False
    adapter._step_count = 0
    adapter._closed = False
    adapter._registered = False
    adapter._record_dir = None
    adapter._record_file_path = None
    adapter._record_file = None
    adapter._video_path = None
    adapter._video_raw_path = None
    adapter._video_writer = None
    adapter._transcode_watchdog_started = False
    adapter._runtime_subtasks = []
    adapter._runtime_subtasks_by_id = {}
    adapter._active_subtask_name = None
    adapter._active_subtask_instruction = None
    adapter._navigation_runtime_state = {}
    adapter._logged_subtask_attempts = set()
    adapter._logged_action_internal_attempts = set()
    adapter._logged_action_internal_replans = set()
    adapter._root_task_instruction = ""
    adapter._policy_backend = ""
    adapter._task_type = TaskType.MANIPULATION
    adapter._hovsg_localizer = None
    adapter._recording_exit_guard_installed = False
    adapter._previous_signal_handlers = {}
    adapter._handling_termination_signal = False
    adapter._active_action_internal_step = None
    adapter._navigation_passable_door_overrides = {}
