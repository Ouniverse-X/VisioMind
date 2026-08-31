from __future__ import annotations

import importlib
from typing import Any


def build_run_end_payload(
    *,
    step_count: int,
    task_success: bool,
    terminated: bool,
    truncated: bool,
    last_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "step_count": step_count,
        "task_success": task_success,
        "terminated": terminated,
        "truncated": truncated,
        "task_progress": last_info.get("task_progress"),
    }


def reset_omnigibson_simulator() -> None:
    try:
        og = importlib.import_module("omnigibson")
    except ModuleNotFoundError:
        return

    sim = getattr(og, "sim", None)
    if sim is None:
        return

    clear = getattr(og, "clear", None)
    if callable(clear):
        clear()
        return

    stop = getattr(sim, "stop", None)
    if not callable(stop):
        return

    is_stopped = getattr(sim, "is_stopped", None)
    if callable(is_stopped) and is_stopped():
        return
    stop()


def close_runtime_environment(
    *,
    closed: bool,
    env: Any | None,
    step_count: int,
    task_success: bool,
    terminated: bool,
    truncated: bool,
    last_info: dict[str, Any],
    remove_recording_exit_guard: Any,
    finalize_recording: Any,
    record_event: Any,
    record_memory_diagnostic: Any | None = None,
) -> dict[str, Any]:
    def _record_memory(stage: str, **extra: Any) -> None:
        if record_memory_diagnostic is None:
            return
        record_memory_diagnostic(stage=stage, extra=extra or None)

    if closed:
        _record_memory("close_already_closed")
        remove_recording_exit_guard()
        _record_memory("before_finalize_recording_already_closed")
        finalize_recording()
        _record_memory("after_finalize_recording_already_closed")
        return {"closed": True, "env": env}

    _record_memory("before_run_end")
    record_event(
        "run_end",
        build_run_end_payload(
            step_count=step_count,
            task_success=task_success,
            terminated=terminated,
            truncated=truncated,
            last_info=last_info,
        ),
    )
    _record_memory("after_run_end")
    remove_recording_exit_guard()
    _record_memory("before_finalize_recording")
    try:
        finalize_recording()
        _record_memory("after_finalize_recording")
    finally:
        if env is not None:
            _record_memory("before_env_close")
            env.close()
            _record_memory("after_env_close")
        _record_memory("before_omnigibson_reset")
        reset_omnigibson_simulator()
        _record_memory("after_omnigibson_reset")
        env = None
    return {"closed": True, "env": env}
