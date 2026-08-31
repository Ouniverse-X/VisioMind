from __future__ import annotations

import atexit
import os
import shutil
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from voltron.integrations.simulator.behavior.artifacts import recorder as behavior_recorder
from voltron.integrations.simulator.behavior.tools import (
    bridge_recording as behavior_bridge_recording,
)
from voltron.integrations.simulator.behavior.tools import (
    bridge_subtasks as behavior_bridge_subtasks,
)
from voltron.integrations.simulator.behavior.tools import (
    memory_diagnostics as behavior_memory_diagnostics,
)
from voltron.integrations.simulator.behavior.tools import (
    runtime_feedback as behavior_runtime_feedback,
)
from voltron.integrations.simulator.behavior.tools import (
    runtime_shutdown as behavior_runtime_shutdown,
)
from voltron.integrations.simulator.behavior.tools import transcode as behavior_transcode
from voltron.integrations.simulator.behavior.observation import (
    frames as behavior_observation_frames,
)
from voltron.runtime.session import episode_session as runtime_episode_session
from voltron.runtime.telemetry import artifact_writer as runtime_artifact_writer
from voltron.runtime.telemetry import process_trace as runtime_process_trace
from voltron.runtime.telemetry.process_verbosity import filter_process_event_for_verbosity
from voltron.runtime.telemetry import video_trace as runtime_video_trace


def record_event(runtime: Any, event: str, payload: dict[str, Any]) -> None:
    filtered = filter_process_event_for_verbosity(
        event=event,
        payload=payload,
        verbose=getattr(runtime, "logging_verbose", True),
        memory_diagnostics=getattr(runtime, "logging_memory_diagnostics", False),
    )
    if filtered is None:
        return
    event, payload = filtered

    record_file = runtime._record_file
    if record_file is None or getattr(record_file, "closed", False):
        record_file_path = getattr(runtime, "_record_file_path", None)
        if record_file_path is None:
            return
        try:
            with Path(record_file_path).open("a", encoding="utf-8") as handle:
                runtime_process_trace.write_event_record(
                    record_file=handle,
                    event=event,
                    payload=payload,
                )
        except Exception:
            pass
        return

    behavior_bridge_recording.record_event(
        record_file=record_file,
        event=event,
        payload=payload,
        write_event_record=runtime_process_trace.write_event_record,
    )


def emit_progress(runtime: Any, message: str) -> None:
    if not getattr(runtime, "logging_verbose", True) and message.startswith("[closed-loop] step "):
        return
    behavior_bridge_recording.emit_progress(
        progress_log_every=runtime.progress_log_every,
        message=message,
    )


def maybe_log_progress(
    runtime: Any,
    *,
    subtask: Any,
    attempt: int,
    control_step: int | None,
    action: dict[str, Any],
    done: bool,
    success_flag: bool,
    subtask_completed: bool,
    subtask_succeeded: bool,
    subtask_completion_reason: str | None,
) -> None:
    behavior_bridge_recording.maybe_log_progress(
        runtime,
        subtask=subtask,
        attempt=attempt,
        control_step=control_step,
        action=action,
        done=done,
        success_flag=success_flag,
        subtask_completed=subtask_completed,
        subtask_succeeded=subtask_succeeded,
        subtask_completion_reason=subtask_completion_reason,
        record_event=lambda event, payload: record_event(runtime, event, payload),
        emit_progress=lambda message: emit_progress(runtime, message),
    )


def transcode_recording(runtime: Any) -> None:
    raw_video_path = getattr(runtime, "_video_raw_path", None)
    output_video_path = getattr(runtime, "_video_path", None)
    if (
        getattr(runtime, "enable_transcode_watchdog", False)
        and getattr(runtime, "_transcode_watchdog_started", False)
        and raw_video_path is not None
        and output_video_path is not None
    ):
        try:
            raw_ready = raw_video_path.exists() and raw_video_path.stat().st_size > 0
            output_ready = output_video_path.exists() and output_video_path.stat().st_size > 0
        except Exception:
            raw_ready = False
            output_ready = False
        if raw_ready and not output_ready:
            behavior_memory_diagnostics.record_memory_diagnostic(
                lambda event, payload: record_event(runtime, event, payload),
                stage="video_transcode_deferred",
                extra={
                    "video_raw_path": str(raw_video_path),
                    "video_path": str(output_video_path),
                },
            )
            record_event(
                runtime,
                "video_transcode_deferred",
                {
                    "reason": "watchdog_enabled",
                    "video_raw_path": str(raw_video_path),
                    "video_path": str(output_video_path),
                },
            )
            return

    behavior_memory_diagnostics.record_memory_diagnostic(
        lambda event, payload: record_event(runtime, event, payload),
        stage="before_video_transcode",
        extra={
            "video_raw_path": str(raw_video_path) if raw_video_path is not None else None,
            "video_path": str(output_video_path) if output_video_path is not None else None,
        },
    )
    try:
        behavior_bridge_recording.transcode_recording(
            runtime,
            which=shutil.which,
            build_transcode_command=behavior_transcode.build_transcode_command,
            run_command=lambda command: subprocess.run(
                command,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=300,
            ),
            transcode_recording_impl=runtime_video_trace.transcode_video_trace,
            remove_raw_recording=runtime_video_trace.remove_raw_recording,
            record_event_callback=lambda event, payload: record_event(runtime, event, payload),
            shorten_text_callback=behavior_bridge_recording.shorten_text,
        )
    finally:
        behavior_memory_diagnostics.record_memory_diagnostic(
            lambda event, payload: record_event(runtime, event, payload),
            stage="after_video_transcode",
            extra={
                "video_raw_path": str(raw_video_path) if raw_video_path is not None else None,
                "video_path": str(output_video_path) if output_video_path is not None else None,
            },
        )


def launch_transcode_watchdog(runtime: Any) -> None:
    runtime._transcode_watchdog_started = bool(
        behavior_bridge_recording.launch_transcode_watchdog(
            runtime,
            python_executable=sys.executable,
            parent_pid=os.getpid(),
            which=shutil.which,
            build_watchdog_command=behavior_transcode.build_transcode_watchdog_command,
            memory_diagnostics_enabled=getattr(runtime, "logging_memory_diagnostics", False),
            popen=subprocess.Popen,
            launch_transcode_watchdog_impl=runtime_video_trace.launch_video_trace_watchdog,
            record_event_callback=lambda event, payload: record_event(runtime, event, payload),
        )
    )


def close_from_exit_guard(runtime: Any) -> None:
    behavior_bridge_recording.close_from_exit_guard(
        runtime,
        close_from_exit_guard_impl=behavior_recorder.close_from_exit_guard,
        close_runtime=runtime.close,
        finalize_recording=lambda: finalize_recording(runtime),
    )


def remove_recording_exit_guard(runtime: Any) -> None:
    behavior_bridge_recording.remove_recording_exit_guard(
        runtime,
        atexit_unregister=atexit.unregister,
        signal_setsignal=signal.signal,
        remove_recording_exit_guard_impl=behavior_recorder.remove_recording_exit_guard,
        close_from_exit_guard=lambda: close_from_exit_guard(runtime),
    )


def handle_termination_signal(runtime: Any, signum: int, frame: Any) -> None:
    behavior_bridge_recording.handle_termination_signal(
        runtime,
        signum,
        frame,
        signal_default=signal.SIG_DFL,
        signal_ignore=signal.SIG_IGN,
        signal_setsignal=signal.signal,
        os_kill=os.kill,
        os_getpid=os.getpid,
        handle_termination_signal_impl=behavior_recorder.handle_termination_signal,
        record_event_callback=lambda event, payload: record_event(runtime, event, payload),
        remove_recording_exit_guard_callback=lambda: remove_recording_exit_guard(runtime),
        finalize_recording_callback=lambda: finalize_recording(runtime),
    )


def install_recording_exit_guard(runtime: Any) -> None:
    behavior_bridge_recording.install_recording_exit_guard(
        runtime,
        atexit_register=atexit.register,
        signal_getsignal=signal.getsignal,
        signal_setsignal=signal.signal,
        signal_numbers=[getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)],
        install_recording_exit_guard_impl=behavior_recorder.install_recording_exit_guard,
        close_from_exit_guard=lambda: close_from_exit_guard(runtime),
        handle_termination_signal=lambda signum, frame: handle_termination_signal(
            runtime, signum, frame
        ),
    )


def finalize_recording(runtime: Any) -> None:
    behavior_bridge_recording.finalize_recording(
        runtime,
        finalize_recording_session=runtime_episode_session.finalize_episode_session,
        transcode_recording=lambda: transcode_recording(runtime),
    )


def start_recording(runtime: Any, request: Any, plan: Any, *, runtime_bridge_file: str) -> None:
    behavior_bridge_recording.start_recording(
        runtime,
        request=request,
        plan=plan,
        runs_root=Path(runtime_bridge_file).resolve().parents[1] / "runs",
        timestamp_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"),
        safe_slug_fn=behavior_bridge_recording.safe_slug,
        open_record_file=lambda path: runtime_artifact_writer.open_artifact_file(path, mode="a"),
        finalize_previous=lambda: finalize_recording(runtime),
        launch_watchdog=lambda: launch_transcode_watchdog(runtime),
        install_exit_guard=lambda: install_recording_exit_guard(runtime),
        record_event_callback=lambda event, payload: record_event(runtime, event, payload),
        start_recording_session=runtime_episode_session.start_episode_session,
    )


def record_frame(runtime: Any, obs: dict[str, Any]) -> None:
    behavior_bridge_recording.record_frame(
        runtime,
        obs,
        record_observation_frame=behavior_recorder.record_observation_frame,
        call_env_method=lambda method_name: runtime._call_behavior_env_method(method_name),
        subtask_name=behavior_bridge_subtasks.recording_subtask_name(
            active_internal_step=runtime._active_action_internal_step,
            last_info=runtime._last_info,
            active_subtask_name=runtime._active_subtask_name,
        ),
        instruction=behavior_bridge_subtasks.recording_subtask_instruction(
            active_internal_step=runtime._active_action_internal_step,
            active_subtask_instruction=runtime._active_subtask_instruction,
        ),
        record_event_callback=lambda event, payload: record_event(runtime, event, payload),
    )


def build_runtime_feedback(
    runtime: Any,
    *,
    subtask: Any | None = None,
    extras: dict[str, Any] | None = None,
) -> Any:
    feedback_extras = _completion_feedback_extras(runtime=runtime, extras=extras)
    return behavior_runtime_feedback.build_runtime_feedback(
        step_count=runtime._step_count,
        reward=runtime._last_reward,
        last_info=runtime._last_info,
        navigation_runtime_state=runtime._navigation_runtime_state,
        subtask=subtask,
        extras=feedback_extras,
    )


def _completion_feedback_extras(*, runtime: Any, extras: dict[str, Any] | None) -> dict[str, Any]:
    feedback_extras = dict(extras or {})
    if "images_b64" in feedback_extras:
        return feedback_extras
    last_obs = getattr(runtime, "_last_obs", None)
    if not isinstance(last_obs, dict):
        return feedback_extras
    images, image_view_order = behavior_observation_frames.extract_images_b64(last_obs)
    if not images:
        return feedback_extras
    if "third_person" not in image_view_order:
        third_person = getattr(runtime, "_last_recording_third_person_rgb", None)
        encoded_third_person = (
            behavior_observation_frames.encode_image_b64(third_person)
            if third_person is not None
            else None
        )
        if encoded_third_person:
            images.append(encoded_third_person)
            image_view_order.append("third_person")
    feedback_extras["images_b64"] = images[:4]
    if image_view_order:
        feedback_extras["image_view_order"] = image_view_order[:4]
    return feedback_extras


def task_succeeded(runtime: Any) -> bool:
    return behavior_runtime_feedback.task_succeeded(
        task_success=runtime._task_success,
        last_info=runtime._last_info,
    )


def build_summary(runtime: Any) -> dict[str, Any]:
    return behavior_runtime_feedback.build_runtime_summary(
        env_id=runtime.env_id,
        step_count=runtime._step_count,
        task_success=task_succeeded(runtime),
        last_info=runtime._last_info,
        terminated=runtime._terminated,
        truncated=runtime._truncated,
        closed=runtime._closed,
        record_dir=runtime._record_dir,
        record_file_path=runtime._record_file_path,
        video_path=runtime._video_path,
        video_raw_path=runtime._video_raw_path,
    )


def close(runtime: Any) -> None:
    result = behavior_runtime_shutdown.close_runtime_environment(
        closed=runtime._closed,
        env=runtime._env,
        step_count=runtime._step_count,
        task_success=task_succeeded(runtime),
        terminated=runtime._terminated,
        truncated=runtime._truncated,
        last_info=runtime._last_info,
        remove_recording_exit_guard=lambda: remove_recording_exit_guard(runtime),
        finalize_recording=lambda: finalize_recording(runtime),
        record_event=lambda event, payload: record_event(runtime, event, payload),
        record_memory_diagnostic=lambda stage,
        extra=None: behavior_memory_diagnostics.record_memory_diagnostic(
            lambda event, payload: record_event(runtime, event, payload),
            stage=stage,
            extra=extra,
        ),
    )
    runtime._closed = result["closed"]
    runtime._env = result["env"]


__all__ = [
    "build_runtime_feedback",
    "build_summary",
    "close",
    "close_from_exit_guard",
    "emit_progress",
    "finalize_recording",
    "handle_termination_signal",
    "install_recording_exit_guard",
    "launch_transcode_watchdog",
    "maybe_log_progress",
    "record_event",
    "record_frame",
    "remove_recording_exit_guard",
    "start_recording",
    "task_succeeded",
    "transcode_recording",
]
