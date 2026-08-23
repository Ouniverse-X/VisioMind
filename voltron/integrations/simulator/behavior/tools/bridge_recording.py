"""Recording and telemetry bridge helpers for the BEHAVIOR runtime facade."""

from __future__ import annotations

from typing import Any, Callable

from voltron.integrations.simulator.behavior.artifacts import process_logger as behavior_process_logger
from voltron.integrations.simulator.behavior.artifacts import recorder as behavior_recorder
from voltron.integrations.simulator.behavior.tools import runtime_inputs as behavior_runtime_inputs


def _fallback_runtime_callback(runtime: Any, name: str) -> Callable[..., Any]:
    callback = getattr(runtime, name, None)
    if not callable(callback):
        raise AttributeError(f"runtime callback {name!r} is not available")
    return callback


def _runtime_record_event_callback(runtime: Any) -> Callable[[str, dict[str, Any]], None]:
    callback = _fallback_runtime_callback(runtime, "_record_event")
    return lambda event, payload: callback(event, payload)


def start_recording(
    runtime: Any,
    *,
    request: Any,
    plan: Any,
    runs_root: Any,
    timestamp_factory: Callable[[], str],
    open_record_file: Callable[[Any], Any],
    start_recording_session: Callable[..., Any],
    safe_slug_fn: Callable[[str], str] | None = None,
    finalize_previous: Callable[[], None] | None = None,
    launch_watchdog: Callable[[], None] | None = None,
    install_exit_guard: Callable[[], None] | None = None,
    record_event_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> None:
    behavior_recorder.start_recording_runtime(
        runtime,
        request=request,
        plan=plan,
        runs_root=runs_root,
        timestamp_factory=timestamp_factory,
        safe_slug=safe_slug_fn or safe_slug,
        open_record_file=open_record_file,
        finalize_previous=finalize_previous or _fallback_runtime_callback(runtime, "_finalize_recording"),
        launch_watchdog=launch_watchdog or _fallback_runtime_callback(runtime, "_launch_transcode_watchdog"),
        install_exit_guard=install_exit_guard or _fallback_runtime_callback(runtime, "_install_recording_exit_guard"),
        record_event=record_event_callback or _runtime_record_event_callback(runtime),
        start_recording_session=start_recording_session,
    )


def finalize_recording(
    runtime: Any,
    *,
    finalize_recording_session: Callable[..., Any],
    transcode_recording: Callable[[], None] | None = None,
) -> None:
    behavior_recorder.finalize_recording_runtime(
        runtime,
        finalize_recording_session=finalize_recording_session,
        transcode_recording=transcode_recording or _fallback_runtime_callback(runtime, "_transcode_recording"),
    )


def transcode_recording(
    runtime: Any,
    *,
    which: Callable[[str], str | None],
    build_transcode_command: Callable[..., list[str]],
    run_command: Callable[[list[str]], Any],
    transcode_recording_impl: Callable[..., Any],
    remove_raw_recording: Callable[..., Any],
    record_event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    shorten_text_callback: Callable[[str], str] | None = None,
) -> None:
    behavior_recorder.transcode_recording_runtime(
        runtime,
        which=which,
        build_transcode_command=build_transcode_command,
        run_command=run_command,
        transcode_recording_impl=transcode_recording_impl,
        remove_raw_recording=remove_raw_recording,
        record_event=record_event_callback or _runtime_record_event_callback(runtime),
        shorten_text=shorten_text_callback or _fallback_runtime_callback(runtime, "_shorten_text"),
    )


def launch_transcode_watchdog(
    runtime: Any,
    *,
    python_executable: str,
    parent_pid: int,
    which: Callable[[str], str | None],
    build_watchdog_command: Callable[..., list[str]],
    memory_diagnostics_enabled: bool,
    popen: Callable[..., Any],
    launch_transcode_watchdog_impl: Callable[..., Any],
    record_event_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> bool:
    return bool(
        behavior_recorder.launch_transcode_watchdog_runtime(
            runtime,
            python_executable=python_executable,
            parent_pid=parent_pid,
            which=which,
            build_watchdog_command=build_watchdog_command,
            memory_diagnostics_enabled=memory_diagnostics_enabled,
            popen=popen,
            launch_transcode_watchdog_impl=launch_transcode_watchdog_impl,
            record_event=record_event_callback or _runtime_record_event_callback(runtime),
        )
    )


def install_recording_exit_guard(
    runtime: Any,
    *,
    atexit_register: Callable[..., Any],
    signal_getsignal: Callable[..., Any],
    signal_setsignal: Callable[..., Any],
    signal_numbers: list[Any],
    install_recording_exit_guard_impl: Callable[..., Any],
    close_from_exit_guard: Callable[[], None] | None = None,
    handle_termination_signal: Callable[[int, Any], None] | None = None,
) -> None:
    behavior_recorder.install_recording_exit_guard_runtime(
        runtime,
        atexit_register=atexit_register,
        signal_getsignal=signal_getsignal,
        signal_setsignal=signal_setsignal,
        signal_numbers=signal_numbers,
        install_recording_exit_guard_impl=install_recording_exit_guard_impl,
        close_from_exit_guard=close_from_exit_guard or _fallback_runtime_callback(runtime, "_close_from_exit_guard"),
        handle_termination_signal=handle_termination_signal or _fallback_runtime_callback(runtime, "_handle_termination_signal"),
    )


def remove_recording_exit_guard(
    runtime: Any,
    *,
    atexit_unregister: Callable[..., Any],
    signal_setsignal: Callable[..., Any],
    remove_recording_exit_guard_impl: Callable[..., Any],
    close_from_exit_guard: Callable[[], None] | None = None,
) -> None:
    behavior_recorder.remove_recording_exit_guard_runtime(
        runtime,
        atexit_unregister=atexit_unregister,
        signal_setsignal=signal_setsignal,
        remove_recording_exit_guard_impl=remove_recording_exit_guard_impl,
        close_from_exit_guard=close_from_exit_guard or _fallback_runtime_callback(runtime, "_close_from_exit_guard"),
    )


def close_from_exit_guard(
    runtime: Any,
    *,
    close_from_exit_guard_impl: Callable[..., Any],
    close_runtime: Callable[[], None] | None = None,
    finalize_recording: Callable[[], None] | None = None,
) -> None:
    behavior_recorder.close_from_exit_guard_runtime(
        runtime,
        close_from_exit_guard_impl=close_from_exit_guard_impl,
        close_runtime=close_runtime or runtime.close,
        finalize_recording=finalize_recording or _fallback_runtime_callback(runtime, "_finalize_recording"),
    )


def handle_termination_signal(
    runtime: Any,
    signum: int,
    frame: Any,
    *,
    signal_default: Any,
    signal_ignore: Any,
    signal_setsignal: Callable[..., Any],
    os_kill: Callable[..., Any],
    os_getpid: Callable[[], int],
    handle_termination_signal_impl: Callable[..., Any],
    record_event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    remove_recording_exit_guard_callback: Callable[[], None] | None = None,
    finalize_recording_callback: Callable[[], None] | None = None,
) -> None:
    behavior_recorder.handle_termination_signal_runtime(
        runtime,
        signum,
        frame,
        signal_default=signal_default,
        signal_ignore=signal_ignore,
        signal_setsignal=signal_setsignal,
        os_kill=os_kill,
        os_getpid=os_getpid,
        handle_termination_signal_impl=handle_termination_signal_impl,
        record_event=record_event_callback or _runtime_record_event_callback(runtime),
        remove_recording_exit_guard=remove_recording_exit_guard_callback
        or _fallback_runtime_callback(runtime, "_remove_recording_exit_guard"),
        finalize_recording=finalize_recording_callback or _fallback_runtime_callback(runtime, "_finalize_recording"),
    )


def record_event(*, record_file: Any, event: str, payload: dict[str, Any], write_event_record: Callable[..., Any]) -> None:
    write_event_record(
        record_file=record_file,
        event=event,
        payload=payload,
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
    record_event: Callable[[str, dict[str, Any]], None],
    emit_progress: Callable[[str], None],
) -> None:
    behavior_process_logger.log_progress_update(
        subtask=subtask,
        attempt=attempt,
        control_step=control_step,
        step_count=runtime._step_count,
        last_reward=runtime._last_reward,
        last_info=runtime._last_info,
        success_flag=success_flag,
        subtask_completed=subtask_completed,
        subtask_succeeded=subtask_succeeded,
        subtask_completion_reason=subtask_completion_reason,
        action=action,
        nav_state=(
            runtime._navigation_runtime_state.get(subtask.runtime_id)
            or runtime._navigation_runtime_state.get(subtask.subtask_id)
        ),
        progress_log_every=runtime.progress_log_every,
        done=done,
        record_event=record_event,
        emit_progress=emit_progress,
    )


def emit_progress(*, progress_log_every: int | None, message: str) -> None:
    if progress_log_every is None:
        return
    print(message, flush=True)


def normalize_progress_log_every(value: Any) -> int | None:
    return behavior_process_logger.normalize_progress_log_every(value)


def normalize_completion_reason(value: Any) -> str | None:
    return behavior_process_logger.normalize_completion_reason(value)


def subtask_failure_reason(completion_reason: str | None) -> str:
    return behavior_process_logger.subtask_failure_reason(completion_reason)


def summarize_sequence(value: Any, *, limit: int = 3) -> str:
    return behavior_process_logger.summarize_sequence(value, limit=limit)


def shorten_text(value: str, *, limit: int = 80) -> str:
    return behavior_process_logger.shorten_text(value, limit=limit)


def format_float(value: Any) -> str:
    return behavior_process_logger.format_float(value)


def summarize_action(action: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return behavior_process_logger.summarize_action(action)


def format_action_summary_text(summary: dict[str, dict[str, Any]]) -> str:
    return behavior_process_logger.format_action_summary_text(summary)


def record_frame(
    runtime: Any,
    obs: dict[str, Any],
    *,
    record_observation_frame: Callable[..., Any],
    call_env_method: Callable[[str], Any] | None = None,
    subtask_name: str | None = None,
    instruction: str | None = None,
    record_event_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> None:
    behavior_recorder.record_frame_runtime(
        runtime,
        obs,
        record_observation_frame=record_observation_frame,
        call_env_method=call_env_method or (lambda method_name: _fallback_runtime_callback(runtime, "_call_behavior_env_method")(method_name)),
        subtask_name=subtask_name if subtask_name is not None else _fallback_runtime_callback(runtime, "_recording_subtask_name")(),
        instruction=instruction if instruction is not None else _fallback_runtime_callback(runtime, "_recording_subtask_instruction")(),
        record_event=record_event_callback or _runtime_record_event_callback(runtime),
    )


def to_uint8_rgb(value: Any):
    return behavior_recorder.to_uint8_rgb(value)


def safe_slug(raw: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)
    slug = slug.strip("_")
    return slug or "task"


def state_snapshot(info: dict[str, Any]) -> dict[str, Any]:
    return behavior_runtime_inputs.state_snapshot(info)
