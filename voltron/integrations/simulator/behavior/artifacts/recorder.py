from __future__ import annotations

from typing import Any, Callable

import numpy as np

from voltron.integrations.simulator.behavior.artifacts import (
    process_logger as behavior_process_logger,
)
from voltron.shared.context import Plan, TaskRequest


def draw_recording_overlay_text(
    *,
    frame: np.ndarray,
    cv2: Any,
    text: str,
    origin: tuple[int, int],
    font: int,
    font_scale: float,
    glow_thickness: int,
    main_thickness: int,
) -> None:
    line_type = getattr(cv2, "LINE_AA", 16)
    glow_color = (255, 210, 120)
    main_color = (160, 72, 12)
    cv2.putText(frame, text, origin, font, font_scale, glow_color, glow_thickness, line_type)
    cv2.putText(frame, text, origin, font, font_scale, main_color, main_thickness, line_type)


def annotate_recording_overlay_lines(
    frame: np.ndarray,
    *,
    cv2: Any,
    lines: list[str],
    video_scale: float = 1.0,
) -> np.ndarray:
    filtered_lines = [str(line).strip() for line in lines if str(line).strip()]
    if not filtered_lines or not hasattr(cv2, "putText"):
        return frame

    font = getattr(cv2, "FONT_HERSHEY_SIMPLEX", 0)
    overlay_scale = recording_overlay_inverse_scale(video_scale)
    font_scale = 0.42 * overlay_scale
    glow_thickness = max(1, int(round(2 * overlay_scale)))
    main_thickness = max(1, int(round(1 * overlay_scale)))
    margin = max(1, int(round(10 * overlay_scale)))
    line_gap = max(1, int(round(6 * overlay_scale)))
    line_sizes = [
        recording_text_box_size(
            text=line,
            cv2=cv2,
            font=font,
            font_scale=font_scale,
            thickness=main_thickness,
        )
        for line in filtered_lines
    ]
    valid_sizes = [(width, height) for width, height in line_sizes if width > 0 and height > 0]
    if not valid_sizes:
        return frame

    total_height = sum(height for _, height in valid_sizes) + (line_gap * (len(valid_sizes) - 1))
    baseline_y = max(valid_sizes[0][1], frame.shape[0] - margin - total_height + valid_sizes[0][1])

    for index, (line, (line_width, line_height)) in enumerate(
        zip(filtered_lines, line_sizes, strict=False)
    ):
        if line_width <= 0 or line_height <= 0:
            continue
        x = max(0, frame.shape[1] - line_width - margin)
        y = baseline_y + sum(height for _, height in valid_sizes[:index]) + (line_gap * index)
        draw_recording_overlay_text(
            frame=frame,
            cv2=cv2,
            text=line,
            origin=(x, y),
            font=font,
            font_scale=font_scale,
            glow_thickness=glow_thickness,
            main_thickness=main_thickness,
        )
    return frame


def compose_recording_frame(
    *,
    left: np.ndarray | None,
    right: np.ndarray | None,
    head: np.ndarray | None,
    third_person: np.ndarray | None,
    cv2: Any,
    overlay_lines: list[str] | None = None,
    video_scale: float = 1.0,
) -> np.ndarray | None:
    base_frame = head if head is not None else third_person
    if base_frame is None:
        base_frame = left if left is not None else right
    if base_frame is None:
        return None

    target_h, target_w = base_frame.shape[:2]
    upper_h = max(1, target_h // 2)
    lower_h = max(1, target_h - upper_h)
    wrist_box_size = max(1, min(target_w, upper_h, lower_h))

    left_top = resize_recording_frame_to_box(
        frame=center_crop_square_frame(left, size=wrist_box_size),
        width=wrist_box_size,
        height=wrist_box_size,
        cv2=cv2,
    )
    left_bottom = resize_recording_frame_to_box(
        frame=center_crop_square_frame(right, size=wrist_box_size),
        width=wrist_box_size,
        height=wrist_box_size,
        cv2=cv2,
    )
    left_column = np.vstack([left_top, left_bottom])
    first_person_box = resize_recording_frame_to_box(
        frame=head, width=target_w, height=target_h, cv2=cv2
    )
    third_person_box = resize_recording_frame_to_box(
        frame=third_person,
        width=target_w,
        height=target_h,
        cv2=cv2,
    )
    tiled_rgb = np.hstack([left_column, first_person_box, third_person_box])
    tiled_bgr = cv2.cvtColor(tiled_rgb, cv2.COLOR_RGB2BGR)
    if overlay_lines:
        return annotate_recording_overlay_lines(
            tiled_bgr, cv2=cv2, lines=overlay_lines, video_scale=video_scale
        )
    return tiled_bgr


def compose_recording_frame_from_observation(
    *,
    obs: dict[str, Any],
    cv2: Any,
    capture_third_person_rgb: Any,
    subtask_name: str | None,
    instruction: str | None,
    video_scale: float = 1.0,
) -> np.ndarray | None:
    left = to_uint8_rgb(obs.get("video.observation.images.rgb.left_wrist_256_256"))
    right = to_uint8_rgb(obs.get("video.observation.images.rgb.right_wrist_256_256"))
    head = to_uint8_rgb(obs.get("video.observation.images.rgb.head_256_256"))
    third_person = to_uint8_rgb(obs.get("video.observation.images.rgb.third_person_256_256"))
    if third_person is None:
        third_person = to_uint8_rgb(capture_third_person_rgb())
    overlay_lines: list[str] = []
    if subtask_name:
        overlay_lines.append(subtask_name)
    if instruction:
        overlay_lines.append(instruction)
    return compose_recording_frame(
        left=left,
        right=right,
        head=head,
        third_person=third_person,
        cv2=cv2,
        overlay_lines=overlay_lines,
        video_scale=video_scale,
    )


def recording_overlay_inverse_scale(video_scale: float) -> float:
    scale = max(0.05, min(1.0, float(video_scale)))
    return 1.0 / scale


def scale_recording_output_frame(
    frame: np.ndarray,
    *,
    cv2: Any,
    video_scale: float,
) -> np.ndarray:
    scale = max(0.05, min(1.0, float(video_scale)))
    if scale >= 1.0:
        return frame
    height, width = frame.shape[:2]
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (target_width, target_height))


def write_recording_frame(
    *,
    tiled_bgr: np.ndarray | None,
    video_writer: Any | None,
    video_raw_path: Any,
    cv2: Any,
    record_event: Any,
    video_scale: float = 1.0,
) -> Any | None:
    if tiled_bgr is None:
        return video_writer

    tiled_bgr = scale_recording_output_frame(tiled_bgr, cv2=cv2, video_scale=video_scale)

    writer = video_writer
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(
            str(video_raw_path), fourcc, 10.0, (tiled_bgr.shape[1], tiled_bgr.shape[0])
        )
        if not writer.isOpened():
            record_event(
                "video_writer_open_failed",
                {
                    "video_path": str(video_raw_path),
                    "frame_width": int(tiled_bgr.shape[1]),
                    "frame_height": int(tiled_bgr.shape[0]),
                },
            )
            writer.release()
            return None

    try:
        writer.write(tiled_bgr)
    except Exception:
        pass
    return writer


def record_observation_frame(
    *,
    obs: dict[str, Any],
    cv2: Any,
    video_writer: Any | None,
    video_raw_path: Any,
    capture_third_person_rgb: Any,
    subtask_name: str | None,
    instruction: str | None,
    record_event: Any,
    video_scale: float = 1.0,
) -> Any | None:
    tiled_bgr = compose_recording_frame_from_observation(
        obs=obs,
        cv2=cv2,
        capture_third_person_rgb=capture_third_person_rgb,
        subtask_name=subtask_name,
        instruction=instruction,
        video_scale=video_scale,
    )
    return write_recording_frame(
        tiled_bgr=tiled_bgr,
        video_writer=video_writer,
        video_raw_path=video_raw_path,
        cv2=cv2,
        record_event=record_event,
        video_scale=video_scale,
    )


def recording_text_box_size(
    *,
    text: str,
    cv2: Any,
    font: int,
    font_scale: float,
    thickness: int,
) -> tuple[int, int]:
    if hasattr(cv2, "getTextSize"):
        (width, height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        return int(width), int(height)
    approx_width = max(1, int(len(text) * 10 * max(font_scale, 0.1)))
    approx_height = max(1, int(18 * max(font_scale, 0.1)))
    return approx_width, approx_height


def resize_recording_frame_to_box(
    *,
    frame: np.ndarray | None,
    width: int,
    height: int,
    cv2: Any,
) -> np.ndarray:
    width = max(1, int(width))
    height = max(1, int(height))
    if frame is None:
        return np.zeros((height, width, 3), dtype=np.uint8)
    if frame.shape[:2] == (height, width):
        return frame
    return cv2.resize(frame, (width, height))


def center_crop_square_frame(frame: np.ndarray | None, *, size: int) -> np.ndarray | None:
    if frame is None:
        return None
    crop_size = max(1, min(int(size), frame.shape[0], frame.shape[1]))
    start_y = max(0, (frame.shape[0] - crop_size) // 2)
    start_x = max(0, (frame.shape[1] - crop_size) // 2)
    return frame[start_y : start_y + crop_size, start_x : start_x + crop_size]


def to_uint8_rgb(value: Any) -> np.ndarray | None:
    arr = behavior_process_logger.to_numpy(value)
    if arr is None or arr.size == 0:
        return None
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4:
        arr = arr[-1]
    if arr.ndim != 3:
        return None
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def install_recording_exit_guard(
    *,
    recording_exit_guard_installed: bool,
    previous_signal_handlers: dict[int, Any],
    close_from_exit_guard: Callable[[], None],
    handle_termination_signal: Callable[[int, Any], None],
    atexit_register: Callable[[Callable[[], None]], None],
    signal_getsignal: Callable[[int], Any],
    signal_setsignal: Callable[[int, Any], None],
    signal_numbers: list[Any],
) -> dict[str, Any]:
    next_installed = recording_exit_guard_installed
    next_handlers = dict(previous_signal_handlers)

    if not next_installed:
        atexit_register(close_from_exit_guard)
        next_installed = True

    if next_handlers:
        return {
            "recording_exit_guard_installed": next_installed,
            "previous_signal_handlers": next_handlers,
        }

    for signum in signal_numbers:
        if signum is None:
            continue
        try:
            next_handlers[signum] = signal_getsignal(signum)
            signal_setsignal(signum, handle_termination_signal)
        except (OSError, RuntimeError, ValueError):
            continue

    return {
        "recording_exit_guard_installed": next_installed,
        "previous_signal_handlers": next_handlers,
    }


def remove_recording_exit_guard(
    *,
    recording_exit_guard_installed: bool,
    previous_signal_handlers: dict[int, Any],
    close_from_exit_guard: Callable[[], None],
    atexit_unregister: Callable[[Callable[[], None]], None],
    signal_setsignal: Callable[[int, Any], None],
) -> dict[str, Any]:
    next_installed = recording_exit_guard_installed
    next_handlers = dict(previous_signal_handlers)

    if next_installed:
        try:
            atexit_unregister(close_from_exit_guard)
        except Exception:
            pass
        next_installed = False

    if not next_handlers:
        return {
            "recording_exit_guard_installed": next_installed,
            "previous_signal_handlers": {},
        }

    for signum, handler in list(next_handlers.items()):
        try:
            signal_setsignal(signum, handler)
        except (OSError, RuntimeError, ValueError):
            pass

    return {
        "recording_exit_guard_installed": next_installed,
        "previous_signal_handlers": {},
    }


def close_from_exit_guard(
    *,
    closed: bool,
    record_file: Any,
    video_writer: Any,
    close_runtime: Callable[[], None],
    finalize_recording: Callable[[], None],
) -> None:
    if closed and record_file is None and video_writer is None:
        return

    try:
        close_runtime()
    except Exception:
        finalize_recording()


def handle_termination_signal(
    *,
    signum: int,
    frame: Any,
    previous_signal_handlers: dict[int, Any],
    step_count: int,
    last_info: dict[str, Any],
    record_event: Callable[[str, dict[str, Any]], None],
    remove_recording_exit_guard: Callable[[], None],
    finalize_recording: Callable[[], None],
    signal_default: Any,
    signal_ignore: Any,
    signal_setsignal: Callable[[int, Any], None],
    os_kill: Callable[[int, int], None],
    os_getpid: Callable[[], int],
) -> None:
    previous_handler = previous_signal_handlers.get(signum, signal_default)
    record_event(
        "process_signal",
        {
            "signal": int(signum),
            "step_count": step_count,
            "task_progress": last_info.get("task_progress"),
        },
    )
    remove_recording_exit_guard()
    finalize_recording()

    if callable(previous_handler):
        previous_handler(signum, frame)
        return

    if previous_handler == signal_ignore:
        return

    signal_setsignal(signum, signal_default)
    os_kill(os_getpid(), signum)


def start_recording_runtime(
    adapter: Any,
    *,
    request: TaskRequest,
    plan: Plan,
    runs_root: Any,
    timestamp_factory: Callable[[], str],
    safe_slug: Callable[[str], str],
    open_record_file: Callable[[Any], Any],
    finalize_previous: Callable[[], None],
    launch_watchdog: Callable[[], None],
    install_exit_guard: Callable[[], None],
    record_event: Callable[[str, dict[str, Any]], None],
    start_recording_session: Callable[..., dict[str, Any]],
) -> None:
    session = start_recording_session(
        runs_root=runs_root,
        task_id=request.task_id,
        timestamp_factory=timestamp_factory,
        safe_slug=safe_slug,
        open_record_file=open_record_file,
        finalize_previous=finalize_previous,
    )
    if hasattr(session, "record_dir"):
        adapter._record_dir = session.record_dir
        adapter._record_file_path = session.record_file_path
        adapter._record_file = session.record_file
        adapter._video_path = session.video_path
        adapter._video_raw_path = session.video_raw_path
        adapter._video_writer = session.video_writer
        adapter._closed = bool(session.closed)
    else:
        adapter._record_dir = session["record_dir"]
        adapter._record_file_path = session["record_file_path"]
        adapter._record_file = session["record_file"]
        adapter._video_path = session["video_path"]
        adapter._video_raw_path = session["video_raw_path"]
        adapter._video_writer = session["video_writer"]
        adapter._closed = bool(session["closed"])

    launch_watchdog()
    install_exit_guard()
    record_event(
        "recording_initialized",
        {
            "record_dir": str(adapter._record_dir),
            "process_log": str(adapter._record_file_path),
            "video_path": str(adapter._video_path),
            "video_raw_path": str(adapter._video_raw_path),
            "video_scale": float(getattr(adapter, "recording_video_scale", 1.0)),
            "plan_subtasks": len(plan.subtasks),
        },
    )


def finalize_recording_runtime(
    adapter: Any,
    *,
    finalize_recording_session: Callable[..., dict[str, Any]],
    transcode_recording: Callable[[], None],
) -> None:
    session = finalize_recording_session(
        video_writer=adapter._video_writer,
        record_file=adapter._record_file,
        transcode_recording=transcode_recording,
    )
    adapter._video_writer = session["video_writer"]
    adapter._record_file = session["record_file"]


def transcode_recording_runtime(
    adapter: Any,
    *,
    which: Callable[[str], str | None],
    build_transcode_command: Callable[..., list[str]],
    run_command: Callable[[list[str]], Any],
    transcode_recording_impl: Callable[..., None],
    remove_raw_recording: Callable[[Any], None],
    record_event: Callable[[str, dict[str, Any]], None],
    shorten_text: Callable[[str], str],
) -> None:
    transcode_recording_impl(
        raw_video_path=adapter._video_raw_path,
        output_video_path=adapter._video_path,
        which=which,
        build_transcode_command=build_transcode_command,
        run_command=run_command,
        record_event=record_event,
        shorten_text=shorten_text,
        remove_raw_recording=remove_raw_recording,
    )


def launch_transcode_watchdog_runtime(
    adapter: Any,
    *,
    python_executable: str,
    parent_pid: int,
    which: Callable[[str], str | None],
    build_watchdog_command: Callable[..., list[str]],
    memory_diagnostics_enabled: bool,
    popen: Callable[..., Any],
    launch_transcode_watchdog_impl: Callable[..., bool],
    record_event: Callable[[str, dict[str, Any]], None],
) -> bool:
    return bool(
        launch_transcode_watchdog_impl(
            enabled=adapter.enable_transcode_watchdog,
            raw_video_path=adapter._video_raw_path,
            output_video_path=adapter._video_path,
            process_log_path=adapter._record_file_path,
            python_executable=python_executable,
            parent_pid=parent_pid,
            which=which,
            build_watchdog_command=build_watchdog_command,
            memory_diagnostics_enabled=memory_diagnostics_enabled,
            popen=popen,
            record_event=record_event,
        )
    )


def record_frame_runtime(
    adapter: Any,
    obs: dict[str, Any],
    *,
    record_observation_frame: Callable[..., Any],
    call_env_method: Callable[[str], Any],
    subtask_name: str | None,
    instruction: str | None,
    record_event: Callable[[str, dict[str, Any]], None],
) -> None:
    if adapter._record_dir is None or adapter._video_raw_path is None:
        return

    try:
        import cv2
    except Exception:
        return

    prefer_live_capture = bool(
        getattr(adapter, "env_kwargs", {}).get("recording_third_person_prefer_live_capture", False)
    )
    recording_obs = obs
    if prefer_live_capture:
        recording_obs = dict(obs)
        recording_obs.pop("video.observation.images.rgb.third_person_256_256", None)

    observed_third_person = obs.get("video.observation.images.rgb.third_person_256_256")
    if observed_third_person is not None:
        adapter._last_recording_third_person_rgb = observed_third_person

    def capture_third_person_rgb() -> Any:
        frame = call_env_method("capture_recording_third_person_rgb")
        if frame is not None:
            adapter._last_recording_third_person_rgb = frame
        return frame

    adapter._video_writer = record_observation_frame(
        obs=recording_obs,
        cv2=cv2,
        video_writer=adapter._video_writer,
        video_raw_path=adapter._video_raw_path,
        capture_third_person_rgb=capture_third_person_rgb,
        subtask_name=subtask_name,
        instruction=instruction,
        record_event=record_event,
        video_scale=getattr(adapter, "recording_video_scale", 1.0),
    )


def install_recording_exit_guard_runtime(
    adapter: Any,
    *,
    atexit_register: Callable[[Callable[[], None]], None],
    signal_getsignal: Callable[[Any], Any],
    signal_setsignal: Callable[[Any, Any], Any],
    signal_numbers: list[Any],
    install_recording_exit_guard_impl: Callable[..., dict[str, Any]],
    close_from_exit_guard: Callable[[], None],
    handle_termination_signal: Callable[[int, Any], None],
) -> None:
    installed = install_recording_exit_guard_impl(
        recording_exit_guard_installed=adapter._recording_exit_guard_installed,
        previous_signal_handlers=adapter._previous_signal_handlers,
        close_from_exit_guard=close_from_exit_guard,
        handle_termination_signal=handle_termination_signal,
        atexit_register=atexit_register,
        signal_getsignal=signal_getsignal,
        signal_setsignal=signal_setsignal,
        signal_numbers=signal_numbers,
    )
    adapter._recording_exit_guard_installed = installed["recording_exit_guard_installed"]
    adapter._previous_signal_handlers = installed["previous_signal_handlers"]


def remove_recording_exit_guard_runtime(
    adapter: Any,
    *,
    atexit_unregister: Callable[[Callable[[], None]], None],
    signal_setsignal: Callable[[Any, Any], Any],
    remove_recording_exit_guard_impl: Callable[..., dict[str, Any]],
    close_from_exit_guard: Callable[[], None],
) -> None:
    removed = remove_recording_exit_guard_impl(
        recording_exit_guard_installed=adapter._recording_exit_guard_installed,
        previous_signal_handlers=adapter._previous_signal_handlers,
        close_from_exit_guard=close_from_exit_guard,
        atexit_unregister=atexit_unregister,
        signal_setsignal=signal_setsignal,
    )
    adapter._recording_exit_guard_installed = removed["recording_exit_guard_installed"]
    adapter._previous_signal_handlers = removed["previous_signal_handlers"]


def close_from_exit_guard_runtime(
    adapter: Any,
    *,
    close_from_exit_guard_impl: Callable[..., None],
    close_runtime: Callable[[], None],
    finalize_recording: Callable[[], None],
) -> None:
    close_from_exit_guard_impl(
        closed=adapter._closed,
        record_file=adapter._record_file,
        video_writer=adapter._video_writer,
        close_runtime=close_runtime,
        finalize_recording=finalize_recording,
    )


def handle_termination_signal_runtime(
    adapter: Any,
    signum: int,
    frame: Any,
    *,
    signal_default: Any,
    signal_ignore: Any,
    signal_setsignal: Callable[[Any, Any], Any],
    os_kill: Callable[[int, int], Any],
    os_getpid: Callable[[], int],
    handle_termination_signal_impl: Callable[..., None],
    record_event: Callable[[str, dict[str, Any]], None],
    remove_recording_exit_guard: Callable[[], None],
    finalize_recording: Callable[[], None],
) -> None:
    if adapter._handling_termination_signal:
        return

    adapter._handling_termination_signal = True
    try:
        handle_termination_signal_impl(
            signum=signum,
            frame=frame,
            previous_signal_handlers=adapter._previous_signal_handlers,
            step_count=adapter._step_count,
            last_info=adapter._last_info,
            record_event=record_event,
            remove_recording_exit_guard=remove_recording_exit_guard,
            finalize_recording=finalize_recording,
            signal_default=signal_default,
            signal_ignore=signal_ignore,
            signal_setsignal=signal_setsignal,
            os_kill=os_kill,
            os_getpid=os_getpid,
        )
    finally:
        adapter._handling_termination_signal = False
