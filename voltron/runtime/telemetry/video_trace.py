"""Video trace lifecycle helpers for runtime telemetry."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable


def _resolve_imageio_ffmpeg_path() -> str | None:
    try:
        import imageio_ffmpeg  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        path = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
    return str(path) if path else None


def resolve_ffmpeg_path(which: Callable[[str], str | None]) -> str | None:
    return which("ffmpeg") or _resolve_imageio_ffmpeg_path()


def remove_raw_recording(raw_video_path: Path | None) -> bool:
    if raw_video_path is None or not raw_video_path.exists():
        return False
    try:
        raw_video_path.unlink()
    except Exception:
        return False
    return True


def transcode_video_trace(
    *,
    raw_video_path: Path | None,
    output_video_path: Path | None,
    which: Callable[[str], str | None],
    build_transcode_command: Callable[..., list[str]],
    run_command: Callable[[list[str]], Any],
    record_event: Callable[[str, dict[str, Any]], None],
    shorten_text: Callable[[str, int], str],
    remove_raw_recording: Callable[[Path | None], bool],
) -> None:
    if raw_video_path is None or output_video_path is None:
        return
    if not raw_video_path.exists() or raw_video_path.stat().st_size == 0:
        return

    ffmpeg_path = resolve_ffmpeg_path(which)
    if not ffmpeg_path:
        record_event(
            "video_transcode_skipped",
            {
                "reason": "ffmpeg_missing",
                "video_raw_path": str(raw_video_path),
                "video_path": str(output_video_path),
            },
        )
        return

    command = build_transcode_command(
        ffmpeg_path=ffmpeg_path,
        raw_video_path=raw_video_path,
        output_video_path=output_video_path,
    )
    try:
        completed = run_command(command)
    except Exception as exc:
        record_event(
            "video_transcode_failed",
            {
                "error": f"{type(exc).__name__}: {exc}",
                "video_raw_path": str(raw_video_path),
                "video_path": str(output_video_path),
            },
        )
        return

    if completed.returncode != 0 or not output_video_path.exists() or output_video_path.stat().st_size == 0:
        stderr = (getattr(completed, "stderr", "") or "").strip()
        record_event(
            "video_transcode_failed",
            {
                "returncode": completed.returncode,
                "stderr": shorten_text(stderr, 240) if stderr else None,
                "video_raw_path": str(raw_video_path),
                "video_path": str(output_video_path),
            },
        )
        return

    raw_deleted = remove_raw_recording(raw_video_path)
    record_event(
        "video_transcode_succeeded",
        {
            "video_raw_path": str(raw_video_path),
            "video_path": str(output_video_path),
            "video_size": int(output_video_path.stat().st_size),
            "raw_deleted": raw_deleted,
        },
    )


def launch_video_trace_watchdog(
    *,
    enabled: bool,
    raw_video_path: Path | None,
    output_video_path: Path | None,
    process_log_path: Path | None,
    python_executable: str,
    parent_pid: int,
    which: Callable[[str], str | None],
    build_watchdog_command: Callable[..., list[str]],
    popen: Callable[..., Any],
    record_event: Callable[[str, dict[str, Any]], None],
    memory_diagnostics_enabled: bool = False,
) -> bool:
    if not enabled:
        return False
    if raw_video_path is None or output_video_path is None:
        return False

    ffmpeg_path = resolve_ffmpeg_path(which)
    if not ffmpeg_path:
        record_event(
            "video_transcode_watchdog_skipped",
            {
                "reason": "ffmpeg_missing",
                "video_raw_path": str(raw_video_path),
                "video_path": str(output_video_path),
            },
        )
        return False

    command = build_watchdog_command(
        python_executable=python_executable,
        parent_pid=parent_pid,
        raw_video_path=raw_video_path,
        output_video_path=output_video_path,
        process_log_path=process_log_path,
        ffmpeg_path=ffmpeg_path,
        memory_diagnostics_enabled=memory_diagnostics_enabled,
    )
    if not command:
        return False

    try:
        helper = popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except Exception as exc:
        record_event(
            "video_transcode_watchdog_failed",
            {
                "error": f"{type(exc).__name__}: {exc}",
                "video_raw_path": str(raw_video_path),
                "video_path": str(output_video_path),
            },
        )
        return False

    record_event(
        "video_transcode_watchdog_started",
        {
            "helper_pid": helper.pid,
            "video_raw_path": str(raw_video_path),
            "video_path": str(output_video_path),
        },
    )
    return True
