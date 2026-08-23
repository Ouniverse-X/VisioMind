"""Episode-session helpers for runtime recording lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class EpisodeSession:
    """Active runtime episode recording session."""

    record_dir: Path
    record_file_path: Path
    record_file: Any
    video_path: Path
    video_raw_path: Path
    video_writer: Any | None = None
    closed: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "record_dir": self.record_dir,
            "record_file_path": self.record_file_path,
            "record_file": self.record_file,
            "video_path": self.video_path,
            "video_raw_path": self.video_raw_path,
            "video_writer": self.video_writer,
            "closed": self.closed,
        }


def start_episode_session(
    *,
    runs_root: Path,
    task_id: str,
    timestamp_factory: Callable[[], str],
    safe_slug: Callable[[str], str],
    open_record_file: Callable[[Path], Any],
    finalize_previous: Callable[[], None],
) -> EpisodeSession:
    finalize_previous()

    runs_root.mkdir(parents=True, exist_ok=True)
    run_name = f"{safe_slug(task_id)}_{timestamp_factory()}"
    record_dir = runs_root / run_name
    record_dir.mkdir(parents=True, exist_ok=True)

    record_file_path = record_dir / "process_data.jsonl"
    video_path = record_dir / "trajectory.mp4"
    video_raw_path = record_dir / "trajectory.avi"
    record_file = open_record_file(record_file_path)

    return EpisodeSession(
        record_dir=record_dir,
        record_file_path=record_file_path,
        record_file=record_file,
        video_path=video_path,
        video_raw_path=video_raw_path,
    )


def finalize_episode_session(
    *,
    video_writer: Any | None,
    record_file: Any | None,
    transcode_recording: Callable[[], None],
) -> dict[str, Any]:
    if video_writer is not None:
        try:
            video_writer.release()
        except Exception:
            pass

    transcode_recording()

    if record_file is not None:
        try:
            record_file.flush()
            record_file.close()
        except Exception:
            pass

    return {"video_writer": None, "record_file": None}
