from __future__ import annotations

from pathlib import Path


def build_transcode_command(
    *,
    ffmpeg_path: str,
    raw_video_path: Path,
    output_video_path: Path,
) -> list[str]:
    return [
        ffmpeg_path,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(raw_video_path),
        "-c:v",
        "libx264",
        "-threads",
        "1",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_video_path),
    ]


def build_transcode_watchdog_command(
    *,
    python_executable: str,
    parent_pid: int,
    raw_video_path: Path,
    output_video_path: Path,
    process_log_path: Path | None,
    ffmpeg_path: str,
    memory_diagnostics_enabled: bool = False,
) -> list[str]:
    process_log = str(process_log_path) if process_log_path is not None else ""
    memory_diagnostics = "1" if memory_diagnostics_enabled else "0"
    helper_code = (
        "import json, os, subprocess, sys, time\n"
        "from datetime import datetime\n"
        "from pathlib import Path\n"
        "ppid = int(sys.argv[1])\n"
        "raw = Path(sys.argv[2])\n"
        "mp4 = Path(sys.argv[3])\n"
        "log_path = Path(sys.argv[4]) if sys.argv[4] else None\n"
        "ffmpeg = sys.argv[5]\n"
        "memory_diagnostics_enabled = sys.argv[6] == '1'\n"
        "def record(event, payload):\n"
        "    if log_path is None:\n"
        "        return\n"
        "    try:\n"
        "        with log_path.open('a', encoding='utf-8') as handle:\n"
        "            handle.write(json.dumps({'ts': datetime.now().isoformat(timespec='seconds'), 'event': event, 'payload': payload}, ensure_ascii=False) + '\\n')\n"
        "    except Exception:\n"
        "        pass\n"
        "def memory_payload(stage, extra=None):\n"
        "    try:\n"
        "        from visiomind.action.integrations.simulator.behavior.tools import memory_diagnostics\n"
        "        return memory_diagnostics.build_memory_diagnostic_payload(stage=stage, extra=extra)\n"
        "    except Exception as exc:\n"
        "        payload = {'stage': stage, 'error': f'{type(exc).__name__}: {exc}', 'pid': os.getpid()}\n"
        "        if extra:\n"
        "            payload['extra'] = extra\n"
        "        return payload\n"
        "def record_memory(stage, extra=None):\n"
        "    if not memory_diagnostics_enabled:\n"
        "        return\n"
        "    record('memory_diagnostic', memory_payload(stage, extra))\n"
        "record_memory('video_transcode_watchdog_started', {'parent_pid': ppid, 'video_raw_path': str(raw), 'video_path': str(mp4)})\n"
        "while True:\n"
        "    try:\n"
        "        os.kill(ppid, 0)\n"
        "    except OSError:\n"
        "        break\n"
        "    time.sleep(1.0)\n"
        "stable_size = -1\n"
        "for _ in range(10):\n"
        "    size = raw.stat().st_size if raw.exists() else 0\n"
        "    if size > 0 and size == stable_size:\n"
        "        break\n"
        "    stable_size = size\n"
        "    time.sleep(1.0)\n"
        "if mp4.exists() and mp4.stat().st_size > 0:\n"
        "    record_memory('video_transcode_watchdog_mp4_already_ready', {'video_size': int(mp4.stat().st_size), 'video_path': str(mp4)})\n"
        "    sys.exit(0)\n"
        "if not raw.exists() or raw.stat().st_size == 0:\n"
        "    record_memory('video_transcode_watchdog_raw_missing', {'video_raw_path': str(raw), 'video_path': str(mp4)})\n"
        "    record('video_transcode_skipped', {'reason': 'raw_video_missing_after_exit', 'video_raw_path': str(raw), 'video_path': str(mp4)})\n"
        "    sys.exit(0)\n"
        "cmd = [\n"
        "    ffmpeg,\n"
        "    '-nostdin',\n"
        "    '-hide_banner',\n"
        "    '-loglevel',\n"
        "    'error',\n"
        "    '-y',\n"
        "    '-i',\n"
        "    str(raw),\n"
        "    '-c:v',\n"
        "    'libx264',\n"
        "    '-threads',\n"
        "    '1',\n"
        "    '-pix_fmt',\n"
        "    'yuv420p',\n"
        "    '-movflags',\n"
        "    '+faststart',\n"
        "    str(mp4),\n"
        "]\n"
        "record_memory('before_video_transcode_watchdog_ffmpeg', {'video_raw_path': str(raw), 'video_path': str(mp4), 'raw_size': int(raw.stat().st_size)})\n"
        "try:\n"
        "    completed = subprocess.run(\n"
        "        cmd,\n"
        "        check=False,\n"
        "        stdout=subprocess.DEVNULL,\n"
        "        stderr=subprocess.PIPE,\n"
        "        text=True,\n"
        "        timeout=300,\n"
        "    )\n"
        "except Exception as exc:\n"
        "    record_memory('after_video_transcode_watchdog_ffmpeg_exception', {'error': f'{type(exc).__name__}: {exc}', 'video_raw_path': str(raw), 'video_path': str(mp4)})\n"
        "    record('video_transcode_failed', {'error': f'{type(exc).__name__}: {exc}', 'video_raw_path': str(raw), 'video_path': str(mp4)})\n"
        "    sys.exit(0)\n"
        "record_memory('after_video_transcode_watchdog_ffmpeg', {'returncode': completed.returncode, 'video_raw_path': str(raw), 'video_path': str(mp4)})\n"
        "if completed.returncode != 0 or not mp4.exists() or mp4.stat().st_size == 0:\n"
        "    stderr = (completed.stderr or '').strip()\n"
        "    record('video_transcode_failed', {'returncode': completed.returncode, 'stderr': stderr[:240] or None, 'video_raw_path': str(raw), 'video_path': str(mp4)})\n"
        "    sys.exit(0)\n"
        "raw_deleted = False\n"
        "try:\n"
        "    raw.unlink()\n"
        "    raw_deleted = True\n"
        "except Exception:\n"
        "    raw_deleted = False\n"
        "record_memory('video_transcode_watchdog_succeeded', {'video_raw_path': str(raw), 'video_path': str(mp4), 'video_size': int(mp4.stat().st_size), 'raw_deleted': raw_deleted})\n"
        "record('video_transcode_succeeded', {'video_raw_path': str(raw), 'video_path': str(mp4), 'video_size': int(mp4.stat().st_size), 'source': 'watchdog', 'raw_deleted': raw_deleted})\n"
    )
    return [
        python_executable,
        "-c",
        helper_code,
        str(parent_pid),
        str(raw_video_path),
        str(output_video_path),
        process_log,
        ffmpeg_path,
        memory_diagnostics,
    ]
