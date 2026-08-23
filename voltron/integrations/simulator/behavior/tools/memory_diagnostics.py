"""Memory diagnostics for BEHAVIOR runtime lifecycle events."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable


PROCESS_MATCH_TERMS = (
    "ffmpeg",
    "trajectory.avi",
    "trajectory.mp4",
    "closed_loop",
    "omnigibson",
    "isaac",
    "kit",
    "vlm.server",
    "serve_b1k_dynamic",
    "openpi",
    "gr00t",
)


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _read_proc_kv(path: Path) -> dict[str, str]:
    text = _read_text(path)
    if not text:
        return {}
    values: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        values[key.strip()] = raw_value.strip()
    return values


def _status_kib(status: dict[str, str], key: str) -> int | None:
    value = status.get(key)
    if not value:
        return None
    first = value.split()[0]
    try:
        return int(first)
    except ValueError:
        return None


def _status_int(status: dict[str, str], key: str) -> int | None:
    value = status.get(key)
    if not value:
        return None
    first = value.split()[0]
    try:
        return int(first)
    except ValueError:
        return None


def _read_cmdline(pid_dir: Path) -> str:
    try:
        raw = (pid_dir / "cmdline").read_bytes()
    except Exception:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def _truncate(value: str, limit: int = 240) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def _process_payload(pid: int, pid_dir: Path) -> dict[str, Any] | None:
    status = _read_proc_kv(pid_dir / "status")
    if not status:
        return None
    cmdline = _read_cmdline(pid_dir)
    name = status.get("Name") or ""
    haystack = f"{name} {cmdline}".lower()
    if pid != os.getpid() and not any(term in haystack for term in PROCESS_MATCH_TERMS):
        return None
    return {
        "pid": pid,
        "ppid": _status_int(status, "PPid"),
        "name": name or None,
        "rss_kib": _status_kib(status, "VmRSS"),
        "hwm_kib": _status_kib(status, "VmHWM"),
        "vms_kib": _status_kib(status, "VmSize"),
        "threads": _status_int(status, "Threads"),
        "cmdline": _truncate(cmdline),
    }


def collect_matching_processes(*, limit: int = 20, proc_root: Path = Path("/proc")) -> list[dict[str, Any]]:
    processes: list[dict[str, Any]] = []
    try:
        children = list(proc_root.iterdir())
    except Exception:
        return processes
    for pid_dir in children:
        if not pid_dir.name.isdigit():
            continue
        payload = _process_payload(int(pid_dir.name), pid_dir)
        if payload is not None:
            processes.append(payload)
    processes.sort(key=lambda item: item.get("rss_kib") or 0, reverse=True)
    return processes[:limit]


def collect_system_memory(*, meminfo_path: Path = Path("/proc/meminfo")) -> dict[str, int | None]:
    meminfo = _read_proc_kv(meminfo_path)
    keys = (
        "MemTotal",
        "MemAvailable",
        "MemFree",
        "Buffers",
        "Cached",
        "SwapTotal",
        "SwapFree",
        "SwapCached",
    )
    return {f"{key.lower()}_kib": _status_kib(meminfo, key) for key in keys}


def collect_cgroup_memory(*, cgroup_root: Path = Path("/sys/fs/cgroup")) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    current = _read_text(cgroup_root / "memory.current")
    maximum = _read_text(cgroup_root / "memory.max")
    events = _read_text(cgroup_root / "memory.events")
    if current is not None:
        try:
            payload["current_bytes"] = int(current.strip())
        except ValueError:
            payload["current_bytes"] = current.strip()
    if maximum is not None:
        raw_maximum = maximum.strip()
        payload["max_bytes"] = raw_maximum
        if raw_maximum != "max":
            try:
                payload["max_bytes"] = int(raw_maximum)
            except ValueError:
                pass
    if events is not None:
        parsed_events: dict[str, int] = {}
        for line in events.splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                parsed_events[parts[0]] = int(parts[1])
            except ValueError:
                continue
        payload["events"] = parsed_events
    return payload


def build_memory_diagnostic_payload(
    *,
    stage: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    self_pid = os.getpid()
    self_status = _read_proc_kv(Path("/proc") / str(self_pid) / "status")
    payload: dict[str, Any] = {
        "stage": stage,
        "pid": self_pid,
        "self": {
            "rss_kib": _status_kib(self_status, "VmRSS"),
            "hwm_kib": _status_kib(self_status, "VmHWM"),
            "vms_kib": _status_kib(self_status, "VmSize"),
            "threads": _status_int(self_status, "Threads"),
        },
        "system": collect_system_memory(),
        "cgroup": collect_cgroup_memory(),
        "processes": collect_matching_processes(),
    }
    if extra:
        payload["extra"] = dict(extra)
    return payload


def record_memory_diagnostic(
    record_event: Callable[[str, dict[str, Any]], None],
    *,
    stage: str,
    extra: dict[str, Any] | None = None,
) -> None:
    try:
        payload = build_memory_diagnostic_payload(stage=stage, extra=extra)
    except Exception as exc:
        payload = {
            "stage": stage,
            "error": f"{type(exc).__name__}: {exc}",
        }
    record_event("memory_diagnostic", payload)
