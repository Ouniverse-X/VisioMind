"""Helpers for Nav2 profile resolution and runtime environment probing."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Mapping, Protocol


class Nav2ProfileLike(Protocol):
    profile_id: str
    ros_distro: str
    setup_script: str
    setup_script_candidates: tuple[str, ...]
    python_bin: str


def resolve_profile_setup_script(profile: Nav2ProfileLike) -> str:
    candidates = [
        *(candidate for candidate in profile.setup_script_candidates if candidate),
        profile.setup_script,
    ]
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        if Path(normalized).is_file():
            return normalized
    return str(profile.setup_script)


def prepend_path(value: str, existing: str | None) -> str:
    existing = (existing or "").strip()
    if not existing:
        return value
    parts = existing.split(":")
    if value in parts:
        return existing
    return f"{value}:{existing}"


def build_overlay_env(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    overlay_prefix = env.get("VOLTRON_NAV2_PREFIX", "").strip()
    overlay_python = env.get("VOLTRON_NAV2_PYTHONPATH", "").strip()
    if not overlay_prefix:
        return env

    env["AMENT_PREFIX_PATH"] = prepend_path(overlay_prefix, env.get("AMENT_PREFIX_PATH"))
    env["CMAKE_PREFIX_PATH"] = prepend_path(overlay_prefix, env.get("CMAKE_PREFIX_PATH"))
    env["COLCON_PREFIX_PATH"] = prepend_path(overlay_prefix, env.get("COLCON_PREFIX_PATH"))
    env["LD_LIBRARY_PATH"] = prepend_path(
        f"{overlay_prefix}/lib",
        env.get("LD_LIBRARY_PATH"),
    )
    if overlay_python:
        env["PYTHONPATH"] = prepend_path(overlay_python, env.get("PYTHONPATH"))
    return env


def inspect_runtime_environment(
    *,
    profile: Nav2ProfileLike,
    worker_script: str | Path,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    worker_path = Path(worker_script)
    env = build_overlay_env(base_env)
    resolved_setup_script = resolve_profile_setup_script(profile)
    command = (
        f"test -f {shlex.quote(resolved_setup_script)} && "
        f"source {shlex.quote(resolved_setup_script)} && "
        f"command -v ros2 >/dev/null && "
        f"{profile.python_bin} - <<'PY'\n"
        "import importlib.util, json\n"
        "from pathlib import Path\n"
        "try:\n"
        "  from ament_index_python.packages import get_package_prefix\n"
        "except Exception:\n"
        "  get_package_prefix = None\n"
        "def _pkg_binary(pkg, relpath):\n"
        "  if get_package_prefix is None:\n"
        "    return False\n"
        "  try:\n"
        "    prefix = Path(get_package_prefix(pkg))\n"
        "  except Exception:\n"
        "    return False\n"
        "  return (prefix / relpath).is_file()\n"
        "print(json.dumps({\n"
        "  'rclpy': bool(importlib.util.find_spec('rclpy')),\n"
        "  'nav2_msgs': bool(importlib.util.find_spec('nav2_msgs')),\n"
        "  'geometry_msgs': bool(importlib.util.find_spec('geometry_msgs')),\n"
        "  'nav2_planner_pkg': bool(importlib.util.find_spec('nav2_planner')),\n"
        "  'nav2_controller_pkg': bool(importlib.util.find_spec('nav2_controller')),\n"
        "  'planner_server_bin': _pkg_binary('nav2_planner', 'lib/nav2_planner/planner_server'),\n"
        "  'controller_server_bin': _pkg_binary('nav2_controller', 'lib/nav2_controller/controller_server'),\n"
        "  'lifecycle_manager_bin': _pkg_binary('nav2_lifecycle_manager', 'lib/nav2_lifecycle_manager/lifecycle_manager'),\n"
        "}))\n"
        "PY"
    )
    result = subprocess.run(
        ["bash", "-lc", command],
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
        env=env,
    )
    summary = {
        "profile_id": profile.profile_id,
        "ros_distro": profile.ros_distro,
        "setup_script": profile.setup_script,
        "setup_script_resolved": resolved_setup_script,
        "ros2_cli": result.returncode == 0,
        "worker_script": str(worker_path),
        "worker_exists": worker_path.is_file(),
        "overlay_prefix": env.get("VOLTRON_NAV2_PREFIX"),
        "overlay_pythonpath": env.get("VOLTRON_NAV2_PYTHONPATH"),
    }
    if result.returncode == 0:
        try:
            summary.update(json.loads(result.stdout.strip() or "{}"))
        except json.JSONDecodeError:
            summary["environment_stdout"] = result.stdout.strip()
    else:
        summary["environment_stderr"] = result.stderr.strip()
        summary.setdefault("rclpy", False)
        summary.setdefault("nav2_msgs", False)
        summary.setdefault("geometry_msgs", False)
    return summary
