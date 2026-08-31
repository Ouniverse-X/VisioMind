from __future__ import annotations

from typing import Any


def capture_reset_runtime_state(*, obs: Any, info: Any) -> dict[str, Any]:
    last_obs = dict(obs) if isinstance(obs, dict) else {}
    last_info = dict(info) if isinstance(info, dict) else {}
    return {
        "last_obs": last_obs,
        "last_info": last_info,
        "last_reward": 0.0,
        "terminated": False,
        "truncated": False,
        "task_success": bool(last_info.get("success", False)),
        "step_count": 0,
        "closed": False,
        "active_subtask_name": None,
        "active_subtask_instruction": None,
        "active_action_internal_step": None,
        "navigation_runtime_state": {},
        "logged_subtask_attempts": set(),
        "logged_action_internal_attempts": set(),
        "logged_action_internal_replans": set(),
    }
