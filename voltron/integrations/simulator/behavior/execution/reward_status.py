from __future__ import annotations

from typing import Any, Callable

from voltron.integrations.simulator.behavior.artifacts import (
    process_logger as behavior_process_logger,
)


def resolve_step_completion_state(
    *,
    last_info: dict[str, Any],
    terminated: bool,
    truncated: bool,
    control_step: int | None,
    subtask_max_steps: int,
) -> dict[str, Any]:
    updated_last_info = dict(last_info)
    success_flag = bool(updated_last_info.get("success", False))
    subtask_completed = bool(updated_last_info.get("subtask_completed", False))
    subtask_succeeded = bool(updated_last_info.get("subtask_succeeded", False))
    subtask_completion_reason = behavior_process_logger.normalize_completion_reason(
        updated_last_info.get("subtask_completion_reason")
    )

    if subtask_completion_reason in {"success", "vlm_success"}:
        subtask_succeeded = True

    if success_flag and not subtask_completed:
        subtask_completed = True
        subtask_succeeded = True
        subtask_completion_reason = subtask_completion_reason or "task_success"
        updated_last_info.update(
            {
                "subtask_completed": True,
                "subtask_succeeded": True,
                "subtask_completion_reason": subtask_completion_reason,
            }
        )

    if (
        subtask_completed
        and not (success_flag or subtask_succeeded)
        and subtask_completion_reason is None
        and not terminated
        and not truncated
        and control_step is not None
        and control_step >= subtask_max_steps
    ):
        subtask_completion_reason = "timeout"

    return {
        "last_info": updated_last_info,
        "success_flag": success_flag,
        "subtask_completed": subtask_completed,
        "subtask_succeeded": subtask_succeeded,
        "subtask_completion_reason": subtask_completion_reason,
    }


def apply_env_step_result(
    *,
    obs: Any,
    reward: Any,
    terminated: Any,
    truncated: Any,
    info: Any,
    step_count: int,
    task_success: bool,
    control_step: int,
    subtask_max_steps: int,
    resolve_step_completion_state: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    next_last_obs = dict(obs) if isinstance(obs, dict) else {}
    next_last_info = dict(info) if isinstance(info, dict) else {}
    next_last_reward = float(reward)
    next_terminated = bool(terminated)
    next_truncated = bool(truncated)
    next_step_count = int(step_count) + 1

    resolved = resolve_step_completion_state(
        last_info=next_last_info,
        terminated=next_terminated,
        truncated=next_truncated,
        control_step=control_step,
        subtask_max_steps=subtask_max_steps,
    )
    success_flag = bool(resolved["success_flag"])

    return {
        "last_obs": next_last_obs,
        "last_info": resolved["last_info"],
        "last_reward": next_last_reward,
        "terminated": next_terminated,
        "truncated": next_truncated,
        "step_count": next_step_count,
        "task_success": bool(task_success) or success_flag,
        "success_flag": success_flag,
        "subtask_completed": bool(resolved["subtask_completed"]),
        "subtask_succeeded": bool(resolved["subtask_succeeded"]),
        "subtask_completion_reason": resolved["subtask_completion_reason"],
    }


def settle_step_completion(
    *,
    last_info: dict[str, Any],
    terminated: bool,
    truncated: bool,
    task_success: bool,
    control_step: int,
    subtask_max_steps: int,
    apply_success_overrides: Callable[[dict[str, Any], bool], dict[str, Any]] | None,
    resolve_step_completion_state: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    resolved = resolve_step_completion_state(
        last_info=last_info,
        terminated=terminated,
        truncated=truncated,
        control_step=control_step,
        subtask_max_steps=subtask_max_steps,
    )

    settled_last_info = resolved["last_info"]
    settled_task_success = bool(task_success)
    if apply_success_overrides is not None:
        overridden = apply_success_overrides(settled_last_info, settled_task_success)
        settled_last_info = overridden["last_info"]
        settled_task_success = bool(overridden["task_success"])
        resolved = resolve_step_completion_state(
            last_info=settled_last_info,
            terminated=terminated,
            truncated=truncated,
            control_step=control_step,
            subtask_max_steps=subtask_max_steps,
        )

    return {
        "last_info": resolved["last_info"],
        "task_success": settled_task_success or bool(resolved["success_flag"]),
        "success_flag": bool(resolved["success_flag"]),
        "subtask_completed": bool(resolved["subtask_completed"]),
        "subtask_succeeded": bool(resolved["subtask_succeeded"]),
        "subtask_completion_reason": resolved["subtask_completion_reason"],
    }


def advance_runtime_step_state(
    *,
    obs: Any,
    reward: Any,
    terminated: Any,
    truncated: Any,
    info: Any,
    step_count: int,
    task_success: bool,
    control_step: int,
    subtask_max_steps: int,
    apply_success_overrides: Callable[[dict[str, Any], bool], dict[str, Any]] | None,
    resolve_step_completion_state: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    state = apply_env_step_result(
        obs=obs,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info=info,
        step_count=step_count,
        task_success=task_success,
        control_step=control_step,
        subtask_max_steps=subtask_max_steps,
        resolve_step_completion_state=resolve_step_completion_state,
    )
    if apply_success_overrides is None:
        return state

    overridden = apply_success_overrides(state["last_info"], bool(state["task_success"]))
    resolved = resolve_step_completion_state(
        last_info=overridden["last_info"],
        terminated=state["terminated"],
        truncated=state["truncated"],
        control_step=control_step,
        subtask_max_steps=subtask_max_steps,
    )
    return {
        "last_obs": state["last_obs"],
        "last_info": resolved["last_info"],
        "last_reward": state["last_reward"],
        "terminated": state["terminated"],
        "truncated": state["truncated"],
        "step_count": state["step_count"],
        "task_success": bool(overridden["task_success"]) or bool(resolved["success_flag"]),
        "success_flag": bool(resolved["success_flag"]),
        "subtask_completed": bool(resolved["subtask_completed"]),
        "subtask_succeeded": bool(resolved["subtask_succeeded"]),
        "subtask_completion_reason": resolved["subtask_completion_reason"],
    }
