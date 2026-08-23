"""Process-logging and progress helpers for the BEHAVIOR runtime bridge."""

from __future__ import annotations

from typing import Any

import numpy as np
from voltron.runtime.telemetry.process_trace import build_event_record, write_event_record  # noqa: F401
from voltron.runtime.telemetry.navigation_payloads import summarize_navigation_progress_entry


def _subtask_identity_payload(subtask: Any) -> dict[str, Any]:
    execution_id = getattr(subtask, "execution_id", None)
    if not execution_id:
        return {}
    return {
        "execution_id": str(execution_id),
        "plan_revision": int(getattr(subtask, "plan_revision", 0)),
    }


def build_subtask_start_payload(
    *,
    subtask: Any,
    attempt: int,
    instruction: str,
    max_steps: int,
) -> dict[str, Any]:
    return {
        "subtask_id": subtask.subtask_id,
        **_subtask_identity_payload(subtask),
        "agent": subtask.agent.value,
        "action": subtask.action,
        "attempt": attempt,
        "instruction": instruction,
        "target": dict(subtask.target),
        "max_steps": max_steps,
    }


def build_vla_internal_plan_created_payload(
    *,
    subtask_id: str,
    active_internal: dict[str, Any],
    execution_plan: dict[str, Any],
    progress: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "subtask_id": subtask_id,
        "parent_subtask_id": active_internal.get("parent_subtask_id"),
        "goal_summary": execution_plan.get("goal_summary"),
        "total_steps": progress.get("total_steps"),
        "source": execution_plan.get("source"),
    }
    steps = execution_plan.get("steps")
    if isinstance(steps, list):
        payload["steps"] = [dict(step) if isinstance(step, dict) else step for step in steps]
    metadata = execution_plan.get("metadata")
    if isinstance(metadata, dict):
        payload["metadata"] = dict(metadata)
    return payload


def build_vla_internal_replan_payload(
    *,
    subtask_id: str,
    active_internal: dict[str, Any],
    execution_plan: dict[str, Any],
    progress: dict[str, Any],
    replan_entry: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "subtask_id": subtask_id,
        "parent_subtask_id": active_internal.get("parent_subtask_id"),
        "active_step_id": replan_entry.get("active_step_id"),
        "reason": replan_entry.get("reason"),
        "replacement_step_ids": list(replan_entry.get("replacement_step_ids") or []),
        "goal_summary": execution_plan.get("goal_summary"),
        "total_steps": progress.get("total_steps"),
        "replan_entry": dict(replan_entry),
    }
    metadata = replan_entry.get("metadata")
    if isinstance(metadata, dict):
        payload["metadata"] = dict(metadata)
    steps = execution_plan.get("steps")
    if isinstance(steps, list):
        payload["steps"] = [dict(step) if isinstance(step, dict) else step for step in steps]
    return payload


def build_vla_internal_step_start_payload(
    *,
    subtask_id: str,
    attempt: int,
    control_step: int | None,
    active_internal: dict[str, Any],
) -> dict[str, Any]:
    return {
        "subtask_id": subtask_id,
        "parent_subtask_id": active_internal.get("parent_subtask_id"),
        "internal_step_id": active_internal.get("internal_step_id"),
        "name": active_internal.get("name"),
        "instruction": active_internal.get("instruction"),
        "action": active_internal.get("action"),
        "attempt": attempt,
        "control_step": control_step,
        "step_index": active_internal.get("step_index"),
        "total_steps": active_internal.get("total_steps"),
        "skill_id": active_internal.get("selected_skill_id") or active_internal.get("preferred_skill_id"),
        "display_name": active_internal.get("display_name"),
    }


def build_vla_internal_step_end_payload(
    *,
    subtask_id: str,
    attempt: int,
    control_step: int | None,
    step_count: int,
    reward: Any,
    terminated: bool,
    truncated: bool,
    success_flag: bool,
    active_internal: dict[str, Any],
    action_progress: dict[str, Any] | None,
) -> dict[str, Any]:
    progress = action_progress or {}
    return {
        "subtask_id": subtask_id,
        "parent_subtask_id": active_internal.get("parent_subtask_id"),
        "internal_step_id": active_internal.get("internal_step_id"),
        "attempt": attempt,
        "control_step": control_step,
        "step_count": step_count,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "plan_completed": progress.get("plan_completed", False),
        "completed_step_ids": list(progress.get("completed_step_ids", [])),
        "pending_step_ids": list(progress.get("pending_step_ids", [])),
        "task_success": success_flag,
    }


def build_env_step_payload(
    *,
    subtask: Any,
    attempt: int,
    control_step: int | None,
    step_count: int,
    reward: Any,
    terminated: bool,
    truncated: bool,
    last_info: dict[str, Any],
    success_flag: bool,
    subtask_completed: bool,
    subtask_succeeded: bool,
    subtask_completion_reason: str | None,
    resolved_subtask_name: str,
    env_subtask_name: str | None,
    active_internal: dict[str, Any] | None,
    action_progress: dict[str, Any] | None,
    motion_diagnostics: dict[str, Any] | None = None,
    environment_success_evidence_only: bool = False,
) -> dict[str, Any]:
    payload = {
        "subtask_id": subtask.subtask_id,
        **_subtask_identity_payload(subtask),
        "agent": subtask.agent.value,
        "attempt": attempt,
        "control_step": control_step,
        "step_count": step_count,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "task_progress": last_info.get("task_progress", 0.0),
        "task_success": success_flag,
        "subtask_completed": subtask_completed,
        "subtask_succeeded": subtask_succeeded,
        "subtask_completion_reason": subtask_completion_reason,
        "subtask_name": resolved_subtask_name,
        "env_subtask_name": env_subtask_name,
        "action_internal_step_id": (active_internal or {}).get("internal_step_id"),
        "action_internal_step_name": (active_internal or {}).get("name"),
        "action_internal_step_index": (active_internal or {}).get("step_index"),
        "action_plan_completed": (action_progress or {}).get("plan_completed"),
    }
    if environment_success_evidence_only:
        payload["environment_success_evidence_only"] = True
    if motion_diagnostics is not None:
        payload["motion_diagnostics"] = motion_diagnostics
    if "action_completion_diagnostics" in last_info:
        payload["action_completion_diagnostics"] = last_info.get("action_completion_diagnostics")
    return payload


def record_step_events(
    *,
    subtask: Any,
    attempt: int,
    control_step: int | None,
    step_count: int,
    reward: Any,
    terminated: bool,
    truncated: bool,
    last_info: dict[str, Any],
    success_flag: bool,
    subtask_completed: bool,
    subtask_succeeded: bool,
    subtask_completion_reason: str | None,
    resolved_subtask_name: str,
    env_subtask_name: str | None,
    active_internal: dict[str, Any] | None,
    action_progress: dict[str, Any] | None,
    record_event: Any,
    motion_diagnostics: dict[str, Any] | None = None,
    environment_success_evidence_only: bool = False,
) -> None:
    record_event(
        "env_step",
        build_env_step_payload(
            subtask=subtask,
            attempt=attempt,
            control_step=control_step,
            step_count=step_count,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            last_info=last_info,
            success_flag=success_flag,
            subtask_completed=subtask_completed,
            subtask_succeeded=subtask_succeeded,
            subtask_completion_reason=subtask_completion_reason,
            resolved_subtask_name=resolved_subtask_name,
            env_subtask_name=env_subtask_name,
            active_internal=active_internal,
            action_progress=action_progress,
            motion_diagnostics=motion_diagnostics,
            environment_success_evidence_only=environment_success_evidence_only,
        ),
    )
    if subtask.agent != "ACTION" and getattr(subtask.agent, "value", None) != "ACTION":
        return
    if active_internal is None:
        return
    record_event(
        "action_internal_step_end",
        build_vla_internal_step_end_payload(
            subtask_id=subtask.subtask_id,
            attempt=attempt,
            control_step=control_step,
            step_count=step_count,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            success_flag=success_flag,
            active_internal=active_internal,
            action_progress=action_progress,
        ),
    )


def format_subtask_start_message(*, subtask: Any, attempt: int, max_steps: int, instruction: str) -> str:
    return (
        "[closed-loop] start "
        f"subtask={subtask.subtask_id} "
        f"agent={subtask.agent.value} "
        f"attempt={attempt} "
        f"max_steps={max_steps} "
        f"instruction={shorten_text(instruction, limit=96)}"
    )


def log_subtask_attempt_start(
    *,
    subtask: Any,
    attempt: int,
    instruction: str,
    max_steps: int,
    logged_subtask_attempts: set[tuple[str, int]],
    record_event: Any,
    emit_progress: Any,
) -> set[tuple[str, int]]:
    marker = (subtask.runtime_id, attempt)
    if marker in logged_subtask_attempts:
        return logged_subtask_attempts

    next_markers = set(logged_subtask_attempts)
    next_markers.add(marker)
    payload = build_subtask_start_payload(
        subtask=subtask,
        attempt=attempt,
        instruction=instruction,
        max_steps=max_steps,
    )
    record_event("subtask_start", payload)
    emit_progress(
        format_subtask_start_message(
            subtask=subtask,
            attempt=attempt,
            max_steps=max_steps,
            instruction=instruction,
        )
    )
    return next_markers


def build_progress_payload(
    *,
    subtask: Any,
    attempt: int,
    control_step: int | None,
    step_count: int,
    last_reward: Any,
    last_info: dict[str, Any],
    success_flag: bool,
    subtask_completed: bool,
    subtask_succeeded: bool,
    subtask_completion_reason: str | None,
    action_summary: dict[str, dict[str, Any]],
    nav_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "subtask_id": subtask.subtask_id,
        **_subtask_identity_payload(subtask),
        "agent": subtask.agent.value,
        "attempt": attempt,
        "control_step": control_step,
        "step_count": step_count,
        "reward": last_reward,
        "task_progress": last_info.get("task_progress", 0.0),
        "task_success": success_flag,
        "subtask_completed": subtask_completed,
        "subtask_succeeded": subtask_succeeded,
        "subtask_completion_reason": subtask_completion_reason,
        "action_summary": action_summary,
    }
    if isinstance(nav_state, dict):
        for key in (
            "active_waypoint_index",
            "global_waypoint_index",
            "dense_waypoint_index",
            "controller_mode",
            "follow_status",
            "recovery_mode",
            "recovery_profile",
            "yaw_source",
            "path_backend",
            "path_tracking_mode",
            "nav2_error",
            "nav2_trav_map_filename",
            "loop_detected",
            "oscillation_detected",
            "tracking_target",
            "target_waypoint",
            "local_goal",
            "execution_goal",
            "nav2_compute_goal",
            "transition_anchor",
            "nav_goal",
            "grounded_goal",
            "interpreted_goal",
            "selected_object_approach",
            "grounding_candidates",
            "selected_grounding_candidate",
            "object_approach_selection",
            "prepared_navigation_payload",
            "navigation_skill_selection",
            "steps_since_progress",
            "best_distance_to_waypoint",
            "path_cross_track_error",
            "path_signed_cross_track_error",
            "path_segment_index",
            "path_tangent_heading",
            "localization_guard",
        ):
            if key in nav_state:
                entry = summarize_navigation_progress_entry(key, nav_state.get(key))
                if entry is not None:
                    payload[entry[0]] = entry[1]
    if "action_completion_diagnostics" in last_info:
        payload["action_completion_diagnostics"] = last_info.get("action_completion_diagnostics")
    for key in ("pose", "current_room", "current_region", "room_id", "floor_id"):
        if key in last_info:
            payload[key] = last_info.get(key)
    return payload


def maybe_log_progress(
    *,
    subtask: Any,
    attempt: int,
    control_step: int | None,
    step_count: int,
    last_reward: Any,
    last_info: dict[str, Any],
    success_flag: bool,
    subtask_completed: bool,
    subtask_succeeded: bool,
    subtask_completion_reason: str | None,
    action_summary: dict[str, dict[str, Any]],
    nav_state: dict[str, Any] | None,
    progress_log_every: int | None,
    done: bool,
    record_event: Any,
    emit_progress: Any,
) -> None:
    if not should_emit_progress_log(
        progress_log_every=progress_log_every,
        control_step=control_step,
        done=done,
    ):
        return

    payload = build_progress_payload(
        subtask=subtask,
        attempt=attempt,
        control_step=control_step,
        step_count=step_count,
        last_reward=last_reward,
        last_info=last_info,
        success_flag=success_flag,
        subtask_completed=subtask_completed,
        subtask_succeeded=subtask_succeeded,
        subtask_completion_reason=subtask_completion_reason,
        action_summary=action_summary,
        nav_state=nav_state,
    )
    record_event("progress_update", payload)
    emit_progress(
        format_progress_message(
            subtask=subtask,
            attempt=attempt,
            control_step=control_step,
            step_count=step_count,
            reward=last_reward,
            task_progress=last_info.get("task_progress", 0.0),
            success_flag=success_flag,
            subtask_completed=subtask_completed,
            subtask_succeeded=subtask_succeeded,
            subtask_completion_reason=subtask_completion_reason,
            action_summary=action_summary,
        )
    )


def log_progress_update(
    *,
    subtask: Any,
    attempt: int,
    control_step: int | None,
    step_count: int,
    last_reward: Any,
    last_info: dict[str, Any],
    success_flag: bool,
    subtask_completed: bool,
    subtask_succeeded: bool,
    subtask_completion_reason: str | None,
    action: dict[str, Any],
    nav_state: dict[str, Any] | None,
    progress_log_every: int | None,
    done: bool,
    record_event: Any,
    emit_progress: Any,
) -> None:
    maybe_log_progress(
        subtask=subtask,
        attempt=attempt,
        control_step=control_step,
        step_count=step_count,
        last_reward=last_reward,
        last_info=last_info,
        success_flag=success_flag,
        subtask_completed=subtask_completed,
        subtask_succeeded=subtask_succeeded,
        subtask_completion_reason=subtask_completion_reason,
        action_summary=summarize_action(action),
        nav_state=nav_state,
        progress_log_every=progress_log_every,
        done=done,
        record_event=record_event,
        emit_progress=emit_progress,
    )


def format_progress_message(
    *,
    subtask: Any,
    attempt: int,
    control_step: int | None,
    step_count: int,
    reward: Any,
    task_progress: Any,
    success_flag: bool,
    subtask_completed: bool,
    subtask_succeeded: bool,
    subtask_completion_reason: str | None,
    action_summary: dict[str, dict[str, Any]],
) -> str:
    return (
        "[closed-loop] step "
        f"subtask={subtask.subtask_id} "
        f"attempt={attempt} "
        f"control_step={control_step or '-'} "
        f"env_step={step_count} "
        f"reward={format_float(reward)} "
        f"task_progress={format_float(task_progress)} "
        f"task_success={success_flag} "
        f"subtask_completed={subtask_completed} "
        f"subtask_succeeded={subtask_succeeded} "
        f"reason={subtask_completion_reason or '-'} "
        f"action={format_action_summary_text(action_summary)}"
    )


def should_emit_progress_log(*, progress_log_every: int | None, control_step: int | None, done: bool) -> bool:
    if progress_log_every is None:
        return False
    if done or control_step == 1:
        return True
    if control_step is None:
        return False
    return control_step % progress_log_every == 0


def normalize_progress_log_every(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def normalize_completion_reason(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def subtask_failure_reason(completion_reason: str | None) -> str:
    if completion_reason == "timeout":
        return "SUBTASK_TIMEOUT"
    return "SUBTASK_FAILED"


def summarize_sequence(value: Any, limit: int = 3) -> str:
    if not isinstance(value, list) or not value:
        return "-"
    items = [str(item) for item in value[:limit]]
    if len(value) > limit:
        items.append("...")
    return ",".join(items)


def shorten_text(value: str, limit: int = 80) -> str:
    compact = " ".join(str(value).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def format_float(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "-"


def summarize_action(action: dict[str, Any]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for key, value in sorted(action.items()):
        arr = to_numpy(value)
        if arr is None or arr.size == 0:
            continue
        flat = arr.astype(np.float32, copy=False).reshape(-1)
        preview = [round(float(item), 4) for item in flat[:3]]
        summary[key] = {
            "shape": list(arr.shape),
            "non_zero": int(np.count_nonzero(np.abs(flat) > 1e-4)),
            "max_abs": round(float(np.max(np.abs(flat))), 4),
            "mean_abs": round(float(np.mean(np.abs(flat))), 4),
            "preview": preview,
        }
    return summary


def build_motion_diagnostics(
    *,
    previous_obs: dict[str, Any] | None,
    obs: dict[str, Any] | None,
    info: dict[str, Any] | None,
    action: dict[str, Any],
    env_action: dict[str, Any],
    runtime_artifacts: dict[str, Any] | None,
    previous_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifacts = runtime_artifacts or {}
    full_action = artifacts.get("full_action") or {}
    projected_action = artifacts.get("projected_action") or action
    before = summarize_motion_state(previous_obs, previous_info)
    after = summarize_motion_state(obs, info)
    return {
        "full_action_summary": summarize_action(full_action),
        "projected_action_summary": summarize_action(projected_action),
        "env_action_summary": summarize_action(env_action),
        "action_transport": summarize_action_transport(
            action=action,
            env_action=env_action,
            full_action=full_action,
            projected_action=projected_action,
        ),
        "local_base_motion_allowed": artifacts.get("local_base_motion_allowed"),
        "control_mode": artifacts.get("control_mode"),
        "state_before": before,
        "state_after": after,
        "state_delta": summarize_motion_delta(before, after),
    }


def summarize_action_transport(
    *,
    action: dict[str, Any],
    env_action: dict[str, Any],
    full_action: dict[str, Any],
    projected_action: dict[str, Any],
) -> dict[str, Any]:
    return {
        "policy_action_keys": sorted(str(key) for key in full_action.keys()),
        "projected_action_keys": sorted(str(key) for key in projected_action.keys()),
        "runtime_action_keys": sorted(str(key) for key in action.keys()),
        "env_action_keys": sorted(str(key) for key in env_action.keys()),
        "policy_robot_r1_slices": summarize_robot_r1_action_slices(full_action),
        "projected_robot_r1_slices": summarize_robot_r1_action_slices(projected_action),
        "runtime_robot_r1_slices": summarize_robot_r1_action_slices(action),
        "env_robot_r1_slices": summarize_robot_r1_action_slices(env_action),
        "policy_split_action_slices": summarize_split_action_slices(full_action),
        "projected_split_action_slices": summarize_split_action_slices(projected_action),
        "runtime_split_action_slices": summarize_split_action_slices(action),
        "env_split_action_slices": summarize_split_action_slices(env_action),
        "env_action_shapes": summarize_action_shapes(env_action),
    }


def summarize_action_shapes(action: dict[str, Any]) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for key, value in sorted(action.items()):
        arr = to_numpy(value)
        if arr is None:
            continue
        shapes[str(key)] = list(arr.shape)
    return shapes


def summarize_robot_r1_action_slices(action: dict[str, Any]) -> dict[str, dict[str, Any]]:
    arr = to_numpy(action.get("robot_r1"))
    if arr is None or arr.size != 23:
        return {}
    flat = arr.astype(np.float32, copy=False).reshape(-1)
    return {
        "base": summarize_flat_action_slice(flat[0:3]),
        "torso": summarize_flat_action_slice(flat[3:7]),
        "left_arm": summarize_flat_action_slice(flat[7:14]),
        "left_gripper": summarize_flat_action_slice(flat[14:15]),
        "right_arm": summarize_flat_action_slice(flat[15:22]),
        "right_gripper": summarize_flat_action_slice(flat[22:23]),
    }


def summarize_split_action_slices(action: dict[str, Any]) -> dict[str, dict[str, Any]]:
    slices: dict[str, dict[str, Any]] = {}
    for source_key, name in (
        ("action.base", "base"),
        ("action.torso", "torso"),
        ("action.left_arm", "left_arm"),
        ("action.left_gripper", "left_gripper"),
        ("action.right_arm", "right_arm"),
        ("action.right_gripper", "right_gripper"),
    ):
        arr = to_numpy(action.get(source_key))
        if arr is None:
            continue
        slices[name] = summarize_flat_action_slice(arr.astype(np.float32, copy=False).reshape(-1))
    return slices


def summarize_flat_action_slice(values: np.ndarray) -> dict[str, Any]:
    return {
        "size": int(values.size),
        "non_zero": int(np.count_nonzero(np.abs(values) > 1e-4)),
        "max_abs": round(float(np.max(np.abs(values))), 6) if values.size else 0.0,
        "mean_abs": round(float(np.mean(np.abs(values))), 6) if values.size else 0.0,
        "values": [round(float(item), 6) for item in values],
    }


def summarize_motion_state(observation: dict[str, Any] | None, info: dict[str, Any] | None) -> dict[str, Any]:
    state: dict[str, Any] = {}
    if isinstance(info, dict):
        for key in ("pose", "current_room", "current_region", "room_id", "floor_id"):
            if key in info:
                state[f"info.{key}"] = json_safe(info.get(key))
    for key in (
        "robot_r1::proprio",
        "proprio",
        "state.robot_pos",
        "state.robot_ori_cos",
        "state.robot_ori_sin",
        "state.base_qpos",
        "state.base_qvel",
        "state.trunk_qpos",
        "state.arm_left_qpos",
        "state.arm_right_qpos",
        "state.gripper_left_qpos",
        "state.gripper_right_qpos",
    ):
        arr = extract_observation_array(observation, key)
        if arr is not None:
            summary = summarize_array(arr)
            if summary is not None:
                state[key] = summary
    return state


def summarize_motion_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key, after_summary in after.items():
        before_summary = before.get(key)
        if not isinstance(before_summary, dict) or not isinstance(after_summary, dict):
            continue
        before_values = before_summary.get("flat_preview_full")
        after_values = after_summary.get("flat_preview_full")
        if not isinstance(before_values, list) or not isinstance(after_values, list):
            continue
        if len(before_values) != len(after_values):
            continue
        diff = np.asarray(after_values, dtype=np.float32) - np.asarray(before_values, dtype=np.float32)
        delta[key] = {
            "size": int(diff.size),
            "max_abs": round(float(np.max(np.abs(diff))), 6) if diff.size else 0.0,
            "mean_abs": round(float(np.mean(np.abs(diff))), 6) if diff.size else 0.0,
            "preview": [round(float(item), 6) for item in diff[:6]],
        }
    return delta


def extract_observation_array(observation: dict[str, Any] | None, key: str) -> np.ndarray | None:
    if not isinstance(observation, dict):
        return None
    for candidate in observation_dict_candidates(observation):
        if key in candidate:
            return to_numpy(candidate[key])
        robot_obs = candidate.get("robot_r1")
        if isinstance(robot_obs, dict):
            if key in robot_obs:
                return to_numpy(robot_obs[key])
            alias = key.removeprefix("robot_r1::")
            if alias in robot_obs:
                return to_numpy(robot_obs[alias])
    return None


def observation_dict_candidates(observation: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [observation]
    raw_observation = observation.get("raw_observation")
    if isinstance(raw_observation, dict):
        candidates.append(raw_observation)
    return candidates


def summarize_array(value: Any) -> dict[str, Any] | None:
    arr = to_numpy(value)
    if arr is None or arr.size == 0:
        return None
    flat = arr.astype(np.float32, copy=False).reshape(-1)
    return {
        "shape": list(arr.shape),
        "size": int(flat.size),
        "max_abs": round(float(np.max(np.abs(flat))), 6),
        "mean_abs": round(float(np.mean(np.abs(flat))), 6),
        "preview": [round(float(item), 6) for item in flat[:6]],
        "flat_preview_full": [round(float(item), 6) for item in flat[: min(258, flat.size)]],
    }


def json_safe(value: Any) -> Any:
    arr = to_numpy(value)
    if arr is not None and arr.ndim > 0:
        try:
            return [round(float(item), 6) for item in arr.astype(np.float32, copy=False).reshape(-1)[:12]]
        except (TypeError, ValueError):
            return [str(item) for item in arr.reshape(-1)[:12]]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value[:12]]
    try:
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


def format_action_summary_text(summary: dict[str, dict[str, Any]]) -> str:
    if not summary:
        return "-"
    parts: list[str] = []
    for key, stats in sorted(summary.items()):
        preview = ",".join(f"{float(item):.3f}" for item in stats.get("preview", []))
        parts.append(
            f"{key}[nz={stats.get('non_zero', 0)},max={stats.get('max_abs', 0.0):.3f},"
            f"mean={stats.get('mean_abs', 0.0):.3f},v={preview}]"
        )
    return "; ".join(parts)


def to_numpy(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.cpu().numpy()
    try:
        return np.asarray(value)
    except Exception:
        return None
