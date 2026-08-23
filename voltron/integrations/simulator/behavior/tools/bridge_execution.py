"""Step-execution orchestration helpers for the BEHAVIOR runtime facade."""

from __future__ import annotations

from typing import Any

from voltron.integrations.simulator.behavior.artifacts import process_logger as behavior_process_logger
from voltron.integrations.simulator.behavior.execution import action_stepper as behavior_action_stepper
from voltron.integrations.simulator.behavior.execution import reward_status as behavior_reward_status
from voltron.integrations.simulator.behavior.tools import bridge_environment as behavior_bridge_environment
from voltron.integrations.simulator.behavior.tools import bridge_lifecycle as behavior_bridge_lifecycle
from voltron.integrations.simulator.behavior.tools import bridge_localization as behavior_bridge_localization
from voltron.integrations.simulator.behavior.tools import bridge_recording as behavior_bridge_recording
from voltron.integrations.simulator.behavior.tools import bridge_subtasks as behavior_bridge_subtasks
from voltron.integrations.simulator.behavior.tools import runtime_adapter_state as behavior_runtime_adapter_state
from voltron.integrations.simulator.behavior.tools import runtime_vla as behavior_runtime_vla
from voltron.integrations.simulator.behavior.tools import step_setup as behavior_step_setup


def on_agent_result(
    runtime: Any,
    *,
    subtask: Any,
    result: Any,
    context: Any,
    runtime_bridge_file: str,
) -> Any:
    subtask_max_steps = behavior_bridge_subtasks.subtask_max_steps(
        subtask,
        runtime.default_subtask_max_steps,
    )
    prepared = behavior_step_setup.prepare_agent_result_runtime_state(
        context=context,
        subtask=subtask,
        result=result,
        navigation_runtime_state=runtime._navigation_runtime_state,
        logged_subtask_attempts=runtime._logged_subtask_attempts,
        logged_action_internal_attempts=runtime._logged_action_internal_attempts,
        logged_action_internal_replans=runtime._logged_action_internal_replans,
        call_env_method=runtime._call_behavior_env_method,
        instruction_for_subtask=runtime._instruction_for_subtask,
        subtask_max_steps=lambda current: behavior_bridge_subtasks.subtask_max_steps(
            current,
            runtime.default_subtask_max_steps,
        ),
        record_event=lambda event, payload: behavior_bridge_lifecycle.record_event(runtime, event, payload),
        emit_progress=lambda message: behavior_bridge_lifecycle.emit_progress(runtime, message),
    )
    prepared_state = behavior_runtime_adapter_state.apply_prepared_step_state(runtime, prepared)
    control_step = prepared_state["control_step"]
    attempt = prepared_state["attempt"]

    resolved_subtask_name = behavior_bridge_subtasks.resolved_subtask_name(
        subtask=subtask,
        active_internal_step=runtime._active_action_internal_step,
        last_info=runtime._last_info,
        active_subtask_name=runtime._active_subtask_name,
        slugify=behavior_bridge_recording.safe_slug,
    )
    env_subtask_name = behavior_bridge_subtasks.env_subtask_name(runtime._last_info)
    instruction = runtime._instruction_for_subtask(subtask)

    terminal_outcome = behavior_action_stepper.handle_terminal_step(
        subtask=subtask,
        result=result,
        attempt=attempt,
        control_step=control_step,
        step_count=runtime._step_count,
        instruction=instruction,
        resolved_subtask_name=resolved_subtask_name,
        env_subtask_name=env_subtask_name,
        summarize_sequence=behavior_bridge_recording.summarize_sequence,
        record_event=lambda event, payload: behavior_bridge_lifecycle.record_event(runtime, event, payload),
        emit_progress=lambda message: behavior_bridge_lifecycle.emit_progress(runtime, message),
    )
    if terminal_outcome is not None:
        return terminal_outcome

    previous_obs = dict(runtime._last_obs) if isinstance(runtime._last_obs, dict) else {}
    previous_info = dict(runtime._last_info) if isinstance(runtime._last_info, dict) else {}
    executed = behavior_action_stepper.execute_agent_step(
        subtask=subtask,
        result=result,
        attempt=attempt,
        control_step=control_step,
        step_count=runtime._step_count,
        ensure_env=lambda: behavior_bridge_environment.ensure_env(
            runtime,
            runtime_bridge_file=runtime_bridge_file,
        ),
        extract_action=behavior_bridge_localization.extract_action,
        format_behavior_action=lambda action: behavior_bridge_environment.format_behavior_action(
            runtime,
            action,
            runtime_bridge_file=runtime_bridge_file,
            hold_grippers_closed=_should_hold_grippers_closed(context=context, subtask=subtask),
        ),
        build_action_missing_response=behavior_action_stepper.build_action_missing_response,
        build_env_step_error_response=behavior_action_stepper.build_env_step_error_response,
        emit_step_response=behavior_action_stepper.emit_step_response,
        record_event=lambda event, payload: behavior_bridge_lifecycle.record_event(runtime, event, payload),
        emit_progress=lambda message: behavior_bridge_lifecycle.emit_progress(runtime, message),
    )
    if executed["terminal_outcome"] is not None:
        return executed["terminal_outcome"]

    action = executed["action"]
    env_action = executed["env_action"]
    obs = executed["obs"]
    reward = executed["reward"]
    terminated = executed["terminated"]
    truncated = executed["truncated"]
    info = executed["info"]
    runtime._last_obs = dict(obs) if isinstance(obs, dict) else {}

    def apply_success_overrides(last_info: dict[str, Any], task_success: bool) -> dict[str, Any]:
        merged_info = behavior_bridge_environment.merge_environment_vlm_heartbeat(
            runtime,
            last_info,
            subtask=subtask,
        )
        nav_override = behavior_bridge_environment.apply_navigation_success_override(
            runtime,
            subtask=subtask,
            last_info=merged_info,
            task_success=task_success,
        )
        return behavior_bridge_environment.apply_action_completion_override(
            runtime,
            subtask=subtask,
            last_info=nav_override["last_info"],
            task_success=bool(nav_override["task_success"]),
        )

    step_state = behavior_reward_status.advance_runtime_step_state(
        obs=obs,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info=info,
        step_count=runtime._step_count,
        task_success=runtime._task_success,
        control_step=control_step,
        subtask_max_steps=subtask_max_steps,
        apply_success_overrides=apply_success_overrides,
        resolve_step_completion_state=behavior_reward_status.resolve_step_completion_state,
    )
    completion_state = behavior_runtime_adapter_state.apply_step_state(runtime, step_state)
    success_flag = bool(completion_state["success_flag"])
    subtask_completed = bool(completion_state["subtask_completed"])
    subtask_succeeded = bool(completion_state["subtask_succeeded"])
    subtask_completion_reason = completion_state["subtask_completion_reason"]
    environment_success_evidence_only = _environment_success_evidence_only(runtime)

    action_progress = behavior_runtime_vla.extract_vla_execution_progress(result)
    active_internal = runtime._active_action_internal_step
    motion_diagnostics = None
    if getattr(subtask.agent, "value", None) == "ACTION":
        motion_diagnostics = behavior_process_logger.build_motion_diagnostics(
            previous_obs=previous_obs,
            previous_info=previous_info,
            obs=obs if isinstance(obs, dict) else {},
            info=info if isinstance(info, dict) else {},
            action=action,
            env_action=env_action,
            runtime_artifacts=result.runtime_artifacts if isinstance(result.runtime_artifacts, dict) else {},
        )
    return behavior_action_stepper.finalize_step(
        subtask=subtask,
        attempt=attempt,
        control_step=control_step,
        action=action,
        step_count=runtime._step_count,
        last_obs=runtime._last_obs,
        last_info=runtime._last_info,
        last_reward=runtime._last_reward,
        terminated=runtime._terminated,
        truncated=runtime._truncated,
        success_flag=success_flag,
        subtask_completed=subtask_completed,
        subtask_succeeded=subtask_succeeded,
        subtask_completion_reason=subtask_completion_reason,
        active_internal=active_internal,
        action_progress=action_progress,
        motion_diagnostics=motion_diagnostics,
        resolved_subtask_name=resolved_subtask_name,
        env_subtask_name=env_subtask_name,
        record_frame=lambda obs: behavior_bridge_lifecycle.record_frame(runtime, obs),
        record_step_events=lambda **kwargs: behavior_process_logger.record_step_events(
            **kwargs,
            record_event=lambda event, payload: behavior_bridge_lifecycle.record_event(runtime, event, payload),
        ),
        maybe_log_progress=lambda **kwargs: behavior_bridge_lifecycle.maybe_log_progress(runtime, **kwargs),
        subtask_failure_reason=behavior_bridge_recording.subtask_failure_reason,
        format_float=behavior_bridge_recording.format_float,
        feedback_factory=lambda extras=None: behavior_bridge_lifecycle.build_runtime_feedback(
            runtime,
            subtask=subtask,
            extras=extras,
        ),
        record_event=lambda event, payload: behavior_bridge_lifecycle.record_event(runtime, event, payload),
        emit_progress=lambda message: behavior_bridge_lifecycle.emit_progress(runtime, message),
        environment_success_evidence_only=environment_success_evidence_only,
    )


def _environment_success_evidence_only(runtime: Any) -> bool:
    if not bool(getattr(runtime, "_task_success", False)):
        return False
    use_environment_success = bool(
        getattr(runtime, "runtime_termination_use_environment_success_signal", True)
    )
    signal_policy = str(
        getattr(runtime, "runtime_termination_environment_signal_policy", "allow_early_success")
        or "allow_early_success"
    )
    return (not use_environment_success) or signal_policy == "evidence_only"


def _should_hold_grippers_closed(*, context: Any, subtask: Any) -> bool:
    runtime_state = getattr(context, "runtime_state", None)
    if not isinstance(runtime_state, dict) or not isinstance(runtime_state.get("carried_object"), dict):
        return False
    action = " ".join(str(getattr(subtask, "action", "") or "").replace("_", " ").replace("-", " ").lower().split())
    return action not in {"place", "put", "put down", "putdown", "drop", "release"}
