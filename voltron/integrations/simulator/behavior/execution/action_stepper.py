"""Action-step execution and terminal response helpers for BEHAVIOR."""

from __future__ import annotations

from typing import Any, Callable

from voltron.shared.enums import AgentName, AgentStatus
from voltron.shared.models import RuntimeFeedback, SubtaskStepOutcome


ACTION_STATE_COMPLETION_REASONS = {
    "object_opened",
    "object_closed",
    "object_toggled_on",
    "object_toggled_off",
}


def _agent_value(subtask: Any) -> str | None:
    return getattr(getattr(subtask, "agent", None), "value", None)


def _action_state_completion_evidence_only(
    *,
    subtask: Any,
    subtask_completed: bool,
    subtask_completion_reason: str | None,
    terminated: bool,
    truncated: bool,
) -> bool:
    return (
        _agent_value(subtask) == "ACTION"
        and bool(subtask_completed)
        and str(subtask_completion_reason or "") in ACTION_STATE_COMPLETION_REASONS
        and not bool(terminated)
        and not bool(truncated)
    )


def emit_step_response(
    *,
    response: dict[str, Any],
    record_event: Callable[[str, dict[str, Any]], None],
    emit_progress: Callable[[str], None],
) -> SubtaskStepOutcome:
    if response["event"] is not None:
        record_event(response["event"], response["payload"])
    if response["progress_message"] is not None:
        emit_progress(response["progress_message"])
    return response["outcome"]


def build_agent_failure_response(
    *,
    subtask: Any,
    error_code: str | None,
    attempt: int,
    control_step: int | None,
    step_count: int,
) -> dict[str, Any]:
    failure_reason = error_code or "AGENT_FAILURE"
    return {
        "event": "agent_failure",
        "payload": {
            "subtask_id": subtask.subtask_id,
            "agent": subtask.agent.value,
            "attempt": attempt,
            "control_step": control_step,
            "error_code": failure_reason,
            "step_count": step_count,
        },
        "progress_message": (
            "[closed-loop] fail "
            f"subtask={subtask.subtask_id} "
            f"agent={subtask.agent.value} "
            f"attempt={attempt} "
            f"control_step={control_step or '-'} "
            f"reason={failure_reason}"
        ),
        "outcome": SubtaskStepOutcome(
            done=True,
            success=False,
            failure_reason=failure_reason,
            feedback=RuntimeFeedback(step_count=step_count),
        ),
    }


def build_vision_step_response(
    *,
    subtask: Any,
    result: Any,
    attempt: int,
    control_step: int | None,
    step_count: int,
    instruction: str,
    resolved_subtask_name: str,
    env_subtask_name: str | None,
    summarize_sequence: Callable[[Any], str],
) -> dict[str, Any]:
    task_complete = bool(result.result.get("task_complete", False))
    scene_report = result.result.get("scene_report")
    raw_text = str(result.result.get("raw_text", ""))
    return {
        "event": "vlm_step",
        "payload": {
            "subtask_id": subtask.subtask_id,
            "agent": subtask.agent.value,
            "attempt": attempt,
            "control_step": control_step,
            "step_count": step_count,
            "task_complete": task_complete,
            "instruction": instruction,
            "objects": list(result.result.get("objects", []))[:5],
            "relations": list(result.result.get("relations", []))[:5],
            "scene_report": dict(scene_report) if isinstance(scene_report, dict) else {},
            "raw_text": raw_text,
            "subtask_name": resolved_subtask_name,
            "env_subtask_name": env_subtask_name,
        },
        "progress_message": (
            "[closed-loop] vlm "
            f"subtask={subtask.subtask_id} "
            f"attempt={attempt} "
            f"control_step={control_step or '-'} "
            f"task_complete={task_complete} "
            f"objects={summarize_sequence(result.result.get('objects'))} "
            f"relations={summarize_sequence(result.result.get('relations'))}"
        ),
        "outcome": SubtaskStepOutcome(
            done=True,
            success=True,
            feedback=RuntimeFeedback(
                step_count=step_count,
                extras={
                    # A Vision observation finishes this Vision subtask even when
                    # it is not allowed to claim the overall task is complete.
                    "task_complete": task_complete,
                    "subtask_completed": True,
                    "subtask_succeeded": True,
                    "subtask_completion_reason": "vision_observation_complete",
                },
            ),
        ),
    }


def build_action_missing_response(
    *,
    subtask: Any,
    attempt: int,
    control_step: int | None,
    step_count: int,
) -> dict[str, Any]:
    return {
        "event": "action_missing",
        "payload": {
            "subtask_id": subtask.subtask_id,
            "agent": subtask.agent.value,
            "attempt": attempt,
            "control_step": control_step,
            "step_count": step_count,
        },
        "progress_message": (
            "[closed-loop] fail "
            f"subtask={subtask.subtask_id} "
            f"agent={subtask.agent.value} "
            f"attempt={attempt} "
            f"control_step={control_step or '-'} "
            "reason=RUNTIME_ACTION_MISSING"
        ),
        "outcome": SubtaskStepOutcome(
            done=True,
            success=False,
            failure_reason="RUNTIME_ACTION_MISSING",
            feedback=RuntimeFeedback(step_count=step_count),
        ),
    }


def _verified_action_free_placement(result: Any) -> dict[str, Any] | None:
    """Return terminal placement evidence for a successful action-free result.

    A placement skill applies its last simulator action on the preceding
    control cycle.  Its terminal ``AgentResult`` therefore intentionally has
    no fresh action.  Treating that result as a missing action either replays
    a stale command or turns a physically verified placement into a runtime
    failure.  The narrow checks below keep ordinary empty ACTION results on
    the existing ``RUNTIME_ACTION_MISSING`` path.
    """

    payload = getattr(result, "result", None)
    artifacts = getattr(result, "runtime_artifacts", None)
    if not isinstance(payload, dict) or not isinstance(artifacts, dict):
        return None
    if payload.get("action_keys") != []:
        return None
    full_action = artifacts.get("full_action")
    projected_action = artifacts.get("projected_action")
    if not isinstance(full_action, dict) or full_action:
        return None
    if not isinstance(projected_action, dict) or projected_action:
        return None
    if payload.get("placement_success") is not True:
        return None
    if payload.get("placement_verified") is not True:
        return None
    evidence = payload.get("physical_evidence")
    if not isinstance(evidence, dict):
        evidence = artifacts.get("physical_evidence")
    if not isinstance(evidence, dict):
        return None
    if evidence.get("released") is not True:
        return None
    if evidence.get("aabb_contained") is not True:
        return None
    return evidence


def build_action_terminal_success_response(
    *,
    subtask: Any,
    result: Any,
    evidence: dict[str, Any],
    attempt: int,
    control_step: int | None,
    step_count: int,
) -> dict[str, Any]:
    """Build the terminal runtime response for a verified placement."""

    result_payload = dict(getattr(result, "result", {}) or {})
    placement_strategy = evidence.get("placement_strategy") or evidence.get("strategy")
    extras = {
        "action_terminal_success": True,
        "action_keys": [],
        "placement_success": True,
        "placement_verified": True,
        "placement_strategy": placement_strategy,
        "released": True,
        "aabb_contained": True,
        "last_applied_action_keys": list(evidence.get("last_applied_action_keys", [])),
        "physical_evidence": dict(evidence),
    }
    return {
        "event": "action_terminal_success",
        "payload": {
            "subtask_id": subtask.subtask_id,
            "agent": subtask.agent.value,
            "attempt": attempt,
            "control_step": control_step,
            "step_count": step_count,
            "status": "success",
            "action_keys": [],
            "destination_object": result_payload.get("destination_object"),
            "sim_steps": result_payload.get("sim_steps"),
            **extras,
        },
        "progress_message": (
            "[closed-loop] terminal-success "
            f"subtask={subtask.subtask_id} "
            f"agent={subtask.agent.value} "
            f"attempt={attempt} "
            f"control_step={control_step or '-'} "
            f"strategy={placement_strategy or '-'} "
            "action_keys=[]"
        ),
        "outcome": SubtaskStepOutcome(
            done=True,
            success=True,
            feedback=RuntimeFeedback(step_count=step_count, extras=extras),
        ),
    }


def build_env_step_error_response(
    *,
    subtask: Any,
    error: str,
    step_count: int,
) -> dict[str, Any]:
    return {
        "event": "env_step_error",
        "payload": {
            "subtask_id": subtask.subtask_id,
            "agent": subtask.agent.value,
            "step_count": step_count,
            "error": error,
        },
        "progress_message": None,
        "outcome": SubtaskStepOutcome(
            done=True,
            success=False,
            failure_reason="ENV_STEP_ERROR",
            feedback=RuntimeFeedback(
                step_count=step_count,
                extras={"message": error},
            ),
        ),
    }


def handle_terminal_step(
    *,
    subtask: Any,
    result: Any,
    attempt: int,
    control_step: int | None,
    step_count: int,
    instruction: str,
    resolved_subtask_name: str,
    env_subtask_name: str | None,
    summarize_sequence: Callable[[Any], str],
    record_event: Callable[[str, dict[str, Any]], None],
    emit_progress: Callable[[str], None],
) -> SubtaskStepOutcome | None:
    if result.status != AgentStatus.SUCCESS:
        return emit_step_response(
            response=build_agent_failure_response(
                subtask=subtask,
                error_code=result.error_code,
                attempt=attempt,
                control_step=control_step,
                step_count=step_count,
            ),
            record_event=record_event,
            emit_progress=emit_progress,
        )

    if subtask.agent == AgentName.ACTION:
        terminal_evidence = _verified_action_free_placement(result)
        if terminal_evidence is not None:
            return emit_step_response(
                response=build_action_terminal_success_response(
                    subtask=subtask,
                    result=result,
                    evidence=terminal_evidence,
                    attempt=attempt,
                    control_step=control_step,
                    step_count=step_count,
                ),
                record_event=record_event,
                emit_progress=emit_progress,
            )

    if subtask.agent != AgentName.VISION:
        return None

    return emit_step_response(
        response=build_vision_step_response(
            subtask=subtask,
            result=result,
            attempt=attempt,
            control_step=control_step,
            step_count=step_count,
            instruction=instruction,
            resolved_subtask_name=resolved_subtask_name,
            env_subtask_name=env_subtask_name,
            summarize_sequence=summarize_sequence,
        ),
        record_event=record_event,
        emit_progress=emit_progress,
    )


def execute_agent_step(
    *,
    subtask: Any,
    result: Any,
    attempt: int,
    control_step: int | None,
    step_count: int,
    ensure_env: Callable[[], Any],
    extract_action: Callable[[Any], dict[str, Any] | None],
    format_behavior_action: Callable[[dict[str, Any]], dict[str, Any]],
    build_action_missing_response: Callable[..., dict[str, Any]],
    build_env_step_error_response: Callable[..., dict[str, Any]],
    emit_step_response: Callable[..., Any],
    record_event: Callable[[str, dict[str, Any]], None],
    emit_progress: Callable[[str], None],
) -> dict[str, Any]:
    action = extract_action(result)
    if action is None:
        return {
            "terminal_outcome": emit_step_response(
                response=build_action_missing_response(
                    subtask=subtask,
                    attempt=attempt,
                    control_step=control_step,
                    step_count=step_count,
                ),
                record_event=record_event,
                emit_progress=emit_progress,
            )
        }

    env_action = format_behavior_action(action)
    try:
        obs, reward, terminated, truncated, info = ensure_env().step(env_action)
    except Exception as exc:
        return {
            "terminal_outcome": emit_step_response(
                response=build_env_step_error_response(
                    subtask=subtask,
                    error=str(exc),
                    step_count=step_count,
                ),
                record_event=record_event,
                emit_progress=emit_progress,
            )
        }

    return {
        "terminal_outcome": None,
        "action": action,
        "env_action": env_action,
        "obs": obs,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": info,
    }


def build_vla_step_response(
    *,
    subtask: Any,
    attempt: int,
    control_step: int | None,
    step_count: int,
    task_progress: float,
    success_flag: bool,
    terminated: bool,
    truncated: bool,
    action_progress: dict[str, Any],
    feedback_factory: Callable[..., Any],
    action_completion_diagnostics: dict[str, Any] | None = None,
    environment_success_evidence_only: bool = False,
) -> dict[str, Any]:
    plan_completed = bool(action_progress.get("plan_completed", False))

    if terminated and success_flag and environment_success_evidence_only:
        return {
            "event": None,
            "payload": None,
            "progress_message": None,
            "outcome": SubtaskStepOutcome(
                done=False,
                success=None,
                feedback=feedback_factory(
                    extras={
                        "task_success": True,
                        "environment_terminated": True,
                        "environment_success_evidence_only": True,
                    },
                ),
            ),
        }

    if truncated and not success_flag:
        payload = {
            "subtask_id": subtask.subtask_id,
            "reason": "ENV_TRUNCATED",
            "attempt": attempt,
            "control_step": control_step,
            "step_count": step_count,
            "task_progress": task_progress,
        }
        if action_completion_diagnostics is not None:
            payload["action_completion_diagnostics"] = action_completion_diagnostics
        return {
            "event": "subtask_failed",
            "payload": payload,
            "progress_message": None,
            "outcome": SubtaskStepOutcome(
                done=True,
                success=False,
                failure_reason="ENV_TRUNCATED",
                feedback=feedback_factory(),
            ),
        }

    if terminated and not success_flag:
        return {
            "event": None,
            "payload": None,
            "progress_message": None,
            "outcome": SubtaskStepOutcome(
                done=True,
                success=False,
                failure_reason="ENV_TERMINATED",
                feedback=feedback_factory(),
            ),
        }

    if terminated and success_flag:
        return {
            "event": None,
            "payload": None,
            "progress_message": None,
            "outcome": SubtaskStepOutcome(
                done=True,
                success=True,
                feedback=feedback_factory(),
            ),
        }

    if not plan_completed:
        return {
            "event": None,
            "payload": None,
            "progress_message": None,
            "outcome": SubtaskStepOutcome(
                done=False,
                feedback=feedback_factory(),
            ),
        }

    payload = {
        "subtask_id": subtask.subtask_id,
        "reason": "VLA_INTERNAL_STEP_LIMIT_REACHED",
        "attempt": attempt,
        "control_step": control_step,
        "step_count": step_count,
        "task_progress": task_progress,
    }
    if action_completion_diagnostics is not None:
        payload["action_completion_diagnostics"] = action_completion_diagnostics
    return {
        "event": "subtask_failed",
        "payload": payload,
        "progress_message": None,
        "outcome": SubtaskStepOutcome(
            done=True,
            success=False,
            failure_reason="VLA_INTERNAL_STEP_LIMIT_REACHED",
            feedback=feedback_factory(extras={"action_plan_completed": True}),
        ),
    }


def build_standard_step_response(
    *,
    subtask: Any,
    attempt: int,
    control_step: int | None,
    step_count: int,
    task_progress: float,
    success_flag: bool,
    subtask_completed: bool,
    subtask_succeeded: bool,
    subtask_completion_reason: str | None,
    terminated: bool,
    truncated: bool,
    resolved_subtask_name: str,
    env_subtask_name: str | None,
    action_completion_diagnostics: dict[str, Any] | None = None,
    environment_success_evidence_only: bool = False,
    subtask_failure_reason: Callable[[str | None], str],
    format_float: Callable[[Any], str],
    feedback_factory: Callable[..., Any],
) -> dict[str, Any]:
    done = terminated or truncated or subtask_completed
    if not done:
        return {
            "event": None,
            "payload": None,
            "progress_message": None,
            "outcome": SubtaskStepOutcome(
                done=False,
                feedback=feedback_factory(),
            ),
        }

    if (
        environment_success_evidence_only
        and subtask_completed
        and subtask_completion_reason == "task_success"
    ):
        return {
            "event": None,
            "payload": None,
            "progress_message": None,
            "outcome": SubtaskStepOutcome(
                done=False,
                success=None,
                feedback=feedback_factory(
                    extras={
                        "subtask_completed": subtask_completed,
                        "subtask_succeeded": subtask_succeeded,
                        "subtask_completion_reason": subtask_completion_reason,
                        "task_success": True,
                        "environment_terminated": bool(terminated),
                        "environment_truncated": bool(truncated),
                        # The completion monitor may use this as evidence, but
                        # it must not advance the subtask without Brain/Vision.
                        "environment_success_evidence_only": True,
                    },
                ),
            ),
        }

    if truncated and not success_flag:
        payload = {
            "subtask_id": subtask.subtask_id,
            "reason": "ENV_TRUNCATED",
            "attempt": attempt,
            "control_step": control_step,
            "step_count": step_count,
            "task_progress": task_progress,
        }
        if action_completion_diagnostics is not None:
            payload["action_completion_diagnostics"] = action_completion_diagnostics
        return {
            "event": "subtask_failed",
            "payload": payload,
            "progress_message": (
                "[closed-loop] fail "
                f"subtask={subtask.subtask_id} "
                f"attempt={attempt} "
                f"control_step={control_step or '-'} "
                f"env_step={step_count} "
                "reason=ENV_TRUNCATED"
            ),
            "outcome": SubtaskStepOutcome(
                done=True,
                success=False,
                failure_reason="ENV_TRUNCATED",
                feedback=feedback_factory(),
            ),
        }

    if subtask_completed and not (success_flag or subtask_succeeded):
        failure_reason = subtask_failure_reason(subtask_completion_reason)
        payload = {
            "subtask_id": subtask.subtask_id,
            "reason": failure_reason,
            "attempt": attempt,
            "control_step": control_step,
            "step_count": step_count,
            "task_progress": task_progress,
            "subtask_completion_reason": subtask_completion_reason,
        }
        if action_completion_diagnostics is not None:
            payload["action_completion_diagnostics"] = action_completion_diagnostics
        return {
            "event": "subtask_failed",
            "payload": payload,
            "progress_message": (
                "[closed-loop] fail "
                f"subtask={subtask.subtask_id} "
                f"attempt={attempt} "
                f"control_step={control_step or '-'} "
                f"env_step={step_count} "
                f"reason={failure_reason}"
            ),
            "outcome": SubtaskStepOutcome(
                done=True,
                success=False,
                failure_reason=failure_reason,
                feedback=feedback_factory(
                    extras={"subtask_completion_reason": subtask_completion_reason},
                ),
            ),
        }

    if terminated and not (success_flag or subtask_completed):
        payload = {
            "subtask_id": subtask.subtask_id,
            "reason": "ENV_TERMINATED",
            "attempt": attempt,
            "control_step": control_step,
            "step_count": step_count,
            "task_progress": task_progress,
        }
        if action_completion_diagnostics is not None:
            payload["action_completion_diagnostics"] = action_completion_diagnostics
        return {
            "event": "subtask_failed",
            "payload": payload,
            "progress_message": (
                "[closed-loop] fail "
                f"subtask={subtask.subtask_id} "
                f"attempt={attempt} "
                f"control_step={control_step or '-'} "
                f"env_step={step_count} "
                "reason=ENV_TERMINATED"
            ),
            "outcome": SubtaskStepOutcome(
                done=True,
                success=False,
                failure_reason="ENV_TERMINATED",
                feedback=feedback_factory(),
            ),
        }

    return {
        "event": "subtask_done",
        "payload": {
            "subtask_id": subtask.subtask_id,
            "attempt": attempt,
            "control_step": control_step,
            "step_count": step_count,
            "task_progress": task_progress,
            "task_success": success_flag,
            "subtask_completed": subtask_completed,
            "subtask_succeeded": subtask_succeeded,
            "subtask_completion_reason": subtask_completion_reason,
            "subtask_name": resolved_subtask_name,
            "env_subtask_name": env_subtask_name,
        },
        "progress_message": (
            "[closed-loop] done "
            f"subtask={subtask.subtask_id} "
            f"attempt={attempt} "
            f"control_step={control_step or '-'} "
            f"env_step={step_count} "
            f"task_progress={format_float(task_progress)} "
            f"task_success={success_flag} "
            f"subtask_completed={subtask_completed} "
            f"subtask_succeeded={subtask_succeeded} "
            f"reason={subtask_completion_reason or '-'}"
        ),
        "outcome": SubtaskStepOutcome(
            done=True,
            success=True,
            feedback=feedback_factory(
                extras={
                    "subtask_completed": subtask_completed,
                    "subtask_succeeded": subtask_succeeded,
                    "subtask_completion_reason": subtask_completion_reason,
                    "task_success": success_flag,
                },
            ),
        ),
    }


def build_final_step_response(
    *,
    subtask: Any,
    attempt: int,
    control_step: int | None,
    step_count: int,
    task_progress: float,
    success_flag: bool,
    subtask_completed: bool,
    subtask_succeeded: bool,
    subtask_completion_reason: str | None,
    terminated: bool,
    truncated: bool,
    resolved_subtask_name: str,
    env_subtask_name: str | None,
    action_progress: dict[str, Any] | None,
    subtask_failure_reason: Callable[[str | None], str],
    format_float: Callable[[Any], str],
    feedback_factory: Callable[..., Any],
    action_completion_diagnostics: dict[str, Any] | None = None,
    environment_success_evidence_only: bool = False,
) -> dict[str, Any]:
    action_plan_completed = bool((action_progress or {}).get("plan_completed", False))
    if _action_state_completion_evidence_only(
        subtask=subtask,
        subtask_completed=subtask_completed,
        subtask_completion_reason=subtask_completion_reason,
        terminated=terminated,
        truncated=truncated,
    ):
        return {
            "event": None,
            "payload": None,
            "progress_message": None,
            "outcome": SubtaskStepOutcome(
                done=False,
                success=None,
                feedback=feedback_factory(
                    extras={
                        "subtask_completed": subtask_completed,
                        "subtask_succeeded": subtask_succeeded,
                        "subtask_completion_reason": subtask_completion_reason,
                        "environment_success_evidence_only": True,
                    },
                ),
            ),
        }
    if (
        _agent_value(subtask) == "ACTION"
        and subtask_completed
        and subtask_completion_reason == "vlm_success"
        and not success_flag
        and not action_plan_completed
    ):
        return build_vla_step_response(
            subtask=subtask,
            attempt=attempt,
            control_step=control_step,
            step_count=step_count,
            task_progress=task_progress,
            success_flag=success_flag,
            terminated=terminated,
            truncated=truncated,
            action_progress=action_progress,
            action_completion_diagnostics=action_completion_diagnostics,
            environment_success_evidence_only=environment_success_evidence_only,
            feedback_factory=feedback_factory,
        )

    if (
        _agent_value(subtask) == "ACTION"
        and action_progress is not None
        and not subtask_completed
    ):
        return build_vla_step_response(
            subtask=subtask,
            attempt=attempt,
            control_step=control_step,
            step_count=step_count,
            task_progress=task_progress,
            success_flag=success_flag,
            terminated=terminated,
            truncated=truncated,
            action_progress=action_progress,
            action_completion_diagnostics=action_completion_diagnostics,
            environment_success_evidence_only=environment_success_evidence_only,
            feedback_factory=feedback_factory,
        )

    return build_standard_step_response(
        subtask=subtask,
        attempt=attempt,
        control_step=control_step,
        step_count=step_count,
        task_progress=task_progress,
        success_flag=success_flag,
        subtask_completed=subtask_completed,
        subtask_succeeded=subtask_succeeded,
        subtask_completion_reason=subtask_completion_reason,
        terminated=terminated,
        truncated=truncated,
        resolved_subtask_name=resolved_subtask_name,
        env_subtask_name=env_subtask_name,
        action_completion_diagnostics=action_completion_diagnostics,
        environment_success_evidence_only=environment_success_evidence_only,
        subtask_failure_reason=subtask_failure_reason,
        format_float=format_float,
        feedback_factory=feedback_factory,
    )


def _logging_completion_state(
    *,
    subtask: Any,
    success_flag: bool,
    subtask_completed: bool,
    subtask_succeeded: bool,
    subtask_completion_reason: str | None,
    terminated: bool,
    truncated: bool,
    environment_success_evidence_only: bool,
) -> dict[str, Any]:
    """Return completion fields for step/progress logs.

    When environment success is evidence-only, the raw simulator signal should
    be available to the monitor but must not make runtime logs look like the
    subtask has already advanced.
    """

    env_success_only = (
        environment_success_evidence_only
        and subtask_completed
        and subtask_completion_reason == "task_success"
    )
    action_state_evidence_only = (
        _agent_value(subtask) == "ACTION"
        and str(subtask_completion_reason or "") in ACTION_STATE_COMPLETION_REASONS
        and not terminated
        and not truncated
    )
    if not (env_success_only or action_state_evidence_only):
        return {
            "done": bool(terminated or truncated or subtask_completed),
            "success_flag": success_flag,
            "subtask_completed": subtask_completed,
            "subtask_succeeded": subtask_succeeded,
            "subtask_completion_reason": subtask_completion_reason,
        }
    return {
        "done": bool(truncated),
        "success_flag": False,
        "subtask_completed": False,
        "subtask_succeeded": False,
        "subtask_completion_reason": None,
    }


def finalize_step(
    *,
    subtask: Any,
    attempt: int,
    control_step: int | None,
    action: dict[str, Any],
    step_count: int,
    last_obs: dict[str, Any],
    last_info: dict[str, Any],
    last_reward: Any,
    terminated: bool,
    truncated: bool,
    success_flag: bool,
    subtask_completed: bool,
    subtask_succeeded: bool,
    subtask_completion_reason: str | None,
    active_internal: dict[str, Any] | None,
    action_progress: dict[str, Any] | None,
    resolved_subtask_name: str,
    env_subtask_name: str | None,
    record_frame: Callable[[dict[str, Any]], None],
    record_step_events: Callable[..., None],
    maybe_log_progress: Callable[..., None],
    subtask_failure_reason: Callable[[str | None], str],
    format_float: Callable[[Any], str],
    feedback_factory: Callable[..., Any],
    record_event: Callable[[str, dict[str, Any]], None],
    emit_progress: Callable[[str], None],
    motion_diagnostics: dict[str, Any] | None = None,
    environment_success_evidence_only: bool = False,
) -> SubtaskStepOutcome:
    logging_state = _logging_completion_state(
        subtask=subtask,
        success_flag=success_flag,
        subtask_completed=subtask_completed,
        subtask_succeeded=subtask_succeeded,
        subtask_completion_reason=subtask_completion_reason,
        terminated=terminated,
        truncated=truncated,
        environment_success_evidence_only=environment_success_evidence_only,
    )
    record_frame(last_obs)
    record_step_events(
        subtask=subtask,
        attempt=attempt,
        control_step=control_step,
        step_count=step_count,
        reward=last_reward,
        terminated=terminated,
        truncated=truncated,
        last_info=last_info,
        success_flag=logging_state["success_flag"],
        subtask_completed=logging_state["subtask_completed"],
        subtask_succeeded=logging_state["subtask_succeeded"],
        subtask_completion_reason=logging_state["subtask_completion_reason"],
        resolved_subtask_name=resolved_subtask_name,
        env_subtask_name=env_subtask_name,
        active_internal=active_internal,
        action_progress=action_progress,
        motion_diagnostics=motion_diagnostics,
        environment_success_evidence_only=environment_success_evidence_only,
    )
    maybe_log_progress(
        subtask=subtask,
        attempt=attempt,
        control_step=control_step,
        action=action,
        done=logging_state["done"],
        success_flag=logging_state["success_flag"],
        subtask_completed=logging_state["subtask_completed"],
        subtask_succeeded=logging_state["subtask_succeeded"],
        subtask_completion_reason=logging_state["subtask_completion_reason"],
    )
    response = build_final_step_response(
        subtask=subtask,
        attempt=attempt,
        control_step=control_step,
        step_count=step_count,
        task_progress=last_info.get("task_progress", 0.0),
        success_flag=success_flag,
        subtask_completed=subtask_completed,
        subtask_succeeded=subtask_succeeded,
        subtask_completion_reason=subtask_completion_reason,
        terminated=terminated,
        truncated=truncated,
        resolved_subtask_name=resolved_subtask_name,
        env_subtask_name=env_subtask_name,
        action_progress=action_progress,
        environment_success_evidence_only=environment_success_evidence_only,
        action_completion_diagnostics=last_info.get("action_completion_diagnostics"),
        subtask_failure_reason=subtask_failure_reason,
        format_float=format_float,
        feedback_factory=feedback_factory,
    )
    return emit_step_response(
        response=response,
        record_event=record_event,
        emit_progress=emit_progress,
    )
