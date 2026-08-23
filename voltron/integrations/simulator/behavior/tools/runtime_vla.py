"""Action runtime-state helpers for the BEHAVIOR runtime bridge."""

from __future__ import annotations

from typing import Any

from voltron.shared.enums import AgentName


def extract_vla_execution_progress(result: Any) -> dict[str, Any] | None:
    progress = result.runtime_artifacts.get("action_execution_progress")
    if not isinstance(progress, dict):
        return None
    return dict(progress)


def extract_vla_active_internal_step(result: Any) -> dict[str, Any] | None:
    payload = result.runtime_artifacts.get("action_active_internal_step")
    if not isinstance(payload, dict):
        return None
    return dict(payload)


def capture_vla_internal_runtime_state(
    *,
    subtask: Any,
    result: Any,
    attempt: int,
    control_step: int | None,
    logged_action_internal_attempts: set[tuple[str, int]],
    logged_action_internal_replans: set[tuple[str, int]],
    display_name_builder: Any,
    build_plan_created_payload: Any,
    build_replan_payload: Any,
    build_step_start_payload: Any,
) -> dict[str, Any]:
    if subtask.agent != AgentName.ACTION:
        return {
            "active_internal_step": None,
            "logged_action_internal_attempts": set(logged_action_internal_attempts),
            "logged_action_internal_replans": set(logged_action_internal_replans),
            "events": [],
        }

    active_internal = extract_vla_active_internal_step(result)
    if active_internal is None:
        return {
            "active_internal_step": None,
            "logged_action_internal_attempts": set(logged_action_internal_attempts),
            "logged_action_internal_replans": set(logged_action_internal_replans),
            "events": [],
        }

    active_internal["display_name"] = display_name_builder(active_internal)
    progress = extract_vla_execution_progress(result) or {}
    execution_plan = dict(result.runtime_artifacts.get("action_execution_plan") or {})
    events: list[dict[str, Any]] = []

    if bool(progress.get("plan_created")):
        events.append(
            {
                "event": "action_internal_plan_created",
                "payload": build_plan_created_payload(
                    subtask_id=subtask.subtask_id,
                    active_internal=active_internal,
                    execution_plan=execution_plan,
                    progress=progress,
                ),
            }
        )

    next_logged_replans = set(logged_action_internal_replans)
    replan_history = result.runtime_artifacts.get("action_replan_history")
    if isinstance(replan_history, list):
        for index, replan_entry in enumerate(replan_history):
            if not isinstance(replan_entry, dict):
                continue
            replan_marker = (str(subtask.runtime_id), index)
            if replan_marker in next_logged_replans:
                continue
            next_logged_replans.add(replan_marker)
            events.append(
                {
                    "event": "action_internal_replan",
                    "payload": build_replan_payload(
                        subtask_id=subtask.subtask_id,
                        active_internal=active_internal,
                        execution_plan=execution_plan,
                        progress=progress,
                        replan_entry=replan_entry,
                    ),
                }
            )

    next_logged = set(logged_action_internal_attempts)
    marker = (str(active_internal.get("internal_step_id") or ""), attempt)
    if marker not in next_logged:
        next_logged.add(marker)
        events.append(
            {
                "event": "action_internal_step_start",
                "payload": build_step_start_payload(
                    subtask_id=subtask.subtask_id,
                    attempt=attempt,
                    control_step=control_step,
                    active_internal=active_internal,
                ),
            }
        )

    return {
        "active_internal_step": active_internal,
        "logged_action_internal_attempts": next_logged,
        "logged_action_internal_replans": next_logged_replans,
        "events": events,
    }


def apply_vla_internal_runtime_state(
    *,
    subtask: Any,
    result: Any,
    attempt: int,
    control_step: int | None,
    logged_action_internal_attempts: set[tuple[str, int]],
    logged_action_internal_replans: set[tuple[str, int]],
    display_name_builder: Any,
    build_plan_created_payload: Any,
    build_replan_payload: Any,
    build_step_start_payload: Any,
    record_event: Any,
) -> dict[str, Any]:
    captured = capture_vla_internal_runtime_state(
        subtask=subtask,
        result=result,
        attempt=attempt,
        control_step=control_step,
        logged_action_internal_attempts=logged_action_internal_attempts,
        logged_action_internal_replans=logged_action_internal_replans,
        display_name_builder=display_name_builder,
        build_plan_created_payload=build_plan_created_payload,
        build_replan_payload=build_replan_payload,
        build_step_start_payload=build_step_start_payload,
    )
    for event in captured["events"]:
        record_event(event["event"], event["payload"])
    return {
        "active_internal_step": captured["active_internal_step"],
        "logged_action_internal_attempts": captured["logged_action_internal_attempts"],
        "logged_action_internal_replans": captured["logged_action_internal_replans"],
    }
