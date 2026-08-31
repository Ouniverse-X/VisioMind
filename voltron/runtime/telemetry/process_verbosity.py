from __future__ import annotations

from typing import Any


_SUMMARY_KEYS = (
    "subtask_id",
    "agent",
    "status",
    "error_code",
    "control_step",
    "attempt",
)
_POLICY_INFO_KEYS = (
    "backend",
    "task_name",
    "task_id",
    "action_mode",
    "controller_mode",
    "goal_reached",
    "selected_candidate_id",
    "selected_candidate_source",
    "target_object_name",
    "nearest_object_name",
)


def filter_process_event_for_verbosity(
    *,
    event: str,
    payload: dict[str, Any],
    verbose: bool,
    memory_diagnostics: bool = False,
) -> tuple[str, dict[str, Any]] | None:
    if event == "memory_diagnostic" and not memory_diagnostics:
        return None
    if event == "orchestrator_task_final":
        return event, _compact_task_final_payload(payload)
    if verbose:
        return event, payload
    if event == "active_subtask_sync":
        return None
    if event == "env_step" and not _is_terminal_step(payload):
        return None
    if event == "progress_update":
        return event, _compact_progress_update_payload(payload)
    if event == "orchestrator_agent_result":
        return event, _compact_agent_result_payload(payload)
    if event == "env_step":
        return event, _compact_env_step_payload(payload)
    return event, payload


def _compact_task_final_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compact = dict(payload)
    environment = compact.get("environment")
    if not isinstance(environment, dict):
        return compact

    compact_environment = dict(environment)
    last_info = compact_environment.get("last_info")
    if isinstance(last_info, dict):
        compact_last_info = dict(last_info)
        compact_last_info.pop("scene_map_seed", None)
        compact_environment["last_info"] = compact_last_info
    compact["environment"] = compact_environment
    return compact


def _is_terminal_step(payload: dict[str, Any]) -> bool:
    if bool(payload.get("environment_success_evidence_only")):
        return bool(
            payload.get("truncated")
            or payload.get("subtask_completed")
            or payload.get("subtask_succeeded")
        )
    return bool(
        payload.get("terminated")
        or payload.get("truncated")
        or payload.get("subtask_completed")
        or payload.get("subtask_succeeded")
        or payload.get("task_success")
    )


def _compact_agent_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compact = {key: payload.get(key) for key in _SUMMARY_KEYS if key in payload}
    result = payload.get("result")
    if isinstance(result, dict):
        compact_result = {
            key: result.get(key)
            for key in (
                "agent",
                "attempt",
                "control_step",
                "action_keys",
                "memory_update",
                "success",
                "summary",
                "message",
                "error_type",
                "error_stage",
            )
            if result.get(key) not in (None, "", [], {})
        }
        policy_info = result.get("policy_info")
        if isinstance(policy_info, dict):
            policy_summary = {
                key: policy_info.get(key)
                for key in _POLICY_INFO_KEYS
                if policy_info.get(key) not in (None, "", [], {})
            }
            if policy_summary:
                compact_result["policy_info"] = policy_summary
        candidate_audit = result.get("candidate_detection_audit")
        if isinstance(candidate_audit, list) and candidate_audit:
            compact_result["candidate_detection_audit"] = candidate_audit
        execution_failure_audit = result.get("execution_failure_audit")
        if isinstance(execution_failure_audit, list) and execution_failure_audit:
            compact_result["execution_failure_audit"] = execution_failure_audit
        if compact_result:
            compact["result"] = compact_result
    return compact


def _compact_env_step_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload.get(key)
        for key in (
            "subtask_id",
            "agent",
            "attempt",
            "control_step",
            "step_count",
            "reward",
            "terminated",
            "truncated",
            "task_progress",
            "task_success",
            "subtask_completed",
            "subtask_succeeded",
            "subtask_completion_reason",
            "action_internal_step_id",
            "action_internal_step_name",
            "action_internal_step_index",
            "action_plan_completed",
            "environment_success_evidence_only",
        )
        if key in payload
    }


def _compact_progress_update_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: payload.get(key)
        for key in (
            "subtask_id",
            "agent",
            "attempt",
            "control_step",
            "step_count",
            "reward",
            "task_progress",
            "task_success",
            "subtask_completed",
            "subtask_succeeded",
            "subtask_completion_reason",
        )
        if key in payload
    }
    diagnostics = payload.get("action_completion_diagnostics")
    if isinstance(diagnostics, dict):
        if diagnostics.get("reason"):
            compact["action_completion_reason"] = diagnostics.get("reason")
        selected = diagnostics.get("selected_candidate")
        if isinstance(selected, dict):
            compact["selected_candidate"] = selected
    for key in (
        "active_waypoint_index",
        "controller_mode",
        "recovery_mode",
        "path_tracking_mode",
        "loop_detected",
        "oscillation_detected",
    ):
        if key in payload:
            compact[key] = payload.get(key)
    return compact
