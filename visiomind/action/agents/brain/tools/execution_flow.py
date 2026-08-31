from __future__ import annotations

from typing import Any

from . import navigation_runtime, planning_runtime
from visiomind.action.shared.context import ExecutionContext, Plan


def build_execution_state_payload(
    *,
    context: ExecutionContext,
    latest_result: Any,
    planner_mode: str,
) -> dict[str, Any]:
    serialized_latest = planning_runtime.serialize_result(
        latest_result,
        environment_state=context.runtime_state.get("environment"),
    )
    serialized_recent_results = [
        planning_runtime.serialize_result(item) for item in context.results[-5:]
    ]
    last_scene_report = planning_runtime.last_scene_report(context.results)
    navigation_state = navigation_runtime.resolve_navigation_state(
        execution_state={
            "latest_result": serialized_latest,
            "recent_results": serialized_recent_results,
        },
        environment_state=context.runtime_state.get("environment"),
    )
    navigation_report = navigation_runtime.build_navigation_report(
        context=context,
        latest_result=serialized_latest,
        navigation_state=navigation_state,
    )
    plan_history = list(context.runtime_state.get("plan_history", []))[-5:]
    active_plan = list(plan_history[-1].get("subtasks") or []) if plan_history else []
    return {
        "task_type": context.task_request.task_type.value,
        "planner_mode": planner_mode,
        "next_subtask_index": planning_runtime.next_subtask_index(context.results, latest_result),
        "latest_result": serialized_latest,
        "recent_results": serialized_recent_results,
        "last_scene_report": last_scene_report,
        "navigation_state": navigation_state,
        "navigation_report": navigation_report,
        "task_progress": planning_runtime.extract_task_progress(latest_result),
        "environment_feedback": navigation_runtime.runtime_feedback_dict(
            getattr(latest_result, "result", {}).get("env_feedback")
        ),
        "completed_subtasks": [item.subtask_id for item in context.results],
        "completed_execution_ids": [
            str(getattr(item, "result", {}).get("execution_id") or item.subtask_id)
            for item in context.results
        ],
        "current_plan_revision": context.runtime_state.get("plan_revision"),
        "current_plan": active_plan,
        "current_plan_execution_ids": list(
            context.runtime_state.get("current_plan_execution_ids") or []
        ),
        "plan_history": plan_history,
    }


def build_task_context_update_payload(
    *,
    context: ExecutionContext,
    plan: Plan,
    reason: str,
    execution_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current_plan = [planning_runtime.serialize_subtask_summary(item) for item in plan.subtasks]
    current_subtask = current_plan[0] if current_plan else {}
    latest_execution = execution_state or {}
    latest_result = dict(latest_execution.get("latest_result") or {})
    latest_feedback = navigation_runtime.runtime_feedback_dict(latest_result.get("env_feedback"))
    return {
        "runtime_namespace": planning_runtime.build_runtime_namespace(context),
        "execution_state": {
            "task_phase": planning_runtime.format_task_phase(current_subtask),
            "current_plan": current_plan,
            "current_subtask": current_subtask,
            "recent_decisions": planning_runtime.recent_plan_decisions(context),
            "robot_state": {
                "current_room": latest_feedback.get("current_room"),
                "current_region": latest_feedback.get("current_region"),
                "pose": latest_feedback.get("pose"),
            },
            "latest_scene_report": dict(latest_execution.get("last_scene_report") or {}),
            "latest_navigation_report": dict(latest_execution.get("navigation_report") or {}),
            "plan_reason": reason,
        },
    }
