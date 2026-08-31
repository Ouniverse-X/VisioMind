from __future__ import annotations

import json

from visiomind.action.shared.context import ExecutionContext, Plan, Subtask, TaskRequest
from visiomind.action.shared.contracts import RuntimeEnvironment
from visiomind.action.shared.results import AgentResult


_TRANSIENT_BACKEND_ERRORS = {
    "NAV_BACKEND_TIMEOUT",
    "NAV_BACKEND_RATE_LIMITED",
    "NAV_BACKEND_CONNECTION_ERROR",
    "ANYGRASP_FAILED",
    "PLACE_EXECUTION_FAILED",
}


def update_environment_plan(
    *,
    environment: RuntimeEnvironment,
    context: ExecutionContext,
    plan: Plan,
) -> None:
    if not plan.subtasks:
        return
    environment.update_plan(plan=plan, context=context)


def plan_changed(*, previous: Plan, current: Plan) -> bool:
    if previous.metadata != current.metadata:
        return True
    if len(previous.subtasks) != len(current.subtasks):
        return True
    for prev, curr in zip(previous.subtasks, current.subtasks):
        if prev != curr:
            return True
    return False


def execute_with_retry(
    *,
    orchestrator: object,
    request: TaskRequest,
    context: ExecutionContext,
    subtask: Subtask,
    environment: RuntimeEnvironment,
) -> tuple[AgentResult, list[Subtask], bool]:
    current_subtask = subtask
    attempts = 0
    last_result: AgentResult | None = None
    replanned_followups: list[Subtask] = []
    replaced_pending_plan = False

    while attempts <= orchestrator.max_retries:
        result = orchestrator._run_subtask_control_loop(
            subtask=current_subtask,
            context=context,
            environment=environment,
            attempt=attempts + 1,
        )
        if result.status.value == "success":
            return result, replanned_followups, replaced_pending_plan

        attempts += 1
        last_result = result
        replanned_followups = []
        if attempts > orchestrator.max_retries:
            break
        if result.error_code in _TRANSIENT_BACKEND_ERRORS:
            orchestrator._emit_event(
                event_type="transient_retry",
                source=current_subtask.agent.value,
                message=(
                    f"retry {current_subtask.subtask_id} after transient "
                    f"{result.error_code} ({attempts}/{orchestrator.max_retries})"
                ),
                payload={
                    "subtask_id": current_subtask.subtask_id,
                    "error_code": result.error_code,
                    "attempt": attempts,
                    "max_retries": max(0, int(orchestrator.max_retries)),
                    "error": dict(result.result),
                },
                task_id=request.task_id,
            )
            continue
        if not context.runtime_state.get("dynamic_execution", False):
            break

        failure_reason = result.error_code or "UNKNOWN_ERROR"
        failure_signature = _failure_signature(
            subtask=current_subtask,
            failure_reason=failure_reason,
        )
        failure_count = _record_failure_signature(
            context=context,
            signature=failure_signature,
        )
        if failure_count > max(0, int(orchestrator.max_retries)):
            result.result.setdefault("replan_suppressed", True)
            result.result.setdefault("replan_suppression_reason", "repeated_failure_signature")
            result.result.setdefault("replan_failure_signature", failure_signature)
            result.result.setdefault("replan_failure_count", failure_count)
            orchestrator._emit_event(
                event_type="replan_suppressed",
                source="BRAIN",
                message=(
                    f"skip replan after {current_subtask.subtask_id}: "
                    f"repeated {failure_reason} for {failure_signature}"
                ),
                payload={
                    "failed_subtask_id": current_subtask.subtask_id,
                    "failure_reason": failure_reason,
                    "failure_signature": failure_signature,
                    "failure_count": failure_count,
                    "max_retries": max(0, int(orchestrator.max_retries)),
                },
                task_id=request.task_id,
            )
            break

        orchestrator._emit_event(
            event_type="replan_start",
            source="BRAIN",
            message=f"replan after {current_subtask.subtask_id}: {failure_reason}",
            payload={
                "failed_subtask_id": current_subtask.subtask_id,
                "failed_execution_id": current_subtask.runtime_id,
                "failed_plan_revision": current_subtask.plan_revision,
                "failure_reason": failure_reason,
                "attempt": attempts,
            },
            task_id=request.task_id,
        )
        try:
            replanned = orchestrator.brain_agent.replan(
                request=request,
                context=context,
                failed_subtask=current_subtask,
                failure_reason=failure_reason,
                latest_result=result,
            )
        except Exception as exc:
            result.result.setdefault("replan_error", f"{type(exc).__name__}: {exc}")
            result.result.setdefault("failed_subtask", current_subtask.subtask_id)
            break
        if not replanned.subtasks:
            break
        orchestrator._emit_event(
            event_type="brain_plan",
            source="BRAIN",
            message=f"replan with {len(replanned.subtasks)} subtasks",
            payload=orchestrator._serialize_plan_event_payload(
                plan=replanned,
                reason="replan",
                failure_reason=failure_reason,
                attempt=attempts,
            ),
            task_id=request.task_id,
        )
        update_environment_plan(environment=environment, context=context, plan=replanned)
        context.runtime_state["current_plan_subtask_ids"] = [
            item.subtask_id for item in replanned.subtasks
        ]
        context.runtime_state["current_plan_execution_ids"] = [
            item.runtime_id for item in replanned.subtasks
        ]
        current_subtask = replanned.subtasks[0]
        replanned_followups = list(replanned.subtasks[1:])
        replaced_pending_plan = True

    if last_result is None:
        raise RuntimeError("_execute_with_retry reached impossible state")
    return last_result, [], replaced_pending_plan


def _failure_signature(*, subtask: Subtask, failure_reason: str) -> str:
    target = json.dumps(subtask.target, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "|".join(
        [
            subtask.agent.value,
            str(subtask.action or ""),
            target,
            str(failure_reason or ""),
        ]
    )


def _record_failure_signature(*, context: ExecutionContext, signature: str) -> int:
    counts = context.runtime_state.setdefault("replan_failure_signature_counts", {})
    if not isinstance(counts, dict):
        counts = {}
        context.runtime_state["replan_failure_signature_counts"] = counts
    count = int(counts.get(signature, 0)) + 1
    counts[signature] = count
    return count
