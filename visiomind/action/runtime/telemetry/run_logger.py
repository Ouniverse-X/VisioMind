from __future__ import annotations

from typing import Any

from visiomind.action.shared.context import ExecutionContext


def build_task_run_response(context: ExecutionContext, final: dict[str, Any]) -> dict[str, Any]:
    return {
        "trace_id": context.trace_id,
        "task_id": context.task_request.task_id,
        "task_description": context.task_request.description,
        "task_type": context.task_request.task_type.value,
        "started_at": context.started_at,
        "results": [
            {
                "subtask_id": result.subtask_id,
                "status": result.status.value,
                "error_code": result.error_code,
                "result": result.result,
                "state_changes": result.state_changes,
                "latency_ms": result.latency_ms,
            }
            for result in context.results
        ],
        "final": final,
    }
