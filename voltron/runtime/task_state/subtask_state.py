"""Runtime helpers for per-subtask execution state."""

from __future__ import annotations

from typing import Any, Callable

from voltron.shared.context import Subtask


def subtask_max_steps(subtask: Subtask, default_subtask_max_steps: int | None) -> int:
    candidates = [
        subtask.parameters.get("max_steps"),
        subtask.context.get("max_steps"),
        default_subtask_max_steps,
        120,
    ]
    for value in candidates:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return 120


def build_runtime_subtask(
    *,
    subtask: Subtask,
    default_subtask_max_steps: int | None,
    instruction_for_subtask: Callable[[Subtask], str],
    planned_subtask_name: Callable[[Subtask], str],
) -> dict[str, Any]:
    payload = {
        "subtask_id": subtask.subtask_id,
        "agent": subtask.agent.value,
        "action": subtask.action,
        "instruction": instruction_for_subtask(subtask),
        "max_steps": subtask_max_steps(subtask, default_subtask_max_steps),
        "name": planned_subtask_name(subtask),
    }
    if subtask.execution_id:
        payload["execution_id"] = subtask.execution_id
        payload["plan_revision"] = subtask.plan_revision
    if subtask.replaces_execution_id:
        payload["replaces_execution_id"] = subtask.replaces_execution_id
    return payload


def sync_runtime_subtask(
    *,
    subtask: Subtask,
    runtime_subtasks_by_id: dict[str, dict[str, Any]],
    instruction_for_subtask: Callable[[Subtask], str],
    planned_subtask_name: Callable[[Subtask], str],
    last_info: dict[str, Any],
    call_env_method: Callable[..., Any],
    record_event: Callable[[str, dict[str, Any]], None],
) -> dict[str, Any]:
    runtime_subtask = runtime_subtasks_by_id.get(subtask.runtime_id)
    if runtime_subtask is None:
        runtime_subtask = runtime_subtasks_by_id.get(subtask.subtask_id)
    instruction = instruction_for_subtask(subtask)
    name = planned_subtask_name(subtask)
    if runtime_subtask is not None:
        instruction = str(runtime_subtask.get("instruction") or instruction)
        name = str(runtime_subtask.get("name") or name)
        call_env_method(
            "set_active_runtime_subtask",
            subtask_id=subtask.subtask_id,
            instruction=instruction,
        )

    if instruction:
        event_payload = {
            "subtask_id": subtask.subtask_id,
            "instruction": instruction,
        }
        if subtask.execution_id:
            event_payload.update(
                {
                    "execution_id": subtask.execution_id,
                    "plan_revision": subtask.plan_revision,
                }
            )
        record_event(
            "active_subtask_sync",
            event_payload,
        )

    updated_last_info = dict(last_info)
    if name:
        updated_last_info["subtask_name"] = name

    return {
        "runtime_subtask": runtime_subtask,
        "active_subtask_name": name or None,
        "active_subtask_instruction": instruction or None,
        "last_info": updated_last_info,
    }
