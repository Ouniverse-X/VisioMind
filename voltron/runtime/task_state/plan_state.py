"""Plan-state helpers used by runtime control flows."""

from __future__ import annotations

from typing import Any, Callable

from voltron.shared.context import Plan, Subtask


def _runtime_subtask_key(item: dict[str, Any]) -> str:
    return str(item.get("execution_id") or item.get("subtask_id") or "")


def configure_runtime_subtasks(
    *,
    plan: Plan,
    env_kwargs: dict[str, Any],
    build_runtime_subtask: Callable[[Subtask], dict[str, Any]],
    call_env_method: Callable[[str, Any], Any],
) -> dict[str, Any]:
    runtime_subtasks = [build_runtime_subtask(subtask) for subtask in plan.subtasks]
    runtime_subtasks_by_id = {
        _runtime_subtask_key(item): item
        for item in runtime_subtasks
        if _runtime_subtask_key(item)
    }

    if runtime_subtasks:
        env_kwargs["runtime_subtasks"] = [dict(item) for item in runtime_subtasks]
    else:
        env_kwargs.pop("runtime_subtasks", None)

    call_env_method("set_runtime_subtasks", [dict(item) for item in runtime_subtasks])
    return {
        "runtime_subtasks": runtime_subtasks,
        "runtime_subtasks_by_id": runtime_subtasks_by_id,
    }


def merge_plan_runtime_subtasks(
    *,
    plan: Plan,
    runtime_subtasks: list[dict[str, Any]],
    runtime_subtasks_by_id: dict[str, dict[str, Any]],
    build_runtime_subtask: Callable[[Subtask], dict[str, Any]],
) -> dict[str, Any]:
    additions = [build_runtime_subtask(subtask) for subtask in plan.subtasks]
    replace_active_plan = bool(plan.metadata.get("replace_active_plan", False))
    if replace_active_plan:
        updated_runtime_subtasks = list(additions)
        updated_runtime_subtasks_by_id = {
            _runtime_subtask_key(item): item
            for item in additions
            if _runtime_subtask_key(item)
        }
        return {
            "runtime_subtasks": updated_runtime_subtasks,
            "runtime_subtasks_by_id": updated_runtime_subtasks_by_id,
            "additions": additions,
            "added_subtask_ids": [item.get("subtask_id") for item in additions],
            "added_execution_ids": [_runtime_subtask_key(item) for item in additions],
            "replaced_subtask_ids": [item.get("subtask_id") for item in runtime_subtasks],
            "replaced_execution_ids": [_runtime_subtask_key(item) for item in runtime_subtasks],
            "replace_active_plan": True,
        }

    updated_runtime_subtasks = list(runtime_subtasks)
    updated_runtime_subtasks_by_id = dict(runtime_subtasks_by_id)

    for item in additions:
        runtime_key = _runtime_subtask_key(item)
        if not runtime_key:
            continue
        if runtime_key in updated_runtime_subtasks_by_id:
            updated_runtime_subtasks_by_id[runtime_key].update(item)
            continue
        updated_runtime_subtasks.append(item)
        updated_runtime_subtasks_by_id[runtime_key] = item

    return {
        "runtime_subtasks": updated_runtime_subtasks,
        "runtime_subtasks_by_id": updated_runtime_subtasks_by_id,
        "additions": additions,
        "added_subtask_ids": [item.get("subtask_id") for item in additions],
        "added_execution_ids": [_runtime_subtask_key(item) for item in additions],
        "replaced_subtask_ids": [],
        "replaced_execution_ids": [],
        "replace_active_plan": False,
    }
