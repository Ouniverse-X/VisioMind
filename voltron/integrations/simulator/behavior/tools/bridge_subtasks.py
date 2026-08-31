from __future__ import annotations

from typing import Any, Callable

from voltron.integrations.simulator.behavior.tools import subtasks as behavior_subtasks
from voltron.runtime.task_state import subtask_state as runtime_subtask_state
from voltron.shared.context import Plan, Subtask


def build_runtime_subtasks(
    *,
    plan: Plan,
    default_subtask_max_steps: int | None,
    slugify: Callable[[str], str],
) -> list[dict[str, Any]]:
    return [
        build_runtime_subtask(
            subtask=subtask,
            default_subtask_max_steps=default_subtask_max_steps,
            slugify=slugify,
        )
        for subtask in plan.subtasks
    ]


def build_runtime_subtask(
    *,
    subtask: Subtask,
    default_subtask_max_steps: int | None,
    slugify: Callable[[str], str],
) -> dict[str, Any]:
    return runtime_subtask_state.build_runtime_subtask(
        subtask=subtask,
        default_subtask_max_steps=default_subtask_max_steps,
        instruction_for_subtask=instruction_for_subtask,
        planned_subtask_name=lambda item: planned_subtask_name(item, slugify=slugify),
    )


def subtask_max_steps(subtask: Subtask, default_subtask_max_steps: int | None) -> int:
    return runtime_subtask_state.subtask_max_steps(subtask, default_subtask_max_steps)


def instruction_for_subtask(subtask: Subtask) -> str:
    return behavior_subtasks.instruction_for_subtask(subtask)


def render_target_phrase(target: dict[str, Any]) -> str:
    return behavior_subtasks.render_target_phrase(target)


def first_target_value(target: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    return behavior_subtasks.first_target_value(target, keys)


def first_target_value_raw(target: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    return behavior_subtasks.first_target_value_raw(target, keys)


def planned_subtask_name(subtask: Subtask, *, slugify: Callable[[str], str]) -> str:
    return behavior_subtasks.planned_subtask_name(subtask, slugify=slugify)


def env_subtask_name(last_info: dict[str, Any]) -> str | None:
    return behavior_subtasks.env_subtask_name(last_info)


def recording_subtask_name(
    *,
    active_internal_step: dict[str, Any] | None,
    last_info: dict[str, Any],
    active_subtask_name: str | None,
) -> str | None:
    return behavior_subtasks.recording_subtask_name(
        active_internal_step=active_internal_step,
        env_subtask_name=env_subtask_name(last_info),
        active_subtask_name=active_subtask_name,
    )


def recording_subtask_instruction(
    *,
    active_internal_step: dict[str, Any] | None,
    active_subtask_instruction: str | None,
) -> str | None:
    return behavior_subtasks.recording_subtask_instruction(
        active_internal_step=active_internal_step,
        active_subtask_instruction=active_subtask_instruction,
    )


def resolved_subtask_name(
    *,
    subtask: Subtask,
    active_internal_step: dict[str, Any] | None,
    last_info: dict[str, Any],
    active_subtask_name: str | None,
    slugify: Callable[[str], str],
) -> str:
    return behavior_subtasks.resolved_subtask_name(
        subtask=subtask,
        active_internal_step=active_internal_step,
        env_subtask_name=env_subtask_name(last_info),
        planned_subtask_name=planned_subtask_name(subtask, slugify=slugify),
    )
