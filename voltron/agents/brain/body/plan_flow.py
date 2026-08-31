from __future__ import annotations

from typing import Any, Callable

from voltron.agents.brain.tools import execution_flow, interaction_flow, planning_runtime
from voltron.shared.context import ExecutionContext, Plan, Subtask, TaskRequest
from voltron.shared.enums import AgentName


def normalize_plan(
    *,
    plan: Plan,
    request: TaskRequest,
    seed_interaction: bool,
    should_apply_runtime_interaction_control: Callable[[TaskRequest, list[Subtask]], bool],
) -> Plan:
    normalized_metadata = dict(plan.metadata)
    normalized_metadata.setdefault("dynamic_execution", True)
    if _uses_native_pi05_policy(request):
        normalized_metadata["dynamic_execution"] = False

    normalized_subtasks: list[Subtask] = []
    for subtask in plan.subtasks:
        parameters = dict(subtask.parameters)
        context = dict(subtask.context)

        if subtask.agent == AgentName.ACTION:
            control_mode = request.metadata.get("action_control_mode")
            if isinstance(control_mode, str) and control_mode.strip():
                parameters.setdefault("control_mode", control_mode.strip())
            else:
                parameters.setdefault("control_mode", "whole_body_local")
            if request.metadata.get("action_allow_base_motion") is True:
                parameters.setdefault("allow_base_motion", True)
            context.setdefault("task_description", request.description)

        normalized_subtasks.append(
            Subtask(
                subtask_id=subtask.subtask_id,
                agent=subtask.agent,
                action=subtask.action,
                target=dict(subtask.target),
                parameters=parameters,
                context=context,
            )
        )

    if _uses_native_pi05_policy(request):
        normalized_subtasks = _canonicalize_pi05_plan(normalized_subtasks)

    if (
        seed_interaction
        and should_apply_runtime_interaction_control(request, normalized_subtasks)
        and normalized_metadata.get("dynamic_execution", False)
    ):
        normalized_subtasks = seed_interaction_plan(request=request, subtasks=normalized_subtasks)

    return Plan(subtasks=normalized_subtasks, metadata=normalized_metadata)


def version_plan(
    *,
    context: ExecutionContext,
    plan: Plan,
    reason: str,
    replaces_execution_id: str | None = None,
) -> Plan:
    if not plan.subtasks:
        return plan

    current_revision = context.runtime_state.get("plan_revision")
    if current_revision is None:
        revision = 0
    else:
        try:
            revision = int(current_revision) + 1
        except (TypeError, ValueError):
            revision = 0

    versioned_subtasks: list[Subtask] = []
    for index, subtask in enumerate(plan.subtasks, start=1):
        local_id = f"st_{index:02d}"
        execution_id = f"r{revision}/{local_id}"
        parameters = dict(subtask.parameters)
        criteria = parameters.get("completion_criteria")
        if isinstance(criteria, list):
            parameters["completion_criteria"] = [
                {
                    **item,
                    "subtask_id": local_id,
                    "execution_id": execution_id,
                    "plan_revision": revision,
                }
                if isinstance(item, dict)
                else item
                for item in criteria
            ]
        versioned_subtasks.append(
            Subtask(
                subtask_id=local_id,
                agent=subtask.agent,
                action=subtask.action,
                target=dict(subtask.target),
                parameters=parameters,
                context=dict(subtask.context),
                plan_revision=revision,
                execution_id=execution_id,
                replaces_execution_id=(replaces_execution_id if index == 1 else None),
            )
        )

    metadata = dict(plan.metadata)
    metadata.update(
        {
            "plan_revision": revision,
            "revision_reason": reason,
            "replace_active_plan": current_revision is not None,
        }
    )
    if replaces_execution_id:
        metadata["replaces_execution_id"] = replaces_execution_id

    context.runtime_state["plan_revision"] = revision
    context.runtime_state["current_plan_execution_ids"] = [
        item.runtime_id for item in versioned_subtasks
    ]
    return Plan(subtasks=versioned_subtasks, metadata=metadata)


def _uses_native_pi05_policy(request: TaskRequest) -> bool:
    return str(request.metadata.get("policy_backend") or "").strip().lower() == "pi05"


def _canonicalize_pi05_plan(subtasks: list[Subtask]) -> list[Subtask]:
    action_target = next(
        (
            dict(subtask.target)
            for subtask in subtasks
            if subtask.agent == AgentName.ACTION and subtask.target.get("object")
        ),
        None,
    )
    if action_target is None:
        return subtasks

    object_name = str(action_target.get("object") or "target object").strip() or "target object"
    canonical: list[Subtask] = []
    converted = False
    for subtask in subtasks:
        if not converted and subtask.agent == AgentName.NAVIGATION:
            target = dict(subtask.target)
            room = (
                target.get("room")
                or target.get("region")
                or action_target.get("room")
                or action_target.get("region")
            )
            target = {"object": object_name}
            if room:
                target["room"] = room
                target["region"] = room
            canonical.append(
                Subtask(
                    subtask_id=subtask.subtask_id,
                    agent=subtask.agent,
                    action="approach_target",
                    target=target,
                    parameters={
                        **subtask.parameters,
                        "instruction": f"Approach the {object_name} for local interaction.",
                    },
                    context=dict(subtask.context),
                )
            )
            converted = True
            continue
        canonical.append(subtask)
    return canonical


def record_plan(*, context: ExecutionContext, plan: Plan, reason: str) -> None:
    planning_runtime.record_plan(context=context, plan=plan, reason=reason)


def sync_working_memory_after_plan(
    *,
    memory: Any,
    context: ExecutionContext,
    plan: Plan,
    reason: str,
    execution_state: dict[str, Any] | None = None,
) -> None:
    memory.update_task_context(
        execution_flow.build_task_context_update_payload(
            context=context,
            plan=plan,
            reason=reason,
            execution_state=execution_state,
        )
    )


def build_execution_state(
    *,
    context: ExecutionContext,
    latest_result: Any,
    planner_mode: str,
) -> dict[str, Any]:
    return execution_flow.build_execution_state_payload(
        context=context,
        latest_result=latest_result,
        planner_mode=planner_mode,
    )


def seed_interaction_plan(*, request: TaskRequest, subtasks: list[Subtask]) -> list[Subtask]:
    return interaction_flow.build_seed_interaction_plan(subtasks=subtasks, request=request)


def serialize_subtask_summary(subtask: Subtask) -> dict[str, Any]:
    return planning_runtime.serialize_subtask_summary(subtask)


def format_task_phase(subtask_summary: dict[str, Any]) -> str | None:
    return planning_runtime.format_task_phase(subtask_summary)


def recent_plan_decisions(context: ExecutionContext) -> list[dict[str, Any]]:
    return planning_runtime.recent_plan_decisions(context)


def build_runtime_namespace(context: ExecutionContext) -> dict[str, Any]:
    return planning_runtime.build_runtime_namespace(context)
