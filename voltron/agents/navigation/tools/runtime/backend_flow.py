"""Backend planning-flow helpers for the Navigation agent runtime."""

from __future__ import annotations

from typing import Any

from . import object_approach
from . import skill_routing
from . import execution_context
from voltron.shared.context import ExecutionContext, Subtask


def resolve_grounded_goal_bundle(
    *,
    navigator: Any,
    memory: Any,
    selector: Any,
    skill_registry: Any,
    approach_point_selector: Any,
    subtask: Subtask,
    context: ExecutionContext,
    scene_id: str | None,
    start_state: dict[str, Any],
    nav_context: dict[str, Any],
    observation: dict[str, Any],
) -> dict[str, Any]:
    grounded_goal: dict[str, Any] | None = None
    path_plan: dict[str, Any] | None = None
    navigation_skill_selection: dict[str, Any] | None = None
    prepared_navigation_payload: dict[str, Any] | None = None
    object_approach_selection: dict[str, Any] | None = None
    selected_object_approach: dict[str, Any] | None = None

    cached_object_approach = object_approach.load_cached_object_approach_state(context=context, subtask=subtask)
    if cached_object_approach is not None:
        restored = object_approach.restore_cached_object_approach_state(cached_object_approach)
        grounded_goal = restored["grounded_goal"]
        navigation_skill_selection = restored["navigation_skill_selection"]
        prepared_navigation_payload = restored["prepared_navigation_payload"]
        object_approach_selection = restored["object_approach_selection"]
        selected_object_approach = restored["selected_object_approach"]
        path_plan = restored["path_plan"]
    else:
        grounded_goal = navigator.ground_goal(
            execution_context.resolve_instruction(subtask),
            context=nav_context,
        )
        if object_approach.should_use_object_approach_flow(subtask=subtask, grounded_goal=grounded_goal):
            history = object_approach.prime_object_approach_history(
                memory=memory,
                subtask=subtask,
                scene_id=scene_id,
                goal=grounded_goal,
            )
            if history is not None:
                context.runtime_state["object_approach_history"] = history
            selection = skill_routing.select_navigation_skill(
                selector=selector,
                registry=skill_registry,
                subtask=subtask,
                context=context,
            )
            navigation_skill_selection = object_approach.serialize_skill_selection(selection)
            skill = skill_routing.resolve_navigation_skill(
                registry=skill_registry,
                subtask=subtask,
                context=context,
                selection=selection,
            )
            if skill is not None:
                prepared_navigation_payload = dict(
                    skill.prepare(
                        subtask=subtask,
                        context=context,
                        navigator=navigator,
                        start=start_state,
                        goal=grounded_goal,
                        navigation_context=nav_context,
                    )
                )
            grounded_goal, object_approach_selection, selected_object_approach = (
                object_approach.apply_prepared_payload_candidates(
                    context=context,
                    subtask=subtask,
                    grounded_goal=grounded_goal,
                    prepared_navigation_payload=prepared_navigation_payload,
                    approach_point_selector=approach_point_selector,
                )
            )
            object_approach.store_cached_object_approach_state(
                context=context,
                subtask=subtask,
                grounded_goal=grounded_goal,
                navigation_skill_selection=navigation_skill_selection,
                prepared_navigation_payload=prepared_navigation_payload,
                object_approach_selection=object_approach_selection,
                selected_object_approach=selected_object_approach,
            )
        else:
            object_approach.clear_cached_object_approach_state(context=context, subtask=subtask)

    should_plan_path = not object_approach.should_reuse_cached_path_plan(
        cached_object_approach=cached_object_approach,
        path_plan=path_plan,
        subtask=subtask,
        observation=observation,
    )
    if should_plan_path:
        path_plan = navigator.plan_path(
            start=start_state,
            goal=grounded_goal,
            context=nav_context,
        )
        if object_approach.should_use_object_approach_flow(subtask=subtask, grounded_goal=grounded_goal):
            object_approach.store_cached_object_approach_state(
                context=context,
                subtask=subtask,
                grounded_goal=grounded_goal,
                navigation_skill_selection=navigation_skill_selection,
                prepared_navigation_payload=prepared_navigation_payload,
                object_approach_selection=object_approach_selection,
                selected_object_approach=selected_object_approach,
                path_plan=path_plan,
            )

    return {
        "cached_object_approach": cached_object_approach,
        "grounded_goal": grounded_goal,
        "path_plan": path_plan,
        "navigation_skill_selection": navigation_skill_selection,
        "prepared_navigation_payload": prepared_navigation_payload,
        "object_approach_selection": object_approach_selection,
        "selected_object_approach": selected_object_approach,
    }
