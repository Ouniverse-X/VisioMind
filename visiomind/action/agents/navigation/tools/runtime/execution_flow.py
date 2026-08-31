from __future__ import annotations

from typing import Any

from . import execution_context as runtime_execution_context
from . import observation as runtime_observation


def collect_runtime_inputs(
    *,
    subtask: Any,
    context: Any,
    observation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "scene_id": runtime_observation.resolve_scene_id(
            subtask=subtask, context=context, observation=observation
        ),
        "current_region": runtime_observation.extract_region(
            subtask=subtask, observation=observation
        ),
        "pose": runtime_observation.extract_pose(subtask=subtask, observation=observation),
        "orientation": runtime_observation.extract_orientation(
            subtask=subtask, observation=observation
        ),
    }


def build_memory_update_payload(
    *,
    scene_id: str | None,
    current_region: str | None,
    target_region: str,
    pose: dict[str, Any] | None,
    orientation: dict[str, Any] | None,
    nav_feedback: dict[str, Any] | None,
    obstacles: list[Any],
    policy_info: dict[str, Any] | None,
    grounded_goal: dict[str, Any] | None,
    path_plan: dict[str, Any] | None,
    navigation_skill_selection: dict[str, Any] | None,
    prepared_navigation_payload: dict[str, Any] | None,
    object_approach_selection: dict[str, Any] | None,
    selected_object_approach: dict[str, Any] | None,
    navigator_backend_name: str | None,
) -> dict[str, Any]:
    return {
        "scene_id": scene_id,
        "region": current_region,
        "target_region": target_region,
        "pose": pose,
        "orientation": orientation,
        "nav_feedback": nav_feedback,
        "obstacles": obstacles,
        "policy_info": policy_info,
        "grounded_goal": grounded_goal,
        "path_plan": path_plan,
        "navigation_skill_selection": navigation_skill_selection,
        "prepared_navigation_payload": prepared_navigation_payload,
        "object_approach_selection": object_approach_selection,
        "selected_object_approach": selected_object_approach,
        "grounding_candidates": (
            list(grounded_goal.get("grounding_candidates"))
            if isinstance(grounded_goal, dict)
            and isinstance(grounded_goal.get("grounding_candidates"), list)
            else []
        ),
        "selected_grounding_candidate": (
            dict(grounded_goal.get("selected_grounding_candidate"))
            if isinstance(grounded_goal, dict)
            and isinstance(grounded_goal.get("selected_grounding_candidate"), dict)
            else None
        ),
        "navigator_backend": navigator_backend_name,
    }


def build_success_payloads(
    *,
    action: dict[str, Any],
    projected_action: dict[str, Any],
    policy_info: dict[str, Any] | None,
    memory_update: dict[str, Any] | None,
    navigator_backend_name: str | None,
    grounded_goal: dict[str, Any] | None,
    scene_id: str | None,
    path_plan: dict[str, Any] | None,
    interpreted_goal: dict[str, Any] | None,
    navigation_grounding_context: dict[str, Any] | None,
    execution_bridge_artifacts: dict[str, Any],
    navigation_skill_selection: dict[str, Any] | None,
    prepared_navigation_payload: dict[str, Any] | None,
    object_approach_selection: dict[str, Any] | None,
    selected_object_approach: dict[str, Any] | None,
    policy_runtime_artifacts: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    result_payload = {
        "action_keys": sorted(projected_action.keys()),
        "policy_info": policy_info,
        "memory_update": memory_update,
        "navigator_backend": navigator_backend_name,
        "grounded_goal": grounded_goal or {},
        "scene_id": scene_id,
        "waypoint_count": runtime_execution_context.waypoint_count(path_plan),
    }
    runtime_artifacts = {
        "full_action": action,
        "projected_action": projected_action,
        "policy_info": policy_info,
        "interpreted_goal": dict(interpreted_goal) if isinstance(interpreted_goal, dict) else None,
        "navigation_grounding_context": (
            dict(navigation_grounding_context)
            if isinstance(navigation_grounding_context, dict)
            else None
        ),
        **execution_bridge_artifacts,
        "navigation_skill_selection": navigation_skill_selection,
        "prepared_navigation_payload": prepared_navigation_payload,
        "object_approach_selection": object_approach_selection,
        "selected_object_approach": selected_object_approach,
        "grounding_candidates": (
            list(grounded_goal.get("grounding_candidates"))
            if isinstance(grounded_goal, dict)
            and isinstance(grounded_goal.get("grounding_candidates"), list)
            else []
        ),
        "selected_grounding_candidate": (
            dict(grounded_goal.get("selected_grounding_candidate"))
            if isinstance(grounded_goal, dict)
            and isinstance(grounded_goal.get("selected_grounding_candidate"), dict)
            else None
        ),
        **policy_runtime_artifacts,
    }
    return result_payload, runtime_artifacts
