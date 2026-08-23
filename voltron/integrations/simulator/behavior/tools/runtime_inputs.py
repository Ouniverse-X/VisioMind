"""Runtime input assembly helpers for the BEHAVIOR simulator integration."""

from __future__ import annotations

from typing import Any

from voltron.shared.enums import AgentName


_BEHAVIOR_TASK_INSTRUCTIONS: dict[str, str] = {
    "turning_on_radio": "Turn on the radio receiver that's on the table in the living room.",
    "bringing_water": "Retrieve the two bottles from the refrigerator in the kitchen, bring them to the living room, and place both on the coffee table. Make sure the refrigerator is closed when you finish.",
}


def behavior_task_instruction_for_env_id(env_id: str | None) -> str | None:
    if not isinstance(env_id, str):
        return None
    task_name = env_id.rsplit("/", 1)[-1].strip()
    return _BEHAVIOR_TASK_INSTRUCTIONS.get(task_name)


def observation_instruction(observation: dict[str, Any]) -> str:
    annotation = observation.get("annotation.human.coarse_action")
    if isinstance(annotation, (list, tuple)) and annotation:
        return str(annotation[0]).strip()
    if isinstance(annotation, str):
        return annotation.strip()
    return ""


def build_vision_runtime_inputs(
    *,
    runtime_subtask: dict[str, Any] | None,
    subtask: Any,
    images: list[str],
    image_view_order: list[str],
    environment_vlm_heartbeat: dict[str, Any] | None = None,
    camera_capture: Any | None = None,
    run_dir: Any | None = None,
) -> dict[str, Any]:
    payload = {
        "images": images,
        "image_view_order": image_view_order,
        "instruction": (runtime_subtask or {}).get(
            "instruction",
            subtask.parameters.get("instruction", subtask.action),
        ),
    }
    if environment_vlm_heartbeat is not None:
        payload["environment_vlm_heartbeat"] = dict(environment_vlm_heartbeat)
    if camera_capture is not None:
        payload["camera_capture"] = camera_capture
    if run_dir is not None:
        payload["run_dir"] = str(run_dir)
    return payload


def build_action_runtime_inputs(
    *,
    observation: dict[str, Any],
    last_obs: dict[str, Any],
    last_info: dict[str, Any],
    policy_instruction: str | None = None,
) -> dict[str, Any]:
    snapshot = state_snapshot(last_info)
    payload = {
        "observation": observation,
        "raw_observation": dict(last_obs),
        "pre_state": snapshot,
        "post_state": snapshot,
    }
    if isinstance(policy_instruction, str) and policy_instruction.strip():
        instruction = policy_instruction.strip()
        payload["instruction"] = instruction
        payload["policy_options"] = {"instruction": instruction}
        payload["vla_prompt"] = instruction
    return payload


def build_vln_runtime_inputs(
    *,
    subtask: Any,
    observation: dict[str, Any],
    env_kwargs: dict[str, Any],
    scene_id: str | None,
    pose: dict[str, Any] | None,
    orientation: dict[str, Any] | None,
    region: str | None,
    nav_feedback: dict[str, Any] | None,
    rgb: dict[str, Any] | None,
    depth: dict[str, Any] | None,
    navigation_runtime_state: dict[str, dict[str, Any]],
    scene_state: dict[str, Any] | None = None,
    simulator_pose: dict[str, Any] | None = None,
    simulator_orientation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    observation = dict(observation)
    if pose is not None:
        observation["pose"] = dict(pose)
    if orientation is not None:
        observation["orientation"] = dict(orientation)
        if "yaw" in orientation:
            observation["state.robot_2d_ori"] = [orientation["yaw"]]
    if simulator_pose is not None:
        observation["simulator_pose"] = dict(simulator_pose)
    if simulator_orientation is not None:
        observation["simulator_orientation"] = dict(simulator_orientation)
    scene_vertical_axis = env_kwargs.get("scene_vertical_axis")
    if isinstance(scene_vertical_axis, str) and scene_vertical_axis in {"x", "y", "z"}:
        observation["vertical_axis"] = scene_vertical_axis
    if isinstance(scene_state, dict) and scene_state:
        # The navigator only sees the inner observation (navigator.update), while
        # grounding context metadata is copied from the outer payload — feed both.
        observation = {**observation, "scene_state": scene_state}
    payload: dict[str, Any] = {"observation": observation}
    if isinstance(scene_state, dict) and scene_state:
        payload["scene_state"] = scene_state

    if scene_id:
        payload["scene_id"] = scene_id
    scene_file = env_kwargs.get("scene_file")
    if isinstance(scene_file, str) and scene_file.strip():
        payload["scene_file"] = scene_file.strip()
    nav2_trav_map_filename = env_kwargs.get("nav2_trav_map_filename")
    if isinstance(nav2_trav_map_filename, str) and nav2_trav_map_filename.strip():
        payload["nav2_trav_map_filename"] = nav2_trav_map_filename.strip()
    for key in ("portal_annotations", "transition_portals", "portals"):
        value = env_kwargs.get(key)
        if isinstance(value, (dict, list)) and value:
            payload[key] = value

    if pose is not None:
        payload["pose"] = pose
    if orientation is not None:
        payload["orientation"] = orientation
    if simulator_pose is not None:
        payload["simulator_pose"] = simulator_pose
    if simulator_orientation is not None:
        payload["simulator_orientation"] = simulator_orientation
    if region:
        payload["region"] = region
    if nav_feedback:
        payload["nav_feedback"] = nav_feedback
    if rgb:
        payload["rgb"] = rgb
    if depth:
        payload["depth"] = depth

    nav_state = navigation_runtime_state.get(subtask.runtime_id)
    if nav_state is None:
        nav_state = navigation_runtime_state.get(subtask.subtask_id)
    if nav_state:
        payload.update({key: value for key, value in nav_state.items() if value is not None})

    return payload


def build_runtime_inputs_for_subtask(
    *,
    subtask: Any,
    runtime_subtask: dict[str, Any] | None,
    last_obs: dict[str, Any],
    last_info: dict[str, Any],
    environment_vlm_heartbeat: dict[str, Any] | None = None,
    camera_capture: Any | None = None,
    run_dir: Any | None = None,
    build_vision_inputs: Any,
    build_action_inputs: Any,
    build_navigation_inputs: Any,
    extract_images_b64: Any,
    to_policy_observation: Any,
    policy_observation_source: Any,
    instruction_for_subtask: Any,
    policy_backend: str | None = None,
    env_id: str | None = None,
) -> dict[str, Any]:
    if subtask.agent == AgentName.VISION:
        images, image_view_order = extract_images_b64(last_obs)
        return build_vision_inputs(
            runtime_subtask=runtime_subtask,
            subtask=subtask,
            images=images,
            image_view_order=image_view_order,
            environment_vlm_heartbeat=environment_vlm_heartbeat,
            camera_capture=camera_capture,
            run_dir=run_dir,
        )

    if subtask.agent == AgentName.ACTION:
        active_instruction = str((runtime_subtask or {}).get("instruction") or "").strip()
        policy_instruction = None
        if policy_backend in {"pi05", "openpi_comet"}:
            source = policy_observation_source(language_instruction=None)
            policy_instruction = (
                active_instruction
                or instruction_for_subtask(subtask)
                or behavior_task_instruction_for_env_id(env_id)
                or observation_instruction(source)
            )
            observation = to_policy_observation(
                policy_observation_source(language_instruction=policy_instruction)
            )
        else:
            observation = to_policy_observation(
                policy_observation_source(
                    language_instruction=active_instruction or instruction_for_subtask(subtask),
                )
            )
        payload = build_action_inputs(
            observation=observation,
            last_obs=last_obs,
            last_info=last_info,
            policy_instruction=policy_instruction,
        )
        if policy_backend in {"pi05", "openpi_comet"}:
            _ensure_whole_body_action_defaults(payload=payload, subtask=subtask)
        return payload

    observation = to_policy_observation(policy_observation_source())
    return build_navigation_inputs(
        subtask=subtask,
        observation=observation,
    )


def _ensure_whole_body_action_defaults(*, payload: dict[str, Any], subtask: Any) -> None:
    parameters = getattr(subtask, "parameters", {})
    context = getattr(subtask, "context", {})
    if not isinstance(parameters, dict):
        parameters = {}
    if not isinstance(context, dict):
        context = {}

    if not any(key in parameters or key in context for key in ("control_mode", "execution_mode")):
        payload["control_mode"] = "whole_body_local"
    if not any(
        key in parameters or key in context
        for key in ("allow_base_motion", "vla_allow_base_motion", "base_motion", "base_motion_mode")
    ):
        payload["allow_base_motion"] = True


def state_snapshot(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_progress": info.get("task_progress"),
        "success": info.get("success"),
        "valid": info.get("valid"),
    }


def capture_navigation_runtime_state(
    *,
    subtask: Any,
    result: Any,
    navigation_runtime_state: dict[str, dict[str, Any]],
    call_env_method: Any,
) -> dict[str, dict[str, Any]]:
    if subtask.agent != AgentName.NAVIGATION:
        return navigation_runtime_state

    artifacts = result.runtime_artifacts
    if not isinstance(artifacts, dict):
        return navigation_runtime_state

    nav_state: dict[str, Any] = {}
    nav_goal = artifacts.get("nav_goal") or artifacts.get("grounded_goal")
    if isinstance(nav_goal, dict) and nav_goal:
        nav_state["nav_goal"] = dict(nav_goal)
    grounded_goal = artifacts.get("grounded_goal")
    if isinstance(grounded_goal, dict) and grounded_goal:
        nav_state["grounded_goal"] = dict(grounded_goal)
    interpreted_goal = artifacts.get("interpreted_goal")
    if isinstance(interpreted_goal, dict) and interpreted_goal:
        nav_state["interpreted_goal"] = dict(interpreted_goal)

    waypoints = artifacts.get("waypoints")
    if not isinstance(waypoints, list):
        path_plan = artifacts.get("path_plan")
        if isinstance(path_plan, dict):
            waypoints = path_plan.get("waypoints")
    if isinstance(waypoints, list):
        nav_state["waypoints"] = list(waypoints)

    for key in (
        "active_waypoint_index",
        "global_waypoint_index",
        "dense_waypoint_index",
        "recovery_mode",
        "recovery_profile",
        "exploration_target",
        "vertical_axis",
        "controller_mode",
        "follow_status",
        "yaw_source",
        "tracking_target",
        "target_waypoint",
        "local_goal",
        "execution_goal",
        "nav2_compute_goal",
        "transition_anchor",
        "selected_object_approach",
        "grounding_candidates",
        "selected_grounding_candidate",
        "object_approach_selection",
        "prepared_navigation_payload",
        "navigation_skill_selection",
        "path_backend",
        "path_tracking_mode",
        "nav2_error",
        "nav2_trav_map_filename",
        "loop_detected",
        "oscillation_detected",
        "steps_since_progress",
        "best_distance_to_waypoint",
        "path_cross_track_error",
        "path_signed_cross_track_error",
        "path_segment_index",
        "path_tangent_heading",
        "goal_reached",
        "localization_guard",
    ):
        if key in artifacts:
            nav_state[key] = artifacts.get(key)

    if not nav_state:
        return navigation_runtime_state

    updated = dict(navigation_runtime_state)
    updated[subtask.runtime_id] = nav_state
    call_env_method(
        "set_navigation_runtime_state",
        subtask_id=subtask.subtask_id,
        navigation_state=dict(nav_state),
    )
    return updated
