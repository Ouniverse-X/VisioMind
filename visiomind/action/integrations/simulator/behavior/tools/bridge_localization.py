from __future__ import annotations

from typing import Any

import numpy as np

from visiomind.action.integrations.simulator.behavior.observation import (
    frames as behavior_observation_frames,
)
from visiomind.action.integrations.simulator.behavior.observation import robot_state as behavior_robot_state
from visiomind.action.integrations.simulator.behavior.tools import (
    navigation_success as behavior_navigation_success,
)
from visiomind.action.integrations.simulator.behavior.tools import (
    runtime_actions as behavior_runtime_actions,
)
from visiomind.action.integrations.simulator.behavior.tools import runtime_inputs as behavior_runtime_inputs
from visiomind.action.integrations.simulator.behavior.tools import (
    runtime_localization as behavior_runtime_localization,
)
from visiomind.action.shared.context import Subtask
from visiomind.action.shared.enums import TaskType
from visiomind.action.shared.results import AgentResult


def build_vln_runtime_inputs(
    *,
    subtask: Subtask,
    observation: dict[str, Any],
    env_kwargs: dict[str, Any],
    last_info: dict[str, Any],
    last_obs: dict[str, Any],
    navigation_runtime_state: dict[str, dict[str, Any]],
    scene_id: str | None,
    scene_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    robot_state = behavior_robot_state.extract_runtime_robot_state(
        last_info=last_info,
        last_obs=last_obs,
        frame_config=env_kwargs,
    )
    return behavior_runtime_inputs.build_vln_runtime_inputs(
        subtask=subtask,
        observation=observation,
        env_kwargs=env_kwargs,
        scene_id=scene_id,
        pose=robot_state["pose"],
        orientation=robot_state["orientation"],
        simulator_pose=robot_state["simulator_pose"],
        simulator_orientation=robot_state["simulator_orientation"],
        region=extract_runtime_region(last_info=last_info, last_obs=last_obs),
        nav_feedback=extract_runtime_nav_feedback(last_info=last_info),
        rgb=extract_modal_frames(last_obs, prefix="video.observation.images.rgb"),
        depth=extract_modal_frames(last_obs, prefix="video.observation.images.depth"),
        navigation_runtime_state=navigation_runtime_state,
        scene_state=scene_state,
    )


def extract_scene_id(
    *, last_info: dict[str, Any], last_obs: dict[str, Any], scene_id: str | None
) -> str | None:
    return behavior_runtime_localization.extract_scene_id(
        last_info=last_info,
        last_obs=last_obs,
        scene_id=scene_id,
    )


def extract_runtime_kwarg(runtime_kwargs: dict[str, Any], key: str) -> str | None:
    return behavior_runtime_localization.extract_runtime_kwarg(runtime_kwargs, key)


def normalize_runtime_str(value: Any) -> str | None:
    return behavior_runtime_localization.normalize_runtime_str(value)


def build_hovsg_localizer(
    *,
    existing_localizer: Any | None,
    last_info: dict[str, Any],
    last_obs: dict[str, Any],
    scene_id: str | None,
    hovsg_graph_path: str | None,
    hovsg_graph_root: str | None,
    hovsg_nav_graph_type: str | None,
) -> Any | None:
    return behavior_runtime_localization.build_hovsg_localizer(
        existing_localizer=existing_localizer,
        last_info=last_info,
        last_obs=last_obs,
        scene_id=scene_id,
        hovsg_graph_path=hovsg_graph_path,
        hovsg_graph_root=hovsg_graph_root,
        hovsg_nav_graph_type=hovsg_nav_graph_type,
    )


def localize_runtime_state_snapshot(
    *,
    existing_localizer: Any | None,
    last_info: dict[str, Any],
    last_obs: dict[str, Any],
    scene_id: str | None,
    hovsg_graph_path: str | None,
    hovsg_graph_root: str | None,
    hovsg_nav_graph_type: str | None,
    resolved_metadata: dict[str, str | None],
    frame_config: dict[str, Any] | None = None,
) -> tuple[Any | None, dict[str, Any]]:
    return behavior_runtime_localization.localize_runtime_state_snapshot(
        existing_localizer=existing_localizer,
        last_info=last_info,
        last_obs=last_obs,
        scene_id=scene_id,
        hovsg_graph_path=hovsg_graph_path,
        hovsg_graph_root=hovsg_graph_root,
        hovsg_nav_graph_type=hovsg_nav_graph_type,
        resolved_metadata=resolved_metadata,
        frame_config=frame_config,
    )


def apply_navigation_success_override(
    *,
    subtask: Subtask,
    last_info: dict[str, Any],
    task_success: bool,
    nav_state: dict[str, Any],
    task_type: TaskType | None,
    localizer: Any | None,
    last_obs: dict[str, Any],
    scene_id: str | None,
    object_goal_distance_tolerance_m: float,
    object_goal_heading_tolerance_rad: float,
    frame_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    robot_state = behavior_robot_state.extract_runtime_robot_state(
        last_info=last_info,
        last_obs=last_obs,
        frame_config=frame_config,
    )
    return behavior_navigation_success.apply_navigation_success_override(
        agent=subtask.agent,
        last_info=last_info,
        task_success=task_success,
        nav_state=nav_state,
        target=subtask.target if isinstance(subtask.target, dict) else {},
        task_type=task_type,
        localizer=localizer,
        pose=robot_state["pose"],
        orientation=robot_state["orientation"],
        scene_id=scene_id,
        object_goal_distance_tolerance_m=object_goal_distance_tolerance_m,
        object_goal_heading_tolerance_rad=object_goal_heading_tolerance_rad,
    )


def format_behavior_action(
    *,
    action: dict[str, Any],
    action_spaces: dict[str, Any],
    reference_observation: dict[str, Any] | None = None,
    hold_grippers_closed: bool = False,
) -> dict[str, Any]:
    return behavior_runtime_actions.format_behavior_action(
        action=action,
        action_spaces=action_spaces,
        reference_observation=reference_observation,
        hold_grippers_closed=hold_grippers_closed,
    )


def normalize_action_dict(action: dict[str, Any]) -> dict[str, Any]:
    return behavior_runtime_actions.normalize_action_dict(action)


def extract_action(result: AgentResult) -> dict[str, Any] | None:
    return behavior_runtime_actions.extract_action(result.runtime_artifacts)


def to_numpy(value: Any) -> np.ndarray | None:
    return behavior_runtime_actions.to_numpy(value)


def select_first_action_step(arr: np.ndarray, *, expected_shape: tuple[int, ...]) -> np.ndarray:
    return behavior_runtime_actions.select_first_action_step(arr, expected_shape=expected_shape)


def build_policy_observation_source(
    *,
    last_obs: dict[str, Any],
    root_task_instruction: str | None,
    language_instruction: str | None = None,
) -> dict[str, Any]:
    return behavior_runtime_actions.build_policy_observation_source(
        last_obs=last_obs,
        root_task_instruction=root_task_instruction,
        language_instruction=language_instruction,
    )


def normalize_label(value: Any) -> str | None:
    return behavior_robot_state.normalize_label(value)


def extract_runtime_pose(
    *,
    last_info: dict[str, Any],
    last_obs: dict[str, Any],
    frame_config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    return behavior_robot_state.extract_runtime_pose(
        last_info=last_info,
        last_obs=last_obs,
        frame_config=frame_config,
    )


def extract_runtime_orientation(
    *,
    last_info: dict[str, Any],
    last_obs: dict[str, Any],
    frame_config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    return behavior_robot_state.extract_runtime_orientation(
        last_info=last_info,
        last_obs=last_obs,
        frame_config=frame_config,
    )


def extract_runtime_region(*, last_info: dict[str, Any], last_obs: dict[str, Any]) -> str | None:
    return behavior_robot_state.extract_runtime_region(last_info=last_info, last_obs=last_obs)


def extract_runtime_nav_feedback(*, last_info: dict[str, Any]) -> dict[str, Any]:
    return behavior_robot_state.extract_runtime_nav_feedback(last_info=last_info)


def extract_modal_frames(last_obs: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    return behavior_observation_frames.extract_modal_frames(last_obs, prefix=prefix)


def array_to_pose(value: Any) -> dict[str, Any] | None:
    return behavior_robot_state.array_to_pose(value)


def array_to_yaw(value: Any) -> float | None:
    return behavior_robot_state.array_to_yaw(value)


def orientation_to_yaw(value: Any) -> float | None:
    return behavior_robot_state.orientation_to_yaw(value)


def quat_to_yaw(x_coord: float, y_coord: float, z_coord: float, w_coord: float) -> float:
    return behavior_robot_state.quat_to_yaw(x_coord, y_coord, z_coord, w_coord)


def wrap_angle(value: float) -> float:
    return behavior_robot_state.wrap_angle(value)


def planar_axes(vertical_axis: Any) -> tuple[str, str]:
    return behavior_robot_state.planar_axes(vertical_axis)


def to_float(value: Any) -> float | None:
    return behavior_robot_state.to_float(value)


def to_policy_observation(obs: dict[str, Any]) -> dict[str, Any]:
    return behavior_observation_frames.to_policy_observation(obs)


def extract_images_b64(obs: dict[str, Any]) -> tuple[list[str], list[str]]:
    return behavior_observation_frames.extract_images_b64(obs)


def image_view_name_from_obs_key(key: str) -> str:
    return behavior_observation_frames.image_view_name_from_obs_key(key)


def encode_image_b64(image: Any) -> str | None:
    return behavior_observation_frames.encode_image_b64(image)
