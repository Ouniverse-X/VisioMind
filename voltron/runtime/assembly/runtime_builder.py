"""Runtime object builders for closed-loop entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from voltron.runtime.interaction.task_request import build_task_request
from voltron.integrations.simulator.behavior.runtime_bridge import (
    BehaviorRuntimeEnvironment,
)


def build_behavior_env_kwargs(
    *, args: Any, hovsg_runtime: dict[str, Any]
) -> dict[str, Any]:
    scene_id = hovsg_runtime.get("scene_id")
    env_kwargs: dict[str, Any] = {
        "env_idx": args.env_idx,
        "total_n_envs": args.total_n_envs,
        "vlm": bool(getattr(args, "behavior_builtin_vlm_detector_enabled", False)),
    }
    if args.behavior_scene_file:
        env_kwargs["scene_file"] = str(Path(args.behavior_scene_file).expanduser())
    if args.behavior_tro_state_file:
        env_kwargs["tro_state_file"] = str(
            Path(args.behavior_tro_state_file).expanduser()
        )
    if args.behavior_task_instance_id is not None:
        env_kwargs["task_instance_id"] = int(args.behavior_task_instance_id)
    if getattr(args, "behavior_scene_state_include_aabb", False):
        env_kwargs["scene_state_include_aabb"] = True
    navigation_role_overrides = getattr(
        args,
        "behavior_scene_state_navigation_role_overrides",
        None,
    )
    if isinstance(navigation_role_overrides, dict) and navigation_role_overrides:
        env_kwargs["scene_state_navigation_role_overrides"] = dict(
            navigation_role_overrides
        )
    if args.behavior_robot_start_position is not None:
        env_kwargs["robot_start_position"] = [
            float(value) for value in args.behavior_robot_start_position
        ]
    if args.behavior_robot_start_orientation is not None:
        env_kwargs["robot_start_orientation"] = [
            float(value) for value in args.behavior_robot_start_orientation
        ]
    if getattr(args, "behavior_post_reset_robot_position", None) is not None:
        env_kwargs["post_reset_robot_position"] = [
            float(value) for value in args.behavior_post_reset_robot_position
        ]
    if getattr(args, "behavior_post_reset_robot_orientation", None) is not None:
        env_kwargs["post_reset_robot_orientation"] = [
            float(value) for value in args.behavior_post_reset_robot_orientation
        ]
    if getattr(args, "behavior_post_reset_object_states", None) is not None:
        env_kwargs["post_reset_object_states"] = dict(
            args.behavior_post_reset_object_states
        )
    if getattr(args, "behavior_post_reset_robot_joint_positions", None) is not None:
        env_kwargs["post_reset_robot_joint_positions"] = [
            float(value)
            for value in args.behavior_post_reset_robot_joint_positions
        ]
    if getattr(args, "behavior_post_reset_robot_joint_velocities", None) is not None:
        env_kwargs["post_reset_robot_joint_velocities"] = [
            float(value)
            for value in args.behavior_post_reset_robot_joint_velocities
        ]
    env_kwargs["post_reset_refresh_observation"] = bool(
        getattr(args, "behavior_post_reset_refresh_observation", True)
    )
    if getattr(args, "behavior_post_reset_settle_steps", None) is not None:
        env_kwargs["post_reset_settle_steps"] = max(
            0, int(args.behavior_post_reset_settle_steps)
        )
    for arg_name, env_name in (
        ("behavior_recording_third_person_local_offset", "recording_third_person_local_offset"),
        ("behavior_recording_third_person_look_at_offset", "recording_third_person_look_at_offset"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            env_kwargs[env_name] = [float(item) for item in value]
    env_kwargs["recording_third_person_prefer_live_capture"] = bool(
        getattr(args, "behavior_recording_third_person_prefer_live_capture", False)
    )
    industrial_overlay = getattr(
        args,
        "behavior_industrial_visual_overlay",
        None,
    )
    if isinstance(industrial_overlay, dict):
        env_kwargs["industrial_visual_overlay"] = dict(industrial_overlay)
    if scene_id:
        env_kwargs["scene_id"] = scene_id
    if args.nav2_trav_map_filename:
        env_kwargs["nav2_trav_map_filename"] = args.nav2_trav_map_filename
    portal_annotations = getattr(args, "navigation_portal_annotations", None)
    if portal_annotations:
        env_kwargs["portal_annotations"] = portal_annotations
    if hovsg_runtime.get("graph_root"):
        env_kwargs["hovsg_graph_root"] = hovsg_runtime["graph_root"]
    if hovsg_runtime.get("graph_path"):
        env_kwargs["hovsg_graph_path"] = hovsg_runtime["graph_path"]
    if hovsg_runtime.get("nav_graph_type"):
        env_kwargs["hovsg_nav_graph_type"] = hovsg_runtime["nav_graph_type"]
    if hovsg_runtime.get("scene_vertical_axis"):
        env_kwargs["scene_vertical_axis"] = hovsg_runtime["scene_vertical_axis"]
    if hovsg_runtime.get("simulator_vertical_axis"):
        env_kwargs["simulator_vertical_axis"] = hovsg_runtime[
            "simulator_vertical_axis"
        ]
    if hovsg_runtime.get("scene_from_simulator_transform"):
        env_kwargs["scene_from_simulator_transform"] = hovsg_runtime[
            "scene_from_simulator_transform"
        ]
    policy_backend = str(getattr(args, "policy_backend", "")).strip().lower()
    anygrasp_action_only = bool(getattr(args, "anygrasp_config", None)) and bool(
        getattr(args, "action_subtask_action", None)
    )
    if policy_backend in {"pi05", "openpi_comet"}:
        env_kwargs["use_low_res_rgb"] = anygrasp_action_only
        env_kwargs["use_openpi_rgb"] = not anygrasp_action_only
    return env_kwargs


def build_behavior_environment(
    *, args: Any, hovsg_runtime: dict[str, Any]
) -> BehaviorRuntimeEnvironment:
    environment = BehaviorRuntimeEnvironment(
        env_id=args.env_id,
        env_kwargs=build_behavior_env_kwargs(args=args, hovsg_runtime=hovsg_runtime),
        auto_register=not args.no_auto_register,
        default_subtask_max_steps=args.max_control_steps,
        progress_log_every=args.progress_log_every,
        recording_video_scale=getattr(args, "recording_video_scale", 1.0),
        logging_verbose=getattr(args, "logging_verbose", True),
        logging_memory_diagnostics=getattr(args, "logging_memory_diagnostics", False),
        runtime_termination_use_environment_success_signal=getattr(
            args,
            "runtime_termination_use_environment_success_signal",
            True,
        ),
        runtime_termination_environment_signal_policy=getattr(
            args,
            "runtime_termination_environment_signal_policy",
            "allow_early_success",
        ),
    )
    environment._policy_backend = str(args.policy_backend)
    return environment


__all__ = [
    "build_behavior_env_kwargs",
    "build_behavior_environment",
    "build_task_request",
]
