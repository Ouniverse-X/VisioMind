from __future__ import annotations

import argparse
import json

from voltron.config_loader import parse_args_with_config
from voltron.shared.context import Plan, Subtask
from voltron.shared.enums import AgentName


def build_parser() -> argparse.ArgumentParser:
    from voltron.entrypoints.examples.closed_loop import parser as entrypoint_parser

    parser = entrypoint_parser.build_closed_loop_parser()
    parser.add_argument("--action-subtask-action", type=str, default=None)
    parser.add_argument("--action-target-object", type=str, default=None)
    parser.add_argument("--action-instruction", type=str, default=None)
    parser.add_argument(
        "--action-sequence",
        type=str,
        default=None,
        help="Optional JSON list of explicit ACTION subtasks; overrides the single action fields.",
    )
    return parser


def _default_action_target_for_env(env_id: str) -> tuple[str, str]:
    task_name = env_id.rsplit("/", 1)[-1]
    if task_name == "turning_on_radio":
        return "turn_on", "radio"
    return "open", "refrigerator"


def _build_action_only_plan(args: argparse.Namespace) -> Plan:
    sequence = getattr(args, "action_sequence", None)
    if isinstance(sequence, str) and sequence.strip():
        sequence = json.loads(sequence)
    if sequence:
        if not isinstance(sequence, list):
            raise ValueError("action_sequence must be a non-empty JSON list")
        subtasks = []
        for index, item in enumerate(sequence, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"action_sequence[{index - 1}] must be an object")
            action = str(item.get("action") or "").strip()
            if not action:
                raise ValueError(f"action_sequence[{index - 1}] is missing action")
            target = item.get("target") or {}
            if isinstance(target, str):
                target = {"object": target}
            if not isinstance(target, dict):
                raise ValueError(f"action_sequence[{index - 1}].target must be an object or string")
            parameters = dict(item.get("parameters") or {})
            parameters.setdefault("instruction", item.get("instruction") or action)
            parameters.setdefault("control_mode", args.action_control_mode)
            parameters.setdefault("allow_base_motion", bool(args.action_allow_base_motion))
            subtasks.append(
                Subtask(
                    subtask_id=f"st_action_{index:02d}",
                    agent=AgentName.ACTION,
                    action=action,
                    target=target,
                    parameters=parameters,
                    context={
                        "task_description": args.task_desc,
                        "execution_mode": args.action_control_mode,
                        "allow_base_motion": bool(args.action_allow_base_motion),
                    },
                )
            )
        return Plan(
            subtasks=subtasks,
            metadata={"planner": "action_sequence_override", "dynamic_execution": False},
        )

    instruction = args.action_instruction or args.task_desc
    default_action, default_target = _default_action_target_for_env(args.env_id)
    action = args.action_subtask_action or default_action
    target_object = args.action_target_object or default_target
    return Plan(
        subtasks=[
            Subtask(
                subtask_id="st_action_01",
                agent=AgentName.ACTION,
                action=action,
                target={"object": target_object},
                parameters={
                    "instruction": instruction,
                    "control_mode": args.action_control_mode,
                    "allow_base_motion": bool(args.action_allow_base_motion),
                },
                context={
                    "task_description": args.task_desc,
                    "execution_mode": args.action_control_mode,
                    "allow_base_motion": bool(args.action_allow_base_motion),
                },
            )
        ],
        metadata={"planner": "action_only_override", "dynamic_execution": False},
    )


def main() -> None:
    from voltron.entrypoints.examples.closed_loop import main as closed_loop_main
    from voltron.runtime.assembly import runtime_builder

    parser = build_parser()
    args = parse_args_with_config(parser)

    hovsg_runtime = closed_loop_main.resolve_hovsg_runtime_config(
        env_id=args.env_id,
        hovsg_scene_map=args.hovsg_scene_map,
        hovsg_graph_root=args.hovsg_graph_root,
        hovsg_scene_id=args.hovsg_scene_id,
        hovsg_graph_path=args.hovsg_graph_path,
        hovsg_nav_graph_type=args.hovsg_nav_graph_type,
    )
    args.pi05_task_id = closed_loop_main._normalize_pi05_task_id(
        policy_backend=args.policy_backend,
        env_id=args.env_id,
        pi05_task_id=args.pi05_task_id,
    )
    args.openpi_comet_task_id = closed_loop_main._normalize_policy_task_id(
        policy_backend=args.policy_backend,
        env_id=args.env_id,
        task_id=args.openpi_comet_task_id,
    )
    orchestrator = closed_loop_main.build_orchestrator(
        embodiment=args.embodiment,
        gr00t_host=args.gr00t_host,
        gr00t_port=args.gr00t_port,
        vision_endpoint=args.vision_endpoint,
        vision_timeout_s=args.vision_timeout_s,
        vision_max_retries=args.vision_max_retries,
        vision_retry_backoff_s=args.vision_retry_backoff_s,
        vision_heartbeat_interval_steps=args.vision_heartbeat_interval_steps,
        memory_agent_endpoint=args.memory_agent_endpoint,
        use_memory_agent=args.memory_mode == "agent",
        memory_agent_enabled=args.memory_agent_enabled,
        memory_llm_backend=args.memory_llm_backend,
        memory_llm_base_url=args.memory_llm_base_url,
        memory_llm_model=args.memory_llm_model,
        memory_llm_api_key=args.memory_llm_api_key,
        memory_llm_api_key_env=args.memory_llm_api_key_env,
        memory_llm_timeout_s=args.memory_llm_timeout_s,
        memory_llm_temperature=args.memory_llm_temperature,
        memory_llm_max_retries=args.memory_llm_max_retries,
        memory_llm_retry_backoff_s=args.memory_llm_retry_backoff_s,
        memory_experience_extraction_enabled=args.memory_experience_extraction_enabled,
        memory_experience_extraction_min_confidence_to_write=(
            args.memory_experience_extraction_min_confidence_to_write
        ),
        memory_experience_extraction_min_confidence_to_promote=(
            args.memory_experience_extraction_min_confidence_to_promote
        ),
        max_retries=args.max_retries,
        max_control_steps_per_subtask=args.max_control_steps,
        action_verify_every_control_steps=args.action_verify_every_control_steps,
        action_max_unverified_internal_step_control_steps=(
            args.action_max_unverified_internal_step_control_steps
        ),
        action_internal_planning_enabled=args.action_internal_planning_enabled,
        action_internal_step_completion_use_vision_completion_monitor=(
            args.action_internal_step_completion_use_vision_completion_monitor
        ),
        action_internal_step_completion_require_verified_completion=(
            args.action_internal_step_completion_require_verified_completion
        ),
        planner_backend=args.brain_planner,
        brain_base_url=args.brain_base_url,
        brain_model=args.brain_model,
        brain_api_key=args.brain_api_key,
        brain_api_key_env=args.brain_api_key_env,
        brain_timeout_s=args.brain_timeout_s,
        brain_temperature=args.brain_temperature,
        brain_max_retries=args.brain_max_retries,
        brain_retry_backoff_s=args.brain_retry_backoff_s,
        action_selector=args.action_selector,
        action_base_url=args.action_base_url,
        action_model=args.action_model,
        action_api_key=args.action_api_key,
        action_api_key_env=args.action_api_key_env,
        action_timeout_s=args.action_timeout_s,
        action_temperature=args.action_temperature,
        action_max_retries=args.action_max_retries,
        action_retry_backoff_s=args.action_retry_backoff_s,
        navigation_backend=args.navigation_backend,
        navigation_base_url=args.navigation_base_url,
        navigation_model=args.navigation_model,
        navigation_api_key=args.navigation_api_key,
        navigation_api_key_env=args.navigation_api_key_env,
        navigation_timeout_s=args.navigation_timeout_s,
        navigation_temperature=args.navigation_temperature,
        navigation_max_retries=args.navigation_max_retries,
        navigation_retry_backoff_s=args.navigation_retry_backoff_s,
        hovsg_graph_root=hovsg_runtime["graph_root"],
        hovsg_scene_id=hovsg_runtime["scene_id"],
        hovsg_graph_path=hovsg_runtime["graph_path"],
        hovsg_nav_graph_type=hovsg_runtime["nav_graph_type"],
        hovsg_direct_room_transition_max_gap_m=args.hovsg_direct_room_transition_max_gap_m,
        hovsg_direct_room_transition_min_span_m=args.hovsg_direct_room_transition_min_span_m,
        hovsg_object_approach_min_portal_stance_clearance_m=args.hovsg_object_approach_min_portal_stance_clearance_m,
        nav2_version_profile=args.nav2_version_profile,
        nav2_action_name=args.nav2_action_name,
        nav2_planner_id=args.nav2_planner_id,
        nav2_frame_id=args.nav2_frame_id,
        nav2_timeout_s=args.nav2_timeout_s,
        nav2_strict=args.nav2_strict,
        nav2_trav_map_filename=args.nav2_trav_map_filename,
        nav2_portal_analysis_map_resolution=args.nav2_portal_analysis_map_resolution,
        nav2_portal_clearance_radius_m=args.nav2_portal_clearance_radius_m,
        nav2_portal_corridor_standoff_m=args.nav2_portal_corridor_standoff_m,
        nav2_portal_sampling_step_m=args.nav2_portal_sampling_step_m,
        nav2_local_path_clearance_radius_m=args.nav2_local_path_clearance_radius_m,
        nav2_local_path_waypoint_spacing_m=args.nav2_local_path_waypoint_spacing_m,
        navigation_prefer_forward_facing_motion=args.navigation_prefer_forward_facing_motion,
        navigation_portal_alignment_distance_threshold=args.navigation_portal_alignment_distance_threshold,
        navigation_portal_prealign_distance_threshold_m=args.navigation_portal_prealign_distance_threshold_m,
        navigation_portal_alignment_footprint_width_m=args.navigation_portal_alignment_footprint_width_m,
        navigation_portal_alignment_min_lateral_deadband_m=args.navigation_portal_alignment_min_lateral_deadband_m,
        navigation_portal_alignment_wide_clearance_margin_m=args.navigation_portal_alignment_wide_clearance_margin_m,
        navigation_max_linear_velocity=args.navigation_max_linear_velocity,
        navigation_linear_gain=args.navigation_linear_gain,
        navigation_local_path_linear_gain=args.navigation_local_path_linear_gain,
        navigation_local_path_max_linear_velocity=args.navigation_local_path_max_linear_velocity,
        navigation_portal_alignment_max_linear_velocity=args.navigation_portal_alignment_max_linear_velocity,
        navigation_object_approach_final_waypoint_tolerance_m=(
            args.navigation_object_approach_final_waypoint_tolerance_m
        ),
        navigation_max_angular_velocity=args.navigation_max_angular_velocity,
        navigation_local_path_angular_gain_scale=args.navigation_local_path_angular_gain_scale,
        policy_backend=args.policy_backend,
        pi05_endpoint=args.pi05_endpoint,
        pi05_timeout_s=args.pi05_timeout_s,
        pi05_task_id=args.pi05_task_id,
        openpi_comet_endpoint=args.openpi_comet_endpoint,
        openpi_comet_timeout_s=args.openpi_comet_timeout_s,
        openpi_comet_task_name=args.openpi_comet_task_name,
        openpi_comet_task_id=args.openpi_comet_task_id,
        openpi_comet_prompt=args.openpi_comet_prompt,
        openpi_comet_action_mode=args.openpi_comet_action_mode,
        log_navigation_candidates=args.log_navigation_candidates,
        logging_nav2_path_snapshots=args.logging_nav2_path_snapshots,
        runtime_termination_use_environment_success_signal=(
            args.runtime_termination_use_environment_success_signal
        ),
        runtime_termination_use_brain_completion_signal=(
            args.runtime_termination_use_brain_completion_signal
        ),
        runtime_termination_environment_signal_policy=(
            args.runtime_termination_environment_signal_policy
        ),
        vision_completion_enabled=args.vision_completion_enabled,
        vision_completion_positive_streak=args.vision_completion_positive_streak,
        vision_completion_stability_steps=args.vision_completion_stability_steps,
        vision_completion_min_confidence=args.vision_completion_min_confidence,
        vision_completion_action_delta_threshold=(args.vision_completion_action_delta_threshold),
        vision_completion_check_interval_steps=(args.vision_completion_check_interval_steps),
        vision_completion_agent_scope=args.vision_completion_agent_scope,
        vision_completion_use_memory_guidance=(args.vision_completion_use_memory_guidance),
        vision_completion_include_third_person=(args.vision_completion_include_third_person),
        vision_completion_max_images=args.vision_completion_max_images,
        vision_completion_max_image_side_px=(args.vision_completion_max_image_side_px),
        vision_completion_jpeg_quality=args.vision_completion_jpeg_quality,
        vision_completion_max_image_b64_chars=(args.vision_completion_max_image_b64_chars),
        vision_completion_image_detail=args.vision_completion_image_detail,
        logging_verbose=args.logging_verbose,
        anygrasp_config=getattr(args, "anygrasp_config", None),
    )

    environment = runtime_builder.build_behavior_environment(args=args, hovsg_runtime=hovsg_runtime)
    orchestrator.event_sink = (
        lambda event: closed_loop_main._record_orchestrator_event_to_environment(environment, event)
    )
    request = runtime_builder.build_task_request(
        args=args, scene_id=hovsg_runtime["scene_id"], hovsg_runtime=hovsg_runtime
    )
    result = orchestrator.run_task(
        request=request, environment=environment, plan_override=_build_action_only_plan(args)
    )
    print(result)


if __name__ == "__main__":
    main()
