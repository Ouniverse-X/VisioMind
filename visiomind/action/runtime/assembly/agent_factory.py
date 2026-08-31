from __future__ import annotations

from visiomind.action.agents import ActionAgent, BrainAgent, MemoryAgent, NavigationAgent, VisionAgent
from visiomind.action.agents.action.body.step_verification import VisionBackedActionStepVerifier
from visiomind.action.agents.action.skills import DefaultActionTaskPlanningSkill
from visiomind.action.agents.vision.body import VLMCompletionEvaluator
from visiomind.action.agents.action.tools.action_projection import ActionProjection
from visiomind.action.integrations.manipulation import (
    Gr00tPolicyAdapter,
    OpenPICometPolicyAdapter,
    Pi05PolicyAdapter,
)
from visiomind.action.integrations.memory.hems.backend import HEMSAdapter
from visiomind.action.integrations.memory.service import MemoryAgentClient
from visiomind.action.integrations.vlm.service.client import VLMHttpAdapter
from visiomind.action.runtime.assembly import backend_factory
from visiomind.action.runtime.assembly.capabilities import inject_agent_capabilities
from visiomind.action.runtime.orchestrator.closed_loop import ClosedLoopOrchestrator


def build_closed_loop_orchestrator(
    embodiment: str,
    gr00t_host: str,
    gr00t_port: int,
    vision_endpoint: str,
    vision_timeout_s: float,
    vision_max_retries: int,
    vision_retry_backoff_s: float,
    vision_heartbeat_interval_steps: int,
    memory_agent_endpoint: str,
    use_memory_agent: bool,
    max_retries: int,
    max_control_steps_per_subtask: int,
    planner_backend: str,
    brain_base_url: str | None,
    brain_model: str | None,
    brain_api_key: str | None,
    brain_api_key_env: str,
    brain_timeout_s: float,
    brain_temperature: float,
    brain_max_retries: int,
    brain_retry_backoff_s: float,
    action_selector: str,
    action_base_url: str | None,
    action_model: str | None,
    action_api_key: str | None,
    action_api_key_env: str,
    action_timeout_s: float,
    action_temperature: float,
    action_max_retries: int,
    action_retry_backoff_s: float,
    navigation_backend: str,
    hovsg_graph_root: str | None,
    hovsg_scene_id: str | None,
    hovsg_graph_path: str | None,
    hovsg_nav_graph_type: str | None,
    hovsg_direct_room_transition_max_gap_m: float,
    hovsg_direct_room_transition_min_span_m: float,
    hovsg_object_approach_min_portal_stance_clearance_m: float,
    nav2_version_profile: str,
    nav2_action_name: str,
    nav2_planner_id: str | None,
    nav2_frame_id: str,
    nav2_timeout_s: float,
    nav2_strict: bool,
    nav2_trav_map_filename: str | None,
    nav2_portal_analysis_map_resolution: float,
    nav2_portal_clearance_radius_m: float,
    nav2_portal_corridor_standoff_m: float,
    nav2_portal_sampling_step_m: float,
    nav2_local_path_clearance_radius_m: float,
    nav2_local_path_waypoint_spacing_m: float,
    navigation_prefer_forward_facing_motion: bool,
    navigation_portal_alignment_distance_threshold: float,
    navigation_portal_prealign_distance_threshold_m: float,
    navigation_portal_alignment_footprint_width_m: float,
    navigation_portal_alignment_min_lateral_deadband_m: float,
    navigation_portal_alignment_wide_clearance_margin_m: float,
    navigation_max_linear_velocity: float,
    navigation_linear_gain: float,
    navigation_local_path_linear_gain: float,
    navigation_local_path_max_linear_velocity: float,
    navigation_portal_alignment_max_linear_velocity: float,
    navigation_object_approach_final_waypoint_tolerance_m: float = 0.45,
    action_verify_every_control_steps: int = 400,
    action_max_unverified_internal_step_control_steps: int = 5000,
    action_internal_planning_enabled: bool = True,
    action_internal_step_completion_use_vision_completion_monitor: bool = True,
    action_internal_step_completion_require_verified_completion: bool = True,
    navigation_max_angular_velocity: float = 0.8,
    navigation_local_path_angular_gain_scale: float = 0.7,
    policy_backend: str = "groot",
    pi05_endpoint: str = "ws://127.0.0.1:9000",
    pi05_timeout_s: float = 15.0,
    pi05_task_id: int | None = None,
    openpi_comet_endpoint: str = "ws://127.0.0.1:9000",
    openpi_comet_timeout_s: float = 60.0,
    openpi_comet_task_name: str | None = None,
    openpi_comet_task_id: int | None = None,
    openpi_comet_prompt: str | None = None,
    openpi_comet_action_mode: str = "raw",
    log_navigation_candidates: bool = False,
    log_nav2_path_snapshots: bool = False,
    navigation_base_url: str | None = None,
    navigation_model: str | None = None,
    navigation_api_key: str | None = None,
    navigation_api_key_env: str = "OPENAI_API_KEY",
    navigation_timeout_s: float = 30.0,
    navigation_temperature: float = 0.1,
    navigation_max_retries: int = 0,
    navigation_retry_backoff_s: float = 1.0,
    memory_agent_enabled: bool = True,
    memory_llm_backend: str | None = None,
    memory_llm_base_url: str | None = None,
    memory_llm_model: str | None = None,
    memory_llm_api_key: str | None = None,
    memory_llm_api_key_env: str = "OPENAI_API_KEY",
    memory_llm_timeout_s: float = 30.0,
    memory_llm_temperature: float = 0.0,
    memory_llm_max_retries: int = 0,
    memory_llm_retry_backoff_s: float = 1.0,
    memory_experience_extraction_enabled: bool = False,
    memory_experience_extraction_min_confidence_to_write: float = 0.4,
    memory_experience_extraction_min_confidence_to_promote: float = 0.7,
    runtime_termination_use_environment_success_signal: bool = True,
    runtime_termination_use_brain_completion_signal: bool = True,
    runtime_termination_environment_signal_policy: str = "allow_early_success",
    vision_completion_enabled: bool = True,
    vision_completion_positive_streak: int = 1,
    vision_completion_stability_steps: int = 1,
    vision_completion_min_confidence: float = 0.75,
    vision_completion_action_delta_threshold: float = 0.03,
    vision_completion_check_interval_steps: int = 200,
    vision_completion_agent_scope: list[str] | tuple[str, ...] | set[str] | None = None,
    vision_completion_use_memory_guidance: bool = True,
    vision_completion_include_third_person: bool = True,
    vision_completion_max_images: int = 4,
    vision_completion_max_image_side_px: int = 1024,
    vision_completion_jpeg_quality: int = 90,
    vision_completion_max_image_b64_chars: int = 900_000,
    vision_completion_image_detail: str = "high",
    logging_verbose: bool = True,
    anygrasp_config: dict | None = None,
) -> ClosedLoopOrchestrator:
    policy_backend = str(policy_backend).strip().lower()
    if use_memory_agent:
        memory = MemoryAgentClient(
            endpoint=memory_agent_endpoint,
            timeout_s=max(15.0, float(memory_llm_timeout_s)),
        )
    else:
        memory_backend = HEMSAdapter()
        if memory_agent_enabled:
            memory_extractor = backend_factory.build_memory_extractor(
                enabled=memory_experience_extraction_enabled,
                backend=memory_llm_backend,
                base_url=memory_llm_base_url,
                model=memory_llm_model,
                api_key=memory_llm_api_key,
                api_key_env=memory_llm_api_key_env,
                timeout_s=memory_llm_timeout_s,
                temperature=memory_llm_temperature,
                max_retries=memory_llm_max_retries,
                retry_backoff_s=memory_llm_retry_backoff_s,
            )
            memory = MemoryAgent(
                backend=memory_backend,
                extractor=memory_extractor,
                experience_extraction_enabled=memory_experience_extraction_enabled,
                min_confidence_to_write=memory_experience_extraction_min_confidence_to_write,
                min_confidence_to_promote=memory_experience_extraction_min_confidence_to_promote,
            )
        else:
            memory = memory_backend
    if policy_backend == "pi05":
        manipulation_policy = Pi05PolicyAdapter(
            endpoint=pi05_endpoint,
            timeout_s=pi05_timeout_s,
            task_id=pi05_task_id,
            chunk_size=1,
        )
    elif policy_backend == "openpi_comet":
        manipulation_policy = OpenPICometPolicyAdapter(
            endpoint=openpi_comet_endpoint,
            timeout_s=openpi_comet_timeout_s,
            task_name=openpi_comet_task_name,
            task_id=openpi_comet_task_id,
            prompt=openpi_comet_prompt,
            action_mode=openpi_comet_action_mode,
            request_diagnostics_enabled=bool(logging_verbose),
        )
    elif policy_backend == "groot":
        manipulation_policy = Gr00tPolicyAdapter(host=gr00t_host, port=gr00t_port, strict=False)
    else:
        raise ValueError(f"Unsupported VLA policy backend '{policy_backend}'.")
    navigation_policy = backend_factory.build_vln_policy(
        backend=navigation_backend,
        gr00t_host=gr00t_host,
        gr00t_port=gr00t_port,
        prefer_forward_facing_motion=navigation_prefer_forward_facing_motion,
        portal_alignment_distance_threshold=navigation_portal_alignment_distance_threshold,
        portal_prealign_distance_threshold_m=navigation_portal_prealign_distance_threshold_m,
        portal_alignment_footprint_width_m=navigation_portal_alignment_footprint_width_m,
        portal_alignment_min_lateral_deadband_m=navigation_portal_alignment_min_lateral_deadband_m,
        portal_alignment_wide_clearance_margin_m=navigation_portal_alignment_wide_clearance_margin_m,
        max_linear_velocity=navigation_max_linear_velocity,
        linear_gain=navigation_linear_gain,
        local_path_linear_gain=navigation_local_path_linear_gain,
        local_path_max_linear_velocity=navigation_local_path_max_linear_velocity,
        portal_alignment_max_linear_velocity=navigation_portal_alignment_max_linear_velocity,
        object_approach_final_waypoint_tolerance_m=navigation_object_approach_final_waypoint_tolerance_m,
        max_angular_velocity=navigation_max_angular_velocity,
        local_path_angular_gain_scale=navigation_local_path_angular_gain_scale,
    )
    vision = VLMHttpAdapter(
        endpoint=vision_endpoint,
        timeout_s=vision_timeout_s,
        max_retries=vision_max_retries,
        retry_backoff_s=vision_retry_backoff_s,
    )
    projector = ActionProjection.from_embodiment(embodiment)
    navigator = backend_factory.build_vln_navigator(
        backend=navigation_backend,
        hovsg_graph_root=hovsg_graph_root,
        hovsg_scene_id=hovsg_scene_id,
        hovsg_graph_path=hovsg_graph_path,
        hovsg_nav_graph_type=hovsg_nav_graph_type,
        hovsg_direct_room_transition_max_gap_m=hovsg_direct_room_transition_max_gap_m,
        hovsg_direct_room_transition_min_span_m=hovsg_direct_room_transition_min_span_m,
        hovsg_object_approach_min_portal_stance_clearance_m=hovsg_object_approach_min_portal_stance_clearance_m,
        nav2_version_profile=nav2_version_profile,
        nav2_action_name=nav2_action_name,
        nav2_planner_id=nav2_planner_id,
        nav2_frame_id=nav2_frame_id,
        nav2_timeout_s=nav2_timeout_s,
        nav2_strict=nav2_strict,
        nav2_trav_map_filename=nav2_trav_map_filename,
        nav2_portal_analysis_map_resolution=nav2_portal_analysis_map_resolution,
        nav2_portal_clearance_radius_m=nav2_portal_clearance_radius_m,
        nav2_portal_corridor_standoff_m=nav2_portal_corridor_standoff_m,
        nav2_portal_sampling_step_m=nav2_portal_sampling_step_m,
        nav2_local_path_clearance_radius_m=nav2_local_path_clearance_radius_m,
        nav2_local_path_waypoint_spacing_m=nav2_local_path_waypoint_spacing_m,
    )
    planner = backend_factory.build_planner(
        planner_backend=planner_backend,
        brain_base_url=brain_base_url,
        brain_model=brain_model,
        brain_api_key=brain_api_key,
        brain_api_key_env=brain_api_key_env,
        brain_timeout_s=brain_timeout_s,
        brain_temperature=brain_temperature,
        brain_max_retries=brain_max_retries,
        brain_retry_backoff_s=brain_retry_backoff_s,
    )
    action_selector_mode = action_selector

    selector = backend_factory.build_vla_selector(
        selector_mode=action_selector_mode,
        action_base_url=action_base_url,
        action_model=action_model,
        action_api_key=action_api_key,
        action_api_key_env=action_api_key_env,
        action_timeout_s=action_timeout_s,
        action_temperature=action_temperature,
        action_max_retries=action_max_retries,
        action_retry_backoff_s=action_retry_backoff_s,
    )
    deliberator = backend_factory.build_vla_deliberator(
        selector_mode=action_selector_mode,
        action_base_url=action_base_url,
        action_model=action_model,
        action_api_key=action_api_key,
        action_api_key_env=action_api_key_env,
        action_timeout_s=action_timeout_s,
        action_temperature=action_temperature,
        action_max_retries=action_max_retries,
        action_retry_backoff_s=action_retry_backoff_s,
    )
    task_planner = (
        backend_factory.build_action_task_planner(
            selector_mode=action_selector_mode,
            action_base_url=action_base_url,
            action_model=action_model,
            action_api_key=action_api_key,
            action_api_key_env=action_api_key_env,
            action_timeout_s=action_timeout_s,
            action_temperature=action_temperature,
            action_max_retries=action_max_retries,
            action_retry_backoff_s=action_retry_backoff_s,
        )
        if action_internal_planning_enabled
        else None
    )
    brain = BrainAgent(memory=memory, planner=planner)
    vision_agent = VisionAgent(memory=memory, vision=vision)
    completion_evaluator = (
        VLMCompletionEvaluator(
            vision=vision,
            min_confidence=vision_completion_min_confidence,
            use_memory_guidance=vision_completion_use_memory_guidance,
            include_third_person=vision_completion_include_third_person,
            max_images=vision_completion_max_images,
            max_image_side_px=vision_completion_max_image_side_px,
            jpeg_quality=vision_completion_jpeg_quality,
            max_image_b64_chars=vision_completion_max_image_b64_chars,
            image_detail=vision_completion_image_detail,
        )
        if vision_completion_enabled
        else None
    )
    task_planning_skill = (
        DefaultActionTaskPlanningSkill() if action_internal_planning_enabled else None
    )
    step_verifier = (
        VisionBackedActionStepVerifier(
            completion_evaluator=completion_evaluator
            or VLMCompletionEvaluator(
                vision=vision,
                min_confidence=vision_completion_min_confidence,
                use_memory_guidance=vision_completion_use_memory_guidance,
                include_third_person=vision_completion_include_third_person,
                max_images=vision_completion_max_images,
                max_image_side_px=vision_completion_max_image_side_px,
                jpeg_quality=vision_completion_jpeg_quality,
                max_image_b64_chars=vision_completion_max_image_b64_chars,
                image_detail=vision_completion_image_detail,
            )
        )
        if action_internal_planning_enabled
        and action_internal_step_completion_use_vision_completion_monitor
        else None
    )
    navigation_selector = backend_factory.build_vln_selector(
        navigation_base_url=navigation_base_url,
        navigation_model=navigation_model,
        navigation_api_key=navigation_api_key,
        navigation_api_key_env=navigation_api_key_env,
        navigation_timeout_s=navigation_timeout_s,
        navigation_temperature=navigation_temperature,
        navigation_max_retries=navigation_max_retries,
        navigation_retry_backoff_s=navigation_retry_backoff_s,
    )
    navigation_point_selector = backend_factory.build_vln_point_selector(
        navigation_base_url=navigation_base_url,
        navigation_model=navigation_model,
        navigation_api_key=navigation_api_key,
        navigation_api_key_env=navigation_api_key_env,
        navigation_timeout_s=navigation_timeout_s,
        navigation_temperature=navigation_temperature,
        navigation_max_retries=navigation_max_retries,
        navigation_retry_backoff_s=navigation_retry_backoff_s,
    )
    navigation_goal_interpreter = backend_factory.build_vln_goal_interpreter(
        navigation_base_url=navigation_base_url,
        navigation_model=navigation_model,
        navigation_api_key=navigation_api_key,
        navigation_api_key_env=navigation_api_key_env,
        navigation_timeout_s=navigation_timeout_s,
        navigation_temperature=navigation_temperature,
        navigation_max_retries=navigation_max_retries,
        navigation_retry_backoff_s=navigation_retry_backoff_s,
    )
    navigation_agent = NavigationAgent(
        memory=memory,
        policy=navigation_policy,
        projector=projector,
        navigator=navigator,
        selector=navigation_selector,
        approach_point_selector=navigation_point_selector,
        goal_interpreter=navigation_goal_interpreter,
    )
    skill_registry = None
    if anygrasp_config:
        from visiomind.action.agents.action.skills import ActionSkillRegistry

        skill_registry = ActionSkillRegistry.build_default(
            memory=memory,
            policy=manipulation_policy,
            projector=projector,
            anygrasp_config=anygrasp_config,
        )
    action_agent = ActionAgent(
        memory=memory,
        policy=manipulation_policy,
        projector=projector,
        selector=selector,
        skill_registry=skill_registry,
        deliberator=deliberator,
        task_planning_skill=task_planning_skill,
        task_planner=task_planner,
        step_verifier=step_verifier,
        verify_every_control_steps=action_verify_every_control_steps,
        verify_after_first_success=action_internal_step_completion_require_verified_completion,
        max_unverified_internal_step_control_steps=action_max_unverified_internal_step_control_steps,
        require_verified_internal_step_completion=action_internal_step_completion_require_verified_completion,
    )
    inject_agent_capabilities(brain, [vision_agent, navigation_agent, action_agent])

    return ClosedLoopOrchestrator(
        brain_agent=brain,
        vision_agent=vision_agent,
        navigation_agent=navigation_agent,
        action_agent=action_agent,
        max_retries=max_retries,
        max_control_steps_per_subtask=max_control_steps_per_subtask,
        log_navigation_candidates=log_navigation_candidates,
        log_nav2_path_snapshots=log_nav2_path_snapshots,
        vision_heartbeat_interval_steps=vision_heartbeat_interval_steps,
        use_environment_success_signal=runtime_termination_use_environment_success_signal,
        use_brain_completion_signal=runtime_termination_use_brain_completion_signal,
        environment_signal_policy=runtime_termination_environment_signal_policy,
        completion_evaluator=completion_evaluator,
        vision_completion_positive_streak=vision_completion_positive_streak,
        vision_completion_stability_steps=vision_completion_stability_steps,
        vision_completion_action_delta_threshold=vision_completion_action_delta_threshold,
        vision_completion_check_interval_steps=vision_completion_check_interval_steps,
        vision_completion_agent_scope=vision_completion_agent_scope,
    )


__all__ = ["build_closed_loop_orchestrator"]
