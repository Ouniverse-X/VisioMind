from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_CONFIG_SECTIONS = {
    "metadata",
    "task",
    "environment",
    "runtime",
    "recording",
    "logging",
    "services",
    "memory",
    "brain",
    "vision",
    "navigation",
    "action",
    "anygrasp",
}

_FLAT_KEYS = {
    "task_id",
    "task_desc",
    "task_type",
    "planner_mode",
    "env_id",
    "env_idx",
    "total_n_envs",
    "behavior_scene_file",
    "behavior_tro_state_file",
    "behavior_task_instance_id",
    "behavior_scene_state_include_aabb",
    "behavior_scene_state_navigation_role_overrides",
    "behavior_robot_start_position",
    "behavior_robot_start_orientation",
    "behavior_post_reset_robot_position",
    "behavior_post_reset_robot_orientation",
    "behavior_post_reset_object_states",
    "behavior_post_reset_robot_joint_positions",
    "behavior_post_reset_robot_joint_velocities",
    "behavior_post_reset_refresh_observation",
    "behavior_post_reset_settle_steps",
    "behavior_recording_third_person_local_offset",
    "behavior_recording_third_person_look_at_offset",
    "behavior_recording_third_person_prefer_live_capture",
    "behavior_industrial_visual_overlay",
    "behavior_builtin_vlm_detector_enabled",
    "embodiment",
    "progress_log_every",
    "recording_video_scale",
    "logging_verbose",
    "logging_memory_diagnostics",
    "logging_nav2_path_snapshots",
    "gr00t_host",
    "gr00t_port",
    "vision_endpoint",
    "vision_timeout_s",
    "memory_agent_endpoint",
    "vision_heartbeat_interval_steps",
    "memory_mode",
    "memory_agent_enabled",
    "memory_llm_backend",
    "memory_llm_base_url",
    "memory_llm_model",
    "memory_llm_api_key",
    "memory_llm_api_key_env",
    "memory_llm_timeout_s",
    "memory_llm_temperature",
    "memory_llm_max_retries",
    "memory_llm_retry_backoff_s",
    "memory_experience_extraction_enabled",
    "memory_experience_extraction_trigger",
    "memory_experience_extraction_min_confidence_to_write",
    "memory_experience_extraction_min_confidence_to_promote",
    "memory_experience_extraction_extract_completion_criteria",
    "memory_experience_extraction_extract_clarification_answers",
    "brain_planner",
    "brain_base_url",
    "brain_model",
    "brain_api_key",
    "brain_api_key_env",
    "brain_timeout_s",
    "brain_temperature",
    "brain_max_retries",
    "brain_retry_backoff_s",
    "brain_interactive_planning_enabled",
    "brain_interactive_planning_require_user_confirmation",
    "brain_interactive_planning_ask_when_uncertain",
    "brain_interactive_planning_max_questions",
    "brain_interactive_planning_reuse_memory_criteria_min_confidence",
    "navigation_backend",
    "hovsg_graph_root",
    "hovsg_scene_id",
    "hovsg_graph_path",
    "hovsg_scene_map",
    "hovsg_nav_graph_type",
    "hovsg_direct_room_transition_max_gap_m",
    "hovsg_direct_room_transition_min_span_m",
    "hovsg_object_approach_min_portal_stance_clearance_m",
    "nav2_version_profile",
    "nav2_action_name",
    "nav2_planner_id",
    "nav2_frame_id",
    "nav2_timeout_s",
    "nav2_strict",
    "nav2_trav_map_filename",
    "nav2_portal_analysis_map_resolution",
    "nav2_portal_clearance_radius_m",
    "nav2_portal_corridor_standoff_m",
    "nav2_portal_sampling_step_m",
    "nav2_local_path_clearance_radius_m",
    "nav2_local_path_waypoint_spacing_m",
    "navigation_prefer_forward_facing_motion",
    "navigation_portal_alignment_distance_threshold",
    "navigation_portal_prealign_distance_threshold_m",
    "navigation_portal_alignment_footprint_width_m",
    "navigation_portal_alignment_min_lateral_deadband_m",
    "navigation_portal_alignment_wide_clearance_margin_m",
    "navigation_max_linear_velocity",
    "navigation_linear_gain",
    "navigation_local_path_linear_gain",
    "navigation_local_path_max_linear_velocity",
    "navigation_portal_alignment_max_linear_velocity",
    "navigation_object_approach_final_waypoint_tolerance_m",
    "navigation_portal_annotations",
    "action_selector",
    "action_base_url",
    "action_model",
    "action_api_key",
    "action_api_key_env",
    "action_timeout_s",
    "action_temperature",
    "action_max_retries",
    "action_retry_backoff_s",
    "action_max_unverified_internal_step_control_steps",
    "vision_max_retries",
    "vision_retry_backoff_s",
    "vision_completion_enabled",
    "vision_completion_check_interval_steps",
    "vision_completion_agent_scope",
    "vision_completion_positive_streak",
    "vision_completion_stability_steps",
    "vision_completion_min_confidence",
    "vision_completion_action_delta_threshold",
    "vision_completion_use_memory_guidance",
    "vision_completion_include_third_person",
    "vision_completion_max_images",
    "vision_completion_max_image_side_px",
    "vision_completion_jpeg_quality",
    "vision_completion_max_image_b64_chars",
    "vision_completion_image_detail",
    "runtime_termination_use_environment_success_signal",
    "runtime_termination_use_brain_completion_signal",
    "runtime_termination_environment_signal_policy",
    "action_internal_planning_enabled",
    "action_internal_step_completion_use_vision_completion_monitor",
    "action_internal_step_completion_require_verified_completion",
    "max_retries",
    "max_control_steps",
    "no_auto_register",
    "policy_backend",
    "pi05_endpoint",
    "pi05_timeout_s",
    "pi05_task_id",
    "openpi_comet_endpoint",
    "openpi_comet_timeout_s",
    "openpi_comet_task_name",
    "openpi_comet_task_id",
    "openpi_comet_prompt",
    "openpi_comet_action_mode",
    "action_subtask_action",
    "action_target_object",
    "action_control_mode",
    "action_instruction",
    "action_sequence",
    "anygrasp_config",
}


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON config file. CLI args override config values.",
    )


def parse_args_with_config(
    parser: argparse.ArgumentParser, argv: list[str] | None = None
) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)

    if pre_args.config:
        config_values = load_config_file(pre_args.config)
        parser.set_defaults(**config_values)

    return parser.parse_args(argv)


def load_config_file(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Config file root must be a JSON object")
    return normalize_config(payload)


def normalize_config(payload: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {}

    unknown_top_level = set(payload) - _CONFIG_SECTIONS - _FLAT_KEYS
    if unknown_top_level:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown_top_level)}")

    for key in _FLAT_KEYS:
        if key in payload:
            config[key] = payload[key]

    if "metadata" in payload:
        metadata = payload["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError("Config section 'metadata' must be an object")
        config["metadata"] = dict(metadata)

    if "task" in payload:
        config.update(_normalize_task_section(payload["task"]))
    if "environment" in payload:
        config.update(_normalize_environment_section(payload["environment"]))
    if "runtime" in payload:
        config.update(_normalize_runtime_section(payload["runtime"]))
    if "recording" in payload:
        config.update(_normalize_recording_section(payload["recording"]))
    if "logging" in payload:
        config.update(_normalize_logging_section(payload["logging"]))
    if "services" in payload:
        config.update(_normalize_services_section(payload["services"]))
    if "memory" in payload:
        config.update(_normalize_memory_section(payload["memory"]))
    if "brain" in payload:
        config.update(_normalize_brain_section(payload["brain"]))
    if "vision" in payload:
        config.update(_normalize_vision_section(payload["vision"]))
    if "navigation" in payload:
        config.update(_normalize_navigation_section(payload["navigation"]))
    if "action" in payload:
        config.update(_normalize_action_section(payload["action"]))
    if "anygrasp" in payload:
        config.update(_normalize_anygrasp_section(payload["anygrasp"]))

    return config


def _normalize_task_section(section: Any) -> dict[str, Any]:
    _validate_section(section, "task", {"task_id", "task_desc", "task_type", "planner_mode"})
    return dict(section)


def _normalize_environment_section(section: Any) -> dict[str, Any]:
    _validate_section(
        section,
        "environment",
        {
            "env_id",
            "env_idx",
            "total_n_envs",
            "behavior_scene_file",
            "behavior_tro_state_file",
            "behavior_task_instance_id",
            "scene_state_include_aabb",
            "scene_state_navigation_role_overrides",
            "robot_start_position",
            "behavior_robot_start_position",
            "robot_start_orientation",
            "behavior_robot_start_orientation",
            "post_reset_robot_position",
            "behavior_post_reset_robot_position",
            "post_reset_robot_orientation",
            "behavior_post_reset_robot_orientation",
            "post_reset_object_states",
            "behavior_post_reset_object_states",
            "post_reset_robot_joint_positions",
            "behavior_post_reset_robot_joint_positions",
            "post_reset_robot_joint_velocities",
            "behavior_post_reset_robot_joint_velocities",
            "post_reset_refresh_observation",
            "behavior_post_reset_refresh_observation",
            "post_reset_settle_steps",
            "behavior_post_reset_settle_steps",
            "recording_third_person_local_offset",
            "behavior_recording_third_person_local_offset",
            "recording_third_person_look_at_offset",
            "behavior_recording_third_person_look_at_offset",
            "recording_third_person_prefer_live_capture",
            "behavior_recording_third_person_prefer_live_capture",
            "industrial_visual_overlay",
            "behavior_industrial_visual_overlay",
            "auto_register",
            "no_auto_register",
        },
    )
    normalized = {k: v for k, v in section.items() if k != "auto_register"}
    if "scene_state_include_aabb" in section:
        normalized["behavior_scene_state_include_aabb"] = bool(section["scene_state_include_aabb"])
    if "scene_state_navigation_role_overrides" in section:
        overrides = section["scene_state_navigation_role_overrides"]
        if not isinstance(overrides, dict):
            raise ValueError(
                "Config environment.scene_state_navigation_role_overrides must be an object"
            )
        valid_roles = {"obstacle", "support_surface", "overhead", "structural"}
        invalid_roles = {
            str(key): value
            for key, value in overrides.items()
            if not isinstance(value, str) or value.strip().lower() not in valid_roles
        }
        if invalid_roles:
            raise ValueError(
                "Config environment.scene_state_navigation_role_overrides contains "
                f"invalid roles: {invalid_roles}"
            )
        normalized["behavior_scene_state_navigation_role_overrides"] = dict(overrides)
    if "robot_start_position" in section and "behavior_robot_start_position" not in normalized:
        normalized["behavior_robot_start_position"] = section["robot_start_position"]
    if (
        "robot_start_orientation" in section
        and "behavior_robot_start_orientation" not in normalized
    ):
        normalized["behavior_robot_start_orientation"] = section["robot_start_orientation"]
    if (
        "post_reset_robot_position" in section
        and "behavior_post_reset_robot_position" not in normalized
    ):
        normalized["behavior_post_reset_robot_position"] = section["post_reset_robot_position"]
    if (
        "post_reset_robot_orientation" in section
        and "behavior_post_reset_robot_orientation" not in normalized
    ):
        normalized["behavior_post_reset_robot_orientation"] = section[
            "post_reset_robot_orientation"
        ]
    if (
        "post_reset_object_states" in section
        and "behavior_post_reset_object_states" not in normalized
    ):
        normalized["behavior_post_reset_object_states"] = section["post_reset_object_states"]
    if (
        "post_reset_robot_joint_positions" in section
        and "behavior_post_reset_robot_joint_positions" not in normalized
    ):
        normalized["behavior_post_reset_robot_joint_positions"] = section[
            "post_reset_robot_joint_positions"
        ]
    if (
        "post_reset_robot_joint_velocities" in section
        and "behavior_post_reset_robot_joint_velocities" not in normalized
    ):
        normalized["behavior_post_reset_robot_joint_velocities"] = section[
            "post_reset_robot_joint_velocities"
        ]
    if (
        "post_reset_refresh_observation" in section
        and "behavior_post_reset_refresh_observation" not in normalized
    ):
        normalized["behavior_post_reset_refresh_observation"] = bool(
            section["post_reset_refresh_observation"]
        )
    if (
        "post_reset_settle_steps" in section
        and "behavior_post_reset_settle_steps" not in normalized
    ):
        normalized["behavior_post_reset_settle_steps"] = section["post_reset_settle_steps"]
    for short_key in (
        "recording_third_person_local_offset",
        "recording_third_person_look_at_offset",
    ):
        behavior_key = f"behavior_{short_key}"
        if short_key in section and behavior_key not in normalized:
            value = section[short_key]
            if not isinstance(value, list) or len(value) != 3:
                raise ValueError(f"Config environment.{short_key} must be a 3-element list")
            normalized[behavior_key] = [float(item) for item in value]
    if (
        "recording_third_person_prefer_live_capture" in section
        and "behavior_recording_third_person_prefer_live_capture" not in normalized
    ):
        normalized["behavior_recording_third_person_prefer_live_capture"] = bool(
            section["recording_third_person_prefer_live_capture"]
        )
    if (
        "industrial_visual_overlay" in section
        and "behavior_industrial_visual_overlay" not in normalized
    ):
        overlay = section["industrial_visual_overlay"]
        if not isinstance(overlay, dict):
            raise ValueError("Config environment.industrial_visual_overlay must be an object")
        normalized["behavior_industrial_visual_overlay"] = dict(overlay)
    normalized.pop("robot_start_position", None)
    normalized.pop("robot_start_orientation", None)
    normalized.pop("post_reset_robot_position", None)
    normalized.pop("post_reset_robot_orientation", None)
    normalized.pop("post_reset_object_states", None)
    normalized.pop("post_reset_robot_joint_positions", None)
    normalized.pop("post_reset_robot_joint_velocities", None)
    normalized.pop("post_reset_refresh_observation", None)
    normalized.pop("post_reset_settle_steps", None)
    normalized.pop("recording_third_person_local_offset", None)
    normalized.pop("recording_third_person_look_at_offset", None)
    normalized.pop("recording_third_person_prefer_live_capture", None)
    normalized.pop("industrial_visual_overlay", None)
    normalized.pop("scene_state_navigation_role_overrides", None)
    if "auto_register" in section and "no_auto_register" not in section:
        normalized["no_auto_register"] = not bool(section["auto_register"])
    return normalized


def _normalize_runtime_section(section: Any) -> dict[str, Any]:
    _validate_section(
        section,
        "runtime",
        {
            "embodiment",
            "max_retries",
            "max_control_steps",
            "progress_log_every",
            "behavior_builtin_vlm_detector_enabled",
            "termination",
        },
    )
    normalized = {k: v for k, v in section.items() if k != "termination"}
    if "termination" in section:
        normalized.update(_normalize_runtime_termination_section(section["termination"]))
    return normalized


def _normalize_runtime_termination_section(section: Any) -> dict[str, Any]:
    _validate_section(
        section,
        "runtime.termination",
        {
            "use_environment_success_signal",
            "use_brain_completion_signal",
            "environment_signal_policy",
        },
    )
    return {f"runtime_termination_{key}": value for key, value in section.items()}


def _normalize_recording_section(section: Any) -> dict[str, Any]:
    _validate_section(section, "recording", {"video_scale"})
    normalized: dict[str, Any] = {}
    if "video_scale" in section:
        normalized["recording_video_scale"] = section["video_scale"]
    return normalized


def _normalize_logging_section(section: Any) -> dict[str, Any]:
    _validate_section(section, "logging", {"verbose", "memory_diagnostics", "nav2_path_snapshots"})
    normalized: dict[str, Any] = {}
    if "verbose" in section:
        normalized["logging_verbose"] = bool(section["verbose"])
    if "memory_diagnostics" in section:
        normalized["logging_memory_diagnostics"] = bool(section["memory_diagnostics"])
    if "nav2_path_snapshots" in section:
        normalized["logging_nav2_path_snapshots"] = bool(section["nav2_path_snapshots"])
    return normalized


def _normalize_services_section(section: Any) -> dict[str, Any]:
    _validate_section(
        section,
        "services",
        {
            "gr00t_host",
            "gr00t_port",
            "vision_endpoint",
            "memory_agent_endpoint",
            "policy_backend",
            "pi05_endpoint",
            "pi05_timeout_s",
            "pi05_task_id",
            "openpi_comet_endpoint",
            "openpi_comet_timeout_s",
            "openpi_comet_task_name",
            "openpi_comet_task_id",
            "openpi_comet_prompt",
            "openpi_comet_action_mode",
        },
    )
    normalized = {k: v for k, v in section.items() if k != "vision_endpoint"}
    if "vision_endpoint" in section:
        normalized["vision_endpoint"] = section["vision_endpoint"]
    return normalized


def _normalize_memory_section(section: Any) -> dict[str, Any]:
    _validate_section(
        section,
        "memory",
        {
            "mode",
            "memory_mode",
            "agent_enabled",
            "agent",
            "llm",
            "experience_extraction",
        },
    )
    normalized = {
        k: v
        for k, v in section.items()
        if k not in {"mode", "agent_enabled", "agent", "llm", "experience_extraction"}
    }
    if "mode" in section and "memory_mode" not in section:
        normalized["memory_mode"] = section["mode"]
    if "agent_enabled" in section:
        normalized["memory_agent_enabled"] = bool(section["agent_enabled"])
    if "agent" in section:
        normalized.update(_normalize_memory_agent_section(section["agent"]))
    if "llm" in section:
        normalized.update(_normalize_memory_llm_section(section["llm"]))
    if "experience_extraction" in section:
        normalized.update(_normalize_memory_experience_section(section["experience_extraction"]))
    return normalized


def _normalize_memory_agent_section(section: Any) -> dict[str, Any]:
    allowed = {"endpoint"}
    _validate_section(section, "memory.agent", allowed)
    aliases = {"endpoint": "memory_agent_endpoint"}
    return {aliases[key]: value for key, value in section.items()}


def _normalize_memory_llm_section(section: Any) -> dict[str, Any]:
    allowed = {
        "backend",
        "base_url",
        "model",
        "api_key",
        "api_key_env",
        "timeout_s",
        "temperature",
        "max_retries",
        "retry_backoff_s",
    }
    _validate_section(section, "memory.llm", allowed)
    aliases = {
        "backend": "memory_llm_backend",
        "base_url": "memory_llm_base_url",
        "model": "memory_llm_model",
        "api_key": "memory_llm_api_key",
        "api_key_env": "memory_llm_api_key_env",
        "timeout_s": "memory_llm_timeout_s",
        "temperature": "memory_llm_temperature",
        "max_retries": "memory_llm_max_retries",
        "retry_backoff_s": "memory_llm_retry_backoff_s",
    }
    return {aliases[key]: value for key, value in section.items()}


def _normalize_memory_experience_section(section: Any) -> dict[str, Any]:
    allowed = {
        "enabled",
        "trigger",
        "min_confidence_to_write",
        "min_confidence_to_promote",
        "extract_completion_criteria",
        "extract_clarification_answers",
    }
    _validate_section(section, "memory.experience_extraction", allowed)
    return {f"memory_experience_extraction_{key}": value for key, value in section.items()}


def _normalize_brain_section(section: Any) -> dict[str, Any]:
    _validate_section(
        section,
        "brain",
        {
            "planner",
            "base_url",
            "model",
            "api_key",
            "api_key_env",
            "timeout_s",
            "temperature",
            "max_retries",
            "retry_backoff_s",
            "heartbeat_interval_steps",
            "interactive_planning",
        },
    )
    normalized = {k: v for k, v in section.items() if k != "interactive_planning"}
    aliases = {
        "planner": "brain_planner",
        "base_url": "brain_base_url",
        "model": "brain_model",
        "api_key": "brain_api_key",
        "api_key_env": "brain_api_key_env",
        "timeout_s": "brain_timeout_s",
        "temperature": "brain_temperature",
        "max_retries": "brain_max_retries",
        "retry_backoff_s": "brain_retry_backoff_s",
    }
    for src, dest in aliases.items():
        if src in section and dest not in normalized:
            normalized[dest] = section[src]
        normalized.pop(src, None)
    if "interactive_planning" in section:
        normalized.update(
            _normalize_brain_interactive_planning_section(section["interactive_planning"])
        )
    return normalized


def _normalize_brain_interactive_planning_section(section: Any) -> dict[str, Any]:
    _validate_section(
        section,
        "brain.interactive_planning",
        {
            "enabled",
            "require_user_confirmation",
            "ask_when_uncertain",
            "max_questions",
            "reuse_memory_criteria_min_confidence",
        },
    )
    return {f"brain_interactive_planning_{key}": value for key, value in section.items()}


def _normalize_vision_section(section: Any) -> dict[str, Any]:
    _validate_section(
        section,
        "vision",
        {
            "provider",
            "base_url",
            "model",
            "api_key",
            "api_key_env",
            "timeout_s",
            "max_retries",
            "retry_backoff_s",
            "heartbeat_interval_steps",
            "completion",
        },
    )
    normalized = {k: v for k, v in section.items() if k != "completion"}
    if "timeout_s" in section and "vision_timeout_s" not in normalized:
        normalized["vision_timeout_s"] = section["timeout_s"]
    if "max_retries" in section and "vision_max_retries" not in normalized:
        normalized["vision_max_retries"] = section["max_retries"]
    if "retry_backoff_s" in section and "vision_retry_backoff_s" not in normalized:
        normalized["vision_retry_backoff_s"] = section["retry_backoff_s"]
    if (
        "heartbeat_interval_steps" in section
        and "vision_heartbeat_interval_steps" not in normalized
    ):
        normalized["vision_heartbeat_interval_steps"] = section["heartbeat_interval_steps"]

    for key in (
        "provider",
        "base_url",
        "model",
        "api_key",
        "api_key_env",
        "timeout_s",
        "max_retries",
        "retry_backoff_s",
        "heartbeat_interval_steps",
    ):
        normalized.pop(key, None)

    if "completion" in section:
        normalized.update(_normalize_vision_completion_section(section["completion"]))

    return normalized


def _normalize_vision_completion_section(section: Any) -> dict[str, Any]:
    _validate_section(
        section,
        "vision.completion",
        {
            "enabled",
            "check_interval_steps",
            "agent_scope",
            "positive_streak",
            "stability_steps",
            "min_confidence",
            "action_delta_threshold",
            "use_memory_guidance",
            "include_third_person",
            "max_images",
            "max_image_side_px",
            "jpeg_quality",
            "max_image_b64_chars",
            "image_detail",
        },
    )
    return {f"vision_completion_{key}": value for key, value in section.items()}


def _normalize_navigation_section(section: Any) -> dict[str, Any]:
    _validate_section(
        section,
        "navigation",
        {
            "llm",
            "backend",
            "graph_root",
            "scene_id",
            "graph_path",
            "scene_map",
            "nav_graph_type",
            "direct_room_transition_max_gap_m",
            "direct_room_transition_min_span_m",
            "object_approach_min_portal_stance_clearance_m",
            "version_profile",
            "action_name",
            "planner_id",
            "frame_id",
            "timeout_s",
            "strict",
            "trav_map_filename",
            "portal_analysis_map_resolution",
            "portal_clearance_radius_m",
            "portal_corridor_standoff_m",
            "portal_sampling_step_m",
            "local_path_clearance_radius_m",
            "local_path_waypoint_spacing_m",
            "prefer_forward_facing_motion",
            "portal_alignment_distance_threshold",
            "portal_prealign_distance_threshold_m",
            "portal_alignment_footprint_width_m",
            "portal_alignment_min_lateral_deadband_m",
            "portal_alignment_wide_clearance_margin_m",
            "max_linear_velocity",
            "linear_gain",
            "local_path_linear_gain",
            "local_path_max_linear_velocity",
            "portal_alignment_max_linear_velocity",
            "object_approach_final_waypoint_tolerance_m",
            "portal_annotations",
        },
    )
    normalized = {k: v for k, v in section.items() if k != "llm"}
    if "llm" in section:
        normalized.update(_normalize_navigation_llm_section(section["llm"]))
    aliases = {
        "backend": "navigation_backend",
        "graph_root": "hovsg_graph_root",
        "scene_id": "hovsg_scene_id",
        "graph_path": "hovsg_graph_path",
        "scene_map": "hovsg_scene_map",
        "nav_graph_type": "hovsg_nav_graph_type",
        "direct_room_transition_max_gap_m": "hovsg_direct_room_transition_max_gap_m",
        "direct_room_transition_min_span_m": "hovsg_direct_room_transition_min_span_m",
        "object_approach_min_portal_stance_clearance_m": (
            "hovsg_object_approach_min_portal_stance_clearance_m"
        ),
        "version_profile": "nav2_version_profile",
        "action_name": "nav2_action_name",
        "planner_id": "nav2_planner_id",
        "frame_id": "nav2_frame_id",
        "timeout_s": "nav2_timeout_s",
        "strict": "nav2_strict",
        "trav_map_filename": "nav2_trav_map_filename",
        "portal_analysis_map_resolution": "nav2_portal_analysis_map_resolution",
        "portal_clearance_radius_m": "nav2_portal_clearance_radius_m",
        "portal_corridor_standoff_m": "nav2_portal_corridor_standoff_m",
        "portal_sampling_step_m": "nav2_portal_sampling_step_m",
        "local_path_clearance_radius_m": "nav2_local_path_clearance_radius_m",
        "local_path_waypoint_spacing_m": "nav2_local_path_waypoint_spacing_m",
        "prefer_forward_facing_motion": "navigation_prefer_forward_facing_motion",
        "portal_alignment_distance_threshold": "navigation_portal_alignment_distance_threshold",
        "portal_prealign_distance_threshold_m": "navigation_portal_prealign_distance_threshold_m",
        "portal_alignment_footprint_width_m": "navigation_portal_alignment_footprint_width_m",
        "portal_alignment_min_lateral_deadband_m": "navigation_portal_alignment_min_lateral_deadband_m",
        "portal_alignment_wide_clearance_margin_m": "navigation_portal_alignment_wide_clearance_margin_m",
        "max_linear_velocity": "navigation_max_linear_velocity",
        "linear_gain": "navigation_linear_gain",
        "local_path_linear_gain": "navigation_local_path_linear_gain",
        "local_path_max_linear_velocity": "navigation_local_path_max_linear_velocity",
        "portal_alignment_max_linear_velocity": "navigation_portal_alignment_max_linear_velocity",
        "object_approach_final_waypoint_tolerance_m": "navigation_object_approach_final_waypoint_tolerance_m",
        "portal_annotations": "navigation_portal_annotations",
    }
    for src, dest in aliases.items():
        if src in section and dest not in normalized:
            normalized[dest] = section[src]
        normalized.pop(src, None)
    return normalized


def _normalize_navigation_llm_section(section: Any) -> dict[str, Any]:
    _validate_section(
        section,
        "navigation.llm",
        {
            "base_url",
            "model",
            "api_key",
            "api_key_env",
            "timeout_s",
            "temperature",
            "max_retries",
            "retry_backoff_s",
        },
    )
    aliases = {
        "base_url": "navigation_base_url",
        "model": "navigation_model",
        "api_key": "navigation_api_key",
        "api_key_env": "navigation_api_key_env",
        "timeout_s": "navigation_timeout_s",
        "temperature": "navigation_temperature",
        "max_retries": "navigation_max_retries",
        "retry_backoff_s": "navigation_retry_backoff_s",
    }
    return {dest: section[src] for src, dest in aliases.items() if src in section}


def _normalize_action_section(section: Any) -> dict[str, Any]:
    _validate_section(
        section,
        "action",
        {
            "selector",
            "base_url",
            "model",
            "api_key",
            "api_key_env",
            "timeout_s",
            "temperature",
            "max_retries",
            "retry_backoff_s",
            "action_verify_every_control_steps",
            "max_unverified_internal_step_control_steps",
            "control_mode",
            "allow_base_motion",
            "internal_planning",
            "internal_step_completion",
        },
    )
    normalized = {
        k: v
        for k, v in section.items()
        if k not in {"internal_planning", "internal_step_completion"}
    }
    aliases = {
        "selector": "action_selector",
        "base_url": "action_base_url",
        "model": "action_model",
        "api_key": "action_api_key",
        "api_key_env": "action_api_key_env",
        "timeout_s": "action_timeout_s",
        "temperature": "action_temperature",
        "max_retries": "action_max_retries",
        "retry_backoff_s": "action_retry_backoff_s",
        "max_unverified_internal_step_control_steps": "action_max_unverified_internal_step_control_steps",
        "control_mode": "action_control_mode",
        "allow_base_motion": "action_allow_base_motion",
    }
    for src, dest in aliases.items():
        if src in section and dest not in normalized:
            normalized[dest] = section[src]
        normalized.pop(src, None)
    if "internal_step_completion" in section:
        normalized.update(
            _normalize_action_internal_step_completion_section(section["internal_step_completion"])
        )
    if "internal_planning" in section:
        normalized.update(_normalize_action_internal_planning_section(section["internal_planning"]))
    return normalized


def _normalize_action_internal_planning_section(section: Any) -> dict[str, Any]:
    _validate_section(section, "action.internal_planning", {"enabled"})
    return {f"action_internal_planning_{key}": value for key, value in section.items()}


def _normalize_action_internal_step_completion_section(section: Any) -> dict[str, Any]:
    _validate_section(
        section,
        "action.internal_step_completion",
        {
            "use_vision_completion_monitor",
            "require_verified_completion",
        },
    )
    return {f"action_internal_step_completion_{key}": value for key, value in section.items()}


def _normalize_anygrasp_section(section: Any) -> dict[str, Any]:
    _validate_section(
        section,
        "anygrasp",
        {
            "sdk_root",
            "checkpoint_path",
            "license_dir",
            "endpoint",
            "max_gripper_width",
            "gripper_height",
            "top_down_grasp",
            "dense_grasp",
            "collision_detection",
            "apply_nms",
            "top_k",
            "request_timeout_s",
            "region_margin",
            "camera_sensor",
            "depth_trunc",
            "sensor_warmup_frames",
            "sensor_read_retries",
            "perception_audit_dir",
            "target_depth_outlier_m",
            "mask_dilation_px",
            "min_target_points",
            "require_target_mask",
            "require_target_object",
            "approach_direction",
            "approach_thresh",
            "max_attempts",
            "arm",
            "curobo_batch_size",
            "allow_fallback",
            "target_anchor_tolerance_m",
            "candidate_target_centroid_tolerance_m",
            "candidate_max_world_approach_z",
            "candidate_force_world_vertical_approach",
            "candidate_world_vertical_jaw_axis",
            "candidate_min_width_m",
            "candidate_min_inner_target_points",
            "candidate_require_center_straddle",
            "candidate_canonical_depth_base_m",
            "candidate_inner_line_gate_enabled",
            "candidate_inner_line_margin_m",
            "candidate_inner_line_min_target_points",
            "candidate_inner_line_min_target_fraction",
            "candidate_inner_line_min_overlap_m",
            "candidate_inner_line_require_center_straddle",
            "candidate_min_open_jaw_clearance_m",
            "candidate_fit_depth_to_robot_inner_line",
            "candidate_fit_depth_min_m",
            "candidate_fit_depth_step_m",
            "candidate_fit_depth_max_m",
            "candidate_fit_depth_selection_mode",
            "candidate_target_collision_geometry_enabled",
            "candidate_non_target_collision_audit_enabled",
            "candidate_non_target_collision_margin_m",
            "candidate_preferred_detector_ranks",
            "candidate_recenter_to_target_centroid",
            "candidate_recenter_axes",
            "candidate_recenter_reference",
            "candidate_recenter_max_translation_m",
            "candidate_detection_refreshes",
            "candidate_detection_only",
            "candidate_min_translation_m",
            "candidate_min_approach_angle_deg",
            "pregrasp_offset_m",
            "whole_body_standoff_m",
            "lift_height_m",
            "post_lift_yaw_deg",
            "post_lift_yaw_cycles",
            "post_lift_place_back",
            "place_back_clearance_m",
            "place_back_retreat_m",
            "place_inside_grid_shape",
            "place_inside_cell_margin_m",
            "skip_standoff_if_within_m",
            "constrained_approach",
            "retry_unconstrained_approach",
            "approach_segment_max_m",
            "approach_target_displacement_tolerance_m",
            "close_target_displacement_tolerance_m",
            "approach_goal_position_tolerance_m",
            "live_open_jaw_y_correction_max_m",
            "grasping_mode_override",
            "collision_workspace_radius_m",
            "verification_steps",
            "verification_min_target_z_rise_m",
            "verification_relative_offset_tolerance_m",
            "verification_relative_orientation_tolerance_deg",
            "verification_require_attachment_valid",
            "physical_require_bilateral_contact_before_lift",
            "physical_staged_close_enabled",
            "physical_close_compression_m",
            "physical_close_stage_count",
            "physical_close_hold_steps",
            "physical_close_stage_displacement_tolerance_m",
            "physical_unilateral_contact_displacement_tolerance_m",
            "fingertip_depth_override_m",
            "eef_approach_offset_m",
        },
    )
    return {"anygrasp_config": dict(section)}


def _validate_section(section: Any, name: str, allowed_keys: set[str]) -> None:
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{name}' must be a JSON object")
    unknown_keys = set(section) - allowed_keys
    if unknown_keys:
        raise ValueError(f"Unknown config keys in section '{name}': {sorted(unknown_keys)}")
