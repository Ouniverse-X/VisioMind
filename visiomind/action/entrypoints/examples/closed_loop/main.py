from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from typing import Any

from visiomind.action.config_loader import parse_args_with_config
from visiomind.action.agents.brain.contracts import PlanConfirmation, UserAnswer
from visiomind.action.entrypoints.examples.closed_loop import parser as entrypoint_parser
from visiomind.action.integrations.navigation.nav2.navigator import DEFAULT_NAV2_VERSION_PROFILE
from visiomind.action.runtime.assembly import agent_factory as runtime_agent_factory
from visiomind.action.runtime.assembly import backend_factory as runtime_backend_factory
from visiomind.action.runtime.assembly import runtime_builder as runtime_runtime_builder
from visiomind.action.runtime.orchestrator.closed_loop import ClosedLoopOrchestrator
from visiomind.action.runtime.session.events import VisioMindActionEvent
from visiomind.action.shared.telemetry.payload_sanitizer import strip_image_payloads


class _GraphModuleStderrFilter:
    _MARKER = "class GraphModule(torch.nn.Module):"

    def __init__(self, wrapped: Any):
        self._wrapped = wrapped

    def write(self, data: str) -> int:
        if self._MARKER not in data:
            return self._wrapped.write(data)
        prefix, _, _ = data.partition(self._MARKER)
        if prefix:
            self._wrapped.write(prefix)
        self._wrapped.write("[visiomind] suppressed verbose Torch GraphModule stderr dump\n")
        return len(data)

    def flush(self) -> None:
        self._wrapped.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


def install_console_noise_filters() -> None:
    value = os.environ.get("VISIOMIND_ACTION_SUPPRESS_TORCH_GRAPH_STDERR", "1").strip().lower()
    if value in {"0", "false", "no"}:
        return
    if not isinstance(sys.stderr, _GraphModuleStderrFilter):
        sys.stderr = _GraphModuleStderrFilter(sys.stderr)


def _behavior_task_name_from_env_id(env_id: str) -> str | None:
    marker = "sim_behavior_r1_pro/"
    if marker not in str(env_id):
        return None
    return str(env_id).split(marker, 1)[1].strip("/") or None


def _expected_pi05_task_id(task_name: str) -> int | None:
    try:
        module = importlib.import_module("gr00t.eval.sim.BEHAVIOR.behavior_env")
    except Exception:
        return None
    mapping = getattr(module, "TASK_NAMES_TO_INDICES", None)
    if not isinstance(mapping, dict):
        return None
    value = mapping.get(task_name)
    return int(value) if value is not None else None


def _normalize_policy_task_id(
    *, policy_backend: str, env_id: str, task_id: int | None
) -> int | None:
    backend = str(policy_backend).strip().lower()
    if backend not in {"pi05", "openpi_comet"}:
        return task_id
    task_name = _behavior_task_name_from_env_id(env_id)
    if task_name is None:
        return task_id
    expected = _expected_pi05_task_id(task_name)
    if expected is None:
        return task_id
    if task_id is None:
        return expected
    if int(task_id) != expected:
        label = "Pi0.5" if backend == "pi05" else "OpenPI Comet"
        raise ValueError(
            f"{label} task_id mismatch for env_id={env_id!r}: got {task_id}, expected {expected}."
        )
    return task_id


def _normalize_pi05_task_id(
    *, policy_backend: str, env_id: str, pi05_task_id: int | None
) -> int | None:
    return _normalize_policy_task_id(
        policy_backend=policy_backend, env_id=env_id, task_id=pi05_task_id
    )


def format_console_result(result: dict[str, Any]) -> str:
    compact = _compact_console_result(strip_image_payloads(result))
    return json.dumps(compact, ensure_ascii=False, indent=2)


def _compact_console_result(result: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: result.get(key)
        for key in ("trace_id", "task_id", "task_description", "task_type", "started_at")
        if result.get(key) not in (None, "", [], {})
    }
    results = result.get("results")
    if isinstance(results, list):
        compact["results"] = [
            _compact_console_subtask_result(item) for item in results if isinstance(item, dict)
        ]
    final = result.get("final")
    if isinstance(final, dict):
        compact["final"] = _compact_console_final(final)
    return compact


def _compact_console_subtask_result(result: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: result.get(key)
        for key in ("subtask_id", "status", "error_code", "latency_ms")
        if result.get(key) not in (None, "", [], {})
    }
    payload = result.get("result")
    if isinstance(payload, dict):
        compact["result"] = _compact_runtime_payload(payload)
    return compact


def _compact_console_final(final: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: final.get(key)
        for key in ("outcome", "failure_reason")
        if final.get(key) not in (None, "", [], {})
    }
    end_info = final.get("end_info")
    if isinstance(end_info, dict):
        compact["end_info"] = {
            key: end_info.get(key)
            for key in ("episode_id", "outcome", "duration_s", "failure_reason")
            if end_info.get(key) not in (None, "", [], {})
        }
    reflection = final.get("reflection")
    if isinstance(reflection, dict):
        compact["reflection"] = {
            key: reflection.get(key)
            for key in (
                "status",
                "episode_id",
                "outcome",
                "failure_reason",
                "recent_observation_count",
                "generated_by",
            )
            if reflection.get(key) not in (None, "", [], {})
        }
        consolidation = reflection.get("memory_consolidation")
        if isinstance(consolidation, dict):
            compact["reflection"]["memory_consolidation"] = {
                key: consolidation.get(key)
                for key in ("job_id", "episode_id", "status", "mode")
                if consolidation.get(key) not in (None, "", [], {})
            }
    cleanup_errors = final.get("cleanup_errors")
    if isinstance(cleanup_errors, list) and cleanup_errors:
        compact["cleanup_errors"] = cleanup_errors
    environment = final.get("environment")
    if isinstance(environment, dict):
        compact["environment"] = _compact_console_environment(environment)
    return compact


def _compact_console_environment(environment: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: environment.get(key)
        for key in (
            "env_id",
            "step_count",
            "task_success",
            "terminated",
            "truncated",
            "task_progress",
            "record_dir",
            "process_log",
            "video_path",
            "closed",
        )
        if environment.get(key) not in (None, "", [], {})
    }
    last_info = environment.get("last_info")
    if isinstance(last_info, dict):
        compact_last_info = _compact_environment_feedback(last_info)
        if compact_last_info:
            compact["last_info"] = compact_last_info
    return compact


def _compact_environment_feedback(feedback: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: feedback.get(key)
        for key in (
            "scene_id",
            "pose",
            "current_room",
            "current_region",
            "room_id",
            "floor_id",
            "subtask_name",
            "task_success",
            "task_progress",
            "subtask_completed",
            "subtask_succeeded",
            "subtask_completion_reason",
            "image_count",
            "rgb_keys",
        )
        if feedback.get(key) not in (None, "", [], {})
    }
    heartbeat = feedback.get("environment_vlm_heartbeat")
    if isinstance(heartbeat, dict):
        compact["environment_vlm_heartbeat"] = {
            key: heartbeat.get(key)
            for key in (
                "last_result",
                "last_success",
                "reported_success",
                "success_confirmation_count",
                "success_confirmation_threshold",
                "subtask_completed",
                "subtask_succeeded",
                "subtask_completion_reason",
            )
            if heartbeat.get(key) not in (None, "", [], {})
        }
    return compact


def _compact_runtime_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: payload.get(key)
        for key in (
            "message",
            "agent",
            "attempt",
            "control_step",
            "action_keys",
            "policy_info",
            "completion_verdict",
            "error",
        )
        if payload.get(key) not in (None, "", [], {})
    }
    env_feedback = payload.get("env_feedback") or payload.get("feedback")
    if isinstance(env_feedback, dict):
        compact_feedback = _compact_environment_feedback(env_feedback)
        if compact_feedback:
            compact["env_feedback"] = compact_feedback
    return compact


def build_planner(
    planner_backend: str,
    brain_base_url: str | None,
    brain_model: str | None,
    brain_api_key: str | None,
    brain_api_key_env: str,
    brain_timeout_s: float,
    brain_temperature: float,
    brain_max_retries: int,
    brain_retry_backoff_s: float,
):
    return runtime_backend_factory.build_planner(
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


def build_orchestrator(
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
    logging_nav2_path_snapshots: bool = False,
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
    return runtime_agent_factory.build_closed_loop_orchestrator(
        embodiment=embodiment,
        gr00t_host=gr00t_host,
        gr00t_port=gr00t_port,
        vision_endpoint=vision_endpoint,
        vision_timeout_s=vision_timeout_s,
        vision_max_retries=vision_max_retries,
        vision_retry_backoff_s=vision_retry_backoff_s,
        vision_heartbeat_interval_steps=vision_heartbeat_interval_steps,
        memory_agent_endpoint=memory_agent_endpoint,
        use_memory_agent=use_memory_agent,
        max_retries=max_retries,
        max_control_steps_per_subtask=max_control_steps_per_subtask,
        action_verify_every_control_steps=action_verify_every_control_steps,
        action_max_unverified_internal_step_control_steps=action_max_unverified_internal_step_control_steps,
        action_internal_planning_enabled=action_internal_planning_enabled,
        action_internal_step_completion_use_vision_completion_monitor=(
            action_internal_step_completion_use_vision_completion_monitor
        ),
        action_internal_step_completion_require_verified_completion=(
            action_internal_step_completion_require_verified_completion
        ),
        planner_backend=planner_backend,
        brain_base_url=brain_base_url,
        brain_model=brain_model,
        brain_api_key=brain_api_key,
        brain_api_key_env=brain_api_key_env,
        brain_timeout_s=brain_timeout_s,
        brain_temperature=brain_temperature,
        brain_max_retries=brain_max_retries,
        brain_retry_backoff_s=brain_retry_backoff_s,
        action_selector=action_selector,
        action_base_url=action_base_url,
        action_model=action_model,
        action_api_key=action_api_key,
        action_api_key_env=action_api_key_env,
        action_timeout_s=action_timeout_s,
        action_temperature=action_temperature,
        action_max_retries=action_max_retries,
        action_retry_backoff_s=action_retry_backoff_s,
        navigation_backend=navigation_backend,
        hovsg_graph_root=hovsg_graph_root,
        hovsg_scene_id=hovsg_scene_id,
        hovsg_graph_path=hovsg_graph_path,
        hovsg_nav_graph_type=hovsg_nav_graph_type,
        hovsg_direct_room_transition_max_gap_m=hovsg_direct_room_transition_max_gap_m,
        hovsg_direct_room_transition_min_span_m=hovsg_direct_room_transition_min_span_m,
        hovsg_object_approach_min_portal_stance_clearance_m=(
            hovsg_object_approach_min_portal_stance_clearance_m
        ),
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
        navigation_prefer_forward_facing_motion=navigation_prefer_forward_facing_motion,
        navigation_portal_alignment_distance_threshold=navigation_portal_alignment_distance_threshold,
        navigation_portal_prealign_distance_threshold_m=navigation_portal_prealign_distance_threshold_m,
        navigation_portal_alignment_footprint_width_m=navigation_portal_alignment_footprint_width_m,
        navigation_portal_alignment_min_lateral_deadband_m=navigation_portal_alignment_min_lateral_deadband_m,
        navigation_portal_alignment_wide_clearance_margin_m=navigation_portal_alignment_wide_clearance_margin_m,
        navigation_max_linear_velocity=navigation_max_linear_velocity,
        navigation_linear_gain=navigation_linear_gain,
        navigation_local_path_linear_gain=navigation_local_path_linear_gain,
        navigation_local_path_max_linear_velocity=navigation_local_path_max_linear_velocity,
        navigation_portal_alignment_max_linear_velocity=navigation_portal_alignment_max_linear_velocity,
        navigation_object_approach_final_waypoint_tolerance_m=navigation_object_approach_final_waypoint_tolerance_m,
        navigation_max_angular_velocity=navigation_max_angular_velocity,
        navigation_local_path_angular_gain_scale=navigation_local_path_angular_gain_scale,
        policy_backend=policy_backend,
        pi05_endpoint=pi05_endpoint,
        pi05_timeout_s=pi05_timeout_s,
        pi05_task_id=pi05_task_id,
        openpi_comet_endpoint=openpi_comet_endpoint,
        openpi_comet_timeout_s=openpi_comet_timeout_s,
        openpi_comet_task_name=openpi_comet_task_name,
        openpi_comet_task_id=openpi_comet_task_id,
        openpi_comet_prompt=openpi_comet_prompt,
        openpi_comet_action_mode=openpi_comet_action_mode,
        log_navigation_candidates=log_navigation_candidates,
        log_nav2_path_snapshots=logging_nav2_path_snapshots,
        navigation_base_url=navigation_base_url,
        navigation_model=navigation_model,
        navigation_api_key=navigation_api_key,
        navigation_api_key_env=navigation_api_key_env,
        navigation_timeout_s=navigation_timeout_s,
        navigation_temperature=navigation_temperature,
        navigation_max_retries=navigation_max_retries,
        navigation_retry_backoff_s=navigation_retry_backoff_s,
        memory_agent_enabled=memory_agent_enabled,
        memory_llm_backend=memory_llm_backend,
        memory_llm_base_url=memory_llm_base_url,
        memory_llm_model=memory_llm_model,
        memory_llm_api_key=memory_llm_api_key,
        memory_llm_api_key_env=memory_llm_api_key_env,
        memory_llm_timeout_s=memory_llm_timeout_s,
        memory_llm_temperature=memory_llm_temperature,
        memory_llm_max_retries=memory_llm_max_retries,
        memory_llm_retry_backoff_s=memory_llm_retry_backoff_s,
        memory_experience_extraction_enabled=memory_experience_extraction_enabled,
        memory_experience_extraction_min_confidence_to_write=memory_experience_extraction_min_confidence_to_write,
        memory_experience_extraction_min_confidence_to_promote=memory_experience_extraction_min_confidence_to_promote,
        runtime_termination_use_environment_success_signal=runtime_termination_use_environment_success_signal,
        runtime_termination_use_brain_completion_signal=runtime_termination_use_brain_completion_signal,
        runtime_termination_environment_signal_policy=runtime_termination_environment_signal_policy,
        vision_completion_enabled=vision_completion_enabled,
        vision_completion_positive_streak=vision_completion_positive_streak,
        vision_completion_stability_steps=vision_completion_stability_steps,
        vision_completion_min_confidence=vision_completion_min_confidence,
        vision_completion_action_delta_threshold=vision_completion_action_delta_threshold,
        vision_completion_check_interval_steps=vision_completion_check_interval_steps,
        vision_completion_agent_scope=vision_completion_agent_scope,
        vision_completion_use_memory_guidance=vision_completion_use_memory_guidance,
        vision_completion_include_third_person=vision_completion_include_third_person,
        vision_completion_max_images=vision_completion_max_images,
        vision_completion_max_image_side_px=vision_completion_max_image_side_px,
        vision_completion_jpeg_quality=vision_completion_jpeg_quality,
        vision_completion_max_image_b64_chars=vision_completion_max_image_b64_chars,
        vision_completion_image_detail=vision_completion_image_detail,
        logging_verbose=logging_verbose,
        anygrasp_config=anygrasp_config,
    )


def build_vln_selector(
    *,
    navigation_base_url: str | None,
    navigation_model: str | None,
    navigation_api_key: str | None,
    navigation_api_key_env: str,
    navigation_timeout_s: float,
    navigation_temperature: float,
    navigation_max_retries: int,
    navigation_retry_backoff_s: float,
):
    return runtime_backend_factory.build_vln_selector(
        navigation_base_url=navigation_base_url,
        navigation_model=navigation_model,
        navigation_api_key=navigation_api_key,
        navigation_api_key_env=navigation_api_key_env,
        navigation_timeout_s=navigation_timeout_s,
        navigation_temperature=navigation_temperature,
        navigation_max_retries=navigation_max_retries,
        navigation_retry_backoff_s=navigation_retry_backoff_s,
    )


def build_vln_point_selector(
    *,
    navigation_base_url: str | None,
    navigation_model: str | None,
    navigation_api_key: str | None,
    navigation_api_key_env: str,
    navigation_timeout_s: float,
    navigation_temperature: float,
    navigation_max_retries: int,
    navigation_retry_backoff_s: float,
):
    return runtime_backend_factory.build_vln_point_selector(
        navigation_base_url=navigation_base_url,
        navigation_model=navigation_model,
        navigation_api_key=navigation_api_key,
        navigation_api_key_env=navigation_api_key_env,
        navigation_timeout_s=navigation_timeout_s,
        navigation_temperature=navigation_temperature,
        navigation_max_retries=navigation_max_retries,
        navigation_retry_backoff_s=navigation_retry_backoff_s,
    )


def build_vln_policy(
    *,
    backend: str,
    gr00t_host: str,
    gr00t_port: int,
    prefer_forward_facing_motion: bool = False,
    portal_alignment_distance_threshold: float = 1.2,
    portal_prealign_distance_threshold_m: float = 1.2,
    portal_alignment_footprint_width_m: float = 0.72,
    portal_alignment_min_lateral_deadband_m: float = 0.01,
    portal_alignment_wide_clearance_margin_m: float = 0.4,
    max_linear_velocity: float = 0.60,
    linear_gain: float = 0.45,
    local_path_linear_gain: float = 0.75,
    local_path_max_linear_velocity: float = 0.85,
    portal_alignment_max_linear_velocity: float = 0.28,
    object_approach_final_waypoint_tolerance_m: float = 0.45,
    max_angular_velocity: float = 0.8,
    local_path_angular_gain_scale: float = 0.7,
):
    return runtime_backend_factory.build_vln_policy(
        backend=backend,
        gr00t_host=gr00t_host,
        gr00t_port=gr00t_port,
        prefer_forward_facing_motion=prefer_forward_facing_motion,
        portal_alignment_distance_threshold=portal_alignment_distance_threshold,
        portal_prealign_distance_threshold_m=portal_prealign_distance_threshold_m,
        portal_alignment_footprint_width_m=portal_alignment_footprint_width_m,
        portal_alignment_min_lateral_deadband_m=portal_alignment_min_lateral_deadband_m,
        portal_alignment_wide_clearance_margin_m=portal_alignment_wide_clearance_margin_m,
        max_linear_velocity=max_linear_velocity,
        linear_gain=linear_gain,
        local_path_linear_gain=local_path_linear_gain,
        local_path_max_linear_velocity=local_path_max_linear_velocity,
        portal_alignment_max_linear_velocity=portal_alignment_max_linear_velocity,
        object_approach_final_waypoint_tolerance_m=object_approach_final_waypoint_tolerance_m,
        max_angular_velocity=max_angular_velocity,
        local_path_angular_gain_scale=local_path_angular_gain_scale,
    )


def build_vln_navigator(
    *,
    backend: str,
    hovsg_graph_root: str | None,
    hovsg_scene_id: str | None,
    hovsg_graph_path: str | None,
    hovsg_nav_graph_type: str | None = None,
    hovsg_direct_room_transition_max_gap_m: float = 0.25,
    hovsg_direct_room_transition_min_span_m: float = 1.0,
    hovsg_object_approach_min_portal_stance_clearance_m: float = 0.45,
    nav2_version_profile: str = DEFAULT_NAV2_VERSION_PROFILE,
    nav2_action_name: str = "compute_path_to_pose",
    nav2_planner_id: str | None = None,
    nav2_frame_id: str = "map",
    nav2_timeout_s: float = 8.0,
    nav2_strict: bool = False,
    nav2_trav_map_filename: str | None = None,
    nav2_portal_analysis_map_resolution: float = 0.05,
    nav2_portal_clearance_radius_m: float = 0.35,
    nav2_portal_corridor_standoff_m: float = 0.18,
    nav2_portal_sampling_step_m: float = 0.05,
    nav2_local_path_clearance_radius_m: float = 0.0,
    nav2_local_path_waypoint_spacing_m: float = 0.35,
):
    return runtime_backend_factory.build_vln_navigator(
        backend=backend,
        hovsg_graph_root=hovsg_graph_root,
        hovsg_scene_id=hovsg_scene_id,
        hovsg_graph_path=hovsg_graph_path,
        hovsg_nav_graph_type=hovsg_nav_graph_type,
        hovsg_direct_room_transition_max_gap_m=hovsg_direct_room_transition_max_gap_m,
        hovsg_direct_room_transition_min_span_m=hovsg_direct_room_transition_min_span_m,
        hovsg_object_approach_min_portal_stance_clearance_m=(
            hovsg_object_approach_min_portal_stance_clearance_m
        ),
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


def _resolve_scene_id(*, hovsg_scene_id: str | None, hovsg_graph_path: str | None) -> str | None:
    return runtime_backend_factory._resolve_scene_id(
        hovsg_scene_id=hovsg_scene_id,
        hovsg_graph_path=hovsg_graph_path,
    )


def resolve_hovsg_runtime_config(
    *,
    env_id: str,
    hovsg_scene_map: str | None,
    hovsg_graph_root: str | None,
    hovsg_scene_id: str | None,
    hovsg_graph_path: str | None,
    hovsg_nav_graph_type: str | None,
) -> dict[str, str | None]:
    return runtime_backend_factory.resolve_hovsg_runtime_config(
        env_id=env_id,
        hovsg_scene_map=hovsg_scene_map,
        hovsg_graph_root=hovsg_graph_root,
        hovsg_scene_id=hovsg_scene_id,
        hovsg_graph_path=hovsg_graph_path,
        hovsg_nav_graph_type=hovsg_nav_graph_type,
    )


def build_vla_selector(
    selector_mode: str,
    action_base_url: str | None,
    action_model: str | None,
    action_api_key: str | None,
    action_api_key_env: str,
    action_timeout_s: float,
    action_temperature: float,
    action_max_retries: int,
    action_retry_backoff_s: float,
):
    return runtime_backend_factory.build_vla_selector(
        selector_mode=selector_mode,
        action_base_url=action_base_url,
        action_model=action_model,
        action_api_key=action_api_key,
        action_api_key_env=action_api_key_env,
        action_timeout_s=action_timeout_s,
        action_temperature=action_temperature,
        action_max_retries=action_max_retries,
        action_retry_backoff_s=action_retry_backoff_s,
    )


def build_vla_deliberator(
    selector_mode: str,
    action_base_url: str | None,
    action_model: str | None,
    action_api_key: str | None,
    action_api_key_env: str,
    action_timeout_s: float,
    action_temperature: float,
    action_max_retries: int,
    action_retry_backoff_s: float,
):
    return runtime_backend_factory.build_vla_deliberator(
        selector_mode=selector_mode,
        action_base_url=action_base_url,
        action_model=action_model,
        action_api_key=action_api_key,
        action_api_key_env=action_api_key_env,
        action_timeout_s=action_timeout_s,
        action_temperature=action_temperature,
        action_max_retries=action_max_retries,
        action_retry_backoff_s=action_retry_backoff_s,
    )


def build_parser() -> argparse.ArgumentParser:
    return entrypoint_parser.build_closed_loop_parser()


def _record_orchestrator_event_to_environment(environment: Any, event: VisioMindActionEvent) -> None:
    recorder = getattr(environment, "record_orchestrator_event", None)
    if not callable(recorder):
        return
    recorder(
        {
            "event": f"orchestrator_{event.event_type}",
            "payload": {
                "source": event.source,
                "message": event.message,
                "task_id": event.task_id,
                **dict(event.payload),
            },
        }
    )


def configure_brain_interactive_planning(
    orchestrator: Any,
    *,
    ask_when_uncertain: bool,
    max_questions: int,
    reuse_memory_criteria_min_confidence: float,
) -> None:
    skill = getattr(
        getattr(getattr(orchestrator, "brain_agent", None), "interactive_planning", None),
        "skill",
        None,
    )
    if skill is None:
        return
    skill.ask_when_uncertain = bool(ask_when_uncertain)
    skill.max_questions = max(0, int(max_questions))
    skill.reuse_memory_criteria_min_confidence = float(reuse_memory_criteria_min_confidence)


def run_closed_loop_task(
    *,
    orchestrator: Any,
    request: Any,
    environment: Any,
    interactive_planning_enabled: bool,
    require_user_confirmation: bool = True,
    input_fn: Any = input,
    output_fn: Any = print,
) -> dict[str, Any]:
    if not interactive_planning_enabled:
        return orchestrator.run_task(request=request, environment=environment)

    try:
        session = orchestrator.brain_agent.begin_interactive_prepare(request)
    except Exception as exc:
        raise RuntimeError(f"Brain could not prepare a detailed interactive plan: {exc}") from exc
    _emit_brain_interactive_event(
        orchestrator,
        event_type="brain_text_plan_draft",
        message="drafted interactive text plan",
        session=session,
    )
    _print_text_plan(output_fn, session.draft.to_dict())

    for question in _unanswered_questions(session):
        _emit_brain_interactive_event(
            orchestrator,
            event_type="brain_question",
            message=str(question.get("text") or "planning clarification requested"),
            session=session,
            payload={"question": dict(question)},
        )
        answer_text = input_fn(_format_question_prompt(output_fn, question)).strip()
        session = orchestrator.brain_agent.answer_planning_question(
            session,
            UserAnswer(
                question_id=str(question.get("question_id") or ""),
                answer=answer_text,
            ),
        )
        _emit_brain_interactive_event(
            orchestrator,
            event_type="brain_user_answer_recorded",
            message="recorded user planning clarification",
            session=session,
            payload={
                "question_id": str(question.get("question_id") or ""),
                "answer": answer_text,
            },
        )
        _emit_brain_interactive_event(
            orchestrator,
            event_type="brain_text_plan_revised",
            message="revised interactive text plan",
            session=session,
        )

    _emit_brain_interactive_event(
        orchestrator,
        event_type="brain_plan_confirmation_requested",
        message="interactive text plan is ready for confirmation",
        session=session,
    )
    _print_text_plan(output_fn, session.draft.to_dict())

    confirmed = True
    user_message: str | None = None
    if require_user_confirmation:
        response = input_fn("Confirm Brain plan and start execution? [y/N]: ").strip()
        confirmed = response.lower() in {"y", "yes"}
        user_message = response
    try:
        context, plan = orchestrator.brain_agent.confirm_interactive_plan_with_context(
            session,
            PlanConfirmation(confirmed=confirmed, user_message=user_message),
        )
    except Exception as exc:
        if confirmed:
            raise
        _emit_brain_interactive_event(
            orchestrator,
            event_type="brain_plan_rejected",
            message="interactive plan rejected",
            session=session,
            payload={"confirmed": False, "user_message": user_message},
        )
        raise RuntimeError("Interactive Brain plan was rejected by the user") from exc
    _emit_brain_interactive_event(
        orchestrator,
        event_type="brain_plan_confirmed" if confirmed else "brain_plan_rejected",
        message="interactive plan confirmed" if confirmed else "interactive plan rejected",
        session=session,
        payload={"confirmed": bool(confirmed), "user_message": user_message},
    )
    if not confirmed:
        raise RuntimeError("Interactive Brain plan was rejected by the user")

    return orchestrator.run_prepared_task(
        request=request,
        environment=environment,
        context=context,
        plan=plan,
        initial_plan_reason="interactive_plan_confirmed",
    )


def _emit_brain_interactive_event(
    orchestrator: Any,
    *,
    event_type: str,
    message: str,
    session: Any,
    payload: dict[str, Any] | None = None,
) -> None:
    sink = getattr(orchestrator, "event_sink", None)
    if not callable(sink):
        return
    text_plan = session.draft.to_dict()
    event_payload = {
        "session_id": session.session_id,
        "status": session.status,
        "text_plan": text_plan,
    }
    if payload:
        event_payload.update(payload)
    sink(
        VisioMindActionEvent(
            event_type=event_type,
            source="BRAIN",
            message=message,
            payload=event_payload,
            task_id=session.task_id,
        )
    )


def _unanswered_questions(session: Any) -> list[dict[str, Any]]:
    answered = {
        str(item.get("question_id"))
        for item in session.dialogue
        if item.get("type") == "answer" and item.get("question_id")
    }
    return [
        dict(item)
        for item in session.dialogue
        if item.get("type") == "question" and str(item.get("question_id") or "") not in answered
    ]


def _format_question_prompt(output_fn: Any, question: dict[str, Any]) -> str:
    question_id = str(question.get("question_id") or "question")
    output_fn(f"Brain question [{question_id}]: {question.get('text') or ''}")
    reason = str(question.get("reason") or "").strip()
    if reason:
        output_fn(f"Reason: {reason}")
    options = [str(item) for item in question.get("options") or [] if str(item).strip()]
    if options:
        output_fn("Options: " + " | ".join(options))
    return f"Answer [{question_id}]: "


def _print_text_plan(output_fn: Any, text_plan: dict[str, Any]) -> None:
    output_fn("Brain text plan:")
    summary = str(text_plan.get("task_summary") or "").strip()
    if summary:
        output_fn(f"Summary: {summary}")
    for index, step in enumerate(text_plan.get("steps") or [], start=1):
        description = step.get("description") or step.get("action") or step
        role = str(step.get("role") or "milestone").strip().lower()
        condition = str(step.get("condition") or "").strip()
        if role == "contingency":
            suffix = f" — {condition}" if condition else ""
            output_fn(f"Optional step {index}: {description}{suffix}")
        else:
            output_fn(f"Step {index}: {description}")
    criteria = text_plan.get("success_criteria") or []
    if criteria:
        output_fn("Success criteria:")
        for item in criteria:
            output_fn(f"- {item.get('description') or item}")


def main() -> None:
    install_console_noise_filters()
    parser = build_parser()
    args = parse_args_with_config(parser)

    hovsg_runtime = resolve_hovsg_runtime_config(
        env_id=args.env_id,
        hovsg_scene_map=args.hovsg_scene_map,
        hovsg_graph_root=args.hovsg_graph_root,
        hovsg_scene_id=args.hovsg_scene_id,
        hovsg_graph_path=args.hovsg_graph_path,
        hovsg_nav_graph_type=args.hovsg_nav_graph_type,
    )

    args.pi05_task_id = _normalize_pi05_task_id(
        policy_backend=args.policy_backend,
        env_id=args.env_id,
        pi05_task_id=args.pi05_task_id,
    )
    args.openpi_comet_task_id = _normalize_policy_task_id(
        policy_backend=args.policy_backend,
        env_id=args.env_id,
        task_id=args.openpi_comet_task_id,
    )

    orchestrator = build_orchestrator(
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
        memory_experience_extraction_min_confidence_to_write=args.memory_experience_extraction_min_confidence_to_write,
        memory_experience_extraction_min_confidence_to_promote=(
            args.memory_experience_extraction_min_confidence_to_promote
        ),
        max_retries=args.max_retries,
        max_control_steps_per_subtask=args.max_control_steps,
        action_verify_every_control_steps=args.action_verify_every_control_steps,
        action_max_unverified_internal_step_control_steps=args.action_max_unverified_internal_step_control_steps,
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
        hovsg_object_approach_min_portal_stance_clearance_m=(
            args.hovsg_object_approach_min_portal_stance_clearance_m
        ),
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
        runtime_termination_use_environment_success_signal=args.runtime_termination_use_environment_success_signal,
        runtime_termination_use_brain_completion_signal=args.runtime_termination_use_brain_completion_signal,
        runtime_termination_environment_signal_policy=args.runtime_termination_environment_signal_policy,
        vision_completion_enabled=args.vision_completion_enabled,
        vision_completion_positive_streak=args.vision_completion_positive_streak,
        vision_completion_stability_steps=args.vision_completion_stability_steps,
        vision_completion_min_confidence=args.vision_completion_min_confidence,
        vision_completion_action_delta_threshold=args.vision_completion_action_delta_threshold,
        vision_completion_check_interval_steps=args.vision_completion_check_interval_steps,
        vision_completion_agent_scope=args.vision_completion_agent_scope,
        vision_completion_use_memory_guidance=args.vision_completion_use_memory_guidance,
        vision_completion_include_third_person=args.vision_completion_include_third_person,
        vision_completion_max_images=args.vision_completion_max_images,
        vision_completion_max_image_side_px=args.vision_completion_max_image_side_px,
        vision_completion_jpeg_quality=args.vision_completion_jpeg_quality,
        vision_completion_max_image_b64_chars=args.vision_completion_max_image_b64_chars,
        vision_completion_image_detail=args.vision_completion_image_detail,
        logging_verbose=args.logging_verbose,
        anygrasp_config=getattr(args, "anygrasp_config", None),
    )
    configure_brain_interactive_planning(
        orchestrator,
        ask_when_uncertain=args.brain_interactive_planning_ask_when_uncertain,
        max_questions=args.brain_interactive_planning_max_questions,
        reuse_memory_criteria_min_confidence=(
            args.brain_interactive_planning_reuse_memory_criteria_min_confidence
        ),
    )

    scene_id = hovsg_runtime["scene_id"]
    environment = runtime_runtime_builder.build_behavior_environment(
        args=args, hovsg_runtime=hovsg_runtime
    )
    orchestrator.event_sink = lambda event: _record_orchestrator_event_to_environment(
        environment, event
    )
    request = runtime_runtime_builder.build_task_request(
        args=args,
        scene_id=scene_id,
        hovsg_runtime=hovsg_runtime,
    )

    result = run_closed_loop_task(
        orchestrator=orchestrator,
        request=request,
        environment=environment,
        interactive_planning_enabled=args.brain_interactive_planning_enabled,
        require_user_confirmation=args.brain_interactive_planning_require_user_confirmation,
    )
    print(format_console_result(result))


if __name__ == "__main__":
    main()
