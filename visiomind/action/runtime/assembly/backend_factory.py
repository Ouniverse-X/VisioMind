from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from visiomind.action.agents.action.body.deliberation import (
    OpenAIActionDeliberatorConfig,
    OpenAICompatibleActionDeliberator,
)
from visiomind.action.agents.action.body.skill_selection import (
    HeuristicActionSkillSelector,
    OpenAIActionSkillSelectorConfig,
    OpenAICompatibleActionSkillSelector,
)
from visiomind.action.agents.action.body.task_planning import (
    HeuristicActionTaskPlanner,
    OpenAIActionTaskPlannerConfig,
    OpenAICompatibleActionTaskPlanner,
)
from visiomind.action.agents.navigation.body.object_approach_selection import (
    HeuristicNavigationApproachPointSelector,
    OpenAICompatibleNavigationApproachPointSelector,
    OpenAINavigationApproachPointSelectorConfig,
)
from visiomind.action.shared.geometry_frames import (
    coerce_frame_transform,
    frame_transform_for_vertical_axes,
    normalize_vertical_axis,
)
from visiomind.action.agents.navigation.body.goal_interpretation import (
    OpenAICompatibleNavigationGoalInterpreter,
    OpenAINavigationGoalInterpreterConfig,
)
from visiomind.action.agents.navigation.body.skill_routing import (
    HeuristicNavigationSkillSelector,
    OpenAICompatibleNavigationSkillSelector,
    OpenAINavigationSkillSelectorConfig,
)
from visiomind.action.agents.brain.body.planner_backend import OpenAICompatiblePlanner, OpenAIPlannerConfig
from visiomind.action.agents.brain.body.rule_based_planner import RuleBasedPlanner
from visiomind.action.agents.memory.policies.llm_experience_extractor import (
    OpenAICompatibleExperienceExtractor,
    OpenAIExperienceExtractorConfig,
)
from visiomind.action.integrations.navigation import (
    HOVSGNavigatorAdapter,
    Nav2NavigatorAdapter,
    Nav2PolicyAdapter,
    WaypointPolicyAdapter,
)
from visiomind.action.integrations.navigation.nav2.navigator import DEFAULT_NAV2_VERSION_PROFILE


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
    if planner_backend == "rule":
        return RuleBasedPlanner()

    base_url = brain_base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    model = brain_model or os.getenv("OPENAI_MODEL")
    if not base_url:
        raise ValueError("OpenAI-compatible planner requires --brain-base-url or OPENAI_BASE_URL")
    if not model:
        raise ValueError("OpenAI-compatible planner requires --brain-model or OPENAI_MODEL")

    return OpenAICompatiblePlanner(
        OpenAIPlannerConfig(
            base_url=base_url,
            model=model,
            api_key=brain_api_key,
            api_key_env=brain_api_key_env,
            timeout_s=brain_timeout_s,
            temperature=brain_temperature,
            max_retries=brain_max_retries,
            retry_backoff_s=brain_retry_backoff_s,
        )
    )


def build_memory_extractor(
    *,
    enabled: bool,
    backend: str | None,
    base_url: str | None,
    model: str | None,
    api_key: str | None,
    api_key_env: str,
    timeout_s: float,
    temperature: float,
    max_retries: int,
    retry_backoff_s: float = 1.0,
):
    if not enabled:
        return None

    backend_name = backend or ("openai_compatible" if base_url or model else None)
    if backend_name is None:
        return None
    if backend_name != "openai_compatible":
        raise ValueError(f"Unsupported MemoryAgent LLM backend '{backend_name}'")

    resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    resolved_model = model or os.getenv("OPENAI_MODEL")
    if not resolved_base_url:
        raise ValueError(
            "Memory experience extraction requires memory_llm_base_url or OPENAI_BASE_URL"
        )
    if not resolved_model:
        raise ValueError("Memory experience extraction requires memory_llm_model or OPENAI_MODEL")

    return OpenAICompatibleExperienceExtractor(
        OpenAIExperienceExtractorConfig(
            base_url=resolved_base_url,
            model=resolved_model,
            api_key=api_key,
            api_key_env=api_key_env,
            timeout_s=timeout_s,
            temperature=temperature,
            max_retries=max_retries,
            retry_backoff_s=retry_backoff_s,
        )
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
    base_url = navigation_base_url
    model = navigation_model
    if not base_url or not model:
        return HeuristicNavigationSkillSelector()
    return OpenAICompatibleNavigationSkillSelector(
        OpenAINavigationSkillSelectorConfig(
            base_url=base_url,
            model=model,
            api_key=navigation_api_key,
            api_key_env=navigation_api_key_env,
            timeout_s=navigation_timeout_s,
            temperature=navigation_temperature,
            max_retries=navigation_max_retries,
            retry_backoff_s=navigation_retry_backoff_s,
        )
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
    base_url = navigation_base_url
    model = navigation_model
    if not base_url or not model:
        return HeuristicNavigationApproachPointSelector()
    return OpenAICompatibleNavigationApproachPointSelector(
        OpenAINavigationApproachPointSelectorConfig(
            base_url=base_url,
            model=model,
            api_key=navigation_api_key,
            api_key_env=navigation_api_key_env,
            timeout_s=navigation_timeout_s,
            temperature=navigation_temperature,
            max_retries=navigation_max_retries,
            retry_backoff_s=navigation_retry_backoff_s,
        )
    )


def build_vln_goal_interpreter(
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
    base_url = navigation_base_url
    model = navigation_model
    if not base_url or not model:
        return None
    return OpenAICompatibleNavigationGoalInterpreter(
        OpenAINavigationGoalInterpreterConfig(
            base_url=base_url,
            model=model,
            api_key=navigation_api_key,
            api_key_env=navigation_api_key_env,
            timeout_s=navigation_timeout_s,
            temperature=navigation_temperature,
            max_retries=navigation_max_retries,
            retry_backoff_s=navigation_retry_backoff_s,
        )
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
    del gr00t_host, gr00t_port
    if backend != "nav2":
        raise ValueError(f"Unsupported VLN backend '{backend}'. Only 'nav2' is supported.")
    return Nav2PolicyAdapter(
        fallback_policy=WaypointPolicyAdapter(
            prefer_forward_facing_motion=prefer_forward_facing_motion,
            portal_alignment_distance_threshold=portal_alignment_distance_threshold,
            portal_prealign_distance_threshold_m=portal_prealign_distance_threshold_m,
            portal_alignment_footprint_width_m=portal_alignment_footprint_width_m,
            portal_alignment_min_lateral_deadband_m=portal_alignment_min_lateral_deadband_m,
            portal_alignment_wide_clearance_margin_m=portal_alignment_wide_clearance_margin_m,
            max_linear_velocity=max_linear_velocity,
            max_angular_velocity=max_angular_velocity,
            linear_gain=linear_gain,
            local_path_linear_gain=local_path_linear_gain,
            local_path_max_linear_velocity=local_path_max_linear_velocity,
            local_path_angular_gain_scale=local_path_angular_gain_scale,
            portal_alignment_max_linear_velocity=portal_alignment_max_linear_velocity,
            object_approach_final_waypoint_tolerance_m=object_approach_final_waypoint_tolerance_m,
        )
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
    if backend != "nav2":
        raise ValueError(f"Unsupported VLN backend '{backend}'. Only 'nav2' is supported.")

    scene_id = _resolve_scene_id(hovsg_scene_id=hovsg_scene_id, hovsg_graph_path=hovsg_graph_path)
    if scene_id is None:
        raise ValueError(
            "HOV-SG navigator requires --hovsg-scene-id or a graph path from which scene id can be inferred"
        )

    if hovsg_graph_path:
        semantic_backend = HOVSGNavigatorAdapter(
            scene_roots={scene_id: hovsg_graph_path},
            default_scene_id=scene_id,
            nav_graph_type=hovsg_nav_graph_type,
            direct_room_transition_max_gap=hovsg_direct_room_transition_max_gap_m,
            direct_room_transition_min_span=hovsg_direct_room_transition_min_span_m,
            object_approach_min_portal_stance_clearance_m=(
                hovsg_object_approach_min_portal_stance_clearance_m
            ),
            object_approach_clearance_radius_m=nav2_local_path_clearance_radius_m,
            portal_analysis_map_resolution=nav2_portal_analysis_map_resolution,
            portal_sampling_step_m=nav2_portal_sampling_step_m,
        )
    elif hovsg_graph_root:
        semantic_backend = HOVSGNavigatorAdapter(
            graph_root=hovsg_graph_root,
            default_scene_id=scene_id,
            nav_graph_type=hovsg_nav_graph_type,
            direct_room_transition_max_gap=hovsg_direct_room_transition_max_gap_m,
            direct_room_transition_min_span=hovsg_direct_room_transition_min_span_m,
            object_approach_min_portal_stance_clearance_m=(
                hovsg_object_approach_min_portal_stance_clearance_m
            ),
            object_approach_clearance_radius_m=nav2_local_path_clearance_radius_m,
            portal_analysis_map_resolution=nav2_portal_analysis_map_resolution,
            portal_sampling_step_m=nav2_portal_sampling_step_m,
        )
    else:
        raise ValueError("HOV-SG navigator requires --hovsg-graph-path or --hovsg-graph-root")

    return Nav2NavigatorAdapter(
        semantic_backend=semantic_backend,
        version_profile=nav2_version_profile,
        action_name=nav2_action_name,
        frame_id=nav2_frame_id,
        planner_id=nav2_planner_id,
        timeout_s=nav2_timeout_s,
        strict=nav2_strict,
        trav_map_filename=nav2_trav_map_filename,
        portal_analysis_map_resolution=nav2_portal_analysis_map_resolution,
        portal_clearance_radius_m=nav2_portal_clearance_radius_m,
        portal_corridor_standoff_m=nav2_portal_corridor_standoff_m,
        portal_sampling_step_m=nav2_portal_sampling_step_m,
        local_path_clearance_radius_m=nav2_local_path_clearance_radius_m,
        local_path_waypoint_spacing_m=nav2_local_path_waypoint_spacing_m,
    )


def _resolve_scene_id(*, hovsg_scene_id: str | None, hovsg_graph_path: str | None) -> str | None:
    if isinstance(hovsg_scene_id, str) and hovsg_scene_id.strip():
        return hovsg_scene_id.strip()
    if not hovsg_graph_path:
        return None

    graph_path = Path(hovsg_graph_path).expanduser()
    if graph_path.name == "graph":
        return graph_path.parent.name or None
    if (graph_path / "floors").exists():
        return graph_path.parent.name or None
    return graph_path.name or None


def resolve_hovsg_runtime_config(
    *,
    env_id: str,
    hovsg_scene_map: str | None,
    hovsg_graph_root: str | None,
    hovsg_scene_id: str | None,
    hovsg_graph_path: str | None,
    hovsg_nav_graph_type: str | None,
) -> dict[str, Any]:
    resolved = {
        "scene_id": hovsg_scene_id,
        "graph_root": hovsg_graph_root,
        "graph_path": hovsg_graph_path,
        "nav_graph_type": hovsg_nav_graph_type,
        "scene_vertical_axis": None,
        "simulator_vertical_axis": None,
        "scene_from_simulator_transform": None,
    }

    if hovsg_scene_map:
        payload = json.loads(Path(hovsg_scene_map).expanduser().read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("HOV-SG scene map must be a JSON object")
        entry = payload.get(env_id, {})
        if not isinstance(entry, dict):
            raise ValueError(f"HOV-SG scene map entry for '{env_id}' must be a JSON object")
        resolved["scene_id"] = resolved["scene_id"] or entry.get("scene_id")
        resolved["graph_root"] = resolved["graph_root"] or entry.get("graph_root")
        resolved["graph_path"] = resolved["graph_path"] or entry.get("graph_path")
        resolved["nav_graph_type"] = resolved["nav_graph_type"] or entry.get("nav_graph_type")
        resolved["scene_vertical_axis"] = entry.get("scene_vertical_axis") or entry.get(
            "vertical_axis"
        )
        resolved["simulator_vertical_axis"] = entry.get("simulator_vertical_axis")
        resolved["scene_from_simulator_transform"] = entry.get(
            "scene_from_simulator_transform"
        ) or entry.get("simulator_to_scene_transform")

    resolved["scene_id"] = _resolve_scene_id(
        hovsg_scene_id=resolved["scene_id"],
        hovsg_graph_path=resolved["graph_path"],
    )
    metadata_contract = _hovsg_frame_contract(
        graph_root=resolved["graph_root"],
        graph_path=resolved["graph_path"],
        scene_id=resolved["scene_id"],
    )
    resolved["scene_vertical_axis"] = normalize_vertical_axis(
        resolved["scene_vertical_axis"] or metadata_contract.get("scene_vertical_axis"),
        default="z",
    )
    resolved["simulator_vertical_axis"] = normalize_vertical_axis(
        resolved["simulator_vertical_axis"] or metadata_contract.get("simulator_vertical_axis"),
        default="z",
    )
    explicit_transform = coerce_frame_transform(
        resolved["scene_from_simulator_transform"]
        or metadata_contract.get("scene_from_simulator_transform")
    )
    resolved["scene_from_simulator_transform"] = explicit_transform or (
        frame_transform_for_vertical_axes(
            source_vertical_axis=resolved["simulator_vertical_axis"],
            target_vertical_axis=resolved["scene_vertical_axis"],
        )
    )
    return resolved


def _hovsg_frame_contract(
    *,
    graph_root: str | None,
    graph_path: str | None,
    scene_id: str | None,
) -> dict[str, Any]:
    candidates = []
    if graph_path:
        path = Path(graph_path).expanduser()
        candidates.extend((path, path / "graph"))
    if graph_root and scene_id:
        root = Path(graph_root).expanduser()
        candidates.extend((root / scene_id / "graph", root / scene_id))
    metadata = None
    for candidate in candidates:
        metadata_path = candidate / "metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            metadata = payload
            break
    if metadata is None:
        return {}
    coord_system = metadata.get("coord_system")
    coord_system = coord_system if isinstance(coord_system, dict) else {}
    return {
        "scene_vertical_axis": metadata.get("vertical_axis") or coord_system.get("vertical_axis"),
        "simulator_vertical_axis": metadata.get("simulator_vertical_axis")
        or coord_system.get("simulator_vertical_axis")
        or "z",
        "scene_from_simulator_transform": metadata.get("scene_from_simulator_transform")
        or metadata.get("simulator_to_scene_transform")
        or coord_system.get("scene_from_simulator_transform")
        or coord_system.get("simulator_to_scene_transform"),
    }


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
    if selector_mode == "heuristic":
        return HeuristicActionSkillSelector()

    base_url = action_base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    model = action_model or os.getenv("OPENAI_MODEL")
    if not base_url:
        raise ValueError(
            "OpenAI-compatible action selector requires --action-base-url or OPENAI_BASE_URL"
        )
    if not model:
        raise ValueError(
            "OpenAI-compatible action selector requires --action-model or OPENAI_MODEL"
        )

    return OpenAICompatibleActionSkillSelector(
        OpenAIActionSkillSelectorConfig(
            base_url=base_url,
            model=model,
            api_key=action_api_key,
            api_key_env=action_api_key_env,
            timeout_s=action_timeout_s,
            temperature=action_temperature,
            max_retries=action_max_retries,
            retry_backoff_s=action_retry_backoff_s,
        )
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
    if selector_mode == "heuristic":
        return None

    base_url = action_base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    model = action_model or os.getenv("OPENAI_MODEL")
    if not base_url:
        raise ValueError(
            "OpenAI-compatible action deliberator requires --action-base-url or OPENAI_BASE_URL"
        )
    if not model:
        raise ValueError(
            "OpenAI-compatible action deliberator requires --action-model or OPENAI_MODEL"
        )

    return OpenAICompatibleActionDeliberator(
        OpenAIActionDeliberatorConfig(
            base_url=base_url,
            model=model,
            api_key=action_api_key,
            api_key_env=action_api_key_env,
            timeout_s=action_timeout_s,
            temperature=action_temperature,
            max_retries=action_max_retries,
            retry_backoff_s=action_retry_backoff_s,
        )
    )


def build_action_task_planner(
    *,
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
    if selector_mode == "heuristic":
        return HeuristicActionTaskPlanner()

    base_url = action_base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    model = action_model or os.getenv("OPENAI_MODEL")
    if not base_url or not model:
        return HeuristicActionTaskPlanner()

    return OpenAICompatibleActionTaskPlanner(
        OpenAIActionTaskPlannerConfig(
            base_url=base_url,
            model=model,
            api_key=action_api_key,
            api_key_env=action_api_key_env,
            timeout_s=action_timeout_s,
            temperature=action_temperature,
            max_retries=action_max_retries,
            retry_backoff_s=action_retry_backoff_s,
        )
    )


__all__ = [
    "build_planner",
    "build_memory_extractor",
    "build_vln_selector",
    "build_vln_point_selector",
    "build_vln_goal_interpreter",
    "build_vln_policy",
    "build_vln_navigator",
    "_resolve_scene_id",
    "resolve_hovsg_runtime_config",
    "build_vla_selector",
    "build_vla_deliberator",
    "build_action_task_planner",
]
