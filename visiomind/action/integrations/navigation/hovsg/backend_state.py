from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from . import scene_loading as hovsg_scene_loading


def normalize_scene_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def build_backend_state(
    *,
    graph_root: str | Path | None = None,
    scene_roots: dict[str, str | Path] | None = None,
    default_scene_id: str | None = None,
    auto_load: bool = False,
    nav_graph_type: str | None = None,
    vertical_axis: str = "y",
    portal_target_offset: float = 0.35,
    direct_room_transition_max_gap: float = 0.25,
    direct_room_transition_min_span: float = 1.0,
    portal_analysis_map_resolution: float = 0.05,
    portal_sampling_step_m: float = 0.05,
    portal_opening_probe_offset_m: float = 0.12,
    portal_opening_clearance_radius_m: float = 0.0,
    room_boundary_tolerance: float = 0.03,
    room_hysteresis_margin: float = 0.08,
    object_approach_min_distance_m: float = 0.65,
    object_approach_preferred_distance_m: float = 1.0,
    object_approach_max_distance_m: float = 2.0,
    object_approach_max_candidates: int = 8,
    object_approach_angle_samples: int = 16,
    object_approach_clearance_radius_m: float = 0.0,
    object_approach_max_graph_handoff_distance_m: float = 1.0,
    object_approach_min_portal_stance_clearance_m: float = 0.45,
    normalize_nav_graph_type: Callable[[Any], str | None] | None = None,
) -> dict[str, Any]:
    normalize_nav_graph_type = (
        normalize_nav_graph_type or hovsg_scene_loading.normalize_nav_graph_type
    )
    normalized_room_boundary_tolerance = max(0.0, float(room_boundary_tolerance))
    normalized_object_approach_min_distance = max(0.1, float(object_approach_min_distance_m))
    normalized_object_approach_preferred_distance = max(
        normalized_object_approach_min_distance,
        float(object_approach_preferred_distance_m),
    )
    normalized_scene_roots = {}
    for scene_id, path in (scene_roots or {}).items():
        normalized_scene_id = normalize_scene_id(scene_id)
        if normalized_scene_id is None:
            continue
        normalized_scene_roots[normalized_scene_id] = Path(path).expanduser()

    return {
        "graph_root": Path(graph_root).expanduser() if graph_root is not None else None,
        "scene_roots": normalized_scene_roots,
        "default_scene_id": normalize_scene_id(default_scene_id),
        "auto_load": auto_load,
        "nav_graph_type": normalize_nav_graph_type(nav_graph_type),
        "vertical_axis": vertical_axis if vertical_axis in {"x", "y", "z"} else "y",
        "portal_target_offset": max(0.0, float(portal_target_offset)),
        "direct_room_transition_max_gap": max(0.0, float(direct_room_transition_max_gap)),
        "direct_room_transition_min_span": max(0.0, float(direct_room_transition_min_span)),
        "portal_analysis_map_resolution": max(0.02, float(portal_analysis_map_resolution)),
        "portal_sampling_step_m": max(0.02, float(portal_sampling_step_m)),
        "portal_opening_probe_offset_m": max(0.02, float(portal_opening_probe_offset_m)),
        "portal_opening_clearance_radius_m": max(0.0, float(portal_opening_clearance_radius_m)),
        "room_boundary_tolerance": normalized_room_boundary_tolerance,
        "room_hysteresis_margin": max(
            normalized_room_boundary_tolerance, float(room_hysteresis_margin)
        ),
        "object_approach_min_distance_m": normalized_object_approach_min_distance,
        "object_approach_preferred_distance_m": normalized_object_approach_preferred_distance,
        "object_approach_max_distance_m": max(
            normalized_object_approach_preferred_distance,
            float(object_approach_max_distance_m),
        ),
        "object_approach_max_candidates": max(1, int(object_approach_max_candidates)),
        "object_approach_angle_samples": max(8, int(object_approach_angle_samples)),
        "object_approach_clearance_radius_m": max(0.0, float(object_approach_clearance_radius_m)),
        "object_approach_max_graph_handoff_distance_m": max(
            0.0,
            float(object_approach_max_graph_handoff_distance_m),
        ),
        "object_approach_min_portal_stance_clearance_m": max(
            0.0,
            float(object_approach_min_portal_stance_clearance_m),
        ),
        "scene_cache": {},
        "last_localized_room_ids": {},
        "portal_analysis_cache": {},
        "runtime_scene_states": {},
    }


def apply_backend_state(adapter: Any, state: dict[str, Any]) -> None:
    adapter.graph_root = state["graph_root"]
    adapter.scene_roots = dict(state["scene_roots"])
    adapter.default_scene_id = state["default_scene_id"]
    adapter.auto_load = state["auto_load"]
    adapter.nav_graph_type = state["nav_graph_type"]
    adapter.vertical_axis = state["vertical_axis"]
    adapter.portal_target_offset = state["portal_target_offset"]
    adapter.direct_room_transition_max_gap = state["direct_room_transition_max_gap"]
    adapter.direct_room_transition_min_span = state["direct_room_transition_min_span"]
    adapter.portal_analysis_map_resolution = state["portal_analysis_map_resolution"]
    adapter.portal_sampling_step_m = state["portal_sampling_step_m"]
    adapter.portal_opening_probe_offset_m = state["portal_opening_probe_offset_m"]
    adapter.portal_opening_clearance_radius_m = state["portal_opening_clearance_radius_m"]
    adapter.room_boundary_tolerance = state["room_boundary_tolerance"]
    adapter.room_hysteresis_margin = state["room_hysteresis_margin"]
    adapter.object_approach_min_distance_m = state["object_approach_min_distance_m"]
    adapter.object_approach_preferred_distance_m = state["object_approach_preferred_distance_m"]
    adapter.object_approach_max_distance_m = state["object_approach_max_distance_m"]
    adapter.object_approach_max_candidates = state["object_approach_max_candidates"]
    adapter.object_approach_angle_samples = state["object_approach_angle_samples"]
    adapter.object_approach_clearance_radius_m = state["object_approach_clearance_radius_m"]
    adapter.object_approach_max_graph_handoff_distance_m = state[
        "object_approach_max_graph_handoff_distance_m"
    ]
    adapter.object_approach_min_portal_stance_clearance_m = state[
        "object_approach_min_portal_stance_clearance_m"
    ]
    adapter._scenes = state["scene_cache"]
    adapter._last_localized_room_ids = state["last_localized_room_ids"]
    adapter._portal_analysis_map_cache = state["portal_analysis_cache"]
    adapter._runtime_scene_states = state.get("runtime_scene_states", {})


def initialize_adapter(adapter: Any, **kwargs: Any) -> dict[str, Any]:
    state = build_backend_state(**kwargs)
    apply_backend_state(adapter, state)
    return state
