from __future__ import annotations

from pathlib import Path
from typing import Any


def configure_adapter(
    adapter: Any,
    *,
    graph_root: str | Path | None,
    scene_roots: dict[str, str | Path] | None,
    default_scene_id: str | None,
    auto_load: bool,
    nav_graph_type: str | None,
    vertical_axis: str,
    portal_target_offset: float,
    direct_room_transition_max_gap: float,
    direct_room_transition_min_span: float,
    portal_analysis_map_resolution: float,
    portal_sampling_step_m: float,
    portal_opening_probe_offset_m: float,
    portal_opening_clearance_radius_m: float,
    room_boundary_tolerance: float,
    room_hysteresis_margin: float,
    object_approach_min_distance_m: float,
    object_approach_preferred_distance_m: float,
    object_approach_max_distance_m: float,
    object_approach_max_candidates: int,
    object_approach_angle_samples: int,
    object_approach_clearance_radius_m: float,
    object_approach_max_graph_handoff_distance_m: float,
    normalize_nav_graph_type: Any,
    object_approach_min_portal_stance_clearance_m: float = 0.45,
) -> None:
    adapter.graph_root = Path(graph_root).expanduser() if graph_root is not None else None
    adapter.scene_roots = {
        scene_id: Path(path).expanduser() for scene_id, path in (scene_roots or {}).items()
    }
    adapter.default_scene_id = default_scene_id
    adapter.auto_load = auto_load
    adapter.nav_graph_type = normalize_nav_graph_type(nav_graph_type)
    adapter.vertical_axis = vertical_axis if vertical_axis in {"x", "y", "z"} else "y"
    adapter.portal_target_offset = max(0.0, float(portal_target_offset))
    adapter.direct_room_transition_max_gap = max(0.0, float(direct_room_transition_max_gap))
    adapter.direct_room_transition_min_span = max(0.0, float(direct_room_transition_min_span))
    adapter.portal_analysis_map_resolution = max(0.02, float(portal_analysis_map_resolution))
    adapter.portal_sampling_step_m = max(0.02, float(portal_sampling_step_m))
    adapter.portal_opening_probe_offset_m = max(0.02, float(portal_opening_probe_offset_m))
    adapter.portal_opening_clearance_radius_m = max(0.0, float(portal_opening_clearance_radius_m))
    adapter.room_boundary_tolerance = max(0.0, float(room_boundary_tolerance))
    adapter.room_hysteresis_margin = max(
        adapter.room_boundary_tolerance, float(room_hysteresis_margin)
    )
    adapter.object_approach_min_distance_m = max(0.1, float(object_approach_min_distance_m))
    adapter.object_approach_preferred_distance_m = max(
        adapter.object_approach_min_distance_m,
        float(object_approach_preferred_distance_m),
    )
    adapter.object_approach_max_distance_m = max(
        adapter.object_approach_preferred_distance_m,
        float(object_approach_max_distance_m),
    )
    adapter.object_approach_max_candidates = max(1, int(object_approach_max_candidates))
    adapter.object_approach_angle_samples = max(8, int(object_approach_angle_samples))
    adapter.object_approach_clearance_radius_m = max(0.0, float(object_approach_clearance_radius_m))
    adapter.object_approach_max_graph_handoff_distance_m = max(
        0.0,
        float(object_approach_max_graph_handoff_distance_m),
    )
    adapter.object_approach_min_portal_stance_clearance_m = max(
        0.0,
        float(object_approach_min_portal_stance_clearance_m),
    )


def initialize_runtime_state(adapter: Any) -> None:
    adapter._scenes = {}
    adapter._last_localized_room_ids = {}
    adapter._portal_analysis_map_cache = {}
    adapter._runtime_scene_states = {}
