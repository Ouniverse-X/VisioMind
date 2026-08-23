"""Compatibility facade for HOV-SG portal helpers."""

from __future__ import annotations

from .portal_candidates import (
    segment_pair_transition_candidates,
    transition_metrics_from_bboxes,
    transition_points_from_bboxes,
)
from .portal_primitives import (
    append_waypoint_if_distinct,
    axis_aligned_segment_axes,
    closest_point_on_segment,
    closest_points_between_segments,
    closest_points_on_axis_aligned_segments,
    lift_horizontal_point,
    midpoint_from_positions,
    polygon_segments,
    portal_plane_point,
    room_polygon_2d,
    segment_entry_point_into_polygon,
    segment_intersection,
    segment_point_at_axis_value,
)
from .portal_refinement import (
    load_portal_analysis_map,
    refine_transition_metrics_with_traversability,
    resolve_traversability_map_filename_from_context,
    room_transition_metrics,
    search_transition_metrics_from_segment_pairs,
)
from .portal_waypoints import (
    edge_transition_target_entry,
    offset_horizontal_point_along_segment_into_room,
    offset_horizontal_point_into_room,
    portal_waypoint_metadata,
    room_transition_center,
    room_transition_points,
    room_transition_target_entry,
    transition_center_from_bboxes,
    transition_waypoint,
)

__all__ = [
    "append_waypoint_if_distinct",
    "axis_aligned_segment_axes",
    "closest_point_on_segment",
    "closest_points_between_segments",
    "closest_points_on_axis_aligned_segments",
    "edge_transition_target_entry",
    "lift_horizontal_point",
    "load_portal_analysis_map",
    "midpoint_from_positions",
    "offset_horizontal_point_along_segment_into_room",
    "offset_horizontal_point_into_room",
    "polygon_segments",
    "portal_plane_point",
    "portal_waypoint_metadata",
    "refine_transition_metrics_with_traversability",
    "resolve_traversability_map_filename_from_context",
    "room_polygon_2d",
    "room_transition_center",
    "room_transition_metrics",
    "room_transition_points",
    "room_transition_target_entry",
    "search_transition_metrics_from_segment_pairs",
    "segment_entry_point_into_polygon",
    "segment_intersection",
    "segment_pair_transition_candidates",
    "segment_point_at_axis_value",
    "transition_center_from_bboxes",
    "transition_metrics_from_bboxes",
    "transition_points_from_bboxes",
    "transition_waypoint",
]
