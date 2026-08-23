"""Lightweight HOV-SG navigator adapter for Voltron VLN integration.

This adapter intentionally avoids importing the full HOV-SG runtime stack.
It only reads exported graph assets from disk:

- ``graph/floors/*.json``
- ``graph/rooms/*.json``
- ``graph/objects/*.json``
- ``graph/nav_graph/*.json`` for room graphs or Voronoi graphs
"""

from __future__ import annotations
from pathlib import Path
from typing import Any

import networkx as nx

from . import backend_state as hovsg_backend_state
from . import explicit_portals as hovsg_explicit_portals
from . import object_approach as hovsg_object_approach
from . import goal_resolution as hovsg_goal_resolution
from . import planning_flow as hovsg_planning_flow
from . import pose_localization as hovsg_pose_localization
from . import portal_candidates as hovsg_portal_candidates
from . import portal_primitives as hovsg_portal_primitives
from . import portal_refinement as hovsg_portal_refinement
from . import portal_waypoints as hovsg_portal_waypoints
from . import public_flow as hovsg_public_flow
from . import scene_loading as hovsg_scene_loading
from . import scene_runtime as hovsg_scene_runtime
from . import waypoint_planning as hovsg_waypoint_planning
from .models import (
    HOVSGFloorAsset,
    HOVSGNavGraphAsset,
    HOVSGObjectAsset,
    HOVSGRoomAsset,
    HOVSGRoomLocalization,
    HOVSGSceneAsset,
)


class HOVSGNavigatorAdapter:
    """Goal grounding and path planning over exported HOV-SG assets."""

    def __init__(
        self,
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
    ) -> None:
        hovsg_backend_state.initialize_adapter(
            self,
            graph_root=graph_root,
            scene_roots=scene_roots,
            default_scene_id=default_scene_id,
            auto_load=auto_load,
            nav_graph_type=nav_graph_type,
            vertical_axis=vertical_axis,
            portal_target_offset=portal_target_offset,
            direct_room_transition_max_gap=direct_room_transition_max_gap,
            direct_room_transition_min_span=direct_room_transition_min_span,
            portal_analysis_map_resolution=portal_analysis_map_resolution,
            portal_sampling_step_m=portal_sampling_step_m,
            portal_opening_probe_offset_m=portal_opening_probe_offset_m,
            portal_opening_clearance_radius_m=portal_opening_clearance_radius_m,
            room_boundary_tolerance=room_boundary_tolerance,
            room_hysteresis_margin=room_hysteresis_margin,
            object_approach_min_distance_m=object_approach_min_distance_m,
            object_approach_preferred_distance_m=object_approach_preferred_distance_m,
            object_approach_max_distance_m=object_approach_max_distance_m,
            object_approach_max_candidates=object_approach_max_candidates,
            object_approach_angle_samples=object_approach_angle_samples,
            object_approach_clearance_radius_m=object_approach_clearance_radius_m,
            object_approach_max_graph_handoff_distance_m=object_approach_max_graph_handoff_distance_m,
            object_approach_min_portal_stance_clearance_m=object_approach_min_portal_stance_clearance_m,
            normalize_nav_graph_type=hovsg_scene_loading.normalize_nav_graph_type,
        )

        if auto_load and default_scene_id:
            self.load_scene(default_scene_id)

    def load_scene(self, scene_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
        return hovsg_public_flow.load_scene(self, scene_id, config=config)

    def update(
        self,
        observation: dict[str, Any],
        *,
        pose: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return hovsg_public_flow.update(
            self,
            observation,
            pose=pose,
        )

    def ground_goal(
        self,
        instruction: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return hovsg_public_flow.ground_goal(
            self,
            instruction,
            context=context,
        )

    def generate_object_approach_candidates(
        self,
        *,
        start: dict[str, Any],
        goal: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return hovsg_public_flow.generate_object_approach_candidates(
            self,
            start=start,
            goal=goal,
            context=context,
        )

    def plan_path(
        self,
        *,
        start: dict[str, Any],
        goal: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return hovsg_public_flow.plan_path(
            self,
            start=start,
            goal=goal,
            context=context,
        )

    def _build_geometric_waypoints(
        self,
        *,
        scene: HOVSGSceneAsset,
        path_nodes: list[Any],
        goal: dict[str, Any],
        goal_position: dict[str, float],
        start: dict[str, Any],
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return hovsg_waypoint_planning.build_geometric_waypoints(
            self,
            scene=scene,
            path_nodes=path_nodes,
            goal=goal,
            goal_position=goal_position,
            start=start,
            context=context,
        )

    def _build_dense_waypoints(
        self,
        *,
        scene: HOVSGSceneAsset,
        node_waypoints: list[dict[str, Any]],
        goal: dict[str, Any],
        goal_position: dict[str, float],
    ) -> list[dict[str, Any]]:
        return hovsg_waypoint_planning.build_dense_waypoints(
            self,
            scene=scene,
            node_waypoints=node_waypoints,
            goal=goal,
            goal_position=goal_position,
        )

    def _build_room_transition_waypoints(
        self,
        *,
        scene: HOVSGSceneAsset,
        room_steps: list[dict[str, Any]],
        goal: dict[str, Any],
        goal_position: dict[str, float],
        start: dict[str, Any],
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return hovsg_waypoint_planning.build_room_transition_waypoints(
            self,
            scene=scene,
            room_steps=room_steps,
            goal=goal,
            goal_position=goal_position,
            start=start,
            context=context,
        )

    @staticmethod
    def _room_steps_from_node_waypoints(node_waypoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return hovsg_waypoint_planning.room_steps_from_node_waypoints(node_waypoints)

    def _collapse_room_steps(
        self,
        *,
        scene: HOVSGSceneAsset,
        room_steps: list[dict[str, Any]],
        goal_room_id: str | None,
        start: dict[str, Any],
        goal: dict[str, Any],
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return hovsg_waypoint_planning.collapse_room_steps(
            self,
            scene=scene,
            room_steps=room_steps,
            goal_room_id=goal_room_id,
            start=start,
            goal=goal,
            context=context,
        )

    def _preferred_direct_room_step_index(
        self,
        *,
        scene: HOVSGSceneAsset,
        room_steps: list[dict[str, Any]],
        current_index: int,
        goal_room_id: str | None,
        start: dict[str, Any],
        goal: dict[str, Any],
        context: dict[str, Any],
    ) -> int | None:
        return hovsg_waypoint_planning.preferred_direct_room_step_index(
            self,
            scene=scene,
            room_steps=room_steps,
            current_index=current_index,
            goal_room_id=goal_room_id,
            start=start,
            goal=goal,
            context=context,
        )

    def _strong_room_transition_metrics(
        self,
        *,
        scene: HOVSGSceneAsset,
        source_room_id: str,
        target_room_id: str,
        start: dict[str, Any],
        goal: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        return hovsg_waypoint_planning.strong_room_transition_metrics(
            self,
            scene=scene,
            source_room_id=source_room_id,
            target_room_id=target_room_id,
            start=start,
            goal=goal,
            context=context,
        )

    def _explicit_transition_portal(
        self,
        *,
        scene: HOVSGSceneAsset,
        source_room: HOVSGRoomAsset | None,
        target_room: HOVSGRoomAsset | None,
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        return hovsg_explicit_portals.explicit_transition_portal(
            self,
            scene=scene,
            source_room=source_room,
            target_room=target_room,
            context=context,
        )

    def _rooms_are_directly_adjacent(
        self,
        scene: HOVSGSceneAsset,
        *,
        source_room_id: str,
        target_room_id: str,
    ) -> bool:
        return hovsg_waypoint_planning.rooms_are_directly_adjacent(
            scene,
            source_room_id=source_room_id,
            target_room_id=target_room_id,
            adapter=self,
        )

    def _ensure_scene(self, scene_id: str | None) -> HOVSGSceneAsset | None:
        return hovsg_scene_runtime.ensure_scene(self, scene_id)

    def _resolve_scene_id(
        self,
        *,
        observation: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> str | None:
        return hovsg_scene_runtime.resolve_scene_id(
            self,
            observation=observation,
            context=context,
        )

    def _resolve_graph_path(self, *, scene_id: str, config: dict[str, Any] | None = None) -> Path:
        return hovsg_scene_loading.resolve_graph_path(
            scene_id=scene_id,
            graph_root=self.graph_root,
            scene_roots=self.scene_roots,
            config=config,
        )

    @staticmethod
    def _load_metadata(path: Path) -> dict[str, Any]:
        return hovsg_scene_loading.load_metadata(path)

    def _resolve_vertical_axis(self, metadata: dict[str, Any], *, config: dict[str, Any] | None = None) -> str:
        return hovsg_scene_loading.resolve_vertical_axis(
            metadata,
            config=config,
            default_vertical_axis=self.vertical_axis,
        )

    def _resolve_scene_map_source(self, metadata: dict[str, Any], *, config: dict[str, Any] | None = None) -> str | None:
        return hovsg_scene_loading.resolve_scene_map_source(metadata, config=config)

    @staticmethod
    def _load_floors(path: Path) -> dict[str, HOVSGFloorAsset]:
        return hovsg_scene_loading.load_floors(path)

    @staticmethod
    def _load_rooms(path: Path) -> dict[str, HOVSGRoomAsset]:
        return hovsg_scene_loading.load_rooms(path)

    @staticmethod
    def _load_room_adjacency(
        *,
        graph_path: Path,
        rooms: dict[str, HOVSGRoomAsset],
    ) -> dict[str, set[str]] | None:
        return hovsg_scene_loading.load_room_adjacency(graph_path=graph_path, rooms=rooms)

    @staticmethod
    def _load_scene_map_payload(graph_path: Path) -> dict[str, Any] | None:
        return hovsg_scene_loading.load_scene_map_payload(graph_path)

    def _validate_scene_graph_consistency(self, scene: HOVSGSceneAsset) -> None:
        hovsg_scene_loading.validate_scene_graph_consistency(scene)

    def _validate_scene_map_graph_path(self, *, scene: HOVSGSceneAsset, payload: dict[str, Any]) -> None:
        hovsg_scene_loading.validate_scene_map_graph_path(scene=scene, payload=payload)

    def _validate_scene_map_nav_graph_topology(self, *, scene: HOVSGSceneAsset) -> None:
        hovsg_scene_loading.validate_scene_map_nav_graph_topology(scene=scene)

    @staticmethod
    def _resolve_room_id_from_nav_node_attrs(
        attrs: dict[str, Any],
        room_id_by_name: dict[str, str],
    ) -> str | None:
        return hovsg_scene_loading.resolve_room_id_from_nav_node_attrs(attrs, room_id_by_name)

    @staticmethod
    def _format_room_edge(scene: HOVSGSceneAsset, edge: tuple[str, str]) -> str:
        return hovsg_scene_loading.format_room_edge(scene, edge)

    @staticmethod
    def _load_objects(path: Path) -> dict[str, HOVSGObjectAsset]:
        return hovsg_scene_loading.load_objects(path)

    @staticmethod
    def _load_nav_graph(
        path: Path,
        *,
        metadata: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> HOVSGNavGraphAsset:
        return hovsg_scene_loading.load_nav_graph(path, metadata=metadata, config=config)

    @staticmethod
    def _nav_graph_entries(path: Path, *, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        return hovsg_scene_loading.nav_graph_entries(path, metadata=metadata)

    @staticmethod
    def _preferred_nav_graph_type(metadata: dict[str, Any], config: dict[str, Any]) -> str | None:
        return hovsg_scene_loading.preferred_nav_graph_type(metadata, config)

    @staticmethod
    def _normalize_nav_graph_type(value: Any) -> str | None:
        return hovsg_scene_loading.normalize_nav_graph_type(value)

    @staticmethod
    def _infer_nav_graph_type_from_filename(filename: str) -> str:
        return hovsg_scene_loading.infer_nav_graph_type_from_filename(filename)

    def _localize_pose(
        self,
        scene: HOVSGSceneAsset,
        pose: dict[str, Any],
        *,
        previous_room_id: str | None = None,
        persist: bool = False,
    ) -> dict[str, Any]:
        return hovsg_pose_localization.localize_pose(
            self,
            scene,
            pose,
            previous_room_id=previous_room_id,
            persist=persist,
        )

    def _containing_room(self, scene: HOVSGSceneAsset, pose: dict[str, Any]) -> HOVSGRoomAsset | None:
        return hovsg_pose_localization.containing_room(self, scene, pose)

    def _localized_room(
        self,
        scene: HOVSGSceneAsset,
        pose: dict[str, Any],
        *,
        previous_room_id: str | None,
    ) -> HOVSGRoomAsset | None:
        return hovsg_pose_localization.localized_room(
            self,
            scene,
            pose,
            previous_room_id=previous_room_id,
        )

    def _select_room_localization(
        self,
        scene: HOVSGSceneAsset,
        pose: dict[str, Any],
        *,
        previous_room_id: str | None,
    ) -> HOVSGRoomLocalization | None:
        return hovsg_pose_localization.select_room_localization(
            self,
            scene,
            pose,
            previous_room_id=previous_room_id,
        )

    def _room_localizations(
        self,
        scene: HOVSGSceneAsset,
        pose: dict[str, Any],
    ) -> list[HOVSGRoomLocalization]:
        return hovsg_pose_localization.room_localizations(self, scene, pose)

    @staticmethod
    def _has_complete_pose(pose: dict[str, Any]) -> bool:
        return hovsg_pose_localization.has_complete_pose(HOVSGNavigatorAdapter, pose)

    def _room_contains_pose(self, scene: HOVSGSceneAsset, room: HOVSGRoomAsset, pose: dict[str, Any]) -> bool:
        return hovsg_pose_localization.room_contains_pose(self, scene, room, pose)

    @staticmethod
    def _horizontal_axes(scene: HOVSGSceneAsset) -> tuple[str, str]:
        return hovsg_pose_localization.horizontal_axes(scene)

    @staticmethod
    def _project_horizontal(scene: HOVSGSceneAsset, point: dict[str, Any]) -> tuple[float, float] | None:
        return hovsg_pose_localization.project_horizontal(HOVSGNavigatorAdapter, scene, point)

    @staticmethod
    def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
        return hovsg_pose_localization.point_in_polygon(point, polygon)

    @classmethod
    def _point_to_polygon_boundary_distance(
        cls,
        point: tuple[float, float],
        polygon: list[tuple[float, float]],
    ) -> float:
        return hovsg_pose_localization.point_to_polygon_boundary_distance(point, polygon)

    @staticmethod
    def _distance_point_to_segment(
        point: tuple[float, float],
        segment_start: tuple[float, float],
        segment_end: tuple[float, float],
    ) -> float:
        return hovsg_pose_localization.distance_point_to_segment(point, segment_start, segment_end)

    def _room_polygon_area(self, scene: HOVSGSceneAsset, room: HOVSGRoomAsset) -> float | None:
        return hovsg_pose_localization.room_polygon_area(self, scene, room)

    @staticmethod
    def _polygon_area(polygon: list[tuple[float, float]]) -> float:
        return hovsg_pose_localization.polygon_area(polygon)

    @staticmethod
    def _centroid_distance_sq(centroid: dict[str, float] | None, pose: dict[str, Any]) -> float:
        return hovsg_pose_localization.centroid_distance_sq(HOVSGNavigatorAdapter, centroid, pose)

    def _match_room(self, scene: HOVSGSceneAsset | None, text: str | None) -> HOVSGRoomAsset | None:
        return hovsg_goal_resolution.match_room(self, scene, text)

    def _match_object(
        self,
        scene: HOVSGSceneAsset | None,
        text: str | None,
        room_id: str | None = None,
    ) -> HOVSGObjectAsset | None:
        return hovsg_goal_resolution.match_object(self, scene, text, room_id=room_id)

    def _match_floor(self, scene: HOVSGSceneAsset | None, text: str | None) -> HOVSGFloorAsset | None:
        return hovsg_goal_resolution.match_floor(self, scene, text)

    def _goal_position(self, scene: HOVSGSceneAsset, goal: dict[str, Any]) -> dict[str, float] | None:
        return hovsg_goal_resolution.goal_position(self, scene, goal)

    def _goal_waypoint(
        self,
        *,
        scene: HOVSGSceneAsset,
        goal: dict[str, Any],
        goal_position: dict[str, float],
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        return hovsg_goal_resolution.goal_waypoint(
            self,
            scene=scene,
            goal=goal,
            goal_position=goal_position,
            fallback=fallback,
        )

    def _resolve_goal_position_and_node(
        self,
        *,
        scene: HOVSGSceneAsset,
        goal: dict[str, Any],
        start_node: Any,
        start: dict[str, Any],
        context: dict[str, Any],
        fallback_goal_position: dict[str, float],
    ) -> tuple[dict[str, Any], Any | None, list[dict[str, Any]], dict[str, Any] | None]:
        return hovsg_goal_resolution.resolve_goal_position_and_node(
            self,
            scene=scene,
            goal=goal,
            start_node=start_node,
            start=start,
            context=context,
            fallback_goal_position=fallback_goal_position,
        )

    def _build_object_approach_candidates(
        self,
        *,
        scene: HOVSGSceneAsset,
        goal: dict[str, Any],
        start: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return hovsg_object_approach.build_object_approach_candidates(
            self,
            scene=scene,
            goal=goal,
            start=start,
            context=context,
        )

    def _build_continuous_object_approach_candidates(
        self,
        *,
        scene: HOVSGSceneAsset,
        goal: dict[str, Any],
        start: dict[str, Any] | None,
        object_xy: tuple[float, float],
        object_position: dict[str, float],
        object_floor_id: str | None,
        object_room_id: str | None,
        object_name: str | None,
        object_polygon: list[tuple[float, float]],
        room: HOVSGRoomAsset | None,
        room_polygon: list[tuple[float, float]],
        context: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        return hovsg_object_approach.build_continuous_object_approach_candidates(
            self,
            scene=scene,
            goal=goal,
            start=start,
            object_xy=object_xy,
            object_position=object_position,
            object_floor_id=object_floor_id,
            object_room_id=object_room_id,
            object_name=object_name,
            object_polygon=object_polygon,
            room=room,
            room_polygon=room_polygon,
            context=context,
        )

    @staticmethod
    def _merge_object_approach_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return hovsg_object_approach.merge_object_approach_candidates(candidates)

    def _score_object_approach_candidates(
        self,
        *,
        scene: HOVSGSceneAsset,
        start_node: Any,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return hovsg_object_approach.score_object_approach_candidates(
            self,
            scene=scene,
            start_node=start_node,
            candidates=candidates,
        )

    def _select_object_approach_candidate(
        self,
        *,
        scene: HOVSGSceneAsset,
        start_node: Any,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        return hovsg_object_approach.select_object_approach_candidate(candidates=candidates)

    def _object_polygon_2d(self, scene: HOVSGSceneAsset, obj: HOVSGObjectAsset) -> list[tuple[float, float]]:
        return hovsg_object_approach.object_polygon_2d(self, scene, obj)

    def _infer_floor_id(self, scene: HOVSGSceneAsset, anchor: dict[str, Any]) -> str | None:
        return hovsg_goal_resolution.infer_floor_id(self, scene, anchor)

    def _refine_start_anchor(
        self,
        *,
        scene: HOVSGSceneAsset,
        start: dict[str, Any],
    ) -> dict[str, Any]:
        return hovsg_goal_resolution.refine_start_anchor(self, scene=scene, start=start)

    def _infer_room_id(self, scene: HOVSGSceneAsset, anchor: dict[str, Any]) -> str | None:
        return hovsg_goal_resolution.infer_room_id(self, scene, anchor)

    def _infer_floor_from_height(self, scene: HOVSGSceneAsset, pose: dict[str, Any]) -> str | None:
        return hovsg_pose_localization.infer_floor_from_height(self, scene, pose)

    @staticmethod
    def _vertical_axis_value(scene: HOVSGSceneAsset, pose: dict[str, Any]) -> float | None:
        return hovsg_pose_localization.vertical_axis_value(HOVSGNavigatorAdapter, scene, pose)

    @staticmethod
    def _nearest_nav_node(
        graph: nx.Graph,
        position: dict[str, Any],
        floor_id: str | None = None,
        room_id: str | None = None,
    ) -> Any | None:
        return hovsg_goal_resolution.nearest_nav_node(
            HOVSGNavigatorAdapter,
            graph,
            position,
            floor_id=floor_id,
            room_id=room_id,
        )

    @staticmethod
    def _node_to_waypoint(graph: nx.Graph, node_id: Any) -> dict[str, Any]:
        return hovsg_goal_resolution.node_to_waypoint(HOVSGNavigatorAdapter, graph, node_id)

    def _transition_waypoint(
        self,
        *,
        scene: HOVSGSceneAsset,
        source_room_id: str,
        target_room_id: str,
        fallback_from: dict[str, Any],
        fallback_to: dict[str, Any],
        start: dict[str, Any],
        goal: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        return hovsg_portal_waypoints.transition_waypoint(
            self,
            scene=scene,
            source_room_id=source_room_id,
            target_room_id=target_room_id,
            fallback_from=fallback_from,
            fallback_to=fallback_to,
            start=start,
            goal=goal,
            context=context,
        )

    def _edge_transition_target_entry(
        self,
        *,
        scene: HOVSGSceneAsset,
        source_room: HOVSGRoomAsset | None,
        target_room: HOVSGRoomAsset | None,
        fallback_from: dict[str, Any],
        fallback_to: dict[str, Any],
    ) -> dict[str, float] | None:
        return hovsg_portal_waypoints.edge_transition_target_entry(
            self,
            scene,
            source_room=source_room,
            target_room=target_room,
            fallback_from=fallback_from,
            fallback_to=fallback_to,
        )

    def _room_transition_target_entry(
        self,
        scene: HOVSGSceneAsset,
        source_room: HOVSGRoomAsset | None,
        target_room: HOVSGRoomAsset | None,
        *,
        start: dict[str, Any],
        goal: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, float] | None:
        return hovsg_portal_waypoints.room_transition_target_entry(
            self,
            scene,
            source_room,
            target_room,
            start=start,
            goal=goal,
            context=context,
        )

    def _room_transition_center(
        self,
        scene: HOVSGSceneAsset,
        source_room: HOVSGRoomAsset | None,
        target_room: HOVSGRoomAsset | None,
        *,
        start: dict[str, Any],
        goal: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, float] | None:
        return hovsg_portal_waypoints.room_transition_center(
            self,
            scene,
            source_room,
            target_room,
            start=start,
            goal=goal,
            context=context,
        )

    def _room_transition_points(
        self,
        scene: HOVSGSceneAsset,
        source_room: HOVSGRoomAsset | None,
        target_room: HOVSGRoomAsset | None,
        *,
        start: dict[str, Any],
        goal: dict[str, Any],
        context: dict[str, Any],
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        return hovsg_portal_waypoints.room_transition_points(
            self,
            scene,
            source_room,
            target_room,
            start=start,
            goal=goal,
            context=context,
        )

    @staticmethod
    def _transition_points_from_bboxes(
        source_polygon: list[tuple[float, float]],
        target_polygon: list[tuple[float, float]],
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        return hovsg_portal_candidates.transition_points_from_bboxes(source_polygon, target_polygon)

    @staticmethod
    def _transition_metrics_from_bboxes(
        source_polygon: list[tuple[float, float]],
        target_polygon: list[tuple[float, float]],
    ) -> dict[str, Any] | None:
        return hovsg_portal_candidates.transition_metrics_from_bboxes(source_polygon, target_polygon)

    def _room_transition_metrics(
        self,
        scene: HOVSGSceneAsset,
        source_room: HOVSGRoomAsset | None,
        target_room: HOVSGRoomAsset | None,
        *,
        start: dict[str, Any],
        goal: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        return hovsg_portal_refinement.room_transition_metrics(
            self,
            scene,
            source_room,
            target_room,
            start=start,
            goal=goal,
            context=context,
        )

    def _portal_waypoint_metadata(
        self,
        *,
        scene: HOVSGSceneAsset,
        source_room: HOVSGRoomAsset | None,
        target_room: HOVSGRoomAsset | None,
        start: dict[str, Any],
        goal: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        return hovsg_portal_waypoints.portal_waypoint_metadata(
            self,
            scene=scene,
            source_room=source_room,
            target_room=target_room,
            start=start,
            goal=goal,
            context=context,
        )

    def _load_portal_analysis_map(
        self,
        *,
        scene_id: str,
        start: dict[str, Any],
        goal: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        return hovsg_portal_refinement.load_portal_analysis_map(
            self,
            scene_id=scene_id,
            start=start,
            goal=goal,
            context=context,
        )

    @staticmethod
    def _resolve_traversability_map_filename_from_context(
        *,
        start: dict[str, Any],
        goal: dict[str, Any],
        context: dict[str, Any],
    ) -> str | None:
        return hovsg_portal_refinement.resolve_traversability_map_filename_from_context(
            start=start,
            goal=goal,
            context=context,
        )

    def _refine_transition_metrics_with_traversability(
        self,
        *,
        scene: HOVSGSceneAsset,
        metrics: dict[str, Any],
        map_spec: dict[str, Any],
    ) -> dict[str, Any] | None:
        return hovsg_portal_refinement.refine_transition_metrics_with_traversability(
            self,
            scene=scene,
            metrics=metrics,
            map_spec=map_spec,
        )

    def _search_transition_metrics_from_segment_pairs(
        self,
        *,
        scene: HOVSGSceneAsset,
        source_polygon: list[tuple[float, float]],
        target_polygon: list[tuple[float, float]],
        map_spec: dict[str, Any],
    ) -> dict[str, Any] | None:
        return hovsg_portal_refinement.search_transition_metrics_from_segment_pairs(
            self,
            scene=scene,
            source_polygon=source_polygon,
            target_polygon=target_polygon,
            map_spec=map_spec,
        )

    def _segment_pair_transition_candidates(
        self,
        *,
        source_start: tuple[float, float],
        source_end: tuple[float, float],
        target_start: tuple[float, float],
        target_end: tuple[float, float],
    ) -> list[dict[str, Any]]:
        return hovsg_portal_candidates.segment_pair_transition_candidates(
            self,
            source_start=source_start,
            source_end=source_end,
            target_start=target_start,
            target_end=target_end,
        )

    @staticmethod
    def _axis_aligned_segment_axes(
        seg_start: tuple[float, float],
        seg_end: tuple[float, float],
    ) -> tuple[int, int] | None:
        return hovsg_portal_primitives.axis_aligned_segment_axes(seg_start, seg_end)

    @staticmethod
    def _segment_point_at_axis_value(
        *,
        seg_start: tuple[float, float],
        seg_end: tuple[float, float],
        axis_index: int,
        axis_value: float,
    ) -> tuple[float, float] | None:
        return hovsg_portal_primitives.segment_point_at_axis_value(
            seg_start=seg_start,
            seg_end=seg_end,
            axis_index=axis_index,
            axis_value=axis_value,
        )

    @staticmethod
    def _portal_plane_point(
        *,
        plane_axes: tuple[str, str],
        span_axis: str,
        normal_axis: str,
        span_value: float,
        normal_value: float,
    ) -> dict[str, float]:
        return hovsg_portal_primitives.portal_plane_point(
            plane_axes=plane_axes,
            span_axis=span_axis,
            normal_axis=normal_axis,
            span_value=span_value,
            normal_value=normal_value,
        )

    def _offset_horizontal_point_along_segment_into_room(
        self,
        scene: HOVSGSceneAsset,
        *,
        boundary_point: tuple[float, float],
        toward_point: tuple[float, float],
        room: HOVSGRoomAsset | None,
        fallback_room: HOVSGRoomAsset | None,
    ) -> dict[str, float] | None:
        return hovsg_portal_waypoints.offset_horizontal_point_along_segment_into_room(
            self,
            scene,
            boundary_point=boundary_point,
            toward_point=toward_point,
            room=room,
            fallback_room=fallback_room,
        )

    def _offset_horizontal_point_into_room(
        self,
        scene: HOVSGSceneAsset,
        *,
        boundary_point: tuple[float, float],
        room: HOVSGRoomAsset | None,
        fallback_room: HOVSGRoomAsset | None,
    ) -> dict[str, float] | None:
        return hovsg_portal_waypoints.offset_horizontal_point_into_room(
            self,
            scene,
            boundary_point=boundary_point,
            room=room,
            fallback_room=fallback_room,
        )

    def _transition_center_from_bboxes(
        self,
        scene: HOVSGSceneAsset,
        source_polygon: list[tuple[float, float]],
        target_polygon: list[tuple[float, float]],
        *,
        source_room: HOVSGRoomAsset,
        target_room: HOVSGRoomAsset,
    ) -> dict[str, float] | None:
        return hovsg_portal_waypoints.transition_center_from_bboxes(
            self,
            scene,
            source_polygon,
            target_polygon,
            source_room=source_room,
            target_room=target_room,
        )

    def _room_polygon_2d(self, scene: HOVSGSceneAsset, room: HOVSGRoomAsset) -> list[tuple[float, float]]:
        return hovsg_portal_primitives.room_polygon_2d(self, scene, room)

    @staticmethod
    def _polygon_segments(polygon: list[tuple[float, float]]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        return hovsg_portal_primitives.polygon_segments(polygon)

    def _segment_entry_point_into_polygon(
        self,
        *,
        start_point: tuple[float, float],
        end_point: tuple[float, float],
        polygon: list[tuple[float, float]],
    ) -> tuple[float, float] | None:
        return hovsg_portal_primitives.segment_entry_point_into_polygon(
            self,
            start_point=start_point,
            end_point=end_point,
            polygon=polygon,
        )

    @staticmethod
    def _segment_intersection(
        a0: tuple[float, float],
        a1: tuple[float, float],
        b0: tuple[float, float],
        b1: tuple[float, float],
    ) -> tuple[tuple[float, float], float] | None:
        return hovsg_portal_primitives.segment_intersection(a0, a1, b0, b1)

    @staticmethod
    def _closest_points_between_segments(
        a0: tuple[float, float],
        a1: tuple[float, float],
        b0: tuple[float, float],
        b1: tuple[float, float],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        return hovsg_portal_primitives.closest_points_between_segments(a0, a1, b0, b1)

    @staticmethod
    def _closest_points_on_axis_aligned_segments(
        a0: tuple[float, float],
        a1: tuple[float, float],
        b0: tuple[float, float],
        b1: tuple[float, float],
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        return hovsg_portal_primitives.closest_points_on_axis_aligned_segments(a0, a1, b0, b1)

    @staticmethod
    def _closest_point_on_segment(
        point: tuple[float, float],
        seg_start: tuple[float, float],
        seg_end: tuple[float, float],
    ) -> tuple[float, float]:
        return hovsg_portal_primitives.closest_point_on_segment(point, seg_start, seg_end)

    def _lift_horizontal_point(
        self,
        scene: HOVSGSceneAsset,
        horizontal_point: tuple[float, float],
        *,
        source_room: HOVSGRoomAsset | None,
        target_room: HOVSGRoomAsset | None,
    ) -> dict[str, float] | None:
        return hovsg_portal_primitives.lift_horizontal_point(
            self,
            scene,
            horizontal_point,
            source_room=source_room,
            target_room=target_room,
        )

    @staticmethod
    def _midpoint_from_positions(
        first: dict[str, Any] | None,
        second: dict[str, Any] | None,
    ) -> dict[str, float] | None:
        return hovsg_portal_primitives.midpoint_from_positions(
            adapter=HOVSGNavigatorAdapter,
            first=first,
            second=second,
        )

    @staticmethod
    def _append_waypoint_if_distinct(
        waypoints: list[dict[str, Any]],
        waypoint: dict[str, Any],
        *,
        epsilon: float = 0.05,
    ) -> None:
        hovsg_portal_primitives.append_waypoint_if_distinct(
            waypoints,
            waypoint,
            epsilon=epsilon,
        )

    @staticmethod
    def _normalize_node_id(node_id: Any) -> Any:
        return hovsg_scene_loading.normalize_node_id(node_id)

    @staticmethod
    def _serialize_node_id(node_id: Any) -> Any:
        if isinstance(node_id, tuple):
            return list(node_id)
        return node_id

    @staticmethod
    def _centroid_from_vertices(vertices: Any) -> dict[str, float] | None:
        return hovsg_scene_loading.centroid_from_vertices(vertices)

    @staticmethod
    def _position_from_metadata(metadata: dict[str, Any]) -> dict[str, float] | None:
        return hovsg_scene_loading.position_from_metadata(metadata)

    @staticmethod
    def _first_non_empty(mapping: dict[str, Any], *keys: str) -> str | None:
        return hovsg_scene_runtime.first_non_empty(mapping, *keys)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return hovsg_scene_loading.read_json(path)

    @staticmethod
    def _to_float(value: Any) -> float | None:
        return hovsg_scene_loading.to_float(value)

    @staticmethod
    def _normalize_text(value: str) -> str:
        return hovsg_scene_runtime.normalize_text(value)

    @staticmethod
    def _compact_text(value: str) -> str:
        return hovsg_scene_runtime.compact_text(value)

    @staticmethod
    def _score_text_match(candidate: str, query: str) -> float:
        return hovsg_scene_runtime.score_text_match(HOVSGNavigatorAdapter, candidate, query)

    @staticmethod
    def _path_cost(graph: nx.Graph, path_nodes: list[Any]) -> float:
        return hovsg_planning_flow.path_cost(graph, path_nodes)

    @staticmethod
    def _room_sequence(waypoints: list[dict[str, Any]]) -> list[str]:
        return hovsg_planning_flow.room_sequence(waypoints)

    def _graph_room_sequence(
        self,
        *,
        scene: HOVSGSceneAsset,
        waypoints: list[dict[str, Any]],
    ) -> list[str]:
        return hovsg_planning_flow.graph_room_sequence(
            self,
            scene=scene,
            waypoints=waypoints,
        )
