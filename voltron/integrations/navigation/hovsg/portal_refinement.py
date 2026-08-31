from __future__ import annotations

import math
from typing import Any

from .models import HOVSGRoomAsset, HOVSGSceneAsset
from .portal_candidates import (
    segment_pair_transition_candidates,
    transition_metrics_from_bboxes,
)
from .portal_primitives import (
    closest_points_between_segments,
    polygon_segments,
    portal_plane_point,
    room_polygon_2d,
    segment_point_at_axis_value,
)


def room_transition_metrics(
    adapter: Any,
    scene: HOVSGSceneAsset,
    source_room: HOVSGRoomAsset | None,
    target_room: HOVSGRoomAsset | None,
    *,
    start: dict[str, Any],
    goal: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any] | None:
    if source_room is None or target_room is None:
        return None
    source_polygon = room_polygon_2d(adapter, scene, source_room)
    target_polygon = room_polygon_2d(adapter, scene, target_room)
    if len(source_polygon) < 2 or len(target_polygon) < 2:
        return None
    metrics = transition_metrics_from_bboxes(source_polygon, target_polygon)
    explicit_portal = adapter._explicit_transition_portal(
        scene=scene,
        source_room=source_room,
        target_room=target_room,
        context=context,
    )
    map_spec = load_portal_analysis_map(
        adapter,
        scene_id=scene.scene_id,
        start=start,
        goal=goal,
        context=context,
        portal_goal=explicit_portal,
    )
    if metrics is not None:
        if map_spec is not None:
            refined_metrics = refine_transition_metrics_with_traversability(
                adapter,
                scene=scene,
                metrics=metrics,
                map_spec=map_spec,
            )
            if refined_metrics is not None:
                return refined_metrics
            segment_metrics = search_transition_metrics_from_segment_pairs(
                adapter,
                scene=scene,
                source_polygon=source_polygon,
                target_polygon=target_polygon,
                map_spec=map_spec,
            )
            if segment_metrics is not None:
                return segment_metrics
            return {
                "source_point": None,
                "target_point": None,
                "gap": float(metrics.get("gap", 0.0)),
                "span": 0.0,
                "traversability_blocked": True,
            }
        return metrics

    if map_spec is not None:
        segment_metrics = search_transition_metrics_from_segment_pairs(
            adapter,
            scene=scene,
            source_polygon=source_polygon,
            target_polygon=target_polygon,
            map_spec=map_spec,
        )
        if segment_metrics is not None:
            return segment_metrics

    best_distance = None
    for source_start, source_end in polygon_segments(source_polygon):
        for target_start, target_end in polygon_segments(target_polygon):
            source_point, target_point = closest_points_between_segments(
                source_start, source_end, target_start, target_end
            )
            distance = math.hypot(
                source_point[0] - target_point[0], source_point[1] - target_point[1]
            )
            if best_distance is None or distance < best_distance:
                best_distance = distance
    if best_distance is None:
        return None
    return {
        "source_point": None,
        "target_point": None,
        "gap": best_distance,
        "span": 0.0,
    }


def load_portal_analysis_map(
    adapter: Any,
    *,
    scene_id: str,
    start: dict[str, Any],
    goal: dict[str, Any],
    context: dict[str, Any],
    portal_goal: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    from . import door_gating as hovsg_door_gating
    from . import object_gating as hovsg_object_gating
    from . import runtime_state as hovsg_runtime_state
    from . import sensor_gating as hovsg_sensor_gating

    trav_map_filename = resolve_traversability_map_filename_from_context(
        start=start, goal=goal, context=context
    )
    base_key = (scene_id, trav_map_filename, adapter.portal_analysis_map_resolution)

    door_signature = hovsg_runtime_state.door_signature(adapter, scene_id)
    object_overlay = hovsg_object_gating.runtime_object_map_overlays(
        adapter,
        scene_id,
        navigation_goal=portal_goal,
        include_unchanged=(
            not trav_map_filename
            or "no_obj" in str(trav_map_filename).lower()
            or "no_object" in str(trav_map_filename).lower()
        ),
    )
    object_signature = str(object_overlay.get("signature") or "")
    sensor_overlay = hovsg_sensor_gating.runtime_sensor_map_overlays(adapter, scene_id)
    sensor_signature = str(sensor_overlay.get("signature") or "")
    overlay_signature = "|".join(
        part for part in (door_signature, object_signature, sensor_signature) if part
    )
    cache_key = base_key + (overlay_signature,) if overlay_signature else base_key
    cached = adapter._portal_analysis_map_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        from ..nav2.nav2_runtime_bridge import (
            clear_exported_object_artifacts_from_map_spec,
            clear_regions_from_map_spec,
            load_scene_traversability_grid,
            stamp_obstacles_into_map_spec,
        )
    except Exception:
        return None

    base_map = adapter._portal_analysis_map_cache.get(base_key)
    if base_map is None:
        try:
            base_map = load_scene_traversability_grid(
                scene_id=scene_id,
                map_resolution=adapter.portal_analysis_map_resolution,
                trav_map_filename=trav_map_filename,
            )
        except Exception:
            return None
        adapter._portal_analysis_map_cache[base_key] = base_map
    if cache_key == base_key:
        return base_map

    closed_door_obstacles = hovsg_door_gating.runtime_door_obstacles(adapter, scene_id)
    moved_object_obstacles = list(object_overlay.get("obstacles") or [])
    sensor_obstacles = list(sensor_overlay.get("obstacles") or [])
    moved_object_clear_regions = list(object_overlay.get("clear_regions") or [])
    open_door_regions = hovsg_door_gating.open_door_clear_regions(adapter, scene_id)
    fixed_portal_region = hovsg_door_gating.open_portal_clear_region(
        portal_goal,
    )
    open_door_regions = hovsg_door_gating.prefer_canonical_portal_clear_region(
        open_door_regions,
        fixed_portal_region,
    )
    map_spec = base_map
    if moved_object_clear_regions:
        map_spec = clear_exported_object_artifacts_from_map_spec(
            map_spec,
            scene_id=scene_id,
            map_resolution=adapter.portal_analysis_map_resolution,
            regions=moved_object_clear_regions,
        )
    if open_door_regions:
        map_spec = clear_regions_from_map_spec(map_spec, open_door_regions)
    obstacles = [*closed_door_obstacles, *moved_object_obstacles, *sensor_obstacles]
    if obstacles:
        map_spec = stamp_obstacles_into_map_spec(map_spec, obstacles)
    _evict_stale_stamped_entries(adapter, base_key=base_key, keep_key=cache_key)
    adapter._portal_analysis_map_cache[cache_key] = map_spec
    return map_spec


def _evict_stale_stamped_entries(adapter: Any, *, base_key: tuple, keep_key: tuple) -> None:
    stale_keys = [
        key
        for key in adapter._portal_analysis_map_cache
        if isinstance(key, tuple)
        and len(key) == len(base_key) + 1
        and key[: len(base_key)] == base_key
        and key != keep_key
    ]
    for key in stale_keys:
        adapter._portal_analysis_map_cache.pop(key, None)


def resolve_traversability_map_filename_from_context(
    *,
    start: dict[str, Any],
    goal: dict[str, Any],
    context: dict[str, Any],
) -> str | None:
    parameters = context.get("parameters", {})
    map_state = context.get("map_state", {})
    for candidate in (
        goal.get("nav2_trav_map_filename"),
        start.get("nav2_trav_map_filename"),
        context.get("nav2_trav_map_filename"),
        parameters.get("nav2_trav_map_filename") if isinstance(parameters, dict) else None,
        map_state.get("nav2_trav_map_filename") if isinstance(map_state, dict) else None,
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    scene_file = None
    for candidate in (
        goal.get("scene_file"),
        start.get("scene_file"),
        context.get("scene_file"),
        parameters.get("scene_file") if isinstance(parameters, dict) else None,
    ):
        if isinstance(candidate, str) and candidate.strip():
            scene_file = candidate.strip().lower()
            break

    if scene_file is None:
        return None
    if any(
        token in scene_file
        for token in (
            "open_door",
            "door_open",
            "doors_open",
            "all_doors_open",
            "sliding_full",
        )
    ):
        return "floor_trav_open_door_0.png"
    if "no_door" in scene_file or "doorless" in scene_file:
        return "floor_trav_no_door_0.png"
    if "no_obj" in scene_file or "no_object" in scene_file:
        return "floor_trav_no_obj_0.png"
    return None


def refine_transition_metrics_with_traversability(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    metrics: dict[str, Any],
    map_spec: dict[str, Any],
) -> dict[str, Any] | None:
    span_axis_index = metrics.get("span_axis_index")
    normal_axis_index = metrics.get("normal_axis_index")
    source_point = metrics.get("source_point")
    target_point = metrics.get("target_point")
    if not isinstance(span_axis_index, int) or not isinstance(normal_axis_index, int):
        return None
    if not isinstance(source_point, tuple) or not isinstance(target_point, tuple):
        return None

    plane_axes = adapter._horizontal_axes(scene)
    if span_axis_index >= len(plane_axes) or normal_axis_index >= len(plane_axes):
        return None
    span_axis = plane_axes[span_axis_index]
    normal_axis = plane_axes[normal_axis_index]
    if span_axis == normal_axis:
        return None

    try:
        from ..nav2.nav2_runtime_bridge import (
            point_has_clearance,
            segment_has_clearance,
        )
    except Exception:
        return None

    span_min = float(metrics.get("span_min", 0.0))
    span_max = float(metrics.get("span_max", 0.0))
    if span_max < span_min:
        span_min, span_max = span_max, span_min
    if (span_max - span_min) < adapter.portal_sampling_step_m:
        return metrics

    boundary_value = float(metrics.get("boundary_value", 0.0))
    gap = float(metrics.get("gap", 0.0))
    normal_sign = (
        1.0
        if float(target_point[normal_axis_index]) >= float(source_point[normal_axis_index])
        else -1.0
    )
    probe_offset = max(
        adapter.portal_opening_probe_offset_m,
        gap * 0.5
        + max(
            0.01,
            float(map_spec.get("resolution", adapter.portal_analysis_map_resolution)),
        ),
    )
    clearance_radius = adapter.portal_opening_clearance_radius_m
    preferred_span_value = (
        float(source_point[span_axis_index]) + float(target_point[span_axis_index])
    ) * 0.5

    valid_samples: list[float] = []
    sample_value = span_min
    while sample_value <= span_max + 1e-6:
        source_xy = portal_plane_point(
            plane_axes=plane_axes,
            span_axis=span_axis,
            normal_axis=normal_axis,
            span_value=sample_value,
            normal_value=boundary_value - normal_sign * probe_offset,
        )
        midpoint_xy = portal_plane_point(
            plane_axes=plane_axes,
            span_axis=span_axis,
            normal_axis=normal_axis,
            span_value=sample_value,
            normal_value=boundary_value,
        )
        target_xy = portal_plane_point(
            plane_axes=plane_axes,
            span_axis=span_axis,
            normal_axis=normal_axis,
            span_value=sample_value,
            normal_value=boundary_value + normal_sign * probe_offset,
        )
        if (
            point_has_clearance(
                map_spec=map_spec,
                point_xy=source_xy,
                clearance_radius_m=clearance_radius,
            )
            and point_has_clearance(
                map_spec=map_spec,
                point_xy=midpoint_xy,
                clearance_radius_m=clearance_radius,
            )
            and point_has_clearance(
                map_spec=map_spec,
                point_xy=target_xy,
                clearance_radius_m=clearance_radius,
            )
            and segment_has_clearance(
                map_spec=map_spec,
                start_xy=source_xy,
                end_xy=target_xy,
                clearance_radius_m=clearance_radius,
                step_m=adapter.portal_sampling_step_m * 0.5,
            )
        ):
            valid_samples.append(sample_value)
        sample_value += adapter.portal_sampling_step_m

    if not valid_samples:
        return None

    sampled_runs: list[tuple[float, float]] = []
    run_start = valid_samples[0]
    previous_value = valid_samples[0]
    for current_value in valid_samples[1:]:
        if (current_value - previous_value) <= adapter.portal_sampling_step_m * 1.5:
            previous_value = current_value
            continue
        sampled_runs.append((run_start, previous_value))
        run_start = current_value
        previous_value = current_value
    sampled_runs.append((run_start, previous_value))
    sampled_runs.sort(
        key=lambda item: (
            -abs(item[1] - item[0]),
            abs(((item[0] + item[1]) * 0.5) - preferred_span_value),
        )
    )
    chosen_run = sampled_runs[0]
    refined_span_value = (chosen_run[0] + chosen_run[1]) * 0.5

    source_coords = list(source_point)
    target_coords = list(target_point)
    source_coords[span_axis_index] = refined_span_value
    target_coords[span_axis_index] = refined_span_value
    return {
        **metrics,
        "source_point": (float(source_coords[0]), float(source_coords[1])),
        "target_point": (float(target_coords[0]), float(target_coords[1])),
        "span": max(0.0, float(chosen_run[1] - chosen_run[0])),
        "span_min": float(chosen_run[0]),
        "span_max": float(chosen_run[1]),
    }


def search_transition_metrics_from_segment_pairs(
    adapter: Any,
    *,
    scene: HOVSGSceneAsset,
    source_polygon: list[tuple[float, float]],
    target_polygon: list[tuple[float, float]],
    map_spec: dict[str, Any],
) -> dict[str, Any] | None:
    try:
        from ..nav2.nav2_runtime_bridge import (
            point_has_clearance,
            segment_has_clearance,
        )
    except Exception:
        return None

    candidates: list[dict[str, Any]] = []
    for source_start, source_end in polygon_segments(source_polygon):
        for target_start, target_end in polygon_segments(target_polygon):
            candidates.extend(
                segment_pair_transition_candidates(
                    adapter,
                    source_start=source_start,
                    source_end=source_end,
                    target_start=target_start,
                    target_end=target_end,
                )
            )

    if not candidates:
        return None

    resolution = max(
        0.01, float(map_spec.get("resolution", adapter.portal_analysis_map_resolution))
    )
    clearance_radius = adapter.portal_opening_clearance_radius_m
    best_metrics: dict[str, Any] | None = None
    best_score: tuple[float, float, float] | None = None
    plane_axes = adapter._horizontal_axes(scene)
    for candidate in candidates:
        span_axis_index = int(candidate["span_axis_index"])
        normal_axis_index = int(candidate["normal_axis_index"])
        span_min = float(candidate["span_min"])
        span_max = float(candidate["span_max"])
        if span_max < span_min:
            span_min, span_max = span_max, span_min
        if (span_max - span_min) < adapter.portal_sampling_step_m:
            continue

        preferred_span_value = float(candidate["preferred_span_value"])
        valid_samples: list[dict[str, Any]] = []
        sample_value = span_min
        while sample_value <= span_max + 1e-6:
            source_point = segment_point_at_axis_value(
                seg_start=candidate["source_segment"][0],
                seg_end=candidate["source_segment"][1],
                axis_index=span_axis_index,
                axis_value=sample_value,
            )
            target_point = segment_point_at_axis_value(
                seg_start=candidate["target_segment"][0],
                seg_end=candidate["target_segment"][1],
                axis_index=span_axis_index,
                axis_value=sample_value,
            )
            if source_point is None or target_point is None:
                sample_value += adapter.portal_sampling_step_m
                continue

            gap = math.hypot(target_point[0] - source_point[0], target_point[1] - source_point[1])
            normal_sign = (
                1.0 if target_point[normal_axis_index] >= source_point[normal_axis_index] else -1.0
            )
            boundary_value = (
                float(source_point[normal_axis_index]) + float(target_point[normal_axis_index])
            ) * 0.5
            probe_offset = max(adapter.portal_opening_probe_offset_m, gap * 0.5 + resolution)
            source_xy = portal_plane_point(
                plane_axes=plane_axes,
                span_axis=plane_axes[span_axis_index],
                normal_axis=plane_axes[normal_axis_index],
                span_value=sample_value,
                normal_value=boundary_value - normal_sign * probe_offset,
            )
            midpoint_xy = portal_plane_point(
                plane_axes=plane_axes,
                span_axis=plane_axes[span_axis_index],
                normal_axis=plane_axes[normal_axis_index],
                span_value=sample_value,
                normal_value=boundary_value,
            )
            target_xy = portal_plane_point(
                plane_axes=plane_axes,
                span_axis=plane_axes[span_axis_index],
                normal_axis=plane_axes[normal_axis_index],
                span_value=sample_value,
                normal_value=boundary_value + normal_sign * probe_offset,
            )
            if (
                point_has_clearance(
                    map_spec=map_spec,
                    point_xy=source_xy,
                    clearance_radius_m=clearance_radius,
                )
                and point_has_clearance(
                    map_spec=map_spec,
                    point_xy=midpoint_xy,
                    clearance_radius_m=clearance_radius,
                )
                and point_has_clearance(
                    map_spec=map_spec,
                    point_xy=target_xy,
                    clearance_radius_m=clearance_radius,
                )
                and segment_has_clearance(
                    map_spec=map_spec,
                    start_xy=source_xy,
                    end_xy=target_xy,
                    clearance_radius_m=clearance_radius,
                    step_m=adapter.portal_sampling_step_m * 0.5,
                )
            ):
                valid_samples.append(
                    {
                        "span_value": float(sample_value),
                        "gap": float(gap),
                        "boundary_value": float(boundary_value),
                        "source_point": source_point,
                        "target_point": target_point,
                    }
                )
            sample_value += adapter.portal_sampling_step_m

        if not valid_samples:
            continue

        sampled_runs: list[tuple[int, int]] = []
        run_start = 0
        previous_value = float(valid_samples[0]["span_value"])
        for index, sample in enumerate(valid_samples[1:], start=1):
            current_value = float(sample["span_value"])
            if (current_value - previous_value) <= adapter.portal_sampling_step_m * 1.5:
                previous_value = current_value
                continue
            sampled_runs.append((run_start, index - 1))
            run_start = index
            previous_value = current_value
        sampled_runs.append((run_start, len(valid_samples) - 1))

        for start_index, end_index in sampled_runs:
            run_start_sample = valid_samples[start_index]
            run_end_sample = valid_samples[end_index]
            run_span = max(
                0.0,
                float(run_end_sample["span_value"]) - float(run_start_sample["span_value"]),
            )
            if run_span <= 0.0:
                continue
            center_span_value = (
                float(run_start_sample["span_value"]) + float(run_end_sample["span_value"])
            ) * 0.5
            center_sample = min(
                valid_samples[start_index : end_index + 1],
                key=lambda sample: abs(float(sample["span_value"]) - center_span_value),
            )
            score = (
                float(run_span),
                -float(center_sample["gap"]),
                -abs(float(center_sample["span_value"]) - preferred_span_value),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_metrics = {
                    "source_point": (
                        float(center_sample["source_point"][0]),
                        float(center_sample["source_point"][1]),
                    ),
                    "target_point": (
                        float(center_sample["target_point"][0]),
                        float(center_sample["target_point"][1]),
                    ),
                    "gap": float(center_sample["gap"]),
                    "span": float(run_span),
                    "span_axis_index": span_axis_index,
                    "span_min": float(run_start_sample["span_value"]),
                    "span_max": float(run_end_sample["span_value"]),
                    "normal_axis_index": normal_axis_index,
                    "boundary_value": float(center_sample["boundary_value"]),
                }
    return best_metrics
