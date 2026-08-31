from __future__ import annotations

from typing import Any

from .portal_primitives import axis_aligned_segment_axes


def transition_points_from_bboxes(
    source_polygon: list[tuple[float, float]],
    target_polygon: list[tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    metrics = transition_metrics_from_bboxes(source_polygon, target_polygon)
    if metrics is None:
        return None
    return metrics["source_point"], metrics["target_point"]


def transition_metrics_from_bboxes(
    source_polygon: list[tuple[float, float]],
    target_polygon: list[tuple[float, float]],
) -> dict[str, Any] | None:
    source_xs = [point[0] for point in source_polygon]
    source_ys = [point[1] for point in source_polygon]
    target_xs = [point[0] for point in target_polygon]
    target_ys = [point[1] for point in target_polygon]

    source_x_min, source_x_max = min(source_xs), max(source_xs)
    source_y_min, source_y_max = min(source_ys), max(source_ys)
    target_x_min, target_x_max = min(target_xs), max(target_xs)
    target_y_min, target_y_max = min(target_ys), max(target_ys)

    overlap_x_min = max(source_x_min, target_x_min)
    overlap_x_max = min(source_x_max, target_x_max)
    overlap_y_min = max(source_y_min, target_y_min)
    overlap_y_max = min(source_y_max, target_y_max)

    candidates: list[dict[str, Any]] = []
    if overlap_x_min <= overlap_x_max:
        horizontal_mid = (overlap_x_min + overlap_x_max) * 0.5
        source_edge = source_y_max if source_y_max <= target_y_min else source_y_min
        target_edge = target_y_min if source_y_max <= target_y_min else target_y_max
        candidates.append(
            {
                "source_point": (horizontal_mid, source_edge),
                "target_point": (horizontal_mid, target_edge),
                "gap": abs(target_edge - source_edge),
                "span": max(0.0, overlap_x_max - overlap_x_min),
                "span_axis_index": 0,
                "span_min": overlap_x_min,
                "span_max": overlap_x_max,
                "normal_axis_index": 1,
                "boundary_value": (source_edge + target_edge) * 0.5,
            }
        )

    if overlap_y_min <= overlap_y_max:
        vertical_mid = (overlap_y_min + overlap_y_max) * 0.5
        source_edge = source_x_max if source_x_max <= target_x_min else source_x_min
        target_edge = target_x_min if source_x_max <= target_x_min else target_x_max
        candidates.append(
            {
                "source_point": (source_edge, vertical_mid),
                "target_point": (target_edge, vertical_mid),
                "gap": abs(target_edge - source_edge),
                "span": max(0.0, overlap_y_max - overlap_y_min),
                "span_axis_index": 1,
                "span_min": overlap_y_min,
                "span_max": overlap_y_max,
                "normal_axis_index": 0,
                "boundary_value": (source_edge + target_edge) * 0.5,
            }
        )

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item["gap"], -item["span"]))
    return candidates[0]


def segment_pair_transition_candidates(
    adapter: Any,
    *,
    source_start: tuple[float, float],
    source_end: tuple[float, float],
    target_start: tuple[float, float],
    target_end: tuple[float, float],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for carrier_start, carrier_end, other_start, other_end in (
        (source_start, source_end, target_start, target_end),
        (target_start, target_end, source_start, source_end),
    ):
        alignment = axis_aligned_segment_axes(carrier_start, carrier_end)
        if alignment is None:
            continue
        span_axis_index, normal_axis_index = alignment
        carrier_span_min = min(carrier_start[span_axis_index], carrier_end[span_axis_index])
        carrier_span_max = max(carrier_start[span_axis_index], carrier_end[span_axis_index])
        other_span_min = min(other_start[span_axis_index], other_end[span_axis_index])
        other_span_max = max(other_start[span_axis_index], other_end[span_axis_index])
        span_min = max(carrier_span_min, other_span_min)
        span_max = min(carrier_span_max, other_span_max)
        if (span_max - span_min) < adapter.portal_sampling_step_m:
            continue
        carrier_mid = (carrier_span_min + carrier_span_max) * 0.5
        other_mid = (other_span_min + other_span_max) * 0.5
        candidates.append(
            {
                "source_segment": (source_start, source_end),
                "target_segment": (target_start, target_end),
                "span_axis_index": span_axis_index,
                "normal_axis_index": normal_axis_index,
                "span_min": span_min,
                "span_max": span_max,
                "preferred_span_value": max(
                    span_min, min(span_max, (carrier_mid + other_mid) * 0.5)
                ),
            }
        )
    return candidates
