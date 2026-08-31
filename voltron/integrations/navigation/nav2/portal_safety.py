from __future__ import annotations

from typing import Any


DEFAULT_PORTAL_EGRESS_DEPTH_M = 0.65
DEFAULT_PORTAL_SPAN_TOLERANCE_M = 0.22


def has_portal_frame(anchor: Any) -> bool:
    if not isinstance(anchor, dict):
        return False
    normal_axis = anchor.get("portal_normal_axis")
    span_axis = anchor.get("portal_span_axis")
    return (
        normal_axis in {"x", "y", "z"}
        and span_axis in {"x", "y", "z"}
        and normal_axis != span_axis
        and isinstance(anchor.get("portal_boundary_value"), (int, float))
        and isinstance(anchor.get("portal_normal_sign"), (int, float))
    )


def required_egress_depth_m(
    anchor: dict[str, Any],
    *,
    default_depth_m: float = DEFAULT_PORTAL_EGRESS_DEPTH_M,
) -> float:
    if not has_portal_frame(anchor):
        return 0.0
    normal_axis = str(anchor["portal_normal_axis"])
    normal_sign = _normal_sign(anchor)
    try:
        target_offset = (
            float(anchor[normal_axis]) - float(anchor["portal_boundary_value"])
        ) * normal_sign
    except (KeyError, TypeError, ValueError):
        target_offset = 0.0
    configured_depth = anchor.get("portal_required_egress_depth_m")
    if isinstance(configured_depth, (int, float)):
        default_depth_m = max(0.0, float(configured_depth))
    return max(max(0.0, float(default_depth_m)), target_offset)


def target_side_depth_m(*, pose: dict[str, Any], anchor: dict[str, Any]) -> float | None:
    if not has_portal_frame(anchor):
        return None
    normal_axis = str(anchor["portal_normal_axis"])
    try:
        return (float(pose[normal_axis]) - float(anchor["portal_boundary_value"])) * _normal_sign(
            anchor
        )
    except (KeyError, TypeError, ValueError):
        return None


def pose_has_sufficient_egress(
    *,
    pose: dict[str, Any],
    anchor: dict[str, Any],
    span_tolerance_m: float = DEFAULT_PORTAL_SPAN_TOLERANCE_M,
    required_depth_m: float | None = None,
) -> bool:
    depth = target_side_depth_m(pose=pose, anchor=anchor)
    required_depth = required_egress_depth_m(
        anchor,
        default_depth_m=(
            DEFAULT_PORTAL_EGRESS_DEPTH_M if required_depth_m is None else required_depth_m
        ),
    )
    if depth is None or depth < required_depth:
        return False
    span_axis = str(anchor["portal_span_axis"])
    try:
        pose_span = float(pose[span_axis])
        anchor_span = float(anchor[span_axis])
        span_min = float(anchor.get("portal_span_min", anchor_span))
        span_max = float(anchor.get("portal_span_max", anchor_span))
    except (KeyError, TypeError, ValueError):
        return True
    tolerance = max(0.0, float(span_tolerance_m))
    return min(span_min, span_max) - tolerance <= pose_span <= max(
        span_min, span_max
    ) + tolerance and abs(pose_span - anchor_span) <= max(tolerance, 0.22)


def egress_waypoint(
    *,
    anchor: dict[str, Any],
    path_points: list[dict[str, float]],
) -> dict[str, Any] | None:
    if not has_portal_frame(anchor):
        return None
    normal_axis = str(anchor["portal_normal_axis"])
    span_axis = str(anchor["portal_span_axis"])
    if normal_axis not in {"x", "y"} or span_axis not in {"x", "y"}:
        return None
    required_depth = required_egress_depth_m(anchor)
    normal_sign = _normal_sign(anchor)
    boundary = float(anchor["portal_boundary_value"])
    anchor_span = float(anchor[span_axis])
    span_min = float(anchor.get("portal_span_min", anchor_span))
    span_max = float(anchor.get("portal_span_max", anchor_span))
    span_tolerance = DEFAULT_PORTAL_SPAN_TOLERANCE_M

    candidates: list[tuple[float, float, dict[str, float]]] = []
    for point in path_points:
        if not isinstance(point, dict):
            continue
        try:
            normal_value = float(point[normal_axis])
            span_value = float(point[span_axis])
        except (KeyError, TypeError, ValueError):
            continue
        depth = (normal_value - boundary) * normal_sign
        if depth < required_depth:
            continue
        if not (
            min(span_min, span_max) - span_tolerance
            <= span_value
            <= max(span_min, span_max) + span_tolerance
        ):
            continue
        candidates.append((depth, abs(span_value - anchor_span), point))

    waypoint = dict(anchor)
    if candidates:
        _, _, chosen = min(candidates, key=lambda item: (item[0], item[1]))
        waypoint[normal_axis] = float(chosen[normal_axis])
        waypoint[span_axis] = float(chosen[span_axis])
        waypoint["portal_egress_source"] = "nav2_path"
    else:
        waypoint[normal_axis] = boundary + normal_sign * required_depth
        waypoint[span_axis] = anchor_span
        waypoint["portal_egress_source"] = "portal_centerline"
    waypoint["waypoint_type"] = "local_path"
    waypoint["portal_egress_guard"] = True
    waypoint["portal_egress_depth_m"] = required_depth
    return waypoint


def _normal_sign(anchor: dict[str, Any]) -> float:
    try:
        return 1.0 if float(anchor.get("portal_normal_sign", 1.0)) >= 0.0 else -1.0
    except (TypeError, ValueError):
        return 1.0


__all__ = [
    "DEFAULT_PORTAL_EGRESS_DEPTH_M",
    "egress_waypoint",
    "has_portal_frame",
    "pose_has_sufficient_egress",
    "required_egress_depth_m",
    "target_side_depth_m",
]
