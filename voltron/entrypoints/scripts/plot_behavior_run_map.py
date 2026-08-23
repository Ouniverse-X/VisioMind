#!/usr/bin/env python3
"""Visualize a BEHAVIOR closed-loop run on the floor layout with portal widths."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from PIL import Image

DEFAULT_LAYOUT_PIXEL_RESOLUTION = 0.01
DEFAULT_R1PRO_NAV_FOOTPRINT: tuple[tuple[float, float], ...] = (
    (0.24, 0.34),
    (0.24, -0.34),
    (-0.40, -0.34),
    (-0.40, 0.34),
)
DEFAULT_R1PRO_NAV_FOOTPRINT_PADDING_M = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scene-map", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--floor-id", type=str, default="0")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--subtask-id",
        type=str,
        default="st_02",
        help="Subtask id used when extracting the direct local-goal line and latest controller state.",
    )
    parser.add_argument(
        "--highlight-room",
        action="append",
        default=[],
        help="Room name to highlight. Repeat this flag for multiple rooms.",
    )
    parser.add_argument(
        "--object-name",
        action="append",
        default=[],
        help="Object name to overlay. Matching is case-insensitive and exact after normalization.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_name(value: str) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def load_trajectory(process_log: Path, *, subtask_id: str) -> tuple[list[dict], list[dict], dict | None, dict | None]:
    poses: list[dict] = []
    portals: dict[tuple, dict] = {}
    first_progress_payload: dict | None = None
    last_progress_payload: dict | None = None

    def _portal_key(portal: dict) -> tuple | None:
        required = (
            portal.get("source_room_name"),
            portal.get("room_name"),
            portal.get("portal_normal_axis"),
            portal.get("portal_boundary_value"),
            portal.get("portal_span_axis"),
            portal.get("portal_span_min"),
            portal.get("portal_span_max"),
        )
        if any(value is None for value in required):
            return None
        return (
            str(required[0]),
            str(required[1]),
            str(required[2]),
            float(required[3]),
            str(required[4]),
            float(required[5]),
            float(required[6]),
        )

    def _register_portal(candidate: dict | None) -> None:
        if not isinstance(candidate, dict):
            return
        if candidate.get("waypoint_type") != "portal":
            return
        portal_key = _portal_key(candidate)
        if portal_key is None:
            return
        portals[portal_key] = candidate

    with process_log.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event") != "progress_update":
                continue
            payload = row.get("payload") or {}
            if str(payload.get("subtask_id") or "") != str(subtask_id):
                continue
            if first_progress_payload is None:
                first_progress_payload = payload
            last_progress_payload = payload
            pose = payload.get("pose") or {}
            if "x" in pose and "y" in pose:
                poses.append(
                    {
                        "control_step": payload.get("control_step"),
                        "x": float(pose["x"]),
                        "y": float(pose["y"]),
                        "room": payload.get("current_room"),
                    }
                )
            _register_portal(payload.get("local_goal"))
            _register_portal(payload.get("tracking_target"))
            _register_portal(payload.get("target_waypoint"))
            target_waypoint = payload.get("target_waypoint") or {}
            _register_portal(target_waypoint.get("transition_anchor"))
    return poses, list(portals.values()), first_progress_payload, last_progress_payload


def load_navigation_candidate_overlay(process_log: Path, *, subtask_id: str) -> dict:
    latest: dict | None = None
    with process_log.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            event = row.get("event")
            payload = row.get("payload") or {}
            if event == "navigation_candidates":
                if str(payload.get("subtask_id") or "") == str(subtask_id):
                    latest = normalize_candidate_overlay(payload)
                continue
            if event != "orchestrator_agent_result":
                continue
            if str(payload.get("subtask_id") or "") != str(subtask_id):
                continue
            result = payload.get("result") or {}
            candidates = None
            selected = None
            for container in navigation_candidate_containers(result):
                if not isinstance(container, dict):
                    continue
                if not isinstance(candidates, list):
                    candidates = container.get("object_approach_candidates")
                selected = selected or container.get("selected_object_approach")
            if not isinstance(candidates, list):
                continue
            latest = normalize_candidate_overlay(
                {
                    "subtask_id": subtask_id,
                    "target": result.get("grounded_goal") or result.get("nav_goal") or {},
                    "selected_candidate_id": (selected or {}).get("candidate_id") if isinstance(selected, dict) else None,
                    "selected_object_approach": selected if isinstance(selected, dict) else {},
                    "candidates": candidates,
                }
            )
    return latest or {"subtask_id": subtask_id, "target": {}, "selected_candidate_id": None, "candidates": []}


def navigation_candidate_containers(result: dict) -> tuple:
    return (
        result,
        result.get("grounded_goal") if isinstance(result.get("grounded_goal"), dict) else None,
        result.get("nav_goal") if isinstance(result.get("nav_goal"), dict) else None,
        result.get("path_plan") if isinstance(result.get("path_plan"), dict) else None,
        result.get("prepared_navigation_payload") if isinstance(result.get("prepared_navigation_payload"), dict) else None,
    )


def normalize_candidate_overlay(payload: dict) -> dict:
    candidates = []
    for candidate in payload.get("candidates") or []:
        point = point_xy(candidate)
        if point is None:
            continue
        normalized = dict(candidate)
        normalized["x"] = point[0]
        normalized["y"] = point[1]
        candidates.append(normalized)
    return {
        "subtask_id": payload.get("subtask_id"),
        "target": dict(payload.get("target") or {}),
        "selected_candidate_id": payload.get("selected_candidate_id"),
        "selected_object_approach": dict(payload.get("selected_object_approach") or {}),
        "candidates": candidates,
    }


def point_xy(payload: dict | None) -> tuple[float, float] | None:
    if not isinstance(payload, dict):
        return None
    source = payload.get("position") if isinstance(payload.get("position"), dict) else payload
    try:
        return float(source["x"]), float(source["y"])
    except (KeyError, TypeError, ValueError):
        return None


def image_foreground_bbox(image: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(image > 0)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def world_to_pixel(
    x: float,
    y: float,
    *,
    image_size: tuple[int, int],
    pixel_resolution: float = DEFAULT_LAYOUT_PIXEL_RESOLUTION,
) -> tuple[float, float]:
    width_px, height_px = image_size
    center_x = width_px / 2.0
    center_y = height_px / 2.0
    # Match the BEHAVIOR layout -> world transform used by scene_graph_export.py:
    # x = (col - map_size / 2) * res, y = (row - map_size / 2) * res
    px = center_x + (x / pixel_resolution)
    py = center_y + (y / pixel_resolution)
    return px, py


def load_room_polygons(graph_root: Path) -> dict[str, np.ndarray]:
    room_dir = graph_root / "rooms"
    polygons: dict[str, np.ndarray] = {}
    for room_file in sorted(room_dir.glob("*.json")):
        room = load_json(room_file)
        polygons[str(room["name"])] = np.asarray([[float(v[0]), float(v[1])] for v in room["vertices"]], dtype=float)
    return polygons


def load_object_polygons(graph_root: Path, *, requested_names: set[str]) -> list[dict]:
    object_dir = graph_root / "objects"
    objects: list[dict] = []
    for object_file in sorted(object_dir.glob("*.json")):
        obj = load_json(object_file)
        obj_name = normalize_name(obj.get("name", ""))
        if requested_names and obj_name not in requested_names:
            continue
        vertices = np.asarray([[float(v[0]), float(v[1])] for v in obj["vertices"]], dtype=float)
        position = obj.get("position") or {}
        objects.append(
            {
                "name": obj_name,
                "raw_name": str(obj.get("name", "")),
                "vertices": vertices,
                "position": {
                    "x": float(position.get("x", float(vertices[:, 0].mean()))),
                    "y": float(position.get("y", float(vertices[:, 1].mean()))),
                },
            }
        )
    return objects


def world_vertices_to_pixel(vertices_w: np.ndarray, *, image_size: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        [world_to_pixel(float(x_coord), float(y_coord), image_size=image_size) for x_coord, y_coord in vertices_w],
        dtype=float,
    )


def expanded_bbox_polygon(vertices_w: np.ndarray, *, padding_m: float) -> np.ndarray:
    min_x = float(np.min(vertices_w[:, 0])) - padding_m
    max_x = float(np.max(vertices_w[:, 0])) + padding_m
    min_y = float(np.min(vertices_w[:, 1])) - padding_m
    max_y = float(np.max(vertices_w[:, 1])) + padding_m
    return np.asarray(
        [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
        ],
        dtype=float,
    )


def footprint_polygon_world(
    *,
    pose_xy: dict[str, float],
    yaw: float,
    footprint: tuple[tuple[float, float], ...] = DEFAULT_R1PRO_NAV_FOOTPRINT,
    padding_m: float = DEFAULT_R1PRO_NAV_FOOTPRINT_PADDING_M,
) -> np.ndarray:
    padded_vertices: list[tuple[float, float]] = []
    for local_x, local_y in footprint:
        sign_x = 1.0 if local_x >= 0.0 else -1.0
        sign_y = 1.0 if local_y >= 0.0 else -1.0
        padded_vertices.append((local_x + sign_x * padding_m, local_y + sign_y * padding_m))

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    origin_x = float(pose_xy["x"])
    origin_y = float(pose_xy["y"])
    vertices_world = []
    for local_x, local_y in padded_vertices:
        world_x = origin_x + (local_x * cos_yaw - local_y * sin_yaw)
        world_y = origin_y + (local_x * sin_yaw + local_y * cos_yaw)
        vertices_world.append((world_x, world_y))
    return np.asarray(vertices_world, dtype=float)


def load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    return load_json(path)


def config_vln(config: dict) -> dict:
    vln = config.get("vln")
    return vln if isinstance(vln, dict) else {}


def format_parameter_summary(config: dict) -> str:
    vln = config_vln(config)
    footprint_x = [vertex[0] for vertex in DEFAULT_R1PRO_NAV_FOOTPRINT]
    footprint_y = [vertex[1] for vertex in DEFAULT_R1PRO_NAV_FOOTPRINT]
    footprint_length = max(footprint_x) - min(footprint_x) + 2.0 * DEFAULT_R1PRO_NAV_FOOTPRINT_PADDING_M
    footprint_width = max(footprint_y) - min(footprint_y) + 2.0 * DEFAULT_R1PRO_NAV_FOOTPRINT_PADDING_M
    items = [
        f"trav_map={vln.get('trav_map_filename', '-')}",
        f"direct_transition_min_span={vln.get('direct_room_transition_min_span_m', '-')}",
        f"portal_clearance={vln.get('portal_clearance_radius_m', '-')}",
        f"local_clearance={vln.get('local_path_clearance_radius_m', '-')}",
        f"footprint_bbox={footprint_length:.2f}x{footprint_width:.2f}m",
        f"portal_align_width={vln.get('portal_alignment_footprint_width_m', '-')}",
        f"max_v={vln.get('max_linear_velocity', '-')}",
        f"local_path_max_v={vln.get('local_path_max_linear_velocity', '-')}",
    ]
    return "\n".join(items)


def segment_intersects_expanded_bbox(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    vertices_w: np.ndarray,
    *,
    padding_m: float,
) -> bool:
    expanded = expanded_bbox_polygon(vertices_w, padding_m=padding_m)
    min_x = float(np.min(expanded[:, 0]))
    max_x = float(np.max(expanded[:, 0]))
    min_y = float(np.min(expanded[:, 1]))
    max_y = float(np.max(expanded[:, 1]))
    x0, y0 = start_xy
    x1, y1 = end_xy

    def _inside(x_coord: float, y_coord: float) -> bool:
        return min_x <= x_coord <= max_x and min_y <= y_coord <= max_y

    if _inside(x0, y0) or _inside(x1, y1):
        return True

    dx = x1 - x0
    dy = y1 - y0
    p = (-dx, dx, -dy, dy)
    q = (x0 - min_x, max_x - x0, y0 - min_y, max_y - y0)
    u1 = 0.0
    u2 = 1.0
    for p_value, q_value in zip(p, q):
        if abs(p_value) <= 1e-9:
            if q_value < 0.0:
                return False
            continue
        t_value = q_value / p_value
        if p_value < 0.0:
            u1 = max(u1, t_value)
        else:
            u2 = min(u2, t_value)
        if u1 > u2:
            return False
    return True


def portal_segment(portal: dict) -> tuple[tuple[float, float], tuple[float, float], str]:
    span_axis = str(portal["portal_span_axis"])
    boundary = float(portal["portal_boundary_value"])
    span_min = float(portal["portal_span_min"])
    span_max = float(portal["portal_span_max"])
    if span_axis == "x":
        p0 = (span_min, boundary)
        p1 = (span_max, boundary)
    else:
        p0 = (boundary, span_min)
        p1 = (boundary, span_max)
    portal_span = float(portal.get("portal_span", span_max - span_min))
    label = f'{portal["source_room_name"]}->{portal["room_name"]} ({portal_span:.1f}m)'
    return p0, p1, label


def crop_bounds(
    points_px: list[tuple[float, float]],
    *,
    image_size: tuple[int, int],
    margin: int = 280,
) -> tuple[int, int, int, int]:
    width, height = image_size
    xs = [p[0] for p in points_px]
    ys = [p[1] for p in points_px]
    left = max(0, int(min(xs) - margin))
    top = max(0, int(min(ys) - margin))
    right = min(width, int(max(xs) + margin))
    bottom = min(height, int(max(ys) + margin))
    return left, top, right, bottom


def main() -> None:
    args = parse_args()
    process_log = args.run_dir / "process_data.jsonl"
    output = args.output or (args.run_dir / "map_trajectory_portals.png")

    load_json(args.scene_map)
    load_json(args.graph_root / "floors" / f"{args.floor_id}.json")
    config = load_config(args.config)
    image = np.asarray(Image.open(args.layout).convert("L"))
    image_size = (image.shape[1], image.shape[0])

    poses, portals, first_payload, last_payload = load_trajectory(process_log, subtask_id=args.subtask_id)
    if not poses:
        raise RuntimeError(f"No trajectory points found in {process_log}")
    candidate_overlay = load_navigation_candidate_overlay(process_log, subtask_id=args.subtask_id)

    room_polygons = load_room_polygons(args.graph_root)
    relevant_rooms = {str(room).strip() for room in args.highlight_room if str(room).strip()}
    requested_objects = {normalize_name(name) for name in args.object_name if normalize_name(name)}
    objects = load_object_polygons(args.graph_root, requested_names=requested_objects)
    vln = config_vln(config)
    local_path_clearance = float(vln.get("local_path_clearance_radius_m", 0.0) or 0.0)
    portal_clearance = float(vln.get("portal_clearance_radius_m", 0.0) or 0.0)

    traj_px = [world_to_pixel(item["x"], item["y"], image_size=image_size) for item in poses]
    portal_segments = []
    for portal in portals:
        p0_w, p1_w, label = portal_segment(portal)
        p0_px = world_to_pixel(p0_w[0], p0_w[1], image_size=image_size)
        p1_px = world_to_pixel(p1_w[0], p1_w[1], image_size=image_size)
        portal_segments.append((p0_px, p1_px, label))

    crop_pts = list(traj_px)
    candidate_points_w = [point_xy(candidate) for candidate in candidate_overlay.get("candidates", [])]
    candidate_points_w = [point for point in candidate_points_w if point is not None]
    target_point_w = point_xy(candidate_overlay.get("target", {}).get("position"))
    if target_point_w is not None:
        crop_pts.append(world_to_pixel(target_point_w[0], target_point_w[1], image_size=image_size))
    for point in candidate_points_w:
        crop_pts.append(world_to_pixel(point[0], point[1], image_size=image_size))
    for p0, p1, _ in portal_segments:
        crop_pts.extend([p0, p1])
    for obj in objects:
        crop_pts.extend(world_vertices_to_pixel(obj["vertices"], image_size=image_size))
    crop_box = crop_bounds(crop_pts, image_size=(image.shape[1], image.shape[0]))

    first_pose = poses[0]
    first_local_goal = None
    if isinstance(first_payload, dict):
        candidate_goal = first_payload.get("local_goal")
        if isinstance(candidate_goal, dict) and "x" in candidate_goal and "y" in candidate_goal:
            first_local_goal = {"x": float(candidate_goal["x"]), "y": float(candidate_goal["y"])}
    initial_yaw = 0.0
    if isinstance(first_payload, dict):
        yaw_value = first_payload.get("yaw")
        if isinstance(yaw_value, (float, int)):
            initial_yaw = float(yaw_value)
    if abs(initial_yaw) <= 1e-6 and len(poses) >= 2:
        initial_yaw = math.atan2(
            float(poses[1]["y"]) - float(poses[0]["y"]),
            float(poses[1]["x"]) - float(poses[0]["x"]),
        )
    footprint_px = None
    if first_pose is not None:
        footprint_world = footprint_polygon_world(
            pose_xy={"x": float(first_pose["x"]), "y": float(first_pose["y"])},
            yaw=initial_yaw,
        )
        footprint_px = world_vertices_to_pixel(footprint_world, image_size=image_size)

    direct_line_intersections: list[str] = []
    if first_local_goal is not None:
        start_xy = (float(first_pose["x"]), float(first_pose["y"]))
        goal_xy = (float(first_local_goal["x"]), float(first_local_goal["y"]))
        for obj in objects:
            if local_path_clearance <= 0.0:
                continue
            if segment_intersects_expanded_bbox(
                start_xy,
                goal_xy,
                obj["vertices"],
                padding_m=local_path_clearance,
            ):
                direct_line_intersections.append(obj["raw_name"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=180)
    titles = ["Full Floor", "Zoom Near Target Doors"]
    views = [None, crop_box]

    for ax, title, crop in zip(axes, titles, views):
        ax.imshow(image, cmap="gray", origin="upper")
        for room_name, vertices_w in room_polygons.items():
            vertices_px = np.asarray(
                [world_to_pixel(x, y, image_size=image_size) for x, y in vertices_w],
                dtype=float,
            )
            edge_color = "#47a3ff" if room_name in relevant_rooms else "#8a8a8a"
            alpha = 0.95 if room_name in relevant_rooms else 0.35
            ax.add_patch(
                Polygon(
                    vertices_px,
                    closed=True,
                    fill=False,
                    linewidth=2.0 if room_name in relevant_rooms else 0.9,
                    edgecolor=edge_color,
                    alpha=alpha,
                )
            )
            if room_name in relevant_rooms:
                center = vertices_px.mean(axis=0)
                ax.text(center[0], center[1], room_name, fontsize=8, color="#8fd0ff", ha="center", va="center")

        traj_x = [p[0] for p in traj_px]
        traj_y = [p[1] for p in traj_px]
        ax.plot(traj_x, traj_y, color="#ff4d4d", linewidth=2.0, label="trajectory")
        ax.scatter(traj_x[0], traj_y[0], color="#00d084", s=50, zorder=5, label="start")
        ax.scatter(traj_x[-1], traj_y[-1], color="#ffd24d", s=50, zorder=5, label="last")

        candidate_label_used = False
        selected_candidate_id = str(candidate_overlay.get("selected_candidate_id") or "")
        selected_px = None
        for candidate in candidate_overlay.get("candidates", []):
            point = point_xy(candidate)
            if point is None:
                continue
            candidate_px = world_to_pixel(point[0], point[1], image_size=image_size)
            is_selected = str(candidate.get("candidate_id") or "") == selected_candidate_id
            if is_selected:
                selected_px = candidate_px
            ax.scatter(
                candidate_px[0],
                candidate_px[1],
                s=92 if is_selected else 42,
                marker="*" if is_selected else "o",
                facecolors="#00f5ff" if is_selected else "none",
                edgecolors="#00f5ff",
                linewidths=1.5,
                zorder=7,
                label="navigation candidates" if not candidate_label_used and crop is None else None,
            )
            candidate_label_used = True
            label = str(candidate.get("candidate_id") or "")
            if label:
                ax.text(
                    candidate_px[0] + 6,
                    candidate_px[1] - 6,
                    label,
                    fontsize=7,
                    color="#00f5ff",
                    ha="left",
                    va="bottom",
                    zorder=8,
                )

        if target_point_w is not None:
            target_px = world_to_pixel(target_point_w[0], target_point_w[1], image_size=image_size)
            ax.scatter(
                target_px[0],
                target_px[1],
                s=80,
                marker="x",
                color="#ff66ff",
                linewidths=2.0,
                zorder=8,
                label="navigation target" if crop is None else None,
            )
            target_label = str(candidate_overlay.get("target", {}).get("object_name") or candidate_overlay.get("target", {}).get("object_id") or "target")
            ax.text(target_px[0] + 8, target_px[1] + 8, target_label, fontsize=8, color="#ff66ff", ha="left", va="top")
            if selected_px is not None:
                ax.plot(
                    [selected_px[0], target_px[0]],
                    [selected_px[1], target_px[1]],
                    color="#ff66ff",
                    linewidth=1.1,
                    linestyle=":",
                    alpha=0.9,
                )

        if first_local_goal is not None:
            goal_px = world_to_pixel(first_local_goal["x"], first_local_goal["y"], image_size=image_size)
            ax.plot(
                [traj_x[0], goal_px[0]],
                [traj_y[0], goal_px[1]],
                color="#ff5ea8",
                linewidth=1.8,
                linestyle="--",
                label="initial direct local_goal line" if crop is None else None,
            )
            ax.scatter(goal_px[0], goal_px[1], color="#ff5ea8", s=36, zorder=5)

        if footprint_px is not None:
            ax.add_patch(
                Polygon(
                    footprint_px,
                    closed=True,
                    fill=False,
                    linewidth=1.5,
                    edgecolor="#66ff66",
                    alpha=0.95,
                )
            )

        object_colors = {
            "coffee table": "#ff884d",
            "fridge": "#59c3ff",
            "sofa": "#ffcf5c",
        }
        for obj in objects:
            vertices_px = world_vertices_to_pixel(obj["vertices"], image_size=image_size)
            color = object_colors.get(obj["name"], "#d0d0d0")
            ax.add_patch(
                Polygon(
                    vertices_px,
                    closed=True,
                    fill=False,
                    linewidth=2.0,
                    edgecolor=color,
                    alpha=0.95,
                )
            )
            center = vertices_px.mean(axis=0)
            ax.text(center[0], center[1] - 10, obj["raw_name"], fontsize=8, color=color, ha="center", va="bottom")
            if local_path_clearance > 0.0 and obj["name"] in {"coffee table", "sofa"}:
                expanded_px = world_vertices_to_pixel(
                    expanded_bbox_polygon(obj["vertices"], padding_m=local_path_clearance),
                    image_size=image_size,
                )
                ax.add_patch(
                    Polygon(
                        expanded_px,
                        closed=True,
                        fill=False,
                        linewidth=1.2,
                        linestyle="--",
                        edgecolor=color,
                        alpha=0.75,
                    )
                )
            if portal_clearance > 0.0 and obj["name"] == "fridge":
                expanded_px = world_vertices_to_pixel(
                    expanded_bbox_polygon(obj["vertices"], padding_m=portal_clearance),
                    image_size=image_size,
                )
                ax.add_patch(
                    Polygon(
                        expanded_px,
                        closed=True,
                        fill=False,
                        linewidth=1.2,
                        linestyle=":",
                        edgecolor=color,
                        alpha=0.65,
                    )
                )

        for idx, (p0, p1, label) in enumerate(portal_segments):
            color = "#00e5ff" if idx == 0 else "#ff9f1a"
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linewidth=5, solid_capstyle="round")
            cx = (p0[0] + p1[0]) / 2
            cy = (p0[1] + p1[1]) / 2
            ax.text(cx, cy - 18, label, fontsize=8, color=color, ha="center", va="bottom")

        if crop is not None:
            left, top, right, bottom = crop
            ax.set_xlim(left, right)
            ax.set_ylim(bottom, top)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].legend(loc="lower right", fontsize=8)
    summary_lines = [format_parameter_summary(config)]
    if direct_line_intersections:
        summary_lines.append(
            "initial_line_hits_local_clearance=" + ", ".join(sorted(set(direct_line_intersections)))
        )
    if isinstance(last_payload, dict):
        summary_lines.append(
            "last_mode="
            + str(last_payload.get("controller_mode", "-"))
            + "/"
            + str(last_payload.get("recovery_mode", "-"))
        )
        summary_lines.append(
            "last_path_backend="
            + str(last_payload.get("path_backend", "-"))
            + " tracking="
            + str(last_payload.get("path_tracking_mode", "-"))
        )
    if candidate_overlay.get("candidates"):
        summary_lines.append(
            "navigation_candidates="
            + str(len(candidate_overlay["candidates"]))
            + " selected="
            + str(candidate_overlay.get("selected_candidate_id") or "-")
        )
    fig.text(
        0.015,
        0.02,
        "\n".join(summary_lines),
        fontsize=8,
        family="monospace",
        va="bottom",
        ha="left",
        bbox={"facecolor": "#111111", "alpha": 0.85, "edgecolor": "#333333", "pad": 8},
        color="#f0f0f0",
    )
    fig.suptitle(
        f"Run: {args.run_dir.name}\nTrajectory, navigation candidates, target, obstacle envelopes, and portal width segments",
        fontsize=12,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
