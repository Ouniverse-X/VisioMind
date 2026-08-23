#!/usr/bin/env python3
"""Render a portal-focused diagnostics figure for a BEHAVIOR closed-loop run."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon
from PIL import Image

DEFAULT_LAYOUT_PIXEL_RESOLUTION = 0.01
DEFAULT_R1PRO_NAV_FOOTPRINT = (
    (0.24, 0.34),
    (0.24, -0.34),
    (-0.40, -0.34),
    (-0.40, 0.34),
)
DEFAULT_R1PRO_NAV_FOOTPRINT_PADDING_M = 0.02


@dataclass(frozen=True)
class PortalKey:
    source_room_name: str
    room_name: str
    normal_axis: str
    boundary_value: float
    span_axis: str
    span_min: float
    span_max: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--layout", type=Path, default=None)
    parser.add_argument("--graph-root", type=Path, default=None)
    parser.add_argument("--floor-id", type=str, default="0")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def world_to_pixel(
    x_coord: float,
    y_coord: float,
    *,
    image_size: tuple[int, int],
    pixel_resolution: float = DEFAULT_LAYOUT_PIXEL_RESOLUTION,
) -> tuple[float, float]:
    width_px, height_px = image_size
    center_x = width_px / 2.0
    center_y = height_px / 2.0
    px = center_x + (x_coord / pixel_resolution)
    py = center_y + (y_coord / pixel_resolution)
    return px, py


def portal_key(candidate: dict[str, Any] | None) -> PortalKey | None:
    if not isinstance(candidate, dict):
        return None
    required = (
        candidate.get("source_room_name"),
        candidate.get("room_name"),
        candidate.get("portal_normal_axis"),
        candidate.get("portal_boundary_value"),
        candidate.get("portal_span_axis"),
        candidate.get("portal_span_min"),
        candidate.get("portal_span_max"),
    )
    if any(value is None for value in required):
        return None
    return PortalKey(
        source_room_name=str(required[0]),
        room_name=str(required[1]),
        normal_axis=str(required[2]),
        boundary_value=float(required[3]),
        span_axis=str(required[4]),
        span_min=float(required[5]),
        span_max=float(required[6]),
    )


def load_progress_rows(process_log: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with process_log.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event") == "progress_update":
                rows.append(row)
    return rows


def load_pose_series(progress_rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    poses: list[dict[str, float]] = []
    for row in progress_rows:
        payload = row.get("payload") or {}
        pose = payload.get("pose") or {}
        if "x" not in pose or "y" not in pose:
            continue
        poses.append(
            {
                "control_step": float(payload.get("control_step", len(poses))),
                "x": float(pose["x"]),
                "y": float(pose["y"]),
            }
        )
    return poses


def choose_active_portal(progress_rows: list[dict[str, Any]]) -> tuple[PortalKey, dict[str, Any], dict[str, Any]]:
    latest_progress = progress_rows[-1]["payload"]
    for key in ("tracking_target", "target_waypoint", "local_goal"):
        candidate = latest_progress.get(key)
        portal = portal_key(candidate)
        if portal is not None:
            return portal, candidate, latest_progress
    raise RuntimeError("No portal metadata found in the latest progress update.")


def collect_portal_candidates(
    progress_rows: list[dict[str, Any]],
    active_key: PortalKey,
) -> dict[str, dict[str, Any]]:
    collected: dict[str, dict[str, Any]] = {}
    desired_waypoint_types = {
        "pre_portal_standoff": "source_anchor",
        "portal_midpoint": "midpoint",
        "portal": "portal",
    }
    for row in progress_rows:
        payload = row.get("payload") or {}
        for field in ("tracking_target", "target_waypoint", "local_goal"):
            candidate = payload.get(field)
            if portal_key(candidate) != active_key:
                continue
            waypoint_type = str(candidate.get("waypoint_type", ""))
            bucket = desired_waypoint_types.get(waypoint_type)
            if bucket is None:
                continue
            collected[bucket] = dict(candidate)
    return collected


def infer_layout_path(config: dict[str, Any]) -> Path:
    graph_root = Path(config["vln"]["graph_path"]).resolve()
    metadata = load_json(graph_root / "metadata.json")
    scene_id = str(metadata["scene_id"])
    trav_map_filename = str(config["vln"]["trav_map_filename"])
    return (
        Path("/mnt/data/huangyixuan/isaac/BEHAVIOR-1K/datasets/behavior-1k-assets/scenes")
        / scene_id
        / "layout"
        / trav_map_filename
    )


def load_voronoi_crossing(
    graph_root: Path,
    *,
    source_room_name: str,
    target_room_name: str,
    door_center_xy: tuple[float, float],
) -> list[tuple[float, float]]:
    nav_graph = load_json(graph_root / "nav_graph" / "sparse_voronoi_graph.json")
    node_by_id = {tuple(node["id"]): node for node in nav_graph.get("nodes", [])}
    best_polyline: list[tuple[float, float]] = []
    best_distance = float("inf")
    room_pair = {source_room_name, target_room_name}
    for link in nav_graph.get("links", []):
        source = node_by_id.get(tuple(link["source"]))
        target = node_by_id.get(tuple(link["target"]))
        if not source or not target:
            continue
        if {source.get("room_name"), target.get("room_name")} != room_pair:
            continue
        polyline = [(float(point[0]), float(point[1])) for point in link.get("polyline", [])]
        if not polyline:
            continue
        distance = min(
            math.hypot(point[0] - door_center_xy[0], point[1] - door_center_xy[1]) for point in polyline
        )
        if distance < best_distance:
            best_distance = distance
            best_polyline = polyline
    return best_polyline


def portal_segment_world(portal: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    span_axis = str(portal["portal_span_axis"])
    boundary = float(portal["portal_boundary_value"])
    span_min = float(portal["portal_span_min"])
    span_max = float(portal["portal_span_max"])
    if span_axis == "x":
        return (span_min, boundary), (span_max, boundary)
    return (boundary, span_min), (boundary, span_max)


def build_footprint_polygon(
    *,
    center_xy: tuple[float, float],
    heading_rad: float,
    footprint: tuple[tuple[float, float], ...] = DEFAULT_R1PRO_NAV_FOOTPRINT,
    padding_m: float = DEFAULT_R1PRO_NAV_FOOTPRINT_PADDING_M,
) -> np.ndarray:
    padded = []
    for x_coord, y_coord in footprint:
        padded_x = x_coord + math.copysign(padding_m, x_coord) if x_coord != 0 else x_coord
        padded_y = y_coord + math.copysign(padding_m, y_coord) if y_coord != 0 else y_coord
        padded.append((padded_x, padded_y))
    cos_yaw = math.cos(heading_rad)
    sin_yaw = math.sin(heading_rad)
    points = []
    for rel_x, rel_y in padded:
        world_x = center_xy[0] + rel_x * cos_yaw - rel_y * sin_yaw
        world_y = center_xy[1] + rel_x * sin_yaw + rel_y * cos_yaw
        points.append((world_x, world_y))
    return np.asarray(points, dtype=float)


def crop_world_bounds(points: list[tuple[float, float]], *, margin_m: float = 1.6) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (
        min(xs) - margin_m,
        max(xs) + margin_m,
        min(ys) - margin_m,
        max(ys) + margin_m,
    )


def set_world_window(
    ax: plt.Axes,
    bounds_world: tuple[float, float, float, float],
    *,
    image_size: tuple[int, int],
) -> None:
    min_x, max_x, min_y, max_y = bounds_world
    left, bottom = world_to_pixel(min_x, min_y, image_size=image_size)
    right, top = world_to_pixel(max_x, max_y, image_size=image_size)
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    graph_root = (args.graph_root or Path(config["vln"]["graph_path"])).resolve()
    layout_path = (args.layout or infer_layout_path(config)).resolve()
    output = args.output or (args.run_dir / "map_overlay_portal_diagnostics.png")

    process_log = args.run_dir / "process_data.jsonl"
    progress_rows = load_progress_rows(process_log)
    if not progress_rows:
        raise RuntimeError(f"No progress_update rows found in {process_log}")

    active_key, active_portal, latest_progress = choose_active_portal(progress_rows)
    portal_candidates = collect_portal_candidates(progress_rows, active_key)
    poses = load_pose_series(progress_rows)

    image = np.asarray(Image.open(layout_path).convert("L"))
    image_size = (image.shape[1], image.shape[0])

    last_pose = poses[-1]
    start_pose = poses[0]
    target_xy = (float(active_portal["x"]), float(active_portal["y"]))
    last_xy = (float(last_pose["x"]), float(last_pose["y"]))
    heading_rad = math.atan2(target_xy[1] - last_xy[1], target_xy[0] - last_xy[0])

    raw_p0, raw_p1 = portal_segment_world(active_portal)
    door_center = ((raw_p0[0] + raw_p1[0]) / 2.0, (raw_p0[1] + raw_p1[1]) / 2.0)
    voronoi_polyline = load_voronoi_crossing(
        graph_root,
        source_room_name=str(active_portal["source_room_name"]),
        target_room_name=str(active_portal["room_name"]),
        door_center_xy=door_center,
    )

    footprint_world = build_footprint_polygon(center_xy=last_xy, heading_rad=heading_rad)
    footprint_px = np.asarray(
        [world_to_pixel(x_coord, y_coord, image_size=image_size) for x_coord, y_coord in footprint_world],
        dtype=float,
    )
    traj_px = np.asarray(
        [world_to_pixel(pose["x"], pose["y"], image_size=image_size) for pose in poses],
        dtype=float,
    )
    raw_segment_px = np.asarray(
        [world_to_pixel(raw_p0[0], raw_p0[1], image_size=image_size), world_to_pixel(raw_p1[0], raw_p1[1], image_size=image_size)],
        dtype=float,
    )
    target_px = world_to_pixel(target_xy[0], target_xy[1], image_size=image_size)
    last_px = world_to_pixel(last_xy[0], last_xy[1], image_size=image_size)
    start_px = world_to_pixel(start_pose["x"], start_pose["y"], image_size=image_size)

    source_anchor = portal_candidates.get("source_anchor")
    source_anchor_px = (
        world_to_pixel(float(source_anchor["x"]), float(source_anchor["y"]), image_size=image_size)
        if source_anchor is not None
        else None
    )
    midpoint = portal_candidates.get("midpoint")
    midpoint_px = (
        world_to_pixel(float(midpoint["x"]), float(midpoint["y"]), image_size=image_size) if midpoint is not None else None
    )
    voronoi_px = np.asarray(
        [world_to_pixel(point[0], point[1], image_size=image_size) for point in voronoi_polyline],
        dtype=float,
    )

    footprint_width_m = 2 * (max(abs(y_coord) for _, y_coord in DEFAULT_R1PRO_NAV_FOOTPRINT) + DEFAULT_R1PRO_NAV_FOOTPRINT_PADDING_M)
    footprint_length_m = (
        max(x_coord for x_coord, _ in DEFAULT_R1PRO_NAV_FOOTPRINT)
        + max(-x_coord for x_coord, _ in DEFAULT_R1PRO_NAV_FOOTPRINT)
        + 2 * DEFAULT_R1PRO_NAV_FOOTPRINT_PADDING_M
    )
    footprint_display_dims = sorted((footprint_width_m, footprint_length_m), reverse=True)
    clearance_radius_m = float(config["vln"]["portal_clearance_radius_m"])
    raw_opening_m = float(active_portal["portal_span"])
    standoff_m = float(config["vln"]["portal_corridor_standoff_m"])
    signed_cross_track_error = float(latest_progress.get("path_signed_cross_track_error", 0.0))
    distance_to_target = math.hypot(last_xy[0] - target_xy[0], last_xy[1] - target_xy[1])

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=180)
    fig.suptitle(
        f"{args.run_dir.name}\nTrajectory + portal geometry + R1Pro footprint overlay",
        fontsize=12,
        y=0.98,
    )

    for ax, title in zip(axes, ("Full Traversability Map", "Door Zoom")):
        ax.imshow(image, cmap="gray", origin="upper")
        ax.plot(traj_px[:, 0], traj_px[:, 1], color="#ff5a5a", linewidth=2.0, label="trajectory")
        ax.scatter(start_px[0], start_px[1], color="#00d084", s=48, zorder=6, label="start")
        ax.scatter(last_px[0], last_px[1], color="#ff9f1a", s=48, zorder=7, label="last pose")
        ax.plot(raw_segment_px[:, 0], raw_segment_px[:, 1], color="#00dfff", linewidth=4.0, label="raw opening")
        if len(voronoi_px) > 1:
            ax.plot(
                voronoi_px[:, 0],
                voronoi_px[:, 1],
                color="#3d8bfd",
                linestyle="--",
                linewidth=2.0,
                alpha=0.95,
                label="cross-room Voronoi edge",
            )
        ax.scatter(target_px[0], target_px[1], marker="x", color="#ffbf00", s=70, zorder=8, label="portal target")
        if source_anchor_px is not None:
            ax.scatter(source_anchor_px[0], source_anchor_px[1], marker="x", color="#c58b00", s=65, zorder=8)
        if midpoint_px is not None:
            ax.scatter(midpoint_px[0], midpoint_px[1], marker="o", color="#ffaa33", s=36, zorder=8)
        ax.add_patch(
            Polygon(footprint_px, closed=True, fill=False, linewidth=2.0, edgecolor="#ff4de3", alpha=0.9, label="padded footprint")
        )
        clearance_px = clearance_radius_m / DEFAULT_LAYOUT_PIXEL_RESOLUTION
        ax.add_patch(
            Circle(last_px, clearance_px, fill=False, linewidth=1.8, linestyle=":", edgecolor="#ff66d1", alpha=0.95, label="clearance radius")
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    door_bounds = crop_world_bounds(
        [
            raw_p0,
            raw_p1,
            last_xy,
            target_xy,
            *((point[0], point[1]) for point in footprint_world),
            *((point[0], point[1]) for point in voronoi_polyline[: min(len(voronoi_polyline), 200)]),
        ],
        margin_m=1.2,
    )
    set_world_window(axes[1], door_bounds, image_size=image_size)

    axes[1].text(
        target_px[0] + 8,
        target_px[1] - 8,
        f"midpoint ({target_xy[0]:.2f}, {target_xy[1]:.2f})",
        fontsize=7,
        color="#ffbf00",
    )
    if source_anchor_px is not None and source_anchor is not None:
        axes[1].text(
            source_anchor_px[0] + 8,
            source_anchor_px[1] - 8,
            f"source ({float(source_anchor['x']):.2f}, {float(source_anchor['y']):.2f})",
            fontsize=7,
            color="#c58b00",
        )
    raw_mid_px = raw_segment_px.mean(axis=0)
    axes[1].text(
        raw_mid_px[0] + 10,
        raw_mid_px[1] - 10,
        f"raw opening {raw_opening_m:.2f}m",
        fontsize=7,
        color="#00dfff",
        bbox={"facecolor": "#222", "alpha": 0.55, "pad": 1.5, "edgecolor": "none"},
    )
    axes[1].text(
        last_px[0] + 10,
        last_px[1] + 14,
        f"footprint {footprint_display_dims[0]:.2f}m x {footprint_display_dims[1]:.2f}m",
        fontsize=7,
        color="#ff9de8",
        bbox={"facecolor": "#222", "alpha": 0.55, "pad": 1.5, "edgecolor": "none"},
    )
    axes[1].text(
        last_px[0] + 10,
        last_px[1] + 30,
        f"clearance r={clearance_radius_m:.2f}m | standoff={standoff_m:.2f}m",
        fontsize=7,
        color="#ff66d1",
        bbox={"facecolor": "#222", "alpha": 0.55, "pad": 1.5, "edgecolor": "none"},
    )
    axes[1].text(
        last_px[0] + 10,
        last_px[1] + 46,
        f"dist to target={distance_to_target:.3f}m | signed XTE={signed_cross_track_error:.3f}m",
        fontsize=7,
        color="#57d66c",
        bbox={"facecolor": "#222", "alpha": 0.55, "pad": 1.5, "edgecolor": "none"},
    )

    axes[1].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "run_dir": str(args.run_dir),
        "output": str(output),
        "layout": str(layout_path),
        "graph_root": str(graph_root),
        "control_step": latest_progress.get("control_step"),
        "controller_mode": latest_progress.get("controller_mode"),
        "tracking_target_waypoint_type": active_portal.get("waypoint_type"),
        "portal_alignment_stage": active_portal.get("portal_alignment_stage"),
        "raw_opening_m": raw_opening_m,
        "portal_gap_m": float(active_portal.get("portal_gap", 0.0)),
        "portal_clearance_radius_m": clearance_radius_m,
        "portal_corridor_standoff_m": standoff_m,
        "footprint_length_m": footprint_length_m,
        "footprint_width_m": footprint_width_m,
        "distance_to_target_m": distance_to_target,
        "signed_cross_track_error_m": signed_cross_track_error,
        "last_pose": {"x": last_xy[0], "y": last_xy[1]},
        "target": {"x": target_xy[0], "y": target_xy[1]},
    }
    summary_path = output.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(str(output))
    print(str(summary_path))


if __name__ == "__main__":
    main()
