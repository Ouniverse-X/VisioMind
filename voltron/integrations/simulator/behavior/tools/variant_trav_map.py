"""Generate a traversability map for a BEHAVIOR scene variant from a base map.

This script is intentionally narrow: it derives a new traversability map by
expanding existing door openings according to width changes encoded in a scene
variant JSON. It keeps the original layout segmentation maps intact and only
patches the traversability image locally around doors whose width changed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

LAYOUT_PIXEL_RESOLUTION_M = 0.01


@dataclass(frozen=True)
class DoorDelta:
    name: str
    category: str
    model: str
    center_xy_m: tuple[float, float]
    yaw_rad: float
    width_delta_m: float
    width_base_m: float
    width_variant_m: float


@dataclass(frozen=True)
class OpeningOverride:
    center_xy_m: tuple[float, float] | None = None
    yaw_rad: float | None = None
    target_opening_width_m: float | None = None


def generate_variant_trav_map(
    *,
    base_scene_file: str | Path,
    variant_scene_file: str | Path,
    behavior_assets_root: str | Path,
    base_trav_map_path: str | Path,
    output_trav_map_path: str | Path,
    opening_overrides_file: str | Path | None = None,
    carve_wall_thickness_m: float = 0.18,
    min_opening_width_m: float = 0.6,
) -> dict[str, Any]:
    base_scene_path = Path(base_scene_file).expanduser().resolve()
    variant_scene_path = Path(variant_scene_file).expanduser().resolve()
    assets_root = Path(behavior_assets_root).expanduser().resolve()
    base_trav_path = Path(base_trav_map_path).expanduser().resolve()
    output_trav_path = Path(output_trav_map_path).expanduser().resolve()
    overrides_path = Path(opening_overrides_file).expanduser().resolve() if opening_overrides_file else None

    base_scene = json.loads(base_scene_path.read_text(encoding="utf-8"))
    variant_scene = json.loads(variant_scene_path.read_text(encoding="utf-8"))
    trav_map = np.asarray(Image.open(base_trav_path).convert("L"))
    opening_overrides = _load_opening_overrides(overrides_path)

    door_deltas = _collect_door_deltas(
        base_scene=base_scene,
        variant_scene=variant_scene,
        behavior_assets_root=assets_root,
    )

    patched_map = trav_map.copy()
    applied: list[dict[str, Any]] = []
    for door in door_deltas:
        opening_override = opening_overrides.get(door.name)
        opening_center_xy = opening_override.center_xy_m if opening_override and opening_override.center_xy_m else door.center_xy_m
        opening_yaw_rad = (
            opening_override.yaw_rad
            if opening_override and opening_override.yaw_rad is not None
            else door.yaw_rad
        )
        opening = _measure_opening_width_px(
            trav_map=patched_map,
            center_rc=_world_to_map_rc(opening_center_xy, map_size=patched_map.shape[0]),
            tangent_rc=_world_to_map_direction_rc(_door_tangent_world(opening_yaw_rad)),
            search_half_width_px=max(
                int(round(door.width_base_m / (2.0 * LAYOUT_PIXEL_RESOLUTION_M))) + 40,
                60,
            ),
            normal_half_thickness_px=max(int(round(0.10 / LAYOUT_PIXEL_RESOLUTION_M)), 8),
        )
        min_half_width_px = max(int(round(min_opening_width_m / (2.0 * LAYOUT_PIXEL_RESOLUTION_M))), 1)
        override_target_width_m = (
            opening_override.target_opening_width_m
            if opening_override and opening_override.target_opening_width_m is not None
            else None
        )
        if override_target_width_m is not None:
            target_half_width_px = max(
                int(round(override_target_width_m / (2.0 * LAYOUT_PIXEL_RESOLUTION_M))),
                min_half_width_px,
            )
        else:
            delta_half_width_px = max(int(round(door.width_delta_m / (2.0 * LAYOUT_PIXEL_RESOLUTION_M))), 0)
            existing_half_width_px = max(
                opening["half_width_px"],
                int(np.ceil(max(opening["width_px"] - 1, 0) / 2.0)),
            )
            target_half_width_px = max(existing_half_width_px + delta_half_width_px, min_half_width_px)
        _carve_oriented_opening(
            trav_map=patched_map,
            center_rc=opening["center_rc"],
            tangent_rc=opening["tangent_rc"],
            half_width_px=target_half_width_px,
            half_thickness_px=max(int(round(carve_wall_thickness_m / (2.0 * LAYOUT_PIXEL_RESOLUTION_M))), 10),
        )
        applied.append(
            {
                "door": door.name,
                "category": door.category,
                "model": door.model,
                "center_xy_m": [door.center_xy_m[0], door.center_xy_m[1]],
                "yaw_rad": door.yaw_rad,
                "opening_center_xy_m": [opening_center_xy[0], opening_center_xy[1]],
                "opening_yaw_rad": opening_yaw_rad,
                "width_base_m": door.width_base_m,
                "width_variant_m": door.width_variant_m,
                "width_delta_m": door.width_delta_m,
                "opening_before_px": opening["width_px"],
                "opening_after_px": 2 * target_half_width_px + 1,
                "opening_override": {
                    "center_xy_m": list(opening_override.center_xy_m) if opening_override and opening_override.center_xy_m else None,
                    "yaw_rad": opening_override.yaw_rad if opening_override else None,
                    "target_opening_width_m": override_target_width_m,
                }
                if opening_override
                else None,
            }
        )

    output_trav_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(patched_map.astype(np.uint8), mode="L").save(output_trav_path)
    return {
        "base_scene_file": str(base_scene_path),
        "variant_scene_file": str(variant_scene_path),
        "base_trav_map": str(base_trav_path),
        "output_trav_map": str(output_trav_path),
        "opening_overrides_file": str(overrides_path) if overrides_path else None,
        "door_count": len(door_deltas),
        "applied": applied,
    }


def _collect_door_deltas(
    *,
    base_scene: dict[str, Any],
    variant_scene: dict[str, Any],
    behavior_assets_root: Path,
) -> list[DoorDelta]:
    base_init = base_scene.get("objects_info", {}).get("init_info", {})
    variant_init = variant_scene.get("objects_info", {}).get("init_info", {})
    variant_state = variant_scene.get("state", {}).get("registry", {}).get("object_registry", {})

    deltas: list[DoorDelta] = []
    for object_name in sorted(set(base_init) & set(variant_init)):
        base_args = base_init[object_name].get("args", {})
        variant_args = variant_init[object_name].get("args", {})
        category = str(variant_args.get("category") or "")
        if category not in {"door", "sliding_door"}:
            continue

        model = str(variant_args.get("model") or "")
        bbox_size = _load_bbox_size(behavior_assets_root=behavior_assets_root, category=category, model=model)
        base_scale = _scale_axis(variant=base_args, axis=1)
        variant_scale = _scale_axis(variant=variant_args, axis=1)
        width_base_m = bbox_size[1] * base_scale
        width_variant_m = bbox_size[1] * variant_scale
        width_delta_m = width_base_m - width_variant_m
        if width_delta_m <= 1e-6:
            continue

        root_link = variant_state.get(object_name, {}).get("root_link", {})
        pos = root_link.get("pos")
        ori = root_link.get("ori")
        if not isinstance(pos, list) or len(pos) < 2 or not isinstance(ori, list) or len(ori) < 4:
            continue

        deltas.append(
            DoorDelta(
                name=object_name,
                category=category,
                model=model,
                center_xy_m=(float(pos[0]), float(pos[1])),
                yaw_rad=_yaw_from_quaternion(ori),
                width_delta_m=float(width_delta_m),
                width_base_m=float(width_base_m),
                width_variant_m=float(width_variant_m),
            )
        )
    return deltas


def _load_opening_overrides(path: Path | None) -> dict[str, OpeningOverride]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Opening overrides must be a JSON object: {path}")

    overrides: dict[str, OpeningOverride] = {}
    for door_name, raw_override in payload.items():
        if not isinstance(raw_override, dict):
            raise ValueError(f"Opening override for {door_name} must be an object")
        center_xy_raw = raw_override.get("center_xy_m")
        center_xy_m = None
        if center_xy_raw is not None:
            if not isinstance(center_xy_raw, list) or len(center_xy_raw) < 2:
                raise ValueError(f"center_xy_m for {door_name} must be a [x, y] list")
            center_xy_m = (float(center_xy_raw[0]), float(center_xy_raw[1]))
        yaw_rad_raw = raw_override.get("yaw_rad")
        target_width_raw = raw_override.get("target_opening_width_m")
        overrides[str(door_name)] = OpeningOverride(
            center_xy_m=center_xy_m,
            yaw_rad=float(yaw_rad_raw) if yaw_rad_raw is not None else None,
            target_opening_width_m=float(target_width_raw) if target_width_raw is not None else None,
        )
    return overrides


def _load_bbox_size(*, behavior_assets_root: Path, category: str, model: str) -> tuple[float, float, float]:
    metadata_path = behavior_assets_root / "objects" / category / model / "misc" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    bbox_size = metadata.get("bbox_size")
    if not isinstance(bbox_size, list) or len(bbox_size) < 3:
        raise ValueError(f"Invalid bbox_size in {metadata_path}")
    return float(bbox_size[0]), float(bbox_size[1]), float(bbox_size[2])


def _scale_axis(*, variant: dict[str, Any], axis: int) -> float:
    scale = variant.get("scale")
    if not isinstance(scale, list) or len(scale) <= axis:
        raise ValueError(f"Missing scale[{axis}] in object args: {variant}")
    return float(scale[axis])


def _yaw_from_quaternion(quat_xyzw: list[float]) -> float:
    x_coord, y_coord, z_coord, w_coord = [float(value) for value in quat_xyzw[:4]]
    siny_cosp = 2.0 * (w_coord * z_coord + x_coord * y_coord)
    cosy_cosp = 1.0 - 2.0 * (y_coord * y_coord + z_coord * z_coord)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _door_tangent_world(yaw_rad: float) -> np.ndarray:
    return np.asarray([-np.sin(yaw_rad), np.cos(yaw_rad)], dtype=float)


def _world_to_map_rc(xy_world_m: tuple[float, float], *, map_size: int) -> np.ndarray:
    x_coord, y_coord = xy_world_m
    row = y_coord / LAYOUT_PIXEL_RESOLUTION_M + map_size / 2.0
    col = x_coord / LAYOUT_PIXEL_RESOLUTION_M + map_size / 2.0
    return np.asarray([row, col], dtype=float)


def _world_to_map_direction_rc(direction_xy: np.ndarray) -> np.ndarray:
    return np.asarray([float(direction_xy[1]), float(direction_xy[0])], dtype=float)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return np.asarray([1.0, 0.0], dtype=float)
    return vec / norm


def _sample_line_support(
    *,
    trav_map: np.ndarray,
    center_rc: np.ndarray,
    tangent_rc: np.ndarray,
    offset_px: int,
    normal_half_thickness_px: int,
) -> bool:
    tangent = _normalize(tangent_rc)
    normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
    point = center_rc + tangent * float(offset_px)
    samples: list[int] = []
    for normal_offset in range(-normal_half_thickness_px, normal_half_thickness_px + 1):
        sample = point + normal * float(normal_offset)
        row = int(round(sample[0]))
        col = int(round(sample[1]))
        if row < 0 or row >= trav_map.shape[0] or col < 0 or col >= trav_map.shape[1]:
            continue
        samples.append(int(trav_map[row, col]))
    if not samples:
        return False
    return float(sum(value == 255 for value in samples)) / float(len(samples)) >= 0.45


def _measure_opening_width_px(
    *,
    trav_map: np.ndarray,
    center_rc: np.ndarray,
    tangent_rc: np.ndarray,
    search_half_width_px: int,
    normal_half_thickness_px: int,
) -> dict[str, Any]:
    support = []
    offsets = range(-search_half_width_px, search_half_width_px + 1)
    for offset_px in offsets:
        support.append(
            _sample_line_support(
                trav_map=trav_map,
                center_rc=center_rc,
                tangent_rc=tangent_rc,
                offset_px=offset_px,
                normal_half_thickness_px=normal_half_thickness_px,
            )
        )

    center_index = search_half_width_px
    left = center_index
    while left > 0 and support[left - 1]:
        left -= 1
    right = center_index
    while right < len(support) - 1 and support[right + 1]:
        right += 1

    if not support[center_index]:
        supported_indices = [index for index, is_supported in enumerate(support) if is_supported]
        if supported_indices:
            closest = min(
                supported_indices,
                key=lambda index: abs(index - center_index),
            )
            left = right = closest
            while left > 0 and support[left - 1]:
                left -= 1
            while right < len(support) - 1 and support[right + 1]:
                right += 1
    if right >= left:
        run_mid_index = (left + right) / 2.0
        center_rc = center_rc + _normalize(tangent_rc) * float(run_mid_index - center_index)
    width_px = right - left + 1 if right >= left else 1
    half_width_px = max(int(np.floor(max(width_px - 1, 0) / 2.0)), 0)
    return {
        "center_rc": center_rc,
        "tangent_rc": _normalize(tangent_rc),
        "half_width_px": half_width_px,
        "width_px": width_px,
    }


def _carve_oriented_opening(
    *,
    trav_map: np.ndarray,
    center_rc: np.ndarray,
    tangent_rc: np.ndarray,
    half_width_px: int,
    half_thickness_px: int,
) -> None:
    tangent = _normalize(tangent_rc)
    normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
    radius = int(np.ceil(np.hypot(half_width_px, half_thickness_px))) + 2
    row_center = int(round(center_rc[0]))
    col_center = int(round(center_rc[1]))
    row_min = max(row_center - radius, 0)
    row_max = min(row_center + radius + 1, trav_map.shape[0])
    col_min = max(col_center - radius, 0)
    col_max = min(col_center + radius + 1, trav_map.shape[1])

    rows, cols = np.meshgrid(
        np.arange(row_min, row_max, dtype=float),
        np.arange(col_min, col_max, dtype=float),
        indexing="ij",
    )
    delta = np.stack((rows - center_rc[0], cols - center_rc[1]), axis=-1)
    tangent_proj = np.abs(delta[..., 0] * tangent[0] + delta[..., 1] * tangent[1])
    normal_proj = np.abs(delta[..., 0] * normal[0] + delta[..., 1] * normal[1])
    mask = (tangent_proj <= float(half_width_px)) & (normal_proj <= float(half_thickness_px))
    trav_map[row_min:row_max, col_min:col_max][mask] = 255


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-scene-file", required=True, help="Base scene JSON used by the original trav map.")
    parser.add_argument("--variant-scene-file", required=True, help="Variant scene JSON with modified door widths.")
    parser.add_argument("--behavior-assets-root", required=True, help="Root directory containing behavior-1k-assets.")
    parser.add_argument("--base-trav-map", required=True, help="Base traversability map PNG.")
    parser.add_argument("--output-trav-map", required=True, help="Output traversability map PNG.")
    parser.add_argument(
        "--opening-overrides-file",
        default=None,
        help="Optional JSON file with per-door opening center / yaw / target width overrides.",
    )
    parser.add_argument(
        "--carve-wall-thickness-m",
        type=float,
        default=0.18,
        help="Thickness of the local wall band to carve around each widened door.",
    )
    parser.add_argument(
        "--min-opening-width-m",
        type=float,
        default=0.6,
        help="Lower bound on the resulting opening width to avoid collapsing narrow measurements.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = generate_variant_trav_map(
        base_scene_file=args.base_scene_file,
        variant_scene_file=args.variant_scene_file,
        behavior_assets_root=args.behavior_assets_root,
        base_trav_map_path=args.base_trav_map,
        output_trav_map_path=args.output_trav_map,
        opening_overrides_file=args.opening_overrides_file,
        carve_wall_thickness_m=args.carve_wall_thickness_m,
        min_opening_width_m=args.min_opening_width_m,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
