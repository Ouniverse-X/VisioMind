from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraspObservation:
    points: np.ndarray
    colors: np.ndarray
    region_mask: np.ndarray | None
    camera_pose_world: np.ndarray
    camera_sensor: str


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        return np.asarray(numpy())
    return np.asarray(value)


def _normalize_name(value: Any) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def _identity_prefix(value: Any) -> str:
    tokens = "".join(ch if ch.isalnum() else " " for ch in str(value).strip().lower()).split()
    return tokens[0] if tokens else ""


def target_mask_from_segmentation(
    segmentation: Any,
    id_to_labels: dict[Any, Any] | None,
    target_name: str,
) -> np.ndarray | None:
    if segmentation is None or not id_to_labels or not str(target_name).strip():
        return None
    seg = np.squeeze(to_numpy(segmentation))
    if seg.ndim != 2:
        raise ValueError(f"segmentation must be HxW, got {seg.shape}")

    wanted = _normalize_name(target_name)
    wanted_prefix = _identity_prefix(target_name)
    matching_ids: list[int] = []
    for raw_id, raw_label in id_to_labels.items():
        label = raw_label
        if isinstance(raw_label, dict):
            label = raw_label.get("name") or raw_label.get("class") or raw_label
        normalized = _normalize_name(label)
        same_prefix = len(wanted_prefix) >= 3 and _identity_prefix(label) == wanted_prefix
        if (
            normalized == wanted
            or normalized.startswith(wanted)
            or wanted.startswith(normalized)
            or same_prefix
        ):
            try:
                matching_ids.append(int(raw_id))
            except (TypeError, ValueError):
                continue
    if not matching_ids:
        return None
    return np.isin(seg, matching_ids)


def target_mask_from_industrial_detector(
    rgb: Any,
    target_name: str,
    detector: Any | None = None,
    conf_threshold: float = 0.35,
) -> np.ndarray | None:
    if rgb is None or not str(target_name).strip():
        return None

    rgb_array = to_numpy(rgb)
    if rgb_array.size == 0 or rgb_array.ndim < 3:
        return None
    if rgb_array.dtype != np.uint8 and float(np.nanmax(rgb_array)) <= 1.0:
        rgb_array = (rgb_array * 255.0).astype(np.uint8)

    if detector is None:
        try:
            from visiomind.perception import IndustrialPartDetector

            detector_weights = (
                Path(__file__).resolve().parents[4] / "models" / "industrial_part_detector.pt"
            )
            detector = IndustrialPartDetector(
                weights_path=detector_weights if detector_weights.exists() else None,
                conf_threshold=conf_threshold,
            )
        except Exception as exc:
            logger.warning("Industrial detector initialization skipped: %s", exc)
            return None

    wanted = _normalize_name(target_name)
    wanted_prefix = _identity_prefix(target_name)
    matched_class = None
    for cls in getattr(detector, "classes", ()):
        if cls == "background":
            continue
        c_norm = _normalize_name(cls)
        if c_norm in wanted or wanted in c_norm or _identity_prefix(cls) == wanted_prefix:
            matched_class = cls
            break

    if not matched_class:
        matched_class = wanted_prefix or wanted

    try:
        res = detector.detect(rgb_array, conf_threshold=conf_threshold)
        det = res.get_highest_confidence(matched_class) or res.get_highest_confidence()
        if det and det.mask is not None and np.any(det.mask):
            return det.mask
    except Exception as exc:
        logger.warning("Industrial detection inference error: %s", exc)

    return None


def dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    radius = max(0, int(radius_px))
    if radius == 0:
        return result.copy()
    padded = np.pad(result, radius, mode="constant", constant_values=False)
    expanded = np.zeros_like(result, dtype=bool)
    height, width = result.shape
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            expanded |= padded[dy : dy + height, dx : dx + width]
    return expanded


def rgbd_to_points(
    depth: Any,
    colors: Any,
    intrinsics: Any,
    *,
    depth_trunc: float,
    target_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    depth_arr = np.squeeze(to_numpy(depth)).astype(np.float32, copy=False)
    color_arr = to_numpy(colors)
    if depth_arr.ndim != 2:
        raise ValueError(f"depth must be HxW, got {depth_arr.shape}")
    if color_arr.ndim != 3 or color_arr.shape[:2] != depth_arr.shape or color_arr.shape[2] < 3:
        raise ValueError(f"colors must be HxWx3 aligned with depth, got {color_arr.shape}")
    color_arr = color_arr[..., :3].astype(np.float32, copy=False)
    if color_arr.size and float(np.nanmax(color_arr)) > 1.0:
        color_arr = color_arr / 255.0
    color_arr = np.clip(color_arr, 0.0, 1.0)

    K = to_numpy(intrinsics).astype(np.float64, copy=False)
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError(f"intrinsics must be a finite 3x3 matrix, got {K}")

    ys, xs = np.indices(depth_arr.shape, dtype=np.float32)
    z = depth_arr
    x = (xs - float(K[0, 2])) * z / float(K[0, 0])
    y = (ys - float(K[1, 2])) * z / float(K[1, 1])
    valid = np.isfinite(z) & (z > 0.0) & (z < float(depth_trunc))
    points = np.stack((x, y, z), axis=-1)[valid].astype(np.float32)
    aligned_colors = color_arr[valid].astype(np.float32)

    aligned_mask = None
    if target_mask is not None:
        pixel_mask = np.asarray(target_mask, dtype=bool)
        if pixel_mask.shape != depth_arr.shape:
            raise ValueError(
                f"target mask shape {pixel_mask.shape} does not match depth {depth_arr.shape}"
            )
        aligned_mask = pixel_mask[valid]
    return points, aligned_colors, aligned_mask


def _quat_pose_matrix(position: Any, quaternion_xyzw: Any) -> np.ndarray:
    position = to_numpy(position).astype(np.float64).reshape(-1)[:3]
    x, y, z, w = to_numpy(quaternion_xyzw).astype(np.float64).reshape(-1)[:4]
    norm = np.linalg.norm([x, y, z, w])
    if norm <= 1e-12:
        raise ValueError("camera quaternion has zero norm")
    x, y, z, w = np.array([x, y, z, w]) / norm
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    matrix[:3, 3] = position
    return matrix


def optical_camera_pose_world(position: Any, quaternion_xyzw: Any) -> np.ndarray:
    world_from_usd = _quat_pose_matrix(position, quaternion_xyzw)
    usd_from_optical = np.eye(4, dtype=np.float64)
    usd_from_optical[:3, :3] = np.diag([1.0, -1.0, -1.0])
    return (world_from_usd @ usd_from_optical).astype(np.float32)


def _find_sensor(robot: Any, sensor_name: str) -> Any:
    sensors = getattr(robot, "sensors", {})
    if sensor_name in sensors:
        return sensors[sensor_name]
    normalized = _normalize_name(sensor_name)
    aliases = {
        "head": ("zedlinkcamera", "headcam"),
        "headcam": ("zedlinkcamera", "headcam"),
        "leftwrist": ("leftrealsenselinkcamera", "leftcam"),
        "leftcam": ("leftrealsenselinkcamera", "leftcam"),
        "rightwrist": ("rightrealsenselinkcamera", "rightcam"),
        "rightcam": ("rightrealsenselinkcamera", "rightcam"),
    }
    markers = aliases.get(normalized, (normalized,))
    matches = [
        sensor
        for name, sensor in sensors.items()
        if any(marker in _normalize_name(name) for marker in markers)
    ]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(f"camera sensor '{sensor_name}' not found; available={sorted(sensors)}")


def _render_sensor_warmup(frames: int) -> bool:
    try:
        import omnigibson as og
    except ImportError:
        return False
    render = getattr(getattr(og, "sim", None), "render", None)
    if not callable(render):
        return False
    for _ in range(max(0, int(frames))):
        render()
    return True


def _read_sensor_with_retries(
    sensor: Any,
    *,
    retries: int,
    warmup_frames: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    attempts = max(1, int(retries))
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            obs, info = sensor.get_obs()
            missing = [
                name
                for name in ("rgb", "depth_linear", "seg_instance")
                if name not in obs or to_numpy(obs[name]).size == 0
            ]
            if missing:
                last_error = RuntimeError(f"empty sensor modalities: {missing}")
            else:
                intrinsics = to_numpy(sensor.intrinsic_matrix).astype(np.float64, copy=False)
                valid_intrinsics = bool(
                    intrinsics.shape == (3, 3)
                    and np.isfinite(intrinsics).all()
                    and intrinsics[0, 0] > 0.0
                    and intrinsics[1, 1] > 0.0
                )
                if valid_intrinsics:
                    return obs, info
                last_error = ValueError(f"invalid sensor intrinsics: {intrinsics}")
        except Exception as exc:
            last_error = exc
        if attempt + 1 < attempts:
            _render_sensor_warmup(max(1, warmup_frames))
    raise RuntimeError(
        f"RGB-D-instance annotators were not ready after {attempts} attempts: {last_error}"
    ) from last_error


def _write_perception_audit(
    audit_dir: str,
    *,
    sensor_name: str,
    rgb: Any,
    depth: Any,
    target_mask: np.ndarray | None,
    aligned_mask: np.ndarray | None,
    points: np.ndarray,
    colors: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose_world: np.ndarray,
    observation_audit: dict[str, Any],
) -> str:
    root = Path(audit_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    sample_dir = root / f"{sensor_name.replace(':', '_').replace('/', '_')}_{time.time_ns()}"
    sample_dir.mkdir(parents=False, exist_ok=False)

    rgb_array = np.squeeze(to_numpy(rgb))
    if rgb_array.ndim == 3 and rgb_array.shape[2] >= 3:
        rgb_uint8 = rgb_array[..., :3]
        if np.nanmax(rgb_uint8) <= 1.0:
            rgb_uint8 = rgb_uint8 * 255.0
        rgb_uint8 = np.clip(rgb_uint8, 0.0, 255.0).astype(np.uint8)
        from PIL import Image

        Image.fromarray(rgb_uint8).save(sample_dir / "rgb.png")
        if target_mask is not None:
            overlay = rgb_uint8.copy()
            overlay[target_mask] = (
                0.5 * overlay[target_mask] + np.array([255, 32, 32]) * 0.5
            ).astype(np.uint8)
            Image.fromarray(overlay).save(sample_dir / "target_overlay.png")

    depth_array = np.squeeze(to_numpy(depth)).astype(np.float32, copy=False)
    np.save(sample_dir / "depth.npy", depth_array)
    finite_depth = depth_array[np.isfinite(depth_array) & (depth_array > 0.0)]
    if finite_depth.size:
        low, high = np.percentile(finite_depth, [2.0, 98.0])
        depth_visual = np.clip((depth_array - low) / max(1e-6, high - low), 0.0, 1.0)
        depth_visual[~np.isfinite(depth_visual)] = 0.0
        from PIL import Image

        Image.fromarray((depth_visual * 255.0).astype(np.uint8)).save(
            sample_dir / "depth_visual.png"
        )
    if target_mask is not None:
        from PIL import Image

        Image.fromarray((target_mask.astype(np.uint8) * 255)).save(sample_dir / "target_mask.png")
    if aligned_mask is not None:
        target_points = points[aligned_mask]
        target_colors = colors[aligned_mask]
    else:
        target_points = np.empty((0, 3), dtype=np.float32)
        target_colors = np.empty((0, 3), dtype=np.float32)
    np.save(sample_dir / "points.npy", points)
    np.save(sample_dir / "colors.npy", colors)
    np.save(sample_dir / "target_points.npy", target_points)
    np.save(sample_dir / "target_colors.npy", target_colors)

    metadata = dict(observation_audit)
    metadata.update(
        {
            "audit_dir": str(sample_dir),
            "camera_pose_world": np.asarray(camera_pose_world, dtype=float).tolist(),
            "target_camera_bounds": (
                None
                if not len(target_points)
                else [target_points.min(axis=0).tolist(), target_points.max(axis=0).tolist()]
            ),
            "target_camera_centroid": (
                None if not len(target_points) else target_points.mean(axis=0).tolist()
            ),
            "target_camera_percentiles": (
                None
                if not len(target_points)
                else {
                    axis: np.percentile(
                        target_points[:, index], [1, 5, 25, 50, 75, 95, 99]
                    ).tolist()
                    for index, axis in enumerate(("x", "y", "z"))
                }
            ),
            "point_count": int(len(points)),
            "target_point_count": int(len(target_points)),
            "intrinsics_matrix": np.asarray(intrinsics, dtype=float).tolist(),
        }
    )
    (sample_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info("AnyGrasp perception audit written to %s", sample_dir)
    return str(sample_dir)


def capture_grasp_observation(
    robot: Any,
    *,
    sensor_name: str,
    target_name: str,
    depth_trunc: float = 2.0,
    mask_dilation_px: int = 1,
    min_target_points: int = 30,
    require_target_mask: bool = True,
    sensor_warmup_frames: int = 3,
    sensor_read_retries: int = 3,
    perception_audit_dir: str | None = None,
    target_depth_outlier_m: float | None = None,
) -> GraspObservation:
    sensor = _find_sensor(robot, sensor_name)
    modalities = set(getattr(sensor, "modalities", ()))
    added_modality = False
    for modality in ("rgb", "depth_linear", "seg_instance"):
        if modality not in modalities:
            sensor.add_modality(modality)
            added_modality = True
    if added_modality:
        _render_sensor_warmup(sensor_warmup_frames)
    sensor_obs, sensor_info = _read_sensor_with_retries(
        sensor,
        retries=sensor_read_retries,
        warmup_frames=sensor_warmup_frames,
    )

    rgb = sensor_obs.get("rgb")
    depth = sensor_obs.get("depth_linear")
    if rgb is None or depth is None:
        raise ValueError(f"sensor '{sensor_name}' did not return rgb and depth_linear")

    instance = sensor_obs.get("seg_instance")
    labels = sensor_info.get("seg_instance", {}) if isinstance(sensor_info, dict) else {}
    target_mask = target_mask_from_segmentation(instance, labels, target_name)
    if target_mask is None:
        target_mask = target_mask_from_industrial_detector(rgb, target_name)
    if target_mask is not None and mask_dilation_px:
        target_mask = dilate_mask(target_mask, mask_dilation_px)
    if target_mask is None and require_target_mask:
        raise ValueError(f"target '{target_name}' is not visible in {sensor_name} instance mask")
    if target_mask is not None and target_depth_outlier_m is not None:
        outlier_limit = float(target_depth_outlier_m)
        if not np.isfinite(outlier_limit) or outlier_limit <= 0.0:
            raise ValueError("target_depth_outlier_m must be finite and positive")
        depth_array = np.squeeze(to_numpy(depth)).astype(np.float32, copy=False)
        target_depth = depth_array[target_mask]
        finite_target_depth = target_depth[np.isfinite(target_depth) & (target_depth > 0.0)]
        if finite_target_depth.size:
            median_depth = float(np.median(finite_target_depth))
            target_mask &= np.abs(depth_array - median_depth) <= outlier_limit
            logger.info(
                "AnyGrasp target depth filter: median=%.4f limit=%.4f kept=%d",
                median_depth,
                outlier_limit,
                int(np.count_nonzero(target_mask)),
            )

    points, colors, aligned_mask = rgbd_to_points(
        depth,
        rgb,
        sensor.intrinsic_matrix,
        depth_trunc=depth_trunc,
        target_mask=target_mask,
    )
    depth_array = np.squeeze(to_numpy(depth))
    finite_positive = depth_array[np.isfinite(depth_array) & (depth_array > 0.0)]
    intrinsics = to_numpy(sensor.intrinsic_matrix).astype(np.float64, copy=False)
    observation_audit = {
        "event": "anygrasp_observation_audit",
        "sensor": sensor_name,
        "depth_dtype": str(depth_array.dtype),
        "depth_shape": list(depth_array.shape),
        "finite_positive_depth_min": (
            None if not finite_positive.size else float(finite_positive.min())
        ),
        "finite_positive_depth_max": (
            None if not finite_positive.size else float(finite_positive.max())
        ),
        "depth_trunc": float(depth_trunc),
        "valid_depth_count": int(len(points)),
        "target_pixel_count": (None if target_mask is None else int(np.count_nonzero(target_mask))),
        "target_aligned_count": (
            None if aligned_mask is None else int(np.count_nonzero(aligned_mask))
        ),
        "intrinsics": {
            "fx": float(intrinsics[0, 0]),
            "fy": float(intrinsics[1, 1]),
            "cx": float(intrinsics[0, 2]),
            "cy": float(intrinsics[1, 2]),
        },
        "depth_unit_assumption": "meters_linear_optical_z",
    }
    logger.info("%s", json.dumps(observation_audit, sort_keys=True, separators=(",", ":")))
    if len(points) == 0:
        raise ValueError(f"sensor '{sensor_name}' produced no valid depth points")
    position, orientation = sensor.get_position_orientation()
    camera_pose_world = optical_camera_pose_world(position, orientation)
    if perception_audit_dir:
        audit_path = _write_perception_audit(
            perception_audit_dir,
            sensor_name=sensor_name,
            rgb=rgb,
            depth=depth,
            target_mask=target_mask,
            aligned_mask=aligned_mask,
            points=points,
            colors=colors,
            intrinsics=intrinsics,
            camera_pose_world=camera_pose_world,
            observation_audit=observation_audit,
        )
        observation_audit["audit_dir"] = audit_path
    if aligned_mask is not None and int(aligned_mask.sum()) < int(min_target_points):
        raise ValueError(
            f"target '{target_name}' has only {int(aligned_mask.sum())} valid points "
            f"(< {int(min_target_points)})"
        )

    return GraspObservation(
        points=points,
        colors=colors,
        region_mask=aligned_mask,
        camera_pose_world=camera_pose_world,
        camera_sensor=sensor_name,
    )
