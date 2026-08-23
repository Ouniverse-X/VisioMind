"""Validated local / HTTP adapter for the licensed AnyGrasp SDK."""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GraspCandidate:
    """A single 6-DoF parallel-jaw grasp in OpenCV camera coordinates."""

    score: float
    translation: np.ndarray
    rotation_matrix: np.ndarray
    width: float
    depth: float
    height: float

    @property
    def approach_direction(self) -> np.ndarray:
        return self.rotation_matrix[:, 0]

    @property
    def tip_position(self) -> np.ndarray:
        return self.translation + self.depth * self.approach_direction

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": float(self.score),
            "translation": self.translation.tolist(),
            "rotation_matrix": self.rotation_matrix.tolist(),
            "width": float(self.width),
            "depth": float(self.depth),
            "height": float(self.height),
            "tip_position": self.tip_position.tolist(),
            "approach_direction": self.approach_direction.tolist(),
        }


def _finite_range(values: Any) -> list[float] | None:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return None
    return [float(finite.min()), float(finite.max())]


def candidate_distribution_summary(
    candidates: Any,
    preferred_approach: Any | None = None,
    *,
    max_depth_unique: int = 16,
) -> dict[str, Any]:
    """Return a bounded, JSON-safe summary without changing candidate order or values."""
    if hasattr(candidates, "scores"):
        scores = np.asarray(candidates.scores, dtype=np.float64).reshape(-1)
        widths = np.asarray(candidates.widths, dtype=np.float64).reshape(-1)
        depths = np.asarray(candidates.depths, dtype=np.float64).reshape(-1)
        rotations = np.asarray(candidates.rotation_matrices, dtype=np.float64)
        approaches = rotations[:, :, 0] if rotations.ndim == 3 else np.empty((0, 3))
    else:
        items = list(candidates or [])
        scores = np.asarray([item.score for item in items], dtype=np.float64)
        widths = np.asarray([item.width for item in items], dtype=np.float64)
        depths = np.asarray([item.depth for item in items], dtype=np.float64)
        approaches = np.asarray(
            [item.approach_direction for item in items], dtype=np.float64
        ).reshape(-1, 3)

    finite_depths = depths[np.isfinite(depths)]
    unique_depths = np.unique(finite_depths)
    limit = max(0, int(max_depth_unique))
    summary: dict[str, Any] = {
        "count": int(len(scores)),
        "score_range": _finite_range(scores),
        "width_range": _finite_range(widths),
        "depth_range": _finite_range(depths),
        "depth_unique": [float(value) for value in unique_depths[:limit]],
        "depth_unique_count": int(len(unique_depths)),
        "depth_unique_truncated": bool(len(unique_depths) > limit),
        "approach_component_ranges": {
            axis: _finite_range(approaches[:, index]) if len(approaches) else None
            for index, axis in enumerate(("x", "y", "z"))
        },
    }
    if preferred_approach is not None:
        preferred = np.asarray(preferred_approach, dtype=np.float64).reshape(-1)
        if preferred.shape == (3,) and np.isfinite(preferred).all():
            preferred_norm = float(np.linalg.norm(preferred))
            norms = np.linalg.norm(approaches, axis=1) if len(approaches) else np.empty(0)
            valid = np.isfinite(approaches).all(axis=1) & (norms > 1e-12)
            if preferred_norm > 1e-12 and valid.any():
                dots = (approaches[valid] / norms[valid, None]) @ (
                    preferred / preferred_norm
                )
                angles = np.rad2deg(np.arccos(np.clip(dots, -1.0, 1.0)))
                quantiles = np.quantile(angles, [0.0, 0.25, 0.5, 0.75, 1.0])
                summary["preferred_approach_angle_deg_quantiles"] = {
                    name: float(value)
                    for name, value in zip(
                        ("min", "p25", "p50", "p75", "max"), quantiles
                    )
                }
            else:
                summary["preferred_approach_angle_deg_quantiles"] = None
    return summary


@dataclass
class AnyGraspConfig:
    sdk_root: str = ""
    checkpoint_path: str = ""
    license_dir: str = ""
    endpoint: str = ""
    max_gripper_width: float = 0.1
    gripper_height: float = 0.03
    top_down_grasp: bool = False
    dense_grasp: bool = False
    collision_detection: bool = True
    apply_nms: bool = True
    top_k: int = 10
    request_timeout_s: float = 60.0
    region_margin: float = 0.04

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "AnyGraspConfig":
        return cls(**{key: value for key, value in cfg.items() if key in cls.__dataclass_fields__})


def validate_detection_inputs(
    points: Any,
    colors: Any,
    region_mask: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Validate and normalize the arrays shared by local and remote modes."""
    points_arr = np.ascontiguousarray(points, dtype=np.float32)
    colors_arr = np.ascontiguousarray(colors, dtype=np.float32)
    if points_arr.ndim != 2 or points_arr.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points_arr.shape}")
    if colors_arr.shape != points_arr.shape:
        raise ValueError(f"colors must match points shape {points_arr.shape}, got {colors_arr.shape}")
    if len(points_arr) == 0:
        raise ValueError("point cloud is empty")
    if not np.isfinite(points_arr).all() or not np.isfinite(colors_arr).all():
        raise ValueError("points and colors must contain only finite values")
    if float(colors_arr.min()) < 0.0 or float(colors_arr.max()) > 1.0:
        raise ValueError("colors must be float RGB values in [0, 1]")

    mask_arr = None
    if region_mask is not None:
        mask_arr = np.ascontiguousarray(region_mask, dtype=np.bool_).reshape(-1)
        if mask_arr.shape != (len(points_arr),):
            raise ValueError(f"region_mask must have shape ({len(points_arr)},), got {mask_arr.shape}")
        if not mask_arr.any():
            raise ValueError("region_mask contains no target points")
    return points_arr, colors_arr, mask_arr


def workspace_limits(
    points: np.ndarray,
    region_mask: np.ndarray | None,
    margin: float,
) -> list[float] | None:
    """Build an SDK workspace around target points while retaining scene geometry."""
    if region_mask is None:
        return None
    target = points[region_mask]
    lower = target.min(axis=0) - max(0.0, float(margin))
    upper = target.max(axis=0) + max(0.0, float(margin))
    return [
        float(lower[0]), float(upper[0]),
        float(lower[1]), float(upper[1]),
        float(lower[2]), float(upper[2]),
    ]


def grasp_group_to_candidates(group: Any) -> list[GraspCandidate]:
    candidates: list[GraspCandidate] = []
    for index in range(len(group)):
        candidate = GraspCandidate(
            score=float(group.scores[index]),
            translation=np.asarray(group.translations[index], dtype=np.float32).copy(),
            rotation_matrix=np.asarray(group.rotation_matrices[index], dtype=np.float32).copy(),
            width=float(group.widths[index]),
            depth=float(group.depths[index]),
            height=float(group.heights[index]),
        )
        if (
            np.isfinite(candidate.translation).all()
            and np.isfinite(candidate.rotation_matrix).all()
            and np.isfinite(candidate.score)
        ):
            candidates.append(candidate)
    return candidates


def filter_candidates_by_approach(
    candidates: list[GraspCandidate],
    approach_direction: Any | None,
    approach_thresh: float,
) -> list[GraspCandidate]:
    """Filter grasps by angle to a preferred camera-frame approach vector."""
    if approach_direction is None or float(approach_thresh) >= np.pi:
        return candidates
    preferred = np.asarray(approach_direction, dtype=np.float64).reshape(-1)
    if preferred.shape != (3,) or not np.isfinite(preferred).all():
        raise ValueError("approach_direction must be a finite three-vector")
    norm = float(np.linalg.norm(preferred))
    if norm <= 1e-8:
        raise ValueError("approach_direction must be non-zero")
    preferred /= norm
    threshold = max(0.0, float(approach_thresh))
    kept: list[GraspCandidate] = []
    for candidate in candidates:
        approach = np.asarray(candidate.approach_direction, dtype=np.float64)
        approach /= np.linalg.norm(approach) + 1e-12
        angle = float(np.arccos(np.clip(np.dot(approach, preferred), -1.0, 1.0)))
        if angle <= threshold:
            kept.append(candidate)
    return kept


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class AnyGraspDetector:
    """Lazy AnyGrasp detector with identical local and remote semantics."""

    def __init__(self, config: AnyGraspConfig | dict[str, Any]) -> None:
        self._config = AnyGraspConfig.from_dict(config) if isinstance(config, dict) else config
        self._detector: Any | None = None
        self._remote = bool(self._config.endpoint)
        self.last_detection_audit: dict[str, Any] | None = None

    def _remote_url(self, path: str) -> str:
        return f"{self._config.endpoint.rstrip('/')}{path}"

    def _ensure_loaded(self) -> None:
        if self._detector is not None:
            return
        sdk_root = Path(self._config.sdk_root).expanduser().resolve()
        detection_dir = sdk_root / "grasp_detection"
        if not detection_dir.is_dir():
            raise RuntimeError(f"AnyGrasp detection directory not found: {detection_dir}")
        checkpoint = Path(self._config.checkpoint_path).expanduser() if self._config.checkpoint_path else detection_dir / "log" / "checkpoint_detection.tar"
        if not checkpoint.is_absolute():
            checkpoint = (detection_dir / checkpoint).resolve()
        if not checkpoint.is_file():
            raise RuntimeError(f"AnyGrasp checkpoint not found: {checkpoint}")
        if str(detection_dir) not in sys.path:
            sys.path.insert(0, str(detection_dir))

        from argparse import Namespace
        try:
            with _working_directory(detection_dir):
                from gsnet import AnyGrasp  # type: ignore[import-not-found]
                cfg = Namespace(
                    checkpoint_path=str(checkpoint),
                    max_gripper_width=min(0.1, max(0.0, self._config.max_gripper_width)),
                    gripper_height=self._config.gripper_height,
                    top_down_grasp=bool(self._config.top_down_grasp),
                    debug=False,
                )
                detector = AnyGrasp(cfg)
                detector.load_net()
        except Exception as exc:
            raise RuntimeError(f"failed to load AnyGrasp SDK from {detection_dir}: {exc}") from exc
        self._detector = detector
        logger.info("AnyGrasp local detector loaded (checkpoint=%s)", checkpoint)


    def _detect_remote(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        *,
        region_mask: np.ndarray | None,
        approach_direction: Any | None,
        approach_thresh: float,
        dense_grasp: bool,
        collision_detection: bool,
        top_k: int,
    ) -> list[GraspCandidate]:
        import requests

        payload: dict[str, Any] = {
            "points_b64": base64.b64encode(points.tobytes()).decode("ascii"),
            "points_shape": list(points.shape),
            "colors_b64": base64.b64encode(colors.tobytes()).decode("ascii"),
            "colors_shape": list(colors.shape),
            "dense_grasp": dense_grasp,
            "collision_detection": collision_detection,
            "apply_nms": bool(self._config.apply_nms),
            "top_k": top_k,
            "region_margin": float(self._config.region_margin),
            "approach_thresh": float(approach_thresh),
        }
        if region_mask is not None:
            payload["region_mask_b64"] = base64.b64encode(region_mask.tobytes()).decode("ascii")
            payload["region_mask_shape"] = list(region_mask.shape)
        if approach_direction is not None:
            payload["approach_direction"] = np.asarray(approach_direction, dtype=float).tolist()

        response = requests.post(
            self._remote_url("/detect"),
            json=payload,
            timeout=float(self._config.request_timeout_s),
        )
        response.raise_for_status()
        response_payload = response.json()
        candidates = []
        for item in response_payload.get("candidates", []):
            candidates.append(
                GraspCandidate(
                    score=float(item["score"]),
                    translation=np.asarray(item["translation"], dtype=np.float32),
                    rotation_matrix=np.asarray(item["rotation_matrix"], dtype=np.float32),
                    width=float(item["width"]),
                    depth=float(item["depth"]),
                    height=float(item["height"]),
                )
            )
        server_audit = response_payload.get("audit")
        if isinstance(server_audit, dict):
            self.last_detection_audit = server_audit
        else:
            self.last_detection_audit = {
                "event": "anygrasp_detection_audit",
                "mode": "remote",
                "server_audit_available": False,
                "network_raw_count": None,
                "network_raw_available": False,
                "network_raw_unavailable_reason": (
                    "remote response omitted optional audit (legacy server compatible)"
                ),
                "post_top_k_count": int(len(candidates)),
                "post_top_k_distribution": candidate_distribution_summary(
                    candidates, approach_direction
                ),
            }
        logger.info(
            "%s",
            json.dumps(self.last_detection_audit, sort_keys=True, separators=(",", ":")),
        )
        return candidates

    def detect(
        self,
        points: Any,
        colors: Any,
        *,
        region_mask: Any | None = None,
        approach_direction: Any | None = None,
        approach_thresh: float = np.pi,
        dense_grasp: bool | None = None,
        collision_detection: bool | None = None,
        top_k: int | None = None,
    ) -> list[GraspCandidate]:
        self.last_detection_audit = None
        points_arr, colors_arr, mask_arr = validate_detection_inputs(points, colors, region_mask)
        dense = self._config.dense_grasp if dense_grasp is None else bool(dense_grasp)
        collision = self._config.collision_detection if collision_detection is None else bool(collision_detection)
        limit = self._config.top_k if top_k is None else int(top_k)
        if not 1 <= limit <= 100:
            raise ValueError(f"top_k must be in [1, 100], got {limit}")
        if self._remote:
            return self._detect_remote(
                points_arr,
                colors_arr,
                region_mask=mask_arr,
                approach_direction=approach_direction,
                approach_thresh=approach_thresh,
                dense_grasp=dense,
                collision_detection=collision,
                top_k=limit,
            )

        self._ensure_loaded()
        limits = workspace_limits(points_arr, mask_arr, self._config.region_margin)
        group, _cloud = self._detector.get_grasp(
            points_arr,
            colors_arr,
            lims=limits,
            apply_object_mask=True,
            dense_grasp=dense,
            collision_detection=collision,
        )
        sdk_count = 0 if group is None else int(len(group))
        audit: dict[str, Any] = {
            "event": "anygrasp_detection_audit",
            "mode": "local",
            "network_raw_count": None,
            "network_raw_available": False,
            "network_raw_unavailable_reason": (
                "raw network proposals are internal to compiled gsnet.so"
            ),
            "input_point_count": int(len(points_arr)),
            "target_point_count": (
                None if mask_arr is None else int(np.count_nonzero(mask_arr))
            ),
            "workspace": limits,
            "dense_grasp": dense,
            "collision_detection": collision,
            "apply_nms": bool(self._config.apply_nms),
            "nms_applied": bool(self._config.apply_nms and not dense),
            "top_k": limit,
            "approach_filter": {
                "preferred": (
                    None
                    if approach_direction is None
                    else np.asarray(approach_direction, dtype=float).reshape(-1).tolist()
                ),
                "threshold_rad": float(approach_thresh),
            },
            "sdk_returned_count": sdk_count,
            "sdk_returned_definition": "count returned by compiled SDK get_grasp",
            "sdk_returned_distribution": (
                candidate_distribution_summary(group, approach_direction)
                if sdk_count
                else candidate_distribution_summary([], approach_direction)
            ),
        }
        if group is None or len(group) == 0:
            for stage in ("post_nms", "post_conversion", "post_approach", "post_top_k"):
                audit[f"{stage}_count"] = 0
                audit[f"{stage}_distribution"] = candidate_distribution_summary(
                    [], approach_direction
                )
            self.last_detection_audit = audit
            logger.info("%s", json.dumps(audit, sort_keys=True, separators=(",", ":")))
            return []
        if self._config.apply_nms and not dense:
            group = group.nms()
        audit["post_nms_count"] = int(len(group))
        audit["post_nms_distribution"] = candidate_distribution_summary(
            group, approach_direction
        )
        group = group.sort_by_score()
        candidates = grasp_group_to_candidates(group)
        audit["post_conversion_count"] = int(len(candidates))
        audit["post_conversion_distribution"] = candidate_distribution_summary(
            candidates, approach_direction
        )
        candidates = filter_candidates_by_approach(
            candidates, approach_direction, approach_thresh
        )
        audit["post_approach_count"] = int(len(candidates))
        audit["post_approach_distribution"] = candidate_distribution_summary(
            candidates, approach_direction
        )
        candidates = candidates[:limit]
        audit["post_top_k_count"] = int(len(candidates))
        audit["post_top_k_distribution"] = candidate_distribution_summary(
            candidates, approach_direction
        )
        self.last_detection_audit = audit
        logger.info("%s", json.dumps(audit, sort_keys=True, separators=(",", ":")))
        return candidates


    def detect_from_depth(
        self,
        depth_image: np.ndarray,
        colors: np.ndarray,
        *,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        scale: float = 1000.0,
        depth_trunc: float = 1.5,
        region_mask: np.ndarray | None = None,
        **kwargs: Any,
    ) -> list[GraspCandidate]:
        from .observation import rgbd_to_points

        depth_m = np.asarray(depth_image, dtype=np.float32) / float(scale)
        intrinsics = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        points, aligned_colors, aligned_mask = rgbd_to_points(
            depth_m,
            colors,
            intrinsics,
            depth_trunc=depth_trunc,
            target_mask=region_mask,
        )
        return self.detect(points, aligned_colors, region_mask=aligned_mask, **kwargs)

    def ping(self) -> bool:
        if self._remote:
            try:
                import requests
                response = requests.get(
                    self._remote_url("/health"),
                    timeout=min(5.0, float(self._config.request_timeout_s)),
                )
                if not response.ok:
                    return False
                payload = response.json()
                if not bool(payload.get("detector_loaded")):
                    return False
                service_top_down = bool(payload.get("top_down_grasp", False))
                requested_top_down = bool(self._config.top_down_grasp)
                if service_top_down != requested_top_down:
                    logger.warning(
                        "AnyGrasp service detection mode mismatch: requested "
                        "top_down_grasp=%s but service was loaded with "
                        "top_down_grasp=%s",
                        requested_top_down,
                        service_top_down,
                    )
                    return False
                return True
            except Exception:
                return False
        try:
            self._ensure_loaded()
            return self._detector is not None
        except Exception:
            return False

    def reset(self) -> None:
        self._detector = None
        self.last_detection_audit = None
