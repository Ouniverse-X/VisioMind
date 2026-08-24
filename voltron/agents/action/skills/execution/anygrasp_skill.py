"""AnyGrasp RGB-D detection and non-blocking CuRobo execution skill."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np

from voltron.agents.action.skills.execution.core import (
    PolicyBackedVLASkill,
    resolve_control_mode,
)
from voltron.agents.action.tools.action_projection import ActionProjection
from voltron.shared.context import ExecutionContext, LocalSkillSelection, Subtask
from voltron.shared.contracts import MemoryAdapter, PolicyAdapter
from voltron.shared.enums import AgentStatus
from voltron.shared.results import AgentResult
from voltron.shared.action_semantics import normalize_action_name

logger = logging.getLogger(__name__)


def _world_vertical_grasp_rotation(
    target_points_camera: Any,
    camera_pose_world: Any,
    original_rotation_camera: Any,
    *,
    jaw_axis: str = "minor",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build an AnyGrasp-frame rotation with a world-vertical approach.

    AnyGrasp uses rotation columns ``(+X approach, +Y jaw articulation, +Z)``.
    The horizontal jaw direction is estimated from the target mask footprint so
    the detector still supplies grasp position / depth while execution uses a
    robot-compatible top-down orientation.
    """
    points_camera = np.asarray(target_points_camera, dtype=np.float64).reshape(-1, 3)
    camera_pose = np.asarray(camera_pose_world, dtype=np.float64)
    original_rotation = np.asarray(original_rotation_camera, dtype=np.float64)
    if len(points_camera) < 3 or not np.isfinite(points_camera).all():
        raise ValueError("world-vertical orientation requires at least 3 finite target points")
    if camera_pose.shape != (4, 4) or not np.isfinite(camera_pose).all():
        raise ValueError("camera pose must be a finite 4x4 transform")
    if original_rotation.shape != (3, 3) or not np.isfinite(original_rotation).all():
        raise ValueError("original grasp rotation must be a finite 3x3 matrix")
    jaw_axis = str(jaw_axis).strip().lower()
    if jaw_axis not in {"minor", "major"}:
        raise ValueError("candidate_world_vertical_jaw_axis must be 'minor' or 'major'")

    camera_rotation = camera_pose[:3, :3]
    points_world = points_camera @ camera_rotation.T + camera_pose[:3, 3]
    centered_xy = points_world[:, :2] - np.mean(points_world[:, :2], axis=0)
    covariance = centered_xy.T @ centered_xy / max(1, len(centered_xy) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis_index = 0 if jaw_axis == "minor" else 1
    jaw_world = np.array(
        [eigenvectors[0, axis_index], eigenvectors[1, axis_index], 0.0],
        dtype=np.float64,
    )
    jaw_norm = float(np.linalg.norm(jaw_world))
    if jaw_norm <= 1e-9:
        original_jaw_world = camera_rotation @ original_rotation[:, 1]
        jaw_world = np.array(
            [original_jaw_world[0], original_jaw_world[1], 0.0], dtype=np.float64
        )
        jaw_norm = float(np.linalg.norm(jaw_world))
    if jaw_norm <= 1e-9:
        jaw_world = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        jaw_world /= jaw_norm

    # Resolve PCA's sign ambiguity using the detector's original jaw direction.
    original_jaw_world = camera_rotation @ original_rotation[:, 1]
    original_jaw_xy = np.array(
        [original_jaw_world[0], original_jaw_world[1], 0.0], dtype=np.float64
    )
    if float(np.dot(jaw_world, original_jaw_xy)) < 0.0:
        jaw_world = -jaw_world

    approach_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    third_world = np.cross(approach_world, jaw_world)
    third_world /= np.linalg.norm(third_world) + 1e-12
    rotation_world = np.column_stack((approach_world, jaw_world, third_world))
    rotation_camera = camera_rotation.T @ rotation_world
    audit = {
        "enabled": True,
        "jaw_axis": jaw_axis,
        "target_point_count": int(len(points_camera)),
        "footprint_eigenvalues_m2": eigenvalues.tolist(),
        "world_approach": approach_world.tolist(),
        "world_jaw_direction": jaw_world.tolist(),
        "original_world_approach": (
            camera_rotation @ original_rotation[:, 0]
        ).tolist(),
    }
    return rotation_camera, audit


def _open_jaw_clearance_passes(
    evidence: dict[str, Any],
    minimum_clearance_m: float,
) -> bool:
    """Require continuous target geometry to fit with usable clearance.

    A zero-clearance containment check is numerically valid but physically
    brittle: millimetre-scale pose error then produces a one-sided collision.
    """
    clearance = evidence.get("open_jaw_continuous_inner_clearance_m")
    return bool(
        evidence.get("available", False)
        and evidence.get("open_jaw_continuous_cross_section_intersects", False)
        and evidence.get("target_between_open_fingers", False)
        and clearance is not None
        and np.isfinite(float(clearance))
        and float(clearance) >= float(minimum_clearance_m)
    )


class AnyGraspSkill(PolicyBackedVLASkill):
    """Target-conditioned AnyGrasp skill with explicit, observable fallback."""

    skill_id = "anygrasp_manipulation_skill"
    supported_actions = (
        "pick_up", "grasp", "lift", "take", "hold",
        "place", "place_inside", "put_inside", "drop", "release",
    )

    def __init__(
        self,
        memory: MemoryAdapter,
        policy: PolicyAdapter,
        projector: ActionProjection,
        anygrasp_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(memory=memory, policy=policy, projector=projector)
        self._anygrasp_config = dict(anygrasp_config or {})
        self._detector: Any | None = None
        self._detector_init_failed = False
        self._executor: Any | None = None
        self._executor_init_failed = False
        self._active_execution: Any | None = None
        self._active_source = ""
        self._pre_detection_release_execution: Any | None = None
        self._pre_detection_release_completed = False
        self._pre_detection_release_failed = False
        self._active_subtask_id = ""
        self._failed_candidates: set[tuple[float, ...]] = set()
        self._candidate_queue: list[Any] = []
        self._candidate_packet: Any | None = None
        self._candidate_batch_loaded = False
        self._candidate_detection_batches = 0
        self._candidate_detection_audits: list[dict[str, Any]] = []
        self._execution_failure_audits: list[dict[str, Any]] = []
        self._pending_candidate_detection_audit: dict[str, Any] | None = None
        self._anygrasp_attempts = 0
        self._builtin_attempted = False
        self._last_execution_error: str | None = None

    @property
    def _candidate_detection_only(self) -> bool:
        return bool(self._anygrasp_config.get("candidate_detection_only", False))

    @property
    def _allow_fallback(self) -> bool:
        return not self._candidate_detection_only and bool(
            self._anygrasp_config.get("allow_fallback", True)
        )

    def _reset_subtask_state(self, subtask_id: str) -> None:
        self._active_execution = None
        self._active_source = ""
        self._pre_detection_release_execution = None
        self._pre_detection_release_completed = False
        self._pre_detection_release_failed = False
        self._active_subtask_id = subtask_id
        self._failed_candidates.clear()
        self._candidate_queue.clear()
        self._candidate_packet = None
        self._candidate_batch_loaded = False
        self._candidate_detection_batches = 0
        self._candidate_detection_audits.clear()
        self._execution_failure_audits.clear()
        self._pending_candidate_detection_audit = None
        self._anygrasp_attempts = 0
        self._builtin_attempted = False
        self._last_execution_error = None


    def _get_detector(self) -> Any | None:
        if self._detector is not None:
            return self._detector
        if self._detector_init_failed or not self._anygrasp_config:
            return None
        try:
            from voltron.integrations.manipulation.anygrasp import AnyGraspDetector
            detector = AnyGraspDetector(self._anygrasp_config)
            if not detector.ping():
                raise RuntimeError("AnyGrasp health check failed or detector is not loaded")
            self._detector = detector
            logger.info("AnyGrasp detector ready")
            return detector
        except Exception as exc:
            self._detector_init_failed = True
            self._last_execution_error = str(exc)
            logger.warning("AnyGrasp detector unavailable: %s", exc)
            return None

    def _get_executor(self) -> Any | None:
        if self._executor is not None:
            return self._executor
        if self._executor_init_failed:
            return None
        try:
            robot = self._get_og_robot()
            if robot is None:
                raise RuntimeError("OmniGibson robot is unavailable")
            from voltron.integrations.manipulation.anygrasp.grasp_executor import GraspExecutor
            self._executor = GraspExecutor(
                robot=robot,
                arm=self._anygrasp_config.get("arm"),
                curobo_batch_size=int(self._anygrasp_config.get("curobo_batch_size", 1)),
                pregrasp_offset_m=float(
                    self._anygrasp_config.get("pregrasp_offset_m", 0.08)
                ),
                whole_body_standoff_m=float(
                    self._anygrasp_config.get("whole_body_standoff_m", 0.35)
                ),
                lift_height_m=float(self._anygrasp_config.get("lift_height_m", 0.15)),
                post_lift_yaw_deg=float(
                    self._anygrasp_config.get("post_lift_yaw_deg", 0.0)
                ),
                post_lift_yaw_cycles=int(
                    self._anygrasp_config.get("post_lift_yaw_cycles", 0)
                ),
                post_lift_place_back=bool(
                    self._anygrasp_config.get("post_lift_place_back", False)
                ),
                place_back_clearance_m=float(
                    self._anygrasp_config.get("place_back_clearance_m", 0.015)
                ),
                place_back_retreat_m=float(
                    self._anygrasp_config.get("place_back_retreat_m", 0.08)
                ),
                skip_standoff_if_within_m=float(
                    self._anygrasp_config.get("skip_standoff_if_within_m", 0.20)
                ),
                constrained_approach=bool(
                    self._anygrasp_config.get("constrained_approach", True)
                ),
                retry_unconstrained_approach=bool(
                    self._anygrasp_config.get("retry_unconstrained_approach", True)
                ),
                approach_segment_max_m=float(
                    self._anygrasp_config.get("approach_segment_max_m", 0.0)
                ),
                approach_target_displacement_tolerance_m=float(
                    self._anygrasp_config.get(
                        "approach_target_displacement_tolerance_m", 0.02
                    )
                ),
                close_target_displacement_tolerance_m=float(
                    self._anygrasp_config.get(
                        "close_target_displacement_tolerance_m", 0.01
                    )
                ),
                approach_goal_position_tolerance_m=float(
                    self._anygrasp_config.get(
                        "approach_goal_position_tolerance_m", 0.015
                    )
                ),
                live_open_jaw_y_correction_max_m=float(
                    self._anygrasp_config.get(
                        "live_open_jaw_y_correction_max_m", 0.0
                    )
                ),
                grasping_mode_override=self._anygrasp_config.get(
                    "grasping_mode_override"
                ),
                collision_workspace_radius_m=self._anygrasp_config.get(
                    "collision_workspace_radius_m"
                ),
                verification_steps=int(
                    self._anygrasp_config.get("verification_steps", 5)
                ),
                verification_min_target_z_rise_m=float(
                    self._anygrasp_config.get(
                        "verification_min_target_z_rise_m", 0.03
                    )
                ),
                verification_relative_offset_tolerance_m=float(
                    self._anygrasp_config.get(
                        "verification_relative_offset_tolerance_m", 0.01
                    )
                ),
                verification_relative_orientation_tolerance_deg=float(
                    self._anygrasp_config.get(
                        "verification_relative_orientation_tolerance_deg", 10.0
                    )
                ),
                verification_require_attachment_valid=bool(
                    self._anygrasp_config.get(
                        "verification_require_attachment_valid", True
                    )
                ),
                physical_require_bilateral_contact_before_lift=bool(
                    self._anygrasp_config.get(
                        "physical_require_bilateral_contact_before_lift", False
                    )
                ),
                physical_staged_close_enabled=bool(
                    self._anygrasp_config.get("physical_staged_close_enabled", True)
                ),
                physical_close_compression_m=float(
                    self._anygrasp_config.get("physical_close_compression_m", 0.004)
                ),
                physical_close_stage_count=int(
                    self._anygrasp_config.get("physical_close_stage_count", 6)
                ),
                physical_close_hold_steps=int(
                    self._anygrasp_config.get("physical_close_hold_steps", 4)
                ),
                physical_close_stage_displacement_tolerance_m=float(
                    self._anygrasp_config.get(
                        "physical_close_stage_displacement_tolerance_m", 0.008
                    )
                ),
                physical_unilateral_contact_displacement_tolerance_m=float(
                    self._anygrasp_config.get(
                        "physical_unilateral_contact_displacement_tolerance_m",
                        0.002,
                    )
                ),
                fingertip_depth_override_m=(
                    None
                    if self._anygrasp_config.get("fingertip_depth_override_m") is None
                    else float(self._anygrasp_config["fingertip_depth_override_m"])
                ),
                eef_approach_offset_m=float(
                    self._anygrasp_config.get("eef_approach_offset_m", 0.0)
                ),
            )
            return self._executor
        except Exception as exc:
            self._executor_init_failed = True
            self._last_execution_error = str(exc)
            logger.warning("AnyGrasp CuRobo executor unavailable: %s", exc)
            return None

    @staticmethod
    def _get_og_robot() -> Any | None:
        try:
            import omnigibson as og
            if og.sim is None or not og.sim.scenes or not og.sim.scenes[0].robots:
                return None
            return og.sim.scenes[0].robots[0]
        except Exception:
            return None

    @staticmethod
    def _target_name(subtask: Subtask) -> str:
        return str(subtask.target.get("object") or subtask.target.get("object_id") or "").strip()

    @staticmethod
    def _destination_name(subtask: Subtask) -> str:
        return str(
            subtask.target.get("container")
            or subtask.target.get("destination")
            or subtask.parameters.get("container")
            or subtask.parameters.get("destination")
            or ""
        ).strip()

    def _capture_observation(self, subtask: Subtask) -> Any:
        from voltron.integrations.manipulation.anygrasp.observation import capture_grasp_observation

        robot = self._get_og_robot()
        if robot is None:
            raise RuntimeError("cannot capture AnyGrasp RGB-D without an OmniGibson robot")
        return capture_grasp_observation(
            robot,
            sensor_name=str(self._anygrasp_config.get("camera_sensor", "head_cam")),
            target_name=self._target_name(subtask),
            depth_trunc=float(self._anygrasp_config.get("depth_trunc", 2.0)),
            mask_dilation_px=int(self._anygrasp_config.get("mask_dilation_px", 1)),
            min_target_points=int(self._anygrasp_config.get("min_target_points", 30)),
            require_target_mask=bool(self._anygrasp_config.get("require_target_mask", True)),
            sensor_warmup_frames=int(self._anygrasp_config.get("sensor_warmup_frames", 3)),
            sensor_read_retries=int(self._anygrasp_config.get("sensor_read_retries", 3)),
            perception_audit_dir=self._anygrasp_config.get("perception_audit_dir"),
            target_depth_outlier_m=(
                None
                if self._anygrasp_config.get("target_depth_outlier_m") is None
                else float(self._anygrasp_config["target_depth_outlier_m"])
            ),
        )
    def _invalidate_candidate_batch(self) -> None:
        self._candidate_queue.clear()
        self._candidate_packet = None
        self._candidate_batch_loaded = False
        self._candidate_detection_batches = 0
        self._pre_detection_release_execution = None
        self._pre_detection_release_completed = False
        self._pre_detection_release_failed = False

    @staticmethod
    def _candidate_key(candidate: Any, camera_pose_world: Any) -> tuple[float, ...]:
        camera_pose = np.asarray(camera_pose_world, dtype=float)
        if camera_pose.shape != (4, 4) or not np.isfinite(camera_pose).all():
            raise ValueError("candidate camera pose must be a finite 4x4 transform")
        camera_rotation = camera_pose[:3, :3]
        translation_world = (
            camera_rotation @ np.asarray(candidate.translation, dtype=float).reshape(3)
            + camera_pose[:3, 3]
        )
        approach_world = (
            camera_rotation @ np.asarray(candidate.approach_direction, dtype=float).reshape(3)
        )
        values = np.concatenate([translation_world, approach_world])
        return tuple(np.round(values, 3).tolist())

    def _detect_candidates(
        self,
        subtask: Subtask,
        target_obj: Any | None = None,
    ) -> tuple[list[Any], Any | None]:
        self._pending_candidate_detection_audit = {
            "event": "anygrasp_skill_candidate_funnel_audit",
            "detector_audit": None,
        }
        detector = self._get_detector()
        if detector is None:
            self._pending_candidate_detection_audit["error"] = (
                self._last_execution_error or "detector unavailable"
            )
            return [], None
        try:
            release_planner = getattr(self._executor, "release_planner_memory", None)
            if callable(release_planner):
                release_planner()
            packet = self._capture_observation(subtask)
            max_world_approach_z = self._anygrasp_config.get(
                "candidate_max_world_approach_z"
            )
            min_candidate_width = float(
                self._anygrasp_config.get("candidate_min_width_m", 0.0)
            )
            if max_world_approach_z is not None:
                max_world_approach_z = float(max_world_approach_z)
                if (
                    not np.isfinite(max_world_approach_z)
                    or not -1.0 <= max_world_approach_z <= 1.0
                ):
                    raise ValueError(
                        "candidate_max_world_approach_z must be finite and in [-1, 1]"
                    )
            if not np.isfinite(min_candidate_width) or min_candidate_width < 0.0:
                raise ValueError("candidate_min_width_m must be finite and non-negative")

            camera_pose = np.asarray(packet.camera_pose_world, dtype=np.float64)
            if camera_pose.shape != (4, 4) or not np.isfinite(camera_pose).all():
                raise ValueError("candidate camera pose must be a finite 4x4 transform")
            force_world_vertical = bool(
                self._anygrasp_config.get(
                    "candidate_force_world_vertical_approach", False
                )
            )

            physical_mode = (
                self._anygrasp_config.get("grasping_mode_override") == "physical"
            )
            collision_geometry_enabled = bool(
                self._anygrasp_config.get(
                    "candidate_target_collision_geometry_enabled", physical_mode
                )
            )
            collision_target_points_camera: np.ndarray | None = None
            collision_geometry_audit: dict[str, Any] = {
                "enabled": collision_geometry_enabled,
                "source": "target_collision_boundary_points_world",
                "fail_closed": collision_geometry_enabled,
                "available": False,
                "point_count": 0,
            }
            if collision_geometry_enabled:
                if target_obj is None:
                    raise RuntimeError(
                        "target collision geometry gate requires a resolved target object"
                    )
                executor = self._executor or self._get_executor()
                collision_point_snapshot = getattr(
                    executor, "target_collision_boundary_points_world", None
                )
                if not callable(collision_point_snapshot):
                    raise RuntimeError(
                        "target collision geometry gate requires collision-boundary preflight"
                    )
                world_points = np.asarray(
                    collision_point_snapshot(target_obj), dtype=np.float64
                ).reshape(-1, 3)
                if not len(world_points) or not np.isfinite(world_points).all():
                    raise RuntimeError(
                        "target collision geometry gate received no finite boundary points"
                    )
                collision_target_points_camera = (
                    world_points - camera_pose[:3, 3]
                ) @ camera_pose[:3, :3]
                collision_geometry_audit.update(
                    {
                        "available": True,
                        "point_count": int(len(world_points)),
                        "world_bounds": [
                            world_points.min(axis=0).tolist(),
                            world_points.max(axis=0).tolist(),
                        ],
                        "camera_bounds": [
                            collision_target_points_camera.min(axis=0).tolist(),
                            collision_target_points_camera.max(axis=0).tolist(),
                        ],
                    }
                )

            approach = subtask.parameters.get("approach_direction")
            if approach is None:
                approach = self._anygrasp_config.get("approach_direction")
            approach_thresh = float(
                self._anygrasp_config.get("approach_thresh", np.pi)
            )
            if (
                approach is None
                and max_world_approach_z is not None
                and not force_world_vertical
            ):
                # The detector expects a camera-frame vector. Applying this request
                # server-side ensures top_k is taken from downward grasps rather than
                # from high-scoring grasps that approach the tabletop from below.
                approach = camera_pose[:3, :3].T @ np.array(
                    [0.0, 0.0, -1.0], dtype=np.float64
                )
                downward_thresh = float(np.arccos(np.clip(-max_world_approach_z, -1.0, 1.0)))
                approach_thresh = min(approach_thresh, downward_thresh)
                logger.warning(
                    "requesting AnyGrasp world-down candidates: camera_direction=%s "
                    "threshold_deg=%.1f max_world_z=%.3f",
                    np.round(approach, 4).tolist(),
                    np.rad2deg(approach_thresh),
                    max_world_approach_z,
                )

            candidates = detector.detect(
                packet.points,
                packet.colors,
                region_mask=packet.region_mask,
                approach_direction=approach,
                approach_thresh=approach_thresh,
            )
            detected_count = len(candidates)
            detector_audit = getattr(detector, "last_detection_audit", None)
            perception_audit_dir = self._anygrasp_config.get("perception_audit_dir")
            if perception_audit_dir:
                audit_root = Path(str(perception_audit_dir)).expanduser()
                audit_root.mkdir(parents=True, exist_ok=True)
                candidate_audit_path = audit_root / (
                    f"candidates_{time.time_ns()}.json"
                )
                candidate_audit_path.write_text(
                    json.dumps(
                        {
                            "event": "anygrasp_candidate_perception_audit",
                            "target": self._target_name(subtask),
                            "candidate_count": int(len(candidates)),
                            "candidates": [candidate.as_dict() for candidate in candidates],
                            "detector_audit": detector_audit,
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                logger.info("AnyGrasp candidate perception audit written to %s", candidate_audit_path)
            self._pending_candidate_detection_audit = {
                "event": "anygrasp_skill_candidate_funnel_audit",
                "detector_audit": (
                    json.loads(json.dumps(detector_audit))
                    if isinstance(detector_audit, dict)
                    else None
                ),
            }
            for rank, candidate in enumerate(candidates, start=1):
                snapshot = {
                    "rank": rank,
                    "score": float(candidate.score),
                    "original_camera_translation": np.asarray(
                        candidate.translation, dtype=np.float64
                    ).reshape(3).tolist(),
                    "original_camera_rotation": np.asarray(
                        candidate.rotation_matrix, dtype=np.float64
                    ).reshape(3, 3).tolist(),
                    "original_camera_approach": np.asarray(
                        candidate.approach_direction, dtype=np.float64
                    ).reshape(3).tolist(),
                    "width": float(candidate.width),
                    "depth": float(candidate.depth),
                    "height": float(candidate.height),
                }
                setattr(candidate, "anygrasp_original_snapshot", snapshot)
            target_points: np.ndarray | None = None
            target_reference_points: np.ndarray | None = None
            non_target_points: np.ndarray | None = None
            if packet.region_mask is not None:
                scene_points = np.asarray(packet.points, dtype=np.float64)
                target_mask = np.asarray(packet.region_mask, dtype=bool)
                target_points = scene_points[target_mask]
                non_target_points = scene_points[~target_mask]
                anchor_tolerance = float(
                    self._anygrasp_config.get(
                        "target_anchor_tolerance_m",
                        max(0.02, float(self._anygrasp_config.get("region_margin", 0.04))),
                    )
                )
                centroid_tolerance_raw = self._anygrasp_config.get(
                    "candidate_target_centroid_tolerance_m"
                )
                centroid_tolerance = (
                    None
                    if centroid_tolerance_raw is None
                    else float(centroid_tolerance_raw)
                )
                if not np.isfinite(anchor_tolerance) or anchor_tolerance <= 0.0:
                    raise ValueError("target_anchor_tolerance_m must be finite and positive")
                if centroid_tolerance is not None and (
                    not np.isfinite(centroid_tolerance) or centroid_tolerance <= 0.0
                ):
                    raise ValueError(
                        "candidate_target_centroid_tolerance_m must be finite and positive"
                    )
                target_centroid = np.mean(target_points, axis=0)
                target_candidates = []
                for candidate in candidates:
                    anchor = np.asarray(candidate.translation, dtype=np.float64).reshape(3)
                    distances_sq = np.sum((target_points - anchor) ** 2, axis=1)
                    anchor_distance = float(np.sqrt(np.min(distances_sq)))
                    centroid_distance = float(np.linalg.norm(anchor - target_centroid))
                    anchor_passed = anchor_distance <= anchor_tolerance
                    centroid_passed = (
                        centroid_tolerance is None
                        or centroid_distance <= centroid_tolerance
                    )
                    if anchor_passed and centroid_passed:
                        target_candidates.append(candidate)
                    else:
                        logger.warning(
                            "rejecting AnyGrasp candidate target anchor=%.3f m "
                            "centroid=%.3f m tolerances=(%.3f,%s)",
                            anchor_distance,
                            centroid_distance,
                            anchor_tolerance,
                            centroid_tolerance,
                        )
                candidates = target_candidates

            if force_world_vertical:
                if target_points is None or len(target_points) < 3:
                    raise ValueError(
                        "candidate_force_world_vertical_approach requires target mask points"
                    )
                jaw_axis = self._anygrasp_config.get(
                    "candidate_world_vertical_jaw_axis", "minor"
                )
                orientation_audits = []
                for candidate in candidates:
                    rotation_camera, orientation_audit = (
                        _world_vertical_grasp_rotation(
                            target_points,
                            camera_pose,
                            candidate.rotation_matrix,
                            jaw_axis=jaw_axis,
                        )
                    )
                    candidate.rotation_matrix = rotation_camera.astype(np.float32)
                    setattr(
                        candidate,
                        "anygrasp_world_vertical_orientation",
                        orientation_audit,
                    )
                    orientation_audits.append(orientation_audit)
                self._pending_candidate_detection_audit[
                    "world_vertical_orientation"
                ] = {
                    "enabled": True,
                    "jaw_axis": str(jaw_axis),
                    "candidate_count": int(len(candidates)),
                    "candidates": orientation_audits,
                }
                logger.warning(
                    "forced %d AnyGrasp candidates to world-vertical approach "
                    "using target PCA jaw_axis=%s",
                    len(candidates),
                    jaw_axis,
                )

            target_cross_section_geometry_source = "rgbd_target_mask_points"
            target_reference_geometry_source = target_cross_section_geometry_source
            target_reference_points = target_points
            if collision_geometry_enabled:
                if collision_target_points_camera is None:
                    raise RuntimeError(
                        "target collision geometry gate failed to produce camera points"
                    )
                if target_points is None or not len(target_points):
                    raise RuntimeError(
                        "target collision geometry gate requires RGB-D target mask points"
                    )
                target_reference_points = collision_target_points_camera
                target_reference_geometry_source = (
                    "target_collision_boundary_points_world"
                )
            collision_geometry_audit.update(
                {
                    "target_cross_section_geometry_source": (
                        target_cross_section_geometry_source
                    ),
                    "target_cross_section_point_count": (
                        0 if target_points is None else int(len(target_points))
                    ),
                    "target_reference_geometry_source": (
                        target_reference_geometry_source
                    ),
                    "target_reference_point_count": (
                        0
                        if target_reference_points is None
                        else int(len(target_reference_points))
                    ),
                    "candidate_bounds_gate": {
                        "required_axis_overlaps": ["x", "y", "z"],
                        "require_y_center_straddle": collision_geometry_enabled,
                        "require_z_center_straddle": collision_geometry_enabled,
                        "uses_point_count": False,
                    },
                }
            )

            min_inner_points = int(
                self._anygrasp_config.get("candidate_min_inner_target_points", 0)
            )
            require_center_straddle = bool(
                self._anygrasp_config.get("candidate_require_center_straddle", False)
            )
            canonical_depth_base_m = float(
                self._anygrasp_config.get("candidate_canonical_depth_base_m", 0.02)
            )
            if min_inner_points < 0:
                raise ValueError("candidate_min_inner_target_points must be non-negative")
            if not np.isfinite(canonical_depth_base_m) or canonical_depth_base_m < 0.0:
                raise ValueError("candidate_canonical_depth_base_m must be finite and non-negative")

            inner_line_gate_enabled = bool(
                self._anygrasp_config.get(
                    "candidate_inner_line_gate_enabled", physical_mode
                )
            )
            inner_line_margin_m = float(
                self._anygrasp_config.get("candidate_inner_line_margin_m", 0.0)
            )
            inner_line_min_points = int(
                self._anygrasp_config.get("candidate_inner_line_min_target_points", 0)
            )
            inner_line_min_fraction = float(
                self._anygrasp_config.get("candidate_inner_line_min_target_fraction", 0.0)
            )
            inner_line_min_overlap_m = float(
                self._anygrasp_config.get("candidate_inner_line_min_overlap_m", 0.0)
            )
            inner_line_require_center = bool(
                self._anygrasp_config.get(
                    "candidate_inner_line_require_center_straddle", False
                )
            )
            min_open_jaw_clearance_m = float(
                self._anygrasp_config.get(
                    "candidate_min_open_jaw_clearance_m", 0.0
                )
            )
            if not np.isfinite(inner_line_margin_m) or inner_line_margin_m < 0.0:
                raise ValueError("candidate_inner_line_margin_m must be finite and non-negative")
            if inner_line_min_points < 0:
                raise ValueError("candidate_inner_line_min_target_points must be non-negative")
            if (
                not np.isfinite(inner_line_min_fraction)
                or not 0.0 <= inner_line_min_fraction <= 1.0
            ):
                raise ValueError(
                    "candidate_inner_line_min_target_fraction must be finite and in [0, 1]"
                )
            if not np.isfinite(inner_line_min_overlap_m) or inner_line_min_overlap_m < 0.0:
                raise ValueError(
                    "candidate_inner_line_min_overlap_m must be finite and non-negative"
                )
            if (
                not np.isfinite(min_open_jaw_clearance_m)
                or min_open_jaw_clearance_m < 0.0
            ):
                raise ValueError(
                    "candidate_min_open_jaw_clearance_m must be finite and "
                    "non-negative"
                )
            depth_fit_enabled = bool(
                self._anygrasp_config.get(
                    "candidate_fit_depth_to_robot_inner_line", False
                )
            )
            depth_fit_min_m = float(
                self._anygrasp_config.get("candidate_fit_depth_min_m", 0.005)
            )
            depth_fit_step_m = float(
                self._anygrasp_config.get("candidate_fit_depth_step_m", 0.005)
            )
            depth_fit_max_raw = self._anygrasp_config.get(
                "candidate_fit_depth_max_m"
            )
            depth_fit_max_m = (
                None if depth_fit_max_raw is None else float(depth_fit_max_raw)
            )
            depth_fit_selection_mode = str(
                self._anygrasp_config.get(
                    "candidate_fit_depth_selection_mode", "shallowest_pass"
                )
            ).strip().lower()
            if not np.isfinite(depth_fit_min_m) or depth_fit_min_m <= 0.0:
                raise ValueError("candidate_fit_depth_min_m must be finite and positive")
            if not np.isfinite(depth_fit_step_m) or depth_fit_step_m <= 0.0:
                raise ValueError("candidate_fit_depth_step_m must be finite and positive")
            if depth_fit_max_m is not None and (
                not np.isfinite(depth_fit_max_m) or depth_fit_max_m <= 0.0
            ):
                raise ValueError("candidate_fit_depth_max_m must be finite and positive")
            if depth_fit_selection_mode not in {"shallowest_pass", "centered_coverage"}:
                raise ValueError(
                    "candidate_fit_depth_selection_mode must be one of: "
                    "shallowest_pass, centered_coverage"
                )
            if depth_fit_enabled and not inner_line_gate_enabled:
                raise ValueError(
                    "candidate depth fitting requires candidate_inner_line_gate_enabled"
                )
            inner_line_preflight = None
            if inner_line_gate_enabled or collision_geometry_enabled:
                executor = self._executor or self._get_executor()
                inner_line_preflight = getattr(
                    executor, "candidate_inner_grasp_line_evidence", None
                )
                if not callable(inner_line_preflight):
                    raise RuntimeError(
                        "actual inner grasp-line and open-jaw gates require robot "
                        "geometry preflight"
                    )

            non_target_collision_audit_enabled = bool(
                self._anygrasp_config.get(
                    "candidate_non_target_collision_audit_enabled", False
                )
            )
            non_target_collision_margin_m = float(
                self._anygrasp_config.get(
                    "candidate_non_target_collision_margin_m", 0.0
                )
            )
            if (
                not np.isfinite(non_target_collision_margin_m)
                or non_target_collision_margin_m < 0.0
            ):
                raise ValueError(
                    "candidate_non_target_collision_margin_m must be finite and "
                    "non-negative"
                )
            non_target_collision_preflight = None
            if non_target_collision_audit_enabled:
                if non_target_points is None or not len(non_target_points):
                    raise ValueError(
                        "non-target collision audit requires non-target scene points"
                    )
                executor = self._executor or self._get_executor()
                non_target_collision_preflight = getattr(
                    executor, "candidate_non_target_collision_evidence", None
                )
                if not callable(non_target_collision_preflight):
                    raise RuntimeError(
                        "non-target collision audit requires robot geometry preflight"
                    )

            raw_preferred_detector_ranks = self._anygrasp_config.get(
                "candidate_preferred_detector_ranks", []
            )
            if raw_preferred_detector_ranks is None:
                raw_preferred_detector_ranks = []
            if not isinstance(raw_preferred_detector_ranks, (list, tuple)):
                raise ValueError("candidate_preferred_detector_ranks must be a list")
            if any(
                isinstance(rank, bool)
                or not isinstance(rank, (int, np.integer))
                or int(rank) <= 0
                for rank in raw_preferred_detector_ranks
            ):
                raise ValueError(
                    "candidate_preferred_detector_ranks must contain positive integers"
                )
            preferred_detector_ranks = tuple(
                int(rank) for rank in raw_preferred_detector_ranks
            )
            if len(set(preferred_detector_ranks)) != len(preferred_detector_ranks):
                raise ValueError(
                    "candidate_preferred_detector_ranks must contain unique ranks"
                )
            preferred_rank_order = {
                rank: order for order, rank in enumerate(preferred_detector_ranks)
            }

            recenter_enabled = bool(
                self._anygrasp_config.get("candidate_recenter_to_target_centroid", False)
            )
            raw_recenter_axes = self._anygrasp_config.get(
                "candidate_recenter_axes", ["y", "z"]
            )
            if isinstance(raw_recenter_axes, str):
                recenter_axes = tuple(
                    axis.strip().lower() for axis in raw_recenter_axes.split(",")
                )
            elif isinstance(raw_recenter_axes, (list, tuple)):
                recenter_axes = tuple(str(axis).strip().lower() for axis in raw_recenter_axes)
            else:
                raise ValueError("candidate_recenter_axes must be a list or comma-separated string")
            if not recenter_axes or len(set(recenter_axes)) != len(recenter_axes):
                raise ValueError("candidate_recenter_axes must contain unique axes")
            axis_indices = {"x": 0, "y": 1, "z": 2}
            invalid_axes = set(recenter_axes) - set(axis_indices)
            if invalid_axes:
                raise ValueError(
                    f"candidate_recenter_axes contains invalid axes: {sorted(invalid_axes)}"
                )
            if depth_fit_enabled and recenter_enabled and "x" in recenter_axes:
                raise ValueError(
                    "candidate depth fitting is incompatible with X-axis recentering"
                )
            recenter_reference = str(
                self._anygrasp_config.get(
                    "candidate_recenter_reference", "target_centroid"
                )
            ).strip().lower()
            if recenter_reference not in {"target_centroid", "axial_centroid"}:
                raise ValueError(
                    "candidate_recenter_reference must be target_centroid or axial_centroid"
                )
            recenter_max_raw = self._anygrasp_config.get(
                "candidate_recenter_max_translation_m"
            )
            recenter_max_translation = (
                None if recenter_max_raw is None else float(recenter_max_raw)
            )
            if recenter_max_translation is not None and (
                not np.isfinite(recenter_max_translation)
                or recenter_max_translation < 0.0
            ):
                raise ValueError(
                    "candidate_recenter_max_translation_m must be finite and non-negative"
                )
            if recenter_enabled and target_points is None:
                raise ValueError("candidate recentering requires target mask points")

            anchored_count = len(candidates)
            geometry_candidates: list[tuple[Any, float, dict[str, Any]]] = []
            depth_fit_audits: list[dict[str, Any]] = []
            rejected_approach = 0
            rejected_target_volume = 0
            rejected_collision_bounds = 0
            rejected_inner_line = 0
            rejected_open_jaw = 0
            rejected_recenter = 0
            narrow_candidates = 0
            target_centroid_camera = (
                np.mean(target_reference_points, axis=0)
                if target_reference_points is not None
                else None
            )
            target_centroid_world = (
                camera_pose[:3, :3] @ target_centroid_camera + camera_pose[:3, 3]
                if target_centroid_camera is not None
                else None
            )

            def target_geometry_for(
                candidate: Any,
                rotation: np.ndarray,
                translation: np.ndarray,
            ) -> dict[str, Any]:
                if target_points is None or target_reference_points is None:
                    return {
                        "target_geometry_source": target_cross_section_geometry_source,
                        "target_cross_section_geometry_source": (
                            target_cross_section_geometry_source
                        ),
                        "target_reference_geometry_source": (
                            target_reference_geometry_source
                        ),
                        "target_point_count": 0,
                        "target_cross_section_point_count": 0,
                        "target_reference_point_count": 0,
                        "axial_target_point_count": 0,
                        "axial_target_fraction": 0.0,
                        "inner_target_point_count": 0,
                        "inner_target_fraction": 0.0,
                        "center_straddled": False,
                        "target_center_offset_y_m": None,
                        "target_span_y_m": None,
                        "target_local_bounds": None,
                        "target_cross_section_local_bounds": None,
                        "target_reference_local_bounds": None,
                        "target_centroid_camera": None,
                        "target_centroid_world": None,
                        "target_centroid_local": None,
                        "axial_target_centroid_local": None,
                        "axial_target_centroid_camera": None,
                        "axial_target_centroid_world": None,
                        "inner_target_centroid_local": None,
                        "inner_target_centroid_camera": None,
                        "inner_target_centroid_world": None,
                        "jaw_box_center_local": None,
                        "jaw_box_center_camera": None,
                        "jaw_box_center_world": None,
                        "jaw_box_center_to_target_centroid_m": None,
                        "collision_bounds_gate": {
                            "enabled": collision_geometry_enabled,
                            "available": False,
                            "gate_passed": not collision_geometry_enabled,
                        },
                        "collision_bounds_gate_passed": (
                            not collision_geometry_enabled
                        ),
                        "recenter_applied": False,
                    }
                local_points = (target_points - translation) @ rotation
                reference_local_points = (
                    target_reference_points - translation
                ) @ rotation
                width = max(0.0, float(candidate.width))
                height = max(0.0, float(candidate.height))
                depth = max(0.0, float(candidate.depth))
                axial_height_mask = (
                    (local_points[:, 0] > -canonical_depth_base_m)
                    & (local_points[:, 0] < depth)
                    & (np.abs(local_points[:, 2]) < height / 2.0)
                )
                inner_mask = axial_height_mask & (
                    np.abs(local_points[:, 1]) < width / 2.0
                )
                axial_points = local_points[axial_height_mask]
                inner_points = local_points[inner_mask]
                y_min = float(axial_points[:, 1].min()) if len(axial_points) else None
                y_max = float(axial_points[:, 1].max()) if len(axial_points) else None
                center_straddled = bool(
                    y_min is not None and y_max is not None and y_min <= 0.0 <= y_max
                )
                center_offset = (
                    abs((y_min + y_max) / 2.0)
                    if y_min is not None and y_max is not None
                    else None
                )
                reference_bounds_min = reference_local_points.min(axis=0)
                reference_bounds_max = reference_local_points.max(axis=0)
                jaw_bounds_min = np.array(
                    [-canonical_depth_base_m, -width / 2.0, -height / 2.0],
                    dtype=np.float64,
                )
                jaw_bounds_max = np.array(
                    [depth, width / 2.0, height / 2.0], dtype=np.float64
                )
                bounds_overlaps = np.maximum(
                    0.0,
                    np.minimum(reference_bounds_max, jaw_bounds_max)
                    - np.maximum(reference_bounds_min, jaw_bounds_min),
                )
                positive_axis_overlaps = bounds_overlaps > 0.0
                collision_y_center_straddled = bool(
                    reference_bounds_min[1] <= 0.0 <= reference_bounds_max[1]
                )
                collision_z_center_straddled = bool(
                    reference_bounds_min[2] <= 0.0 <= reference_bounds_max[2]
                )
                collision_bounds_gate_passed = bool(
                    not collision_geometry_enabled
                    or (
                        np.all(positive_axis_overlaps)
                        and collision_y_center_straddled
                        and collision_z_center_straddled
                    )
                )
                collision_bounds_gate = {
                    "enabled": collision_geometry_enabled,
                    "available": bool(len(reference_local_points)),
                    "source": target_reference_geometry_source,
                    "uses_point_count": False,
                    "candidate_jaw_intervals_m": {
                        "x": [float(jaw_bounds_min[0]), float(jaw_bounds_max[0])],
                        "y": [float(jaw_bounds_min[1]), float(jaw_bounds_max[1])],
                        "z": [float(jaw_bounds_min[2]), float(jaw_bounds_max[2])],
                    },
                    "target_reference_bounds_m": [
                        reference_bounds_min.tolist(),
                        reference_bounds_max.tolist(),
                    ],
                    "axis_overlap_m": {
                        "x": float(bounds_overlaps[0]),
                        "y": float(bounds_overlaps[1]),
                        "z": float(bounds_overlaps[2]),
                    },
                    "positive_axis_overlap": {
                        "x": bool(positive_axis_overlaps[0]),
                        "y": bool(positive_axis_overlaps[1]),
                        "z": bool(positive_axis_overlaps[2]),
                    },
                    "y_center_straddled": collision_y_center_straddled,
                    "z_center_straddled": collision_z_center_straddled,
                    "gate_passed": collision_bounds_gate_passed,
                }
                target_centroid_local = np.mean(reference_local_points, axis=0)
                axial_centroid_local = (
                    np.mean(axial_points, axis=0) if len(axial_points) else None
                )
                axial_centroid_camera = (
                    translation + rotation @ axial_centroid_local
                    if axial_centroid_local is not None
                    else None
                )
                axial_centroid_world = (
                    camera_pose[:3, :3] @ axial_centroid_camera + camera_pose[:3, 3]
                    if axial_centroid_camera is not None
                    else None
                )
                inner_centroid_local = (
                    np.mean(inner_points, axis=0) if len(inner_points) else None
                )
                inner_centroid_camera = (
                    translation + rotation @ inner_centroid_local
                    if inner_centroid_local is not None
                    else None
                )
                inner_centroid_world = (
                    camera_pose[:3, :3] @ inner_centroid_camera + camera_pose[:3, 3]
                    if inner_centroid_camera is not None
                    else None
                )
                jaw_box_center_local = np.array(
                    [(depth - canonical_depth_base_m) / 2.0, 0.0, 0.0],
                    dtype=np.float64,
                )
                jaw_box_center_camera = translation + rotation @ jaw_box_center_local
                jaw_box_center_world = (
                    camera_pose[:3, :3] @ jaw_box_center_camera + camera_pose[:3, 3]
                )
                cross_section_local_bounds = [
                    np.round(local_points.min(axis=0), 5).tolist(),
                    np.round(local_points.max(axis=0), 5).tolist(),
                ]
                reference_local_bounds = [
                    np.round(reference_bounds_min, 5).tolist(),
                    np.round(reference_bounds_max, 5).tolist(),
                ]
                return {
                    "target_geometry_source": target_cross_section_geometry_source,
                    "target_cross_section_geometry_source": (
                        target_cross_section_geometry_source
                    ),
                    "target_reference_geometry_source": (
                        target_reference_geometry_source
                    ),
                    "target_point_count": int(len(target_points)),
                    "target_cross_section_point_count": int(len(target_points)),
                    "target_reference_point_count": int(len(target_reference_points)),
                    "axial_target_point_count": int(len(axial_points)),
                    "axial_target_fraction": float(len(axial_points) / len(target_points)),
                    "inner_target_point_count": int(inner_mask.sum()),
                    "inner_target_fraction": float(inner_mask.sum() / len(target_points)),
                    "center_straddled": center_straddled,
                    "target_center_offset_y_m": center_offset,
                    "target_span_y_m": (
                        y_max - y_min
                        if y_min is not None and y_max is not None
                        else None
                    ),
                    "target_local_bounds": reference_local_bounds,
                    "target_cross_section_local_bounds": cross_section_local_bounds,
                    "target_reference_local_bounds": reference_local_bounds,
                    "target_centroid_camera": target_centroid_camera.tolist(),
                    "target_centroid_world": target_centroid_world.tolist(),
                    "target_centroid_local": target_centroid_local.tolist(),
                    "axial_target_centroid_local": (
                        None if axial_centroid_local is None else axial_centroid_local.tolist()
                    ),
                    "axial_target_centroid_camera": (
                        None if axial_centroid_camera is None else axial_centroid_camera.tolist()
                    ),
                    "axial_target_centroid_world": (
                        None if axial_centroid_world is None else axial_centroid_world.tolist()
                    ),
                    "inner_target_centroid_local": (
                        None if inner_centroid_local is None else inner_centroid_local.tolist()
                    ),
                    "inner_target_centroid_camera": (
                        None if inner_centroid_camera is None else inner_centroid_camera.tolist()
                    ),
                    "inner_target_centroid_world": (
                        None if inner_centroid_world is None else inner_centroid_world.tolist()
                    ),
                    "jaw_box_center_local": jaw_box_center_local.tolist(),
                    "jaw_box_center_camera": jaw_box_center_camera.tolist(),
                    "jaw_box_center_world": jaw_box_center_world.tolist(),
                    "jaw_box_center_to_target_centroid_m": float(
                        np.linalg.norm(jaw_box_center_camera - target_centroid_camera)
                    ),
                    "collision_bounds_gate": collision_bounds_gate,
                    "collision_bounds_gate_passed": collision_bounds_gate_passed,
                    "recenter_applied": False,
                }

            def inner_line_gate_for(
                candidate: Any,
                rotation: np.ndarray,
                translation: np.ndarray,
            ) -> tuple[bool, dict[str, Any]]:
                if target_points is None or not len(target_points):
                    return False, {
                        "available": False,
                        "gate_enabled": True,
                        "gate_passed": False,
                        "target_geometry_source": (
                            target_cross_section_geometry_source
                        ),
                        "reason": "RGB-D target mask points are unavailable",
                    }
                target_local_points = (target_points - translation) @ rotation
                evidence = inner_line_preflight(
                    candidate,
                    target_local_points,
                    margin_m=inner_line_margin_m,
                )
                line_count = int(
                    evidence.get("target_points_in_inner_line_count", 0)
                )
                line_fraction = float(
                    evidence.get("target_points_in_inner_line_fraction", 0.0)
                )
                line_overlap = float(evidence.get("target_inner_line_overlap_m", 0.0))
                line_centered = bool(
                    evidence.get("inner_line_center_straddled", False)
                )
                passed = bool(
                    evidence.get("available", False)
                    and line_count >= inner_line_min_points
                    and line_fraction >= inner_line_min_fraction
                    and line_overlap >= inner_line_min_overlap_m
                    and (not inner_line_require_center or line_centered)
                )
                evidence.update(
                    {
                        "gate_enabled": True,
                        "gate_passed": passed,
                        "target_geometry_source": (
                            target_cross_section_geometry_source
                        ),
                        "required_target_points": inner_line_min_points,
                        "required_target_fraction": inner_line_min_fraction,
                        "required_overlap_m": inner_line_min_overlap_m,
                        "require_center_straddle": inner_line_require_center,
                    }
                )
                return passed, evidence

            def open_jaw_gate_for(
                candidate: Any,
                rotation: np.ndarray,
                translation: np.ndarray,
            ) -> tuple[bool, dict[str, Any]]:
                if not collision_geometry_enabled:
                    return True, {
                        "available": False,
                        "gate_enabled": False,
                        "gate_passed": True,
                    }
                if target_reference_points is None or not len(target_reference_points):
                    return False, {
                        "available": False,
                        "gate_enabled": True,
                        "gate_passed": False,
                        "reason": "target collision boundary points are unavailable",
                    }
                target_local_points = (
                    target_reference_points - translation
                ) @ rotation
                evidence = inner_line_preflight(
                    candidate,
                    target_local_points,
                    margin_m=inner_line_margin_m,
                )
                passed = _open_jaw_clearance_passes(
                    evidence,
                    min_open_jaw_clearance_m,
                )
                evidence.update(
                    {
                        "gate_enabled": True,
                        "gate_passed": passed,
                        "target_geometry_source": target_reference_geometry_source,
                        "requires_continuous_cross_section_geometry": True,
                        "required_minimum_inner_clearance_m": (
                            min_open_jaw_clearance_m
                        ),
                        "sampled_point_count_is_diagnostic_only": True,
                        "uses_tunable_point_count_threshold": False,
                    }
                )
                return passed, evidence

            def target_volume_gate_for(
                geometry: dict[str, Any],
            ) -> tuple[bool, bool, bool]:
                dense_passed = bool(
                    int(geometry["inner_target_point_count"])
                    >= min_inner_points
                    and (
                        not require_center_straddle
                        or bool(geometry["center_straddled"])
                    )
                )
                collision_passed = bool(
                    geometry.get("collision_bounds_gate_passed", False)
                )
                combined_passed = dense_passed and collision_passed
                geometry["dense_target_volume_gate"] = {
                    "gate_passed": dense_passed,
                    "required_target_points": min_inner_points,
                    "require_center_straddle": require_center_straddle,
                    "geometry_source": target_cross_section_geometry_source,
                }
                geometry["target_volume_gate_passed"] = combined_passed
                return combined_passed, dense_passed, collision_passed

            for candidate in candidates:
                approach_world = camera_pose[:3, :3] @ np.asarray(
                    candidate.approach_direction, dtype=np.float64
                ).reshape(3)
                approach_world /= np.linalg.norm(approach_world) + 1e-12
                approach_world_z = float(approach_world[2])
                if (
                    max_world_approach_z is not None
                    and approach_world_z > max_world_approach_z
                ):
                    rejected_approach += 1
                    continue
                if float(candidate.width) < min_candidate_width:
                    narrow_candidates += 1

                rotation = np.asarray(candidate.rotation_matrix, dtype=np.float64)
                translation = np.asarray(candidate.translation, dtype=np.float64).reshape(3)
                if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
                    rejected_target_volume += 1
                    logger.warning("rejecting AnyGrasp candidate with invalid rotation")
                    continue
                original_translation = translation.copy()
                original_geometry = target_geometry_for(candidate, rotation, translation)
                original_snapshot = dict(
                    getattr(candidate, "anygrasp_original_snapshot", {})
                )
                original_geometry["candidate_original_snapshot"] = original_snapshot
                target_geometry = original_geometry

                if recenter_enabled:
                    reference_key = (
                        "axial_target_centroid_local"
                        if recenter_reference == "axial_centroid"
                        else "target_centroid_local"
                    )
                    reference_value = original_geometry[reference_key]
                    if reference_value is None:
                        rejected_recenter += 1
                        logger.warning(
                            "rejecting AnyGrasp candidate without %s recenter reference",
                            recenter_reference,
                        )
                        continue
                    recenter_reference_local = np.asarray(
                        reference_value, dtype=np.float64
                    )
                    jaw_box_center_local = np.asarray(
                        original_geometry["jaw_box_center_local"], dtype=np.float64
                    )
                    recenter_delta_local = np.zeros(3, dtype=np.float64)
                    for axis in recenter_axes:
                        axis_index = axis_indices[axis]
                        recenter_delta_local[axis_index] = (
                            recenter_reference_local[axis_index]
                            - jaw_box_center_local[axis_index]
                        )
                    recenter_delta_camera = rotation @ recenter_delta_local
                    recenter_distance = float(np.linalg.norm(recenter_delta_camera))
                    if (
                        recenter_max_translation is not None
                        and recenter_distance > recenter_max_translation
                    ):
                        rejected_recenter += 1
                        logger.warning(
                            "rejecting AnyGrasp candidate recenter shift %.4f m exceeds "
                            "maximum %.4f m axes=%s original_translation=%s",
                            recenter_distance,
                            recenter_max_translation,
                            recenter_axes,
                            np.round(original_translation, 5).tolist(),
                        )
                        continue
                    translation = original_translation + recenter_delta_camera
                    candidate.translation = translation.astype(np.float32)
                    target_geometry = target_geometry_for(candidate, rotation, translation)
                    target_geometry["candidate_original_snapshot"] = original_snapshot
                    target_geometry.update(
                        {
                            "recenter_applied": True,
                            "recenter_axes": list(recenter_axes),
                            "recenter_reference": recenter_reference,
                            "recenter_reference_local": recenter_reference_local.tolist(),
                            "recenter_original_translation_camera": (
                                original_translation.tolist()
                            ),
                            "recenter_delta_local": recenter_delta_local.tolist(),
                            "recenter_delta_camera": recenter_delta_camera.tolist(),
                            "recenter_translation_m": recenter_distance,
                            "recentered_translation_camera": translation.tolist(),
                            "recenter_original_inner_target_point_count": int(
                                original_geometry["inner_target_point_count"]
                            ),
                            "recenter_original_inner_target_fraction": float(
                                original_geometry["inner_target_fraction"]
                            ),
                            "recenter_original_jaw_center_distance_m": float(
                                original_geometry["jaw_box_center_to_target_centroid_m"]
                            ),
                            "recenter_original_target_local_bounds": original_geometry[
                                "target_local_bounds"
                            ],
                        }
                    )
                    logger.warning(
                        "recentered AnyGrasp candidate reference=%s axes=%s shift=%.4f m "
                        "inner_points=%d->%d jaw_center_distance=%.4f->%.4f",
                        recenter_reference,
                        recenter_axes,
                        recenter_distance,
                        int(original_geometry["inner_target_point_count"]),
                        int(target_geometry["inner_target_point_count"]),
                        float(original_geometry["jaw_box_center_to_target_centroid_m"]),
                        float(target_geometry["jaw_box_center_to_target_centroid_m"]),
                    )

                if depth_fit_enabled:
                    sdk_depth = float(original_snapshot.get("depth", candidate.depth))
                    probe_ceiling_depth = (
                        sdk_depth
                        if depth_fit_max_m is None
                        else min(sdk_depth, depth_fit_max_m)
                    )
                    depth_fit_audit: dict[str, Any] = {
                        "enabled": True,
                        "sdk_original_depth_m": sdk_depth,
                        "minimum_probe_depth_m": depth_fit_min_m,
                        "maximum_probe_depth_m": depth_fit_max_m,
                        "effective_probe_ceiling_depth_m": probe_ceiling_depth,
                        "probe_step_m": depth_fit_step_m,
                        "selection_mode": depth_fit_selection_mode,
                        "selected_depth_m": None,
                        "selected_probe_metrics": None,
                        "applied": False,
                        "probes": [],
                    }
                    if not np.isfinite(sdk_depth) or sdk_depth <= 0.0:
                        depth_fit_audit["rejection_reason"] = (
                            "SDK candidate depth must be finite and positive"
                        )
                        depth_fit_audits.append(
                            {
                                "detector_rank": original_snapshot.get("rank"),
                                "score": float(candidate.score),
                                **depth_fit_audit,
                            }
                        )
                        rejected_target_volume += 1
                        continue

                    first_probe_depth = min(depth_fit_min_m, probe_ceiling_depth)
                    probe_depths: list[float] = []
                    probe_depth = first_probe_depth
                    while probe_depth < probe_ceiling_depth - 1e-9:
                        probe_depths.append(float(probe_depth))
                        if len(probe_depths) >= 1000:
                            raise ValueError(
                                "candidate depth fitting requires more than 1000 probes"
                            )
                        probe_depth += depth_fit_step_m
                    if not probe_depths or not np.isclose(
                        probe_depths[-1], probe_ceiling_depth, atol=1e-9, rtol=0.0
                    ):
                        probe_depths.append(probe_ceiling_depth)

                    selected_depth = None
                    selected_geometry = None
                    selected_translation = None
                    selected_probe_audit = None
                    passing_probes: list[
                        tuple[
                            tuple[float, ...],
                            float,
                            np.ndarray,
                            dict[str, Any],
                            dict[str, Any],
                        ]
                    ] = []
                    any_target_volume_passed = False
                    any_target_volume_and_inner_line_passed = False
                    any_all_gates_passed = False
                    any_dense_collision_rejection = False
                    for probe_depth in probe_depths:
                        probe_candidate = copy.copy(candidate)
                        probe_candidate.depth = float(probe_depth)
                        probe_translation = translation.copy()
                        probe_geometry = target_geometry_for(
                            probe_candidate, rotation, probe_translation
                        )
                        (
                            target_volume_passed,
                            dense_target_volume_passed,
                            collision_bounds_passed,
                        ) = target_volume_gate_for(probe_geometry)
                        line_passed, line_evidence = inner_line_gate_for(
                            probe_candidate, rotation, probe_translation
                        )
                        probe_geometry["robot_inner_grasp_line"] = line_evidence
                        open_jaw_passed, open_jaw_evidence = open_jaw_gate_for(
                            probe_candidate, rotation, probe_translation
                        )
                        probe_geometry["robot_open_jaw_containment"] = (
                            open_jaw_evidence
                        )
                        open_jaw_y_fit: dict[str, Any] = {
                            "enabled": bool(
                                recenter_enabled and "y" in recenter_axes
                            ),
                            "attempted": False,
                            "applied": False,
                            "axis": "y",
                        }
                        continuous_y_bounds = open_jaw_evidence.get(
                            "open_jaw_continuous_cross_section_y_bounds_m"
                        )
                        open_y_interval = open_jaw_evidence.get(
                            "open_jaw_inner_surface_y_interval_m"
                        )
                        if (
                            target_volume_passed
                            and line_passed
                            and not open_jaw_passed
                            and open_jaw_y_fit["enabled"]
                            and open_jaw_evidence.get(
                                "open_jaw_continuous_cross_section_intersects",
                                False,
                            )
                            and isinstance(continuous_y_bounds, (list, tuple))
                            and len(continuous_y_bounds) == 2
                            and isinstance(open_y_interval, (list, tuple))
                            and len(open_y_interval) == 2
                        ):
                            target_y = np.asarray(
                                continuous_y_bounds, dtype=np.float64
                            )
                            open_y = np.asarray(
                                open_y_interval, dtype=np.float64
                            )
                            target_span_y = float(target_y[1] - target_y[0])
                            open_span_y = float(open_y[1] - open_y[0])
                            shift_local_y = float(
                                target_y.mean() - open_y.mean()
                            )
                            shift_local = np.array(
                                [0.0, shift_local_y, 0.0], dtype=np.float64
                            )
                            shift_camera = rotation @ shift_local
                            proposed_translation = (
                                probe_translation + shift_camera
                            )
                            cumulative_recenter = float(
                                np.linalg.norm(
                                    proposed_translation - original_translation
                                )
                            )
                            fits_open_span = bool(
                                target_span_y <= open_span_y + 1e-8
                            )
                            within_recenter_limit = bool(
                                recenter_max_translation is None
                                or cumulative_recenter
                                <= recenter_max_translation
                            )
                            open_jaw_y_fit.update(
                                {
                                    "attempted": True,
                                    "target_span_y_m": target_span_y,
                                    "open_span_y_m": open_span_y,
                                    "shift_local_y_m": shift_local_y,
                                    "shift_camera_m": shift_camera.tolist(),
                                    "cumulative_recenter_m": cumulative_recenter,
                                    "maximum_recenter_m": (
                                        recenter_max_translation
                                    ),
                                    "fits_open_span": fits_open_span,
                                    "within_recenter_limit": (
                                        within_recenter_limit
                                    ),
                                }
                            )
                            if fits_open_span and within_recenter_limit:
                                proposed_geometry = target_geometry_for(
                                    probe_candidate,
                                    rotation,
                                    proposed_translation,
                                )
                                (
                                    proposed_target_volume_passed,
                                    proposed_dense_passed,
                                    proposed_collision_passed,
                                ) = target_volume_gate_for(proposed_geometry)
                                (
                                    proposed_line_passed,
                                    proposed_line_evidence,
                                ) = inner_line_gate_for(
                                    probe_candidate,
                                    rotation,
                                    proposed_translation,
                                )
                                proposed_geometry["robot_inner_grasp_line"] = (
                                    proposed_line_evidence
                                )
                                (
                                    proposed_open_jaw_passed,
                                    proposed_open_jaw_evidence,
                                ) = open_jaw_gate_for(
                                    probe_candidate,
                                    rotation,
                                    proposed_translation,
                                )
                                proposed_geometry[
                                    "robot_open_jaw_containment"
                                ] = proposed_open_jaw_evidence
                                proposed_combined_passed = bool(
                                    proposed_target_volume_passed
                                    and proposed_line_passed
                                    and proposed_open_jaw_passed
                                )
                                open_jaw_y_fit.update(
                                    {
                                        "proposed_target_volume_passed": (
                                            proposed_target_volume_passed
                                        ),
                                        "proposed_inner_line_passed": (
                                            proposed_line_passed
                                        ),
                                        "proposed_open_jaw_passed": (
                                            proposed_open_jaw_passed
                                        ),
                                        "proposed_combined_passed": (
                                            proposed_combined_passed
                                        ),
                                    }
                                )
                                if proposed_combined_passed:
                                    probe_translation = proposed_translation
                                    probe_geometry = proposed_geometry
                                    target_volume_passed = (
                                        proposed_target_volume_passed
                                    )
                                    dense_target_volume_passed = (
                                        proposed_dense_passed
                                    )
                                    collision_bounds_passed = (
                                        proposed_collision_passed
                                    )
                                    line_passed = proposed_line_passed
                                    line_evidence = proposed_line_evidence
                                    open_jaw_passed = (
                                        proposed_open_jaw_passed
                                    )
                                    open_jaw_evidence = (
                                        proposed_open_jaw_evidence
                                    )
                                    open_jaw_y_fit["applied"] = True
                        probe_geometry["open_jaw_y_fit"] = open_jaw_y_fit
                        combined_passed = bool(
                            target_volume_passed
                            and line_passed
                            and open_jaw_passed
                        )
                        any_target_volume_passed |= target_volume_passed
                        any_target_volume_and_inner_line_passed |= bool(
                            target_volume_passed and line_passed
                        )
                        any_all_gates_passed |= combined_passed
                        any_dense_collision_rejection |= bool(
                            dense_target_volume_passed
                            and not collision_bounds_passed
                        )

                        cross_section_bounds = line_evidence.get(
                            "target_cross_section_x_bounds_m"
                        )
                        target_cross_section_center_m = None
                        if (
                            isinstance(cross_section_bounds, (list, tuple))
                            and len(cross_section_bounds) == 2
                        ):
                            bounds_array = np.asarray(cross_section_bounds, dtype=float)
                            if np.all(np.isfinite(bounds_array)):
                                target_cross_section_center_m = float(
                                    bounds_array.mean()
                                )
                        inner_line_center_m = line_evidence.get(
                            "inner_line_center_x_m"
                        )
                        if inner_line_center_m is not None:
                            inner_line_center_m = float(inner_line_center_m)
                            if not np.isfinite(inner_line_center_m):
                                inner_line_center_m = None
                        center_offset_m = None
                        if (
                            target_cross_section_center_m is not None
                            and inner_line_center_m is not None
                        ):
                            center_offset_m = abs(
                                inner_line_center_m
                                - target_cross_section_center_m
                            )
                        overlap_m = float(
                            line_evidence.get("target_inner_line_overlap_m", 0.0)
                        )
                        overlap_fraction = float(
                            line_evidence.get(
                                "target_inner_line_overlap_fraction", 0.0
                            )
                        )
                        line_target_count = int(
                            line_evidence.get(
                                "target_points_in_inner_line_count", 0
                            )
                        )
                        collision_bounds_evidence = probe_geometry.get(
                            "collision_bounds_gate", {}
                        )
                        probe_audit = {
                            "depth_m": float(probe_depth),
                            "effective_translation_camera": (
                                probe_translation.tolist()
                            ),
                            "open_jaw_y_fit": open_jaw_y_fit,
                            "target_volume_passed": target_volume_passed,
                            "dense_target_volume_passed": (
                                dense_target_volume_passed
                            ),
                            "target_cross_section_geometry_source": (
                                target_cross_section_geometry_source
                            ),
                            "target_reference_geometry_source": (
                                target_reference_geometry_source
                            ),
                            "collision_bounds_passed": collision_bounds_passed,
                            "collision_local_bounds": collision_bounds_evidence.get(
                                "target_reference_bounds_m"
                            ),
                            "collision_axis_overlap_m": (
                                collision_bounds_evidence.get("axis_overlap_m")
                            ),
                            "collision_positive_axis_overlap": (
                                collision_bounds_evidence.get(
                                    "positive_axis_overlap"
                                )
                            ),
                            "collision_y_center_straddled": (
                                collision_bounds_evidence.get(
                                    "y_center_straddled"
                                )
                            ),
                            "collision_z_center_straddled": (
                                collision_bounds_evidence.get(
                                    "z_center_straddled"
                                )
                            ),
                            "inner_target_point_count": int(
                                probe_geometry["inner_target_point_count"]
                            ),
                            "target_center_straddled": bool(
                                probe_geometry["center_straddled"]
                            ),
                            "inner_line_available": bool(
                                line_evidence.get("available", False)
                            ),
                            "inner_line_passed": line_passed,
                            "inner_line_target_point_count": line_target_count,
                            "inner_line_target_fraction": float(
                                line_evidence.get(
                                    "target_points_in_inner_line_fraction", 0.0
                                )
                            ),
                            "inner_line_overlap_m": overlap_m,
                            "inner_line_overlap_fraction": overlap_fraction,
                            "inner_line_center_m": inner_line_center_m,
                            "target_cross_section_center_m": (
                                target_cross_section_center_m
                            ),
                            "inner_line_center_offset_m": center_offset_m,
                            "inner_line_center_straddled": bool(
                                line_evidence.get(
                                    "inner_line_center_straddled", False
                                )
                            ),
                            "open_jaw_available": bool(
                                open_jaw_evidence.get("available", False)
                            ),
                            "open_jaw_passed": open_jaw_passed,
                            "open_jaw_gap_m": open_jaw_evidence.get(
                                "open_jaw_gap_m"
                            ),
                            "open_jaw_sampled_cross_section_point_count": (
                                open_jaw_evidence.get(
                                    "open_jaw_target_cross_section_point_count"
                                )
                            ),
                            "open_jaw_sampled_cross_section_y_bounds_m": (
                                open_jaw_evidence.get(
                                    "open_jaw_target_cross_section_y_bounds_m"
                                )
                            ),
                            "open_jaw_continuous_cross_section_intersects": (
                                open_jaw_evidence.get(
                                    "open_jaw_continuous_cross_section_intersects"
                                )
                            ),
                            "open_jaw_continuous_cross_section_y_bounds_m": (
                                open_jaw_evidence.get(
                                    "open_jaw_continuous_cross_section_y_bounds_m"
                                )
                            ),
                            "open_jaw_continuous_inner_clearance_m": (
                                open_jaw_evidence.get(
                                    "open_jaw_continuous_inner_clearance_m"
                                )
                            ),
                            "open_jaw_geometry_method": open_jaw_evidence.get(
                                "open_jaw_geometry_method"
                            ),
                            "target_between_open_fingers": open_jaw_evidence.get(
                                "target_between_open_fingers"
                            ),
                            "combined_gate_passed": combined_passed,
                            "selected": False,
                        }
                        depth_fit_audit["probes"].append(probe_audit)
                        if combined_passed:
                            if depth_fit_selection_mode == "centered_coverage":
                                selection_key = (
                                    float("inf")
                                    if center_offset_m is None
                                    else center_offset_m,
                                    -overlap_fraction,
                                    -overlap_m,
                                    -float(line_target_count),
                                    float(probe_depth),
                                )
                            else:
                                selection_key = (float(probe_depth),)
                            passing_probes.append(
                                (
                                    selection_key,
                                    float(probe_depth),
                                    probe_translation.copy(),
                                    probe_geometry,
                                    probe_audit,
                                )
                            )

                    if passing_probes:
                        (
                            _selection_key,
                            selected_depth,
                            selected_translation,
                            selected_geometry,
                            selected_probe_audit,
                        ) = min(passing_probes, key=lambda item: item[0])
                        selected_probe_audit["selected"] = True
                        depth_fit_audit["selected_probe_metrics"] = {
                            "inner_line_center_offset_m": selected_probe_audit[
                                "inner_line_center_offset_m"
                            ],
                            "inner_line_overlap_m": selected_probe_audit[
                                "inner_line_overlap_m"
                            ],
                            "inner_line_overlap_fraction": selected_probe_audit[
                                "inner_line_overlap_fraction"
                            ],
                            "inner_line_target_point_count": selected_probe_audit[
                                "inner_line_target_point_count"
                            ],
                        }

                    if (
                        selected_depth is None
                        or selected_translation is None
                        or selected_geometry is None
                    ):
                        if not any_target_volume_passed:
                            rejection_stage = "target_volume"
                            depth_fit_audit["rejection_reason"] = (
                                "no probe depth passed the target-volume gate"
                            )
                            rejected_target_volume += 1
                            if any_dense_collision_rejection:
                                rejected_collision_bounds += 1
                        elif not any_target_volume_and_inner_line_passed:
                            rejection_stage = "inner_line"
                            depth_fit_audit["rejection_reason"] = (
                                "target-volume passed, but no probe depth passed the "
                                "robot inner-line gate"
                            )
                            rejected_inner_line += 1
                        else:
                            rejection_stage = "open_jaw_continuous_geometry"
                            depth_fit_audit["rejection_reason"] = (
                                "target-volume and robot inner-line passed, but no "
                                "probe depth passed continuous open-jaw containment"
                            )
                            rejected_open_jaw += 1
                        depth_fit_audit["rejection_stage"] = rejection_stage
                        depth_fit_audit["any_target_volume_passed"] = (
                            any_target_volume_passed
                        )
                        depth_fit_audit["any_target_volume_and_inner_line_passed"] = (
                            any_target_volume_and_inner_line_passed
                        )
                        depth_fit_audit["any_all_gates_passed"] = any_all_gates_passed
                        depth_fit_audits.append(
                            {
                                "detector_rank": original_snapshot.get("rank"),
                                "score": float(candidate.score),
                                **depth_fit_audit,
                            }
                        )
                        logger.warning(
                            "rejecting AnyGrasp candidate after robot-aware depth sweep: "
                            "rank=%s score=%.4f sdk_depth=%.4f probes=%d stage=%s",
                            original_snapshot.get("rank"),
                            float(candidate.score),
                            sdk_depth,
                            len(probe_depths),
                            rejection_stage,
                        )
                        continue

                    depth_fit_audit["selected_depth_m"] = selected_depth
                    depth_fit_audit["applied"] = not np.isclose(
                        selected_depth, sdk_depth, atol=1e-9, rtol=0.0
                    )
                    depth_fit_audits.append(
                        {
                            "detector_rank": original_snapshot.get("rank"),
                            "score": float(candidate.score),
                            **depth_fit_audit,
                        }
                    )
                    recenter_audit = {
                        key: value
                        for key, value in target_geometry.items()
                        if key.startswith("recenter")
                    }
                    candidate.depth = selected_depth
                    translation = np.asarray(
                        selected_translation, dtype=np.float64
                    ).reshape(3)
                    candidate.translation = translation.astype(np.float32)
                    selected_geometry["candidate_original_snapshot"] = original_snapshot
                    selected_geometry.update(recenter_audit)
                    selected_geometry["depth_fit"] = depth_fit_audit
                    target_geometry = selected_geometry
                    logger.warning(
                        "fit AnyGrasp candidate to robot inner line: rank=%s "
                        "sdk_depth=%.4f fitted_depth=%.4f probes=%d mode=%s applied=%s",
                        original_snapshot.get("rank"),
                        sdk_depth,
                        selected_depth,
                        len(probe_depths),
                        depth_fit_selection_mode,
                        bool(depth_fit_audit["applied"]),
                    )

                inner_count = int(target_geometry["inner_target_point_count"])
                center_straddled = bool(target_geometry["center_straddled"])
                jaw_center_distance = target_geometry[
                    "jaw_box_center_to_target_centroid_m"
                ]
                (
                    target_volume_passed,
                    dense_target_volume_passed,
                    collision_bounds_passed,
                ) = target_volume_gate_for(target_geometry)
                if not target_volume_passed:
                    rejected_target_volume += 1
                    if dense_target_volume_passed and not collision_bounds_passed:
                        rejected_collision_bounds += 1
                    collision_bounds_evidence = target_geometry.get(
                        "collision_bounds_gate", {}
                    )
                    logger.warning(
                        "rejecting AnyGrasp candidate outside target closing volume: "
                        "inner_points=%d/%d required=%d center_straddled=%s required=%s "
                        "dense_passed=%s collision_bounds_passed=%s "
                        "collision_overlap=%s collision_yz_center=(%s,%s) "
                        "jaw_center_distance=%.4f recentered=%s "
                        "cross_section_bounds=%s reference_bounds=%s",
                        inner_count,
                        int(target_geometry["target_cross_section_point_count"]),
                        min_inner_points,
                        center_straddled,
                        require_center_straddle,
                        dense_target_volume_passed,
                        collision_bounds_passed,
                        collision_bounds_evidence.get("axis_overlap_m"),
                        collision_bounds_evidence.get("y_center_straddled"),
                        collision_bounds_evidence.get("z_center_straddled"),
                        float(jaw_center_distance or 0.0),
                        bool(target_geometry["recenter_applied"]),
                        target_geometry["target_cross_section_local_bounds"],
                        target_geometry["target_reference_local_bounds"],
                    )
                    continue

                if inner_line_gate_enabled:
                    existing_evidence = target_geometry.get("robot_inner_grasp_line")
                    if isinstance(existing_evidence, dict):
                        inner_line_evidence = existing_evidence
                        line_passed = bool(inner_line_evidence.get("gate_passed", False))
                    else:
                        line_passed, inner_line_evidence = inner_line_gate_for(
                            candidate, rotation, translation
                        )
                        target_geometry["robot_inner_grasp_line"] = inner_line_evidence
                    line_count = int(
                        inner_line_evidence.get("target_points_in_inner_line_count", 0)
                    )
                    line_fraction = float(
                        inner_line_evidence.get(
                            "target_points_in_inner_line_fraction", 0.0
                        )
                    )
                    line_overlap = float(
                        inner_line_evidence.get("target_inner_line_overlap_m", 0.0)
                    )
                    line_centered = bool(
                        inner_line_evidence.get("inner_line_center_straddled", False)
                    )
                    if not line_passed:
                        rejected_inner_line += 1
                        snapshot = getattr(
                            candidate, "anygrasp_original_snapshot", {}
                        )
                        logger.warning(
                            "rejecting robot-incompatible AnyGrasp candidate: "
                            "rank=%s score=%.4f depth=%.4f original_translation=%s "
                            "current_translation=%s available=%s line_points=%d/%d "
                            "fraction=%.4f/%.4f overlap=%.4f/%.4f "
                            "center_straddled=%s required=%s inner_line=%s "
                            "target_x=%s reason=%s",
                            snapshot.get("rank"),
                            float(candidate.score),
                            float(candidate.depth),
                            snapshot.get("original_camera_translation"),
                            np.asarray(candidate.translation, dtype=float).reshape(3).tolist(),
                            bool(inner_line_evidence.get("available", False)),
                            line_count,
                            inner_line_min_points,
                            line_fraction,
                            inner_line_min_fraction,
                            line_overlap,
                            inner_line_min_overlap_m,
                            line_centered,
                            inner_line_require_center,
                            inner_line_evidence.get(
                                "effective_inner_line_x_interval_m"
                            ),
                            inner_line_evidence.get("target_cross_section_x_bounds_m"),
                            inner_line_evidence.get("unavailable_reason"),
                        )
                        continue

                if collision_geometry_enabled:
                    existing_open_jaw = target_geometry.get(
                        "robot_open_jaw_containment"
                    )
                    if isinstance(existing_open_jaw, dict):
                        open_jaw_evidence = existing_open_jaw
                        open_jaw_passed = bool(
                            open_jaw_evidence.get("gate_passed", False)
                        )
                    else:
                        open_jaw_passed, open_jaw_evidence = open_jaw_gate_for(
                            candidate, rotation, translation
                        )
                        target_geometry["robot_open_jaw_containment"] = (
                            open_jaw_evidence
                        )
                    if not open_jaw_passed:
                        rejected_open_jaw += 1
                        snapshot = getattr(
                            candidate, "anygrasp_original_snapshot", {}
                        )
                        logger.warning(
                            "rejecting AnyGrasp candidate outside actual open jaw: "
                            "rank=%s score=%.4f depth=%.4f available=%s "
                            "gap=%s sampled_points=%s sampled_y=%s "
                            "continuous_intersects=%s continuous_y=%s "
                            "continuous_clearance=%s between_fingers=%s reason=%s",
                            snapshot.get("rank"),
                            float(candidate.score),
                            float(candidate.depth),
                            bool(open_jaw_evidence.get("available", False)),
                            open_jaw_evidence.get("open_jaw_gap_m"),
                            open_jaw_evidence.get(
                                "open_jaw_target_cross_section_point_count"
                            ),
                            open_jaw_evidence.get(
                                "open_jaw_target_cross_section_y_bounds_m"
                            ),
                            open_jaw_evidence.get(
                                "open_jaw_continuous_cross_section_intersects"
                            ),
                            open_jaw_evidence.get(
                                "open_jaw_continuous_cross_section_y_bounds_m"
                            ),
                            open_jaw_evidence.get(
                                "open_jaw_continuous_inner_clearance_m"
                            ),
                            open_jaw_evidence.get("target_between_open_fingers"),
                            open_jaw_evidence.get("unavailable_reason")
                            or open_jaw_evidence.get("reason"),
                        )
                        continue

                if non_target_collision_audit_enabled:
                    non_target_local_points = (
                        non_target_points - translation
                    ) @ rotation
                    non_target_collision = non_target_collision_preflight(
                        candidate,
                        non_target_local_points,
                        margin_m=non_target_collision_margin_m,
                    )
                    target_geometry["non_target_collision_audit"] = (
                        non_target_collision
                    )
                    logger.warning(
                        "AnyGrasp non-target collision audit: rank=%s depth=%.4f "
                        "available=%s points_in_component_aabb=%s/%d components=%s",
                        getattr(candidate, "anygrasp_original_snapshot", {}).get(
                            "rank"
                        ),
                        float(candidate.depth),
                        bool(non_target_collision.get("available", False)),
                        non_target_collision.get(
                            "non_target_points_in_any_component_aabb"
                        ),
                        int(len(non_target_points)),
                        [
                            {
                                "role": component.get("role"),
                                "name": component.get("name"),
                                "points": component.get(
                                    "non_target_points_in_aabb"
                                ),
                                "distance_m": component.get(
                                    "minimum_non_target_distance_to_aabb_m"
                                ),
                            }
                            for component in non_target_collision.get(
                                "components", []
                            )
                        ],
                    )

                setattr(candidate, "target_geometry_evidence", target_geometry)
                geometry_candidates.append((candidate, approach_world_z, target_geometry))

            def detector_rank_preference(candidate: Any) -> int:
                snapshot = getattr(candidate, "anygrasp_original_snapshot", {})
                raw_rank = snapshot.get("rank") if isinstance(snapshot, dict) else None
                if isinstance(raw_rank, bool) or not isinstance(
                    raw_rank, (int, np.integer)
                ):
                    return len(preferred_rank_order)
                return preferred_rank_order.get(
                    int(raw_rank), len(preferred_rank_order)
                )

            # The explicit detector-rank preference is a diagnostic ordering override,
            # not a geometry gate. All candidates reaching this point already passed
            # the target-volume and robot inner-line checks. Preserve the existing
            # geometry ordering for candidates with the same preference priority.
            geometry_candidates.sort(
                key=lambda item: (
                    detector_rank_preference(item[0]),
                    0 if float(item[0].width) >= min_candidate_width else 1,
                    -float(
                        (item[2].get("robot_inner_grasp_line") or {}).get(
                            "target_inner_line_overlap_fraction", 0.0
                        )
                    ),
                    -int(
                        (item[2].get("robot_inner_grasp_line") or {}).get(
                            "target_points_in_inner_line_count", 0
                        )
                    ),
                    0 if item[2]["center_straddled"] else 1,
                    (
                        float(item[2]["jaw_box_center_to_target_centroid_m"])
                        if item[2]["jaw_box_center_to_target_centroid_m"] is not None
                        else float("inf")
                    ),
                    -int(item[2]["inner_target_point_count"]),
                    (
                        float(item[2]["target_center_offset_y_m"])
                        if item[2]["target_center_offset_y_m"] is not None
                        else float("inf")
                    ),
                    item[1],
                    -float(
                        getattr(item[0], "anygrasp_original_snapshot", {}).get(
                            "depth", item[0].depth
                        )
                    ),
                    -float(item[0].width),
                    -float(item[0].score),
                )
            )
            candidates = [candidate for candidate, _world_z, _geometry in geometry_candidates]
            logger.warning(
                "AnyGrasp candidate geometry: detected=%d anchored=%d "
                "rejected_approach=%d rejected_recenter=%d rejected_target_volume=%d "
                "rejected_collision_bounds=%d rejected_inner_line=%d "
                "rejected_open_jaw=%d narrow_deprioritized=%d remaining=%d",
                detected_count,
                anchored_count,
                rejected_approach,
                rejected_recenter,
                rejected_target_volume,
                rejected_collision_bounds,
                rejected_inner_line,
                rejected_open_jaw,
                narrow_candidates,
                len(candidates),
            )
            for rank, (candidate, approach_world_z, target_geometry) in enumerate(
                geometry_candidates[:5], start=1
            ):
                translation_world = (
                    camera_pose[:3, :3]
                    @ np.asarray(candidate.translation, dtype=np.float64).reshape(3)
                    + camera_pose[:3, 3]
                )
                line_evidence = target_geometry.get("robot_inner_grasp_line") or {}
                logger.warning(
                    "AnyGrasp ranked candidate %d score=%.3f width=%.4f depth=%.4f "
                    "world_z=%.4f grasp_origin=%s inner_target_points=%d/%d "
                    "inner_fraction=%.4f center_straddled=%s center_offset_y=%.4f "
                    "jaw_center_distance=%.4f recentered=%s recenter_shift=%.4f "
                    "robot_line=%s line_points=%d line_fraction=%.4f "
                    "line_overlap=%.4f line_center_straddled=%s local_bounds=%s",
                    rank,
                    float(candidate.score),
                    float(candidate.width),
                    float(candidate.depth),
                    approach_world_z,
                    np.round(translation_world, 4).tolist(),
                    int(target_geometry["inner_target_point_count"]),
                    int(target_geometry["target_point_count"]),
                    float(target_geometry["inner_target_fraction"]),
                    bool(target_geometry["center_straddled"]),
                    float(target_geometry["target_center_offset_y_m"] or 0.0),
                    float(target_geometry["jaw_box_center_to_target_centroid_m"] or 0.0),
                    bool(target_geometry["recenter_applied"]),
                    float(target_geometry.get("recenter_translation_m", 0.0)),
                    line_evidence.get("effective_inner_line_x_interval_m"),
                    int(line_evidence.get("target_points_in_inner_line_count", 0)),
                    float(line_evidence.get("target_points_in_inner_line_fraction", 0.0)),
                    float(line_evidence.get("target_inner_line_overlap_m", 0.0)),
                    bool(line_evidence.get("inner_line_center_straddled", False)),
                    target_geometry["target_local_bounds"],
                )

            min_translation = float(
                self._anygrasp_config.get("candidate_min_translation_m", 0.015)
            )
            min_angle_deg = float(
                self._anygrasp_config.get("candidate_min_approach_angle_deg", 10.0)
            )
            if not np.isfinite(min_translation) or min_translation < 0.0:
                raise ValueError("candidate_min_translation_m must be finite and non-negative")
            if not np.isfinite(min_angle_deg) or not 0.0 <= min_angle_deg <= 180.0:
                raise ValueError(
                    "candidate_min_approach_angle_deg must be finite and in [0, 180]"
                )
            min_angle = np.deg2rad(min_angle_deg)

            def is_similar(first: np.ndarray, second: np.ndarray) -> bool:
                if np.linalg.norm(first[:3] - second[:3]) >= min_translation:
                    return False
                first_approach = first[3:] / (np.linalg.norm(first[3:]) + 1e-12)
                second_approach = second[3:] / (np.linalg.norm(second[3:]) + 1e-12)
                angle = np.arccos(np.clip(np.dot(first_approach, second_approach), -1.0, 1.0))
                return bool(angle < min_angle)

            failed_signatures = [np.asarray(key, dtype=float) for key in self._failed_candidates]
            unique_candidates: list[Any] = []
            unique_signatures: list[np.ndarray] = []
            rejected_similar = 0
            for candidate in candidates:
                signature = np.asarray(
                    self._candidate_key(candidate, packet.camera_pose_world), dtype=float
                )
                if any(is_similar(signature, failed) for failed in failed_signatures):
                    rejected_similar += 1
                    continue
                if any(is_similar(signature, kept) for kept in unique_signatures):
                    rejected_similar += 1
                    continue
                unique_candidates.append(candidate)
                unique_signatures.append(signature)
            candidates = unique_candidates
            skill_funnel = {
                "detector_returned_count": detected_count,
                "anchored_count": anchored_count,
                "rejected_approach_count": rejected_approach,
                "rejected_recenter_count": rejected_recenter,
                "rejected_target_volume_count": rejected_target_volume,
                "rejected_collision_bounds_count": rejected_collision_bounds,
                "rejected_inner_line_count": rejected_inner_line,
                "rejected_open_jaw_count": rejected_open_jaw,
                "narrow_deprioritized_count": narrow_candidates,
                "post_geometry_count": int(len(geometry_candidates)),
                "dedupe_rejected_count": rejected_similar,
                "post_dedupe_count": int(len(candidates)),
                "queued_count": None,
                "ranking_preference": {
                    "preferred_detector_ranks": list(preferred_detector_ranks),
                    "diagnostic_only": True,
                    "hard_gate": False,
                },
                "target_collision_geometry": collision_geometry_audit,
                "depth_fit": {
                    "enabled": depth_fit_enabled,
                    "minimum_probe_depth_m": depth_fit_min_m,
                    "maximum_probe_depth_m": depth_fit_max_m,
                    "probe_step_m": depth_fit_step_m,
                    "selection_mode": depth_fit_selection_mode,
                    "audited_candidate_count": len(depth_fit_audits),
                    "selected_candidate_count": sum(
                        audit.get("selected_depth_m") is not None
                        for audit in depth_fit_audits
                    ),
                    "candidates": depth_fit_audits,
                },
                "non_target_collision_audit": {
                    "enabled": non_target_collision_audit_enabled,
                    "margin_m": non_target_collision_margin_m,
                    "audited_candidate_count": (
                        len(geometry_candidates)
                        if non_target_collision_audit_enabled
                        else 0
                    ),
                    "candidates": [
                        {
                            "detector_rank": getattr(
                                candidate, "anygrasp_original_snapshot", {}
                            ).get("rank"),
                            "score": float(candidate.score),
                            "sdk_original_depth_m": float(
                                getattr(
                                    candidate, "anygrasp_original_snapshot", {}
                                ).get("depth", candidate.depth)
                            ),
                            "effective_depth_m": float(candidate.depth),
                            "evidence": geometry.get(
                                "non_target_collision_audit"
                            ),
                        }
                        for candidate, _world_z, geometry in geometry_candidates
                    ],
                },
            }
            if self._pending_candidate_detection_audit is None:
                self._pending_candidate_detection_audit = {
                    "event": "anygrasp_skill_candidate_funnel_audit",
                    "detector_audit": None,
                }
            self._pending_candidate_detection_audit["skill_funnel"] = skill_funnel
            if rejected_similar:
                logger.info(
                    "rejected %d duplicate/previously-failed AnyGrasp candidates "
                    "(translation<%.3f m, approach<%.1f deg)",
                    rejected_similar,
                    min_translation,
                    min_angle_deg,
                )
            if candidates:
                logger.info(
                    "AnyGrasp detected %d target-conditioned candidates; best score=%.3f",
                    len(candidates),
                    candidates[0].score,
                )
            else:
                if rejected_open_jaw:
                    self._last_execution_error = (
                        "AnyGrasp returned no robot-compatible candidate: continuous "
                        "open-jaw geometry rejected "
                        f"{rejected_open_jaw} candidate(s) after target-volume and "
                        "inner-line gates passed"
                    )
                elif rejected_inner_line:
                    self._last_execution_error = (
                        "AnyGrasp returned no robot-compatible candidate: actual inner "
                        f"grasp-line gate rejected {rejected_inner_line} candidate(s)"
                    )
                elif detected_count or not self._last_execution_error:
                    self._last_execution_error = (
                        "AnyGrasp returned no unused target grasp candidate"
                    )
            return candidates, packet
        except Exception as exc:
            self._last_execution_error = f"AnyGrasp observation/detection failed: {exc}"
            if self._pending_candidate_detection_audit is None:
                self._pending_candidate_detection_audit = {
                    "event": "anygrasp_skill_candidate_funnel_audit",
                    "detector_audit": None,
                }
            self._pending_candidate_detection_audit["error"] = self._last_execution_error
            logger.warning(self._last_execution_error)
            return [], None

    def _find_target_object(self, subtask: Subtask) -> Any | None:
        target_name = self._target_name(subtask)
        if not target_name:
            return None

        def normalized(value: Any) -> str:
            return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())

        def prefix(value: Any) -> str:
            tokens = "".join(
                ch if ch.isalnum() else " " for ch in str(value).strip().lower()
            ).split()
            return tokens[0] if tokens else ""

        try:
            import omnigibson as og
            scene = og.sim.scenes[0]
            objects = list(getattr(scene, "objects", ()))
            target_normalized = normalized(target_name)
            target_prefix = prefix(target_name)
            prefix_matches = []
            for obj in objects:
                identities = (
                    getattr(obj, "name", ""),
                    getattr(obj, "category", ""),
                    getattr(obj, "model", ""),
                    getattr(obj, "prim_path", ""),
                )
                if any(normalized(identity) == target_normalized for identity in identities):
                    return obj
                if len(target_prefix) >= 3 and any(
                    prefix(identity) == target_prefix for identity in identities
                ):
                    prefix_matches.append(obj)
            if len(prefix_matches) == 1:
                matched = prefix_matches[0]
                logger.info(
                    "matched grasp target '%s' to scene object '%s' (category=%s)",
                    target_name,
                    getattr(matched, "name", ""),
                    getattr(matched, "category", ""),
                )
                return matched
            if prefix_matches:
                self._last_execution_error = (
                    f"target '{target_name}' is ambiguous across "
                    f"{[getattr(obj, 'name', '') for obj in prefix_matches]}"
                )
        except Exception as exc:
            logger.debug("target lookup failed: %s", exc)
        return None

    def _find_scene_object_by_name(self, name: str) -> Any | None:
        if not name:
            return None
        normalized_name = "".join(ch for ch in name.lower() if ch.isalnum())
        prefix_name = "".join(ch if ch.isalnum() else " " for ch in name.lower()).split()
        prefix_name = prefix_name[0] if prefix_name else ""
        try:
            import omnigibson as og

            objects = list(getattr(og.sim.scenes[0], "objects", ()))
            prefix_matches = []
            for obj in objects:
                identities = (
                    getattr(obj, "name", ""),
                    getattr(obj, "category", ""),
                    getattr(obj, "model", ""),
                    getattr(obj, "prim_path", ""),
                )
                if any(
                    "".join(ch for ch in str(identity).lower() if ch.isalnum())
                    == normalized_name
                    for identity in identities
                ):
                    return obj
                if prefix_name and any(
                    "".join(ch if ch.isalnum() else " " for ch in str(identity).lower()).split()[:1]
                    == [prefix_name]
                    for identity in identities
                ):
                    prefix_matches.append(obj)
            return prefix_matches[0] if len(prefix_matches) == 1 else None
        except Exception as exc:
            logger.debug("destination lookup failed: %s", exc)
            return None

    def _build_place_result(
        self,
        subtask: Subtask,
        selection: LocalSkillSelection,
        outcome: Any,
        final_action: dict[str, Any],
        start: float,
    ) -> AgentResult:
        evidence = dict(getattr(outcome, "physical_evidence", {}) or {})
        success = bool(getattr(outcome, "success", False))
        return AgentResult(
            subtask_id=subtask.subtask_id,
            status=AgentStatus.SUCCESS if success else AgentStatus.FAILURE,
            error_code=None if success else "PLACE_EXECUTION_FAILED",
            result={
                "action_keys": sorted(final_action),
                "skill_id": self.skill_id,
                "skill_source": "anygrasp_place_inside",
                "placement_success": success,
                "placement_verified": bool(getattr(outcome, "placement_verified", success)),
                "destination_object": self._destination_name(subtask),
                "sim_steps": int(getattr(outcome, "total_sim_steps", 0)),
                "placement_error": getattr(outcome, "error", None),
                "physical_evidence": evidence,
                "selector_confidence": selection.confidence,
            },
            runtime_artifacts={
                "full_action": final_action,
                "projected_action": final_action,
                "skill_selection": {
                    "skill_id": self.skill_id,
                    "source": "anygrasp_place_inside",
                    "confidence": selection.confidence,
                    "reason": selection.reason,
                },
                "physical_evidence": evidence,
                "placement_verified": bool(getattr(outcome, "placement_verified", success)),
            },
            latency_ms=self._latency_ms(start),
        )

    def _advance_place_execution(
        self, subtask: Subtask, selection: LocalSkillSelection, start: float
    ) -> AgentResult | None:
        if self._active_execution is None:
            return None
        try:
            action, outcome = self._active_execution.advance()
        except Exception as exc:
            self._last_execution_error = f"placement generator failed: {type(exc).__name__}: {exc}"
            logger.exception("PLACE_INSIDE execution could not return its terminal evidence")
            self._active_execution = None
            return None
        if action is not None:
            return self._build_action_result(
                subtask, selection, action, start, source="anygrasp_place_inside"
            )
        if outcome is None:
            self._last_execution_error = "placement execution produced neither action nor outcome"
            return None
        # ``last_action`` has already been applied by the environment on the
        # preceding control step.  A terminal placement result must not carry
        # it as a fresh runtime action: doing so makes the closed-loop runner
        # execute the stale frame once more and call this skill again after
        # the gripper has released the object.  Keep only the action keys as
        # audit evidence and return an action-free terminal result.
        evidence = getattr(outcome, "physical_evidence", None)
        if isinstance(evidence, dict):
            evidence.setdefault(
                "last_applied_action_keys",
                sorted(self._active_execution.last_action),
            )
        final_action: dict[str, Any] = {}
        self._active_execution = None
        return self._build_place_result(
            subtask, selection, outcome, final_action, start
        )

    def _start_place_execution(self, subtask: Subtask, executor: Any) -> bool:
        destination = self._find_scene_object_by_name(self._destination_name(subtask))
        if destination is None:
            self._last_execution_error = (
                f"placement destination '{self._destination_name(subtask)}' not found"
            )
            return False
        begin_place = getattr(executor, "begin_place_inside", None)
        if not callable(begin_place):
            self._last_execution_error = "executor does not support PLACE_INSIDE"
            return False
        raw_cell_index = subtask.target.get("cell_index")
        if raw_cell_index is None:
            raw_cell_index = subtask.parameters.get("cell_index")
        cell_index = None if raw_cell_index is None else int(raw_cell_index)
        grid_shape = self._anygrasp_config.get(
            "place_inside_grid_shape", [1, 3]
        )
        if not isinstance(grid_shape, (list, tuple)) or len(grid_shape) != 2:
            self._last_execution_error = (
                "place_inside_grid_shape must contain [rows, columns]"
            )
            return False
        self._active_execution = begin_place(
            destination,
            cell_index=cell_index,
            grid_shape=[int(value) for value in grid_shape],
            cell_margin_m=float(
                self._anygrasp_config.get("place_inside_cell_margin_m", 0.005)
            ),
        )
        self._active_source = "anygrasp_place_inside"
        return True

    def _start_anygrasp_execution(self, subtask: Subtask, executor: Any) -> bool:
        max_attempts = max(1, int(self._anygrasp_config.get("max_attempts", 3)))
        if self._anygrasp_attempts >= max_attempts:
            return False
        max_detection_batches = max(
            1, int(self._anygrasp_config.get("candidate_detection_refreshes", 1))
        )
        if max_detection_batches > 20:
            raise ValueError("candidate_detection_refreshes must be at most 20")
        target_obj = self._find_target_object(subtask)
        if target_obj is None and bool(
            self._anygrasp_config.get("require_target_object", True)
        ):
            self._last_execution_error = (
                f"target object '{self._target_name(subtask)}' not found in scene"
            )
            return False
        while (
            not self._candidate_queue
            and self._candidate_detection_batches < max_detection_batches
        ):
            previous_execution_error = self._last_execution_error
            candidates, packet = self._detect_candidates(subtask, target_obj)
            self._candidate_detection_batches += 1
            if packet is None and previous_execution_error:
                refresh_error = self._last_execution_error
                self._last_execution_error = (
                    f"{previous_execution_error}; candidate refresh failed: {refresh_error}"
                )
            self._candidate_batch_loaded = packet is not None
            remaining_attempts = max_attempts - self._anygrasp_attempts
            compatible_candidate_count = len(candidates)
            if self._candidate_detection_only:
                self._candidate_queue = []
            else:
                self._candidate_queue = candidates[:remaining_attempts]
            self._candidate_packet = packet
            audit = self._pending_candidate_detection_audit or {
                "event": "anygrasp_skill_candidate_funnel_audit",
                "detector_audit": None,
            }
            audit["detection_batch"] = self._candidate_detection_batches
            audit["max_detection_batches"] = max_detection_batches
            funnel = audit.setdefault("skill_funnel", {})
            funnel["candidate_detection_only"] = self._candidate_detection_only
            funnel["audited_compatible_candidate_count"] = compatible_candidate_count
            funnel["queued_count"] = int(len(self._candidate_queue))
            funnel["remaining_attempt_slots"] = int(remaining_attempts)
            saved_audit = json.loads(json.dumps(audit))
            self._candidate_detection_audits.append(saved_audit)
            self._pending_candidate_detection_audit = None
            logger.info(
                "%s", json.dumps(saved_audit, sort_keys=True, separators=(",", ":"))
            )
            if self._candidate_detection_only and packet is not None:
                self._last_execution_error = (
                    "AnyGrasp candidate detection-only audit completed; "
                    "robot execution intentionally disabled"
                )
                logger.info(
                    "AnyGrasp detection-only batch %d/%d audited %d compatible "
                    "candidate(s); no candidate was queued",
                    self._candidate_detection_batches,
                    max_detection_batches,
                    compatible_candidate_count,
                )
            elif self._candidate_detection_only:
                logger.warning(
                    "AnyGrasp detection-only batch %d/%d failed before a detection "
                    "packet was produced: %s",
                    self._candidate_detection_batches,
                    max_detection_batches,
                    self._last_execution_error,
                )
            elif not self._candidate_queue:
                logger.warning(
                    "AnyGrasp detection batch %d/%d produced no usable candidate",
                    self._candidate_detection_batches,
                    max_detection_batches,
                )
        if not self._candidate_queue or self._candidate_packet is None:
            return False
        candidate = self._candidate_queue.pop(0)
        self._failed_candidates.add(
            self._candidate_key(candidate, self._candidate_packet.camera_pose_world)
        )
        self._anygrasp_attempts += 1
        self._active_execution = executor.begin_grasp(
            candidate,
            camera_pose_world=self._candidate_packet.camera_pose_world,
            target_obj=target_obj,
        )
        self._active_source = "anygrasp_curobo"
        return True


    def _start_builtin_execution(self, subtask: Subtask, executor: Any) -> bool:
        if self._builtin_attempted:
            return False
        self._builtin_attempted = True
        target_obj = self._find_target_object(subtask)
        if target_obj is None:
            self._last_execution_error = "OmniGibson built-in fallback target was not found"
            return False
        self._active_execution = executor.begin_grasp_by_object(target_obj)
        self._active_source = "og_builtin_curobo"
        return True

    def _build_action_result(
        self,
        subtask: Subtask,
        selection: LocalSkillSelection,
        action: dict[str, Any],
        start: float,
        *,
        source: str | None = None,
    ) -> AgentResult:
        skill_source = source or self._active_source
        return AgentResult(
            subtask_id=subtask.subtask_id,
            status=AgentStatus.SUCCESS,
            result={
                "action_keys": sorted(action),
                "control_mode": resolve_control_mode(subtask),
                "skill_id": self.skill_id,
                "skill_source": skill_source,
                "grasp_plan_completed": False,
                "grasp_success": False,
                "grasp_attempt": self._anygrasp_attempts,
                "selector_confidence": selection.confidence,
            },
            runtime_artifacts={
                "full_action": action,
                "projected_action": action,
                "skill_selection": {
                    "skill_id": self.skill_id,
                    "source": skill_source,
                    "confidence": selection.confidence,
                    "reason": selection.reason,
                },
            },
            latency_ms=self._latency_ms(start),
        )

    def _build_grasp_result(
        self,
        subtask: Subtask,
        selection: LocalSkillSelection,
        outcome: Any,
        source: str,
        final_action: dict[str, Any],
        start: float,
    ) -> AgentResult:
        target = self._target_name(subtask)
        physical_grasp_verified = bool(
            getattr(outcome, "physical_grasp_verified", False)
        )
        physical_evidence = dict(getattr(outcome, "physical_evidence", {}) or {})
        try:
            self.memory.record_action(
                {
                    "action_type": subtask.action,
                    "target": target,
                    "success": outcome.success,
                    "skill_id": self.skill_id,
                    "skill_source": source,
                    "anygrasp_score": outcome.anygrasp_score,
                    "sim_steps": outcome.total_sim_steps,
                    "physical_grasp_verified": physical_grasp_verified,
                    "physical_evidence": physical_evidence,
                }
            )
        except Exception:
            pass
        return AgentResult(
            subtask_id=subtask.subtask_id,
            status=AgentStatus.SUCCESS if outcome.success else AgentStatus.FAILURE,
            error_code=None if outcome.success else "GRASP_EXECUTION_FAILED",
            result={
                "action_keys": sorted(final_action),
                "control_mode": resolve_control_mode(subtask),
                "skill_id": self.skill_id,
                "skill_source": source,
                "grasp_plan_completed": True,
                "grasp_success": bool(outcome.success),
                "physical_grasp_verified": physical_grasp_verified,
                "physical_evidence": physical_evidence,
                "object_in_hand": outcome.object_in_hand,
                "target_object": target,
                "anygrasp_score": outcome.anygrasp_score,
                "sim_steps": outcome.total_sim_steps,
                "grasp_error": outcome.error,
                "selector_confidence": selection.confidence,
            },
            runtime_artifacts={
                "full_action": final_action,
                "projected_action": final_action,
                "skill_selection": {
                    "skill_id": self.skill_id,
                    "source": source,
                    "confidence": selection.confidence,
                    "reason": selection.reason,
                },
                "grasp_pose_world": np.asarray(outcome.grasp_pos_world).tolist(),
                "physical_grasp_verified": physical_grasp_verified,
                "physical_evidence": physical_evidence,
            },
            latency_ms=self._latency_ms(start),
        )


    def _advance_pre_detection_release(
        self,
        subtask: Subtask,
        selection: LocalSkillSelection,
        start: float,
        executor: Any,
    ) -> AgentResult | None:
        if self._pre_detection_release_completed or self._pre_detection_release_failed:
            return None
        begin_release = getattr(executor, "begin_release", None)
        if not callable(begin_release):
            # Preserve compatibility with external executors that pre-open internally.
            self._pre_detection_release_completed = True
            return None
        if self._pre_detection_release_execution is None:
            self._pre_detection_release_execution = begin_release()
        try:
            action, outcome = self._pre_detection_release_execution.advance()
        except Exception as exc:
            self._last_execution_error = (
                "AnyGrasp pre-detection release generator failed: "
                f"{type(exc).__name__}: {exc}"
            )
            logger.warning(self._last_execution_error)
            self._pre_detection_release_execution = None
            self._pre_detection_release_failed = True
            return None
        if action is not None:
            return self._build_action_result(
                subtask,
                selection,
                action,
                start,
                source="anygrasp_pre_detection_release",
            )
        self._pre_detection_release_execution = None
        if outcome is None:
            self._last_execution_error = (
                "AnyGrasp pre-detection release produced neither action nor outcome"
            )
            self._pre_detection_release_failed = True
        elif outcome.success:
            self._pre_detection_release_completed = True
            logger.info(
                "AnyGrasp pre-detection release completed: %s",
                getattr(outcome, "physical_evidence", {}),
            )
        else:
            self._last_execution_error = outcome.error or (
                "AnyGrasp pre-detection release failed"
            )
            self._pre_detection_release_failed = True
            logger.warning(self._last_execution_error)
        return None

    def _advance_active_execution(
        self,
        subtask: Subtask,
        selection: LocalSkillSelection,
        start: float,
    ) -> AgentResult | None:
        execution = self._active_execution
        source = self._active_source
        if execution is None:
            return None
        try:
            action, outcome = execution.advance()
        except Exception as exc:
            self._last_execution_error = f"{source} generator failed: {type(exc).__name__}: {exc}"
            logger.warning(self._last_execution_error)
            self._active_execution = None
            self._active_source = ""
            self._invalidate_candidate_batch()
            return None
        if action is not None:
            return self._build_action_result(subtask, selection, action, start)
        if outcome is None:
            raise RuntimeError("grasp execution produced neither action nor outcome")
        final_action = dict(execution.last_action)
        self._active_execution = None
        self._active_source = ""
        if outcome.success:
            # The final simulator action was already applied on the preceding
            # control cycle.  Replaying it here makes the closed-loop runner
            # execute the same subtask again while it waits for the overall
            # BDDL goal (which normally cannot become true until PLACE_INSIDE).
            # Return an action-free terminal grasp result and retain only the
            # keys of the already-applied action as audit evidence.  The
            # runtime accepts this narrow terminal form only when the full
            # physical grasp evidence passes its independent checks.
            physical_evidence = getattr(outcome, "physical_evidence", None)
            if isinstance(physical_evidence, dict):
                physical_evidence.setdefault(
                    "last_applied_action_keys",
                    sorted(final_action),
                )
            final_action = {}
            self._candidate_queue.clear()
            return self._build_grasp_result(
                subtask,
                selection,
                outcome,
                source,
                final_action,
                start,
            )
        self._last_execution_error = outcome.error or f"{source} failed"
        physical_evidence = dict(getattr(outcome, "physical_evidence", {}) or {})
        failure_audit = {
            "skill_source": source,
            "error": self._last_execution_error,
            "failure_phase": getattr(outcome, "failure_phase", None),
            "scene_changed": bool(getattr(outcome, "scene_changed", False)),
            "object_in_hand": getattr(outcome, "object_in_hand", None),
            "anygrasp_score": float(getattr(outcome, "anygrasp_score", 0.0)),
            "sim_steps": int(getattr(outcome, "total_sim_steps", 0)),
            "physical_grasp_verified": bool(
                getattr(outcome, "physical_grasp_verified", False)
            ),
            "grasp_pose_world": {
                "position": np.asarray(
                    getattr(outcome, "grasp_pos_world", []), dtype=float
                ).tolist(),
                "orientation": np.asarray(
                    getattr(outcome, "grasp_quat_world", []), dtype=float
                ).tolist(),
            },
            "physical_evidence": physical_evidence,
        }
        self._execution_failure_audits.append(failure_audit)
        logger.warning("%s attempt failed: %s", source, self._last_execution_error)
        if failure_audit["scene_changed"]:
            logger.info(
                "invalidating AnyGrasp candidates after physical failure phase=%s",
                failure_audit["failure_phase"],
            )
            self._invalidate_candidate_batch()
        return None

    def _failure_result(self, subtask: Subtask, start: float) -> AgentResult:
        execution_failure_audit = list(self._execution_failure_audits)
        return AgentResult(
            subtask_id=subtask.subtask_id,
            status=AgentStatus.FAILURE,
            error_code="ANYGRASP_FAILED",
            result={
                "skill_id": self.skill_id,
                "skill_source": "anygrasp_curobo",
                "grasp_plan_completed": True,
                "grasp_success": False,
                "grasp_error": self._last_execution_error or "AnyGrasp failed",
                "anygrasp_attempts": self._anygrasp_attempts,
                "candidate_detection_audit": list(self._candidate_detection_audits),
                "execution_failure_audit": execution_failure_audit,
            },
            runtime_artifacts={
                "candidate_detection_audit": list(self._candidate_detection_audits),
                "execution_failure_audit": execution_failure_audit,
            },
            latency_ms=self._latency_ms(start),
        )

    def execute(
        self,
        subtask: Subtask,
        context: ExecutionContext,
        selection: LocalSkillSelection,
    ) -> AgentResult:
        start = time.time()
        observation = subtask.parameters.get("observation")
        if not isinstance(observation, dict):
            return AgentResult(
                subtask_id=subtask.subtask_id,
                status=AgentStatus.FAILURE,
                error_code="MANIP_OBSERVATION_MISSING",
                result={"message": "subtask.parameters['observation'] is required"},
                latency_ms=self._latency_ms(start),
            )
        if self._active_subtask_id != subtask.runtime_id:
            self._reset_subtask_state(subtask.runtime_id)

        canonical_action = normalize_action_name(subtask.action)
        if canonical_action in {"place", "place_inside", "put_inside", "drop", "release"}:
            if self._active_execution is not None:
                result = self._advance_place_execution(subtask, selection, start)
                if result is not None:
                    return result
                # A placement may already have opened the gripper before an
                # exception is raised while assembling terminal evidence.
                # Starting a fresh PLACE_INSIDE in that state destroys the
                # original failure context and reports the misleading error
                # "no object is attached".  Placement is deliberately
                # single-shot per subtask: surface the first execution error
                # and let the orchestrator decide whether a new recovery
                # subtask is safe.
                return AgentResult(
                    subtask_id=subtask.subtask_id,
                    status=AgentStatus.FAILURE,
                    error_code="PLACE_EXECUTION_FAILED",
                    result={
                        "skill_id": self.skill_id,
                        "skill_source": "anygrasp_place_inside",
                        "placement_success": False,
                        "placement_error": self._last_execution_error
                        or "placement execution ended without terminal evidence",
                    },
                    latency_ms=self._latency_ms(start),
                )
            executor = self._get_executor()
            if executor is not None and self._start_place_execution(subtask, executor):
                result = self._advance_place_execution(subtask, selection, start)
                if result is not None:
                    return result
            return AgentResult(
                subtask_id=subtask.subtask_id,
                status=AgentStatus.FAILURE,
                error_code="PLACE_EXECUTION_FAILED",
                result={
                    "skill_id": self.skill_id,
                    "placement_success": False,
                    "placement_error": self._last_execution_error or "placement unavailable",
                },
                latency_ms=self._latency_ms(start),
            )

        if self._active_execution is not None:
            result = self._advance_active_execution(subtask, selection, start)
            if result is not None:
                return result

        executor = None if self._candidate_detection_only else self._get_executor()
        if (
            executor is not None
            and not self._candidate_detection_only
            and not self._pre_detection_release_completed
            and not self._pre_detection_release_failed
        ):
            result = self._advance_pre_detection_release(
                subtask,
                selection,
                start,
                executor,
            )
            if result is not None:
                return result

        detection_ready = bool(
            self._candidate_detection_only or self._pre_detection_release_completed
        )
        if (
            detection_ready
            and (self._candidate_detection_only or executor is not None)
            and self._start_anygrasp_execution(subtask, executor)
        ):
            result = self._advance_active_execution(subtask, selection, start)
            if result is not None:
                return result

        if not self._allow_fallback:
            return self._failure_result(subtask, start)

        if executor is not None and self._start_builtin_execution(subtask, executor):
            result = self._advance_active_execution(subtask, selection, start)
            if result is not None:
                return result

        logger.warning(
            "AnyGrasp and OG grasp unavailable; falling back to VLA for %s: %s",
            subtask.subtask_id,
            self._last_execution_error,
        )
        return super().execute(subtask, context, selection)
