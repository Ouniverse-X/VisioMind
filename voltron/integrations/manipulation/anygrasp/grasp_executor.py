"""Non-blocking CuRobo execution for AnyGrasp candidates.

The executor never steps OmniGibson itself.  It exposes one native robot
action at a time so Voltron's normal RuntimeEnvironment.step path retains
observations, rewards, termination, recording, and telemetry.
"""

from __future__ import annotations

import gc
import inspect
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Generator

import numpy as np

from .frame_adapter import AnyGraspFrameAdapter

logger = logging.getLogger(__name__)
_PREGRASP_OFFSET_M = 0.08
_WHOLE_BODY_STANDOFF_M = 0.35
_LIFT_HEIGHT_M = 0.15
_TARGET_DISPLACEMENT_TOLERANCE_M = 0.005


def _as_numpy_vector(value: Any, expected_size: int = 3) -> np.ndarray | None:
    """Convert torch/NumPy-like vectors without importing torch."""
    try:
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        if vector.size == int(expected_size) and np.isfinite(vector).all():
            return vector
    except Exception:
        pass
    return None


def _object_world_aabb(obj: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Read an OmniGibson object's world-space AABB when exposed."""
    aabb = getattr(obj, "aabb", None)
    if not isinstance(aabb, (tuple, list)) or len(aabb) != 2:
        return None
    lower = _as_numpy_vector(aabb[0])
    upper = _as_numpy_vector(aabb[1])
    if lower is None or upper is None or np.any(upper < lower):
        return None
    return lower, upper


def _aabb_contains(
    inner: tuple[np.ndarray, np.ndarray],
    outer: tuple[np.ndarray, np.ndarray],
    margin_m: float = 0.0,
) -> bool:
    """Return whether ``inner`` is fully contained by ``outer``."""
    inner_min, inner_max = inner
    outer_min, outer_max = outer
    margin = float(max(0.0, margin_m))
    return bool(
        np.all(inner_min >= outer_min + margin)
        and np.all(inner_max <= outer_max - margin)
    )


def _grid_cell_aabb(
    container_aabb: tuple[np.ndarray, np.ndarray],
    *,
    grid_shape: tuple[int, int] | list[int],
    cell_index: int,
    margin_m: float = 0.0,
) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, Any]]:
    """Split a container AABB into deterministic, one-based workcell slots.

    Columns follow the longer horizontal world-AABB axis so a ``1 x 3``
    parts bin remains meaningful even when the source asset is rotated by
    ninety degrees.  Rows use the remaining axis.  The returned audit makes
    this convention explicit for competition evidence and downstream video
    overlays.
    """

    shape = tuple(int(value) for value in grid_shape)
    if len(shape) != 2 or any(value < 1 for value in shape):
        raise ValueError("grid_shape must contain two positive integers")
    rows, columns = shape
    total_cells = rows * columns
    index = int(cell_index)
    if not 1 <= index <= total_cells:
        raise ValueError(f"cell_index must be in [1, {total_cells}], got {cell_index}")
    margin = float(margin_m)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("cell margin must be finite and non-negative")

    container_min = np.asarray(container_aabb[0], dtype=np.float64).copy()
    container_max = np.asarray(container_aabb[1], dtype=np.float64).copy()
    horizontal_size = container_max[:2] - container_min[:2]
    column_axis = int(np.argmax(horizontal_size))
    row_axis = 1 - column_axis
    row_index = (index - 1) // columns
    column_index = (index - 1) % columns

    cell_min = container_min.copy()
    cell_max = container_max.copy()
    column_width = horizontal_size[column_axis] / columns
    row_width = horizontal_size[row_axis] / rows
    cell_min[column_axis] += column_index * column_width
    cell_max[column_axis] = cell_min[column_axis] + column_width
    cell_min[row_axis] += row_index * row_width
    cell_max[row_axis] = cell_min[row_axis] + row_width
    cell_min[:2] += margin
    cell_max[:2] -= margin
    if np.any(cell_max[:2] <= cell_min[:2]):
        raise ValueError("cell margin consumes the complete horizontal cell")

    audit = {
        "cell_index": index,
        "grid_shape": [rows, columns],
        "indexing": "one_based_row_major",
        "column_axis_world": "x" if column_axis == 0 else "y",
        "row_axis_world": "x" if row_axis == 0 else "y",
        "cell_margin_m": margin,
        "target_cell_aabb_world": [cell_min.tolist(), cell_max.tolist()],
    }

    try:
        from voltron.shared.compartment_geometry import MultiCompartmentBinGeometry

        fine_geom = MultiCompartmentBinGeometry(
            container_aabb=container_aabb,
            grid_shape=shape,
            cell_margin_m=margin,
        )
        slot = fine_geom.get_slot(index)
        audit["multi_compartment_geometry"] = fine_geom.export_audit()
        audit["slot_inner_aabb_world"] = slot.inner_aabb_world
        audit["bounding_divider_ids"] = slot.bounding_divider_ids
        audit["preplace_entry_pose_world"] = slot.preplace_entry_pose_world
        audit["divider_collision_aabbs"] = [
            [d.aabb_world[0], d.aabb_world[1]] for d in fine_geom.get_all_dividers()
        ]
    except Exception as exc:
        audit["fine_geometry_error"] = str(exc)

    return (cell_min, cell_max), audit


def _xy_containment_correction(
    inner: tuple[np.ndarray, np.ndarray],
    outer: tuple[np.ndarray, np.ndarray],
    *,
    margin_m: float,
) -> tuple[np.ndarray, bool]:
    """Return the smallest XY translation that places ``inner`` in ``outer``.

    The correction is computed independently on each axis.  It is only valid
    when the inner AABB can physically fit between the requested margins; the
    boolean result makes that precondition explicit to callers.
    """
    inner_min, inner_max = inner
    outer_min, outer_max = outer
    margin = float(max(0.0, margin_m))
    lower = np.asarray(outer_min[:2], dtype=np.float64) + margin
    upper = np.asarray(outer_max[:2], dtype=np.float64) - margin
    inner_size = np.asarray(inner_max[:2], dtype=np.float64) - np.asarray(
        inner_min[:2], dtype=np.float64
    )
    if np.any(inner_size > upper - lower + 1e-9):
        return np.zeros(2, dtype=np.float64), False

    correction = np.zeros(2, dtype=np.float64)
    for axis in range(2):
        if inner_min[axis] < lower[axis]:
            correction[axis] = lower[axis] - inner_min[axis]
        elif inner_max[axis] > upper[axis]:
            correction[axis] = upper[axis] - inner_max[axis]
    return correction, True


def _pose_to_matrix(pos: Any, quat: Any) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float64).reshape(-1)[:3]
    x, y, z, w = np.asarray(quat, dtype=np.float64).reshape(-1)[:4]
    norm = np.linalg.norm([x, y, z, w])
    if norm <= 1e-12:
        raise ValueError("quaternion has zero norm")
    x, y, z, w = np.array([x, y, z, w]) / norm
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    matrix[:3, 3] = pos
    return matrix


def _mat_to_quat_xyzw(rotation: np.ndarray) -> np.ndarray:
    R = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(R))
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        quat = np.array(
            [
                (R[2, 1] - R[1, 2]) * s,
                (R[0, 2] - R[2, 0]) * s,
                (R[1, 0] - R[0, 1]) * s,
                0.25 / s,
            ]
        )
    else:
        index = int(np.argmax(np.diag(R)))
        if index == 0:
            s = 2.0 * math.sqrt(max(1e-12, 1.0 + R[0, 0] - R[1, 1] - R[2, 2]))
            quat = np.array(
                [
                    0.25 * s,
                    (R[0, 1] + R[1, 0]) / s,
                    (R[0, 2] + R[2, 0]) / s,
                    (R[2, 1] - R[1, 2]) / s,
                ]
            )
        elif index == 1:
            s = 2.0 * math.sqrt(max(1e-12, 1.0 + R[1, 1] - R[0, 0] - R[2, 2]))
            quat = np.array(
                [
                    (R[0, 1] + R[1, 0]) / s,
                    0.25 * s,
                    (R[1, 2] + R[2, 1]) / s,
                    (R[0, 2] - R[2, 0]) / s,
                ]
            )
        else:
            s = 2.0 * math.sqrt(max(1e-12, 1.0 + R[2, 2] - R[0, 0] - R[1, 1]))
            quat = np.array(
                [
                    (R[0, 2] + R[2, 0]) / s,
                    (R[1, 2] + R[2, 1]) / s,
                    0.25 * s,
                    (R[1, 0] - R[0, 1]) / s,
                ]
            )
    return (quat / np.linalg.norm(quat)).astype(np.float32)


def _quat_multiply_xyzw(left: Any, right: Any) -> np.ndarray:
    """Compose XYZW quaternions with ``left`` applied in world coordinates."""

    lx, ly, lz, lw = _as_numpy_vector(left, expected_size=4)
    rx, ry, rz, rw = _as_numpy_vector(right, expected_size=4)
    result = np.array(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(result))
    if norm <= 1e-12:
        raise ValueError("quaternion composition produced zero norm")
    return (result / norm).astype(np.float32)


def _quat_slerp_xyzw(start: Any, end: Any, fraction: float) -> np.ndarray:
    """Interpolate unit XYZW quaternions along the shortest rotation arc."""

    start_quat = np.asarray(start, dtype=np.float64).reshape(-1)[:4]
    end_quat = np.asarray(end, dtype=np.float64).reshape(-1)[:4]
    start_norm = float(np.linalg.norm(start_quat))
    end_norm = float(np.linalg.norm(end_quat))
    if start_norm <= 1e-12 or end_norm <= 1e-12:
        raise ValueError("cannot interpolate a zero-norm quaternion")
    start_quat /= start_norm
    end_quat /= end_norm
    dot = float(np.dot(start_quat, end_quat))
    if dot < 0.0:
        end_quat = -end_quat
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    amount = float(np.clip(fraction, 0.0, 1.0))
    if dot > 0.9995:
        result = start_quat + amount * (end_quat - start_quat)
    else:
        angle = math.acos(dot)
        sin_angle = math.sin(angle)
        result = (
            math.sin((1.0 - amount) * angle) / sin_angle * start_quat
            + math.sin(amount * angle) / sin_angle * end_quat
        )
    return (result / np.linalg.norm(result)).astype(np.float32)


def _quat_shortest_angle_rad_xyzw(start: Any, end: Any) -> float:
    """Return the unsigned shortest rotation angle between two quaternions."""

    start_quat = np.asarray(start, dtype=np.float64).reshape(-1)[:4]
    end_quat = np.asarray(end, dtype=np.float64).reshape(-1)[:4]
    start_quat /= max(float(np.linalg.norm(start_quat)), 1e-12)
    end_quat /= max(float(np.linalg.norm(end_quat)), 1e-12)
    return 2.0 * math.acos(float(np.clip(abs(np.dot(start_quat, end_quat)), 0.0, 1.0)))


@dataclass(frozen=True)
class GripperGeometryAdapter:
    """Map AnyGrasp's canonical parallel-jaw frame to an OmniGibson EEF frame."""

    fingertip_depth_m: float
    eef_approach_offset_m: float = 0.0
    source: str = "robot_collision_geometry_mean"

    @classmethod
    def from_robot(
        cls,
        robot: Any,
        arm: str,
        *,
        fingertip_depth_override_m: float | None = None,
        eef_approach_offset_m: float = 0.0,
    ) -> "GripperGeometryAdapter":
        if fingertip_depth_override_m is not None:
            override = float(fingertip_depth_override_m)
            if not np.isfinite(override) or override <= 0.0:
                raise ValueError(
                    "fingertip_depth_override_m must be finite and positive"
                )
            approach_offset = float(eef_approach_offset_m)
            if not np.isfinite(approach_offset):
                raise ValueError("eef_approach_offset_m must be finite")
            return cls(
                fingertip_depth_m=override,
                eef_approach_offset_m=approach_offset,
                source="config_override",
            )
        lengths = getattr(robot, "eef_to_fingertip_lengths", {}).get(arm, {})
        values = np.asarray(list(lengths.values()), dtype=np.float64)
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size == 0:
            raise ValueError(
                f"robot arm '{arm}' has no valid EEF-to-fingertip geometry"
            )
        approach_offset = float(eef_approach_offset_m)
        if not np.isfinite(approach_offset):
            raise ValueError("eef_approach_offset_m must be finite")
        return cls(
            fingertip_depth_m=float(np.mean(values)),
            eef_approach_offset_m=approach_offset,
            source="robot_collision_geometry_mean",
        )

    def eef_position(
        self,
        grasp_origin_world: np.ndarray,
        approach_world: np.ndarray,
        canonical_depth_m: float,
    ) -> np.ndarray:
        """Align robot fingertips with the AnyGrasp canonical fingertips.

        AnyGrasp's translation is the canonical gripper origin and its fingers
        extend ``depth`` along the approach axis. OmniGibson's EEF +Z axis
        likewise points from the EEF origin towards its fingertips.
        """
        origin = np.asarray(grasp_origin_world, dtype=np.float32).reshape(3)
        approach = np.asarray(approach_world, dtype=np.float32).reshape(3)
        return origin + approach * self.eef_origin_candidate_x(canonical_depth_m)

    def eef_origin_candidate_x(self, canonical_depth_m: float) -> float:
        """Return the EEF origin on AnyGrasp's canonical approach axis.

        Candidate preflight, runtime contact geometry, and the commanded EEF
        pose must use this exact same calibrated transform. Keeping the formula
        here prevents an approach offset from being applied only during one
        phase of the grasp pipeline.
        """
        depth = float(canonical_depth_m)
        if not np.isfinite(depth) or depth < 0.0:
            raise ValueError(
                f"AnyGrasp depth must be finite and non-negative, got {depth}"
            )
        return float(depth - self.fingertip_depth_m + self.eef_approach_offset_m)


@dataclass
class GraspResult:
    success: bool
    object_in_hand: str | None
    grasp_pos_world: np.ndarray
    grasp_quat_world: np.ndarray
    anygrasp_score: float
    total_sim_steps: int
    error: str | None = None
    failure_phase: str | None = None
    scene_changed: bool = False
    physical_grasp_verified: bool = False
    physical_evidence: dict[str, Any] = field(default_factory=dict)
    placement_verified: bool = False


class GraspExecution:
    """Stateful primitive generator advanced exactly once per Voltron step."""

    def __init__(self, generator: Generator[Any, None, GraspResult]) -> None:
        self._generator = generator
        self.done = False
        self.result: GraspResult | None = None
        self.last_action: dict[str, np.ndarray] = {}

    @staticmethod
    def _native_action(action: Any) -> dict[str, np.ndarray]:
        detach = getattr(action, "detach", None)
        if callable(detach):
            action = detach()
        cpu = getattr(action, "cpu", None)
        if callable(cpu):
            action = cpu()
        numpy = getattr(action, "numpy", None)
        if callable(numpy):
            action = numpy()
        array = np.asarray(action, dtype=np.float32).reshape(-1)
        if array.size == 0:
            raise ValueError("action primitive yielded an empty action")
        return {"robot_r1": array}

    def advance(self) -> tuple[dict[str, np.ndarray] | None, GraspResult | None]:
        if self.done:
            return None, self.result
        while True:
            try:
                action = next(self._generator)
            except StopIteration as stop:
                self.done = True
                self.result = stop.value
                if not isinstance(self.result, GraspResult):
                    raise RuntimeError("grasp generator ended without a GraspResult")
                return None, self.result
            if action is None:
                continue
            self.last_action = self._native_action(action)
            return self.last_action, None


class _OGEnvProxy:
    def __init__(self, scene: Any, robots: Any) -> None:
        self.scene = scene
        self.robots = robots


class GraspExecutor:
    """Build non-blocking StarterSemanticActionPrimitives executions."""

    def __init__(
        self,
        robot: Any,
        arm: str | None = None,
        primitives: Any | None = None,
        curobo_batch_size: int = 1,
        pregrasp_offset_m: float = _PREGRASP_OFFSET_M,
        whole_body_standoff_m: float = _WHOLE_BODY_STANDOFF_M,
        lift_height_m: float = _LIFT_HEIGHT_M,
        post_lift_yaw_deg: float = 0.0,
        post_lift_yaw_cycles: int = 0,
        post_lift_place_back: bool = False,
        place_back_clearance_m: float = 0.015,
        place_back_retreat_m: float = 0.08,
        skip_standoff_if_within_m: float = 0.20,
        constrained_approach: bool = True,
        retry_unconstrained_approach: bool = True,
        approach_segment_max_m: float = 0.0,
        approach_target_displacement_tolerance_m: float = 0.02,
        close_target_displacement_tolerance_m: float = 0.01,
        approach_goal_position_tolerance_m: float = 0.015,
        live_open_jaw_y_correction_max_m: float = 0.0,
        grasping_mode_override: str | None = None,
        collision_workspace_radius_m: float | None = None,
        verification_steps: int = 5,
        verification_min_target_z_rise_m: float = 0.03,
        verification_relative_offset_tolerance_m: float = 0.01,
        verification_relative_orientation_tolerance_deg: float = 10.0,
        verification_require_attachment_valid: bool = True,
        physical_require_bilateral_contact_before_lift: bool = False,
        physical_staged_close_enabled: bool = True,
        physical_close_compression_m: float = 0.004,
        physical_close_stage_count: int = 6,
        physical_close_hold_steps: int = 4,
        physical_close_stage_displacement_tolerance_m: float = 0.008,
        physical_unilateral_contact_displacement_tolerance_m: float = 0.002,
        fingertip_depth_override_m: float | None = None,
        eef_approach_offset_m: float = 0.0,
    ) -> None:
        if int(curobo_batch_size) < 1:
            raise ValueError("curobo_batch_size must be at least 1")
        if int(verification_steps) < 2:
            raise ValueError("verification_steps must be at least 2")
        if int(physical_close_stage_count) < 1:
            raise ValueError("physical_close_stage_count must be at least 1")
        if int(physical_close_hold_steps) < 0:
            raise ValueError("physical_close_hold_steps must be non-negative")
        if int(post_lift_yaw_cycles) < 0:
            raise ValueError("post_lift_yaw_cycles must be non-negative")
        if grasping_mode_override not in {None, "physical", "assisted", "sticky"}:
            raise ValueError(
                "grasping_mode_override must be physical, assisted, sticky, or null"
            )
        if collision_workspace_radius_m is not None and (
            not np.isfinite(collision_workspace_radius_m)
            or float(collision_workspace_radius_m) <= 0.0
        ):
            raise ValueError("collision_workspace_radius_m must be finite and positive")
        distances = {
            "pregrasp_offset_m": pregrasp_offset_m,
            "whole_body_standoff_m": whole_body_standoff_m,
            "lift_height_m": lift_height_m,
            "post_lift_yaw_deg": post_lift_yaw_deg,
            "place_back_clearance_m": place_back_clearance_m,
            "place_back_retreat_m": place_back_retreat_m,
            "skip_standoff_if_within_m": skip_standoff_if_within_m,
            "approach_segment_max_m": approach_segment_max_m,
            "approach_target_displacement_tolerance_m": (
                approach_target_displacement_tolerance_m
            ),
            "close_target_displacement_tolerance_m": (
                close_target_displacement_tolerance_m
            ),
            "approach_goal_position_tolerance_m": approach_goal_position_tolerance_m,
            "live_open_jaw_y_correction_max_m": live_open_jaw_y_correction_max_m,
            "verification_min_target_z_rise_m": verification_min_target_z_rise_m,
            "verification_relative_offset_tolerance_m": (
                verification_relative_offset_tolerance_m
            ),
            "verification_relative_orientation_tolerance_deg": (
                verification_relative_orientation_tolerance_deg
            ),
            "physical_close_compression_m": physical_close_compression_m,
            "physical_close_stage_displacement_tolerance_m": (
                physical_close_stage_displacement_tolerance_m
            ),
            "physical_unilateral_contact_displacement_tolerance_m": (
                physical_unilateral_contact_displacement_tolerance_m
            ),
        }
        for name, value in distances.items():
            if not np.isfinite(value) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
        if float(whole_body_standoff_m) < float(pregrasp_offset_m):
            raise ValueError(
                "whole_body_standoff_m must not be smaller than pregrasp_offset_m"
            )
        self._robot = robot
        self._arm = arm or str(getattr(robot, "default_arm", "right"))
        self._frame_adapter = AnyGraspFrameAdapter()
        self._fingertip_depth_override_m = (
            None
            if fingertip_depth_override_m is None
            else float(fingertip_depth_override_m)
        )
        self._eef_approach_offset_m = float(eef_approach_offset_m)
        if not np.isfinite(self._eef_approach_offset_m):
            raise ValueError("eef_approach_offset_m must be finite")
        if grasping_mode_override is not None:
            current_mode = getattr(robot, "grasping_mode", None)
            if current_mode != grasping_mode_override:
                logger.warning(
                    "AnyGrasp overriding robot grasping mode from %s to %s",
                    current_mode,
                    grasping_mode_override,
                )
                robot._grasping_mode = grasping_mode_override
        self._primitives = primitives
        self._owns_primitives = primitives is None
        self._primitives_init_failed = False
        self._curobo_batch_size = int(curobo_batch_size)
        self._pregrasp_offset_m = float(pregrasp_offset_m)
        self._whole_body_standoff_m = float(whole_body_standoff_m)
        self._lift_height_m = float(lift_height_m)
        self._post_lift_yaw_deg = float(post_lift_yaw_deg)
        self._post_lift_yaw_cycles = int(post_lift_yaw_cycles)
        self._post_lift_place_back = bool(post_lift_place_back)
        self._place_back_clearance_m = float(place_back_clearance_m)
        self._place_back_retreat_m = float(place_back_retreat_m)
        self._skip_standoff_if_within_m = float(skip_standoff_if_within_m)
        self._constrained_approach = bool(constrained_approach)
        self._retry_unconstrained_approach = bool(retry_unconstrained_approach)
        self._approach_segment_max_m = float(approach_segment_max_m)
        self._approach_target_displacement_tolerance_m = float(
            approach_target_displacement_tolerance_m
        )
        self._close_target_displacement_tolerance_m = float(
            close_target_displacement_tolerance_m
        )
        self._approach_goal_position_tolerance_m = float(
            approach_goal_position_tolerance_m
        )
        self._live_open_jaw_y_correction_max_m = float(live_open_jaw_y_correction_max_m)
        if (
            self._live_open_jaw_y_correction_max_m
            > self._approach_target_displacement_tolerance_m
        ):
            raise ValueError(
                "live_open_jaw_y_correction_max_m must not exceed "
                "approach_target_displacement_tolerance_m"
            )
        self._collision_workspace_radius_m = (
            None
            if collision_workspace_radius_m is None
            else float(collision_workspace_radius_m)
        )
        self._verification_steps = int(verification_steps)
        self._verification_min_target_z_rise_m = float(verification_min_target_z_rise_m)
        self._verification_relative_offset_tolerance_m = float(
            verification_relative_offset_tolerance_m
        )
        self._verification_relative_orientation_tolerance_deg = float(
            verification_relative_orientation_tolerance_deg
        )
        self._verification_require_attachment_valid = bool(
            verification_require_attachment_valid
        )
        self._physical_require_bilateral_contact_before_lift = bool(
            physical_require_bilateral_contact_before_lift
        )
        self._physical_staged_close_enabled = bool(physical_staged_close_enabled)
        self._physical_close_compression_m = float(physical_close_compression_m)
        self._physical_close_stage_count = int(physical_close_stage_count)
        self._physical_close_hold_steps = int(physical_close_hold_steps)
        self._physical_close_stage_displacement_tolerance_m = float(
            physical_close_stage_displacement_tolerance_m
        )
        self._physical_unilateral_contact_displacement_tolerance_m = float(
            physical_unilateral_contact_displacement_tolerance_m
        )

    @staticmethod
    def post_lift_yaw_sequence(yaw_deg: float, cycles: int) -> list[float]:
        """Return alternating world-yaw test targets followed by neutral."""
        yaw = float(yaw_deg)
        count = int(cycles)
        if not np.isfinite(yaw) or yaw < 0.0:
            raise ValueError("yaw_deg must be finite and non-negative")
        if count < 0:
            raise ValueError("cycles must be non-negative")
        if yaw == 0.0 or count == 0:
            return []
        return [angle for _ in range(count) for angle in (yaw, -yaw)] + [0.0]

    @staticmethod
    def physical_staged_close_plan(
        *,
        open_qpos: Any,
        lower_qpos: Any,
        open_gap_m: float,
        target_y_bounds_m: Any,
        compression_m: float,
        stage_count: int,
        lower_limit_margin_m: float = 0.001,
    ) -> dict[str, Any]:
        """Map measured jaw geometry to safe, intermediate gripper joint targets.

        R1's two finger positions have a one-to-one total travel relationship with
        the inner-jaw gap.  Planning from the measured open gap avoids treating an
        AnyGrasp width as a command and, importantly, never drives to the mechanical
        lower limit merely because contact sensing is late.
        """
        open_values = np.asarray(open_qpos, dtype=np.float64).reshape(-1)
        lower_values = np.asarray(lower_qpos, dtype=np.float64).reshape(-1)
        bounds = np.asarray(target_y_bounds_m, dtype=np.float64).reshape(-1)
        if not len(open_values) or open_values.shape != lower_values.shape:
            raise ValueError("open_qpos and lower_qpos must be non-empty and aligned")
        if len(bounds) != 2 or not np.isfinite(bounds).all():
            raise ValueError("target_y_bounds_m must contain two finite values")
        if not np.isfinite(open_values).all() or not np.isfinite(lower_values).all():
            raise ValueError("gripper qpos values must be finite")
        if not np.isfinite(open_gap_m) or float(open_gap_m) <= 0.0:
            raise ValueError("open_gap_m must be finite and positive")
        if not np.isfinite(compression_m) or float(compression_m) < 0.0:
            raise ValueError("compression_m must be finite and non-negative")
        if int(stage_count) < 1:
            raise ValueError("stage_count must be at least 1")

        target_span_m = float(bounds[1] - bounds[0])
        if target_span_m <= 0.0:
            raise ValueError("target cross-section span must be positive")
        desired_gap_m = max(0.0, target_span_m - float(compression_m))
        if desired_gap_m >= float(open_gap_m):
            raise ValueError(
                "target close gap must be smaller than the measured open jaw gap"
            )

        travel = open_values - lower_values
        available_total_travel = float(np.sum(np.abs(travel)))
        requested_gap_reduction = float(open_gap_m) - desired_gap_m
        if available_total_travel <= 1e-9:
            raise ValueError("gripper has no available closing travel")
        close_fraction = requested_gap_reduction / available_total_travel

        movable = np.abs(travel) > 1e-9
        safe_fraction = np.ones_like(travel)
        safe_fraction[movable] = np.maximum(
            0.0,
            1.0 - float(lower_limit_margin_m) / np.abs(travel[movable]),
        )
        maximum_safe_fraction = float(np.min(safe_fraction[movable]))
        applied_fraction = float(np.clip(close_fraction, 0.0, maximum_safe_fraction))
        target_qpos = open_values - applied_fraction * travel
        stages = [
            (open_values + (target_qpos - open_values) * (index / int(stage_count)))
            for index in range(1, int(stage_count) + 1)
        ]
        achieved_gap_m = float(open_gap_m) - applied_fraction * available_total_travel
        return {
            "open_qpos": open_values.tolist(),
            "lower_qpos": lower_values.tolist(),
            "target_span_m": target_span_m,
            "compression_m": float(compression_m),
            "desired_gap_m": desired_gap_m,
            "open_gap_m": float(open_gap_m),
            "requested_gap_reduction_m": requested_gap_reduction,
            "close_fraction": close_fraction,
            "applied_close_fraction": applied_fraction,
            "target_qpos": target_qpos.tolist(),
            "stage_qpos": [stage.tolist() for stage in stages],
            "achieved_gap_m": achieved_gap_m,
            "clamped_above_lower_limit": bool(applied_fraction < close_fraction),
            "lower_limit_margin_m": float(lower_limit_margin_m),
        }

    @staticmethod
    def physical_staged_close_should_stop(
        evidence: dict[str, Any],
        *,
        target_displacement_m: float | None,
        displacement_tolerance_m: float,
        stage_index: int,
        unilateral_contact_displacement_tolerance_m: float | None = None,
    ) -> bool:
        """Apply the contact stop and object-motion abort policy for one sample."""
        if (
            target_displacement_m is not None
            and target_displacement_m > displacement_tolerance_m
        ):
            raise RuntimeError(
                "target moved during staged gripper close by "
                f"{target_displacement_m:.4f} m at stage {stage_index}"
            )
        unilateral_tolerance = unilateral_contact_displacement_tolerance_m
        if (
            unilateral_tolerance is not None
            and int(evidence.get("target_finger_contact_count", 0)) == 1
            and target_displacement_m is not None
            and target_displacement_m > float(unilateral_tolerance)
        ):
            raise RuntimeError(
                "single-finger contact pushed target during staged gripper close by "
                f"{target_displacement_m:.4f} m at stage {stage_index}"
            )
        return bool(
            evidence.get("bilateral_finger_contact", False)
            or evidence.get("grasp_state_passed", False)
        )

    @staticmethod
    def target_collision_boundary_points_world(target_obj: Any) -> np.ndarray:
        """Snapshot finite target collision-boundary points in world coordinates.

        This is read-only and deliberately fails closed when simulator collision
        geometry is unavailable. Candidate preflight and execution diagnostics
        can therefore use the same physical geometry source.
        """
        if target_obj is None:
            raise ValueError("target object is required for collision geometry")
        target_links = getattr(target_obj, "links", {})
        links = (
            list(target_links.values())
            if isinstance(target_links, dict)
            else list(target_links or [])
        )
        point_sets: list[np.ndarray] = []
        for link in links:
            boundary = getattr(link, "collision_boundary_points_world", None)
            if boundary is None:
                continue
            if hasattr(boundary, "detach"):
                boundary = boundary.detach().cpu().numpy()
            points = np.asarray(boundary, dtype=np.float64).reshape(-1, 3)
            points = points[np.isfinite(points).all(axis=1)]
            if len(points):
                point_sets.append(points)
        if not point_sets:
            raise ValueError("target collision boundary points are unavailable")
        return np.concatenate(point_sets, axis=0)

    def candidate_inner_grasp_line_evidence(
        self,
        candidate: Any,
        target_local_points: Any,
        *,
        margin_m: float = 0.0,
    ) -> dict[str, Any]:
        """Compare target points with the robot's actual inner finger-line X span.

        ``target_local_points`` must already be in the AnyGrasp candidate frame.
        This method only reads robot geometry; it does not step the simulator or
        alter the gripper state.
        """
        evidence: dict[str, Any] = {
            "available": False,
            "source": "omnigibson_assisted_grasp_points",
            "arm": self._arm,
        }
        try:
            margin = float(margin_m)
            if not np.isfinite(margin) or margin < 0.0:
                raise ValueError(
                    "inner grasp-line margin must be finite and non-negative"
                )
            target_points = np.asarray(target_local_points, dtype=np.float64).reshape(
                -1, 3
            )
            if not len(target_points) or not np.isfinite(target_points).all():
                raise ValueError(
                    "target candidate-frame points must be finite and non-empty"
                )

            geometry = GripperGeometryAdapter.from_robot(
                self._robot,
                self._arm,
                fingertip_depth_override_m=self._fingertip_depth_override_m,
                eef_approach_offset_m=self._eef_approach_offset_m,
            )
            eef_matrix = _pose_to_matrix(*self._robot.get_eef_pose(self._arm))
            world_to_eef = np.linalg.inv(eef_matrix)
            robot_links = getattr(self._robot, "links", {})
            candidate_eef_origin_x = geometry.eef_origin_candidate_x(candidate.depth)
            line_intervals: dict[str, list[float]] = {}
            line_y_positions: dict[str, float] = {}
            line_points: dict[str, list[dict[str, Any]]] = {}

            for label, attribute in (
                ("start", "assisted_grasp_start_points"),
                ("end", "assisted_grasp_end_points"),
            ):
                by_arm = getattr(self._robot, attribute, {})
                grasp_points = (
                    by_arm.get(self._arm, []) if isinstance(by_arm, dict) else []
                )
                candidate_x_values: list[float] = []
                candidate_y_values: list[float] = []
                point_records: list[dict[str, Any]] = []
                for grasp_point in grasp_points or []:
                    link_name = str(getattr(grasp_point, "link_name", ""))
                    link = (
                        robot_links.get(link_name)
                        if isinstance(robot_links, dict)
                        else None
                    )
                    get_pose = getattr(link, "get_position_orientation", None)
                    if not callable(get_pose):
                        continue
                    local = getattr(grasp_point, "position")
                    if hasattr(local, "detach"):
                        local = local.detach().cpu().numpy()
                    local_array = np.asarray(local, dtype=np.float64).reshape(3)
                    world = _pose_to_matrix(*get_pose()) @ np.append(local_array, 1.0)
                    eef_local = (world_to_eef @ world)[:3]
                    candidate_local = np.array(
                        [
                            eef_local[2] + candidate_eef_origin_x,
                            eef_local[1],
                            -eef_local[0],
                        ],
                        dtype=np.float64,
                    )
                    candidate_x_values.append(float(candidate_local[0]))
                    candidate_y_values.append(float(candidate_local[1]))
                    point_records.append(
                        {
                            "link_name": link_name,
                            "candidate_local": candidate_local.tolist(),
                            "candidate_local_x_m": float(candidate_local[0]),
                        }
                    )
                if len(candidate_x_values) < 2:
                    raise ValueError(
                        f"{attribute} did not resolve two inner-line points"
                    )
                if not candidate_y_values:
                    raise ValueError(f"{attribute} did not resolve an inner-surface Y")
                line_intervals[label] = [
                    float(min(candidate_x_values)),
                    float(max(candidate_x_values)),
                ]
                line_y_positions[label] = float(np.median(candidate_y_values))
                line_points[label] = point_records

            common_min = max(interval[0] for interval in line_intervals.values())
            common_max = min(interval[1] for interval in line_intervals.values())
            effective_min = common_min + margin
            effective_max = common_max - margin
            if not common_min < common_max or not effective_min < effective_max:
                raise ValueError(
                    "finger inner grasp-line intervals have no usable intersection"
                )

            width = max(0.0, float(candidate.width))
            height = max(0.0, float(candidate.height))
            cross_section_mask = (np.abs(target_points[:, 1]) < width / 2.0) & (
                np.abs(target_points[:, 2]) < height / 2.0
            )
            cross_section_x = target_points[cross_section_mask, 0]
            if len(cross_section_x):
                target_x_min = float(cross_section_x.min())
                target_x_max = float(cross_section_x.max())
                overlap_m = max(
                    0.0,
                    min(target_x_max, effective_max) - max(target_x_min, effective_min),
                )
            else:
                target_x_min = target_x_max = None
                overlap_m = 0.0
            inner_mask = cross_section_mask & (
                (target_points[:, 0] >= effective_min)
                & (target_points[:, 0] <= effective_max)
            )
            line_center = (effective_min + effective_max) / 2.0
            effective_span = effective_max - effective_min
            open_y_min = min(line_y_positions.values()) + margin
            open_y_max = max(line_y_positions.values()) - margin
            if not open_y_min < open_y_max:
                raise ValueError("finger inner surfaces have no usable open-jaw gap")
            actual_cross_section_mask = (
                (target_points[:, 0] >= effective_min)
                & (target_points[:, 0] <= effective_max)
                & (np.abs(target_points[:, 2]) <= height / 2.0)
            )
            actual_cross_section_y = target_points[actual_cross_section_mask, 1]
            if len(actual_cross_section_y):
                sampled_y_min = float(actual_cross_section_y.min())
                sampled_y_max = float(actual_cross_section_y.max())
                sampled_target_between_open_fingers = bool(
                    sampled_y_min >= open_y_min and sampled_y_max <= open_y_max
                )
                sampled_y_bounds: list[float] | None = [
                    sampled_y_min,
                    sampled_y_max,
                ]
            else:
                sampled_target_between_open_fingers = False
                sampled_y_bounds = None

            # Boundary samples are sparse and can miss an X/Z slab even when the
            # target's continuous convex volume intersects it. Optimize directly
            # over convex-combination weights to find the full intersection's Y
            # extrema without requiring a sampled point inside the slab.
            from scipy.optimize import linprog

            z_min = -height / 2.0
            z_max = height / 2.0
            if not z_min < z_max:
                raise ValueError("candidate jaw height has no usable interval")
            constraint_matrix = np.vstack(
                (
                    target_points[:, 0],
                    -target_points[:, 0],
                    target_points[:, 2],
                    -target_points[:, 2],
                )
            )
            constraint_bounds = np.array(
                [effective_max, -effective_min, z_max, -z_min],
                dtype=np.float64,
            )
            equality_matrix = np.ones((1, len(target_points)), dtype=np.float64)
            equality_bounds = np.ones(1, dtype=np.float64)
            variable_bounds = [(0.0, None)] * len(target_points)
            minimum_y_result = linprog(
                target_points[:, 1],
                A_ub=constraint_matrix,
                b_ub=constraint_bounds,
                A_eq=equality_matrix,
                b_eq=equality_bounds,
                bounds=variable_bounds,
                method="highs",
            )
            solver_status: dict[str, Any] = {
                "minimum_y": {
                    "status": int(minimum_y_result.status),
                    "message": str(minimum_y_result.message),
                }
            }
            if minimum_y_result.status == 2:
                continuous_intersects = False
                continuous_y_bounds = None
                continuous_inner_clearance = None
                target_between_open_fingers = False
                open_jaw_center_straddled = False
            else:
                if minimum_y_result.status != 0 or minimum_y_result.fun is None:
                    raise RuntimeError(
                        "continuous open-jaw minimum-Y LP failed: "
                        f"status={minimum_y_result.status} "
                        f"message={minimum_y_result.message}"
                    )
                maximum_y_result = linprog(
                    -target_points[:, 1],
                    A_ub=constraint_matrix,
                    b_ub=constraint_bounds,
                    A_eq=equality_matrix,
                    b_eq=equality_bounds,
                    bounds=variable_bounds,
                    method="highs",
                )
                solver_status["maximum_y"] = {
                    "status": int(maximum_y_result.status),
                    "message": str(maximum_y_result.message),
                }
                if maximum_y_result.status != 0 or maximum_y_result.fun is None:
                    raise RuntimeError(
                        "continuous open-jaw maximum-Y LP failed: "
                        f"status={maximum_y_result.status} "
                        f"message={maximum_y_result.message}"
                    )
                continuous_intersects = True
                continuous_y_min = float(minimum_y_result.fun)
                continuous_y_max = float(-maximum_y_result.fun)
                continuous_y_bounds = [continuous_y_min, continuous_y_max]
                lower_clearance = continuous_y_min - open_y_min
                upper_clearance = open_y_max - continuous_y_max
                continuous_inner_clearance = float(
                    min(lower_clearance, upper_clearance)
                )
                numerical_tolerance = 1e-8
                target_between_open_fingers = bool(
                    lower_clearance >= -numerical_tolerance
                    and upper_clearance >= -numerical_tolerance
                )
                open_jaw_center = (open_y_min + open_y_max) / 2.0
                open_jaw_center_straddled = bool(
                    continuous_y_min <= open_jaw_center <= continuous_y_max
                )
            evidence.update(
                {
                    "available": True,
                    "eef_fingertip_depth_m": geometry.fingertip_depth_m,
                    "eef_origin_candidate_local_x_m": candidate_eef_origin_x,
                    "per_finger_inner_line_x_intervals_m": line_intervals,
                    "per_finger_inner_surface_y_m": line_y_positions,
                    "inner_line_points": line_points,
                    "common_inner_line_x_interval_m": [common_min, common_max],
                    "effective_inner_line_x_interval_m": [effective_min, effective_max],
                    "effective_inner_line_span_m": effective_span,
                    "open_jaw_inner_surface_y_interval_m": [open_y_min, open_y_max],
                    "open_jaw_gap_m": open_y_max - open_y_min,
                    "open_jaw_target_cross_section_definition": (
                        "convex hull of target collision boundary points intersected "
                        "with actual inner-line X and candidate jaw-height Z intervals"
                    ),
                    "open_jaw_geometry_method": (
                        "convex_combination_linear_program_scipy_highs"
                    ),
                    "open_jaw_continuous_cross_section_intersects": (
                        continuous_intersects
                    ),
                    "open_jaw_continuous_cross_section_y_bounds_m": (
                        continuous_y_bounds
                    ),
                    "open_jaw_continuous_inner_clearance_m": (
                        continuous_inner_clearance
                    ),
                    "open_jaw_continuous_solver_status": solver_status,
                    "open_jaw_continuous_x_interval_m": [
                        effective_min,
                        effective_max,
                    ],
                    "open_jaw_continuous_z_interval_m": [z_min, z_max],
                    "open_jaw_target_cross_section_point_count": int(
                        actual_cross_section_mask.sum()
                    ),
                    "open_jaw_target_cross_section_y_bounds_m": sampled_y_bounds,
                    "open_jaw_sampled_target_between_open_fingers": (
                        sampled_target_between_open_fingers
                    ),
                    "target_between_open_fingers": target_between_open_fingers,
                    "open_jaw_center_straddled": open_jaw_center_straddled,
                    "margin_m": margin,
                    "target_point_count": int(len(target_points)),
                    "target_cross_section_point_count": int(cross_section_mask.sum()),
                    "target_cross_section_x_bounds_m": (
                        None if target_x_min is None else [target_x_min, target_x_max]
                    ),
                    "target_points_in_inner_line_count": int(inner_mask.sum()),
                    "target_points_in_inner_line_fraction": float(
                        inner_mask.sum() / len(target_points)
                    ),
                    "target_inner_line_overlap_m": overlap_m,
                    "target_inner_line_overlap_fraction": float(
                        overlap_m / effective_span
                    ),
                    "inner_line_center_x_m": line_center,
                    "inner_line_center_straddled": bool(
                        target_x_min is not None
                        and target_x_min <= line_center <= target_x_max
                    ),
                }
            )
        except Exception as exc:
            evidence["unavailable_reason"] = f"{type(exc).__name__}: {exc}"
        return evidence

    def candidate_non_target_collision_evidence(
        self,
        candidate: Any,
        non_target_local_points: Any,
        *,
        margin_m: float = 0.0,
    ) -> dict[str, Any]:
        """Audit scene points against actual finger and palm collision AABBs.

        ``non_target_local_points`` must be in the AnyGrasp candidate frame.
        This is a conservative read-only diagnostic, not a collision predicate:
        visible RGB-D points and component AABBs both approximate true geometry.
        """
        evidence: dict[str, Any] = {
            "available": False,
            "source": "rgbd_non_target_points_vs_robot_collision_aabbs",
            "arm": self._arm,
            "diagnostic_only": True,
            "hard_gate": False,
        }
        try:
            margin = float(margin_m)
            if not np.isfinite(margin) or margin < 0.0:
                raise ValueError(
                    "non-target collision margin must be finite and non-negative"
                )
            scene_points = np.asarray(
                non_target_local_points, dtype=np.float64
            ).reshape(-1, 3)
            if not len(scene_points) or not np.isfinite(scene_points).all():
                raise ValueError(
                    "non-target candidate-frame points must be finite and non-empty"
                )

            geometry = GripperGeometryAdapter.from_robot(
                self._robot,
                self._arm,
                fingertip_depth_override_m=self._fingertip_depth_override_m,
                eef_approach_offset_m=self._eef_approach_offset_m,
            )
            eef_matrix = _pose_to_matrix(*self._robot.get_eef_pose(self._arm))
            world_to_eef = np.linalg.inv(eef_matrix)
            candidate_eef_origin_x = geometry.eef_origin_candidate_x(candidate.depth)

            finger_links = list(
                getattr(self._robot, "finger_links", {}).get(self._arm, [])
            )
            if len(finger_links) != 2:
                raise ValueError(
                    f"expected two finger links, resolved {len(finger_links)}"
                )
            collision_links: list[tuple[str, Any]] = [
                ("finger", link) for link in finger_links
            ]
            eef_link_names = getattr(self._robot, "eef_link_names", {})
            eef_link_name = (
                eef_link_names.get(self._arm)
                if isinstance(eef_link_names, dict)
                else None
            )
            robot_links = getattr(self._robot, "links", {})
            eef_link = (
                robot_links.get(eef_link_name)
                if isinstance(robot_links, dict) and eef_link_name is not None
                else None
            )
            if eef_link is not None and all(
                eef_link is not link for link in finger_links
            ):
                collision_links.append(("palm_or_eef", eef_link))

            def to_numpy(value: Any) -> np.ndarray:
                if hasattr(value, "detach"):
                    value = value.detach().cpu().numpy()
                return np.asarray(value, dtype=np.float64)

            def eef_to_candidate(points: np.ndarray) -> np.ndarray:
                return np.column_stack(
                    (
                        points[:, 2] + candidate_eef_origin_x,
                        points[:, 1],
                        -points[:, 0],
                    )
                )

            component_records: list[dict[str, Any]] = []
            union_inside = np.zeros(len(scene_points), dtype=bool)
            for role, link in collision_links:
                boundary = getattr(link, "collision_boundary_points_world", None)
                if boundary is None:
                    component_records.append(
                        {
                            "role": role,
                            "name": str(getattr(link, "name", "")),
                            "available": False,
                            "unavailable_reason": "collision boundary points unavailable",
                        }
                    )
                    continue
                world_points = to_numpy(boundary).reshape(-1, 3)
                if not len(world_points):
                    continue
                eef_points = (
                    np.column_stack((world_points, np.ones(len(world_points))))
                    @ world_to_eef.T
                )[:, :3]
                candidate_points = eef_to_candidate(eef_points)
                bounds_min = candidate_points.min(axis=0) - margin
                bounds_max = candidate_points.max(axis=0) + margin
                inside = np.all(
                    (scene_points >= bounds_min) & (scene_points <= bounds_max),
                    axis=1,
                )
                union_inside |= inside
                separation = np.maximum(
                    np.maximum(bounds_min - scene_points, scene_points - bounds_max),
                    0.0,
                )
                component_records.append(
                    {
                        "role": role,
                        "name": str(getattr(link, "name", "")),
                        "prim_path": str(getattr(link, "prim_path", "")),
                        "available": True,
                        "candidate_local_aabb": [
                            bounds_min.tolist(),
                            bounds_max.tolist(),
                        ],
                        "non_target_points_in_aabb": int(inside.sum()),
                        "non_target_fraction_in_aabb": float(inside.mean()),
                        "minimum_non_target_distance_to_aabb_m": float(
                            np.linalg.norm(separation, axis=1).min()
                        ),
                    }
                )

            available_components = [
                record for record in component_records if record.get("available")
            ]
            if len(available_components) < 2:
                raise ValueError(
                    "fewer than two robot collision components have boundary points"
                )
            evidence.update(
                {
                    "available": True,
                    "eef_fingertip_depth_m": geometry.fingertip_depth_m,
                    "eef_origin_candidate_local_x_m": candidate_eef_origin_x,
                    "margin_m": margin,
                    "non_target_point_count": int(len(scene_points)),
                    "component_count": len(available_components),
                    "components": component_records,
                    "non_target_points_in_any_component_aabb": int(union_inside.sum()),
                    "non_target_fraction_in_any_component_aabb": float(
                        union_inside.mean()
                    ),
                    "interpretation": (
                        "AABB occupancy is conservative diagnostic evidence only; "
                        "it is not proof of mesh collision"
                    ),
                }
            )
        except Exception as exc:
            evidence["unavailable_reason"] = f"{type(exc).__name__}: {exc}"
        return evidence

    def _local_collision_ignore_objects(self, target_obj: Any | None) -> list[Any]:
        """Exclude scene objects outside the robot-target CuRobo workspace."""
        radius = self._collision_workspace_radius_m
        if radius is None:
            return []

        def as_array(value: Any) -> np.ndarray:
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=np.float64)

        anchors = [
            as_array(self._robot.get_position_orientation()[0]).reshape(3),
        ]
        if target_obj is not None:
            anchors.append(
                as_array(target_obj.get_position_orientation()[0]).reshape(3)
            )

        ignored: list[Any] = []
        ignored_meshes = 0
        for obj in self._robot.scene.objects:
            if (
                obj is self._robot
                or obj is target_obj
                or getattr(obj, "visual_only", False)
            ):
                continue
            distance: float | None = None
            try:
                bounds = as_array(obj.aabb).reshape(2, 3)
                low = np.minimum(bounds[0], bounds[1])
                high = np.maximum(bounds[0], bounds[1])
                distance = min(
                    float(np.linalg.norm(anchor - np.clip(anchor, low, high)))
                    for anchor in anchors
                )
            except Exception:
                try:
                    position = as_array(obj.get_position_orientation()[0]).reshape(3)
                    distance = min(
                        float(np.linalg.norm(anchor - position)) for anchor in anchors
                    )
                except Exception:
                    pass
            if distance is not None and distance > radius:
                ignored.append(obj)
                ignored_meshes += sum(
                    len(link.collision_meshes) for link in obj.links.values()
                )

        logger.warning(
            "AnyGrasp local CuRobo workspace radius=%.2f m ignored_objects=%d ignored_meshes=%d",
            radius,
            len(ignored),
            ignored_meshes,
        )
        return ignored

    def _install_controller_compatibility(self, primitives: Any) -> None:
        """Adapt Starter primitives to R1's one-dimensional smooth grippers."""
        import torch as th

        robot = self._robot

        def q_to_action(q: Any) -> Any:
            q_tensor = th.as_tensor(q)
            control_dict = robot.get_control_dict()
            current_q = th.as_tensor(
                robot.get_joint_positions(),
                dtype=q_tensor.dtype,
                device=q_tensor.device,
            )
            action = th.zeros(
                robot.action_dim, dtype=q_tensor.dtype, device=q_tensor.device
            )
            inactive_arm = "left" if self._arm == "right" else "right"
            default_q = getattr(robot, "default_joint_positions", None)
            if default_q is None:
                default_q = getattr(robot, "_default_joint_positions", None)
            default_q_tensor = (
                th.as_tensor(default_q, dtype=q_tensor.dtype, device=q_tensor.device)
                if default_q is not None
                else None
            )

            for name, controller in robot.controllers.items():
                controller_type = type(controller).__name__
                is_absolute_joint = controller_type == "JointController" and not bool(
                    getattr(controller, "use_delta_commands", False)
                )
                is_inactive_arm = (
                    name in {f"arm_{inactive_arm}", f"{inactive_arm}_arm", f"gripper_{inactive_arm}", f"{inactive_arm}_gripper"}
                    or f"arm_{inactive_arm}" in name.lower()
                    or f"{inactive_arm}_arm" in name.lower()
                )
                if is_inactive_arm and default_q_tensor is not None and is_absolute_joint:
                    # Firmly lock inactive arm at its resting tucked home pose to prevent flailing
                    command = default_q_tensor[controller.dof_idx]
                    partial_action = controller._reverse_preprocess_command(command)
                elif is_absolute_joint:
                    command = q_tensor[controller.dof_idx]
                    partial_action = controller._reverse_preprocess_command(command)
                elif controller_type == "HolonomicBaseJointController":
                    target = q_tensor[controller.dof_idx]
                    current = current_q[controller.dof_idx]
                    delta_yaw = th.atan2(
                        th.sin(target[2] - current[2]),
                        th.cos(target[2] - current[2]),
                    )
                    cos_yaw = th.cos(current[2])
                    sin_yaw = th.sin(current[2])
                    delta_x = target[0] - current[0]
                    delta_y = target[1] - current[1]
                    local_error = th.stack(
                        (
                            cos_yaw * delta_x + sin_yaw * delta_y,
                            -sin_yaw * delta_x + cos_yaw * delta_y,
                        )
                    )
                    if getattr(controller, "motor_type", "position") == "position":
                        command = th.cat((local_error, delta_yaw.reshape(1)))
                    else:
                        frequency = float(getattr(controller, "control_freq", 30.0))
                        linear_velocity = local_error * frequency
                        speed = th.linalg.vector_norm(linear_velocity)
                        if float(speed) > 0.3:
                            linear_velocity = linear_velocity * (0.3 / speed)
                        angular_velocity = th.clamp(delta_yaw * frequency, -0.2, 0.2)
                        command = th.cat((linear_velocity, angular_velocity.reshape(1)))
                    partial_action = controller._reverse_preprocess_command(command)
                else:
                    partial_action = controller.compute_no_op_action(control_dict)
                action[robot.controller_action_idx[name]] = th.as_tensor(
                    partial_action,
                    dtype=action.dtype,
                    device=action.device,
                )
            return action

        def move_fingers_to_limit(limit_type: str):
            target_builder = getattr(
                primitives, "_get_joint_position_with_fingers_at_limit", None
            )
            gripper_control_idx = getattr(robot, "gripper_control_idx", {})
            if not callable(target_builder) or self._arm not in gripper_control_idx:
                # Lightweight test doubles do not expose physical joint limits.
                command = -1.0 if limit_type == "lower" else 1.0
                action_idx = robot.controller_action_idx[f"gripper_{self._arm}"]
                close_steps_after_attachment = 0
                for _ in range(35):
                    action = primitives._empty_action(follow_arm_targets=False)
                    action[action_idx] = command
                    yield primitives._postprocess_action(action)
                    if (
                        limit_type == "lower"
                        and primitives._get_obj_in_hand() is not None
                    ):
                        close_steps_after_attachment += 1
                        if close_steps_after_attachment >= 15:
                            return
                return

            target_joint_positions = th.as_tensor(target_builder(limit_type))
            joint_idx = gripper_control_idx[self._arm]
            target_qpos = target_joint_positions[joint_idx]
            action_idx = robot.controller_action_idx[f"gripper_{self._arm}"]
            close_steps_after_attachment = 0
            for _ in range(250):
                current_joint_positions = th.as_tensor(
                    robot.get_joint_positions(),
                    dtype=target_qpos.dtype,
                    device=target_qpos.device,
                )
                current_qpos = current_joint_positions[joint_idx]
                if th.allclose(current_qpos, target_qpos, atol=0.005):
                    return
                if limit_type == "lower" and primitives._get_obj_in_hand() is not None:
                    close_steps_after_attachment += 1
                    # Ensure fingers physically continue closing to clamp tightly across both sides
                    if close_steps_after_attachment >= 25:
                        return
                command = (
                    1.0 if float(th.mean(target_qpos - current_qpos)) >= 0.0 else -1.0
                )
                action = primitives._empty_action(follow_arm_targets=False)
                action[action_idx] = command
                yield primitives._postprocess_action(action)

            current_qpos = th.as_tensor(robot.get_joint_positions())[joint_idx]
            raise RuntimeError(
                f"gripper failed to reach {limit_type} limit: "
                f"current={current_qpos.tolist()} target={target_qpos.tolist()}"
            )

        def settle_robot():
            for _ in range(20):
                action = primitives._empty_action(follow_arm_targets=False)
                yield primitives._postprocess_action(action)

        robot.q_to_action = q_to_action
        primitives._move_fingers_to_limit = move_fingers_to_limit
        primitives._settle_robot = settle_robot

    def _ensure_primitives(self) -> Any | None:
        if self._primitives is not None:
            return self._primitives
        if self._primitives_init_failed:
            return None
        try:
            import omnigibson as og
            from omnigibson.action_primitives import curobo as og_curobo
            from omnigibson.action_primitives.starter_semantic_action_primitives import (
                StarterSemanticActionPrimitives,
            )

            scene = og.sim.scenes[0]
            scene_mesh_count = int(og.sim.floor_plane is not None)
            for obj in scene.objects:
                if obj is self._robot or getattr(obj, "visual_only", False):
                    continue
                scene_mesh_count += sum(
                    len(link.collision_meshes) for link in obj.links.values()
                )

            # OmniGibson fixes CuRobo's initial mesh cache at 2048. This full-house
            # scene has more meshes, and CuRobo's first-plan dynamic cache growth can
            # corrupt the CUDA context. Preallocate for every collision mesh instead
            # of dropping remote objects from the collision world.
            original_create_world_mesh_collision = og_curobo.create_world_mesh_collision
            mesh_cache_capacity = max(2048, scene_mesh_count)

            def create_world_mesh_collision_with_scene_capacity(
                tensor_args: Any,
                obb_cache_size: int = 10,
                mesh_cache_size: int = 2048,
                max_distance: float = 0.05,
            ) -> Any:
                return original_create_world_mesh_collision(
                    tensor_args,
                    obb_cache_size=obb_cache_size,
                    mesh_cache_size=max(mesh_cache_size, mesh_cache_capacity),
                    max_distance=max_distance,
                )

            logger.warning(
                "AnyGrasp preallocating CuRobo mesh cache capacity=%d "
                "scene_collision_meshes=%d; no scene objects omitted",
                mesh_cache_capacity,
                scene_mesh_count,
            )
            og_curobo.create_world_mesh_collision = (
                create_world_mesh_collision_with_scene_capacity
            )
            try:
                self._primitives = StarterSemanticActionPrimitives(
                    env=_OGEnvProxy(scene=scene, robots=scene.robots),
                    robot=self._robot,
                    enable_head_tracking=False,
                    curobo_batch_size=self._curobo_batch_size,
                )
            finally:
                og_curobo.create_world_mesh_collision = (
                    original_create_world_mesh_collision
                )
            self._arm = str(getattr(self._primitives, "arm", self._arm))
            self._install_controller_compatibility(self._primitives)
            return self._primitives
        except Exception as exc:
            logger.warning("StarterSemanticActionPrimitives init failed: %s", exc)
            self._primitives_init_failed = True
            return None

    def release_planner_memory(self) -> None:
        """Drop owned CuRobo state so isolated AnyGrasp inference can use the GPU."""
        if not self._owns_primitives or self._primitives is None:
            return
        primitives = self._primitives
        self._primitives = None
        self._primitives_init_failed = False
        motion_generator = getattr(primitives, "_motion_generator", None)
        if motion_generator is not None:
            primitives._motion_generator = None
        del motion_generator
        del primitives
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            logger.debug("Unable to release CuRobo CUDA memory", exc_info=True)

    def camera_to_world(
        self,
        translation: np.ndarray,
        rotation_matrix: np.ndarray,
        *,
        camera_pose_world: np.ndarray | None = None,
        camera_sensor_name: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform an AnyGrasp pose using the capture-time optical pose."""
        if camera_pose_world is None:
            position, orientation = self._get_camera_pose(
                camera_sensor_name or "head_cam"
            )
            world_from_camera = _pose_to_matrix(position, orientation)
        else:
            world_from_camera = np.asarray(camera_pose_world, dtype=np.float64)
        frame_pose = self._frame_adapter.camera_candidate_to_world(
            translation,
            rotation_matrix,
            world_from_camera,
        )
        return (
            frame_pose.canonical_origin_world,
            frame_pose.eef_quaternion_xyzw,
        )

    def _get_camera_pose(self, sensor_name: str) -> tuple[Any, Any]:
        sensors = getattr(self._robot, "sensors", {})
        if sensor_name in sensors:
            return sensors[sensor_name].get_position_orientation()
        matches = [sensor for name, sensor in sensors.items() if sensor_name in name]
        if len(matches) == 1:
            return matches[0].get_position_orientation()
        links = getattr(self._robot, "links", {})
        if sensor_name in links:
            logger.warning(
                "using camera link pose instead of an optical sensor pose: %s",
                sensor_name,
            )
            return links[sensor_name].get_position_orientation()
        raise KeyError(f"camera sensor '{sensor_name}' not found")

    @staticmethod
    def _immediate_result(result: GraspResult) -> Generator[Any, None, GraspResult]:
        if False:
            yield None
        return result

    def _release_generator(
        self,
        primitives: Any,
    ) -> Generator[Any, None, GraspResult]:
        steps = 0
        try:
            for action in primitives._execute_release():
                steps += 1
                yield action
            joint_idx = self._robot.gripper_control_idx[self._arm]
            current_qpos = self._robot.get_joint_positions()[joint_idx]
            target_qpos = primitives._get_joint_position_with_fingers_at_limit("upper")[
                joint_idx
            ]
            evidence = {
                "gripper_qpos": np.asarray(current_qpos, dtype=float).tolist(),
                "target_gripper_qpos": np.asarray(target_qpos, dtype=float).tolist(),
                "reached_open_limit": True,
            }
            logger.warning("AnyGrasp pre-detection release evidence=%s", evidence)
            return GraspResult(
                success=True,
                object_in_hand=None,
                grasp_pos_world=np.zeros(3, dtype=np.float32),
                grasp_quat_world=np.array([0, 0, 0, 1], dtype=np.float32),
                anygrasp_score=0.0,
                total_sim_steps=steps,
                physical_evidence=evidence,
            )
        except Exception as exc:
            return GraspResult(
                success=False,
                object_in_hand=None,
                grasp_pos_world=np.zeros(3, dtype=np.float32),
                grasp_quat_world=np.array([0, 0, 0, 1], dtype=np.float32),
                anygrasp_score=0.0,
                total_sim_steps=steps,
                error=f"pre-detection release failed: {type(exc).__name__}: {exc}",
                failure_phase="pre_detection_release",
                scene_changed=False,
            )

    def begin_release(self) -> GraspExecution:
        """Open the physical gripper before detection through normal environment steps."""
        primitives = self._ensure_primitives()
        if primitives is None:
            result = GraspResult(
                success=False,
                object_in_hand=None,
                grasp_pos_world=np.zeros(3, dtype=np.float32),
                grasp_quat_world=np.array([0, 0, 0, 1], dtype=np.float32),
                anygrasp_score=0.0,
                total_sim_steps=0,
                error="StarterSemanticActionPrimitives init failed before detection",
                failure_phase="pre_detection_release",
            )
            return GraspExecution(self._immediate_result(result))
        return GraspExecution(self._release_generator(primitives))

    def begin_place_inside(
        self,
        target_obj: Any,
        *,
        cell_index: int | None = None,
        grid_shape: tuple[int, int] | list[int] = (1, 3),
        cell_margin_m: float = 0.005,
    ) -> GraspExecution:
        """Place the currently held object inside ``target_obj``."""
        primitives = self._ensure_primitives()
        if primitives is None:
            result = GraspResult(
                success=False,
                object_in_hand=None,
                grasp_pos_world=np.zeros(3, dtype=np.float32),
                grasp_quat_world=np.array([0, 0, 0, 1], dtype=np.float32),
                anygrasp_score=0.0,
                total_sim_steps=0,
                error="StarterSemanticActionPrimitives init failed before placement",
                failure_phase="place_inside",
            )
            return GraspExecution(self._immediate_result(result))
        return GraspExecution(
            self._place_inside_generator(
                primitives,
                target_obj,
                cell_index=cell_index,
                grid_shape=grid_shape,
                cell_margin_m=cell_margin_m,
            )
        )

    def _place_inside_generator(
        self,
        primitives: Any,
        target_obj: Any,
        *,
        cell_index: int | None,
        grid_shape: tuple[int, int] | list[int],
        cell_margin_m: float,
    ) -> Generator[Any, None, GraspResult]:
        import torch as th
        from omnigibson import object_states
        from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection

        try:
            execute_motion_parameters = inspect.signature(
                primitives._execute_motion_plan
            ).parameters
            supports_cartesian_tail_audit = (
                "ignore_failure" in execute_motion_parameters
            )
        except (TypeError, ValueError):
            supports_cartesian_tail_audit = False

        steps = 0
        pre_navigation_steps = 0
        pre_navigation_carry_steps = 0
        placement_steps = 0
        release_steps = 0
        settle_steps = 0
        destination_name = getattr(target_obj, "name", None)
        # Keep enough clearance for the R1 base to turn and settle without
        # contacting the workbench.  In the industrial cell, 0.45 m left the
        # commanded base pose inside the collision-limited region; 0.70 m is
        # reachable while still keeping the placement hand target within the
        # arm workspace.
        navigation_standoff_m = 0.70
        navigation_base_pose_world = None
        navigation_path_world = None
        navigation_candidate_count = 0
        navigation_geodesic_distance_m = None
        navigation_waypoint_index = None
        navigation_waypoint_pose_world = None
        navigation_final_base_pose_world = None
        navigation_terminal_clearance_m = None
        navigation_arm_hold_mode = "primitive_default"
        navigation_mode = "direct_same_side_standoff"
        sampled_object_pose_world = None
        placement_hand_pose_world = None
        preplace_hand_pose_world = None
        placement_pose_sample_count = 0
        preplan_base_to_hand_xy_m = None
        preplan_base_pose_world = None
        preplan_eef_pose_world = None
        placement_orientation_mode = "sampled_world_aligned"
        placement_strategy = "curobo_staged_top_entry"
        placement_waypoints_world: list[dict[str, Any]] = []
        pre_release_drop_evidence: dict[str, Any] | None = None
        drop_alignment_attempts: list[dict[str, Any]] = []
        drop_alignment_steps = 0
        placement_phase = "precondition"
        planning_attempts: list[dict[str, Any]] = []
        carry_planning_attempts: list[dict[str, Any]] = []
        carry_waypoints_world: list[dict[str, Any]] = []
        target_cell_aabb = None
        target_cell_audit: dict[str, Any] | None = None
        cell_pose_audits: list[dict[str, Any]] = []
        try:
            held_before = primitives._get_obj_in_hand()
            if held_before is None:
                raise RuntimeError("no object is attached before PLACE_INSIDE")
            destination_aabb_before = _object_world_aabb(target_obj)
            if destination_aabb_before is None:
                raise RuntimeError(
                    "destination AABB unavailable for deterministic pre-navigation"
                )

            # Sample the semantic placement first so navigation can target the
            # actual hand position rather than merely the container centre.
            # Sampling is kinematic and does not disturb the attached object.
            destination_min, destination_max = destination_aabb_before
            destination_center_xy = (destination_min[:2] + destination_max[:2]) / 2.0
            if cell_index is not None:
                target_cell_aabb, target_cell_audit = _grid_cell_aabb(
                    destination_aabb_before,
                    grid_shape=grid_shape,
                    cell_index=cell_index,
                    margin_m=cell_margin_m,
                )
                destination_center_xy = (
                    target_cell_aabb[0][:2] + target_cell_aabb[1][:2]
                ) / 2.0

            # The grasp finishes with the arm close to its maximum horizontal
            # reach. Holding those joint targets throughout base navigation
            # preserves the grasp, but leaves no IK margin for a vertical
            # clearance move at the destination. Before leaving the pickup
            # workcell, retract and lift diagonally into a compact carry pose.
            # This is both safer for a long tool during navigation and gives
            # the placement planner usable elbow margin at the toolbox.
            eef_link = getattr(self._robot, "eef_links", {}).get(self._arm)
            get_eef_pose = getattr(eef_link, "get_position_orientation", None)
            if callable(get_eef_pose):
                carry_eef_pose = get_eef_pose()
                carry_eef_pos = _as_numpy_vector(carry_eef_pose[0])
                carry_eef_quat = _as_numpy_vector(
                    carry_eef_pose[1], expected_size=4
                )
                carry_base_pos = _as_numpy_vector(
                    self._robot.get_position_orientation()[0]
                )
                carry_object_aabb = _object_world_aabb(held_before)
                if (
                    carry_eef_pos is not None
                    and carry_eef_quat is not None
                    and carry_base_pos is not None
                    and carry_object_aabb is not None
                ):
                    carry_relative_xy = carry_eef_pos[:2] - carry_base_pos[:2]
                    carry_radius_m = float(np.linalg.norm(carry_relative_xy))
                    compact_radius_m = 0.75
                    if carry_radius_m > 0.88:
                        carry_target_pos = carry_eef_pose[0].clone()
                        carry_scale = compact_radius_m / carry_radius_m
                        carry_target_pos[:2] = th.as_tensor(
                            carry_base_pos[:2] + carry_relative_xy * carry_scale,
                            dtype=carry_target_pos.dtype,
                            device=carry_target_pos.device,
                        )
                        hand_to_object_bottom_m = max(
                            0.0,
                            float(carry_eef_pos[2] - carry_object_aabb[0][2]),
                        )
                        carry_target_pos[2] = max(
                            float(carry_target_pos[2]),
                            float(destination_max[2])
                            + hand_to_object_bottom_m
                            + 0.10,
                        )
                        carry_delta = carry_target_pos - carry_eef_pose[0]
                        carry_distance_m = float(th.linalg.vector_norm(carry_delta))
                        # The first compact-carry attempt from the fully
                        # extended grasp can sit on a narrow CuRobo basin.
                        # A 50 mm Cartesian hop repeatedly failed at waypoint
                        # 3 in the real R1 scene; use 25 mm guarded hops so
                        # every interpolation starts from the measured joint
                        # state reached by the previous segment.
                        carry_count = max(
                            1, int(math.ceil(carry_distance_m / 0.025))
                        )
                        carry_local_ignores = self._local_collision_ignore_objects(
                            held_before
                        )
                        for ignored_obj in (held_before, target_obj):
                            if all(
                                ignored_obj is not item
                                for item in carry_local_ignores
                            ):
                                carry_local_ignores.append(ignored_obj)
                        carry_semantic_ignores = [
                            obj
                            for obj in self._robot.scene.objects
                            if obj is not self._robot
                        ]
                        carry_profiles = (
                            (
                                CuRoboEmbodimentSelection.ARM,
                                carry_local_ignores,
                                "local_obstacles",
                            ),
                            (
                                CuRoboEmbodimentSelection.DEFAULT,
                                carry_local_ignores,
                                "local_obstacles",
                            ),
                            (
                                CuRoboEmbodimentSelection.ARM,
                                carry_semantic_ignores,
                                "semantic_carry_recovery",
                            ),
                            (
                                CuRoboEmbodimentSelection.DEFAULT,
                                carry_semantic_ignores,
                                "semantic_carry_recovery",
                            ),
                        )
                        carry_obstacle_profile = None
                        carry_waypoint_quat = carry_eef_pose[1].clone()
                        for index in range(1, carry_count + 1):
                            placement_phase = (
                                f"plan_arm_compact_carry_{index}_of_{carry_count}"
                            )
                            carry_pose_pos = (
                                carry_eef_pose[0]
                                + carry_delta * index / carry_count
                            )
                            carry_segment = (
                                f"arm_compact_carry_{index}_of_{carry_count}"
                            )
                            # Keep the requested orientation for the post-motion
                            # tail audit before replacing the next seed with the
                            # measured physics pose.
                            planned_carry_waypoint_quat = carry_waypoint_quat.clone()
                            carry_waypoint_record = {
                                "segment": carry_segment,
                                "position": _as_numpy_vector(
                                    carry_pose_pos
                                ).tolist(),
                                "orientation_xyzw": _as_numpy_vector(
                                    carry_waypoint_quat, expected_size=4
                                ).tolist(),
                            }
                            carry_waypoints_world.append(carry_waypoint_record)
                            trajectory = None
                            last_carry_error: Exception | None = None
                            for (
                                embodiment,
                                ignore_objects,
                                collision_profile,
                            ) in carry_profiles:
                                planner_get_obj_in_hand = primitives._get_obj_in_hand
                                try:
                                    primitives._get_obj_in_hand = lambda: None
                                    trajectory = primitives._plan_joint_motion(
                                        target_pos={
                                            self._robot.eef_link_names[self._arm]: (
                                                carry_pose_pos
                                            )
                                        },
                                        target_quat={
                                            self._robot.eef_link_names[self._arm]: (
                                                carry_waypoint_quat
                                            )
                                        },
                                        embodiment_selection=embodiment,
                                        ignore_objects=ignore_objects,
                                        skip_obstacle_update=(
                                            carry_obstacle_profile
                                            == collision_profile
                                        ),
                                    )
                                except Exception as exc:
                                    last_carry_error = exc
                                    carry_planning_attempts.append(
                                        {
                                            "segment": carry_segment,
                                            "embodiment": str(
                                                getattr(
                                                    embodiment,
                                                    "value",
                                                    embodiment,
                                                )
                                            ),
                                            "collision_profile": collision_profile,
                                            "success": False,
                                            "error": (
                                                f"{type(exc).__name__}: {exc}"
                                            ),
                                        }
                                    )
                                    # Recovery for narrow CuRobo orientation
                                    # basins: first try the initial stable grasp
                                    # quaternion, then relax orientation with
                                    # motion_constraint.  The resulting physics
                                    # pose remains subject to the strict tail audit below.
                                    try:
                                        trajectory = primitives._plan_joint_motion(
                                            target_pos={
                                                self._robot.eef_link_names[self._arm]: (
                                                    carry_pose_pos
                                                )
                                            },
                                            target_quat={
                                                self._robot.eef_link_names[self._arm]: (
                                                    carry_eef_pose[1]
                                                )
                                            },
                                            embodiment_selection=embodiment,
                                            ignore_objects=ignore_objects,
                                            skip_obstacle_update=(
                                                carry_obstacle_profile
                                                == collision_profile
                                            ),
                                        )
                                    except Exception:
                                        try:
                                            trajectory = primitives._plan_joint_motion(
                                                target_pos={
                                                    self._robot.eef_link_names[self._arm]: (
                                                        carry_pose_pos
                                                    )
                                                },
                                                target_quat={
                                                    self._robot.eef_link_names[self._arm]: (
                                                        carry_waypoint_quat
                                                    )
                                                },
                                                embodiment_selection=embodiment,
                                                motion_constraint=[
                                                    1.0,
                                                    1.0,
                                                    1.0,
                                                    0.0,
                                                    0.0,
                                                    0.0,
                                                ],
                                                ignore_objects=ignore_objects,
                                                skip_obstacle_update=(
                                                    carry_obstacle_profile
                                                    == collision_profile
                                                ),
                                            )
                                        except Exception as recovery_exc:
                                            last_carry_error = recovery_exc
                                            carry_planning_attempts.append(
                                                {
                                                    "segment": carry_segment,
                                                    "embodiment": str(
                                                        getattr(
                                                            embodiment,
                                                            "value",
                                                            embodiment,
                                                        )
                                                    ),
                                                    "collision_profile": (
                                                        f"{collision_profile}:orientation_recovery"
                                                    ),
                                                    "success": False,
                                                    "error": (
                                                        f"{type(recovery_exc).__name__}: "
                                                        f"{recovery_exc}"
                                                    ),
                                                }
                                            )
                                        else:
                                            carry_planning_attempts.append(
                                                {
                                                    "segment": carry_segment,
                                                    "embodiment": str(
                                                        getattr(
                                                            embodiment,
                                                            "value",
                                                            embodiment,
                                                        )
                                                    ),
                                                    "collision_profile": (
                                                        f"{collision_profile}:orientation_recovery"
                                                    ),
                                                    "success": True,
                                                    "trajectory_steps": len(trajectory),
                                                }
                                            )
                                            carry_waypoint_record[
                                                "orientation_constraint_recovery"
                                            ] = "motion_constraint"
                                            break
                                    else:
                                        carry_planning_attempts.append(
                                            {
                                                "segment": carry_segment,
                                                "embodiment": str(
                                                    getattr(
                                                        embodiment,
                                                        "value",
                                                        embodiment,
                                                    )
                                                ),
                                                "collision_profile": (
                                                    f"{collision_profile}:initial_quaternion_recovery"
                                                ),
                                                "success": True,
                                                "trajectory_steps": len(trajectory),
                                            }
                                        )
                                        carry_waypoint_record[
                                            "orientation_constraint_recovery"
                                        ] = "initial_quaternion"
                                        break
                                else:
                                    carry_planning_attempts.append(
                                        {
                                            "segment": carry_segment,
                                            "embodiment": str(
                                                getattr(
                                                    embodiment,
                                                    "value",
                                                    embodiment,
                                                )
                                            ),
                                            "collision_profile": collision_profile,
                                            "success": True,
                                            "trajectory_steps": len(trajectory),
                                        }
                                    )
                                    break
                                finally:
                                    primitives._get_obj_in_hand = (
                                        planner_get_obj_in_hand
                                    )
                                    carry_obstacle_profile = collision_profile
                            if trajectory is None:
                                if last_carry_error is None:
                                    raise RuntimeError(
                                        "no CuRobo embodiment configured for "
                                        f"{carry_segment}"
                                    )
                                raise last_carry_error
                            placement_phase = (
                                f"execute_arm_compact_carry_{index}_of_{carry_count}"
                            )
                            carry_execution_kwargs = (
                                {"ignore_failure": True}
                                if supports_cartesian_tail_audit
                                else {}
                            )
                            for action in primitives._execute_motion_plan(
                                trajectory,
                                **carry_execution_kwargs,
                            ):
                                if action is not None:
                                    steps += 1
                                    pre_navigation_carry_steps += 1
                                    yield action
                            attachment_same = (
                                primitives._get_obj_in_hand() is held_before
                            )
                            if supports_cartesian_tail_audit:
                                actual_carry_pose = get_eef_pose()
                                actual_carry_pos = _as_numpy_vector(
                                    actual_carry_pose[0]
                                )
                                actual_carry_quat = _as_numpy_vector(
                                    actual_carry_pose[1], expected_size=4
                                )
                                target_carry_pos = _as_numpy_vector(carry_pose_pos)
                                position_error_m = (
                                    None
                                    if actual_carry_pos is None
                                    or target_carry_pos is None
                                    else float(
                                        np.linalg.norm(
                                            actual_carry_pos - target_carry_pos
                                        )
                                    )
                                )
                                orientation_error_deg = (
                                    None
                                    if actual_carry_quat is None
                                    else math.degrees(
                                        _quat_shortest_angle_rad_xyzw(
                                            actual_carry_quat,
                                            planned_carry_waypoint_quat,
                                        )
                                    )
                                )
                                # Physics and the joint controller can leave a
                                # small wrist orientation residual after each
                                # short segment. Carry that measured pose into
                                # the next planning seed instead of repeatedly
                                # asking CuRobo to undo the accumulated residual
                                # while also moving the payload.
                                if actual_carry_quat is not None:
                                    carry_waypoint_quat = th.as_tensor(
                                        actual_carry_quat,
                                        dtype=carry_eef_pose[1].dtype,
                                        device=carry_eef_pose[1].device,
                                    )
                                carry_waypoint_record.update(
                                    {
                                        "actual_position": (
                                            None
                                            if actual_carry_pos is None
                                            else actual_carry_pos.tolist()
                                        ),
                                        "actual_orientation_xyzw": (
                                            None
                                            if actual_carry_quat is None
                                            else actual_carry_quat.tolist()
                                        ),
                                        "position_error_m": position_error_m,
                                        "orientation_error_deg": (
                                            orientation_error_deg
                                        ),
                                        "attachment_same": attachment_same,
                                        "cartesian_tail_audit_passed": bool(
                                            position_error_m is not None
                                            and position_error_m <= 0.025
                                            and orientation_error_deg is not None
                                            and orientation_error_deg <= 6.0
                                            and attachment_same
                                        ),
                                    }
                                )
                                if not carry_waypoint_record[
                                    "cartesian_tail_audit_passed"
                                ]:
                                    raise RuntimeError(
                                        "compact carry Cartesian tail audit failed: "
                                        f"{carry_waypoint_record}"
                                    )
                            if not attachment_same:
                                raise RuntimeError(
                                    "held-object attachment changed during carry "
                                    "posture transition"
                                )
                        logger.warning(
                            "PLACE_INSIDE compact carry radius %.3f->%.3f m "
                            "waypoints=%d target=%s",
                            carry_radius_m,
                            compact_radius_m,
                            carry_count,
                            carry_waypoints_world[-1],
                        )

            def build_deterministic_cell_pose(phase: str) -> Any:
                if target_cell_aabb is None:
                    raise RuntimeError("deterministic cell pose requires cell bounds")
                current_position, current_quaternion = (
                    held_before.get_position_orientation()
                )
                current_position_np = _as_numpy_vector(current_position)
                current_quaternion_np = _as_numpy_vector(
                    current_quaternion, expected_size=4
                )
                if current_position_np is None or current_quaternion_np is None:
                    raise RuntimeError(
                        "held-object pose is unavailable for cell alignment"
                    )
                boundary_points = self.target_collision_boundary_points_world(
                    held_before
                )
                geometry_min = np.min(boundary_points, axis=0)
                geometry_max = np.max(boundary_points, axis=0)
                geometry_center = (geometry_min + geometry_max) / 2.0
                centered_xy = boundary_points[:, :2] - np.mean(
                    boundary_points[:, :2], axis=0
                )
                covariance = centered_xy.T @ centered_xy / max(1, len(centered_xy) - 1)
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                major_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
                current_major_angle = math.atan2(
                    float(major_axis[1]), float(major_axis[0])
                )
                cell_span = target_cell_aabb[1] - target_cell_aabb[0]
                target_major_axis = int(np.argmax(cell_span[:2]))
                target_major_angle = 0.0 if target_major_axis == 0 else math.pi / 2.0
                # A principal axis is undirected.  Wrap modulo pi to select
                # the smallest yaw that aligns the long tool with the long
                # dimension of the requested cell.
                yaw_delta = (
                    (target_major_angle - current_major_angle + math.pi / 2.0) % math.pi
                ) - math.pi / 2.0
                cosine = math.cos(yaw_delta)
                sine = math.sin(yaw_delta)
                yaw_rotation = np.array(
                    [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                )
                rotated_centered_points = (
                    yaw_rotation @ (boundary_points - geometry_center).T
                ).T
                predicted_span = np.ptp(rotated_centered_points, axis=0)
                fit_clearance_m = 0.001
                fits_cell_xy = bool(
                    np.all(predicted_span[:2] + 2.0 * fit_clearance_m <= cell_span[:2])
                )
                if not fits_cell_xy:
                    raise RuntimeError(
                        "held object collision geometry cannot fit requested grid cell: "
                        f"object_span_xy={predicted_span[:2].tolist()} "
                        f"cell_span_xy={cell_span[:2].tolist()}"
                    )

                cell_center = (target_cell_aabb[0] + target_cell_aabb[1]) / 2.0
                target_geometry_center = geometry_center.copy()
                target_geometry_center[:2] = cell_center[:2]
                target_geometry_center[2] = (
                    target_cell_aabb[0][2] + predicted_span[2] / 2.0 + 0.015
                )
                origin_to_geometry_center = geometry_center - current_position_np
                rotated_origin_offset = yaw_rotation @ origin_to_geometry_center
                desired_origin = target_geometry_center - rotated_origin_offset
                desired_position = current_position.clone()
                desired_position[:] = th.as_tensor(
                    desired_origin,
                    dtype=desired_position.dtype,
                    device=desired_position.device,
                )
                yaw_quaternion = np.array(
                    [0.0, 0.0, math.sin(yaw_delta / 2.0), math.cos(yaw_delta / 2.0)],
                    dtype=np.float32,
                )
                desired_quaternion_np = _quat_multiply_xyzw(
                    yaw_quaternion,
                    current_quaternion_np,
                )
                desired_quaternion = current_quaternion.clone()
                desired_quaternion[:] = th.as_tensor(
                    desired_quaternion_np,
                    dtype=desired_quaternion.dtype,
                    device=desired_quaternion.device,
                )
                audit = {
                    "phase": phase,
                    "source": "collision_boundary_pca_cell_alignment",
                    "boundary_point_count": int(len(boundary_points)),
                    "principal_eigenvalues_m2": eigenvalues.tolist(),
                    "current_major_axis_world_xy": major_axis.tolist(),
                    "target_major_axis_world": "x" if target_major_axis == 0 else "y",
                    "yaw_delta_deg": math.degrees(yaw_delta),
                    "predicted_object_span_world_m": predicted_span.tolist(),
                    "target_cell_span_world_m": cell_span.tolist(),
                    "fit_clearance_m": fit_clearance_m,
                    "fits_cell_xy": fits_cell_xy,
                    "target_geometry_center_world": target_geometry_center.tolist(),
                    "desired_object_origin_world": desired_origin.tolist(),
                }
                cell_pose_audits.append(audit)
                logger.warning("PLACE_INSIDE deterministic cell pose=%s", audit)
                return desired_position, desired_quaternion

            placement_phase = "sample_placement_pose"
            if target_cell_aabb is not None:
                desired_object_pose = build_deterministic_cell_pose("pre_navigation")
                placement_orientation_mode = "cell_principal_axis_alignment"
            else:
                desired_object_pose = primitives._sample_pose_with_object_and_predicate(
                    object_states.Inside,
                    held_before,
                    target_obj,
                    world_aligned=True,
                )
            placement_pose_sample_count += 1
            desired_hand_pose = primitives._get_hand_pose_for_object_pose(
                desired_object_pose
            )
            desired_object_pos = _as_numpy_vector(desired_object_pose[0])
            desired_object_quat = _as_numpy_vector(
                desired_object_pose[1], expected_size=4
            )
            desired_hand_pos = _as_numpy_vector(desired_hand_pose[0])
            desired_hand_quat = _as_numpy_vector(desired_hand_pose[1], expected_size=4)
            if any(
                value is None
                for value in (
                    desired_object_pos,
                    desired_object_quat,
                    desired_hand_pos,
                    desired_hand_quat,
                )
            ):
                raise RuntimeError("sampled placement pose is unavailable")
            sampled_object_pose_world = {
                "position": desired_object_pos.tolist(),
                "orientation_xyzw": desired_object_quat.tolist(),
            }
            placement_hand_pose_world = {
                "position": desired_hand_pos.tolist(),
                "orientation_xyzw": desired_hand_quat.tolist(),
            }
            logger.warning(
                "PLACE_INSIDE sampled object_pose=%s hand_pose=%s",
                sampled_object_pose_world,
                placement_hand_pose_world,
            )
            preplace_pos = desired_hand_pose[0].clone()
            hand_offset_z = float(
                desired_hand_pose[0][2] - desired_object_pose[0][2]
            )
            clearance_offset_z = (
                hand_offset_z + 0.10 if target_cell_aabb is not None else 0.10
            )
            preplace_pos[2] = max(
                float(preplace_pos[2]),
                float(destination_max[2]) + clearance_offset_z,
            )
            preplace_pose = (preplace_pos, desired_hand_pose[1])
            preplace_hand_pose_world = {
                "position": _as_numpy_vector(preplace_pos).tolist(),
                "orientation_xyzw": desired_hand_quat.tolist(),
            }

            # The stock semantic primitive rebuilds thousands of scene meshes
            # before yielding its first navigation action.  Instead, use the
            # already-loaded traversability map to test close approach poses
            # around the container.  This matters for worktops: the pose on
            # the robot's current side can be inside the cabinet even though
            # its radial standoff is correct.  A* selects a collision-free
            # side and supplies waypoints around the worktop without a global
            # CuRobo obstacle rebuild.
            base_position = _as_numpy_vector(self._robot.get_position_orientation()[0])
            if base_position is None:
                raise RuntimeError("robot base position is unavailable")
            approach_direction = base_position[:2] - destination_center_xy
            approach_norm = float(np.linalg.norm(approach_direction))
            if approach_norm < 1e-6:
                approach_direction = np.array([1.0, 0.0], dtype=np.float64)
            else:
                approach_direction /= approach_norm
            navigation_xy = (
                destination_center_xy + approach_direction * navigation_standoff_m
            )
            scene = getattr(self._robot, "scene", None)
            get_shortest_path = getattr(scene, "get_shortest_path", None)
            best_navigation = None
            if callable(get_shortest_path):
                current_angle = math.atan2(approach_direction[1], approach_direction[0])
                # Search the nearest proven-reachable radii first, while
                # covering both sides and diagonals of the container.
                angle_offsets = (
                    0.0,
                    math.pi,
                    math.pi / 2.0,
                    -math.pi / 2.0,
                    math.pi / 4.0,
                    -math.pi / 4.0,
                    3.0 * math.pi / 4.0,
                    -3.0 * math.pi / 4.0,
                )
                trav_map = getattr(scene, "_trav_map", None)
                floor_maps = getattr(trav_map, "floor_map", None)
                world_to_map = getattr(trav_map, "world_to_map", None)
                map_to_world = getattr(trav_map, "map_to_world", None)
                raw_map_available = False
                raw_clearance_map = None
                trav_map_resolution_m = float(getattr(trav_map, "map_resolution", 0.1))
                if (
                    floor_maps is not None
                    and callable(world_to_map)
                    and callable(map_to_world)
                ):
                    try:
                        import cv2
                        from omnigibson.utils.motion_planning_utils import astar

                        floor_map_tensor = floor_maps[0]
                        detach = getattr(floor_map_tensor, "detach", None)
                        if callable(detach):
                            floor_map_tensor = detach()
                        cpu = getattr(floor_map_tensor, "cpu", None)
                        floor_map_cpu = cpu() if callable(cpu) else floor_map_tensor
                        floor_map = np.asarray(floor_map_cpu)
                        raw_clearance_map = (
                            cv2.distanceTransform(
                                (floor_map == 255).astype(np.uint8),
                                cv2.DIST_L2,
                                5,
                            )
                            * trav_map_resolution_m
                        )
                        # The raw map's shortest path can skim worktop corners.
                        # A 7x7 kernel still admitted an approximately 0.30 m
                        # clearance turn where the R1 stalled while sweeping a
                        # long attached tool through the corner.  Use a 9x9
                        # kernel for approximately 0.40 m carried-object
                        # clearance.  This is intentionally more conservative
                        # than unloaded navigation and keeps the tool pose
                        # outside the cabinet corner during the aisle transfer.
                        planning_map = cv2.erode(
                            floor_map.astype(np.uint8),
                            np.ones((9, 9), dtype=np.uint8),
                        )
                        source_map = np.asarray(
                            world_to_map(base_position[:2]), dtype=np.int64
                        ).reshape(2)
                        _, component_labels = cv2.connectedComponents(
                            (planning_map == 255).astype(np.uint8), connectivity=4
                        )
                        source_label = int(
                            component_labels[int(source_map[0]), int(source_map[1])]
                        )
                        raw_candidate = None
                        raw_candidate_radii = (
                            0.45,
                            0.50,
                            0.55,
                            0.60,
                            0.65,
                            0.70,
                            0.75,
                            0.80,
                        )
                        raw_candidate_angles = tuple(
                            math.radians(degrees) for degrees in range(0, 360, 15)
                        )
                        for radius in raw_candidate_radii:
                            for angle in raw_candidate_angles:
                                candidate_xy = (
                                    destination_center_xy
                                    + radius
                                    * np.array(
                                        [math.cos(angle), math.sin(angle)],
                                        dtype=np.float64,
                                    )
                                )
                                navigation_candidate_count += 1
                                map_point = np.asarray(
                                    world_to_map(candidate_xy), dtype=np.int64
                                ).reshape(2)
                                row, col = int(map_point[0]), int(map_point[1])
                                if (
                                    row < 0
                                    or col < 0
                                    or row >= planning_map.shape[0]
                                    or col >= planning_map.shape[1]
                                    or planning_map[row, col] != 255
                                    or source_label == 0
                                    or int(component_labels[row, col]) != source_label
                                ):
                                    continue
                                score = float(
                                    np.linalg.norm(candidate_xy - base_position[:2])
                                ) + 2.0 * (radius - navigation_standoff_m)
                                if raw_candidate is None or score < raw_candidate[0]:
                                    raw_candidate = (
                                        score,
                                        candidate_xy,
                                        radius,
                                        map_point,
                                    )
                        raw_map_available = True
                        if raw_candidate is not None:
                            _, candidate_xy, radius, target_map = raw_candidate
                            path_map = astar(
                                planning_map,
                                tuple(source_map.tolist()),
                                tuple(target_map.tolist()),
                            )
                            if path_map is not None:
                                path_world = map_to_world(path_map)
                                path_np = np.asarray(path_world, dtype=np.float64)
                                distance_value = float(
                                    np.linalg.norm(
                                        path_np[1:] - path_np[:-1], axis=1
                                    ).sum()
                                )
                                best_navigation = (
                                    raw_candidate[0],
                                    candidate_xy,
                                    radius,
                                    path_np,
                                    distance_value,
                                )
                    except Exception as exc:
                        raw_map_available = False
                        logger.warning(
                            "PLACE_INSIDE raw traversability analysis failed: %s",
                            exc,
                        )

                generic_candidate_radii = (
                    ()
                    if raw_map_available
                    else (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
                )
                for radius in generic_candidate_radii:
                    for angle_offset in angle_offsets:
                        angle = current_angle + angle_offset
                        candidate_xy = destination_center_xy + radius * np.array(
                            [math.cos(angle), math.sin(angle)], dtype=np.float64
                        )
                        navigation_candidate_count += 1
                        path = None
                        geodesic_distance = None
                        # The full R1 erosion can reject the current pose
                        # after grasping because it is already close to the
                        # worktop. Try it first, then fall back to the raw map
                        # only when the destination itself has at least a
                        # 0.20 m square-map clearance from obstacles.
                        for path_robot in (self._robot, None):
                            if path_robot is None:
                                clearance_ok = False
                                if floor_maps is not None and callable(world_to_map):
                                    try:
                                        map_point = np.asarray(
                                            world_to_map(candidate_xy), dtype=np.int64
                                        ).reshape(2)
                                        floor_map = np.asarray(floor_maps[0])
                                        row, col = int(map_point[0]), int(map_point[1])
                                        clearance_cells = 2
                                        patch = floor_map[
                                            row - clearance_cells : row
                                            + clearance_cells
                                            + 1,
                                            col - clearance_cells : col
                                            + clearance_cells
                                            + 1,
                                        ]
                                        clearance_ok = patch.shape == (5, 5) and bool(
                                            np.all(patch == 255)
                                        )
                                    except Exception:
                                        clearance_ok = False
                                if not clearance_ok:
                                    continue
                            try:
                                path, geodesic_distance = get_shortest_path(
                                    0,
                                    base_position[:2],
                                    candidate_xy,
                                    entire_path=True,
                                    robot=path_robot,
                                )
                            except Exception as exc:
                                logger.debug(
                                    "PLACE_INSIDE traversability candidate failed "
                                    "xy=%s robot_erosion=%s error=%s",
                                    np.round(candidate_xy, 4).tolist(),
                                    path_robot is not None,
                                    exc,
                                )
                                path = None
                                geodesic_distance = None
                            if path is not None:
                                break
                        path_np = (
                            None if path is None else np.asarray(path, dtype=np.float64)
                        )
                        if path_np is None or path_np.ndim != 2 or len(path_np) == 0:
                            continue
                        distance_value = float(geodesic_distance)
                        # Prefer short routes, with a small penalty for moving
                        # farther from the arm's known reachable workspace.
                        score = distance_value + 2.0 * (radius - navigation_standoff_m)
                        if best_navigation is None or score < best_navigation[0]:
                            best_navigation = (
                                score,
                                candidate_xy,
                                radius,
                                path_np,
                                distance_value,
                            )
            if best_navigation is not None:
                _, navigation_xy, navigation_standoff_m, path_np, distance_value = (
                    best_navigation
                )
                navigation_mode = "traversability_multi_side_standoff"
                # Keep the complete 0.30 m-clearance A* route to the selected
                # 0.80 m cell, then permit a short radial terminal approach to
                # 0.75 m only when the *raw* map independently confirms at
                # least 0.30 m clearance.  This recovers arm reach without
                # reintroducing the proven-unsafe 0.20 m-clearance cell.
                if (
                    raw_clearance_map is not None
                    and callable(world_to_map)
                    and navigation_standoff_m > 0.70
                ):
                    terminal_direction = navigation_xy - destination_center_xy
                    terminal_norm = float(np.linalg.norm(terminal_direction))
                    if terminal_norm > 1e-6:
                        terminal_xy = destination_center_xy + (
                            terminal_direction / terminal_norm * 0.70
                        )
                        terminal_map = np.asarray(
                            world_to_map(terminal_xy), dtype=np.int64
                        ).reshape(2)
                        row, col = int(terminal_map[0]), int(terminal_map[1])
                        if (
                            0 <= row < raw_clearance_map.shape[0]
                            and 0 <= col < raw_clearance_map.shape[1]
                        ):
                            navigation_terminal_clearance_m = float(
                                raw_clearance_map[row, col]
                            )
                            if navigation_terminal_clearance_m >= 0.30 - 1e-6:
                                distance_value += float(
                                    np.linalg.norm(terminal_xy - navigation_xy)
                                )
                                navigation_xy = terminal_xy
                                navigation_standoff_m = 0.70
                                navigation_mode += "_terminal_approach"
                navigation_geodesic_distance_m = distance_value
                # ``_navigate_to_pose_direct`` is a straight-line velocity
                # controller, rather than a collision-aware planner.  Do not
                # sparsify the A* path: skipping grid corners creates a
                # diagonal shortcut that can cut through a worktop/cabinet
                # even when every individual A* cell is free.  Keep the
                # original grid-resolution waypoints (typically ~0.10 m) so
                # each executed segment follows a collision-free map edge.
                sparse_path: list[np.ndarray] = []
                for point in path_np:
                    point_xy = np.asarray(point[:2], dtype=np.float64)
                    if (
                        not sparse_path
                        and float(np.linalg.norm(point_xy - base_position[:2])) < 0.15
                    ):
                        continue
                    if (
                        not sparse_path
                        or float(np.linalg.norm(point_xy - sparse_path[-1])) >= 1e-4
                    ):
                        sparse_path.append(point_xy)
                if (
                    not sparse_path
                    or float(np.linalg.norm(navigation_xy - sparse_path[-1])) > 1e-4
                ):
                    sparse_path.append(np.asarray(navigation_xy, dtype=np.float64))
                navigation_path_world = [point.tolist() for point in sparse_path]
            else:
                sparse_path = [np.asarray(navigation_xy, dtype=np.float64)]

            # Bias the semantic Inside pose toward the selected aisle-side
            # approach. This recovers arm reach while the base remains far
            # enough from the counter. Every candidate is independently
            # generated by OmniGibson's predicate sampler, so the selected
            # pose remains a valid Inside proposal.
            best_pose_distance = float(
                np.linalg.norm(desired_hand_pos[:2] - navigation_xy)
            )
            # The collision-safe aisle pose is farther from the container than
            # the old unsafe goal.  Eight random Inside samples left the hand
            # target 0.86 m from the actual robot root; 64 reduced it to 0.79 m
            # but remained outside the proven workspace.  Draw 256 legal
            # predicate samples to cover the near/aisle-side container volume.
            for _ in range(0 if target_cell_aabb is not None else 255):
                try:
                    candidate_object_pose = (
                        primitives._sample_pose_with_object_and_predicate(
                            object_states.Inside,
                            held_before,
                            target_obj,
                            world_aligned=True,
                        )
                    )
                    placement_pose_sample_count += 1
                    candidate_hand_pose = primitives._get_hand_pose_for_object_pose(
                        candidate_object_pose
                    )
                    candidate_object_pos = _as_numpy_vector(candidate_object_pose[0])
                    candidate_object_quat = _as_numpy_vector(
                        candidate_object_pose[1], expected_size=4
                    )
                    candidate_hand_pos = _as_numpy_vector(candidate_hand_pose[0])
                    candidate_hand_quat = _as_numpy_vector(
                        candidate_hand_pose[1], expected_size=4
                    )
                    if any(
                        value is None
                        for value in (
                            candidate_object_pos,
                            candidate_object_quat,
                            candidate_hand_pos,
                            candidate_hand_quat,
                        )
                    ):
                        continue
                    candidate_distance = float(
                        np.linalg.norm(candidate_hand_pos[:2] - navigation_xy)
                    )
                    if candidate_distance < best_pose_distance:
                        best_pose_distance = candidate_distance
                        desired_object_pose = candidate_object_pose
                        desired_hand_pose = candidate_hand_pose
                        desired_object_pos = candidate_object_pos
                        desired_object_quat = candidate_object_quat
                        desired_hand_pos = candidate_hand_pos
                        desired_hand_quat = candidate_hand_quat
                except Exception as exc:
                    logger.debug("PLACE_INSIDE extra pose sample failed: %s", exc)

            sampled_object_pose_world = {
                "position": desired_object_pos.tolist(),
                "orientation_xyzw": desired_object_quat.tolist(),
            }
            placement_hand_pose_world = {
                "position": desired_hand_pos.tolist(),
                "orientation_xyzw": desired_hand_quat.tolist(),
            }
            preplace_pos = desired_hand_pose[0].clone()
            hand_offset_z = float(
                desired_hand_pose[0][2] - desired_object_pose[0][2]
            )
            clearance_offset_z = (
                hand_offset_z + 0.10 if target_cell_aabb is not None else 0.10
            )
            preplace_pos[2] = max(
                float(preplace_pos[2]),
                float(destination_max[2]) + clearance_offset_z,
            )
            preplace_pose = (preplace_pos, desired_hand_pose[1])
            preplace_hand_pose_world = {
                "position": _as_numpy_vector(preplace_pos).tolist(),
                "orientation_xyzw": desired_hand_quat.tolist(),
            }
            logger.warning(
                "PLACE_INSIDE selected object_pose=%s hand_pose=%s approach_hand_xy_m=%.4f",
                sampled_object_pose_world,
                placement_hand_pose_world,
                best_pose_distance,
            )

            navigation_yaw = math.atan2(
                desired_hand_pos[1] - navigation_xy[1],
                desired_hand_pos[0] - navigation_xy[0],
            )
            navigation_pose = th.tensor(
                [navigation_xy[0], navigation_xy[1], navigation_yaw],
                dtype=th.float32,
            )
            navigation_base_pose_world = navigation_pose.tolist()
            placement_phase = "pre_navigation"
            logger.warning(
                "PLACE_INSIDE pre-navigation destination=%s mode=%s "
                "base_pose_xyyaw=%s standoff_m=%.3f candidates=%d "
                "geodesic_m=%s waypoints=%d",
                destination_name,
                navigation_mode,
                navigation_base_pose_world,
                navigation_standoff_m,
                navigation_candidate_count,
                navigation_geodesic_distance_m,
                len(sparse_path),
            )
            navigate_direct = getattr(primitives, "_navigate_to_pose_direct", None)
            if not callable(navigate_direct):
                raise RuntimeError("direct base navigation primitive is unavailable")
            # The stock navigation helper calls ``_empty_action()`` with its
            # default ``follow_arm_targets=True``.  CuRobo execution does not
            # refresh those saved primitive targets, so navigation otherwise
            # pulls a successfully lifted arm back toward a stale pre-grasp
            # pose while carrying the object.  During base-only navigation,
            # ask every non-base controller for its true current-state no-op.
            original_empty_action = getattr(primitives, "_empty_action", None)
            if callable(original_empty_action):
                primitives._empty_action = lambda *args, **kwargs: (
                    original_empty_action(follow_arm_targets=False)
                )
                navigation_arm_hold_mode = "current_joint_no_op"

            # R1 has a holonomic base, but OmniGibson's stock direct helper
            # rotates to every 0.10 m A* edge before translating.  With a long
            # attached tool this repeatedly sweeps the payload through nearby
            # cabinet corners; the observed result was a 500-step rotation
            # timeout at an otherwise traversable waypoint.  Track each A*
            # edge in the base frame with zero angular velocity and preserve
            # the carried-tool orientation.  Rotate only once, at the final
            # high-clearance standoff.  Mocks and non-holonomic robots retain
            # the stock primitive through the capability-gated fallback.
            get_robot_pose_from_2d = getattr(
                primitives, "_get_robot_pose_from_2d_pose", None
            )
            world_pose_to_robot_pose = getattr(
                primitives, "_world_pose_to_robot_pose", None
            )
            postprocess_action = getattr(primitives, "_postprocess_action", None)
            rotate_in_place = getattr(primitives, "_rotate_in_place", None)
            base_action_index = getattr(self._robot, "controller_action_idx", {}).get(
                "base"
            )
            use_holonomic_carry_navigation = (
                all(
                    callable(value)
                    for value in (
                        get_robot_pose_from_2d,
                        world_pose_to_robot_pose,
                        postprocess_action,
                        rotate_in_place,
                        original_empty_action,
                    )
                )
                and base_action_index is not None
            )

            def navigate_holonomic_carry_pose(
                pose_2d: Any,
                *,
                final_waypoint: bool,
                distance_threshold_m: float | None = None,
                min_speed_mps: float = 0.08,
            ) -> Generator[Any, None, None]:
                current_pose = self._robot.get_position_orientation()
                current_quat = _as_numpy_vector(current_pose[1], expected_size=4)
                if current_quat is None:
                    raise RuntimeError(
                        "robot orientation unavailable during carry navigation"
                    )
                x, y, z, w = current_quat
                current_yaw = math.atan2(
                    2.0 * (w * z + x * y),
                    1.0 - 2.0 * (y * y + z * z),
                )
                translation_pose = pose_2d.clone()
                if not final_waypoint:
                    translation_pose[2] = float(current_yaw)
                end_pose = get_robot_pose_from_2d(translation_pose)
                if distance_threshold_m is None:
                    distance_threshold_m = 0.06 if final_waypoint else 0.075
                maximum_steps = 800
                for _ in range(maximum_steps):
                    body_target_pose = world_pose_to_robot_pose(end_pose)
                    local_delta = _as_numpy_vector(body_target_pose[0])
                    if local_delta is None:
                        raise RuntimeError(
                            "local waypoint delta unavailable during carry navigation"
                        )
                    distance_m = float(np.linalg.norm(local_delta[:2]))
                    if distance_m < distance_threshold_m:
                        break
                    action = original_empty_action(follow_arm_targets=False)
                    base_action = action[base_action_index]
                    if int(np.asarray(base_action).size) != 3:
                        raise RuntimeError(
                            "holonomic carry navigation requires a 3-DoF base action"
                        )
                    speed_mps = float(
                        np.clip(1.2 * distance_m, min_speed_mps, 0.25)
                    )
                    direction = local_delta[:2] / max(distance_m, 1e-9)
                    base_action[0] = float(direction[0] * speed_mps)
                    base_action[1] = float(direction[1] * speed_mps)
                    base_action[2] = 0.0
                    action[base_action_index] = base_action
                    yield postprocess_action(action)
                else:
                    raise RuntimeError(
                        "holonomic carry navigation could not reach waypoint "
                        f"within {maximum_steps} steps"
                    )

                stop_action = original_empty_action(follow_arm_targets=False)
                yield postprocess_action(stop_action)
                if final_waypoint:
                    # Match the stock low-precision terminal tolerance.  The
                    # final standoff has the carried-object clearance enforced
                    # by the eroded A* map above.
                    yield from rotate_in_place(end_pose, angle_threshold=0.2)

            if use_holonomic_carry_navigation:
                navigation_mode += "_holonomic_payload_preserving"
            try:
                for waypoint_index, waypoint_xy in enumerate(sparse_path):
                    if waypoint_index + 1 < len(sparse_path):
                        next_xy = sparse_path[waypoint_index + 1]
                        waypoint_yaw = math.atan2(
                            next_xy[1] - waypoint_xy[1],
                            next_xy[0] - waypoint_xy[0],
                        )
                    else:
                        waypoint_yaw = navigation_yaw
                    waypoint_pose = th.tensor(
                        [waypoint_xy[0], waypoint_xy[1], waypoint_yaw],
                        dtype=th.float32,
                    )
                    navigation_waypoint_index = waypoint_index
                    navigation_waypoint_pose_world = waypoint_pose.tolist()
                    is_final_waypoint = waypoint_index + 1 == len(sparse_path)
                    navigation_actions = (
                        navigate_holonomic_carry_pose(
                            waypoint_pose,
                            final_waypoint=is_final_waypoint,
                        )
                        if use_holonomic_carry_navigation
                        else navigate_direct(waypoint_pose, low_precision=True)
                    )
                    for action in navigation_actions:
                        if action is not None:
                            steps += 1
                            pre_navigation_steps += 1
                            yield action
            finally:
                if callable(original_empty_action):
                    primitives._empty_action = original_empty_action

            # Base navigation terminates within a controller tolerance, which
            # can leave a thin object a few millimetres across the container
            # wall even when the sampled Inside pose is valid.  Before asking
            # CuRobo for a much larger arm motion, make at most two small,
            # measured base corrections.  The 25 mm alignment margin leaves
            # 15 mm of reserve over the 10 mm release gate below, so the
            # controller's positional tolerance cannot turn a near miss into
            # an unsafe drop.  Large corrections are deliberately refused.
            placement_phase = "pre_release_xy_alignment"
            alignment_margin_m = 0.005 if target_cell_aabb is not None else 0.025
            maximum_alignment_m = 0.08
            for alignment_index in range(2):
                current_object_aabb = _object_world_aabb(held_before)
                current_destination_aabb = _object_world_aabb(target_obj)
                if current_object_aabb is None or current_destination_aabb is None:
                    break
                alignment_target_aabb = (
                    target_cell_aabb
                    if target_cell_aabb is not None
                    else current_destination_aabb
                )
                correction_xy, can_fit = _xy_containment_correction(
                    current_object_aabb,
                    alignment_target_aabb,
                    margin_m=alignment_margin_m,
                )
                correction_norm_m = float(np.linalg.norm(correction_xy))
                attempt_evidence = {
                    "attempt": alignment_index + 1,
                    "margin_m": alignment_margin_m,
                    "correction_xy_m": correction_xy.tolist(),
                    "correction_norm_m": correction_norm_m,
                    "can_fit": can_fit,
                    "executed": False,
                }
                drop_alignment_attempts.append(attempt_evidence)
                if not can_fit or correction_norm_m <= 0.002:
                    break
                if correction_norm_m > maximum_alignment_m:
                    attempt_evidence["refused_reason"] = "correction_exceeds_safe_limit"
                    break
                # A grid-cell goal already has a deterministic hand target
                # and strict final AABB verification.  For a small residual,
                # let the staged arm waypoints absorb the translation instead
                # of rotating / translating the base next to the toolbox wall.
                # The observed 46.5 mm correction repeatedly timed out in
                # rotate_in_place even though it is well inside the arm's
                # incremental Cartesian workspace.
                if target_cell_aabb is not None and correction_norm_m <= 0.06:
                    attempt_evidence["executed_by"] = "staged_arm_waypoints"
                    attempt_evidence["deferred_to_arm_planner"] = True
                    break

                robot_pose_before_alignment = self._robot.get_position_orientation()
                robot_position_before_alignment = _as_numpy_vector(
                    robot_pose_before_alignment[0]
                )
                if robot_position_before_alignment is None:
                    attempt_evidence["refused_reason"] = "robot_pose_unavailable"
                    break
                alignment_pose = th.tensor(
                    [
                        robot_position_before_alignment[0] + correction_xy[0],
                        robot_position_before_alignment[1] + correction_xy[1],
                        navigation_yaw,
                    ],
                    dtype=th.float32,
                )
                attempt_evidence["target_base_pose_xyyaw"] = alignment_pose.tolist()
                original_empty_action = getattr(primitives, "_empty_action", None)
                if callable(original_empty_action):
                    primitives._empty_action = lambda *args, **kwargs: (
                        original_empty_action(follow_arm_targets=False)
                    )
                try:
                    alignment_actions = (
                        navigate_holonomic_carry_pose(
                            alignment_pose,
                            final_waypoint=False,
                            distance_threshold_m=0.005,
                            min_speed_mps=0.03,
                        )
                        if use_holonomic_carry_navigation
                        else navigate_direct(alignment_pose, low_precision=False)
                    )
                    for action in alignment_actions:
                        if action is not None:
                            steps += 1
                            pre_navigation_steps += 1
                            drop_alignment_steps += 1
                            yield action
                finally:
                    if callable(original_empty_action):
                        primitives._empty_action = original_empty_action
                attempt_evidence["executed"] = True
                logger.warning(
                    "PLACE_INSIDE drop-alignment attempt=%s",
                    attempt_evidence,
                )

            final_base_position = _as_numpy_vector(
                self._robot.get_position_orientation()[0]
            )
            if final_base_position is not None:
                navigation_final_base_pose_world = final_base_position.tolist()

            # Base yaw changes the world orientation of the attached object
            # during navigation. Recompute the hand goal at the sampled
            # in-container position while preserving that post-navigation
            # object orientation. This avoids an unnecessary, often
            # unreachable wrist flip; the sampled XYZ still comes from the
            # semantic Inside predicate and final AABB containment remains the
            # hard acceptance criterion.
            if target_cell_aabb is not None:
                desired_object_pose = build_deterministic_cell_pose("post_navigation")
                placement_orientation_mode = "cell_principal_axis_alignment"
            else:
                get_held_pose = getattr(held_before, "get_position_orientation", None)
                if callable(get_held_pose):
                    held_pose_after_navigation = get_held_pose()
                    desired_object_pose = (
                        desired_object_pose[0],
                        held_pose_after_navigation[1].clone(),
                    )
                    placement_orientation_mode = (
                        "preserve_post_navigation_grasp_orientation"
                    )
                else:
                    # Lightweight adapters may expose only the attachment and
                    # AABB contract.  For non-grid placement, retaining the
                    # sampler orientation preserves the established generic
                    # Inside behavior.  Grid placement never takes this
                    # fallback because its deterministic geometry path
                    # requires a real object pose and fails closed above.
                    placement_orientation_mode = "sampled_world_aligned"
            desired_hand_pose = primitives._get_hand_pose_for_object_pose(
                desired_object_pose
            )
            desired_object_pos = _as_numpy_vector(desired_object_pose[0])
            desired_object_quat = _as_numpy_vector(
                desired_object_pose[1], expected_size=4
            )
            desired_hand_pos = _as_numpy_vector(desired_hand_pose[0])
            desired_hand_quat = _as_numpy_vector(desired_hand_pose[1], expected_size=4)
            if any(
                value is None
                for value in (
                    desired_object_pos,
                    desired_object_quat,
                    desired_hand_pos,
                    desired_hand_quat,
                )
            ):
                raise RuntimeError("post-navigation placement pose is unavailable")
            sampled_object_pose_world = {
                "position": desired_object_pos.tolist(),
                "orientation_xyzw": desired_object_quat.tolist(),
            }
            placement_hand_pose_world = {
                "position": desired_hand_pos.tolist(),
                "orientation_xyzw": desired_hand_quat.tolist(),
            }
            preplace_pos = desired_hand_pose[0].clone()
            hand_offset_z = float(
                desired_hand_pose[0][2] - desired_object_pose[0][2]
            )
            clearance_offset_z = (
                hand_offset_z + 0.10 if target_cell_aabb is not None else 0.10
            )
            preplace_pos[2] = max(
                float(preplace_pos[2]),
                float(destination_max[2]) + clearance_offset_z,
            )
            preplace_pose = (preplace_pos, desired_hand_pose[1])
            preplace_hand_pose_world = {
                "position": _as_numpy_vector(preplace_pos).tolist(),
                "orientation_xyzw": desired_hand_quat.tolist(),
            }

            # Plan two arm motions (above the opening, then inside) against
            # only the local workcell meshes. The held object and destination
            # are ignored by CuRobo for this short top-entry motion; the
            # simulator still enforces their physical geometry and attachment.
            base_pose_before_plan = self._robot.get_position_orientation()
            base_before_plan = _as_numpy_vector(base_pose_before_plan[0])
            base_quat_before_plan = _as_numpy_vector(
                base_pose_before_plan[1], expected_size=4
            )
            if base_before_plan is not None:
                preplan_base_to_hand_xy_m = float(
                    np.linalg.norm(base_before_plan[:2] - desired_hand_pos[:2])
                )
            preplan_base_pose_world = {
                "position": (
                    None if base_before_plan is None else base_before_plan.tolist()
                ),
                "orientation_xyzw": (
                    None
                    if base_quat_before_plan is None
                    else base_quat_before_plan.tolist()
                ),
            }
            eef_pose_before_plan = None
            eef_link = getattr(self._robot, "eef_links", {}).get(self._arm)
            get_eef_pose = getattr(eef_link, "get_position_orientation", None)
            if callable(get_eef_pose):
                eef_pose_before_plan = get_eef_pose()
                eef_pos_before_plan = _as_numpy_vector(eef_pose_before_plan[0])
                eef_quat_before_plan = _as_numpy_vector(
                    eef_pose_before_plan[1], expected_size=4
                )
                preplan_eef_pose_world = {
                    "position": (
                        None
                        if eef_pos_before_plan is None
                        else eef_pos_before_plan.tolist()
                    ),
                    "orientation_xyzw": (
                        None
                        if eef_quat_before_plan is None
                        else eef_quat_before_plan.tolist()
                    ),
                }
            object_aabb_for_clearance = _object_world_aabb(held_before)
            if (
                target_cell_aabb is not None
                and eef_pose_before_plan is not None
                and eef_pos_before_plan is not None
                and object_aabb_for_clearance is not None
            ):
                hand_to_object_bottom_m = max(
                    0.0,
                    float(eef_pos_before_plan[2] - object_aabb_for_clearance[0][2]),
                )
                preplace_pos[2] = max(
                    float(preplace_pos[2]),
                    float(destination_max[2]) + hand_to_object_bottom_m + 0.10,
                )
                preplace_pose = (preplace_pos, desired_hand_pose[1])
                preplace_hand_pose_world["position"] = _as_numpy_vector(
                    preplace_pos
                ).tolist()
            logger.warning(
                "PLACE_INSIDE preplace_pose=%s base_to_hand_xy_m=%s base_pose=%s eef_pose=%s",
                preplace_hand_pose_world,
                (
                    None
                    if preplan_base_to_hand_xy_m is None
                    else round(preplan_base_to_hand_xy_m, 4)
                ),
                preplan_base_pose_world,
                preplan_eef_pose_world,
            )
            local_ignores = self._local_collision_ignore_objects(target_obj)
            planner_ignores = list(local_ignores)
            for ignored_obj in (held_before, target_obj):
                if all(ignored_obj is not item for item in planner_ignores):
                    planner_ignores.append(ignored_obj)

            # CuRobo can conservatively voxelize an open container / counter
            # assembly as a closed obstacle. Keep normal local collision
            # planning as the primary path. If both fixed-base and whole-body
            # planning fail, retry this explicitly top-entry-only motion with
            # scene meshes ignored. Robot self-collision remains enabled and
            # Isaac physics still executes the trajectory against the real
            # geometry; release plus AABB containment remain mandatory.
            semantic_top_entry_ignores = [
                obj for obj in self._robot.scene.objects if obj is not self._robot
            ]
            for ignored_obj in (held_before, target_obj):
                if all(ignored_obj is not item for item in semantic_top_entry_ignores):
                    semantic_top_entry_ignores.append(ignored_obj)

            # If base navigation has already carried the object directly over
            # the opening, avoid an unnecessary boundary-reach arm motion.
            # This is a guarded gravity placement, not an unconditional drop:
            # require the complete object AABB to fit inside the destination
            # XY bounds with wall clearance, require a short downward fall,
            # and still accept success only after release, settle, and full
            # 3D AABB containment below.
            object_aabb_before_drop = _object_world_aabb(held_before)
            destination_aabb_before_drop = _object_world_aabb(target_obj)
            drop_target_aabb = (
                target_cell_aabb
                if target_cell_aabb is not None
                else destination_aabb_before_drop
            )
            gravity_drop_ready = False
            if object_aabb_before_drop is not None and drop_target_aabb is not None:
                object_min_drop, object_max_drop = object_aabb_before_drop
                destination_min_drop, destination_max_drop = drop_target_aabb
                wall_margin_m = 0.005
                xy_contained_before_drop = bool(
                    np.all(
                        object_min_drop[:2] >= destination_min_drop[:2] + wall_margin_m
                    )
                    and np.all(
                        object_max_drop[:2] <= destination_max_drop[:2] - wall_margin_m
                    )
                )
                drop_height_m = float(object_min_drop[2] - destination_max_drop[2])
                gravity_drop_ready = bool(
                    xy_contained_before_drop and -0.02 <= drop_height_m <= 0.35
                )
                pre_release_drop_evidence = {
                    "xy_contained_with_margin": xy_contained_before_drop,
                    "wall_margin_m": wall_margin_m,
                    "drop_height_m": drop_height_m,
                    "object_aabb_world": [
                        object_min_drop.tolist(),
                        object_max_drop.tolist(),
                    ],
                    "destination_aabb_world": [
                        destination_min_drop.tolist(),
                        destination_max_drop.tolist(),
                    ],
                    "target_is_grid_cell": target_cell_aabb is not None,
                    "ready": gravity_drop_ready,
                }
                logger.warning(
                    "PLACE_INSIDE gravity-drop gate=%s",
                    pre_release_drop_evidence,
                )

            standard_profiles = (
                (
                    CuRoboEmbodimentSelection.ARM,
                    planner_ignores,
                    "local_obstacles",
                ),
                (
                    CuRoboEmbodimentSelection.DEFAULT,
                    planner_ignores,
                    "local_obstacles",
                ),
                (
                    CuRoboEmbodimentSelection.ARM,
                    semantic_top_entry_ignores,
                    "semantic_top_entry",
                ),
                (
                    CuRoboEmbodimentSelection.DEFAULT,
                    semantic_top_entry_ignores,
                    "semantic_top_entry",
                ),
            )
            descend_profiles = (
                (
                    CuRoboEmbodimentSelection.ARM,
                    semantic_top_entry_ignores,
                    "semantic_top_entry",
                ),
            )

            # A single 0.4--0.6 m Cartesian displacement can have a valid
            # endpoint IK solution yet fail CuRobo's trajectory search from
            # the post-navigation joint state.  Build short, auditable
            # waypoints instead: retract and lift together while retaining
            # the current grasp orientation, reorient only after leaving the
            # near-maximum-reach posture, then descend in small increments.
            # Changing the wrist orientation or requesting a pure vertical
            # move at maximum reach failed IK in real Isaac / CuRobo runs.
            placement_segments: list[tuple[str, Any, Any]] = []
            current_pos = (
                eef_pose_before_plan[0].clone()
                if eef_pose_before_plan is not None
                else preplace_pose[0].clone()
            )
            current_waypoint_quat = (
                eef_pose_before_plan[1].clone()
                if eef_pose_before_plan is not None
                else desired_hand_pose[1].clone()
            )
            desired_waypoint_quat = desired_hand_pose[1].clone()
            # At a stretched posture, a pure vertical request can be outside
            # the arm's instantaneous reachable direction even when both the
            # start and final poses are valid. Retract toward the opening and
            # lift at the same time, giving the elbow margin on every short
            # Cartesian segment. Grid-cell placement uses finer segments
            # because a long industrial tool is close to the bin wall.
            clearance_delta = preplace_pose[0] - current_pos
            clearance_distance = float(th.linalg.vector_norm(clearance_delta))
            # The industrial plier reaches the toolbox from a near-boundary
            # wrist configuration.  In the real R1 scene, a 90--100 mm first
            # Cartesian hop was rejected by CuRobo even though the endpoint
            # IK was valid (the planner could not find a collision-free joint
            # interpolation from the stretched grasp state).  Use shorter
            # guarded hops for grid placement so the planner can leave that
            # singularity incrementally while preserving the grasp pose.
            clearance_segment_max_m = (
                0.035 if target_cell_aabb is not None else 0.18
            )
            clearance_count = min(
                6,
                max(
                    1,
                    int(math.ceil(clearance_distance / clearance_segment_max_m)),
                ),
            )
            for index in range(1, clearance_count + 1):
                pose_pos = (
                    current_pos + clearance_delta * index / clearance_count
                )
                segment_name = (
                    "arm_above_opening"
                    if index == clearance_count
                    else (
                        f"arm_retract_lift_clearance_{index}_of_"
                        f"{clearance_count}"
                    )
                )
                placement_segments.append(
                    (
                        segment_name,
                        (pose_pos, current_waypoint_quat.clone()),
                        standard_profiles,
                    )
                )

            reorient_position = preplace_pose[0].clone()
            current_quat_np = _as_numpy_vector(current_waypoint_quat, expected_size=4)
            desired_quat_np = _as_numpy_vector(desired_waypoint_quat, expected_size=4)
            reorient_angle_rad = _quat_shortest_angle_rad_xyzw(
                current_quat_np, desired_quat_np
            )
            reorient_count = (
                max(
                    1,
                    int(math.ceil(reorient_angle_rad / math.radians(25.0))),
                )
                if reorient_angle_rad > math.radians(2.0)
                else 0
            )
            for index in range(1, reorient_count + 1):
                interpolated_quat = _quat_slerp_xyzw(
                    current_quat_np,
                    desired_quat_np,
                    index / reorient_count,
                )
                pose_quat = desired_waypoint_quat.clone()
                pose_quat[:] = th.as_tensor(
                    interpolated_quat,
                    dtype=pose_quat.dtype,
                    device=pose_quat.device,
                )
                placement_segments.append(
                    (
                        f"arm_reorient_clearance_{index}_of_{reorient_count}",
                        (reorient_position.clone(), pose_quat),
                        standard_profiles,
                    )
                )

            descend_delta = desired_hand_pose[0] - preplace_pose[0]
            descend_distance = float(th.linalg.vector_norm(descend_delta))
            descend_count = (
                max(1, int(math.ceil(descend_distance / 0.08)))
                if eef_pose_before_plan is not None
                else 1
            )
            for index in range(1, descend_count + 1):
                pose_pos = preplace_pose[0] + descend_delta * index / descend_count
                segment_name = (
                    "arm_descend_inside"
                    if index == descend_count
                    else f"arm_descend_inside_{index}_of_{descend_count}"
                )
                placement_segments.append(
                    (
                        segment_name,
                        (pose_pos, desired_waypoint_quat.clone()),
                        descend_profiles,
                    )
                )

            placement_waypoints_world = [
                {
                    "segment": segment_name,
                    "position": _as_numpy_vector(pose[0]).tolist(),
                    "orientation_xyzw": _as_numpy_vector(
                        pose[1], expected_size=4
                    ).tolist(),
                }
                for segment_name, pose, _ in placement_segments
            ]
            placement_waypoint_records = {
                record["segment"]: record
                for record in placement_waypoints_world
            }
            if gravity_drop_ready:
                placement_strategy = "guarded_gravity_drop"
                placement_segments = []
                placement_waypoints_world = []
            logger.warning(
                "PLACE_INSIDE strategy=%s staged waypoints=%s",
                placement_strategy,
                placement_waypoints_world,
            )
            obstacle_snapshot_profile = None
            for segment_name, pose, planning_profiles in placement_segments:
                placement_phase = f"plan_{segment_name}"
                trajectory = None
                last_planning_error: Exception | None = None
                for embodiment, ignore_objects, collision_profile in planning_profiles:
                    planner_get_obj_in_hand = primitives._get_obj_in_hand
                    try:
                        primitives._get_obj_in_hand = lambda: None
                        trajectory = primitives._plan_joint_motion(
                            target_pos={self._robot.eef_link_names[self._arm]: pose[0]},
                            target_quat={
                                self._robot.eef_link_names[self._arm]: pose[1]
                            },
                            embodiment_selection=embodiment,
                            ignore_objects=ignore_objects,
                            skip_obstacle_update=(
                                obstacle_snapshot_profile == collision_profile
                            ),
                        )
                    except Exception as exc:
                        last_planning_error = exc
                        planning_attempts.append(
                            {
                                "segment": segment_name,
                                "embodiment": str(
                                    getattr(embodiment, "value", embodiment)
                                ),
                                "collision_profile": collision_profile,
                                "success": False,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                    else:
                        planning_attempts.append(
                            {
                                "segment": segment_name,
                                "embodiment": str(
                                    getattr(embodiment, "value", embodiment)
                                ),
                                "collision_profile": collision_profile,
                                "success": True,
                                "trajectory_steps": len(trajectory),
                            }
                        )
                        break
                    finally:
                        primitives._get_obj_in_hand = planner_get_obj_in_hand
                        obstacle_snapshot_profile = collision_profile
                if trajectory is None:
                    if last_planning_error is None:
                        raise RuntimeError(
                            f"no CuRobo embodiment configured for {segment_name}"
                        )
                    raise last_planning_error
                if primitives._get_obj_in_hand() is not held_before:
                    raise RuntimeError(
                        "held-object attachment changed during placement"
                    )
                placement_phase = f"execute_{segment_name}"
                placement_execution_kwargs = (
                    {"ignore_failure": True}
                    if supports_cartesian_tail_audit
                    else {}
                )
                for action in primitives._execute_motion_plan(
                    trajectory,
                    **placement_execution_kwargs,
                ):
                    if action is not None:
                        steps += 1
                        placement_steps += 1
                        yield action
                if supports_cartesian_tail_audit:
                    actual_segment_pose = get_eef_pose()
                    actual_segment_pos = _as_numpy_vector(
                        actual_segment_pose[0]
                    )
                    actual_segment_quat = _as_numpy_vector(
                        actual_segment_pose[1], expected_size=4
                    )
                    target_segment_pos = _as_numpy_vector(pose[0])
                    target_segment_quat = _as_numpy_vector(
                        pose[1], expected_size=4
                    )
                    position_error_m = (
                        None
                        if actual_segment_pos is None
                        or target_segment_pos is None
                        else float(
                            np.linalg.norm(
                                actual_segment_pos - target_segment_pos
                            )
                        )
                    )
                    orientation_error_deg = (
                        None
                        if actual_segment_quat is None
                        or target_segment_quat is None
                        else math.degrees(
                            _quat_shortest_angle_rad_xyzw(
                                actual_segment_quat,
                                target_segment_quat,
                            )
                        )
                    )
                    segment_attachment_same = (
                        primitives._get_obj_in_hand() is held_before
                    )
                    segment_record = placement_waypoint_records[segment_name]
                    segment_record.update(
                        {
                            "actual_position": (
                                None
                                if actual_segment_pos is None
                                else actual_segment_pos.tolist()
                            ),
                            "actual_orientation_xyzw": (
                                None
                                if actual_segment_quat is None
                                else actual_segment_quat.tolist()
                            ),
                            "position_error_m": position_error_m,
                            "orientation_error_deg": orientation_error_deg,
                            "attachment_same": segment_attachment_same,
                            "cartesian_tail_audit_passed": bool(
                                position_error_m is not None
                                and position_error_m <= 0.025
                                and orientation_error_deg is not None
                                and orientation_error_deg <= 5.0
                                and segment_attachment_same
                            ),
                        }
                    )
                    if not segment_record["cartesian_tail_audit_passed"]:
                        raise RuntimeError(
                            "placement Cartesian tail audit failed: "
                            f"{segment_record}"
                        )

            placement_phase = "release"
            for action in primitives._execute_release():
                if action is not None:
                    steps += 1
                    release_steps += 1
                    yield action
            placement_phase = "settle"
            for action in primitives._settle_robot():
                if action is not None:
                    steps += 1
                    settle_steps += 1
                    yield action
            remaining = primitives._get_obj_in_hand()
            remaining_name = getattr(remaining, "name", None)
            released = remaining is None
            object_aabb = _object_world_aabb(held_before)
            destination_aabb = _object_world_aabb(target_obj)
            containment_available = (
                object_aabb is not None and destination_aabb is not None
            )
            aabb_contained = (
                _aabb_contains(object_aabb, destination_aabb, margin_m=0.001)
                if containment_available
                else None
            )
            cell_containment_available = (
                object_aabb is not None and target_cell_aabb is not None
            )
            cell_aabb_contained = (
                _aabb_contains(object_aabb, target_cell_aabb, margin_m=0.001)
                if cell_containment_available
                else None
            )
            placement_verified = (
                released
                and (not containment_available or bool(aabb_contained))
                and (not cell_containment_available or bool(cell_aabb_contained))
            )
            evidence = {
                "placement_mode": "place_inside",
                "destination_object": destination_name,
                "pre_navigation_carry_steps": pre_navigation_carry_steps,
                "carry_planning_attempts": carry_planning_attempts,
                "carry_waypoints_world": carry_waypoints_world,
                "pre_navigation_steps": pre_navigation_steps,
                "pre_navigation_mode": navigation_mode,
                "pre_navigation_base_pose_world": navigation_base_pose_world,
                "pre_navigation_standoff_m": navigation_standoff_m,
                "pre_navigation_path_world": navigation_path_world,
                "pre_navigation_candidate_count": navigation_candidate_count,
                "pre_navigation_geodesic_distance_m": (navigation_geodesic_distance_m),
                "pre_navigation_last_waypoint_index": navigation_waypoint_index,
                "pre_navigation_last_waypoint_pose_world": (
                    navigation_waypoint_pose_world
                ),
                "pre_navigation_final_base_pose_world": (
                    navigation_final_base_pose_world
                ),
                "pre_navigation_terminal_clearance_m": (
                    navigation_terminal_clearance_m
                ),
                "pre_navigation_arm_hold_mode": navigation_arm_hold_mode,
                "drop_alignment_steps": drop_alignment_steps,
                "drop_alignment_attempts": drop_alignment_attempts,
                "placement_steps": placement_steps,
                "planning_attempts": planning_attempts,
                "release_steps": release_steps,
                "settle_steps": settle_steps,
                "primitive_attempts": 1,
                "sampled_object_pose_world": sampled_object_pose_world,
                "placement_hand_pose_world": placement_hand_pose_world,
                "preplace_hand_pose_world": preplace_hand_pose_world,
                "placement_pose_sample_count": placement_pose_sample_count,
                "preplan_base_to_hand_xy_m": preplan_base_to_hand_xy_m,
                "preplan_base_pose_world": preplan_base_pose_world,
                "preplan_eef_pose_world": preplan_eef_pose_world,
                "placement_orientation_mode": placement_orientation_mode,
                "placement_strategy": placement_strategy,
                "placement_waypoints_world": placement_waypoints_world,
                "pre_release_drop_evidence": pre_release_drop_evidence,
                "object_in_hand_after_release": remaining_name,
                "released": released,
                "containment_check_available": containment_available,
                "aabb_contained": aabb_contained,
                "cell_index": cell_index,
                "grid_shape": (
                    list(target_cell_audit["grid_shape"])
                    if target_cell_audit is not None
                    else None
                ),
                "target_cell_aabb_world": (
                    target_cell_audit["target_cell_aabb_world"]
                    if target_cell_audit is not None
                    else None
                ),
                "cell_axis_convention": (
                    {
                        "indexing": target_cell_audit["indexing"],
                        "column_axis_world": target_cell_audit["column_axis_world"],
                        "row_axis_world": target_cell_audit["row_axis_world"],
                        "cell_margin_m": target_cell_audit["cell_margin_m"],
                    }
                    if target_cell_audit is not None
                    else None
                ),
                "cell_pose_audits": cell_pose_audits,
                "cell_containment_check_available": cell_containment_available,
                "cell_aabb_contained": cell_aabb_contained,
                "object_aabb_world": (
                    [object_aabb[0].tolist(), object_aabb[1].tolist()]
                    if object_aabb is not None
                    else None
                ),
                "destination_aabb_world": (
                    [destination_aabb[0].tolist(), destination_aabb[1].tolist()]
                    if destination_aabb is not None
                    else None
                ),
            }
            return GraspResult(
                success=placement_verified,
                object_in_hand=remaining_name,
                grasp_pos_world=np.zeros(3, dtype=np.float32),
                grasp_quat_world=np.array([0, 0, 0, 1], dtype=np.float32),
                anygrasp_score=0.0,
                total_sim_steps=steps,
                error=(
                    None
                    if placement_verified
                    else (
                        "object remained in hand after PLACE_INSIDE"
                        if not released
                        else (
                            "object AABB is not contained by requested grid cell"
                            if cell_containment_available and not cell_aabb_contained
                            else "object AABB is not contained by destination AABB"
                        )
                    )
                ),
                failure_phase=None if placement_verified else "place_inside",
                physical_grasp_verified=placement_verified,
                physical_evidence=evidence,
                placement_verified=placement_verified,
            )
        except Exception as exc:
            try:
                final_base_position = _as_numpy_vector(
                    self._robot.get_position_orientation()[0]
                )
                if final_base_position is not None:
                    navigation_final_base_pose_world = final_base_position.tolist()
            except Exception:
                pass
            logger.exception(
                "PLACE_INSIDE failed destination=%s phase=%s after_steps=%d",
                destination_name,
                placement_phase,
                steps,
            )
            remaining = None
            try:
                remaining = primitives._get_obj_in_hand()
            except Exception:
                pass
            remaining_name = getattr(remaining, "name", None)
            evidence = {
                "placement_mode": "place_inside",
                "destination_object": destination_name,
                "failure_phase": placement_phase,
                "pre_navigation_carry_steps": pre_navigation_carry_steps,
                "carry_planning_attempts": carry_planning_attempts,
                "carry_waypoints_world": carry_waypoints_world,
                "pre_navigation_steps": pre_navigation_steps,
                "pre_navigation_mode": navigation_mode,
                "pre_navigation_base_pose_world": navigation_base_pose_world,
                "pre_navigation_standoff_m": navigation_standoff_m,
                "pre_navigation_path_world": navigation_path_world,
                "pre_navigation_candidate_count": navigation_candidate_count,
                "pre_navigation_geodesic_distance_m": (navigation_geodesic_distance_m),
                "pre_navigation_last_waypoint_index": navigation_waypoint_index,
                "pre_navigation_last_waypoint_pose_world": (
                    navigation_waypoint_pose_world
                ),
                "pre_navigation_final_base_pose_world": (
                    navigation_final_base_pose_world
                ),
                "pre_navigation_terminal_clearance_m": (
                    navigation_terminal_clearance_m
                ),
                "pre_navigation_arm_hold_mode": navigation_arm_hold_mode,
                "drop_alignment_steps": drop_alignment_steps,
                "drop_alignment_attempts": drop_alignment_attempts,
                "sampled_object_pose_world": sampled_object_pose_world,
                "placement_hand_pose_world": placement_hand_pose_world,
                "preplace_hand_pose_world": preplace_hand_pose_world,
                "placement_pose_sample_count": placement_pose_sample_count,
                "preplan_base_to_hand_xy_m": preplan_base_to_hand_xy_m,
                "preplan_base_pose_world": preplan_base_pose_world,
                "preplan_eef_pose_world": preplan_eef_pose_world,
                "placement_orientation_mode": placement_orientation_mode,
                "placement_strategy": placement_strategy,
                "placement_waypoints_world": placement_waypoints_world,
                "pre_release_drop_evidence": pre_release_drop_evidence,
                "planning_attempts": planning_attempts,
                "placement_steps": placement_steps,
                "release_steps": release_steps,
                "settle_steps": settle_steps,
                "object_in_hand_after_failure": remaining_name,
                "released": remaining is None,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            }
            return GraspResult(
                success=False,
                object_in_hand=remaining_name,
                grasp_pos_world=np.zeros(3, dtype=np.float32),
                grasp_quat_world=np.array([0, 0, 0, 1], dtype=np.float32),
                anygrasp_score=0.0,
                total_sim_steps=steps,
                error=f"place inside failed: {type(exc).__name__}: {exc}",
                failure_phase="place_inside",
                physical_evidence=evidence,
                placement_verified=False,
            )

    def begin_grasp(
        self,
        grasp_candidate: Any,
        *,
        camera_pose_world: np.ndarray,
        target_obj: Any | None = None,
    ) -> GraspExecution:
        primitives = self._ensure_primitives()
        if primitives is None:
            result = GraspResult(
                success=False,
                object_in_hand=None,
                grasp_pos_world=np.zeros(3, dtype=np.float32),
                grasp_quat_world=np.array([0, 0, 0, 1], dtype=np.float32),
                anygrasp_score=float(grasp_candidate.score),
                total_sim_steps=0,
                error="StarterSemanticActionPrimitives init failed",
            )
            return GraspExecution(self._immediate_result(result))
        return GraspExecution(
            self._grasp_generator(
                primitives,
                grasp_candidate,
                camera_pose_world=camera_pose_world,
                target_obj=target_obj,
            )
        )

    def _grasp_generator(
        self,
        primitives: Any,
        candidate: Any,
        *,
        camera_pose_world: np.ndarray,
        target_obj: Any | None,
    ) -> Generator[Any, None, GraspResult]:
        import torch as th
        from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection

        grasp_pos, grasp_quat = self.camera_to_world(
            candidate.translation,
            candidate.rotation_matrix,
            camera_pose_world=camera_pose_world,
        )
        grasp_quat_t = th.as_tensor(grasp_quat, dtype=th.float32)
        rotation = _pose_to_matrix(grasp_pos, grasp_quat)[:3, :3]
        approach = rotation[:, 2]
        approach /= np.linalg.norm(approach) + 1e-12

        geometry = GripperGeometryAdapter.from_robot(
            self._robot,
            self._arm,
            fingertip_depth_override_m=self._fingertip_depth_override_m,
            eef_approach_offset_m=self._eef_approach_offset_m,
        )
        grasp_eef_pos = geometry.eef_position(grasp_pos, approach, candidate.depth)
        grasp_tip_pos = grasp_pos + approach.astype(np.float32) * float(candidate.depth)
        original_snapshot = getattr(candidate, "anygrasp_original_snapshot", {})
        target_geometry_evidence = getattr(candidate, "target_geometry_evidence", {})
        depth_fit = (
            target_geometry_evidence.get("depth_fit", {})
            if isinstance(target_geometry_evidence, dict)
            else {}
        )
        execution_pose_audit = {
            "event": "anygrasp_execution_pose_audit",
            "detector_original_camera_translation": original_snapshot.get(
                "original_camera_translation",
                np.asarray(candidate.translation, dtype=np.float64).reshape(3).tolist(),
            ),
            "detector_recentered_camera_translation": np.asarray(
                candidate.translation, dtype=np.float64
            )
            .reshape(3)
            .tolist(),
            "world_canonical_origin": np.asarray(grasp_pos, dtype=float).tolist(),
            "world_approach": np.asarray(approach, dtype=float).tolist(),
            "detector_original_canonical_depth": float(
                original_snapshot.get("depth", candidate.depth)
            ),
            "canonical_depth": float(candidate.depth),
            "fitted_canonical_depth": depth_fit.get("selected_depth_m"),
            "depth_fit_applied": bool(depth_fit.get("applied", False)),
            "fingertip_length": float(geometry.fingertip_depth_m),
            "fingertip_length_source": geometry.source,
            "eef_approach_offset_m": float(geometry.eef_approach_offset_m),
            "world_eef": np.asarray(grasp_eef_pos, dtype=float).tolist(),
            "world_tip": np.asarray(grasp_tip_pos, dtype=float).tolist(),
            "frame_adapter_basis_mapping": self._frame_adapter.validate_basis_mapping(),
        }
        logger.info(
            "%s",
            json.dumps(execution_pose_audit, sort_keys=True, separators=(",", ":")),
        )
        grasp_pose = (th.as_tensor(grasp_eef_pos, dtype=th.float32), grasp_quat_t)
        pregrasp_pos = (
            grasp_eef_pos - approach.astype(np.float32) * self._pregrasp_offset_m
        )
        pregrasp_pose = (
            th.as_tensor(pregrasp_pos, dtype=th.float32),
            grasp_quat_t.clone(),
        )
        standoff_pos = (
            grasp_eef_pos - approach.astype(np.float32) * self._whole_body_standoff_m
        )
        standoff_pose = (
            th.as_tensor(standoff_pos, dtype=th.float32),
            grasp_quat_t.clone(),
        )
        local_collision_ignores = self._local_collision_ignore_objects(target_obj)

        def planning_ignore_objects(
            additional: list[Any] | None = None,
        ) -> list[Any] | None:
            ignored = list(local_collision_ignores)
            for obj in additional or []:
                if all(obj is not item for item in ignored):
                    ignored.append(obj)
            return ignored or None

        attached_ignore_objects = planning_ignore_objects(
            [target_obj] if target_obj is not None else None
        )
        steps = 0
        last_action: Any | None = None
        approach_target_origin: np.ndarray | None = None
        close_target_origin: np.ndarray | None = None
        lift_started = False
        lift_completed = False
        lift_joint_tail_tolerance_enabled = False

        def actions(
            generator: Any,
            *,
            target_guard_origin: np.ndarray | None = None,
            target_guard_tolerance_m: float = _TARGET_DISPLACEMENT_TOLERANCE_M,
        ) -> Generator[Any, None, Any]:
            nonlocal last_action, steps
            iterator = iter(generator)
            while True:
                try:
                    action = next(iterator)
                except StopIteration as stop:
                    return stop.value
                if action is not None:
                    steps += 1
                    last_action = action
                    yield action
                    if target_guard_origin is not None:
                        position = target_position_array()
                        if (
                            position is not None
                            and np.linalg.norm(position - target_guard_origin)
                            > target_guard_tolerance_m
                        ):
                            raise RuntimeError(
                                "target moved during contact-guarded motion "
                                f"by {np.linalg.norm(position - target_guard_origin):.4f} m"
                            )

        def move_to_pose_whole_body(
            pose: tuple[Any, Any],
            *,
            ignore_objects: list[Any] | None = None,
            low_precision: bool = False,
            stop_on_contact: bool = False,
        ) -> Generator[Any, None, None]:
            eef_name = self._robot.eef_link_names[self._arm]
            plan_kwargs: dict[str, Any] = {
                "target_pos": {eef_name: pose[0]},
                "target_quat": {eef_name: pose[1]},
                "embodiment_selection": CuRoboEmbodimentSelection.DEFAULT,
            }
            ignored = planning_ignore_objects(ignore_objects)
            if ignored is not None:
                plan_kwargs["ignore_objects"] = ignored
            trajectory = primitives._plan_joint_motion(**plan_kwargs)
            yield from primitives._execute_motion_plan(
                trajectory,
                low_precision=low_precision,
                stop_on_contact=stop_on_contact,
            )

        def move_sticky_attached_pose(
            pose: tuple[Any, Any],
            held_obj: Any,
            *,
            tolerate_joint_tail: bool = False,
        ) -> Generator[Any, None, None]:
            """Plan held-object motion without asking CuRobo to mesh the object.

            The sticky constraint and assisted-grasp bookkeeping must remain active so
            the object follows the hand and can still be verified. Only hide the object
            from Starter's planner query while it builds this arm-only trajectory.
            """
            objects_in_hand = getattr(self._robot, "_ag_obj_in_hand", None)
            if (
                not isinstance(objects_in_hand, dict)
                or objects_in_hand.get(self._arm) is not held_obj
            ):
                raise RuntimeError(
                    "sticky motion requires the expected attached object"
                )

            eef_name = self._robot.eef_link_names[self._arm]
            planner_get_obj_in_hand = primitives._get_obj_in_hand
            try:
                primitives._get_obj_in_hand = lambda: None
                trajectory = primitives._plan_joint_motion(
                    target_pos={eef_name: pose[0]},
                    target_quat={eef_name: pose[1]},
                    embodiment_selection=CuRoboEmbodimentSelection.ARM,
                    ignore_objects=planning_ignore_objects([held_obj]),
                )
            finally:
                primitives._get_obj_in_hand = planner_get_obj_in_hand

            if primitives._get_obj_in_hand() is not held_obj:
                raise RuntimeError("sticky attachment changed during motion planning")
            execute_kwargs: dict[str, Any] = {}
            if tolerate_joint_tail:
                try:
                    execute_parameters = inspect.signature(
                        primitives._execute_motion_plan
                    ).parameters
                except (TypeError, ValueError):
                    execute_parameters = {}
                if "ignore_failure" in execute_parameters:
                    execute_kwargs["ignore_failure"] = True
            yield from primitives._execute_motion_plan(trajectory, **execute_kwargs)

        def target_pose_array() -> tuple[np.ndarray, np.ndarray] | None:
            if target_obj is None:
                return None
            get_target_pose = getattr(target_obj, "get_position_orientation", None)
            if not callable(get_target_pose):
                return None
            position, orientation = get_target_pose()
            return (
                np.asarray(position, dtype=np.float64).reshape(3),
                np.asarray(orientation, dtype=np.float64).reshape(4),
            )

        def target_position_array() -> np.ndarray | None:
            pose = target_pose_array()
            return None if pose is None else pose[0]

        initial_target_position = target_position_array()

        def target_position() -> list[float] | None:
            position = target_position_array()
            return None if position is None else np.round(position, 4).tolist()

        def target_has_moved(
            tolerance_m: float = _TARGET_DISPLACEMENT_TOLERANCE_M,
        ) -> bool:
            position = target_position_array()
            return bool(
                initial_target_position is not None
                and position is not None
                and np.linalg.norm(position - initial_target_position) > tolerance_m
            )

        def attachment_evidence() -> dict[str, Any]:
            mode = getattr(self._robot, "grasping_mode", None)
            required = self._verification_require_attachment_valid and mode in {
                "assisted",
                "sticky",
            }
            constraints = getattr(self._robot, "_ag_obj_constraints", None)
            attachment = (
                constraints.get(self._arm) if isinstance(constraints, dict) else None
            )
            present = attachment is not None
            valid: bool | None = None
            enabled: bool | None = None
            path: str | None = None
            if attachment is not None:
                is_valid = getattr(attachment, "IsValid", None)
                valid = bool(is_valid()) if callable(is_valid) else bool(attachment)
                get_path = getattr(attachment, "GetPath", None)
                if callable(get_path):
                    path = str(get_path())
                get_attribute = getattr(attachment, "GetAttribute", None)
                if callable(get_attribute):
                    enabled_attribute = get_attribute("physics:jointEnabled")
                    attribute_valid = getattr(enabled_attribute, "IsValid", None)
                    if not callable(attribute_valid) or attribute_valid():
                        get_enabled = getattr(enabled_attribute, "Get", None)
                        if callable(get_enabled):
                            value = get_enabled()
                            enabled = None if value is None else bool(value)
            if mode == "physical":
                passed = not present
            else:
                passed = not required or (
                    present and valid is True and enabled is not False
                )
            return {
                "attachment_required": required,
                "attachment_expected_absent": mode == "physical",
                "attachment_present": present,
                "attachment_valid": valid,
                "attachment_enabled": enabled,
                "attachment_path": path,
                "attachment_passed": passed,
            }

        def to_numpy(value: Any) -> np.ndarray:
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            if hasattr(value, "numpy"):
                value = value.numpy()
            return np.asarray(value)

        def physical_contact_evidence() -> dict[str, Any]:
            def prim_path(value: Any) -> str:
                return str(getattr(value, "prim_path", value))

            expected_name = getattr(target_obj, "name", None)
            target_paths = {
                prim_path(path) for path in getattr(target_obj, "link_prim_paths", [])
            }
            target_prim_path = prim_path(getattr(target_obj, "prim_path", ""))
            finger_links = list(
                getattr(self._robot, "finger_links", {}).get(self._arm, [])
            )
            finger_paths = {prim_path(link) for link in finger_links}
            contact_positions: list[list[float]] = []
            target_finger_paths: set[str] = set()
            raw_contact_paths: list[str] = []
            raw_contact_finger_paths: dict[str, list[str]] = {}

            def belongs_to_target(path: Any) -> bool:
                path_str = prim_path(path)
                return path_str in target_paths or bool(
                    target_prim_path
                    and (
                        path_str == target_prim_path
                        or path_str.startswith(f"{target_prim_path}/")
                    )
                )

            find_contacts = getattr(self._robot, "_find_gripper_contacts", None)
            if callable(find_contacts):
                contacts, contact_links = find_contacts(
                    arm=self._arm,
                    return_contact_positions=True,
                )
                raw_contact_paths = sorted(
                    {prim_path(contact[0]) for contact in contacts}
                )
                raw_contact_finger_paths = {
                    prim_path(other_path): sorted(prim_path(link) for link in links)
                    for other_path, links in contact_links.items()
                }
                for other_path, links in contact_links.items():
                    if belongs_to_target(other_path):
                        target_finger_paths.update(prim_path(link) for link in links)
                for contact in contacts:
                    if len(contact) == 2 and belongs_to_target(contact[0]):
                        contact_positions.append(
                            to_numpy(contact[1]).astype(float).reshape(-1).tolist()
                        )

            is_grasping = getattr(self._robot, "is_grasping", None)
            grasp_state = (
                is_grasping(self._arm, candidate_obj=target_obj)
                if callable(is_grasping)
                else None
            )
            grasp_state_name = getattr(grasp_state, "name", str(grasp_state)).lower()
            grasp_state_passed = grasp_state_name == "true"

            qpos: list[float] = []
            gripper_indices = getattr(self._robot, "gripper_control_idx", {}).get(
                self._arm
            )
            if gripper_indices is not None:
                all_qpos = to_numpy(self._robot.get_joint_positions()).reshape(-1)
                indices = to_numpy(gripper_indices).astype(int).reshape(-1)
                qpos = all_qpos[indices].astype(float).tolist()

            target_position_now = target_position_array()
            eef_pose = self._robot.get_eef_pose(self._arm)
            eef_position = np.asarray(eef_pose[0], dtype=np.float64).reshape(3)
            eef_matrix = _pose_to_matrix(*eef_pose)
            world_to_eef = np.linalg.inv(eef_matrix)

            candidate_eef_origin_x = geometry.eef_origin_candidate_x(candidate.depth)

            def eef_to_candidate_points(points: np.ndarray) -> np.ndarray:
                points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
                return np.column_stack(
                    (
                        points[:, 2] + candidate_eef_origin_x,
                        points[:, 1],
                        -points[:, 0],
                    )
                )

            def world_to_candidate_points(points: np.ndarray) -> np.ndarray:
                points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
                homogeneous = np.column_stack((points, np.ones(len(points))))
                return eef_to_candidate_points((homogeneous @ world_to_eef.T)[:, :3])

            target_collision_points: list[np.ndarray] = []
            try:
                target_collision_points.append(
                    self.target_collision_boundary_points_world(target_obj)
                )
            except ValueError:
                pass
            target_boundary_points_world: np.ndarray | None = None
            target_collision_aabb: list[list[float]] | None = None
            target_candidate_points: np.ndarray | None = None
            target_candidate_aabb: list[list[float]] | None = None
            target_aabb_min: np.ndarray | None = None
            target_aabb_max: np.ndarray | None = None
            if target_collision_points:
                all_target_points = np.concatenate(target_collision_points, axis=0)
                target_boundary_points_world = all_target_points
                target_aabb_min = all_target_points.min(axis=0)
                target_aabb_max = all_target_points.max(axis=0)
                target_collision_aabb = [
                    target_aabb_min.tolist(),
                    target_aabb_max.tolist(),
                ]
                target_candidate_points = world_to_candidate_points(all_target_points)
                target_candidate_aabb = [
                    target_candidate_points.min(axis=0).tolist(),
                    target_candidate_points.max(axis=0).tolist(),
                ]

            finger_details: list[dict[str, Any]] = []
            finger_positions: list[np.ndarray] = []
            finger_candidate_bounds: list[np.ndarray] = []
            for link in finger_links:
                detail: dict[str, Any] = {
                    "name": str(getattr(link, "name", "")),
                    "prim_path": prim_path(link),
                }
                get_pose = getattr(link, "get_position_orientation", None)
                if callable(get_pose):
                    link_pose = get_pose()
                    link_position = np.asarray(link_pose[0], dtype=np.float64).reshape(
                        3
                    )
                    finger_positions.append(link_position)
                    detail["position"] = link_position.tolist()
                    detail["eef_local_position"] = (
                        world_to_eef @ np.append(link_position, 1.0)
                    )[:3].tolist()
                collision_points = getattr(
                    link, "collision_boundary_points_world", None
                )
                if collision_points is not None:
                    points = to_numpy(collision_points).astype(float).reshape(-1, 3)
                    if len(points):
                        finger_aabb_min = points.min(axis=0)
                        finger_aabb_max = points.max(axis=0)
                        local_points = (
                            np.column_stack((points, np.ones(len(points))))
                            @ world_to_eef.T
                        )[:, :3]
                        candidate_points = eef_to_candidate_points(local_points)
                        candidate_bounds = np.stack(
                            (candidate_points.min(axis=0), candidate_points.max(axis=0))
                        )
                        finger_candidate_bounds.append(candidate_bounds)
                        detail["collision_aabb"] = [
                            finger_aabb_min.tolist(),
                            finger_aabb_max.tolist(),
                        ]
                        detail["collision_eef_local_aabb"] = [
                            local_points.min(axis=0).tolist(),
                            local_points.max(axis=0).tolist(),
                        ]
                        detail["collision_candidate_local_aabb"] = (
                            candidate_bounds.tolist()
                        )
                        if target_boundary_points_world is not None:
                            target_to_finger = (
                                target_boundary_points_world[:, None, :]
                                - points[None, :, :]
                            )
                            detail["minimum_target_boundary_distance_m"] = float(
                                np.linalg.norm(target_to_finger, axis=2).min()
                            )
                        if target_position_now is not None:
                            detail["minimum_boundary_distance_to_target_m"] = float(
                                np.min(
                                    np.linalg.norm(
                                        points - target_position_now.reshape(1, 3),
                                        axis=1,
                                    )
                                )
                            )
                        if target_aabb_min is not None and target_aabb_max is not None:
                            separation = np.maximum(
                                np.maximum(
                                    target_aabb_min - finger_aabb_max,
                                    finger_aabb_min - target_aabb_max,
                                ),
                                0.0,
                            )
                            detail["target_collision_aabb_overlap"] = bool(
                                np.all(separation == 0.0)
                            )
                            detail["target_collision_aabb_separation_m"] = float(
                                np.linalg.norm(separation)
                            )
                finger_details.append(detail)

            def assisted_points(attribute: str) -> list[dict[str, Any]]:
                by_arm = getattr(self._robot, attribute, {})
                grasp_points = (
                    by_arm.get(self._arm, []) if isinstance(by_arm, dict) else []
                )
                robot_links = getattr(self._robot, "links", {})
                result: list[dict[str, Any]] = []
                for grasp_point in grasp_points or []:
                    link_name = str(getattr(grasp_point, "link_name", ""))
                    link = (
                        robot_links.get(link_name)
                        if isinstance(robot_links, dict)
                        else None
                    )
                    get_pose = getattr(link, "get_position_orientation", None)
                    if not callable(get_pose):
                        continue
                    local = (
                        to_numpy(getattr(grasp_point, "position"))
                        .astype(float)
                        .reshape(3)
                    )
                    world = _pose_to_matrix(*get_pose()) @ np.append(local, 1.0)
                    candidate_point = world_to_candidate_points(world[:3])[0]
                    result.append(
                        {
                            "link_name": link_name,
                            "candidate_local": candidate_point.tolist(),
                        }
                    )
                return result

            assisted_start = assisted_points("assisted_grasp_start_points")
            assisted_end = assisted_points("assisted_grasp_end_points")
            candidate_geometry = getattr(candidate, "target_geometry_evidence", {})
            preflight_open_jaw = (
                candidate_geometry.get("robot_open_jaw_containment", {})
                if isinstance(candidate_geometry, dict)
                else {}
            )
            open_jaw_margin_m = float(preflight_open_jaw.get("margin_m", 0.0))
            actual_inner_line_evidence = self.candidate_inner_grasp_line_evidence(
                candidate,
                (
                    target_candidate_points
                    if target_candidate_points is not None
                    else np.empty((0, 3), dtype=np.float64)
                ),
                margin_m=open_jaw_margin_m,
            )
            common_collision_bounds: list[list[float]] | None = None
            target_in_common_collision_count: int | None = None
            if len(finger_candidate_bounds) == 2:
                common_min = np.maximum(
                    finger_candidate_bounds[0][0], finger_candidate_bounds[1][0]
                )
                common_max = np.minimum(
                    finger_candidate_bounds[0][1], finger_candidate_bounds[1][1]
                )
                common_collision_bounds = [common_min.tolist(), common_max.tolist()]
                if target_candidate_points is not None and np.all(
                    common_min <= common_max
                ):
                    in_common = np.all(
                        (target_candidate_points >= common_min)
                        & (target_candidate_points <= common_max),
                        axis=1,
                    )
                    target_in_common_collision_count = int(in_common.sum())

            current_inner_y_interval: list[float] | None = None
            target_current_y_bounds: list[float] | None = None
            target_between_current_inner_surfaces: bool | None = None
            if len(finger_candidate_bounds) == 2:
                ordered_bounds = sorted(
                    finger_candidate_bounds,
                    key=lambda bounds: float(np.mean(bounds[:, 1])),
                )
                current_inner_y_interval = [
                    float(ordered_bounds[1][0, 1]),
                    float(ordered_bounds[0][1, 1]),
                ]
                if target_candidate_points is not None and len(target_candidate_points):
                    target_current_y_bounds = [
                        float(target_candidate_points[:, 1].min()),
                        float(target_candidate_points[:, 1].max()),
                    ]
                    target_between_current_inner_surfaces = bool(
                        target_current_y_bounds[0] >= current_inner_y_interval[0]
                        and target_current_y_bounds[1] <= current_inner_y_interval[1]
                    )

            jaw_link_midpoint = (
                np.mean(finger_positions, axis=0) if finger_positions else None
            )
            contacted_finger_paths = sorted(
                target_finger_paths.intersection(finger_paths)
            )
            required_finger_contacts = min(2, len(finger_paths))
            bilateral_contact = bool(
                required_finger_contacts == 2
                and len(contacted_finger_paths) >= required_finger_contacts
            )
            candidate_frame_geometry = {
                "candidate_depth_m": float(candidate.depth),
                "eef_fingertip_depth_m": geometry.fingertip_depth_m,
                "eef_origin_candidate_local_x_m": candidate_eef_origin_x,
                "target_collision_candidate_local_aabb": target_candidate_aabb,
                "common_finger_collision_candidate_local_aabb": (
                    common_collision_bounds
                ),
                "target_collision_points_in_common_finger_aabb": (
                    target_in_common_collision_count
                ),
                "target_collision_point_count": (
                    None
                    if target_candidate_points is None
                    else len(target_candidate_points)
                ),
                "current_finger_inner_surface_y_interval_m": current_inner_y_interval,
                "target_current_candidate_y_bounds_m": target_current_y_bounds,
                "target_between_current_inner_surfaces": (
                    target_between_current_inner_surfaces
                ),
                "assisted_grasp_start_points": assisted_start,
                "assisted_grasp_end_points": assisted_end,
                "actual_inner_grasp_line": actual_inner_line_evidence,
                "actual_open_jaw_containment": actual_inner_line_evidence,
            }
            return {
                "grasp_state": grasp_state_name,
                "grasp_state_passed": grasp_state_passed,
                "finger_inventory_valid": len(finger_paths) == 2,
                "finger_paths": sorted(finger_paths),
                "finger_details": finger_details,
                "candidate_frame_geometry": candidate_frame_geometry,
                "jaw_link_midpoint": (
                    None if jaw_link_midpoint is None else jaw_link_midpoint.tolist()
                ),
                "target_to_jaw_link_midpoint_m": (
                    None
                    if jaw_link_midpoint is None or target_position_now is None
                    else float(np.linalg.norm(target_position_now - jaw_link_midpoint))
                ),
                "eef_position": eef_position.tolist(),
                "target_position": (
                    None
                    if target_position_now is None
                    else target_position_now.tolist()
                ),
                "target_collision_aabb": target_collision_aabb,
                "raw_contact_paths": raw_contact_paths,
                "raw_contact_finger_paths": raw_contact_finger_paths,
                "target_contacted_finger_paths": contacted_finger_paths,
                "target_finger_contact_count": len(contacted_finger_paths),
                "required_target_finger_contact_count": required_finger_contacts,
                "bilateral_finger_contact": bilateral_contact,
                "target_between_current_inner_surfaces": (
                    target_between_current_inner_surfaces
                ),
                "target_current_candidate_y_bounds_m": target_current_y_bounds,
                "current_finger_inner_surface_y_interval_m": current_inner_y_interval,
                "minimum_target_boundary_distance_m": (
                    None
                    if not finger_details
                    else min(
                        (
                            detail["minimum_target_boundary_distance_m"]
                            for detail in finger_details
                            if "minimum_target_boundary_distance_m" in detail
                        ),
                        default=None,
                    )
                ),
                "target_contact_positions": contact_positions,
                "gripper_qpos": qpos,
                "expected_object": expected_name,
            }

        def staged_physical_close(
            after_approach_evidence: dict[str, Any],
            target_origin: np.ndarray | None,
        ) -> Generator[Any, None, dict[str, Any]]:
            geometry_evidence = after_approach_evidence.get(
                "candidate_frame_geometry", {}
            ).get("actual_open_jaw_containment", {})
            open_gap_m = geometry_evidence.get("open_jaw_gap_m")
            target_y_bounds_m = geometry_evidence.get(
                "open_jaw_continuous_cross_section_y_bounds_m"
            )
            open_qpos = after_approach_evidence.get("gripper_qpos")
            gripper_indices = getattr(self._robot, "gripper_control_idx", {}).get(
                self._arm
            )
            target_builder = getattr(
                primitives, "_get_joint_position_with_fingers_at_limit", None
            )
            if (
                not callable(target_builder)
                or gripper_indices is None
                or not open_qpos
                or open_gap_m is None
                or target_y_bounds_m is None
            ):
                raise RuntimeError(
                    "staged physical close requires gripper limits, qpos, and "
                    "measured open-jaw target geometry"
                )

            lower_joint_positions = to_numpy(target_builder("lower")).reshape(-1)
            index_values = to_numpy(gripper_indices).astype(int).reshape(-1)
            lower_qpos = lower_joint_positions[index_values]
            plan = self.physical_staged_close_plan(
                open_qpos=open_qpos,
                lower_qpos=lower_qpos,
                open_gap_m=float(open_gap_m),
                target_y_bounds_m=target_y_bounds_m,
                compression_m=self._physical_close_compression_m,
                stage_count=self._physical_close_stage_count,
            )
            plan.update(
                {
                    "enabled": True,
                    "hold_steps": self._physical_close_hold_steps,
                    "stage_displacement_tolerance_m": (
                        self._physical_close_stage_displacement_tolerance_m
                    ),
                    "unilateral_contact_displacement_tolerance_m": (
                        self._physical_unilateral_contact_displacement_tolerance_m
                    ),
                    "completed_stages": 0,
                    "stopped_on_contact": False,
                    "stage_samples": [],
                }
            )
            logger.warning("AnyGrasp staged physical close plan=%s", plan)

            action_idx = self._robot.controller_action_idx[f"gripper_{self._arm}"]

            def evaluate(stage_index: int, sample_kind: str) -> bool:
                evidence = physical_contact_evidence()
                target_now = target_position_array()
                displacement = (
                    None
                    if target_origin is None or target_now is None
                    else float(np.linalg.norm(target_now - target_origin))
                )
                sample = {
                    "stage": stage_index,
                    "sample_kind": sample_kind,
                    "gripper_qpos": evidence.get("gripper_qpos"),
                    "bilateral_finger_contact": evidence.get(
                        "bilateral_finger_contact", False
                    ),
                    "target_finger_contact_count": evidence.get(
                        "target_finger_contact_count", 0
                    ),
                    "grasp_state_passed": evidence.get("grasp_state_passed", False),
                    "target_displacement_m": displacement,
                }
                plan["stage_samples"].append(sample)
                contacted = self.physical_staged_close_should_stop(
                    evidence,
                    target_displacement_m=displacement,
                    displacement_tolerance_m=(
                        self._physical_close_stage_displacement_tolerance_m
                    ),
                    stage_index=stage_index,
                    unilateral_contact_displacement_tolerance_m=(
                        self._physical_unilateral_contact_displacement_tolerance_m
                    ),
                )
                if contacted:
                    plan["stopped_on_contact"] = True
                    plan["contact_stage"] = stage_index
                    plan["contact_evidence"] = evidence
                return contacted

            for stage_index, stage_qpos_values in enumerate(
                plan["stage_qpos"], start=1
            ):
                stage_qpos = np.asarray(stage_qpos_values, dtype=np.float64)
                for _ in range(80):
                    all_qpos = to_numpy(self._robot.get_joint_positions()).reshape(-1)
                    current_qpos = all_qpos[index_values]
                    error = stage_qpos - current_qpos
                    if np.max(np.abs(error)) <= 0.003:
                        break
                    command = 1.0 if float(np.mean(error)) >= 0.0 else -1.0
                    action = primitives._empty_action(follow_arm_targets=False)
                    action[action_idx] = command
                    yield primitives._postprocess_action(action)
                    if evaluate(stage_index, "motion"):
                        plan["completed_stages"] = stage_index
                        return plan
                else:
                    raise RuntimeError(
                        "gripper failed to reach staged close target "
                        f"{stage_index}: target={stage_qpos.tolist()}"
                    )

                plan["completed_stages"] = stage_index
                for _ in range(self._physical_close_hold_steps):
                    action = primitives._empty_action(follow_arm_targets=False)
                    yield primitives._postprocess_action(action)
                    if evaluate(stage_index, "hold"):
                        return plan
            plan["final_evidence"] = physical_contact_evidence()
            return plan

        def verify_physical_grasp() -> Generator[Any, None, dict[str, Any]]:
            nonlocal steps
            samples: list[dict[str, Any]] = []
            mode = getattr(self._robot, "grasping_mode", None)
            physical_mode = mode == "physical"
            if last_action is not None:
                for _ in range(self._verification_steps):
                    steps += 1
                    yield last_action
                    target_pose = target_pose_array()
                    eef_pose = self._robot.get_eef_pose(self._arm)
                    if target_pose is None:
                        continue
                    target_matrix = _pose_to_matrix(*target_pose)
                    eef_matrix = _pose_to_matrix(*eef_pose)
                    relative_matrix = np.linalg.inv(eef_matrix) @ target_matrix
                    sample = {
                        "target_position": target_pose[0].tolist(),
                        "eef_position": np.asarray(eef_pose[0], dtype=float).tolist(),
                        "relative_position": relative_matrix[:3, 3].tolist(),
                        "relative_rotation": relative_matrix[:3, :3].tolist(),
                    }
                    if physical_mode:
                        sample["physical_contact"] = physical_contact_evidence()
                    samples.append(sample)

            sample_count_ok = len(samples) == self._verification_steps
            rises = (
                [
                    sample["target_position"][2] - initial_target_position[2]
                    for sample in samples
                ]
                if initial_target_position is not None
                else []
            )
            minimum_rise = float(min(rises)) if rises else None
            final_rise = float(rises[-1]) if rises else None
            rise_passed = bool(
                minimum_rise is not None
                and minimum_rise >= self._verification_min_target_z_rise_m
            )

            relative_position_drift = None
            relative_orientation_drift_deg = None
            if samples:
                reference_position = np.asarray(
                    samples[0]["relative_position"], dtype=float
                )
                reference_rotation = np.asarray(
                    samples[0]["relative_rotation"], dtype=float
                )
                relative_position_drift = max(
                    float(
                        np.linalg.norm(
                            np.asarray(sample["relative_position"], dtype=float)
                            - reference_position
                        )
                    )
                    for sample in samples
                )
                orientation_drifts = []
                for sample in samples:
                    rotation = np.asarray(sample["relative_rotation"], dtype=float)
                    cosine = (np.trace(reference_rotation.T @ rotation) - 1.0) / 2.0
                    orientation_drifts.append(
                        math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
                    )
                relative_orientation_drift_deg = max(orientation_drifts)
            relative_pose_stable = bool(
                relative_position_drift is not None
                and relative_position_drift
                <= self._verification_relative_offset_tolerance_m
                and relative_orientation_drift_deg is not None
                and relative_orientation_drift_deg
                <= self._verification_relative_orientation_tolerance_deg
            )

            lift_start = phase_contact_evidence.get("after_settle", {})
            lift_end = phase_contact_evidence.get("after_lift", {})
            lift_target_start = lift_start.get("target_position")
            lift_target_end = lift_end.get("target_position")
            lift_eef_start = lift_start.get("eef_position")
            lift_eef_end = lift_end.get("eef_position")
            lift_target_delta: np.ndarray | None = None
            lift_eef_delta: np.ndarray | None = None
            lift_motion_error_m: float | None = None
            lift_motion_passed: bool | None = None
            if all(
                value is not None
                for value in (
                    lift_target_start,
                    lift_target_end,
                    lift_eef_start,
                    lift_eef_end,
                )
            ):
                lift_target_delta = np.asarray(lift_target_end) - np.asarray(
                    lift_target_start
                )
                lift_eef_delta = np.asarray(lift_eef_end) - np.asarray(lift_eef_start)
                lift_motion_error_m = float(
                    np.linalg.norm(lift_target_delta - lift_eef_delta)
                )
                lift_motion_passed = bool(
                    lift_target_delta[2] >= self._verification_min_target_z_rise_m
                    and lift_eef_delta[2] >= self._verification_min_target_z_rise_m
                    and lift_motion_error_m
                    <= self._verification_relative_offset_tolerance_m
                )

            expected_name = getattr(target_obj, "name", None)
            if physical_mode:
                contact_samples = [
                    sample.get("physical_contact", {}) for sample in samples
                ]
                physical_contact_passed = bool(
                    sample_count_ok
                    and all(
                        sample.get("grasp_state_passed") is True
                        and sample.get("bilateral_finger_contact") is True
                        for sample in contact_samples
                    )
                )
                object_name = expected_name if physical_contact_passed else None
                object_identity_matches = bool(
                    expected_name and physical_contact_passed
                )
            else:
                contact_samples = []
                physical_contact_passed = None
                object_in_hand = primitives._get_obj_in_hand()
                object_name = getattr(object_in_hand, "name", None)
                object_identity_matches = bool(
                    object_in_hand is not None
                    and expected_name is not None
                    and object_name == expected_name
                )
            attachment = attachment_evidence()
            passed = bool(
                sample_count_ok
                and rise_passed
                and relative_pose_stable
                and object_identity_matches
                and attachment["attachment_passed"]
                and (not physical_mode or physical_contact_passed)
                and (not physical_mode or lift_motion_passed is True)
            )
            return {
                "passed": passed,
                "grasping_mode": mode,
                "sample_count": len(samples),
                "required_sample_count": self._verification_steps,
                "initial_target_position": (
                    None
                    if initial_target_position is None
                    else initial_target_position.tolist()
                ),
                "final_target_position": (
                    samples[-1]["target_position"] if samples else None
                ),
                "minimum_target_z_rise_m": minimum_rise,
                "final_target_z_rise_m": final_rise,
                "required_target_z_rise_m": self._verification_min_target_z_rise_m,
                "target_z_rise_passed": rise_passed,
                "max_relative_position_drift_m": relative_position_drift,
                "relative_position_tolerance_m": (
                    self._verification_relative_offset_tolerance_m
                ),
                "max_relative_orientation_drift_deg": relative_orientation_drift_deg,
                "relative_orientation_tolerance_deg": (
                    self._verification_relative_orientation_tolerance_deg
                ),
                "relative_pose_stable": relative_pose_stable,
                "lift_target_delta": (
                    None if lift_target_delta is None else lift_target_delta.tolist()
                ),
                "lift_eef_delta": (
                    None if lift_eef_delta is None else lift_eef_delta.tolist()
                ),
                "lift_motion_error_m": lift_motion_error_m,
                "lift_motion_passed": lift_motion_passed,
                "lift_started": lift_started,
                "lift_completed": lift_completed,
                "lift_joint_tail_tolerance_enabled": (
                    lift_joint_tail_tolerance_enabled
                ),
                "lift_success_decided_by_physical_verification": True,
                "phase_contact_evidence": phase_contact_evidence,
                "approach_segment_audit": approach_segment_audit,
                "physical_contact_required": physical_mode,
                "physical_contact_passed": physical_contact_passed,
                "physical_contact_samples": contact_samples,
                "object_in_hand": object_name,
                "expected_object": expected_name,
                "candidate_target_geometry": getattr(
                    candidate, "target_geometry_evidence", None
                ),
                "object_identity_matches": object_identity_matches,
                "samples": samples,
                **attachment,
            }

        approach_segment_audit: list[dict[str, Any]] = []

        def collision_aware_approach(
            *,
            motion_constraint: list[float] | None = None,
        ) -> Generator[Any, None, None]:
            move_kwargs: dict[str, Any] = {
                "ignore_objects": attached_ignore_objects,
            }
            if motion_constraint is not None:
                move_kwargs["motion_constraint"] = motion_constraint
            if self._approach_segment_max_m <= 0.0:
                yield from primitives._move_hand(grasp_pose, **move_kwargs)
                return

            approach_delta = np.asarray(grasp_eef_pos - pregrasp_pos, dtype=np.float64)
            approach_distance = float(np.linalg.norm(approach_delta))
            segment_count = max(
                1,
                int(np.ceil(approach_distance / self._approach_segment_max_m)),
            )
            for segment_index in range(1, segment_count + 1):
                fraction = segment_index / segment_count
                segment_pos = np.asarray(pregrasp_pos, dtype=np.float64) + (
                    approach_delta * fraction
                )
                segment_pose = (
                    th.as_tensor(segment_pos, dtype=th.float32),
                    grasp_quat_t.clone(),
                )
                audit = {
                    "segment_index": segment_index,
                    "segment_count": segment_count,
                    "fraction": fraction,
                    "planned_eef_position": segment_pos.tolist(),
                    "maximum_segment_length_m": self._approach_segment_max_m,
                    "scene_collision_enabled": True,
                    "ignored_target_only": not local_collision_ignores,
                    "status": "planning",
                }
                approach_segment_audit.append(audit)
                logger.warning(
                    "AnyGrasp CuRobo approach segment=%d/%d endpoint=%s",
                    segment_index,
                    segment_count,
                    np.round(segment_pos, 4).tolist(),
                )
                try:
                    yield from primitives._move_hand(segment_pose, **move_kwargs)
                except Exception as exc:
                    audit.update(
                        {
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    raise
                actual_eef_pos = np.asarray(
                    self._robot.get_eef_pose(self._arm)[0], dtype=np.float64
                )
                audit.update(
                    {
                        "status": "completed",
                        "actual_eef_position": actual_eef_pos.tolist(),
                        "endpoint_error_m": float(
                            np.linalg.norm(actual_eef_pos - segment_pos)
                        ),
                    }
                )

        def contact_approach_actions(
            generator: Any,
            target_guard_origin: np.ndarray | None,
        ) -> Generator[Any, None, None]:
            try:
                yield from actions(
                    generator,
                    target_guard_origin=target_guard_origin,
                    target_guard_tolerance_m=(
                        self._approach_target_displacement_tolerance_m
                    ),
                )
            except Exception as exc:
                current_eef_pos = np.asarray(
                    self._robot.get_eef_pose(self._arm)[0], dtype=float
                )
                goal_error = float(np.linalg.norm(current_eef_pos - grasp_eef_pos))
                target_position_now = target_position_array()
                target_displacement = (
                    float(np.linalg.norm(target_position_now - target_guard_origin))
                    if target_position_now is not None
                    and target_guard_origin is not None
                    else 0.0
                )
                if (
                    goal_error > self._approach_goal_position_tolerance_m
                    or target_displacement
                    > self._approach_target_displacement_tolerance_m
                ):
                    raise
                logger.warning(
                    "AnyGrasp accepting near-goal approach after %s: %s "
                    "(EEF position error=%.4f m, target displacement=%.4f m)",
                    type(exc).__name__,
                    exc,
                    goal_error,
                    target_displacement,
                )

        def log_state(label: str) -> None:
            current_eef = self._robot.get_eef_pose(self._arm)
            logger.warning(
                "AnyGrasp phase=%s eef_pos=%s base_pos=%s target_pos=%s",
                label,
                np.round(np.asarray(current_eef[0]), 4).tolist(),
                np.round(
                    np.asarray(self._robot.get_position_orientation()[0]), 4
                ).tolist(),
                target_position(),
            )

        logger.warning(
            "AnyGrasp geometry score=%.3f width=%.4f depth=%.4f camera_pos=%s "
            "grasp_origin=%s grasp_tip_pos=%s grasp_eef_pos=%s fingertip_depth=%.4f "
            "pregrasp_pos=%s standoff_pos=%s approach=%s target_geometry=%s",
            float(candidate.score),
            float(candidate.width),
            float(candidate.depth),
            np.round(np.asarray(candidate.translation), 4).tolist(),
            np.round(grasp_pos, 4).tolist(),
            np.round(grasp_tip_pos, 4).tolist(),
            np.round(grasp_eef_pos, 4).tolist(),
            geometry.fingertip_depth_m,
            np.round(pregrasp_pos, 4).tolist(),
            np.round(standoff_pos, 4).tolist(),
            np.round(approach, 4).tolist(),
            getattr(candidate, "target_geometry_evidence", None),
        )
        log_state("initial")

        physical_evidence: dict[str, Any] = {}
        place_back_evidence: dict[str, Any] = {}
        phase_contact_evidence: dict[str, dict[str, Any]] = {}
        scene_changing_phases = {
            "close_before_sticky_approach",
            "sticky_approach",
            "close_gripper",
            "settle_after_grasp",
            "lift",
            "physical_verification",
            "place_back_lower",
            "place_back_release",
            "place_back_retreat",
        }
        contact_approach_phases = {
            "assisted_approach_constrained",
            "assisted_approach_unconstrained",
            "post_approach_pre_close_check",
            "live_open_jaw_y_correction",
        }
        phase = "release"
        try:
            yield from actions(primitives._execute_release())
            current_eef_pos = np.asarray(
                self._robot.get_eef_pose(self._arm)[0], dtype=float
            )
            pregrasp_distance = float(np.linalg.norm(current_eef_pos - pregrasp_pos))
            if pregrasp_distance > self._skip_standoff_if_within_m:
                phase = "whole_body_standoff"
                yield from actions(
                    move_to_pose_whole_body(standoff_pose, low_precision=True)
                )
                log_state("after_whole_body_standoff")
                if target_has_moved():
                    raise RuntimeError(
                        "target moved during collision-aware whole-body standoff"
                    )
            else:
                logger.warning(
                    "AnyGrasp skipping whole-body standoff: EEF is %.3f m from pregrasp "
                    "(threshold %.3f m)",
                    pregrasp_distance,
                    self._skip_standoff_if_within_m,
                )

            phase = "precise_pregrasp"
            pregrasp_target_origin = target_position_array()
            yield from actions(
                move_to_pose_whole_body(
                    pregrasp_pose,
                    ignore_objects=attached_ignore_objects,
                    stop_on_contact=True,
                ),
                target_guard_origin=pregrasp_target_origin,
            )
            log_state("after_precise_pregrasp")
            if target_has_moved():
                raise RuntimeError(
                    "target moved during contact-guarded precise pregrasp"
                )

            if getattr(self._robot, "grasping_mode", None) == "sticky":
                phase = "close_before_sticky_approach"
                yield from actions(primitives._execute_grasp())
                phase = "sticky_approach"
                yield from actions(
                    primitives._move_hand(
                        grasp_pose,
                        motion_constraint=(
                            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
                            if self._constrained_approach
                            else None
                        ),
                        stop_on_ag=True,
                        ignore_objects=attached_ignore_objects,
                    )
                )
            else:
                approach_target_origin = target_position_array()
                approach_tolerance = self._approach_target_displacement_tolerance_m
                if self._constrained_approach:
                    phase = "assisted_approach_constrained"
                    try:
                        yield from contact_approach_actions(
                            collision_aware_approach(
                                motion_constraint=[
                                    1.0,
                                    1.0,
                                    1.0,
                                    1.0,
                                    1.0,
                                    0.0,
                                ],
                            ),
                            approach_target_origin,
                        )
                    except Exception as constrained_error:
                        if not self._retry_unconstrained_approach or target_has_moved(
                            approach_tolerance
                        ):
                            raise
                        logger.warning(
                            "AnyGrasp constrained approach failed (%s: %s); "
                            "retrying collision-aware unconstrained approach",
                            type(constrained_error).__name__,
                            constrained_error,
                        )
                        phase = "assisted_approach_unconstrained"
                        yield from contact_approach_actions(
                            collision_aware_approach(),
                            approach_target_origin,
                        )
                else:
                    phase = "assisted_approach_unconstrained"
                    yield from contact_approach_actions(
                        collision_aware_approach(),
                        approach_target_origin,
                    )
                if getattr(self._robot, "grasping_mode", None) == "physical":
                    phase = "post_approach_pre_close_check"

                    def annotate_approach_evidence(
                        evidence: dict[str, Any],
                        expected_eef_position: np.ndarray,
                    ) -> tuple[float | None, float]:
                        target_now = target_position_array()
                        displacement = (
                            None
                            if target_now is None or approach_target_origin is None
                            else float(
                                np.linalg.norm(target_now - approach_target_origin)
                            )
                        )
                        current_position = np.asarray(
                            self._robot.get_eef_pose(self._arm)[0], dtype=float
                        )
                        goal_error = float(
                            np.linalg.norm(current_position - expected_eef_position)
                        )
                        frame_geometry = evidence.get("candidate_frame_geometry", {})
                        open_jaw = frame_geometry.get("actual_open_jaw_containment", {})
                        evidence.update(
                            {
                                "approach_baseline_target_position": (
                                    None
                                    if approach_target_origin is None
                                    else approach_target_origin.tolist()
                                ),
                                "approach_target_displacement_m": displacement,
                                "approach_target_displacement_tolerance_m": (
                                    approach_tolerance
                                ),
                                "approach_target_displacement_passed": bool(
                                    displacement is not None
                                    and displacement <= approach_tolerance
                                ),
                                "approach_expected_eef_position": (
                                    expected_eef_position.tolist()
                                ),
                                "approach_goal_error_m": goal_error,
                                "approach_goal_position_tolerance_m": (
                                    self._approach_goal_position_tolerance_m
                                ),
                                "approach_goal_passed": bool(
                                    goal_error
                                    <= self._approach_goal_position_tolerance_m
                                ),
                                "open_jaw_containment_passed": bool(
                                    open_jaw.get("available", False)
                                    and open_jaw.get(
                                        "open_jaw_continuous_cross_section_intersects",
                                        False,
                                    )
                                    and open_jaw.get(
                                        "target_between_open_fingers", False
                                    )
                                ),
                            }
                        )
                        return displacement, goal_error

                    after_approach = physical_contact_evidence()
                    approach_displacement, approach_goal_error = (
                        annotate_approach_evidence(
                            after_approach,
                            grasp_eef_pos,
                        )
                    )
                    phase_contact_evidence["after_approach"] = after_approach
                    logger.warning(
                        "AnyGrasp physical contact phase=after_approach evidence=%s",
                        after_approach,
                    )
                    if approach_displacement is None:
                        raise RuntimeError(
                            "target pose unavailable after physical approach"
                        )
                    if approach_displacement > approach_tolerance:
                        raise RuntimeError(
                            "target moved during physical approach by "
                            f"{approach_displacement:.4f} m"
                        )
                    if approach_goal_error > self._approach_goal_position_tolerance_m:
                        raise RuntimeError(
                            "physical approach did not reach grasp endpoint; EEF "
                            f"position error={approach_goal_error:.4f} m"
                        )

                    if (
                        not after_approach["open_jaw_containment_passed"]
                        and self._live_open_jaw_y_correction_max_m > 0.0
                    ):
                        open_jaw = after_approach.get(
                            "candidate_frame_geometry", {}
                        ).get("actual_open_jaw_containment", {})
                        target_y_bounds = open_jaw.get(
                            "open_jaw_continuous_cross_section_y_bounds_m"
                        )
                        open_y_interval = open_jaw.get(
                            "open_jaw_inner_surface_y_interval_m"
                        )
                        correction_audit: dict[str, Any] = {
                            "enabled": True,
                            "axis": "candidate_y_only",
                            "x_correction_m": 0.0,
                            "maximum_correction_m": (
                                self._live_open_jaw_y_correction_max_m
                            ),
                            "attempted": False,
                            "applied": False,
                        }
                        if (
                            open_jaw.get(
                                "open_jaw_continuous_cross_section_intersects",
                                False,
                            )
                            and isinstance(target_y_bounds, (list, tuple))
                            and len(target_y_bounds) == 2
                            and isinstance(open_y_interval, (list, tuple))
                            and len(open_y_interval) == 2
                        ):
                            target_y = np.asarray(target_y_bounds, dtype=np.float64)
                            open_y = np.asarray(open_y_interval, dtype=np.float64)
                            target_span = float(target_y[1] - target_y[0])
                            open_span = float(open_y[1] - open_y[0])
                            correction_y = float(target_y.mean() - open_y.mean())
                            fits_open_span = bool(target_span <= open_span + 1e-8)
                            within_limit = bool(
                                abs(correction_y)
                                <= self._live_open_jaw_y_correction_max_m
                            )
                            correction_audit.update(
                                {
                                    "attempted": True,
                                    "target_y_bounds_m": target_y.tolist(),
                                    "open_y_interval_m": open_y.tolist(),
                                    "target_span_y_m": target_span,
                                    "open_span_y_m": open_span,
                                    "correction_local_y_m": correction_y,
                                    "fits_open_span": fits_open_span,
                                    "within_correction_limit": within_limit,
                                }
                            )
                            if fits_open_span and within_limit:
                                current_eef_pose = self._robot.get_eef_pose(self._arm)
                                current_eef_matrix = _pose_to_matrix(*current_eef_pose)
                                correction_local = np.array(
                                    [0.0, correction_y, 0.0], dtype=np.float64
                                )
                                correction_world = (
                                    current_eef_matrix[:3, :3] @ correction_local
                                )
                                corrected_eef_pos = (
                                    np.asarray(current_eef_pose[0], dtype=np.float64)
                                    + correction_world
                                )
                                correction_audit.update(
                                    {
                                        "correction_local_m": correction_local.tolist(),
                                        "correction_world_m": correction_world.tolist(),
                                        "corrected_eef_position": (
                                            corrected_eef_pos.tolist()
                                        ),
                                    }
                                )
                                phase = "live_open_jaw_y_correction"
                                corrected_pose = (
                                    th.as_tensor(corrected_eef_pos, dtype=th.float32),
                                    th.as_tensor(
                                        current_eef_pose[1], dtype=th.float32
                                    ).clone(),
                                )
                                yield from actions(
                                    primitives._move_hand(
                                        corrected_pose,
                                        ignore_objects=attached_ignore_objects,
                                    ),
                                    target_guard_origin=approach_target_origin,
                                    target_guard_tolerance_m=approach_tolerance,
                                )
                                phase = "post_approach_pre_close_check"
                                corrected_evidence = physical_contact_evidence()
                                (
                                    correction_displacement,
                                    correction_goal_error,
                                ) = annotate_approach_evidence(
                                    corrected_evidence,
                                    corrected_eef_pos,
                                )
                                correction_audit.update(
                                    {
                                        "applied": True,
                                        "correction_goal_error_m": (
                                            correction_goal_error
                                        ),
                                        "target_displacement_m": (
                                            correction_displacement
                                        ),
                                    }
                                )
                                corrected_evidence["live_open_jaw_y_correction"] = (
                                    correction_audit
                                )
                                phase_contact_evidence[
                                    "after_live_open_jaw_y_correction"
                                ] = corrected_evidence
                                after_approach = corrected_evidence
                                phase_contact_evidence["after_approach"] = (
                                    after_approach
                                )
                                approach_displacement = correction_displacement
                                approach_goal_error = correction_goal_error
                                logger.warning(
                                    "AnyGrasp physical contact "
                                    "phase=after_live_open_jaw_y_correction "
                                    "evidence=%s",
                                    corrected_evidence,
                                )
                        after_approach.setdefault(
                            "live_open_jaw_y_correction", correction_audit
                        )

                    if approach_displacement is None:
                        raise RuntimeError(
                            "target pose unavailable after live open-jaw correction"
                        )
                    if approach_displacement > approach_tolerance:
                        raise RuntimeError(
                            "target moved during live open-jaw correction by "
                            f"{approach_displacement:.4f} m"
                        )
                    if approach_goal_error > self._approach_goal_position_tolerance_m:
                        raise RuntimeError(
                            "live open-jaw correction did not reach its endpoint; EEF "
                            f"position error={approach_goal_error:.4f} m"
                        )
                    if not after_approach["open_jaw_containment_passed"]:
                        raise RuntimeError(
                            "target collision cross-section is not contained between "
                            "the actual open finger inner surfaces"
                        )
                phase = "close_gripper"
                close_target_origin = target_position_array()
                if (
                    getattr(self._robot, "grasping_mode", None) == "physical"
                    and self._physical_staged_close_enabled
                ):
                    staged_close_audit = yield from actions(
                        staged_physical_close(after_approach, close_target_origin),
                        target_guard_origin=close_target_origin,
                        target_guard_tolerance_m=(
                            self._physical_close_stage_displacement_tolerance_m
                        ),
                    )
                    phase_contact_evidence["staged_close"] = staged_close_audit
                    logger.warning(
                        "AnyGrasp physical contact phase=staged_close evidence=%s",
                        staged_close_audit,
                    )
                else:
                    yield from actions(
                        primitives._execute_grasp(),
                        target_guard_origin=close_target_origin,
                        target_guard_tolerance_m=(
                            self._close_target_displacement_tolerance_m
                        ),
                    )
            log_state("after_grasp_close")
            if getattr(self._robot, "grasping_mode", None) == "physical":
                phase_contact_evidence["after_grasp_close"] = (
                    physical_contact_evidence()
                )
                logger.warning(
                    "AnyGrasp physical contact phase=after_grasp_close evidence=%s",
                    phase_contact_evidence["after_grasp_close"],
                )

            phase = "settle_after_grasp"
            yield from actions(
                primitives._settle_robot(),
                target_guard_origin=close_target_origin,
                target_guard_tolerance_m=self._close_target_displacement_tolerance_m,
            )
            if close_target_origin is not None:
                target_after_close = target_position_array()
                close_displacement = (
                    None
                    if target_after_close is None
                    else float(np.linalg.norm(target_after_close - close_target_origin))
                )
                if (
                    close_displacement is not None
                    and close_displacement > self._close_target_displacement_tolerance_m
                ):
                    raise RuntimeError(
                        f"target moved during gripper close/settle by {close_displacement:.4f} m"
                    )
            if getattr(self._robot, "grasping_mode", None) == "physical":
                phase_contact_evidence["after_settle"] = physical_contact_evidence()
                logger.warning(
                    "AnyGrasp physical contact phase=after_settle evidence=%s",
                    phase_contact_evidence["after_settle"],
                )
                if (
                    self._physical_require_bilateral_contact_before_lift
                    and not phase_contact_evidence["after_settle"].get(
                        "bilateral_finger_contact", False
                    )
                ):
                    raise RuntimeError(
                        "physical grasp has no bilateral finger contact after close; "
                        "skipping empty lift and trying another candidate"
                    )

            if getattr(self._robot, "grasping_mode", None) == "sticky":
                object_in_hand = primitives._get_obj_in_hand()
                object_name = getattr(object_in_hand, "name", None)
                expected_name = getattr(target_obj, "name", None)
                logger.warning("AnyGrasp object after sticky approach=%s", object_name)
                if object_in_hand is None or (
                    expected_name is not None and object_name != expected_name
                ):
                    raise RuntimeError(
                        f"sticky approach attached {object_name}, expected {expected_name}"
                    )

            lift_pos = grasp_eef_pos.copy()
            lift_pos[2] += self._lift_height_m
            lift_pose = (th.as_tensor(lift_pos, dtype=th.float32), grasp_quat_t.clone())
            phase = "lift"
            lift_started = True
            if getattr(self._robot, "grasping_mode", None) == "sticky":
                logger.warning(
                    "AnyGrasp executing sticky lift without attached-mesh collision planning"
                )
                try:
                    execute_parameters = inspect.signature(
                        primitives._execute_motion_plan
                    ).parameters
                except (TypeError, ValueError):
                    execute_parameters = {}
                lift_joint_tail_tolerance_enabled = (
                    "ignore_failure" in execute_parameters
                )
                yield from actions(
                    move_sticky_attached_pose(
                        lift_pose,
                        object_in_hand,
                        tolerate_joint_tail=True,
                    )
                )
            else:
                lift_kwargs: dict[str, Any] = {}
                # At lift time the target is already enclosed by / touching the
                # fingers.  Keeping the same object in CuRobo's world obstacles
                # makes the current state collision-invalid, which prevents the
                # planner from finding even a straight upward retreat in physical
                # mode.  Assisted mode represents it separately as an attached
                # object, so it must not also remain as a duplicate world obstacle.
                lift_ignores = attached_ignore_objects
                if lift_ignores is not None:
                    lift_kwargs["ignore_objects"] = lift_ignores
                yield from actions(primitives._move_hand(lift_pose, **lift_kwargs))
            lift_completed = True
            log_state("after_lift")
            if getattr(self._robot, "grasping_mode", None) == "physical":
                phase_contact_evidence["after_lift"] = physical_contact_evidence()
                logger.warning(
                    "AnyGrasp physical contact phase=after_lift evidence=%s",
                    phase_contact_evidence["after_lift"],
                )
            yaw_sequence = self.post_lift_yaw_sequence(
                self._post_lift_yaw_deg,
                self._post_lift_yaw_cycles,
            )
            if yaw_sequence:
                phase = "post_lift_stability_test"
                base_rotation = _pose_to_matrix(
                    np.zeros(3, dtype=np.float32),
                    grasp_quat_t.detach().cpu().numpy(),
                )[:3, :3]
                for index, yaw_deg in enumerate(yaw_sequence, start=1):
                    yaw_rad = math.radians(yaw_deg)
                    world_yaw = np.array(
                        [
                            [math.cos(yaw_rad), -math.sin(yaw_rad), 0.0],
                            [math.sin(yaw_rad), math.cos(yaw_rad), 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                        dtype=np.float64,
                    )
                    yaw_quat = th.as_tensor(
                        _mat_to_quat_xyzw(world_yaw @ base_rotation),
                        dtype=th.float32,
                    )
                    yaw_pose = (lift_pose[0].clone(), yaw_quat)
                    logger.warning(
                        "AnyGrasp post-lift stability yaw step=%d/%d yaw_deg=%.1f",
                        index,
                        len(yaw_sequence),
                        yaw_deg,
                    )
                    if getattr(self._robot, "grasping_mode", None) == "sticky":
                        yield from actions(
                            move_sticky_attached_pose(yaw_pose, object_in_hand)
                        )
                    else:
                        yaw_kwargs: dict[str, Any] = {}
                        if attached_ignore_objects is not None:
                            yaw_kwargs["ignore_objects"] = attached_ignore_objects
                        yield from actions(
                            primitives._move_hand(yaw_pose, **yaw_kwargs)
                        )
                    log_state(f"after_stability_yaw_{index}")
            phase = "physical_verification"
            physical_evidence = yield from verify_physical_grasp()
            logger.warning(
                "AnyGrasp physical verification evidence=%s", physical_evidence
            )
            if self._post_lift_place_back:
                phase = "place_back_lower"
                place_pos = np.asarray(grasp_eef_pos, dtype=np.float32).copy()
                place_pos[2] += self._place_back_clearance_m
                place_pose = (
                    th.as_tensor(place_pos, dtype=th.float32),
                    grasp_quat_t.clone(),
                )
                logger.warning(
                    "AnyGrasp lowering grasped object for place-back eef_pos=%s",
                    np.round(place_pos, 4).tolist(),
                )
                if getattr(self._robot, "grasping_mode", None) == "sticky":
                    yield from actions(
                        move_sticky_attached_pose(place_pose, object_in_hand)
                    )
                else:
                    place_kwargs: dict[str, Any] = {}
                    if attached_ignore_objects is not None:
                        place_kwargs["ignore_objects"] = attached_ignore_objects
                    yield from actions(
                        primitives._move_hand(place_pose, **place_kwargs)
                    )
                log_state("after_place_back_lower")

                phase = "place_back_release"
                yield from actions(primitives._execute_release())
                yield from actions(primitives._settle_robot())
                released_object = primitives._get_obj_in_hand()
                released_target_position = target_position_array()
                release_passed = released_object is None
                place_back_evidence = {
                    "enabled": True,
                    "release_passed": release_passed,
                    "object_in_hand_after_release": getattr(
                        released_object, "name", None
                    ),
                    "target_position_after_release": (
                        None
                        if released_target_position is None
                        else released_target_position.tolist()
                    ),
                    "planned_place_eef_position": place_pos.tolist(),
                    "clearance_m": self._place_back_clearance_m,
                }
                if not release_passed:
                    raise RuntimeError(
                        "object remained attached after place-back release"
                    )

                phase = "place_back_retreat"
                retreat_pos = place_pos.copy()
                retreat_pos[2] += self._place_back_retreat_m
                retreat_pose = (
                    th.as_tensor(retreat_pos, dtype=th.float32),
                    grasp_quat_t.clone(),
                )
                yield from actions(primitives._move_hand(retreat_pose))
                place_back_evidence["retreat_eef_position"] = retreat_pos.tolist()
                place_back_evidence["passed"] = True
                physical_evidence["place_back"] = dict(place_back_evidence)
                logger.warning(
                    "AnyGrasp place-back verification evidence=%s",
                    place_back_evidence,
                )
        except Exception as exc:
            current_target_position = target_position_array()

            def displacement_from(origin: np.ndarray | None) -> float | None:
                if origin is None or current_target_position is None:
                    return None
                return float(np.linalg.norm(current_target_position - origin))

            failure_evidence = dict(physical_evidence)
            failure_evidence.update(
                {
                    "passed": False,
                    "failure_phase": phase,
                    "phase_contact_evidence": phase_contact_evidence,
                    "approach_segment_audit": approach_segment_audit,
                    "initial_target_position": (
                        None
                        if initial_target_position is None
                        else initial_target_position.tolist()
                    ),
                    "failure_target_position": (
                        None
                        if current_target_position is None
                        else current_target_position.tolist()
                    ),
                    "target_displacement_from_initial_m": displacement_from(
                        initial_target_position
                    ),
                    "approach_baseline_target_position": (
                        None
                        if approach_target_origin is None
                        else approach_target_origin.tolist()
                    ),
                    "approach_target_displacement_m": displacement_from(
                        approach_target_origin
                    ),
                    "approach_target_displacement_tolerance_m": (
                        self._approach_target_displacement_tolerance_m
                    ),
                    "close_baseline_target_position": (
                        None
                        if close_target_origin is None
                        else close_target_origin.tolist()
                    ),
                    "close_target_displacement_m": displacement_from(
                        close_target_origin
                    ),
                    "close_target_displacement_tolerance_m": (
                        self._close_target_displacement_tolerance_m
                    ),
                    "lift_started": lift_started,
                    "lift_completed": lift_completed,
                }
            )
            try:
                live_contact = physical_contact_evidence()
                failure_evidence["failure_contact_evidence"] = live_contact
                failure_evidence["failure_gripper_qpos"] = live_contact.get(
                    "gripper_qpos"
                )
            except Exception as evidence_error:
                failure_evidence["failure_contact_evidence_error"] = (
                    f"{type(evidence_error).__name__}: {evidence_error}"
                )
            try:
                failure_evidence.update(attachment_evidence())
            except Exception as attachment_error:
                failure_evidence["attachment_evidence_error"] = (
                    f"{type(attachment_error).__name__}: {attachment_error}"
                )
            try:
                object_in_hand = primitives._get_obj_in_hand()
            except Exception as object_error:
                object_in_hand = None
                failure_evidence["object_in_hand_error"] = (
                    f"{type(object_error).__name__}: {object_error}"
                )
            return GraspResult(
                success=False,
                object_in_hand=getattr(object_in_hand, "name", None),
                grasp_pos_world=grasp_pos,
                grasp_quat_world=grasp_quat,
                anygrasp_score=float(candidate.score),
                total_sim_steps=steps,
                error=(
                    f"grasp execution failed during {phase}: {type(exc).__name__}: {exc}"
                ),
                failure_phase=phase,
                scene_changed=(
                    phase in scene_changing_phases
                    or target_has_moved(
                        self._approach_target_displacement_tolerance_m
                        if phase in contact_approach_phases
                        else _TARGET_DISPLACEMENT_TOLERANCE_M
                    )
                ),
                physical_evidence=failure_evidence,
            )

        mode = getattr(self._robot, "grasping_mode", None)
        expected_name = getattr(target_obj, "name", None)
        if self._post_lift_place_back:
            object_name = physical_evidence.get("object_in_hand")
            identity_matches = bool(physical_evidence.get("object_identity_matches"))
        elif mode == "physical":
            object_name = physical_evidence.get("object_in_hand")
            identity_matches = bool(physical_evidence.get("object_identity_matches"))
        else:
            object_in_hand = primitives._get_obj_in_hand()
            object_name = getattr(object_in_hand, "name", None)
            identity_matches = bool(
                object_in_hand is not None
                and expected_name is not None
                and object_name == expected_name
            )
        physical_grasp_verified = bool(physical_evidence.get("passed"))
        success = identity_matches and physical_grasp_verified
        if success:
            error = None
        elif not identity_matches:
            error = (
                f"wrong object grasped: {object_name}, expected {expected_name}"
                if object_name is not None
                else "no object in hand after lift"
            )
        else:
            error = "physical grasp verification failed after lift"
        return GraspResult(
            success=success,
            object_in_hand=object_name,
            grasp_pos_world=grasp_pos,
            grasp_quat_world=grasp_quat,
            anygrasp_score=float(candidate.score),
            total_sim_steps=steps,
            error=error,
            failure_phase=None if success else "physical_verification",
            scene_changed=not success,
            physical_grasp_verified=physical_grasp_verified,
            physical_evidence=physical_evidence,
        )

    def begin_grasp_by_object(self, target_obj: Any) -> GraspExecution:
        primitives = self._ensure_primitives()
        if primitives is None:
            result = GraspResult(
                success=False,
                object_in_hand=None,
                grasp_pos_world=np.zeros(3, dtype=np.float32),
                grasp_quat_world=np.array([0, 0, 0, 1], dtype=np.float32),
                anygrasp_score=0.0,
                total_sim_steps=0,
                error="StarterSemanticActionPrimitives init failed",
            )
            return GraspExecution(self._immediate_result(result))
        return GraspExecution(self._builtin_grasp_generator(primitives, target_obj))

    def _builtin_grasp_generator(
        self,
        primitives: Any,
        target_obj: Any,
    ) -> Generator[Any, None, GraspResult]:
        from omnigibson.action_primitives.starter_semantic_action_primitives import (
            StarterSemanticActionPrimitiveSet,
        )

        steps = 0
        try:
            for action in primitives.apply_ref(
                StarterSemanticActionPrimitiveSet.GRASP, target_obj
            ):
                if action is not None:
                    steps += 1
                    yield action
        except Exception as exc:
            return GraspResult(
                success=False,
                object_in_hand=None,
                grasp_pos_world=np.zeros(3, dtype=np.float32),
                grasp_quat_world=np.array([0, 0, 0, 1], dtype=np.float32),
                anygrasp_score=0.0,
                total_sim_steps=steps,
                error=f"built-in grasp failed: {type(exc).__name__}: {exc}",
            )
        object_in_hand = primitives._get_obj_in_hand()
        object_name = getattr(object_in_hand, "name", None)
        expected = getattr(target_obj, "name", None)
        success = object_in_hand is not None and object_name == expected
        return GraspResult(
            success=success,
            object_in_hand=object_name,
            grasp_pos_world=np.zeros(3, dtype=np.float32),
            grasp_quat_world=np.array([0, 0, 0, 1], dtype=np.float32),
            anygrasp_score=0.0,
            total_sim_steps=steps,
            error=None
            if success
            else f"object in hand is {object_name}, expected {expected}",
        )
