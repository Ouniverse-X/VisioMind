from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


ANYGRASP_TO_EEF_ROTATION = np.array(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
    dtype=np.float64,
)


def _pose_to_matrix(pos: Any, quat: Any) -> np.ndarray:
    position = np.asarray(pos, dtype=np.float64).reshape(-1)[:3]
    x, y, z, w = np.asarray(quat, dtype=np.float64).reshape(-1)[:4]
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm <= 1e-12:
        raise ValueError("quaternion has zero norm")
    x, y, z, w = np.asarray([x, y, z, w], dtype=np.float64) / norm
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    matrix[:3, 3] = position
    return matrix


def _mat_to_quat_xyzw(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = 0.5 / np.sqrt(trace + 1.0)
        quat = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) * scale,
                (matrix[0, 2] - matrix[2, 0]) * scale,
                (matrix[1, 0] - matrix[0, 1]) * scale,
                0.25 / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = 2.0 * np.sqrt(max(1e-12, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]))
            quat = np.array(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ]
            )
        elif index == 1:
            scale = 2.0 * np.sqrt(max(1e-12, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]))
            quat = np.array(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(max(1e-12, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]))
            quat = np.array(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
    return (quat / np.linalg.norm(quat)).astype(np.float32)


@dataclass(frozen=True)
class GraspFramePose:
    canonical_origin_world: np.ndarray
    eef_rotation_world: np.ndarray
    approach_world: np.ndarray
    jaw_world: np.ndarray

    @property
    def eef_quaternion_xyzw(self) -> np.ndarray:
        return _mat_to_quat_xyzw(self.eef_rotation_world)


@dataclass(frozen=True)
class AnyGraspFrameAdapter:
    anygrasp_to_eef_rotation: np.ndarray = field(
        default_factory=lambda: ANYGRASP_TO_EEF_ROTATION.copy()
    )

    def __post_init__(self) -> None:
        correction = np.asarray(self.anygrasp_to_eef_rotation, dtype=np.float64)
        if correction.shape != (3, 3) or not np.isfinite(correction).all():
            raise ValueError("AnyGrasp-to-EEF rotation must be a finite 3x3 matrix")
        if not np.allclose(correction.T @ correction, np.eye(3), atol=1e-6):
            raise ValueError("AnyGrasp-to-EEF rotation must be orthonormal")
        if not np.isclose(np.linalg.det(correction), 1.0, atol=1e-6):
            raise ValueError("AnyGrasp-to-EEF rotation must have determinant +1")

    def camera_candidate_to_world(
        self,
        translation_camera: Any,
        rotation_camera: Any,
        camera_pose_world: Any,
    ) -> GraspFramePose:
        translation = np.asarray(translation_camera, dtype=np.float64).reshape(-1)
        rotation = np.asarray(rotation_camera, dtype=np.float64)
        world_from_camera = np.asarray(camera_pose_world, dtype=np.float64)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("AnyGrasp translation must be a finite three-vector")
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError("AnyGrasp rotation must be a finite 3x3 matrix")
        if world_from_camera.shape != (4, 4) or not np.isfinite(world_from_camera).all():
            raise ValueError("camera_pose_world must be a finite 4x4 matrix")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
            raise ValueError("AnyGrasp rotation must be approximately orthonormal")
        world_from_grasp = world_from_camera @ np.block(
            [
                [rotation, translation[:, None]],
                [np.zeros((1, 3)), np.ones((1, 1))],
            ]
        )
        eef_rotation = world_from_grasp[:3, :3] @ self.anygrasp_to_eef_rotation
        return GraspFramePose(
            canonical_origin_world=world_from_grasp[:3, 3].astype(np.float32),
            eef_rotation_world=eef_rotation,
            approach_world=world_from_grasp[:3, :3][:, 0].astype(np.float32),
            jaw_world=world_from_grasp[:3, :3][:, 1].astype(np.float32),
        )

    def validate_basis_mapping(self) -> dict[str, Any]:
        correction = np.asarray(self.anygrasp_to_eef_rotation, dtype=np.float64)
        return {
            "anygrasp_approach_to_eef_z": correction[:, 2].tolist(),
            "anygrasp_jaw_to_eef_y": correction[:, 1].tolist(),
            "anygrasp_completion_to_eef_x": correction[:, 0].tolist(),
            "determinant": float(np.linalg.det(correction)),
        }
