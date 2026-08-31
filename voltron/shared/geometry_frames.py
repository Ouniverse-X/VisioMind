from __future__ import annotations

import math
from typing import Any


def normalize_vertical_axis(value: Any, *, default: str = "z") -> str:
    axis = str(value or "").strip().lower()
    return axis if axis in {"x", "y", "z"} else default


def frame_transform_for_vertical_axes(
    *,
    source_vertical_axis: str,
    target_vertical_axis: str,
) -> list[list[float]]:
    source = _canonical_from_frame(normalize_vertical_axis(source_vertical_axis))
    target = _canonical_from_frame(normalize_vertical_axis(target_vertical_axis))
    rotation = _multiply_3x3(_transpose_3x3(target), source)
    return [
        [*rotation[0], 0.0],
        [*rotation[1], 0.0],
        [*rotation[2], 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def coerce_frame_transform(value: Any) -> list[list[float]] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    matrix = []
    for row in value[:4]:
        if not isinstance(row, (list, tuple)) or len(row) < 4:
            return None
        try:
            matrix.append([float(item) for item in row[:4]])
        except (TypeError, ValueError):
            return None
    return matrix


def resolve_frame_contract(*containers: Any) -> dict[str, Any]:
    scene_vertical_axis = None
    simulator_vertical_axis = None
    explicit_transform = None
    for container in containers:
        if not isinstance(container, dict):
            continue
        scene_vertical_axis = (
            scene_vertical_axis
            or container.get("scene_vertical_axis")
            or container.get("hovsg_vertical_axis")
            or container.get("vertical_axis")
        )
        simulator_vertical_axis = simulator_vertical_axis or container.get(
            "simulator_vertical_axis"
        )
        explicit_transform = (
            explicit_transform
            or container.get("scene_from_simulator_transform")
            or container.get("simulator_to_scene_transform")
        )
    simulator_axis = normalize_vertical_axis(simulator_vertical_axis, default="z")
    scene_axis = normalize_vertical_axis(scene_vertical_axis, default=simulator_axis)
    transform = coerce_frame_transform(explicit_transform) or (
        frame_transform_for_vertical_axes(
            source_vertical_axis=simulator_axis,
            target_vertical_axis=scene_axis,
        )
    )
    return {
        "simulator_vertical_axis": simulator_axis,
        "scene_vertical_axis": scene_axis,
        "scene_from_simulator_transform": transform,
    }


def transform_point(
    point: list[float] | tuple[float, ...],
    transform: list[list[float]],
) -> list[float]:
    x_coord, y_coord, z_coord = (float(point[index]) for index in range(3))
    return [
        transform[row][0] * x_coord
        + transform[row][1] * y_coord
        + transform[row][2] * z_coord
        + transform[row][3]
        for row in range(3)
    ]


def transform_vector(
    vector: list[float] | tuple[float, ...],
    transform: list[list[float]],
) -> list[float]:
    x_coord, y_coord, z_coord = (float(vector[index]) for index in range(3))
    return [
        transform[row][0] * x_coord + transform[row][1] * y_coord + transform[row][2] * z_coord
        for row in range(3)
    ]


def transform_position(
    position: dict[str, float] | None,
    transform: list[list[float]],
) -> dict[str, float] | None:
    if not isinstance(position, dict):
        return None
    try:
        point = transform_point(
            [position["x"], position["y"], position.get("z", 0.0)],
            transform,
        )
    except (KeyError, TypeError, ValueError):
        return None
    return {"x": point[0], "y": point[1], "z": point[2]}


def transform_orientation(
    orientation: dict[str, Any] | None,
    transform: list[list[float]],
    *,
    source_vertical_axis: str = "z",
    target_vertical_axis: str = "z",
) -> dict[str, float] | None:
    if not isinstance(orientation, dict):
        return None
    if is_identity_transform(transform) and normalize_vertical_axis(
        source_vertical_axis
    ) == normalize_vertical_axis(target_vertical_axis):
        try:
            return {key: float(value) for key, value in orientation.items()}
        except (TypeError, ValueError):
            return dict(orientation)
    source_rotation = _orientation_rotation_matrix(
        orientation,
        vertical_axis=source_vertical_axis,
    )
    if source_rotation is None:
        return dict(orientation)
    frame_rotation = _rotation_from_transform(transform)
    scene_rotation = _multiply_3x3(frame_rotation, source_rotation)
    quaternion = _quaternion_from_rotation_matrix(scene_rotation)
    horizontal_indices = horizontal_axis_indices(target_vertical_axis)
    forward = [scene_rotation[row][0] for row in range(3)]
    yaw = math.atan2(
        forward[horizontal_indices[1]],
        forward[horizontal_indices[0]],
    )
    return {
        "x": quaternion[0],
        "y": quaternion[1],
        "z": quaternion[2],
        "w": quaternion[3],
        "yaw": yaw,
    }


def transform_aabb(
    aabb: dict[str, list[float]] | None,
    transform: list[list[float]],
) -> dict[str, list[float]] | None:
    if not isinstance(aabb, dict):
        return None
    minimum = aabb.get("min")
    maximum = aabb.get("max")
    if not isinstance(minimum, (list, tuple)) or not isinstance(maximum, (list, tuple)):
        return None
    try:
        points = [
            transform_point([x_coord, y_coord, z_coord], transform)
            for x_coord in (minimum[0], maximum[0])
            for y_coord in (minimum[1], maximum[1])
            for z_coord in (minimum[2], maximum[2])
        ]
    except (IndexError, TypeError, ValueError):
        return None
    return {
        "min": [min(point[index] for point in points) for index in range(3)],
        "max": [max(point[index] for point in points) for index in range(3)],
    }


def multiply_transforms(
    left: list[list[float]],
    right: list[list[float]],
) -> list[list[float]]:
    return [
        [sum(left[row][index] * right[index][column] for index in range(4)) for column in range(4)]
        for row in range(4)
    ]


def horizontal_axis_indices(vertical_axis: str) -> tuple[int, int]:
    return {
        "x": (1, 2),
        "y": (0, 2),
        "z": (0, 1),
    }.get(normalize_vertical_axis(vertical_axis), (0, 1))


def vertical_axis_index(vertical_axis: str) -> int:
    return {"x": 0, "y": 1, "z": 2}.get(
        normalize_vertical_axis(vertical_axis),
        2,
    )


def _canonical_from_frame(vertical_axis: str) -> list[list[float]]:
    if vertical_axis == "y":
        return [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    if vertical_axis == "x":
        return [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def _transpose_3x3(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[column][row] for column in range(3)] for row in range(3)]


def _multiply_3x3(
    left: list[list[float]],
    right: list[list[float]],
) -> list[list[float]]:
    return [
        [sum(left[row][index] * right[index][column] for index in range(3)) for column in range(3)]
        for row in range(3)
    ]


def _orientation_rotation_matrix(
    orientation: dict[str, Any],
    *,
    vertical_axis: str,
) -> list[list[float]] | None:
    quaternion = [
        orientation.get("x"),
        orientation.get("y"),
        orientation.get("z"),
        orientation.get("w"),
    ]
    if all(value is not None for value in quaternion):
        try:
            return _rotation_matrix_from_quaternion(*(float(value) for value in quaternion))
        except (TypeError, ValueError):
            return None
    if all(key in orientation for key in ("roll", "pitch", "yaw")):
        try:
            roll = float(orientation["roll"])
            pitch = float(orientation["pitch"])
            yaw = float(orientation["yaw"])
        except (TypeError, ValueError):
            return None
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        return _rotation_matrix_from_quaternion(
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )
    if "yaw" in orientation:
        try:
            angle = float(orientation["yaw"])
        except (TypeError, ValueError):
            return None
        axis = [0.0, 0.0, 0.0]
        axis[vertical_axis_index(vertical_axis)] = 1.0
        half_angle = 0.5 * angle
        scale = math.sin(half_angle)
        return _rotation_matrix_from_quaternion(
            axis[0] * scale,
            axis[1] * scale,
            axis[2] * scale,
            math.cos(half_angle),
        )
    return None


def _rotation_matrix_from_quaternion(
    x_coord: float,
    y_coord: float,
    z_coord: float,
    w_coord: float,
) -> list[list[float]]:
    norm = math.sqrt(x_coord * x_coord + y_coord * y_coord + z_coord * z_coord + w_coord * w_coord)
    if norm <= 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    x_coord /= norm
    y_coord /= norm
    z_coord /= norm
    w_coord /= norm
    return [
        [
            1.0 - 2.0 * (y_coord * y_coord + z_coord * z_coord),
            2.0 * (x_coord * y_coord - z_coord * w_coord),
            2.0 * (x_coord * z_coord + y_coord * w_coord),
        ],
        [
            2.0 * (x_coord * y_coord + z_coord * w_coord),
            1.0 - 2.0 * (x_coord * x_coord + z_coord * z_coord),
            2.0 * (y_coord * z_coord - x_coord * w_coord),
        ],
        [
            2.0 * (x_coord * z_coord - y_coord * w_coord),
            2.0 * (y_coord * z_coord + x_coord * w_coord),
            1.0 - 2.0 * (x_coord * x_coord + y_coord * y_coord),
        ],
    ]


def _quaternion_from_rotation_matrix(
    matrix: list[list[float]],
) -> tuple[float, float, float, float]:
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = (
            (matrix[2][1] - matrix[1][2]) / scale,
            (matrix[0][2] - matrix[2][0]) / scale,
            (matrix[1][0] - matrix[0][1]) / scale,
            0.25 * scale,
        )
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
        quaternion = (
            0.25 * scale,
            (matrix[0][1] + matrix[1][0]) / scale,
            (matrix[0][2] + matrix[2][0]) / scale,
            (matrix[2][1] - matrix[1][2]) / scale,
        )
    elif matrix[1][1] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
        quaternion = (
            (matrix[0][1] + matrix[1][0]) / scale,
            0.25 * scale,
            (matrix[1][2] + matrix[2][1]) / scale,
            (matrix[0][2] - matrix[2][0]) / scale,
        )
    else:
        scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
        quaternion = (
            (matrix[0][2] + matrix[2][0]) / scale,
            (matrix[1][2] + matrix[2][1]) / scale,
            0.25 * scale,
            (matrix[1][0] - matrix[0][1]) / scale,
        )
    norm = math.sqrt(sum(value * value for value in quaternion))
    return tuple(value / norm for value in quaternion)


def _rotation_from_transform(transform: list[list[float]]) -> list[list[float]]:
    first = _normalize_vector([transform[row][0] for row in range(3)])
    second_raw = [transform[row][1] for row in range(3)]
    projection = sum(first[index] * second_raw[index] for index in range(3))
    second = _normalize_vector(
        [second_raw[index] - projection * first[index] for index in range(3)]
    )
    third = _cross(first, second)
    original_third = [transform[row][2] for row in range(3)]
    if sum(third[index] * original_third[index] for index in range(3)) < 0.0:
        second = [-value for value in second]
        third = _cross(first, second)
    return [[first[row], second[row], third[row]] for row in range(3)]


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-12:
        return [1.0, 0.0, 0.0]
    return [value / norm for value in vector]


def _cross(left: list[float], right: list[float]) -> list[float]:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def is_identity_transform(transform: list[list[float]]) -> bool:
    return all(
        abs(float(transform[row][column]) - (1.0 if row == column else 0.0)) <= 1e-12
        for row in range(4)
        for column in range(4)
    )


__all__ = [
    "coerce_frame_transform",
    "frame_transform_for_vertical_axes",
    "horizontal_axis_indices",
    "is_identity_transform",
    "multiply_transforms",
    "normalize_vertical_axis",
    "resolve_frame_contract",
    "transform_aabb",
    "transform_orientation",
    "transform_point",
    "transform_position",
    "transform_vector",
    "vertical_axis_index",
]
