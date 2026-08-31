from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)
DEFAULT_ROOT_PATH = "/World/scene_0/VisioMindIndustrialWorkcell"


def _as_vector(value: Any, size: int) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size != size or not np.isfinite(vector).all():
        raise ValueError(f"expected {size} finite values")
    return vector


def _quaternion_rotation_xyzw(quaternion: Any) -> np.ndarray:
    x, y, z, w = _as_vector(quaternion, 4)
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm <= 1e-12:
        raise ValueError("robot orientation quaternion has zero norm")
    x, y, z, w = np.array([x, y, z, w]) / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", str(value)).strip("_")
    return name or "part"


def _environment_candidates(env: Any):
    seen: set[int] = set()
    current = env
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = getattr(current, "env", None)
    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and id(unwrapped) not in seen:
        yield unwrapped


def _first_robot(env: Any) -> Any | None:
    for candidate in _environment_candidates(env):
        robots = getattr(candidate, "robots", None)
        if robots:
            return robots[0]
    return None


def _scene_objects(env: Any) -> list[Any]:
    for candidate in _environment_candidates(env):
        scene = getattr(candidate, "scene", None)
        objects = getattr(scene, "objects", None)
        if objects:
            return list(objects)
        scenes = getattr(candidate, "scenes", None)
        if scenes:
            objects = getattr(scenes[0], "objects", None)
            if objects:
                return list(objects)
    return []


def _find_object(env: Any, exact_name: str | None) -> Any | None:
    if not exact_name:
        return None
    for obj in _scene_objects(env):
        if str(getattr(obj, "name", "")) == str(exact_name):
            return obj
    return None


def _world_aabb(obj: Any) -> tuple[np.ndarray, np.ndarray] | None:
    aabb = getattr(obj, "aabb", None)
    if not isinstance(aabb, (tuple, list)) or len(aabb) != 2:
        return None
    lower = _as_vector(aabb[0], 3)
    upper = _as_vector(aabb[1], 3)
    if np.any(upper <= lower):
        return None
    return lower, upper


def _cube_spec(
    name: str,
    local_position: list[float],
    dimensions: list[float],
    color: list[float],
    opacity: float = 1.0,
) -> dict[str, Any]:
    return {
        "name": name,
        "local_position": local_position,
        "dimensions": dimensions,
        "color": color,
        "opacity": opacity,
    }


def default_workcell_parts() -> list[dict[str, Any]]:
    steel = [0.055, 0.075, 0.105]
    blue = [0.025, 0.16, 0.31]
    yellow = [0.95, 0.62, 0.02]
    return [
        _cube_spec("floor_mat", [0.70, 0.0, 0.006], [2.8, 3.0, 0.012], [0.12, 0.14, 0.16]),
        _cube_spec("rear_backdrop", [1.75, 0.0, 1.25], [0.04, 3.2, 2.5], steel),
        _cube_spec("rear_blue_panel", [1.72, 0.0, 1.45], [0.025, 2.55, 0.75], blue),
        _cube_spec("left_upright", [1.68, 1.42, 1.25], [0.12, 0.12, 2.5], blue),
        _cube_spec("right_upright", [1.68, -1.42, 1.25], [0.12, 0.12, 2.5], blue),
        _cube_spec("top_beam", [1.68, 0.0, 2.43], [0.12, 2.95, 0.12], blue),
        _cube_spec("safety_front", [-0.62, 0.0, 0.018], [0.08, 3.0, 0.025], yellow),
        _cube_spec("safety_left", [0.70, 1.46, 0.018], [2.7, 0.08, 0.025], yellow),
        _cube_spec("safety_right", [0.70, -1.46, 0.018], [2.7, 0.08, 0.025], yellow),
        _cube_spec("header_light", [1.62, 0.0, 2.25], [0.05, 1.15, 0.08], [0.15, 0.85, 1.0]),
    ]


def _define_cube(
    stage: Any,
    path: str,
    *,
    position: np.ndarray,
    orientation_xyzw: np.ndarray,
    dimensions: np.ndarray,
    color: np.ndarray,
    opacity: float,
) -> None:
    from pxr import Gf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color.astype(float).tolist())])
    cube.CreateDisplayOpacityAttr([float(opacity)])
    xformable = UsdGeom.Xformable(cube.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*position.astype(float).tolist()))
    x, y, z, w = orientation_xyzw.astype(float).tolist()
    xformable.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    xformable.AddScaleOp().Set(Gf.Vec3f(*dimensions.astype(float).tolist()))


def _add_cell_rims(
    stage: Any,
    root_path: str,
    *,
    container_aabb: tuple[np.ndarray, np.ndarray],
    grid_shape: tuple[int, int],
    highlighted_cell: int,
) -> list[str]:
    lower, upper = container_aabb
    horizontal = upper[:2] - lower[:2]
    column_axis = int(np.argmax(horizontal))
    row_axis = 1 - column_axis
    rows, columns = grid_shape
    total = rows * columns
    if not 1 <= highlighted_cell <= total:
        raise ValueError(f"highlighted_cell must be in [1, {total}]")
    row = (highlighted_cell - 1) // columns
    column = (highlighted_cell - 1) % columns
    cell_lower = lower.copy()
    cell_upper = upper.copy()
    cell_lower[column_axis] += column * horizontal[column_axis] / columns
    cell_upper[column_axis] = cell_lower[column_axis] + horizontal[column_axis] / columns
    cell_lower[row_axis] += row * horizontal[row_axis] / rows
    cell_upper[row_axis] = cell_lower[row_axis] + horizontal[row_axis] / rows
    center = (cell_lower + cell_upper) / 2.0
    center[2] = upper[2] + 0.012
    thickness = 0.012
    height = 0.024
    green = np.array([0.12, 0.95, 0.32], dtype=np.float64)
    created: list[str] = []
    for suffix, axis, sign in (
        ("column_min", column_axis, -1),
        ("column_max", column_axis, 1),
        ("row_min", row_axis, -1),
        ("row_max", row_axis, 1),
    ):
        position = center.copy()
        dimensions = np.array(
            [cell_upper[0] - cell_lower[0], cell_upper[1] - cell_lower[1], height],
            dtype=np.float64,
        )
        position[axis] = cell_lower[axis] if sign < 0 else cell_upper[axis]
        dimensions[axis] = thickness
        other_axis = 1 - axis
        dimensions[other_axis] += thickness
        path = f"{root_path}/target_cell_{_safe_name(suffix)}"
        _define_cube(
            stage,
            path,
            position=position,
            orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            dimensions=dimensions,
            color=green,
            opacity=0.9,
        )
        created.append(path)
    return created


def _add_physical_dividers(
    stage: Any,
    root_path: str,
    *,
    container_aabb: tuple[np.ndarray, np.ndarray],
    grid_shape: tuple[int, int],
    divider_thickness: float = 0.008,
) -> list[str]:
    from voltron.shared.compartment_geometry import MultiCompartmentBinGeometry

    geom = MultiCompartmentBinGeometry(
        container_aabb=container_aabb,
        grid_shape=grid_shape,
        divider_thickness_m=divider_thickness,
    )
    steel = np.array([0.45, 0.48, 0.52], dtype=np.float64)
    created: list[str] = []
    for div in geom.get_all_dividers():
        path = f"{root_path}/divider_{_safe_name(div.divider_id)}"
        _define_cube(
            stage,
            path,
            position=np.asarray(div.center_world, dtype=np.float64),
            orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            dimensions=np.asarray(div.dimensions_world, dtype=np.float64),
            color=steel,
            opacity=1.0,
        )
        created.append(path)
    return created


def install_industrial_workcell(env: Any, config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict) or not bool(config.get("enabled", False)):
        return {"enabled": False, "created": False}
    robot = _first_robot(env)
    if robot is None:
        raise RuntimeError("industrial workcell requires a loaded robot")
    robot_position, robot_orientation = robot.get_position_orientation()
    robot_position = _as_vector(robot_position, 3)
    robot_orientation = _as_vector(robot_orientation, 4)
    rotation = _quaternion_rotation_xyzw(robot_orientation)

    try:
        import omnigibson as og

        stage = getattr(og.sim, "stage", None)
    except Exception as exc:
        raise RuntimeError("industrial workcell requires OmniGibson") from exc
    if stage is None:
        raise RuntimeError("industrial workcell could not resolve the active USD stage")

    root_path = str(config.get("root_path", DEFAULT_ROOT_PATH))
    if not root_path.startswith("/World/scene_0/"):
        raise ValueError("industrial workcell root_path must be under /World/scene_0")
    if stage.GetPrimAtPath(root_path).IsValid():
        stage.RemovePrim(root_path)

    from pxr import UsdGeom

    UsdGeom.Xform.Define(stage, root_path)
    created_paths: list[str] = []
    parts = config.get("parts", default_workcell_parts())
    if not isinstance(parts, list):
        raise ValueError("industrial workcell parts must be a list")
    for index, part in enumerate(parts):
        if not isinstance(part, dict):
            raise ValueError("industrial workcell part must be an object")
        name = _safe_name(str(part.get("name", f"part_{index}")))
        local_position = _as_vector(part["local_position"], 3)
        world_position = robot_position + rotation @ local_position
        dimensions = _as_vector(part["dimensions"], 3)
        if np.any(dimensions <= 0.0):
            raise ValueError(f"industrial part {name!r} dimensions must be positive")
        color = _as_vector(part["color"], 3)
        if np.any(color < 0.0) or np.any(color > 1.0):
            raise ValueError(f"industrial part {name!r} color must be in [0, 1]")
        path = f"{root_path}/{name}"
        _define_cube(
            stage,
            path,
            position=world_position,
            orientation_xyzw=robot_orientation,
            dimensions=dimensions,
            color=color,
            opacity=float(part.get("opacity", 1.0)),
        )
        created_paths.append(path)

    container_name = config.get("cell_container")
    container = _find_object(env, str(container_name) if container_name else None)
    cell_rim_paths: list[str] = []
    divider_paths: list[str] = []
    container_aabb = _world_aabb(container) if container is not None else None
    if container_name and container_aabb is None:
        raise RuntimeError(
            f"industrial cell container {container_name!r} was not found or has no AABB"
        )
    if container_aabb is not None:
        raw_shape = config.get("grid_shape", [1, 3])
        shape = tuple(int(value) for value in raw_shape)
        if len(shape) != 2 or any(value < 1 for value in shape):
            raise ValueError("industrial grid_shape must contain two positive integers")
        try:
            divider_paths = _add_physical_dividers(
                stage,
                root_path,
                container_aabb=container_aabb,
                grid_shape=shape,
                divider_thickness=float(config.get("divider_thickness_m", 0.008)),
            )
            created_paths.extend(divider_paths)
        except Exception as exc:
            logger.warning("Industrial divider generation skipped: %s", exc)

        cell_rim_paths = _add_cell_rims(
            stage,
            root_path,
            container_aabb=container_aabb,
            grid_shape=shape,
            highlighted_cell=int(config.get("highlighted_cell", 3)),
        )
        created_paths.extend(cell_rim_paths)

    return {
        "enabled": True,
        "created": True,
        "root_path": root_path,
        "visual_only": True,
        "collision_api_applied": False,
        "robot_anchor_position_world": robot_position.tolist(),
        "part_count": len(created_paths),
        "created_paths": created_paths,
        "cell_container": container_name,
        "cell_container_aabb_world": (
            [container_aabb[0].tolist(), container_aabb[1].tolist()]
            if container_aabb is not None
            else None
        ),
        "cell_rim_count": len(cell_rim_paths),
        "physical_divider_count": len(divider_paths),
        "physical_divider_paths": divider_paths,
    }
