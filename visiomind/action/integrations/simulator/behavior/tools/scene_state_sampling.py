from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from visiomind.action.integrations.simulator.behavior.tools.bridge_environment import (
    _collect_scene_objects,
    _env_candidates,
    _object_open_state,
)
from visiomind.action.integrations.simulator.behavior.tools.door_navigation_passability import (
    apply_runtime_override as apply_door_navigation_passability_override,
)
from visiomind.action.integrations.simulator.behavior.tools.scene_relation_sampling import (
    sample_scene_relations,
)
from visiomind.action.shared.geometry_frames import (
    frame_transform_for_vertical_axes,
    horizontal_axis_indices,
    multiply_transforms,
    resolve_frame_contract,
    transform_aabb,
    transform_point,
    transform_position,
    vertical_axis_index,
)
from visiomind.action.shared.models.scene_state import (
    NAVIGATION_ROLE_OBSTACLE,
    NAVIGATION_ROLE_OVERHEAD,
    NAVIGATION_ROLE_STRUCTURAL,
    NAVIGATION_ROLE_SUPPORT_SURFACE,
    RuntimeDoorState,
    RuntimeObjectState,
    RuntimeObstacleState,
    SceneRuntimeState,
    compute_relation_signature,
    compute_scene_state_signature,
    is_door_category,
    navigation_role_from_category,
    scene_runtime_state_from_payload,
)

_VALID_NAVIGATION_ROLES = {
    NAVIGATION_ROLE_OBSTACLE,
    NAVIGATION_ROLE_OVERHEAD,
    NAVIGATION_ROLE_STRUCTURAL,
    NAVIGATION_ROLE_SUPPORT_SURFACE,
}


def sample_scene_runtime_state(
    runtime: Any,
    *,
    scene_id: str | None,
) -> dict[str, Any] | None:
    env = getattr(runtime, "_env", None)
    if env is None:
        return None
    cache = getattr(runtime, "_scene_runtime_state_cache", None)
    step = int(cache.get("step", 0)) + 1 if isinstance(cache, dict) else 1

    tracked_names = _tracked_object_names(runtime, env)
    frame_contract = _scene_frame_contract(runtime)
    scene_transform = frame_contract["scene_from_simulator_transform"]
    scene_vertical_axis = frame_contract["scene_vertical_axis"]
    include_aabb = _include_aabb(runtime)
    track_all_collision_objects = include_aabb and _track_all_collision_objects(runtime)
    objects: dict[str, RuntimeObjectState] = {}
    doors: dict[str, RuntimeDoorState] = {}
    temporary_obstacles = _temporary_navigation_obstacles(runtime, current_step=step)
    controlled_robot_objects, controlled_robot_names = _controlled_robot_identity(env)
    scene_objects = list(_collect_scene_objects(env))
    for obj in scene_objects:
        name = str(getattr(obj, "name", "") or "").strip()
        if not name:
            continue
        if id(obj) in controlled_robot_objects or name in controlled_robot_names:
            continue
        category = str(getattr(obj, "category", "") or getattr(obj, "class_name", "") or "").strip()
        navigation_role = _navigation_role(runtime, name=name, category=category)
        if is_door_category(category):
            doors[name] = apply_door_navigation_passability_override(
                runtime,
                door=_door_state(
                    obj,
                    runtime=runtime,
                    name=name,
                    category=category,
                    include_aabb=include_aabb,
                    scene_transform=scene_transform,
                ),
                current_step=step,
            )
        elif (
            (track_all_collision_objects and navigation_role == NAVIGATION_ROLE_OBSTACLE)
            or name in tracked_names
            or _normalize(name) in tracked_names
        ):
            objects[name] = RuntimeObjectState(
                name=name,
                category=category or None,
                navigation_role=navigation_role,
                position=transform_position(_object_position_3d(obj), scene_transform),
                aabb=(transform_aabb(_object_aabb(obj), scene_transform) if include_aabb else None),
                collision_parts=(
                    _cached_collision_parts(
                        runtime,
                        obj,
                        name=name,
                        include_root=True,
                    )
                    if include_aabb
                    else []
                ),
                room_hint=_object_room_hint(obj),
                floor_hint=_object_floor_hint(obj),
            )
    relations = sample_scene_relations(
        runtime,
        scene_objects=scene_objects,
        objects=objects,
        doors=doors,
        step=step,
        vertical_axis=scene_vertical_axis,
    )
    if not objects and not doors and not temporary_obstacles and not isinstance(cache, dict):
        return None

    signature = compute_scene_state_signature(
        objects=objects,
        doors=doors,
        relations=relations,
        temporary_obstacles=temporary_obstacles,
        simulator_vertical_axis=frame_contract["simulator_vertical_axis"],
        scene_vertical_axis=scene_vertical_axis,
        scene_from_simulator_transform=scene_transform,
    )
    if (
        isinstance(cache, dict)
        and cache.get("signature") == signature
        and isinstance(cache.get("payload"), dict)
    ):
        payload = {**cache["payload"], "step": step}
        runtime._scene_runtime_state_cache = {
            "signature": signature,
            "payload": payload,
            "step": step,
        }
        return payload

    state = SceneRuntimeState(
        scene_id=scene_id,
        step=step,
        simulator_vertical_axis=frame_contract["simulator_vertical_axis"],
        scene_vertical_axis=scene_vertical_axis,
        scene_from_simulator_transform=scene_transform,
        objects=objects,
        doors=doors,
        relations=relations,
        relation_signature=compute_relation_signature(
            relations,
            current_step=step,
        ),
        temporary_obstacles=temporary_obstacles,
        signature=signature,
    )
    payload = state.to_payload()
    runtime._scene_runtime_state_cache = {
        "signature": signature,
        "payload": payload,
        "step": step,
    }
    return payload


def _temporary_navigation_obstacles(
    runtime: Any,
    *,
    current_step: int,
) -> list[RuntimeObstacleState]:
    for container in (
        getattr(runtime, "_temporary_navigation_obstacles", None),
        getattr(runtime, "temporary_navigation_obstacles", None),
    ):
        if not isinstance(container, (list, tuple, dict)):
            continue
        parsed = scene_runtime_state_from_payload({"temporary_obstacles": container})
        if parsed is not None:
            return [
                obstacle
                for obstacle in parsed.temporary_obstacles
                if obstacle.expires_at_step is None or current_step <= obstacle.expires_at_step
            ]
    return []


def _tracked_object_names(runtime: Any, env: Any) -> set[str]:
    names: set[str] = set()
    env_kwargs = getattr(runtime, "env_kwargs", None)
    if isinstance(env_kwargs, dict):
        configured = env_kwargs.get("scene_state_tracked_objects")
        if isinstance(configured, (list, tuple)):
            names.update(_normalize(item) for item in configured if str(item or "").strip())
    names.update(_task_relevant_object_names(env))
    return names


def _controlled_robot_identity(env: Any) -> tuple[set[int], set[str]]:
    for candidate in _env_candidates(env):
        robots = getattr(candidate, "robots", None)
        if not isinstance(robots, (list, tuple)) or not robots:
            continue
        robot = robots[0]
        names = {
            str(value).strip()
            for value in (
                getattr(robot, "name", None),
                getattr(robot, "prim_path", None),
            )
            if isinstance(value, str) and value.strip()
        }
        return {id(robot)}, names
    return set(), set()


def _include_aabb(runtime: Any) -> bool:
    for container in (
        getattr(runtime, "env_kwargs", None),
        getattr(runtime, "runtime_kwargs", None),
        getattr(runtime, "config", None),
        getattr(runtime, "_last_info", None),
        getattr(runtime, "last_info", None),
    ):
        if not isinstance(container, dict):
            continue
        value = container.get("scene_state_include_aabb")
        if isinstance(value, bool):
            return value
    return False


def _track_all_collision_objects(runtime: Any) -> bool:
    containers = (
        getattr(runtime, "env_kwargs", None),
        getattr(runtime, "runtime_kwargs", None),
        getattr(runtime, "config", None),
    )
    for container in containers:
        if not isinstance(container, dict):
            continue
        filename = container.get("nav2_trav_map_filename")
        if isinstance(filename, str) and filename.strip():
            normalized = filename.lower()
            return "no_obj" in normalized or "no_object" in normalized

    return True


def _task_relevant_object_names(env: Any) -> set[str]:
    names: set[str] = set()
    for candidate in _env_candidates(env):
        task = getattr(candidate, "task", None)
        object_scope = getattr(task, "object_scope", None)
        if not isinstance(object_scope, dict):
            continue
        for entry in object_scope.values():
            wrapped = getattr(entry, "wrapped_obj", entry)
            name = str(getattr(wrapped, "name", "") or "").strip()
            if name and not _is_structural_category(str(getattr(wrapped, "category", "") or "")):
                names.add(name)
                names.add(_normalize(name))
    return names


def _is_structural_category(category: str) -> bool:
    return _navigation_role_from_category(category) == NAVIGATION_ROLE_STRUCTURAL


def _navigation_role(runtime: Any, *, name: str, category: str) -> str:
    override = _navigation_role_override(runtime, name=name, category=category)
    if override is not None:
        return override
    return _navigation_role_from_category(category)


def _navigation_role_from_category(category: str) -> str:
    return navigation_role_from_category(category)


def _navigation_role_override(
    runtime: Any,
    *,
    name: str,
    category: str,
) -> str | None:
    lookup_keys = (
        name,
        _normalize(name),
        category,
        _normalize(category),
    )
    for container in (
        getattr(runtime, "env_kwargs", None),
        getattr(runtime, "runtime_kwargs", None),
        getattr(runtime, "config", None),
    ):
        if not isinstance(container, dict):
            continue
        overrides = container.get("scene_state_navigation_role_overrides")
        if not isinstance(overrides, dict):
            continue
        for key in lookup_keys:
            role = overrides.get(key)
            if isinstance(role, str) and role.strip().lower() in _VALID_NAVIGATION_ROLES:
                return role.strip().lower()
    return None


def _door_state(
    obj: Any,
    *,
    runtime: Any,
    name: str,
    category: str,
    include_aabb: bool = False,
    scene_transform: list[list[float]],
) -> RuntimeDoorState:
    del category
    return RuntimeDoorState(
        name=name,
        in_rooms=_object_in_rooms(obj),
        is_open=_object_open_state(obj),
        openness=_door_openness(obj),
        position=transform_position(_object_position_3d(obj), scene_transform),
        aabb=(transform_aabb(_object_aabb(obj), scene_transform) if include_aabb else None),
        collision_parts=(
            _cached_collision_parts(runtime, obj, name=name, include_root=False)
            if include_aabb
            else []
        ),
    )


def _object_in_rooms(obj: Any) -> list[str]:
    in_rooms = getattr(obj, "in_rooms", None)
    if isinstance(in_rooms, str):
        return [in_rooms] if in_rooms.strip() else []
    if isinstance(in_rooms, (list, tuple, set)):
        return sorted({str(room) for room in in_rooms if str(room or "").strip()})
    return []


def _door_openness(obj: Any) -> float | None:
    getter = getattr(obj, "get_joint_positions", None)
    if not callable(getter):
        return None
    try:
        values = [abs(float(item)) for item in getter()]
    except Exception:
        return None
    return max(values) if values else None


def _object_position_3d(obj: Any) -> dict[str, float] | None:
    getter = getattr(obj, "get_position_orientation", None)
    if callable(getter):
        try:
            position, _ = getter()
            return _position_from_sequence(position)
        except Exception:
            pass
    getter = getattr(obj, "get_position", None)
    if callable(getter):
        try:
            return _position_from_sequence(getter())
        except Exception:
            pass
    for attr in ("position", "pos"):
        position = _position_from_sequence(getattr(obj, attr, None))
        if position is not None:
            return position
    return None


def _position_from_sequence(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    try:
        return {
            "x": float(value[0]),
            "y": float(value[1]),
            "z": float(value[2]) if len(value) >= 3 else 0.0,
        }
    except (IndexError, TypeError, ValueError):
        return None


def _object_aabb(obj: Any) -> dict[str, list[float]] | None:
    aabb = getattr(obj, "aabb", None)
    if not isinstance(aabb, tuple) or len(aabb) != 2:
        return None
    corners: dict[str, list[float]] = {}
    for key, corner in zip(("min", "max"), aabb):
        try:
            corners[key] = [
                float(corner[0]),
                float(corner[1]),
                float(corner[2]) if len(corner) >= 3 else 0.0,
            ]
        except (IndexError, TypeError, ValueError):
            return None
    return corners


def _collision_parts(
    obj: Any,
    *,
    include_root: bool = False,
) -> list[dict[str, Any]]:
    links = getattr(obj, "links", None)
    if not isinstance(links, dict) or not links:
        return []
    root_link_name = str(getattr(obj, "root_link_name", "") or "")
    omit_root = not include_root and len(links) > 1 and bool(root_link_name)
    parts: list[dict[str, Any]] = []
    for link_name, link in sorted(links.items(), key=lambda item: str(item[0])):
        link_name = str(link_name)
        if omit_root and link_name == root_link_name:
            continue
        bounds = _collision_points_aabb(link)
        if bounds is not None:
            parts.append({"link": link_name, **bounds})
    return parts


def _cached_collision_parts(
    runtime: Any,
    obj: Any,
    *,
    name: str,
    include_root: bool,
) -> list[dict[str, Any]]:
    frame_contract = _scene_frame_contract(runtime)
    simulator_vertical_axis = frame_contract["simulator_vertical_axis"]
    scene_vertical_axis = frame_contract["scene_vertical_axis"]
    scene_transform = frame_contract["scene_from_simulator_transform"]
    enhanced = _collision_parts_from_cached_local_geometry(
        runtime,
        obj,
        name=name,
        include_root=include_root,
        simulator_vertical_axis=simulator_vertical_axis,
        scene_vertical_axis=scene_vertical_axis,
        scene_transform=scene_transform,
    )
    if enhanced is not None:
        return enhanced

    cache = getattr(runtime, "_scene_collision_parts_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        runtime._scene_collision_parts_cache = cache
    key = (
        name,
        bool(include_root),
        simulator_vertical_axis,
        scene_vertical_axis,
        _transform_signature(scene_transform),
    )
    signature = _collision_pose_signature(obj)
    cached = cache.get(key)
    if isinstance(cached, dict) and cached.get("signature") == signature:
        return list(cached.get("parts") or [])
    parts = _collision_parts(obj, include_root=include_root)
    parts = [
        transformed
        for part in parts
        for transformed in [
            _transform_collision_bounds(
                part,
                scene_transform=scene_transform,
                scene_vertical_axis=scene_vertical_axis,
                simulator_vertical_axis=simulator_vertical_axis,
            )
        ]
        if transformed is not None
    ]
    cache[key] = {"signature": signature, "parts": parts}
    return list(parts)


def _scene_frame_contract(runtime: Any) -> dict[str, Any]:
    return resolve_frame_contract(
        getattr(runtime, "env_kwargs", None),
        getattr(runtime, "runtime_kwargs", None),
        getattr(runtime, "config", None),
    )


def _collision_parts_from_cached_local_geometry(
    runtime: Any,
    obj: Any,
    *,
    name: str,
    include_root: bool,
    simulator_vertical_axis: str,
    scene_vertical_axis: str,
    scene_transform: list[list[float]],
) -> list[dict[str, Any]] | None:
    links = getattr(obj, "links", None)
    if not isinstance(links, dict) or not links:
        return None
    root_link_name = str(getattr(obj, "root_link_name", "") or "")
    omit_root = not include_root and len(links) > 1 and bool(root_link_name)
    geometry_cache = getattr(runtime, "_scene_collision_geometry_cache", None)
    if not isinstance(geometry_cache, dict):
        geometry_cache = {}
        runtime._scene_collision_geometry_cache = geometry_cache

    parts: list[dict[str, Any]] = []
    enhanced_any = False
    for link_name_raw, link in sorted(links.items(), key=lambda item: str(item[0])):
        link_name = str(link_name_raw)
        if omit_root and link_name == root_link_name:
            continue
        pose = _link_position_orientation(link)
        cache_key = (
            id(obj),
            name,
            link_name,
            simulator_vertical_axis,
            scene_vertical_axis,
            _transform_signature(scene_transform),
            str(
                getattr(link, "geometry_revision", None)
                or getattr(obj, "geometry_revision", None)
                or getattr(obj, "model", None)
                or ""
            ),
        )
        cached_geometry = geometry_cache.get(cache_key)
        if pose is not None and not isinstance(cached_geometry, dict):
            world_points = _collision_points_world(link)
            if world_points:
                position, orientation = pose
                local_points = [
                    _inverse_transform_point(point, position, orientation) for point in world_points
                ]
                local_polygon = _convex_hull_projected(
                    local_points,
                    vertical_axis=simulator_vertical_axis,
                )
                geometry_hash = _geometry_hash(local_points)
                cached_geometry = {
                    "local_points": local_points,
                    "local_polygons": [local_polygon] if local_polygon else [],
                    "geometry_hash": geometry_hash,
                }
                geometry_cache[cache_key] = cached_geometry

        if pose is not None and isinstance(cached_geometry, dict):
            local_points = list(cached_geometry.get("local_points") or [])
            if local_points:
                position, orientation = pose
                world_points = [
                    _transform_point(point, position, orientation) for point in local_points
                ]
                part = _collision_geometry_part(
                    obj,
                    name=name,
                    link_name=link_name,
                    world_points=world_points,
                    local_points=local_points,
                    local_polygons=list(cached_geometry.get("local_polygons") or []),
                    geometry_hash=str(cached_geometry.get("geometry_hash") or ""),
                    world_transform=_transform_matrix(position, orientation),
                    simulator_vertical_axis=simulator_vertical_axis,
                    scene_vertical_axis=scene_vertical_axis,
                    scene_transform=scene_transform,
                )
                if part is not None:
                    parts.append(part)
                    enhanced_any = True
                    continue

        world_points = _collision_points_world(link)
        if not world_points:
            continue
        part = _collision_geometry_part(
            obj,
            name=name,
            link_name=link_name,
            world_points=world_points,
            local_points=[],
            local_polygons=[],
            geometry_hash=hashlib.sha1(
                f"{name}:{link_name}:{len(world_points)}".encode("utf-8")
            ).hexdigest()[:16],
            world_transform=None,
            simulator_vertical_axis=simulator_vertical_axis,
            scene_vertical_axis=scene_vertical_axis,
            scene_transform=scene_transform,
        )
        if part is not None:
            parts.append(part)
    return parts if enhanced_any else None


def _collision_geometry_part(
    obj: Any,
    *,
    name: str,
    link_name: str,
    world_points: list[list[float]],
    local_points: list[list[float]],
    local_polygons: list[list[list[float]]],
    geometry_hash: str,
    world_transform: list[list[float]] | None,
    simulator_vertical_axis: str = "z",
    scene_vertical_axis: str = "z",
    scene_transform: list[list[float]] | None = None,
) -> dict[str, Any] | None:
    transform = scene_transform or frame_transform_for_vertical_axes(
        source_vertical_axis=simulator_vertical_axis,
        target_vertical_axis=scene_vertical_axis,
    )
    scene_points = [transform_point(point, transform) for point in world_points]
    bounds = _points_aabb(scene_points)
    world_polygon = _convex_hull_projected(
        scene_points,
        vertical_axis=scene_vertical_axis,
    )
    if bounds is None or not world_polygon:
        return None
    joint_metadata = _joint_metadata(obj, link_name)
    part: dict[str, Any] = {
        "link": link_name,
        **bounds,
        "geometry_id": f"{name}:{link_name}",
        "geometry_hash": geometry_hash,
        "geometry_source": "collision_mesh",
        "local_frame": link_name,
        "local_polygons": local_polygons,
        "world_polygons": [world_polygon],
        "height_min": min(
            point[vertical_axis_index(scene_vertical_axis)] for point in scene_points
        ),
        "height_max": max(
            point[vertical_axis_index(scene_vertical_axis)] for point in scene_points
        ),
        "vertical_axis": scene_vertical_axis,
        "source_vertical_axis": simulator_vertical_axis,
        "frame_id": "scene",
        "source_frame": "simulator",
        "geometry_revision": geometry_hash,
    }
    if local_points:
        part["local_points"] = local_points
    if world_transform is not None:
        part["world_transform"] = multiply_transforms(transform, world_transform)
        part["pose_revision"] = hashlib.sha1(
            json.dumps(part["world_transform"], separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
    part.update(joint_metadata)
    return part


def _collision_points_world(link: Any) -> list[list[float]]:
    try:
        points = getattr(link, "collision_boundary_points_world", None)
    except Exception:
        return []
    if callable(points):
        try:
            points = points()
        except Exception:
            return []
    if points is None:
        return []
    detach = getattr(points, "detach", None)
    if callable(detach):
        try:
            points = detach().cpu().tolist()
        except Exception:
            return []
    coordinates: list[list[float]] = []
    try:
        for point in points:
            coordinates.append([float(point[0]), float(point[1]), float(point[2])])
    except (IndexError, TypeError, ValueError):
        return []
    return coordinates


def _link_position_orientation(
    link: Any,
) -> tuple[list[float], list[float]] | None:
    getter = getattr(link, "get_position_orientation", None)
    if callable(getter):
        try:
            position, orientation = getter()
            normalized = _normalize_pose(position, orientation)
            if normalized is not None:
                return normalized
        except Exception:
            pass
    position = getattr(link, "position", None)
    orientation = getattr(link, "orientation", None)
    return _normalize_pose(position, orientation)


def _normalize_pose(
    position: Any,
    orientation: Any,
) -> tuple[list[float], list[float]] | None:
    try:
        normalized_position = [float(position[index]) for index in range(3)]
        normalized_orientation = [float(orientation[index]) for index in range(4)]
    except (IndexError, TypeError, ValueError):
        return None
    norm = math.sqrt(sum(value * value for value in normalized_orientation))
    if norm <= 1e-9:
        return None
    return normalized_position, [value / norm for value in normalized_orientation]


def _transform_point(
    point: list[float],
    position: list[float],
    orientation: list[float],
) -> list[float]:
    rotated = _rotate_vector(point, orientation)
    return [rotated[index] + position[index] for index in range(3)]


def _inverse_transform_point(
    point: list[float],
    position: list[float],
    orientation: list[float],
) -> list[float]:
    translated = [point[index] - position[index] for index in range(3)]
    inverse = [-orientation[0], -orientation[1], -orientation[2], orientation[3]]
    return list(_rotate_vector(translated, inverse))


def _rotate_vector(vector: Any, orientation: list[float]) -> tuple[float, float, float]:
    qx, qy, qz, qw = orientation
    vx, vy, vz = (float(vector[index]) for index in range(3))
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def _transform_matrix(
    position: list[float],
    orientation: list[float],
) -> list[list[float]]:
    qx, qy, qz, qw = orientation
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), position[0]],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), position[1]],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), position[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _convex_hull_projected(
    points: list[list[float]],
    *,
    vertical_axis: str,
) -> list[list[float]]:
    horizontal_indices = horizontal_axis_indices(vertical_axis)
    unique = sorted(
        {
            (
                round(point[horizontal_indices[0]], 6),
                round(point[horizontal_indices[1]], 6),
            )
            for point in points
        }
    )
    if len(unique) < 3:
        return []

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (
            second[0] - origin[0]
        )

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return [[float(x_coord), float(y_coord)] for x_coord, y_coord in lower[:-1] + upper[:-1]]


def _transform_signature(transform: list[list[float]]) -> str:
    return hashlib.sha1(json.dumps(transform, separators=(",", ":")).encode("utf-8")).hexdigest()[
        :16
    ]


def _transform_collision_bounds(
    part: dict[str, Any],
    *,
    scene_transform: list[list[float]],
    scene_vertical_axis: str,
    simulator_vertical_axis: str,
) -> dict[str, Any] | None:
    bounds = transform_aabb(
        {"min": part.get("min"), "max": part.get("max")},
        scene_transform,
    )
    if bounds is None:
        return None
    return {
        **part,
        **bounds,
        "vertical_axis": scene_vertical_axis,
        "source_vertical_axis": simulator_vertical_axis,
        "frame_id": "scene",
        "source_frame": "simulator",
    }


def _points_aabb(points: list[list[float]]) -> dict[str, list[float]] | None:
    if not points:
        return None
    return {
        "min": [min(point[axis] for point in points) for axis in range(3)],
        "max": [max(point[axis] for point in points) for axis in range(3)],
    }


def _geometry_hash(local_points: list[list[float]]) -> str:
    normalized = sorted([round(float(value), 6) for value in point[:3]] for point in local_points)
    return hashlib.sha1(json.dumps(normalized, separators=(",", ":")).encode("utf-8")).hexdigest()[
        :16
    ]


def _joint_metadata(obj: Any, link_name: str) -> dict[str, Any]:
    joints = getattr(obj, "joints", None)
    if not isinstance(joints, dict):
        return {}
    normalized_link = _normalize(link_name)
    for joint_name, joint in joints.items():
        tokens = " ".join(
            str(value or "")
            for value in (
                joint_name,
                getattr(joint, "name", None),
                getattr(joint, "child_link_name", None),
                getattr(joint, "body1", None),
            )
        ).lower()
        if normalized_link and normalized_link not in tokens:
            continue
        metadata: dict[str, Any] = {"parent_link": str(joint_name)}
        joint_type = getattr(joint, "joint_type", None) or getattr(joint, "type", None)
        if joint_type is not None:
            metadata["joint_type"] = str(joint_type)
        for getter_name in ("get_state", "get_position", "get_joint_position"):
            getter = getattr(joint, getter_name, None)
            if not callable(getter):
                continue
            try:
                value = getter()
                if isinstance(value, (list, tuple)):
                    value = value[0] if value else None
                if isinstance(value, (int, float)):
                    metadata["joint_position"] = float(value)
                    break
            except Exception:
                continue
        return metadata
    return {}


def _collision_pose_signature(obj: Any) -> tuple[Any, ...]:
    values: list[Any] = [id(obj)]
    getter = getattr(obj, "get_position_orientation", None)
    if callable(getter):
        try:
            position, orientation = getter()
            values.extend(_rounded_sequence(position))
            values.extend(_rounded_sequence(orientation))
        except Exception:
            pass
    joint_getter = getattr(obj, "get_joint_positions", None)
    if callable(joint_getter):
        try:
            values.extend(_rounded_sequence(joint_getter()))
        except Exception:
            pass
    return tuple(values)


def _rounded_sequence(value: Any) -> tuple[float, ...]:
    try:
        return tuple(round(float(item), 4) for item in value)
    except (TypeError, ValueError):
        return ()


def _collision_points_aabb(link: Any) -> dict[str, list[float]] | None:
    try:
        points = getattr(link, "collision_boundary_points_world", None)
    except Exception:
        return None
    if callable(points):
        try:
            points = points()
        except Exception:
            return None
    if points is None:
        return None
    detach = getattr(points, "detach", None)
    if callable(detach):
        try:
            points = detach().cpu().tolist()
        except Exception:
            return None
    coordinates: list[tuple[float, float, float]] = []
    try:
        for point in points:
            coordinates.append((float(point[0]), float(point[1]), float(point[2])))
    except (IndexError, TypeError, ValueError):
        return None
    if not coordinates:
        return None
    return {
        "min": [min(point[axis] for point in coordinates) for axis in range(3)],
        "max": [max(point[axis] for point in coordinates) for axis in range(3)],
    }


def _object_room_hint(obj: Any) -> str | None:
    rooms = _object_in_rooms(obj)
    return rooms[0] if rooms else None


def _object_floor_hint(obj: Any) -> str | None:
    for attribute in ("floor_id", "floor", "on_floor"):
        value = getattr(obj, attribute, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
        identifier = getattr(value, "floor_id", None)
        if isinstance(identifier, str) and identifier.strip():
            return identifier.strip()
    return None


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower()


__all__ = ["sample_scene_runtime_state"]
