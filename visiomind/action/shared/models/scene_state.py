from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

DOOR_OPEN_JOINT_THRESHOLD = 0.1
SIGNATURE_POSITION_QUANTUM_M = 0.1
NAVIGATION_ROLE_OBSTACLE = "obstacle"
NAVIGATION_ROLE_SUPPORT_SURFACE = "support_surface"
NAVIGATION_ROLE_STRUCTURAL = "structural"
NAVIGATION_ROLE_OVERHEAD = "overhead"
NON_BLOCKING_NAVIGATION_ROLES = {
    NAVIGATION_ROLE_SUPPORT_SURFACE,
    NAVIGATION_ROLE_STRUCTURAL,
    NAVIGATION_ROLE_OVERHEAD,
}
_STRUCTURAL_CATEGORY_PREFIXES = (
    "walls",
    "floors",
    "ceilings",
    "wall",
    "floor",
    "ceiling",
    "background",
)
_SUPPORT_SURFACE_CATEGORY_TOKENS = {
    "carpet",
    "deck",
    "driveway",
    "grass",
    "ground",
    "lawn",
    "mat",
    "patio",
    "pavement",
    "paver",
    "road",
    "rug",
    "sidewalk",
    "soil",
    "stair",
    "staircase",
    "terrain",
    "walkway",
}
_OVERHEAD_CATEGORY_TOKENS = {
    "canopy",
    "roof",
}


def is_door_category(value: Any) -> bool:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return False
    return normalized == "door" or normalized.split("_")[-1] == "door"


def navigation_role_from_category(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    tokens = set(normalized.split("_"))
    if any(normalized.startswith(prefix) for prefix in _STRUCTURAL_CATEGORY_PREFIXES):
        return NAVIGATION_ROLE_STRUCTURAL
    if tokens & _OVERHEAD_CATEGORY_TOKENS:
        return NAVIGATION_ROLE_OVERHEAD
    if tokens & _SUPPORT_SURFACE_CATEGORY_TOKENS:
        return NAVIGATION_ROLE_SUPPORT_SURFACE
    return NAVIGATION_ROLE_OBSTACLE


def door_is_open_from_joints(
    joint_positions: Any, *, threshold: float = DOOR_OPEN_JOINT_THRESHOLD
) -> bool | None:
    if not isinstance(joint_positions, (list, tuple)) or not joint_positions:
        return None
    values = []
    for value in joint_positions:
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return any(abs(value) >= threshold for value in values)


def door_is_navigation_passable(door: "RuntimeDoorState") -> bool | None:
    if door.navigation_passable is not None:
        return bool(door.navigation_passable)
    return door.is_open


def _navigation_passability_text(door: "RuntimeDoorState") -> str:
    state = door_is_navigation_passable(door)
    state_text = "passable" if state is True else "blocked" if state is False else "unknown"
    return "@".join(
        (
            state_text,
            door.navigation_passable_source or "physical_state",
            str(door.navigation_passable_revision or ""),
        )
    )


@dataclass
class RuntimeObjectState:
    name: str
    category: str | None = None
    navigation_role: str | None = None
    position: dict[str, float] | None = None
    aabb: dict[str, list[float]] | None = None
    oriented_bbox: dict[str, Any] | None = None
    covariance_xy: list[list[float]] | None = None
    collision_parts: list[dict[str, Any]] = field(default_factory=list)
    room_hint: str | None = None
    floor_hint: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name}
        if self.category:
            payload["category"] = self.category
        if self.navigation_role:
            payload["navigation_role"] = self.navigation_role
        if self.position is not None:
            payload["position"] = dict(self.position)
        if self.aabb is not None:
            payload["aabb"] = {key: list(value) for key, value in self.aabb.items()}
        if self.oriented_bbox is not None:
            payload["oriented_bbox"] = dict(self.oriented_bbox)
        if self.covariance_xy is not None:
            payload["covariance_xy"] = [list(row) for row in self.covariance_xy]
        if self.collision_parts:
            payload["collision_parts"] = _collision_parts_payload(self.collision_parts)
        if self.room_hint:
            payload["room_hint"] = self.room_hint
        if self.floor_hint:
            payload["floor_hint"] = self.floor_hint
        return payload


@dataclass
class RuntimeDoorState:
    name: str
    in_rooms: list[str] = field(default_factory=list)
    is_open: bool | None = None
    openness: float | None = None
    navigation_passable: bool | None = None
    navigation_passable_source: str | None = None
    navigation_passable_observed_at_step: int | None = None
    navigation_passable_revision: str | int | None = None
    position: dict[str, float] | None = None
    aabb: dict[str, list[float]] | None = None
    collision_parts: list[dict[str, Any]] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"name": self.name, "in_rooms": list(self.in_rooms)}
        if self.is_open is not None:
            payload["is_open"] = bool(self.is_open)
        if self.openness is not None:
            payload["openness"] = float(self.openness)
        if self.navigation_passable is not None:
            payload["navigation_passable"] = bool(self.navigation_passable)
        if self.navigation_passable_source:
            payload["navigation_passable_source"] = self.navigation_passable_source
        if self.navigation_passable_observed_at_step is not None:
            payload["navigation_passable_observed_at_step"] = int(
                self.navigation_passable_observed_at_step
            )
        if self.navigation_passable_revision is not None:
            payload["navigation_passable_revision"] = self.navigation_passable_revision
        if self.position is not None:
            payload["position"] = dict(self.position)
        if self.aabb is not None:
            payload["aabb"] = {key: list(value) for key, value in self.aabb.items()}
        if self.collision_parts:
            payload["collision_parts"] = _collision_parts_payload(self.collision_parts)
        return payload


@dataclass
class RuntimeObstacleState:
    obstacle_id: str
    polygons: list[list[list[float]]] = field(default_factory=list)
    source: str = "sensor"
    confidence: float | None = None
    covariance_xy: list[list[float]] | None = None
    expires_at_step: int | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "obstacle_id": self.obstacle_id,
            "source": self.source,
            "polygons": _coerce_polygons(self.polygons),
        }
        if self.confidence is not None:
            payload["confidence"] = float(self.confidence)
        if self.covariance_xy is not None:
            payload["covariance_xy"] = [list(row) for row in self.covariance_xy]
        if self.expires_at_step is not None:
            payload["expires_at_step"] = int(self.expires_at_step)
        return payload


@dataclass
class RuntimeRelationState:
    subject_id: str = ""
    relation: str = ""
    object_id: str | None = None
    confidence: float | None = None
    source: str | None = None
    observed_at_step: int | None = None
    expires_at_step: int | None = None
    revision: str | int | None = None
    removed: bool = False

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "subject_id": self.subject_id,
            "relation": self.relation,
        }
        if self.object_id is not None:
            payload["object_id"] = self.object_id
        if self.confidence is not None:
            payload["confidence"] = float(self.confidence)
        if self.source:
            payload["source"] = self.source
        if self.observed_at_step is not None:
            payload["observed_at_step"] = int(self.observed_at_step)
        if self.expires_at_step is not None:
            payload["expires_at_step"] = int(self.expires_at_step)
        if self.revision is not None:
            payload["revision"] = self.revision
        if self.removed:
            payload["removed"] = True
        return payload


@dataclass
class SceneRuntimeState:
    scene_id: str | None
    step: int = 0
    simulator_vertical_axis: str = "z"
    scene_vertical_axis: str | None = None
    scene_from_simulator_transform: list[list[float]] | None = None
    objects: dict[str, RuntimeObjectState] = field(default_factory=dict)
    doors: dict[str, RuntimeDoorState] = field(default_factory=dict)
    relations: list[RuntimeRelationState] = field(default_factory=list)
    relation_signature: str = ""
    temporary_obstacles: list[RuntimeObstacleState] = field(default_factory=list)
    signature: str = ""

    def door_signature(self) -> str:
        parts = []
        for name, door in sorted(self.doors.items()):
            state_text = (
                "open" if door.is_open is True else "closed" if door.is_open is False else "unknown"
            )
            parts.append(
                f"{name}={state_text}"
                f"/nav={_navigation_passability_text(door)}"
                f"@{_collision_geometry_text(door.collision_parts, door.aabb, SIGNATURE_POSITION_QUANTUM_M)}"
            )
        return hashlib.sha1(";".join(parts).encode("utf-8")).hexdigest()[:16] if parts else ""

    def to_payload(self) -> dict[str, Any]:
        relation_signature = self.relation_signature or compute_relation_signature(
            self.relations,
            current_step=self.step,
        )
        return {
            "scene_id": self.scene_id,
            "step": int(self.step),
            "simulator_vertical_axis": self.simulator_vertical_axis,
            "scene_vertical_axis": self.scene_vertical_axis,
            "scene_from_simulator_transform": self.scene_from_simulator_transform,
            "objects": {name: state.to_payload() for name, state in self.objects.items()},
            "doors": {name: state.to_payload() for name, state in self.doors.items()},
            "relations": [relation.to_payload() for relation in self.relations],
            "relation_signature": relation_signature,
            "temporary_obstacles": [obstacle.to_payload() for obstacle in self.temporary_obstacles],
            "signature": self.signature,
        }


def compute_scene_state_signature(
    *,
    objects: dict[str, RuntimeObjectState],
    doors: dict[str, RuntimeDoorState],
    relations: list[RuntimeRelationState] | None = None,
    temporary_obstacles: list[RuntimeObstacleState] | None = None,
    simulator_vertical_axis: str | None = None,
    scene_vertical_axis: str | None = None,
    scene_from_simulator_transform: list[list[float]] | None = None,
    position_quantum_m: float = SIGNATURE_POSITION_QUANTUM_M,
) -> str:
    quantum = max(1e-6, float(position_quantum_m))
    parts: list[str] = []
    if simulator_vertical_axis or scene_vertical_axis or scene_from_simulator_transform:
        parts.append(
            "frame:"
            f"{simulator_vertical_axis or ''}->{scene_vertical_axis or ''}@"
            f"{_quantize_nested_numbers(scene_from_simulator_transform, quantum)}"
        )
    for name, door in sorted(doors.items()):
        state_text = "open" if door.is_open else "closed" if door.is_open is not None else "unknown"
        geometry_text = _collision_geometry_text(door.collision_parts, door.aabb, quantum)
        parts.append(
            f"door:{name}={state_text}/nav={_navigation_passability_text(door)}@{geometry_text}"
        )
    for name, obj in sorted(objects.items()):
        geometry_text = _collision_geometry_text(obj.collision_parts, obj.aabb, quantum)
        parts.append(
            f"obj:{name}@{obj.navigation_role or ''}@"
            f"{_quantized_position_text(obj.position, quantum)}@{geometry_text}@"
            f"{_oriented_bbox_text(obj.oriented_bbox, quantum)}@"
            f"{_covariance_text(obj.covariance_xy, quantum)}@"
            f"{obj.room_hint or ''}@{obj.floor_hint or ''}"
        )
    relation_signature = compute_relation_signature(relations or [])
    if relation_signature:
        parts.append(f"relations:{relation_signature}")
    for obstacle in sorted(temporary_obstacles or [], key=lambda item: item.obstacle_id):
        parts.append(
            "sensor:"
            f"{obstacle.obstacle_id}@{obstacle.source}@"
            f"{_quantized_polygons_text(obstacle.polygons, quantum)}@"
            f"{obstacle.expires_at_step}"
        )
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


def compute_relation_signature(
    relations: list[RuntimeRelationState],
    *,
    current_step: int | None = None,
) -> str:
    parts = []
    for relation in relations:
        if (
            current_step is not None
            and relation.expires_at_step is not None
            and current_step > relation.expires_at_step
        ):
            continue
        if not relation.subject_id or not relation.relation:
            continue
        parts.append(
            "@".join(
                (
                    relation.subject_id,
                    relation.relation,
                    relation.object_id or "",
                    relation.source or "",
                    str(relation.revision or ""),
                    "removed" if relation.removed else "active",
                )
            )
        )
    if not parts:
        return ""
    return hashlib.sha1("|".join(sorted(parts)).encode("utf-8")).hexdigest()[:16]


def _quantized_position_text(position: dict[str, float] | None, quantum: float) -> str:
    if not isinstance(position, dict):
        return "none"
    cells = []
    for axis in ("x", "y", "z"):
        try:
            cells.append(str(int(round(float(position.get(axis, 0.0)) / quantum))))
        except (TypeError, ValueError):
            cells.append("0")
    return ",".join(cells)


def _quantized_aabb_text(aabb: dict[str, list[float]] | None, quantum: float) -> str:
    if not isinstance(aabb, dict):
        return "none"
    corners: list[str] = []
    for key in ("min", "max"):
        corner = aabb.get(key)
        if not isinstance(corner, (list, tuple)) or len(corner) < 2:
            return "none"
        values: list[str] = []
        for index in range(3):
            try:
                value = float(corner[index]) if index < len(corner) else 0.0
                values.append(str(int(round(value / quantum))))
            except (TypeError, ValueError):
                values.append("0")
        corners.append(",".join(values))
    return ":".join(corners)


def _oriented_bbox_text(value: dict[str, Any] | None, quantum: float) -> str:
    if not isinstance(value, dict):
        return "none"
    return hashlib.sha1(repr(_quantize_nested_numbers(value, quantum)).encode("utf-8")).hexdigest()[
        :12
    ]


def _covariance_text(value: list[list[float]] | None, quantum: float) -> str:
    if not isinstance(value, list):
        return "none"
    return repr(_quantize_nested_numbers(value, quantum))


def _quantize_nested_numbers(value: Any, quantum: float) -> Any:
    if isinstance(value, dict):
        return tuple(
            (str(key), _quantize_nested_numbers(item, quantum))
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_quantize_nested_numbers(item, quantum) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(round(float(value) / quantum))
    return str(value)


def _collision_geometry_text(
    collision_parts: list[dict[str, Any]],
    fallback_aabb: dict[str, list[float]] | None,
    quantum: float,
) -> str:
    if collision_parts:
        parts = []
        for part in sorted(collision_parts, key=lambda item: str(item.get("link") or "")):
            link = str(part.get("link") or "")
            geometry_hash = str(part.get("geometry_hash") or "")
            polygons_text = _quantized_polygons_text(part.get("world_polygons"), quantum)
            geometry_text = (
                polygons_text if polygons_text != "none" else _quantized_aabb_text(part, quantum)
            )
            parts.append(f"{link}:{geometry_hash}:{geometry_text}")
        return ";".join(parts)
    return _quantized_aabb_text(fallback_aabb, quantum)


def _quantized_polygons_text(value: Any, quantum: float) -> str:
    polygons = _coerce_polygons(value)
    if not polygons:
        return "none"
    encoded = []
    for polygon in polygons:
        encoded.append(
            ";".join(
                f"{int(round(point[0] / quantum))},{int(round(point[1] / quantum))}"
                for point in polygon
            )
        )
    return "/".join(encoded)


def _collision_parts_payload(
    collision_parts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for part in collision_parts:
        if not isinstance(part, dict):
            continue
        payload: dict[str, Any] = {"link": str(part.get("link") or "")}
        bounds = _coerce_aabb(part)
        if bounds is not None:
            payload.update(bounds)
        for key in (
            "geometry_id",
            "geometry_hash",
            "geometry_source",
            "local_frame",
            "parent_link",
            "joint_type",
            "vertical_axis",
            "source_vertical_axis",
            "frame_id",
            "source_frame",
        ):
            value = _optional_text(part.get(key))
            if value is not None:
                payload[key] = value
        for key in ("height_min", "height_max", "joint_position"):
            value = part.get(key)
            if isinstance(value, (int, float)):
                payload[key] = float(value)
        for key in ("geometry_revision", "pose_revision"):
            value = part.get(key)
            if isinstance(value, (str, int)):
                payload[key] = value
        local_points = _coerce_points_3d(part.get("local_points"))
        if local_points:
            payload["local_points"] = local_points
        for key in ("local_polygons", "world_polygons"):
            polygons = _coerce_polygons(part.get(key))
            if polygons:
                payload[key] = polygons
        transform = _coerce_transform(part.get("world_transform"))
        if transform is not None:
            payload["world_transform"] = transform
        if len(payload) > 1:
            payloads.append(payload)
    return payloads


def scene_runtime_state_from_payload(payload: Any) -> SceneRuntimeState | None:
    if not isinstance(payload, dict):
        return None
    objects: dict[str, RuntimeObjectState] = {}
    for name, entry in (
        (payload.get("objects") or {}).items() if isinstance(payload.get("objects"), dict) else []
    ):
        state = _object_state_from_entry(name, entry)
        if state is not None:
            objects[state.name] = state
    doors: dict[str, RuntimeDoorState] = {}
    for name, entry in (
        (payload.get("doors") or {}).items() if isinstance(payload.get("doors"), dict) else []
    ):
        state = _door_state_from_entry(name, entry)
        if state is not None:
            doors[state.name] = state
    temporary_obstacles = _coerce_temporary_obstacles(payload.get("temporary_obstacles"))
    relations = _coerce_relations(payload.get("relations"))
    if (
        not objects
        and not doors
        and not relations
        and not temporary_obstacles
        and not any(
            key in payload for key in ("scene_id", "step", "signature", "relation_signature")
        )
    ):
        return None
    signature = str(payload.get("signature") or "")
    if not signature:
        simulator_vertical_axis = _optional_axis(payload.get("simulator_vertical_axis")) or "z"
        scene_vertical_axis = _optional_axis(payload.get("scene_vertical_axis"))
        scene_from_simulator_transform = _coerce_transform(
            payload.get("scene_from_simulator_transform")
        )
        signature = compute_scene_state_signature(
            objects=objects,
            doors=doors,
            relations=relations,
            temporary_obstacles=temporary_obstacles,
            simulator_vertical_axis=simulator_vertical_axis,
            scene_vertical_axis=scene_vertical_axis,
            scene_from_simulator_transform=scene_from_simulator_transform,
        )
    else:
        simulator_vertical_axis = _optional_axis(payload.get("simulator_vertical_axis")) or "z"
        scene_vertical_axis = _optional_axis(payload.get("scene_vertical_axis"))
        scene_from_simulator_transform = _coerce_transform(
            payload.get("scene_from_simulator_transform")
        )
    step = _to_int(payload.get("step"))
    relation_signature = str(payload.get("relation_signature") or "")
    if not relation_signature:
        relation_signature = compute_relation_signature(
            relations,
            current_step=step,
        )
    return SceneRuntimeState(
        scene_id=_optional_text(payload.get("scene_id")),
        step=step,
        simulator_vertical_axis=simulator_vertical_axis,
        scene_vertical_axis=scene_vertical_axis,
        scene_from_simulator_transform=scene_from_simulator_transform,
        objects=objects,
        doors=doors,
        relations=relations,
        relation_signature=relation_signature,
        temporary_obstacles=temporary_obstacles,
        signature=signature,
    )


def _object_state_from_entry(name: Any, entry: Any) -> RuntimeObjectState | None:
    if not isinstance(entry, dict):
        return None
    object_name = _optional_text(entry.get("name")) or _optional_text(name)
    if object_name is None:
        return None
    return RuntimeObjectState(
        name=object_name,
        category=_optional_text(entry.get("category")),
        navigation_role=_optional_text(entry.get("navigation_role")),
        position=_coerce_position(entry.get("position")),
        aabb=_coerce_aabb(entry.get("aabb")),
        oriented_bbox=_coerce_oriented_bbox(entry.get("oriented_bbox")),
        covariance_xy=_coerce_covariance_xy(entry.get("covariance_xy")),
        collision_parts=_coerce_collision_parts(entry.get("collision_parts")),
        room_hint=_optional_text(entry.get("room_hint")),
        floor_hint=_optional_text(entry.get("floor_hint")),
    )


def _door_state_from_entry(name: Any, entry: Any) -> RuntimeDoorState | None:
    if not isinstance(entry, dict):
        return None
    door_name = _optional_text(entry.get("name")) or _optional_text(name)
    if door_name is None:
        return None
    in_rooms_raw = entry.get("in_rooms")
    in_rooms = (
        [str(room) for room in in_rooms_raw if str(room or "").strip()]
        if isinstance(in_rooms_raw, (list, tuple))
        else []
    )
    is_open = entry.get("is_open")
    openness = entry.get("openness")
    navigation_passable = entry.get("navigation_passable")
    navigation_passable_observed_at_step = entry.get("navigation_passable_observed_at_step")
    navigation_passable_revision = entry.get("navigation_passable_revision")
    return RuntimeDoorState(
        name=door_name,
        in_rooms=in_rooms,
        is_open=bool(is_open) if isinstance(is_open, bool) else None,
        openness=float(openness) if isinstance(openness, (int, float)) else None,
        navigation_passable=(
            bool(navigation_passable) if isinstance(navigation_passable, bool) else None
        ),
        navigation_passable_source=_optional_text(entry.get("navigation_passable_source")),
        navigation_passable_observed_at_step=(
            int(navigation_passable_observed_at_step)
            if isinstance(navigation_passable_observed_at_step, (int, float))
            else None
        ),
        navigation_passable_revision=(
            navigation_passable_revision
            if isinstance(navigation_passable_revision, (str, int))
            else None
        ),
        position=_coerce_position(entry.get("position")),
        aabb=_coerce_aabb(entry.get("aabb")),
        collision_parts=_coerce_collision_parts(entry.get("collision_parts")),
    )


def _coerce_position(value: Any) -> dict[str, float] | None:
    if isinstance(value, dict):
        try:
            return {
                "x": float(value["x"]),
                "y": float(value["y"]),
                "z": float(value.get("z", 0.0)),
            }
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return {
                "x": float(value[0]),
                "y": float(value[1]),
                "z": float(value[2]) if len(value) >= 3 else 0.0,
            }
        except (TypeError, ValueError):
            return None
    return None


def _coerce_aabb(value: Any) -> dict[str, list[float]] | None:
    if not isinstance(value, dict):
        return None
    corners: dict[str, list[float]] = {}
    for key in ("min", "max"):
        corner = value.get(key)
        if not isinstance(corner, (list, tuple)) or len(corner) < 2:
            return None
        try:
            corners[key] = [float(item) for item in corner[:3]]
        except (TypeError, ValueError):
            return None
        if len(corners[key]) == 2:
            corners[key].append(0.0)
    return corners


def _coerce_collision_parts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    parts: list[dict[str, Any]] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        bounds = _coerce_aabb(entry)
        local_polygons = _coerce_polygons(entry.get("local_polygons"))
        world_polygons = _coerce_polygons(entry.get("world_polygons"))
        if bounds is None and not local_polygons and not world_polygons:
            continue
        part: dict[str, Any] = {
            "link": _optional_text(entry.get("link")) or "",
        }
        if bounds is not None:
            part.update(bounds)
        for key in (
            "geometry_id",
            "geometry_hash",
            "geometry_source",
            "local_frame",
            "parent_link",
            "joint_type",
            "vertical_axis",
            "source_vertical_axis",
            "frame_id",
            "source_frame",
        ):
            text = _optional_text(entry.get(key))
            if text is not None:
                part[key] = text
        for key in ("height_min", "height_max", "joint_position"):
            raw = entry.get(key)
            if isinstance(raw, (int, float)):
                part[key] = float(raw)
        for key in ("geometry_revision", "pose_revision"):
            raw = entry.get(key)
            if isinstance(raw, (str, int)):
                part[key] = raw
        local_points = _coerce_points_3d(entry.get("local_points"))
        if local_points:
            part["local_points"] = local_points
        if local_polygons:
            part["local_polygons"] = local_polygons
        if world_polygons:
            part["world_polygons"] = world_polygons
        transform = _coerce_transform(entry.get("world_transform"))
        if transform is not None:
            part["world_transform"] = transform
        parts.append(part)
    return parts


def _coerce_polygons(value: Any) -> list[list[list[float]]]:
    if not isinstance(value, (list, tuple)):
        return []
    polygons: list[list[list[float]]] = []
    for raw_polygon in value:
        if not isinstance(raw_polygon, (list, tuple)):
            continue
        polygon: list[list[float]] = []
        for raw_point in raw_polygon:
            if isinstance(raw_point, dict):
                try:
                    polygon.append([float(raw_point["x"]), float(raw_point["y"])])
                except (KeyError, TypeError, ValueError):
                    continue
            elif isinstance(raw_point, (list, tuple)) and len(raw_point) >= 2:
                try:
                    polygon.append([float(raw_point[0]), float(raw_point[1])])
                except (TypeError, ValueError):
                    continue
        if len(polygon) >= 3:
            polygons.append(polygon)
    return polygons


def _coerce_points_3d(value: Any) -> list[list[float]]:
    if not isinstance(value, (list, tuple)):
        return []
    points: list[list[float]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 3:
            continue
        try:
            points.append([float(point[0]), float(point[1]), float(point[2])])
        except (TypeError, ValueError):
            continue
    return points


def _coerce_transform(value: Any) -> list[list[float]] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    matrix: list[list[float]] = []
    for row in value[:4]:
        if not isinstance(row, (list, tuple)) or len(row) < 4:
            return None
        try:
            matrix.append([float(item) for item in row[:4]])
        except (TypeError, ValueError):
            return None
    return matrix


def _coerce_temporary_obstacles(value: Any) -> list[RuntimeObstacleState]:
    entries = list(value.values()) if isinstance(value, dict) else value
    if not isinstance(entries, (list, tuple)):
        return []
    obstacles: list[RuntimeObstacleState] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        polygons = _coerce_polygons(entry.get("polygons"))
        if not polygons:
            continue
        obstacle_id = (
            _optional_text(entry.get("obstacle_id"))
            or _optional_text(entry.get("id"))
            or f"sensor_{index}"
        )
        confidence = entry.get("confidence")
        covariance = _coerce_covariance_xy(entry.get("covariance_xy"))
        expires_at_step = entry.get("expires_at_step")
        obstacles.append(
            RuntimeObstacleState(
                obstacle_id=obstacle_id,
                polygons=polygons,
                source=_optional_text(entry.get("source")) or "sensor",
                confidence=(float(confidence) if isinstance(confidence, (int, float)) else None),
                covariance_xy=covariance,
                expires_at_step=(
                    int(expires_at_step) if isinstance(expires_at_step, (int, float)) else None
                ),
            )
        )
    return obstacles


def _coerce_relations(value: Any) -> list[RuntimeRelationState]:
    entries = list(value.values()) if isinstance(value, dict) else value
    if not isinstance(entries, (list, tuple)):
        return []
    relations: list[RuntimeRelationState] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        subject_id = _optional_text(entry.get("subject_id") or entry.get("subject"))
        relation = _optional_text(entry.get("relation") or entry.get("predicate"))
        if subject_id is None or relation is None:
            continue
        confidence = entry.get("confidence")
        observed_at_step = entry.get("observed_at_step")
        expires_at_step = entry.get("expires_at_step")
        revision = entry.get("revision")
        relations.append(
            RuntimeRelationState(
                subject_id=subject_id,
                relation=relation.strip().lower(),
                object_id=_optional_text(entry.get("object_id") or entry.get("object")),
                confidence=(float(confidence) if isinstance(confidence, (int, float)) else None),
                source=_optional_text(entry.get("source")),
                observed_at_step=(
                    int(observed_at_step) if isinstance(observed_at_step, (int, float)) else None
                ),
                expires_at_step=(
                    int(expires_at_step) if isinstance(expires_at_step, (int, float)) else None
                ),
                revision=(revision if isinstance(revision, (str, int)) else None),
                removed=bool(entry.get("removed") or entry.get("tombstone")),
            )
        )
    return relations


def _coerce_oriented_bbox(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    payload: dict[str, Any] = {}
    center = _coerce_position(value.get("center"))
    if center is not None:
        payload["center"] = center
    for key in ("size", "half_extents"):
        raw = value.get(key)
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            try:
                payload[key] = [float(item) for item in raw[:3]]
            except (TypeError, ValueError):
                pass
    corners = value.get("world_corners") or value.get("corners")
    if isinstance(corners, (list, tuple)):
        points = []
        for point in corners:
            coerced = _coerce_position(point)
            if coerced is not None:
                points.append(coerced)
        if points:
            payload["world_corners"] = points
    yaw = value.get("yaw")
    if isinstance(yaw, (int, float)):
        payload["yaw"] = float(yaw)
    return payload or None


def _coerce_covariance_xy(value: Any) -> list[list[float]] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    matrix: list[list[float]] = []
    for row in value[:2]:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            return None
        try:
            matrix.append([float(row[0]), float(row[1])])
        except (TypeError, ValueError):
            return None
    return matrix


def _optional_text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _optional_axis(value: Any) -> str | None:
    text = _optional_text(value)
    return text.lower() if text and text.lower() in {"x", "y", "z"} else None


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


__all__ = [
    "NAVIGATION_ROLE_OBSTACLE",
    "NAVIGATION_ROLE_OVERHEAD",
    "NAVIGATION_ROLE_STRUCTURAL",
    "NAVIGATION_ROLE_SUPPORT_SURFACE",
    "NON_BLOCKING_NAVIGATION_ROLES",
    "DOOR_OPEN_JOINT_THRESHOLD",
    "RuntimeDoorState",
    "RuntimeObjectState",
    "RuntimeObstacleState",
    "RuntimeRelationState",
    "SceneRuntimeState",
    "compute_relation_signature",
    "compute_scene_state_signature",
    "door_is_navigation_passable",
    "door_is_open_from_joints",
    "is_door_category",
    "navigation_role_from_category",
    "scene_runtime_state_from_payload",
]
