from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from visiomind.action.integrations.simulator.behavior.tools.bridge_environment import (
    _env_candidates,
    _object_open_state,
)
from visiomind.action.shared.geometry_frames import (
    horizontal_axis_indices,
    vertical_axis_index,
)
from visiomind.action.shared.models.scene_state import (
    RuntimeDoorState,
    RuntimeObjectState,
    RuntimeRelationState,
)

_BUCKET_SIZE_M = 2.0
_ON_TOP_ESTABLISH_GAP_M = 0.08
_ON_TOP_RETAIN_GAP_M = 0.15
_ON_TOP_MIN_OVERLAP = 0.25
_INSIDE_ESTABLISH_RATIO = 0.85
_INSIDE_RETAIN_RATIO = 0.70
_NEAR_ESTABLISH_DISTANCE_M = 1.0
_NEAR_RETAIN_DISTANCE_M = 1.25
_CONTAINER_CATEGORY_TOKENS = {
    "basket",
    "bin",
    "cabinet",
    "closet",
    "container",
    "cupboard",
    "dishwasher",
    "drawer",
    "freezer",
    "fridge",
    "refrigerator",
    "suitcase",
    "trunk",
    "wardrobe",
}
_SUPPORT_CATEGORY_TOKENS = {
    "bench",
    "counter",
    "countertop",
    "desk",
    "platform",
    "rack",
    "shelf",
    "stand",
    "table",
}


def sample_scene_relations(
    runtime: Any,
    *,
    scene_objects: list[Any],
    objects: dict[str, RuntimeObjectState],
    doors: dict[str, RuntimeDoorState],
    step: int,
    vertical_axis: str = "z",
) -> list[RuntimeRelationState]:
    raw_by_name = {
        str(getattr(obj, "name", "") or "").strip(): obj
        for obj in scene_objects
        if str(getattr(obj, "name", "") or "").strip()
    }
    local_pairs = _local_object_pairs(objects, vertical_axis=vertical_axis)
    relations = _location_relations(objects, step=step)
    explicit_relations, authoritative_relative_states = _explicit_object_relations(
        raw_by_name=raw_by_name,
        objects=objects,
        local_pairs=local_pairs,
        step=step,
    )
    relations.extend(explicit_relations)
    relations.extend(
        _held_relations(
            runtime,
            scene_objects=scene_objects,
            objects=objects,
            step=step,
        )
    )
    relations.extend(
        _geometric_relations(
            runtime,
            objects=objects,
            raw_by_name=raw_by_name,
            authoritative_relative_states=authoritative_relative_states,
            local_pairs=local_pairs,
            step=step,
            vertical_axis=vertical_axis,
        )
    )
    for door in doors.values():
        if door.is_open is None:
            continue
        relations.append(
            RuntimeRelationState(
                subject_id=door.name,
                relation="open" if door.is_open else "closed",
                confidence=1.0,
                source="simulator",
                observed_at_step=step,
                revision="open" if door.is_open else "closed",
            )
        )
    return relations


def _location_relations(
    objects: dict[str, RuntimeObjectState],
    *,
    step: int,
) -> list[RuntimeRelationState]:
    relations = []
    for obj in objects.values():
        if obj.room_hint:
            relations.append(
                RuntimeRelationState(
                    subject_id=obj.name,
                    relation="in_room",
                    object_id=obj.room_hint,
                    confidence=1.0,
                    source="simulator",
                    observed_at_step=step,
                    revision=obj.room_hint,
                )
            )
        if obj.floor_hint:
            relations.append(
                RuntimeRelationState(
                    subject_id=obj.name,
                    relation="on_floor",
                    object_id=obj.floor_hint,
                    confidence=1.0,
                    source="simulator",
                    observed_at_step=step,
                    revision=obj.floor_hint,
                )
            )
    return relations


def _explicit_object_relations(
    *,
    raw_by_name: dict[str, Any],
    objects: dict[str, RuntimeObjectState],
    local_pairs: list[tuple[RuntimeObjectState, RuntimeObjectState]],
    step: int,
) -> tuple[list[RuntimeRelationState], set[tuple[str, str, str]]]:
    relation_attributes = {
        "inside": ("inside_of", "container", "contained_in"),
        "on_top": ("on_top_of", "supporting_object", "support_object"),
        "attached_to": ("attached_to", "attachment_parent"),
        "held_by": ("held_by", "grasped_by"),
    }
    relations = []
    authoritative_relative_states: set[tuple[str, str, str]] = set()
    pairs_by_subject: dict[str, list[RuntimeObjectState]] = defaultdict(list)
    for subject_state, owner_state in local_pairs:
        pairs_by_subject[subject_state.name].append(owner_state)
    for name in objects:
        raw = raw_by_name.get(name)
        if raw is None:
            continue
        for relation_name, attributes in relation_attributes.items():
            target = next(
                (
                    _object_reference_name(getattr(raw, attribute, None))
                    for attribute in attributes
                    if _object_reference_name(getattr(raw, attribute, None))
                ),
                None,
            )
            if target is None:
                continue
            relations.append(
                RuntimeRelationState(
                    subject_id=name,
                    relation=relation_name,
                    object_id=target,
                    confidence=1.0,
                    source="simulator",
                    observed_at_step=step,
                    revision=f"{relation_name}:{target}",
                )
            )
        for owner_state in pairs_by_subject.get(name, []):
            owner_name = owner_state.name
            owner = raw_by_name.get(owner_name)
            if owner is None:
                continue
            for relation_name, state_name in (
                ("inside", "Inside"),
                ("on_top", "OnTop"),
                ("attached_to", "AttachedTo"),
            ):
                official_value = _object_relative_state(raw, state_name, owner)
                if official_value is None:
                    continue
                authoritative_relative_states.add((name, relation_name, owner_name))
                if official_value:
                    relations.append(
                        RuntimeRelationState(
                            subject_id=name,
                            relation=relation_name,
                            object_id=owner_name,
                            confidence=1.0,
                            source="simulator_object_state",
                            observed_at_step=step,
                            revision=f"{relation_name}:{owner_name}",
                        )
                    )
                else:
                    relations = [
                        relation
                        for relation in relations
                        if not (
                            relation.subject_id == name
                            and relation.relation == relation_name
                            and relation.object_id == owner_name
                        )
                    ]
        open_state = _object_open_state(raw)
        if open_state is not None:
            relations.append(
                RuntimeRelationState(
                    subject_id=name,
                    relation="open" if open_state else "closed",
                    confidence=1.0,
                    source="simulator",
                    observed_at_step=step,
                    revision="open" if open_state else "closed",
                )
            )
    return relations, authoritative_relative_states


def _held_relations(
    runtime: Any,
    *,
    scene_objects: list[Any],
    objects: dict[str, RuntimeObjectState],
    step: int,
) -> list[RuntimeRelationState]:
    controlled_robot = None
    for candidate in _env_candidates(getattr(runtime, "_env", None)):
        robots = getattr(candidate, "robots", None)
        if isinstance(robots, (list, tuple)) and robots:
            controlled_robot = robots[0]
            break
    if controlled_robot is None:
        return []
    held_names = _held_object_names(
        controlled_robot,
        scene_objects=scene_objects,
    )
    if not held_names:
        return []
    known_names = set(objects)
    known_names.update(str(getattr(obj, "name", "") or "").strip() for obj in scene_objects)
    return [
        RuntimeRelationState(
            subject_id=name,
            relation="held_by",
            object_id="self_robot",
            confidence=1.0,
            source="task_state",
            observed_at_step=step,
            revision="held_by:self_robot",
        )
        for name in sorted(held_names & known_names)
    ]


def _held_object_names(robot: Any, *, scene_objects: list[Any]) -> set[str]:
    candidates = []
    for attribute in ("grasped_objects", "held_objects", "objects_in_hand"):
        value = getattr(robot, attribute, None)
        if value is not None:
            candidates.append(value)
    assisted_grasp_objects = getattr(robot, "_ag_obj_in_hand", None)
    if isinstance(assisted_grasp_objects, dict):
        for arm, obj in assisted_grasp_objects.items():
            if obj is None:
                continue
            if _robot_reports_grasping(robot, arm=str(arm), candidate_obj=obj) is False:
                continue
            candidates.append(obj)
    for method_name in ("get_grasped_objects", "get_held_objects"):
        method = getattr(robot, method_name, None)
        if callable(method):
            try:
                candidates.append(method())
            except Exception:
                continue
    names = set()
    for value in candidates:
        entries = value.values() if isinstance(value, dict) else value
        if not isinstance(entries, (list, tuple, set)):
            entries = [entries]
        for entry in entries:
            if isinstance(entry, (list, tuple, set)):
                for nested in entry:
                    name = _object_reference_name(nested)
                    if name:
                        names.add(name)
                continue
            name = _object_reference_name(entry)
            if name:
                names.add(name)
    arms = getattr(robot, "arm_names", None)
    arm_names = list(arms) if isinstance(arms, (list, tuple)) and arms else ["default"]
    for arm in arm_names:
        if (
            _robot_reports_grasping(
                robot,
                arm=str(arm),
                candidate_obj=None,
            )
            is not True
        ):
            continue
        for obj in scene_objects:
            if _robot_reports_grasping(
                robot,
                arm=str(arm),
                candidate_obj=obj,
            ):
                name = _object_reference_name(obj)
                if name:
                    names.add(name)
    return names


def _geometric_relations(
    runtime: Any,
    *,
    objects: dict[str, RuntimeObjectState],
    raw_by_name: dict[str, Any],
    authoritative_relative_states: set[tuple[str, str, str]],
    local_pairs: list[tuple[RuntimeObjectState, RuntimeObjectState]],
    step: int,
    vertical_axis: str,
) -> list[RuntimeRelationState]:
    relations = []
    del objects
    for subject, owner in local_pairs:
        if subject.aabb is None or owner.aabb is None:
            continue
        inside_key = (subject.name, "inside", owner.name)
        inside_authoritative = inside_key in authoritative_relative_states
        if inside_authoritative:
            _clear_relation_hysteresis(runtime, inside_key)
        inside_established = _relation_is_established(runtime, inside_key)
        inside_ratio = _aabb_containment_ratio(subject.aabb, owner.aabb)
        inside_observed = (
            not inside_authoritative
            and _is_container_capable(raw_by_name.get(owner.name), owner)
            and inside_ratio
            >= (_INSIDE_RETAIN_RATIO if inside_established else _INSIDE_ESTABLISH_RATIO)
        )
        if not inside_authoritative and _update_relation_hysteresis(
            runtime,
            inside_key,
            inside_observed,
        ):
            relations.append(
                RuntimeRelationState(
                    subject_id=subject.name,
                    relation="inside",
                    object_id=owner.name,
                    confidence=min(1.0, inside_ratio),
                    source="geometry",
                    observed_at_step=step,
                    revision=f"inside:{owner.name}",
                )
            )
            continue

        on_top_key = (subject.name, "on_top", owner.name)
        on_top_authoritative = on_top_key in authoritative_relative_states
        if on_top_authoritative:
            _clear_relation_hysteresis(runtime, on_top_key)
        on_top_established = _relation_is_established(runtime, on_top_key)
        gap, overlap = _on_top_metrics(
            subject.aabb,
            owner.aabb,
            vertical_axis=vertical_axis,
        )
        gap_limit = _ON_TOP_RETAIN_GAP_M if on_top_established else _ON_TOP_ESTABLISH_GAP_M
        on_top_observed = (
            not on_top_authoritative
            and _is_reliable_support(raw_by_name.get(owner.name), owner)
            and abs(gap) <= gap_limit
            and overlap >= _ON_TOP_MIN_OVERLAP
        )
        if not on_top_authoritative and _update_relation_hysteresis(
            runtime,
            on_top_key,
            on_top_observed,
        ):
            relations.append(
                RuntimeRelationState(
                    subject_id=subject.name,
                    relation="on_top",
                    object_id=owner.name,
                    confidence=min(1.0, overlap),
                    source="geometry",
                    observed_at_step=step,
                    revision=f"on_top:{owner.name}",
                )
            )

        near_key = (subject.name, "near", owner.name)
        near_established = _relation_is_established(runtime, near_key)
        distance = _aabb_center_distance(subject.aabb, owner.aabb)
        near_observed = distance <= (
            _NEAR_RETAIN_DISTANCE_M if near_established else _NEAR_ESTABLISH_DISTANCE_M
        )
        if _update_relation_hysteresis(runtime, near_key, near_observed):
            relations.append(
                RuntimeRelationState(
                    subject_id=subject.name,
                    relation="near",
                    object_id=owner.name,
                    confidence=max(0.0, 1.0 - distance / _NEAR_RETAIN_DISTANCE_M),
                    source="geometry",
                    observed_at_step=step,
                    revision=f"near:{owner.name}",
                )
            )
    return relations


def _local_object_pairs(
    objects: dict[str, RuntimeObjectState],
    *,
    vertical_axis: str = "z",
) -> list[tuple[RuntimeObjectState, RuntimeObjectState]]:
    buckets: dict[tuple[str, int, int], list[RuntimeObjectState]] = defaultdict(list)
    for obj in objects.values():
        center = _aabb_center(obj.aabb)
        if center is None:
            continue
        horizontal_indices = horizontal_axis_indices(vertical_axis)
        bucket = (
            str(obj.room_hint or ""),
            int(math.floor(center[horizontal_indices[0]] / _BUCKET_SIZE_M)),
            int(math.floor(center[horizontal_indices[1]] / _BUCKET_SIZE_M)),
        )
        buckets[bucket].append(obj)
    pairs = []
    seen = set()
    for (room_id, bucket_x, bucket_y), subjects in buckets.items():
        neighbors = [
            item
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for item in buckets.get((room_id, bucket_x + dx, bucket_y + dy), [])
        ]
        for subject in subjects:
            for owner in neighbors:
                if subject.name == owner.name:
                    continue
                key = (subject.name, owner.name)
                if key in seen:
                    continue
                seen.add(key)
                if _aabb_volume(subject.aabb) > _aabb_volume(owner.aabb) * 1.2:
                    continue
                pairs.append((subject, owner))
    return pairs


def _update_relation_hysteresis(
    runtime: Any,
    key: tuple[str, str, str],
    observed: bool,
    *,
    establish_frames: int = 2,
    release_frames: int = 2,
) -> bool:
    store = getattr(runtime, "_scene_relation_hysteresis", None)
    if not isinstance(store, dict):
        store = {}
        runtime._scene_relation_hysteresis = store
    state = dict(store.get(key) or {})
    established = bool(state.get("established"))
    positive = int(state.get("positive", 0))
    negative = int(state.get("negative", 0))
    if observed:
        positive += 1
        negative = 0
        if positive >= establish_frames:
            established = True
    else:
        negative += 1
        positive = 0
        if negative >= release_frames:
            established = False
    store[key] = {
        "established": established,
        "positive": positive,
        "negative": negative,
    }
    return established


def _relation_is_established(runtime: Any, key: tuple[str, str, str]) -> bool:
    store = getattr(runtime, "_scene_relation_hysteresis", None)
    state = store.get(key) if isinstance(store, dict) else None
    return bool(state.get("established")) if isinstance(state, dict) else False


def _clear_relation_hysteresis(
    runtime: Any,
    key: tuple[str, str, str],
) -> None:
    store = getattr(runtime, "_scene_relation_hysteresis", None)
    if isinstance(store, dict):
        store.pop(key, None)


def _on_top_metrics(
    subject: dict[str, list[float]],
    owner: dict[str, list[float]],
    *,
    vertical_axis: str,
) -> tuple[float, float]:
    height_index = vertical_axis_index(vertical_axis)
    horizontal_indices = horizontal_axis_indices(vertical_axis)
    gap = float(subject["min"][height_index]) - float(owner["max"][height_index])
    overlap = _horizontal_overlap_area(
        subject,
        owner,
        horizontal_indices=horizontal_indices,
    )
    subject_area = max(
        1e-9,
        (
            float(subject["max"][horizontal_indices[0]])
            - float(subject["min"][horizontal_indices[0]])
        )
        * (
            float(subject["max"][horizontal_indices[1]])
            - float(subject["min"][horizontal_indices[1]])
        ),
    )
    return gap, overlap / subject_area


def _aabb_containment_ratio(
    subject: dict[str, list[float]],
    owner: dict[str, list[float]],
) -> float:
    intersection = 1.0
    subject_volume = 1.0
    for axis in range(3):
        subject_span = max(
            0.0,
            float(subject["max"][axis]) - float(subject["min"][axis]),
        )
        overlap = max(
            0.0,
            min(float(subject["max"][axis]), float(owner["max"][axis]))
            - max(float(subject["min"][axis]), float(owner["min"][axis])),
        )
        subject_volume *= max(subject_span, 1e-6)
        intersection *= overlap
    return intersection / subject_volume if subject_volume > 0.0 else 0.0


def _horizontal_overlap_area(
    left: dict[str, list[float]],
    right: dict[str, list[float]],
    *,
    horizontal_indices: tuple[int, int],
) -> float:
    overlap_first = max(
        0.0,
        min(
            float(left["max"][horizontal_indices[0]]),
            float(right["max"][horizontal_indices[0]]),
        )
        - max(
            float(left["min"][horizontal_indices[0]]),
            float(right["min"][horizontal_indices[0]]),
        ),
    )
    overlap_second = max(
        0.0,
        min(
            float(left["max"][horizontal_indices[1]]),
            float(right["max"][horizontal_indices[1]]),
        )
        - max(
            float(left["min"][horizontal_indices[1]]),
            float(right["min"][horizontal_indices[1]]),
        ),
    )
    return overlap_first * overlap_second


def _aabb_center(aabb: dict[str, list[float]] | None) -> tuple[float, float, float] | None:
    if not isinstance(aabb, dict):
        return None
    try:
        return tuple(
            0.5 * (float(aabb["min"][axis]) + float(aabb["max"][axis])) for axis in range(3)
        )
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _aabb_center_distance(
    left: dict[str, list[float]],
    right: dict[str, list[float]],
) -> float:
    left_center = _aabb_center(left)
    right_center = _aabb_center(right)
    if left_center is None or right_center is None:
        return float("inf")
    return math.sqrt(sum((left_center[index] - right_center[index]) ** 2 for index in range(3)))


def _aabb_volume(aabb: dict[str, list[float]] | None) -> float:
    if not isinstance(aabb, dict):
        return float("inf")
    try:
        return math.prod(
            max(0.0, float(aabb["max"][axis]) - float(aabb["min"][axis])) for axis in range(3)
        )
    except (KeyError, IndexError, TypeError, ValueError):
        return float("inf")


def _object_reference_name(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    name = getattr(value, "name", None)
    return str(name).strip() if isinstance(name, str) and name.strip() else None


def _object_relative_state(
    subject: Any,
    state_name: str,
    owner: Any,
) -> bool | None:
    states = getattr(subject, "states", None)
    if not isinstance(states, dict):
        return None
    for key, state in states.items():
        key_name = getattr(key, "__name__", str(key)).split(".")[-1]
        if key_name != state_name:
            continue
        getter = getattr(state, "get_value", None)
        if not callable(getter):
            return None
        try:
            return bool(getter(owner))
        except Exception:
            return None
    return None


def _is_container_capable(raw_owner: Any, owner: RuntimeObjectState) -> bool:
    links = getattr(raw_owner, "links", None)
    if isinstance(links, dict):
        for link in links.values():
            if not bool(getattr(link, "is_meta_link", False)):
                continue
            if str(getattr(link, "meta_link_type", "") or "").lower() in {
                "fillable",
                "openfillable",
            }:
                return True
    return bool(_semantic_tokens(owner) & _CONTAINER_CATEGORY_TOKENS)


def _is_reliable_support(raw_owner: Any, owner: RuntimeObjectState) -> bool:
    del raw_owner
    return bool(_semantic_tokens(owner) & _SUPPORT_CATEGORY_TOKENS)


def _semantic_tokens(obj: RuntimeObjectState) -> set[str]:
    text = " ".join(value for value in (obj.name, obj.category) if value)
    return set(text.lower().replace("-", "_").replace(" ", "_").split("_"))


def _robot_reports_grasping(
    robot: Any,
    *,
    arm: str,
    candidate_obj: Any,
) -> bool | None:
    method = getattr(robot, "is_grasping", None)
    if not callable(method):
        return None
    try:
        result = method(arm=arm, candidate_obj=candidate_obj)
    except TypeError:
        try:
            result = method(candidate_obj=candidate_obj)
        except Exception:
            return None
    except Exception:
        return None
    if isinstance(result, bool):
        return result
    name = str(getattr(result, "name", "") or "").upper()
    if name == "TRUE":
        return True
    if name == "FALSE":
        return False
    try:
        numeric = int(result)
    except (TypeError, ValueError):
        return None
    if numeric == 1:
        return True
    if numeric == -1:
        return False
    return None


__all__ = ["sample_scene_relations"]
