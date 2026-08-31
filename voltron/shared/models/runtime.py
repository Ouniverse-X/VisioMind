from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class NavigationRuntimeState:
    active_waypoint_index: int | None = None
    global_waypoint_index: int | None = None
    dense_waypoint_index: int | None = None
    controller_mode: str | None = None
    follow_status: str | None = None
    recovery_mode: str | None = None
    recovery_profile: str | None = None
    yaw_source: str | None = None
    path_backend: str | None = None
    path_tracking_mode: str | None = None
    nav2_error: str | None = None
    nav2_trav_map_filename: str | None = None
    loop_detected: bool | None = None
    oscillation_detected: bool | None = None
    tracking_target: dict[str, Any] | None = None
    target_waypoint: dict[str, Any] | None = None
    local_goal: dict[str, Any] | None = None
    execution_goal: dict[str, Any] | None = None
    steps_since_progress: int | None = None
    best_distance_to_waypoint: float | None = None
    path_cross_track_error: float | None = None
    path_signed_cross_track_error: float | None = None
    path_segment_index: int | None = None
    path_tangent_heading: float | None = None
    goal_reached: bool | None = None

    _KEYS = (
        "active_waypoint_index",
        "global_waypoint_index",
        "dense_waypoint_index",
        "controller_mode",
        "follow_status",
        "recovery_mode",
        "recovery_profile",
        "yaw_source",
        "path_backend",
        "path_tracking_mode",
        "nav2_error",
        "nav2_trav_map_filename",
        "loop_detected",
        "oscillation_detected",
        "tracking_target",
        "target_waypoint",
        "local_goal",
        "execution_goal",
        "steps_since_progress",
        "best_distance_to_waypoint",
        "path_cross_track_error",
        "path_signed_cross_track_error",
        "path_segment_index",
        "path_tangent_heading",
        "goal_reached",
    )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key in self._KEYS:
            value = getattr(self, key)
            if value not in (None, "", {}):
                payload[key] = value
        return payload

    @classmethod
    def from_value(cls, value: Any) -> NavigationRuntimeState | None:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            return None
        payload = {key: value.get(key) for key in cls._KEYS if value.get(key) not in (None, "", {})}
        if not payload:
            return None
        return cls(**payload)


@dataclass(frozen=True)
class RuntimeFeedback:
    step_count: int | None = None
    reward: float | None = None
    task_progress: float | None = None
    current_room: str | None = None
    current_region: str | None = None
    room_id: str | None = None
    floor_id: str | None = None
    pose: dict[str, Any] | None = None
    navigation: NavigationRuntimeState | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    _BASE_KEYS = (
        "step_count",
        "reward",
        "task_progress",
        "current_room",
        "current_region",
        "room_id",
        "floor_id",
        "pose",
    )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key in self._BASE_KEYS:
            value = getattr(self, key)
            if value not in (None, "", {}):
                payload[key] = value
        if self.navigation is not None:
            payload.update(self.navigation.to_dict())
        for key, value in self.extras.items():
            if value not in (None, "", {}):
                payload[key] = value
        return payload

    def with_extras(self, **extras: Any) -> RuntimeFeedback:
        merged = dict(self.extras)
        merged.update(extras)
        return RuntimeFeedback(
            step_count=self.step_count,
            reward=self.reward,
            task_progress=self.task_progress,
            current_room=self.current_room,
            current_region=self.current_region,
            room_id=self.room_id,
            floor_id=self.floor_id,
            pose=dict(self.pose) if isinstance(self.pose, dict) else self.pose,
            navigation=self.navigation,
            extras=merged,
        )

    @classmethod
    def from_value(cls, value: Any) -> RuntimeFeedback | None:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            return None
        navigation = NavigationRuntimeState.from_value(value)
        extras = {
            key: item
            for key, item in value.items()
            if key not in cls._BASE_KEYS and key not in NavigationRuntimeState._KEYS
        }
        payload = {key: value.get(key) for key in cls._BASE_KEYS}
        if (
            all(item in (None, "", {}) for item in payload.values())
            and navigation is None
            and not extras
        ):
            return None
        return cls(
            step_count=payload.get("step_count"),
            reward=payload.get("reward"),
            task_progress=payload.get("task_progress"),
            current_room=payload.get("current_room"),
            current_region=payload.get("current_region"),
            room_id=payload.get("room_id"),
            floor_id=payload.get("floor_id"),
            pose=payload.get("pose"),
            navigation=navigation,
            extras=extras,
        )

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self.to_dict()


@dataclass
class SubtaskStepOutcome:
    done: bool
    success: bool | None = None
    failure_reason: str | None = None
    feedback: RuntimeFeedback | dict[str, Any] = field(default_factory=dict)
