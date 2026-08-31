from __future__ import annotations

from typing import Any

from . import localization as behavior_localization
from visiomind.action.integrations.simulator.behavior.observation import robot_state as behavior_robot_state


def normalize_runtime_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def extract_runtime_kwarg(runtime_kwargs: dict[str, Any], key: str) -> str | None:
    return normalize_runtime_str(runtime_kwargs.pop(key, None))


def extract_scene_id(
    *,
    last_info: dict[str, Any],
    last_obs: dict[str, Any],
    scene_id: str | None,
) -> str | None:
    for candidate in (
        last_info.get("scene_id"),
        last_obs.get("scene_id"),
        scene_id,
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def build_hovsg_localizer(
    *,
    existing_localizer: Any | None,
    last_info: dict[str, Any],
    last_obs: dict[str, Any],
    scene_id: str | None,
    hovsg_graph_path: str | None,
    hovsg_graph_root: str | None,
    hovsg_nav_graph_type: str | None,
) -> Any | None:
    resolved_scene_id = extract_scene_id(
        last_info=last_info,
        last_obs=last_obs,
        scene_id=scene_id,
    )

    from visiomind.action.integrations.navigation.hovsg import HOVSGNavigatorAdapter

    return behavior_localization.build_hovsg_localizer(
        scene_id=resolved_scene_id,
        existing_localizer=existing_localizer,
        hovsg_graph_path=hovsg_graph_path,
        hovsg_graph_root=hovsg_graph_root,
        hovsg_nav_graph_type=hovsg_nav_graph_type,
        localizer_factory=HOVSGNavigatorAdapter,
    )


def localize_runtime_state_snapshot(
    *,
    existing_localizer: Any | None,
    last_info: dict[str, Any],
    last_obs: dict[str, Any],
    scene_id: str | None,
    hovsg_graph_path: str | None,
    hovsg_graph_root: str | None,
    hovsg_nav_graph_type: str | None,
    resolved_metadata: dict[str, str | None],
    frame_config: dict[str, Any] | None = None,
) -> tuple[Any | None, dict[str, Any]]:
    localizer = build_hovsg_localizer(
        existing_localizer=existing_localizer,
        last_info=last_info,
        last_obs=last_obs,
        scene_id=scene_id,
        hovsg_graph_path=hovsg_graph_path,
        hovsg_graph_root=hovsg_graph_root,
        hovsg_nav_graph_type=hovsg_nav_graph_type,
    )
    robot_state = behavior_robot_state.extract_runtime_robot_state(
        last_info=last_info,
        last_obs=last_obs,
        frame_config=frame_config,
    )
    localized_state = behavior_localization.localize_runtime_state(
        localizer=localizer,
        pose=robot_state["pose"],
        scene_id=resolved_metadata.get("scene_id"),
        last_info=last_info,
    )
    for key, value in (
        ("scene_pose", robot_state["pose"]),
        ("scene_orientation", robot_state["orientation"]),
        ("simulator_pose", robot_state["simulator_pose"]),
        ("simulator_orientation", robot_state["simulator_orientation"]),
    ):
        if value is not None:
            localized_state[key] = value
    if robot_state["pose"] is not None:
        localized_state["pose_frame"] = "scene"
    if robot_state["orientation"] is not None:
        localized_state["orientation_frame"] = "scene"
    return localizer, localized_state
