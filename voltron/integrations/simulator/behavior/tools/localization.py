"""Localization helpers for the BEHAVIOR simulator integration."""

from __future__ import annotations

from typing import Any, Callable


def build_hovsg_localizer(
    *,
    scene_id: str | None,
    existing_localizer: Any,
    hovsg_graph_path: str | None,
    hovsg_graph_root: str | None,
    hovsg_nav_graph_type: str | None,
    localizer_factory: Callable[..., Any],
) -> Any | None:
    if scene_id is None:
        return None
    if existing_localizer is not None:
        return existing_localizer
    if hovsg_graph_path is None and hovsg_graph_root is None:
        return None

    if hovsg_graph_path is not None:
        return localizer_factory(
            scene_roots={scene_id: hovsg_graph_path},
            default_scene_id=scene_id,
            nav_graph_type=hovsg_nav_graph_type,
        )

    return localizer_factory(
        graph_root=hovsg_graph_root,
        default_scene_id=scene_id,
        nav_graph_type=hovsg_nav_graph_type,
    )


def localize_runtime_state(
    *,
    localizer: Any | None,
    pose: dict[str, Any] | None,
    scene_id: str | None,
    last_info: dict[str, Any],
) -> dict[str, Any]:
    if localizer is None or pose is None or scene_id is None:
        return dict(last_info)

    try:
        localized = localizer.update({"scene_id": scene_id}, pose=pose)
    except Exception:
        return dict(last_info)

    if not isinstance(localized, dict) or not localized:
        return dict(last_info)

    return {**dict(last_info), **localized, "scene_id": scene_id}
