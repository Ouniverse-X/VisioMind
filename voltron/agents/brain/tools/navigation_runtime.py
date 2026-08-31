from __future__ import annotations

import re
from typing import Any

from voltron.shared.context import ExecutionContext
from voltron.shared.models import RuntimeFeedback


def runtime_feedback_dict(value: Any) -> dict[str, Any]:
    feedback = RuntimeFeedback.from_value(value)
    return feedback.to_dict() if feedback is not None else {}


def resolve_navigation_state(
    *,
    execution_state: dict[str, Any] | None,
    environment_state: dict[str, Any] | None,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    if isinstance(execution_state, dict):
        navigation_state = execution_state.get("navigation_state")
        if isinstance(navigation_state, dict):
            candidates.append(navigation_state)
        latest = execution_state.get("latest_result")
        if isinstance(latest, dict):
            feedback = runtime_feedback_dict(latest.get("env_feedback"))
            if feedback:
                candidates.append(feedback)
        feedback = runtime_feedback_dict(execution_state.get("environment_feedback"))
        if feedback:
            candidates.append(feedback)
        for item in reversed(execution_state.get("recent_results", [])):
            if not isinstance(item, dict):
                continue
            feedback = runtime_feedback_dict(item.get("env_feedback"))
            if feedback:
                candidates.append(feedback)
    environment_feedback = runtime_feedback_dict(environment_state)
    if environment_feedback:
        candidates.append(environment_feedback)
    elif isinstance(environment_state, dict):
        candidates.append(environment_state)

    merged: dict[str, Any] = {}
    for candidate in candidates:
        for key in (
            "current_room",
            "current_region",
            "room_id",
            "floor_id",
            "pose",
            "path_backend",
            "controller_mode",
            "follow_status",
            "best_distance_to_waypoint",
            "distance_to_waypoint",
            "goal_reached",
            "nav2_error",
            "execution_goal",
            "local_goal",
            "target_waypoint",
            "tracking_target",
            "loop_detected",
            "oscillation_detected",
            "steps_since_progress",
        ):
            value = candidate.get(key)
            if key not in merged and value not in (None, "", {}):
                merged[key] = value
    return merged


def build_navigation_report(
    *,
    context: ExecutionContext,
    latest_result: dict[str, Any],
    navigation_state: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(navigation_state or {})
    latest_feedback = runtime_feedback_dict(latest_result.get("env_feedback"))
    if latest_feedback:
        merged.update(
            {key: value for key, value in latest_feedback.items() if value not in (None, "", {})}
        )

    distance_to_execution_goal_m = coerce_float(
        merged.get("best_distance_to_waypoint"),
        fallback=merged.get("distance_to_waypoint"),
    )
    target_hints = (
        context.runtime_state.get("planning_context", {}).get("interaction_target_hints", {})
        if isinstance(context.runtime_state.get("planning_context"), dict)
        else {}
    )
    if not target_room_available(target_hints):
        target_room_status = "no_room_constraint"
    elif not room_state_available(merged):
        target_room_status = "target_room_unknown"
    elif room_state_matches_target(target_hints=target_hints, navigation_state=merged):
        target_room_status = "already_in_target_room"
    else:
        target_room_status = "outside_target_room"

    follow_status = str(merged.get("follow_status") or "").strip().lower()
    path_backend = str(merged.get("path_backend") or "").strip().lower()
    nav_error = merged.get("nav2_error")
    has_approach_goal = any(
        isinstance(merged.get(key), dict) and merged.get(key)
        for key in ("execution_goal", "local_goal", "target_waypoint", "tracking_target")
    )
    approach_ready = bool(
        merged.get("goal_reached")
        or follow_status in {"goal_reached", "reached", "arrived"}
        or path_backend == "global_goal_reached"
    )
    approach_reachable: bool | None
    if nav_error not in (None, "", {}) or follow_status in {"failed", "unreachable", "blocked"}:
        approach_reachable = False
    elif has_approach_goal:
        approach_reachable = True
    else:
        approach_reachable = None

    steps_since_progress = coerce_int(merged.get("steps_since_progress"))
    approach_stalled = bool(
        merged.get("loop_detected")
        or merged.get("oscillation_detected")
        or follow_status in {"stalled", "stuck", "timeout"}
        or (steps_since_progress is not None and steps_since_progress >= 60)
    )
    return {
        "current_room": merged.get("current_room"),
        "current_region": merged.get("current_region"),
        "target_room_status": target_room_status,
        "distance_to_execution_goal_m": distance_to_execution_goal_m,
        "approach_reachable": approach_reachable,
        "approach_ready": approach_ready,
        "approach_stalled": approach_stalled,
        "path_backend": merged.get("path_backend"),
        "controller_mode": merged.get("controller_mode"),
        "follow_status": merged.get("follow_status"),
    }


def coerce_float(value: Any, *, fallback: Any = None) -> float | None:
    for candidate in (value, fallback):
        try:
            if candidate is None:
                continue
            return float(candidate)
        except (TypeError, ValueError):
            continue
    return None


def coerce_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def first_non_empty(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", {}):
            return value
    return None


def normalize_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.replace("_", " ").lower().split()).strip()
    return normalized or None


def canonical_room_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    region_match = re.fullmatch(r"region_(.+)_\d+", text.strip().lower())
    if region_match:
        text = region_match.group(1)
    normalized = re.sub(r"_+", "_", re.sub(r"\s+", "_", text.strip().lower())).strip("_")
    return normalized or None


def room_display_label(
    *,
    room: Any = None,
    region: Any = None,
    canonical_room_name_value: Any = None,
    room_name: Any = None,
    room_label: Any = None,
) -> str | None:
    explicit = normalize_label(room_label)
    if explicit:
        return explicit

    semantic = normalize_label(first_non_empty(room, region))
    canonical = first_non_empty(
        canonical_room_name(canonical_room_name_value)
        if canonical_room_name_value not in (None, "", {})
        else None,
        canonical_room_name(room_name),
    )
    if semantic:
        suffix_match = re.search(r"_(\d+)$", canonical or "")
        if suffix_match:
            return f"{semantic} {suffix_match.group(1)}"
        return semantic
    if canonical:
        return canonical.replace("_", " ")
    return None


def target_room_available(target_hints: dict[str, Any]) -> bool:
    return bool(
        first_non_empty(
            target_hints.get("room_id"),
            target_hints.get("room"),
            target_hints.get("region"),
            target_hints.get("room_label"),
            target_hints.get("canonical_room_name"),
            target_hints.get("room_name"),
        )
    )


def room_state_available(navigation_state: dict[str, Any]) -> bool:
    return bool(
        first_non_empty(
            navigation_state.get("room_id"),
            navigation_state.get("current_room"),
            navigation_state.get("current_region"),
        )
    )


def room_state_matches_target(
    *,
    target_hints: dict[str, Any],
    navigation_state: dict[str, Any],
) -> bool:
    target_room_id = first_non_empty(target_hints.get("room_id"))
    current_room_id = first_non_empty(navigation_state.get("room_id"))
    if (
        target_room_id
        and current_room_id
        and str(target_room_id).strip() == str(current_room_id).strip()
    ):
        return True

    target_aliases = room_aliases(
        room=target_hints.get("room"),
        region=target_hints.get("region"),
        room_label=target_hints.get("room_label"),
        canonical_room_name=target_hints.get("canonical_room_name"),
        room_name=target_hints.get("room_name"),
    )
    current_aliases = room_aliases(
        room=navigation_state.get("current_room"),
        region=navigation_state.get("current_region"),
        canonical_room_name=canonical_room_name(
            first_non_empty(
                navigation_state.get("current_room"),
                navigation_state.get("current_region"),
            )
        ),
    )
    return bool(
        target_aliases and current_aliases and not target_aliases.isdisjoint(current_aliases)
    )


def label_variants(value: Any) -> set[str]:
    normalized = normalize_label(value)
    if normalized is None:
        return set()

    variants = {normalized}
    trimmed_numeric_suffix = re.sub(r"\s+\d+$", "", normalized).strip()
    if trimmed_numeric_suffix:
        variants.add(trimmed_numeric_suffix)
    canonical = canonical_room_name(value)
    canonical_display = normalize_label(canonical)
    if canonical_display:
        variants.add(canonical_display)
        canonical_trimmed = re.sub(r"\s+\d+$", "", canonical_display).strip()
        if canonical_trimmed:
            variants.add(canonical_trimmed)
    return variants


def room_aliases(
    *,
    room: Any = None,
    region: Any = None,
    room_label: Any = None,
    canonical_room_name: Any = None,
    room_name: Any = None,
) -> set[str]:
    aliases: set[str] = set()
    for value in (room, region, room_label, canonical_room_name, room_name):
        aliases.update(label_variants(value))
    return aliases
