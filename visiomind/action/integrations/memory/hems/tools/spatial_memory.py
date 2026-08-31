from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any

from visiomind.action.shared.object_approach_signature import candidate_signature


PROVENANCE_KEYS = (
    "source_agent",
    "source_episode_id",
    "source_action_id",
    "confidence_source",
    "run_id",
)


def ensure_map(maps: dict[str, dict[str, Any]], scene_id: str) -> dict[str, Any]:
    return maps.setdefault(scene_id, {"map_payload": {}, "metadata": {}})


def clone_map_entry(scene_id: str, entry: dict[str, Any], *, status: str) -> dict[str, Any]:
    return {
        "scene_id": scene_id,
        "status": status,
        "map_payload": deepcopy(entry["map_payload"]),
        "metadata": deepcopy(entry["metadata"]),
    }


def build_provenance(
    *sources: dict[str, Any] | None,
    now_string: str | None = None,
    default_source_agent: str | None = None,
) -> dict[str, Any]:
    provenance: dict[str, Any] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in PROVENANCE_KEYS:
            value = source.get(key)
            if value is not None:
                provenance[key] = deepcopy(value)
        if source.get("updated_at") is not None:
            provenance["updated_at"] = deepcopy(source["updated_at"])
    if default_source_agent and "source_agent" not in provenance:
        provenance["source_agent"] = default_source_agent
    provenance.setdefault("updated_at", now_string or datetime.now().isoformat())
    return provenance


def record_map_metadata_update(
    entry: dict[str, Any],
    *,
    key: str,
    provenance: dict[str, Any],
) -> None:
    metadata = entry.setdefault("metadata", {})
    metadata[key] = deepcopy(provenance)
    metadata["updated_at"] = provenance["updated_at"]


def merge_dicts(base: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in delta.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = merge_dicts(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def update_navigation_map(
    *, maps: dict[str, dict[str, Any]], scene_id: str, payload: dict[str, Any]
) -> None:
    entry = ensure_map(maps, scene_id)
    nav_state = entry["map_payload"].setdefault("navigation", {})
    provenance = build_provenance(payload, default_source_agent="NAVIGATION")
    nav_state["last_update_provenance"] = deepcopy(provenance)
    record_map_metadata_update(
        entry,
        key="last_navigation_update",
        provenance=provenance,
    )

    region_name = str(payload.get("region", "")).strip()
    if region_name:
        nav_state["last_region"] = region_name
        visited_regions = nav_state.setdefault("visited_regions", [])
        if region_name not in visited_regions:
            visited_regions.append(region_name)

    if "pose" in payload:
        nav_state["last_pose"] = deepcopy(payload["pose"])
    if "grounded_goal" in payload:
        nav_state["last_grounded_goal"] = deepcopy(payload["grounded_goal"])
    if "path_plan" in payload:
        nav_state["last_path_plan"] = deepcopy(payload["path_plan"])

    obstacles = entry["map_payload"].setdefault("obstacles", [])
    existing_by_name = {
        str(item.get("name") or item.get("id") or ""): index for index, item in enumerate(obstacles)
    }
    for obstacle in payload.get("obstacles", []):
        obstacle_name = str(obstacle.get("name") or obstacle.get("id") or "")
        if obstacle_name and obstacle_name in existing_by_name:
            obstacles[existing_by_name[obstacle_name]] = deepcopy(obstacle)
        else:
            obstacles.append(deepcopy(obstacle))

    if "frontiers" in payload:
        exploration = entry["map_payload"].setdefault("exploration", {})
        exploration["frontiers"] = deepcopy(payload["frontiers"])


def query_topology(
    *, maps: dict[str, dict[str, Any]], start: dict[str, Any], goal: dict[str, Any]
) -> dict[str, Any]:
    scene_id = str(start.get("scene_id") or goal.get("scene_id") or "").strip()
    if not scene_id:
        if len(maps) == 1:
            scene_id = next(iter(maps))
        else:
            return {
                "scene_id": None,
                "start": extract_anchor_name(start),
                "goal": extract_anchor_name(goal),
                "path": [],
                "found": False,
                "source": "unavailable",
            }

    entry = maps.get(scene_id)
    if entry is None:
        return {
            "scene_id": scene_id,
            "start": extract_anchor_name(start),
            "goal": extract_anchor_name(goal),
            "path": [],
            "found": False,
            "source": "missing_map",
        }

    start_name = extract_anchor_name(start)
    goal_name = extract_anchor_name(goal)
    map_payload = entry["map_payload"]
    route_path = find_topology_route(map_payload, start_name, goal_name)
    if route_path:
        return {
            "scene_id": scene_id,
            "start": start_name,
            "goal": goal_name,
            "path": route_path,
            "found": True,
            "source": "routes",
        }

    adjacency = map_payload.get("topology", {}).get("adjacency", {})
    bfs_result = bfs_path(adjacency, start_name, goal_name)
    if bfs_result:
        return {
            "scene_id": scene_id,
            "start": start_name,
            "goal": goal_name,
            "path": bfs_result,
            "found": True,
            "source": "adjacency",
        }

    return {
        "scene_id": scene_id,
        "start": start_name,
        "goal": goal_name,
        "path": [],
        "found": False,
        "source": "not_found",
    }


def find_topology_route(map_payload: dict[str, Any], start: str, goal: str) -> list[str]:
    routes = map_payload.get("topology", {}).get("routes", [])
    for route in routes:
        route_start = extract_anchor_name(route.get("start", {}))
        route_goal = extract_anchor_name(route.get("goal", {}))
        path = route.get("path", [])
        if route_start == start and route_goal == goal and path:
            return [str(node) for node in path]
    return []


def extract_anchor_name(anchor: Any) -> str:
    if isinstance(anchor, str):
        return anchor
    if not isinstance(anchor, dict):
        return str(anchor)
    return str(
        anchor.get("region")
        or anchor.get("node_id")
        or anchor.get("name")
        or anchor.get("id")
        or ""
    )


def object_approach_target_key(target: dict[str, Any]) -> str:
    for key in ("object_id", "object", "object_name", "target", "name", "room_id", "room_name"):
        value = target.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def object_approach_candidate_signature(candidate: dict[str, Any]) -> dict[str, Any]:
    return candidate_signature(candidate)


def bfs_path(adjacency: dict[str, Any], start: str, goal: str) -> list[str]:
    if not start or not goal or start not in adjacency:
        return []
    if start == goal:
        return [start]

    queue: list[list[str]] = [[start]]
    visited = {start}

    while queue:
        path = queue.pop(0)
        node = path[-1]
        for neighbor in adjacency.get(node, []):
            neighbor_name = str(neighbor)
            if neighbor_name in visited:
                continue
            next_path = path + [neighbor_name]
            if neighbor_name == goal:
                return next_path
            visited.add(neighbor_name)
            queue.append(next_path)
    return []
