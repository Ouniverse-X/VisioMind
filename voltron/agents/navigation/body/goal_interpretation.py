from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

import requests

from voltron.agents.navigation.body.json_response import extract_json_object

_SYSTEM_PROMPT = """You are the Navigation agent's natural-language goal interpreter.
Return valid JSON only. Do not include markdown fences or extra text.

Convert any navigation instruction into one structured goal. Do not choose exact
object IDs unless the user explicitly names one. Prefer semantic intent and
constraints that can be grounded by the scene graph and navigation backend.

If the context contains grounding_candidates, choose the candidate that best
matches the full navigation intent, follow-up task, current pose, visible or
likely visible landmarks, room/region context, and interaction readiness. The
Navigation agent is responsible for this candidate choice. The grounding tool
only supplies candidates and evidence. Prefer the candidate that makes the
whole instruction executable, not merely the first semantic name match.
For follow-up interaction, the selected candidate must match the requested
action affordance or operable part/category; choose the control, handle,
button, switch, knob, lever, or other actionable part when the task requires
one, not merely the device or fixture affected by that control.

Supported goal kinds:
- relative_motion: metric displacement commands with a direction and distance.
- object: approach a named or implied physical object, part, device, control, surface, fixture, or tool.
- region: navigate to a named or implied room, zone, area, side, entrance area, passage, or boundary region.
- landmark: navigate relative to a salient spatial landmark such as an opening, threshold, passage, boundary, or fixture.
- exploration: inspect, search, scan, or continue when the destination is underspecified.

Use generic spatial fields instead of adding special-case goal kinds.
- spatial_relation: at, near, inside, outside, before, beyond, through, facing,
  alongside, between, around, reachable_from, visible_from.
- stop_condition: distance_reached, same_side, other_side, inside_region,
  object_reachable, object_visible, interaction_ready, view_ready, blocked,
  timeout.

Interpretation principles:
- Infer object categories and action-relevant parts from affordance language, including controls, switches, buttons, handles, knobs, and appliance controls.
- Infer landmarks and regions from language about passages, entrances, thresholds, room boundaries, sides, and transition areas.
- Preserve follow-up task context that changes where the robot should stop, such as whether it should be ready to see, reach, cross, avoid crossing, or interact.
- Preserve ambiguity as natural-language constraints instead of fabricating IDs.
- Use the current pose, current room, recent observations, and scene graph summaries from context.
- Output relative distances and directions for metric commands; the backend computes coordinates.

Return this JSON schema:
{
  "goal_kind": "relative_motion | object | region | landmark | exploration",
  "target_query": {
    "object": "optional object/category phrase",
    "part": "optional part phrase",
    "room": "optional room phrase",
    "region": "optional region phrase",
    "landmark": "optional landmark phrase"
  },
  "spatial_relation": "optional generic relation",
  "stop_condition": ["optional generic termination conditions"],
  "relative_motion": {
    "direction": "forward | backward | left | right",
    "distance_m": 0.0
  },
  "constraints": {"free-form": "short natural-language constraints"},
  "followup_context": {"free-form": "short natural-language context for downstream agents"},
  "selected_grounding_candidate": {
    "object_id": "optional id from grounding_candidates",
    "reason": "why this candidate best satisfies the navigation instruction"
  },
  "reason": "short explanation"
}
"""

_RETRIABLE_STATUS_CODES = {
    HTTPStatus.REQUEST_TIMEOUT,
    HTTPStatus.TOO_MANY_REQUESTS,
    HTTPStatus.INTERNAL_SERVER_ERROR,
    HTTPStatus.BAD_GATEWAY,
    HTTPStatus.SERVICE_UNAVAILABLE,
    HTTPStatus.GATEWAY_TIMEOUT,
}

_MAX_CONTEXT_STRING_CHARS = 1000
_MAX_CONTEXT_LIST_ITEMS = 40
_MAX_CONTEXT_MAPPING_ITEMS = 80
_MAX_GROUNDING_CANDIDATES = 200
_PARAMETER_CONTEXT_KEYS = (
    "instruction",
    "scene_id",
    "scene_file",
    "behavior_scene_file",
    "nav2_trav_map_filename",
    "pose",
    "orientation",
    "region",
    "room",
    "room_id",
    "floor_id",
    "nav_feedback",
    "constraints",
    "followup_context",
    "completion_criteria",
    "policy_options",
)
_NAVIGATION_STATE_CONTEXT_KEYS = (
    "scene_id",
    "pose",
    "orientation",
    "current_room",
    "current_region",
    "room_id",
    "floor_id",
    "vertical_axis",
    "active_waypoint_index",
    "global_waypoint_index",
    "dense_waypoint_index",
    "recovery_mode",
    "controller_mode",
    "follow_status",
    "localization_guard",
    "scene_state",
)
_HEAVY_CONTEXT_KEYS = {
    "image",
    "images",
    "raw_image",
    "raw_images",
    "image_b64",
    "images_b64",
    "rgb",
    "depth",
    "raw_observation",
    "embedding",
    "embeddings",
    "point_cloud",
    "point_clouds",
    "collision_parts",
    "scene_map_seed",
    "nav2_raw_path_points",
    "nav2_path_points",
    "dense_waypoints",
    "global_waypoints",
}


@dataclass(frozen=True)
class OpenAINavigationGoalInterpreterConfig:
    base_url: str
    model: str
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: float = 20.0
    temperature: float = 0.0
    max_retries: int = 0
    retry_backoff_s: float = 1.0


class OpenAICompatibleNavigationGoalInterpreter:
    def __init__(self, config: OpenAINavigationGoalInterpreterConfig) -> None:
        self.config = config
        self.session = requests.Session()

    def interpret_goal(self, *, instruction: str, context: dict[str, Any]) -> dict[str, Any]:
        prompt = self._build_prompt(instruction=instruction, context=context)
        content = self._request_chat_completion(prompt)
        payload = extract_json_object(content, label="Navigation goal interpreter")
        if not isinstance(payload, dict):
            raise ValueError("Navigation goal interpreter response must be a JSON object")
        grounding_candidates = context.get("grounding_candidates")
        allow_grounding_candidate_selection = isinstance(grounding_candidates, list) and bool(
            grounding_candidates
        )
        return self._normalize_payload(
            payload,
            allow_grounding_candidate_selection=allow_grounding_candidate_selection,
        )

    def _request_chat_completion(self, user_prompt: str) -> str:
        url = self._completion_url(self.config.base_url)
        api_key = (
            self.config.api_key or os.getenv(self.config.api_key_env) or os.getenv("OPENAI_API_KEY")
        )
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
        }
        attempts = max(1, int(self.config.max_retries) + 1)
        response = None
        last_content_error: ValueError | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = self.session.post(
                    url, headers=headers, json=payload, timeout=self.config.timeout_s
                )
                if response.status_code in _RETRIABLE_STATUS_CODES and attempt < attempts:
                    time.sleep(max(0.0, float(self.config.retry_backoff_s)))
                    continue
                response.raise_for_status()
                try:
                    return self._response_content(response)
                except ValueError as exc:
                    last_content_error = exc
                    if attempt < attempts:
                        time.sleep(max(0.0, float(self.config.retry_backoff_s)))
                        continue
                    raise
            except requests.exceptions.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code in _RETRIABLE_STATUS_CODES and attempt < attempts:
                    time.sleep(max(0.0, float(self.config.retry_backoff_s)))
                    continue
                detail = self._extract_http_error_detail(exc.response)
                raise RuntimeError(
                    f"Navigation goal interpreter HTTP {status_code or 'unknown'} "
                    f"on attempt {attempt}/{attempts}: {detail}"
                ) from exc
            except requests.exceptions.RequestException:
                if attempt < attempts:
                    time.sleep(max(0.0, float(self.config.retry_backoff_s)))
                    continue
                raise
        if response is None:
            raise RuntimeError("Navigation goal interpreter request produced no response")
        if last_content_error is not None:
            raise last_content_error
        return self._response_content(response)

    @staticmethod
    def _response_content(response: requests.Response) -> str:
        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise ValueError("Navigation goal interpreter response did not include choices")
        content = choices[0].get("message", {}).get("content")
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item) for item in content
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Navigation goal interpreter response content is empty")
        return content

    @staticmethod
    def _completion_url(base_url: str) -> str:
        normalized = base_url.rstrip("/")
        if normalized.endswith("/chat/completions"):
            return normalized
        return f"{normalized}/chat/completions"

    @staticmethod
    def _extract_http_error_detail(response: requests.Response | None) -> str:
        if response is None:
            return "empty error response"
        try:
            payload = response.json()
        except ValueError:
            text = response.text.strip()
            return text or "empty error response"
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict) and error.get("message"):
                return str(error["message"])
            if payload.get("detail"):
                return str(payload["detail"])
        return str(payload)

    @staticmethod
    def _build_prompt(*, instruction: str, context: dict[str, Any]) -> str:
        target = context.get("target")
        payload = {
            "instruction": instruction,
            "task_description": context.get("task_description"),
            "action": context.get("action"),
            "target": _compact_context_value(target),
            "parameters": _compact_parameters(context.get("parameters")),
            "map_state": _compact_navigation_state(context.get("map_state")),
            "backend_state": _compact_navigation_state(context.get("backend_state")),
            "scene_map": _compact_context_value(context.get("scene_map")),
            "working_state": _compact_context_value(context.get("working_state")),
            "observation_context": _compact_observation_context(
                context.get("observation_context"),
                target=target,
            ),
            "start": _compact_context_value(context.get("start")),
            "interpreted_goal": _compact_context_value(context.get("interpreted_goal")),
            "grounded_goal": _compact_context_value(context.get("grounded_goal")),
            "grounding_candidates": _compact_grounding_candidates(
                context.get("grounding_candidates")
            ),
        }
        if context.get("grounding_candidates"):
            task = (
                "Interpret this navigation instruction and choose exactly one grounding candidate "
                "for the Navigation agent to execute."
            )
        else:
            task = "Interpret this navigation instruction for grounding and path planning."
        return f"{task}\nContext JSON: {json.dumps(payload, ensure_ascii=False, default=str)}\nReturn JSON only."

    @staticmethod
    def _normalize_payload(
        payload: dict[str, Any],
        *,
        allow_grounding_candidate_selection: bool = False,
    ) -> dict[str, Any]:
        normalized = dict(payload)
        goal_kind = str(normalized.get("goal_kind") or "exploration").strip().lower()
        if not isinstance(normalized.get("target_query"), dict):
            normalized["target_query"] = {}
        target_query = dict(normalized["target_query"])
        if goal_kind == "room":
            goal_kind = "region"
            if "region" not in target_query and isinstance(target_query.get("room"), str):
                target_query["region"] = target_query["room"]
        elif goal_kind == "doorway":
            goal_kind = "landmark"
            target_query.setdefault("landmark", "threshold")
        if goal_kind not in {
            "relative_motion",
            "object",
            "region",
            "landmark",
            "exploration",
            "position",
        }:
            goal_kind = "exploration"
        normalized["goal_kind"] = goal_kind
        normalized["target_query"] = target_query
        if not isinstance(normalized.get("relative_motion"), dict):
            normalized["relative_motion"] = {}
        if not isinstance(normalized.get("constraints"), dict):
            normalized["constraints"] = {}
        if not isinstance(normalized.get("followup_context"), dict):
            normalized["followup_context"] = {}
        spatial_relation = str(normalized.get("spatial_relation") or "").strip().lower()
        if spatial_relation not in {
            "at",
            "near",
            "inside",
            "outside",
            "before",
            "beyond",
            "through",
            "facing",
            "alongside",
            "between",
            "around",
            "reachable_from",
            "visible_from",
        }:
            spatial_relation = ""
        normalized["spatial_relation"] = spatial_relation
        stop_condition = normalized.get("stop_condition")
        if isinstance(stop_condition, str):
            stop_condition = [stop_condition]
        if not isinstance(stop_condition, list):
            stop_condition = []
        valid_stop_conditions = {
            "distance_reached",
            "same_side",
            "other_side",
            "inside_region",
            "object_reachable",
            "object_visible",
            "interaction_ready",
            "view_ready",
            "blocked",
            "timeout",
        }
        normalized["stop_condition"] = [
            value
            for item in stop_condition
            if (value := str(item).strip().lower()) in valid_stop_conditions
        ]
        normalized.setdefault("reason", "")
        selected_grounding_candidate = normalized.get("selected_grounding_candidate")
        if allow_grounding_candidate_selection and isinstance(selected_grounding_candidate, dict):
            selected = {
                "object_id": str(selected_grounding_candidate.get("object_id") or "").strip(),
                "reason": str(selected_grounding_candidate.get("reason") or "").strip(),
            }
            normalized["selected_grounding_candidate"] = selected if selected["object_id"] else {}
        else:
            normalized["selected_grounding_candidate"] = {}
        normalized["source"] = "openai_compatible_navigation_goal_interpreter"
        return normalized


def _compact_parameters(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: _compact_context_value(value[key])
        for key in _PARAMETER_CONTEXT_KEYS
        if key in value and value[key] is not None
    }


def _compact_observation_context(value: Any, *, target: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    compacted: dict[str, Any] = {}
    for key in ("metadata", "scene_report", "navigation_state", "observation_keys"):
        if key in value and value[key] is not None:
            compacted[key] = _compact_context_value(value[key])
    observation = value.get("observation")
    if isinstance(observation, dict):
        observation_summary = _compact_observation(observation, target=target)
        if observation_summary:
            compacted["observation_summary"] = observation_summary
    return compacted


def _compact_observation(value: dict[str, Any], *, target: Any) -> dict[str, Any]:
    semantic: dict[str, Any] = {}
    tensor_metadata: dict[str, Any] = {}
    for key, item in value.items():
        normalized_key = str(key)
        if normalized_key == "scene_state" and isinstance(item, dict):
            semantic["scene_state_summary"] = _compact_scene_state(item, target=target)
            continue
        array_metadata = _array_metadata(item)
        if array_metadata is not None:
            tensor_metadata[normalized_key] = array_metadata
            continue
        if normalized_key in _HEAVY_CONTEXT_KEYS or normalized_key.startswith("video."):
            continue
        if isinstance(item, (str, int, float, bool)) or item is None:
            semantic[normalized_key] = _compact_context_value(item)
        elif normalized_key in {
            "pose",
            "orientation",
            "nav_feedback",
            "current_room",
            "current_region",
            "region",
            "room",
            "scene_report",
        }:
            semantic[normalized_key] = _compact_context_value(item)
    if tensor_metadata:
        semantic["tensor_metadata"] = tensor_metadata
    return semantic


def _compact_navigation_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    compacted: dict[str, Any] = {}
    for key in _NAVIGATION_STATE_CONTEXT_KEYS:
        if key not in value or value[key] is None:
            continue
        if key == "scene_state" and isinstance(value[key], dict):
            compacted[key] = _compact_scene_state(value[key], target=None)
        else:
            compacted[key] = _compact_context_value(value[key])
    return compacted


def _compact_scene_state(value: dict[str, Any], *, target: Any) -> dict[str, Any]:
    objects = value.get("objects")
    doors = value.get("doors")
    temporary_obstacles = value.get("temporary_obstacles")
    summary: dict[str, Any] = {
        key: _compact_context_value(value[key])
        for key in ("scene_id", "step", "signature", "door_signature")
        if key in value and value[key] is not None
    }
    summary["object_count"] = (
        len(objects) if isinstance(objects, (dict, list)) else int(value.get("object_count") or 0)
    )
    summary["door_count"] = (
        len(doors) if isinstance(doors, (dict, list)) else int(value.get("door_count") or 0)
    )
    summary["temporary_obstacle_count"] = (
        len(temporary_obstacles) if isinstance(temporary_obstacles, (dict, list)) else 0
    )
    closed_doors = value.get("closed_doors")
    if isinstance(closed_doors, list):
        summary["closed_doors"] = _compact_context_value(closed_doors)
    if isinstance(doors, dict):
        summary["doors"] = [
            _compact_scene_entity(name, door)
            for name, door in list(doors.items())[:_MAX_CONTEXT_LIST_ITEMS]
            if isinstance(door, dict)
        ]
    relevant_objects = _relevant_scene_objects(objects, target=target)
    if relevant_objects:
        summary["relevant_objects"] = relevant_objects
    return summary


def _relevant_scene_objects(value: Any, *, target: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    target_terms = _semantic_terms(target)
    candidates: list[dict[str, Any]] = []
    for name, item in value.items():
        if not isinstance(item, dict):
            continue
        haystack = _semantic_terms(
            " ".join(
                str(candidate)
                for candidate in (name, item.get("name"), item.get("category"))
                if candidate is not None
            )
        )
        if target_terms and not target_terms.intersection(haystack):
            continue
        candidates.append(_compact_scene_entity(str(name), item))
        if len(candidates) >= _MAX_CONTEXT_LIST_ITEMS:
            break
    return candidates


def _compact_scene_entity(name: str, value: dict[str, Any]) -> dict[str, Any]:
    compacted: dict[str, Any] = {"name": value.get("name") or name}
    for key in ("category", "position", "room_hint", "in_rooms", "is_open", "openness"):
        if key in value and value[key] is not None:
            compacted[key] = _compact_context_value(value[key])
    return compacted


def _compact_grounding_candidates(value: Any) -> list[Any]:
    if not isinstance(value, list):
        return []
    return [
        _compact_context_value(item, max_mapping_items=_MAX_CONTEXT_MAPPING_ITEMS)
        for item in value[:_MAX_GROUNDING_CANDIDATES]
    ]


def _compact_context_value(
    value: Any,
    *,
    depth: int = 0,
    max_mapping_items: int = _MAX_CONTEXT_MAPPING_ITEMS,
) -> Any:
    if isinstance(value, str):
        return value[:_MAX_CONTEXT_STRING_CHARS]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    array_metadata = _array_metadata(value)
    if array_metadata is not None:
        return array_metadata
    if depth >= 4:
        return {"type": type(value).__name__, "omitted_at_depth": depth}
    if isinstance(value, dict):
        compacted: dict[str, Any] = {}
        kept = 0
        for key, item in value.items():
            normalized_key = str(key)
            if normalized_key in _HEAVY_CONTEXT_KEYS or normalized_key.startswith("video."):
                continue
            compacted[normalized_key] = _compact_context_value(
                item,
                depth=depth + 1,
                max_mapping_items=max_mapping_items,
            )
            kept += 1
            if kept >= max_mapping_items:
                break
        omitted = max(0, len(value) - kept)
        if omitted:
            compacted["omitted_mapping_items"] = omitted
        return compacted
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        compacted = [
            _compact_context_value(
                item,
                depth=depth + 1,
                max_mapping_items=max_mapping_items,
            )
            for item in items[:_MAX_CONTEXT_LIST_ITEMS]
        ]
        if len(items) > len(compacted):
            compacted.append({"omitted_list_items": len(items) - len(compacted)})
        return compacted
    return str(value)[:_MAX_CONTEXT_STRING_CHARS]


def _array_metadata(value: Any) -> dict[str, Any] | None:
    shape = getattr(value, "shape", None)
    if shape is None or isinstance(value, (str, bytes, dict, list, tuple, set)):
        return None
    try:
        normalized_shape = [int(item) for item in shape]
    except (TypeError, ValueError):
        return None
    metadata: dict[str, Any] = {
        "type": type(value).__name__,
        "shape": normalized_shape,
    }
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        metadata["dtype"] = str(dtype)
    return metadata


def _semantic_terms(value: Any) -> set[str]:
    if isinstance(value, dict):
        text = " ".join(str(item) for item in value.values() if isinstance(item, (str, int, float)))
    else:
        text = str(value or "")
    return {term for term in re.sub(r"[^a-z0-9]+", " ", text.lower()).split() if term}


__all__ = [
    "OpenAICompatibleNavigationGoalInterpreter",
    "OpenAINavigationGoalInterpreterConfig",
]
