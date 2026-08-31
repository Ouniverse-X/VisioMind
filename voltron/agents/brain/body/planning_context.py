from __future__ import annotations

from typing import Any, Callable

from voltron.shared.context import MemorySnapshot, TaskRequest


def build_planning_context(
    *,
    memory: Any,
    request: TaskRequest,
    planner_mode_from_request: Callable[[TaskRequest], str],
) -> dict[str, Any]:
    memory_snapshot = MemorySnapshot.from_memory(memory, recent_observation_limit=8)
    return {
        "objects": memory.find_object(request.description, top_k=5),
        "similar_episodes": memory.find_similar_episodes(request.description, top_k=5),
        "skills": memory.find_applicable_skills(current_state={}, top_k=5),
        "memory_consolidation": _find_experience_hints(memory, request),
        "failure_patterns": _find_failure_patterns(memory, request, min_confidence=0.7),
        "memory_evidence_summary": _get_memory_evidence_summary(memory, request),
        **memory_snapshot.to_planning_context(),
        "metadata": request.metadata,
        "task_type": request.task_type.value,
        "task_type_hint": request.task_type.value,
        "planner_mode": planner_mode_from_request(request),
    }


def refresh_runtime_planning_context(
    *,
    memory: Any,
    planning_context: dict[str, Any],
    execution_state: dict[str, Any] | None,
    environment_state: dict[str, Any] | None,
    resolve_navigation_state: Callable[
        [dict[str, Any] | None, dict[str, Any] | None], dict[str, Any]
    ],
) -> None:
    nav_state = resolve_navigation_state(execution_state, environment_state)
    if nav_state:
        planning_context["navigation_state"] = nav_state
    if execution_state:
        navigation_report = execution_state.get("navigation_report")
        if isinstance(navigation_report, dict) and navigation_report:
            planning_context["navigation_report"] = dict(navigation_report)
        scene_report = execution_state.get("last_scene_report")
        if isinstance(scene_report, dict) and scene_report:
            planning_context["last_scene_report"] = dict(scene_report)

    planning_context.update(
        MemorySnapshot.from_memory(memory, recent_observation_limit=8).to_planning_context()
    )


def attach_counterfactual_evidence(
    *,
    memory: Any,
    planning_context: dict[str, Any],
    task_description: str,
    failed_subtask: Any,
    top_k: int = 3,
) -> None:
    query = getattr(memory, "counterfactual_query", None)
    if not callable(query):
        return

    subtask_context = dict(getattr(failed_subtask, "context", {}) or {})
    parameters = dict(getattr(failed_subtask, "parameters", {}) or {})
    result = query(
        episode_id=_first_nonempty(
            subtask_context.get("source_episode_id"),
            subtask_context.get("episode_id"),
            parameters.get("source_episode_id"),
        ),
        task_description=task_description,
        failed_action_idx=_optional_int(
            _first_nonempty(
                subtask_context.get("action_index"),
                subtask_context.get("failed_action_idx"),
                parameters.get("failed_action_idx"),
            )
        ),
        desired_effect=_first_nonempty(
            parameters.get("desired_effect"),
            parameters.get("expected_effect"),
            subtask_context.get("desired_effect"),
        ),
        current_state=dict(parameters.get("pre_state") or subtask_context.get("pre_state") or {}),
        top_k=top_k,
    )
    if not isinstance(result, dict):
        return
    summary = planning_context.setdefault("memory_evidence_summary", {})
    if not isinstance(summary, dict):
        summary = {}
        planning_context["memory_evidence_summary"] = summary
    retrieval = summary.setdefault("retrieval", {})
    if not isinstance(retrieval, dict):
        retrieval = {}
        summary["retrieval"] = retrieval
    retrieval["counterfactual"] = _compact_counterfactual_result(result, top_k=top_k)


def _get_memory_evidence_summary(memory: Any, request: TaskRequest) -> dict[str, Any]:
    finder = getattr(memory, "get_memory_evidence_summary", None)
    scene_id = _request_scene_id(request)
    target = _request_target(request)
    fallback = {
        "query_type": "memory_evidence_summary",
        "query": {
            "task_description": request.description,
            "task_type": request.task_type.value,
            "scene_id": scene_id,
            "target": target,
        },
        "retrieval": {},
        "navigation_guidance": {
            "object_approach_history": {
                "scene_id": scene_id,
                "target_key": None,
                "entries": [],
            },
            "avoid_object_approach_candidates": [],
            "prefer_object_approach_candidates": [],
            "risk_reasons": [],
        },
        "runtime": {},
        "metadata": {"available": False, "top_k": 5},
    }
    if not callable(finder):
        return fallback
    result = finder(
        request.description,
        task_type=request.task_type.value,
        scene_id=scene_id,
        target=target,
        top_k=5,
    )
    return result if isinstance(result, dict) else fallback


def _find_experience_hints(memory: Any, request: TaskRequest) -> dict[str, Any]:
    finder = getattr(memory, "find_experience_hints", None)
    if not callable(finder):
        return {
            "query_type": "experience_hints",
            "results": [],
            "scores": [],
            "metadata": {"available": False},
        }
    return finder(request.description, task_type=request.task_type.value, top_k=5)


def _find_failure_patterns(
    memory: Any,
    request: TaskRequest,
    *,
    min_confidence: float,
) -> dict[str, Any]:
    finder = getattr(memory, "find_failure_patterns", None)
    if not callable(finder):
        return {
            "query_type": "failure_patterns",
            "results": [],
            "scores": [],
            "metadata": {"available": False, "min_confidence": min_confidence},
        }
    result = finder(request.description, task_type=request.task_type.value, top_k=5)
    if not isinstance(result, dict):
        return {
            "query_type": "failure_patterns",
            "results": [],
            "scores": [],
            "metadata": {"available": False, "min_confidence": min_confidence},
        }

    filtered_results = []
    filtered_scores = []
    for item in result.get("results", []):
        if not isinstance(item, dict):
            continue
        confidence = _confidence(item)
        if confidence >= min_confidence:
            filtered_results.append(item)
            filtered_scores.append(confidence)

    return {
        **result,
        "results": filtered_results,
        "scores": filtered_scores,
        "metadata": {
            **dict(result.get("metadata", {})),
            "min_confidence": min_confidence,
        },
    }


def _confidence(item: dict[str, Any]) -> float:
    try:
        return float(item.get("confidence", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _compact_counterfactual_result(result: dict[str, Any], *, top_k: int) -> dict[str, Any]:
    compact = {
        "query_type": result.get("query_type", "counterfactual"),
        "query": dict(result.get("query", {})) if isinstance(result.get("query"), dict) else {},
        "results": [],
        "explanation": result.get("explanation", ""),
        "metadata": dict(result.get("metadata", {}))
        if isinstance(result.get("metadata"), dict)
        else {},
    }
    raw_results = result.get("results", [])
    if isinstance(raw_results, list):
        compact["results"] = [
            _strip_counterfactual_item(item)
            for item in raw_results[:top_k]
            if isinstance(item, dict)
        ]
    return compact


def _strip_counterfactual_item(item: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "decision_point",
        "original_action",
        "alternative_action",
        "predicted_effects",
        "supporting_episodes",
        "skills",
        "causal_edges",
        "confidence",
        "explanation",
    )
    return {key: item[key] for key in keep if key in item}


def _first_nonempty(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _request_scene_id(request: TaskRequest) -> str | None:
    value = request.metadata.get("scene_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    environment = request.metadata.get("environment")
    if isinstance(environment, dict):
        value = environment.get("scene_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _request_target(request: TaskRequest) -> dict[str, Any]:
    metadata = request.metadata
    target = metadata.get("target")
    if isinstance(target, dict):
        return dict(target)

    extracted: dict[str, Any] = {}
    for source_key, target_key in (
        ("target_object_id", "object_id"),
        ("object_id", "object_id"),
        ("target_object", "object"),
        ("object", "object"),
        ("target_room_id", "room_id"),
        ("room_id", "room_id"),
        ("target_room", "room"),
        ("room", "room"),
        ("target_region", "region"),
        ("region", "region"),
        ("floor_id", "floor_id"),
    ):
        value = metadata.get(source_key)
        if isinstance(value, str) and value.strip():
            extracted[target_key] = value.strip()
    return extracted
