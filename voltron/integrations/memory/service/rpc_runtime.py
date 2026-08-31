from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from voltron.shared.enums import TaskType
from voltron.shared.contracts import MemoryAdapter
from voltron.shared.models import PerceptionObject, PerceptionRelation, PerceptionReport


@dataclass
class RpcResponse:
    ok: bool
    result: Any = None
    error: str | None = None


def build_rpc_method_table(backend: MemoryAdapter) -> dict[str, Callable[..., Any]]:
    methods = {
        "start_task": backend.start_task,
        "end_task": backend.end_task,
        "reflect": backend.reflect,
        "find_object": backend.find_object,
        "find_objects_near": backend.find_objects_near,
        "find_similar_episodes": backend.find_similar_episodes,
        "find_applicable_skills": backend.find_applicable_skills,
        "predict_action_effects": backend.predict_action_effects,
        "diagnose_effect_cause": backend.diagnose_effect_cause,
        "load_map": backend.load_map,
        "save_map": backend.save_map,
        "update_map": backend.update_map,
        "query_semantic_region": backend.query_semantic_region,
        "query_topology": backend.query_topology,
        "mark_explored": backend.mark_explored,
        "get_exploration_frontiers": backend.get_exploration_frontiers,
        "get_working_state": backend.get_working_state,
        "get_active_regions": backend.get_active_regions,
        "get_recent_observations": backend.get_recent_observations,
        "get_task_context": backend.get_task_context,
        "update_task_context": backend.update_task_context,
        "record_working_observation": backend.record_working_observation,
        "get_completed_episode_context": backend.get_completed_episode_context,
        "annotate_completed_episode": backend.annotate_completed_episode,
        "store_experience_hint": backend.store_experience_hint,
        "get_experience_hint": backend.get_experience_hint,
        "find_experience_hints": backend.find_experience_hints,
        "get_skill_candidate": backend.get_skill_candidate,
        "get_causal_hypothesis": backend.get_causal_hypothesis,
        "record_perception": backend.record_perception,
        "record_navigation_update": backend.record_navigation_update,
        "record_navigation_event": backend.record_navigation_event,
        "record_action": backend.record_action,
    }
    for method_name in (
        "get_object_approach_history",
        "record_object_approach_outcome",
        "get_memory_evidence_summary",
        "counterfactual_query",
        "record_monitor_summary",
        "store_failure_pattern_candidate",
        "get_failure_pattern_candidate",
        "find_failure_patterns",
        "store_semantic_update_candidate",
        "get_semantic_update_candidate",
        "find_semantic_update_candidates",
        "store_skill_candidate",
        "store_causal_hypothesis",
        "promote_skill_candidate",
        "promote_causal_hypothesis",
        "get_consolidation_job",
        "wait_for_consolidation_jobs",
    ):
        method = getattr(backend, method_name, None)
        if callable(method):
            methods[method_name] = method
    return methods


def normalize_rpc_kwargs(*, method: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(kwargs)

    if method == "start_task" and "task_type" in normalized:
        raw_task_type = normalized["task_type"]
        if isinstance(raw_task_type, str):
            normalized["task_type"] = TaskType(raw_task_type)

    if method == "record_perception" and "report" in normalized:
        report_payload = normalized["report"]
        if isinstance(report_payload, dict):
            normalized["report"] = _to_perception_report(report_payload)

    if method == "find_objects_near" and "position" in normalized:
        position = normalized["position"]
        if isinstance(position, list):
            normalized["position"] = tuple(position)

    return normalized


def dispatch_rpc_call(
    *, methods: dict[str, Callable[..., Any]], method: str, kwargs: dict[str, Any]
) -> dict[str, Any]:
    if method not in methods:
        return RpcResponse(ok=False, error=f"unknown_method: {method}").__dict__

    try:
        result = methods[method](**kwargs)
        return RpcResponse(ok=True, result=result).__dict__
    except Exception as exc:
        return RpcResponse(ok=False, error=str(exc)).__dict__


def _to_perception_report(value: dict[str, Any]) -> PerceptionReport:
    objects = [PerceptionObject(**item) for item in value.get("objects", [])]
    relations = [PerceptionRelation(**item) for item in value.get("relations", [])]
    return PerceptionReport(
        objects=objects,
        relations=relations,
        task_complete=bool(value.get("task_complete", False)),
        raw_text=str(value.get("raw_text", "")),
        metadata=dict(value.get("metadata", {})),
    )
