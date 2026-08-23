"""HEMS adapter that exposes a stable memory interface for Voltron agents."""

from __future__ import annotations

import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from . import recording, retrieval
from .tools import (
    bootstrap_runtime,
    episode_context,
    episode_summary,
    persistence,
    persistence_runtime,
    query_runtime,
    scene_map_ingestion,
    serialization,
    spatial_memory,
    task_lifecycle,
    working_memory,
)
from voltron.shared.enums import TaskType
from voltron.shared.errors import AdapterError
from voltron.shared.models import PerceptionReport

LOGGER = logging.getLogger(__name__)


def _scene_map_ingested(ingestion: dict[str, Any]) -> bool:
    return any(int(ingestion.get(key, 0)) > 0 for key in ("regions", "objects", "edges"))


def _navigation_event_target(event: dict[str, Any]) -> str:
    target = event.get("target")
    if isinstance(target, dict):
        for key in ("object", "object_name", "room", "room_name", "region", "name"):
            value = target.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for key in ("object", "object_name", "room", "room_name", "region", "target"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _compact_navigation_event_for_episode(event: dict[str, Any]) -> dict[str, Any]:
    event_type = str(event.get("event_type") or "event").strip() or "event"
    path_plan = event.get("path_plan")
    path_plan = path_plan if isinstance(path_plan, dict) else {}
    target = event.get("target")
    target = target if isinstance(target, dict) else {}
    selected_candidate = _first_mapping(
        event.get("selected_object_approach"),
        path_plan.get("selected_object_approach"),
        target.get("selected_object_approach"),
    )
    candidates = _first_list(
        path_plan.get("object_approach_candidates"),
        target.get("object_approach_candidates"),
    )
    compact = {
        "event_type": event_type,
        "scene_id": event.get("scene_id"),
        "target": _compact_navigation_target(target),
        "success": bool(event.get("success", True)),
        "failure_reason": event.get("failure_reason"),
    }
    if selected_candidate:
        compact["selected_object_approach"] = _compact_navigation_candidate(selected_candidate)
    if candidates or selected_candidate:
        compact["candidate_summary"] = _navigation_candidate_summary(
            candidates=candidates,
            selected_candidate=selected_candidate,
        )
    if path_plan:
        compact["route_summary"] = _navigation_route_summary(path_plan)
    backend_state = event.get("backend_state")
    if isinstance(backend_state, dict):
        compact["backend"] = _compact_navigation_backend_state(backend_state)
    return {key: value for key, value in compact.items() if value not in (None, {}, [])}


def _compact_navigation_target(target: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in ("object", "object_id", "object_name", "room", "room_id", "room_name", "floor_id", "region"):
        value = target.get(key)
        if value not in (None, ""):
            compact[key] = value
    return compact


def _compact_navigation_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in (
        "candidate_id",
        "room_id",
        "room_name",
        "floor_id",
        "approach_distance_m",
        "handoff_distance_m",
        "path_cost",
        "candidate_geometry_score",
        "history_penalty",
        "blocked_by_history",
        "selection_source",
    ):
        value = candidate.get(key)
        if value not in (None, ""):
            compact[key] = value
    signature = spatial_memory.object_approach_candidate_signature(candidate)
    if signature:
        compact["candidate_signature"] = signature
    clearance = _compact_candidate_clearance(candidate.get("nearby_object_evidence"))
    if clearance:
        compact["clearance"] = clearance
    history_summary = candidate.get("history_summary")
    if isinstance(history_summary, dict):
        compact["history_summary"] = {
            key: history_summary.get(key)
            for key in ("failure_count", "success_count", "recent_failure", "last_reason")
            if history_summary.get(key) is not None
        }
    return compact


def _compact_candidate_clearance(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    compact = {}
    for key in (
        "nearest_object_id",
        "nearest_object_name",
        "nearest_object_distance_m",
        "path_nearest_object_id",
        "path_nearest_object_name",
        "path_nearest_object_distance_m",
    ):
        if value.get(key) is not None:
            compact[key] = value.get(key)
    return compact


def _navigation_candidate_summary(
    *,
    candidates: list[Any],
    selected_candidate: dict[str, Any],
) -> dict[str, Any]:
    candidate_ids = [
        item.get("candidate_id")
        for item in candidates
        if isinstance(item, dict) and item.get("candidate_id") is not None
    ]
    return {
        "candidate_count": len(candidates),
        "candidate_ids": candidate_ids[:10],
        "selected_candidate_id": selected_candidate.get("candidate_id"),
    }


def _navigation_route_summary(path_plan: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "found": path_plan.get("found", True),
            "planner": path_plan.get("planner") or path_plan.get("path_backend"),
            "path_cost": path_plan.get("path_cost"),
            "waypoint_count": _list_length(path_plan.get("waypoints")),
            "global_waypoint_count": _list_length(path_plan.get("global_waypoints")),
            "dense_waypoint_count": _list_length(path_plan.get("dense_waypoints")),
            "path_node_count": _list_length(path_plan.get("path_nodes")),
            "failure_reason": path_plan.get("reason") or path_plan.get("error"),
        }.items()
        if value is not None
    }


def _compact_navigation_backend_state(backend_state: dict[str, Any]) -> dict[str, Any]:
    compact = {}
    for key in ("scene_id", "current_room", "current_region", "room_id", "floor_id", "nav_backend", "nav2_profile"):
        value = backend_state.get(key)
        if value not in (None, ""):
            compact[key] = value
    return compact


def _first_mapping(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict) and value:
            return dict(value)
    return {}


def _first_list(*values: Any) -> list[Any]:
    for value in values:
        if isinstance(value, list):
            return list(value)
    return []


def _list_length(value: Any) -> int | None:
    return len(value) if isinstance(value, list) else None


class HEMSAdapter:
    """Adapter over `UnifiedMemorySystem` + `RetrievalAPI`.

    All agents should depend on this adapter instead of direct HEMS internals.
    This keeps the memory backend replaceable.
    """

    def __init__(
        self,
        memory_system: Any | None = None,
        retrieval_api: Any | None = None,
        auto_initialize: bool = True,
        persistence_dir: str | None = None,
        auto_persist: bool = True,
    ):
        deps = self._load_hems_symbols()
        self._deps = deps
        self._auto_persist = bool(auto_persist)
        self._auto_initialize = bool(auto_initialize)
        self._owns_memory_backend = memory_system is None
        self._owns_retrieval_api = retrieval_api is None
        self._persistence_dir = Path(
            persistence_dir
            or os.getenv("VOLTRON_HEMS_MEMORY_DIR")
            or (Path(__file__).resolve().parents[1] / "data" / "hems_memory" / "global")
        )
        self._maps_path = self._persistence_dir / "maps.json"

        if memory_system is None:
            config = deps["HEMSConfig"]()
            memory_system = deps["UnifiedMemorySystem"](config)
            if auto_initialize and not memory_system.is_initialized:
                memory_system.initialize()
        self.memory = memory_system

        if retrieval_api is None:
            retrieval_api = deps["RetrievalAPI"](self.memory)
        self.retrieval = retrieval_api

        self._last_completed_episode: Any | None = None
        self._experience_hints: list[dict[str, Any]] = []
        self._maps: dict[str, dict[str, Any]] = {}
        self._load_persistent_state()

    # ------------------------ Task lifecycle ------------------------

    def start_task(self, task_description: str, task_type: TaskType) -> str:
        hems_task_type = self._map_task_type(task_type)
        episode = self.memory.start_task(task_description, hems_task_type)
        return episode.episode_id

    def end_task(self, outcome: str, failure_reason: str | None = None) -> dict[str, Any]:
        self._annotate_current_episode_before_end()
        hems_outcome = self._map_outcome(outcome)
        episode = self.memory.end_task(hems_outcome, failure_reason=failure_reason)
        self._last_completed_episode = episode
        self._persist_state()
        return task_lifecycle.build_end_task_result(episode=episode, outcome_label=outcome)

    def reflect(self, similar_top_k: int = 5) -> dict[str, Any]:
        return {
            "status": "unsupported",
            "reason": "reflection_owned_by_memory_agent",
            "similar_top_k": similar_top_k,
        }

    # ------------------------ Retrieval ------------------------

    def find_object(self, name: str, attributes: dict[str, Any] | None = None, top_k: int = 5) -> Any:
        return retrieval.find_object(
            retrieval_api=self.retrieval,
            name=name,
            attributes=attributes,
            top_k=top_k,
            serialize_retrieval=self._serialize_retrieval,
        )

    def find_objects_near(self, position: tuple[float, float, float], radius: float = 2.0) -> Any:
        return retrieval.find_objects_near(
            retrieval_api=self.retrieval,
            position=position,
            radius=radius,
            serialize_retrieval=self._serialize_retrieval,
        )

    def find_similar_episodes(self, description: str, top_k: int = 5) -> Any:
        return retrieval.find_similar_episodes(
            retrieval_api=self.retrieval,
            description=description,
            top_k=top_k,
            serialize_retrieval=self._serialize_retrieval,
        )

    def find_applicable_skills(self, current_state: dict[str, Any], top_k: int = 5) -> Any:
        return retrieval.find_applicable_skills(
            retrieval_api=self.retrieval,
            current_state=current_state,
            top_k=top_k,
            serialize_retrieval=self._serialize_retrieval,
        )

    def predict_action_effects(
        self,
        action: str,
        target: str,
        conditions: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
        match_mode: str = "strict",
    ) -> Any:
        return retrieval.predict_action_effects(
            retrieval_api=self.retrieval,
            action=action,
            target=target,
            conditions=conditions,
            parameters=parameters,
            match_mode=match_mode,
            serialize_retrieval=self._serialize_retrieval,
        )

    def diagnose_effect_cause(self, effect: str, value: Any = None) -> Any:
        return retrieval.diagnose_effect_cause(
            retrieval_api=self.retrieval,
            effect=effect,
            value=value,
            serialize_retrieval=self._serialize_retrieval,
        )

    def load_map(self, scene_id: str) -> dict[str, Any]:
        return recording.load_map(maps=self._maps, scene_id=scene_id)

    def save_map(
        self,
        scene_id: str,
        map_payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = recording.save_map(
            maps=self._maps,
            scene_id=scene_id,
            map_payload=map_payload,
            metadata=metadata,
            persist_state=self._persist_state,
            clone_map_entry=spatial_memory.clone_map_entry,
        )
        ingestion = self._ingest_scene_map(scene_id, map_payload, metadata)
        if _scene_map_ingested(ingestion):
            self._maps[scene_id]["metadata"]["semantic_ingestion"] = ingestion
            self._persist_state()
            return recording.load_map(maps=self._maps, scene_id=scene_id)
        return result

    def update_map(
        self,
        scene_id: str,
        delta: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = recording.update_map(
            maps=self._maps,
            scene_id=scene_id,
            delta=delta,
            metadata=metadata,
            persist_state=self._persist_state,
            ensure_map=spatial_memory.ensure_map,
            merge_dicts=spatial_memory.merge_dicts,
            clone_map_entry=spatial_memory.clone_map_entry,
        )
        entry = self._maps.get(scene_id, {})
        ingestion = self._ingest_scene_map(
            scene_id,
            entry.get("map_payload", {}),
            entry.get("metadata", {}),
        )
        if _scene_map_ingested(ingestion):
            entry.setdefault("metadata", {})["semantic_ingestion"] = ingestion
            self._persist_state()
            return recording.load_map(maps=self._maps, scene_id=scene_id)
        return result

    def query_semantic_region(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        return retrieval.query_semantic_region(
            semantic_memory=self.memory.semantic,
            name=name,
            attributes=attributes,
            top_k=top_k,
            serializer=self._serialize_obj,
        )

    def query_topology(self, start: dict[str, Any], goal: dict[str, Any]) -> dict[str, Any]:
        return spatial_memory.query_topology(maps=self._maps, start=start, goal=goal)

    def mark_explored(self, scene_id: str, evidence: dict[str, Any]) -> dict[str, Any]:
        return recording.mark_explored(
            maps=self._maps,
            scene_id=scene_id,
            evidence=evidence,
            persist_state=self._persist_state,
            ensure_map=spatial_memory.ensure_map,
        )

    def get_exploration_frontiers(self, scene_id: str) -> list[dict[str, Any]]:
        return recording.get_exploration_frontiers(maps=self._maps, scene_id=scene_id)

    def get_working_state(self) -> dict[str, Any]:
        return working_memory.get_working_state(
            working_memory=self.memory.working,
            serializer=self._serialize_obj,
        )

    def get_active_regions(self) -> list[str]:
        return working_memory.get_active_regions(working_memory=self.memory.working)

    def get_recent_observations(self, n: int = 10) -> list[dict[str, Any]]:
        return working_memory.get_recent_observations(
            working_memory=self.memory.working,
            n=n,
            serializer=self._serialize_obj,
        )

    def get_task_context(self) -> dict[str, Any]:
        return working_memory.get_task_context(
            working_memory=self.memory.working,
            serializer=self._serialize_obj,
        )

    def update_task_context(self, updates: dict[str, Any]) -> dict[str, Any]:
        return working_memory.update_task_context(
            working_memory=self.memory.working,
            updates=updates,
            merge_dicts=self._merge_dicts,
            serializer=self._serialize_obj,
        )

    def record_working_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        return working_memory.record_working_observation(
            working_memory=self.memory.working,
            observation=observation,
            serializer=self._serialize_obj,
        )

    def get_completed_episode_context(
        self,
        episode_id: str | None = None,
        *,
        recent_observation_limit: int = 20,
    ) -> dict[str, Any]:
        episode = self._resolve_completed_episode(episode_id)
        if episode is None:
            return {
                "episode_id": episode_id,
                "source_integrity": {
                    "from_completed_episode": False,
                    "missing_fields": ["episode"],
                },
            }

        episode_payload = self._serialize_obj(episode)
        if not isinstance(episode_payload, dict):
            episode_payload = {"value": episode_payload}
        context = {
            **episode_payload,
            "episode": episode_payload,
            "episode_id": str(episode_payload.get("episode_id") or getattr(episode, "episode_id", "")),
            "recent_observations": self.get_recent_observations(n=recent_observation_limit),
            "scene_maps": self._summarize_scene_maps(),
            "scene_memory_context": episode_context.build_scene_memory_context(
                maps=self._maps,
                serializer=self._serialize_obj,
            ),
            "source_integrity": {
                "from_completed_episode": True,
                "from_live_task_context": False,
                "missing_fields": self._missing_episode_fields(episode_payload),
            },
        }
        context.setdefault("task_description", getattr(episode, "task_description", ""))
        context.setdefault("failure_reason", getattr(episode, "failure_reason", None))
        outcome = getattr(episode, "outcome", None)
        context.setdefault("outcome", getattr(outcome, "value", outcome))
        return context

    def annotate_completed_episode(self, episode_id: str, annotation: dict[str, Any]) -> dict[str, Any]:
        annotate_episode = getattr(self.memory, "annotate_episode", None)
        if callable(annotate_episode):
            result = annotate_episode(episode_id, dict(annotation))
        else:
            episode = self._resolve_completed_episode(episode_id)
            if episode is None:
                raise KeyError(f"Episode not found: {episode_id}")
            final_state = getattr(episode, "final_state", None)
            if not isinstance(final_state, dict):
                final_state = {}
                setattr(episode, "final_state", final_state)
            annotations = final_state.setdefault("memory_annotations", [])
            stored_annotation = dict(annotation)
            annotations.append(stored_annotation)
            result = {
                "episode_id": episode_id,
                "annotation": stored_annotation,
                "annotation_count": len(annotations),
            }
        self._persist_state()
        return self._serialize_obj(result)

    def store_experience_hint(self, hint: dict[str, Any]) -> dict[str, Any]:
        payload = dict(hint)
        store_hint = getattr(self.memory, "store_experience_hint", None)
        if callable(store_hint):
            hint_id = store_hint(payload)
        else:
            hint_id = str(payload.get("hint_id") or f"hint_{uuid.uuid4().hex[:8]}")
            payload["hint_id"] = hint_id
            self._experience_hints.append(payload)
        self._persist_state()
        return {"hint_id": hint_id, "stored": True}

    def get_experience_hint(self, hint_id: str) -> dict[str, Any] | None:
        get_hint = getattr(self.memory, "get_experience_hint", None)
        if callable(get_hint):
            return self._serialize_obj(get_hint(hint_id))
        for hint in self._experience_hints:
            if hint.get("hint_id") == hint_id:
                return self._serialize_obj(hint)
        return None

    def find_experience_hints(
        self,
        task_description: str,
        task_type: str | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        find_hints = getattr(self.memory, "find_experience_hints", None)
        if callable(find_hints):
            hints = find_hints(task_description, task_type=task_type, top_k=top_k)
            results = [self._serialize_obj(item) for item in hints]
        else:
            results = self._find_fallback_experience_hints(task_description, task_type, top_k)
        score_components = self._experience_hint_score_components(results)
        return {
            "query_type": "experience_hints",
            "query": {"task_description": task_description, "task_type": task_type},
            "results": results,
            "scores": [float(item.get("confidence", 0.0)) for item in results if isinstance(item, dict)],
            "metadata": {"top_k": top_k, "score_components": score_components},
        }

    def store_failure_pattern_candidate(self, pattern: dict[str, Any]) -> dict[str, Any]:
        store_pattern = getattr(self.memory, "store_failure_pattern_candidate", None)
        if not callable(store_pattern):
            return {
                "pattern_id": str(pattern.get("pattern_id") or ""),
                "stored": False,
                "reason": "backend_missing_store_failure_pattern_candidate",
            }
        pattern_id = store_pattern(dict(pattern))
        self._persist_state()
        return {"pattern_id": pattern_id, "stored": True}

    def get_failure_pattern_candidate(self, pattern_id: str) -> dict[str, Any] | None:
        get_pattern = getattr(self.memory, "get_failure_pattern_candidate", None)
        if not callable(get_pattern):
            return None
        return self._serialize_obj(get_pattern(pattern_id))

    def find_failure_patterns(
        self,
        query: str,
        task_type: str | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        find_patterns = getattr(self.memory, "find_failure_patterns", None)
        if not callable(find_patterns):
            return {
                "query_type": "failure_patterns",
                "query": {"query": query, "task_type": task_type},
                "results": [],
                "scores": [],
                "metadata": {"top_k": top_k, "available": False},
            }
        patterns = find_patterns(query, task_type=task_type, top_k=top_k)
        results = [self._serialize_obj(item) for item in patterns]
        return {
            "query_type": "failure_patterns",
            "query": {"query": query, "task_type": task_type},
            "results": results,
            "scores": [float(item.get("confidence", 0.0)) for item in results if isinstance(item, dict)],
            "metadata": {"top_k": top_k},
        }

    def get_memory_evidence_summary(
        self,
        task_description: str,
        task_type: str | None = None,
        scene_id: str | None = None,
        target: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Aggregate bounded memory evidence for planner consumption."""
        normalized_target = dict(target or {})
        hints = self.find_experience_hints(task_description, task_type=task_type, top_k=top_k)
        failures = self.find_failure_patterns(task_description, task_type=task_type, top_k=top_k)
        object_history = self._memory_evidence_object_approach_history(
            scene_id=scene_id,
            target=normalized_target,
            top_k=top_k,
        )
        navigation_guidance = self._build_navigation_guidance(
            object_history=object_history,
            failure_patterns=failures,
            top_k=top_k,
        )
        return {
            "query_type": "memory_evidence_summary",
            "query": {
                "task_description": task_description,
                "task_type": task_type,
                "scene_id": scene_id,
                "target": normalized_target,
            },
            "retrieval": {
                "experience_hints": hints,
                "failure_patterns": failures,
                "causal": self._build_causal_summary(top_k=top_k),
            },
            "navigation_guidance": navigation_guidance,
            "runtime": {
                "working_state": self.get_working_state(),
                "working_evidence": self._get_native_working_evidence(top_k=top_k),
                "task_context": self.get_task_context(),
                "recent_observations": self.get_recent_observations(n=min(max(int(top_k), 1), 10)),
            },
            "metadata": {"available": True, "top_k": top_k},
        }

    def _get_native_working_evidence(self, *, top_k: int) -> dict[str, Any]:
        get_summary = getattr(self.memory.working, "get_evidence_summary", None)
        if not callable(get_summary):
            return {}
        return self._serialize_obj(get_summary(max_items=min(max(int(top_k), 1), 10)))

    def _build_causal_summary(self, *, top_k: int) -> dict[str, Any]:
        recent_observations = self.get_recent_observations(n=min(max(int(top_k), 1), 10))
        negative_evidence = []
        chain_summaries = []
        for observation in recent_observations:
            if not isinstance(observation, dict):
                continue
            action_type = str(observation.get("action_type") or "").strip()
            target = str(observation.get("target") or "").strip()
            for item in observation.get("negative_evidence", []):
                if not isinstance(item, dict):
                    continue
                evidence = {
                    "action_type": action_type,
                    "target": target,
                    **{
                        key: item[key]
                        for key in ("attribute", "expected", "observed", "reason", "edge_id")
                        if key in item
                    },
                }
                negative_evidence.append(evidence)
                if not action_type or not evidence.get("attribute"):
                    continue
                chain_summaries.extend(
                    self._find_causal_chain_summaries(
                        start_action=action_type,
                        end_effect=str(evidence["attribute"]),
                        max_depth=3,
                    )
                )

        chain_summaries = sorted(
            chain_summaries,
            key=lambda item: float(item.get("cumulative_strength", 0.0)),
            reverse=True,
        )[:top_k]
        return {
            "negative_evidence": negative_evidence[:top_k],
            "chain_summaries": chain_summaries,
        }

    def _find_causal_chain_summaries(
        self,
        *,
        start_action: str,
        end_effect: str,
        max_depth: int,
    ) -> list[dict[str, Any]]:
        finder = getattr(self.memory, "find_causal_chain_summaries", None)
        if not callable(finder):
            return []
        result = finder(
            start_action=start_action,
            end_effect=end_effect,
            max_depth=max_depth,
        )
        if not isinstance(result, list):
            return []
        return [
            self._serialize_obj(item)
            for item in result
            if isinstance(item, dict)
        ]

    def counterfactual_query(
        self,
        episode_id: str | None = None,
        task_description: str | None = None,
        failed_action_idx: int | None = None,
        desired_effect: str | None = None,
        current_state: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        query = getattr(self.memory, "counterfactual_query", None)
        if not callable(query):
            return {
                "query_type": "counterfactual",
                "query": {
                    "episode_id": episode_id,
                    "task_description": task_description,
                    "failed_action_idx": failed_action_idx,
                    "desired_effect": desired_effect,
                    "current_state": dict(current_state or {}),
                    "top_k": top_k,
                },
                "results": [],
                "explanation": "backend_missing_counterfactual_query",
                "metadata": {"available": False},
            }
        result = query(
            episode_id=episode_id,
            task_description=task_description,
            failed_action_idx=failed_action_idx,
            desired_effect=desired_effect,
            current_state=dict(current_state or {}),
            top_k=top_k,
        )
        return self._serialize_obj(result)

    def store_semantic_update_candidate(self, update: dict[str, Any]) -> dict[str, Any]:
        store_update = getattr(self.memory, "store_semantic_update_candidate", None)
        if not callable(store_update):
            return {
                "update_id": str(update.get("update_id") or ""),
                "stored": False,
                "reason": "backend_missing_store_semantic_update_candidate",
            }
        update_id = store_update(dict(update))
        self._persist_state()
        return {"update_id": update_id, "stored": True}

    def get_semantic_update_candidate(self, update_id: str) -> dict[str, Any] | None:
        get_update = getattr(self.memory, "get_semantic_update_candidate", None)
        if not callable(get_update):
            return None
        return self._serialize_obj(get_update(update_id))

    def find_semantic_update_candidates(
        self,
        query: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        find_updates = getattr(self.memory, "find_semantic_update_candidates", None)
        if not callable(find_updates):
            return {
                "query_type": "semantic_update_candidates",
                "query": {"query": query},
                "results": [],
                "scores": [],
                "metadata": {"top_k": top_k, "available": False},
            }
        updates = find_updates(query, top_k=top_k)
        results = [self._serialize_obj(item) for item in updates]
        return {
            "query_type": "semantic_update_candidates",
            "query": {"query": query},
            "results": results,
            "scores": [float(item.get("confidence", 0.0)) for item in results if isinstance(item, dict)],
            "metadata": {"top_k": top_k},
        }

    def store_skill_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        store_candidate = getattr(self.memory, "store_skill_candidate", None)
        if not callable(store_candidate):
            return {
                "candidate_id": str(candidate.get("candidate_id") or ""),
                "stored": False,
                "reason": "backend_missing_store_skill_candidate",
            }
        candidate_id = store_candidate(dict(candidate))
        self._persist_state()
        return {"candidate_id": candidate_id, "stored": True}

    def get_skill_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        get_candidate = getattr(self.memory, "get_skill_candidate", None)
        if not callable(get_candidate):
            return None
        return self._serialize_obj(get_candidate(candidate_id))

    def store_causal_hypothesis(self, hypothesis: dict[str, Any]) -> dict[str, Any]:
        store_hypothesis = getattr(self.memory, "store_causal_hypothesis", None)
        if not callable(store_hypothesis):
            return {
                "hypothesis_id": str(hypothesis.get("hypothesis_id") or ""),
                "stored": False,
                "reason": "backend_missing_store_causal_hypothesis",
            }
        hypothesis_id = store_hypothesis(dict(hypothesis))
        self._persist_state()
        return {"hypothesis_id": hypothesis_id, "stored": True}

    def get_causal_hypothesis(self, hypothesis_id: str) -> dict[str, Any] | None:
        get_hypothesis = getattr(self.memory, "get_causal_hypothesis", None)
        if not callable(get_hypothesis):
            return None
        return self._serialize_obj(get_hypothesis(hypothesis_id))

    def promote_skill_candidate(
        self,
        candidate_id: str,
        min_confidence: float = 0.7,
        min_supporting_episodes: int = 1,
        max_contradictions: int = 0,
        allow_conflicts: bool = False,
    ) -> dict[str, Any]:
        promote_candidate = getattr(self.memory, "promote_skill_candidate", None)
        if not callable(promote_candidate):
            return {
                "candidate_id": candidate_id,
                "promoted": False,
                "reason": "backend_missing_promote_skill_candidate",
            }
        try:
            candidate = promote_candidate(
                candidate_id,
                min_confidence=min_confidence,
                min_supporting_episodes=min_supporting_episodes,
                max_contradictions=max_contradictions,
                allow_conflicts=allow_conflicts,
            )
        except (KeyError, ValueError) as exc:
            return {"candidate_id": candidate_id, "promoted": False, "reason": str(exc)}
        self._persist_state()
        return {
            "candidate_id": candidate_id,
            "promoted": True,
            "candidate": self._serialize_obj(candidate),
        }

    def promote_causal_hypothesis(
        self,
        hypothesis_id: str,
        min_confidence: float = 0.7,
        min_supporting_episodes: int = 1,
        max_contradictions: int = 0,
        allow_conflicts: bool = False,
    ) -> dict[str, Any]:
        promote_hypothesis = getattr(self.memory, "promote_causal_hypothesis", None)
        if not callable(promote_hypothesis):
            return {
                "hypothesis_id": hypothesis_id,
                "promoted": False,
                "reason": "backend_missing_promote_causal_hypothesis",
            }
        try:
            hypothesis = promote_hypothesis(
                hypothesis_id,
                min_confidence=min_confidence,
                min_supporting_episodes=min_supporting_episodes,
                max_contradictions=max_contradictions,
                allow_conflicts=allow_conflicts,
            )
        except (KeyError, ValueError) as exc:
            return {"hypothesis_id": hypothesis_id, "promoted": False, "reason": str(exc)}
        self._persist_state()
        return {
            "hypothesis_id": hypothesis_id,
            "promoted": True,
            "hypothesis": self._serialize_obj(hypothesis),
        }

    def get_object_approach_history(
        self,
        scene_id: str,
        target: dict[str, Any],
        top_k: int = 10,
    ) -> dict[str, Any]:
        return recording.get_object_approach_history(
            maps=self._maps,
            scene_id=scene_id,
            target=target,
            top_k=top_k,
            get_task_context=self.get_task_context,
            target_key_builder=spatial_memory.object_approach_target_key,
        )

    def record_object_approach_outcome(
        self,
        scene_id: str,
        target: dict[str, Any],
        candidate: dict[str, Any],
        outcome: str,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return recording.record_object_approach_outcome(
            maps=self._maps,
            scene_id=scene_id,
            target=target,
            candidate=candidate,
            outcome=outcome,
            reason=reason,
            metadata=metadata,
            now_string=(
                str(self.memory.config.clock.now())
                if hasattr(self.memory.config, "clock")
                else None
            ),
            persist_state=self._persist_state,
            ensure_map=spatial_memory.ensure_map,
            merge_dicts=spatial_memory.merge_dicts,
            target_key_builder=spatial_memory.object_approach_target_key,
            candidate_signature_builder=spatial_memory.object_approach_candidate_signature,
        )

    # ------------------------ Updates ------------------------

    def record_perception(self, report: PerceptionReport) -> dict[str, Any]:
        return recording.record_perception(
            report=report,
            deps=self._deps,
            resolve_node=self._resolve_node,
            resolve_node_id_by_name=self._resolve_node_id_by_name,
            parse_relation_type=self._parse_relation_type,
            new_node_id=lambda name: self._new_node_id(name),
            store_memory=self.memory.store,
            update_node=self.memory.semantic.update_node,
            get_edge=self.memory.semantic.get_edge,
            verify_edge=self.memory.semantic.verify_edge,
            add_observation=self.memory.working.add_observation,
        )

    def record_navigation_update(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Update spatial memory evidence from VLN execution."""
        return recording.record_navigation_update(
            payload=payload,
            deps=self._deps,
            resolve_node=self._resolve_node,
            new_node_id=lambda name, prefix="obj": self._new_node_id(name, prefix=prefix),
            store_node=self.memory.store,
            activate_region=self.memory.working.activate_region,
            update_node=self.memory.semantic.update_node,
            add_observation=self.memory.working.add_observation,
            maps=self._maps,
            update_navigation_map=spatial_memory.update_navigation_map,
            persist_state=self._persist_state,
        )

    def record_navigation_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Record an important navigation event as an episode action."""
        event = dict(payload)
        event_type = str(event.get("event_type") or "event").strip() or "event"
        episode_event = _compact_navigation_event_for_episode(event)
        action_payload = {
            "action_type": f"navigation.{event_type}",
            "target": _navigation_event_target(event),
            "parameters": episode_event,
            "success": bool(event.get("success", True)),
            "failure_reason": event.get("failure_reason"),
            "episodic_record": True,
        }
        recorded = self.record_action(action_payload)
        if isinstance(recorded, dict):
            return {
                **recorded,
                "event_type": event_type,
                "source_agent": "NAVIGATION",
            }
        return {"recorded": False, "event_type": event_type}

    def record_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Record VLA execution into current episode and causal memory."""
        return recording.record_action(
            payload=payload,
            action_record_cls=self._deps["ActionRecord"],
            get_current_episode=self.memory.working.get_current_episode,
            update_causal=self.memory.causal.update_from_observation,
            add_observation=self.memory.working.add_observation,
        )

    def record_monitor_summary(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Record a coarse Vision monitor summary into the current episode."""
        return recording.record_monitor_summary(
            payload=payload,
            get_current_episode=self.memory.working.get_current_episode,
            add_observation=self.memory.working.add_observation,
        )

    # ------------------------ Internal helpers ------------------------

    def _map_task_type(self, task_type: TaskType) -> Any:
        hems_task_type = self._deps["TaskType"]
        mapping = {
            TaskType.MANIPULATION: hems_task_type.MANIPULATION,
            TaskType.NAVIGATION: hems_task_type.NAVIGATION,
            TaskType.INTERACTION: hems_task_type.INTERACTION,
            TaskType.OBSERVATION: hems_task_type.OBSERVATION,
        }
        return mapping[task_type]

    def _map_outcome(self, outcome: str) -> Any:
        hems_outcome = self._deps["Outcome"]
        normalized = outcome.lower()
        if normalized == "success":
            return hems_outcome.SUCCESS
        if normalized == "failure":
            return hems_outcome.FAILURE
        return hems_outcome.PARTIAL

    def _resolve_node(self, node_id: str | None, name: str | None) -> Any | None:
        return query_runtime.resolve_node(
            semantic_memory=self.memory.semantic,
            node_id=node_id,
            name=name,
        )

    def _resolve_node_id_by_name(self, name: str) -> str | None:
        return query_runtime.resolve_node_id_by_name(
            semantic_memory=self.memory.semantic,
            name=name,
        )

    def _parse_relation_type(self, relation: str, relation_enum: Any) -> Any:
        return query_runtime.parse_relation_type(relation, relation_enum)

    def _ensure_map(self, scene_id: str) -> dict[str, Any]:
        return spatial_memory.ensure_map(self._maps, scene_id)

    def _annotate_current_episode_before_end(self) -> None:
        working_memory.annotate_current_episode(
            working_memory=self.memory.working,
            get_task_context=self.get_task_context,
            get_working_state=self.get_working_state,
            annotate_episode=episode_summary.annotate_episode,
        )

    def _resolve_completed_episode(self, episode_id: str | None) -> Any | None:
        if episode_id is None:
            return self._last_completed_episode
        if self._last_completed_episode is not None:
            last_id = getattr(self._last_completed_episode, "episode_id", None)
            if str(last_id) == str(episode_id):
                return self._last_completed_episode
        episodic = getattr(self.memory, "episodic", None)
        get_episode = getattr(episodic, "get_episode", None)
        if callable(get_episode):
            return get_episode(episode_id)
        return None

    def _summarize_scene_maps(self) -> list[dict[str, Any]]:
        summaries = []
        for scene_id, entry in self._maps.items():
            map_payload = entry.get("map_payload", {})
            summaries.append(
                {
                    "scene_id": scene_id,
                    "metadata": self._serialize_obj(entry.get("metadata", {})),
                    "map_keys": sorted(map_payload) if isinstance(map_payload, dict) else [],
                }
            )
        return summaries

    @staticmethod
    def _missing_episode_fields(episode_payload: dict[str, Any]) -> list[str]:
        required = ("action_sequence", "initial_state", "final_state")
        return [field for field in required if field not in episode_payload]

    def _find_fallback_experience_hints(
        self,
        task_description: str,
        task_type: str | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        terms = [term for term in task_description.lower().split() if term]
        matches = []
        for hint in self._experience_hints:
            if task_type is not None and hint.get("task_type") not in (None, task_type):
                continue
            haystack = " ".join(
                str(hint.get(key, ""))
                for key in ("task_description", "summary", "hint_type")
            ).lower()
            if not terms or all(term in haystack for term in terms):
                matches.append(dict(hint))
        matches.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        return matches[:top_k]

    def _experience_hint_score_components(
        self,
        results: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        get_components = getattr(self.memory, "get_experience_hint_score_components", None)
        if not callable(get_components):
            return {}
        components: dict[str, dict[str, Any]] = {}
        for result in results:
            if not isinstance(result, dict):
                continue
            hint_id = result.get("hint_id")
            if not hint_id:
                continue
            item_components = get_components(str(hint_id))
            if isinstance(item_components, dict) and item_components:
                components[str(hint_id)] = self._serialize_obj(item_components)
        return components

    def _memory_evidence_object_approach_history(
        self,
        *,
        scene_id: str | None,
        target: dict[str, Any],
        top_k: int,
    ) -> dict[str, Any]:
        normalized_scene_id = str(scene_id or "").strip()
        if not normalized_scene_id or not target:
            return {
                "scene_id": normalized_scene_id or None,
                "target_key": spatial_memory.object_approach_target_key(target) or None,
                "entries": [],
            }
        return self.get_object_approach_history(
            scene_id=normalized_scene_id,
            target=target,
            top_k=top_k,
        )

    def _build_navigation_guidance(
        self,
        *,
        object_history: dict[str, Any],
        failure_patterns: dict[str, Any],
        top_k: int,
    ) -> dict[str, Any]:
        grouped: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}
        for entry in object_history.get("entries", []):
            if not isinstance(entry, dict):
                continue
            signature = entry.get("candidate_signature")
            if not isinstance(signature, dict) or not signature:
                signature = spatial_memory.object_approach_candidate_signature(dict(entry.get("candidate") or {}))
            if not signature:
                continue
            key = tuple(sorted((str(name), repr(value)) for name, value in signature.items()))
            bucket = grouped.setdefault(
                key,
                {
                    "candidate_signature": dict(signature),
                    "failure_count": 0,
                    "success_count": 0,
                    "last_outcome": None,
                    "reason": None,
                },
            )
            outcome = str(entry.get("outcome") or "").strip().lower()
            bucket["last_outcome"] = outcome or None
            if outcome == "success":
                bucket["success_count"] += 1
            elif outcome in {"failure", "failed", "timeout", "blocked"}:
                bucket["failure_count"] += 1
                if entry.get("reason"):
                    bucket["reason"] = entry.get("reason")
            elif entry.get("reason") and bucket.get("reason") is None:
                bucket["reason"] = entry.get("reason")

        avoid = []
        prefer = []
        for bucket in grouped.values():
            if int(bucket.get("failure_count", 0)) > 0:
                avoid.append(
                    {
                        "candidate_signature": dict(bucket["candidate_signature"]),
                        "reason": bucket.get("reason"),
                        "failure_count": int(bucket.get("failure_count", 0)),
                        "last_outcome": bucket.get("last_outcome"),
                    }
                )
            if int(bucket.get("success_count", 0)) > 0:
                prefer.append(
                    {
                        "candidate_signature": dict(bucket["candidate_signature"]),
                        "success_count": int(bucket.get("success_count", 0)),
                        "last_outcome": bucket.get("last_outcome"),
                    }
                )

        avoid.sort(key=lambda item: int(item.get("failure_count", 0)), reverse=True)
        prefer.sort(key=lambda item: int(item.get("success_count", 0)), reverse=True)
        risk_reasons = self._memory_evidence_risk_reasons(avoid, failure_patterns, top_k=top_k)
        limit = max(int(top_k), 0)
        return {
            "object_approach_history": object_history,
            "avoid_object_approach_candidates": avoid[:limit],
            "prefer_object_approach_candidates": prefer[:limit],
            "risk_reasons": risk_reasons,
        }

    @staticmethod
    def _memory_evidence_risk_reasons(
        avoid_candidates: list[dict[str, Any]],
        failure_patterns: dict[str, Any],
        *,
        top_k: int,
    ) -> list[str]:
        reasons: list[str] = []
        for item in avoid_candidates:
            reason = item.get("reason")
            if reason and str(reason) not in reasons:
                reasons.append(str(reason))
        for item in failure_patterns.get("results", []):
            if not isinstance(item, dict):
                continue
            summary = item.get("summary") or item.get("description") or item.get("reason")
            if summary and str(summary) not in reasons:
                reasons.append(str(summary))
        return reasons[: max(int(top_k), 0)]

    def _load_persistent_state(self) -> None:
        persistence_runtime.load_persistent_state(
            auto_persist=self._auto_persist,
            persistence_dir=self._persistence_dir,
            memory=self.memory,
            maps_path=self._maps_path,
            maps=self._maps,
            load_maps_payload=persistence.load_maps_payload,
            quarantine_persistence_dir=persistence.quarantine_persistence_dir,
            reset_runtime_memory=self._reset_runtime_memory,
            logger=LOGGER,
        )
        ingested_any = False
        for scene_id, entry in self._maps.items():
            ingestion = self._ingest_scene_map(
                scene_id,
                entry.get("map_payload", {}),
                entry.get("metadata", {}),
            )
            if _scene_map_ingested(ingestion):
                entry.setdefault("metadata", {})["semantic_ingestion"] = ingestion
                ingested_any = True
        if ingested_any:
            self._persist_state()

    def _reset_runtime_memory(self) -> None:
        reset = bootstrap_runtime.reset_runtime_memory(
            deps=self._deps,
            auto_initialize=self._auto_initialize,
            owns_memory_backend=self._owns_memory_backend,
            owns_retrieval_api=self._owns_retrieval_api,
            memory=self.memory,
            retrieval=self.retrieval,
        )
        self.memory = reset["memory"]
        self.retrieval = reset["retrieval"]

    def _persist_state(self) -> None:
        persistence_runtime.persist_state(
            auto_persist=self._auto_persist,
            persistence_dir=self._persistence_dir,
            memory=self.memory,
            maps_path=self._maps_path,
            maps=self._maps,
            serializer=self._serialize_obj,
            write_maps_payload=persistence.write_maps_payload,
        )

    def _ingest_scene_map(
        self,
        scene_id: str,
        map_payload: dict[str, Any],
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        store_memory = getattr(self.memory, "store", None)
        semantic_memory = getattr(self.memory, "semantic", None)
        get_node = getattr(semantic_memory, "get_node", None)
        update_node = getattr(semantic_memory, "update_node", None)
        if not callable(store_memory) or not isinstance(map_payload, dict):
            return {"scene_id": scene_id, "regions": 0, "objects": 0, "edges": 0}
        return scene_map_ingestion.ingest_scene_map(
            scene_id=scene_id,
            map_payload=map_payload,
            metadata=metadata,
            deps=self._deps,
            store_memory=store_memory,
            get_node=get_node if callable(get_node) else None,
            update_node=update_node if callable(update_node) else None,
        )

    def _merge_dicts(self, base: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
        return spatial_memory.merge_dicts(base, delta)

    @staticmethod
    def _new_node_id(name: str, prefix: str = "obj") -> str:
        slug = "_".join(name.lower().strip().split()) or "unnamed"
        return f"{prefix}_{slug}_{uuid.uuid4().hex[:6]}"

    def _serialize_retrieval(self, retrieval_result: Any) -> dict[str, Any]:
        return serialization.serialize_retrieval(
            retrieval_result=retrieval_result,
            serializer=self._serialize_obj,
        )

    def _serialize_obj(self, value: Any) -> Any:
        return serialization.serialize_obj(value)

    @staticmethod
    def _load_hems_symbols() -> dict[str, Any]:
        return bootstrap_runtime.load_hems_symbols(
            import_symbols=HEMSAdapter._import_symbols,
            repo_root=Path(__file__).resolve().parents[2],
            sys_path=sys.path,
            adapter_error_cls=AdapterError,
        )

    @staticmethod
    def _import_symbols() -> dict[str, Any]:
        return bootstrap_runtime.import_symbols()
