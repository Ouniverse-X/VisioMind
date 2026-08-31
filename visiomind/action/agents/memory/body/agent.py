from __future__ import annotations

import copy
import itertools
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from visiomind.action.shared.enums import TaskType
from visiomind.action.integrations.memory.hems.backend import HEMSAdapter
from visiomind.action.shared.models import PerceptionReport
from . import consolidation, executor, reflection as native_reflection


class MemoryAgent:
    def __init__(
        self,
        backend: Any | None = None,
        *,
        extractor: Any | None = None,
        experience_extraction_enabled: bool = False,
        experience_consolidation_async: bool = False,
        min_confidence_to_write: float = 0.4,
        min_confidence_to_promote: float = 0.7,
        **backend_kwargs: Any,
    ) -> None:
        self._backend = executor.build_backend(
            backend=backend,
            backend_factory=HEMSAdapter,
            backend_kwargs=backend_kwargs,
        )
        self._extractor = extractor
        self._experience_extraction_enabled = bool(experience_extraction_enabled)
        self._experience_consolidation_async = bool(experience_consolidation_async)
        self._min_confidence_to_write = float(min_confidence_to_write)
        self._min_confidence_to_promote = float(min_confidence_to_promote)
        self._last_completed_episode_id: str | None = None
        self._consolidation_lock = threading.Lock()
        self._consolidation_counter = itertools.count(1)
        self._consolidation_executor: ThreadPoolExecutor | None = None
        self._consolidation_futures: dict[str, Future] = {}
        self._consolidation_jobs: dict[str, dict[str, Any]] = {}

    @property
    def backend(self) -> Any:
        return self._backend

    @property
    def last_completed_episode_id(self) -> str | None:
        return self._last_completed_episode_id

    def start_task(self, task_description: str, task_type: TaskType) -> str:
        return executor.call_backend_method(
            self._backend, "start_task", task_description, task_type
        )

    def end_task(self, outcome: str, failure_reason: str | None = None) -> dict[str, Any]:
        result = executor.call_backend_method(
            self._backend,
            "end_task",
            outcome,
            failure_reason=failure_reason,
        )
        if isinstance(result, dict):
            episode_id = result.get("episode_id")
            self._last_completed_episode_id = str(episode_id) if episode_id else None
        return result

    def reflect(self, similar_top_k: int = 5) -> dict[str, Any]:
        episode_context = executor.call_backend_method(
            self._backend,
            "get_completed_episode_context",
            episode_id=self._last_completed_episode_id,
        )
        similar_episodes = self._find_similar_for_reflection(
            episode_context, similar_top_k=similar_top_k
        )
        reflection = native_reflection.build_reflection_evidence(
            episode_context,
            similar_episodes=similar_episodes,
        )
        self._annotate_reflection_evidence(reflection)
        if not self._experience_extraction_enabled:
            return reflection
        if self._experience_consolidation_async:
            consolidation_result = self._schedule_consolidation(
                episode_context=episode_context,
                reflection_evidence=reflection,
            )
        else:
            consolidation_result = self._consolidate_completed_episode(
                episode_context=episode_context,
                reflection_evidence=reflection,
            )
        if isinstance(reflection, dict):
            return {**reflection, "memory_consolidation": consolidation_result}
        return {"reflection": reflection, "memory_consolidation": consolidation_result}

    def get_consolidation_job(
        self, job_id: str | None = None
    ) -> dict[str, Any] | list[dict[str, Any]]:
        with self._consolidation_lock:
            if job_id is None:
                return [dict(job) for job in self._consolidation_jobs.values()]
            job = self._consolidation_jobs.get(job_id)
            return dict(job) if job is not None else {"job_id": job_id, "status": "not_found"}

    def wait_for_consolidation_jobs(self, timeout_s: float | None = None) -> list[dict[str, Any]]:
        deadline = None if timeout_s is None else time.monotonic() + max(0.0, float(timeout_s))
        with self._consolidation_lock:
            futures = list(self._consolidation_futures.items())
        for _, future in futures:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                future.result(timeout=remaining)
            except Exception:
                pass
        jobs = self.get_consolidation_job()
        return jobs if isinstance(jobs, list) else [jobs]

    def _consolidate_completed_episode(
        self,
        *,
        episode_context: dict[str, Any],
        reflection_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        return consolidation.consolidate_completed_episode(
            backend=self._backend,
            extractor=self._extractor,
            episode_id=self._last_completed_episode_id,
            episode_context=episode_context,
            reflection_evidence=reflection_evidence,
            min_confidence_to_write=self._min_confidence_to_write,
            min_confidence_to_promote=self._min_confidence_to_promote,
        )

    def _schedule_consolidation(
        self,
        *,
        episode_context: dict[str, Any],
        reflection_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        episode_id = str(
            episode_context.get("episode_id") or self._last_completed_episode_id or "unknown"
        )
        job_id = f"consolidation_{episode_id}_{next(self._consolidation_counter)}"
        job = {
            "ok": None,
            "job_id": job_id,
            "episode_id": episode_id,
            "status": "scheduled",
            "mode": "async",
        }
        with self._consolidation_lock:
            self._consolidation_jobs[job_id] = dict(job)
        future = self._ensure_consolidation_executor().submit(
            self._run_consolidation_job,
            job_id,
            copy.deepcopy(episode_context),
            copy.deepcopy(reflection_evidence),
        )
        with self._consolidation_lock:
            self._consolidation_futures[job_id] = future
        return job

    def _ensure_consolidation_executor(self) -> ThreadPoolExecutor:
        if self._consolidation_executor is None:
            self._consolidation_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="memory-consolidation",
            )
        return self._consolidation_executor

    def _run_consolidation_job(
        self,
        job_id: str,
        episode_context: dict[str, Any],
        reflection_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        with self._consolidation_lock:
            self._consolidation_jobs[job_id]["status"] = "running"
        result = self._consolidate_completed_episode(
            episode_context=episode_context,
            reflection_evidence=reflection_evidence,
        )
        with self._consolidation_lock:
            self._consolidation_jobs[job_id].update(
                {
                    "ok": bool(result.get("ok")) if isinstance(result, dict) else False,
                    "status": "completed"
                    if isinstance(result, dict) and result.get("ok")
                    else "failed",
                    "result": result,
                }
            )
            return dict(self._consolidation_jobs[job_id])

    def _find_similar_for_reflection(
        self, episode_context: dict[str, Any], *, similar_top_k: int
    ) -> Any:
        task_description = str(episode_context.get("task_description") or "").strip()
        find_similar = getattr(self._backend, "find_similar_episodes", None)
        if not task_description or not callable(find_similar) or similar_top_k <= 0:
            return None
        try:
            return find_similar(task_description, top_k=similar_top_k)
        except Exception:
            return None

    def _annotate_reflection_evidence(self, reflection: dict[str, Any]) -> None:
        episode_id = str(reflection.get("episode_id") or "")
        if not episode_id:
            return
        annotate = getattr(self._backend, "annotate_completed_episode", None)
        if not callable(annotate):
            return
        try:
            annotate(episode_id, native_reflection.reflection_annotation(reflection))
        except Exception:
            reflection["annotation_error"] = "reflection_evidence_write_failed"

    def find_object(
        self, name: str, attributes: dict[str, Any] | None = None, top_k: int = 5
    ) -> Any:
        return executor.call_backend_method(
            self._backend,
            "find_object",
            name,
            attributes=attributes,
            top_k=top_k,
        )

    def find_objects_near(self, position: tuple[float, float, float], radius: float = 2.0) -> Any:
        return executor.call_backend_method(
            self._backend, "find_objects_near", position, radius=radius
        )

    def find_similar_episodes(self, description: str, top_k: int = 5) -> Any:
        return executor.call_backend_method(
            self._backend, "find_similar_episodes", description, top_k=top_k
        )

    def find_applicable_skills(self, current_state: dict[str, Any], top_k: int = 5) -> Any:
        return executor.call_backend_method(
            self._backend, "find_applicable_skills", current_state, top_k=top_k
        )

    def predict_action_effects(
        self,
        action: str,
        target: str,
        conditions: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
        match_mode: str = "strict",
    ) -> Any:
        return executor.call_backend_method(
            self._backend,
            "predict_action_effects",
            action,
            target,
            conditions=conditions,
            parameters=parameters,
            match_mode=match_mode,
        )

    def diagnose_effect_cause(self, effect: str, value: Any = None) -> Any:
        return executor.call_backend_method(
            self._backend, "diagnose_effect_cause", effect, value=value
        )

    def load_map(self, scene_id: str) -> dict[str, Any]:
        return executor.call_backend_method(self._backend, "load_map", scene_id)

    def save_map(
        self,
        scene_id: str,
        map_payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return executor.call_backend_method(
            self._backend,
            "save_map",
            scene_id,
            map_payload,
            metadata=metadata,
        )

    def update_map(
        self,
        scene_id: str,
        delta: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return executor.call_backend_method(
            self._backend,
            "update_map",
            scene_id,
            delta,
            metadata=metadata,
        )

    def query_semantic_region(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> Any:
        return executor.call_backend_method(
            self._backend,
            "query_semantic_region",
            name,
            attributes=attributes,
            top_k=top_k,
        )

    def query_topology(self, start: dict[str, Any], goal: dict[str, Any]) -> dict[str, Any]:
        return executor.call_backend_method(self._backend, "query_topology", start, goal)

    def mark_explored(self, scene_id: str, evidence: dict[str, Any]) -> dict[str, Any]:
        return executor.call_backend_method(self._backend, "mark_explored", scene_id, evidence)

    def get_exploration_frontiers(self, scene_id: str) -> list[dict[str, Any]]:
        return executor.call_backend_method(self._backend, "get_exploration_frontiers", scene_id)

    def get_working_state(self) -> dict[str, Any]:
        return executor.call_backend_method(self._backend, "get_working_state")

    def get_active_regions(self) -> list[str]:
        return executor.call_backend_method(self._backend, "get_active_regions")

    def get_recent_observations(self, n: int = 10) -> list[dict[str, Any]]:
        return executor.call_backend_method(self._backend, "get_recent_observations", n=n)

    def get_task_context(self) -> dict[str, Any]:
        return executor.call_backend_method(self._backend, "get_task_context")

    def update_task_context(self, updates: dict[str, Any]) -> dict[str, Any]:
        return executor.call_backend_method(self._backend, "update_task_context", updates=updates)

    def record_working_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        return executor.call_backend_method(
            self._backend,
            "record_working_observation",
            observation=observation,
        )

    def get_completed_episode_context(
        self,
        episode_id: str | None = None,
        *,
        recent_observation_limit: int = 20,
    ) -> dict[str, Any]:
        return executor.call_backend_method(
            self._backend,
            "get_completed_episode_context",
            episode_id=episode_id,
            recent_observation_limit=recent_observation_limit,
        )

    def annotate_completed_episode(
        self, episode_id: str, annotation: dict[str, Any]
    ) -> dict[str, Any]:
        return executor.call_backend_method(
            self._backend, "annotate_completed_episode", episode_id, annotation
        )

    def store_experience_hint(self, hint: dict[str, Any]) -> dict[str, Any]:
        return executor.call_backend_method(self._backend, "store_experience_hint", hint)

    def get_experience_hint(self, hint_id: str) -> dict[str, Any] | None:
        return executor.call_backend_method(self._backend, "get_experience_hint", hint_id)

    def store_failure_pattern_candidate(self, pattern: dict[str, Any]) -> dict[str, Any]:
        return executor.call_backend_method(
            self._backend, "store_failure_pattern_candidate", pattern
        )

    def get_failure_pattern_candidate(self, pattern_id: str) -> dict[str, Any] | None:
        return executor.call_backend_method(
            self._backend, "get_failure_pattern_candidate", pattern_id
        )

    def find_failure_patterns(
        self,
        query: str,
        task_type: str | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        return executor.call_backend_method(
            self._backend,
            "find_failure_patterns",
            query,
            task_type=task_type,
            top_k=top_k,
        )

    def store_semantic_update_candidate(self, update: dict[str, Any]) -> dict[str, Any]:
        return executor.call_backend_method(
            self._backend, "store_semantic_update_candidate", update
        )

    def get_semantic_update_candidate(self, update_id: str) -> dict[str, Any] | None:
        return executor.call_backend_method(
            self._backend, "get_semantic_update_candidate", update_id
        )

    def find_semantic_update_candidates(
        self,
        query: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        return executor.call_backend_method(
            self._backend,
            "find_semantic_update_candidates",
            query,
            top_k=top_k,
        )

    def store_skill_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        return executor.call_backend_method(self._backend, "store_skill_candidate", candidate)

    def get_skill_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        return executor.call_backend_method(self._backend, "get_skill_candidate", candidate_id)

    def store_causal_hypothesis(self, hypothesis: dict[str, Any]) -> dict[str, Any]:
        return executor.call_backend_method(self._backend, "store_causal_hypothesis", hypothesis)

    def get_causal_hypothesis(self, hypothesis_id: str) -> dict[str, Any] | None:
        return executor.call_backend_method(self._backend, "get_causal_hypothesis", hypothesis_id)

    def promote_skill_candidate(
        self,
        candidate_id: str,
        min_confidence: float = 0.7,
        min_supporting_episodes: int = 1,
        max_contradictions: int = 0,
        allow_conflicts: bool = False,
    ) -> dict[str, Any]:
        return executor.call_backend_method(
            self._backend,
            "promote_skill_candidate",
            candidate_id,
            min_confidence=min_confidence,
            min_supporting_episodes=min_supporting_episodes,
            max_contradictions=max_contradictions,
            allow_conflicts=allow_conflicts,
        )

    def promote_causal_hypothesis(
        self,
        hypothesis_id: str,
        min_confidence: float = 0.7,
        min_supporting_episodes: int = 1,
        max_contradictions: int = 0,
        allow_conflicts: bool = False,
    ) -> dict[str, Any]:
        return executor.call_backend_method(
            self._backend,
            "promote_causal_hypothesis",
            hypothesis_id,
            min_confidence=min_confidence,
            min_supporting_episodes=min_supporting_episodes,
            max_contradictions=max_contradictions,
            allow_conflicts=allow_conflicts,
        )

    def find_experience_hints(
        self,
        task_description: str,
        task_type: str | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        return executor.call_backend_method(
            self._backend,
            "find_experience_hints",
            task_description,
            task_type=task_type,
            top_k=top_k,
        )

    def get_memory_evidence_summary(
        self,
        task_description: str,
        task_type: str | None = None,
        scene_id: str | None = None,
        target: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        return executor.call_backend_method(
            self._backend,
            "get_memory_evidence_summary",
            task_description,
            task_type=task_type,
            scene_id=scene_id,
            target=target,
            top_k=top_k,
        )

    def counterfactual_query(
        self,
        episode_id: str | None = None,
        task_description: str | None = None,
        failed_action_idx: int | None = None,
        desired_effect: str | None = None,
        current_state: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        return executor.call_backend_method(
            self._backend,
            "counterfactual_query",
            episode_id=episode_id,
            task_description=task_description,
            failed_action_idx=failed_action_idx,
            desired_effect=desired_effect,
            current_state=current_state,
            top_k=top_k,
        )

    def get_object_approach_history(
        self,
        scene_id: str,
        target: dict[str, Any],
        top_k: int = 10,
    ) -> dict[str, Any]:
        return executor.call_backend_method(
            self._backend,
            "get_object_approach_history",
            scene_id=scene_id,
            target=target,
            top_k=top_k,
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
        return executor.call_backend_method(
            self._backend,
            "record_object_approach_outcome",
            scene_id=scene_id,
            target=target,
            candidate=candidate,
            outcome=outcome,
            reason=reason,
            metadata=metadata,
        )

    def record_perception(self, report: PerceptionReport) -> dict[str, Any]:
        return executor.call_backend_method(self._backend, "record_perception", report)

    def record_navigation_update(self, payload: dict[str, Any]) -> dict[str, Any]:
        return executor.call_backend_method(self._backend, "record_navigation_update", payload)

    def record_navigation_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        return executor.call_backend_method(self._backend, "record_navigation_event", payload)

    def record_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        return executor.call_backend_method(self._backend, "record_action", payload)

    def record_monitor_summary(self, payload: dict[str, Any]) -> dict[str, Any]:
        return executor.call_backend_method(self._backend, "record_monitor_summary", payload)
