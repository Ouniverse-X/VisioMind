from __future__ import annotations

from dataclasses import asdict
from typing import Any

import requests

from voltron.shared.errors import AdapterError
from voltron.shared.models import PerceptionReport


class MemoryAgentClient:
    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8070/rpc",
        timeout_s: float = 15,
        session: requests.Session | None = None,
    ):
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.session = session or requests.Session()

    def start_task(self, task_description: str, task_type: Any) -> str:
        task_type_value = getattr(task_type, "value", task_type)
        return self._rpc("start_task", task_description=task_description, task_type=task_type_value)

    def end_task(self, outcome: str, failure_reason: str | None = None) -> dict[str, Any]:
        return self._rpc("end_task", outcome=outcome, failure_reason=failure_reason)

    def reflect(self, similar_top_k: int = 5) -> dict[str, Any]:
        return self._rpc("reflect", similar_top_k=similar_top_k)

    def get_consolidation_job(
        self, job_id: str | None = None
    ) -> dict[str, Any] | list[dict[str, Any]]:
        return self._rpc("get_consolidation_job", job_id=job_id)

    def wait_for_consolidation_jobs(self, timeout_s: float | None = None) -> list[dict[str, Any]]:
        return self._rpc("wait_for_consolidation_jobs", timeout_s=timeout_s)

    def find_object(
        self, name: str, attributes: dict[str, Any] | None = None, top_k: int = 5
    ) -> Any:
        return self._rpc("find_object", name=name, attributes=attributes, top_k=top_k)

    def find_objects_near(self, position: tuple[float, float, float], radius: float = 2.0) -> Any:
        return self._rpc("find_objects_near", position=list(position), radius=radius)

    def find_similar_episodes(self, description: str, top_k: int = 5) -> Any:
        return self._rpc("find_similar_episodes", description=description, top_k=top_k)

    def find_applicable_skills(self, current_state: dict[str, Any], top_k: int = 5) -> Any:
        return self._rpc("find_applicable_skills", current_state=current_state, top_k=top_k)

    def predict_action_effects(
        self,
        action: str,
        target: str,
        conditions: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
        match_mode: str = "strict",
    ) -> Any:
        return self._rpc(
            "predict_action_effects",
            action=action,
            target=target,
            conditions=conditions,
            parameters=parameters,
            match_mode=match_mode,
        )

    def diagnose_effect_cause(self, effect: str, value: Any = None) -> Any:
        return self._rpc("diagnose_effect_cause", effect=effect, value=value)

    def load_map(self, scene_id: str) -> dict[str, Any]:
        return self._rpc("load_map", scene_id=scene_id)

    def save_map(
        self,
        scene_id: str,
        map_payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._rpc("save_map", scene_id=scene_id, map_payload=map_payload, metadata=metadata)

    def update_map(
        self,
        scene_id: str,
        delta: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._rpc(
            "update_map",
            scene_id=scene_id,
            delta=delta,
            metadata=metadata,
        )

    def query_semantic_region(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> Any:
        return self._rpc("query_semantic_region", name=name, attributes=attributes, top_k=top_k)

    def query_topology(self, start: dict[str, Any], goal: dict[str, Any]) -> dict[str, Any]:
        return self._rpc("query_topology", start=start, goal=goal)

    def mark_explored(self, scene_id: str, evidence: dict[str, Any]) -> dict[str, Any]:
        return self._rpc("mark_explored", scene_id=scene_id, evidence=evidence)

    def get_exploration_frontiers(self, scene_id: str) -> list[dict[str, Any]]:
        return self._rpc("get_exploration_frontiers", scene_id=scene_id)

    def get_working_state(self) -> dict[str, Any]:
        return self._rpc("get_working_state")

    def get_active_regions(self) -> list[str]:
        return self._rpc("get_active_regions")

    def get_recent_observations(self, n: int = 10) -> list[dict[str, Any]]:
        return self._rpc("get_recent_observations", n=n)

    def get_task_context(self) -> dict[str, Any]:
        return self._rpc("get_task_context")

    def update_task_context(self, updates: dict[str, Any]) -> dict[str, Any]:
        return self._rpc("update_task_context", updates=updates)

    def record_working_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        return self._rpc("record_working_observation", observation=observation)

    def get_completed_episode_context(
        self,
        episode_id: str | None = None,
        *,
        recent_observation_limit: int = 20,
    ) -> dict[str, Any]:
        return self._rpc(
            "get_completed_episode_context",
            episode_id=episode_id,
            recent_observation_limit=recent_observation_limit,
        )

    def annotate_completed_episode(
        self, episode_id: str, annotation: dict[str, Any]
    ) -> dict[str, Any]:
        return self._rpc("annotate_completed_episode", episode_id=episode_id, annotation=annotation)

    def store_experience_hint(self, hint: dict[str, Any]) -> dict[str, Any]:
        return self._rpc("store_experience_hint", hint=hint)

    def get_experience_hint(self, hint_id: str) -> dict[str, Any] | None:
        return self._rpc("get_experience_hint", hint_id=hint_id)

    def store_failure_pattern_candidate(self, pattern: dict[str, Any]) -> dict[str, Any]:
        return self._rpc("store_failure_pattern_candidate", pattern=pattern)

    def get_failure_pattern_candidate(self, pattern_id: str) -> dict[str, Any] | None:
        return self._rpc("get_failure_pattern_candidate", pattern_id=pattern_id)

    def find_failure_patterns(
        self,
        query: str,
        task_type: str | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        return self._rpc(
            "find_failure_patterns",
            query=query,
            task_type=task_type,
            top_k=top_k,
        )

    def store_semantic_update_candidate(self, update: dict[str, Any]) -> dict[str, Any]:
        return self._rpc("store_semantic_update_candidate", update=update)

    def get_semantic_update_candidate(self, update_id: str) -> dict[str, Any] | None:
        return self._rpc("get_semantic_update_candidate", update_id=update_id)

    def find_semantic_update_candidates(
        self,
        query: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        return self._rpc("find_semantic_update_candidates", query=query, top_k=top_k)

    def store_skill_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        return self._rpc("store_skill_candidate", candidate=candidate)

    def get_skill_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        return self._rpc("get_skill_candidate", candidate_id=candidate_id)

    def store_causal_hypothesis(self, hypothesis: dict[str, Any]) -> dict[str, Any]:
        return self._rpc("store_causal_hypothesis", hypothesis=hypothesis)

    def get_causal_hypothesis(self, hypothesis_id: str) -> dict[str, Any] | None:
        return self._rpc("get_causal_hypothesis", hypothesis_id=hypothesis_id)

    def promote_skill_candidate(
        self,
        candidate_id: str,
        min_confidence: float = 0.7,
        min_supporting_episodes: int = 1,
        max_contradictions: int = 0,
        allow_conflicts: bool = False,
    ) -> dict[str, Any]:
        return self._rpc(
            "promote_skill_candidate",
            candidate_id=candidate_id,
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
        return self._rpc(
            "promote_causal_hypothesis",
            hypothesis_id=hypothesis_id,
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
        return self._rpc(
            "find_experience_hints",
            task_description=task_description,
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
        return self._rpc(
            "get_memory_evidence_summary",
            task_description=task_description,
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
        return self._rpc(
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
        return self._rpc(
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
        return self._rpc(
            "record_object_approach_outcome",
            scene_id=scene_id,
            target=target,
            candidate=candidate,
            outcome=outcome,
            reason=reason,
            metadata=metadata,
        )

    def record_perception(self, report: PerceptionReport) -> dict[str, Any]:
        return self._rpc("record_perception", report=asdict(report))

    def record_navigation_update(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._rpc("record_navigation_update", payload=payload)

    def record_navigation_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._rpc("record_navigation_event", payload=payload)

    def record_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._rpc("record_action", payload=payload)

    def record_monitor_summary(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._rpc("record_monitor_summary", payload=payload)

    def _rpc(self, method: str, **kwargs) -> Any:
        request_payload = {"method": method, "kwargs": kwargs}

        try:
            response = self.session.post(
                self.endpoint, json=request_payload, timeout=self.timeout_s
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise AdapterError(f"Memory Agent request failed: {exc}") from exc

        if not isinstance(data, dict):
            raise AdapterError(f"Memory Agent invalid response: {data}")

        if not data.get("ok", False):
            raise AdapterError(data.get("error") or "Memory Agent unknown error")

        return data.get("result")
