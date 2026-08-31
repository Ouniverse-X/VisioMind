from __future__ import annotations

from typing import Any, Protocol

from visiomind.action.shared.enums import TaskType
from visiomind.action.shared.models import PerceptionReport


class MemoryAdapter(Protocol):
    def start_task(self, task_description: str, task_type: TaskType) -> str:
        pass

    def end_task(self, outcome: str, failure_reason: str | None = None) -> dict[str, Any]:
        pass

    def reflect(self, similar_top_k: int = 5) -> dict[str, Any]:
        pass

    def find_object(
        self, name: str, attributes: dict[str, Any] | None = None, top_k: int = 5
    ) -> Any:
        pass

    def find_objects_near(self, position: tuple[float, float, float], radius: float = 2.0) -> Any:
        pass

    def find_similar_episodes(self, description: str, top_k: int = 5) -> Any:
        pass

    def find_applicable_skills(self, current_state: dict[str, Any], top_k: int = 5) -> Any:
        pass

    def predict_action_effects(
        self,
        action: str,
        target: str,
        conditions: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
        match_mode: str = "strict",
    ) -> Any:
        pass

    def diagnose_effect_cause(self, effect: str, value: Any = None) -> Any:
        pass

    def load_map(self, scene_id: str) -> dict[str, Any]:
        pass

    def save_map(
        self,
        scene_id: str,
        map_payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        pass

    def update_map(
        self,
        scene_id: str,
        delta: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        pass

    def query_semantic_region(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> Any:
        pass

    def query_topology(self, start: dict[str, Any], goal: dict[str, Any]) -> dict[str, Any]:
        pass

    def mark_explored(self, scene_id: str, evidence: dict[str, Any]) -> dict[str, Any]:
        pass

    def get_exploration_frontiers(self, scene_id: str) -> list[dict[str, Any]]:
        pass

    def get_working_state(self) -> dict[str, Any]:
        pass

    def get_active_regions(self) -> list[str]:
        pass

    def get_recent_observations(self, n: int = 10) -> list[dict[str, Any]]:
        pass

    def get_task_context(self) -> dict[str, Any]:
        pass

    def update_task_context(self, updates: dict[str, Any]) -> dict[str, Any]:
        pass

    def record_working_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        pass

    def get_completed_episode_context(
        self,
        episode_id: str | None = None,
        *,
        recent_observation_limit: int = 20,
    ) -> dict[str, Any]:
        pass

    def annotate_completed_episode(
        self, episode_id: str, annotation: dict[str, Any]
    ) -> dict[str, Any]:
        pass

    def store_experience_hint(self, hint: dict[str, Any]) -> dict[str, Any]:
        pass

    def get_experience_hint(self, hint_id: str) -> dict[str, Any] | None:
        pass

    def store_failure_pattern_candidate(self, pattern: dict[str, Any]) -> dict[str, Any]:
        pass

    def get_failure_pattern_candidate(self, pattern_id: str) -> dict[str, Any] | None:
        pass

    def find_failure_patterns(
        self,
        query: str,
        task_type: str | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        pass

    def store_semantic_update_candidate(self, update: dict[str, Any]) -> dict[str, Any]:
        pass

    def get_semantic_update_candidate(self, update_id: str) -> dict[str, Any] | None:
        pass

    def find_semantic_update_candidates(
        self,
        query: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        pass

    def store_skill_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        pass

    def get_skill_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        pass

    def store_causal_hypothesis(self, hypothesis: dict[str, Any]) -> dict[str, Any]:
        pass

    def get_causal_hypothesis(self, hypothesis_id: str) -> dict[str, Any] | None:
        pass

    def promote_skill_candidate(
        self,
        candidate_id: str,
        min_confidence: float = 0.7,
        min_supporting_episodes: int = 1,
        max_contradictions: int = 0,
        allow_conflicts: bool = False,
    ) -> dict[str, Any]:
        pass

    def promote_causal_hypothesis(
        self,
        hypothesis_id: str,
        min_confidence: float = 0.7,
        min_supporting_episodes: int = 1,
        max_contradictions: int = 0,
        allow_conflicts: bool = False,
    ) -> dict[str, Any]:
        pass

    def find_experience_hints(
        self,
        task_description: str,
        task_type: str | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        pass

    def get_memory_evidence_summary(
        self,
        task_description: str,
        task_type: str | None = None,
        scene_id: str | None = None,
        target: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        pass

    def counterfactual_query(
        self,
        episode_id: str | None = None,
        task_description: str | None = None,
        failed_action_idx: int | None = None,
        desired_effect: str | None = None,
        current_state: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        pass

    def get_object_approach_history(
        self,
        scene_id: str,
        target: dict[str, Any],
        top_k: int = 10,
    ) -> dict[str, Any]:
        pass

    def record_object_approach_outcome(
        self,
        scene_id: str,
        target: dict[str, Any],
        candidate: dict[str, Any],
        outcome: str,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        pass

    def record_perception(self, report: PerceptionReport) -> dict[str, Any]:
        pass

    def record_navigation_update(self, payload: dict[str, Any]) -> dict[str, Any]:
        pass

    def record_navigation_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        pass

    def record_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        pass

    def record_monitor_summary(self, payload: dict[str, Any]) -> dict[str, Any]:
        pass
