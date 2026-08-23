"""Memory adapter interface used by all agents."""

from __future__ import annotations

from typing import Any, Protocol

from voltron.shared.enums import TaskType
from voltron.shared.models import PerceptionReport


class MemoryAdapter(Protocol):
    """Stable memory API exposed to all agents via the adapter boundary."""

    def start_task(self, task_description: str, task_type: TaskType) -> str:
        """Start a new task and return an episode/task id."""

    def end_task(self, outcome: str, failure_reason: str | None = None) -> dict[str, Any]:
        """Finalize the active task and persist the episode."""

    def reflect(self, similar_top_k: int = 5) -> dict[str, Any]:
        """Run task-level reflection on the latest episode."""

    def find_object(self, name: str, attributes: dict[str, Any] | None = None, top_k: int = 5) -> Any:
        """Query semantic memory for matching objects."""

    def find_objects_near(self, position: tuple[float, float, float], radius: float = 2.0) -> Any:
        """Query semantic memory by spatial neighborhood."""

    def find_similar_episodes(self, description: str, top_k: int = 5) -> Any:
        """Query episodic memory for similar history."""

    def find_applicable_skills(self, current_state: dict[str, Any], top_k: int = 5) -> Any:
        """Query procedural memory for reusable skills."""

    def predict_action_effects(
        self,
        action: str,
        target: str,
        conditions: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
        match_mode: str = "strict",
    ) -> Any:
        """Query the causal graph for forward effect prediction."""

    def diagnose_effect_cause(self, effect: str, value: Any = None) -> Any:
        """Query the causal graph for backward diagnosis."""

    def load_map(self, scene_id: str) -> dict[str, Any]:
        """Load a persisted scene map asset."""

    def save_map(
        self,
        scene_id: str,
        map_payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist a scene map asset."""

    def update_map(
        self,
        scene_id: str,
        delta: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Merge incremental updates into a scene map asset."""

    def query_semantic_region(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> Any:
        """Query region-level semantic memory."""

    def query_topology(self, start: dict[str, Any], goal: dict[str, Any]) -> dict[str, Any]:
        """Query connectivity/path hints between two anchors."""

    def mark_explored(self, scene_id: str, evidence: dict[str, Any]) -> dict[str, Any]:
        """Record exploration evidence into scene-map memory."""

    def get_exploration_frontiers(self, scene_id: str) -> list[dict[str, Any]]:
        """Return currently known frontiers for exploration."""

    def get_working_state(self) -> dict[str, Any]:
        """Return the current working-memory snapshot."""

    def get_active_regions(self) -> list[str]:
        """Return region ids currently held in working memory."""

    def get_recent_observations(self, n: int = 10) -> list[dict[str, Any]]:
        """Return the most recent working-memory observations."""

    def get_task_context(self) -> dict[str, Any]:
        """Return the current task context from working memory."""

    def update_task_context(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Merge structured runtime/task updates into working-memory task context."""

    def record_working_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Write a compact runtime observation into task-local working memory."""

    def get_completed_episode_context(
        self,
        episode_id: str | None = None,
        *,
        recent_observation_limit: int = 20,
    ) -> dict[str, Any]:
        """Return RPC-serializable context for a completed episode."""

    def annotate_completed_episode(self, episode_id: str, annotation: dict[str, Any]) -> dict[str, Any]:
        """Attach MemoryAgent consolidation output to a completed episode."""

    def store_experience_hint(self, hint: dict[str, Any]) -> dict[str, Any]:
        """Persist a lightweight experience hint for later retrieval."""

    def get_experience_hint(self, hint_id: str) -> dict[str, Any] | None:
        """Return one experience hint by exact id."""

    def store_failure_pattern_candidate(self, pattern: dict[str, Any]) -> dict[str, Any]:
        """Persist a failure pattern candidate for later retrieval."""

    def get_failure_pattern_candidate(self, pattern_id: str) -> dict[str, Any] | None:
        """Return one failure pattern candidate by exact id."""

    def find_failure_patterns(
        self,
        query: str,
        task_type: str | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Retrieve failure pattern candidates for planning."""

    def store_semantic_update_candidate(self, update: dict[str, Any]) -> dict[str, Any]:
        """Persist a semantic update candidate without applying it immediately."""

    def get_semantic_update_candidate(self, update_id: str) -> dict[str, Any] | None:
        """Return one semantic update candidate by exact id."""

    def find_semantic_update_candidates(
        self,
        query: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Retrieve semantic update candidates for inspection or validation."""

    def store_skill_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Persist a procedural skill candidate without promoting it to a Skill."""

    def get_skill_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        """Return one procedural skill candidate by exact id."""

    def store_causal_hypothesis(self, hypothesis: dict[str, Any]) -> dict[str, Any]:
        """Persist a causal hypothesis without promoting it to a CausalEdge."""

    def get_causal_hypothesis(self, hypothesis_id: str) -> dict[str, Any] | None:
        """Return one causal hypothesis by exact id."""

    def promote_skill_candidate(
        self,
        candidate_id: str,
        min_confidence: float = 0.7,
        min_supporting_episodes: int = 1,
        max_contradictions: int = 0,
        allow_conflicts: bool = False,
    ) -> dict[str, Any]:
        """Promote a procedural candidate to a formal Skill if policy gates pass."""

    def promote_causal_hypothesis(
        self,
        hypothesis_id: str,
        min_confidence: float = 0.7,
        min_supporting_episodes: int = 1,
        max_contradictions: int = 0,
        allow_conflicts: bool = False,
    ) -> dict[str, Any]:
        """Promote a causal hypothesis to a formal CausalEdge if policy gates pass."""

    def find_experience_hints(
        self,
        task_description: str,
        task_type: str | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Retrieve prior experience hints for a task."""

    def get_memory_evidence_summary(
        self,
        task_description: str,
        task_type: str | None = None,
        scene_id: str | None = None,
        target: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Return bounded task evidence for planning and replanning."""

    def counterfactual_query(
        self,
        episode_id: str | None = None,
        task_description: str | None = None,
        failed_action_idx: int | None = None,
        desired_effect: str | None = None,
        current_state: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Return structured counterfactual alternatives for replanning."""

    def get_object_approach_history(
        self,
        scene_id: str,
        target: dict[str, Any],
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Return historical approach outcomes for one object-target navigation goal."""

    def record_object_approach_outcome(
        self,
        scene_id: str,
        target: dict[str, Any],
        candidate: dict[str, Any],
        outcome: str,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist one object-approach anchor outcome for later retrieval."""

    def record_perception(self, report: PerceptionReport) -> dict[str, Any]:
        """Write a VLM perception report to semantic/working memory."""

    def record_navigation_update(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Write VLN navigation evidence to semantic/working memory."""

    def record_navigation_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Record a navigation lineage event as an episode action."""

    def record_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Write VLA action execution and causal update."""

    def record_monitor_summary(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Write a coarse Vision monitor summary for the active episode."""
