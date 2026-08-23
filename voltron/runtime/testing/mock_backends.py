"""Mock runtime backends shared by tests and local example entrypoints."""

from __future__ import annotations

from typing import Any

import numpy as np

from voltron.shared.enums import TaskType
from voltron.shared.models import PerceptionObject, PerceptionReport
from voltron.shared.object_approach_signature import candidate_signature


class MockMemoryAdapter:
    def __init__(self):
        self.started = False
        self.maps: dict[str, dict[str, Any]] = {}
        self.active_regions: list[str] = []
        self.task_context: dict[str, Any] = {}
        self.recent_observations: list[dict[str, Any]] = []
        self.task_context_updates: list[dict[str, Any]] = []

    def start_task(self, task_description: str, task_type: TaskType) -> str:
        self.started = True
        self.task_context = {"task_description": task_description, "task_type": task_type.value}
        self.task_context_updates = [dict(self.task_context)]
        return "ep_mock_001"

    def end_task(self, outcome: str, failure_reason: str | None = None) -> dict[str, Any]:
        return {"episode_id": "ep_mock_001", "outcome": outcome, "failure_reason": failure_reason}

    def reflect(self, similar_top_k: int = 5) -> dict[str, Any]:
        return {"lessons": ["mock lesson"], "suggestions": []}

    def find_object(self, name: str, attributes: dict[str, Any] | None = None, top_k: int = 5) -> Any:
        return {"query": name, "results": []}

    def find_objects_near(self, position: tuple[float, float, float], radius: float = 2.0) -> Any:
        return {"query": {"position": position, "radius": radius}, "results": []}

    def find_similar_episodes(self, description: str, top_k: int = 5) -> Any:
        return {"query": description, "results": []}

    def find_applicable_skills(self, current_state: dict[str, Any], top_k: int = 5) -> Any:
        return {"results": [{"skill_name": "mock_skill"}]}

    def predict_action_effects(self, action: str, target: str, conditions: dict[str, Any] | None = None) -> Any:
        return {"results": [{"effect": "object_held", "prob": 0.9}]}

    def diagnose_effect_cause(self, effect: str, value: Any = None) -> Any:
        return {"results": []}

    def load_map(self, scene_id: str) -> dict[str, Any]:
        entry = self.maps.get(scene_id)
        if entry is None:
            return {"scene_id": scene_id, "status": "missing", "map_payload": None, "metadata": {}}
        return {
            "scene_id": scene_id,
            "status": "loaded",
            "map_payload": dict(entry["map_payload"]),
            "metadata": dict(entry["metadata"]),
        }

    def save_map(
        self,
        scene_id: str,
        map_payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        entry = {"map_payload": dict(map_payload), "metadata": dict(metadata or {})}
        self.maps[scene_id] = entry
        return {"scene_id": scene_id, "status": "saved", **entry}

    def update_map(
        self,
        scene_id: str,
        delta: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        current = self.maps.setdefault(scene_id, {"map_payload": {}, "metadata": {}})
        current["map_payload"].update(delta)
        current["metadata"].update(dict(metadata or {}))
        return {"scene_id": scene_id, "status": "updated", **current}

    def query_semantic_region(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> Any:
        return {
            "query_type": "region",
            "query": {"name": name, "attributes": attributes},
            "results": [],
            "scores": [],
            "metadata": {"top_k": top_k},
        }

    def query_topology(self, start: dict[str, Any], goal: dict[str, Any]) -> dict[str, Any]:
        return {"scene_id": start.get("scene_id") or goal.get("scene_id"), "start": start, "goal": goal, "path": []}

    def mark_explored(self, scene_id: str, evidence: dict[str, Any]) -> dict[str, Any]:
        current = self.maps.setdefault(scene_id, {"map_payload": {}, "metadata": {}})
        explored = current["map_payload"].setdefault("explored", [])
        explored.append(dict(evidence))
        return {"scene_id": scene_id, "status": "marked", "evidence_count": len(explored)}

    def get_exploration_frontiers(self, scene_id: str) -> list[dict[str, Any]]:
        current = self.maps.get(scene_id, {"map_payload": {}})
        return list(current["map_payload"].get("frontiers", []))

    def get_working_state(self) -> dict[str, Any]:
        return {
            "active_regions": list(self.active_regions),
            "task_context": dict(self.task_context),
            "recent_observations": len(self.recent_observations),
        }

    def get_active_regions(self) -> list[str]:
        return list(self.active_regions)

    def get_recent_observations(self, n: int = 10) -> list[dict[str, Any]]:
        return list(self.recent_observations[-n:])

    def get_task_context(self) -> dict[str, Any]:
        return dict(self.task_context)

    def update_task_context(self, updates: dict[str, Any]) -> dict[str, Any]:
        self.task_context = self._merge_dicts(self.task_context, updates)
        self.task_context_updates.append(dict(self.task_context))
        return dict(self.task_context)

    def get_object_approach_history(
        self,
        scene_id: str,
        target: dict[str, Any],
        top_k: int = 10,
    ) -> dict[str, Any]:
        scene_entry = self.maps.get(scene_id, {"map_payload": {}})
        target_key = str(
            target.get("object_id")
            or target.get("object")
            or target.get("object_name")
            or target.get("room_id")
            or ""
        ).strip()
        history_store = scene_entry["map_payload"].get("object_approach_memory", {})
        entries = list(history_store.get(target_key, {}).get("entries", []))
        if not entries:
            runtime_history = self.task_context.get("object_approach_history", {})
            if (
                isinstance(runtime_history, dict)
                and runtime_history.get("scene_id") == scene_id
                and runtime_history.get("target_key") == target_key
            ):
                entries = list(runtime_history.get("entries", []))
        if top_k > 0:
            entries = entries[-int(top_k) :]
        return {
            "scene_id": scene_id,
            "target_key": target_key or None,
            "entries": entries,
        }

    def record_object_approach_outcome(
        self,
        scene_id: str,
        target: dict[str, Any],
        candidate: dict[str, Any],
        outcome: str,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        current = self.maps.setdefault(scene_id, {"map_payload": {}, "metadata": {}})
        history_store = current["map_payload"].setdefault("object_approach_memory", {})
        target_key = str(
            target.get("object_id")
            or target.get("object")
            or target.get("object_name")
            or target.get("room_id")
            or ""
        ).strip()
        bucket = history_store.setdefault(target_key, {"target": dict(target), "entries": []})
        bucket["entries"].append(
            {
                "outcome": outcome,
                "reason": reason,
                "candidate": dict(candidate),
                "candidate_signature": candidate_signature(candidate),
                "metadata": dict(metadata or {}),
            }
        )
        return {"status": "recorded", "scene_id": scene_id, "target_key": target_key}

    def record_perception(self, report: PerceptionReport) -> dict[str, Any]:
        self.recent_observations.append({"source": "vlm", "objects": len(report.objects)})
        return {"objects": len(report.objects)}

    def record_navigation_update(self, payload: dict[str, Any]) -> dict[str, Any]:
        region = payload.get("region")
        if isinstance(region, str) and region and region not in self.active_regions:
            self.active_regions.append(region)
        self.recent_observations.append({"source": "vln", "payload": dict(payload)})
        return {"updated": True, "region": payload.get("region")}

    def record_working_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        self.recent_observations.append(dict(observation))
        return {"recorded": True, "source": observation.get("source")}

    def record_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.recent_observations.append({"source": "vla", "payload": dict(payload)})
        return {"recorded": True, "action_type": payload.get("action_type")}

    def record_monitor_summary(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.recent_observations.append({"source": "vision_monitor", "payload": dict(payload)})
        return {"recorded": True, "summary_index": len(self.recent_observations) - 1}

    def get_modality_config(self) -> dict[str, Any]:
        return {}

    def _merge_dicts(self, base: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in delta.items():
            current = merged.get(key)
            if isinstance(current, dict) and isinstance(value, dict):
                merged[key] = self._merge_dicts(current, value)
            else:
                merged[key] = value
        return merged


class MockVisionAdapter:
    def analyze(
        self,
        images_b64: list[str],
        instruction: str,
        task_name: str,
        image_view_order: list[str] | None = None,
    ) -> PerceptionReport:
        return PerceptionReport(
            objects=[PerceptionObject(name="red_cup", confidence=0.95)],
            task_complete=False,
            raw_text="mock detection",
        )


class MockPolicyAdapter:
    def ping(self) -> bool:
        return True

    def get_action(self, observation: dict[str, Any], options: dict[str, Any] | None = None):
        action = {
            "base": np.zeros((1, 1, 3), dtype=np.float32),
            "torso": np.zeros((1, 1, 4), dtype=np.float32),
            "left_arm": np.zeros((1, 1, 7), dtype=np.float32),
            "left_gripper": np.zeros((1, 1, 2), dtype=np.float32),
            "right_arm": np.zeros((1, 1, 7), dtype=np.float32),
            "right_gripper": np.zeros((1, 1, 2), dtype=np.float32),
        }
        return action, {"backend": "mock"}

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return {}


__all__ = [
    "MockMemoryAdapter",
    "MockPolicyAdapter",
    "MockVisionAdapter",
]
