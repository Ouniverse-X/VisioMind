from __future__ import annotations

import json
from typing import Any

from visiomind.action.agents.memory.contracts.experience import (
    ExperienceExtractionResult,
    normalize_extraction_result,
)
from visiomind.action.agents.navigation.body.json_response import extract_json_object

_MAX_COMPACT_STRING_CHARS = 800
_MAX_COMPACT_LIST_ITEMS = 5
_MAX_COMPACT_DICT_ITEMS = 32

_SYSTEM_PROMPT = """You are the VisioMindAction Memory Agent experience extractor.
Return valid JSON only. Do not include markdown fences or extra text.

Extract reusable task experience from a completed embodied-robot episode.
Only write candidates supported by the episode evidence. Use conservative confidence scores.

Return this top-level JSON object:
{
  "schema_version": "memory_experience_v2",
  "episode_summary": "short factual summary",
  "task_outcome": "success | failure | partial | unknown",
  "source_episode_id": "episode id",
  "confidence": 0.0,
  "rule_derived_evidence": {
    "critical_failure_action": "copy or summarize the provided reflection_evidence.rule_derived_evidence when relevant",
    "failure_classification": {"kind": "environment_truncation | action_failure | vla_backend_failure | simulator_task_failure | unknown"},
    "object_approach_selection": {"candidate": "selected object approach and history penalty when present"},
    "visual_state": {"target_visible": "bool when observed", "target_part_visible": "bool when observed"},
    "action_backend_status": {"backend": "vla backend name when known", "status": "backend_failure | ok | unknown"},
    "action_interaction_context": {
      "selected_candidate": "candidate id and compact candidate_signature",
      "distance_context": "approach, handoff, and visual distance when observed",
      "visual_affordance": "target/switch visibility and view quality",
      "contact_context": "contact/reachability evidence when observed",
      "environment_outcome": "simulator predicate, progress, truncation, and goal status",
      "vlm_predicate_mismatch": "VLM reported success while simulator predicate stayed false"
    }
  },
  "procedural_skills": [
    {
      "name": "candidate skill name",
      "description": "when and how to reuse it",
      "trigger": {"task_type": "interaction", "object": "fridge"},
      "steps": [{"agent": "NAVIGATION | VISION | ACTION", "action": "short action"}],
      "confidence": 0.0,
      "source_action_ids": ["action ids supporting the candidate"]
    }
  ],
  "causal_hypotheses": [
    {
      "action": "action name",
      "target": "target object or region",
      "expected_effect": "observed effect",
      "conditions": {"condition": "value"},
      "confidence": 0.0,
      "source_action_ids": ["action ids supporting the hypothesis"]
    }
  ],
  "retrieval_hints": [
    {
      "hint_type": "procedural | failure_avoidance | causal | semantic | completion_criteria",
      "summary": "short retrieval-ready lesson",
      "confidence": 0.0,
      "source_action_ids": ["action ids supporting the hint"],
      "content": {"optional": "structured details"}
    }
  ],
  "failure_patterns": [
    {
      "pattern_type": "navigation_blockage | action_failure | perception_gap | other",
      "summary": "retrieval-ready failure pattern",
      "conditions": {"scene_id": "optional", "object": "optional"},
      "recommended_response": "how future planning should avoid or recover from it",
      "confidence": 0.0,
      "source_action_ids": ["action ids supporting the pattern"]
    }
  ],
  "semantic_updates": [
    {
      "update_type": "object_location | relation | affordance | attribute | other",
      "subject": "entity being updated",
      "relation": "relation or attribute name",
      "object": "new value or related entity",
      "content": {"optional": "structured details"},
      "confidence": 0.0,
      "source_action_ids": ["action ids supporting the update"]
    }
  ],
  "object_approach_priors": []
}

Separate candidate types strictly:
- retrieval_hints are lightweight planning/retrieval lessons.
- failure_patterns describe repeatable failure conditions and recovery advice.
- semantic_updates are world-state candidates only; do not invent facts or override conflicting scene memory.
- procedural_skills are reusable successful action sequences. Do not create procedural_skills from failed-only evidence.
- causal_hypotheses require state_deltas or causal_observations that show the effect; source_action_ids alone are not enough.
- object_approach_priors should come from selected object approach evidence and include history penalties when present.
- completion_criteria hints preserve user-confirmed and episode-verified success conditions for similar tasks.
- Only write completion_criteria when the episode succeeded, the text plan was confirmed, and completion evidence shows the condition was achieved.
- Completion criteria memories should preserve the target object, target part, user-confirmed condition, and whether monitor evidence verified it.
- Do not promote environment task_success alone as a visual completion condition; treat it as environment evidence unless monitor evidence supports the same condition.
- If action_interaction_context shows VLM success but environment predicates remain false, write failure_avoidance hints and action_failure patterns tied to the candidate_signature, visible target part, contact, and distance context.
"""


class DefaultMemoryExperienceExtractionSkill:
    @property
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    def build_prompt(self, episode_context: dict[str, Any]) -> str:
        payload = _compact_episode_context(episode_context)
        return (
            "Extract reusable MemoryAgent experience from this completed episode.\n"
            f"Episode context JSON: {json.dumps(payload, ensure_ascii=False, default=str)}\n"
            "Return JSON only."
        )

    def parse_extraction_response(self, content: str) -> ExperienceExtractionResult:
        payload = extract_json_object(content, label="Memory experience extractor")
        _normalize_list_fields(payload)
        result = normalize_extraction_result(payload)
        result.procedural_skills = _dict_items(result.procedural_skills)
        result.causal_hypotheses = _dict_items(result.causal_hypotheses)
        result.failure_patterns = _dict_items(result.failure_patterns)
        result.semantic_updates = _dict_items(result.semantic_updates)
        result.object_approach_priors = _dict_items(result.object_approach_priors)
        return result


def _compact_episode_context(episode_context: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = (
        "episode_id",
        "task_id",
        "task_description",
        "task_type",
        "outcome",
        "failure_reason",
        "initial_state",
        "final_state",
        "actions",
        "action_sequence",
        "state_transitions",
        "causal_annotations",
        "recent_observations",
        "lessons_learned",
        "improvement_suggestions",
        "reflection_evidence",
        "rule_derived_evidence",
        "scene_memory_context",
        "source_integrity",
        "interactive_planning",
        "completion_monitor",
        "completion_criteria",
    )
    return {
        key: _compact_top_level_value(key, episode_context[key])
        for key in allowed_keys
        if key in episode_context and episode_context[key] is not None
    }


def _compact_top_level_value(key: str, value: Any) -> Any:
    if key in {"actions", "action_sequence"} and isinstance(value, (list, tuple)):
        return _compact_action_list(value)
    if key in {"initial_state", "final_state"} and isinstance(value, dict):
        return _compact_episode_state(value)
    if key == "scene_memory_context" and isinstance(value, dict):
        return _compact_scene_memory_context(value)
    return _compact_value(value)


def _compact_action_list(actions: list[Any] | tuple[Any, ...]) -> list[Any]:
    compacted = [_compact_action(action) for action in list(actions)[:_MAX_COMPACT_LIST_ITEMS]]
    omitted = len(actions) - len(compacted)
    if omitted > 0:
        compacted.append({"omitted_items": omitted})
    return compacted


def _compact_action(action: Any) -> Any:
    if not isinstance(action, dict):
        return _compact_value(action)
    keep_keys = (
        "action_id",
        "subtask_id",
        "agent",
        "action_type",
        "type",
        "target",
        "status",
        "success",
        "error_code",
        "failure_reason",
        "latency_ms",
        "selected_object_approach",
        "execution_goal",
        "result",
        "state_changes",
    )
    compacted = {
        key: _compact_value(action[key])
        for key in keep_keys
        if key in action and action[key] is not None
    }
    return compacted or _compact_value(action)


def _compact_episode_state(state: dict[str, Any]) -> dict[str, Any]:
    keep_keys = (
        "episode_id",
        "outcome",
        "failure_reason",
        "success",
        "task_success",
        "task_progress",
        "subtask_completed",
        "subtask_succeeded",
        "subtask_completion_reason",
        "subtask_name",
        "goal_status",
        "reward_breakdown",
        "source_integrity",
        "environment_vlm_heartbeat",
        "interactive_planning",
        "completion_monitor",
        "completion_criteria",
    )
    compacted = {
        key: _compact_value(state[key])
        for key in keep_keys
        if key in state and state[key] is not None
    }
    environment = state.get("environment")
    if isinstance(environment, dict):
        compacted["environment"] = _compact_value(
            {
                key: environment[key]
                for key in (
                    "env_id",
                    "step_count",
                    "task_success",
                    "terminated",
                    "truncated",
                    "success",
                    "goal_status",
                )
                if key in environment and environment[key] is not None
            }
        )
    working_summary = state.get("working_memory_summary")
    if isinstance(working_summary, dict):
        compacted["working_memory_summary"] = _compact_value(
            {
                key: working_summary[key]
                for key in (
                    "task_phase",
                    "current_subtask",
                    "recent_observations",
                    "history_insights",
                )
                if key in working_summary and working_summary[key] is not None
            }
        )
    annotations = state.get("memory_annotations")
    if isinstance(annotations, list):
        compacted["memory_annotations"] = {
            "count": len(annotations),
            "latest_types": [
                item.get("annotation_type")
                for item in annotations[-3:]
                if isinstance(item, dict) and item.get("annotation_type")
            ],
        }
    return compacted


def _compact_scene_memory_context(context: dict[str, Any]) -> dict[str, Any]:
    compacted: dict[str, Any] = {}
    for key in ("summary", "scene_count", "target", "task_description"):
        if key in context and context[key] is not None:
            compacted[key] = _compact_value(context[key])
    scenes = context.get("scenes")
    if isinstance(scenes, list):
        compacted["scenes"] = [
            _compact_scene(scene) for scene in scenes[:2] if isinstance(scene, dict)
        ]
        omitted = len(scenes) - len(compacted["scenes"])
        if omitted > 0:
            compacted["scenes"].append({"omitted_items": omitted})
    elif scenes is not None:
        compacted["scenes"] = _compact_value(scenes)
    if not compacted:
        compacted = _compact_dict(
            {
                key: value
                for key, value in context.items()
                if key not in {"raw_scene_dump", "graph", "objects", "edges"}
            },
            depth=1,
        )
    return compacted


def _compact_scene(scene: dict[str, Any]) -> dict[str, Any]:
    metadata = scene.get("metadata") if isinstance(scene.get("metadata"), dict) else {}
    navigation = scene.get("navigation") if isinstance(scene.get("navigation"), dict) else {}
    approach_memory = (
        scene.get("object_approach_memory")
        if isinstance(scene.get("object_approach_memory"), dict)
        else {}
    )
    compacted: dict[str, Any] = {"scene_id": scene.get("scene_id")}
    semantic_ingestion = metadata.get("semantic_ingestion") if isinstance(metadata, dict) else None
    if isinstance(semantic_ingestion, dict):
        compacted["semantic_ingestion"] = _compact_value(
            {
                key: semantic_ingestion[key]
                for key in ("regions", "objects", "edges", "confidence")
                if key in semantic_ingestion
            }
        )
    compacted["navigation"] = _compact_navigation_summary(navigation)
    compacted["object_approach_memory"] = _compact_object_approach_memory(approach_memory)
    return {key: value for key, value in compacted.items() if value is not None}


def _compact_navigation_summary(navigation: dict[str, Any]) -> dict[str, Any]:
    summary = {
        key: _compact_value(navigation[key])
        for key in ("last_region", "visited_regions", "last_pose")
        if key in navigation and navigation[key] is not None
    }
    grounded_goal = navigation.get("last_grounded_goal")
    if isinstance(grounded_goal, dict):
        summary["last_grounded_goal"] = _compact_value(
            {
                key: grounded_goal[key]
                for key in (
                    "goal_type",
                    "object_id",
                    "object_name",
                    "room_id",
                    "room_name",
                    "floor_id",
                    "position",
                    "grounding_query",
                )
                if key in grounded_goal and grounded_goal[key] is not None
            }
        )
    selected = navigation.get("selected_object_approach")
    if isinstance(selected, dict):
        summary["selected_object_approach"] = _compact_approach_candidate(selected)
    return summary


def _compact_object_approach_memory(memory: dict[str, Any]) -> dict[str, Any]:
    targets = memory.get("targets")
    summary: dict[str, Any] = {}
    if "target_count" in memory:
        summary["target_count"] = memory["target_count"]
    if isinstance(targets, list):
        compacted_targets = []
        for target in targets[:2]:
            if not isinstance(target, dict):
                continue
            entries = target.get("entries") if isinstance(target.get("entries"), list) else []
            compacted_targets.append(
                {
                    "target_key": target.get("target_key"),
                    "target": _compact_value(target.get("target")),
                    "entry_count": target.get("entry_count", len(entries)),
                    "recent_entries": [
                        _compact_object_approach_entry(entry)
                        for entry in entries[-3:]
                        if isinstance(entry, dict)
                    ],
                }
            )
        summary["targets"] = compacted_targets
        if len(targets) > len(compacted_targets):
            summary["omitted_targets"] = len(targets) - len(compacted_targets)
    return summary


def _compact_object_approach_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "outcome": entry.get("outcome"),
            "reason": entry.get("reason"),
            "candidate_signature": _compact_value(entry.get("candidate_signature")),
            "candidate": _compact_approach_candidate(entry.get("candidate"))
            if isinstance(entry.get("candidate"), dict)
            else None,
        }.items()
        if value is not None
    }


def _compact_approach_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    keep_keys = (
        "candidate_id",
        "nav_node",
        "floor_id",
        "room_id",
        "room_name",
        "object_id",
        "object_name",
        "approach_distance_m",
        "approach_boundary_distance_m",
        "desired_heading",
        "path_cost",
        "selection_source",
        "handoff_distance_m",
        "history_penalty",
        "blocked_by_history",
    )
    return {
        key: _compact_value(candidate[key])
        for key in keep_keys
        if key in candidate and candidate[key] is not None
    }


def _compact_value(value: Any, *, depth: int = 0) -> Any:
    if isinstance(value, str):
        return _compact_string(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return _compact_dict(value, depth=depth + 1)
    if isinstance(value, (list, tuple)):
        return _compact_list(value, depth=depth + 1)
    return _compact_string(str(value))


def _compact_string(value: str) -> str | dict[str, Any]:
    if len(value) <= _MAX_COMPACT_STRING_CHARS:
        return value
    return {
        "text": value[:_MAX_COMPACT_STRING_CHARS],
        "truncated": True,
        "original_chars": len(value),
    }


def _compact_list(values: list[Any] | tuple[Any, ...], *, depth: int) -> list[Any]:
    compacted = [
        _compact_value(item, depth=depth) for item in list(values)[:_MAX_COMPACT_LIST_ITEMS]
    ]
    omitted = len(values) - len(compacted)
    if omitted > 0:
        compacted.append({"omitted_items": omitted})
    return compacted


def _compact_dict(value: dict[str, Any], *, depth: int) -> dict[str, Any]:
    compacted: dict[str, Any] = {}
    items = list(value.items())
    for key, item in items[:_MAX_COMPACT_DICT_ITEMS]:
        if item is not None:
            compacted[str(key)] = _compact_value(item, depth=depth)
    omitted = len(items) - len(compacted)
    if omitted > 0:
        compacted["omitted_keys"] = omitted
    return compacted


def _normalize_list_fields(payload: dict[str, Any]) -> None:
    for key in (
        "procedural_skills",
        "failure_patterns",
        "causal_hypotheses",
        "semantic_updates",
        "retrieval_hints",
        "object_approach_priors",
        "validation_warnings",
    ):
        value = payload.get(key)
        if value is None:
            payload[key] = []
        elif not isinstance(value, list):
            payload[key] = [value]


def _dict_items(items: list[Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in items if isinstance(item, dict)]
