from __future__ import annotations

import json
from json import JSONDecoder
from typing import Any

from voltron.shared.context import Plan, Subtask
from voltron.shared.enums import AgentName
from voltron.shared.telemetry.payload_sanitizer import strip_image_payloads

_ALLOWED_AGENTS = ", ".join(
    agent.value for agent in (AgentName.NAVIGATION, AgentName.VISION, AgentName.ACTION)
)
NAVIGATION_INSTRUCTION_GUIDANCE = (
    "For every NAVIGATION subtask, write `parameters.instruction` as a task-refined natural-language goal. "
    "When the navigation destination is chosen to enable later work, include the follow-up object, part, or action "
    "that determines where the robot should stop. Express the needed spatial relation or stop intent semantically, "
    "such as a usable boundary, nearby approach point, view point, reachable side, or interaction-ready pose; avoid "
    "generic location-only wording when downstream interaction depends on a specific object or part. Keep this as "
    "natural language and do not fabricate graph object IDs, room IDs, or waypoint IDs."
)
ACTION_INSTRUCTION_GUIDANCE = (
    "For every ACTION subtask, write `parameters.instruction` as a task-refined natural-language command for the "
    "local interaction. The instruction must name the target object and, when applicable, the actionable part or "
    "control such as a button, switch, handle, knob, or door. Do not rely on runtime fallback wording."
)
_SYSTEM_PROMPT = """You are the Voltron task planner for an embodied robot.
Return valid JSON only. Do not include markdown fences or extra text.

You must produce a top-level object with this schema:
{
  "subtasks": [
    {
      "subtask_id": "st_01",
      "agent": "NAVIGATION" | "VISION" | "ACTION",
      "action": "short_action_name",
      "target": {
        "object": "target object name when applicable",
        "part": "target part for local interaction when applicable",
        "region": "target region when applicable",
        "room": "target room when applicable",
        "room_name": "numbered internal room instance name when available",
        "room_id": "internal room id when available"
      },
      "parameters": {"optional": "planner/runtime parameters"},
      "context": {"optional": "extra context"}
    }
  ],
  "metadata": {"planner": "openai_compatible"}
}

Use canonical target keys only: `object`, `part`, `region`, `room`, `room_name`, `room_id`.
Do not use `target.name`.

Agent meanings:
- NAVIGATION: global navigation / locomotion / long-range approach / re-localization.
- VISION: visual observation / localization / verification from images.
- ACTION: local interaction / grasp / press / toggle / place. ACTION may use local whole-body base adjustment together with the arms when the target is visible in the same room and only local approach/alignment is needed.

Use `agent_capabilities` from planning context as the preferred source for tool- or skill-specific routing. When a user request matches a declared capability, emit a subtask for that capability's `agent` using one of its `action_names` and compatible `parameters`.

Planning rules:
- Keep plans concise and executable.
- Use ordered subtasks that match the user task.
- Read `planner_mode` from context:
  - `auto`: freely compose NAVIGATION/VISION/ACTION from runtime state and task intent.
  - `scripted`: stay close to explicit templates or task-type hints when provided.
  - `benchmark`: prefer conservative, reproducible, low-branching plans.
- A final VISION verification subtask such as `verify`, `check`, or `confirm` may appear after an execution step. Only set `"parameters": {"allow_task_complete": true}` on this final verification subtask when its purpose is to verify the overall task completion condition.
- For tasks that involve local interaction, prefer dynamic execution:
  - Treat runtime navigation state such as `navigation_state.current_room` / `current_region` as authoritative for coarse room-level localization when available.
  - Treat `scene_report` as visual evidence only. `target_visible` and `target_part_visible` come from VISION.
  - Treat `navigation_report` as authoritative for room status, distance, approach readiness, and reachability.
  - If navigation state says the robot is not yet in the target room/region, emit `NAVIGATION` first.
  - Distinguish room-level navigation from object-level approach:
    - Room-level relocation: use `NAVIGATION navigate` toward the target room/region.
    - Object-level approach inside the correct room: use `NAVIGATION approach_target` toward the target object.
    - Local manipulation once `navigation_report.approach_ready=true`: use `ACTION`.
    - Ambiguous local perception: use `VISION` inspect/find to resolve it.
  - If `target_visible=true` and `navigation_report.approach_ready=true`, do not emit NAVIGATION navigate/approach.
  - If `target_visible=true` and `navigation_report.approach_ready=false`, prefer `NAVIGATION approach_target` when `navigation_report.approach_reachable=true`.
  - For any object-targeted `VISION`, `ACTION`, or object-level `NAVIGATION approach_target` subtask, the natural-language instruction must name the target object and/or part. Do not use room names, room instance names, or room ids as the acted-on item when `target.object` or `target.part` is available.
  - Use VISION only for ambiguous inspection or verification, not as a substitute for local execution.
  - After NAVIGATION, usually inspect again with VISION.
  - After ACTION, usually verify with VISION.
- When emitting a local interaction ACTION step, set `"parameters": {"control_mode": "whole_body_local"}` unless already specified.
- Keep the executable plan complete, but distinguish user goals from runtime-enabling work. When an ACTION is needed only under a condition rather than explicitly requested by the user, annotate it with `parameters.outline_role="contingency"`, `parameters.required=false`, and a concrete `parameters.condition`. Use `parameters.outline_role="support"` for implementation-only ACTION details. Do not mark an explicitly requested world-state change as optional.
- Prefer explicit target names and short machine-readable actions.
- __ACTION_INSTRUCTION_GUIDANCE__
- __NAVIGATION_INSTRUCTION_GUIDANCE__
- Use subtask ids in order: st_01, st_02, ...
- For initial plans, always output at least one subtask.
""".replace("__ACTION_INSTRUCTION_GUIDANCE__", ACTION_INSTRUCTION_GUIDANCE).replace(
    "__NAVIGATION_INSTRUCTION_GUIDANCE__", NAVIGATION_INSTRUCTION_GUIDANCE
)


class DefaultBrainPlanningSkill:
    def system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    def build_prompt(self, task_description: str, context: dict[str, Any]) -> str:
        interactive_request = context.get("interactive_planning_request")
        interactive_request = interactive_request if isinstance(interactive_request, dict) else {}
        require_complete_plan = bool(interactive_request.get("require_complete_plan"))
        interactive_guidance = ""
        vision_guidance = (
            "If you choose a VISION observation subtask such as observe/find/inspect, return exactly one subtask in "
            "this response and wait for runtime observation results before planning any NAVIGATION/ACTION follow-up. A VISION "
            "verification subtask may still be used later as the final step after execution.\n"
        )
        if require_complete_plan:
            interactive_guidance = (
                "This is an interactive planning request: return a complete ordered plan covering the entire user "
                "instruction. Do not return only a seed subtask or defer required later stages. This requirement "
                "overrides the short seed plan preference for this request.\n"
            )
            vision_guidance = (
                "If a VISION observation subtask such as observe/find/inspect is needed, include it in the complete "
                "ordered preview plan and include later NAVIGATION/ACTION stages in the same preview plan when required "
                "to complete the user instruction. A VISION verification subtask may still be used later as the final "
                "step after execution.\n"
            )
            if interactive_request.get("phase") == "refinement":
                interactive_guidance += (
                    "For refinement, treat the confirmed text plan as a human-facing outline rather than a "
                    "one-to-one executable subtask list. Preserve the semantic coverage and relative order of "
                    "required milestone ACTION steps, and attach parameters.collaborative_step_id to their final "
                    "state-establishing descendants when practical. Supporting implementation steps may be expanded "
                    "or inserted. Conditional contingency steps may be omitted when unnecessary or moved to the "
                    "point where runtime reachability requires them; for example, open a door earlier when it blocks "
                    "access to a later milestone. Do not add unrelated world-changing actions.\n"
                )
        return (
            "Create an executable Voltron plan.\n"
            f"Task description: {task_description}\n"
            f"Available execution agents: {_ALLOWED_AGENTS}\n"
            f"Planning context JSON: {self.serialize_context(context)}\n"
            "Interpret natural-language intent yourself from the task and runtime context. Do not rely on "
            "pre-grounded metadata target IDs, object IDs, or room IDs as the answer. Use metadata only as "
            "non-authoritative context.\n"
            "Decompose arbitrary user language by semantic type: metric displacement commands should become "
            "NAVIGATION instructions with measurable direction and distance preserved; portal or threshold commands "
            "should become NAVIGATION instructions toward a passage, entrance, doorway, boundary, or transition area; "
            "affordance-bearing object interaction commands should become ACTION or object-approach instructions that "
            "name the target control, switch, button, handle, knob, or appliance control naturally, not as a hard-coded "
            "graph object id.\n"
            f"{ACTION_INSTRUCTION_GUIDANCE}\n"
            f"{NAVIGATION_INSTRUCTION_GUIDANCE}\n"
            "Prefer dynamic execution and set `metadata.dynamic_execution` to true unless there is a clear reason "
            "to disable it. When the task requires local interaction, prefer a short seed plan or concise mixed-agent plan chunk.\n"
            f"{interactive_guidance}"
            f"{vision_guidance}"
            "When Planning context JSON contains `agent_capabilities`, choose tool- or skill-specific Agents and actions "
            "from that list whenever the user intent matches a declared capability.\n"
            "Return JSON only."
        )

    @classmethod
    def parse_plan_response(
        cls,
        content: str,
        *,
        model: str,
        allow_empty: bool,
        default_dynamic: bool,
    ) -> Plan:
        payload = cls.extract_json(content)
        raw_subtasks = payload.get("subtasks")
        if not isinstance(raw_subtasks, list):
            raise ValueError("LLM planner returned invalid subtasks")
        if not raw_subtasks and not allow_empty:
            raise ValueError("LLM planner returned no subtasks")

        subtasks: list[Subtask] = []
        for index, item in enumerate(raw_subtasks, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"Subtask #{index} must be an object")
            agent = cls.parse_agent(item.get("agent"))
            action = str(item.get("action", "")).strip()
            if not action:
                raise ValueError(f"Subtask #{index} is missing action")
            subtask_id = str(item.get("subtask_id") or f"st_{index:02d}")
            target = item.get("target") if isinstance(item.get("target"), dict) else {}
            parameters = item.get("parameters") if isinstance(item.get("parameters"), dict) else {}
            context = item.get("context") if isinstance(item.get("context"), dict) else {}
            subtasks.append(
                Subtask(
                    subtask_id=subtask_id,
                    agent=agent,
                    action=action,
                    target=target,
                    parameters=parameters,
                    context=context,
                )
            )

        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        metadata.setdefault("planner", "openai_compatible")
        metadata.setdefault("model", model)
        metadata.setdefault("dynamic_execution", default_dynamic)
        return Plan(subtasks=subtasks, metadata=metadata)

    @classmethod
    def validate_plan_semantics(cls, plan: Plan) -> None:
        for subtask in plan.subtasks:
            error = cls.validate_subtask_instruction_semantics(subtask)
            if error is not None:
                raise ValueError(error)

    @classmethod
    def validate_subtask_instruction_semantics(cls, subtask: Subtask) -> str | None:
        if subtask.agent not in {AgentName.NAVIGATION, AgentName.VISION, AgentName.ACTION}:
            return None

        instruction = str(subtask.parameters.get("instruction") or "").strip()
        if subtask.agent == AgentName.ACTION and not instruction:
            return (
                f"Subtask {subtask.subtask_id} ACTION is missing parameters.instruction. "
                "Every ACTION subtask must include a natural-language command naming the target object and "
                "actionable part/control when applicable."
            )
        if not instruction:
            return None

        object_name = cls.normalized_text(subtask.target.get("object"))
        part_name = cls.normalized_text(subtask.target.get("part"))
        if not object_name:
            return None

        instruction_text = cls.normalized_text(instruction)
        if not instruction_text:
            return None

        room_terms = {
            cls.normalized_text(subtask.target.get("room")),
            cls.normalized_text(subtask.target.get("room_name")),
            cls.normalized_text(subtask.target.get("room_id")),
        }
        room_terms = {term for term in room_terms if term}
        if not room_terms:
            return None

        has_room_reference = any(term in instruction_text for term in room_terms)
        if not has_room_reference:
            return None

        mentions_object = object_name in instruction_text
        mentions_part = bool(part_name) and part_name in instruction_text
        if mentions_object or mentions_part:
            return None

        if subtask.agent == AgentName.NAVIGATION and not cls.requires_object_named_instruction(
            subtask
        ):
            return None

        return (
            f"Subtask {subtask.subtask_id} instruction '{instruction}' does not match target "
            f"{json.dumps(subtask.target, ensure_ascii=False, default=str)}"
        )

    @classmethod
    def build_validation_retry_prompt(cls, base_prompt: str, *, error: str) -> str:
        return (
            f"{base_prompt}\n"
            "Previous response was invalid.\n"
            f"Validation error: {error}\n"
            "Regenerate the full JSON plan. Ensure each subtask instruction matches its structured target. "
            "Every ACTION subtask must include `parameters.instruction` as a concrete local interaction command. "
            "For object-targeted VISION/ACTION subtasks and object-level NAVIGATION `approach_target` subtasks, do not use room "
            "names, room instance names, or room ids as the acted-on item when `target.object` or `target.part` is available.\n"
            "Return JSON only."
        )

    @staticmethod
    def parse_agent(value: Any) -> AgentName:
        try:
            return AgentName.parse(value)
        except ValueError as exc:
            raise ValueError(f"Unsupported planner agent '{value}'") from exc

    @staticmethod
    def normalized_text(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        return " ".join(value.replace("_", " ").lower().split()).strip()

    @classmethod
    def normalize_room_label(cls, value: Any) -> str:
        text = cls.normalized_text(value)
        if not text:
            return ""
        parts = text.split()
        if parts and parts[-1].isdigit():
            return " ".join(parts[:-1]).strip()
        return text

    @classmethod
    def labels_match(cls, left: Any, right: Any) -> bool:
        left_text = cls.normalized_text(left)
        right_text = cls.normalized_text(right)
        if not left_text or not right_text:
            return False
        return left_text == right_text or cls.normalize_room_label(
            left_text
        ) == cls.normalize_room_label(right_text)

    @classmethod
    def requires_object_named_instruction(cls, subtask: Subtask) -> bool:
        if subtask.agent in {AgentName.VISION, AgentName.ACTION}:
            return True
        action = cls.normalized_text(subtask.action)
        return "approach" in action

    @staticmethod
    def first_non_empty(*values: Any) -> str | None:
        for value in values:
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def coerce_mapping(value: Any, fallback: Any | None = None) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if isinstance(fallback, dict):
            return dict(fallback)
        return {}

    @classmethod
    def compact_scene_report(cls, value: Any) -> dict[str, Any]:
        report = value if isinstance(value, dict) else {}
        return {
            "target_visible": report.get("target_visible"),
            "target_part_visible": report.get("target_part_visible"),
            "target_part_name": report.get("target_part_name"),
            "task_complete": report.get("task_complete"),
        }

    @classmethod
    def compact_navigation_report(cls, value: Any) -> dict[str, Any]:
        report = value if isinstance(value, dict) else {}
        return {
            "current_room": report.get("current_room"),
            "current_region": report.get("current_region"),
            "target_room_status": report.get("target_room_status"),
            "distance_to_execution_goal_m": report.get("distance_to_execution_goal_m"),
            "approach_reachable": report.get("approach_reachable"),
            "approach_ready": report.get("approach_ready"),
            "approach_stalled": report.get("approach_stalled"),
            "path_backend": report.get("path_backend"),
            "controller_mode": report.get("controller_mode"),
            "follow_status": report.get("follow_status"),
        }

    @classmethod
    def compact_navigation_state(cls, value: Any) -> dict[str, Any]:
        state = value if isinstance(value, dict) else {}
        return {
            "current_room": state.get("current_room"),
            "current_region": state.get("current_region"),
            "room_id": state.get("room_id"),
            "floor_id": state.get("floor_id"),
        }

    @classmethod
    def compact_latest_result(cls, value: Any) -> dict[str, Any]:
        result = value if isinstance(value, dict) else {}
        return {
            "subtask_id": result.get("subtask_id"),
            "execution_id": result.get("execution_id"),
            "plan_revision": result.get("plan_revision"),
            "agent": result.get("agent"),
            "status": result.get("status"),
            "error_code": result.get("error_code"),
            "task_complete": result.get("task_complete"),
            "scene_report": cls.compact_scene_report(result.get("scene_report")),
            "navigation_failure_context": cls.compact_navigation_failure_context(
                result.get("navigation_failure_context")
            ),
        }

    @classmethod
    def compact_navigation_failure_context(cls, value: Any) -> dict[str, Any]:
        context = value if isinstance(value, dict) else {}
        door_candidates = context.get("door_candidates")
        return {
            "failure_type": context.get("failure_type"),
            "path_backend": context.get("path_backend"),
            "nav2_error": context.get("nav2_error"),
            "portal_block_reason": context.get("portal_block_reason"),
            "blocked_transition": (
                cls._select_fields(
                    context.get("blocked_transition"),
                    (
                        "source_room_id",
                        "source_room_name",
                        "target_room_id",
                        "target_room_name",
                    ),
                )
                if isinstance(context.get("blocked_transition"), dict)
                else {}
            ),
            "local_goal": cls._compact_navigation_goal(context.get("local_goal")),
            "transition_anchor": cls._compact_navigation_goal(context.get("transition_anchor")),
            "execution_goal": cls._compact_navigation_goal(context.get("execution_goal")),
            "door_candidates": [
                cls._select_fields(
                    item,
                    (
                        "id",
                        "name",
                        "room",
                        "floor_id",
                        "position",
                        "is_open",
                        "in_rooms",
                        "source_room_id",
                        "source_room_name",
                        "target_room_id",
                        "target_room_name",
                        "distance_to_transition_m",
                    ),
                )
                for item in door_candidates[:5]
                if isinstance(item, dict)
            ]
            if isinstance(door_candidates, list)
            else [],
        }

    @classmethod
    def _compact_navigation_goal(cls, value: Any) -> dict[str, Any]:
        return cls._select_fields(
            value,
            (
                "goal_type",
                "waypoint_type",
                "object_id",
                "object_name",
                "room_id",
                "room_name",
                "source_room_name",
                "target_room_name",
                "floor_id",
                "x",
                "y",
                "z",
                "portal_gap",
                "portal_source_point",
                "portal_target_point",
            ),
        )

    @classmethod
    def _select_fields(cls, value: Any, keys: tuple[str, ...]) -> dict[str, Any]:
        mapping = value if isinstance(value, dict) else {}
        return {
            key: cls._strip_heavy_fields(mapping[key])
            for key in keys
            if mapping.get(key) not in (None, "", [], {})
        }

    @classmethod
    def compact_failed_subtask(cls, value: Any) -> dict[str, Any]:
        failed = value if isinstance(value, dict) else {}
        return {
            "subtask_id": failed.get("subtask_id"),
            "agent": failed.get("agent"),
            "action": failed.get("action"),
            "target": dict(failed.get("target")) if isinstance(failed.get("target"), dict) else {},
            "instruction": (
                dict(failed.get("parameters")).get("instruction")
                if isinstance(failed.get("parameters"), dict)
                else None
            ),
        }

    @classmethod
    def compact_similar_episodes(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [cls.compact_episode_summary(item) for item in value[:2]]
        if isinstance(value, dict):
            compact = dict(value)
            results = value.get("results")
            if isinstance(results, list):
                compact["results"] = [cls.compact_episode_summary(item) for item in results[:2]]
            return compact
        return value

    @classmethod
    def compact_memory_evidence_summary(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        retrieval = value.get("retrieval")
        retrieval = retrieval if isinstance(retrieval, dict) else {}
        navigation = value.get("navigation_guidance")
        navigation = navigation if isinstance(navigation, dict) else {}
        runtime = value.get("runtime")
        runtime = runtime if isinstance(runtime, dict) else {}
        history = navigation.get("object_approach_history")
        history = history if isinstance(history, dict) else {}
        entries = history.get("entries")
        compact_history = dict(history)
        if isinstance(entries, list):
            compact_history["entries"] = [cls._strip_heavy_fields(item) for item in entries[-5:]]
            compact_history["entry_count"] = len(entries)
        return {
            "query_type": value.get("query_type"),
            "query": cls._strip_heavy_fields(value.get("query", {})),
            "retrieval": {
                "experience_hints": cls._compact_retrieval_results(
                    retrieval.get("experience_hints"), limit=3
                ),
                "failure_patterns": cls._compact_retrieval_results(
                    retrieval.get("failure_patterns"), limit=3
                ),
                "causal": cls._compact_causal_evidence(retrieval.get("causal")),
                "counterfactual": cls._compact_counterfactual_evidence(
                    retrieval.get("counterfactual")
                ),
            },
            "navigation_guidance": {
                "object_approach_history": compact_history,
                "avoid_object_approach_candidates": [
                    cls._strip_heavy_fields(item)
                    for item in navigation.get("avoid_object_approach_candidates", [])[:5]
                    if isinstance(item, dict)
                ],
                "prefer_object_approach_candidates": [
                    cls._strip_heavy_fields(item)
                    for item in navigation.get("prefer_object_approach_candidates", [])[:5]
                    if isinstance(item, dict)
                ],
                "risk_reasons": list(navigation.get("risk_reasons", [])[:5])
                if isinstance(navigation.get("risk_reasons"), list)
                else [],
            },
            "runtime": {
                "working_state": cls._strip_heavy_fields(runtime.get("working_state", {})),
                "working_evidence": cls._strip_heavy_fields(runtime.get("working_evidence", {})),
                "task_context": cls._strip_heavy_fields(runtime.get("task_context", {})),
                "recent_observations": [
                    cls._strip_heavy_fields(item)
                    for item in runtime.get("recent_observations", [])[-5:]
                    if isinstance(item, dict)
                ],
            },
            "metadata": cls._strip_heavy_fields(value.get("metadata", {})),
        }

    @classmethod
    def _compact_causal_evidence(cls, value: Any) -> dict[str, Any]:
        causal = value if isinstance(value, dict) else {}
        negative = causal.get("negative_evidence")
        chains = causal.get("chain_summaries")
        negative_items = negative if isinstance(negative, list) else []
        chain_items = chains if isinstance(chains, list) else []
        return {
            "negative_evidence": [
                cls._strip_heavy_fields(item)
                for item in negative_items[:5]
                if isinstance(item, dict)
            ],
            "chain_summaries": [
                {
                    key: cls._strip_heavy_fields(item[key])
                    for key in ("edge_ids", "cumulative_strength", "length", "explanation")
                    if key in item
                }
                for item in chain_items[:5]
                if isinstance(item, dict)
            ],
        }

    @classmethod
    def _compact_counterfactual_evidence(cls, value: Any) -> dict[str, Any]:
        counterfactual = value if isinstance(value, dict) else {}
        results = counterfactual.get("results")
        result_items = results if isinstance(results, list) else []
        keep_keys = (
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
        return {
            "query_type": counterfactual.get("query_type", "counterfactual"),
            "query": cls._strip_heavy_fields(counterfactual.get("query", {})),
            "results": [
                {key: cls._strip_heavy_fields(item[key]) for key in keep_keys if key in item}
                for item in result_items[:3]
                if isinstance(item, dict)
            ],
            "metadata": cls._strip_heavy_fields(counterfactual.get("metadata", {})),
        }

    @classmethod
    def _compact_retrieval_results(cls, value: Any, *, limit: int) -> Any:
        if not isinstance(value, dict):
            return value
        compact = dict(value)
        results = value.get("results")
        if isinstance(results, list):
            compact["results"] = [cls._strip_heavy_fields(item) for item in results[:limit]]
        scores = value.get("scores")
        if isinstance(scores, list):
            compact["scores"] = scores[:limit]
        return cls._strip_heavy_fields(compact)

    @classmethod
    def _strip_heavy_fields(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [cls._strip_heavy_fields(item) for item in value[:20]]
        if not isinstance(value, dict):
            return value
        value = strip_image_payloads(value)
        heavy_keys = {
            "embedding",
            "embeddings",
            "vector",
            "raw_image",
            "raw_images",
            "raw_image_omitted",
            "raw_images_count",
            "image",
            "image_omitted",
            "image_b64_omitted",
            "action_sequence",
        }
        return {
            key: cls._strip_heavy_fields(item)
            for key, item in value.items()
            if key not in heavy_keys
        }

    @classmethod
    def compact_episode_summary(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        keep_keys = (
            "episode_id",
            "task_description",
            "task_type",
            "start_time",
            "end_time",
            "outcome",
            "failure_reason",
            "failure_action_idx",
            "objects_involved",
            "locations_visited",
            "initial_state",
            "final_state",
            "causal_annotations",
            "lessons_learned",
            "improvement_suggestions",
            "importance",
            "access_count",
            "last_accessed",
        )
        compact = {key: value.get(key) for key in keep_keys if key in value}
        action_sequence = value.get("action_sequence")
        if isinstance(action_sequence, list):
            compact["action_count"] = len(action_sequence)
        state_transitions = value.get("state_transitions")
        if isinstance(state_transitions, list):
            compact["state_transition_count"] = len(state_transitions)
        return cls._strip_heavy_fields(compact)

    @classmethod
    def serialize_context(cls, context: dict[str, Any]) -> str:
        compact = {
            "objects": context.get("objects", [])[:3]
            if isinstance(context.get("objects"), list)
            else context.get("objects"),
            "similar_episodes": cls.compact_similar_episodes(context.get("similar_episodes")),
            "skills": context.get("skills", [])[:3]
            if isinstance(context.get("skills"), list)
            else context.get("skills"),
            "memory_evidence_summary": cls.compact_memory_evidence_summary(
                context.get("memory_evidence_summary")
            ),
            "metadata": context.get("metadata", {}),
            "task_type": context.get("task_type"),
            "task_type_hint": context.get("task_type_hint"),
            "planner_mode": context.get("planner_mode", "auto"),
            "interactive_alignment_error": context.get("interactive_alignment_error"),
            "agent_capabilities": context.get("agent_capabilities", []),
            "interaction_target_hints": context.get("interaction_target_hints", {}),
            "navigation_state": context.get("navigation_state", {}),
            "navigation_report": cls.compact_navigation_report(context.get("navigation_report")),
            "last_scene_report": cls.compact_scene_report(context.get("last_scene_report")),
            "working_state": context.get("working_state", {}),
            "task_context": context.get("task_context", {}),
            "available_tools": context.get("available_tools", []),
            "tool_trace": context.get("tool_trace", [])[-5:]
            if isinstance(context.get("tool_trace"), list)
            else [],
            "external_constraints": context.get("external_constraints", {}),
            "schedule_state": context.get("schedule_state", {}),
            "recent_observations": (
                context.get("recent_observations", [])[-5:]
                if isinstance(context.get("recent_observations"), list)
                else context.get("recent_observations")
            ),
        }
        return json.dumps(compact, ensure_ascii=False, default=str)

    @classmethod
    def serialize_execution_state(cls, execution_state: dict[str, Any]) -> str:
        compact = {
            "task_type": execution_state.get("task_type"),
            "planner_mode": execution_state.get("planner_mode"),
            "next_subtask_index": execution_state.get("next_subtask_index"),
            "latest_result": cls.compact_latest_result(execution_state.get("latest_result")),
            "last_scene_report": cls.compact_scene_report(execution_state.get("last_scene_report")),
            "navigation_state": cls.compact_navigation_state(
                execution_state.get("navigation_state")
            ),
            "navigation_report": cls.compact_navigation_report(
                execution_state.get("navigation_report")
            ),
            "task_progress": execution_state.get("task_progress"),
            "failure_reason": execution_state.get("failure_reason"),
            "failed_subtask": cls.compact_failed_subtask(execution_state.get("failed_subtask")),
            "completed_subtasks": execution_state.get("completed_subtasks"),
            "plan_history": execution_state.get("plan_history"),
        }
        return json.dumps(compact, ensure_ascii=False, default=str)

    @classmethod
    def planning_decision_summary(
        cls, *, context: dict[str, Any], execution_state: dict[str, Any]
    ) -> dict[str, Any]:
        target_hints = (
            context.get("interaction_target_hints")
            if isinstance(context.get("interaction_target_hints"), dict)
            else {}
        )
        navigation_state = cls.coerce_mapping(
            execution_state.get("navigation_state"),
            fallback=context.get("navigation_state"),
        )
        navigation_report = cls.coerce_mapping(
            execution_state.get("navigation_report"),
            fallback=context.get("navigation_report"),
        )
        scene_report = cls.coerce_mapping(
            execution_state.get("last_scene_report"),
            fallback=context.get("last_scene_report"),
        )
        failed_subtask = cls.coerce_mapping(execution_state.get("failed_subtask"))
        target_room = cls.first_non_empty(
            target_hints.get("room_name"),
            target_hints.get("room"),
            target_hints.get("region"),
        )
        room_level_status = cls.room_level_status(
            target_room=target_room,
            navigation_state=navigation_state,
            navigation_report=navigation_report,
        )
        object_level_status = cls.object_level_status(
            scene_report,
            navigation_report=navigation_report,
            room_level_status=room_level_status,
        )
        failure_reason = str(execution_state.get("failure_reason") or "").strip() or None
        summary = {
            "target_object": target_hints.get("object"),
            "target_part": target_hints.get("part"),
            "target_room": target_room,
            "current_room": cls.first_non_empty(
                navigation_state.get("current_room"),
                navigation_state.get("current_region"),
            ),
            "room_level_status": room_level_status,
            "object_level_status": object_level_status,
            "approach_ready": navigation_report.get("approach_ready"),
            "approach_reachable": navigation_report.get("approach_reachable"),
            "approach_stalled": navigation_report.get("approach_stalled"),
            "failure_reason": failure_reason,
        }
        summary["planning_hint"] = cls.planning_hint(
            room_level_status=room_level_status,
            object_level_status=object_level_status,
            failed_subtask=failed_subtask,
            failure_reason=failure_reason,
            navigation_report=navigation_report,
        )
        return summary

    @staticmethod
    def starting_subtask_id(execution_state: dict[str, Any]) -> str:
        try:
            next_index = int(execution_state.get("next_subtask_index", 1))
        except (TypeError, ValueError):
            next_index = 1
        return f"st_{max(1, next_index):02d}"

    @classmethod
    def room_level_status(
        cls,
        *,
        target_room: str | None,
        navigation_state: dict[str, Any],
        navigation_report: dict[str, Any],
    ) -> str:
        explicit = cls.normalized_text(navigation_report.get("target_room_status"))
        if explicit in {
            "no_room_constraint",
            "target_room_unknown",
            "already_in_target_room",
            "outside_target_room",
        }:
            return explicit
        if not target_room:
            return "no_room_constraint"
        current_room = cls.first_non_empty(
            navigation_state.get("current_room"),
            navigation_state.get("current_region"),
        )
        if not current_room:
            return "target_room_unknown"
        if cls.labels_match(target_room, current_room):
            return "already_in_target_room"
        return "outside_target_room"

    @classmethod
    def object_level_status(
        cls,
        scene_report: dict[str, Any],
        *,
        navigation_report: dict[str, Any],
        room_level_status: str,
    ) -> str:
        if bool(scene_report.get("task_complete", False)):
            return "task_complete"
        target_visible = bool(scene_report.get("target_visible", False))
        approach_ready = bool(navigation_report.get("approach_ready", False))
        in_target_room = room_level_status in {"already_in_target_room", "no_room_constraint"}
        if target_visible and approach_ready:
            return "local_manipulation_ready"
        if target_visible and in_target_room and not approach_ready:
            return "object_level_approach_required"
        if in_target_room and not target_visible:
            return "visual_disambiguation_required"
        if target_visible and room_level_status == "outside_target_room":
            return "room_mismatch_detected"
        return "state_ambiguous"

    @classmethod
    def planning_hint(
        cls,
        *,
        room_level_status: str,
        object_level_status: str,
        failed_subtask: dict[str, Any],
        failure_reason: str | None,
        navigation_report: dict[str, Any],
    ) -> str:
        failed_agent = cls.normalized_text(failed_subtask.get("agent"))
        failed_action = cls.normalized_text(failed_subtask.get("action"))
        approach_reachable = navigation_report.get("approach_reachable")
        approach_stalled = bool(navigation_report.get("approach_stalled", False))
        if object_level_status == "task_complete":
            return "Task appears complete. Prefer an empty plan or a final VISION verification only if the system still needs confirmation."
        if room_level_status == "outside_target_room":
            return "Use NAVIGATION navigate to reach the target room or region before local inspection or manipulation."
        if object_level_status == "local_manipulation_ready":
            return 'Use ACTION for local interaction with parameters.control_mode = "whole_body_local". Do not insert room-level NAVIGATION.'
        if object_level_status == "object_level_approach_required":
            if approach_reachable is False:
                return "The target is visible, but navigation_report does not show a reachable local approach yet. Prefer one clarifying VISION inspect only if the target/part is ambiguous; otherwise replan conservatively without room-only navigation."
            if approach_stalled:
                return "The target is visible but the object-level approach appears stalled. Do not repeat room-only navigation. Either retry NAVIGATION approach_target with an object-centered instruction or use one VISION clarification step if the target part is still ambiguous."
            if (
                failed_agent == "vln"
                and "approach" in failed_action
                and failure_reason in {"SUBTASK_TIMEOUT", "NO_PROGRESS", "STALLED"}
            ):
                return "Same-room is already satisfied. Do not repeat room-only navigation. Either approach the object with an object-centered NAVIGATION instruction or use one VISION clarification step if the target part is still ambiguous."
            return "Use NAVIGATION approach_target toward the object. The instruction must name the target object or part, not the room instance name."
        if object_level_status == "visual_disambiguation_required":
            return "Use exactly one VISION inspect/find step to localize the object or target part within the current room before NAVIGATION/ACTION."
        return "Prefer the shortest clarifying step. Use NAVIGATION only for room-level relocation, ACTION for local interaction, and VISION for ambiguity or verification."

    @staticmethod
    def extract_json(content: str) -> dict[str, Any]:
        stripped = content.strip()
        candidates = [stripped]
        if "```" in stripped:
            parts = stripped.split("```")
            candidates.extend(part.strip() for part in parts if part.strip())

        decoder = JSONDecoder()
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                for idx, char in enumerate(candidate):
                    if char != "{":
                        continue
                    try:
                        parsed, end = decoder.raw_decode(candidate[idx:])
                    except json.JSONDecodeError:
                        continue
                    if isinstance(parsed, dict):
                        return parsed
                    if end:
                        continue
            else:
                if isinstance(parsed, dict):
                    return parsed

        raise ValueError("Failed to parse JSON from Brain planner response")


__all__ = ["DefaultBrainPlanningSkill", "_SYSTEM_PROMPT"]
