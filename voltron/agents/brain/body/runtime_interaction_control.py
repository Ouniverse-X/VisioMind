"""Runtime interaction control helpers owned by the Brain agent body."""

from __future__ import annotations

from typing import Any

from voltron.agents.brain.tools import interaction_targeting, navigation_runtime
from voltron.shared.enums import AgentName, TaskType
from voltron.shared.context import ExecutionContext, Subtask, TaskRequest


class RuntimeInteractionControlPolicy:
    @classmethod
    def deterministic_interaction_plan(
        cls,
        *,
        request: TaskRequest,
        context: ExecutionContext,
        next_index: int,
        execution_state: dict[str, Any] | None,
        fallback_subtasks: list[Subtask] | None = None,
    ) -> list[Subtask] | None:
        if not cls.should_apply_runtime_interaction_control(
            request=request,
            subtasks=[
                *list(getattr(context.runtime_state.get("planner_plan"), "subtasks", []) or []),
                *list(fallback_subtasks or []),
            ],
        ):
            return None

        target_hints = interaction_targeting.interaction_target_hints(
            request=request,
            subtasks=[
                *list(getattr(context.runtime_state.get("planner_plan"), "subtasks", []) or []),
                *list(fallback_subtasks or []),
            ],
        )
        navigation_state = navigation_runtime.resolve_navigation_state(
            execution_state=execution_state,
            environment_state=context.runtime_state.get("environment"),
        )
        has_current_room = navigation_runtime.room_state_available(navigation_state)
        in_target_room = navigation_runtime.room_state_matches_target(
            target_hints=target_hints,
            navigation_state=navigation_state,
        )

        latest_result = execution_state.get("latest_result", {}) if isinstance(execution_state, dict) else {}
        latest_agent = cls.canonical_agent_label(latest_result.get("agent"))
        latest_scene_report = (
            dict(latest_result.get("scene_report") or execution_state.get("last_scene_report") or {})
            if isinstance(execution_state, dict)
            else {}
        )
        latest_task_complete = bool(latest_result.get("task_complete", False)) or bool(
            latest_scene_report.get("task_complete", False)
        )
        navigation_report = (
            dict(execution_state.get("navigation_report") or {})
            if isinstance(execution_state, dict)
            else {}
        )

        if latest_task_complete:
            return []

        if has_current_room and navigation_runtime.target_room_available(target_hints) and not in_target_room:
            return [
                cls.build_interaction_room_navigation_subtask(
                    target_hints=target_hints,
                    task_description=request.description,
                    index=next_index,
                )
            ]

        if execution_state is None:
            if in_target_room:
                return [
                    cls.build_interaction_inspect_subtask(
                        target_hints=target_hints,
                        task_description=request.description,
                        index=next_index,
                    )
                ]
            return None

        if latest_agent == AgentName.NAVIGATION.value:
            return [
                cls.build_interaction_inspect_subtask(
                    target_hints=target_hints,
                    task_description=request.description,
                    index=next_index,
                )
            ]

        if latest_agent == AgentName.VISION.value:
            target_visible = bool(
                latest_scene_report.get("target_visible", False)
                or latest_scene_report.get("target_part_visible", False)
            )
            if target_visible:
                if bool(navigation_report.get("approach_ready", False)):
                    return [
                        cls.build_interaction_action_subtask(
                            target_hints=target_hints,
                            task_description=request.description,
                            index=next_index,
                        )
                    ]
                if in_target_room:
                    return [
                        cls.build_interaction_approach_subtask(
                            target_hints=target_hints,
                            task_description=request.description,
                            index=next_index,
                            memory_navigation_guidance=cls.memory_navigation_guidance(context),
                        )
                    ]

        return None

    @classmethod
    def build_interaction_room_navigation_subtask(
        cls,
        *,
        target_hints: dict[str, str],
        task_description: str,
        index: int,
    ) -> Subtask:
        target_room = navigation_runtime.first_non_empty(
            target_hints.get("room"),
            target_hints.get("region"),
            "target_room",
        )
        target: dict[str, Any] = {"room": target_room, "region": target_room}
        canonical_room_name = navigation_runtime.first_non_empty(
            target_hints.get("canonical_room_name"),
            navigation_runtime.canonical_room_name(target_hints.get("room_name")),
        )
        room_label = navigation_runtime.first_non_empty(
            target_hints.get("room_label"),
            navigation_runtime.room_display_label(
                room=target_hints.get("room"),
                region=target_hints.get("region"),
                canonical_room_name_value=canonical_room_name,
                room_name=target_hints.get("room_name"),
            ),
        )
        room_name = navigation_runtime.first_non_empty(target_hints.get("room_name"))
        room_id = navigation_runtime.first_non_empty(target_hints.get("room_id"))
        if isinstance(room_label, str) and room_label.strip():
            target["room_label"] = room_label.strip()
        if isinstance(canonical_room_name, str) and canonical_room_name.strip():
            target["canonical_room_name"] = canonical_room_name.strip()
        if isinstance(room_name, str) and room_name.strip():
            target["room_name"] = room_name.strip()
        if isinstance(room_id, str) and room_id.strip():
            target["room_id"] = room_id.strip()
        target_instruction = navigation_runtime.first_non_empty(
            target.get("room_label"),
            target_room,
            target.get("canonical_room_name"),
            target.get("room_name"),
        )
        return Subtask(
            subtask_id=cls.subtask_id(index),
            agent=AgentName.NAVIGATION,
            action="navigate",
            target=target,
            parameters={"instruction": f"navigate to {target_instruction}", "mode": "to_target_room"},
            context={"task_description": task_description, "navigation_reason": "room_mismatch"},
        )

    @classmethod
    def build_interaction_approach_subtask(
        cls,
        *,
        target_hints: dict[str, str],
        task_description: str,
        index: int,
        memory_navigation_guidance: dict[str, Any] | None = None,
    ) -> Subtask:
        target: dict[str, Any] = {}
        if target_hints.get("object"):
            target["object"] = target_hints["object"]
        if target_hints.get("part"):
            target["part"] = target_hints["part"]
        if target_hints.get("room"):
            target["room"] = target_hints["room"]
        if target_hints.get("room_label"):
            target["room_label"] = target_hints["room_label"]
        if target_hints.get("canonical_room_name"):
            target["canonical_room_name"] = target_hints["canonical_room_name"]
        if target_hints.get("room_name"):
            target["room_name"] = target_hints["room_name"]
        if target_hints.get("room_id"):
            target["room_id"] = target_hints["room_id"]
        subtask_context: dict[str, Any] = {
            "task_description": task_description,
            "navigation_reason": "not_locally_operable",
        }
        if memory_navigation_guidance:
            subtask_context["memory_navigation_guidance"] = memory_navigation_guidance
        return Subtask(
            subtask_id=cls.subtask_id(index),
            agent=AgentName.NAVIGATION,
            action="approach_target",
            target=target,
            parameters={
                "instruction": f"Approach the {target.get('object', 'target object')} for local interaction.",
                "mode": "local_approach",
            },
            context=subtask_context,
        )

    @classmethod
    def build_interaction_action_subtask(
        cls,
        *,
        target_hints: dict[str, str],
        task_description: str,
        index: int,
    ) -> Subtask:
        target: dict[str, Any] = {}
        if target_hints.get("object"):
            target["object"] = target_hints["object"]
        if target_hints.get("part"):
            target["part"] = target_hints["part"]
        return Subtask(
            subtask_id=cls.subtask_id(index),
            agent=AgentName.ACTION,
            action="toggle_on",
            target=target,
            parameters={
                "instruction": f"turn on the {target.get('object', 'target object')}",
                "control_mode": "whole_body_local",
            },
            context={"task_description": task_description, "execution_mode": "local_interaction"},
        )

    @classmethod
    def build_interaction_inspect_subtask(
        cls,
        *,
        target_hints: dict[str, str],
        task_description: str,
        index: int,
    ) -> Subtask:
        target: dict[str, Any] = {}
        if target_hints.get("object"):
            target["object"] = target_hints["object"]
        if target_hints.get("part"):
            target["part"] = target_hints["part"]
        if target_hints.get("room"):
            target["room"] = target_hints["room"]
        if target_hints.get("room_name"):
            target["room_name"] = target_hints["room_name"]
        if target_hints.get("room_id"):
            target["room_id"] = target_hints["room_id"]
        return Subtask(
            subtask_id=cls.subtask_id(index),
            agent=AgentName.VISION,
            action="inspect_scene",
            target=target,
            parameters={
                "instruction": interaction_targeting.interaction_seed_instruction(target),
                "allow_task_complete": False,
            },
            context={"task_description": task_description, "seed_plan": True},
        )

    @staticmethod
    def subtask_id(index: int) -> str:
        return f"st_{index:02d}"

    @classmethod
    def memory_navigation_guidance(cls, context: ExecutionContext) -> dict[str, Any]:
        planning_context = context.runtime_state.get("planning_context", {})
        if not isinstance(planning_context, dict):
            return {}
        summary = planning_context.get("memory_evidence_summary", {})
        if not isinstance(summary, dict):
            return {}
        guidance = summary.get("navigation_guidance", {})
        if not isinstance(guidance, dict):
            return {}
        compact: dict[str, Any] = {}
        for key in (
            "avoid_object_approach_candidates",
            "prefer_object_approach_candidates",
            "risk_reasons",
        ):
            value = guidance.get(key)
            if isinstance(value, list) and value:
                compact[key] = [dict(item) if isinstance(item, dict) else item for item in value[:5]]
        return compact

    @staticmethod
    def coerce_next_index(value: Any) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 1
        return max(1, parsed)

    @staticmethod
    def planner_mode_from_request(request: TaskRequest) -> str:
        metadata = request.metadata if isinstance(request.metadata, dict) else {}
        mode = str(metadata.get("planner_mode", "auto")).strip().lower()
        if mode in {"auto", "scripted", "benchmark"}:
            return mode
        return "auto"

    @classmethod
    def should_apply_runtime_interaction_control(
        cls,
        *,
        request: TaskRequest,
        subtasks: list[Subtask],
    ) -> bool:
        planner_mode = cls.planner_mode_from_request(request)
        if planner_mode == "scripted":
            return request.task_type == TaskType.INTERACTION

        has_action = any(subtask.agent == AgentName.ACTION for subtask in subtasks)
        if has_action:
            return True

        if interaction_targeting.infer_object_from_text(request.description):
            return True

        return request.task_type == TaskType.INTERACTION and planner_mode == "benchmark"

    @staticmethod
    def canonical_agent_label(value: Any) -> str:
        normalized = str(value or "").strip().upper()
        legacy_map = {
            "LLM": AgentName.BRAIN.value,
            "VLM": AgentName.VISION.value,
            "VLN": AgentName.NAVIGATION.value,
            "VLA": AgentName.ACTION.value,
        }
        return legacy_map.get(normalized, normalized)
