"""Rule-based planner backend owned by the Brain agent body."""

from __future__ import annotations

import re
from typing import Any

from voltron.shared.enums import AgentName
from voltron.shared.context import Plan, Subtask


class RuleBasedPlanner:
    """Generate baseline manipulation plans from text using simple heuristics."""

    def plan(self, task_description: str, context: dict[str, Any]) -> Plan:
        planner_mode = self._planner_mode(context)
        if planner_mode == "scripted":
            return self._plan_scripted(task_description, context)
        if planner_mode == "benchmark":
            plan = self._plan_auto(task_description, context)
            metadata = dict(plan.metadata)
            metadata["mode"] = f"benchmark_{metadata.get('mode', 'auto')}"
            metadata["planner_mode"] = "benchmark"
            return Plan(subtasks=plan.subtasks, metadata=metadata)
        return self._plan_auto(task_description, context)

    def _plan_scripted(self, task_description: str, context: dict[str, Any]) -> Plan:
        task_type = str(context.get("task_type", "")).strip().lower()
        if task_type == "navigation":
            target = self._extract_navigation_target(task_description)
            return Plan(
                subtasks=[self._build_navigation_subtask(target=target, task_description=task_description)],
                metadata={"planner": "rule_based", "dynamic_execution": False, "mode": "navigation_direct", "planner_mode": "scripted"},
            )
        if task_type == "interaction":
            slots = self._extract_interaction_slots(task_description)
            return Plan(
                subtasks=[self._build_interaction_inspect_subtask(slots, task_description, index=1)],
                metadata={"planner": "rule_based", "dynamic_execution": True, "mode": "interaction_stepwise", "planner_mode": "scripted"},
            )

        slots = self._extract_slots(task_description)
        return Plan(
            subtasks=self._build_transfer_plan(slots, task_description),
            metadata={"planner": "rule_based", "dynamic_execution": False, "mode": "transfer_static", "planner_mode": "scripted"},
        )

    def _plan_auto(self, task_description: str, context: dict[str, Any]) -> Plan:
        photo_subtask = self._build_photo_subtask_if_requested(task_description, context)
        if photo_subtask is not None:
            return Plan(
                subtasks=[photo_subtask],
                metadata={
                    "planner": "rule_based",
                    "dynamic_execution": False,
                    "mode": "photo_capture",
                    "planner_mode": "auto",
                },
            )

        long_range_slots = self._extract_long_range_interaction_slots(task_description, context)
        if long_range_slots:
            if self._uses_native_pi05_policy(context):
                subtasks = [
                    self._build_interaction_approach_subtask(
                        slots=long_range_slots,
                        task_description=task_description,
                        index=1,
                    ),
                    self._build_long_range_interaction_action_subtask(
                        slots=long_range_slots,
                        task_description=task_description,
                        index=2,
                    ),
                ]
                return Plan(
                    subtasks=subtasks,
                    metadata={"planner": "rule_based", "dynamic_execution": False, "mode": "auto_pi05_long_range_interaction", "planner_mode": "auto"},
                )

            subtasks = [
                self._build_navigation_subtask(
                    target=long_range_slots["room"],
                    task_description=task_description,
                ),
                self._build_long_range_interaction_action_subtask(
                    slots=long_range_slots,
                    task_description=task_description,
                    index=2,
                ),
            ]
            return Plan(
                subtasks=subtasks,
                metadata={"planner": "rule_based", "dynamic_execution": True, "mode": "auto_long_range_interaction", "planner_mode": "auto"},
            )

        transfer_slots = self._extract_slots(task_description)
        if {"object", "source", "destination"}.issubset(transfer_slots):
            return Plan(
                subtasks=self._build_transfer_plan(transfer_slots, task_description),
                metadata={"planner": "rule_based", "dynamic_execution": False, "mode": "transfer_static", "planner_mode": "auto"},
            )

        interaction_slots = self._extract_interaction_slots(task_description)
        if interaction_slots.get("object") != "target_object":
            return Plan(
                subtasks=[self._build_interaction_inspect_subtask(interaction_slots, task_description, index=1)],
                metadata={"planner": "rule_based", "dynamic_execution": True, "mode": "auto_local_interaction", "planner_mode": "auto"},
            )

        navigation_target = self._extract_navigation_target(task_description)
        if navigation_target and navigation_target != task_description.strip():
            return Plan(
                subtasks=[self._build_navigation_subtask(target=navigation_target, task_description=task_description)],
                metadata={"planner": "rule_based", "dynamic_execution": False, "mode": "navigation_direct", "planner_mode": "auto"},
            )

        return Plan(
            subtasks=[
                Subtask(
                    subtask_id="st_01",
                    agent=AgentName.VISION,
                    action="observe",
                    target={},
                    parameters={"instruction": f"Observe the scene relevant to: {task_description}"},
                    context={"task_description": task_description},
                )
            ],
            metadata={"planner": "rule_based", "dynamic_execution": True, "mode": "auto_observe_seed", "planner_mode": "auto"},
        )

    def plan_next(
        self,
        task_description: str,
        context: dict[str, Any],
        execution_state: dict[str, Any],
    ) -> Plan:
        planner_mode = self._planner_mode(context)
        if not self._should_use_dynamic_interaction_flow(task_description, context):
            return Plan(subtasks=[], metadata={"planner": "rule_based", "dynamic_execution": False, "planner_mode": planner_mode})

        slots = self._extract_interaction_slots(task_description)
        latest_result = execution_state.get("latest_result", {})
        latest_agent = self._canonical_agent_label(latest_result.get("agent"))
        latest_action_keys = latest_result.get("action_keys", [])
        latest_task_complete = bool(latest_result.get("task_complete", False))
        scene_report = dict(latest_result.get("scene_report") or execution_state.get("last_scene_report") or {})
        navigation_report = self._navigation_report(execution_state)
        next_index = self._coerce_next_index(execution_state.get("next_subtask_index"))
        vla_attempts = self._count_agent_results(
            execution_state.get("recent_results", []),
            agent=AgentName.ACTION.value,
            action_keys_required=True,
        )

        if latest_task_complete:
            return Plan(subtasks=[], metadata={"planner": "rule_based", "dynamic_execution": True, "planner_mode": planner_mode})

        if latest_agent == AgentName.ACTION.value and latest_action_keys:
            return Plan(
                subtasks=[self._build_interaction_verify_subtask(slots, task_description, next_index)],
                metadata={"planner": "rule_based", "dynamic_execution": True, "planner_mode": planner_mode},
            )

        if latest_agent == AgentName.NAVIGATION.value and latest_action_keys:
            return Plan(
                subtasks=[self._build_interaction_inspect_subtask(slots, task_description, next_index)],
                metadata={"planner": "rule_based", "dynamic_execution": True, "planner_mode": planner_mode},
            )

        if latest_agent == AgentName.VISION.value and latest_result.get("scene_report"):
            if bool(scene_report.get("task_complete", False)):
                return Plan(subtasks=[], metadata={"planner": "rule_based", "dynamic_execution": True})
            if self._in_target_room(navigation_report) and bool(scene_report.get("target_visible", False)) and bool(navigation_report.get("approach_ready", False)) and vla_attempts < 2:
                return Plan(
                    subtasks=[
                        self._build_interaction_action_subtask(slots, task_description, next_index),
                        self._build_interaction_verify_subtask(slots, task_description, next_index + 1),
                    ],
                    metadata={"planner": "rule_based", "dynamic_execution": True, "planner_mode": planner_mode},
                )
            if self._outside_target_room(navigation_report) and slots.get("room") not in (None, "", "target_room"):
                return Plan(
                    subtasks=[
                        Subtask(
                            subtask_id=self._subtask_id(next_index),
                            agent=AgentName.NAVIGATION,
                            action="navigate",
                            target={"region": slots["room"], "room": slots["room"]},
                            parameters={"instruction": f"navigate to {slots['room']}"},
                            context={"task_description": task_description},
                        )
                    ],
                    metadata={"planner": "rule_based", "dynamic_execution": True, "planner_mode": planner_mode},
                )
            if bool(scene_report.get("target_visible", False)) and navigation_report.get("approach_reachable") is not False:
                return Plan(
                    subtasks=[
                        self._build_interaction_approach_subtask(slots, task_description, next_index),
                        self._build_interaction_inspect_subtask(slots, task_description, next_index + 1),
                    ],
                    metadata={"planner": "rule_based", "dynamic_execution": True, "planner_mode": planner_mode},
                )
            return Plan(
                subtasks=[self._build_interaction_inspect_subtask(slots, task_description, next_index)],
                metadata={"planner": "rule_based", "dynamic_execution": True, "planner_mode": planner_mode},
            )

        return Plan(
            subtasks=[self._build_interaction_inspect_subtask(slots, task_description, next_index)],
            metadata={"planner": "rule_based", "dynamic_execution": True, "planner_mode": planner_mode},
        )

    def replan(
        self,
        task_description: str,
        context: dict[str, Any],
        failed_subtask: Subtask,
        failure_reason: str,
        execution_state: dict[str, Any],
    ) -> Plan:
        planner_mode = self._planner_mode(context)
        if self._should_use_dynamic_interaction_flow(task_description, context):
            slots = self._extract_interaction_slots(task_description)
            next_index = self._coerce_next_index(execution_state.get("next_subtask_index"))
            latest_result = execution_state.get("latest_result", {})
            scene_report = dict(latest_result.get("scene_report") or execution_state.get("last_scene_report") or {})
            navigation_report = self._navigation_report(execution_state)

            if failed_subtask.agent == AgentName.NAVIGATION and failure_reason == "SUBTASK_TIMEOUT":
                if bool(scene_report.get("target_visible", False)) and bool(navigation_report.get("approach_ready", False)):
                    return Plan(
                        subtasks=[
                            self._build_interaction_action_subtask(slots, task_description, next_index),
                            self._build_interaction_verify_subtask(slots, task_description, next_index + 1),
                        ],
                        metadata={"planner": "rule_based", "dynamic_execution": True, "mode": "timeout_local_recovery", "planner_mode": planner_mode},
                    )
                return Plan(
                    subtasks=[self._build_interaction_inspect_subtask(slots, task_description, next_index)],
                    metadata={"planner": "rule_based", "dynamic_execution": True, "mode": "timeout_reinspect", "planner_mode": planner_mode},
                )

        retry_subtask = Subtask(
            subtask_id=f"{failed_subtask.subtask_id}_retry",
            agent=failed_subtask.agent,
            action=failed_subtask.action,
            target=failed_subtask.target,
            parameters={
                **failed_subtask.parameters,
                "retry": True,
                "previous_failure": failure_reason,
            },
            context={**failed_subtask.context, "replan": True},
        )
        return Plan(
            subtasks=[retry_subtask],
            metadata={
                "planner": "rule_based",
                "dynamic_execution": self._should_use_dynamic_interaction_flow(task_description, context),
                "mode": "single_retry",
                "planner_mode": planner_mode,
            },
        )

    @staticmethod
    def _navigation_report(execution_state: dict[str, Any]) -> dict[str, Any]:
        report = execution_state.get("navigation_report")
        if isinstance(report, dict):
            return dict(report)
        return {}

    def _build_photo_subtask_if_requested(
        self,
        task_description: str,
        context: dict[str, Any],
    ) -> Subtask | None:
        capability = self._photo_capture_capability(context)
        if capability is None or not self._matches_photo_intent(task_description):
            return None

        action_names = capability.get("action_names")
        if not isinstance(action_names, (list, tuple)):
            return None
        declared_actions = [
            action_name.strip()
            for action_name in action_names
            if isinstance(action_name, str) and action_name.strip()
        ]
        if not declared_actions:
            return None

        action = "take_photo" if "take_photo" in declared_actions else declared_actions[0]
        capability_id = str(capability.get("capability_id") or "vision.photo.capture")
        return Subtask(
            subtask_id="st_01",
            agent=AgentName.VISION,
            action=action,
            target={},
            parameters={
                "views": self._extract_camera_views(task_description),
                "instruction": task_description,
            },
            context={"capability_id": capability_id},
        )

    @staticmethod
    def _photo_capture_capability(context: dict[str, Any]) -> dict[str, Any] | None:
        capabilities = context.get("agent_capabilities")
        if not isinstance(capabilities, list):
            return None
        for capability in capabilities:
            if isinstance(capability, dict) and capability.get("capability_id") == "vision.photo.capture":
                return capability
        return None

    @staticmethod
    def _matches_photo_intent(task_description: str) -> bool:
        text = task_description.lower()
        return bool(
            re.search(
                r"\b(capture|save|record|snap)\b.{0,32}\b(photos?|pictures?|snapshots?|images?)\b",
                text,
            )
            or re.search(r"\btake\s+(a|an)?\s*(photo|picture|snapshot|image)\b(?!\s+[a-z])", text)
            or re.search(
                r"\b(photos?|snapshots?|images?)\b.{0,32}\b(take|capture|save|record|snap)\b",
                text,
            )
            or re.search(r"\bcamera\b.{0,32}\b(photos?|pictures?|snapshots?|images?)\b", text)
            or re.search(r"\b(photos?|pictures?|snapshots?|images?)\b.{0,32}\bcamera\b", text)
            or re.search(r"(拍|拍摄|保存|捕获|记录).{0,12}(照|照片|图像|相机)", text)
            or re.search(r"(照片|图像).{0,12}(保存|拍摄|记录)", text)
        )

    @staticmethod
    def _extract_camera_views(task_description: str) -> list[str]:
        text = task_description.lower()
        if re.search(r"\bwrist cameras?\b", text) or "腕部相机" in text or "手腕相机" in text:
            return ["left_wrist", "right_wrist"]
        view_terms = {
            "head": ("head", "head camera", "头部", "主相机"),
            "left_wrist": ("left_wrist", "left wrist", "左腕", "左手"),
            "right_wrist": ("right_wrist", "right wrist", "右腕", "右手"),
        }
        matches: list[tuple[int, str]] = []
        for view, terms in view_terms.items():
            positions = [text.find(term) for term in terms if text.find(term) >= 0]
            if positions:
                matches.append((min(positions), view))
        if not matches:
            return ["head"]

        ordered = [view for _, view in sorted(matches, key=lambda item: item[0])]
        if "head" in ordered:
            return ["head", *[view for view in ordered if view != "head"]]
        return ordered

    @staticmethod
    def _in_target_room(navigation_report: dict[str, Any]) -> bool:
        return str(navigation_report.get("target_room_status") or "").strip().lower() in {
            "already_in_target_room",
            "no_room_constraint",
        }

    @staticmethod
    def _outside_target_room(navigation_report: dict[str, Any]) -> bool:
        return str(navigation_report.get("target_room_status") or "").strip().lower() == "outside_target_room"

    @staticmethod
    def _count_agent_results(
        results: list[dict[str, Any]],
        *,
        agent: str | None = None,
        action_keys_required: bool = False,
    ) -> int:
        count = 0
        for item in results:
            if not isinstance(item, dict):
                continue
            if agent and RuleBasedPlanner._canonical_agent_label(item.get("agent")) != RuleBasedPlanner._canonical_agent_label(agent):
                continue
            action_keys = item.get("action_keys", [])
            if action_keys_required and not action_keys:
                continue
            count += 1
        return count

    def _build_transfer_plan(self, slots: dict[str, str], task_description: str) -> list[Subtask]:
        """Build a standard Navigation->Vision->Action pipeline."""
        source = slots.get("source", "source_region")
        destination = slots.get("destination", "target_region")
        obj = slots.get("object", "target_object")

        return [
            Subtask(
                subtask_id="st_01",
                agent=AgentName.NAVIGATION,
                action="navigate",
                target={"region": source},
                parameters={"mode": "to_source"},
            ),
            Subtask(
                subtask_id="st_02",
                agent=AgentName.VISION,
                action="observe",
                target={"object": obj, "region": source},
                parameters={"instruction": f"locate {obj} in {source}"},
            ),
            Subtask(
                subtask_id="st_03",
                agent=AgentName.ACTION,
                action="grasp",
                target={"object": obj},
                parameters={"instruction": f"grasp {obj}"},
            ),
            Subtask(
                subtask_id="st_04",
                agent=AgentName.NAVIGATION,
                action="navigate",
                target={"region": destination},
                parameters={"mode": "to_destination"},
            ),
            Subtask(
                subtask_id="st_05",
                agent=AgentName.VISION,
                action="observe",
                target={"region": destination},
                parameters={"instruction": f"verify destination {destination}"},
            ),
            Subtask(
                subtask_id="st_06",
                agent=AgentName.ACTION,
                action="place",
                target={"object": obj, "region": destination},
                parameters={"instruction": f"place {obj} at {destination}"},
                context={"task_description": task_description},
            ),
        ]

    def _build_navigation_subtask(self, *, target: str, task_description: str) -> Subtask:
        target_region = target or "target_region"
        return Subtask(
            subtask_id="st_01",
            agent=AgentName.NAVIGATION,
            action="navigate",
            target={"region": target_region, "room": target_region},
            parameters={"mode": "navigation_only", "instruction": f"navigate to {target_region}"},
            context={"task_description": task_description},
        )

    def _extract_slots(self, task_description: str) -> dict[str, str]:
        """Extract rough object/source/destination hints from Chinese or English text."""
        text = task_description.strip()
        slots: dict[str, str] = {}

        cn_pattern = r"把(?P<object>.+?)从(?P<source>.+?)(?:拿到|放到|移动到|转移到)(?P<destination>.+)$"
        cn_match = re.search(cn_pattern, text)
        if cn_match:
            return {
                "object": cn_match.group("object").strip(),
                "source": cn_match.group("source").strip(),
                "destination": cn_match.group("destination").strip(),
            }

        en_pattern = r"move\s+(?P<object>.+?)\s+from\s+(?P<source>.+?)\s+to\s+(?P<destination>.+)$"
        en_match = re.search(en_pattern, text, flags=re.IGNORECASE)
        if en_match:
            return {
                "object": en_match.group("object").strip(),
                "source": en_match.group("source").strip(),
                "destination": en_match.group("destination").strip(),
            }

        if "把" in text:
            after_ba = text.split("把", 1)[-1]
            slots["object"] = after_ba.split("从", 1)[0].strip() or "target_object"

        return slots

    @staticmethod
    def _planner_mode(context: dict[str, Any]) -> str:
        mode = str(context.get("planner_mode", "auto")).strip().lower()
        if mode in {"auto", "scripted", "benchmark"}:
            return mode
        return "auto"

    def _extract_navigation_target(self, task_description: str) -> str:
        text = task_description.strip()
        patterns = (
            r"(?:导航到|前往|去到|去往|去)(?P<target>.+)$",
            r"(?:navigate to|go to|head to|move to)\s+(?P<target>.+)$",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                target = match.group("target").strip()
                if target:
                    return target
        return text or "target_region"

    def _extract_navigation_room_hint(self, task_description: str, context: dict[str, Any]) -> str | None:
        metadata = context.get("metadata", {}) if isinstance(context.get("metadata"), dict) else {}
        for key in ("target_room", "room", "target_region", "region"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        text = task_description.strip()
        patterns = (
            r"(?:导航到|前往|去到|去往|去)(?P<room>.+?)(?:并|然后|再|后|，|,|。|$)",
            r"(?:go to|head to|navigate to|move to)\s+(?:the\s+)?(?P<room>.+?)(?:\s+and\s+|\s+then\s+|[,.]|$)",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                room = match.group("room").strip()
                if room:
                    return room
        return None

    def _extract_long_range_interaction_slots(
        self,
        task_description: str,
        context: dict[str, Any],
    ) -> dict[str, str] | None:
        slots = self._extract_interaction_slots(task_description)
        if slots.get("object") == "target_object":
            return None

        room = self._extract_navigation_room_hint(task_description, context)
        if not room:
            return None

        return {**slots, "room": room, "region": room}

    @staticmethod
    def _uses_native_pi05_policy(context: dict[str, Any]) -> bool:
        metadata = context.get("metadata") if isinstance(context.get("metadata"), dict) else {}
        return str(metadata.get("policy_backend") or "").strip().lower() == "pi05"

    def _should_use_dynamic_interaction_flow(self, task_description: str, context: dict[str, Any]) -> bool:
        if self._uses_native_pi05_policy(context):
            return False
        task_type = str(context.get("task_type", "")).strip().lower()
        if self._planner_mode(context) == "scripted":
            return task_type == "interaction"
        slots = self._extract_interaction_slots(task_description)
        return slots.get("object") != "target_object"

    def _extract_interaction_slots(self, task_description: str) -> dict[str, str]:
        text = task_description.strip()
        lowered = text.lower()
        slots: dict[str, str] = {
            "object": "target_object",
            "desired_state": "on",
            "action": "toggle_on",
        }

        if any(token in lowered for token in ("turn off", "switch off", "power off")) or "关闭" in text:
            slots["desired_state"] = "off"
            slots["action"] = "toggle_off"

        cn_match = re.search(r"(打开|开启|关闭)(?P<object>.+)$", text)
        if cn_match:
            slots["object"] = self._normalize_object_name(cn_match.group("object").strip())
        else:
            en_match = re.search(
                r"(turn on|turn off|switch on|switch off|open|close)\s+(the\s+)?(?P<object>.+)$",
                lowered,
            )
            if en_match:
                slots["object"] = self._normalize_object_name(en_match.group("object").strip())

        if slots["object"] == "radio":
            slots["part"] = "power button"

        return slots

    @staticmethod
    def _normalize_object_name(value: str) -> str:
        normalized = value.strip().lower()
        aliases = {
            "收音机": "radio",
            "radio": "radio",
            "音响": "speaker",
            "扬声器": "speaker",
            "微波炉": "microwave",
            "电视": "tv",
            "电灯": "lamp",
            "灯": "lamp",
            "门": "door",
        }
        return aliases.get(normalized, value.strip() or "target_object")

    @staticmethod
    def _coerce_next_index(value: Any) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 1
        return max(1, parsed)

    @staticmethod
    def _subtask_id(index: int) -> str:
        return f"st_{index:02d}"

    def _build_interaction_inspect_subtask(
        self,
        slots: dict[str, str],
        task_description: str,
        index: int,
    ) -> Subtask:
        target = {"object": slots.get("object", "target_object")}
        if slots.get("part"):
            target["part"] = slots["part"]
        return Subtask(
            subtask_id=self._subtask_id(index),
            agent=AgentName.VISION,
            action="inspect_scene",
            target=target,
            parameters={
                "instruction": (
                    f"Inspect the {target['object']} and determine whether the object or target part is visible."
                )
            },
            context={"task_description": task_description},
        )

    def _build_interaction_approach_subtask(
        self,
        slots: dict[str, str],
        task_description: str,
        index: int,
    ) -> Subtask:
        target = {"object": slots.get("object", "target_object")}
        if slots.get("room"):
            target["room"] = slots["room"]
            target["region"] = slots["room"]
        return Subtask(
            subtask_id=self._subtask_id(index),
            agent=AgentName.NAVIGATION,
            action="approach_target",
            target=target,
            parameters={"instruction": f"Approach the {target['object']} for local interaction."},
            context={"task_description": task_description},
        )

    def _build_interaction_action_subtask(
        self,
        slots: dict[str, str],
        task_description: str,
        index: int,
    ) -> Subtask:
        target = {"object": slots.get("object", "target_object")}
        if slots.get("part"):
            target["part"] = slots["part"]
        return Subtask(
            subtask_id=self._subtask_id(index),
            agent=AgentName.ACTION,
            action=slots.get("action", "toggle_on"),
            target=target,
            parameters={
                "instruction": self._local_interaction_instruction(
                    action=slots.get("action", "toggle_on"),
                    target=target,
                ),
                "control_mode": "whole_body_local",
            },
            context={"task_description": task_description, "execution_mode": "local_interaction"},
        )

    def _build_long_range_interaction_action_subtask(
        self,
        slots: dict[str, str],
        task_description: str,
        index: int,
    ) -> Subtask:
        target = {"object": slots.get("object", "target_object")}
        if slots.get("part"):
            target["part"] = slots["part"]
        if slots.get("room"):
            target["room"] = slots["room"]
            target["region"] = slots["room"]
        return Subtask(
            subtask_id=self._subtask_id(index),
            agent=AgentName.ACTION,
            action=slots.get("action", "toggle_on"),
            target=target,
            parameters={
                "instruction": self._local_interaction_instruction(
                    action=slots.get("action", "toggle_on"),
                    target=target,
                ),
                "control_mode": "whole_body_local",
            },
            context={"task_description": task_description, "execution_mode": "long_range_interaction"},
        )

    def _build_interaction_verify_subtask(
        self,
        slots: dict[str, str],
        task_description: str,
        index: int,
    ) -> Subtask:
        target = {"object": slots.get("object", "target_object")}
        if slots.get("part"):
            target["part"] = slots["part"]
        desired_state = slots.get("desired_state", "on")
        return Subtask(
            subtask_id=self._subtask_id(index),
            agent=AgentName.VISION,
            action="verify_state",
            target=target,
            parameters={
                "instruction": f"Verify whether the {target['object']} is {desired_state}.",
                "allow_task_complete": True,
            },
            context={"task_description": task_description},
        )

    @staticmethod
    def _canonical_agent_label(value: Any) -> str:
        normalized = str(value or "").strip().upper()
        legacy_map = {
            "LLM": AgentName.BRAIN.value,
            "VLM": AgentName.VISION.value,
            "VLN": AgentName.NAVIGATION.value,
            "VLA": AgentName.ACTION.value,
        }
        return legacy_map.get(normalized, normalized)

    @staticmethod
    def _local_interaction_instruction(*, action: str, target: dict[str, Any]) -> str:
        obj = str(target.get("object") or "target object").strip()
        part = str(target.get("part") or "").strip()

        if action == "toggle_on":
            instruction = f"turn on the {obj}"
        elif action == "toggle_off":
            instruction = f"turn off the {obj}"
        elif action == "open":
            instruction = f"open the {obj}"
        elif action == "close":
            instruction = f"close the {obj}"
        else:
            instruction = f"{action.replace('_', ' ')} the {obj}"

        if part:
            instruction = f"{instruction} using the {part}"
        return instruction
