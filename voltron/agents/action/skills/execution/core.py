from __future__ import annotations

import time

from voltron.agents.action.tools.action_projection import ActionProjection
from voltron.shared.action_semantics import action_instruction, normalize_action_name
from voltron.shared.enums import AgentStatus
from voltron.shared.context import ExecutionContext, LocalSkillSelection, Subtask
from voltron.shared.results import AgentResult
from voltron.shared.contracts import MemoryAdapter, PolicyAdapter


def resolve_control_mode(subtask: Subtask) -> str:
    for source in (subtask.parameters, subtask.context):
        value = source.get("control_mode") or source.get("execution_mode")
        if isinstance(value, str) and value.strip():
            normalized = value.strip().lower()
            if normalized in {"whole_body_local", "local_interaction"}:
                return "whole_body_local"
    return "manipulation_only"


def allow_local_base_motion(subtask: Subtask) -> bool:
    for source in (subtask.parameters, subtask.context):
        for key in ("allow_base_motion", "vla_allow_base_motion"):
            value = source.get(key)
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.strip():
                return value.strip().lower() in {"1", "true", "yes", "free", "enabled"}
        mode = source.get("base_motion") or source.get("base_motion_mode")
        if isinstance(mode, str) and mode.strip():
            return mode.strip().lower() in {"free", "enabled", "allow", "allowed"}
    return False


class PolicyBackedVLASkill:
    skill_id = "default_manipulation_skill"
    supported_actions: tuple[str, ...] = ()

    def __init__(
        self,
        memory: MemoryAdapter,
        policy: PolicyAdapter,
        projector: ActionProjection,
    ) -> None:
        self.memory = memory
        self.policy = policy
        self.projector = projector

    def can_handle(self, subtask: Subtask, context: ExecutionContext) -> bool:
        if not self.supported_actions:
            return True
        return normalize_action_name(subtask.action) in self.supported_actions

    def execute(
        self,
        subtask: Subtask,
        context: ExecutionContext,
        selection: LocalSkillSelection,
    ) -> AgentResult:
        start = time.time()

        observation = subtask.parameters.get("observation")
        if not isinstance(observation, dict):
            return AgentResult(
                subtask_id=subtask.subtask_id,
                status=AgentStatus.FAILURE,
                error_code="MANIP_OBSERVATION_MISSING",
                result={"message": "subtask.parameters['observation'] is required"},
                latency_ms=self._latency_ms(start),
            )

        raw_obs = subtask.parameters.get("raw_observation")
        if raw_obs is not None:
            observation["raw_observation"] = raw_obs

        target = str(subtask.target.get("object", subtask.target.get("object_id", "")))
        conditions = subtask.parameters.get("conditions", {})

        try:
            prediction = self.memory.predict_action_effects(
                action=subtask.action,
                target=target,
                conditions=conditions,
            )

            canonical_action = normalize_action_name(subtask.action)
            options = dict(subtask.parameters.get("policy_options") or {})
            options.setdefault("action", canonical_action)
            options.setdefault("action_type", canonical_action)
            options.setdefault("raw_action", subtask.action)
            options.setdefault(
                "instruction",
                str(
                    subtask.parameters.get("instruction")
                    or subtask.context.get("instruction")
                    or action_instruction(action=canonical_action, target=dict(subtask.target))
                ),
            )

            action, info = self.policy.get_action(
                observation,
                options=options,
            )
            control_mode = resolve_control_mode(subtask)
            local_base_motion_allowed = allow_local_base_motion(subtask)
            if "robot_r1" in action or (
                control_mode == "whole_body_local" and local_base_motion_allowed
            ):
                executed_action = action
            else:
                executed_action = self.projector.project_manipulation(action)
            self.projector.update_last_safe_action(executed_action)

            record_payload = {
                "action_type": subtask.action,
                "target": target,
                "parameters": subtask.parameters.get("action_parameters", {}),
                "pre_state": subtask.parameters.get("pre_state", {}),
                "post_state": subtask.parameters.get("post_state", {}),
                "success": bool(subtask.parameters.get("success", True)),
                "failure_reason": subtask.parameters.get("failure_reason"),
                "duration": float(subtask.parameters.get("duration", 0.0)),
                "skill_id": selection.skill_id,
                "skill_source": selection.source,
                "selector_confidence": selection.confidence,
            }
            action_stats = self.memory.record_action(record_payload)
        except Exception as exc:
            return AgentResult(
                subtask_id=subtask.subtask_id,
                status=AgentStatus.FAILURE,
                error_code="VLA_SKILL_EXECUTION_FAILED",
                result={
                    "message": str(exc),
                    "skill_id": selection.skill_id,
                    "skill_source": selection.source,
                },
                latency_ms=self._latency_ms(start),
            )

        return AgentResult(
            subtask_id=subtask.subtask_id,
            status=AgentStatus.SUCCESS,
            result={
                "action_keys": sorted(executed_action.keys()),
                "control_mode": control_mode,
                "local_base_motion_allowed": local_base_motion_allowed,
                "prediction": prediction,
                "policy_info": info,
                "memory_update": action_stats,
                "skill_id": selection.skill_id,
                "skill_source": selection.source,
                "selector_confidence": selection.confidence,
                "selector_reason": selection.reason,
                "fallback_skill_candidates": selection.fallback_skill_candidates,
            },
            runtime_artifacts={
                "full_action": action,
                "projected_action": executed_action,
                "policy_info": info,
                "local_base_motion_allowed": local_base_motion_allowed,
                "skill_selection": {
                    "skill_id": selection.skill_id,
                    "source": selection.source,
                    "confidence": selection.confidence,
                    "reason": selection.reason,
                    "fallback_skill_candidates": selection.fallback_skill_candidates,
                },
            },
            latency_ms=self._latency_ms(start),
        )

    @staticmethod
    def _latency_ms(start: float) -> int:
        return int((time.time() - start) * 1000)


class PolicyBackedActionSkill(PolicyBackedVLASkill):
    pass


__all__ = [
    "PolicyBackedActionSkill",
    "PolicyBackedVLASkill",
    "allow_local_base_motion",
    "resolve_control_mode",
]
