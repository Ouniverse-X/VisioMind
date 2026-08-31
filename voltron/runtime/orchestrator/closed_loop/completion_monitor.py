from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from voltron.shared.context import ExecutionContext, Subtask
from voltron.shared.models import (
    CompletionCriterion,
    CompletionEvaluationContext,
    CompletionVerdict,
    RuntimeFeedback,
    SubtaskStepOutcome,
)
from voltron.shared.results import AgentResult

logger = logging.getLogger(__name__)


@dataclass
class CompletionDecision:
    done: bool
    success: bool | None
    verdict: CompletionVerdict
    failure_reason: str | None = None
    feedback: RuntimeFeedback | dict[str, Any] | None = None

    def to_step_outcome(self) -> SubtaskStepOutcome:
        return SubtaskStepOutcome(
            done=self.done,
            success=self.success,
            failure_reason=self.failure_reason,
            feedback=self.feedback or {},
        )


class CompletionMonitor:
    def __init__(
        self,
        *,
        use_environment_success_signal: bool = True,
        use_brain_completion_signal: bool = True,
        environment_signal_policy: str = "allow_early_success",
        evaluator: Any | None = None,
        positive_streak: int = 1,
        stability_steps: int = 1,
        action_delta_threshold: float = 0.03,
        check_interval_steps: int = 1,
        completion_agent_scope: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.use_environment_success_signal = bool(use_environment_success_signal)
        self.use_brain_completion_signal = bool(use_brain_completion_signal)
        self.environment_signal_policy = str(environment_signal_policy or "allow_early_success")
        self.evaluator = evaluator
        self.positive_streak = max(1, int(positive_streak))
        self.stability_steps = max(1, int(stability_steps))
        self.action_delta_threshold = max(0.0, float(action_delta_threshold))
        self.check_interval_steps = max(1, int(check_interval_steps))
        self.completion_agent_scope = self._normalize_agent_scope(completion_agent_scope)
        self._scope_state: dict[str, dict[str, Any]] = {}

    def evaluate_subtask_step(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        result: AgentResult,
        environment_outcome: SubtaskStepOutcome,
        control_step: int,
    ) -> CompletionDecision:
        env_done = bool(environment_outcome.done)
        env_success = environment_outcome.success
        feedback = environment_outcome.feedback
        env_success_evidence_only = self._environment_success_evidence_only(environment_outcome)
        scope_state = self._state_for_scope("subtask", subtask.runtime_id)
        if self._agent_in_completion_scope(subtask):
            self._update_stability_steps(scope_state, environment_outcome)

        if env_done and env_success is False:
            verdict = CompletionVerdict(
                scope="subtask",
                scope_id=subtask.runtime_id,
                completed=False,
                confidence=1.0,
                reason=environment_outcome.failure_reason or "environment marked subtask failure",
                evidence=self._environment_evidence(environment_outcome, control_step),
                should_continue=False,
                source="environment",
            )
            return CompletionDecision(
                done=True,
                success=False,
                failure_reason=environment_outcome.failure_reason,
                feedback=feedback,
                verdict=verdict,
            )

        result_payload = result.result if isinstance(result.result, dict) else {}
        anygrasp_execution_pending = (
            result_payload.get("skill_id") == "anygrasp_manipulation_skill"
            and result_payload.get("grasp_plan_completed") is False
        )
        if anygrasp_execution_pending:
            evidence = self._environment_evidence(environment_outcome, control_step)
            evidence.update(
                {
                    "skill_id": result_payload.get("skill_id"),
                    "skill_source": result_payload.get("skill_source"),
                    "grasp_plan_completed": False,
                }
            )
            verdict = CompletionVerdict(
                scope="subtask",
                scope_id=subtask.runtime_id,
                completed=False,
                confidence=1.0 if env_success is True else 0.0,
                reason=(
                    "environment success observed; waiting for AnyGrasp execution result"
                    if env_success is True
                    else "waiting for AnyGrasp execution result"
                ),
                evidence=evidence,
                should_continue=True,
                source="pending_action_result",
            )
            return CompletionDecision(
                done=False,
                success=None,
                feedback=feedback,
                verdict=verdict,
            )

        is_anygrasp_result = result_payload.get("skill_id") == "anygrasp_manipulation_skill"
        anygrasp_execution_completed = bool(
            is_anygrasp_result and result_payload.get("grasp_plan_completed")
        )
        physical_evidence = result_payload.get("physical_evidence")
        if not isinstance(physical_evidence, dict):
            physical_evidence = {}
        verified_grasp = bool(
            anygrasp_execution_completed
            and result_payload.get("grasp_success")
            and result_payload.get("physical_grasp_verified") is True
            and physical_evidence.get("passed") is True
            and physical_evidence.get("target_z_rise_passed") is True
            and physical_evidence.get("relative_pose_stable") is True
            and physical_evidence.get("object_identity_matches") is True
            and physical_evidence.get("attachment_passed") is True
            and physical_evidence.get("sample_count")
            == physical_evidence.get("required_sample_count")
            and result_payload.get("object_in_hand") == result_payload.get("target_object")
            and str(result_payload.get("skill_source", "")).endswith("curobo")
        )
        if verified_grasp and env_done and env_success is True:
            verdict = CompletionVerdict(
                scope="subtask",
                scope_id=subtask.runtime_id,
                completed=True,
                confidence=1.0,
                reason=(
                    f"physically verified grasp completed with object "
                    f"{result_payload.get('object_in_hand')} in hand"
                ),
                evidence={
                    "source": "action_agent_and_environment",
                    "control_step": control_step,
                    "skill_source": result_payload.get("skill_source"),
                    "object_in_hand": result_payload.get("object_in_hand"),
                    "environment_success": True,
                    "physical_grasp_verified": True,
                    "physical_evidence": physical_evidence,
                },
                should_continue=False,
                source="verified_physical_grasp_and_environment",
            )
            return CompletionDecision(done=True, success=True, feedback=feedback, verdict=verdict)

        if verified_grasp:
            evidence = self._environment_evidence(environment_outcome, control_step)
            evidence.update(
                {
                    "physical_grasp_verified": True,
                    "physical_evidence": physical_evidence,
                }
            )
            verdict = CompletionVerdict(
                scope="subtask",
                scope_id=subtask.runtime_id,
                completed=False,
                confidence=1.0,
                reason="physical grasp verified; waiting for environment/BDDL success",
                evidence=evidence,
                should_continue=True,
                source="pending_environment_result",
            )
            return CompletionDecision(
                done=False,
                success=None,
                feedback=feedback,
                verdict=verdict,
            )

        if anygrasp_execution_completed:
            evidence = self._environment_evidence(environment_outcome, control_step)
            evidence.update(
                {
                    "skill_source": result_payload.get("skill_source"),
                    "object_in_hand": result_payload.get("object_in_hand"),
                    "target_object": result_payload.get("target_object"),
                    "physical_grasp_verified": result_payload.get("physical_grasp_verified"),
                    "physical_evidence": physical_evidence,
                }
            )
            verdict = CompletionVerdict(
                scope="subtask",
                scope_id=subtask.runtime_id,
                completed=False,
                confidence=1.0,
                reason="AnyGrasp completed without independent physical grasp verification",
                evidence=evidence,
                should_continue=False,
                source="physical_grasp_verification_failed",
            )
            return CompletionDecision(
                done=True,
                success=False,
                failure_reason="GRASP_PHYSICAL_VERIFICATION_FAILED",
                feedback=feedback,
                verdict=verdict,
            )

        if (
            env_done
            and env_success is True
            and self._runtime_subtask_success_allowed(subtask, feedback)
        ):
            verdict = CompletionVerdict(
                scope="subtask",
                scope_id=subtask.runtime_id,
                completed=True,
                confidence=1.0,
                reason="runtime marked non-Vision subtask complete",
                evidence=self._environment_evidence(environment_outcome, control_step),
                should_continue=False,
                source="runtime_subtask",
            )
            return CompletionDecision(done=True, success=True, feedback=feedback, verdict=verdict)

        evaluator_verdict = self._evaluate_with_brain_vision(
            subtask=subtask,
            context=context,
            result=result,
            environment_outcome=environment_outcome,
            control_step=control_step,
        )
        if evaluator_verdict is not None and evaluator_verdict.completed:
            gated_verdict = self._apply_evaluator_gates(
                verdict=evaluator_verdict,
                control_step=control_step,
            )
            if gated_verdict.completed:
                if env_success_evidence_only:
                    gated_verdict = self._with_terminal_environment_evidence(
                        gated_verdict,
                        environment_outcome=environment_outcome,
                        control_step=control_step,
                        should_continue=False,
                    )
                return CompletionDecision(
                    done=True,
                    success=True,
                    feedback=feedback,
                    verdict=gated_verdict,
                )
            if env_success_evidence_only:
                gated_verdict = self._with_terminal_environment_evidence(
                    gated_verdict,
                    environment_outcome=environment_outcome,
                    control_step=control_step,
                    should_continue=True,
                )
            return CompletionDecision(
                done=False, success=None, feedback=feedback, verdict=gated_verdict
            )
        if evaluator_verdict is not None and self._should_reset_positive_streak(evaluator_verdict):
            scope_state["positive_streak"] = 0

        if env_done and self._environment_success_allowed():
            verdict = CompletionVerdict(
                scope="subtask",
                scope_id=subtask.runtime_id,
                completed=True,
                confidence=1.0,
                reason="environment marked subtask complete",
                evidence=self._environment_evidence(environment_outcome, control_step),
                should_continue=False,
                source="environment",
            )
            return CompletionDecision(
                done=True, success=env_success, feedback=feedback, verdict=verdict
            )

        if env_success_evidence_only:
            verdict = evaluator_verdict or CompletionVerdict(
                scope="subtask",
                scope_id=subtask.runtime_id,
                completed=False,
                confidence=0.0,
                reason="environment success is evidence only; waiting for Brain/Vision completion",
                evidence=self._environment_evidence(environment_outcome, control_step),
                should_continue=True,
                source="environment_evidence",
            )
            verdict = self._with_terminal_environment_evidence(
                verdict,
                environment_outcome=environment_outcome,
                control_step=control_step,
                should_continue=True,
            )
            return CompletionDecision(done=False, success=None, feedback=feedback, verdict=verdict)

        if env_done and self._environment_success_allowed():
            verdict = self._terminal_environment_without_completion_verdict(
                subtask=subtask,
                environment_outcome=environment_outcome,
                control_step=control_step,
                evaluator_verdict=evaluator_verdict,
            )
            return CompletionDecision(
                done=True,
                success=False,
                failure_reason="ENVIRONMENT_TERMINATED_BEFORE_BRAIN_COMPLETION",
                feedback=feedback,
                verdict=verdict,
            )

        source = "environment_evidence" if env_done else "completion_monitor"
        verdict = evaluator_verdict or CompletionVerdict(
            scope="subtask",
            scope_id=subtask.runtime_id,
            completed=False,
            confidence=0.0,
            reason="completion evidence is insufficient",
            evidence=self._environment_evidence(environment_outcome, control_step),
            should_continue=True,
            source=source,
        )
        if env_done and not self._environment_success_allowed():
            verdict.source = "environment_evidence"
            verdict.evidence.update(self._environment_evidence(environment_outcome, control_step))
        return CompletionDecision(done=False, success=None, feedback=feedback, verdict=verdict)

    def _terminal_environment_without_completion_verdict(
        self,
        *,
        subtask: Subtask,
        environment_outcome: SubtaskStepOutcome,
        control_step: int,
        evaluator_verdict: CompletionVerdict | None,
    ) -> CompletionVerdict:
        if evaluator_verdict is not None:
            return self._with_terminal_environment_evidence(
                evaluator_verdict,
                environment_outcome=environment_outcome,
                control_step=control_step,
                should_continue=False,
            )
        return CompletionVerdict(
            scope="subtask",
            scope_id=subtask.runtime_id,
            completed=False,
            confidence=0.0,
            reason="environment terminated before Brain/Vision completion was confirmed",
            evidence=self._environment_evidence(environment_outcome, control_step),
            should_continue=False,
            source="environment_evidence",
        )

    def _with_terminal_environment_evidence(
        self,
        verdict: CompletionVerdict,
        *,
        environment_outcome: SubtaskStepOutcome,
        control_step: int,
        should_continue: bool,
    ) -> CompletionVerdict:
        evidence = dict(verdict.evidence)
        evidence.update(self._environment_evidence(environment_outcome, control_step))
        return CompletionVerdict(
            scope=verdict.scope,
            scope_id=verdict.scope_id,
            completed=verdict.completed,
            confidence=verdict.confidence,
            reason=verdict.reason,
            evidence=evidence,
            missing_evidence=list(verdict.missing_evidence),
            should_continue=should_continue,
            should_replan=verdict.should_replan,
            source=verdict.source,
        )

    def _apply_evaluator_gates(
        self,
        *,
        verdict: CompletionVerdict,
        control_step: int,
    ) -> CompletionVerdict:
        state = self._state_for_scope(verdict.scope, verdict.scope_id)
        state["positive_streak"] = int(state.get("positive_streak", 0)) + 1
        stable_steps = int(state.get("stable_steps", 0))
        positive_streak = int(state["positive_streak"])
        gates_met = positive_streak >= self.positive_streak and stable_steps >= self.stability_steps
        evidence = dict(verdict.evidence)
        evidence.update(
            {
                "positive_streak": positive_streak,
                "positive_streak_required": self.positive_streak,
                "stable_steps": stable_steps,
                "stable_steps_required": self.stability_steps,
                "control_step": control_step,
            }
        )
        if gates_met:
            verdict.evidence = evidence
            return verdict
        return CompletionVerdict(
            scope=verdict.scope,
            scope_id=verdict.scope_id,
            completed=False,
            confidence=verdict.confidence,
            reason="completion verdict is positive but streak/stability gates are not yet satisfied",
            evidence=evidence,
            missing_evidence=[
                item
                for item, missing in (
                    ("positive_streak", positive_streak < self.positive_streak),
                    ("stable_steps", stable_steps < self.stability_steps),
                )
                if missing
            ],
            should_continue=True,
            should_replan=verdict.should_replan,
            source=verdict.source,
        )

    def _state_for_scope(self, scope: str, scope_id: str) -> dict[str, Any]:
        return self._scope_state.setdefault(
            f"{scope}:{scope_id}",
            {"positive_streak": 0, "stable_steps": 0, "last_pose": None},
        )

    @staticmethod
    def _should_reset_positive_streak(verdict: CompletionVerdict) -> bool:
        if verdict.completed:
            return False
        return verdict.source not in {
            "completion_monitor_interval",
            "completion_monitor_agent_scope",
            "completion_monitor_evaluator_error",
        }

    def _update_stability_steps(
        self,
        state: dict[str, Any],
        outcome: SubtaskStepOutcome,
    ) -> int:
        pose = self._pose_from_feedback(outcome.feedback)
        if pose is None:
            state["stable_steps"] = int(state.get("stable_steps", 0)) + 1
            return int(state["stable_steps"])
        last_pose = state.get("last_pose")
        state["last_pose"] = pose
        if last_pose is None:
            state["stable_steps"] = 1
            return 1
        if self._pose_delta(last_pose, pose) <= self.action_delta_threshold:
            state["stable_steps"] = int(state.get("stable_steps", 0)) + 1
        else:
            state["stable_steps"] = 1
        return int(state["stable_steps"])

    @staticmethod
    def _pose_from_feedback(feedback: Any) -> dict[str, float] | None:
        payload = CompletionMonitor._serialize_feedback(feedback)
        pose = payload.get("pose")
        if not isinstance(pose, dict):
            return None
        numeric: dict[str, float] = {}
        for key in ("x", "y", "z"):
            try:
                numeric[key] = float(pose.get(key, 0.0))
            except (TypeError, ValueError):
                return None
        return numeric

    @staticmethod
    def _pose_delta(previous: dict[str, float], current: dict[str, float]) -> float:
        return max(
            abs(float(current.get(key, 0.0)) - float(previous.get(key, 0.0)))
            for key in ("x", "y", "z")
        )

    def _environment_success_allowed(self) -> bool:
        return (
            self.use_environment_success_signal
            and self.environment_signal_policy != "evidence_only"
        )

    def _environment_success_evidence_only(self, outcome: SubtaskStepOutcome) -> bool:
        if self._environment_success_allowed():
            return False
        payload = self._serialize_feedback(outcome.feedback)
        return bool(payload.get("environment_success_evidence_only") or payload.get("task_success"))

    def _runtime_subtask_success_allowed(self, subtask: Subtask, feedback: Any) -> bool:
        if self._agent_in_completion_scope(subtask):
            return False
        payload = self._serialize_feedback(feedback)
        if not bool(payload.get("subtask_completed") or payload.get("subtask_succeeded")):
            return False
        completion_reason = str(payload.get("subtask_completion_reason") or "").strip()
        if bool(payload.get("task_success")) and completion_reason in {"", "task_success"}:
            return False
        return True

    def _evaluate_with_brain_vision(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        result: AgentResult,
        environment_outcome: SubtaskStepOutcome,
        control_step: int,
    ) -> CompletionVerdict | None:
        if not self.use_brain_completion_signal or self.evaluator is None:
            return None
        if not self._agent_in_completion_scope(subtask):
            return CompletionVerdict(
                scope="subtask",
                scope_id=subtask.runtime_id,
                completed=False,
                confidence=0.0,
                reason="Brain/Vision completion check skipped for this agent",
                evidence={
                    "control_step": control_step,
                    "agent": subtask.agent.value,
                    "completion_agent_scope": sorted(self.completion_agent_scope),
                },
                should_continue=True,
                source="completion_monitor_agent_scope",
            )
        if not self._should_run_brain_vision_check(environment_outcome, control_step):
            return CompletionVerdict(
                scope="subtask",
                scope_id=subtask.runtime_id,
                completed=False,
                confidence=0.0,
                reason=(
                    "Brain/Vision completion check skipped until configured interval "
                    f"{self.check_interval_steps}"
                ),
                evidence={
                    "control_step": control_step,
                    "check_interval_steps": self.check_interval_steps,
                    "next_check_control_step": self._next_check_control_step(control_step),
                },
                should_continue=True,
                source="completion_monitor_interval",
            )
        evaluation_context = self._evaluation_context(
            subtask=subtask,
            context=context,
            result=result,
            environment_outcome=environment_outcome,
            control_step=control_step,
        )
        try:
            return self.evaluator.evaluate(evaluation_context)
        except Exception as exc:
            logger.warning("Brain/Vision completion evaluator failed: %s", exc)
            return CompletionVerdict(
                scope=evaluation_context.scope,
                scope_id=evaluation_context.scope_id,
                completed=False,
                confidence=0.0,
                reason=f"Brain/Vision completion evaluator failed: {exc}",
                evidence={
                    "control_step": control_step,
                    "agent": subtask.agent.value,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
                missing_evidence=["vision_completion"],
                should_continue=True,
                source="completion_monitor_evaluator_error",
            )

    def _should_run_brain_vision_check(
        self, outcome: SubtaskStepOutcome, control_step: int
    ) -> bool:
        if bool(outcome.done):
            return True
        return int(control_step) % self.check_interval_steps == 0

    def _next_check_control_step(self, control_step: int) -> int:
        step = max(0, int(control_step))
        remainder = step % self.check_interval_steps
        if remainder == 0:
            return step
        return step + (self.check_interval_steps - remainder)

    def _agent_in_completion_scope(self, subtask: Subtask) -> bool:
        return str(subtask.agent.value).upper() in self.completion_agent_scope

    @staticmethod
    def _normalize_agent_scope(scope: set[str] | list[str] | tuple[str, ...] | None) -> set[str]:
        if scope is None:
            return {"ACTION"}
        normalized = {str(item).strip().upper() for item in scope if str(item).strip()}
        return normalized or {"ACTION"}

    def _evaluation_context(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        result: AgentResult,
        environment_outcome: SubtaskStepOutcome,
        control_step: int,
    ) -> CompletionEvaluationContext:
        task_context = self._task_context(context)
        return CompletionEvaluationContext(
            task_description=context.task_request.description,
            scope="subtask",
            scope_id=subtask.runtime_id,
            completion_criteria=self._completion_criteria(task_context, subtask),
            confirmed_text_plan=dict(
                (task_context.get("interactive_planning") or {}).get("text_plan") or {}
            ),
            current_subtask={
                "subtask_id": subtask.subtask_id,
                "agent": subtask.agent.value,
                "action": subtask.action,
                "target": dict(subtask.target),
                "instruction": str(subtask.parameters.get("instruction", "")),
            },
            runtime_feedback=self._serialize_feedback(environment_outcome.feedback),
            task_context=task_context,
            memory_evidence=dict(task_context.get("memory_evidence") or {}),
            action_stability={
                "control_step": control_step,
                "agent_status": result.status.value,
            },
        )

    @staticmethod
    def _task_context(context: ExecutionContext) -> dict[str, Any]:
        memory_context = context.runtime_state.get("task_context")
        return dict(memory_context) if isinstance(memory_context, dict) else {}

    @staticmethod
    def _completion_criteria(
        task_context: dict[str, Any], subtask: Subtask
    ) -> list[CompletionCriterion]:
        raw_items: list[Any] = []
        raw_items.extend(task_context.get("completion_criteria") or [])
        raw_items.extend(subtask.parameters.get("completion_criteria") or [])
        interactive = task_context.get("interactive_planning")
        if isinstance(interactive, dict):
            text_plan = interactive.get("text_plan")
            if isinstance(text_plan, dict):
                raw_items.extend(text_plan.get("success_criteria") or [])
        criteria: list[CompletionCriterion] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            payload = dict(item)
            payload.setdefault("scope", payload.get("scope") or "subtask")
            payload.setdefault("subtask_id", payload.get("subtask_id") or subtask.subtask_id)
            try:
                criterion = CompletionCriterion.from_value(payload)
            except (TypeError, ValueError):
                continue
            if not CompletionMonitor._criterion_applies_to_subtask(criterion, subtask):
                continue
            criteria.append(criterion)
        return criteria

    @staticmethod
    def _criterion_applies_to_subtask(criterion: CompletionCriterion, subtask: Subtask) -> bool:
        scope = str(criterion.scope or "").strip().lower()
        if scope == "task":
            return True
        if scope == "subtask":
            return criterion.subtask_id in (None, "", subtask.subtask_id)
        return criterion.subtask_id in (None, "", subtask.subtask_id)

    @classmethod
    def _environment_evidence(
        cls, outcome: SubtaskStepOutcome, control_step: int
    ) -> dict[str, Any]:
        feedback = cls._serialize_feedback(outcome.feedback)
        evidence = {
            "environment_done": bool(outcome.done),
            "environment_success": outcome.success,
            "control_step": control_step,
        }
        if outcome.failure_reason:
            evidence["failure_reason"] = outcome.failure_reason
        if feedback:
            evidence["feedback"] = feedback
        return evidence

    @staticmethod
    def _serialize_feedback(feedback: Any) -> dict[str, Any]:
        normalized = RuntimeFeedback.from_value(feedback)
        if normalized is not None:
            return normalized.to_dict()
        if isinstance(feedback, dict):
            return dict(feedback)
        return {}


__all__ = ["CompletionDecision", "CompletionMonitor"]
