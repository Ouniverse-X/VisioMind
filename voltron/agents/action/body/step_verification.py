"""High-level step-verification helpers owned by the Action agent body."""

from __future__ import annotations
from typing import Any

from voltron.agents.action.models import ActionStepVerification
from voltron.agents.vision.body import VLMCompletionEvaluator
from voltron.shared.contracts import VisionAdapter
from voltron.shared.context import ExecutionContext, Subtask
from voltron.shared.models import CompletionCriterion, CompletionEvaluationContext


class VisionBackedActionStepVerifier:
    """Use the shared VLM backend to verify Action internal-step completion."""

    def __init__(
        self,
        vision: VisionAdapter | None = None,
        *,
        completion_evaluator: Any | None = None,
    ) -> None:
        if completion_evaluator is None and vision is None:
            raise ValueError("VisionBackedActionStepVerifier requires vision or completion_evaluator")
        self.vision = vision
        self.completion_evaluator = completion_evaluator or VLMCompletionEvaluator(vision=vision)

    def verify_step(
        self,
        *,
        parent_subtask: Subtask,
        internal_subtask: Subtask,
        step_payload,
        context: ExecutionContext,
        executed_control_steps: int,
        verification_index: int,
    ) -> ActionStepVerification:
        images = internal_subtask.parameters.get("images")
        if not isinstance(images, list) or not images:
            return ActionStepVerification(
                step_completed=False,
                reason="verification skipped because images are unavailable",
                indeterminate=True,
                metadata={"verification_index": verification_index},
            )

        evaluation_context = self._evaluation_context(
            parent_subtask=parent_subtask,
            internal_subtask=internal_subtask,
            step_payload=step_payload,
            context=context,
            executed_control_steps=executed_control_steps,
            verification_index=verification_index,
        )
        verdict = self.completion_evaluator.evaluate(evaluation_context)
        raw_text = str(verdict.evidence.get("raw_text") or verdict.reason or "")
        return ActionStepVerification(
            step_completed=bool(verdict.completed),
            confidence=max(0.0, min(1.0, float(verdict.confidence))),
            reason=verdict.reason,
            should_replan=bool(verdict.should_replan),
            observed_success_cues=_observed_success_cues(verdict.evidence),
            scene_report=dict(verdict.evidence.get("scene_report") or {}),
            raw_text=raw_text,
            metadata={
                "completion_verdict": verdict.to_dict(),
                "completion_context": self._completion_context(context),
            },
        )

    @staticmethod
    def _evaluation_context(
        *,
        parent_subtask: Subtask,
        internal_subtask: Subtask,
        step_payload: Any,
        context: ExecutionContext,
        executed_control_steps: int,
        verification_index: int,
    ) -> CompletionEvaluationContext:
        task_context = VisionBackedActionStepVerifier._task_context(context)
        completion_context = VisionBackedActionStepVerifier._completion_context(context)
        criteria = VisionBackedActionStepVerifier._completion_criteria(task_context, internal_subtask)
        criteria.extend(
            VisionBackedActionStepVerifier._internal_step_criteria(
                internal_subtask=internal_subtask,
                step_payload=step_payload,
            )
        )
        return CompletionEvaluationContext(
            task_description=context.task_request.description,
            scope="action_internal_step",
            scope_id=str(getattr(step_payload, "internal_step_id", "") or internal_subtask.subtask_id),
            completion_criteria=criteria,
            confirmed_text_plan=dict((task_context.get("interactive_planning") or {}).get("text_plan") or {}),
            current_subtask={
                "subtask_id": parent_subtask.subtask_id,
                "agent": parent_subtask.agent.value,
                "action": parent_subtask.action,
                "target": dict(parent_subtask.target),
                "instruction": str(parent_subtask.parameters.get("instruction") or parent_subtask.action),
            },
            current_internal_step={
                "internal_step_id": str(getattr(step_payload, "internal_step_id", "") or internal_subtask.subtask_id),
                "name": str(getattr(step_payload, "name", "") or ""),
                "action": internal_subtask.action,
                "target": dict(internal_subtask.target),
                "instruction": str(internal_subtask.parameters.get("instruction") or internal_subtask.action),
                "success_cues": list(internal_subtask.parameters.get("success_cues") or []),
            },
            runtime_feedback={
                "extras": {
                    "images_b64": list(internal_subtask.parameters.get("images") or []),
                    "image_view_order": list(internal_subtask.parameters.get("image_view_order") or []),
                }
            },
            task_context=task_context,
            memory_evidence=dict(task_context.get("memory_evidence") or {}),
            action_stability={
                "executed_control_steps": executed_control_steps,
                "verification_index": verification_index,
                "completion_context": completion_context,
            },
        )

    @staticmethod
    def _task_context(context: ExecutionContext) -> dict[str, Any]:
        task_context = context.runtime_state.get("task_context")
        return dict(task_context) if isinstance(task_context, dict) else {}

    @staticmethod
    def _completion_criteria(task_context: dict[str, Any], internal_subtask: Subtask) -> list[CompletionCriterion]:
        raw_items: list[Any] = []
        raw_items.extend(task_context.get("completion_criteria") or [])
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
            payload.setdefault("scope", payload.get("scope") or "task")
            payload.setdefault("subtask_id", payload.get("subtask_id") or internal_subtask.subtask_id)
            try:
                criteria.append(CompletionCriterion.from_value(payload))
            except (TypeError, ValueError):
                continue
        return criteria

    @staticmethod
    def _internal_step_criteria(*, internal_subtask: Subtask, step_payload: Any) -> list[CompletionCriterion]:
        instruction = str(internal_subtask.parameters.get("instruction") or internal_subtask.action)
        success_cues = [str(item) for item in internal_subtask.parameters.get("success_cues") or []]
        return [
            CompletionCriterion(
                criterion_id=f"crit_{getattr(step_payload, 'internal_step_id', internal_subtask.subtask_id)}",
                scope="action_internal_step",
                subtask_id=internal_subtask.subtask_id,
                internal_step_id=str(getattr(step_payload, "internal_step_id", "") or internal_subtask.subtask_id),
                description=instruction,
                positive_evidence=success_cues,
                required_observations=success_cues,
                success_verifier="vision_completion_evaluator",
            )
        ]

    @staticmethod
    def _completion_context(context: ExecutionContext) -> dict[str, Any]:
        task_context = context.runtime_state.get("task_context")
        if not isinstance(task_context, dict):
            return {}
        criteria = []
        explicit = task_context.get("completion_criteria")
        if isinstance(explicit, list):
            criteria.extend(item for item in explicit if isinstance(item, dict))
        interactive = task_context.get("interactive_planning")
        if isinstance(interactive, dict):
            text_plan = interactive.get("text_plan")
            if isinstance(text_plan, dict):
                criteria.extend(
                    item for item in text_plan.get("success_criteria") or [] if isinstance(item, dict)
                )
        completion_monitor = task_context.get("completion_monitor")
        latest_verdict = {}
        if isinstance(completion_monitor, dict) and isinstance(completion_monitor.get("latest_verdict"), dict):
            latest_verdict = dict(completion_monitor["latest_verdict"])
        payload: dict[str, Any] = {}
        if criteria:
            payload["criteria"] = [_compact_criterion(item) for item in criteria[:5]]
        if latest_verdict:
            payload["latest_verdict"] = {
                key: latest_verdict.get(key)
                for key in ("completed", "confidence", "reason", "source", "missing_evidence")
                if latest_verdict.get(key) not in (None, "", [], {})
            }
        return payload


def _compact_criterion(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item.get(key)
        for key in (
            "criterion_id",
            "scope",
            "description",
            "positive_evidence",
            "required_observations",
            "source",
            "user_confirmed",
        )
        if item.get(key) not in (None, "", [], {})
    }


__all__ = ["VisionBackedActionStepVerifier"]


def _observed_success_cues(evidence: dict[str, Any]) -> list[str]:
    cues = evidence.get("observed_success_cues")
    if not isinstance(cues, list):
        return []
    return [str(item).strip() for item in cues if str(item).strip()]
