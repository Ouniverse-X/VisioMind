"""Action agent for dual-arm manipulation execution."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

from voltron.agents.action.models import (
    ActionExecutionPlan,
    ActionStepVerification,
    VLADeliberation,
    VLATargetRefinement,
)
from voltron.agents.action.body.skill_selection import HeuristicActionSkillSelector
from voltron.shared.enums import AgentStatus
from voltron.shared.context import ExecutionContext, LocalSkillSelection, Subtask
from voltron.shared.results import AgentResult
from voltron.agents.action.skills import ActionSkillRegistry, StructuredTargetRefiner
from voltron.agents.action.contracts import (
    ActionStepVerifier,
    ActionTaskPlanner,
    ActionTaskPlanningSkill,
    LocalSkillSelector,
    VLADeliberator,
    VLATargetRefiner,
)
from voltron.agents.action.tools.action_projection import ActionProjection
from voltron.agents.action.tools import decision_flow, execution_runtime
from voltron.shared.contracts import MemoryAdapter, PolicyAdapter


class ActionAgent:
    """Execute manipulation subtasks through local skill selection."""

    def __init__(
        self,
        memory: MemoryAdapter,
        policy: PolicyAdapter,
        projector: ActionProjection,
        selector: LocalSkillSelector | None = None,
        skill_registry: ActionSkillRegistry | None = None,
        deliberator: VLADeliberator | None = None,
        target_refiner: VLATargetRefiner | None = None,
        task_planning_skill: ActionTaskPlanningSkill | None = None,
        task_planner: ActionTaskPlanner | None = None,
        step_verifier: ActionStepVerifier | None = None,
        verify_every_control_steps: int = 400,
        verify_after_first_success: bool = False,
        verification_positive_streak: int = 1,
        max_verification_failures_before_replan: int = 3,
        max_unverified_internal_step_control_steps: int = 1,
        require_verified_internal_step_completion: bool = True,
    ) -> None:
        self.memory = memory
        self.policy = policy
        self.projector = projector
        self.selector = selector or HeuristicActionSkillSelector()
        self.skill_registry = skill_registry or ActionSkillRegistry.build_default(
            memory=memory,
            policy=policy,
            projector=projector,
        )
        self.deliberator = deliberator
        self.target_refiner = target_refiner or StructuredTargetRefiner(memory)
        self.task_planning_skill = task_planning_skill
        self.task_planner = task_planner
        self.step_verifier = step_verifier
        self.verify_every_control_steps = max(1, verify_every_control_steps)
        self.verify_after_first_success = verify_after_first_success
        self.verification_positive_streak = max(1, verification_positive_streak)
        self.max_verification_failures_before_replan = max(1, max_verification_failures_before_replan)
        self.max_unverified_internal_step_control_steps = max(1, int(max_unverified_internal_step_control_steps))
        self.require_verified_internal_step_completion = bool(require_verified_internal_step_completion)
        self._deliberation_cache: dict[str, VLADeliberation] = {}
        self._target_refinement_cache: dict[str, VLATargetRefinement] = {}
        self._selection_cache: dict[str, LocalSkillSelection] = {}

    def execute(self, subtask: Subtask, context: ExecutionContext) -> AgentResult:
        if self.task_planning_skill is None or self.task_planner is None:
            return self._execute_single_subtask(subtask=subtask, context=context)

        sessions = self._execution_sessions(context)
        session = sessions.get(subtask.subtask_id)
        plan_created = False

        if session is None:
            plan_created = True
            planning_subtask = self._subtask_with_applicable_memory_skills(subtask, context)
            try:
                planning_prompt = self.task_planning_skill.build_plan_prompt(subtask=planning_subtask, context=context)
                planning_response = self.task_planner.generate_plan(
                    subtask=planning_subtask,
                    context=context,
                    prompt=planning_prompt,
                )
                execution_plan = self.task_planning_skill.parse_plan_response(
                    content=planning_response,
                    subtask=planning_subtask,
                    context=context,
                )
            except Exception as exc:
                return self._decorate_execution_result(
                    parent_subtask=subtask,
                    base_result=self._execute_single_subtask(subtask=subtask, context=context),
                    execution_mode="legacy_fallback",
                    execution_plan=ActionExecutionPlan(
                        parent_subtask_id=subtask.subtask_id,
                        goal_summary=str(subtask.parameters.get("instruction") or subtask.action),
                        source="legacy_fallback",
                        metadata={"planning_error": str(exc)},
                    ),
                    internal_steps=[],
                    active_internal_step=None,
                    replan_history=[],
                    execution_progress={
                        "plan_created": False,
                        "current_step_index": 0,
                        "total_steps": 0,
                        "completed_step_ids": [],
                        "pending_step_ids": [],
                        "plan_completed": True,
                    },
                )
            session = self._create_execution_session(execution_plan)
            sessions[subtask.subtask_id] = session
        else:
            execution_plan = session["execution_plan"]

        return self._execute_plan(
            subtask=subtask,
            context=context,
            execution_plan=execution_plan,
            session=session,
            plan_created=plan_created,
        )

    def run_episode(self, *, subtask: Subtask, context: ExecutionContext, runtime: Any) -> AgentResult:
        """Run a complete action subtask episode using the agent's internal plan state."""

        static_parameters = dict(subtask.parameters)
        last_result: AgentResult | None = None

        for control_step in range(1, int(runtime.max_control_steps) + 1):
            runtime.prepare_control_step(
                subtask=subtask,
                context=context,
                static_parameters=static_parameters,
                control_step=control_step,
            )
            result = self.execute(subtask, context)
            last_result = runtime.publish_agent_result(
                subtask=subtask,
                context=context,
                result=result,
                control_step=control_step,
            )

            if result.status == AgentStatus.FAILURE:
                if hasattr(runtime, "record_agent_failure"):
                    runtime.record_agent_failure(
                        subtask=subtask,
                        context=context,
                        result=result,
                        failure_reason=result.error_code or "AGENT_FAILURE",
                    )
                return result

            step_outcome = runtime.apply_agent_result(subtask=subtask, result=result, context=context)
            if getattr(step_outcome, "feedback", None):
                runtime.update_feedback(
                    subtask=subtask,
                    context=context,
                    result=result,
                    control_step=control_step,
                    feedback=step_outcome.feedback,
                )
            if not bool(getattr(step_outcome, "done", False)):
                continue

            if getattr(step_outcome, "success", None) is False:
                return runtime.environment_failure_result(
                    subtask=subtask,
                    context=context,
                    result=result,
                    control_step=control_step,
                    feedback=getattr(step_outcome, "feedback", {}),
                    failure_reason=getattr(step_outcome, "failure_reason", None),
                )

            if hasattr(runtime, "record_agent_success"):
                runtime.record_agent_success(subtask=subtask, context=context, result=result)
            return result

        if last_result is not None and hasattr(runtime, "record_agent_failure"):
            runtime.record_agent_failure(
                subtask=subtask,
                context=context,
                result=last_result,
                failure_reason="SUBTASK_TIMEOUT",
            )
        return runtime.timeout_result(subtask=subtask)

    @staticmethod
    def _execution_sessions(context: ExecutionContext) -> dict[str, dict[str, Any]]:
        return execution_runtime.execution_sessions(context)

    @staticmethod
    def _create_execution_session(execution_plan: ActionExecutionPlan) -> dict[str, Any]:
        return execution_runtime.create_execution_session(execution_plan)

    @staticmethod
    def _serialize_internal_step(step_payload: Any, *, selected_skill_id: str | None = None) -> dict[str, Any]:
        return execution_runtime.serialize_internal_step(step_payload, selected_skill_id=selected_skill_id)

    def _subtask_with_applicable_memory_skills(
        self,
        subtask: Subtask,
        context: ExecutionContext,
    ) -> Subtask:
        finder = getattr(self.memory, "find_applicable_skills", None)
        if not callable(finder):
            return subtask

        current_state = {
            "task_description": context.task_request.description,
            "task_type": context.task_request.task_type.value,
            "subtask_id": subtask.subtask_id,
            "action": subtask.action,
            "target": dict(subtask.target),
            "parameters": dict(subtask.parameters),
            "context": dict(subtask.context),
        }
        try:
            skills = finder(current_state=current_state, top_k=5)
        except Exception:
            return subtask

        compact_skills = self._compact_applicable_memory_skills(skills)
        if not compact_skills.get("results"):
            return subtask

        subtask_context = dict(subtask.context)
        subtask_context["applicable_skills"] = compact_skills
        return replace(subtask, context=subtask_context)

    @staticmethod
    def _compact_applicable_memory_skills(skills: Any) -> dict[str, Any]:
        if isinstance(skills, dict):
            raw_results = skills.get("results", [])
            metadata = skills.get("metadata", {})
            query_type = skills.get("query_type", "applicable_skills")
        elif isinstance(skills, list):
            raw_results = skills
            metadata = {}
            query_type = "applicable_skills"
        else:
            raw_results = []
            metadata = {}
            query_type = "applicable_skills"

        compact_results = []
        for item in raw_results[:5] if isinstance(raw_results, list) else []:
            if not isinstance(item, dict):
                continue
            compact = {
                key: item[key]
                for key in (
                    "skill_id",
                    "skill_name",
                    "name",
                    "description",
                    "success_rate",
                    "execution_count",
                    "confidence",
                    "source_episodes",
                    "parameters",
                    "preconditions",
                    "postconditions",
                )
                if key in item
            }
            action_template = item.get("action_template")
            if isinstance(action_template, list):
                compact["action_template"] = [
                    {
                        key: step[key]
                        for key in ("action_type", "target_param", "parameters")
                        if isinstance(step, dict) and key in step
                    }
                    for step in action_template[:4]
                    if isinstance(step, dict)
                ]
            compact_results.append(compact)

        return {
            "query_type": query_type,
            "results": compact_results,
            "metadata": dict(metadata) if isinstance(metadata, dict) else {},
        }

    def _execute_single_subtask(self, *, subtask: Subtask, context: ExecutionContext) -> AgentResult:
        execution_subtask = subtask
        deliberation = VLADeliberation()
        target_refinement = VLATargetRefinement()
        if self.deliberator is not None:
            deliberation = decision_flow.deliberate_with_cache(
                deliberator=self.deliberator,
                cache=self._deliberation_cache,
                subtask=subtask,
                context=context,
            )
            if deliberation.use_tool and deliberation.tool_name == "refine_target":
                target_refinement = decision_flow.refine_target_with_cache(
                    target_refiner=self.target_refiner,
                    cache=self._target_refinement_cache,
                    subtask=subtask,
                    context=context,
                )
                execution_subtask = decision_flow.apply_target_refinement(
                    subtask=subtask,
                    deliberation=deliberation,
                    target_refinement=target_refinement,
                )

        available_skill_ids = self.skill_registry.available_skill_ids()
        cache_key = decision_flow.build_selection_cache_key(
            subtask=execution_subtask,
            context=context,
            available_skill_ids=available_skill_ids,
        )
        selection = self._selection_cache.get(cache_key)
        if selection is None:
            selection = self.selector.select_skill(
                subtask=execution_subtask,
                context=context,
                available_skill_ids=available_skill_ids,
            )
            self._selection_cache[cache_key] = selection
        skill = self.skill_registry.resolve(subtask=execution_subtask, context=context, selection=selection)
        if skill is None:
            return AgentResult(
                subtask_id=execution_subtask.subtask_id,
                status=AgentStatus.FAILURE,
                error_code="VLA_SKILL_NOT_FOUND",
                result={"message": "No VLA skill could handle the subtask"},
            )
        result = skill.execute(
            subtask=execution_subtask,
            context=context,
            selection=decision_flow.normalize_selection(selection, skill),
        )
        return decision_flow.decorate_skill_result(
            result=result,
            deliberation=deliberation,
            target_refinement=target_refinement,
        )

    def _execute_plan(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        execution_plan: ActionExecutionPlan,
        session: dict[str, Any],
        plan_created: bool,
        allow_local_replan: bool = True,
        skip_step_verification_once: bool = False,
    ) -> AgentResult:
        completed_steps = list(session.get("completed_steps", []))
        next_step_index = int(session.get("next_step_index", 0))
        session, execution_plan, completed_steps, next_step_index = self._apply_pending_verification_outcome(
            subtask=subtask,
            context=context,
            session=session,
            execution_plan=execution_plan,
            completed_steps=completed_steps,
            next_step_index=next_step_index,
            allow_local_replan=allow_local_replan,
        )
        total_steps = len(execution_plan.steps)
        completed_steps, next_step_index = self._advance_verified_step_if_ready(
            session=session,
            execution_plan=execution_plan,
            completed_steps=completed_steps,
            next_step_index=next_step_index,
        )
        session["completed_steps"] = completed_steps
        session["next_step_index"] = next_step_index
        step_advanced_this_call = bool(session.pop("step_advanced_this_call", False))
        applied_step_verification = self._serialize_step_verification(session.pop("applied_step_verification", None))

        if total_steps == 0:
            return self._decorate_execution_result(
                parent_subtask=subtask,
                base_result=AgentResult(
                    subtask_id=subtask.subtask_id,
                    status=AgentStatus.SUCCESS,
                    result={"message": "VLA execution plan completed with no internal steps"},
                ),
                execution_mode="planned",
                execution_plan=execution_plan,
                internal_steps=completed_steps,
                active_internal_step=None,
                replan_history=list(session.get("replan_history", [])),
                execution_progress={
                    "plan_created": plan_created,
                    "current_step_index": 0,
                    "total_steps": 0,
                    "completed_step_ids": [],
                    "pending_step_ids": [],
                    "plan_completed": True,
                    },
                )

        if next_step_index >= total_steps:
            last_success_record = session.get("active_step_last_success_record")
            active_internal_step = None
            if isinstance(last_success_record, dict):
                active_internal_step = {
                    "internal_step_id": last_success_record.get("internal_step_id"),
                    "name": last_success_record.get("name"),
                    "instruction": last_success_record.get("instruction"),
                    "action": last_success_record.get("action"),
                    "target": dict(last_success_record.get("target", {})),
                    "step_index": total_steps,
                    "total_steps": total_steps,
                }
                active_internal_step = self._decorate_active_internal_step(
                    active_internal_step=active_internal_step,
                    session=session,
                    step_verification=applied_step_verification
                    or self._serialize_step_verification(session.get("last_step_verification")),
                )
            self._execution_sessions(context).pop(subtask.subtask_id, None)
            return self._decorate_execution_result(
                parent_subtask=subtask,
                base_result=AgentResult(
                    subtask_id=subtask.subtask_id,
                    status=AgentStatus.SUCCESS,
                    result={"message": "VLA execution plan completed"},
                ),
                execution_mode="planned",
                execution_plan=execution_plan,
                internal_steps=completed_steps,
                active_internal_step=active_internal_step,
                replan_history=list(session.get("replan_history", [])),
                execution_progress={
                    "plan_created": plan_created,
                    "current_step_index": total_steps,
                    "total_steps": total_steps,
                    "completed_step_ids": [item["internal_step_id"] for item in completed_steps],
                    "pending_step_ids": [],
                    "plan_completed": True,
                },
            )

        step = execution_plan.steps[next_step_index]
        self._ensure_active_step_tracking(session=session, active_step_id=step.internal_step_id)
        visible_completed_steps = list(completed_steps)
        internal_subtask = self._build_internal_subtask(parent_subtask=subtask, step_payload=step)
        step_result = self._execute_single_subtask(subtask=internal_subtask, context=context)

        selected_skill_id = None
        skill_selection = step_result.runtime_artifacts.get("skill_selection")
        if isinstance(skill_selection, dict):
            selected_skill_id = str(skill_selection.get("skill_id") or "").strip() or None

        active_internal_step = self._serialize_internal_step(step, selected_skill_id=selected_skill_id)
        active_internal_step["step_index"] = next_step_index + 1
        active_internal_step["total_steps"] = total_steps

        step_record = {
            "internal_step_id": step.internal_step_id,
            "name": step.name,
            "instruction": step.instruction,
            "action": step.action,
            "target": dict(step.target),
            "status": step_result.status.value,
            "result": dict(step_result.result),
            "error_code": step_result.error_code,
            "runtime_artifacts": dict(step_result.runtime_artifacts),
        }

        step_verification: dict[str, Any] | None = None
        if step_result.status != AgentStatus.SUCCESS and allow_local_replan:
            session["active_step_last_success_record"] = None
            session["last_step_verification"] = None
            replan_decision = self.task_planning_skill.replan(  # type: ignore[union-attr]
                subtask=subtask,
                context=context,
                active_step_id=step.internal_step_id,
                reason=step_result.error_code or "VLA_INTERNAL_STEP_FAILED",
            )
            if replan_decision.should_replan and replan_decision.replacement_steps:
                replacement_plan = self._replace_pending_steps(
                    execution_plan=execution_plan,
                    start_index=next_step_index,
                    replacement_steps=replan_decision.replacement_steps,
                )
                session["execution_plan"] = replacement_plan
                session["replan_history"] = list(session.get("replan_history", [])) + [
                    {
                        "active_step_id": step.internal_step_id,
                        "reason": replan_decision.reason or step_result.error_code or "replan",
                        "replacement_step_ids": [
                            replacement_step.internal_step_id for replacement_step in replan_decision.replacement_steps
                        ],
                        "metadata": dict(replan_decision.metadata),
                    }
                ]
                session["next_step_index"] = next_step_index
                replacement_active_step_id = (
                    replan_decision.replacement_steps[0].internal_step_id if replan_decision.replacement_steps else None
                )
                execution_runtime.reset_active_step_tracking(session, replacement_active_step_id)
                return self._execute_plan(
                    subtask=subtask,
                    context=context,
                    execution_plan=replacement_plan,
                    session=session,
                    plan_created=plan_created,
                    allow_local_replan=False,
                )

        if step_result.status == AgentStatus.SUCCESS:
            session["active_step_control_steps"] = int(session.get("active_step_control_steps", 0)) + 1
            session["active_step_last_success_record"] = step_record
            if self.step_verifier is None:
                active_step_control_steps = int(session.get("active_step_control_steps", 0))
                if active_step_control_steps >= self.max_unverified_internal_step_control_steps:
                    completed_steps.append(step_record)
                    session["completed_steps"] = completed_steps
                    session["next_step_index"] = next_step_index + 1
                    if next_step_index + 1 < total_steps:
                        execution_runtime.reset_active_step_tracking(
                            session,
                            execution_plan.steps[next_step_index + 1].internal_step_id,
                        )
                        active_internal_step = self._decorate_active_internal_step(
                            active_internal_step=active_internal_step,
                            session=session,
                            step_verification=None,
                        )
                    else:
                        session["last_step_verification"] = None
            else:
                verification_result = self._verify_step_if_due(
                    parent_subtask=subtask,
                    internal_subtask=internal_subtask,
                    step_payload=step,
                    context=context,
                    session=session,
                    skip_step_verification=skip_step_verification_once or step_advanced_this_call,
                )
                if verification_result is not None:
                    step_verification = asdict(verification_result)
                    if verification_result.step_completed:
                        if next_step_index + 1 >= total_steps and total_steps > 1:
                            completed_steps.append(step_record)
                            session["completed_steps"] = completed_steps
                            session["next_step_index"] = next_step_index + 1
                        elif next_step_index + 1 < total_steps and not plan_created:
                            completed_steps.append(step_record)
                            session["completed_steps"] = completed_steps
                            session["next_step_index"] = next_step_index + 1
                            session["last_step_verification"] = None
                            execution_runtime.reset_active_step_tracking(
                                session,
                                execution_plan.steps[next_step_index + 1].internal_step_id,
                            )
                            session["step_advanced_this_call"] = True
                            return self._execute_plan(
                                subtask=subtask,
                                context=context,
                                execution_plan=execution_plan,
                                session=session,
                                plan_created=plan_created,
                                allow_local_replan=allow_local_replan,
                                skip_step_verification_once=True,
                            )
                        else:
                            session["last_step_verification"] = step_verification
                elif not self.require_verified_internal_step_completion:
                    active_step_control_steps = int(session.get("active_step_control_steps", 0))
                    if active_step_control_steps >= self.max_unverified_internal_step_control_steps:
                        completed_steps.append(step_record)
                        session["completed_steps"] = completed_steps
                        session["next_step_index"] = next_step_index + 1
                        if next_step_index + 1 < total_steps:
                            execution_runtime.reset_active_step_tracking(
                                session,
                                execution_plan.steps[next_step_index + 1].internal_step_id,
                            )
                        else:
                            session["last_step_verification"] = None

        active_internal_step = self._decorate_active_internal_step(
            active_internal_step=active_internal_step,
            session=session,
            step_verification=step_verification or applied_step_verification,
        )
        plan_completed = step_result.status == AgentStatus.SUCCESS and session["next_step_index"] >= total_steps
        if plan_completed or step_result.status != AgentStatus.SUCCESS:
            self._execution_sessions(context).pop(subtask.subtask_id, None)

        completed_step_ids = [item["internal_step_id"] for item in completed_steps]
        pending_step_ids = [
            step_payload.internal_step_id
            for step_payload in execution_plan.steps[session["next_step_index"] :]
            if step_payload.internal_step_id not in completed_step_ids
        ]

        return self._decorate_execution_result(
            parent_subtask=subtask,
            base_result=step_result,
            execution_mode="planned",
            execution_plan=execution_plan,
            internal_steps=(
                completed_steps
                if plan_completed
                else visible_completed_steps
                if step_result.status == AgentStatus.SUCCESS
                else visible_completed_steps + [step_record]
            ),
            active_internal_step=active_internal_step,
            replan_history=list(session.get("replan_history", [])),
            execution_progress={
                "plan_created": plan_created,
                "current_step_index": next_step_index + 1,
                "total_steps": total_steps,
                "completed_step_ids": completed_step_ids,
                "pending_step_ids": pending_step_ids,
                "plan_completed": plan_completed,
            },
            step_verification=step_verification or applied_step_verification,
        )

    def _decorate_execution_result(
        self,
        *,
        parent_subtask: Subtask,
        base_result: AgentResult,
        execution_mode: str,
        execution_plan: ActionExecutionPlan,
        internal_steps: list[dict[str, Any]],
        active_internal_step: dict[str, Any] | None,
        replan_history: list[dict[str, Any]],
        execution_progress: dict[str, Any],
        step_verification: dict[str, Any] | None = None,
        latency_ms: int | None = None,
    ) -> AgentResult:
        return execution_runtime.decorate_execution_result(
            parent_subtask=parent_subtask,
            base_result=base_result,
            execution_mode=execution_mode,
            execution_plan=execution_plan,
            internal_steps=internal_steps,
            active_internal_step=active_internal_step,
            replan_history=replan_history,
            execution_progress=execution_progress,
            step_verification=step_verification,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _build_internal_subtask(*, parent_subtask: Subtask, step_payload: Any) -> Subtask:
        return execution_runtime.build_internal_subtask(parent_subtask=parent_subtask, step_payload=step_payload)

    @staticmethod
    def _replace_pending_steps(
        *,
        execution_plan: ActionExecutionPlan,
        start_index: int,
        replacement_steps: list[Any],
    ) -> ActionExecutionPlan:
        return execution_runtime.replace_pending_steps(
            execution_plan=execution_plan,
            start_index=start_index,
            replacement_steps=replacement_steps,
        )

    @staticmethod
    def _ensure_active_step_tracking(*, session: dict[str, Any], active_step_id: str) -> None:
        if session.get("active_step_id") != active_step_id:
            execution_runtime.reset_active_step_tracking(session, active_step_id)

    @staticmethod
    def _serialize_step_verification(verification: ActionStepVerification | dict[str, Any] | None) -> dict[str, Any] | None:
        if verification is None:
            return None
        if isinstance(verification, dict):
            return dict(verification)
        return asdict(verification)

    def _advance_verified_step_if_ready(
        self,
        *,
        session: dict[str, Any],
        execution_plan: ActionExecutionPlan,
        completed_steps: list[dict[str, Any]],
        next_step_index: int,
    ) -> tuple[list[dict[str, Any]], int]:
        if self.step_verifier is None:
            return completed_steps, next_step_index
        if next_step_index >= len(execution_plan.steps):
            return completed_steps, next_step_index
        last_step_verification = session.get("last_step_verification")
        last_success_record = session.get("active_step_last_success_record")
        active_step_id = session.get("active_step_id")
        if not isinstance(last_step_verification, dict) or not last_step_verification.get("step_completed"):
            return completed_steps, next_step_index
        if not isinstance(last_success_record, dict):
            return completed_steps, next_step_index
        if last_success_record.get("internal_step_id") != active_step_id:
            return completed_steps, next_step_index
        session["applied_step_verification"] = dict(last_step_verification)
        completed_steps = completed_steps + [dict(last_success_record)]
        next_step_index += 1
        session["completed_steps"] = completed_steps
        session["next_step_index"] = next_step_index
        if next_step_index < len(execution_plan.steps):
            execution_runtime.reset_active_step_tracking(session, execution_plan.steps[next_step_index].internal_step_id)
            session["step_advanced_this_call"] = True
        else:
            session["active_step_id"] = active_step_id
        return completed_steps, next_step_index

    def _verify_step_if_due(
        self,
        *,
        parent_subtask: Subtask,
        internal_subtask: Subtask,
        step_payload: Any,
        context: ExecutionContext,
        session: dict[str, Any],
        skip_step_verification: bool,
    ) -> ActionStepVerification | None:
        if self.step_verifier is None:
            return None
        if skip_step_verification:
            return None
        executed_control_steps = int(session.get("active_step_control_steps", 0))
        verification_checks = int(session.get("active_step_verification_checks", 0))
        if not self._should_verify_step(
            executed_control_steps=executed_control_steps,
            verification_checks=verification_checks,
            last_step_verification=session.get("last_step_verification"),
        ):
            return None
        self._refresh_task_context_from_memory(context)
        verification_result = self.step_verifier.verify_step(
            parent_subtask=parent_subtask,
            internal_subtask=internal_subtask,
            step_payload=step_payload,
            context=context,
            executed_control_steps=executed_control_steps,
            verification_index=verification_checks + 1,
        )
        session["active_step_verification_checks"] = verification_checks + 1
        if verification_result.step_completed:
            session["active_step_positive_streak"] = int(session.get("active_step_positive_streak", 0)) + 1
            session["active_step_negative_streak"] = 0
        else:
            session["active_step_positive_streak"] = 0
            session["active_step_negative_streak"] = int(session.get("active_step_negative_streak", 0)) + 1
        session["last_step_verification"] = asdict(verification_result)
        return verification_result

    def _refresh_task_context_from_memory(self, context: ExecutionContext) -> None:
        getter = getattr(self.memory, "get_task_context", None)
        if not callable(getter):
            return
        try:
            memory_context = getter()
        except Exception:
            return
        if not isinstance(memory_context, dict):
            return
        task_context = context.runtime_state.setdefault("task_context", {})
        if isinstance(task_context, dict):
            _deep_merge(task_context, memory_context)

    def _should_verify_step(
        self,
        *,
        executed_control_steps: int,
        verification_checks: int,
        last_step_verification: Any,
    ) -> bool:
        if self.verify_after_first_success and verification_checks == 0 and executed_control_steps >= 1:
            return True
        if isinstance(last_step_verification, dict) and not last_step_verification.get("step_completed"):
            return True
        return executed_control_steps > 0 and executed_control_steps % self.verify_every_control_steps == 0

    def _should_replan_after_verification(
        self,
        *,
        verification: ActionStepVerification,
        session: dict[str, Any],
    ) -> bool:
        if verification.should_replan:
            return True
        if verification.indeterminate:
            return False
        return int(session.get("active_step_negative_streak", 0)) >= self.max_verification_failures_before_replan

    def _decorate_active_internal_step(
        self,
        *,
        active_internal_step: dict[str, Any],
        session: dict[str, Any],
        step_verification: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload = dict(active_internal_step)
        payload["executed_control_steps"] = int(session.get("active_step_control_steps", 0))
        payload["verification_checks"] = int(session.get("active_step_verification_checks", 0))
        payload["verification_positive_streak"] = int(session.get("active_step_positive_streak", 0))
        payload["verification_negative_streak"] = int(session.get("active_step_negative_streak", 0))
        payload["last_step_verification"] = self._serialize_step_verification(
            step_verification or session.get("last_step_verification")
        )
        return payload

    def _apply_pending_verification_outcome(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        session: dict[str, Any],
        execution_plan: ActionExecutionPlan,
        completed_steps: list[dict[str, Any]],
        next_step_index: int,
        allow_local_replan: bool,
    ) -> tuple[dict[str, Any], ActionExecutionPlan, list[dict[str, Any]], int]:
        if self.step_verifier is None or self.task_planning_skill is None:
            return session, execution_plan, completed_steps, next_step_index
        last_step_verification = session.get("last_step_verification")
        if not isinstance(last_step_verification, dict) or last_step_verification.get("step_completed"):
            return session, execution_plan, completed_steps, next_step_index
        verification = ActionStepVerification(**last_step_verification)
        if not allow_local_replan or not self._should_replan_after_verification(verification=verification, session=session):
            return session, execution_plan, completed_steps, next_step_index
        if next_step_index >= len(execution_plan.steps):
            return session, execution_plan, completed_steps, next_step_index
        active_step_id = execution_plan.steps[next_step_index].internal_step_id
        replan_decision = self.task_planning_skill.replan(
            subtask=subtask,
            context=context,
            active_step_id=active_step_id,
            reason=verification.reason or "ACTION_STEP_VERIFICATION_FAILED",
        )
        if not replan_decision.should_replan or not replan_decision.replacement_steps:
            return session, execution_plan, completed_steps, next_step_index
        replacement_plan = self._replace_pending_steps(
            execution_plan=execution_plan,
            start_index=next_step_index,
            replacement_steps=replan_decision.replacement_steps,
        )
        session["execution_plan"] = replacement_plan
        session["replan_history"] = list(session.get("replan_history", [])) + [
            {
                "active_step_id": active_step_id,
                "reason": replan_decision.reason or verification.reason or "verification_replan",
                "replacement_step_ids": [
                    replacement_step.internal_step_id for replacement_step in replan_decision.replacement_steps
                ],
                "metadata": dict(replan_decision.metadata),
            }
        ]
        session["next_step_index"] = next_step_index
        execution_runtime.reset_active_step_tracking(session, replan_decision.replacement_steps[0].internal_step_id)
        session["step_advanced_this_call"] = True
        return session, replacement_plan, completed_steps, next_step_index


def _deep_merge(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
