"""Brain coordinator agent.

This module only handles planning lifecycle and memory orchestration.
The planner itself is injected via `TaskPlanner` to keep model/runtime pluggable.
"""

from __future__ import annotations

from typing import Any

from . import plan_flow as brain_plan_flow
from . import planning_context as brain_planning_context
from .interactive_planning import BrainInteractivePlanningController
from .planning_loop import BrainPlanningLoop
from .runtime_interaction_control import RuntimeInteractionControlPolicy
from voltron.agents.brain.contracts import BrainPlanningSession, PlanConfirmation, TaskPlanner, UserAnswer
from voltron.agents.brain.skills.planning.interactive_alignment import (
    action_contract_steps,
    align_refined_plan,
)
from voltron.agents.brain.tools.cron import CronTool
from voltron.agents.brain.tools import interaction_targeting, navigation_runtime
from voltron.agents.brain.tools.web_search import WebSearchTool
from voltron.shared.context import ExecutionContext, Plan, Subtask, TaskRequest
from voltron.shared.contracts import AgentCapability, MemoryAdapter, ToolInvocation, serialize_agent_capabilities
from voltron.shared.registries import ToolCatalog
from voltron.shared.results import ToolResult


class BrainAgent:
    """Top-level coordinator for plan/replan/finalize flow."""

    def __init__(
        self,
        memory: MemoryAdapter,
        planner: TaskPlanner,
        tools: ToolCatalog | None = None,
        agent_capabilities: list[AgentCapability] | None = None,
    ):
        self.memory = memory
        self.planner = planner
        self.tools = tools or self._build_default_tools()
        self.agent_capabilities = list(agent_capabilities or [])
        self.planning_loop = BrainPlanningLoop(memory=memory, planner=planner, tools=self.tools)
        self.interactive_planning = BrainInteractivePlanningController(memory=memory)
        self._interactive_contexts: dict[str, ExecutionContext] = {}

    def set_agent_capabilities(self, capabilities: list[AgentCapability]) -> None:
        self.agent_capabilities = list(capabilities)

    def invoke_tool(self, invocation: ToolInvocation) -> ToolResult:
        """Execute a Brain-owned runtime tool through the shared invocation contract."""

        try:
            tool = self.tools.get(invocation.tool_name)
        except KeyError:
            return ToolResult(
                tool_name=invocation.tool_name,
                ok=False,
                payload={"message": f"Unknown Brain tool {invocation.tool_name!r}"},
                error_code="unknown_tool",
            )
        if hasattr(tool, "invoke"):
            return tool.invoke(invocation)
        if callable(tool):
            return tool(invocation)
        return ToolResult(
            tool_name=invocation.tool_name,
            ok=False,
            payload={"message": f"Registered Brain tool {invocation.tool_name!r} is not invokable"},
            error_code="invalid_tool",
        )

    @staticmethod
    def _build_default_tools() -> ToolCatalog:
        catalog = ToolCatalog()
        cron_tool = CronTool()
        for tool_name in cron_tool.tool_names:
            catalog.register(tool_name, cron_tool)
        web_search_tool = WebSearchTool()
        for tool_name in web_search_tool.tool_names:
            catalog.register(tool_name, web_search_tool)
        return catalog

    def prepare(self, request: TaskRequest, plan_override: Plan | None = None) -> tuple[ExecutionContext, Plan]:
        trace_id = request.metadata.get("trace_id", request.task_id)
        context = ExecutionContext(trace_id=trace_id, task_request=request)

        episode_id = self.memory.start_task(request.description, request.task_type)
        context.runtime_state["episode_id"] = episode_id

        planning_context = self._build_planning_context(request)
        context.runtime_state["planning_context"] = planning_context

        raw_plan = plan_override or self._plan_initial(request=request, planning_context=planning_context)
        if plan_override is not None and "dynamic_execution" not in raw_plan.metadata:
            raw_plan = Plan(
                subtasks=raw_plan.subtasks,
                metadata={**raw_plan.metadata, "dynamic_execution": False},
            )
        context.runtime_state["planner_plan"] = raw_plan
        planning_context["interaction_target_hints"] = interaction_targeting.interaction_target_hints(
            request=request,
            subtasks=raw_plan.subtasks,
        )

        plan = raw_plan
        plan = self._normalize_plan(plan, request, seed_interaction=True)
        plan = self._attach_interactive_completion_criteria(context=context, plan=plan)
        plan = self._version_plan(context=context, plan=plan, reason="initial_plan")
        self._record_plan(context=context, plan=plan, reason="initial_plan")
        self._sync_working_memory_after_plan(context=context, plan=plan, reason="initial_plan")
        return context, plan

    def begin_interactive_prepare(self, request: TaskRequest) -> BrainPlanningSession:
        """Begin an opt-in interactive planning session without executing a plan."""

        trace_id = request.metadata.get("trace_id", request.task_id)
        context = ExecutionContext(trace_id=trace_id, task_request=request)

        episode_id = self.memory.start_task(request.description, request.task_type)
        context.runtime_state["episode_id"] = episode_id

        try:
            planning_context = self._build_planning_context(request)
            context.runtime_state["planning_context"] = planning_context

            planning_context["interactive_planning_request"] = {
                "phase": "draft",
                "require_complete_plan": True,
            }
            provisional_plan = self._plan_initial(request=request, planning_context=planning_context)
            if not provisional_plan.subtasks:
                raise ValueError("Interactive planning produced no provisional subtasks")
            context.runtime_state["provisional_plan"] = provisional_plan

            session = self.interactive_planning.begin(
                request,
                planning_context,
                provisional_plan=provisional_plan,
            )
            context.runtime_state["interactive_planning"] = session.to_task_context()
            self._interactive_contexts[session.session_id] = context
            return session
        except Exception as exc:
            try:
                self.memory.end_task(outcome="failure", failure_reason=str(exc))
            except Exception:
                pass
            raise

    def answer_planning_question(
        self,
        session: BrainPlanningSession,
        answer: UserAnswer,
    ) -> BrainPlanningSession:
        """Record a user clarification for an opt-in interactive planning session."""

        updated = self.interactive_planning.answer(session, answer)
        context = self._interactive_contexts.get(updated.session_id)
        if context is not None:
            context.runtime_state["interactive_planning"] = updated.to_task_context()
        return updated

    def confirm_interactive_plan(
        self,
        session: BrainPlanningSession,
        confirmation: PlanConfirmation,
    ) -> Plan:
        """Compile an executable plan only after the user confirms the text draft."""

        _, plan = self.confirm_interactive_plan_with_context(session, confirmation)
        return plan

    def confirm_interactive_plan_with_context(
        self,
        session: BrainPlanningSession,
        confirmation: PlanConfirmation,
    ) -> tuple[ExecutionContext, Plan]:
        """Compile an executable plan and expose the session's original context."""

        if confirmation.confirmed:
            unanswered = session.unanswered_required_questions()
            if unanswered:
                raise ValueError("Cannot confirm interactive plan with unanswered required clarification")

        updated = self.interactive_planning.confirm(session, confirmation)
        context = self._interactive_contexts.get(updated.session_id)
        if context is None:
            raise ValueError(f"Unknown interactive planning session {updated.session_id!r}")
        context.runtime_state["interactive_planning"] = updated.to_task_context()
        if not confirmation.confirmed:
            failure_reason = "Interactive plan was not confirmed"
            self._close_failed_interactive_session(
                session_id=updated.session_id,
                failure_reason=failure_reason,
            )
            raise ValueError(failure_reason)

        planning_context = context.runtime_state.get("planning_context", {})
        planning_context["interactive_planning"] = updated.to_task_context()
        planning_context["interactive_planning_request"] = {
            "phase": "refinement",
            "require_complete_plan": True,
            "preserve_confirmed_milestones": True,
            "allow_conditional_support_reordering": True,
        }

        try:
            request = context.task_request
            confirmed_steps = action_contract_steps(
                planning_context["interactive_planning"]["text_plan"]
            )
            raw_plan = self._plan_confirmed_refinement(
                request=request,
                planning_context=planning_context,
                confirmed_steps=confirmed_steps,
                fallback_plan=context.runtime_state.get("provisional_plan"),
            )
            context.runtime_state["planner_plan"] = raw_plan
            planning_context["interaction_target_hints"] = (
                interaction_targeting.interaction_target_hints(
                    request=request,
                    subtasks=raw_plan.subtasks,
                )
            )

            plan = self._normalize_plan(raw_plan, request, seed_interaction=False)
            plan = self._attach_interactive_completion_criteria(
                context=context,
                plan=plan,
            )
            plan = self._version_plan(
                context=context,
                plan=plan,
                reason="interactive_plan_confirmed",
            )
            self._record_plan(
                context=context,
                plan=plan,
                reason="interactive_plan_confirmed",
            )
            self._sync_working_memory_after_plan(
                context=context,
                plan=plan,
                reason="interactive_plan_confirmed",
            )
            return context, plan
        except Exception as exc:
            self._close_failed_interactive_session(
                session_id=updated.session_id,
                failure_reason=str(exc),
            )
            raise

    def _close_failed_interactive_session(
        self,
        *,
        session_id: str,
        failure_reason: str,
    ) -> None:
        self._interactive_contexts.pop(session_id, None)
        try:
            self.memory.end_task(outcome="failure", failure_reason=failure_reason)
        except Exception:
            pass

    def bootstrap_after_reset(
        self,
        request: TaskRequest,
        context: ExecutionContext,
        initial_plan: Plan,
        environment_state: dict[str, Any] | None,
    ) -> Plan:
        if isinstance(environment_state, dict) and environment_state:
            context.runtime_state["environment"] = dict(environment_state)
        planning_context = context.runtime_state.get("planning_context", {})
        self._refresh_runtime_planning_context(
            planning_context=planning_context,
            execution_state=None,
            environment_state=environment_state,
        )
        if self._uses_autonomous_planner():
            return initial_plan

        guarded = RuntimeInteractionControlPolicy.deterministic_interaction_plan(
            request=request,
            context=context,
            next_index=1,
            execution_state=None,
            fallback_subtasks=initial_plan.subtasks,
        )
        if guarded is None:
            return initial_plan

        plan = Plan(subtasks=guarded, metadata={"planner": "runtime_room_gate", "dynamic_execution": True})
        plan = self._version_plan(context=context, plan=plan, reason="runtime_bootstrap")
        self._record_plan(context=context, plan=plan, reason="runtime_bootstrap")
        self._sync_working_memory_after_plan(context=context, plan=plan, reason="runtime_bootstrap")
        return plan

    def should_bootstrap_after_reset(self, request: TaskRequest, plan: Plan) -> bool:
        return RuntimeInteractionControlPolicy.should_apply_runtime_interaction_control(
            request=request,
            subtasks=plan.subtasks,
        )

    def next_step(
        self,
        request: TaskRequest,
        context: ExecutionContext,
        latest_result: Any,
    ) -> Plan:
        planning_context = context.runtime_state.get("planning_context", {})
        execution_state = self._build_execution_state(context=context, latest_result=latest_result)
        self._refresh_runtime_planning_context(
            planning_context=planning_context,
            execution_state=execution_state,
            environment_state=context.runtime_state.get("environment"),
        )

        if not self._uses_autonomous_planner():
            guarded = RuntimeInteractionControlPolicy.deterministic_interaction_plan(
                request=request,
                context=context,
                next_index=RuntimeInteractionControlPolicy.coerce_next_index(execution_state.get("next_subtask_index")),
                execution_state=execution_state,
            )
            if guarded is not None:
                plan = Plan(subtasks=guarded, metadata={"planner": "runtime_room_gate", "dynamic_execution": True})
                plan = self._version_plan(context=context, plan=plan, reason="next_step")
                self._record_plan(context=context, plan=plan, reason="next_step")
                self._sync_working_memory_after_plan(
                    context=context,
                    plan=plan,
                    reason="next_step",
                    execution_state=execution_state,
                )
                return plan

        plan = self._plan_next_step(
            request=request,
            planning_context=planning_context,
            execution_state=execution_state,
        )
        plan = self._normalize_plan(plan, request, seed_interaction=False)
        plan = self._attach_interactive_completion_criteria(context=context, plan=plan)
        plan = self._version_plan(context=context, plan=plan, reason="next_step")
        self._record_plan(context=context, plan=plan, reason="next_step")
        self._sync_working_memory_after_plan(
            context=context,
            plan=plan,
            reason="next_step",
            execution_state=execution_state,
        )
        return plan

    def replan(
        self,
        request: TaskRequest,
        context: ExecutionContext,
        failed_subtask: Subtask,
        failure_reason: str,
        latest_result: Any | None = None,
    ) -> Plan:
        planning_context = context.runtime_state.get("planning_context", {})
        execution_state = self._build_execution_state(context=context, latest_result=latest_result)
        execution_state["failed_subtask"] = {
            "subtask_id": failed_subtask.subtask_id,
            "execution_id": failed_subtask.runtime_id,
            "plan_revision": failed_subtask.plan_revision,
            "agent": failed_subtask.agent.value,
            "action": failed_subtask.action,
            "target": dict(failed_subtask.target),
            "parameters": dict(failed_subtask.parameters),
            "context": dict(failed_subtask.context),
        }
        execution_state["failure_reason"] = failure_reason
        brain_planning_context.attach_counterfactual_evidence(
            memory=self.memory,
            planning_context=planning_context,
            task_description=request.description,
            failed_subtask=failed_subtask,
            top_k=3,
        )
        plan = self._plan_replacement(
            request=request,
            planning_context=planning_context,
            failed_subtask=failed_subtask,
            failure_reason=failure_reason,
            execution_state=execution_state,
        )
        plan = self._normalize_plan(plan, request, seed_interaction=False)
        plan = self._attach_interactive_completion_criteria(context=context, plan=plan)
        plan = self._version_plan(
            context=context,
            plan=plan,
            reason="replan",
            replaces_execution_id=failed_subtask.runtime_id,
        )
        self._record_plan(context=context, plan=plan, reason="replan")
        self._sync_working_memory_after_plan(
            context=context,
            plan=plan,
            reason="replan",
            execution_state=execution_state,
        )
        return plan

    def finalize(self, success: bool, failure_reason: str | None = None) -> dict[str, Any]:
        outcome = "success" if success else "failure"
        end_info = self.memory.end_task(outcome=outcome, failure_reason=failure_reason)

        try:
            reflection = self.memory.reflect()
        except Exception as exc:
            reflection = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        return {
            "outcome": outcome,
            "end_info": end_info,
            "reflection": reflection,
        }

    def _build_planning_context(self, request: TaskRequest) -> dict[str, Any]:
        planning_context = brain_planning_context.build_planning_context(
            memory=self.memory,
            request=request,
            planner_mode_from_request=RuntimeInteractionControlPolicy.planner_mode_from_request,
        )
        planning_context["agent_capabilities"] = serialize_agent_capabilities(self.agent_capabilities)
        return planning_context

    def _plan_initial(self, *, request: TaskRequest, planning_context: dict[str, Any]) -> Plan:
        if self.planning_loop.is_enabled():
            return self.planning_loop.run_initial(
                task_description=request.description,
                planning_context=planning_context,
                invoke_tool=self.invoke_tool,
            )
        return self.planner.plan(request.description, planning_context)

    def _plan_confirmed_refinement(
        self,
        *,
        request: TaskRequest,
        planning_context: dict[str, Any],
        confirmed_steps: list[Any],
        fallback_plan: Plan | None = None,
    ) -> Plan:
        """Plan and align a confirmed draft, retrying only failed refinements.

        The provisional plan is the executable source from which the user-facing
        draft was produced. If nondeterministic refinement attempts all violate
        that confirmed contract, retain the aligned provisional plan instead of
        deleting a user-confirmed world-changing milestone.
        """

        config = getattr(self.planner, "config", None)
        retries = getattr(config, "semantic_validation_retries", 0)
        attempts = max(1, int(retries) + 1)
        last_error: ValueError | None = None

        for attempt in range(attempts):
            raw_plan = self._plan_initial(request=request, planning_context=planning_context)
            try:
                return align_refined_plan(confirmed_steps, raw_plan)
            except ValueError as exc:
                last_error = exc
                if attempt + 1 >= attempts:
                    break
                planning_context["interactive_alignment_error"] = str(exc)

        if last_error is not None:
            if fallback_plan is not None:
                try:
                    fallback = align_refined_plan(confirmed_steps, fallback_plan)
                except ValueError:
                    pass
                else:
                    fallback.metadata["interactive_refinement_fallback"] = (
                        "confirmed_provisional_plan"
                    )
                    fallback.metadata["interactive_alignment_error"] = str(last_error)
                    return fallback
            raise last_error
        raise RuntimeError("Confirmed refinement failed without an alignment error")

    def _plan_next_step(
        self,
        *,
        request: TaskRequest,
        planning_context: dict[str, Any],
        execution_state: dict[str, Any],
    ) -> Plan:
        if self.planning_loop.is_enabled():
            return self.planning_loop.run_next_step(
                task_description=request.description,
                planning_context=planning_context,
                execution_state=execution_state,
                invoke_tool=self.invoke_tool,
            )
        return self.planner.plan_next(
            task_description=request.description,
            context=planning_context,
            execution_state=execution_state,
        )

    def _plan_replacement(
        self,
        *,
        request: TaskRequest,
        planning_context: dict[str, Any],
        failed_subtask: Subtask,
        failure_reason: str,
        execution_state: dict[str, Any],
    ) -> Plan:
        if self.planning_loop.is_enabled():
            return self.planning_loop.run_replan(
                task_description=request.description,
                planning_context=planning_context,
                failed_subtask=failed_subtask,
                failure_reason=failure_reason,
                execution_state=execution_state,
                invoke_tool=self.invoke_tool,
            )
        return self.planner.replan(
            task_description=request.description,
            context=planning_context,
            failed_subtask=failed_subtask,
            failure_reason=failure_reason,
            execution_state=execution_state,
        )

    def _refresh_runtime_planning_context(
        self,
        *,
        planning_context: dict[str, Any],
        execution_state: dict[str, Any] | None,
        environment_state: dict[str, Any] | None,
    ) -> None:
        brain_planning_context.refresh_runtime_planning_context(
            memory=self.memory,
            planning_context=planning_context,
            execution_state=execution_state,
            environment_state=environment_state,
            resolve_navigation_state=lambda state, env: navigation_runtime.resolve_navigation_state(
                execution_state=state,
                environment_state=env,
            ),
        )

    def _normalize_plan(
        self,
        plan: Plan,
        request: TaskRequest,
        *,
        seed_interaction: bool,
    ) -> Plan:
        return brain_plan_flow.normalize_plan(
            plan=plan,
            request=request,
            seed_interaction=seed_interaction,
            should_apply_runtime_interaction_control=lambda req, subtasks: (
                False
                if self._uses_autonomous_planner()
                else RuntimeInteractionControlPolicy.should_apply_runtime_interaction_control(
                    request=req,
                    subtasks=subtasks,
                )
            ),
        )

    @staticmethod
    def _version_plan(
        *,
        context: ExecutionContext,
        plan: Plan,
        reason: str,
        replaces_execution_id: str | None = None,
    ) -> Plan:
        return brain_plan_flow.version_plan(
            context=context,
            plan=plan,
            reason=reason,
            replaces_execution_id=replaces_execution_id,
        )

    def _uses_autonomous_planner(self) -> bool:
        return self.planning_loop.is_enabled()

    @staticmethod
    def _attach_interactive_completion_criteria(
        *,
        context: ExecutionContext,
        plan: Plan,
    ) -> Plan:
        interactive = context.runtime_state.get("interactive_planning")
        if not isinstance(interactive, dict):
            return plan
        text_plan = interactive.get("text_plan")
        if not isinstance(text_plan, dict):
            return plan
        raw_criteria = [item for item in text_plan.get("success_criteria") or [] if isinstance(item, dict)]
        if not raw_criteria:
            return plan

        criteria_by_subtask: dict[str, list[dict[str, Any]]] = {}
        for item in raw_criteria:
            scope = str(item.get("scope") or "").strip().lower()
            if scope == "collaborative_step":
                target_subtask = BrainAgent._resolve_collaborative_criterion_subtask(
                    plan=plan,
                    criterion=item,
                )
            elif scope == "subtask":
                target_subtask = BrainAgent._resolve_interactive_criterion_subtask(
                    plan=plan,
                    criterion=item,
                )
            else:
                continue
            if target_subtask is None:
                continue
            criterion = dict(item)
            criterion["scope"] = "subtask"
            criterion["subtask_id"] = target_subtask.subtask_id
            criteria_by_subtask.setdefault(target_subtask.subtask_id, []).append(criterion)
        if not criteria_by_subtask:
            return plan

        updated_subtasks: list[Subtask] = []
        changed = False
        for subtask in plan.subtasks:
            criteria = criteria_by_subtask.get(subtask.subtask_id)
            if not criteria:
                updated_subtasks.append(subtask)
                continue
            parameters = dict(subtask.parameters)
            existing = [dict(item) for item in parameters.get("completion_criteria") or [] if isinstance(item, dict)]
            seen = {str(item.get("criterion_id") or item.get("description") or "") for item in existing}
            for criterion in criteria:
                key = str(criterion.get("criterion_id") or criterion.get("description") or "")
                if key and key in seen:
                    continue
                existing.append(dict(criterion))
                if key:
                    seen.add(key)
            parameters["completion_criteria"] = existing
            updated_subtasks.append(
                Subtask(
                    subtask_id=subtask.subtask_id,
                    agent=subtask.agent,
                    action=subtask.action,
                    target=dict(subtask.target),
                    parameters=parameters,
                    context=dict(subtask.context),
                )
            )
            changed = True
        if not changed:
            return plan
        return Plan(subtasks=updated_subtasks, metadata=dict(plan.metadata))

    @staticmethod
    def _resolve_interactive_criterion_subtask(
        *,
        plan: Plan,
        criterion: dict[str, Any],
    ) -> Subtask | None:
        original_subtask_id = str(criterion.get("subtask_id") or "").strip()
        metadata = criterion.get("metadata")
        desired_agent = ""
        if isinstance(metadata, dict):
            desired_agent = str(metadata.get("agent") or "").strip().upper()
        desired_agent = desired_agent or str(criterion.get("agent") or "").strip().upper()

        exact_index = None
        exact_subtask = None
        for index, subtask in enumerate(plan.subtasks):
            if subtask.subtask_id == original_subtask_id:
                exact_index = index
                exact_subtask = subtask
                break
        if exact_subtask is not None and (
            not desired_agent or exact_subtask.agent.value.upper() == desired_agent
        ):
            return exact_subtask

        if desired_agent:
            start = (exact_index + 1) if exact_index is not None else 0
            for subtask in plan.subtasks[start:]:
                if subtask.agent.value.upper() == desired_agent:
                    return subtask
            for subtask in plan.subtasks:
                if subtask.agent.value.upper() == desired_agent:
                    return subtask
        return exact_subtask

    @staticmethod
    def _resolve_collaborative_criterion_subtask(
        *,
        plan: Plan,
        criterion: dict[str, Any],
    ) -> Subtask | None:
        desired_step_id = str(criterion.get("collaborative_step_id") or "").strip()
        if desired_step_id:
            for subtask in plan.subtasks:
                if str(subtask.parameters.get("collaborative_step_id") or "").strip() == desired_step_id:
                    return subtask
        return None

    @staticmethod
    def _seed_interaction_plan(subtasks: list[Subtask], request: TaskRequest) -> list[Subtask]:
        return brain_plan_flow.seed_interaction_plan(subtasks=subtasks, request=request)

    @staticmethod
    def _record_plan(context: ExecutionContext, plan: Plan, reason: str) -> None:
        brain_plan_flow.record_plan(context=context, plan=plan, reason=reason)

    def _sync_working_memory_after_plan(
        self,
        *,
        context: ExecutionContext,
        plan: Plan,
        reason: str,
        execution_state: dict[str, Any] | None = None,
    ) -> None:
        brain_plan_flow.sync_working_memory_after_plan(
            memory=self.memory,
            context=context,
            plan=plan,
            reason=reason,
            execution_state=execution_state,
        )

    @staticmethod
    def _serialize_subtask_summary(subtask: Subtask) -> dict[str, Any]:
        return brain_plan_flow.serialize_subtask_summary(subtask)

    @staticmethod
    def _format_task_phase(subtask_summary: dict[str, Any]) -> str | None:
        return brain_plan_flow.format_task_phase(subtask_summary)

    @staticmethod
    def _recent_plan_decisions(context: ExecutionContext) -> list[dict[str, Any]]:
        return brain_plan_flow.recent_plan_decisions(context)

    @staticmethod
    def _build_runtime_namespace(context: ExecutionContext) -> dict[str, Any]:
        return brain_plan_flow.build_runtime_namespace(context)

    def _build_execution_state(self, context: ExecutionContext, latest_result: Any) -> dict[str, Any]:
        return brain_plan_flow.build_execution_state(
            context=context,
            latest_result=latest_result,
            planner_mode=RuntimeInteractionControlPolicy.planner_mode_from_request(context.task_request),
        )

    @classmethod
    def _build_interaction_room_navigation_subtask(
        cls,
        *,
        target_hints: dict[str, str],
        task_description: str,
        index: int,
    ) -> Subtask:
        return RuntimeInteractionControlPolicy.build_interaction_room_navigation_subtask(
            target_hints=target_hints,
            task_description=task_description,
            index=index,
        )

    @classmethod
    def _build_interaction_approach_subtask(
        cls,
        *,
        target_hints: dict[str, str],
        task_description: str,
        index: int,
    ) -> Subtask:
        return RuntimeInteractionControlPolicy.build_interaction_approach_subtask(
            target_hints=target_hints,
            task_description=task_description,
            index=index,
        )

    @classmethod
    def _build_interaction_action_subtask(
        cls,
        *,
        target_hints: dict[str, str],
        task_description: str,
        index: int,
    ) -> Subtask:
        return RuntimeInteractionControlPolicy.build_interaction_action_subtask(
            target_hints=target_hints,
            task_description=task_description,
            index=index,
        )

    @classmethod
    def _build_interaction_inspect_subtask(
        cls,
        *,
        target_hints: dict[str, str],
        task_description: str,
        index: int,
    ) -> Subtask:
        return RuntimeInteractionControlPolicy.build_interaction_inspect_subtask(
            target_hints=target_hints,
            task_description=task_description,
            index=index,
        )
