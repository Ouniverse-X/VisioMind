from __future__ import annotations

import logging
from collections import deque
from typing import Any

from . import completion_policy, replanning_flow, step_runner
from .completion_monitor import CompletionDecision, CompletionMonitor
from .vision_heartbeat import VisionHeartbeatRunner
from voltron.runtime.orchestrator.agent_bindings import resolve_orchestrator_agents
from voltron.runtime.session.events import VoltronEvent
from voltron.runtime.telemetry.run_logger import build_task_run_response
from voltron.shared.context import ExecutionContext, Plan, Subtask, TaskRequest
from voltron.shared.contracts import RuntimeEnvironment, SubtaskAgent
from voltron.shared.enums import AgentName, AgentStatus
from voltron.shared.results import AgentResult
from voltron.shared.telemetry.payload_sanitizer import strip_image_payloads

logger = logging.getLogger(__name__)


class ClosedLoopOrchestrator:
    def __init__(
        self,
        brain_agent: Any | None = None,
        vision_agent: SubtaskAgent | None = None,
        navigation_agent: SubtaskAgent | None = None,
        action_agent: SubtaskAgent | None = None,
        max_retries: int = 1,
        max_control_steps_per_subtask: int = 120,
        event_sink: Any | None = None,
        log_navigation_candidates: bool = False,
        log_nav2_path_snapshots: bool = False,
        vision_heartbeat_interval_steps: int = 200,
        use_environment_success_signal: bool = True,
        use_brain_completion_signal: bool = True,
        environment_signal_policy: str = "allow_early_success",
        completion_evaluator: Any | None = None,
        vision_completion_positive_streak: int = 1,
        vision_completion_stability_steps: int = 1,
        vision_completion_action_delta_threshold: float = 0.03,
        vision_completion_check_interval_steps: int = 200,
        vision_completion_agent_scope: list[str] | tuple[str, ...] | set[str] | None = None,
    ):
        brain_agent, vision_agent, navigation_agent, action_agent = resolve_orchestrator_agents(
            brain_agent=brain_agent,
            vision_agent=vision_agent,
            navigation_agent=navigation_agent,
            action_agent=action_agent,
        )
        self.brain_agent = brain_agent
        self.vision_agent = vision_agent
        self.navigation_agent = navigation_agent
        self.action_agent = action_agent
        self.max_retries = max_retries
        self.max_control_steps_per_subtask = max_control_steps_per_subtask
        self.event_sink = event_sink
        self.log_navigation_candidates = log_navigation_candidates
        self.log_nav2_path_snapshots = log_nav2_path_snapshots
        self.vision_heartbeat_interval_steps = max(0, int(vision_heartbeat_interval_steps))
        self.completion_monitor = CompletionMonitor(
            use_environment_success_signal=use_environment_success_signal,
            use_brain_completion_signal=use_brain_completion_signal,
            environment_signal_policy=environment_signal_policy,
            evaluator=completion_evaluator,
            positive_streak=vision_completion_positive_streak,
            stability_steps=vision_completion_stability_steps,
            action_delta_threshold=vision_completion_action_delta_threshold,
            check_interval_steps=vision_completion_check_interval_steps,
            completion_agent_scope=vision_completion_agent_scope,
        )
        self._vision_heartbeat_runner: VisionHeartbeatRunner | None = None
        self._agents: dict[AgentName, SubtaskAgent] = {
            AgentName.VISION: self.vision_agent,
            AgentName.NAVIGATION: self.navigation_agent,
            AgentName.ACTION: self.action_agent,
        }

    def run_task(
        self,
        request: TaskRequest,
        environment: RuntimeEnvironment,
        plan_override: Plan | None = None,
    ) -> dict[str, Any]:
        context, plan = self.brain_agent.prepare(request, plan_override=plan_override)
        return self.run_prepared_task(
            request=request,
            environment=environment,
            context=context,
            plan=plan,
            initial_plan_reason="initial_plan",
        )

    def run_prepared_task(
        self,
        request: TaskRequest,
        environment: RuntimeEnvironment,
        context: ExecutionContext,
        plan: Plan,
        *,
        initial_plan_reason: str = "initial_plan",
    ) -> dict[str, Any]:
        context.runtime_state.setdefault(
            "log_navigation_candidates", self.log_navigation_candidates
        )
        context.runtime_state.setdefault("log_nav2_path_snapshots", self.log_nav2_path_snapshots)
        context.runtime_state["current_plan_subtask_ids"] = [
            item.subtask_id for item in plan.subtasks
        ]
        context.runtime_state["current_plan_execution_ids"] = [
            item.runtime_id for item in plan.subtasks
        ]
        self._emit_event(
            event_type="brain_plan",
            source="BRAIN",
            message=f"initial plan with {len(plan.subtasks)} subtasks",
            payload=_serialize_plan_event_payload(plan=plan, reason=initial_plan_reason),
            task_id=request.task_id,
        )
        response: dict[str, Any] | None = None
        cleanup_errors: list[dict[str, str]] = []
        pending_subtasks = deque(plan.subtasks)
        dynamic_execution = bool(plan.metadata.get("dynamic_execution", False))
        context.runtime_state["dynamic_execution"] = dynamic_execution

        try:
            context.runtime_state["environment"] = environment.reset(
                request=request,
                plan=plan,
                context=context,
            )
            self._start_vision_heartbeat(context=context, environment=environment)
            self._emit_event(
                event_type="environment_reset",
                source="RUNTIME",
                message="environment reset",
                payload=dict(context.runtime_state["environment"]),
                task_id=request.task_id,
            )
            self._update_working_memory_task_context(
                context=context,
                updates={
                    "runtime_namespace": {
                        "scene_id": context.runtime_state["environment"].get("scene_id"),
                    },
                    "execution_state": {
                        "robot_state": self._extract_robot_state(
                            runtime_inputs={},
                            env_feedback=context.runtime_state["environment"],
                        )
                    },
                },
            )
            context.runtime_state["vla_policy_reset_pending"] = self._vla_policy_needs_reset()

            if dynamic_execution and self.brain_agent.should_bootstrap_after_reset(request, plan):
                bootstrapped_plan = self.brain_agent.bootstrap_after_reset(
                    request=request,
                    context=context,
                    initial_plan=plan,
                    environment_state=context.runtime_state.get("environment"),
                )
                if self._plan_changed(plan, bootstrapped_plan):
                    plan = bootstrapped_plan
                    context.runtime_state["current_plan_subtask_ids"] = [
                        item.subtask_id for item in plan.subtasks
                    ]
                    context.runtime_state["current_plan_execution_ids"] = [
                        item.runtime_id for item in plan.subtasks
                    ]
                    pending_subtasks = deque(plan.subtasks)
                    dynamic_execution = bool(
                        plan.metadata.get("dynamic_execution", dynamic_execution)
                    )
                    replanning_flow.update_environment_plan(
                        environment=environment, context=context, plan=plan
                    )
                    self._emit_event(
                        event_type="brain_plan",
                        source="BRAIN",
                        message=f"runtime bootstrap plan with {len(plan.subtasks)} subtasks",
                        payload=_serialize_plan_event_payload(
                            plan=plan, reason="runtime_bootstrap"
                        ),
                        task_id=request.task_id,
                    )

            while pending_subtasks:
                subtask = pending_subtasks.popleft()
                result, replanned_followups, replaced_pending_plan = self._execute_with_retry(
                    request=request,
                    context=context,
                    subtask=subtask,
                    environment=environment,
                )
                context.results.append(result)

                if result.status == AgentStatus.FAILURE:
                    self._flush_pending_object_approach_outcome(
                        context=context,
                        success=False,
                        reason="task_failed_after_approach",
                    )
                    final = self.brain_agent.finalize(
                        success=False, failure_reason=result.error_code
                    )
                    response = build_task_run_response(context, final)
                    break

                if replaced_pending_plan:
                    pending_subtasks.clear()
                if replanned_followups:
                    pending_subtasks.extendleft(reversed(replanned_followups))

                dynamic_execution = bool(
                    context.runtime_state.get("dynamic_execution", dynamic_execution)
                )
                if (
                    dynamic_execution
                    and not pending_subtasks
                    and not self._task_succeeded(
                        context=context,
                        environment=environment,
                    )
                ):
                    next_plan = self.brain_agent.next_step(
                        request=request,
                        context=context,
                        latest_result=result,
                    )
                    self._emit_event(
                        event_type="brain_plan",
                        source="BRAIN",
                        message=f"next plan with {len(next_plan.subtasks)} subtasks",
                        payload=_serialize_plan_event_payload(plan=next_plan, reason="next_step"),
                        task_id=request.task_id,
                    )
                    replanning_flow.update_environment_plan(
                        environment=environment, context=context, plan=next_plan
                    )
                    context.runtime_state["current_plan_subtask_ids"] = [
                        item.subtask_id for item in next_plan.subtasks
                    ]
                    context.runtime_state["current_plan_execution_ids"] = [
                        item.runtime_id for item in next_plan.subtasks
                    ]
                    pending_subtasks.extend(next_plan.subtasks)

            if response is None:
                task_success = self._task_succeeded(context=context, environment=environment)
                self._flush_pending_object_approach_outcome(
                    context=context,
                    success=task_success,
                    reason="task_completed_after_approach"
                    if task_success
                    else "task_failed_after_approach",
                )
                final = self.brain_agent.finalize(
                    success=task_success,
                    failure_reason=None if task_success else "TASK_NOT_COMPLETED",
                )
                response = build_task_run_response(context, final)
        finally:
            cleanup_errors = self._cleanup_after_run(environment=environment)

        if response is None:
            raise RuntimeError("closed loop finished without response")

        if cleanup_errors:
            response["final"]["cleanup_errors"] = cleanup_errors
        response["final"]["environment"] = environment.summary()
        self._emit_event(
            event_type="task_final",
            source="RUNTIME",
            message=f"task finished with {response['final'].get('outcome')}",
            payload=dict(response["final"]),
            task_id=request.task_id,
        )
        return response

    def _execute_with_retry(
        self,
        request: TaskRequest,
        context: ExecutionContext,
        subtask: Subtask,
        environment: RuntimeEnvironment,
    ) -> tuple[AgentResult, list[Subtask], bool]:
        return replanning_flow.execute_with_retry(
            orchestrator=self,
            request=request,
            context=context,
            subtask=subtask,
            environment=environment,
        )

    def _run_subtask_control_loop(
        self,
        subtask: Subtask,
        context: ExecutionContext,
        environment: RuntimeEnvironment,
        attempt: int,
    ) -> AgentResult:
        return step_runner.run_subtask_control_loop(
            orchestrator=self,
            subtask=subtask,
            context=context,
            environment=environment,
            attempt=attempt,
        )

    def _start_vision_heartbeat(
        self, *, context: ExecutionContext, environment: RuntimeEnvironment
    ) -> None:
        if self.vision_heartbeat_interval_steps <= 0:
            return
        runner = VisionHeartbeatRunner(
            vision_agent=self.vision_agent,
            interval_steps=self.vision_heartbeat_interval_steps,
            emit_event=self._emit_event,
        )
        runner.start(context=context, environment=environment)
        self._vision_heartbeat_runner = runner

    def _evaluate_completion_step(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        environment: RuntimeEnvironment,
        result: AgentResult,
        environment_outcome: Any,
        control_step: int,
    ) -> CompletionDecision:
        decision = self.completion_monitor.evaluate_subtask_step(
            subtask=subtask,
            context=context,
            result=result,
            environment_outcome=environment_outcome,
            control_step=control_step,
        )
        self._record_completion_decision(
            context=context,
            subtask=subtask,
            decision=decision,
            control_step=control_step,
        )
        self._notify_environment_completion_decision(
            environment=environment,
            subtask=subtask,
            context=context,
            decision=decision,
        )
        return decision

    @staticmethod
    def _notify_environment_completion_decision(
        *,
        environment: RuntimeEnvironment,
        subtask: Subtask,
        context: ExecutionContext,
        decision: CompletionDecision,
    ) -> None:
        callback = getattr(environment, "on_subtask_completion_decision", None)
        if not callable(callback):
            return
        try:
            callback(
                subtask,
                {
                    "done": bool(decision.done),
                    "success": decision.success,
                    "failure_reason": decision.failure_reason,
                    "feedback": decision.feedback,
                    "verdict": decision.verdict.to_dict(),
                },
                context,
            )
        except Exception as exc:
            logger.warning("Environment completion-decision hook failed: %s", exc)

    def _stop_vision_heartbeat(self, *, flush: bool) -> None:
        runner = self._vision_heartbeat_runner
        self._vision_heartbeat_runner = None
        if runner is not None:
            runner.stop(flush=flush)

    def _on_environment_step(
        self,
        *,
        context: ExecutionContext,
        environment: RuntimeEnvironment,
        env_step: int,
        source_subtask: Subtask,
        feedback: dict[str, Any],
    ) -> None:
        runner = self._vision_heartbeat_runner
        if runner is None:
            return
        runner.on_environment_step(
            context=context,
            environment=environment,
            env_step=env_step,
            source_subtask=source_subtask,
            feedback=feedback,
        )

    def _task_succeeded(
        self, *, context: ExecutionContext, environment: RuntimeEnvironment
    ) -> bool:
        if self.completion_monitor._environment_success_allowed():
            return environment.task_succeeded(context)
        if bool(context.runtime_state.get("dynamic_execution", False)):
            return False
        monitor_state = context.runtime_state.get("completion_monitor")
        if not isinstance(monitor_state, dict):
            return False
        planned = set(
            context.runtime_state.get("current_plan_execution_ids")
            or context.runtime_state.get("current_plan_subtask_ids")
            or []
        )
        completed = set(
            monitor_state.get("completed_execution_ids")
            or monitor_state.get("completed_subtasks")
            or []
        )
        return bool(planned) and planned.issubset(completed)

    def _vla_policy_needs_reset(self) -> bool:
        vla_agent = self._agents.get(AgentName.ACTION)
        return bool(
            vla_agent is not None
            and hasattr(vla_agent, "policy")
            and hasattr(vla_agent.policy, "reset")
        )

    def _maybe_reset_vla_policy(self, *, subtask: Subtask, context: ExecutionContext) -> None:
        if subtask.agent != AgentName.ACTION:
            return
        if not context.runtime_state.get("vla_policy_reset_pending", False):
            return

        vla_agent = self._agents.get(AgentName.ACTION)
        if vla_agent is not None and hasattr(vla_agent, "policy"):
            try:
                vla_agent.policy.reset()
            except Exception as exc:
                logger.warning("VLA policy reset failed: %s", exc)
        context.runtime_state["vla_policy_reset_pending"] = False

    def _update_working_memory_task_context(
        self, *, context: ExecutionContext, updates: dict[str, Any]
    ) -> None:
        _merge_runtime_state(context.runtime_state, updates)
        try:
            self.brain_agent.memory.update_task_context(updates)
        except Exception:
            pass

    def _record_completion_decision(
        self,
        *,
        context: ExecutionContext,
        subtask: Subtask,
        decision: CompletionDecision,
        control_step: int,
    ) -> None:
        verdict = strip_image_payloads(decision.verdict.to_dict())
        state = context.runtime_state.setdefault("completion_monitor", {})
        completed = set(
            state.get("completed_execution_ids") or state.get("completed_subtasks") or []
        )
        completed_subtasks = set(state.get("completed_subtasks") or [])
        if decision.done and decision.success is not False:
            completed.add(subtask.runtime_id)
            completed_subtasks.add(subtask.subtask_id)
        state.update(
            {
                "latest_verdict": verdict,
                "completed_execution_ids": sorted(completed),
                "completed_subtasks": sorted(completed_subtasks),
            }
        )
        self._update_working_memory_task_context(
            context=context,
            updates={
                "completion_monitor": {
                    "latest_verdict": verdict,
                    "completed_execution_ids": sorted(completed),
                    "completed_subtasks": sorted(completed_subtasks),
                }
            },
        )
        self._record_working_observation(
            {
                "source": "runtime_completion_monitor",
                "subtask_id": subtask.subtask_id,
                "control_step": control_step,
                "verdict": verdict,
            }
        )
        if self._should_emit_completion_monitor_event(
            decision=decision,
            verdict=verdict,
            control_step=control_step,
        ):
            self._emit_event(
                event_type="completion_monitor_decision",
                source="RUNTIME",
                message=(
                    f"completion monitor {subtask.subtask_id} "
                    f"done={decision.done} source={verdict.get('source')}"
                ),
                payload={
                    "subtask_id": subtask.subtask_id,
                    "agent": subtask.agent.value,
                    "action": subtask.action,
                    **self._compact_completion_metadata(subtask),
                    "control_step": control_step,
                    "done": decision.done,
                    "success": decision.success,
                    "verdict": self._compact_completion_verdict(verdict),
                    "completion_criteria": self._compact_completion_criteria(
                        subtask.parameters.get("completion_criteria")
                    ),
                },
                task_id=context.task_request.task_id,
            )

    def _should_emit_completion_monitor_event(
        self,
        *,
        decision: CompletionDecision,
        verdict: dict[str, Any],
        control_step: int,
    ) -> bool:
        if decision.done:
            return True
        source = str(verdict.get("source") or "")
        if source in {
            "environment",
            "environment_evidence",
            "runtime_subtask",
            "completion_monitor_evaluator_error",
        }:
            return True
        if source in {
            "completion_monitor",
            "completion_monitor_agent_scope",
            "completion_monitor_interval",
        }:
            return False
        interval = max(1, int(self.completion_monitor.check_interval_steps))
        return int(control_step) % interval == 0

    @staticmethod
    def _compact_completion_verdict(verdict: dict[str, Any]) -> dict[str, Any]:
        compact = {
            key: verdict.get(key)
            for key in (
                "scope",
                "scope_id",
                "completed",
                "confidence",
                "reason",
                "missing_evidence",
                "should_continue",
                "should_replan",
                "source",
            )
            if verdict.get(key) not in (None, "", [], {})
        }
        evidence = verdict.get("evidence")
        if isinstance(evidence, dict):
            compact["evidence"] = strip_image_payloads(
                {
                    key: evidence.get(key)
                    for key in (
                        "control_step",
                        "environment_done",
                        "environment_success",
                        "positive_streak",
                        "positive_streak_required",
                        "stable_steps",
                        "stable_steps_required",
                        "failure_reason",
                        "feedback",
                    )
                    if evidence.get(key) not in (None, "", [], {})
                }
            )
        return compact

    @staticmethod
    def _compact_completion_metadata(subtask: Subtask) -> dict[str, Any]:
        metadata = {
            "collaborative_step_id": subtask.parameters.get("collaborative_step_id"),
            "completion_condition_source": subtask.parameters.get("completion_condition_source"),
        }
        return {key: value for key, value in metadata.items() if value not in (None, "", [], {})}

    @staticmethod
    def _compact_completion_criteria(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        criteria: list[dict[str, Any]] = []
        for item in value[:5]:
            if not isinstance(item, dict):
                continue
            compact = {
                key: item.get(key)
                for key in (
                    "criterion_id",
                    "scope",
                    "subtask_id",
                    "agent",
                    "description",
                    "confidence",
                    "source",
                    "collaborative_step_id",
                    "completion_condition_source",
                )
                if item.get(key) not in (None, "", [], {})
            }
            if compact:
                criteria.append(compact)
        return criteria

    def _update_working_memory_for_agent_result(
        self,
        *,
        context: ExecutionContext,
        subtask: Subtask,
        result: AgentResult,
    ) -> None:
        active_internal = result.runtime_artifacts.get("action_active_internal_step")
        execution_plan = result.runtime_artifacts.get("action_execution_plan")
        execution_progress = result.runtime_artifacts.get("action_execution_progress")
        if not isinstance(active_internal, dict) or not isinstance(execution_plan, dict):
            return

        internal_phase = f"{active_internal.get('internal_step_id')}:{subtask.agent.value}:{active_internal.get('action')}"
        progress = execution_progress if isinstance(execution_progress, dict) else {}
        completed_step_ids = list(progress.get("completed_step_ids", []))
        pending_step_ids = list(progress.get("pending_step_ids", []))
        updates = {
            "execution_state": {
                "task_phase": internal_phase,
                "parent_task_phase": f"{subtask.subtask_id}:{subtask.agent.value}:{subtask.action}",
                "current_internal_subtask": {
                    "internal_step_id": active_internal.get("internal_step_id"),
                    "name": active_internal.get("name"),
                    "action": active_internal.get("action"),
                    "instruction": active_internal.get("instruction"),
                    "target": dict(active_internal.get("target") or {}),
                    "step_index": active_internal.get("step_index"),
                    "total_steps": active_internal.get("total_steps"),
                    "selected_skill_id": active_internal.get("selected_skill_id"),
                    "preferred_skill_id": active_internal.get("preferred_skill_id"),
                },
                "action_internal_plan": {
                    "parent_subtask_id": execution_plan.get("parent_subtask_id"),
                    "goal_summary": execution_plan.get("goal_summary"),
                    "source": execution_plan.get("source"),
                    "current_step_index": progress.get("current_step_index"),
                    "total_steps": progress.get("total_steps"),
                    "completed_count": len(completed_step_ids),
                    "pending_count": len(pending_step_ids),
                    "plan_completed": bool(progress.get("plan_completed", False)),
                },
            }
        }
        self._update_working_memory_task_context(context=context, updates=updates)

    def _record_working_observation_for_agent_result(
        self,
        *,
        context: ExecutionContext,
        subtask: Subtask,
        result: AgentResult,
        control_step: int,
    ) -> None:
        if subtask.agent != AgentName.VISION or result.status != AgentStatus.SUCCESS:
            return
        payload = {
            "source": "runtime_vision_result",
            "agent": subtask.agent.value,
            "subtask_id": subtask.subtask_id,
            "control_step": control_step,
            "status": result.status.value,
            "objects": list(result.result.get("objects") or []),
            "relations": list(result.result.get("relations") or []),
            "task_complete": bool(result.result.get("task_complete", False)),
            "room_id": self._first_text(
                subtask.target.get("room"),
                subtask.parameters.get("room_id"),
                subtask.parameters.get("room"),
            ),
            "region_id": self._first_text(
                subtask.target.get("region"),
                subtask.parameters.get("region_id"),
                subtask.target.get("room"),
            ),
        }
        raw_text = result.result.get("raw_text")
        if isinstance(raw_text, str) and raw_text.strip():
            payload["summary"] = raw_text[:240]
        self._record_working_observation(payload)

    def _record_working_observation_for_runtime_feedback(
        self,
        *,
        context: ExecutionContext,
        subtask: Subtask,
        result: AgentResult,
        feedback: dict[str, Any],
        control_step: int,
    ) -> None:
        if subtask.agent != AgentName.NAVIGATION:
            return
        room_id = self._first_text(
            feedback.get("room_id"),
            feedback.get("current_room"),
            subtask.target.get("room"),
        )
        region_id = self._first_text(
            feedback.get("current_region"),
            feedback.get("region_id"),
            subtask.target.get("region"),
            subtask.target.get("room"),
        )
        payload = {
            "source": "runtime_navigation_feedback",
            "agent": subtask.agent.value,
            "subtask_id": subtask.subtask_id,
            "control_step": control_step,
            "status": result.status.value,
            "room_id": room_id,
            "region_id": region_id,
            "object_id": self._first_text(
                subtask.target.get("object_id"),
                subtask.target.get("object"),
                subtask.target.get("object_name"),
            ),
            "nav_feedback": self._compact_navigation_feedback(feedback),
        }
        selected = result.runtime_artifacts.get("selected_object_approach")
        if isinstance(selected, dict) and selected:
            payload["selected_object_approach"] = self._compact_runtime_mapping(
                selected,
                keys=(
                    "candidate_id",
                    "room_id",
                    "room_name",
                    "floor_id",
                    "approach_distance_m",
                    "handoff_distance_m",
                    "path_cost",
                    "blocked_by_history",
                ),
            )
        self._record_working_observation(payload)

    def _record_working_observation(self, observation: dict[str, Any]) -> None:
        clean = {
            key: value for key, value in observation.items() if value not in (None, "", [], {})
        }
        record = getattr(self.brain_agent.memory, "record_working_observation", None)
        if not clean or not callable(record):
            return
        try:
            record(clean)
        except Exception as exc:
            logger.debug("working-memory observation write failed: %s", exc)

    @staticmethod
    def _compact_navigation_feedback(feedback: dict[str, Any]) -> dict[str, Any]:
        return ClosedLoopOrchestrator._compact_runtime_mapping(
            feedback,
            keys=(
                "step_count",
                "current_room",
                "current_region",
                "room_id",
                "floor_id",
                "path_backend",
                "best_distance_to_waypoint",
                "goal_reached",
                "loop_detected",
                "oscillation_detected",
                "steps_since_progress",
                "global_step",
                "subtask_steps",
                "budget",
            ),
        )

    @staticmethod
    def _compact_runtime_mapping(
        value: dict[str, Any],
        *,
        keys: tuple[str, ...],
    ) -> dict[str, Any]:
        return {key: value.get(key) for key in keys if value.get(key) not in (None, "", [], {})}

    @staticmethod
    def _first_text(*values: Any) -> str | None:
        for value in values:
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def _extract_robot_state(
        *,
        runtime_inputs: dict[str, Any],
        env_feedback: dict[str, Any],
    ) -> dict[str, Any]:
        return step_runner.extract_robot_state(
            runtime_inputs=runtime_inputs,
            env_feedback=env_feedback,
        )

    @staticmethod
    def _serialize_runtime_feedback(feedback: Any) -> dict[str, Any]:
        return step_runner.serialize_runtime_feedback(feedback)

    def _record_success_from_result(
        self,
        *,
        context: ExecutionContext,
        subtask: Subtask,
        result: AgentResult,
        latest_object_approach: dict[str, Any] | None,
    ) -> None:
        completion_policy.record_success_from_result(
            memory=self.brain_agent.memory,
            context=context,
            subtask=subtask,
            result=result,
            latest_object_approach=latest_object_approach,
        )

    def _record_failure_from_result(
        self,
        *,
        context: ExecutionContext,
        subtask: Subtask,
        failure_reason: str,
        latest_object_approach: dict[str, Any] | None,
    ) -> None:
        completion_policy.record_failure_from_result(
            memory=self.brain_agent.memory,
            context=context,
            subtask=subtask,
            result=None,
            failure_reason=failure_reason,
            latest_object_approach=latest_object_approach,
        )

    def _flush_pending_object_approach_outcome(
        self,
        *,
        context: ExecutionContext,
        success: bool,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        completion_policy.flush_pending_object_approach_outcome(
            memory=self.brain_agent.memory,
            context=context,
            success=success,
            reason=reason,
            metadata=metadata,
        )

    def _record_object_approach_outcome(
        self,
        *,
        approach: dict[str, Any],
        outcome: str,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        completion_policy.record_object_approach_outcome(
            memory=self.brain_agent.memory,
            approach=approach,
            outcome=outcome,
            reason=reason,
            metadata=metadata,
        )

    @staticmethod
    def _extract_object_approach_context(
        *,
        subtask: Subtask,
        result: AgentResult,
    ) -> dict[str, Any] | None:
        return completion_policy.extract_object_approach_context(
            subtask=subtask,
            result=result,
        )

    def _close_runtime_resources(self) -> None:
        closed: set[int] = set()
        for agent in self._agents.values():
            for attr in ("policy", "navigator"):
                resource = getattr(agent, attr, None)
                if resource is None or id(resource) in closed:
                    continue
                close_fn = getattr(resource, "close", None)
                if callable(close_fn):
                    close_fn()
                    closed.add(id(resource))

    def _cleanup_after_run(self, *, environment: RuntimeEnvironment) -> list[dict[str, str]]:
        cleanup_errors: list[dict[str, str]] = []
        cleanup_steps = (
            ("vision_heartbeat", lambda: self._stop_vision_heartbeat(flush=True)),
            ("runtime_resources", self._close_runtime_resources),
            ("environment_close", environment.close),
        )
        for step_name, cleanup_fn in cleanup_steps:
            try:
                cleanup_fn()
            except Exception as exc:
                logger.warning(
                    "Closed-loop cleanup step failed: %s: %s: %s",
                    step_name,
                    type(exc).__name__,
                    exc,
                )
                cleanup_errors.append(
                    {
                        "step": step_name,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        return cleanup_errors

    def _emit_event(
        self,
        *,
        event_type: str,
        source: str,
        message: str,
        payload: dict[str, Any] | None = None,
        task_id: str | None = None,
    ) -> None:
        if self.event_sink is None:
            return
        self.event_sink(
            VoltronEvent(
                event_type=event_type,
                source=source,
                message=message,
                payload=dict(payload or {}),
                task_id=task_id,
            )
        )

    @staticmethod
    def _update_environment_plan(
        environment: RuntimeEnvironment,
        context: ExecutionContext,
        plan: Plan,
    ) -> None:
        replanning_flow.update_environment_plan(
            environment=environment,
            context=context,
            plan=plan,
        )

    @staticmethod
    def _plan_changed(previous: Plan, current: Plan) -> bool:
        return replanning_flow.plan_changed(previous=previous, current=current)

    @staticmethod
    def _serialize_plan_event_payload(
        *,
        plan: Plan,
        reason: str,
        failure_reason: str | None = None,
        attempt: int | None = None,
    ) -> dict[str, Any]:
        return _serialize_plan_event_payload(
            plan=plan,
            reason=reason,
            failure_reason=failure_reason,
            attempt=attempt,
        )


def _serialize_subtask(subtask: Subtask) -> dict[str, Any]:
    payload = {
        "subtask_id": subtask.subtask_id,
        "agent": subtask.agent.value,
        "action": subtask.action,
        "target": dict(subtask.target),
        "parameters": dict(subtask.parameters),
        "context": dict(subtask.context),
    }
    if subtask.execution_id:
        payload["execution_id"] = subtask.execution_id
        payload["plan_revision"] = subtask.plan_revision
    if subtask.replaces_execution_id:
        payload["replaces_execution_id"] = subtask.replaces_execution_id
    return payload


def _merge_runtime_state(runtime_state: dict[str, Any], updates: dict[str, Any]) -> None:
    task_context = runtime_state.setdefault("task_context", {})
    if isinstance(task_context, dict):
        _deep_merge(task_context, updates)


def _deep_merge(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def _serialize_plan_event_payload(
    *,
    plan: Plan,
    reason: str,
    failure_reason: str | None = None,
    attempt: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "reason": reason,
        "metadata": dict(plan.metadata),
        "subtask_count": len(plan.subtasks),
        "subtasks": [_serialize_subtask(item) for item in plan.subtasks],
    }
    if failure_reason:
        payload["failure_reason"] = failure_reason
    if attempt is not None:
        payload["attempt"] = attempt
    return payload
