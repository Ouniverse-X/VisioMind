"""Task orchestration engine for open-loop Voltron execution."""

from __future__ import annotations

from typing import Any

from .agent_episode_runtime import OpenLoopAgentEpisodeRuntime
from voltron.runtime.orchestrator.agent_bindings import resolve_orchestrator_agents
from voltron.runtime.telemetry.run_logger import build_task_run_response
from voltron.shared.context import ExecutionContext, Plan, Subtask, TaskRequest
from voltron.shared.contracts import SubtaskAgent
from voltron.shared.enums import AgentName, AgentStatus
from voltron.shared.results import AgentResult


class VoltronOrchestrator:
    """Coordinate Brain/Vision/Navigation/Action agents under one task lifecycle."""

    def __init__(
        self,
        brain_agent: Any | None = None,
        vision_agent: SubtaskAgent | None = None,
        navigation_agent: SubtaskAgent | None = None,
        action_agent: SubtaskAgent | None = None,
        max_retries: int = 1,
        max_control_steps_per_subtask: int = 1,
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
        self.max_control_steps_per_subtask = max(1, int(max_control_steps_per_subtask))
        self._agents: dict[AgentName, SubtaskAgent] = {
            AgentName.VISION: self.vision_agent,
            AgentName.NAVIGATION: self.navigation_agent,
            AgentName.ACTION: self.action_agent,
        }

    def run_task(
        self,
        request: TaskRequest,
        runtime_inputs: dict[str, dict[str, Any]] | None = None,
        plan_override: Plan | None = None,
    ) -> dict[str, Any]:
        runtime_inputs = runtime_inputs or {}
        context, plan = self.brain_agent.prepare(request, plan_override=plan_override)

        for subtask in plan.subtasks:
            self._inject_runtime_inputs(subtask, runtime_inputs.get(subtask.subtask_id, {}))
            result = self._execute_with_retry(
                request=request,
                context=context,
                subtask=subtask,
                runtime_inputs=runtime_inputs,
            )
            context.results.append(result)

            if result.status == AgentStatus.FAILURE:
                final = self.brain_agent.finalize(success=False, failure_reason=result.error_code)
                return build_task_run_response(context, final)

        final = self.brain_agent.finalize(success=True)
        return build_task_run_response(context, final)

    def _execute_with_retry(
        self,
        request: TaskRequest,
        context: ExecutionContext,
        subtask: Subtask,
        runtime_inputs: dict[str, dict[str, Any]],
    ) -> AgentResult:
        current_subtask = subtask
        attempts = 0
        last_result: AgentResult | None = None

        while attempts <= self.max_retries:
            agent = self._agents[current_subtask.agent]
            runtime_payload = runtime_inputs.get(current_subtask.subtask_id, {})
            result = self._run_agent_episode_or_execute(
                agent=agent,
                subtask=current_subtask,
                context=context,
                runtime_payload=runtime_payload,
                attempt=attempts + 1,
            )
            result.result.setdefault("agent", current_subtask.agent.value)
            result.result.setdefault("attempt", attempts + 1)

            if result.status == AgentStatus.SUCCESS:
                return result

            attempts += 1
            last_result = result
            if attempts > self.max_retries:
                break

            failure_reason = result.error_code or "UNKNOWN_ERROR"
            replanned = self.brain_agent.replan(
                request=request,
                context=context,
                failed_subtask=current_subtask,
                failure_reason=failure_reason,
                latest_result=result,
            )
            if not replanned.subtasks:
                break

            current_subtask = replanned.subtasks[0]
            self._inject_runtime_inputs(
                current_subtask,
                runtime_inputs.get(current_subtask.subtask_id, {}),
            )

        if last_result is None:
            raise RuntimeError("_execute_with_retry reached impossible state")
        return last_result

    def _run_agent_episode_or_execute(
        self,
        *,
        agent: SubtaskAgent,
        subtask: Subtask,
        context: ExecutionContext,
        runtime_payload: dict[str, Any],
        attempt: int,
    ) -> AgentResult:
        episode_runner = getattr(agent, "run_episode", None)
        if callable(episode_runner):
            runtime = OpenLoopAgentEpisodeRuntime(
                runtime_inputs=dict(runtime_payload),
                attempt=attempt,
                max_control_steps=self.max_control_steps_per_subtask,
            )
            return episode_runner(subtask=subtask, context=context, runtime=runtime)
        return agent.execute(subtask, context)

    @staticmethod
    def _inject_runtime_inputs(subtask: Subtask, runtime_payload: dict[str, Any]) -> None:
        if not runtime_payload:
            return
        subtask.parameters = {**subtask.parameters, **runtime_payload}
