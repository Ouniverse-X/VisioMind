"""Tool-aware planning loop for the Brain agent."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from voltron.agents.brain.body.planner_backend import PlannerResponse
from voltron.agents.brain.tools.base import ContextPatch, default_context_patch, default_tool_trace_entry
from voltron.shared.context import Plan, Subtask
from voltron.shared.contracts import MemoryAdapter, ToolInvocation
from voltron.shared.registries import ToolCatalog
from voltron.shared.results import ToolResult


@dataclass(frozen=True)
class BrainPlanningEvent:
    """Observable planning-loop event for CLI/debug surfaces."""

    event_type: str
    message: str
    payload: dict[str, Any] = field(default_factory=dict)


class BrainPlanningLoop:
    """Drive Brain planner iterations until a final Plan is produced."""

    def __init__(
        self,
        *,
        memory: MemoryAdapter,
        planner: Any,
        tools: ToolCatalog,
        max_iterations: int = 6,
        event_sink: Any | None = None,
    ) -> None:
        self.memory = memory
        self.planner = planner
        self.tools = tools
        self.max_iterations = max(1, int(max_iterations))
        self.event_sink = event_sink

    def is_enabled(self) -> bool:
        return callable(getattr(self.planner, "plan_structured", None))

    def run_initial(
        self,
        *,
        task_description: str,
        planning_context: dict[str, Any],
        invoke_tool: Any,
    ) -> Plan:
        return self.run(
            mode="initial",
            task_description=task_description,
            planning_context=planning_context,
            invoke_tool=invoke_tool,
        )

    def run_next_step(
        self,
        *,
        task_description: str,
        planning_context: dict[str, Any],
        execution_state: dict[str, Any],
        invoke_tool: Any,
    ) -> Plan:
        return self.run(
            mode="next_step",
            task_description=task_description,
            planning_context=planning_context,
            execution_state=execution_state,
            invoke_tool=invoke_tool,
        )

    def run_replan(
        self,
        *,
        task_description: str,
        planning_context: dict[str, Any],
        failed_subtask: Subtask,
        failure_reason: str,
        execution_state: dict[str, Any],
        invoke_tool: Any,
    ) -> Plan:
        return self.run(
            mode="replan",
            task_description=task_description,
            planning_context=planning_context,
            execution_state=execution_state,
            failed_subtask=failed_subtask,
            failure_reason=failure_reason,
            invoke_tool=invoke_tool,
        )

    def run(
        self,
        *,
        mode: str,
        task_description: str,
        planning_context: dict[str, Any],
        invoke_tool: Any,
        execution_state: dict[str, Any] | None = None,
        failed_subtask: Subtask | None = None,
        failure_reason: str | None = None,
    ) -> Plan:
        if not self.is_enabled():
            raise RuntimeError("BrainPlanningLoop requires planner.plan_structured")

        self._seed_loop_context(planning_context)
        self._emit(
            "loop_start",
            f"Brain planning started in {mode} mode",
            {"mode": mode, "available_tools": planning_context.get("available_tools", [])},
        )

        for iteration in range(1, self.max_iterations + 1):
            response: PlannerResponse = self.planner.plan_structured(
                mode=mode,
                task_description=task_description,
                context=planning_context,
                execution_state=execution_state,
                failed_subtask=failed_subtask,
                failure_reason=failure_reason,
            )
            self._emit(
                "planner_step",
                f"Planner produced {response.kind}",
                {
                    "iteration": iteration,
                    "kind": response.kind,
                    "tool_name": response.tool_name,
                    "thinking_summary": response.thinking_summary,
                },
            )
            if response.kind == "final_plan":
                if response.plan is None:
                    raise ValueError("PlannerResponse final_plan is missing plan")
                self._emit(
                    "final_plan",
                    "Planner emitted final plan",
                    {
                        "subtask_count": len(response.plan.subtasks),
                        "metadata": dict(response.plan.metadata),
                    },
                )
                return response.plan
            if response.kind != "tool_call" or not response.tool_name:
                raise ValueError(f"Unsupported planner response kind {response.kind!r}")

            invocation = ToolInvocation(
                tool_name=response.tool_name,
                payload=dict(response.tool_payload or {}),
                metadata={"planner_mode": mode},
            )
            self._emit(
                "tool_call",
                f"Calling Brain tool {invocation.tool_name}",
                {"tool_name": invocation.tool_name, "tool_payload": dict(invocation.payload)},
            )
            result = invoke_tool(invocation)
            self._emit(
                "tool_result",
                f"Brain tool {invocation.tool_name} returned {'ok' if result.ok else 'error'}",
                {
                    "tool_name": invocation.tool_name,
                    "ok": result.ok,
                    "error_code": result.error_code,
                    "payload": dict(result.payload),
                    "metadata": dict(result.metadata),
                },
            )
            self._apply_tool_result(
                planning_context=planning_context,
                invocation=invocation,
                result=result,
            )

        raise RuntimeError(f"Brain planner exceeded {self.max_iterations} tool-aware iterations")

    def _seed_loop_context(self, planning_context: dict[str, Any]) -> None:
        planning_context.setdefault("available_tools", self._available_tool_specs())
        planning_context.setdefault("tool_trace", [])
        planning_context.setdefault("external_constraints", {})
        planning_context.setdefault("schedule_state", {})

    def _available_tool_specs(self) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []
        for tool_name in self.tools.names():
            tool = self.tools.get(tool_name)
            describe_tool = getattr(tool, "describe_tool", None)
            if callable(describe_tool):
                spec = describe_tool(tool_name)
                if isinstance(spec, dict):
                    specs.append(dict(spec))
                    continue
            specs.append({"name": tool_name})
        return specs

    def _apply_tool_result(
        self,
        *,
        planning_context: dict[str, Any],
        invocation: ToolInvocation,
        result: ToolResult,
    ) -> None:
        patch = self._build_context_patch(invocation=invocation, result=result)
        trace_entry = patch.tool_trace_entry or default_tool_trace_entry(invocation.tool_name, result)
        planning_context.setdefault("tool_trace", []).append(trace_entry)
        planning_context_updates = dict(patch.planning_context_updates)
        if planning_context_updates:
            _deep_update(planning_context, planning_context_updates)

        brain_tool_state = {
            "external_constraints": deepcopy(planning_context.get("external_constraints", {})),
            "schedule_state": deepcopy(planning_context.get("schedule_state", {})),
            "tool_outputs": deepcopy(planning_context.get("tool_outputs", {})),
            "tool_trace": deepcopy(planning_context.get("tool_trace", [])),
        }
        task_context_updates = _merge_dicts(dict(patch.task_context_updates), {"brain_tool_state": brain_tool_state})
        task_context = planning_context.get("task_context")
        merged_task_context = _merge_dicts(
            task_context if isinstance(task_context, dict) else {},
            task_context_updates,
        )
        planning_context["task_context"] = merged_task_context
        self.memory.update_task_context(task_context_updates)
        self._emit(
            "context_patch",
            f"Applied context patch from {invocation.tool_name}",
            {
                "tool_name": invocation.tool_name,
                "planning_context_keys": sorted(patch.planning_context_updates.keys()),
                "task_context_keys": sorted(task_context_updates.keys()),
            },
        )

    def _build_context_patch(self, *, invocation: ToolInvocation, result: ToolResult) -> ContextPatch:
        try:
            tool = self.tools.get(invocation.tool_name)
        except KeyError:
            return default_context_patch(invocation, result)

        build_context_patch = getattr(tool, "build_context_patch", None)
        if not callable(build_context_patch):
            return default_context_patch(invocation, result)

        patch = build_context_patch(invocation, result)
        if not isinstance(patch, ContextPatch):
            raise TypeError(f"Brain tool {invocation.tool_name!r} returned invalid context patch")
        return patch

    def _emit(self, event_type: str, message: str, payload: dict[str, Any] | None = None) -> None:
        if self.event_sink is None:
            return
        self.event_sink(BrainPlanningEvent(event_type=event_type, message=message, payload=dict(payload or {})))


def _merge_dicts(base: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in delta.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(current, value)
        else:
            merged[key] = value
    return merged


def _deep_update(base: dict[str, Any], delta: dict[str, Any]) -> None:
    for key, value in delta.items():
        current = base.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            _deep_update(current, value)
        else:
            base[key] = value


__all__ = ["BrainPlanningEvent", "BrainPlanningLoop"]
