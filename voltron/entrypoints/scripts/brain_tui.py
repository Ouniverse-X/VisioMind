"""Minimal Brain-agent TUI inspired by Claude Code's event transcript."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, TextIO

from voltron.agents import BrainAgent
from voltron.agents.brain.body.planner_backend import OpenAICompatiblePlanner, OpenAIPlannerConfig, PlannerResponse
from voltron.agents.brain.body.planning_loop import BrainPlanningEvent
from voltron.runtime.testing import MockMemoryAdapter
from voltron.shared.context import Plan, Subtask, TaskRequest
from voltron.shared.enums import AgentName, TaskType


_RESET = "\033[0m"
_COLORS = {
    "USER": "\033[36m",
    "THINK": "\033[35m",
    "TOOL": "\033[33m",
    "RESULT": "\033[32m",
    "PATCH": "\033[34m",
    "PLAN": "\033[1m",
    "ERROR": "\033[31m",
    "SYSTEM": "\033[2m",
}


class DemoToolPlanner:
    """Deterministic tool-aware planner for local TUI demos."""

    def plan(self, task_description: str, context: dict[str, Any]) -> Plan:
        del task_description, context
        return self._final_plan()

    def plan_next(self, task_description: str, context: dict[str, Any], execution_state: dict[str, Any]) -> Plan:
        del task_description, context, execution_state
        return Plan(subtasks=[], metadata={"planner": "brain_tui_demo", "dynamic_execution": True})

    def replan(
        self,
        task_description: str,
        context: dict[str, Any],
        failed_subtask: Subtask,
        failure_reason: str,
        execution_state: dict[str, Any],
    ) -> Plan:
        del task_description, context, failed_subtask, failure_reason, execution_state
        return self._final_plan()

    def plan_structured(
        self,
        *,
        mode: str,
        task_description: str,
        context: dict[str, Any],
        execution_state: dict[str, Any] | None = None,
        failed_subtask: Subtask | None = None,
        failure_reason: str | None = None,
    ) -> PlannerResponse:
        del mode, execution_state, failed_subtask, failure_reason
        tool_trace = context.get("tool_trace") if isinstance(context.get("tool_trace"), list) else []
        called_tools = {str(item.get("tool_name")) for item in tool_trace if isinstance(item, dict)}
        lowered = task_description.lower()

        if _needs_schedule(lowered) and "cron.check_schedule" not in called_tools:
            return PlannerResponse(
                kind="tool_call",
                tool_name="cron.check_schedule",
                tool_payload={
                    "mode": "mock",
                    "deadlines": [{"time": "tomorrow 10:00", "requirement": "morning deadline"}],
                    "phase": "deadline_check",
                },
                thinking_summary="Need schedule state before choosing the task phase.",
            )
        if _needs_external_constraints(lowered) and "web_search.lookup_constraints" not in called_tools:
            return PlannerResponse(
                kind="tool_call",
                tool_name="web_search.lookup_constraints",
                tool_payload={"query": task_description, "mode": "mock"},
                thinking_summary="Need external constraints before selecting concrete items.",
            )
        return PlannerResponse(
            kind="final_plan",
            plan=self._final_plan(),
            thinking_summary="Tool context is sufficient; emit a short executable seed plan.",
        )

    @staticmethod
    def _final_plan() -> Plan:
        return Plan(
            subtasks=[
                Subtask(
                    subtask_id="st_01",
                    agent=AgentName.VISION,
                    action="inspect_scene",
                    target={"object": "task relevant items"},
                    parameters={"instruction": "Inspect the scene for items relevant to the requested task."},
                )
            ],
            metadata={"planner": "brain_tui_demo", "dynamic_execution": True},
        )


@dataclass
class TuiRenderer:
    use_color: bool = True
    stream: TextIO = sys.stdout

    def write(self, label: str, text: str) -> None:
        self.stream.write(self._format(label, text) + "\n")
        self.stream.flush()

    def write_event(self, event: BrainPlanningEvent) -> None:
        self.stream.write(self.render_event(event) + "\n")
        self.stream.flush()

    def render_event(self, event: BrainPlanningEvent) -> str:
        if event.event_type == "planner_step":
            summary = event.payload.get("thinking_summary") or event.message
            return self._format("THINK", str(summary))
        if event.event_type == "tool_call":
            return self._format(
                "TOOL",
                f"{event.payload.get('tool_name')} {json.dumps(event.payload.get('tool_payload', {}), ensure_ascii=False)}",
            )
        if event.event_type == "tool_result":
            preview = _compact_json(event.payload.get("payload", {}))
            return self._format("RESULT", f"{event.payload.get('tool_name')} ok={event.payload.get('ok')} {preview}")
        if event.event_type == "context_patch":
            return self._format(
                "PATCH",
                f"{event.payload.get('tool_name')} context={event.payload.get('planning_context_keys', [])}",
            )
        if event.event_type == "final_plan":
            return self._format("PLAN", f"final plan with {event.payload.get('subtask_count', 0)} subtasks")
        return self._format("SYSTEM", event.message)

    def _format(self, label: str, text: str) -> str:
        prefix = f"[{label}]"
        if not self.use_color:
            return f"{prefix} {text}"
        color = _COLORS.get(label, "")
        return f"{color}{prefix}{_RESET} {text}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive Voltron Brain tool-loop TUI")
    parser.add_argument("--planner", choices=("demo", "openai"), default="demo")
    parser.add_argument("--base-url", default=os.getenv("VOLTRON_OPENAI_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--model", default=os.getenv("VOLTRON_OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--no-color", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    renderer = TuiRenderer(use_color=not args.no_color)
    memory = MockMemoryAdapter()
    planner = _build_planner(args)
    brain = BrainAgent(memory=memory, planner=planner)
    brain.planning_loop.event_sink = renderer.write_event

    renderer.write("SYSTEM", "Voltron Brain TUI. Type /help for commands, /quit to exit.")
    while True:
        try:
            user_text = input("voltron> ").strip()
        except (EOFError, KeyboardInterrupt):
            renderer.write("SYSTEM", "exit")
            return 0
        if not user_text:
            continue
        if user_text in {"/quit", "/exit"}:
            return 0
        if user_text == "/help":
            renderer.write("SYSTEM", "Enter a task request. Commands: /help, /tools, /context, /quit")
            continue
        if user_text == "/tools":
            renderer.write("SYSTEM", ", ".join(brain.tools.names()))
            continue
        if user_text == "/context":
            renderer.write("SYSTEM", _compact_json(memory.get_task_context()))
            continue

        renderer.write("USER", user_text)
        try:
            request = TaskRequest(
                task_id=f"tui_task_{len(memory.task_context_updates):03d}",
                description=user_text,
                task_type=TaskType.MANIPULATION,
            )
            _, plan = brain.prepare(request)
            renderer.write("PLAN", _format_plan(plan))
        except Exception as exc:
            renderer.write("ERROR", f"{type(exc).__name__}: {exc}")
    return 0


def _build_planner(args: argparse.Namespace) -> Any:
    if args.planner == "openai":
        return OpenAICompatiblePlanner(
            OpenAIPlannerConfig(
                base_url=args.base_url,
                model=args.model,
                api_key_env=args.api_key_env,
            )
        )
    return DemoToolPlanner()


def _needs_schedule(text: str) -> bool:
    return any(term in text for term in ("tomorrow", "morning", "10", "schedule", "deadline", "明早", "十点", "时间"))


def _needs_external_constraints(text: str) -> bool:
    return any(term in text for term in ("field trip", "lunch", "food", "web", "search", "外出", "午餐", "食物", "联网"))


def _format_plan(plan: Plan) -> str:
    rows = []
    for subtask in plan.subtasks:
        rows.append(f"{subtask.subtask_id}:{subtask.agent.value}:{subtask.action}:{subtask.target}")
    return "; ".join(rows) if rows else "no remaining subtasks"


def _compact_json(value: Any, *, max_len: int = 320) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


if __name__ == "__main__":
    raise SystemExit(main())
