"""Scenario-clock and scheduled-event tools for the Brain agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

from voltron.shared.contracts import ToolInvocation
from voltron.shared.results import ToolResult
from voltron.agents.brain.tools.base import ContextPatch, default_tool_trace_entry

TOOL_VERSION = "cron/0.1.0"
_DEFAULT_MOCK_NOW = "2026-04-26T19:00:00"


@dataclass
class ScheduledEvent:
    event_id: str
    task: str
    scheduled_time: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "task": self.task,
            "scheduled_time": self.scheduled_time,
            "metadata": dict(self.metadata),
        }


class CronTool:
    """Brain-owned time constraint tool with deterministic and real-clock modes."""

    tool_names = ("cron.check_schedule", "cron.schedule_event", "cron.list_events", "cron.cancel_event")

    def __init__(self, now: Callable[[], datetime] | None = None) -> None:
        self._now = now or datetime.now
        self._events: dict[str, ScheduledEvent] = {}

    def describe_tool(self, tool_name: str) -> dict[str, Any]:
        descriptions = {
            "cron.check_schedule": {
                "description": "Inspect current simulated or real time, deadlines, and due scheduled events.",
                "input_schema": {
                    "mode": '"mock" | "real"',
                    "current_time": "optional ISO timestamp or relative time anchor",
                    "deadlines": [{"time": "ISO or relative time", "requirement": "constraint label"}],
                },
            },
            "cron.schedule_event": {
                "description": "Create a scheduled reminder or deferred task for a future time.",
                "input_schema": {
                    "mode": '"mock" | "real"',
                    "event_id": "stable event id",
                    "task": "human-readable task description",
                    "time": "ISO timestamp or relative time like tomorrow 10:00",
                },
            },
            "cron.list_events": {
                "description": "List scheduled events already registered in the Brain runtime.",
                "input_schema": {},
            },
            "cron.cancel_event": {
                "description": "Cancel one scheduled event by id.",
                "input_schema": {"event_id": "stable event id"},
            },
        }
        spec = descriptions.get(tool_name, {"description": "Brain cron tool", "input_schema": {}})
        return {"name": tool_name, **spec}

    def build_context_patch(self, invocation: ToolInvocation, result: ToolResult) -> ContextPatch:
        return ContextPatch(
            planning_context_updates={
                "schedule_state": {invocation.tool_name: dict(result.payload)},
            },
            tool_trace_entry=default_tool_trace_entry(invocation.tool_name, result),
        )

    def invoke(self, invocation: ToolInvocation) -> ToolResult:
        if invocation.tool_name == "cron.check_schedule":
            return self._check_schedule(invocation)
        if invocation.tool_name == "cron.schedule_event":
            return self._schedule_event(invocation)
        if invocation.tool_name == "cron.list_events":
            return self._list_events(invocation)
        if invocation.tool_name == "cron.cancel_event":
            return self._cancel_event(invocation)
        return self._error(invocation.tool_name, "unsupported_tool", f"Unsupported cron tool {invocation.tool_name!r}")

    def _check_schedule(self, invocation: ToolInvocation) -> ToolResult:
        payload = invocation.payload
        mode = _mode(payload)
        current_dt = self._current_time(payload, mode)
        current_iso = _iso(current_dt)
        deadlines = self._serialize_deadlines(payload.get("deadlines", []), current_dt)
        phase = str(payload.get("phase") or _phase_from_deadlines(deadlines) or "scheduled_execution")
        due_events = self._due_events(current_dt)

        return ToolResult(
            tool_name=invocation.tool_name,
            ok=True,
            payload={
                "mode": mode,
                "current_time": current_iso,
                "phase": phase,
                "deadlines": deadlines,
                "due_events": due_events,
                "scheduled_events": [event.to_payload() for event in self._sorted_events()],
            },
            metadata={"tool_version": TOOL_VERSION},
        )

    def _schedule_event(self, invocation: ToolInvocation) -> ToolResult:
        payload = invocation.payload
        event_id = str(payload.get("event_id") or "").strip()
        task = str(payload.get("task") or "").strip()
        time_value = payload.get("time") or payload.get("scheduled_time")
        if not event_id:
            return self._error(invocation.tool_name, "missing_event_id", "cron.schedule_event requires event_id")
        if not task:
            return self._error(invocation.tool_name, "missing_task", "cron.schedule_event requires task")
        if not time_value:
            return self._error(invocation.tool_name, "missing_time", "cron.schedule_event requires time")

        mode = _mode(payload)
        base_dt = self._current_time({"current_time": payload.get("base_time")}, mode)
        scheduled_dt = _parse_time(str(time_value), base_dt=base_dt)
        event = ScheduledEvent(
            event_id=event_id,
            task=task,
            scheduled_time=_iso(scheduled_dt),
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata"), dict) else {},
        )
        self._events[event_id] = event
        return ToolResult(
            tool_name=invocation.tool_name,
            ok=True,
            payload={"mode": mode, "event": event.to_payload()},
            metadata={"tool_version": TOOL_VERSION},
        )

    def _list_events(self, invocation: ToolInvocation) -> ToolResult:
        return ToolResult(
            tool_name=invocation.tool_name,
            ok=True,
            payload={"events": [event.to_payload() for event in self._sorted_events()]},
            metadata={"tool_version": TOOL_VERSION},
        )

    def _cancel_event(self, invocation: ToolInvocation) -> ToolResult:
        event_id = str(invocation.payload.get("event_id") or "").strip()
        if not event_id:
            return self._error(invocation.tool_name, "missing_event_id", "cron.cancel_event requires event_id")
        removed = self._events.pop(event_id, None)
        if removed is None:
            return self._error(invocation.tool_name, "event_not_found", f"Scheduled event {event_id!r} was not found")
        return ToolResult(
            tool_name=invocation.tool_name,
            ok=True,
            payload={"cancelled_event_id": event_id},
            metadata={"tool_version": TOOL_VERSION},
        )

    def _current_time(self, payload: dict[str, Any], mode: str) -> datetime:
        explicit_time = payload.get("current_time")
        if explicit_time:
            base_dt = _parse_time(_DEFAULT_MOCK_NOW, base_dt=None)
            return _parse_time(str(explicit_time), base_dt=base_dt)
        if mode == "real":
            return _normalize_datetime(self._now())
        return _parse_time(_DEFAULT_MOCK_NOW, base_dt=None)

    def _serialize_deadlines(self, deadlines: Any, current_dt: datetime) -> list[dict[str, Any]]:
        if not isinstance(deadlines, list):
            return []
        serialized = []
        for item in deadlines:
            if not isinstance(item, dict) or not item.get("time"):
                continue
            deadline_dt = _parse_time(str(item["time"]), base_dt=current_dt)
            status = _temporal_status(deadline_dt, current_dt)
            serialized.append(
                {
                    **dict(item),
                    "time": _iso(deadline_dt),
                    "status": status,
                    "overdue": status == "overdue",
                }
            )
        return serialized

    def _due_events(self, current_dt: datetime) -> list[dict[str, Any]]:
        due = []
        for event in self._sorted_events():
            event_dt = _parse_time(event.scheduled_time, base_dt=current_dt)
            if event_dt <= current_dt:
                payload = event.to_payload()
                payload["status"] = "due" if event_dt == current_dt else "overdue"
                due.append(payload)
        return due

    def _sorted_events(self) -> list[ScheduledEvent]:
        return sorted(self._events.values(), key=lambda event: (event.scheduled_time, event.event_id))

    @staticmethod
    def _error(tool_name: str, error_code: str, message: str) -> ToolResult:
        return ToolResult(
            tool_name=tool_name,
            ok=False,
            payload={"message": message},
            error_code=error_code,
            metadata={"tool_version": TOOL_VERSION},
        )


def _mode(payload: dict[str, Any]) -> str:
    mode = str(payload.get("mode") or "mock").strip().lower()
    return mode if mode in {"mock", "real"} else "mock"


def _parse_time(value: str, *, base_dt: datetime | None) -> datetime:
    normalized = value.strip()
    lower = normalized.lower()
    if lower.startswith("tomorrow "):
        if base_dt is None:
            base_dt = _parse_time(_DEFAULT_MOCK_NOW, base_dt=None)
        hour, minute = _parse_hour_minute(normalized.split(None, 1)[1])
        return base_dt.replace(hour=hour, minute=minute, second=0, microsecond=0) + timedelta(days=1)
    if lower.startswith("today "):
        if base_dt is None:
            base_dt = _parse_time(_DEFAULT_MOCK_NOW, base_dt=None)
        hour, minute = _parse_hour_minute(normalized.split(None, 1)[1])
        return base_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if len(normalized) == 5 and normalized[2] == ":":
        if base_dt is None:
            base_dt = _parse_time(_DEFAULT_MOCK_NOW, base_dt=None)
        hour, minute = _parse_hour_minute(normalized)
        return base_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return _normalize_datetime(datetime.fromisoformat(normalized.replace("Z", "+00:00")))


def _parse_hour_minute(value: str) -> tuple[int, int]:
    hour_text, minute_text = value.strip().split(":", 1)
    return int(hour_text), int(minute_text)


def _normalize_datetime(value: datetime) -> datetime:
    return value.replace(tzinfo=None, microsecond=0)


def _iso(value: datetime) -> str:
    return _normalize_datetime(value).isoformat()


def _temporal_status(target_dt: datetime, current_dt: datetime) -> str:
    if target_dt < current_dt:
        return "overdue"
    if target_dt == current_dt:
        return "due"
    return "upcoming"


def _phase_from_deadlines(deadlines: list[dict[str, Any]]) -> str | None:
    due_or_overdue = [deadline for deadline in deadlines if deadline.get("status") in {"due", "overdue"}]
    if due_or_overdue:
        return str(due_or_overdue[-1].get("requirement") or "deadline_due")
    if deadlines:
        return str(deadlines[0].get("requirement") or "scheduled_execution")
    return None


__all__ = ["CronTool", "ScheduledEvent", "TOOL_VERSION"]
