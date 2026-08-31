from __future__ import annotations

import os
from typing import Any, Callable

import requests

from visiomind.action.shared.contracts import ToolInvocation
from visiomind.action.shared.results import ToolResult
from visiomind.action.agents.brain.tools.base import ContextPatch, default_tool_trace_entry

TOOL_VERSION = "web_search/0.1.0"

_FIELD_TRIP_FIXTURE = {
    "facts": {
        "event": "outdoor_field_trip",
        "weather": "warm",
        "microwave_available": False,
    },
    "constraints": {
        "required_items": ["water_bottle", "portable_lunch_container", "backpack"],
        "preferred_categories": ["sandwich", "fruit", "snack", "field_trip_note"],
        "avoid_categories": ["leftovers_requiring_reheating", "open_liquid_container"],
    },
}


class WebSearchTool:
    tool_names = ("web_search.lookup_constraints",)

    def __init__(self, http_get: Callable[..., Any] | None = None) -> None:
        self._http_get = http_get or requests.get

    def describe_tool(self, tool_name: str) -> dict[str, Any]:
        return {
            "name": tool_name,
            "description": "Look up external planning constraints from fixture data or a configured HTTP search endpoint.",
            "input_schema": {
                "query": "natural-language search query",
                "mode": '"mock" | "real"',
                "endpoint": "required only for real mode when VISIOMIND_ACTION_WEB_SEARCH_ENDPOINT is unset",
            },
        }

    def build_context_patch(self, invocation: ToolInvocation, result: ToolResult) -> ContextPatch:
        return ContextPatch(
            planning_context_updates={
                "external_constraints": {invocation.tool_name: dict(result.payload)},
            },
            tool_trace_entry=default_tool_trace_entry(invocation.tool_name, result),
        )

    def invoke(self, invocation: ToolInvocation) -> ToolResult:
        if invocation.tool_name != "web_search.lookup_constraints":
            return self._error(
                invocation.tool_name,
                "unsupported_tool",
                f"Unsupported web search tool {invocation.tool_name!r}",
            )
        payload = invocation.payload
        mode = str(payload.get("mode") or "mock").strip().lower()
        if mode == "real":
            return self._lookup_real(invocation)
        return self._lookup_mock(invocation)

    def _lookup_mock(self, invocation: ToolInvocation) -> ToolResult:
        query = str(invocation.payload.get("query") or "").strip()
        fixture = _FIELD_TRIP_FIXTURE
        return ToolResult(
            tool_name=invocation.tool_name,
            ok=True,
            payload={
                "mode": "mock",
                "query": query,
                "facts": dict(fixture["facts"]),
                "constraints": _copy_constraints(fixture["constraints"]),
                "source": "fixture",
            },
            metadata={"tool_version": TOOL_VERSION},
        )

    def _lookup_real(self, invocation: ToolInvocation) -> ToolResult:
        payload = invocation.payload
        query = str(payload.get("query") or "").strip()
        endpoint = str(
            payload.get("endpoint") or os.getenv("VISIOMIND_ACTION_WEB_SEARCH_ENDPOINT") or ""
        ).strip()
        if not endpoint:
            return self._error(
                invocation.tool_name,
                "missing_real_search_endpoint",
                "Real web_search mode requires endpoint or VISIOMIND_ACTION_WEB_SEARCH_ENDPOINT",
            )

        api_key = str(
            payload.get("api_key") or os.getenv("VISIOMIND_ACTION_WEB_SEARCH_API_KEY") or ""
        ).strip()
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            response = self._http_get(
                endpoint,
                params={"q": query},
                headers=headers,
                timeout=float(payload.get("timeout_s", 10.0)),
            )
            response.raise_for_status()
            raw = response.json()
        except requests.RequestException as exc:
            return self._error(invocation.tool_name, "real_search_request_failed", str(exc))
        except ValueError as exc:
            return self._error(invocation.tool_name, "real_search_invalid_json", str(exc))

        facts, constraints = _extract_constraints(raw)
        return ToolResult(
            tool_name=invocation.tool_name,
            ok=True,
            payload={
                "mode": "real",
                "query": query,
                "facts": facts,
                "constraints": constraints,
                "raw": raw,
                "source": endpoint,
            },
            metadata={"tool_version": TOOL_VERSION},
        )

    @staticmethod
    def _error(tool_name: str, error_code: str, message: str) -> ToolResult:
        return ToolResult(
            tool_name=tool_name,
            ok=False,
            payload={"message": message},
            error_code=error_code,
            metadata={"tool_version": TOOL_VERSION},
        )


def _extract_constraints(raw: Any) -> tuple[dict[str, Any], dict[str, list[str]]]:
    if isinstance(raw, dict):
        raw_facts = raw.get("facts")
        raw_constraints = raw.get("constraints")
        if isinstance(raw_facts, dict) and isinstance(raw_constraints, dict):
            return dict(raw_facts), _normalize_constraints(raw_constraints)
        text = _flatten_search_text(raw)
    else:
        text = str(raw)

    lower = text.lower()
    facts: dict[str, Any] = {
        "event": "outdoor_field_trip" if "field trip" in lower or "outdoor" in lower else "unknown",
        "weather": "warm" if "warm" in lower or "hot" in lower else "unknown",
        "microwave_available": False
        if "no microwave" in lower or "without microwave" in lower
        else None,
    }
    constraints = _copy_constraints(_FIELD_TRIP_FIXTURE["constraints"])
    if "nut" in lower and ("free" in lower or "allergy" in lower):
        constraints["avoid_categories"].append("nuts")
    return facts, constraints


def _flatten_search_text(raw: dict[str, Any]) -> str:
    values: list[str] = []
    for key in ("answer", "summary", "snippet", "content"):
        value = raw.get(key)
        if isinstance(value, str):
            values.append(value)
    for list_key in ("results", "organic_results", "items"):
        value = raw.get(list_key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    values.extend(str(item.get(key, "")) for key in ("title", "snippet", "content"))
                elif isinstance(item, str):
                    values.append(item)
    return "\n".join(values)


def _normalize_constraints(raw_constraints: dict[str, Any]) -> dict[str, list[str]]:
    normalized = {}
    for key in ("required_items", "preferred_categories", "avoid_categories"):
        value = raw_constraints.get(key, [])
        normalized[key] = [str(item) for item in value] if isinstance(value, list) else []
    return normalized


def _copy_constraints(constraints: dict[str, Any]) -> dict[str, list[str]]:
    return {key: list(value) for key, value in _normalize_constraints(constraints).items()}


__all__ = ["TOOL_VERSION", "WebSearchTool"]
