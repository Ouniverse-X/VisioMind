from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

import requests

from visiomind.action.agents.brain.contracts import TaskPlanner
from visiomind.action.agents.brain.skills import (
    DefaultBrainNextStepSkill,
    DefaultBrainPlanningSkill,
    DefaultBrainReplanningSkill,
)
from visiomind.action.agents.brain.skills.planning.skill import (
    ACTION_INSTRUCTION_GUIDANCE,
    NAVIGATION_INSTRUCTION_GUIDANCE,
)
from visiomind.action.shared.context import Plan, Subtask

_RETRIABLE_STATUS_CODES = {
    HTTPStatus.REQUEST_TIMEOUT,
    HTTPStatus.TOO_MANY_REQUESTS,
    HTTPStatus.INTERNAL_SERVER_ERROR,
    HTTPStatus.BAD_GATEWAY,
    HTTPStatus.SERVICE_UNAVAILABLE,
    HTTPStatus.GATEWAY_TIMEOUT,
}

_TOOL_LOOP_SYSTEM_PROMPT = """You are the VisioMindAction task planner for an embodied robot.
Return valid JSON only. Do not include markdown fences or extra text.

You may produce exactly one of these top-level JSON objects:
1. A tool request:
{
  "kind": "tool_call",
  "tool_name": "registered Brain tool name",
  "tool_payload": {"tool-specific": "arguments"}
}
2. A final plan:
{
  "kind": "final_plan",
  "plan": {
    "subtasks": [
      {
        "subtask_id": "st_01",
        "agent": "NAVIGATION" | "VISION" | "ACTION",
        "action": "short_action_name",
        "target": {
          "object": "target object name when applicable",
          "part": "target part for local interaction when applicable",
          "region": "target region when applicable",
          "room": "target room when applicable",
          "room_name": "numbered internal room instance name when available",
          "room_id": "internal room id when available"
        },
        "parameters": {"optional": "planner/runtime parameters"},
        "context": {"optional": "extra context"}
      }
    ],
    "metadata": {"planner": "openai_compatible"}
  }
}

Use canonical target keys only: `object`, `part`, `region`, `room`, `room_name`, `room_id`.
Do not use `target.name`.

Agent meanings:
- NAVIGATION: global navigation / locomotion / long-range approach / re-localization.
- VISION: visual observation / localization / verification from images.
- ACTION: local interaction / grasp / press / toggle / place. ACTION may use local whole-body base adjustment together with the arms when the target is visible in the same room and only local approach/alignment is needed.

Use `agent_capabilities` from planning context as the preferred source for tool- or skill-specific routing. When a user request matches a declared capability, emit a subtask for that capability's `agent` using one of its `action_names` and compatible `parameters`.

Planning rules:
- Keep plans concise and executable.
- Use ordered subtasks that match the user task.
- Use at most one tool call per response.
- If existing tool_trace or external constraints already answer the question, do not call the same tool again.
- Interpret natural-language intent yourself from task text and runtime context. Do not rely on pre-grounded metadata target IDs, object IDs, or room IDs as the answer.
- Preserve metric displacement commands as Navigation instructions with measurable direction and distance.
- Preserve portal or threshold commands as Navigation instructions toward a passage, entrance, doorway, boundary, or transition area.
- Infer affordance-bearing object interaction commands semantically, including controls, switches, buttons, handles, knobs, and appliance controls, without fabricating graph object IDs.
- __NAVIGATION_INSTRUCTION_GUIDANCE__
- A final VISION verification subtask such as `verify`, `check`, or `confirm` may appear after an execution step. Only set `"parameters": {"allow_task_complete": true}` on this final verification subtask when its purpose is to verify the overall task completion condition.
- For tasks that involve local interaction, prefer dynamic execution.
- When emitting a local interaction ACTION step, set `"parameters": {"control_mode": "whole_body_local"}` unless already specified.
- __ACTION_INSTRUCTION_GUIDANCE__
""".replace("__ACTION_INSTRUCTION_GUIDANCE__", ACTION_INSTRUCTION_GUIDANCE).replace(
    "__NAVIGATION_INSTRUCTION_GUIDANCE__", NAVIGATION_INSTRUCTION_GUIDANCE
)


@dataclass(frozen=True)
class OpenAIPlannerConfig:
    base_url: str
    model: str
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: float = 30.0
    temperature: float = 0.1
    max_retries: int = 0
    retry_backoff_s: float = 1.0
    semantic_validation_retries: int = 1


@dataclass(frozen=True)
class PlannerResponse:
    kind: str
    plan: Plan | None = None
    tool_name: str | None = None
    tool_payload: dict[str, Any] | None = None
    thinking_summary: str | None = None


class OpenAICompatiblePlanner(TaskPlanner):
    def __init__(self, config: OpenAIPlannerConfig):
        self.config = config
        self.session = requests.Session()
        self.planning_skill = DefaultBrainPlanningSkill()
        self.next_step_skill = DefaultBrainNextStepSkill()
        self.replanning_skill = DefaultBrainReplanningSkill()

    def plan(self, task_description: str, context: dict) -> Plan:
        prompt = self.planning_skill.build_prompt(
            task_description=task_description, context=context
        )
        return self._request_and_validate_plan(
            prompt,
            allow_empty=False,
            default_dynamic=self._should_enable_dynamic_execution(context),
        )

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
        prompt = self._build_structured_prompt(
            mode=mode,
            task_description=task_description,
            context=context,
            execution_state=execution_state,
            failed_subtask=failed_subtask,
            failure_reason=failure_reason,
        )
        attempts = max(1, int(self.config.semantic_validation_retries) + 1)
        current_prompt = prompt
        last_error: ValueError | None = None
        for attempt in range(1, attempts + 1):
            content = self._request_chat_completion(
                current_prompt, system_prompt=_TOOL_LOOP_SYSTEM_PROMPT
            )
            try:
                return self._parse_structured_response(
                    content,
                    context=context,
                    allow_empty=(mode == "next_step"),
                    default_dynamic=self._should_enable_dynamic_execution(context),
                )
            except ValueError as exc:
                last_error = exc
                if attempt >= attempts:
                    raise
                current_prompt = self.planning_skill.build_validation_retry_prompt(
                    prompt, error=str(exc)
                )

        if last_error is not None:
            raise last_error
        raise RuntimeError("Brain structured planner validation failed without error detail")

    def plan_next(
        self,
        task_description: str,
        context: dict,
        execution_state: dict,
    ) -> Plan:
        prompt = self.next_step_skill.build_prompt(
            task_description=task_description,
            context=context,
            execution_state=execution_state,
        )
        return self._request_and_validate_plan(
            prompt,
            allow_empty=True,
            default_dynamic=self._should_enable_dynamic_execution(context),
        )

    def replan(
        self,
        task_description: str,
        context: dict,
        failed_subtask: Subtask,
        failure_reason: str,
        execution_state: dict,
    ) -> Plan:
        prompt = self.replanning_skill.build_prompt(
            task_description=task_description,
            context=context,
            failed_subtask=failed_subtask,
            failure_reason=failure_reason,
            execution_state=execution_state,
        )
        return self._request_and_validate_plan(
            prompt,
            allow_empty=False,
            default_dynamic=self._should_enable_dynamic_execution(context),
        )

    def _request_and_validate_plan(
        self,
        prompt: str,
        *,
        allow_empty: bool,
        default_dynamic: bool,
    ) -> Plan:
        attempts = max(1, int(self.config.semantic_validation_retries) + 1)
        current_prompt = prompt
        last_error: ValueError | None = None

        for attempt in range(1, attempts + 1):
            content = self._request_chat_completion(current_prompt)
            try:
                plan = self.planning_skill.parse_plan_response(
                    content,
                    model=self.config.model,
                    allow_empty=allow_empty,
                    default_dynamic=default_dynamic,
                )
                self.planning_skill.validate_plan_semantics(plan)
                return plan
            except ValueError as exc:
                last_error = exc
                if attempt >= attempts:
                    raise
                current_prompt = self.planning_skill.build_validation_retry_prompt(
                    prompt, error=str(exc)
                )

        if last_error is not None:
            raise last_error
        raise RuntimeError("Brain planner validation failed without error detail")

    def _parse_structured_response(
        self,
        content: str,
        *,
        context: dict[str, Any],
        allow_empty: bool,
        default_dynamic: bool,
    ) -> PlannerResponse:
        payload = self.planning_skill.extract_json(content)
        kind = str(payload.get("kind") or "final_plan").strip().lower()
        thinking_summary = _coerce_optional_text(
            payload.get("thinking_summary") or payload.get("thought")
        )
        if kind == "tool_call":
            tool_name = str(payload.get("tool_name") or "").strip()
            if not tool_name:
                raise ValueError("LLM planner tool_call is missing tool_name")
            tool_payload = payload.get("tool_payload")
            if tool_payload is None:
                tool_payload = payload.get("payload")
            if not isinstance(tool_payload, dict):
                raise ValueError("LLM planner tool_call must provide object tool_payload")
            return PlannerResponse(
                kind="tool_call",
                tool_name=tool_name,
                tool_payload=dict(tool_payload),
                thinking_summary=thinking_summary,
            )

        plan_payload = payload.get("plan") if kind == "final_plan" else payload
        if not isinstance(plan_payload, dict):
            raise ValueError("LLM planner final_plan payload must be an object")
        plan = self.planning_skill.parse_plan_response(
            json.dumps(plan_payload, ensure_ascii=False, default=str),
            model=self.config.model,
            allow_empty=allow_empty,
            default_dynamic=default_dynamic,
        )
        self.planning_skill.validate_plan_semantics(plan)
        return PlannerResponse(kind="final_plan", plan=plan, thinking_summary=thinking_summary)

    def _request_chat_completion(self, user_prompt: str, system_prompt: str | None = None) -> str:
        url = self._completion_url(self.config.base_url)
        api_key = (
            self.config.api_key or os.getenv(self.config.api_key_env) or os.getenv("OPENAI_API_KEY")
        )

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt or self.planning_skill.system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
        }
        attempts = max(1, int(self.config.max_retries) + 1)
        last_error: Exception | None = None
        response: requests.Response | None = None

        for attempt in range(1, attempts + 1):
            try:
                response = self.session.post(
                    url, headers=headers, json=payload, timeout=self.config.timeout_s
                )
                if response.status_code in _RETRIABLE_STATUS_CODES and attempt < attempts:
                    time.sleep(max(0.0, float(self.config.retry_backoff_s)))
                    continue
                response.raise_for_status()
                break
            except requests.exceptions.Timeout as exc:
                last_error = exc
                if attempt < attempts:
                    time.sleep(max(0.0, float(self.config.retry_backoff_s)))
                    continue
                raise TimeoutError(
                    f"Brain planner timeout after {self.config.timeout_s:.1f}s "
                    f"(attempt {attempt}/{attempts}) calling {url}"
                ) from exc
            except requests.exceptions.ConnectionError as exc:
                last_error = exc
                if attempt < attempts:
                    time.sleep(max(0.0, float(self.config.retry_backoff_s)))
                    continue
                raise RuntimeError(
                    f"Brain planner connection error on attempt {attempt}/{attempts} calling {url}: {exc}"
                ) from exc
            except requests.exceptions.HTTPError as exc:
                last_error = exc
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code in _RETRIABLE_STATUS_CODES and attempt < attempts:
                    time.sleep(max(0.0, float(self.config.retry_backoff_s)))
                    continue
                detail = self._extract_http_error_detail(exc.response)
                raise RuntimeError(
                    f"Brain planner HTTP {status_code or 'unknown'} on attempt {attempt}/{attempts}: {detail}"
                ) from exc
            except requests.exceptions.RequestException as exc:
                raise RuntimeError(f"Brain planner request failed calling {url}: {exc}") from exc
        else:
            raise RuntimeError(f"Brain planner request failed without response: {last_error}")

        if response is None:
            raise RuntimeError("Brain planner request produced no response")

        body = response.json()

        choices = body.get("choices") or []
        if not choices:
            raise ValueError("Brain planner response did not include choices")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item) for item in content
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM planner response content is empty")
        return content

    @staticmethod
    def _extract_http_error_detail(response: requests.Response | None) -> str:
        if response is None:
            return "empty error response"
        try:
            payload = response.json()
        except ValueError:
            text = response.text.strip()
            return text or "empty error response"
        if isinstance(payload, dict):
            detail = payload.get("detail")
            if detail:
                return str(detail)
        return str(payload)

    @staticmethod
    def _completion_url(base_url: str) -> str:
        normalized = base_url.rstrip("/")
        if normalized.endswith("/chat/completions"):
            return normalized
        return f"{normalized}/chat/completions"

    @staticmethod
    def _should_enable_dynamic_execution(context: dict) -> bool:
        del context
        return True

    def _build_structured_prompt(
        self,
        *,
        mode: str,
        task_description: str,
        context: dict[str, Any],
        execution_state: dict[str, Any] | None,
        failed_subtask: Subtask | None,
        failure_reason: str | None,
    ) -> str:
        if mode == "next_step":
            base_prompt = self.next_step_skill.build_prompt(
                task_description=task_description,
                context=context,
                execution_state=execution_state or {},
            )
        elif mode == "replan":
            if failed_subtask is None or failure_reason is None:
                raise ValueError("replan mode requires failed_subtask and failure_reason")
            base_prompt = self.replanning_skill.build_prompt(
                task_description=task_description,
                context=context,
                failed_subtask=failed_subtask,
                failure_reason=failure_reason,
                execution_state=execution_state or {},
            )
        else:
            base_prompt = self.planning_skill.build_prompt(
                task_description=task_description, context=context
            )

        return (
            f"{base_prompt}\n"
            f"Available Brain tools JSON: {json.dumps(context.get('available_tools', []), ensure_ascii=False, default=str)}\n"
            "If you need external constraints or schedule state before committing to a plan, return exactly one "
            "`tool_call` object.\n"
            "If you already have enough information, return a `final_plan` object whose `plan` field matches the "
            "VisioMindAction plan schema.\n"
            "If tool_trace already contains a successful tool result that answers your question, reuse it instead of "
            "calling the same tool again.\n"
            "Return JSON only."
        )

    @staticmethod
    def _serialize_context(context: dict) -> str:
        return DefaultBrainPlanningSkill.serialize_context(context)


def _coerce_optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split()).strip()
    return text or None


__all__ = ["OpenAICompatiblePlanner", "OpenAIPlannerConfig", "PlannerResponse", "time"]
