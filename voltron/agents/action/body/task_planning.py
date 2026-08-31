from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from http import HTTPStatus

import requests

from voltron.shared.context import ExecutionContext, Subtask

_SYSTEM_PROMPT = """You are the internal task planner inside the Action agent for an embodied robot.
Return valid JSON only. Do not include markdown fences or extra text.

Your job is to decompose a local manipulation subtask into semantically meaningful coarse subtasks.
The output must be useful for execution, concise, and directly tied to the task goal.
Decompose local manipulation intent yourself from the instruction, target, visual context, and task description.
affordance-bearing controls include buttons, switches, toggles, knobs, handles, panels, and appliance controls.
Treat pressable or toggleable controls as a button or switch target class when decomposing local execution.
When the task asks for press/toggle/turn/open/close behavior, produce a short sequence that orients to the relevant
control, reaches the actionable area, performs the operation, and exposes clear success cues without requiring a
pre-grounded graph object id.
"""
_RETRIABLE_STATUS_CODES = {
    HTTPStatus.REQUEST_TIMEOUT,
    HTTPStatus.TOO_MANY_REQUESTS,
    HTTPStatus.INTERNAL_SERVER_ERROR,
    HTTPStatus.BAD_GATEWAY,
    HTTPStatus.SERVICE_UNAVAILABLE,
    HTTPStatus.GATEWAY_TIMEOUT,
}


@dataclass(frozen=True)
class OpenAIActionTaskPlannerConfig:
    base_url: str
    model: str
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: float = 10.0
    temperature: float = 0.0
    max_retries: int = 0
    retry_backoff_s: float = 1.0


class OpenAICompatibleActionTaskPlanner:
    def __init__(self, config: OpenAIActionTaskPlannerConfig) -> None:
        self.config = config
        self.session = requests.Session()

    def generate_plan(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        prompt: str,
    ) -> str:
        del subtask, context
        return self._request_chat_completion(prompt)

    def _request_chat_completion(self, user_prompt: str) -> str:
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
                {"role": "system", "content": _SYSTEM_PROMPT},
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
                    f"Action task planner timeout after {self.config.timeout_s:.1f}s "
                    f"(attempt {attempt}/{attempts}) calling {url}"
                ) from exc
            except requests.exceptions.ConnectionError as exc:
                last_error = exc
                if attempt < attempts:
                    time.sleep(max(0.0, float(self.config.retry_backoff_s)))
                    continue
                raise RuntimeError(
                    f"Action task planner connection error on attempt {attempt}/{attempts} calling {url}: {exc}"
                ) from exc
            except requests.exceptions.HTTPError as exc:
                last_error = exc
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code in _RETRIABLE_STATUS_CODES and attempt < attempts:
                    time.sleep(max(0.0, float(self.config.retry_backoff_s)))
                    continue
                detail = self._extract_http_error_detail(exc.response)
                raise RuntimeError(
                    f"Action task planner HTTP {status_code or 'unknown'} on attempt {attempt}/{attempts}: {detail}"
                ) from exc
            except requests.exceptions.RequestException as exc:
                raise RuntimeError(
                    f"Action task planner request failed calling {url}: {exc}"
                ) from exc
        else:
            raise RuntimeError(f"Action task planner request failed without response: {last_error}")

        if response is None:
            raise RuntimeError("Action task planner request produced no response")

        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise ValueError("Action task planner response did not include choices")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item) for item in content
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Action task planner response content is empty")
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


class HeuristicActionTaskPlanner:
    def generate_plan(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        prompt: str,
    ) -> str:
        del context, prompt
        instruction = (
            str(subtask.parameters.get("instruction") or subtask.action).strip() or subtask.action
        )
        return json.dumps(
            {
                "goal_summary": instruction,
                "steps": [
                    {
                        "name": f"perform_{subtask.action}",
                        "instruction": instruction,
                        "action": subtask.action,
                    }
                ],
            }
        )


__all__ = [
    "HeuristicActionTaskPlanner",
    "OpenAIActionTaskPlannerConfig",
    "OpenAICompatibleActionTaskPlanner",
    "time",
]
