"""High-level deliberation logic owned by the Action agent body."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from http import HTTPStatus
from json import JSONDecoder
from typing import Any

import requests

from voltron.agents.action.models import VLADeliberation
from voltron.shared.context import ExecutionContext, Subtask

_SYSTEM_PROMPT = """You are the core decision model inside the Action agent for an embodied robot.
Return valid JSON only. Do not include markdown fences or extra text.

Decide whether the Action agent should call the internal `refine_target` tool before local skill selection.
Only request `refine_target` when the current target/instruction is too coarse for safe or precise local execution.
Interpret local manipulation intent yourself from the instruction, target, visual report, and runtime context.
affordance-bearing controls include buttons, switches, toggles, knobs, handles, panels, and appliance controls.
Treat a requested press/toggle/switch operation as a button or switch target class when the wording indicates a
pressable or toggleable control.
Treat press/toggle/turn/open/close language as a request to identify the actionable control or part, even when no
graph object id is provided. Request `refine_target` when language names a coarse device but the actionable part,
affordance, or success cue still needs visual/contextual refinement.

Return a top-level JSON object with this schema:
{
  "use_tool": true,
  "tool_name": "refine_target",
  "reason": "short explanation",
  "selector_hints": {"optional": "structured hints"},
  "policy_hints": {"optional": "structured hints"}
}

Rules:
- `tool_name` must be `refine_target` when `use_tool=true`.
- If the current target is already precise enough, return `use_tool=false` and `tool_name=null`.
- Keep `selector_hints` and `policy_hints` compact and JSON-serializable.
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
class OpenAIActionDeliberatorConfig:
    base_url: str
    model: str
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: float = 10.0
    temperature: float = 0.0
    max_retries: int = 0
    retry_backoff_s: float = 1.0


class OpenAICompatibleActionDeliberator:
    """Internal Action deliberator backed by an OpenAI-compatible endpoint."""

    def __init__(self, config: OpenAIActionDeliberatorConfig) -> None:
        self.config = config
        self.session = requests.Session()

    def deliberate(
        self,
        subtask: Subtask,
        context: ExecutionContext,
    ) -> VLADeliberation:
        try:
            prompt = self._build_prompt(subtask=subtask, context=context)
            content = self._request_chat_completion(prompt)
            return self._parse_response(content=content)
        except Exception as exc:
            return VLADeliberation(
                use_tool=False,
                tool_name=None,
                reason=f"fallback after openai deliberator error: {exc}",
                source="heuristic_vla_deliberator_fallback",
                metadata={"deliberator_error": str(exc)},
            )

    def _request_chat_completion(self, user_prompt: str) -> str:
        url = self._completion_url(self.config.base_url)
        api_key = self.config.api_key or os.getenv(self.config.api_key_env) or os.getenv("OPENAI_API_KEY")

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
                response = self.session.post(url, headers=headers, json=payload, timeout=self.config.timeout_s)
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
                    f"Action deliberator timeout after {self.config.timeout_s:.1f}s "
                    f"(attempt {attempt}/{attempts}) calling {url}"
                ) from exc
            except requests.exceptions.ConnectionError as exc:
                last_error = exc
                if attempt < attempts:
                    time.sleep(max(0.0, float(self.config.retry_backoff_s)))
                    continue
                raise RuntimeError(
                    f"Action deliberator connection error on attempt {attempt}/{attempts} calling {url}: {exc}"
                ) from exc
            except requests.exceptions.HTTPError as exc:
                last_error = exc
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code in _RETRIABLE_STATUS_CODES and attempt < attempts:
                    time.sleep(max(0.0, float(self.config.retry_backoff_s)))
                    continue
                detail = self._extract_http_error_detail(exc.response)
                raise RuntimeError(
                    f"Action deliberator HTTP {status_code or 'unknown'} on attempt {attempt}/{attempts}: {detail}"
                ) from exc
            except requests.exceptions.RequestException as exc:
                raise RuntimeError(f"Action deliberator request failed calling {url}: {exc}") from exc
        else:
            raise RuntimeError(f"Action deliberator request failed without response: {last_error}")

        if response is None:
            raise RuntimeError("Action deliberator request produced no response")

        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise ValueError("Action deliberator response did not include choices")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Action deliberator response content is empty")
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
    def _build_prompt(
        subtask: Subtask,
        context: ExecutionContext,
    ) -> str:
        last_result = context.results[-1].result if context.results else {}
        payload = {
            "task_description": context.task_request.description,
            "task_type": context.task_request.task_type.value,
            "subtask": {
                "subtask_id": subtask.subtask_id,
                "action": subtask.action,
                "target": subtask.target,
                "parameters": {
                    "instruction": subtask.parameters.get("instruction"),
                    "control_mode": subtask.parameters.get("control_mode"),
                    "policy_options": subtask.parameters.get("policy_options"),
                    "observation_keys": sorted(subtask.parameters.get("observation", {}).keys())
                    if isinstance(subtask.parameters.get("observation"), dict)
                    else [],
                },
                "context": subtask.context,
            },
            "latest_result": last_result,
        }
        return (
            "Decide whether the Action agent should call the internal refine_target tool before skill selection.\n"
            f"Deliberation context JSON: {json.dumps(payload, ensure_ascii=False, default=str)}\n"
            "Return JSON only."
        )

    @staticmethod
    def _parse_response(
        content: str,
    ) -> VLADeliberation:
        payload = OpenAICompatibleActionDeliberator._extract_json(content)
        use_tool = bool(payload.get("use_tool", False))
        tool_name = payload.get("tool_name")
        normalized_tool_name = str(tool_name).strip() if tool_name is not None else None
        if use_tool and normalized_tool_name != "refine_target":
            raise ValueError(f"Unsupported Action tool {normalized_tool_name!r}")
        selector_hints = payload.get("selector_hints")
        policy_hints = payload.get("policy_hints")
        return VLADeliberation(
            use_tool=use_tool,
            tool_name=normalized_tool_name if use_tool else None,
            reason=str(payload.get("reason", "")).strip(),
            source="openai_compatible_vla_deliberator",
            selector_hints=dict(selector_hints) if isinstance(selector_hints, dict) else {},
            policy_hints=dict(policy_hints) if isinstance(policy_hints, dict) else {},
            metadata={"raw_response": payload},
        )

    @staticmethod
    def _extract_json(content: str) -> dict[str, Any]:
        stripped = content.strip()
        candidates = [stripped]
        if "```" in stripped:
            parts = stripped.split("```")
            candidates.extend(part.strip() for part in parts if part.strip())

        decoder = JSONDecoder()
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                for idx, char in enumerate(candidate):
                    if char != "{":
                        continue
                    try:
                        parsed, end = decoder.raw_decode(candidate[idx:])
                    except json.JSONDecodeError:
                        continue
                    if isinstance(parsed, dict):
                        return parsed
                    if end:
                        continue
            else:
                if isinstance(parsed, dict):
                    return parsed
        raise ValueError("Failed to parse JSON from Action deliberator response")


__all__ = [
    "OpenAIActionDeliberatorConfig",
    "OpenAICompatibleActionDeliberator",
    "time",
]
