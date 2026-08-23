"""High-level skill-routing helpers owned by the Navigation agent body."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from http import HTTPStatus

import requests

from voltron.agents.navigation.body.json_response import extract_json_object
from voltron.shared.context import ExecutionContext, LocalSkillSelection, Subtask

_SYSTEM_PROMPT = """You are the local Navigation skill selector for an embodied robot.
Return valid JSON only. Do not include markdown fences or extra text.

Choose exactly one primary skill for the current Navigation subtask.
Allowed skill ids are:
- object_approach_selection_skill
- direct_navigation_skill

Return a top-level JSON object with this schema:
{
  "skill_id": "one allowed skill id",
  "confidence": 0.0,
  "reason": "short explanation",
  "fallback_skill_candidates": ["optional alternate skill ids"]
}

Selection rules:
- Use `object_approach_selection_skill` when the grounded goal is an object or the instruction asks to stop where an object can be observed, reached, manipulated, or used by a following skill.
- Use `direct_navigation_skill` for room / floor / ordinary navigation.
- If `target.object` or `target.object_id` is present and the destination depends on a physical object, strongly prefer `object_approach_selection_skill`.
"""
_RETRIABLE_STATUS_CODES = {
    HTTPStatus.REQUEST_TIMEOUT,
    HTTPStatus.TOO_MANY_REQUESTS,
    HTTPStatus.INTERNAL_SERVER_ERROR,
    HTTPStatus.BAD_GATEWAY,
    HTTPStatus.SERVICE_UNAVAILABLE,
    HTTPStatus.GATEWAY_TIMEOUT,
}


class HeuristicNavigationSkillSelector:
    """Route object-level approach tasks into the object-approach skill."""

    def select_skill(
        self,
        subtask: Subtask,
        context: ExecutionContext,
        available_skill_ids: list[str],
    ) -> LocalSkillSelection:
        del context
        wants_object_approach = (
            bool(str(subtask.target.get("object") or subtask.target.get("object_id") or "").strip())
            and subtask.action.strip().lower() not in {"room_navigation", "floor_navigation"}
        )
        selected_skill_id = "direct_navigation_skill"
        reason = "default navigation path"
        fallback_candidates: list[str] = []
        if wants_object_approach and "object_approach_selection_skill" in available_skill_ids:
            selected_skill_id = "object_approach_selection_skill"
            reason = "object-target approach requires anchor generation before navigation execution"
            if "direct_navigation_skill" in available_skill_ids:
                fallback_candidates.append("direct_navigation_skill")
        elif "direct_navigation_skill" not in available_skill_ids and available_skill_ids:
            selected_skill_id = available_skill_ids[0]
            reason = "fallback to first registered navigation skill"

        return LocalSkillSelection(
            skill_id=selected_skill_id,
            confidence=0.9 if selected_skill_id == "object_approach_selection_skill" else 0.65,
            reason=reason,
            source="heuristic_navigation_skill_selector",
            fallback_skill_candidates=fallback_candidates,
        )


@dataclass(frozen=True)
class OpenAINavigationSkillSelectorConfig:
    base_url: str
    model: str
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: float = 10.0
    temperature: float = 0.0
    max_retries: int = 0
    retry_backoff_s: float = 1.0


class OpenAICompatibleNavigationSkillSelector:
    """OpenAI-compatible high-level skill router used by the Navigation body."""

    def __init__(self, config: OpenAINavigationSkillSelectorConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.fallback_selector = HeuristicNavigationSkillSelector()

    def select_skill(
        self,
        subtask: Subtask,
        context: ExecutionContext,
        available_skill_ids: list[str],
    ) -> LocalSkillSelection:
        try:
            prompt = self._build_prompt(subtask=subtask, context=context, available_skill_ids=available_skill_ids)
            content = self._request_chat_completion(prompt)
            return self._parse_selection_response(content=content, available_skill_ids=available_skill_ids)
        except Exception as exc:
            fallback = self.fallback_selector.select_skill(
                subtask=subtask,
                context=context,
                available_skill_ids=available_skill_ids,
            )
            return LocalSkillSelection(
                skill_id=fallback.skill_id,
                confidence=fallback.confidence,
                reason=f"fallback after openai selector error: {exc}; {fallback.reason}",
                source="heuristic_navigation_skill_selector_fallback",
                fallback_skill_candidates=fallback.fallback_skill_candidates,
                metadata={
                    **fallback.metadata,
                    "selector_error": str(exc),
                    "fallback_from": "openai_compatible_navigation_skill_selector",
                },
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
        response = None
        for attempt in range(1, attempts + 1):
            try:
                response = self.session.post(url, headers=headers, json=payload, timeout=self.config.timeout_s)
                if response.status_code in _RETRIABLE_STATUS_CODES and attempt < attempts:
                    time.sleep(max(0.0, float(self.config.retry_backoff_s)))
                    continue
                response.raise_for_status()
                break
            except requests.exceptions.RequestException:
                if attempt < attempts:
                    time.sleep(max(0.0, float(self.config.retry_backoff_s)))
                    continue
                raise
        if response is None:
            raise RuntimeError("Navigation skill selector request produced no response")
        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise ValueError("Navigation skill selector response did not include choices")
        content = choices[0].get("message", {}).get("content")
        if isinstance(content, list):
            content = "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Navigation skill selector response content is empty")
        return content

    @staticmethod
    def _completion_url(base_url: str) -> str:
        normalized = base_url.rstrip("/")
        if normalized.endswith("/chat/completions"):
            return normalized
        return f"{normalized}/chat/completions"

    @staticmethod
    def _build_prompt(
        *,
        subtask: Subtask,
        context: ExecutionContext,
        available_skill_ids: list[str],
    ) -> str:
        payload = {
            "task_description": context.task_request.description,
            "task_type": context.task_request.task_type.value,
            "subtask": {
                "subtask_id": subtask.subtask_id,
                "action": subtask.action,
                "target": subtask.target,
                "instruction": subtask.parameters.get("instruction"),
            },
            "available_skill_ids": available_skill_ids,
        }
        return (
            "Choose the primary Navigation skill for this subtask.\n"
            f"Selection context JSON: {json.dumps(payload, ensure_ascii=False, default=str)}\n"
            "Return JSON only."
        )

    @staticmethod
    def _parse_selection_response(content: str, available_skill_ids: list[str]) -> LocalSkillSelection:
        payload = extract_json_object(content, label="Navigation skill selector")
        skill_id = str(payload.get("skill_id", "")).strip()
        if skill_id not in available_skill_ids:
            raise ValueError(f"Unsupported skill_id {skill_id!r}")
        fallback_candidates = []
        for item in payload.get("fallback_skill_candidates") or []:
            candidate = str(item).strip()
            if candidate and candidate in available_skill_ids and candidate != skill_id:
                fallback_candidates.append(candidate)
        return LocalSkillSelection(
            skill_id=skill_id,
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            reason=str(payload.get("reason", "")).strip(),
            source="openai_compatible_navigation_skill_selector",
            fallback_skill_candidates=fallback_candidates,
        )


__all__ = [
    "HeuristicNavigationSkillSelector",
    "OpenAICompatibleNavigationSkillSelector",
    "OpenAINavigationSkillSelectorConfig",
    "time",
]
