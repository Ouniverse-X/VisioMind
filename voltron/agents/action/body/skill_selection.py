from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from http import HTTPStatus
from json import JSONDecoder
from typing import Any

import requests

from voltron.shared.action_semantics import normalize_action_name
from voltron.shared.context import ExecutionContext, LocalSkillSelection, Subtask

_SYSTEM_PROMPT = """You are the local Action skill selector for an embodied robot.
Return valid JSON only. Do not include markdown fences or extra text.

You must choose exactly one primary skill for the current Action subtask.
Allowed skill ids are:
- button_interaction_skill
- grasp_manipulation_skill
- anygrasp_manipulation_skill
- placement_skill
- handle_operation_skill
- local_reposition_skill
- default_manipulation_skill

Return a top-level JSON object with this schema:
{
  "skill_id": "one allowed skill id",
  "confidence": 0.0,
  "reason": "short explanation",
  "fallback_skill_candidates": ["optional alternate skill ids"]
}

Selection rules:
- Choose by local interaction semantics, not by model/backend name.
- Infer the local skill from natural-language intent, target names, parts, and visual context. Do not require
  pre-filled object ids or rule-extracted action names.
- affordance-bearing controls such as switches, buttons, toggles, knobs, panels, and appliance controls should
  normally map to `button_interaction_skill` when the intended operation is press/toggle/turn on/off/switch.
- `button_interaction_skill`: press, toggle, switch, turn on/off, push button.
- `grasp_manipulation_skill`: grasp, pick up, lift, hold, take.
- `placement_skill`: generic place, put down, release, drop when no object-aware
  grasp backend is available.
- `anygrasp_manipulation_skill`: pick/grasp and object-aware container placement
  such as place_inside or put_inside when AnyGrasp is registered.
- `handle_operation_skill`: open, close, pull, push, rotate, handle-like articulation.
- `local_reposition_skill`: bounded local base/body repositioning for the current interaction target, such as
  approach, align, step back, adjust pose, or move to an interaction-ready pose. Do not use it for room-scale or
  object-search navigation.
- `default_manipulation_skill`: only when no other skill clearly matches.
- Fallback candidates must also be allowed skill ids and should be ordered by plausibility.
"""
_RETRIABLE_STATUS_CODES = {
    HTTPStatus.REQUEST_TIMEOUT,
    HTTPStatus.TOO_MANY_REQUESTS,
    HTTPStatus.INTERNAL_SERVER_ERROR,
    HTTPStatus.BAD_GATEWAY,
    HTTPStatus.SERVICE_UNAVAILABLE,
    HTTPStatus.GATEWAY_TIMEOUT,
}


class HeuristicActionSkillSelector:
    _ACTION_TO_SKILL = {
        "press": "button_interaction_skill",
        "push_button": "button_interaction_skill",
        "toggle_on": "button_interaction_skill",
        "toggle_off": "button_interaction_skill",
        "turn_on": "button_interaction_skill",
        "turn_off": "button_interaction_skill",
        "switch_on": "button_interaction_skill",
        "switch_off": "button_interaction_skill",
        "pick_up": "grasp_manipulation_skill",
        "grasp": "grasp_manipulation_skill",
        "lift": "grasp_manipulation_skill",
        "take": "grasp_manipulation_skill",
        "hold": "grasp_manipulation_skill",
        "place": "placement_skill",
        "place_inside": "anygrasp_manipulation_skill",
        "put_inside": "anygrasp_manipulation_skill",
        "put_down": "placement_skill",
        "drop": "placement_skill",
        "release": "placement_skill",
        "open": "handle_operation_skill",
        "close": "handle_operation_skill",
        "pull": "handle_operation_skill",
        "push": "handle_operation_skill",
        "turn": "handle_operation_skill",
        "rotate": "handle_operation_skill",
        "move_to_interaction_pose": "local_reposition_skill",
        "align": "local_reposition_skill",
        "approach": "local_reposition_skill",
        "adjust_pose": "local_reposition_skill",
        "step_back": "local_reposition_skill",
    }

    _SKILL_ALTERNATIVES: dict[str, str] = {
        "grasp_manipulation_skill": "anygrasp_manipulation_skill",
        "anygrasp_manipulation_skill": "placement_skill",
    }

    def select_skill(
        self,
        subtask: Subtask,
        context: ExecutionContext,
        available_skill_ids: list[str],
    ) -> LocalSkillSelection:
        action = normalize_action_name(subtask.action)
        selected_skill_id = self._ACTION_TO_SKILL.get(action, "default_manipulation_skill")

        if selected_skill_id not in available_skill_ids:
            alt = self._SKILL_ALTERNATIVES.get(selected_skill_id)
            if alt and alt in available_skill_ids:
                selected_skill_id = alt
            else:
                selected_skill_id = "default_manipulation_skill"

        task_type = context.task_request.task_type.value
        target = str(subtask.target.get("object", ""))
        fallback_candidates = [
            skill_id
            for skill_id in (
                "default_manipulation_skill",
                "local_reposition_skill",
                "grasp_manipulation_skill",
                "button_interaction_skill",
                "placement_skill",
                "anygrasp_manipulation_skill",
            )
            if skill_id in available_skill_ids and skill_id != selected_skill_id
        ]

        return LocalSkillSelection(
            skill_id=selected_skill_id,
            confidence=0.82 if selected_skill_id != "default_manipulation_skill" else 0.45,
            reason=f"selected from action={action!r}, target={target!r}, task_type={task_type!r}",
            source="heuristic_local_skill_selector",
            fallback_skill_candidates=fallback_candidates,
            metadata={"action": action, "target": target, "task_type": task_type},
        )


@dataclass(frozen=True)
class OpenAIActionSkillSelectorConfig:
    base_url: str
    model: str
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: float = 10.0
    temperature: float = 0.0
    max_retries: int = 0
    retry_backoff_s: float = 1.0


class OpenAICompatibleActionSkillSelector:
    def __init__(self, config: OpenAIActionSkillSelectorConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.fallback_selector = HeuristicActionSkillSelector()

    def select_skill(
        self,
        subtask: Subtask,
        context: ExecutionContext,
        available_skill_ids: list[str],
    ) -> LocalSkillSelection:
        try:
            prompt = self._build_prompt(
                subtask=subtask, context=context, available_skill_ids=available_skill_ids
            )
            content = self._request_chat_completion(prompt)
            return self._parse_selection_response(
                content=content,
                available_skill_ids=available_skill_ids,
            )
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
                source="heuristic_local_skill_selector_fallback",
                fallback_skill_candidates=fallback.fallback_skill_candidates,
                metadata={
                    **fallback.metadata,
                    "selector_error": str(exc),
                    "fallback_from": "openai_compatible_local_skill_selector",
                },
            )

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
                    f"Action selector timeout after {self.config.timeout_s:.1f}s "
                    f"(attempt {attempt}/{attempts}) calling {url}"
                ) from exc
            except requests.exceptions.ConnectionError as exc:
                last_error = exc
                if attempt < attempts:
                    time.sleep(max(0.0, float(self.config.retry_backoff_s)))
                    continue
                raise RuntimeError(
                    f"Action selector connection error on attempt {attempt}/{attempts} calling {url}: {exc}"
                ) from exc
            except requests.exceptions.HTTPError as exc:
                last_error = exc
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code in _RETRIABLE_STATUS_CODES and attempt < attempts:
                    time.sleep(max(0.0, float(self.config.retry_backoff_s)))
                    continue
                detail = self._extract_http_error_detail(exc.response)
                raise RuntimeError(
                    f"Action selector HTTP {status_code or 'unknown'} on attempt {attempt}/{attempts}: {detail}"
                ) from exc
            except requests.exceptions.RequestException as exc:
                raise RuntimeError(f"Action selector request failed calling {url}: {exc}") from exc
        else:
            raise RuntimeError(f"Action selector request failed without response: {last_error}")

        if response is None:
            raise RuntimeError("Action selector request produced no response")

        body = response.json()

        choices = body.get("choices") or []
        if not choices:
            raise ValueError("Action selector response did not include choices")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item) for item in content
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Action selector response content is empty")
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
        available_skill_ids: list[str],
    ) -> str:
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
                },
                "context": subtask.context,
            },
            "available_skill_ids": available_skill_ids,
        }
        return (
            "Choose the primary local Action skill for this subtask.\n"
            f"Selection context JSON: {json.dumps(payload, ensure_ascii=False, default=str)}\n"
            "Return JSON only."
        )

    @staticmethod
    def _parse_selection_response(
        content: str,
        available_skill_ids: list[str],
    ) -> LocalSkillSelection:
        payload = OpenAICompatibleActionSkillSelector._extract_json(content)
        skill_id = str(payload.get("skill_id", "")).strip()
        if skill_id not in available_skill_ids:
            raise ValueError(f"Unsupported skill_id {skill_id!r}")

        fallback_raw = payload.get("fallback_skill_candidates")
        fallback_skill_candidates = []
        if isinstance(fallback_raw, list):
            for item in fallback_raw:
                candidate = str(item).strip()
                if candidate and candidate in available_skill_ids and candidate != skill_id:
                    fallback_skill_candidates.append(candidate)

        confidence = payload.get("confidence", 0.0)
        try:
            normalized_confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            normalized_confidence = 0.0

        reason = str(payload.get("reason", "")).strip()
        return LocalSkillSelection(
            skill_id=skill_id,
            confidence=normalized_confidence,
            reason=reason,
            source="openai_compatible_local_skill_selector",
            fallback_skill_candidates=fallback_skill_candidates,
            metadata={"raw_response": payload},
        )

    @staticmethod
    def _extract_json(content: str) -> dict[str, Any]:
        stripped = content.strip()
        candidates = [stripped]
        if "```" in stripped:
            for block in stripped.split("```"):
                candidate = block.strip()
                if not candidate:
                    continue
                if candidate.startswith("json"):
                    candidate = candidate[4:].strip()
                candidates.append(candidate)

        decoder = JSONDecoder()
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                for start_index, char in enumerate(candidate):
                    if char not in "{[":
                        continue
                    try:
                        payload, _ = decoder.raw_decode(candidate[start_index:])
                    except json.JSONDecodeError:
                        continue
                    break
                else:
                    continue
            if isinstance(payload, dict):
                return payload
        raise ValueError("Failed to parse JSON from Action selector response")


__all__ = [
    "HeuristicActionSkillSelector",
    "OpenAIActionSkillSelectorConfig",
    "OpenAICompatibleActionSkillSelector",
    "time",
]
