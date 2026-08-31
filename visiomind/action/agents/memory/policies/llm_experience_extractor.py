from __future__ import annotations

import os
import time
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

import requests

from visiomind.action.agents.memory.contracts.experience import ExperienceExtractionResult
from visiomind.action.agents.memory.contracts.extractor import ExperienceExtractor
from visiomind.action.agents.memory.skills.experience_extraction import (
    DefaultMemoryExperienceExtractionSkill,
)

_RETRIABLE_STATUS_CODES = {
    HTTPStatus.REQUEST_TIMEOUT,
    HTTPStatus.TOO_MANY_REQUESTS,
    HTTPStatus.INTERNAL_SERVER_ERROR,
    HTTPStatus.BAD_GATEWAY,
    HTTPStatus.SERVICE_UNAVAILABLE,
    HTTPStatus.GATEWAY_TIMEOUT,
}


@dataclass(frozen=True)
class OpenAIExperienceExtractorConfig:
    base_url: str
    model: str
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    timeout_s: float = 30.0
    temperature: float = 0.0
    max_retries: int = 0
    retry_backoff_s: float = 1.0


class OpenAICompatibleExperienceExtractor(ExperienceExtractor):
    def __init__(
        self,
        config: OpenAIExperienceExtractorConfig,
        skill: DefaultMemoryExperienceExtractionSkill | None = None,
    ) -> None:
        self.config = config
        self.session = requests.Session()
        self.skill = skill or DefaultMemoryExperienceExtractionSkill()

    def extract(self, episode_context: dict[str, Any]) -> ExperienceExtractionResult:
        prompt = self.skill.build_prompt(episode_context)
        attempts = max(1, int(self.config.max_retries) + 1)
        current_prompt = prompt
        last_error: ValueError | None = None
        for attempt in range(1, attempts + 1):
            content = self._request_chat_completion(current_prompt)
            try:
                result = self.skill.parse_extraction_response(content)
                if not result.source_episode_id:
                    result.source_episode_id = str(episode_context.get("episode_id") or "")
                return result
            except ValueError as exc:
                last_error = exc
                if attempt >= attempts:
                    raise
                current_prompt = self._build_schema_retry_prompt(prompt=prompt, error=str(exc))

        if last_error is not None:
            raise last_error
        raise RuntimeError("Memory experience extractor failed without error detail")

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
                {"role": "system", "content": self.skill.system_prompt},
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
                    f"Memory experience extractor timeout after {self.config.timeout_s:.1f}s "
                    f"(attempt {attempt}/{attempts}) calling {url}"
                ) from exc
            except requests.exceptions.ConnectionError as exc:
                last_error = exc
                if attempt < attempts:
                    time.sleep(max(0.0, float(self.config.retry_backoff_s)))
                    continue
                raise RuntimeError(
                    f"Memory experience extractor connection error on attempt {attempt}/{attempts} "
                    f"calling {url}: {exc}"
                ) from exc
            except requests.exceptions.HTTPError as exc:
                last_error = exc
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code in _RETRIABLE_STATUS_CODES and attempt < attempts:
                    time.sleep(max(0.0, float(self.config.retry_backoff_s)))
                    continue
                detail = self._extract_http_error_detail(exc.response)
                raise RuntimeError(
                    f"Memory experience extractor HTTP {status_code or 'unknown'} "
                    f"on attempt {attempt}/{attempts}: {detail}"
                ) from exc
            except requests.exceptions.RequestException as exc:
                raise RuntimeError(
                    f"Memory experience extractor request failed calling {url}: {exc}"
                ) from exc
        else:
            raise RuntimeError(
                f"Memory experience extractor request failed without response: {last_error}"
            )

        if response is None:
            raise RuntimeError("Memory experience extractor request produced no response")

        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise ValueError("Memory experience extractor response did not include choices")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                item.get("text", "") if isinstance(item, dict) else str(item) for item in content
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Memory experience extractor response content is empty")
        return content

    @staticmethod
    def _completion_url(base_url: str) -> str:
        normalized = base_url.rstrip("/")
        if normalized.endswith("/chat/completions"):
            return normalized
        return f"{normalized}/chat/completions"

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
    def _build_schema_retry_prompt(*, prompt: str, error: str) -> str:
        return (
            "Previous extraction response could not be parsed as the required JSON schema.\n"
            f"Parser error: {error}\n"
            "Retry the same extraction task and return only one valid JSON object.\n\n"
            f"Original prompt:\n{prompt}"
        )


__all__ = [
    "OpenAICompatibleExperienceExtractor",
    "OpenAIExperienceExtractorConfig",
    "time",
]
