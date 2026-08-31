from __future__ import annotations

from http import HTTPStatus
from json import JSONDecoder
import re
import time
from typing import Any, Protocol

import requests

from .models import VLMBackendConfig, VLMProcessRequest, VLMProcessResponse
from .prompting import build_dashscope_messages, build_openai_messages


_JSON_DECODER = JSONDecoder()
_SCENE_REPORT_KEYS = (
    "target_visible",
    "target_part_visible",
    "target_part_name",
)
_RETRIABLE_STATUS_CODES = {
    HTTPStatus.REQUEST_TIMEOUT,
    HTTPStatus.TOO_MANY_REQUESTS,
    HTTPStatus.INTERNAL_SERVER_ERROR,
    HTTPStatus.BAD_GATEWAY,
    HTTPStatus.SERVICE_UNAVAILABLE,
    HTTPStatus.GATEWAY_TIMEOUT,
}


class VLMBackendError(RuntimeError):
    pass


class VLMTimeoutError(VLMBackendError):
    pass


class VLMHTTPError(VLMBackendError):
    pass


class VLMBackend(Protocol):
    def analyze(self, request: VLMProcessRequest) -> VLMProcessResponse: ...


class OpenAICompatibleVLMBackend:
    def __init__(self, config: VLMBackendConfig, session: requests.Session | None = None):
        if not config.base_url:
            raise ValueError("OpenAI-compatible VLM backend requires base_url")
        self.config = config
        self.session = session or requests.Session()

    def analyze(self, request: VLMProcessRequest) -> VLMProcessResponse:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        url = completion_url(self.config.base_url)
        payload = {
            "model": self.config.model,
            "messages": build_openai_messages(request),
            "temperature": 0.0,
        }
        last_error: Exception | None = None
        deadline = time.monotonic() + max(1.0, float(self.config.timeout_s))
        max_attempts = max(1, int(self.config.max_retries) + 1)
        retry_backoff_s = max(0.0, float(self.config.retry_backoff_s))

        for attempt in range(1, max_attempts + 1):
            request_timeout = _request_timeout_budget(
                deadline, attempt, max_attempts, retry_backoff_s
            )
            if request_timeout <= 0.0:
                raise VLMTimeoutError(
                    f"VLM backend timeout after {self.config.timeout_s:.1f}s "
                    f"(attempt {attempt}/{max_attempts}) calling {url}"
                ) from last_error

            try:
                response = self.session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=request_timeout,
                )
            except requests.exceptions.Timeout as exc:
                last_error = exc
                if attempt < max_attempts:
                    time.sleep(retry_backoff_s)
                    continue
                raise VLMTimeoutError(
                    f"VLM backend timeout after {self.config.timeout_s:.1f}s "
                    f"(attempt {attempt}/{max_attempts}) calling {url}"
                ) from exc
            except requests.exceptions.ConnectionError as exc:
                last_error = exc
                if attempt < max_attempts:
                    time.sleep(retry_backoff_s)
                    continue
                raise VLMBackendError(
                    f"VLM backend connection error on attempt {attempt}/{max_attempts} "
                    f"calling {url}: {exc}"
                ) from exc
            except requests.exceptions.RequestException as exc:
                raise VLMBackendError(f"VLM backend request failed calling {url}: {exc}") from exc

            if response.status_code in _RETRIABLE_STATUS_CODES and attempt < max_attempts:
                time.sleep(retry_backoff_s)
                continue

            if response.status_code >= HTTPStatus.BAD_REQUEST:
                message = (
                    f"VLM backend HTTP {response.status_code} from {url}: "
                    f"{_response_preview(response)}"
                )
                if response.status_code in _RETRIABLE_STATUS_CODES:
                    raise VLMHTTPError(f"{message} (attempt {attempt}/{max_attempts})")
                raise VLMHTTPError(message)

            body = response.json()
            content = extract_message_content(body)
            return _build_process_response(content, instruction=request.instruction)

        raise VLMBackendError(f"VLM backend request failed without response: {last_error}")


class DashScopeVLMBackend:
    def __init__(self, config: VLMBackendConfig):
        self.config = config

    def analyze(self, request: VLMProcessRequest) -> VLMProcessResponse:
        try:
            import dashscope
        except ModuleNotFoundError as exc:
            raise RuntimeError("dashscope is not installed in the current environment") from exc

        dashscope.api_key = self.config.api_key
        response = dashscope.MultiModalConversation.call(
            model=self.config.model,
            messages=build_dashscope_messages(request),
            stream=False,
            extra_body={
                "enable_reasoning": True,
                "thinking_budget": 4000,
            },
        )
        if response.status_code != HTTPStatus.OK:
            raise RuntimeError(str(response.message))

        raw_content = response.output.choices[0].message.content
        if isinstance(raw_content, list):
            content = "".join(item.get("text", "") for item in raw_content if "text" in item)
        else:
            content = str(raw_content)
        return _build_process_response(content)


def build_backend(config: VLMBackendConfig) -> VLMBackend:
    if config.provider == "openai":
        return OpenAICompatibleVLMBackend(config)
    if config.provider == "dashscope":
        return DashScopeVLMBackend(config)
    raise RuntimeError(f"Unsupported VLM provider: {config.provider}")


def completion_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def extract_message_content(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("VLM response did not include choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item) for item in content
        )
    if not isinstance(content, str):
        content = str(content)
    return content


def _build_process_response(content: str, *, instruction: str = "") -> VLMProcessResponse:
    payload = _extract_json_payload(content)
    if payload is None:
        is_success = "SUCCESS" in content.upper()
        task_complete = is_success
        if _contradicts_verification_success(instruction=instruction, summary=content):
            is_success = False
            task_complete = False
        return VLMProcessResponse(
            status="ok",
            result=content,
            is_success=is_success,
            task_complete=task_complete,
            raw_text=content,
        )

    summary = str(
        payload.get("summary") or payload.get("result") or payload.get("raw_text") or content
    ).strip()
    raw_text = str(payload.get("raw_text") or summary or content).strip()
    task_complete = _coerce_bool(payload.get("task_complete"))
    if task_complete is None:
        task_complete = _coerce_bool(payload.get("is_success"))
    if task_complete is None:
        task_complete = "SUCCESS" in raw_text.upper() or "SUCCESS" in summary.upper()

    is_success = _coerce_bool(payload.get("is_success"))
    if is_success is None:
        is_success = bool(task_complete)
    if _contradicts_verification_success(instruction=instruction, summary=f"{summary}\n{raw_text}"):
        task_complete = False
        is_success = False

    scene_report = (
        payload.get("scene_report") if isinstance(payload.get("scene_report"), dict) else {}
    )
    if not scene_report:
        scene_report = {key: payload.get(key) for key in _SCENE_REPORT_KEYS if key in payload}
    else:
        scene_report = {
            key: scene_report.get(key) for key in _SCENE_REPORT_KEYS if key in scene_report
        }
    if scene_report and "task_complete" not in scene_report:
        scene_report["task_complete"] = bool(task_complete)

    objects = payload.get("objects") if isinstance(payload.get("objects"), list) else []
    relations = payload.get("relations") if isinstance(payload.get("relations"), list) else []

    return VLMProcessResponse(
        status="ok",
        result=summary or raw_text,
        is_success=bool(is_success),
        task_complete=bool(task_complete),
        raw_text=raw_text or summary or content,
        scene_report=scene_report,
        objects=[item for item in objects if isinstance(item, dict)],
        relations=[item for item in relations if isinstance(item, dict)],
    )


def _contradicts_verification_success(*, instruction: str, summary: str) -> bool:
    normalized_instruction = instruction.strip().lower()
    if not any(token in normalized_instruction for token in ("verify", "check", "confirm")):
        return False

    normalized_summary = summary.strip().lower()
    if re.search(r"\bon\b", normalized_instruction) and re.search(r"\boff\b", normalized_summary):
        return True

    negative_patterns = (
        " appears to be off",
        " appears off",
        " verified as off",
        " confirmed off",
        " is off",
        " still off",
        " turned off",
        " switched off",
        " powered off",
        " not on",
        " remains off",
    )
    return any(pattern in normalized_summary for pattern in negative_patterns)


def _extract_json_payload(content: str) -> dict[str, Any] | None:
    for index, char in enumerate(content):
        if char != "{":
            continue
        try:
            payload, _ = _JSON_DECODER.raw_decode(content[index:])
        except ValueError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "success"}:
            return True
        if normalized in {"false", "no", "failure"}:
            return False
    return None


def _response_preview(response: requests.Response) -> str:
    text = response.text.strip()
    if not text:
        return "<empty response body>"
    if len(text) > 240:
        return f"{text[:240]}..."
    return text


def _request_timeout_budget(
    deadline: float,
    attempt: int,
    max_attempts: int,
    retry_backoff_s: float,
) -> float:
    remaining_budget = deadline - time.monotonic()
    remaining_attempts = max_attempts - attempt + 1
    if remaining_budget <= 0.0 or remaining_attempts <= 0:
        return 0.0
    reserved_backoff = retry_backoff_s * max(0, remaining_attempts - 1)
    request_budget = (remaining_budget - reserved_backoff) / remaining_attempts
    return max(1.0, request_budget)
