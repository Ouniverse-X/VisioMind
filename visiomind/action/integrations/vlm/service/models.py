from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_DASHSCOPE_MODEL = "qwen3-vl-plus"
DEFAULT_OPENAI_VLM_MODEL = "gemini-3-flash-preview"
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_RETRIES = 1
DEFAULT_RETRY_BACKOFF_S = 2.0


@dataclass(frozen=True)
class VLMBackendConfig:
    provider: str
    model: str
    api_key: str | None
    base_url: str | None = None
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_backoff_s: float = DEFAULT_RETRY_BACKOFF_S


@dataclass(frozen=True)
class VLMProcessRequest:
    images: list[str]
    instruction: str
    task_name: str = "unknown"
    image_view_order: list[str] = field(default_factory=list)
    image_detail: str = "low"


@dataclass(frozen=True)
class VLMProcessResponse:
    status: str
    result: str
    is_success: bool
    task_complete: bool = False
    raw_text: str = ""
    scene_report: dict[str, Any] = field(default_factory=dict)
    objects: list[dict[str, Any]] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "result": self.result,
            "is_success": self.is_success,
            "task_complete": self.task_complete,
            "raw_text": self.raw_text or self.result,
            "scene_report": dict(self.scene_report),
            "objects": list(self.objects),
            "relations": list(self.relations),
        }
