"""VLM service integration surface."""

from .backends import DashScopeVLMBackend, OpenAICompatibleVLMBackend, VLMBackend, build_backend
from .client import VLMHttpAdapter
from .config import load_backend_config, load_json_config, resolve_api_key
from .models import (
    DEFAULT_DASHSCOPE_MODEL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_OPENAI_VLM_MODEL,
    DEFAULT_RETRY_BACKOFF_S,
    DEFAULT_TIMEOUT_S,
    VLMBackendConfig,
    VLMProcessRequest,
    VLMProcessResponse,
)
from .prompting import (
    build_dashscope_messages,
    build_openai_messages,
    build_prompt,
    build_system_prompt,
)

__all__ = [
    "DEFAULT_DASHSCOPE_MODEL",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_OPENAI_VLM_MODEL",
    "DEFAULT_RETRY_BACKOFF_S",
    "DEFAULT_TIMEOUT_S",
    "DashScopeVLMBackend",
    "OpenAICompatibleVLMBackend",
    "VLMBackend",
    "VLMBackendConfig",
    "VLMHttpAdapter",
    "VLMProcessRequest",
    "VLMProcessResponse",
    "build_backend",
    "build_dashscope_messages",
    "build_openai_messages",
    "build_prompt",
    "build_system_prompt",
    "load_backend_config",
    "load_json_config",
    "resolve_api_key",
]
