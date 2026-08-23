"""Configuration loading for the Voltron VLM service integration."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from .models import (
    DEFAULT_DASHSCOPE_MODEL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_OPENAI_VLM_MODEL,
    DEFAULT_RETRY_BACKOFF_S,
    DEFAULT_TIMEOUT_S,
    VLMBackendConfig,
)


def load_backend_config(
    config_path: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> VLMBackendConfig:
    payload = load_json_config(config_path)
    env = environ or os.environ
    llm_cfg = payload.get("brain") if isinstance(payload.get("brain"), dict) else {}
    vlm_cfg = payload.get("vision") if isinstance(payload.get("vision"), dict) else {}

    merged = dict(llm_cfg)
    merged.update(vlm_cfg)

    provider = (
        env.get("VLM_PROVIDER")
        or str(merged.get("provider") or "").strip().lower()
        or ("openai" if merged.get("base_url") else "dashscope")
    )

    if provider == "openai":
        base_url = env.get("VLM_BASE_URL") or merged.get("base_url")
        model = env.get("VLM_MODEL") or merged.get("model") or DEFAULT_OPENAI_VLM_MODEL
        api_key = resolve_api_key(
            direct_key=env.get("VLM_API_KEY") or merged.get("api_key"),
            env_name=env.get("VLM_API_KEY_ENV") or merged.get("api_key_env") or "OPENAI_API_KEY",
            environ=env,
        )
        timeout_s = float(env.get("VLM_TIMEOUT_S") or merged.get("timeout_s") or DEFAULT_TIMEOUT_S)
        max_retries = int(env.get("VLM_MAX_RETRIES") or merged.get("max_retries") or DEFAULT_MAX_RETRIES)
        retry_backoff_s = float(
            env.get("VLM_RETRY_BACKOFF_S") or merged.get("retry_backoff_s") or DEFAULT_RETRY_BACKOFF_S
        )
        if not base_url:
            raise RuntimeError("OpenAI-compatible VLM backend requires base_url")
        return VLMBackendConfig(
            provider="openai",
            model=str(model),
            api_key=api_key,
            base_url=str(base_url),
            timeout_s=timeout_s,
            max_retries=max(0, max_retries),
            retry_backoff_s=max(0.0, retry_backoff_s),
        )

    if provider == "dashscope":
        model = env.get("VLM_MODEL") or merged.get("model") or DEFAULT_DASHSCOPE_MODEL
        api_key = resolve_api_key(
            direct_key=env.get("VLM_API_KEY") or merged.get("api_key"),
            env_name=env.get("VLM_API_KEY_ENV") or merged.get("api_key_env") or "DASHSCOPE_API_KEY",
            environ=env,
        )
        timeout_s = float(env.get("VLM_TIMEOUT_S") or merged.get("timeout_s") or DEFAULT_TIMEOUT_S)
        max_retries = int(env.get("VLM_MAX_RETRIES") or merged.get("max_retries") or DEFAULT_MAX_RETRIES)
        retry_backoff_s = float(
            env.get("VLM_RETRY_BACKOFF_S") or merged.get("retry_backoff_s") or DEFAULT_RETRY_BACKOFF_S
        )
        return VLMBackendConfig(
            provider="dashscope",
            model=str(model),
            api_key=api_key,
            timeout_s=timeout_s,
            max_retries=max(0, max_retries),
            retry_backoff_s=max(0.0, retry_backoff_s),
        )

    raise RuntimeError(f"Unsupported VLM provider: {provider}")


def load_json_config(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError("VLM config file root must be a JSON object")
    return payload


def resolve_api_key(
    direct_key: str | None,
    env_name: str,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    if direct_key:
        return str(direct_key)
    env = environ or os.environ
    return env.get(env_name)
