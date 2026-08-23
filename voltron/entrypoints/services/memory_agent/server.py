"""Memory Agent service.

Exposes a thin RPC boundary in front of `MemoryAgent(HEMSAdapter)` by default.
This isolates all business agents from direct HEMS coupling.
"""

from __future__ import annotations

from typing import Any

from voltron.agents.memory import MemoryAgent
from voltron.integrations.memory.hems.backend import HEMSAdapter
from voltron.integrations.memory.service import rpc_runtime
from voltron.runtime.assembly import backend_factory
from voltron.shared.contracts import MemoryAdapter

try:
    from pydantic import BaseModel, Field
except ModuleNotFoundError:  # pragma: no cover - dependency dependent
    BaseModel = None  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]


if BaseModel is not None:

    class RpcRequest(BaseModel):
        method: str
        kwargs: dict[str, Any] = Field(default_factory=dict)


def create_app(
    memory_backend: MemoryAdapter | None = None,
    *,
    memory_agent_enabled: bool = True,
    memory_llm_backend: str | None = None,
    memory_llm_base_url: str | None = None,
    memory_llm_model: str | None = None,
    memory_llm_api_key: str | None = None,
    memory_llm_api_key_env: str = "OPENAI_API_KEY",
    memory_llm_timeout_s: float = 30.0,
    memory_llm_temperature: float = 0.0,
    memory_llm_max_retries: int = 0,
    memory_llm_retry_backoff_s: float = 1.0,
    memory_experience_extraction_enabled: bool = False,
    memory_experience_extraction_min_confidence_to_write: float = 0.4,
    memory_experience_extraction_min_confidence_to_promote: float = 0.7,
):
    """Create FastAPI app instance for Memory Agent service.

    Imported lazily so users without FastAPI can still use local-only mode.
    """
    try:
        from fastapi import FastAPI
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency dependent
        raise RuntimeError(
            "fastapi and pydantic are required for Memory Agent service mode."
        ) from exc
    if BaseModel is None:
        raise RuntimeError("pydantic is required for Memory Agent service mode.")

    if memory_backend is not None:
        backend = memory_backend
    else:
        hems_backend = HEMSAdapter()
        if memory_agent_enabled:
            extractor = backend_factory.build_memory_extractor(
                enabled=memory_experience_extraction_enabled,
                backend=memory_llm_backend,
                base_url=memory_llm_base_url,
                model=memory_llm_model,
                api_key=memory_llm_api_key,
                api_key_env=memory_llm_api_key_env,
                timeout_s=memory_llm_timeout_s,
                temperature=memory_llm_temperature,
                max_retries=memory_llm_max_retries,
                retry_backoff_s=memory_llm_retry_backoff_s,
            )
            backend = MemoryAgent(
                backend=hems_backend,
                extractor=extractor,
                experience_extraction_enabled=memory_experience_extraction_enabled,
                experience_consolidation_async=True,
                min_confidence_to_write=memory_experience_extraction_min_confidence_to_write,
                min_confidence_to_promote=memory_experience_extraction_min_confidence_to_promote,
            )
        else:
            backend = hems_backend
    methods = rpc_runtime.build_rpc_method_table(backend)

    app = FastAPI(title="Voltron Memory Agent", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "service": "memory_agent",
            "memory_agent_enabled": isinstance(backend, MemoryAgent),
            "experience_extraction_enabled": memory_experience_extraction_enabled
            if isinstance(backend, MemoryAgent)
            else False,
        }

    @app.post("/rpc")
    def rpc(request: RpcRequest) -> dict[str, Any]:
        method = request.method
        kwargs = rpc_runtime.normalize_rpc_kwargs(method=method, kwargs=dict(request.kwargs))
        return rpc_runtime.dispatch_rpc_call(methods=methods, method=method, kwargs=kwargs)

    return app


def main() -> None:
    """Run Memory Agent service from CLI.

    Example:
        python -m voltron.entrypoints.services.memory_agent.server --host 0.0.0.0 --port 8070
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8070)
    parser.add_argument("--memory-agent-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--memory-llm-backend", type=str, default=None)
    parser.add_argument("--memory-llm-base-url", type=str, default=None)
    parser.add_argument("--memory-llm-model", type=str, default=None)
    parser.add_argument("--memory-llm-api-key", type=str, default=None)
    parser.add_argument("--memory-llm-api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--memory-llm-timeout-s", type=float, default=30.0)
    parser.add_argument("--memory-llm-temperature", type=float, default=0.0)
    parser.add_argument("--memory-llm-max-retries", type=int, default=0)
    parser.add_argument("--memory-llm-retry-backoff-s", type=float, default=1.0)
    parser.add_argument(
        "--memory-experience-extraction-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--memory-experience-extraction-min-confidence-to-write", type=float, default=0.4)
    parser.add_argument("--memory-experience-extraction-min-confidence-to-promote", type=float, default=0.7)
    args = parser.parse_args()

    try:
        import uvicorn
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency dependent
        raise RuntimeError("uvicorn is required to run Memory Agent server") from exc

    app = create_app(
        memory_agent_enabled=args.memory_agent_enabled,
        memory_llm_backend=args.memory_llm_backend,
        memory_llm_base_url=args.memory_llm_base_url,
        memory_llm_model=args.memory_llm_model,
        memory_llm_api_key=args.memory_llm_api_key,
        memory_llm_api_key_env=args.memory_llm_api_key_env,
        memory_llm_timeout_s=args.memory_llm_timeout_s,
        memory_llm_temperature=args.memory_llm_temperature,
        memory_llm_max_retries=args.memory_llm_max_retries,
        memory_llm_retry_backoff_s=args.memory_llm_retry_backoff_s,
        memory_experience_extraction_enabled=args.memory_experience_extraction_enabled,
        memory_experience_extraction_min_confidence_to_write=args.memory_experience_extraction_min_confidence_to_write,
        memory_experience_extraction_min_confidence_to_promote=(
            args.memory_experience_extraction_min_confidence_to_promote
        ),
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
