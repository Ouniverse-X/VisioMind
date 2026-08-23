"""Runtime session helpers and models."""

from .episode_session import EpisodeSession, finalize_episode_session, start_episode_session
from .runtime_context_store import RuntimeContextStore, merge_runtime_context
from .task_session import (
    build_reset_ok_payload,
    build_reset_result_payload,
    build_run_start_payload,
    resolve_request_runtime_metadata,
)
from .events import VoltronEvent

_VOLTRON_SESSION_EXPORTS = {
    "VoltronSession",
    "build_configured_voltron_session",
    "build_mock_voltron_session",
}

__all__ = [
    "EpisodeSession",
    "RuntimeContextStore",
    "VoltronEvent",
    "VoltronSession",
    "build_reset_ok_payload",
    "build_reset_result_payload",
    "build_run_start_payload",
    "build_configured_voltron_session",
    "build_mock_voltron_session",
    "finalize_episode_session",
    "merge_runtime_context",
    "resolve_request_runtime_metadata",
    "start_episode_session",
]


def __getattr__(name: str):
    if name in _VOLTRON_SESSION_EXPORTS:
        from . import voltron_session

        return getattr(voltron_session, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
