from __future__ import annotations

from typing import Any


def build_end_task_result(*, episode: Any | None, outcome_label: str) -> dict[str, Any]:
    if episode is None:
        return {"episode_id": None, "outcome": outcome_label, "warning": "no_active_episode"}

    return {
        "episode_id": episode.episode_id,
        "outcome": episode.outcome.value,
        "duration_s": episode.duration(),
        "failure_reason": episode.failure_reason,
    }


__all__ = ["build_end_task_result"]
