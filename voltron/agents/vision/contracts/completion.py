"""Vision completion-evaluation contract."""

from __future__ import annotations

from typing import Protocol

from voltron.shared.models import CompletionEvaluationContext, CompletionVerdict


class VisionCompletionEvaluator(Protocol):
    """Evaluate whether a task, subtask, or internal action step is complete."""

    def evaluate(self, context: CompletionEvaluationContext) -> CompletionVerdict:
        """Return a structured completion verdict."""


__all__ = ["VisionCompletionEvaluator"]
