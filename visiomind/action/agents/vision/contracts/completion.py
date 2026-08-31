from __future__ import annotations

from typing import Protocol

from visiomind.action.shared.models import CompletionEvaluationContext, CompletionVerdict


class VisionCompletionEvaluator(Protocol):
    def evaluate(self, context: CompletionEvaluationContext) -> CompletionVerdict:
        pass


__all__ = ["VisionCompletionEvaluator"]
