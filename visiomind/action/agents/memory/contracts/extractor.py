from __future__ import annotations

from typing import Any, Protocol

from .experience import ExperienceExtractionResult


class ExperienceExtractor(Protocol):
    def extract(
        self, episode_context: dict[str, Any]
    ) -> ExperienceExtractionResult | dict[str, Any]:
        pass
