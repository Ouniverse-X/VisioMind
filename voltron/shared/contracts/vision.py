"""Vision adapter interface for VLM service backends."""

from __future__ import annotations

from typing import Protocol

from voltron.shared.models import PerceptionReport


class VisionAdapter(Protocol):
    """Protocol implemented by VLM backends."""

    def analyze(
        self,
        images_b64: list[str],
        instruction: str,
        task_name: str,
    ) -> PerceptionReport:
        """Analyze image sequence and return a structured perception report."""
