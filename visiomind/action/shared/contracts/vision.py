from __future__ import annotations

from typing import Protocol

from visiomind.action.shared.models import PerceptionReport


class VisionAdapter(Protocol):
    def analyze(
        self,
        images_b64: list[str],
        instruction: str,
        task_name: str,
    ) -> PerceptionReport:
        pass
