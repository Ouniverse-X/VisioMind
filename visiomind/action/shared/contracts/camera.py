from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class CameraFrame:
    view: str
    data: Any
    mime_type: str | None = None


class CameraCaptureAdapter(Protocol):
    def capture(self, views: list[str]) -> dict[str, CameraFrame]:
        pass
