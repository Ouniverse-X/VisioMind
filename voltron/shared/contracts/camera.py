"""Camera capture contracts for vision-facing adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class CameraFrame:
    """Captured camera frame payload."""

    view: str
    data: Any
    mime_type: str | None = None


class CameraCaptureAdapter(Protocol):
    """Protocol implemented by camera capture backends."""

    def capture(self, views: list[str]) -> dict[str, CameraFrame]:
        """Capture frames for the requested camera views."""
