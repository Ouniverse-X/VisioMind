from __future__ import annotations

import base64
import binascii
import re
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from voltron.shared.contracts import CameraFrame, ToolInvocation
from voltron.shared.results import ToolResult

TOOL_NAME = "vision.photo.capture"
TOOL_VERSION = "vision.photo.capture/1"
DEFAULT_VIEWS = ["head"]
ALLOWED_VIEWS = {"head", "left_wrist", "right_wrist"}


class VisionPhotoCaptureTool:
    tool_names = (TOOL_NAME,)

    def invoke(self, invocation: ToolInvocation) -> ToolResult:
        payload = invocation.payload
        camera_capture = payload.get("camera_capture")
        if not callable(getattr(camera_capture, "capture", None)):
            return self._error(
                invocation.tool_name,
                "camera_adapter_missing",
                "camera_capture with callable capture(views) is required",
            )

        try:
            views = self._normalize_views(payload.get("views"))
            unavailable = [view for view in views if view not in ALLOWED_VIEWS]
            if unavailable:
                return self._error(
                    invocation.tool_name,
                    "camera_view_unavailable",
                    f"Unsupported camera view(s): {', '.join(unavailable)}",
                )

            trace_id = self._safe_path_component(payload.get("trace_id"), "unknown_trace")
            subtask_id = self._safe_path_component(payload.get("subtask_id"), "unknown_subtask")
            run_dir = (
                Path(payload["run_dir"])
                if payload.get("run_dir")
                else self._default_run_dir(trace_id)
            )
            photos_base = (run_dir / "photos").resolve()
            output_dir = (photos_base / subtask_id).resolve()
            if not output_dir.is_relative_to(photos_base):
                raise ValueError("photo output directory escaped run photos directory")
            output_dir.mkdir(parents=True, exist_ok=True)

            frames = camera_capture.capture(views)
            timestamp = time.strftime("%Y%m%dT%H%M%S")
            capture_id = uuid.uuid4().hex[:8]
            photos: list[dict[str, str]] = []
            for index, view in enumerate(views):
                if view not in frames:
                    return self._error(
                        invocation.tool_name,
                        "camera_view_unavailable",
                        f"Camera capture did not return frame for view {view!r}",
                    )
                png_bytes = self._frame_to_png_bytes(frames[view])
                path = output_dir / f"{view}_{timestamp}_{capture_id}_{index:02d}.png"
                path.write_bytes(png_bytes)
                photos.append(
                    {
                        "view": view,
                        "path": str(path),
                        "mime_type": "image/png",
                    }
                )
        except ValueError as exc:
            return self._error(invocation.tool_name, "invalid_camera_frame", str(exc))
        except OSError as exc:
            return self._error(invocation.tool_name, "photo_write_failed", str(exc))
        except Exception as exc:
            return self._error(invocation.tool_name, "camera_capture_failed", str(exc))

        return ToolResult(
            tool_name=invocation.tool_name,
            ok=True,
            payload={"task_complete": True, "photos": photos},
            metadata={"tool_version": TOOL_VERSION},
        )

    @staticmethod
    def _normalize_views(value: Any) -> list[str]:
        if isinstance(value, str):
            views = [value.strip()]
        elif isinstance(value, list):
            views = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        else:
            return list(DEFAULT_VIEWS)
        return views or list(DEFAULT_VIEWS)

    @staticmethod
    def _default_run_dir(trace_id: str) -> Path:
        voltron_root = Path(__file__).resolve().parents[3]
        return voltron_root / "integrations" / "simulator" / "runs" / trace_id

    @staticmethod
    def _safe_path_component(value: Any, default: str) -> str:
        raw = str(value or "").strip()
        sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_")
        return sanitized or default

    @classmethod
    def _frame_to_png_bytes(cls, frame: Any) -> bytes:
        data = frame.data if isinstance(frame, CameraFrame) else frame
        mime_type = frame.mime_type if isinstance(frame, CameraFrame) else None

        if isinstance(data, bytes):
            return cls._image_bytes_to_png_bytes(data)
        if isinstance(data, str):
            return cls._image_bytes_to_png_bytes(cls._decode_base64_frame(data, mime_type))
        return cls._array_to_png_bytes(data)

    @staticmethod
    def _decode_base64_frame(data: str, mime_type: str | None) -> bytes:
        encoded = data.strip()
        if encoded.startswith("data:"):
            try:
                _, encoded = encoded.split(",", 1)
            except ValueError as exc:
                raise ValueError("invalid data URL camera frame") from exc
        elif mime_type not in {None, "image/png;base64"}:
            encoded = encoded.strip()

        try:
            return base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("camera frame string is not valid base64") from exc

    @staticmethod
    def _array_to_png_bytes(data: Any) -> bytes:
        array = np.asarray(data)
        if array.ndim == 4 and array.shape[0] == 1:
            array = np.squeeze(array, axis=0)
        if array.ndim != 3:
            raise ValueError(f"camera frame array must be 3D, got shape {array.shape}")
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)

        try:
            from PIL import Image
        except ImportError as exc:
            raise ValueError("PIL is required to encode array camera frames") from exc

        buffer = BytesIO()
        Image.fromarray(array).save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def _image_bytes_to_png_bytes(data: bytes) -> bytes:
        try:
            from PIL import Image, UnidentifiedImageError
        except ImportError as exc:
            raise ValueError("PIL is required to validate camera frame image bytes") from exc

        try:
            with Image.open(BytesIO(data)) as image:
                image.load()
                if image.mode not in {"1", "L", "P", "RGB", "RGBA"}:
                    image = image.convert("RGB")
                output = BytesIO()
                image.save(output, format="PNG")
                return output.getvalue()
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("camera frame bytes are not a valid image") from exc

    @staticmethod
    def _error(tool_name: str, error_code: str, message: str) -> ToolResult:
        return ToolResult(
            tool_name=tool_name,
            ok=False,
            payload={"message": message},
            error_code=error_code,
            metadata={"tool_version": TOOL_VERSION},
        )
