from __future__ import annotations

import base64
import io
import json
import math
from typing import Any

from visiomind.action.shared.models import CompletionEvaluationContext, CompletionVerdict
from visiomind.action.shared.telemetry.payload_sanitizer import strip_image_payloads


class VLMCompletionEvaluator:
    def __init__(
        self,
        *,
        vision: Any,
        min_confidence: float = 0.75,
        use_memory_guidance: bool = True,
        max_images: int = 4,
        include_third_person: bool = True,
        max_image_side_px: int = 1024,
        jpeg_quality: int = 90,
        max_image_b64_chars: int = 900_000,
        image_detail: str = "high",
    ) -> None:
        self.vision = vision
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self.use_memory_guidance = bool(use_memory_guidance)
        self.max_images = max(1, int(max_images))
        self.include_third_person = bool(include_third_person)
        self.max_image_side_px = max(32, int(max_image_side_px))
        self.jpeg_quality = max(1, min(95, int(jpeg_quality)))
        self.max_image_b64_chars = max(1_024, int(max_image_b64_chars))
        self.image_detail = self._normalize_image_detail(image_detail)

    def evaluate(self, context: CompletionEvaluationContext) -> CompletionVerdict:
        images, image_view_order = self._extract_images(context.runtime_feedback)
        if not images:
            return CompletionVerdict(
                scope=context.scope,
                scope_id=context.scope_id,
                completed=False,
                confidence=0.0,
                reason="no images were available for Vision completion evaluation",
                evidence={"has_images": False},
                missing_evidence=["images"],
                should_continue=True,
            )

        original_image_count = len(images)
        images, image_view_order = self._limit_images(images, image_view_order)
        selected_image_count = len(images)
        selected_view_order = list(image_view_order)
        images, image_view_order, composite_layout = self._prepare_completion_images(
            images,
            image_view_order,
        )
        report = self.vision.analyze(
            images_b64=images,
            instruction=self._instruction(context, composite_layout=composite_layout),
            task_name=self._task_name(context),
            image_view_order=image_view_order or None,
            image_detail=self.image_detail,
        )
        confidence = self._confidence(report)
        completed = (
            bool(getattr(report, "task_complete", False)) and confidence >= self.min_confidence
        )
        raw_text = str(getattr(report, "raw_text", "") or "")
        evidence = {
            "has_images": True,
            "image_count": len(images),
            "original_image_count": original_image_count,
            "selected_image_count": selected_image_count,
            "selected_image_view_order": selected_view_order,
            "image_view_order": image_view_order,
            "include_third_person": self.include_third_person,
            "composite_layout": composite_layout,
            "max_image_side_px": self.max_image_side_px,
            "image_detail": self.image_detail,
            "vlm_task_complete": bool(getattr(report, "task_complete", False)),
            "min_confidence": self.min_confidence,
        }
        if raw_text:
            evidence["raw_text"] = raw_text[:1000]

        return CompletionVerdict(
            scope=context.scope,
            scope_id=context.scope_id,
            completed=completed,
            confidence=confidence,
            reason=raw_text
            or (
                "completion criteria satisfied"
                if completed
                else "completion criteria not satisfied"
            ),
            evidence=evidence,
            missing_evidence=[] if completed else ["vision_confidence_or_completion"],
            should_continue=not completed,
            source="vision_completion_evaluator",
        )

    def _limit_images(
        self, images: list[str], image_view_order: list[str]
    ) -> tuple[list[str], list[str]]:
        paired = [
            (
                image,
                image_view_order[index] if index < len(image_view_order) else f"view_{index + 1}",
            )
            for index, image in enumerate(images)
        ]
        if not self.include_third_person:
            paired = [
                (image, view_name)
                for image, view_name in paired
                if self._normalize_view_name(view_name) != "third_person"
            ]
        limited = paired[: self.max_images]
        limited_images = [image for image, _ in limited]
        limited_order = [view_name for _, view_name in limited]
        return limited_images, limited_order

    def _prepare_completion_images(
        self,
        images: list[str],
        image_view_order: list[str],
    ) -> tuple[list[str], list[str], str | None]:
        if len(images) <= 1:
            return (
                [self._compact_image_b64(image) for image in images],
                image_view_order,
                None,
            )

        composite = self._compose_images_b64(images)
        if composite is None:
            return [self._compact_image_b64(images[0])], image_view_order[:1], None
        layout = self._composite_layout_description(image_view_order)
        return (
            [self._compact_image_b64(composite)],
            [f"composite: {layout}"],
            layout,
        )

    def _compose_images_b64(self, images: list[str]) -> str | None:
        try:
            from PIL import Image, ImageOps

            decoded = [
                Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
                for image_b64 in images
            ]
            columns = max(1, int(math.ceil(math.sqrt(len(decoded)))))
            rows = int(math.ceil(len(decoded) / columns))
            tile_width = min(self.max_image_side_px, max(image.width for image in decoded))
            tile_height = min(self.max_image_side_px, max(image.height for image in decoded))
            canvas = Image.new(
                "RGB",
                (columns * tile_width, rows * tile_height),
                color=(24, 24, 24),
            )
            for index, image in enumerate(decoded):
                tile = ImageOps.pad(
                    image,
                    (tile_width, tile_height),
                    method=Image.Resampling.LANCZOS,
                    color=(24, 24, 24),
                    centering=(0.5, 0.5),
                )
                canvas.paste(
                    tile,
                    ((index % columns) * tile_width, (index // columns) * tile_height),
                )
            buffer = io.BytesIO()
            canvas.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("ascii")
        except Exception:
            return None

    @classmethod
    def _composite_layout_description(cls, image_view_order: list[str]) -> str:
        count = len(image_view_order)
        columns = max(1, int(math.ceil(math.sqrt(count))))
        rows = int(math.ceil(count / columns))
        tiles = []
        for index, view_name in enumerate(image_view_order):
            position = cls._grid_position(
                row=index // columns,
                column=index % columns,
                rows=rows,
                columns=columns,
            )
            tiles.append(f"{position}: {cls._humanize_view_name(view_name)}")
        return f"{rows}x{columns} grid; " + "; ".join(tiles)

    @staticmethod
    def _grid_position(*, row: int, column: int, rows: int, columns: int) -> str:
        if rows == 1:
            return (
                "left" if column == 0 and columns > 1 else "right" if columns > 1 else "full image"
            )
        vertical = "top" if row == 0 else "bottom" if row == rows - 1 else f"row {row + 1}"
        horizontal = (
            "left" if column == 0 else "right" if column == columns - 1 else f"column {column + 1}"
        )
        return f"{vertical}-{horizontal}"

    @classmethod
    def _humanize_view_name(cls, view_name: str) -> str:
        mapping = {
            "head": "head camera",
            "left_wrist": "left wrist camera",
            "right_wrist": "right wrist camera",
            "third_person": "third-person camera",
        }
        normalized = cls._normalize_view_name(view_name)
        return mapping.get(normalized, normalized.replace("_", " "))

    @staticmethod
    def _normalize_view_name(view_name: str) -> str:
        return str(view_name or "").strip().lower().replace("-", "_").replace(" ", "_")

    @staticmethod
    def _task_name(context: CompletionEvaluationContext) -> str:
        task_name = f"{context.scope}:{context.scope_id}:completion"
        try:
            control_step = int((context.action_stability or {}).get("control_step"))
        except (TypeError, ValueError):
            return task_name
        return f"{task_name}:step_{control_step:04d}"

    def _compact_image_b64(self, image_b64: str) -> str:
        try:
            from PIL import Image

            raw = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            side_px = self.max_image_side_px
            quality = self.jpeg_quality
            best = image_b64
            while side_px >= 32:
                compact = image.copy()
                compact.thumbnail((side_px, side_px))
                buffer = io.BytesIO()
                compact.save(buffer, format="JPEG", quality=quality, optimize=True)
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                best = encoded
                if len(encoded) <= self.max_image_b64_chars:
                    return encoded
                if quality > 35:
                    quality = max(35, quality - 10)
                else:
                    side_px //= 2
            return best
        except Exception:
            return image_b64

    def _instruction(
        self,
        context: CompletionEvaluationContext,
        *,
        composite_layout: str | None = None,
    ) -> str:
        payload = context.to_prompt_payload()
        if not self.use_memory_guidance:
            payload.pop("memory_evidence", None)
        implicit_criteria = self._implicit_completion_criteria(context)
        if implicit_criteria:
            payload["implicit_completion_criteria"] = implicit_criteria
        if composite_layout:
            payload["completion_image_layout"] = composite_layout
        payload["runtime_feedback"] = self._prompt_safe_runtime_feedback(
            payload.get("runtime_feedback"),
            original=context.runtime_feedback,
        )
        payload = strip_image_payloads(payload)
        layout_instruction = (
            "The single provided completion image is a multi-view composite arranged as: "
            f"{composite_layout}. Use every tile and respect these view labels.\n"
            if composite_layout
            else ""
        )
        return (
            "Decide whether the current VisioMindAction task scope is complete using the images and "
            "the structured context. Return a concise result through the VLM service fields: "
            "task_complete/is_success, optional confidence, and raw_text explanation. "
            "Do not mark complete unless the observable completion criteria are satisfied.\n"
            f"{layout_instruction}"
            f"Context JSON:\n{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
        )

    @classmethod
    def _implicit_completion_criteria(
        cls, context: CompletionEvaluationContext
    ) -> list[dict[str, Any]]:
        subtask = dict(context.current_subtask or {})
        action = str(subtask.get("action") or "").strip().lower()
        if action != "open":
            return []
        target = subtask.get("target")
        target_payload = target if isinstance(target, dict) else {}
        target_text = " ".join(
            str(value).strip().lower()
            for key in ("object", "object_name", "category", "part")
            for value in (target_payload.get(key),)
            if value not in (None, "")
        )
        instruction = str(subtask.get("instruction") or "").strip().lower()
        if not cls._is_portal_opening_target(f"{target_text} {instruction}"):
            return []
        return [
            {
                "criterion_id": "implicit_portal_opening_passable",
                "scope": context.scope,
                "subtask_id": context.scope_id,
                "description": (
                    "For opening a door, gate, or doorway used for navigation, success requires a passable "
                    "opening: the doorway must be open wide enough that the robot can pass through, not just "
                    "a narrow gap between the door and frame."
                ),
                "positive_evidence": [
                    "The door leaf is clearly swung or slid out of the doorway path.",
                    "The visible opening is wide enough for the robot body and carried object to pass through.",
                    "The floor path through the doorway is not blocked by the door panel.",
                ],
                "negative_evidence": [
                    "Only a narrow gap is visible.",
                    "The door remains mostly closed or still blocks the doorway.",
                    "The handle moved but the robot cannot pass through the opening.",
                ],
                "success_verifier": "vision_completion_evaluator",
            }
        ]

    @staticmethod
    def _is_portal_opening_target(text: str) -> bool:
        normalized = str(text or "").replace("_", " ").replace("-", " ").lower()
        tokens = set(normalized.split())
        return bool(tokens & {"door", "doorway", "gate", "gateway", "portal"})

    @classmethod
    def _prompt_safe_runtime_feedback(
        cls, value: Any, *, original: dict[str, Any]
    ) -> dict[str, Any]:
        payload = dict(value) if isinstance(value, dict) else {}
        images, image_view_order = cls._extract_images(original)
        cls._remove_image_payloads(payload)
        extras = payload.get("extras")
        if isinstance(extras, dict):
            cls._remove_image_payloads(extras)
            if image_view_order:
                extras["image_view_order"] = list(image_view_order)
        if images:
            payload["image_count"] = len(images)
        if image_view_order:
            payload["image_view_order"] = list(image_view_order)
        return payload

    @classmethod
    def _remove_image_payloads(cls, payload: dict[str, Any]) -> None:
        for key in ("images_b64", "images", "rgb"):
            payload.pop(key, None)

    @classmethod
    def _extract_images(cls, runtime_feedback: dict[str, Any]) -> tuple[list[str], list[str]]:
        candidates = [runtime_feedback]
        extras = runtime_feedback.get("extras")
        if isinstance(extras, dict):
            candidates.insert(0, extras)
        for payload in candidates:
            images = cls._image_list(payload)
            if images:
                order = payload.get("image_view_order")
                return images, [str(item) for item in order] if isinstance(order, list) else []
        return [], []

    @staticmethod
    def _image_list(payload: dict[str, Any]) -> list[str]:
        for key in ("images_b64", "images"):
            value = payload.get(key)
            if isinstance(value, list):
                images = [str(item) for item in value if str(item).strip()]
                if images:
                    return images
        rgb = payload.get("rgb")
        if isinstance(rgb, dict):
            images = [
                str(value) for value in rgb.values() if isinstance(value, str) and value.strip()
            ]
            if images:
                return images
        return []

    @staticmethod
    def _normalize_image_detail(value: str) -> str:
        normalized = str(value or "high").strip().lower()
        return normalized if normalized in {"low", "high", "auto"} else "high"

    @staticmethod
    def _confidence(report: Any) -> float:
        metadata = getattr(report, "metadata", {}) if report is not None else {}
        raw = metadata.get("raw_response") if isinstance(metadata, dict) else {}
        if isinstance(raw, dict):
            for key in ("confidence", "completion_confidence", "score"):
                try:
                    return max(0.0, min(1.0, float(raw[key])))
                except (KeyError, TypeError, ValueError):
                    pass
        return 1.0 if bool(getattr(report, "task_complete", False)) else 0.0


__all__ = ["VLMCompletionEvaluator"]
