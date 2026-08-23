"""FastAPI entrypoint for the Voltron VLM service."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import re
from typing import Any

from voltron.integrations.vlm.service.backends import VLMBackend, VLMHTTPError, build_backend
from voltron.integrations.vlm.service.config import load_backend_config
from voltron.integrations.vlm.service.debug_utils import save_debug_images
from voltron.integrations.vlm.service.models import VLMBackendConfig, VLMProcessRequest

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.getenv("VLM_CONFIG_PATH"))
    parser.add_argument("--host", type=str, default=os.getenv("VLM_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("VLM_PORT", "8081")))
    parser.add_argument(
        "--debug-image-root",
        type=str,
        default=os.getenv("VLM_DEBUG_IMAGE_ROOT", "debug_images"),
    )
    parser.add_argument("--disable-debug-image-save", action="store_true")
    return parser


def create_app(
    *,
    backend_config: VLMBackendConfig | None = None,
    backend: VLMBackend | None = None,
    debug_image_root: str | Path | None = "debug_images",
):
    try:
        from fastapi import Body, FastAPI, HTTPException
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency dependent
        raise RuntimeError("fastapi and pydantic are required for VLM service mode.") from exc

    resolved_config = backend_config
    resolved_backend = backend
    if resolved_backend is None:
        resolved_config = resolved_config or load_backend_config()
        resolved_backend = build_backend(resolved_config)

    app = FastAPI(title="Voltron VLM Service", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "service": "vlm_service",
            "provider": resolved_config.provider if resolved_config is not None else "custom",
            "model": resolved_config.model if resolved_config is not None else None,
        }

    @app.post("/process")
    def process_video_stream(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        images = payload.get("images")
        instruction = payload.get("instruction")
        if not isinstance(images, list):
            raise HTTPException(status_code=422, detail="'images' must be a list")
        if not isinstance(instruction, str):
            raise HTTPException(status_code=422, detail="'instruction' must be a string")

        vlm_request = VLMProcessRequest(
            images=list(images),
            instruction=instruction,
            task_name=str(payload.get("task_name") or "unknown"),
            image_view_order=[
                str(item)
                for item in payload.get("image_view_order", [])
                if isinstance(item, str) and str(item).strip()
            ],
            image_detail=str(payload.get("image_detail") or "low"),
        )
        if not vlm_request.images:
            return {"status": "skipped", "message": "No images provided"}

        if debug_image_root is not None:
            save_debug_images(
                vlm_request.images,
                root_dir=debug_image_root,
                task_name=vlm_request.task_name,
                instruction=vlm_request.instruction,
                image_view_order=vlm_request.image_view_order,
            )

        LOGGER.info(
            "Processing VLM request task=%s frames=%s provider=%s model=%s",
            vlm_request.task_name,
            len(vlm_request.images),
            resolved_config.provider if resolved_config is not None else "custom",
            resolved_config.model if resolved_config is not None else None,
        )

        try:
            result = resolved_backend.analyze(vlm_request)
        except VLMHTTPError as exc:
            LOGGER.exception("VLM processing failed")
            raise HTTPException(status_code=_backend_http_status(exc), detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("VLM processing failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        payload = result.to_payload()
        LOGGER.info(
            "VLM request completed success=%s preview=%s",
            payload["is_success"],
            str(payload["result"])[:160],
        )
        return payload

    return app


def _backend_http_status(exc: VLMHTTPError) -> int:
    match = re.search(r"\bHTTP\s+(\d{3})\b", str(exc))
    if not match:
        return 502
    status_code = int(match.group(1))
    if 400 <= status_code <= 599:
        return status_code
    return 502


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    debug_image_root = None
    if not args.disable_debug_image_save and str(args.debug_image_root).strip():
        debug_image_root = args.debug_image_root

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    try:
        import uvicorn
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency dependent
        raise RuntimeError("uvicorn is required to run VLM service") from exc

    backend_config = load_backend_config(args.config)
    LOGGER.info(
        "Starting VLM service provider=%s model=%s base_url=%s timeout_s=%s",
        backend_config.provider,
        backend_config.model,
        backend_config.base_url,
        backend_config.timeout_s,
    )
    app = create_app(backend_config=backend_config, debug_image_root=debug_image_root)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
