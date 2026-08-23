"""Standalone HTTP service that isolates the licensed AnyGrasp environment."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from .detector import (
        candidate_distribution_summary,
        filter_candidates_by_approach,
        grasp_group_to_candidates,
        validate_detection_inputs,
        workspace_limits,
    )
except ImportError:  # Direct ``python server.py`` execution.
    from detector import (  # type: ignore[no-redef]
        candidate_distribution_summary,
        filter_candidates_by_approach,
        grasp_group_to_candidates,
        validate_detection_inputs,
        workspace_limits,
    )

logger = logging.getLogger("anygrasp_server")
app = FastAPI(title="AnyGrasp Detection Server")
_detector: Any | None = None
_detector_config: dict[str, Any] = {}
_MAX_POINTS = 2_000_000


class DetectRequest(BaseModel):
    points_b64: str
    points_shape: list[int]
    colors_b64: str | None = None
    colors_shape: list[int] | None = None
    region_mask_b64: str | None = None
    region_mask_shape: list[int] | None = None
    region_margin: float = 0.04
    approach_direction: list[float] | None = None
    approach_thresh: float = float(np.pi)
    dense_grasp: bool = False
    collision_detection: bool = True
    apply_nms: bool = True
    top_k: int = 10


class GraspResult(BaseModel):
    score: float
    translation: list[float]
    rotation_matrix: list[list[float]]
    width: float
    depth: float
    height: float


class DetectResponse(BaseModel):
    candidates: list[GraspResult]
    audit: dict[str, Any] | None = None


def _validated_shape(shape: list[int], *, columns: int | None, name: str) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in shape)
    if columns is None:
        if len(normalized) != 1:
            raise ValueError(f"{name} shape must be one-dimensional, got {normalized}")
    elif len(normalized) != 2 or normalized[1] != columns:
        raise ValueError(f"{name} shape must be (N, {columns}), got {normalized}")
    if not normalized or normalized[0] <= 0 or normalized[0] > _MAX_POINTS:
        raise ValueError(f"{name} point count must be in [1, {_MAX_POINTS}]")
    return normalized


def _decode_array(encoded: str, shape: list[int], *, name: str) -> np.ndarray:
    normalized = _validated_shape(shape, columns=3, name=name)
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{name} is not valid base64") from exc
    expected = int(np.prod(normalized)) * np.dtype(np.float32).itemsize
    if len(raw) != expected:
        raise ValueError(f"{name} byte length {len(raw)} does not match expected {expected}")
    return np.frombuffer(raw, dtype=np.float32).reshape(normalized).copy()


def _decode_mask(encoded: str, shape: list[int]) -> np.ndarray:
    normalized = _validated_shape(shape, columns=None, name="region_mask")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("region_mask is not valid base64") from exc
    if len(raw) != normalized[0]:
        raise ValueError("region_mask byte length does not match its shape")
    return np.frombuffer(raw, dtype=np.bool_).reshape(normalized).copy()


def _release_cuda_cache() -> None:
    """Release inference workspaces while keeping the detector weights resident."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        logger.debug("Unable to release the CUDA allocator cache", exc_info=True)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "detector_loaded": _detector is not None,
        **_detector_config,
    }


@app.post("/detect", response_model=DetectResponse)
def detect(request: DetectRequest) -> DetectResponse:
    request_id = str(uuid.uuid4())
    if _detector is None:
        raise HTTPException(503, "detector is not loaded")
    try:
        if request.colors_b64 is None or request.colors_shape is None:
            raise ValueError("colors_b64 and colors_shape are required")
        points = _decode_array(request.points_b64, request.points_shape, name="points")
        colors = _decode_array(request.colors_b64, request.colors_shape, name="colors")
        region_mask = None
        if request.region_mask_b64 is not None or request.region_mask_shape is not None:
            if request.region_mask_b64 is None or request.region_mask_shape is None:
                raise ValueError("region_mask_b64 and region_mask_shape must be provided together")
            region_mask = _decode_mask(request.region_mask_b64, request.region_mask_shape)
        points, colors, region_mask = validate_detection_inputs(points, colors, region_mask)
        if not 1 <= int(request.top_k) <= 100:
            raise ValueError("top_k must be in [1, 100]")
        if not 0.0 <= float(request.approach_thresh) <= float(np.pi):
            raise ValueError("approach_thresh must be in [0, pi]")
        limits = workspace_limits(points, region_mask, request.region_margin)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    audit: dict[str, Any] = {
        "event": "anygrasp_server_detection_audit",
        "request_id": request_id,
        "network_raw_count": None,
        "network_raw_available": False,
        "network_raw_unavailable_reason": (
            "raw network proposals are internal to compiled gsnet.so"
        ),
        "input_point_count": int(len(points)),
        "target_point_count": (
            None if region_mask is None else int(np.count_nonzero(region_mask))
        ),
        "workspace": limits,
        "dense_grasp": bool(request.dense_grasp),
        "collision_detection": bool(request.collision_detection),
        "apply_nms": bool(request.apply_nms),
        "nms_applied": bool(request.apply_nms and not request.dense_grasp),
        "top_k": int(request.top_k),
        "detector_config": dict(_detector_config),
        "approach_filter": {
            "preferred": request.approach_direction,
            "threshold_rad": float(request.approach_thresh),
        },
    }
    candidates = []
    try:
        group, _cloud = _detector.get_grasp(
            points,
            colors,
            lims=limits,
            apply_object_mask=True,
            dense_grasp=request.dense_grasp,
            collision_detection=request.collision_detection,
        )
        sdk_count = 0 if group is None else int(len(group))
        audit["sdk_returned_count"] = sdk_count
        audit["sdk_returned_definition"] = "count returned by compiled SDK get_grasp"
        audit["sdk_returned_distribution"] = (
            candidate_distribution_summary(group, request.approach_direction)
            if sdk_count
            else candidate_distribution_summary([], request.approach_direction)
        )
        if sdk_count:
            if request.apply_nms and not request.dense_grasp:
                group = group.nms()
            audit["post_nms_count"] = int(len(group))
            audit["post_nms_distribution"] = candidate_distribution_summary(
                group, request.approach_direction
            )
            group = group.sort_by_score()
            converted = grasp_group_to_candidates(group)
            audit["post_conversion_count"] = int(len(converted))
            audit["post_conversion_distribution"] = candidate_distribution_summary(
                converted, request.approach_direction
            )
            candidates = filter_candidates_by_approach(
                converted,
                request.approach_direction,
                request.approach_thresh,
            )
            audit["post_approach_count"] = int(len(candidates))
            audit["post_approach_distribution"] = candidate_distribution_summary(
                candidates, request.approach_direction
            )
            candidates = candidates[: request.top_k]
        else:
            empty_distribution = candidate_distribution_summary(
                [], request.approach_direction
            )
            audit["post_nms_count"] = 0
            audit["post_nms_distribution"] = empty_distribution
            audit["post_conversion_count"] = 0
            audit["post_conversion_distribution"] = empty_distribution
            audit["post_approach_count"] = 0
            audit["post_approach_distribution"] = empty_distribution
        audit["post_top_k_count"] = int(len(candidates))
        audit["post_top_k_distribution"] = candidate_distribution_summary(
            candidates, request.approach_direction
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("AnyGrasp inference failed request_id=%s", request_id)
        raise HTTPException(500, f"AnyGrasp inference failed: {type(exc).__name__}: {exc}") from exc
    finally:
        _release_cuda_cache()

    logger.info("%s", json.dumps(audit, sort_keys=True, separators=(",", ":")))
    return DetectResponse(
        candidates=[
            GraspResult(
                score=item.score,
                translation=item.translation.tolist(),
                rotation_matrix=item.rotation_matrix.tolist(),
                width=item.width,
                depth=item.depth,
                height=item.height,
            )
            for item in candidates
        ],
        audit=audit,
    )


def _build_detector(
    sdk_root: str,
    checkpoint: str | None = None,
    *,
    max_gripper_width: float = 0.1,
    gripper_height: float = 0.03,
    top_down_grasp: bool = False,
) -> None:
    global _detector, _detector_config
    detection_dir = Path(sdk_root).expanduser().resolve() / "grasp_detection"
    checkpoint_path = Path(checkpoint).expanduser() if checkpoint else detection_dir / "log" / "checkpoint_detection.tar"
    if not checkpoint_path.is_absolute():
        checkpoint_path = (detection_dir / checkpoint_path).resolve()
    if not detection_dir.is_dir():
        raise RuntimeError(f"AnyGrasp detection directory not found: {detection_dir}")
    if not checkpoint_path.is_file():
        raise RuntimeError(f"AnyGrasp checkpoint not found: {checkpoint_path}")
    if str(detection_dir) not in sys.path:
        sys.path.insert(0, str(detection_dir))

    from argparse import Namespace
    previous_cwd = Path.cwd()
    try:
        os.chdir(detection_dir)
        from gsnet import AnyGrasp  # type: ignore[import-not-found]
        config = Namespace(
            checkpoint_path=str(checkpoint_path),
            max_gripper_width=min(0.1, max(0.0, max_gripper_width)),
            gripper_height=float(gripper_height),
            top_down_grasp=bool(top_down_grasp),
            debug=False,
        )
        detector = AnyGrasp(config)
        detector.load_net()
    finally:
        os.chdir(previous_cwd)
    _detector = detector
    _detector_config = {
        "max_gripper_width": float(config.max_gripper_width),
        "gripper_height": float(config.gripper_height),
        "top_down_grasp": bool(config.top_down_grasp),
    }
    logger.info("AnyGrasp detector loaded (checkpoint=%s)", checkpoint_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="AnyGrasp detection server")
    parser.add_argument("--sdk-root", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max-gripper-width", type=float, default=0.1)
    parser.add_argument("--gripper-height", type=float, default=0.03)
    parser.add_argument("--top-down-grasp", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    _build_detector(
        args.sdk_root,
        args.checkpoint,
        max_gripper_width=args.max_gripper_width,
        gripper_height=args.gripper_height,
        top_down_grasp=args.top_down_grasp,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
