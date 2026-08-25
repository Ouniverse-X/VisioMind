"""Unit and integration tests for the industrial vision detection and segmentation model."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from visiomind.perception.evaluator import (
    DetectionMetricsEvaluator,
    compute_iou,
    compute_mask_iou,
)
from visiomind.perception.industrial_detector import (
    CLASS_TO_ID,
    ID_TO_CLASS,
    INDUSTRIAL_CLASSES,
    IndustrialDetection,
    IndustrialDetectorNetwork,
    IndustrialPartDetector,
    MockIndustrialDetector,
    nms_boxes,
)
from voltron.integrations.manipulation.anygrasp.observation import (
    target_mask_from_industrial_detector,
)


def test_classes_and_mapping() -> None:
    assert "bolt" in INDUSTRIAL_CLASSES
    assert "wrench" in INDUSTRIAL_CLASSES
    assert "roller" in INDUSTRIAL_CLASSES
    assert "screwdriver" in INDUSTRIAL_CLASSES
    assert "pliers" in INDUSTRIAL_CLASSES
    assert "toolbox" in INDUSTRIAL_CLASSES
    assert "parts_bin" in INDUSTRIAL_CLASSES

    for idx, name in enumerate(INDUSTRIAL_CLASSES):
        assert CLASS_TO_ID[name] == idx
        assert ID_TO_CLASS[idx] == name


def test_iou_calculations() -> None:
    box1 = [10.0, 10.0, 50.0, 50.0]
    box2 = [10.0, 10.0, 50.0, 50.0]
    assert compute_iou(box1, box2) == pytest.approx(1.0)

    box3 = [30.0, 10.0, 70.0, 50.0]
    iou = compute_iou(box1, box3)
    # Area1: 1600, Area2: 1600, Inter: 20 * 40 = 800, Union: 3200 - 800 = 2400 -> 1/3
    assert iou == pytest.approx(1.0 / 3.0)

    # Disjoint boxes
    box4 = [100.0, 100.0, 150.0, 150.0]
    assert compute_iou(box1, box4) == pytest.approx(0.0)

    # Mask IoU
    m1 = np.zeros((10, 10), dtype=bool)
    m2 = np.zeros((10, 10), dtype=bool)
    m1[2:6, 2:6] = True
    m2[2:6, 2:6] = True
    assert compute_mask_iou(m1, m2) == pytest.approx(1.0)


def test_nms_filtering() -> None:
    boxes = np.array([
        [10.0, 10.0, 50.0, 50.0],
        [12.0, 11.0, 49.0, 51.0],  # Heavy overlap
        [100.0, 100.0, 150.0, 150.0],  # Disjoint
    ], dtype=np.float32)
    scores = np.array([0.90, 0.75, 0.85], dtype=np.float32)

    keep = nms_boxes(boxes, scores, iou_threshold=0.45)
    assert 0 in keep
    assert 2 in keep
    assert 1 not in keep  # Suppressed


def test_detector_network_forward() -> None:
    net = IndustrialDetectorNetwork(num_classes=len(INDUSTRIAL_CLASSES))
    dummy_input = torch.randn(2, 3, 256, 256)
    cls_logits, box_preds, mask_logits = net(dummy_input)

    assert cls_logits.shape == (2, len(INDUSTRIAL_CLASSES), 16, 16)
    assert box_preds.shape == (2, 4, 16, 16)
    assert mask_logits.shape == (2, len(INDUSTRIAL_CLASSES), 256, 256)


def test_industrial_detector_inference_and_3d_lifting() -> None:
    detector = IndustrialPartDetector(device="cpu", conf_threshold=0.20)
    rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    depth = np.ones((256, 256), dtype=np.float32) * 1.2  # 1.2m depth
    intrinsics = np.array([[200.0, 0.0, 128.0], [0.0, 200.0, 128.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    camera_pose = np.eye(4, dtype=np.float64)
    camera_pose[2, 3] = 0.5  # z offset

    res = detector.detect(
        rgb_image=rgb,
        depth_image=depth,
        camera_intrinsics=intrinsics,
        camera_pose_world=camera_pose,
    )
    assert res.latency_ms > 0.0
    assert res.image_shape == (256, 256)
    assert isinstance(res.detections, list)


def test_mock_industrial_detector() -> None:
    gt_objs = [
        {
            "name": "wrench_001",
            "bbox_xyxy": [20.0, 30.0, 80.0, 90.0],
            "position": [0.5, -0.2, 0.8],
            "aabb": [[0.4, -0.25, 0.75], [0.6, -0.15, 0.85]],
            "confidence": 0.98,
        },
        {
            "name": "bolt_002",
            "bbox_xyxy": [120.0, 140.0, 150.0, 170.0],
            "position": [0.3, 0.1, 0.8],
            "confidence": 0.95,
        },
    ]
    mock_det = MockIndustrialDetector(ground_truth_objects=gt_objs)
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    res = mock_det.detect(rgb)

    assert len(res.detections) == 2
    wrench_det = res.get_highest_confidence("wrench")
    assert wrench_det is not None
    assert wrench_det.class_name == "wrench"
    assert wrench_det.confidence == 0.98

    bolt_det = res.get_highest_confidence("bolt")
    assert bolt_det is not None
    assert bolt_det.class_name == "bolt"


def test_evaluation_metrics_computation() -> None:
    evaluator = DetectionMetricsEvaluator()
    gt = [
        {"class_name": "wrench", "bbox_xyxy": [10.0, 10.0, 50.0, 50.0], "centroid_world": [0.5, 0.0, 0.8]},
        {"class_name": "bolt", "bbox_xyxy": [60.0, 60.0, 90.0, 90.0], "centroid_world": [0.6, 0.1, 0.8]},
    ]
    preds = [
        IndustrialDetection(
            class_name="wrench",
            class_id=CLASS_TO_ID["wrench"],
            confidence=0.95,
            bbox_xyxy=[11.0, 10.0, 51.0, 50.0],
            centroid_world=[0.51, 0.005, 0.8],
        ),
        IndustrialDetection(
            class_name="bolt",
            class_id=CLASS_TO_ID["bolt"],
            confidence=0.92,
            bbox_xyxy=[62.0, 60.0, 88.0, 90.0],
            centroid_world=[0.605, 0.1, 0.8],
        ),
    ]
    evaluator.add_frame_evaluation(predicted=preds, ground_truth=gt, image_id=1)
    metrics = evaluator.compute_metrics()

    assert metrics["mAP_0.5"] == 1.0
    assert metrics["localization_3d_error_cm"]["p50_median"] < 2.0  # < 2 cm


def test_target_mask_from_industrial_detector_fallback() -> None:
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    # Target mask for mock or real detector
    gt_objs = [{"name": "pliers_01", "bbox_xyxy": [20, 20, 60, 80], "confidence": 0.99}]
    mock = MockIndustrialDetector(ground_truth_objects=gt_objs)
    mask = target_mask_from_industrial_detector(rgb, "pliers_01", detector=mock)
    assert mask is not None
    assert mask.shape == (120, 160)
    assert bool(mask[30, 30]) is True
    assert bool(mask[0, 0]) is False
