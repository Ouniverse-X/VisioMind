"""Industrial perception, visual detection, and 3D segmentation package."""

from __future__ import annotations

from visiomind.perception.industrial_detector import (
    INDUSTRIAL_CLASSES,
    CLASS_TO_ID,
    ID_TO_CLASS,
    IndustrialDetection,
    IndustrialPerceptionResult,
    IndustrialPartDetector,
)
from visiomind.perception.evaluator import (
    DetectionMetricsEvaluator,
    compute_iou,
    compute_mask_iou,
)

__all__ = [
    "INDUSTRIAL_CLASSES",
    "CLASS_TO_ID",
    "ID_TO_CLASS",
    "IndustrialDetection",
    "IndustrialPerceptionResult",
    "IndustrialPartDetector",
    "DetectionMetricsEvaluator",
    "compute_iou",
    "compute_mask_iou",
]
