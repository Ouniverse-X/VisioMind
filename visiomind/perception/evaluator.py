from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from visiomind.perception.industrial_detector import (
    INDUSTRIAL_CLASSES,
    IndustrialDetection,
    IndustrialPerceptionResult,
)


def compute_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union <= 1e-9:
        return 0.0
    return float(intersection / union)


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    m1 = np.asarray(mask1, dtype=bool)
    m2 = np.asarray(mask2, dtype=bool)
    if m1.shape != m2.shape:
        return 0.0
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


@dataclass
class DetectionMetricsEvaluator:
    classes: tuple[str, ...] = INDUSTRIAL_CLASSES
    iou_thresholds: tuple[float, ...] = tuple(np.arange(0.5, 1.0, 0.05).round(2).tolist())
    records: list[dict[str, Any]] = field(default_factory=list)

    def add_frame_evaluation(
        self,
        predicted: IndustrialPerceptionResult | list[IndustrialDetection],
        ground_truth: list[dict[str, Any]],
        image_id: str | int = 0,
    ) -> dict[str, Any]:
        preds = (
            predicted.detections if isinstance(predicted, IndustrialPerceptionResult) else predicted
        )

        frame_record: dict[str, Any] = {
            "image_id": image_id,
            "predictions": [p.to_dict() for p in preds],
            "ground_truth": ground_truth,
        }
        self.records.append(frame_record)
        return frame_record

    def compute_metrics(self) -> dict[str, Any]:
        per_class_stats: dict[str, dict[str, Any]] = {}
        localization_errors_3d_cm: list[float] = []

        all_target_classes = [c for c in self.classes if c != "background"]

        for cls in all_target_classes:
            tp_at_iou50 = 0
            fp_at_iou50 = 0
            fn_at_iou50 = 0
            ious_list: list[float] = []
            map_ious: dict[float, list[int]] = {thresh: [] for thresh in self.iou_thresholds}
            total_gt = 0

            for record in self.records:
                gt_boxes = [g for g in record["ground_truth"] if g.get("class_name") == cls]
                pr_boxes = [p for p in record["predictions"] if p.get("class_name") == cls]
                total_gt += len(gt_boxes)

                matched_gt = set()

                pr_sorted = sorted(pr_boxes, key=lambda x: x.get("confidence", 0.0), reverse=True)

                for pr in pr_sorted:
                    best_iou = 0.0
                    best_gt_idx = -1
                    for idx, gt in enumerate(gt_boxes):
                        iou = compute_iou(pr["bbox_xyxy"], gt["bbox_xyxy"])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = idx

                    ious_list.append(best_iou)

                    if best_gt_idx >= 0 and best_gt_idx not in matched_gt and best_iou >= 0.5:
                        matched_gt.add(best_gt_idx)
                        gt_match = gt_boxes[best_gt_idx]
                        if pr.get("centroid_world") and gt_match.get("centroid_world"):
                            p_3d = np.array(pr["centroid_world"])
                            g_3d = np.array(gt_match["centroid_world"])
                            err_cm = float(np.linalg.norm(p_3d - g_3d) * 100.0)
                            localization_errors_3d_cm.append(err_cm)

                    if best_iou >= 0.5:
                        tp_at_iou50 += 1
                    else:
                        fp_at_iou50 += 1

                    for thresh in self.iou_thresholds:
                        map_ious[thresh].append(1 if best_iou >= thresh else 0)

                fn_at_iou50 += max(0, len(gt_boxes) - len(matched_gt))

            precision = tp_at_iou50 / max(1, (tp_at_iou50 + fp_at_iou50))
            recall = tp_at_iou50 / max(1, total_gt)
            f1 = 2 * precision * recall / max(1e-6, (precision + recall))

            ap_50 = precision if total_gt > 0 else 0.0
            ap_list = [
                np.mean(map_ious[t]) if len(map_ious[t]) > 0 else 0.0 for t in self.iou_thresholds
            ]
            map_50_95 = float(np.mean(ap_list)) if ap_list else 0.0

            per_class_stats[cls] = {
                "total_gt": total_gt,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "ap_50": round(ap_50, 4),
                "map_50_95": round(map_50_95, 4),
                "mean_iou": round(float(np.mean(ious_list)) if ious_list else 0.0, 4),
            }

        valid_classes = [c for c, stats in per_class_stats.items() if stats["total_gt"] > 0]
        mean_ap_50 = (
            float(np.mean([per_class_stats[c]["ap_50"] for c in valid_classes]))
            if valid_classes
            else 0.0
        )
        mean_map_50_95 = (
            float(np.mean([per_class_stats[c]["map_50_95"] for c in valid_classes]))
            if valid_classes
            else 0.0
        )

        loc_p50 = (
            float(np.percentile(localization_errors_3d_cm, 50))
            if localization_errors_3d_cm
            else 0.0
        )
        loc_p95 = (
            float(np.percentile(localization_errors_3d_cm, 95))
            if localization_errors_3d_cm
            else 0.0
        )
        loc_max = float(np.max(localization_errors_3d_cm)) if localization_errors_3d_cm else 0.0

        summary = {
            "mAP_0.5": round(mean_ap_50, 4),
            "mAP_0.5_0.95": round(mean_map_50_95, 4),
            "localization_3d_error_cm": {
                "p50_median": round(loc_p50, 2),
                "p95": round(loc_p95, 2),
                "max": round(loc_max, 2),
                "samples": len(localization_errors_3d_cm),
            },
            "per_class": per_class_stats,
            "evaluated_frames": len(self.records),
        }
        return summary
