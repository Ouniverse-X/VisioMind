"""Train / Fine-tune the Industrial Parts Vision Detection & Segmentation Model.

Trains on industrial workbench synthetic & benchmark datasets, evaluates mAP@0.5,
mAP@0.5:0.95 and 3D localization accuracy, saves model weights, updates manifest.json,
and generates reports/industrial_vision_metrics.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from visiomind.perception.evaluator import DetectionMetricsEvaluator
from visiomind.perception.industrial_detector import (
    CLASS_TO_ID,
    ID_TO_CLASS,
    INDUSTRIAL_CLASSES,
    IndustrialDetection,
    IndustrialDetectorNetwork,
    IndustrialPartDetector,
    nms_boxes,
)
from training.generate_industrial_vision_dataset import generate_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_targets(
    gt_objects: list[dict[str, Any]],
    feat_size: tuple[int, int] = (16, 16),
    img_size: tuple[int, int] = (256, 256),
    num_classes: int = len(INDUSTRIAL_CLASSES),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert ground truth boxes and classes to grid target tensors."""
    h_feat, w_feat = feat_size
    h_img, w_img = img_size

    cls_target = torch.zeros((h_feat, w_feat), dtype=torch.long)
    box_target = torch.zeros((4, h_feat, w_feat), dtype=torch.float32)
    box_mask = torch.zeros((h_feat, w_feat), dtype=torch.bool)
    seg_target = torch.zeros((num_classes, h_img, w_img), dtype=torch.float32)

    stride_x = w_img / w_feat
    stride_y = h_img / h_feat

    for obj in gt_objects:
        cls_id = int(obj["class_id"])
        x1, y1, x2, y2 = obj["bbox_xyxy"]

        # 2D Segmentation target
        ix1 = int(max(0, min(w_img - 1, x1)))
        iy1 = int(max(0, min(h_img - 1, y1)))
        ix2 = int(max(ix1 + 1, min(w_img, x2)))
        iy2 = int(max(iy1 + 1, min(h_img, y2)))
        seg_target[cls_id, iy1:iy2, ix1:ix2] = 1.0

        # Grid cell target
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = max(2.0, x2 - x1)
        bh = max(2.0, y2 - y1)

        grid_x = int(cx / stride_x)
        grid_y = int(cy / stride_y)

        if 0 <= grid_x < w_feat and 0 <= grid_y < h_feat:
            cls_target[grid_y, grid_x] = cls_id
            dx = (cx / stride_x) - (grid_x + 0.5)
            dy = (cy / stride_y) - (grid_y + 0.5)
            dw = np.log(max(1e-4, bw / 32.0))
            dh = np.log(max(1e-4, bh / 32.0))

            box_target[:, grid_y, grid_x] = torch.tensor([dx, dy, dw, dh], dtype=torch.float32)
            box_mask[grid_y, grid_x] = True

    return cls_target, box_target, box_mask, seg_target


def train_epoch(
    model: nn.Module,
    samples: list[dict[str, Any]],
    data_dir: Path,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    cls_criterion = nn.CrossEntropyLoss()
    box_criterion = nn.SmoothL1Loss()
    seg_criterion = nn.BCEWithLogitsLoss()

    for sample in samples:
        npz_file = data_dir / sample["file"]
        data = np.load(npz_file)
        rgb = data["rgb"]  # (256, 256, 3)

        img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

        cls_target, box_target, box_mask, seg_target = build_targets(
            sample["objects"], feat_size=(16, 16), img_size=(256, 256)
        )
        cls_target = cls_target.unsqueeze(0).to(device)
        box_target = box_target.unsqueeze(0).to(device)
        box_mask = box_mask.unsqueeze(0).to(device)
        seg_target = seg_target.unsqueeze(0).to(device)

        optimizer.zero_grad()
        cls_logits, box_preds, mask_logits = model(img_tensor)

        # Classification loss
        l_cls = cls_criterion(cls_logits, cls_target)

        # Box regression loss (only on active cells)
        if box_mask.sum() > 0:
            active_preds = box_preds.permute(0, 2, 3, 1)[box_mask]
            active_targets = box_target.permute(0, 2, 3, 1)[box_mask]
            l_box = box_criterion(active_preds, active_targets)
        else:
            l_box = torch.tensor(0.0, device=device)

        # Mask segmentation loss
        l_seg = seg_criterion(mask_logits, seg_target)

        loss = l_cls + 2.0 * l_box + 1.5 * l_seg
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(1, len(samples))


def evaluate_dataset(
    detector: IndustrialPartDetector,
    samples: list[dict[str, Any]],
    data_dir: Path,
) -> dict[str, Any]:
    evaluator = DetectionMetricsEvaluator()
    for sample in samples:
        npz_file = data_dir / sample["file"]
        data = np.load(npz_file)
        rgb = data["rgb"]
        depth = data["depth"]
        intrinsics = data["intrinsics"]
        cam_pose = data["camera_pose"]

        res = detector.detect(
            rgb_image=rgb,
            depth_image=depth,
            camera_intrinsics=intrinsics,
            camera_pose_world=cam_pose,
            conf_threshold=0.35,
        )
        evaluator.add_frame_evaluation(
            predicted=res,
            ground_truth=sample["objects"],
            image_id=sample["id"],
        )

    return evaluator.compute_metrics()


def run_training(
    data_dir: Path,
    output_model_path: Path,
    epochs: int = 15,
    lr: float = 1e-3,
) -> dict[str, Any]:
    """Train industrial part detection model and return evaluation metrics."""
    data_dir = Path(data_dir)
    train_anno = data_dir / "train_annotations.json"
    val_anno = data_dir / "val_annotations.json"
    test_anno = data_dir / "test_annotations.json"

    if not train_anno.exists() or not val_anno.exists():
        logger.info("Dataset annotations not found, synthesizing dataset now...")
        generate_dataset(data_dir, num_train=120, num_val=30, num_test=30)

    train_samples = json.loads(train_anno.read_text(encoding="utf-8"))
    val_samples = json.loads(val_anno.read_text(encoding="utf-8"))
    test_samples = json.loads(test_anno.read_text(encoding="utf-8")) if test_anno.exists() else val_samples

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s for industrial vision training", device)

    detector = IndustrialPartDetector(device=device, conf_threshold=0.35)
    model = detector.model
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    logger.info("Beginning training for %d epochs...", epochs)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, train_samples, data_dir, optimizer, device)
        scheduler.step()
        duration = time.time() - t0
        if ep % 3 == 0 or ep == epochs:
            logger.info("Epoch %d/%d | Loss: %.4f | Time: %.2fs", ep, epochs, loss, duration)

    # Evaluate on test split
    logger.info("Evaluating fine-tuned detector on test dataset (%d scenes)...", len(test_samples))
    metrics = evaluate_dataset(detector, test_samples, data_dir)
    logger.info("Test Evaluation Metrics: mAP@0.5 = %.4f, mAP@0.5:0.95 = %.4f, 3D P50 Error = %.2f cm",
                metrics["mAP_0.5"], metrics["mAP_0.5_0.95"], metrics["localization_3d_error_cm"]["p50_median"])

    # Save weights
    sha256 = detector.save_weights(
        output_model_path,
        extra_metadata={
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(test_samples),
            "epochs": epochs,
            "metrics": metrics,
        },
    )

    return {
        "model_path": str(output_model_path),
        "sha256": sha256,
        "metrics": metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train industrial vision detector")
    parser.add_argument("--data-dir", type=str, default="data/industrial_vision")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="models/industrial_part_detector.pt")
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    out_model = ROOT / args.output
    result = run_training(data_dir, out_model, epochs=args.epochs, lr=args.lr)

    report_path = ROOT / "reports" / "industrial_vision_metrics.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result["metrics"], indent=2), encoding="utf-8")
    print(f"Industrial vision training complete. Report saved to {report_path}")
