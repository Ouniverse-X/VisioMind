from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import logging
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

INDUSTRIAL_CLASSES: tuple[str, ...] = (
    "background",
    "bolt",
    "wrench",
    "roller",
    "screwdriver",
    "pliers",
    "nut",
    "screw",
    "allen_wrench",
    "drill",
    "flashlight",
    "parts_bin",
    "toolbox",
    "packing_box",
)

CLASS_TO_ID: dict[str, int] = {cls: idx for idx, cls in enumerate(INDUSTRIAL_CLASSES)}
ID_TO_CLASS: dict[int, str] = {idx: cls for idx, cls in enumerate(INDUSTRIAL_CLASSES)}


@dataclass
class IndustrialDetection:
    class_name: str
    class_id: int
    confidence: float
    bbox_xyxy: list[float]
    mask: np.ndarray | None = None

    centroid_camera: list[float] | None = None
    centroid_world: list[float] | None = None
    aabb_world: list[list[float]] | None = None
    obb_extents: list[float] | None = None
    obb_rotation_matrix: list[list[float]] | None = None
    point_count_3d: int = 0
    occlusion_rate: float = 0.0
    graspable: bool = True

    def to_dict(self) -> dict[str, Any]:
        result = {
            "class_name": self.class_name,
            "class_id": self.class_id,
            "confidence": float(self.confidence),
            "bbox_xyxy": [float(v) for v in self.bbox_xyxy],
            "has_mask": self.mask is not None,
            "centroid_camera": self.centroid_camera,
            "centroid_world": self.centroid_world,
            "aabb_world": self.aabb_world,
            "obb_extents": self.obb_extents,
            "obb_rotation_matrix": self.obb_rotation_matrix,
            "point_count_3d": self.point_count_3d,
            "occlusion_rate": float(self.occlusion_rate),
            "graspable": self.graspable,
        }
        return result


@dataclass
class IndustrialPerceptionResult:
    detections: list[IndustrialDetection] = field(default_factory=list)
    timestamp: float = 0.0
    latency_ms: float = 0.0
    model_version: str = "industrial_part_detector_v1.0"
    image_shape: tuple[int, int] = (720, 1280)
    device: str = "cpu"

    def get_by_class(self, class_name: str) -> list[IndustrialDetection]:
        target = class_name.strip().lower()
        return [d for d in self.detections if d.class_name.lower() == target]

    def get_highest_confidence(self, class_name: str | None = None) -> IndustrialDetection | None:
        candidates = self.detections
        if class_name:
            candidates = self.get_by_class(class_name)
        if not candidates:
            return None
        return max(candidates, key=lambda d: d.confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_detections": len(self.detections),
            "detections": [d.to_dict() for d in self.detections],
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
            "model_version": self.model_version,
            "image_shape": list(self.image_shape),
            "device": self.device,
        }


class IndustrialDetectorNetwork(nn.Module):
    def __init__(self, num_classes: int = len(INDUSTRIAL_CLASSES), in_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
        )

        self.head_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
        )
        self.cls_head = nn.Conv2d(128, num_classes, kernel_size=1)
        self.box_head = nn.Conv2d(128, 4, kernel_size=1)

        self.mask_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.Conv2d(16, num_classes, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat0 = self.stem(x)
        feat1 = self.layer1(feat0)
        feat2 = self.layer2(feat1)

        head_f = self.head_conv(feat2)
        cls_logits = self.cls_head(head_f)
        box_preds = self.box_head(head_f)
        mask_logits = self.mask_decoder(feat2)

        return cls_logits, box_preds, mask_logits


def nms_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
) -> list[int]:
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h

        union = areas[i] + areas[order[1:]] - intersection
        iou = np.zeros_like(intersection)
        valid = union > 0
        iou[valid] = intersection[valid] / union[valid]

        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return keep


class IndustrialPartDetector:
    def __init__(
        self,
        weights_path: str | Path | None = None,
        device: str | torch.device | None = None,
        conf_threshold: float = 0.40,
        iou_threshold: float = 0.45,
        version: str = "industrial_part_detector_v1.0",
    ):
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.version = str(version)
        self.classes = INDUSTRIAL_CLASSES
        self.num_classes = len(self.classes)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = IndustrialDetectorNetwork(num_classes=self.num_classes)
        self.model.to(self.device)
        self.model.eval()

        self.weights_path = Path(weights_path) if weights_path else None
        self.is_trained = False

        if self.weights_path and self.weights_path.exists():
            self.load_weights(self.weights_path)

    def load_weights(self, path: Path | str) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Industrial detector weights not found at: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
            self.version = checkpoint.get("version", self.version)
        elif isinstance(checkpoint, dict) and "model_state" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state"])
        elif isinstance(checkpoint, dict):
            self.model.load_state_dict(checkpoint)
        else:
            raise ValueError(f"Invalid checkpoint format in {path}")

        self.is_trained = True
        logger.info(
            "Loaded fine-tuned industrial vision detector from %s (version: %s)", path, self.version
        )

    def save_weights(self, path: Path | str, extra_metadata: dict[str, Any] | None = None) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "version": self.version,
            "num_classes": self.num_classes,
            "classes": list(self.classes),
            "state_dict": self.model.state_dict(),
            "timestamp": time.time(),
        }
        if extra_metadata:
            payload.update(extra_metadata)

        torch.save(payload, path)
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        logger.info("Saved industrial detector weights to %s (sha256: %s)", path, sha256)
        return sha256

    def preprocess(
        self, rgb_image: np.ndarray, target_size: tuple[int, int] = (256, 256)
    ) -> tuple[torch.Tensor, float, float]:
        img = np.asarray(rgb_image, dtype=np.float32)
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        h, w = img.shape[:2]
        target_h, target_w = target_size
        scale_y = h / target_h
        scale_x = w / target_w

        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        resized = F.interpolate(tensor, size=target_size, mode="bilinear", align_corners=False)
        normalized = resized / 255.0
        return normalized.to(self.device), scale_x, scale_y

    def detect(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray | None = None,
        camera_intrinsics: np.ndarray | None = None,
        camera_pose_world: np.ndarray | None = None,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
        target_classes: Sequence[str] | None = None,
    ) -> IndustrialPerceptionResult:
        start_time = time.time()
        conf_thresh = self.conf_threshold if conf_threshold is None else float(conf_threshold)
        iou_thresh = self.iou_threshold if iou_threshold is None else float(iou_threshold)

        h_orig, w_orig = rgb_image.shape[:2]
        tensor, scale_x, scale_y = self.preprocess(rgb_image, target_size=(256, 256))

        with torch.no_grad():
            cls_logits, box_preds, mask_logits = self.model(tensor)
            cls_probs = torch.softmax(cls_logits, dim=1)
            mask_probs = torch.sigmoid(mask_logits)

        mask_probs_orig = (
            F.interpolate(mask_probs, size=(h_orig, w_orig), mode="bilinear", align_corners=False)
            .squeeze(0)
            .cpu()
            .numpy()
        )

        cls_probs_np = cls_probs.squeeze(0).cpu().numpy()
        box_preds_np = box_preds.squeeze(0).cpu().numpy()
        _, h_feat, w_feat = cls_probs_np.shape

        raw_detections: list[IndustrialDetection] = []
        stride_x = 256.0 / w_feat
        stride_y = 256.0 / h_feat

        for c in range(1, self.num_classes):
            c_name = ID_TO_CLASS[c]
            if target_classes and c_name not in target_classes:
                continue

            class_prob_map = cls_probs_np[c]
            cy_idx, cx_idx = np.where(class_prob_map >= conf_thresh)

            for py, px in zip(cy_idx, cx_idx):
                conf = float(class_prob_map[py, px])
                dx, dy, dw, dh = box_preds_np[:, py, px]

                cx_256 = (px + 0.5 + float(dx)) * stride_x
                cy_256 = (py + 0.5 + float(dy)) * stride_y
                box_w_256 = np.exp(float(dw)) * 32.0
                box_h_256 = np.exp(float(dh)) * 32.0

                x1 = (cx_256 - box_w_256 / 2.0) * scale_x
                y1 = (cy_256 - box_h_256 / 2.0) * scale_y
                x2 = (cx_256 + box_w_256 / 2.0) * scale_x
                y2 = (cy_256 + box_h_256 / 2.0) * scale_y

                x1 = max(0.0, min(float(w_orig - 1), x1))
                y1 = max(0.0, min(float(h_orig - 1), y1))
                x2 = max(x1 + 2.0, min(float(w_orig), x2))
                y2 = max(y1 + 2.0, min(float(h_orig), y2))

                class_mask_full = mask_probs_orig[c] >= 0.5
                instance_mask = np.zeros((h_orig, w_orig), dtype=bool)
                ix1, iy1, ix2, iy2 = int(x1), int(y1), int(np.ceil(x2)), int(np.ceil(y2))
                instance_mask[iy1:iy2, ix1:ix2] = class_mask_full[iy1:iy2, ix1:ix2]

                if not np.any(instance_mask):
                    instance_mask[iy1:iy2, ix1:ix2] = True

                detection = IndustrialDetection(
                    class_name=c_name,
                    class_id=c,
                    confidence=conf,
                    bbox_xyxy=[x1, y1, x2, y2],
                    mask=instance_mask,
                )
                raw_detections.append(detection)

        filtered_detections: list[IndustrialDetection] = []
        for c in range(1, self.num_classes):
            c_dets = [d for d in raw_detections if d.class_id == c]
            if not c_dets:
                continue
            boxes = np.array([d.bbox_xyxy for d in c_dets], dtype=np.float32)
            scores = np.array([d.confidence for d in c_dets], dtype=np.float32)
            keep_indices = nms_boxes(boxes, scores, iou_threshold=iou_thresh)
            for idx in keep_indices:
                filtered_detections.append(c_dets[idx])

        if depth_image is not None and camera_intrinsics is not None:
            self._lift_detections_to_3d(
                filtered_detections,
                depth_image=depth_image,
                camera_intrinsics=camera_intrinsics,
                camera_pose_world=camera_pose_world,
            )

        latency_ms = (time.time() - start_time) * 1000.0
        return IndustrialPerceptionResult(
            detections=filtered_detections,
            timestamp=time.time(),
            latency_ms=latency_ms,
            model_version=self.version,
            image_shape=(h_orig, w_orig),
            device=str(self.device),
        )

    def _lift_detections_to_3d(
        self,
        detections: list[IndustrialDetection],
        depth_image: np.ndarray,
        camera_intrinsics: np.ndarray,
        camera_pose_world: np.ndarray | None = None,
    ) -> None:
        depth = np.asarray(depth_image, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[:, :, 0]

        fx = float(camera_intrinsics[0, 0])
        fy = float(camera_intrinsics[1, 1])
        cx = float(camera_intrinsics[0, 2])
        cy = float(camera_intrinsics[1, 2])

        h, w = depth.shape
        u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))

        for det in detections:
            if det.mask is None:
                continue
            mask = det.mask & (depth > 0.05) & (depth < 3.0) & np.isfinite(depth)
            v_pts, u_pts = np.where(mask)
            if len(u_pts) < 5:
                continue

            z_pts = depth[v_pts, u_pts]
            x_pts = (u_pts - cx) * z_pts / fx
            y_pts = (v_pts - cy) * z_pts / fy

            points_camera = np.column_stack([x_pts, y_pts, z_pts])

            med_z = np.median(z_pts)
            valid_z = np.abs(z_pts - med_z) < 0.15
            if np.sum(valid_z) >= 5:
                points_camera = points_camera[valid_z]

            det.point_count_3d = int(len(points_camera))
            centroid_cam = np.mean(points_camera, axis=0)
            det.centroid_camera = centroid_cam.tolist()

            if camera_pose_world is not None:
                t_cam_world = np.asarray(camera_pose_world, dtype=np.float64)
                rot_cam_world = t_cam_world[:3, :3]
                trans_cam_world = t_cam_world[:3, 3]

                points_world = points_camera @ rot_cam_world.T + trans_cam_world
                centroid_world = np.mean(points_world, axis=0)
                det.centroid_world = centroid_world.tolist()

                min_world = np.min(points_world, axis=0)
                max_world = np.max(points_world, axis=0)
                det.aabb_world = [min_world.tolist(), max_world.tolist()]

                if len(points_world) >= 6:
                    centered = points_world - centroid_world
                    cov = centered.T @ centered / max(1, len(centered) - 1)
                    eigenvals, eigenvecs = np.linalg.eigh(cov)

                    sort_idx = np.argsort(eigenvals)[::-1]
                    rot_mat = eigenvecs[:, sort_idx]
                    rotated_pts = centered @ rot_mat
                    extents = np.ptp(rotated_pts, axis=0)

                    det.obb_extents = extents.tolist()
                    det.obb_rotation_matrix = rot_mat.tolist()


class MockIndustrialDetector(IndustrialPartDetector):
    def __init__(
        self,
        ground_truth_objects: list[dict[str, Any]] | None = None,
        conf_threshold: float = 0.50,
    ):
        super().__init__(conf_threshold=conf_threshold)
        self.ground_truth_objects = ground_truth_objects or []
        self.is_trained = True

    def set_scene_objects(self, objects: list[dict[str, Any]]) -> None:
        self.ground_truth_objects = objects

    def detect(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray | None = None,
        camera_intrinsics: np.ndarray | None = None,
        camera_pose_world: np.ndarray | None = None,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
        target_classes: Sequence[str] | None = None,
    ) -> IndustrialPerceptionResult:
        start_time = time.time()
        h, w = rgb_image.shape[:2]
        detections: list[IndustrialDetection] = []

        for obj in self.ground_truth_objects:
            name = str(obj.get("name", "")).lower()
            class_name = None
            for cls in INDUSTRIAL_CLASSES[1:]:
                if cls in name or name.startswith(cls):
                    class_name = cls
                    break
            if class_name is None:
                continue
            if target_classes and class_name not in target_classes:
                continue

            bbox = obj.get("bbox_xyxy", [w * 0.3, h * 0.3, w * 0.7, h * 0.7])
            conf = float(obj.get("confidence", 0.96))
            x1, y1, x2, y2 = [int(v) for v in bbox]
            mask = np.zeros((h, w), dtype=bool)
            mask[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)] = True

            det = IndustrialDetection(
                class_name=class_name,
                class_id=CLASS_TO_ID.get(class_name, 1),
                confidence=conf,
                bbox_xyxy=[float(v) for v in bbox],
                mask=mask,
                centroid_world=obj.get("position"),
                aabb_world=obj.get("aabb"),
            )
            detections.append(det)

        latency_ms = (time.time() - start_time) * 1000.0
        return IndustrialPerceptionResult(
            detections=detections,
            timestamp=time.time(),
            latency_ms=latency_ms,
            model_version="mock_oracle_v1.0",
            image_shape=(h, w),
            device="cpu",
        )
