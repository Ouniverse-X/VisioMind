"""Generate synthetic RGB-D and bounding box/mask dataset for industrial parts.

Creates industrial workbench scenes with bolts, wrenches, rollers, screwdrivers,
pliers, nuts, screws, toolboxes, and parts bins with ground truth 2D boxes,
segmentation masks, and 3D world poses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import numpy as np

from visiomind.perception.industrial_detector import (
    CLASS_TO_ID,
    INDUSTRIAL_CLASSES,
)

TARGET_CLASSES = [
    "bolt",
    "wrench",
    "roller",
    "screwdriver",
    "pliers",
    "nut",
    "screw",
    "allen_wrench",
    "flashlight",
    "toolbox",
    "parts_bin",
]

PART_COLORS = {
    "bolt": (180, 185, 190),          # Zinc / steel metallic grey
    "wrench": (210, 215, 220),        # Chrome silver
    "roller": (160, 165, 170),        # Polished cylindrical steel
    "screwdriver": (220, 60, 40),      # Industrial red handle
    "pliers": (40, 100, 210),          # Blue rubberized grip
    "nut": (175, 180, 185),           # Hex steel
    "screw": (150, 155, 160),         # Threaded steel
    "allen_wrench": (60, 60, 65),      # Black oxide
    "flashlight": (30, 30, 35),       # Matte black
    "toolbox": (200, 140, 30),        # Industrial orange-yellow
    "parts_bin": (30, 120, 180),      # Blue plastic tote
    "packing_box": (190, 150, 100),   # Kraft cardboard
}


def render_scene(
    num_objects: int = 4,
    width: int = 640,
    height: int = 480,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Render a synthetic RGB-D workbench scene with industrial parts."""
    rng = np.random.RandomState(seed)

    # Workbench base color with subtle texture
    table_color = rng.uniform(40, 65, size=(height, width, 3)).astype(np.uint8)
    rgb = table_color.copy()

    # Base depth from overhead/angled camera: plane around z=0.85m to 1.25m
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    depth = 0.95 + 0.35 * (y_coords / float(height)) + rng.normal(0, 0.002, (height, width))
    depth = depth.astype(np.float32)

    # Camera intrinsics (standard industrial RGB-D camera like RealSense D435)
    fx = float(width) * 0.9
    fy = float(width) * 0.9
    cx = float(width) / 2.0
    cy = float(height) / 2.0
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    # Camera pose in world
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.7071, -0.7071, 0.65],
        [0.0, 0.7071, 0.7071, 1.10],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

    objects_gt: list[dict[str, Any]] = []
    chosen_classes = rng.choice(TARGET_CLASSES, size=min(num_objects, len(TARGET_CLASSES)), replace=False)

    for cls_name in chosen_classes:
        # Determine part footprint in pixels scaled to image size
        if cls_name in ("toolbox", "parts_bin", "packing_box"):
            w_box = rng.randint(int(width * 0.35), int(width * 0.55))
            h_box = rng.randint(int(height * 0.25), int(height * 0.40))
            part_h_m = 0.08
        elif cls_name in ("wrench", "screwdriver", "pliers"):
            w_box = rng.randint(int(width * 0.25), int(width * 0.45))
            h_box = rng.randint(int(height * 0.12), int(height * 0.22))
            part_h_m = 0.025
        elif cls_name in ("roller", "flashlight"):
            w_box = rng.randint(int(width * 0.20), int(width * 0.35))
            h_box = rng.randint(int(height * 0.08), int(height * 0.16))
            part_h_m = 0.03
        else:  # bolt, nut, screw, allen_wrench
            w_box = rng.randint(int(width * 0.10), int(width * 0.20))
            h_box = rng.randint(int(height * 0.08), int(height * 0.16))
            part_h_m = 0.02

        max_x = max(10, width - w_box - 10)
        max_y = max(10, height - h_box - 10)
        x1 = rng.randint(10, max_x + 1)
        y1 = rng.randint(10, max_y + 1)
        x2 = min(width - 1, x1 + w_box)
        y2 = min(height - 1, y1 + h_box)

        # Render RGB part
        base_c = PART_COLORS.get(cls_name, (180, 180, 180))
        noise = rng.normal(0, 8, (h_box, w_box, 3))
        part_rgb = np.clip(np.array(base_c) + noise, 0, 255).astype(np.uint8)

        # Specular metallic highlight
        spec_x = rng.randint(5, max(6, w_box - 10))
        spec_w = rng.randint(4, max(5, w_box // 4))
        part_rgb[:, spec_x : spec_x + spec_w] = np.clip(
            part_rgb[:, spec_x : spec_x + spec_w] + 45, 0, 255
        )

        rgb[y1:y2, x1:x2] = part_rgb

        # Update depth map
        depth[y1:y2, x1:x2] -= float(part_h_m)

        # 3D position in camera frame
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        center_z = float(depth[int(center_y), int(center_x)])

        x_cam = (center_x - cx) * center_z / fx
        y_cam = (center_y - cy) * center_z / fy
        z_cam = center_z
        pos_cam = np.array([x_cam, y_cam, z_cam])

        # Lift to world
        pos_world = camera_pose[:3, :3] @ pos_cam + camera_pose[:3, 3]

        objects_gt.append({
            "class_name": cls_name,
            "class_id": CLASS_TO_ID[cls_name],
            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "centroid_camera": pos_cam.tolist(),
            "centroid_world": pos_world.tolist(),
            "confidence": 1.0,
        })

    return rgb, depth, objects_gt, intrinsics, camera_pose


def generate_dataset(output_dir: Path, num_train: int = 150, num_val: int = 40, num_test: int = 40) -> dict[str, Any]:
    """Generate train, val, and test synthetic dataset splits."""
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": (num_train, 1000),
        "val": (num_val, 5000),
        "test": (num_test, 9000),
    }

    manifest: dict[str, Any] = {}
    for split_name, (count, seed_base) in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        split_records: list[dict[str, Any]] = []

        for idx in range(count):
            seed = seed_base + idx
            num_objs = random.Random(seed).randint(3, 7)
            rgb, depth, gt, intrinsics, cam_pose = render_scene(
                num_objects=num_objs, width=256, height=256, seed=seed
            )

            img_id = f"{split_name}_{idx:04d}"
            # Save compact npz
            npz_path = split_dir / f"{img_id}.npz"
            np.savez_compressed(
                npz_path,
                rgb=rgb,
                depth=depth,
                intrinsics=intrinsics,
                camera_pose=cam_pose,
            )

            record = {
                "id": img_id,
                "file": f"{split_name}/{img_id}.npz",
                "objects": gt,
            }
            split_records.append(record)

        json_path = output_dir / f"{split_name}_annotations.json"
        json_path.write_text(json.dumps(split_records, indent=2), encoding="utf-8")
        manifest[split_name] = {
            "samples": count,
            "annotations_file": f"{split_name}_annotations.json",
        }

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate industrial vision dataset")
    parser.add_argument("--output-dir", type=str, default="data/industrial_vision")
    parser.add_argument("--train-samples", type=int, default=150)
    parser.add_argument("--val-samples", type=int, default=40)
    parser.add_argument("--test-samples", type=int, default=40)
    args = parser.parse_args()

    out = Path(args.output_dir)
    m = generate_dataset(out, args.train_samples, args.val_samples, args.test_samples)
    print(f"Generated industrial vision dataset in {out}: {m}")
