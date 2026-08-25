"""Fine-grained 3D multi-compartment and physical divider geometry system.

Provides rigorous 3D spatial partitioning for multi-compartment storage bins,
toolboxes, and tote boxes:
  - Exact 3D physical divider plates (thickness, height, collision AABBs).
  - Usable inner slot volumes (factoring in outer wall & divider plate thicknesses).
  - Safe placement / insertion corridors and vertical pre-placement waypoints.
  - Strict divider-collision and slot-containment auditing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Sequence

import numpy as np


def _as_vector3(value: Any) -> np.ndarray:
    vec = np.asarray(value, dtype=np.float64).reshape(-1)
    if vec.size != 3 or not np.isfinite(vec).all():
        raise ValueError(f"expected 3 finite float values, got {value}")
    return vec


@dataclass(frozen=True)
class PhysicalDivider3D:
    """A physical 3D partition plate dividing two adjacent rows or columns."""

    divider_id: str
    divider_type: str  # "column_divider" (vertical plate) or "row_divider"
    split_axis: int    # 0 for X-axis split, 1 for Y-axis split
    split_index: int   # 1, 2, ...
    center_world: list[float]
    dimensions_world: list[float]  # [dx, dy, dz]
    aabb_world: list[list[float]]  # [[x_min, y_min, z_min], [x_max, y_max, z_max]]
    thickness_m: float
    height_m: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def intersects_aabb(self, query_aabb: tuple[np.ndarray, np.ndarray] | list[list[float]], tolerance_m: float = 0.0) -> bool:
        """Check whether query AABB intersects this divider plate within tolerance."""
        q_min = np.asarray(query_aabb[0], dtype=np.float64) - float(tolerance_m)
        q_max = np.asarray(query_aabb[1], dtype=np.float64) + float(tolerance_m)
        d_min = np.asarray(self.aabb_world[0], dtype=np.float64)
        d_max = np.asarray(self.aabb_world[1], dtype=np.float64)

        overlap_x = (q_min[0] <= d_max[0]) and (q_max[0] >= d_min[0])
        overlap_y = (q_min[1] <= d_max[1]) and (q_max[1] >= d_min[1])
        overlap_z = (q_min[2] <= d_max[2]) and (q_max[2] >= d_min[2])
        return bool(overlap_x and overlap_y and overlap_z)


@dataclass(frozen=True)
class CompartmentSlot3D:
    """A usable 3D compartment slot inside a container."""

    cell_index: int            # 1-based index (e.g. 1..3 for 1x3 bin)
    row_index: int             # 0-based
    column_index: int          # 0-based
    center_world: list[float]  # [x, y, z]
    inner_aabb_world: list[list[float]]  # [[x_min, y_min, z_min], [x_max, y_max, z_max]]
    safe_placement_aabb_world: list[list[float]]
    spans_world: list[float]   # [dx, dy, dz]
    preplace_entry_pose_world: list[float]  # Stand-off point above the compartment top rim
    release_pose_world: list[float]         # Drop / release position inside cavity
    bounding_divider_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def contains_aabb(self, query_aabb: tuple[np.ndarray, np.ndarray] | list[list[float]], strict_xy: bool = True) -> bool:
        """Check whether query AABB is fully contained inside the slot's inner AABB."""
        q_min = np.asarray(query_aabb[0], dtype=np.float64)
        q_max = np.asarray(query_aabb[1], dtype=np.float64)
        s_min = np.asarray(self.inner_aabb_world[0], dtype=np.float64)
        s_max = np.asarray(self.inner_aabb_world[1], dtype=np.float64)

        xy_contained = bool(
            (q_min[0] >= s_min[0] - 1e-4) and (q_max[0] <= s_max[0] + 1e-4) and
            (q_min[1] >= s_min[1] - 1e-4) and (q_max[1] <= s_max[1] + 1e-4)
        )
        if not strict_xy:
            return xy_contained
        z_contained = bool((q_min[2] >= s_min[2] - 0.05) and (q_min[2] <= s_max[2] + 0.10))
        return xy_contained and z_contained


class MultiCompartmentBinGeometry:
    """Computes fine-grained 3D partition plates and usable slots for multi-compartment containers."""

    def __init__(
        self,
        container_aabb: tuple[np.ndarray, np.ndarray] | list[list[float]],
        grid_shape: tuple[int, int] | list[int] = (1, 3),
        divider_thickness_m: float = 0.008,
        wall_thickness_m: float = 0.010,
        bottom_thickness_m: float = 0.008,
        divider_height_ratio: float = 0.95,
        cell_margin_m: float = 0.005,
    ):
        raw_min = _as_vector3(container_aabb[0])
        raw_max = _as_vector3(container_aabb[1])
        if np.any(raw_max <= raw_min):
            raise ValueError(f"invalid container AABB bounds: min={raw_min}, max={raw_max}")

        self.container_min = raw_min
        self.container_max = raw_max
        self.container_span = raw_max - raw_min

        shape = tuple(int(v) for v in grid_shape)
        if len(shape) != 2 or any(v < 1 for v in shape):
            raise ValueError(f"grid_shape must contain two positive integers, got {grid_shape}")
        self.rows, self.columns = shape
        self.total_cells = self.rows * self.columns

        self.divider_thickness_m = float(max(0.001, divider_thickness_m))
        self.wall_thickness_m = float(max(0.001, wall_thickness_m))
        self.bottom_thickness_m = float(max(0.001, bottom_thickness_m))
        self.divider_height_ratio = float(np.clip(divider_height_ratio, 0.5, 1.0))
        self.cell_margin_m = float(max(0.0, cell_margin_m))

        # Long axis convention: columns align with longer horizontal axis
        horiz = self.container_span[:2]
        self.column_axis = int(np.argmax(horiz))
        self.row_axis = 1 - self.column_axis

        # Compute inner container cavity
        self.cavity_min = self.container_min.copy()
        self.cavity_max = self.container_max.copy()
        self.cavity_min[:2] += self.wall_thickness_m
        self.cavity_max[:2] -= self.wall_thickness_m
        self.cavity_min[2] += self.bottom_thickness_m

        if np.any(self.cavity_max <= self.cavity_min):
            raise ValueError("outer wall / bottom thickness exceeds container dimensions")

        self.cavity_span = self.cavity_max - self.cavity_min

        # Compute physical dividers & slots
        self.dividers: list[PhysicalDivider3D] = []
        self.slots: dict[int, CompartmentSlot3D] = {}
        self._build_geometry()

    def _build_geometry(self) -> None:
        col_axis = self.column_axis
        row_axis = self.row_axis

        col_span_total = self.cavity_span[col_axis]
        row_span_total = self.cavity_span[row_axis]
        depth_total = self.cavity_span[2]
        divider_height = depth_total * self.divider_height_ratio
        divider_z_min = self.cavity_min[2]
        divider_z_max = divider_z_min + divider_height

        # Calculate slot pitch and usable width
        usable_col_width = (col_span_total - (self.columns - 1) * self.divider_thickness_m) / self.columns
        usable_row_width = (row_span_total - (self.rows - 1) * self.divider_thickness_m) / self.rows

        if usable_col_width <= 0.005 or usable_row_width <= 0.005:
            raise ValueError(f"divider thickness ({self.divider_thickness_m}m) leaves insufficient slot space")

        # 1. Generate column divider partition plates (between columns)
        for c in range(1, self.columns):
            pos_col = self.cavity_min[col_axis] + c * usable_col_width + (c - 0.5) * self.divider_thickness_m
            d_min = np.zeros(3, dtype=np.float64)
            d_max = np.zeros(3, dtype=np.float64)

            d_min[col_axis] = pos_col - self.divider_thickness_m / 2.0
            d_max[col_axis] = pos_col + self.divider_thickness_m / 2.0
            d_min[row_axis] = self.cavity_min[row_axis]
            d_max[row_axis] = self.cavity_max[row_axis]
            d_min[2] = divider_z_min
            d_max[2] = divider_z_max

            center = (d_min + d_max) / 2.0
            dims = d_max - d_min
            div = PhysicalDivider3D(
                divider_id=f"col_divider_{c}",
                divider_type="column_divider",
                split_axis=col_axis,
                split_index=c,
                center_world=center.tolist(),
                dimensions_world=dims.tolist(),
                aabb_world=[d_min.tolist(), d_max.tolist()],
                thickness_m=self.divider_thickness_m,
                height_m=divider_height,
            )
            self.dividers.append(div)

        # 2. Generate row divider partition plates (between rows)
        for r in range(1, self.rows):
            pos_row = self.cavity_min[row_axis] + r * usable_row_width + (r - 0.5) * self.divider_thickness_m
            d_min = np.zeros(3, dtype=np.float64)
            d_max = np.zeros(3, dtype=np.float64)

            d_min[row_axis] = pos_row - self.divider_thickness_m / 2.0
            d_max[row_axis] = pos_row + self.divider_thickness_m / 2.0
            d_min[col_axis] = self.cavity_min[col_axis]
            d_max[col_axis] = self.cavity_max[col_axis]
            d_min[2] = divider_z_min
            d_max[2] = divider_z_max

            center = (d_min + d_max) / 2.0
            dims = d_max - d_min
            div = PhysicalDivider3D(
                divider_id=f"row_divider_{r}",
                divider_type="row_divider",
                split_axis=row_axis,
                split_index=r,
                center_world=center.tolist(),
                dimensions_world=dims.tolist(),
                aabb_world=[d_min.tolist(), d_max.tolist()],
                thickness_m=self.divider_thickness_m,
                height_m=divider_height,
            )
            self.dividers.append(div)

        # 3. Generate individual compartment slots
        for idx in range(1, self.total_cells + 1):
            r = (idx - 1) // self.columns
            c = (idx - 1) % self.columns

            s_min = np.zeros(3, dtype=np.float64)
            s_max = np.zeros(3, dtype=np.float64)

            s_min[col_axis] = self.cavity_min[col_axis] + c * (usable_col_width + self.divider_thickness_m)
            s_max[col_axis] = s_min[col_axis] + usable_col_width
            s_min[row_axis] = self.cavity_min[row_axis] + r * (usable_row_width + self.divider_thickness_m)
            s_max[row_axis] = s_min[row_axis] + usable_row_width
            s_min[2] = self.cavity_min[2]
            s_max[2] = self.container_max[2]

            safe_min = s_min.copy()
            safe_max = s_max.copy()
            safe_min[:2] += self.cell_margin_m
            safe_max[:2] -= self.cell_margin_m

            center = (s_min + s_max) / 2.0
            spans = s_max - s_min

            # Standoff entry pose (0.08m above top rim) and release pose (0.02m above bottom)
            preplace_entry = center.copy()
            preplace_entry[2] = self.container_max[2] + 0.08

            release_pose = center.copy()
            release_pose[2] = s_min[2] + 0.025

            # Find adjacent divider IDs
            adj_divs: list[str] = []
            if c > 0:
                adj_divs.append(f"col_divider_{c}")
            if c < self.columns - 1:
                adj_divs.append(f"col_divider_{c+1}")
            if r > 0:
                adj_divs.append(f"row_divider_{r}")
            if r < self.rows - 1:
                adj_divs.append(f"row_divider_{r+1}")

            slot = CompartmentSlot3D(
                cell_index=idx,
                row_index=r,
                column_index=c,
                center_world=center.tolist(),
                inner_aabb_world=[s_min.tolist(), s_max.tolist()],
                safe_placement_aabb_world=[safe_min.tolist(), safe_max.tolist()],
                spans_world=spans.tolist(),
                preplace_entry_pose_world=preplace_entry.tolist(),
                release_pose_world=release_pose.tolist(),
                bounding_divider_ids=adj_divs,
            )
            self.slots[idx] = slot

    def get_slot(self, cell_index: int) -> CompartmentSlot3D:
        idx = int(cell_index)
        if idx not in self.slots:
            raise ValueError(f"cell_index must be in [1, {self.total_cells}], got {cell_index}")
        return self.slots[idx]

    def get_all_slots(self) -> list[CompartmentSlot3D]:
        return [self.slots[idx] for idx in range(1, self.total_cells + 1)]

    def get_all_dividers(self) -> list[PhysicalDivider3D]:
        return list(self.dividers)

    def get_divider_collision_aabbs(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(np.array(d.aabb_world[0]), np.array(d.aabb_world[1])) for d in self.dividers]

    def check_containment(
        self,
        object_aabb: tuple[np.ndarray, np.ndarray] | list[list[float]],
        cell_index: int,
        strict_divider_clearance: bool = True,
    ) -> tuple[bool, dict[str, Any]]:
        """Verify whether object is strictly inside the requested cell without intersecting dividers."""
        slot = self.get_slot(cell_index)
        inside_slot = slot.contains_aabb(object_aabb, strict_xy=True)

        intersected_dividers: list[str] = []
        for div in self.dividers:
            if div.intersects_aabb(object_aabb, tolerance_m=0.0):
                intersected_dividers.append(div.divider_id)

        passed = bool(inside_slot and (not strict_divider_clearance or len(intersected_dividers) == 0))

        audit = {
            "cell_index": int(cell_index),
            "inside_slot_inner_bounds": inside_slot,
            "intersected_dividers": intersected_dividers,
            "divider_collision_free": len(intersected_dividers) == 0,
            "containment_passed": passed,
            "slot_inner_aabb": slot.inner_aabb_world,
            "object_aabb": [
                np.asarray(object_aabb[0], dtype=float).tolist(),
                np.asarray(object_aabb[1], dtype=float).tolist(),
            ],
        }
        return passed, audit

    def export_audit(self) -> dict[str, Any]:
        """Export comprehensive geometry and divider audit for inspection and logging."""
        return {
            "grid_shape": [self.rows, self.columns],
            "total_cells": self.total_cells,
            "column_axis_world": "x" if self.column_axis == 0 else "y",
            "row_axis_world": "x" if self.row_axis == 0 else "y",
            "container_outer_aabb": [self.container_min.tolist(), self.container_max.tolist()],
            "container_cavity_aabb": [self.cavity_min.tolist(), self.cavity_max.tolist()],
            "divider_count": len(self.dividers),
            "dividers": [d.to_dict() for d in self.dividers],
            "slots": {str(k): v.to_dict() for k, v in self.slots.items()},
        }
