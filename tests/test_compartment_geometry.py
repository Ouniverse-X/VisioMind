"""Unit tests for the 3D multi-compartment and physical divider geometry system."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from voltron.shared.compartment_geometry import (
    CompartmentSlot3D,
    MultiCompartmentBinGeometry,
    PhysicalDivider3D,
)


def test_multi_compartment_1x3_layout_generates_two_column_dividers() -> None:
    container_aabb = (
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.90, 0.30, 0.20], dtype=np.float64),  # X is long axis (0.90m)
    )
    geom = MultiCompartmentBinGeometry(
        container_aabb,
        grid_shape=(1, 3),
        divider_thickness_m=0.010,
        wall_thickness_m=0.010,
    )

    assert geom.total_cells == 3
    assert geom.column_axis == 0  # X axis
    assert geom.row_axis == 1     # Y axis

    dividers = geom.get_all_dividers()
    assert len(dividers) == 2  # 2 internal column dividers for 3 slots
    assert dividers[0].divider_id == "col_divider_1"
    assert dividers[1].divider_id == "col_divider_2"

    slots = geom.get_all_slots()
    assert len(slots) == 3

    # Check slot 1, 2, 3 ordering along X axis
    s1 = geom.get_slot(1)
    s2 = geom.get_slot(2)
    s3 = geom.get_slot(3)

    assert s1.inner_aabb_world[1][0] <= s2.inner_aabb_world[0][0] + 0.015
    assert s2.inner_aabb_world[1][0] <= s3.inner_aabb_world[0][0] + 0.015

    # Check that slot 3's preplace entry is vertically elevated above top rim (z >= 0.20)
    assert s3.preplace_entry_pose_world[2] >= 0.25


def test_multi_compartment_2x2_layout_generates_row_and_column_dividers() -> None:
    container_aabb = (
        np.array([0.0, 0.0, 0.0]),
        np.array([0.60, 0.60, 0.25]),
    )
    geom = MultiCompartmentBinGeometry(
        container_aabb,
        grid_shape=(2, 2),
        divider_thickness_m=0.008,
    )

    assert geom.total_cells == 4
    dividers = geom.get_all_dividers()
    assert len(dividers) == 2  # 1 col divider + 1 row divider
    div_types = {d.divider_type for d in dividers}
    assert "column_divider" in div_types
    assert "row_divider" in div_types


def test_containment_and_divider_collision_checking() -> None:
    container_aabb = (
        np.array([0.0, 0.0, 0.0]),
        np.array([0.90, 0.30, 0.20]),
    )
    geom = MultiCompartmentBinGeometry(
        container_aabb,
        grid_shape=(1, 3),
        divider_thickness_m=0.010,
        wall_thickness_m=0.010,
    )

    # Object properly inside slot 3
    s3 = geom.get_slot(3)
    s3_min = np.array(s3.inner_aabb_world[0])
    s3_max = np.array(s3.inner_aabb_world[1])

    valid_obj_aabb = (
        s3_min + np.array([0.02, 0.02, 0.01]),
        s3_max - np.array([0.02, 0.02, 0.05]),
    )
    passed, audit = geom.check_containment(valid_obj_aabb, cell_index=3)
    assert passed is True
    assert audit["divider_collision_free"] is True
    assert audit["inside_slot_inner_bounds"] is True

    # Object straddling divider between slot 2 and slot 3
    div2 = geom.dividers[1]  # col_divider_2
    div2_center = np.array(div2.center_world)
    straddling_obj_aabb = (
        div2_center - np.array([0.04, 0.04, 0.02]),
        div2_center + np.array([0.04, 0.04, 0.02]),
    )
    passed, audit = geom.check_containment(straddling_obj_aabb, cell_index=3)
    assert passed is False
    assert "col_divider_2" in audit["intersected_dividers"]


def test_audit_export_is_serializable() -> None:
    container_aabb = (
        np.array([0.0, 0.0, 0.0]),
        np.array([0.90, 0.30, 0.20]),
    )
    geom = MultiCompartmentBinGeometry(container_aabb, grid_shape=(1, 3))
    audit = geom.export_audit()

    assert audit["total_cells"] == 3
    assert audit["divider_count"] == 2
    assert "1" in audit["slots"]
    assert "2" in audit["slots"]
    assert "3" in audit["slots"]
