"""Render deterministic 3D presentation images for the VisioMind workcell.

The renderer mirrors the procedural geometry in ``visiomind.simulation`` and
``voltron.shared.compartment_geometry``.  It is intentionally dependency-light
so it can be run without Isaac Sim while preparing a competition poster/demo.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# Permit ``python scripts/render_industrial_workcell_3d.py`` from the project
# root without requiring an editable install.
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voltron.shared.compartment_geometry import MultiCompartmentBinGeometry


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "demo"


def box(ax, center, size, color, *, alpha=1.0, edge="#1a2633", linewidth=0.8):
    """Draw an axis-aligned cuboid using center and full dimensions."""
    c = np.asarray(center, dtype=float)
    s = np.asarray(size, dtype=float) / 2.0
    x0, x1 = c[0] - s[0], c[0] + s[0]
    y0, y1 = c[1] - s[1], c[1] + s[1]
    z0, z1 = c[2] - s[2], c[2] + s[2]
    vertices = [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
    ]
    poly = Poly3DCollection(vertices, facecolors=color, edgecolors=edge, linewidths=linewidth, alpha=alpha)
    ax.add_collection3d(poly)


def segment(ax, p0, p1, radius, color):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], color=color, linewidth=radius, solid_capstyle="round")


def cylinder(ax, center, radius, height, color, *, axis="z", alpha=1.0):
    """Draw a shaded cylinder for rollers, bolts, nuts and flashlight bodies."""
    cx, cy, cz = np.asarray(center, float)
    theta = np.linspace(0, 2 * np.pi, 32)
    t = np.linspace(-height / 2, height / 2, 8)
    theta, t = np.meshgrid(theta, t)
    if axis == "x":
        xs, ys, zs = cx + t, cy + radius * np.cos(theta), cz + radius * np.sin(theta)
    elif axis == "y":
        xs, ys, zs = cx + radius * np.cos(theta), cy + t, cz + radius * np.sin(theta)
    else:
        xs, ys, zs = cx + radius * np.cos(theta), cy + radius * np.sin(theta), cz + t
    ax.plot_surface(xs, ys, zs, color=color, alpha=alpha, linewidth=0, shade=True)


def ring_top(ax, center, outer_radius, inner_radius, color, *, z_offset=0.0, alpha=1.0):
    """Add an annular top face, useful for showing a real nut or roller bore."""
    cx, cy, cz = np.asarray(center, float)
    theta = np.linspace(0, 2 * np.pi, 48)
    radii = np.array([inner_radius, outer_radius], dtype=float)
    tt, rr = np.meshgrid(theta, radii)
    xs = cx + rr * np.cos(tt)
    ys = cy + rr * np.sin(tt)
    zs = np.full_like(xs, cz + z_offset)
    ax.plot_surface(xs, ys, zs, color=color, alpha=alpha, linewidth=0, shade=True)


def helical_thread(ax, center, radius, height, color, *, turns=7):
    """Draw a fine helical thread line around a vertical bolt shaft."""
    cx, cy, cz = np.asarray(center, float)
    theta = np.linspace(0, 2 * np.pi * turns, 220)
    z = cz - height / 2 + height * theta / (2 * np.pi * turns)
    ax.plot(cx + radius * np.cos(theta), cy + radius * np.sin(theta), z, color=color, linewidth=0.9, alpha=0.9)


def draw_part(ax, kind):
    """Draw a compact 3D proxy for one industrial part on a catalog tile."""
    if kind == "pliers":
        box(ax, (0.0, 0.0, 0.03), (0.34, 0.07, 0.06), "#b8423d", edge="#762c2b", linewidth=1.0)
        segment(ax, (-0.13, 0.0, 0.03), (-0.30, 0.10, 0.17), 5, "#b8423d")
        segment(ax, (0.13, 0.0, 0.03), (0.30, 0.10, 0.17), 5, "#b8423d")
        segment(ax, (-0.12, 0.0, 0.03), (-0.23, -0.08, -0.13), 4, "#b8423d")
        segment(ax, (0.12, 0.0, 0.03), (0.23, -0.08, -0.13), 4, "#b8423d")
        cylinder(ax, (0.0, 0.0, 0.03), 0.045, 0.025, "#e4b33c")
        ring_top(ax, (0.0, 0.0, 0.03), 0.045, 0.012, "#7c5e21", z_offset=0.014)
    elif kind == "screwdriver":
        cylinder(ax, (-0.08, 0.0, 0.04), 0.025, 0.35, "#aeb9c2", axis="x")
        cylinder(ax, (0.16, 0.0, 0.04), 0.075, 0.17, "#e86e34", axis="x")
        cylinder(ax, (0.08, 0.0, 0.04), 0.078, 0.018, "#f2c04b", axis="x")
        cylinder(ax, (0.265, 0.0, 0.04), 0.035, 0.018, "#596977", axis="x")
    elif kind == "wrench":
        segment(ax, (-0.25, 0.0, 0.03), (0.25, 0.0, 0.03), 7, "#aeb9c2")
        segment(ax, (0.25, 0.0, 0.03), (0.34, 0.09, 0.03), 5, "#aeb9c2")
        segment(ax, (0.25, 0.0, 0.03), (0.34, -0.09, 0.03), 5, "#aeb9c2")
        cylinder(ax, (-0.25, 0.0, 0.03), 0.07, 0.025, "#aeb9c2")
        ring_top(ax, (-0.25, 0.0, 0.03), 0.07, 0.035, "#53616b", z_offset=0.014)
    elif kind == "allen_wrench":
        segment(ax, (-0.23, -0.12, 0.03), (0.23, -0.12, 0.03), 5, "#626e79")
        segment(ax, (-0.23, -0.12, 0.03), (-0.23, 0.20, 0.03), 5, "#626e79")
    elif kind == "bolt":
        cylinder(ax, (0.0, 0.0, 0.13), 0.065, 0.09, "#83909b")
        cylinder(ax, (0.0, 0.0, -0.02), 0.032, 0.22, "#aab5bd")
        helical_thread(ax, (0.0, 0.0, -0.02), 0.036, 0.20, "#64717b", turns=6)
        ring_top(ax, (0.0, 0.0, 0.13), 0.065, 0.0, "#83909b", z_offset=0.046)
    elif kind == "nut":
        cylinder(ax, (0.0, 0.0, 0.04), 0.105, 0.09, "#c28b2e")
        ring_top(ax, (0.0, 0.0, 0.04), 0.105, 0.043, "#e0a638", z_offset=0.046)
        cylinder(ax, (0.0, 0.0, 0.045), 0.043, 0.095, "#f5f8fb", alpha=0.82)
    elif kind == "roller":
        cylinder(ax, (0.0, 0.0, 0.04), 0.10, 0.32, "#3d84b5", axis="x")
        cylinder(ax, (-0.17, 0.0, 0.04), 0.04, 0.02, "#9eabb5", axis="x")
        cylinder(ax, (0.17, 0.0, 0.04), 0.04, 0.02, "#9eabb5", axis="x")
        for x in (-0.12, -0.04, 0.04, 0.12):
            segment(ax, (x, -0.102, 0.04), (x, 0.102, 0.04), 1.2, "#8fc0d9")
    elif kind == "flashlight":
        cylinder(ax, (-0.03, 0.0, 0.04), 0.065, 0.28, "#3e566d", axis="x")
        cylinder(ax, (0.15, 0.0, 0.04), 0.075, 0.06, "#d1a52f", axis="x")


def render_parts_catalog(path: Path):
    """Render a presentation sheet of the supported industrial object set."""
    parts = [
        ("Pliers", "pliers"), ("Screwdriver", "screwdriver"),
        ("Wrench", "wrench"), ("Allen wrench", "allen_wrench"),
        ("Bolt", "bolt"), ("Nut", "nut"),
        ("Roller", "roller"), ("Flashlight", "flashlight"),
    ]
    fig = plt.figure(figsize=(16, 9), dpi=170)
    fig.patch.set_facecolor("#f5f8fb")
    fig.suptitle("VisioMind Industrial Object Set — 3D Proxies", fontsize=22, fontweight="bold", color="#152536", y=0.98)
    for index, (label, kind) in enumerate(parts, 1):
        ax = fig.add_subplot(2, 4, index, projection="3d")
        ax.set_xlim(-0.42, 0.42); ax.set_ylim(-0.30, 0.30); ax.set_zlim(-0.18, 0.25)
        ax.view_init(elev=22, azim=-58)
        ax.set_box_aspect((1.4, 1.0, 0.75))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_facecolor("#ffffff")
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_facecolor((0.97, 0.98, 0.99, 1.0))
        draw_part(ax, kind)
        ax.set_title(label, fontsize=13, fontweight="bold", color="#28465b", pad=8)
    fig.text(0.04, 0.035, "Object classes used by IndustrialPartDetector / scene grounding: pliers, screwdriver, wrench, allen wrench, bolt, nut, roller and flashlight", fontsize=11, color="#425466")
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def setup(ax, title):
    ax.set_title(title, fontsize=18, fontweight="bold", pad=18, color="#152536")
    ax.set_xlabel("X / m", labelpad=8)
    ax.set_ylabel("Y / m", labelpad=8)
    ax.set_zlabel("Z / m", labelpad=8)
    ax.set_xlim(-1.0, 2.1)
    ax.set_ylim(-1.75, 1.75)
    ax.set_zlim(0.0, 2.75)
    # Look from the operator side so the rear safety wall stays behind the
    # worktop instead of occluding the tools and toolbox.
    ax.view_init(elev=24, azim=122)
    ax.set_box_aspect((3.1, 3.5, 2.5))
    ax.grid(True, alpha=0.18)
    ax.set_facecolor("#f5f8fb")


def draw_workcell(ax):
    # Values are the robot-local dimensions from default_workcell_parts().
    box(ax, (0.70, 0.0, 0.006), (2.8, 3.0, 0.012), "#56636d", alpha=0.16)
    # Keep the rear panel translucent in the presentation render; the actual
    # USD asset remains opaque, but transparency prevents depth-sort occlusion
    # in Matplotlib and exposes the work area behind it.
    box(ax, (1.75, 0.0, 1.25), (0.04, 3.2, 2.5), "#374654", alpha=0.16)
    box(ax, (1.72, 0.0, 1.45), (0.025, 2.55, 0.75), "#123f76", alpha=0.22)
    box(ax, (1.68, 1.42, 1.25), (0.12, 0.12, 2.5), "#12619b")
    box(ax, (1.68, -1.42, 1.25), (0.12, 0.12, 2.5), "#12619b")
    box(ax, (1.68, 0.0, 2.43), (0.12, 2.95, 0.12), "#12619b")
    box(ax, (-0.62, 0.0, 0.018), (0.08, 3.0, 0.025), "#f3b51b")
    box(ax, (0.70, 1.46, 0.018), (2.7, 0.08, 0.025), "#f3b51b")
    box(ax, (0.70, -1.46, 0.018), (2.7, 0.08, 0.025), "#f3b51b")
    box(ax, (1.62, 0.0, 2.25), (0.05, 1.15, 0.08), "#4ad7ee", alpha=0.95)

    # Existing physical worktop and a compact R1 Pro visual proxy for context.
    box(ax, (0.20, 0.0, 0.83), (1.55, 1.55, 0.10), "#8696a6", alpha=0.42)
    box(ax, (-0.34, 0.0, 0.40), (0.52, 0.52, 0.75), "#263544")
    segment(ax, (-0.34, 0.0, 0.78), (-0.05, 0.0, 1.18), 7, "#4d83b4")
    segment(ax, (-0.05, 0.0, 1.18), (0.20, -0.10, 1.30), 6, "#4d83b4")
    segment(ax, (0.20, -0.10, 1.30), (0.35, -0.10, 1.10), 5, "#d5e1eb")

    # Mixed industrial tools (schematic geometry, placed on the worktop).
    box(ax, (0.05, 0.40, 1.02), (0.38, 0.08, 0.08), "#be4b45")  # pliers body
    segment(ax, (-0.14, 0.40, 1.02), (-0.32, 0.48, 1.15), 4, "#be4b45")
    segment(ax, (0.23, 0.40, 1.02), (0.42, 0.48, 1.15), 4, "#be4b45")
    segment(ax, (0.05, -0.34, 1.03), (0.52, -0.34, 1.03), 4, "#d4a528")
    segment(ax, (-0.18, -0.55, 1.04), (0.25, -0.55, 1.04), 4, "#8c9aa6")
    box(ax, (0.60, 0.25, 1.04), (0.10, 0.10, 0.10), "#5b6874")
    box(ax, (0.75, -0.05, 1.04), (0.10, 0.10, 0.10), "#5b6874")
    cylinder(ax, (0.34, 0.08, 1.04), 0.065, 0.24, "#3d84b5", axis="x")  # roller
    cylinder(ax, (0.56, 0.42, 1.10), 0.055, 0.08, "#8997a2")  # bolt head
    cylinder(ax, (0.56, 0.42, 1.02), 0.028, 0.16, "#aab5bd")  # bolt shaft
    cylinder(ax, (0.68, -0.40, 1.04), 0.065, 0.07, "#c28b2e")  # nut
    cylinder(ax, (-0.18, -0.20, 1.04), 0.05, 0.25, "#3e566d", axis="x")  # flashlight

    # Three-compartment toolbox near the workcell center.
    lower = np.array([0.63, -0.42, 0.88])
    upper = np.array([1.28, 0.18, 1.20])
    box(ax, (lower + upper) / 2, upper - lower, "#1f2e3b", alpha=0.26, edge="#a9bac8", linewidth=1.2)
    geom = MultiCompartmentBinGeometry((lower, upper), (1, 3), divider_thickness_m=0.008)
    for div in geom.get_all_dividers():
        box(ax, div.center_world, div.dimensions_world, "#aeb8c2", alpha=0.98, edge="#66737e")
    slot = geom.get_slot(3)
    smin, smax = np.asarray(slot.safe_placement_aabb_world[0]), np.asarray(slot.safe_placement_aabb_world[1])
    box(ax, (smin + smax) / 2, smax - smin, "#25bf64", alpha=0.28, edge="#18a34a", linewidth=1.5)
    # Highlighted third-cell rim.
    box(ax, ((smin + smax) / 2) + np.array([0, 0, 0.012]), (smax[0] - smin[0], 0.012, 0.024), "#26d96b", alpha=0.9)

    ax.text(1.50, 0.80, 2.42, "Industrial safety frame", fontsize=10, color="#12619b")
    ax.text(0.63, -0.50, 1.34, "3-compartment toolbox", fontsize=10, color="#1e475e")
    ax.text(0.96, -0.49, 1.25, "CELL 3", fontsize=10, fontweight="bold", color="#11833b")
    ax.text(-0.05, 0.46, 1.12, "target: pliers", fontsize=10, color="#9c302a")


def render_overview(path: Path):
    fig = plt.figure(figsize=(14, 10), dpi=150)
    fig.patch.set_facecolor("#f5f8fb")
    ax = fig.add_subplot(111, projection="3d")
    setup(ax, "VisioMind Industrial Workcell — Procedural 3D Model")
    draw_workcell(ax)
    fig.text(0.04, 0.035, "Added geometry: safety frame  |  mixed-tool workbench  |  3-cell toolbox  |  Cell 3 highlighted", fontsize=12, color="#425466")
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _draw_injected_shell(ax, *, exploded=False):
    """Draw only geometry dynamically injected by industrial_workcell.py."""
    offset = 0.22 if exploded else 0.0
    # Exact centers and dimensions from default_workcell_parts().
    box(ax, (0.70 - offset, 0.0, 0.006), (2.8, 3.0, 0.012), "#353d45", alpha=0.62, edge="#141b22")
    box(ax, (1.75 + offset, 0.0, 1.25), (0.04, 3.2, 2.5), "#334453", alpha=0.32, edge="#17232c")
    box(ax, (1.72 + offset * 1.15, 0.0, 1.45), (0.025, 2.55, 0.75), "#15508a", alpha=0.58, edge="#082b4c")
    box(ax, (1.68, 1.42 + offset, 1.25), (0.12, 0.12, 2.5), "#0c609e", edge="#063252")
    box(ax, (1.68, -1.42 - offset, 1.25), (0.12, 0.12, 2.5), "#0c609e", edge="#063252")
    box(ax, (1.68, 0.0, 2.43 + offset), (0.12, 2.95, 0.12), "#0c609e", edge="#063252")
    box(ax, (-0.62 - offset, 0.0, 0.018), (0.08, 3.0, 0.025), "#f6bd1e", edge="#855d00", linewidth=1.2)
    box(ax, (0.70, 1.46 + offset, 0.018), (2.7, 0.08, 0.025), "#f6bd1e", edge="#855d00", linewidth=1.2)
    box(ax, (0.70, -1.46 - offset, 0.018), (2.7, 0.08, 0.025), "#f6bd1e", edge="#855d00", linewidth=1.2)
    box(ax, (1.62, 0.0, 2.25), (0.05, 1.15, 0.08), "#42d7ef", alpha=0.96, edge="#0b6575")


def render_injected_workcell_components(path: Path):
    """Render an assembled/exploded technical plate of injected USD parts."""
    fig = plt.figure(figsize=(16, 9), dpi=170)
    fig.patch.set_facecolor("#f3f7fa")
    fig.suptitle(
        "Dynamically Injected Industrial Workcell Geometry",
        fontsize=23,
        fontweight="bold",
        color="#142638",
        y=0.975,
    )
    for index, (title, exploded) in enumerate((("Assembled USD shell", False), ("Exploded component view", True)), 1):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        _draw_injected_shell(ax, exploded=exploded)
        ax.set_title(title, fontsize=16, fontweight="bold", color="#23475f", pad=14)
        ax.set_xlim(-1.15, 2.25); ax.set_ylim(-1.90, 1.90); ax.set_zlim(0.0, 2.90)
        ax.view_init(elev=24, azim=124)
        ax.set_box_aspect((3.4, 3.8, 2.9))
        ax.set_xlabel("X / m"); ax.set_ylabel("Y / m"); ax.set_zlabel("Z / m")
        ax.grid(True, alpha=0.16)
        ax.set_facecolor("#f8fafc")
    left, right = fig.axes
    left.text(0.35, 1.32, 0.08, "industrial safety mat\n2.80 × 3.00 m", color="#27343e", fontweight="bold")
    left.text(1.78, 0.90, 1.82, "rear backdrop + blue panel", color="#164f7a", fontweight="bold")
    left.text(1.68, -1.63, 2.55, "uprights + top beam", color="#0b5b94", fontweight="bold")
    left.text(-0.72, -0.50, 0.12, "yellow safety boundary", color="#8f6500", fontweight="bold")
    left.text(1.60, 0.05, 2.36, "status light", color="#087488", fontweight="bold")
    right.text(0.10, 1.68, 0.07, "floor mat", color="#27343e", fontweight="bold")
    right.text(2.02, 0.75, 1.55, "backdrop", color="#164f7a", fontweight="bold")
    right.text(1.64, -1.79, 2.70, "steel frame", color="#0b5b94", fontweight="bold")
    right.text(-0.93, -0.45, 0.13, "safety perimeter", color="#8f6500", fontweight="bold")
    fig.text(
        0.04,
        0.035,
        "Injection anchor: robot-local frame  |  Created after environment reset  |  Visual USD geometry  |  Collision API: disabled",
        fontsize=11.5,
        color="#42586a",
    )
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def render_toolbox_detail(path: Path):
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([0.90, 0.60, 0.32])
    geom = MultiCompartmentBinGeometry((lower, upper), (1, 3), divider_thickness_m=0.008)
    fig = plt.figure(figsize=(14, 8), dpi=170)
    fig.patch.set_facecolor("#f5f8fb")
    ax = fig.add_subplot(121, projection="3d")
    ax.set_title("Parametric 3D compartment geometry", fontsize=16, fontweight="bold", pad=16)
    ax.set_xlim(-0.08, 0.98); ax.set_ylim(-0.08, 0.68); ax.set_zlim(0.0, 0.42)
    ax.set_xlabel("X / m"); ax.set_ylabel("Y / m"); ax.set_zlabel("Z / m")
    ax.view_init(elev=25, azim=-58); ax.set_box_aspect((1.5, 1.0, 0.7)); ax.grid(True, alpha=0.18)
    box(ax, (upper + lower) / 2, upper - lower, "#263746", alpha=0.18, edge="#7e93a4", linewidth=1.0)
    for div in geom.get_all_dividers():
        box(ax, div.center_world, div.dimensions_world, "#9caab5", edge="#52626e")
    for idx, slot in enumerate(geom.get_all_slots(), 1):
        smin, smax = np.asarray(slot.safe_placement_aabb_world[0]), np.asarray(slot.safe_placement_aabb_world[1])
        color = "#25bf64" if idx == 3 else "#6095c2"
        box(ax, (smin + smax) / 2, smax - smin, color, alpha=0.34, edge=color, linewidth=1.4)
        ax.text(*(np.asarray(slot.center_world) + [0, 0, 0.02]), f"Cell {idx}", color=color, fontweight="bold")
    ax.text(0.42, 0.22, 0.36, "divider plates", color="#455a67", fontsize=9)

    ax2 = fig.add_subplot(122)
    ax2.axis("off")
    rows = [
        ("Outer AABB", "0.90 × 0.60 × 0.32 m"),
        ("Grid", "1 row × 3 columns"),
        ("Divider thickness", "8 mm"),
        ("Wall / bottom", "10 mm / 8 mm"),
        ("Cell-3 clearance", "5 mm XY margin"),
        ("Release pose", "25 mm above cavity bottom"),
        ("Verification", "3D AABB + divider collision audit"),
    ]
    ax2.text(0.02, 0.96, "Cell-level safety model", fontsize=18, fontweight="bold", color="#152536")
    y = 0.85
    for key, value in rows:
        ax2.text(0.03, y, key, fontsize=12, color="#536878")
        ax2.text(0.55, y, value, fontsize=12, color="#152536", ha="left")
        ax2.plot([0.02, 0.98], [y - 0.025, y - 0.025], color="#d9e2e8", linewidth=0.8)
        y -= 0.105
    ax2.text(0.03, 0.10, "Green = safe placement volume\nGray plates = physical dividers\nUsed for planning, collision audit and post-release verification", fontsize=12, color="#1b6e3b", linespacing=1.6)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overview = args.output_dir / "industrial_workcell_3d_overview.png"
    detail = args.output_dir / "industrial_toolbox_cell3_geometry.png"
    catalog = args.output_dir / "industrial_parts_catalog_3d.png"
    injected = args.output_dir / "industrial_workcell_injected_components_3d.png"
    render_overview(overview)
    render_toolbox_detail(detail)
    render_parts_catalog(catalog)
    render_injected_workcell_components(injected)
    print(overview)
    print(detail)
    print(catalog)
    print(injected)


if __name__ == "__main__":
    main()
