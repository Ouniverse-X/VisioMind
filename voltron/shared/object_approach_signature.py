"""Shared helpers for object-approach candidate identity."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any


DISTANCE_BUCKET_M = 0.25
HEADING_SECTORS = 16


def candidate_signature(candidate: dict[str, Any]) -> dict[str, Any]:
    """Build a stable identity for object-approach history lookup."""
    explicit = candidate.get("candidate_signature")
    if isinstance(explicit, dict) and explicit:
        return deepcopy(explicit)

    signature: dict[str, Any] = {}
    nav_node = candidate.get("nav_node")
    if nav_node is not None:
        signature["nav_node"] = deepcopy(nav_node)

    for key in ("floor_id", "room_id"):
        value = candidate.get(key)
        if value not in (None, ""):
            signature[key] = deepcopy(value)

    distance = _float_value(
        candidate.get("approach_boundary_distance_m"),
        candidate.get("approach_distance_m"),
    )
    if distance is not None:
        signature["distance_bucket_m"] = _bucket_float(distance, DISTANCE_BUCKET_M)

    heading = _float_value(candidate.get("desired_heading"))
    if heading is not None:
        signature["approach_heading_sector"] = _heading_sector(heading)

    if not signature:
        candidate_id = candidate.get("candidate_id")
        if candidate_id not in (None, ""):
            signature["candidate_id"] = deepcopy(candidate_id)
    return signature


def signature_values_match(current: dict[str, Any], stored: dict[str, Any]) -> bool:
    return bool(current and stored and current == stored)


def _bucket_float(value: float, bucket_size: float) -> float:
    return round(round(float(value) / bucket_size) * bucket_size, 2)


def _heading_sector(heading: float) -> int:
    normalized = (float(heading) + math.pi) % (2.0 * math.pi)
    return int(normalized / (2.0 * math.pi / float(HEADING_SECTORS)))


def _float_value(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str) and value.strip():
            try:
                return float(value)
            except ValueError:
                continue
    return None


__all__ = ["candidate_signature", "signature_values_match"]
