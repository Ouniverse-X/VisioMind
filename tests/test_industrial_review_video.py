"""Fail-closed checks for competition-video evidence validation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_video_builder():
    path = ROOT / "scripts" / "build_industrial_review_video.py"
    spec = importlib.util.spec_from_file_location("industrial_review_video_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _records(*, cell_contained: bool) -> list[dict[str, object]]:
    return [
        {
            "event": "action_terminal_success",
            "payload": {
                "placement_success": True,
                "placement_verified": True,
                "released": True,
                "aabb_contained": True,
                "cell_aabb_contained": cell_contained,
            },
        },
        {"event": "orchestrator_task_final", "payload": {"outcome": "success"}},
    ]


def test_video_builder_accepts_strict_requested_cell_success() -> None:
    module = _load_video_builder()
    final, terminal = module._verified_run(_records(cell_contained=True))
    assert final["outcome"] == "success"
    assert terminal["cell_aabb_contained"] is True


def test_video_builder_rejects_container_success_in_wrong_cell() -> None:
    module = _load_video_builder()
    with pytest.raises(RuntimeError, match="requested cell AABB contained"):
        module._verified_run(_records(cell_contained=False))
