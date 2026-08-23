"""Synthetic benchmark helpers for navigation run summaries.

This module keeps the benchmark surface small and deterministic so the test
suite can validate report generation without depending on heavyweight runtime
evaluation code.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from random import Random
from typing import Any


def _group_run_name(run_name: str) -> str:
    lowered = run_name.lower()
    if "hovsg_nav" in lowered:
        return "hovsg_nav"
    if "nav" in lowered:
        return "task_nav"
    return "other"


def _discover_run_groups(run_root: str) -> dict[str, int]:
    root = Path(run_root)
    counts: Counter[str] = Counter()
    if not root.exists():
        return {"task_nav": 0, "hovsg_nav": 0}

    for path in root.iterdir():
        if path.is_dir():
            counts[_group_run_name(path.name)] += 1

    # Keep the report surface stable even if one group is absent locally.
    counts.setdefault("task_nav", 0)
    counts.setdefault("hovsg_nav", 0)
    return dict(counts)


def run_navigation_generalization_benchmark(
    run_root: str,
    random_seed: int = 7,
) -> dict[str, Any]:
    """Build a deterministic synthetic benchmark summary from the run folder."""

    rng = Random(random_seed)
    run_groups = _discover_run_groups(run_root)

    baseline_train = {"success_rate": 0.56, "avg_steps": 132}
    baseline_eval = {"success_rate": 0.44, "avg_steps": 167}
    robust_train = {
        "success_rate": round(max(baseline_train["success_rate"], 0.71 + rng.random() * 0.03), 3),
        "avg_steps": 116,
    }
    robust_eval = {
        "success_rate": round(max(baseline_eval["success_rate"], 0.62 + rng.random() * 0.03), 3),
        "avg_steps": 145,
    }

    return {
        "title": "Synthetic Benchmark",
        "run_root": run_root,
        "run_groups": run_groups,
        "controllers": {
            "baseline_rule": {"train": baseline_train, "eval": baseline_eval},
            "robust_tuned": {"train": robust_train, "eval": robust_eval},
        },
    }


def format_markdown_report(results: dict[str, Any]) -> str:
    """Render a compact markdown report for the synthetic benchmark output."""

    run_groups = results.get("run_groups", {})
    lines = [
        "# Synthetic Benchmark",
        "",
        "## Run Groups",
        f"- task_nav: {run_groups.get('task_nav', 0)}",
        f"- hovsg_nav: {run_groups.get('hovsg_nav', 0)}",
        "",
        "## Controllers",
    ]

    controllers = results.get("controllers", {})
    for controller_name, partitions in controllers.items():
        lines.append(f"### {controller_name}")
        for split_name, metrics in partitions.items():
            lines.append(
                f"- {split_name}: success_rate={metrics['success_rate']}, avg_steps={metrics['avg_steps']}"
            )
        lines.append("")

    return "\n".join(lines).strip()
