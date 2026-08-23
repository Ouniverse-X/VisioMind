#!/usr/bin/env python3
"""Extract compact, reproducible evidence from VisioMind Isaac run logs.

The large ``process_data.jsonl`` files remain runtime artifacts outside this
repository.  This tool hashes them and records only the decisive terminal
events, allowing reviewers to distinguish strict physical placement success
from an environment-level task flag that may already be true after grasping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def load_decisive_events(log_path: Path) -> dict[str, Any]:
    run_start = None
    run_end = None
    terminal = None
    failures: list[dict[str, Any]] = []
    forbidden_events: list[str] = []
    with log_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {log_path}:{line_number}") from exc
            event = record.get("event")
            payload = record.get("payload") or {}
            if event == "run_start":
                run_start = record
            elif event == "run_end":
                run_end = record
            elif event == "action_terminal_success":
                terminal = record
            elif event == "orchestrator_agent_result" and payload.get("status") == "failure":
                failures.append(
                    {
                        "ts": record.get("ts"),
                        "subtask_id": payload.get("subtask_id"),
                        "error_code": payload.get("error_code"),
                        "control_step": payload.get("control_step"),
                    }
                )
            if event in {"action_missing", "runtime_action_missing"}:
                forbidden_events.append(str(event))
            if payload.get("error_code") == "RUNTIME_ACTION_MISSING":
                forbidden_events.append("RUNTIME_ACTION_MISSING")
    return {
        "run_start": run_start,
        "run_end": run_end,
        "terminal": terminal,
        "failures": failures,
        "forbidden_events": sorted(set(forbidden_events)),
    }


def summarize_run(run_dir: Path) -> dict[str, Any]:
    log_path = run_dir / "process_data.jsonl"
    if not log_path.is_file():
        raise FileNotFoundError(f"missing run log: {log_path}")
    events = load_decisive_events(log_path)
    terminal_payload = (events["terminal"] or {}).get("payload") or {}
    physical = terminal_payload.get("physical_evidence") or {}
    run_end_payload = (events["run_end"] or {}).get("payload") or {}

    strict_success = bool(
        terminal_payload.get("status") == "success"
        and terminal_payload.get("action_keys") == []
        and terminal_payload.get("placement_success") is True
        and terminal_payload.get("placement_verified") is True
        and terminal_payload.get("placement_strategy") == "guarded_gravity_drop"
        and terminal_payload.get("released") is True
        and terminal_payload.get("aabb_contained") is True
        and terminal_payload.get("last_applied_action_keys") == ["robot_r1"]
        and not events["forbidden_events"]
    )
    start_time = parse_timestamp((events["run_start"] or {}).get("ts"))
    end_time = parse_timestamp((events["run_end"] or {}).get("ts"))
    duration_s = (
        round((end_time - start_time).total_seconds(), 3)
        if start_time is not None and end_time is not None
        else None
    )

    video_path = run_dir / "trajectory.mp4"
    drop = physical.get("pre_release_drop_evidence") or {}
    return {
        "run_id": run_dir.name,
        "strict_physical_success": strict_success,
        "started_at": (events["run_start"] or {}).get("ts"),
        "ended_at": (events["run_end"] or {}).get("ts"),
        "wall_clock_duration_s": duration_s,
        "process_data": {
            "bytes": log_path.stat().st_size,
            "sha256": sha256_file(log_path),
        },
        "video": (
            {
                "filename": video_path.name,
                "bytes": video_path.stat().st_size,
                "sha256": sha256_file(video_path),
            }
            if video_path.is_file()
            else None
        ),
        "terminal": {
            "control_step": terminal_payload.get("control_step"),
            "step_count": terminal_payload.get("step_count"),
            "status": terminal_payload.get("status"),
            "action_keys": terminal_payload.get("action_keys"),
            "placement_success": terminal_payload.get("placement_success"),
            "placement_verified": terminal_payload.get("placement_verified"),
            "placement_strategy": terminal_payload.get("placement_strategy"),
            "released": terminal_payload.get("released"),
            "aabb_contained": terminal_payload.get("aabb_contained"),
            "last_applied_action_keys": terminal_payload.get(
                "last_applied_action_keys"
            ),
        },
        "physical_evidence": {
            "pre_navigation_steps": physical.get("pre_navigation_steps"),
            "pre_navigation_mode": physical.get("pre_navigation_mode"),
            "pre_navigation_geodesic_distance_m": physical.get(
                "pre_navigation_geodesic_distance_m"
            ),
            "drop_alignment_steps": physical.get("drop_alignment_steps"),
            "drop_alignment_attempts": physical.get("drop_alignment_attempts"),
            "drop_height_m": drop.get("drop_height_m"),
            "wall_margin_m": drop.get("wall_margin_m"),
            "xy_contained_with_margin": drop.get("xy_contained_with_margin"),
            "release_steps": physical.get("release_steps"),
            "settle_steps": physical.get("settle_steps"),
            "object_aabb_world_after_settle": physical.get("object_aabb_world"),
            "destination_aabb_world_after_settle": physical.get(
                "destination_aabb_world"
            ),
        },
        "environment_run_end": {
            "task_success": run_end_payload.get("task_success"),
            "task_progress": run_end_payload.get("task_progress"),
            "step_count": run_end_payload.get("step_count"),
        },
        "failure_results": events["failures"],
        "forbidden_runtime_events": events["forbidden_events"],
    }


def render_markdown(report: dict[str, Any]) -> str:
    failure_label = (
        "failure" if report["strict_failure_count"] == 1 else "failures"
    )
    lines = [
        "# Real Isaac Sim run evidence",
        "",
        "Strict success requires a verified action-free terminal placement, "
        "guarded release, and post-settle 3-D AABB containment. The broader "
        "environment `task_success` flag is recorded but is not used as the "
        "acceptance criterion.",
        "",
        f"Generated from {report['run_count']} selected engineering runs: "
        f"{report['strict_success_count']} strict successes and "
        f"{report['strict_failure_count']} strict {failure_label}. This is a "
        "reproducibility audit, not a statistically powered benchmark.",
        "",
        "| Run | Strict | Control / env steps | Navigation | Drop | Released / contained | Video |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report["runs"]:
        terminal = run["terminal"]
        physical = run["physical_evidence"]
        video = run["video"]
        nav = physical["pre_navigation_geodesic_distance_m"]
        drop = physical["drop_height_m"]
        lines.append(
            "| {run_id} | {strict} | {control} / {steps} | {nav} m | "
            "{drop} m | {released} / {contained} | {video} |".format(
                run_id=run["run_id"],
                strict="PASS" if run["strict_physical_success"] else "FAIL",
                control=terminal["control_step"] or "-",
                steps=terminal["step_count"] or run["environment_run_end"]["step_count"] or "-",
                nav="-" if nav is None else f"{nav:.3f}",
                drop="-" if drop is None else f"{drop:.3f}",
                released=terminal["released"] if terminal["released"] is not None else "-",
                contained=terminal["aabb_contained"] if terminal["aabb_contained"] is not None else "-",
                video="-" if video is None else f"{video['bytes'] / 1_000_000:.1f} MB",
            )
        )
    lines.extend(
        [
            "",
            "The JSON companion contains SHA-256 hashes, exact geometry, "
            "negative-run failure codes, and all terminal gate fields.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--json-output", type=Path, default=Path("reports/real_isaac_runs.json")
    )
    parser.add_argument(
        "--markdown-output", type=Path, default=Path("reports/real_isaac_runs.md")
    )
    args = parser.parse_args()

    runs = [summarize_run(path.resolve()) for path in args.run_dirs]
    success_count = sum(run["strict_physical_success"] for run in runs)
    report = {
        "schema_version": 1,
        "acceptance_criterion": "strict_guarded_physical_place_inside_v1",
        "run_count": len(runs),
        "strict_success_count": success_count,
        "strict_failure_count": len(runs) - success_count,
        "runs": runs,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(
        f"wrote {args.json_output} and {args.markdown_output}: "
        f"{success_count}/{len(runs)} strict successes"
    )


if __name__ == "__main__":
    main()
