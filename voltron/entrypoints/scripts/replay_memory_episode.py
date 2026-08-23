#!/usr/bin/env python3
"""Deterministic offline replay helpers for memory-guided navigation episodes."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from voltron.agents.navigation.body.object_approach_selection import HeuristicNavigationApproachPointSelector
from voltron.agents.navigation.skills.object_approach.skill import ObjectApproachSelectionSkill
from voltron.shared.context import ExecutionContext, Subtask, TaskRequest
from voltron.shared.enums import AgentName, TaskType


def replay_object_approach_memory(
    *,
    memory: Any,
    scene_id: str,
    task_description: str,
    target: dict[str, Any],
    candidates: list[dict[str, Any]],
    failure_reason: str,
) -> dict[str, Any]:
    """Replay two object-approach selections and verify memory changes the second run."""
    first_run = _select_object_approach(
        memory=memory,
        scene_id=scene_id,
        task_description=task_description,
        target=target,
        candidates=candidates,
        memory_navigation_guidance=None,
    )
    first_candidate = first_run.get("selected_candidate")
    if isinstance(first_candidate, dict) and first_candidate:
        memory.record_object_approach_outcome(
            scene_id=scene_id,
            target=target,
            candidate=first_candidate,
            outcome="failure",
            reason=failure_reason,
            metadata={"source_agent": "NAVIGATION", "replay_phase": "first_run"},
        )

    evidence_summary = _memory_evidence_summary(
        memory=memory,
        task_description=task_description,
        task_type="navigation",
        scene_id=scene_id,
        target=target,
        top_k=5,
    )
    guidance = evidence_summary.get("navigation_guidance")
    second_run = _select_object_approach(
        memory=memory,
        scene_id=scene_id,
        task_description=task_description,
        target=target,
        candidates=candidates,
        memory_navigation_guidance=guidance if isinstance(guidance, dict) else None,
    )
    return {
        "scene_id": scene_id,
        "target": deepcopy(target),
        "first_run": first_run,
        "second_run": second_run,
        "memory_evidence_summary": evidence_summary,
        "memory_changed_candidate": first_run.get("selected_candidate_id") != second_run.get("selected_candidate_id"),
    }


def _select_object_approach(
    *,
    memory: Any,
    scene_id: str,
    task_description: str,
    target: dict[str, Any],
    candidates: list[dict[str, Any]],
    memory_navigation_guidance: dict[str, Any] | None,
) -> dict[str, Any]:
    context = ExecutionContext(
        trace_id="memory_replay",
        task_request=TaskRequest(
            task_id="memory_replay",
            description=task_description,
            task_type=TaskType.NAVIGATION,
        ),
    )
    subtask_context = {"task_description": task_description}
    if memory_navigation_guidance:
        subtask_context["memory_navigation_guidance"] = deepcopy(memory_navigation_guidance)
    subtask = Subtask(
        subtask_id="st_memory_replay_approach",
        agent=AgentName.NAVIGATION,
        action="approach_target",
        target=deepcopy(target),
        parameters={"instruction": task_description, "scene_id": scene_id},
        context=subtask_context,
    )
    goal = {
        "scene_id": scene_id,
        "goal_type": "object",
        "object": target.get("object"),
        "object_id": target.get("object_id"),
        "object_name": target.get("object_name") or target.get("object"),
        "room_id": target.get("room_id"),
        "room_name": target.get("room_name") or target.get("room"),
        "floor_id": target.get("floor_id"),
    }
    prepared = ObjectApproachSelectionSkill(memory).prepare(
        subtask=subtask,
        context=context,
        navigator=_CandidateReplayNavigator(candidates),
        start={"scene_id": scene_id},
        goal=goal,
        navigation_context={"scene_id": scene_id},
    )
    selection = HeuristicNavigationApproachPointSelector().select_candidate(
        subtask=subtask,
        context=context,
        goal=goal,
        prepared_payload=prepared,
    )
    selected_candidate = selection.get("candidate")
    selected_candidate = deepcopy(selected_candidate) if isinstance(selected_candidate, dict) else None
    return {
        "selected_candidate_id": selected_candidate.get("candidate_id") if selected_candidate else None,
        "selected_candidate": selected_candidate,
        "object_approach_candidates": deepcopy(prepared.get("candidates", [])),
        "object_approach_selection": deepcopy(selection),
    }


def _memory_evidence_summary(
    *,
    memory: Any,
    task_description: str,
    task_type: str,
    scene_id: str,
    target: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    summary = getattr(memory, "get_memory_evidence_summary", None)
    if callable(summary):
        result = summary(
            task_description,
            task_type=task_type,
            scene_id=scene_id,
            target=target,
            top_k=top_k,
        )
        if isinstance(result, dict):
            return result
    history = memory.get_object_approach_history(scene_id=scene_id, target=target, top_k=top_k)
    return {
        "query_type": "memory_evidence_summary",
        "query": {
            "task_description": task_description,
            "task_type": task_type,
            "scene_id": scene_id,
            "target": deepcopy(target),
        },
        "retrieval": {},
        "navigation_guidance": {
            "object_approach_history": history,
            "avoid_object_approach_candidates": _avoid_candidates_from_history(history, top_k=top_k),
            "prefer_object_approach_candidates": [],
            "risk_reasons": [],
        },
        "runtime": {},
        "metadata": {"available": True, "top_k": top_k, "fallback": "object_approach_history"},
    }


def _avoid_candidates_from_history(history: dict[str, Any], *, top_k: int) -> list[dict[str, Any]]:
    avoid: list[dict[str, Any]] = []
    for entry in history.get("entries", []):
        if not isinstance(entry, dict):
            continue
        if str(entry.get("outcome") or "").strip().lower() != "failure":
            continue
        signature = entry.get("candidate_signature")
        if not isinstance(signature, dict) or not signature:
            continue
        avoid.append(
            {
                "candidate_signature": deepcopy(signature),
                "reason": entry.get("reason"),
                "failure_count": 1,
                "last_outcome": "failure",
            }
        )
    return avoid[: max(int(top_k), 0)]


class _CandidateReplayNavigator:
    def __init__(self, candidates: list[dict[str, Any]]) -> None:
        self._candidates = [deepcopy(candidate) for candidate in candidates]

    def generate_object_approach_candidates(
        self,
        *,
        start: dict[str, Any],
        goal: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        del start, goal, context
        return [deepcopy(candidate) for candidate in self._candidates]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON replay input payload.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    from voltron.integrations.memory.hems.backend import HEMSAdapter

    args = parse_args()
    with args.input.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    memory = HEMSAdapter(
        auto_initialize=True,
        persistence_dir=payload.get("persistence_dir"),
    )
    result = replay_object_approach_memory(
        memory=memory,
        scene_id=payload["scene_id"],
        task_description=payload["task_description"],
        target=payload["target"],
        candidates=payload["candidates"],
        failure_reason=payload.get("failure_reason", "replay_failure"),
    )
    output = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    if args.output is None:
        print(output)
    else:
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
