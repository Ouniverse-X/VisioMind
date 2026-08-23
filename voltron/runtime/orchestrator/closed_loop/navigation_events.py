"""Closed-loop navigation telemetry event helpers."""

from __future__ import annotations

from typing import Any

from voltron.runtime.telemetry.navigation_payloads import (
    build_nav2_path_snapshot,
    build_navigation_candidates_snapshot,
    nav2_path_snapshot_signature,
    navigation_candidates_snapshot_signature,
)
from voltron.shared.context import ExecutionContext, Subtask


def emit_navigation_candidates_snapshot_if_new(
    *,
    orchestrator: object,
    subtask: Subtask,
    context: ExecutionContext,
    control_step: int | None,
    result: dict[str, Any],
) -> None:
    if not context.runtime_state.get("log_navigation_candidates", False):
        return

    snapshot = build_navigation_candidates_snapshot(
        subtask_id=subtask.subtask_id,
        control_step=control_step,
        result=result,
    )
    signature = navigation_candidates_snapshot_signature(snapshot)
    if snapshot is None or signature is None:
        return

    key = "_logged_navigation_candidate_snapshot_signatures"
    logged = context.runtime_state.get(key)
    if not isinstance(logged, list):
        logged = []
    if signature in logged:
        return

    context.runtime_state[key] = [*logged, signature]
    orchestrator._emit_event(
        event_type="navigation_candidates",
        source=subtask.agent.value,
        message=f"navigation candidates for {subtask.subtask_id}",
        payload=snapshot,
        task_id=context.task_request.task_id,
    )


def emit_nav2_path_snapshot_if_new(
    *,
    orchestrator: object,
    subtask: Subtask,
    context: ExecutionContext,
    control_step: int | None,
    result: dict[str, Any],
    runtime_artifacts: dict[str, Any] | None = None,
) -> None:
    if not context.runtime_state.get("log_nav2_path_snapshots", False):
        return

    snapshot = build_nav2_path_snapshot(
        subtask_id=subtask.subtask_id,
        control_step=control_step,
        result=result,
        runtime_artifacts=runtime_artifacts,
    )
    signature = nav2_path_snapshot_signature(snapshot)
    if snapshot is None or signature is None:
        return

    key = "_logged_nav2_path_snapshot_signatures"
    logged = context.runtime_state.get(key)
    if not isinstance(logged, list):
        logged = []
    if signature in logged:
        return

    context.runtime_state[key] = [*logged, signature]
    orchestrator._emit_event(
        event_type="navigation_path_snapshot",
        source=subtask.agent.value,
        message=f"Nav2 path snapshot for {subtask.subtask_id}",
        payload=snapshot,
        task_id=context.task_request.task_id,
    )
