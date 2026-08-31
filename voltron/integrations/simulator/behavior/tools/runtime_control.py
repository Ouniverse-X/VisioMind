from __future__ import annotations

from typing import Any, Callable

from voltron.runtime.session import task_session
from voltron.shared.context import Plan, TaskRequest


def reset_runtime_session(
    *,
    request: TaskRequest,
    plan: Plan,
    env_id: str,
    metadata_scene_id: str | None,
    metadata_hovsg_graph_root: str | None,
    metadata_hovsg_graph_path: str | None,
    metadata_hovsg_nav_graph_type: str | None,
    normalize_runtime_str: Callable[[Any], str | None],
    start_recording: Callable[[TaskRequest, Plan], None],
    record_event: Callable[[str, dict[str, Any]], None],
    configure_runtime_subtasks: Callable[[Plan], dict[str, Any]],
    ensure_env: Callable[[], Any],
    sync_runtime_subtasks: Callable[[list[dict[str, Any]]], None],
    capture_reset_runtime_state: Callable[..., dict[str, Any]],
    record_frame: Callable[[dict[str, Any]], None],
    localize_runtime_state: Callable[
        [dict[str, Any], dict[str, Any], dict[str, str | None]], dict[str, Any]
    ],
    extract_pose: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any] | None],
    apply_post_reset_state: Callable[[Any, Any, Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    resolved_metadata = task_session.resolve_request_runtime_metadata(
        metadata=request.metadata,
        scene_id=metadata_scene_id,
        hovsg_graph_root=metadata_hovsg_graph_root,
        hovsg_graph_path=metadata_hovsg_graph_path,
        hovsg_nav_graph_type=metadata_hovsg_nav_graph_type,
        normalize_runtime_str=normalize_runtime_str,
    )
    start_recording(request, plan)
    record_event(
        "run_start",
        task_session.build_run_start_payload(
            request=request,
            env_id=env_id,
            plan=plan,
        ),
    )
    record_event(
        "orchestrator_brain_plan",
        {
            "source": "BRAIN",
            "message": f"initial plan with {len(plan.subtasks)} subtasks",
            **task_session.build_brain_plan_payload(
                plan=plan,
                reason="initial_plan",
                request=request,
            ),
        },
    )
    configured = configure_runtime_subtasks(plan)

    try:
        env = ensure_env()
        sync_runtime_subtasks(configured["runtime_subtasks"])
        obs, info = env.reset()
        if apply_post_reset_state is not None:
            post_reset = apply_post_reset_state(env, obs, info)
            if isinstance(post_reset, dict):
                obs = post_reset.get("obs", obs)
                info = post_reset.get("info", info)
                event_payload = post_reset.get("event_payload")
                if isinstance(event_payload, dict):
                    record_event("post_reset_state", event_payload)
    except Exception as exc:
        record_event(
            "reset_error",
            {"error": str(exc), "env_id": env_id},
        )
        raise

    reset_state = capture_reset_runtime_state(obs=obs, info=info)
    record_frame(reset_state["last_obs"])
    localized_info = localize_runtime_state(
        reset_state["last_obs"],
        reset_state["last_info"],
        resolved_metadata,
    )
    if isinstance(localized_info, dict):
        reset_state["last_info"] = localized_info
    record_event(
        "reset_ok",
        task_session.build_reset_ok_payload(info=reset_state["last_info"]),
    )

    result_payload = task_session.build_reset_result_payload(
        env_id=env_id,
        request=request,
        plan=plan,
        last_info=reset_state["last_info"],
        pose=extract_pose(reset_state["last_info"], reset_state["last_obs"]),
    )
    return {
        "resolved_metadata": resolved_metadata,
        "runtime_subtasks": configured["runtime_subtasks"],
        "runtime_subtasks_by_id": configured["runtime_subtasks_by_id"],
        "reset_state": reset_state,
        "result_payload": result_payload,
    }


def apply_plan_update(
    *,
    plan: Plan,
    runtime_subtasks: list[dict[str, Any]],
    runtime_subtasks_by_id: dict[str, dict[str, Any]],
    env_kwargs: dict[str, Any],
    merge_plan_runtime_subtasks: Callable[..., dict[str, Any]],
    build_runtime_subtask: Callable[[Any], dict[str, Any]],
    sync_runtime_subtasks: Callable[[list[dict[str, Any]]], None],
    record_event: Callable[[str, dict[str, Any]], None],
) -> dict[str, Any]:
    merged = merge_plan_runtime_subtasks(
        plan=plan,
        runtime_subtasks=runtime_subtasks,
        runtime_subtasks_by_id=runtime_subtasks_by_id,
        build_runtime_subtask=build_runtime_subtask,
    )
    updated_env_kwargs = dict(env_kwargs)
    updated_env_kwargs["runtime_subtasks"] = [dict(item) for item in merged["runtime_subtasks"]]
    sync_runtime_subtasks(merged["runtime_subtasks"])
    event_payload = {
        "added_subtasks": merged["added_subtask_ids"],
        "runtime_subtask_count": len(merged["runtime_subtasks"]),
    }
    if plan.metadata.get("plan_revision") is not None or merged.get("replace_active_plan"):
        event_payload.update(
            {
                "added_execution_ids": merged.get("added_execution_ids", []),
                "replaced_subtasks": merged.get("replaced_subtask_ids", []),
                "replaced_execution_ids": merged.get("replaced_execution_ids", []),
                "replace_active_plan": bool(merged.get("replace_active_plan", False)),
                "plan_revision": plan.metadata.get("plan_revision"),
            }
        )
    record_event("plan_update", event_payload)
    return {
        "runtime_subtasks": merged["runtime_subtasks"],
        "runtime_subtasks_by_id": merged["runtime_subtasks_by_id"],
        "env_kwargs": updated_env_kwargs,
    }


__all__ = [
    "apply_plan_update",
    "reset_runtime_session",
]
