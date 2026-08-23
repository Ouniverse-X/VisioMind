"""Task-request builders for runtime entry surfaces."""

from __future__ import annotations

from typing import Any

from voltron.shared.context import TaskRequest
from voltron.shared.enums import TaskType


def build_task_request(
    *,
    args: Any,
    scene_id: str | None,
    hovsg_runtime: dict[str, Any],
) -> TaskRequest:
    metadata = {
        key: value
        for key, value in {
            "planner_mode": args.planner_mode,
            "scene_id": scene_id,
            "hovsg_graph_root": hovsg_runtime.get("graph_root"),
            "hovsg_graph_path": hovsg_runtime.get("graph_path"),
            "hovsg_nav_graph_type": hovsg_runtime.get("nav_graph_type"),
            "action_control_mode": getattr(args, "action_control_mode", None),
            "policy_backend": getattr(args, "policy_backend", None),
        }.items()
        if value
    }
    if bool(getattr(args, "action_allow_base_motion", False)):
        metadata["action_allow_base_motion"] = True
    return TaskRequest(
        task_id=args.task_id,
        description=args.task_desc,
        task_type=TaskType(args.task_type),
        metadata=metadata,
    )
