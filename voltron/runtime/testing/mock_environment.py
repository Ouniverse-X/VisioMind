"""Mock runtime environment adapter for closed-loop local testing."""

from __future__ import annotations

from typing import Any

from voltron.shared.enums import AgentName
from voltron.shared.context import ExecutionContext, Plan, Subtask, TaskRequest
from voltron.shared.results import AgentResult
from voltron.shared.contracts import RuntimeEnvironment
from voltron.shared.models import RuntimeFeedback, SubtaskStepOutcome


class MockRuntimeEnvironment(RuntimeEnvironment):
    """Deterministic mock environment for closed-loop orchestrator tests/examples.

    Each subtask finishes after `step_budget_per_subtask` control iterations unless
    agent reports failure.
    """

    def __init__(
        self,
        step_budget_per_subtask: int = 2,
        subtask_budget_overrides: dict[str, int] | None = None,
    ):
        self.step_budget_per_subtask = max(1, step_budget_per_subtask)
        self.subtask_budget_overrides = dict(subtask_budget_overrides or {})
        self._steps: dict[str, int] = {}
        self._completed: set[str] = set()
        self._completed_subtask_labels: list[str] = []
        self._failed: str | None = None
        self._closed = False
        self._plan_subtasks: list[str] = []
        self._global_step = 0
        self._dynamic_execution = False
        self._task_success = False
        self._navigation_runtime_state: dict[str, dict[str, Any]] = {}
        self._current_room: str | None = None
        self._current_region: str | None = None

    def reset(self, request: TaskRequest, plan: Plan, context: ExecutionContext) -> dict[str, Any]:
        self._steps.clear()
        self._completed.clear()
        self._completed_subtask_labels.clear()
        self._failed = None
        self._closed = False
        self._global_step = 0
        self._plan_subtasks = [item.runtime_id for item in plan.subtasks]
        self._dynamic_execution = bool(plan.metadata.get("dynamic_execution", False))
        self._task_success = False
        self._navigation_runtime_state.clear()
        self._current_room = None
        self._current_region = None
        return {"mode": "mock_runtime_environment", "task_id": request.task_id}

    def update_plan(self, plan: Plan, context: ExecutionContext) -> None:
        self._dynamic_execution = self._dynamic_execution or bool(plan.metadata.get("dynamic_execution", False))
        if plan.metadata.get("replace_active_plan"):
            self._plan_subtasks = [item.runtime_id for item in plan.subtasks]
            return
        for item in plan.subtasks:
            if item.runtime_id not in self._plan_subtasks:
                self._plan_subtasks.append(item.runtime_id)

    def build_runtime_inputs(self, subtask: Subtask, context: ExecutionContext) -> dict[str, Any]:
        if subtask.agent == AgentName.VISION:
            return {
                "images": ["ZmFrZV9pbWFnZQ=="],  # "fake_image" base64 placeholder
                "instruction": subtask.parameters.get("instruction", subtask.action),
            }
        if subtask.agent == AgentName.ACTION:
            step = self._steps.get(subtask.runtime_id, 0)
            return {
                "observation": {"mock_step": step, "trace_id": context.trace_id},
                "pre_state": {"holding_object": step > 0},
                "post_state": {"holding_object": step >= 1},
            }
        # default for Navigation
        payload = {
            "observation": {"mock_step": self._steps.get(subtask.runtime_id, 0)},
            "pose": {"x": float(self._global_step), "y": 0.0, "z": 0.0},
            "nav_feedback": {"stuck": False, "collision": False},
        }
        nav_state = self._navigation_runtime_state.get(subtask.runtime_id)
        if nav_state:
            payload.update(nav_state)
        return payload

    def on_agent_result(
        self,
        subtask: Subtask,
        result: AgentResult,
        context: ExecutionContext,
    ) -> SubtaskStepOutcome:
        self._global_step += 1

        if result.status.value != "success":
            self._failed = result.error_code or "AGENT_FAILURE"
            return SubtaskStepOutcome(
                done=True,
                success=False,
                failure_reason=self._failed,
                feedback=RuntimeFeedback(step_count=self._global_step, extras={"global_step": self._global_step}),
            )

        if subtask.agent == AgentName.NAVIGATION:
            nav_goal = result.runtime_artifacts.get("nav_goal") or result.runtime_artifacts.get("grounded_goal")
            waypoints = result.runtime_artifacts.get("waypoints")
            if not isinstance(waypoints, list):
                path_plan = result.runtime_artifacts.get("path_plan")
                if isinstance(path_plan, dict):
                    waypoints = path_plan.get("waypoints")
            self._navigation_runtime_state[subtask.runtime_id] = {
                "nav_goal": dict(nav_goal) if isinstance(nav_goal, dict) else None,
                "waypoints": list(waypoints) if isinstance(waypoints, list) else None,
                "active_waypoint_index": result.runtime_artifacts.get("active_waypoint_index"),
                "recovery_mode": result.runtime_artifacts.get("recovery_mode"),
                "exploration_target": result.runtime_artifacts.get("exploration_target"),
            }

        steps = self._steps.get(subtask.runtime_id, 0) + 1
        self._steps[subtask.runtime_id] = steps
        budget = self.subtask_budget_overrides.get(
            subtask.runtime_id,
            self.subtask_budget_overrides.get(subtask.subtask_id, self.step_budget_per_subtask),
        )
        done = steps >= budget
        if subtask.agent == AgentName.ACTION:
            action_progress = result.runtime_artifacts.get("action_execution_progress")
            if isinstance(action_progress, dict):
                done = bool(action_progress.get("plan_completed", False))
        if done and subtask.runtime_id not in self._completed:
            self._completed.add(subtask.runtime_id)
            self._completed_subtask_labels.append(subtask.subtask_id)
        if bool(result.result.get("task_complete", False)):
            self._task_success = True

        feedback_extras = {"global_step": self._global_step, "subtask_steps": steps, "budget": budget}
        if subtask.agent == AgentName.NAVIGATION:
            navigation_feedback = self._build_navigation_feedback(
                subtask=subtask,
                result=result,
                done=done,
            )
            feedback_extras.update(navigation_feedback)

        return SubtaskStepOutcome(
            done=done,
            success=True if done else None,
            feedback=RuntimeFeedback(
                step_count=self._global_step,
                current_room=self._current_room,
                current_region=self._current_region,
                extras=feedback_extras,
            ),
        )

    def task_succeeded(self, context: ExecutionContext) -> bool:
        if self._failed is not None:
            return False
        if self._dynamic_execution:
            return self._task_success
        return set(self._plan_subtasks).issubset(self._completed)

    def summary(self) -> dict[str, Any]:
        return {
            "closed": self._closed,
            "failed_reason": self._failed,
            "completed_subtasks": list(self._completed_subtask_labels),
            "completed_execution_ids": sorted(self._completed),
            "subtask_steps": dict(self._steps),
            "global_steps": self._global_step,
            "task_success": self._task_success,
        }

    def close(self) -> None:
        self._closed = True

    def _build_navigation_feedback(
        self,
        *,
        subtask: Subtask,
        result: AgentResult,
        done: bool,
    ) -> dict[str, Any]:
        target_room = self._coerce_text(
            subtask.target.get("room"),
            fallback=subtask.target.get("region"),
        )
        if done and target_room:
            self._current_room = target_room
            self._current_region = target_room

        goal_payload = self._navigation_goal_payload(subtask=subtask, result=result)
        feedback: dict[str, Any] = {}
        if target_room:
            feedback["current_room"] = target_room
            feedback["current_region"] = target_room

        if subtask.action == "approach_target":
            feedback["path_backend"] = "mock_object_approach"
            feedback["best_distance_to_waypoint"] = 0.0 if done else 0.5
            feedback["goal_reached"] = done
            if goal_payload:
                feedback["execution_goal"] = dict(goal_payload)
                feedback["local_goal"] = dict(goal_payload)
            return feedback

        if subtask.action == "navigate":
            feedback["path_backend"] = "global_goal_reached" if done else "mock_navigation"
            feedback["best_distance_to_waypoint"] = 0.0 if done else 1.0
            feedback["goal_reached"] = done
            if goal_payload:
                feedback["execution_goal"] = dict(goal_payload)
            return feedback

        return feedback

    @staticmethod
    def _navigation_goal_payload(
        *,
        subtask: Subtask,
        result: AgentResult,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key in ("execution_goal", "local_goal", "target_waypoint", "nav_goal", "grounded_goal"):
            candidate = result.runtime_artifacts.get(key) or result.result.get(key)
            if isinstance(candidate, dict) and candidate:
                payload.update(candidate)
                break

        for key in ("object", "object_id", "room", "region"):
            value = subtask.target.get(key)
            if value not in (None, "", {}):
                payload.setdefault(key, value)
        return payload

    @staticmethod
    def _coerce_text(value: Any, *, fallback: Any = None) -> str | None:
        for candidate in (value, fallback):
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return None
