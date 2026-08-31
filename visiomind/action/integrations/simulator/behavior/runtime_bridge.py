from __future__ import annotations

from typing import Any, Callable

from visiomind.action.shared.context import ExecutionContext, Plan, Subtask, TaskRequest
from visiomind.action.shared.results import AgentResult
from visiomind.action.runtime.task_state import execution_state as runtime_execution_state
from visiomind.action.runtime.task_state import plan_state as runtime_plan_state
from visiomind.action.runtime.task_state import subtask_state as runtime_subtask_state
from visiomind.action.integrations.simulator.behavior.environment import (
    client as behavior_environment_client,
)
from visiomind.action.integrations.simulator.behavior.tools import (
    bridge_execution as behavior_bridge_execution,
)
from visiomind.action.integrations.simulator.behavior.tools import (
    bridge_environment as behavior_bridge_environment,
)
from visiomind.action.integrations.simulator.behavior.tools import bridge_inputs as behavior_bridge_inputs
from visiomind.action.integrations.simulator.behavior.tools import (
    bridge_lifecycle as behavior_bridge_lifecycle,
)
from visiomind.action.integrations.simulator.behavior.tools import (
    bridge_localization as behavior_bridge_localization,
)
from visiomind.action.integrations.simulator.behavior.tools import (
    bridge_recording as behavior_bridge_recording,
)
from visiomind.action.integrations.simulator.behavior.tools import (
    bridge_subtasks as behavior_bridge_subtasks,
)
from visiomind.action.integrations.simulator.behavior.tools import (
    door_navigation_passability as behavior_door_navigation_passability,
)
from visiomind.action.integrations.simulator.behavior.tools import (
    runtime_adapter_state as behavior_runtime_adapter_state,
)
from visiomind.action.integrations.simulator.behavior.tools import runtime_config as behavior_runtime_config
from visiomind.action.integrations.simulator.behavior.tools import (
    runtime_control as behavior_runtime_control,
)
from visiomind.action.shared.contracts import RuntimeEnvironment
from visiomind.action.shared.models import SubtaskStepOutcome

_RUNTIME_BRIDGE_FILE = __file__


class BehaviorRuntimeEnvironment(RuntimeEnvironment):
    def __init__(
        self,
        env_id: str,
        env_kwargs: dict[str, Any] | None = None,
        env_factory: Callable[[], Any] | None = None,
        auto_register: bool = True,
        default_subtask_max_steps: int | None = None,
        progress_log_every: int | None = None,
        recording_video_scale: float = 1.0,
        logging_verbose: bool = True,
        logging_memory_diagnostics: bool = False,
        enable_transcode_watchdog: bool | None = None,
        runtime_termination_use_environment_success_signal: bool = True,
        runtime_termination_environment_signal_policy: str = "allow_early_success",
        object_goal_distance_tolerance_m: float = 0.9,
        object_goal_heading_tolerance_rad: float = 0.65,
    ):
        behavior_runtime_config.configure_adapter(
            self,
            env_id=env_id,
            env_kwargs=env_kwargs,
            env_factory=env_factory,
            auto_register=auto_register,
            default_subtask_max_steps=default_subtask_max_steps,
            progress_log_every=progress_log_every,
            recording_video_scale=recording_video_scale,
            logging_verbose=logging_verbose,
            logging_memory_diagnostics=logging_memory_diagnostics,
            enable_transcode_watchdog=enable_transcode_watchdog,
            runtime_termination_use_environment_success_signal=runtime_termination_use_environment_success_signal,
            runtime_termination_environment_signal_policy=runtime_termination_environment_signal_policy,
            object_goal_distance_tolerance_m=object_goal_distance_tolerance_m,
            object_goal_heading_tolerance_rad=object_goal_heading_tolerance_rad,
            extract_runtime_kwarg=behavior_bridge_localization.extract_runtime_kwarg,
            normalize_progress_log_every=behavior_bridge_recording.normalize_progress_log_every,
        )
        behavior_runtime_config.configure_tempdir()
        behavior_runtime_config.initialize_runtime_state(self)

    def reset(self, request: TaskRequest, plan: Plan, context: ExecutionContext) -> dict[str, Any]:
        behavior_door_navigation_passability.clear_navigation_passability_overrides(self)
        self._root_task_instruction = request.description.strip()
        self._task_type = request.task_type
        reset = behavior_runtime_control.reset_runtime_session(
            request=request,
            plan=plan,
            env_id=self.env_id,
            metadata_scene_id=self._scene_id,
            metadata_hovsg_graph_root=self._hovsg_graph_root,
            metadata_hovsg_graph_path=self._hovsg_graph_path,
            metadata_hovsg_nav_graph_type=self._hovsg_nav_graph_type,
            normalize_runtime_str=behavior_bridge_localization.normalize_runtime_str,
            start_recording=lambda request, plan: behavior_bridge_lifecycle.start_recording(
                self,
                request,
                plan,
                runtime_bridge_file=_RUNTIME_BRIDGE_FILE,
            ),
            record_event=lambda event, payload: behavior_bridge_lifecycle.record_event(
                self, event, payload
            ),
            configure_runtime_subtasks=self._configure_runtime_subtasks,
            ensure_env=lambda: behavior_bridge_environment.ensure_env(
                self,
                runtime_bridge_file=_RUNTIME_BRIDGE_FILE,
            ),
            sync_runtime_subtasks=lambda runtime_subtasks: self._call_behavior_env_method(
                "set_runtime_subtasks",
                [dict(item) for item in runtime_subtasks],
            ),
            capture_reset_runtime_state=runtime_execution_state.capture_reset_runtime_state,
            record_frame=lambda obs: behavior_bridge_lifecycle.record_frame(self, obs),
            localize_runtime_state=lambda last_obs, last_info, resolved_metadata: (
                behavior_bridge_environment.localize_runtime_state_snapshot(
                    self,
                    last_obs,
                    last_info,
                    resolved_metadata,
                )
            ),
            extract_pose=lambda last_info,
            last_obs: behavior_bridge_localization.extract_runtime_pose(
                last_info=last_info,
                last_obs=last_obs,
                frame_config=self.env_kwargs,
            ),
            apply_post_reset_state=lambda env,
            obs,
            info: behavior_bridge_environment.apply_post_reset_state(
                self,
                env=env,
                obs=obs,
                info=info,
            ),
        )
        return behavior_runtime_adapter_state.apply_reset_result(self, reset)

    def update_plan(self, plan: Plan, context: ExecutionContext) -> None:
        if not plan.subtasks:
            return

        updated = behavior_runtime_control.apply_plan_update(
            plan=plan,
            runtime_subtasks=self._runtime_subtasks,
            runtime_subtasks_by_id=self._runtime_subtasks_by_id,
            env_kwargs=self.env_kwargs,
            merge_plan_runtime_subtasks=runtime_plan_state.merge_plan_runtime_subtasks,
            build_runtime_subtask=self._build_runtime_subtask,
            sync_runtime_subtasks=lambda runtime_subtasks: self._call_behavior_env_method(
                "set_runtime_subtasks",
                [dict(item) for item in runtime_subtasks],
            ),
            record_event=lambda event, payload: behavior_bridge_lifecycle.record_event(
                self,
                event=event,
                payload=payload,
            ),
        )
        behavior_runtime_adapter_state.apply_plan_update(self, updated)

    def build_runtime_inputs(self, subtask: Subtask, context: ExecutionContext) -> dict[str, Any]:
        runtime_inputs = behavior_bridge_inputs.build_runtime_inputs(
            self,
            subtask=subtask,
        )
        context.runtime_state["navigation_door_passability"] = (
            behavior_door_navigation_passability.active_override_payload(self)
        )
        return runtime_inputs

    def _configure_runtime_subtasks(self, plan: Plan) -> dict[str, Any]:
        configured = runtime_plan_state.configure_runtime_subtasks(
            plan=plan,
            env_kwargs=self.env_kwargs,
            build_runtime_subtask=self._build_runtime_subtask,
            call_env_method=lambda method_name, payload: self._call_behavior_env_method(
                method_name, payload
            ),
        )
        return behavior_runtime_adapter_state.apply_configured_runtime_subtasks(self, configured)

    def _sync_runtime_subtask(self, subtask: Subtask) -> dict[str, Any] | None:
        synced = runtime_subtask_state.sync_runtime_subtask(
            subtask=subtask,
            runtime_subtasks_by_id=self._runtime_subtasks_by_id,
            instruction_for_subtask=self._instruction_for_subtask,
            planned_subtask_name=lambda item: behavior_bridge_subtasks.planned_subtask_name(
                item,
                slugify=behavior_bridge_recording.safe_slug,
            ),
            last_info=self._last_info,
            call_env_method=self._call_behavior_env_method,
            record_event=lambda event, payload: behavior_bridge_lifecycle.record_event(
                self, event, payload
            ),
        )
        return behavior_runtime_adapter_state.apply_synced_runtime_subtask(self, synced)

    def _call_behavior_env_method(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        return behavior_environment_client.call_env_method(self._env, method_name, *args, **kwargs)

    def _build_runtime_subtask(self, subtask: Subtask) -> dict[str, Any]:
        return behavior_bridge_subtasks.build_runtime_subtask(
            subtask=subtask,
            default_subtask_max_steps=self.default_subtask_max_steps,
            slugify=behavior_bridge_recording.safe_slug,
        )

    @staticmethod
    def _instruction_for_subtask(subtask: Subtask) -> str:
        return behavior_bridge_subtasks.instruction_for_subtask(subtask)

    def _recording_subtask_name(self) -> str | None:
        return behavior_bridge_subtasks.recording_subtask_name(
            active_internal_step=self._active_action_internal_step,
            last_info=self._last_info,
            active_subtask_name=self._active_subtask_name,
        )

    def _recording_subtask_instruction(self) -> str | None:
        return behavior_bridge_subtasks.recording_subtask_instruction(
            active_internal_step=self._active_action_internal_step,
            active_subtask_instruction=self._active_subtask_instruction,
        )

    def on_agent_result(
        self,
        subtask: Subtask,
        result: AgentResult,
        context: ExecutionContext,
    ) -> SubtaskStepOutcome:
        return behavior_bridge_execution.on_agent_result(
            self,
            subtask=subtask,
            result=result,
            context=context,
            runtime_bridge_file=_RUNTIME_BRIDGE_FILE,
        )

    def record_orchestrator_event(self, event: dict[str, Any]) -> None:
        event_name = str(event.get("event") or "").strip()
        payload = event.get("payload")
        if not event_name:
            return
        behavior_bridge_lifecycle.record_event(
            self,
            event=event_name,
            payload=dict(payload) if isinstance(payload, dict) else {},
        )

    def on_subtask_completion_decision(
        self,
        subtask: Subtask,
        decision: dict[str, Any],
        context: ExecutionContext,
    ) -> None:
        update = behavior_door_navigation_passability.apply_completion_decision(
            self,
            subtask=subtask,
            decision=decision,
        )
        if update is None:
            return
        context.runtime_state["navigation_door_passability"] = dict(update["active_overrides"])
        behavior_bridge_lifecycle.record_event(
            self,
            event="navigation_door_passability_updated",
            payload={key: value for key, value in update.items() if key != "active_overrides"},
        )

    def task_succeeded(self, context: ExecutionContext) -> bool:
        return behavior_bridge_lifecycle.task_succeeded(self)

    def summary(self) -> dict[str, Any]:
        return behavior_bridge_lifecycle.build_summary(self)

    def close(self) -> None:
        behavior_bridge_lifecycle.close(self)
