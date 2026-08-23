"""Long-lived user-command session for Voltron runtime control."""

from __future__ import annotations

import importlib
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Callable

from voltron.agents import ActionAgent, BrainAgent, NavigationAgent, VisionAgent
from voltron.agents.action.tools.action_projection import ActionProjection
from voltron.agents.brain.contracts import BrainPlanningSession, PlanConfirmation, UserAnswer
from voltron.agents.brain.body.planning_loop import BrainPlanningEvent
from voltron.agents.brain.body.rule_based_planner import RuleBasedPlanner
from voltron.runtime.orchestrator.closed_loop import ClosedLoopOrchestrator
from voltron.runtime.session.events import VoltronEvent
from voltron.runtime.testing import MockMemoryAdapter, MockPolicyAdapter, MockRuntimeEnvironment, MockVisionAdapter
from voltron.shared.context import TaskRequest
from voltron.shared.enums import TaskType
from voltron.shared.models import PerceptionObject, PerceptionReport


EventSink = Callable[[VoltronEvent], None]
RequestFactory = Callable[[str, TaskType, str], TaskRequest]


@dataclass
class VoltronSession:
    """Session wrapper that accepts user commands and runs closed-loop tasks."""

    orchestrator: ClosedLoopOrchestrator
    environment_factory: Callable[[], Any]
    event_sink: EventSink | None = None
    session_id: str = "voltron_session"
    default_task_type: TaskType = TaskType.MANIPULATION
    request_factory: RequestFactory | None = None
    task_id_template: str = "{session_id}_task_{counter:03d}"
    _task_counter: int = field(default=0, init=False)
    _latest_environment: Any | None = field(default=None, init=False)
    _closed_environment_ids: set[int] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self.orchestrator.event_sink = self._emit
        brain_loop = getattr(getattr(self.orchestrator, "brain_agent", None), "planning_loop", None)
        if brain_loop is not None:
            brain_loop.event_sink = self._emit_brain_event

    def run_user_command(self, text: str, *, task_type: str | TaskType | None = None) -> dict[str, Any]:
        self._task_counter += 1
        parsed_task_type = _parse_task_type(task_type) if task_type is not None else self.default_task_type
        request = self._build_request(text=text, task_type=parsed_task_type)
        self._emit_user_command(request)
        environment = self.environment_factory()
        self._latest_environment = environment
        return self.orchestrator.run_task(request=request, environment=environment)

    def begin_interactive_user_command(
        self,
        text: str,
        *,
        task_type: str | TaskType | None = None,
    ) -> BrainPlanningSession:
        """Start Brain's text-plan clarification flow without resetting the environment."""

        self._task_counter += 1
        parsed_task_type = _parse_task_type(task_type) if task_type is not None else self.default_task_type
        request = self._build_request(text=text, task_type=parsed_task_type)
        self._emit_user_command(request)
        session = self.orchestrator.brain_agent.begin_interactive_prepare(request)
        self._emit_interactive_session_events(session, include_questions=True)
        return session

    def answer_interactive_planning_question(
        self,
        session: BrainPlanningSession,
        *,
        question_id: str,
        answer: str,
    ) -> BrainPlanningSession:
        updated = self.orchestrator.brain_agent.answer_planning_question(
            session,
            UserAnswer(question_id=question_id, answer=answer),
        )
        self._emit(
            VoltronEvent(
                event_type="brain_user_answer_recorded",
                source="BRAIN",
                message="recorded user planning clarification",
                payload={
                    "session_id": updated.session_id,
                    "question_id": question_id,
                    "answer": answer,
                },
                task_id=updated.task_id,
            )
        )
        self._emit(
            VoltronEvent(
                event_type="brain_text_plan_revised",
                source="BRAIN",
                message="revised interactive text plan",
                payload={
                    "session_id": updated.session_id,
                    "status": updated.status,
                    "text_plan": updated.draft.to_dict(),
                },
                task_id=updated.task_id,
            )
        )
        self._emit_plan_confirmation_request(updated)
        return updated

    def confirm_interactive_plan(
        self,
        session: BrainPlanningSession,
        *,
        confirmed: bool,
        user_message: str | None = None,
    ):
        context, plan = self.orchestrator.brain_agent.confirm_interactive_plan_with_context(
            session,
            PlanConfirmation(confirmed=confirmed, user_message=user_message),
        )
        self._emit(
            VoltronEvent(
                event_type="brain_plan_confirmed" if confirmed else "brain_plan_rejected",
                source="BRAIN",
                message="interactive plan confirmed" if confirmed else "interactive plan rejected",
                payload={
                    "session_id": session.session_id,
                    "confirmed": bool(confirmed),
                    "user_message": user_message,
                    "text_plan": session.draft.to_dict(),
                },
                task_id=session.task_id,
            )
        )
        return context, plan

    def run_user_command_with_events(
        self,
        text: str,
        *,
        task_type: str | TaskType | None = None,
    ) -> tuple[dict[str, Any], list[VoltronEvent]]:
        events: list[VoltronEvent] = []
        previous_sink = self.event_sink
        self.event_sink = events.append
        try:
            result = self.run_user_command(text, task_type=task_type)
        finally:
            self.event_sink = previous_sink
        return result, events

    def _emit(self, event: VoltronEvent) -> None:
        if self.event_sink is not None:
            self.event_sink(event)

    def _build_request(self, *, text: str, task_type: TaskType) -> TaskRequest:
        task_id = self.task_id_template.format(session_id=self.session_id, counter=self._task_counter)
        return (
            self.request_factory(text, task_type, task_id)
            if self.request_factory is not None
            else TaskRequest(
                task_id=task_id,
                description=text,
                task_type=task_type,
            )
        )

    def _emit_user_command(self, request: TaskRequest) -> None:
        self._emit(
            VoltronEvent(
                event_type="user_command",
                source="USER",
                message=request.description,
                payload={"task_type": request.task_type.value},
                task_id=request.task_id,
            )
        )

    def _emit_interactive_session_events(
        self,
        session: BrainPlanningSession,
        *,
        include_questions: bool,
    ) -> None:
        self._emit(
            VoltronEvent(
                event_type="brain_text_plan_draft",
                source="BRAIN",
                message="drafted interactive text plan",
                payload={
                    "session_id": session.session_id,
                    "status": session.status,
                    "text_plan": session.draft.to_dict(),
                },
                task_id=session.task_id,
            )
        )
        if include_questions:
            for item in session.dialogue:
                if item.get("type") == "question":
                    self._emit(
                        VoltronEvent(
                            event_type="brain_question",
                            source="BRAIN",
                            message=str(item.get("text") or "planning clarification requested"),
                            payload={
                                "session_id": session.session_id,
                                "question": dict(item),
                            },
                            task_id=session.task_id,
                        )
                    )
        if session.status == "awaiting_confirmation":
            self._emit_plan_confirmation_request(session)

    def _emit_plan_confirmation_request(self, session: BrainPlanningSession) -> None:
        self._emit(
            VoltronEvent(
                event_type="brain_plan_confirmation_requested",
                source="BRAIN",
                message="interactive text plan is ready for confirmation",
                payload={
                    "session_id": session.session_id,
                    "status": session.status,
                    "text_plan": session.draft.to_dict(),
                },
                task_id=session.task_id,
            )
        )

    def close(self) -> None:
        self._close_environment(self._latest_environment)
        self._latest_environment = None

    def _close_environment(self, environment: Any | None) -> None:
        if environment is None:
            return
        marker = id(environment)
        if marker in self._closed_environment_ids:
            return
        close = getattr(environment, "close", None)
        if not callable(close):
            return
        close()
        self._closed_environment_ids.add(marker)

    def _emit_brain_event(self, event: BrainPlanningEvent) -> None:
        self._emit(
            VoltronEvent(
                event_type=f"brain_{event.event_type}",
                source="BRAIN",
                message=event.message,
                payload=dict(event.payload),
            )
        )


def build_mock_voltron_session(
    *,
    event_sink: EventSink | None = None,
    planner: Any | None = None,
    radio_demo: bool = False,
    step_budget_per_subtask: int = 1,
) -> VoltronSession:
    memory = MockMemoryAdapter()
    vision = RadioDemoVisionAdapter() if radio_demo else MockVisionAdapter()
    policy = MockPolicyAdapter()
    projector = ActionProjection.from_embodiment("behavior_r1_pro")

    brain = BrainAgent(memory=memory, planner=planner or RuleBasedPlanner())
    vision_agent = VisionAgent(memory=memory, vision=vision)
    navigation_agent = NavigationAgent(memory=memory, policy=policy, projector=projector)
    action_agent = ActionAgent(memory=memory, policy=policy, projector=projector)
    orchestrator = ClosedLoopOrchestrator(
        brain_agent=brain,
        vision_agent=vision_agent,
        navigation_agent=navigation_agent,
        action_agent=action_agent,
        max_retries=0,
        max_control_steps_per_subtask=4,
    )
    return VoltronSession(
        orchestrator=orchestrator,
        environment_factory=lambda: MockRuntimeEnvironment(step_budget_per_subtask=step_budget_per_subtask),
        event_sink=event_sink,
        session_id="mock",
    )


def build_configured_voltron_session(
    args: Any,
    *,
    event_sink: EventSink | None = None,
) -> VoltronSession:
    """Build a session from the canonical closed-loop runtime args/config."""

    from voltron.entrypoints.examples.closed_loop import main as closed_loop_main

    runtime_runtime_builder = importlib.import_module("voltron.runtime.assembly.runtime_builder")

    hovsg_runtime = closed_loop_main.resolve_hovsg_runtime_config(
        env_id=args.env_id,
        hovsg_scene_map=args.hovsg_scene_map,
        hovsg_graph_root=args.hovsg_graph_root,
        hovsg_scene_id=args.hovsg_scene_id,
        hovsg_graph_path=args.hovsg_graph_path,
        hovsg_nav_graph_type=args.hovsg_nav_graph_type,
    )
    orchestrator = _build_orchestrator_from_args(args=args, hovsg_runtime=hovsg_runtime)

    def build_request(text: str, task_type: TaskType, task_id: str) -> TaskRequest:
        request_args = _copy_args_with_task(args, task_id=task_id, task_desc=text, task_type=task_type.value)
        return runtime_runtime_builder.build_task_request(
            args=request_args,
            scene_id=hovsg_runtime["scene_id"],
            hovsg_runtime=hovsg_runtime,
        )

    return VoltronSession(
        orchestrator=orchestrator,
        environment_factory=lambda: runtime_runtime_builder.build_behavior_environment(
            args=args,
            hovsg_runtime=hovsg_runtime,
        ),
        event_sink=event_sink,
        session_id=args.task_id,
        default_task_type=_parse_task_type(args.task_type),
        request_factory=build_request,
        task_id_template="{session_id}_{counter:03d}",
    )


def _copy_args_with_task(args: Any, *, task_id: str, task_desc: str, task_type: str) -> Any:
    copied = Namespace(**vars(args))
    copied.task_id = task_id
    copied.task_desc = task_desc
    copied.task_type = task_type
    return copied


def _build_orchestrator_from_args(
    *,
    args: Any,
    hovsg_runtime: dict[str, Any],
) -> ClosedLoopOrchestrator:
    from voltron.entrypoints.examples.closed_loop import main as closed_loop_main

    return closed_loop_main.build_orchestrator(
        embodiment=args.embodiment,
        gr00t_host=args.gr00t_host,
        gr00t_port=args.gr00t_port,
        vision_endpoint=args.vision_endpoint,
        vision_timeout_s=args.vision_timeout_s,
        vision_max_retries=args.vision_max_retries,
        vision_retry_backoff_s=args.vision_retry_backoff_s,
        vision_heartbeat_interval_steps=args.vision_heartbeat_interval_steps,
        memory_agent_endpoint=args.memory_agent_endpoint,
        use_memory_agent=args.memory_mode == "agent",
        max_retries=args.max_retries,
        max_control_steps_per_subtask=args.max_control_steps,
        planner_backend=args.brain_planner,
        brain_base_url=args.brain_base_url,
        brain_model=args.brain_model,
        brain_api_key=args.brain_api_key,
        brain_api_key_env=args.brain_api_key_env,
        brain_timeout_s=args.brain_timeout_s,
        brain_temperature=args.brain_temperature,
        brain_max_retries=args.brain_max_retries,
        brain_retry_backoff_s=args.brain_retry_backoff_s,
        action_selector=args.action_selector,
        action_max_unverified_internal_step_control_steps=args.action_max_unverified_internal_step_control_steps,
        action_internal_planning_enabled=args.action_internal_planning_enabled,
        action_internal_step_completion_use_vision_completion_monitor=(
            args.action_internal_step_completion_use_vision_completion_monitor
        ),
        action_internal_step_completion_require_verified_completion=(
            args.action_internal_step_completion_require_verified_completion
        ),
        action_base_url=args.action_base_url,
        action_model=args.action_model,
        action_api_key=args.action_api_key,
        action_api_key_env=args.action_api_key_env,
        action_timeout_s=args.action_timeout_s,
        action_temperature=args.action_temperature,
        action_max_retries=args.action_max_retries,
        action_retry_backoff_s=args.action_retry_backoff_s,
        navigation_backend=args.navigation_backend,
        navigation_base_url=args.navigation_base_url,
        navigation_model=args.navigation_model,
        navigation_api_key=args.navigation_api_key,
        navigation_api_key_env=args.navigation_api_key_env,
        navigation_timeout_s=args.navigation_timeout_s,
        navigation_temperature=args.navigation_temperature,
        navigation_max_retries=args.navigation_max_retries,
        navigation_retry_backoff_s=args.navigation_retry_backoff_s,
        hovsg_graph_root=hovsg_runtime["graph_root"],
        hovsg_scene_id=hovsg_runtime["scene_id"],
        hovsg_graph_path=hovsg_runtime["graph_path"],
        hovsg_nav_graph_type=hovsg_runtime["nav_graph_type"],
        hovsg_direct_room_transition_max_gap_m=args.hovsg_direct_room_transition_max_gap_m,
        hovsg_direct_room_transition_min_span_m=args.hovsg_direct_room_transition_min_span_m,
        hovsg_object_approach_min_portal_stance_clearance_m=(
            args.hovsg_object_approach_min_portal_stance_clearance_m
        ),
        nav2_version_profile=args.nav2_version_profile,
        nav2_action_name=args.nav2_action_name,
        nav2_planner_id=args.nav2_planner_id,
        nav2_frame_id=args.nav2_frame_id,
        nav2_timeout_s=args.nav2_timeout_s,
        nav2_strict=args.nav2_strict,
        nav2_trav_map_filename=args.nav2_trav_map_filename,
        nav2_portal_analysis_map_resolution=args.nav2_portal_analysis_map_resolution,
        nav2_portal_clearance_radius_m=args.nav2_portal_clearance_radius_m,
        nav2_portal_corridor_standoff_m=args.nav2_portal_corridor_standoff_m,
        nav2_portal_sampling_step_m=args.nav2_portal_sampling_step_m,
        nav2_local_path_clearance_radius_m=args.nav2_local_path_clearance_radius_m,
        nav2_local_path_waypoint_spacing_m=args.nav2_local_path_waypoint_spacing_m,
        navigation_prefer_forward_facing_motion=args.navigation_prefer_forward_facing_motion,
        navigation_portal_alignment_distance_threshold=args.navigation_portal_alignment_distance_threshold,
        navigation_portal_prealign_distance_threshold_m=args.navigation_portal_prealign_distance_threshold_m,
        navigation_portal_alignment_footprint_width_m=args.navigation_portal_alignment_footprint_width_m,
        navigation_portal_alignment_min_lateral_deadband_m=args.navigation_portal_alignment_min_lateral_deadband_m,
        navigation_portal_alignment_wide_clearance_margin_m=args.navigation_portal_alignment_wide_clearance_margin_m,
        navigation_max_linear_velocity=args.navigation_max_linear_velocity,
        navigation_linear_gain=args.navigation_linear_gain,
        navigation_local_path_linear_gain=args.navigation_local_path_linear_gain,
        navigation_local_path_max_linear_velocity=args.navigation_local_path_max_linear_velocity,
        navigation_portal_alignment_max_linear_velocity=args.navigation_portal_alignment_max_linear_velocity,
        navigation_max_angular_velocity=args.navigation_max_angular_velocity,
        navigation_local_path_angular_gain_scale=args.navigation_local_path_angular_gain_scale,
        policy_backend=args.policy_backend,
        pi05_endpoint=args.pi05_endpoint,
        pi05_timeout_s=args.pi05_timeout_s,
        pi05_task_id=args.pi05_task_id,
        log_navigation_candidates=getattr(args, "log_navigation_candidates", False),
        logging_nav2_path_snapshots=getattr(args, "logging_nav2_path_snapshots", False),
        runtime_termination_use_environment_success_signal=getattr(
            args,
            "runtime_termination_use_environment_success_signal",
            True,
        ),
        runtime_termination_use_brain_completion_signal=getattr(
            args,
            "runtime_termination_use_brain_completion_signal",
            True,
        ),
        runtime_termination_environment_signal_policy=getattr(
            args,
            "runtime_termination_environment_signal_policy",
            "allow_early_success",
        ),
        vision_completion_positive_streak=getattr(args, "vision_completion_positive_streak", 1),
        vision_completion_stability_steps=getattr(args, "vision_completion_stability_steps", 1),
        vision_completion_action_delta_threshold=getattr(args, "vision_completion_action_delta_threshold", 0.03),
        vision_completion_check_interval_steps=getattr(args, "vision_completion_check_interval_steps", 200),
        vision_completion_agent_scope=getattr(args, "vision_completion_agent_scope", ["ACTION"]),
        vision_completion_include_third_person=getattr(
            args,
            "vision_completion_include_third_person",
            True,
        ),
        vision_completion_max_images=getattr(args, "vision_completion_max_images", 4),
        vision_completion_max_image_side_px=getattr(args, "vision_completion_max_image_side_px", 1024),
        vision_completion_jpeg_quality=getattr(args, "vision_completion_jpeg_quality", 90),
        vision_completion_max_image_b64_chars=getattr(args, "vision_completion_max_image_b64_chars", 900_000),
        vision_completion_image_detail=getattr(args, "vision_completion_image_detail", "high"),
        logging_verbose=getattr(args, "logging_verbose", True),
    )


class RadioDemoVisionAdapter:
    """Vision fixture that lets the radio interaction closed loop complete."""

    def __init__(self) -> None:
        self.calls = 0

    def analyze(
        self,
        images_b64: list[str],
        instruction: str,
        task_name: str,
        image_view_order: list[str] | None = None,
    ) -> PerceptionReport:
        del images_b64, instruction, task_name, image_view_order
        self.calls += 1
        if self.calls == 1:
            return PerceptionReport(
                objects=[
                    PerceptionObject(name="radio", confidence=0.99),
                    PerceptionObject(name="power_button", confidence=0.98),
                ],
                task_complete=False,
                raw_text="radio is visible and within reach",
                metadata={
                    "raw_response": {
                        "scene_report": {
                            "target_visible": True,
                            "target_part_visible": True,
                            "target_part_name": "power button",
                        }
                    }
                },
            )
        return PerceptionReport(
            objects=[PerceptionObject(name="radio", confidence=0.99)],
            task_complete=True,
            raw_text="radio is on",
            metadata={
                "raw_response": {
                    "scene_report": {
                        "target_visible": True,
                        "target_part_visible": True,
                        "target_part_name": "power button",
                    }
                }
            },
        )


def _parse_task_type(value: str | TaskType) -> TaskType:
    if isinstance(value, TaskType):
        return value
    normalized = str(value).strip().lower()
    for task_type in TaskType:
        if task_type.value == normalized:
            return task_type
    raise ValueError(f"Unsupported task type {value!r}")


__all__ = [
    "RadioDemoVisionAdapter",
    "VoltronEvent",
    "VoltronSession",
    "build_configured_voltron_session",
    "build_mock_voltron_session",
]
