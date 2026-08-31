from __future__ import annotations

import re
import time
from typing import Any

from voltron.agents.action.tools.action_projection import ActionProjection
from voltron.agents.navigation.body.object_approach_selection import (
    HeuristicNavigationApproachPointSelector,
)
from voltron.agents.navigation.body.skill_routing import HeuristicNavigationSkillSelector
from voltron.agents.navigation.tools.navigation_bridge import GoalConditionedNavigationBridge
from voltron.agents.navigation.tools.runtime import execution_context as runtime_execution_context
from voltron.agents.navigation.tools.runtime import execution_flow as runtime_execution_flow
from voltron.agents.navigation.tools.runtime import object_approach as runtime_object_approach
from voltron.agents.navigation.tools.runtime import observation as runtime_observation
from voltron.agents.navigation.tools.runtime import skill_routing as runtime_skill_routing
from voltron.shared.enums import AgentStatus
from voltron.shared.context import ExecutionContext, LocalSkillSelection, Subtask
from voltron.shared.results import AgentResult
from voltron.agents.navigation.skills import NavigationSkillRegistry
from voltron.shared.contracts import MemoryAdapter, NavigatorBackend, PolicyAdapter


_AFFORDANCE_ALIASES: dict[str, set[str]] = {
    "switch": {"switch", "control", "toggle"},
    "button": {"button", "control", "pushbutton"},
    "knob": {"knob", "dial", "control"},
    "dial": {"dial", "knob", "control"},
    "handle": {"handle", "grip"},
    "lever": {"lever", "control"},
    "control": {"control", "switch", "button", "knob", "dial", "lever", "handle", "toggle"},
}


_FOLLOWUP_AFFORDANCE_TEXT_KEYS = (
    "task",
    "instruction",
    "objective",
    "intent",
    "purpose",
    "action",
    "intended_action",
    "next_action",
    "description",
    "free-form",
    "free_form",
    "freeform",
)


def _navigation_backend_error_code(exc: Exception) -> str:
    error_text = f"{type(exc).__name__}: {exc}".lower()
    if "429" in error_text or "too many requests" in error_text or "rate limit" in error_text:
        return "NAV_BACKEND_RATE_LIMITED"
    if "timeout" in error_text or "timed out" in error_text:
        return "NAV_BACKEND_TIMEOUT"
    if "connection" in error_text or "dns" in error_text or "name resolution" in error_text:
        return "NAV_BACKEND_CONNECTION_ERROR"
    if "http" in error_text or "client error" in error_text or "server error" in error_text:
        return "NAV_BACKEND_HTTP_ERROR"
    return "NAV_BACKEND_ERROR"


def _navigation_backend_error_payload(exc: Exception, *, stage: str) -> dict[str, str]:
    return {
        "message": str(exc),
        "error_type": type(exc).__name__,
        "error_stage": stage,
    }


def _normalize_grounding_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split()).strip()


def _text_contains_term(text: str, term: str) -> bool:
    normalized = _normalize_grounding_text(text)
    if not normalized:
        return False
    tokens = set(normalized.split())
    compact = re.sub(r"[^a-z0-9]+", "", normalized)
    compact_term = re.sub(r"[^a-z0-9]+", "", term.lower())
    return term in tokens or bool(compact_term and compact_term in compact)


def _iter_candidate_semantic_text(candidate: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for key in (
        "object_name",
        "name",
        "category",
        "object_category",
        "semantic_class",
        "class_name",
        "label",
        "description",
    ):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())
    return texts


def _iter_grounding_target_texts(
    *,
    instruction: str,
    nav_context: dict[str, Any],
    grounded_goal: dict[str, Any],
) -> list[str]:
    del instruction
    texts: list[str] = []
    for value in (
        grounded_goal.get("grounding_query"),
        grounded_goal.get("object_name"),
    ):
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())
    for goal in (nav_context.get("interpreted_goal"), grounded_goal.get("interpreted_goal")):
        if not isinstance(goal, dict):
            continue
        target_query = goal.get("target_query")
        if isinstance(target_query, dict):
            for key in ("object", "part", "target", "item"):
                value = target_query.get(key)
                if isinstance(value, str) and value.strip():
                    texts.append(value.strip())
        followup_context = goal.get("followup_context")
        if isinstance(followup_context, dict):
            for key in ("target", "object", "part", "item"):
                value = followup_context.get(key)
                if isinstance(value, str) and value.strip():
                    texts.append(value.strip())
            for key in _FOLLOWUP_AFFORDANCE_TEXT_KEYS:
                value = followup_context.get(key)
                if isinstance(value, str) and value.strip():
                    texts.append(value.strip())
    return texts


def _required_affordance_terms(
    *,
    instruction: str,
    nav_context: dict[str, Any],
    grounded_goal: dict[str, Any],
) -> list[str]:
    required: list[str] = []
    for text in _iter_grounding_target_texts(
        instruction=instruction,
        nav_context=nav_context,
        grounded_goal=grounded_goal,
    ):
        for term in _AFFORDANCE_ALIASES:
            if _text_contains_term(text, term) and term not in required:
                required.append(term)
    return required


def _candidate_satisfies_affordance(candidate: dict[str, Any], required_terms: list[str]) -> bool:
    if not required_terms:
        return True
    candidate_text = " ".join(_iter_candidate_semantic_text(candidate))
    if not candidate_text:
        return False
    for term in required_terms:
        aliases = _AFFORDANCE_ALIASES.get(term, {term})
        if any(_text_contains_term(candidate_text, alias) for alias in aliases):
            return True
    return False


class NavigationAgent:
    def __init__(
        self,
        memory: MemoryAdapter,
        policy: PolicyAdapter,
        projector: ActionProjection,
        navigator: NavigatorBackend | None = None,
        execution_bridge: GoalConditionedNavigationBridge | None = None,
        selector: Any | None = None,
        skill_registry: Any | None = None,
        approach_point_selector: Any | None = None,
        goal_interpreter: Any | None = None,
    ):
        self.memory = memory
        self.policy = policy
        self.projector = projector
        self.navigator = navigator
        self.execution_bridge = execution_bridge or GoalConditionedNavigationBridge()
        self.selector = selector or HeuristicNavigationSkillSelector()
        self.skill_registry = skill_registry or NavigationSkillRegistry.build_default(memory=memory)
        self.approach_point_selector = (
            approach_point_selector or HeuristicNavigationApproachPointSelector()
        )
        self.goal_interpreter = goal_interpreter
        self._map_state: dict[str, Any] = {}
        self._last_path_plan: dict[str, Any] | None = None
        self._route_localization_override: dict[str, Any] | None = None

    def execute(self, subtask: Subtask, context: ExecutionContext) -> AgentResult:
        start = time.time()

        observation = subtask.parameters.get("observation")
        if not isinstance(observation, dict):
            return AgentResult(
                subtask_id=subtask.subtask_id,
                status=AgentStatus.FAILURE,
                error_code="NAV_OBSERVATION_MISSING",
                result={"message": "subtask.parameters['observation'] is required"},
                latency_ms=self._latency_ms(start),
            )

        target_region = str(subtask.target.get("region", ""))
        if target_region:
            _ = self.memory.find_object(target_region, top_k=1)

        policy_options = subtask.parameters.get("policy_options")
        grounded_goal: dict[str, Any] | None = None
        path_plan: dict[str, Any] | None = None
        backend_state: dict[str, Any] | None = None
        interpreted_goal: dict[str, Any] | None = None
        navigation_grounding_context: dict[str, Any] | None = None
        navigation_skill_selection: dict[str, Any] | None = None
        prepared_navigation_payload: dict[str, Any] | None = None
        object_approach_selection: dict[str, Any] | None = None
        selected_object_approach: dict[str, Any] | None = None
        runtime_inputs = runtime_execution_flow.collect_runtime_inputs(
            subtask=subtask,
            context=context,
            observation=observation,
        )
        scene_id = runtime_inputs["scene_id"]
        current_region = runtime_inputs["current_region"]
        pose = runtime_inputs["pose"]
        orientation = runtime_inputs["orientation"]

        if self.navigator is not None:
            try:
                nav_context = self._build_navigation_context(
                    subtask=subtask,
                    context=context,
                    observation=observation,
                    scene_id=scene_id,
                )
                previous_map_state = dict(self._map_state)
                backend_state = self.navigator.update(
                    observation,
                    pose=pose,
                )
                backend_state = self._stabilize_route_localization(
                    backend_state=backend_state,
                    previous_map_state=previous_map_state,
                )
                self._merge_map_state(backend_state)
                self._refresh_navigation_context(nav_context, backend_state=backend_state)
                start_state = self._build_start_state(
                    subtask=subtask,
                    context=context,
                    observation=observation,
                    backend_state=backend_state,
                )
                start_state, backend_state = (
                    self._apply_route_localization_override_to_replan_start(
                        start_state=start_state,
                        backend_state=backend_state,
                    )
                )
                if (
                    isinstance(backend_state, dict)
                    and backend_state.get("localization_guard") is not None
                ):
                    self._merge_map_state(backend_state)
                    self._refresh_navigation_context(nav_context, backend_state=backend_state)
                backend_bundle = self._resolve_grounded_goal_bundle(
                    navigator=self.navigator,
                    subtask=subtask,
                    context=context,
                    scene_id=scene_id,
                    start_state=start_state,
                    nav_context=nav_context,
                    observation=observation,
                )
                grounded_goal = backend_bundle["grounded_goal"]
                path_plan = backend_bundle["path_plan"]
                if isinstance(path_plan, dict):
                    self._last_path_plan = dict(path_plan)
                if self._path_plan_unavailable(path_plan):
                    return self._path_unavailable_result(
                        subtask=subtask,
                        path_plan=path_plan,
                        start=start,
                    )
                interpreted_goal = backend_bundle["interpreted_goal"]
                navigation_grounding_context = backend_bundle["navigation_grounding_context"]
                navigation_skill_selection = backend_bundle["navigation_skill_selection"]
                prepared_navigation_payload = backend_bundle["prepared_navigation_payload"]
                object_approach_selection = backend_bundle["object_approach_selection"]
                selected_object_approach = backend_bundle["selected_object_approach"]
                self._sync_scene_map(
                    scene_id=scene_id,
                    backend_state=backend_state,
                    grounded_goal=grounded_goal,
                    path_plan=path_plan,
                )
                policy_options = self.execution_bridge.build_policy_options(
                    existing_options=policy_options,
                    grounded_goal=grounded_goal,
                    path_plan=path_plan,
                )
                current_region = (
                    runtime_observation.extract_runtime_region(backend_state=backend_state)
                    or current_region
                )
            except Exception as exc:
                return AgentResult(
                    subtask_id=subtask.subtask_id,
                    status=AgentStatus.FAILURE,
                    error_code=_navigation_backend_error_code(exc),
                    result=_navigation_backend_error_payload(exc, stage="prepare_navigation_plan"),
                    latency_ms=self._latency_ms(start),
                )

        policy_options = runtime_observation.merge_runtime_navigation_options(
            subtask=subtask,
            existing_options=policy_options,
        )
        policy_observation = runtime_observation.build_policy_observation(
            observation=observation,
            scene_id=scene_id,
            pose=pose,
            orientation=orientation,
            current_region=current_region,
        )

        try:
            action, info = self.policy.get_action(policy_observation, options=policy_options)
            nav_action = self.projector.project_navigation(action)
            self.projector.update_last_safe_action(nav_action)

            nav_stats = self.memory.record_navigation_update(
                runtime_execution_flow.build_memory_update_payload(
                    scene_id=scene_id,
                    current_region=current_region,
                    target_region=target_region,
                    pose=pose,
                    orientation=orientation,
                    nav_feedback=runtime_observation.extract_nav_feedback(
                        subtask=subtask, observation=observation
                    ),
                    obstacles=subtask.parameters.get("obstacles", []),
                    policy_info=info,
                    grounded_goal=grounded_goal,
                    path_plan=path_plan,
                    navigation_skill_selection=navigation_skill_selection,
                    prepared_navigation_payload=prepared_navigation_payload,
                    object_approach_selection=object_approach_selection,
                    selected_object_approach=selected_object_approach,
                    navigator_backend_name=type(self.navigator).__name__
                    if self.navigator is not None
                    else None,
                )
            )
        except Exception as exc:
            return AgentResult(
                subtask_id=subtask.subtask_id,
                status=AgentStatus.FAILURE,
                error_code="NAV_BLOCKED",
                result={"message": str(exc)},
                latency_ms=self._latency_ms(start),
            )

        result_payload, runtime_artifacts = runtime_execution_flow.build_success_payloads(
            action=action,
            projected_action=nav_action,
            policy_info=info,
            memory_update=nav_stats,
            navigator_backend_name=type(self.navigator).__name__
            if self.navigator is not None
            else None,
            grounded_goal=grounded_goal,
            scene_id=scene_id,
            path_plan=path_plan,
            interpreted_goal=interpreted_goal,
            navigation_grounding_context=navigation_grounding_context,
            execution_bridge_artifacts=self.execution_bridge.build_runtime_artifacts(
                grounded_goal=grounded_goal,
                path_plan=path_plan,
                backend_state=backend_state,
            ),
            navigation_skill_selection=navigation_skill_selection,
            prepared_navigation_payload=prepared_navigation_payload,
            object_approach_selection=object_approach_selection,
            selected_object_approach=selected_object_approach,
            policy_runtime_artifacts=runtime_observation.extract_policy_runtime_artifacts(info),
        )
        return AgentResult(
            subtask_id=subtask.subtask_id,
            status=AgentStatus.SUCCESS,
            result=result_payload,
            runtime_artifacts=runtime_artifacts,
            latency_ms=self._latency_ms(start),
        )

    def run_episode(
        self, *, subtask: Subtask, context: ExecutionContext, runtime: Any
    ) -> AgentResult:
        static_parameters = dict(subtask.parameters)
        episode_state: dict[str, Any] | None = None
        last_result: AgentResult | None = None

        for control_step in range(1, int(runtime.max_control_steps) + 1):
            runtime.prepare_control_step(
                subtask=subtask,
                context=context,
                static_parameters=static_parameters,
                control_step=control_step,
            )
            if episode_state is not None:
                self._merge_episode_policy_runtime_state(
                    subtask=subtask, episode_state=episode_state
                )

            if episode_state is None or self._episode_requires_replan(
                subtask, episode_state=episode_state
            ):
                result = self.execute(subtask, context)
            else:
                result = self._execute_navigation_policy_step(
                    subtask=subtask,
                    context=context,
                    episode_state=episode_state,
                )
            if result.status == AgentStatus.SUCCESS:
                episode_state = self._navigation_episode_state_from_result(
                    result, previous_state=episode_state
                )

            last_result = runtime.publish_agent_result(
                subtask=subtask,
                context=context,
                result=result,
                control_step=control_step,
            )
            if result.status == AgentStatus.FAILURE:
                if hasattr(runtime, "record_agent_failure"):
                    runtime.record_agent_failure(
                        subtask=subtask,
                        context=context,
                        result=result,
                        failure_reason=result.error_code or "AGENT_FAILURE",
                    )
                return result

            step_outcome = runtime.apply_agent_result(
                subtask=subtask, result=result, context=context
            )
            if getattr(step_outcome, "feedback", None):
                runtime.update_feedback(
                    subtask=subtask,
                    context=context,
                    result=result,
                    control_step=control_step,
                    feedback=step_outcome.feedback,
                )
            if not bool(getattr(step_outcome, "done", False)):
                continue

            if getattr(step_outcome, "success", None) is False:
                return runtime.environment_failure_result(
                    subtask=subtask,
                    context=context,
                    result=result,
                    control_step=control_step,
                    feedback=getattr(step_outcome, "feedback", {}),
                    failure_reason=getattr(step_outcome, "failure_reason", None),
                )

            if hasattr(runtime, "record_agent_success"):
                runtime.record_agent_success(subtask=subtask, context=context, result=result)
            return result

        if last_result is not None and hasattr(runtime, "record_agent_failure"):
            runtime.record_agent_failure(
                subtask=subtask,
                context=context,
                result=last_result,
                failure_reason="SUBTASK_TIMEOUT",
            )
        return runtime.timeout_result(subtask=subtask)

    @staticmethod
    def _latency_ms(start: float) -> int:
        return int((time.time() - start) * 1000)

    @staticmethod
    def _episode_requires_replan(
        subtask: Subtask, episode_state: dict[str, Any] | None = None
    ) -> bool:
        policy_runtime_state = (
            episode_state.get("policy_runtime_state") if isinstance(episode_state, dict) else None
        )
        if isinstance(policy_runtime_state, dict) and bool(
            policy_runtime_state.get("requires_replan")
            or policy_runtime_state.get("local_segment_complete")
        ):
            return True
        observation = subtask.parameters.get("observation")
        if isinstance(observation, dict):
            nav_feedback = observation.get("nav_feedback")
            if isinstance(nav_feedback, dict) and bool(
                nav_feedback.get("stuck")
                or nav_feedback.get("collision")
                or nav_feedback.get("path_invalid")
            ):
                return True
            grounded_goal = (
                episode_state.get("grounded_goal") if isinstance(episode_state, dict) else None
            )
            if (
                isinstance(episode_state, dict)
                and runtime_object_approach.should_use_object_approach_flow(
                    subtask=subtask,
                    grounded_goal=grounded_goal,
                )
                and runtime_object_approach.should_replan_cached_object_approach(
                    subtask=subtask,
                    observation=observation,
                )
            ):
                return True
        grounded_goal = (
            episode_state.get("grounded_goal") if isinstance(episode_state, dict) else None
        )
        if (
            isinstance(episode_state, dict)
            and runtime_object_approach.should_use_object_approach_flow(
                subtask=subtask,
                grounded_goal=grounded_goal,
            )
            and runtime_object_approach.cached_dynamic_local_segment_completed(
                subtask=subtask,
                path_plan=episode_state.get("path_plan"),
            )
        ):
            return True
        return False

    @staticmethod
    def _path_plan_unavailable(path_plan: dict[str, Any] | None) -> bool:
        if not isinstance(path_plan, dict) or not path_plan:
            return False
        waypoints = path_plan.get("waypoints")
        has_waypoints = isinstance(waypoints, list) and bool(waypoints)
        if path_plan.get("found") is False and not has_waypoints:
            return True
        path_backend = str(path_plan.get("path_backend") or "").strip().lower()
        nav2_error = str(path_plan.get("nav2_error") or "").strip().lower()
        return (
            not has_waypoints
            and path_backend == "fallback_blocked_for_clearance"
            and bool(nav2_error)
        )

    def _path_unavailable_result(
        self,
        *,
        subtask: Subtask,
        path_plan: dict[str, Any],
        start: float,
    ) -> AgentResult:
        nav2_error = str(path_plan.get("nav2_error") or path_plan.get("path_backend") or "no_path")
        return AgentResult(
            subtask_id=subtask.subtask_id,
            status=AgentStatus.FAILURE,
            error_code="NAV_PATH_UNAVAILABLE",
            result={
                "message": f"navigation backend returned no executable path: {nav2_error}",
                "path_plan": dict(path_plan),
            },
            runtime_artifacts={"path_plan": dict(path_plan)},
            latency_ms=self._latency_ms(start),
        )

    @staticmethod
    def _navigation_episode_state_from_result(
        result: AgentResult,
        previous_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        previous = dict(previous_state or {})
        policy_runtime_state: dict[str, Any] = {}
        policy_runtime_state.update(
            runtime_observation.extract_policy_runtime_artifacts(
                result.runtime_artifacts.get("policy_info")
            )
        )
        policy_runtime_state.update(
            runtime_observation.extract_policy_runtime_artifacts(result.runtime_artifacts)
        )
        return {
            "grounded_goal": dict(
                result.runtime_artifacts.get("grounded_goal")
                or result.runtime_artifacts.get("nav_goal")
                or result.result.get("grounded_goal")
                or previous.get("grounded_goal")
                or {}
            ),
            "path_plan": dict(
                result.runtime_artifacts.get("path_plan")
                or result.result.get("path_plan")
                or previous.get("path_plan")
                or {}
            ),
            "navigation_skill_selection": result.runtime_artifacts.get("navigation_skill_selection")
            or previous.get("navigation_skill_selection"),
            "interpreted_goal": result.runtime_artifacts.get("interpreted_goal")
            or previous.get("interpreted_goal"),
            "navigation_grounding_context": result.runtime_artifacts.get(
                "navigation_grounding_context"
            )
            or previous.get("navigation_grounding_context"),
            "prepared_navigation_payload": result.runtime_artifacts.get(
                "prepared_navigation_payload"
            )
            or previous.get("prepared_navigation_payload"),
            "object_approach_selection": result.runtime_artifacts.get("object_approach_selection")
            or previous.get("object_approach_selection"),
            "selected_object_approach": result.runtime_artifacts.get("selected_object_approach")
            or previous.get("selected_object_approach"),
            "policy_runtime_state": policy_runtime_state,
        }

    @staticmethod
    def _merge_episode_policy_runtime_state(
        *, subtask: Subtask, episode_state: dict[str, Any]
    ) -> None:
        policy_runtime_state = episode_state.get("policy_runtime_state")
        if not isinstance(policy_runtime_state, dict) or not policy_runtime_state:
            return
        subtask.parameters = {**subtask.parameters, **policy_runtime_state}

    def _execute_navigation_policy_step(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        episode_state: dict[str, Any],
    ) -> AgentResult:
        start = time.time()
        observation = subtask.parameters.get("observation")
        if not isinstance(observation, dict):
            return AgentResult(
                subtask_id=subtask.subtask_id,
                status=AgentStatus.FAILURE,
                error_code="NAV_OBSERVATION_MISSING",
                result={"message": "subtask.parameters['observation'] is required"},
                latency_ms=self._latency_ms(start),
            )

        runtime_inputs = runtime_execution_flow.collect_runtime_inputs(
            subtask=subtask,
            context=context,
            observation=observation,
        )
        scene_id = runtime_inputs["scene_id"]
        current_region = runtime_inputs["current_region"]
        pose = runtime_inputs["pose"]
        orientation = runtime_inputs["orientation"]

        backend_state: dict[str, Any] | None = None
        if self.navigator is not None:
            try:
                previous_map_state = dict(self._map_state)
                backend_state = self.navigator.update(observation, pose=pose)
                backend_state = self._stabilize_route_localization(
                    backend_state=backend_state,
                    previous_map_state=previous_map_state,
                )
                self._merge_map_state(backend_state)
                current_region = (
                    runtime_observation.extract_runtime_region(backend_state=backend_state)
                    or current_region
                )
            except Exception as exc:
                return AgentResult(
                    subtask_id=subtask.subtask_id,
                    status=AgentStatus.FAILURE,
                    error_code=_navigation_backend_error_code(exc),
                    result=_navigation_backend_error_payload(exc, stage="update_navigation_state"),
                    latency_ms=self._latency_ms(start),
                )

        grounded_goal = dict(episode_state.get("grounded_goal") or {})
        path_plan = dict(episode_state.get("path_plan") or {})
        if self._path_plan_unavailable(path_plan):
            return self._path_unavailable_result(
                subtask=subtask,
                path_plan=path_plan,
                start=start,
            )
        policy_options = self.execution_bridge.build_policy_options(
            existing_options=subtask.parameters.get("policy_options"),
            grounded_goal=grounded_goal,
            path_plan=path_plan,
        )
        policy_options = runtime_observation.merge_runtime_navigation_options(
            subtask=subtask,
            existing_options=policy_options,
        )
        policy_observation = runtime_observation.build_policy_observation(
            observation=observation,
            scene_id=scene_id,
            pose=pose,
            orientation=orientation,
            current_region=current_region,
        )

        try:
            action, info = self.policy.get_action(policy_observation, options=policy_options)
            nav_action = self.projector.project_navigation(action)
            self.projector.update_last_safe_action(nav_action)
            nav_stats = self.memory.record_navigation_update(
                runtime_execution_flow.build_memory_update_payload(
                    scene_id=scene_id,
                    current_region=current_region,
                    target_region=str(subtask.target.get("region", "")),
                    pose=pose,
                    orientation=orientation,
                    nav_feedback=runtime_observation.extract_nav_feedback(
                        subtask=subtask, observation=observation
                    ),
                    obstacles=subtask.parameters.get("obstacles", []),
                    policy_info=info,
                    grounded_goal=grounded_goal,
                    path_plan=path_plan,
                    navigation_skill_selection=episode_state.get("navigation_skill_selection"),
                    prepared_navigation_payload=episode_state.get("prepared_navigation_payload"),
                    object_approach_selection=episode_state.get("object_approach_selection"),
                    selected_object_approach=episode_state.get("selected_object_approach"),
                    navigator_backend_name=type(self.navigator).__name__
                    if self.navigator is not None
                    else None,
                )
            )
        except Exception as exc:
            return AgentResult(
                subtask_id=subtask.subtask_id,
                status=AgentStatus.FAILURE,
                error_code="NAV_BLOCKED",
                result={"message": str(exc)},
                latency_ms=self._latency_ms(start),
            )

        result_payload, runtime_artifacts = runtime_execution_flow.build_success_payloads(
            action=action,
            projected_action=nav_action,
            policy_info=info,
            memory_update=nav_stats,
            navigator_backend_name=type(self.navigator).__name__
            if self.navigator is not None
            else None,
            grounded_goal=grounded_goal,
            scene_id=scene_id,
            path_plan=path_plan,
            interpreted_goal=episode_state.get("interpreted_goal"),
            navigation_grounding_context=episode_state.get("navigation_grounding_context"),
            execution_bridge_artifacts=self.execution_bridge.build_runtime_artifacts(
                grounded_goal=grounded_goal,
                path_plan=path_plan,
                backend_state=backend_state,
            ),
            navigation_skill_selection=episode_state.get("navigation_skill_selection"),
            prepared_navigation_payload=episode_state.get("prepared_navigation_payload"),
            object_approach_selection=episode_state.get("object_approach_selection"),
            selected_object_approach=episode_state.get("selected_object_approach"),
            policy_runtime_artifacts=runtime_observation.extract_policy_runtime_artifacts(info),
        )
        return AgentResult(
            subtask_id=subtask.subtask_id,
            status=AgentStatus.SUCCESS,
            result=result_payload,
            runtime_artifacts=runtime_artifacts,
            latency_ms=self._latency_ms(start),
        )

    def _build_navigation_context(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        observation: dict[str, Any],
        scene_id: str | None,
    ) -> dict[str, Any]:
        return runtime_execution_context.build_navigation_context(
            memory=self.memory,
            subtask=subtask,
            context=context,
            observation=observation,
            scene_id=scene_id,
            map_state=self._map_state,
        )

    def _build_start_state(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        observation: dict[str, Any],
        backend_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return runtime_execution_context.build_start_state(
            subtask=subtask,
            context=context,
            observation=observation,
            backend_state=backend_state,
        )

    def _merge_map_state(self, backend_state: dict[str, Any] | None) -> None:
        self._map_state = runtime_execution_context.merge_map_state(self._map_state, backend_state)

    def _stabilize_route_localization(
        self,
        *,
        backend_state: dict[str, Any] | None,
        previous_map_state: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not isinstance(backend_state, dict) or not isinstance(previous_map_state, dict):
            return backend_state
        path_plan = self._last_path_plan or previous_map_state.get("last_path_plan")
        if not isinstance(path_plan, dict):
            return backend_state

        current_room = self._normalized_room_label(
            backend_state.get("current_room") or backend_state.get("current_region")
        )
        previous_room = self._normalized_room_label(
            previous_map_state.get("current_room") or previous_map_state.get("current_region")
        )
        if not current_room:
            return backend_state

        pose = backend_state.get("pose")
        distance = self._distance_to_route_path(pose=pose, path_plan=path_plan)
        if distance is None or distance > 0.75:
            self._route_localization_override = None
            return backend_state

        active_segment_rooms = self._active_segment_room_labels(
            path_plan=path_plan,
            backend_state=backend_state,
            previous_map_state=previous_map_state,
        )
        override_state = self._apply_route_localization_override(
            backend_state=backend_state,
            current_room=current_room,
            active_segment_rooms=active_segment_rooms,
            distance=distance,
        )
        if override_state is not None:
            return override_state

        if not previous_room or current_room == previous_room:
            return backend_state

        if (
            active_segment_rooms
            and previous_room in active_segment_rooms
            and current_room not in active_segment_rooms
        ):
            return self._localization_guarded_state(
                backend_state=backend_state,
                previous_map_state=previous_map_state,
                reason="active_segment_room_mismatch_near_path",
                distance=distance,
            )

        route_rooms = self._route_room_labels(path_plan)
        if current_room in route_rooms or previous_room not in route_rooms:
            return backend_state

        return self._localization_guarded_state(
            backend_state=backend_state,
            previous_map_state=previous_map_state,
            reason="off_route_room_near_active_path",
            distance=distance,
        )

    def _apply_route_localization_override(
        self,
        *,
        backend_state: dict[str, Any],
        current_room: str,
        active_segment_rooms: set[str],
        distance: float,
    ) -> dict[str, Any] | None:
        override = self._route_localization_override
        if not isinstance(override, dict):
            return None
        rejected_room = self._normalized_room_label(override.get("rejected_room"))
        kept_room = self._normalized_room_label(
            override.get("current_room") or override.get("current_region")
        )
        if not rejected_room:
            return None
        if rejected_room in active_segment_rooms:
            self._route_localization_override = None
            return None
        if current_room != rejected_room:
            return None
        if not active_segment_rooms or not kept_room or kept_room not in active_segment_rooms:
            return None
        return self._localization_guarded_state(
            backend_state=backend_state,
            previous_map_state=override,
            reason=str(override.get("reason") or "active_segment_room_mismatch_near_path"),
            distance=distance,
        )

    def _apply_route_localization_override_to_replan_start(
        self,
        *,
        start_state: dict[str, Any],
        backend_state: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        override = self._route_localization_override
        if not isinstance(override, dict):
            return start_state, backend_state
        current_room = self._normalized_room_label(
            start_state.get("current_room") or start_state.get("current_region")
        )
        rejected_room = self._normalized_room_label(override.get("rejected_room"))
        if not current_room or not rejected_room or current_room != rejected_room:
            return start_state, backend_state

        stabilized_start = dict(start_state)
        stabilized_backend = dict(backend_state or {})
        for target in (stabilized_start, stabilized_backend):
            for key in ("current_room", "current_region", "room_id", "floor_id"):
                if override.get(key) is not None:
                    target[key] = override[key]
            target["localization_guard"] = {
                "reason": str(override.get("reason") or "active_segment_room_mismatch_near_path"),
                "rejected_room": override.get("rejected_room"),
                "kept_room": override.get("current_room") or override.get("current_region"),
                "source": "replan_start_override",
            }
        return stabilized_start, stabilized_backend

    def _localization_guarded_state(
        self,
        *,
        backend_state: dict[str, Any],
        previous_map_state: dict[str, Any],
        reason: str,
        distance: float,
    ) -> dict[str, Any]:
        stabilized = dict(backend_state)
        for key in ("current_room", "current_region", "room_id", "floor_id"):
            if previous_map_state.get(key) is not None:
                stabilized[key] = previous_map_state[key]
        stabilized["localization_guard"] = {
            "reason": reason,
            "rejected_room": backend_state.get("current_room")
            or backend_state.get("current_region"),
            "kept_room": previous_map_state.get("current_room")
            or previous_map_state.get("current_region"),
            "distance_to_route_m": round(float(distance), 3),
        }
        self._route_localization_override = {
            key: stabilized[key]
            for key in ("current_room", "current_region", "room_id", "floor_id")
            if key in stabilized
        }
        self._route_localization_override.update(
            {
                "reason": reason,
                "rejected_room": backend_state.get("current_room")
                or backend_state.get("current_region"),
            }
        )
        return stabilized

    @staticmethod
    def _normalized_room_label(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = " ".join(value.lower().replace("_", " ").split()).strip()
        return normalized or None

    @classmethod
    def _route_room_labels(cls, path_plan: dict[str, Any]) -> set[str]:
        labels: set[str] = set()

        def add_label(value: Any) -> None:
            normalized = cls._normalized_room_label(value)
            if normalized:
                labels.add(normalized)

        def visit_waypoint(waypoint: Any) -> None:
            if not isinstance(waypoint, dict):
                return
            for key in ("room_name", "current_room", "current_region", "source_room_name"):
                add_label(waypoint.get(key))
            transition_anchor = waypoint.get("transition_anchor")
            if isinstance(transition_anchor, dict):
                visit_waypoint(transition_anchor)

        for key in (
            "local_goal",
            "execution_goal",
            "nav2_compute_goal",
            "transition_anchor",
            "goal",
        ):
            visit_waypoint(path_plan.get(key))
        for key in ("waypoints", "global_waypoints", "dense_waypoints"):
            values = path_plan.get(key)
            if isinstance(values, list):
                for waypoint in values:
                    visit_waypoint(waypoint)
        global_plan = path_plan.get("global_plan")
        if isinstance(global_plan, dict):
            for key in ("room_sequence", "graph_room_sequence"):
                values = global_plan.get(key)
                if isinstance(values, list):
                    for value in values:
                        add_label(value)
            for key in ("waypoints", "dense_waypoints"):
                values = global_plan.get(key)
                if isinstance(values, list):
                    for waypoint in values:
                        visit_waypoint(waypoint)
        return labels

    @classmethod
    def _active_segment_room_labels(
        cls,
        *,
        path_plan: dict[str, Any],
        backend_state: dict[str, Any],
        previous_map_state: dict[str, Any],
    ) -> set[str]:
        labels: set[str] = set()

        def add_label(value: Any) -> None:
            normalized = cls._normalized_room_label(value)
            if normalized:
                labels.add(normalized)

        def visit_waypoint(waypoint: Any) -> None:
            if not isinstance(waypoint, dict):
                return
            for key in ("room_name", "current_room", "current_region", "source_room_name"):
                add_label(waypoint.get(key))
            transition_anchor = waypoint.get("transition_anchor")
            if isinstance(transition_anchor, dict):
                visit_waypoint(transition_anchor)

        waypoints = path_plan.get("waypoints")
        if not isinstance(waypoints, list) or not waypoints:
            return labels

        active_index = cls._first_waypoint_index(
            backend_state.get("active_waypoint_index"),
            previous_map_state.get("active_waypoint_index"),
            path_plan.get("active_waypoint_index"),
            path_plan.get("dense_waypoint_index"),
        )
        if active_index is None:
            return labels

        for key in ("local_goal", "nav2_compute_goal", "transition_anchor"):
            visit_waypoint(path_plan.get(key))
        for index in range(max(0, active_index - 1), min(len(waypoints), active_index + 2)):
            visit_waypoint(waypoints[index])
        return labels

    @classmethod
    def _first_waypoint_index(cls, *values: Any) -> int | None:
        for value in values:
            index = cls._coerce_waypoint_index(value)
            if index is not None:
                return index
        return None

    @staticmethod
    def _coerce_waypoint_index(value: Any) -> int | None:
        try:
            index = int(value)
        except (TypeError, ValueError):
            return None
        return index if index >= 0 else None

    @classmethod
    def _distance_to_route_path(cls, *, pose: Any, path_plan: dict[str, Any]) -> float | None:
        if not isinstance(pose, dict):
            return None
        vertical_axis = str(path_plan.get("vertical_axis") or "z")
        pose_xy = cls._project_route_point(pose, vertical_axis=vertical_axis)
        if pose_xy is None:
            return None

        points: list[tuple[float, float]] = []
        for key in ("waypoints", "nav2_path_points", "dense_waypoints"):
            values = path_plan.get(key)
            if not isinstance(values, list):
                continue
            for value in values:
                projected = cls._project_route_point(value, vertical_axis=vertical_axis)
                if projected is not None:
                    points.append(projected)
        if not points:
            return None
        return min(((pose_xy[0] - x) ** 2 + (pose_xy[1] - y) ** 2) ** 0.5 for x, y in points)

    @staticmethod
    def _project_route_point(point: Any, *, vertical_axis: str) -> tuple[float, float] | None:
        if not isinstance(point, dict):
            return None
        axes = {
            "x": ("y", "z"),
            "y": ("x", "z"),
            "z": ("x", "y"),
        }.get(vertical_axis, ("x", "y"))
        try:
            return float(point[axes[0]]), float(point[axes[1]])
        except (KeyError, TypeError, ValueError):
            if axes == ("x", "y"):
                return None
            try:
                return float(point["x"]), float(point["y"])
            except (KeyError, TypeError, ValueError):
                return None

    def _resolve_grounded_goal_bundle(
        self,
        *,
        navigator: NavigatorBackend,
        subtask: Subtask,
        context: ExecutionContext,
        scene_id: str | None,
        start_state: dict[str, Any],
        nav_context: dict[str, Any],
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        grounded_goal: dict[str, Any] | None = None
        path_plan: dict[str, Any] | None = None
        navigation_skill_selection: dict[str, Any] | None = None
        interpreted_goal: dict[str, Any] | None = None
        navigation_grounding_context: dict[str, Any] | None = None
        prepared_navigation_payload: dict[str, Any] | None = None
        object_approach_selection: dict[str, Any] | None = None
        selected_object_approach: dict[str, Any] | None = None

        cached_object_approach = runtime_object_approach.load_cached_object_approach_state(
            context=context,
            subtask=subtask,
        )
        if cached_object_approach is not None:
            restored = runtime_object_approach.restore_cached_object_approach_state(
                cached_object_approach
            )
            grounded_goal = restored["grounded_goal"]
            navigation_skill_selection = restored["navigation_skill_selection"]
            prepared_navigation_payload = restored["prepared_navigation_payload"]
            object_approach_selection = restored["object_approach_selection"]
            selected_object_approach = restored["selected_object_approach"]
            path_plan = restored["path_plan"]
            interpreted_goal = restored.get("interpreted_goal")
            navigation_grounding_context = restored.get("navigation_grounding_context")
        else:
            instruction = runtime_execution_context.resolve_instruction(subtask)
            nav_context["start"] = dict(start_state)
            interpreted_goal = self._interpret_navigation_goal(
                instruction=instruction,
                nav_context=nav_context,
            )
            if isinstance(interpreted_goal, dict) and interpreted_goal:
                nav_context["interpreted_goal"] = dict(interpreted_goal)
            navigation_grounding_context = dict(nav_context)
            grounded_goal = navigator.ground_goal(
                instruction,
                context=nav_context,
            )
            grounding_selection = self._select_grounding_candidate_with_llm(
                instruction=instruction,
                nav_context=nav_context,
                grounded_goal=grounded_goal,
            )
            if grounding_selection is not None:
                grounded_goal = grounding_selection
                nav_context["grounded_goal"] = dict(grounded_goal)
            if runtime_object_approach.should_use_object_approach_flow(
                subtask=subtask, grounded_goal=grounded_goal
            ):
                history = runtime_object_approach.prime_object_approach_history(
                    memory=self.memory,
                    subtask=subtask,
                    scene_id=scene_id,
                    goal=grounded_goal,
                )
                if history is not None:
                    context.runtime_state["object_approach_history"] = history

                selection = self._select_navigation_skill(subtask=subtask, context=context)
                navigation_skill_selection = runtime_object_approach.serialize_skill_selection(
                    selection
                )
                context.runtime_state["navigation_grounded_goal_for_skill"] = dict(grounded_goal)
                skill = self._resolve_navigation_skill(
                    subtask=subtask, context=context, selection=selection
                )
                if skill is not None:
                    prepared_navigation_payload = dict(
                        skill.prepare(
                            subtask=subtask,
                            context=context,
                            navigator=navigator,
                            start=start_state,
                            goal=grounded_goal,
                            navigation_context=nav_context,
                        )
                    )
                grounded_goal, object_approach_selection, selected_object_approach = (
                    self._select_object_approach_candidate(
                        subtask=subtask,
                        context=context,
                        grounded_goal=grounded_goal,
                        prepared_navigation_payload=prepared_navigation_payload,
                    )
                )
                runtime_object_approach.store_cached_object_approach_state(
                    context=context,
                    subtask=subtask,
                    grounded_goal=grounded_goal,
                    navigation_skill_selection=navigation_skill_selection,
                    prepared_navigation_payload=prepared_navigation_payload,
                    object_approach_selection=object_approach_selection,
                    selected_object_approach=selected_object_approach,
                    interpreted_goal=interpreted_goal,
                    navigation_grounding_context=navigation_grounding_context,
                )
            else:
                runtime_object_approach.clear_cached_object_approach_state(
                    context=context, subtask=subtask
                )

        should_plan_path = not runtime_object_approach.should_reuse_cached_path_plan(
            cached_object_approach=cached_object_approach,
            path_plan=path_plan,
            subtask=subtask,
            observation=observation,
        )
        if should_plan_path:
            path_plan = navigator.plan_path(
                start=start_state,
                goal=grounded_goal,
                context=nav_context,
            )
            nav2_selected = (
                path_plan.get("selected_object_approach") if isinstance(path_plan, dict) else None
            )
            if isinstance(nav2_selected, dict) and nav2_selected:
                previous_selected = selected_object_approach
                selected_object_approach = dict(nav2_selected)
                grounded_goal = {
                    **grounded_goal,
                    "selected_object_approach": dict(selected_object_approach),
                }
                context.runtime_state["selected_object_approach"] = dict(selected_object_approach)
                if isinstance(object_approach_selection, dict):
                    object_approach_selection = {
                        **object_approach_selection,
                        "candidate": dict(selected_object_approach),
                        "pre_nav2_candidate_id": (
                            previous_selected.get("candidate_id")
                            if isinstance(previous_selected, dict)
                            else None
                        ),
                        "selected_candidate_id": selected_object_approach.get("candidate_id"),
                        "source": selected_object_approach.get("selection_source")
                        or "nav2_candidate_validation",
                    }
            if runtime_object_approach.should_use_object_approach_flow(
                subtask=subtask, grounded_goal=grounded_goal
            ):
                runtime_object_approach.store_cached_object_approach_state(
                    context=context,
                    subtask=subtask,
                    grounded_goal=grounded_goal,
                    navigation_skill_selection=navigation_skill_selection,
                    prepared_navigation_payload=prepared_navigation_payload,
                    object_approach_selection=object_approach_selection,
                    selected_object_approach=selected_object_approach,
                    path_plan=path_plan,
                    interpreted_goal=interpreted_goal,
                    navigation_grounding_context=navigation_grounding_context,
                )

        return {
            "cached_object_approach": cached_object_approach,
            "grounded_goal": grounded_goal,
            "path_plan": path_plan,
            "interpreted_goal": interpreted_goal,
            "navigation_grounding_context": navigation_grounding_context,
            "navigation_skill_selection": navigation_skill_selection,
            "prepared_navigation_payload": prepared_navigation_payload,
            "object_approach_selection": object_approach_selection,
            "selected_object_approach": selected_object_approach,
        }

    def _interpret_navigation_goal(
        self,
        *,
        instruction: str,
        nav_context: dict[str, Any],
    ) -> dict[str, Any] | None:
        if self.goal_interpreter is None:
            return None
        interpreted_goal = self.goal_interpreter.interpret_goal(
            instruction=instruction,
            context=nav_context,
        )
        if not isinstance(interpreted_goal, dict):
            return interpreted_goal
        grounding_candidates = nav_context.get("grounding_candidates")
        if not isinstance(grounding_candidates, list) or not grounding_candidates:
            interpreted_goal = dict(interpreted_goal)
            interpreted_goal["selected_grounding_candidate"] = {}
        return interpreted_goal

    def _select_grounding_candidate_with_llm(
        self,
        *,
        instruction: str,
        nav_context: dict[str, Any],
        grounded_goal: dict[str, Any],
    ) -> dict[str, Any] | None:
        candidates = grounded_goal.get("grounding_candidates")
        if (
            self.goal_interpreter is None
            or not isinstance(candidates, list)
            or len(candidates) <= 1
        ):
            return None
        selection_context = dict(nav_context)
        selection_context["grounded_goal"] = dict(grounded_goal)
        original_candidates = [
            dict(candidate) for candidate in candidates if isinstance(candidate, dict)
        ]
        required_affordance_terms = _required_affordance_terms(
            instruction=instruction,
            nav_context=nav_context,
            grounded_goal=grounded_goal,
        )
        affordance_matched_candidates = [
            candidate
            for candidate in original_candidates
            if _candidate_satisfies_affordance(candidate, required_affordance_terms)
        ]
        selection_candidates = affordance_matched_candidates or original_candidates
        selection_context["grounding_candidates"] = selection_candidates

        def build_selected_goal(
            *,
            candidate: dict[str, Any],
            selected_payload: dict[str, Any],
            decision_payload: dict[str, Any],
        ) -> dict[str, Any]:
            selected_object_id = str(candidate.get("object_id") or "").strip()
            next_goal = dict(grounded_goal)
            validation = {
                "required_affordance_terms": list(required_affordance_terms),
                "candidate_pool_size": len(original_candidates),
                "affordance_matched_candidate_count": len(affordance_matched_candidates),
            }
            next_goal.update(
                {
                    "object_id": candidate.get("object_id"),
                    "object_name": candidate.get("object_name"),
                    "room_id": candidate.get("room_id"),
                    "room_name": candidate.get("room_name"),
                    "floor_id": candidate.get("floor_id"),
                    "position": dict(candidate.get("position") or {}),
                    "selected_grounding_candidate": {
                        **dict(selected_payload),
                        "object_id": selected_object_id,
                        "candidate": dict(candidate),
                        "validation": validation,
                    },
                    "grounding_selection_decision": dict(decision_payload),
                    "grounding_selection_source": decision_payload.get("source")
                    or "navigation_goal_interpreter",
                }
            )
            next_goal["grounding_candidates"] = [
                {
                    **dict(item),
                    "selection_status": (
                        "selected"
                        if str(item.get("object_id") or "").strip() == selected_object_id
                        else "rejected"
                    ),
                }
                for item in selection_context["grounding_candidates"]
            ]
            return next_goal

        if len(selection_context["grounding_candidates"]) == 1 and affordance_matched_candidates:
            candidate = selection_context["grounding_candidates"][0]
            return build_selected_goal(
                candidate=candidate,
                selected_payload={
                    "object_id": str(candidate.get("object_id") or "").strip(),
                    "reason": "Only grounding candidate compatible with the requested interaction affordance.",
                },
                decision_payload={"source": "navigation_affordance_filter"},
            )
        if len(selection_context["grounding_candidates"]) <= 1:
            return None
        decision = self._interpret_navigation_goal(
            instruction=instruction, nav_context=selection_context
        )
        if not isinstance(decision, dict):
            return None
        selected = decision.get("selected_grounding_candidate")
        if not isinstance(selected, dict):
            return None
        selected_object_id = str(selected.get("object_id") or "").strip()
        if not selected_object_id:
            return None
        selected_candidate: dict[str, Any] | None = None
        for candidate in selection_context["grounding_candidates"]:
            if str(candidate.get("object_id") or "").strip() != selected_object_id:
                continue
            selected_candidate = candidate
            break
        if selected_candidate is None and affordance_matched_candidates:
            selected_candidate = affordance_matched_candidates[0]
            selected_object_id = str(selected_candidate.get("object_id") or "").strip()
            selected = {
                **dict(selected),
                "object_id": selected_object_id,
                "reason": (
                    "LLM candidate choice did not satisfy the requested interaction affordance; "
                    "using the first affordance-compatible candidate."
                ),
            }
        if selected_candidate is not None:
            return build_selected_goal(
                candidate=selected_candidate,
                selected_payload=selected,
                decision_payload=decision,
            )
        return None

    def _select_navigation_skill(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
    ) -> LocalSkillSelection:
        return runtime_skill_routing.select_navigation_skill(
            selector=self.selector,
            registry=self.skill_registry,
            subtask=subtask,
            context=context,
        )

    def _resolve_navigation_skill(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        selection: LocalSkillSelection,
    ) -> Any | None:
        return runtime_skill_routing.resolve_navigation_skill(
            registry=self.skill_registry,
            subtask=subtask,
            context=context,
            selection=selection,
        )

    def _select_object_approach_candidate(
        self,
        *,
        subtask: Subtask,
        context: ExecutionContext,
        grounded_goal: dict[str, Any],
        prepared_navigation_payload: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
        next_goal = dict(grounded_goal)
        object_approach_selection: dict[str, Any] | None = None
        selected_object_approach: dict[str, Any] | None = None

        if not isinstance(prepared_navigation_payload, dict):
            return next_goal, object_approach_selection, selected_object_approach

        prepared_history = prepared_navigation_payload.get("history")
        if isinstance(prepared_history, dict):
            context.runtime_state["object_approach_history"] = prepared_history

        candidates = list(prepared_navigation_payload.get("candidates") or [])
        if not candidates:
            return next_goal, object_approach_selection, selected_object_approach

        candidates, room_rejected_candidates = (
            runtime_object_approach.filter_candidates_for_goal_room(
                candidates=[dict(candidate) for candidate in candidates],
                goal=next_goal,
            )
        )
        filtered_navigation_payload = dict(prepared_navigation_payload)
        filtered_navigation_payload["candidates"] = candidates
        next_goal["object_approach_candidates"] = candidates
        object_approach_selection = dict(
            self.approach_point_selector.select_candidate(
                subtask=subtask,
                context=context,
                goal=next_goal,
                prepared_payload=filtered_navigation_payload,
            )
        )
        if room_rejected_candidates:
            object_approach_selection["room_rejected_candidate_ids"] = [
                candidate.get("candidate_id") for candidate in room_rejected_candidates
            ]
        selected_candidate = object_approach_selection.get("candidate")
        if isinstance(selected_candidate, dict) and selected_candidate:
            selected_object_approach = dict(selected_candidate)
            next_goal["selected_object_approach"] = selected_object_approach
            context.runtime_state["selected_object_approach"] = dict(selected_object_approach)
        elif selected_candidate is None:
            next_goal["object_approach_selection_failed"] = True
            next_goal["object_approach_selection_failure_reason"] = str(
                object_approach_selection.get("reason")
                or "object-approach candidate selection failed"
            )
            context.runtime_state.pop("selected_object_approach", None)

        return next_goal, object_approach_selection, selected_object_approach

    def _refresh_navigation_context(
        self,
        nav_context: dict[str, Any],
        *,
        backend_state: dict[str, Any] | None,
    ) -> None:
        runtime_execution_context.refresh_navigation_context(
            nav_context,
            map_state=self._map_state,
            backend_state=backend_state,
        )

    def _sync_scene_map(
        self,
        *,
        scene_id: str | None,
        backend_state: dict[str, Any] | None,
        grounded_goal: dict[str, Any] | None,
        path_plan: dict[str, Any] | None,
    ) -> None:
        runtime_execution_context.sync_scene_map(
            memory=self.memory,
            scene_id=scene_id,
            backend_state=backend_state,
            grounded_goal=grounded_goal,
            path_plan=path_plan,
        )

    @staticmethod
    def _extract_policy_runtime_artifacts(info: dict[str, Any] | None) -> dict[str, Any]:
        return runtime_observation.extract_policy_runtime_artifacts(info)
