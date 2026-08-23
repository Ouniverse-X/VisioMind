"""Agent assembly helpers for open-loop runtime entrypoints."""

from __future__ import annotations

from voltron.agents import ActionAgent, BrainAgent, MemoryAgent, NavigationAgent, VisionAgent
from voltron.agents.action.body.step_verification import VisionBackedActionStepVerifier
from voltron.agents.action.skills import DefaultActionTaskPlanningSkill
from voltron.agents.action.tools.action_projection import ActionProjection
from voltron.integrations.manipulation import Gr00tPolicyAdapter
from voltron.integrations.memory.hems.backend import HEMSAdapter
from voltron.integrations.memory.service import MemoryAgentClient
from voltron.integrations.vlm.service.client import VLMHttpAdapter
from voltron.runtime.assembly import backend_factory
from voltron.runtime.assembly.capabilities import inject_agent_capabilities
from voltron.runtime.orchestrator.open_loop import VoltronOrchestrator


def build_open_loop_orchestrator(
    embodiment: str,
    gr00t_host: str,
    gr00t_port: int,
    vision_endpoint: str,
    vision_timeout_s: float,
    vision_max_retries: int,
    vision_retry_backoff_s: float,
    memory_agent_endpoint: str,
    use_memory_agent: bool,
    planner_mode: str,
    brain_base_url: str | None,
    brain_model: str | None,
    brain_api_key: str | None,
    brain_api_key_env: str,
    brain_timeout_s: float,
    brain_temperature: float,
    brain_max_retries: int,
    brain_retry_backoff_s: float,
    action_selector: str,
    action_base_url: str | None,
    action_model: str | None,
    action_api_key: str | None,
    action_api_key_env: str,
    action_timeout_s: float,
    action_temperature: float,
    action_max_retries: int,
    action_retry_backoff_s: float,
    navigation_base_url: str | None = None,
    navigation_model: str | None = None,
    navigation_api_key: str | None = None,
    navigation_api_key_env: str = "OPENAI_API_KEY",
    navigation_timeout_s: float = 30.0,
    navigation_temperature: float = 0.1,
    navigation_max_retries: int = 0,
    navigation_retry_backoff_s: float = 1.0,
    memory_agent_enabled: bool = True,
    memory_llm_backend: str | None = None,
    memory_llm_base_url: str | None = None,
    memory_llm_model: str | None = None,
    memory_llm_api_key: str | None = None,
    memory_llm_api_key_env: str = "OPENAI_API_KEY",
    memory_llm_timeout_s: float = 30.0,
    memory_llm_temperature: float = 0.0,
    memory_llm_max_retries: int = 0,
    memory_llm_retry_backoff_s: float = 1.0,
    memory_experience_extraction_enabled: bool = False,
    memory_experience_extraction_min_confidence_to_write: float = 0.4,
    memory_experience_extraction_min_confidence_to_promote: float = 0.7,
) -> VoltronOrchestrator:
    if use_memory_agent:
        memory = MemoryAgentClient(
            endpoint=memory_agent_endpoint,
            timeout_s=max(15.0, float(memory_llm_timeout_s)),
        )
    else:
        memory_backend = HEMSAdapter()
        if memory_agent_enabled:
            memory_extractor = backend_factory.build_memory_extractor(
                enabled=memory_experience_extraction_enabled,
                backend=memory_llm_backend,
                base_url=memory_llm_base_url,
                model=memory_llm_model,
                api_key=memory_llm_api_key,
                api_key_env=memory_llm_api_key_env,
                timeout_s=memory_llm_timeout_s,
                temperature=memory_llm_temperature,
                max_retries=memory_llm_max_retries,
                retry_backoff_s=memory_llm_retry_backoff_s,
            )
            memory = MemoryAgent(
                backend=memory_backend,
                extractor=memory_extractor,
                experience_extraction_enabled=memory_experience_extraction_enabled,
                min_confidence_to_write=memory_experience_extraction_min_confidence_to_write,
                min_confidence_to_promote=memory_experience_extraction_min_confidence_to_promote,
            )
        else:
            memory = memory_backend
    policy = Gr00tPolicyAdapter(host=gr00t_host, port=gr00t_port, strict=False)
    vision = VLMHttpAdapter(
        endpoint=vision_endpoint,
        timeout_s=vision_timeout_s,
        max_retries=vision_max_retries,
        retry_backoff_s=vision_retry_backoff_s,
    )
    projector = ActionProjection.from_embodiment(embodiment)
    planner = backend_factory.build_planner(
        planner_backend=planner_mode,
        brain_base_url=brain_base_url,
        brain_model=brain_model,
        brain_api_key=brain_api_key,
        brain_api_key_env=brain_api_key_env,
        brain_timeout_s=brain_timeout_s,
        brain_temperature=brain_temperature,
        brain_max_retries=brain_max_retries,
        brain_retry_backoff_s=brain_retry_backoff_s,
    )
    selector = backend_factory.build_vla_selector(
        selector_mode=action_selector,
        action_base_url=action_base_url,
        action_model=action_model,
        action_api_key=action_api_key,
        action_api_key_env=action_api_key_env,
        action_timeout_s=action_timeout_s,
        action_temperature=action_temperature,
        action_max_retries=action_max_retries,
        action_retry_backoff_s=action_retry_backoff_s,
    )
    deliberator = backend_factory.build_vla_deliberator(
        selector_mode=action_selector,
        action_base_url=action_base_url,
        action_model=action_model,
        action_api_key=action_api_key,
        action_api_key_env=action_api_key_env,
        action_timeout_s=action_timeout_s,
        action_temperature=action_temperature,
        action_max_retries=action_max_retries,
        action_retry_backoff_s=action_retry_backoff_s,
    )
    task_planner = backend_factory.build_action_task_planner(
        selector_mode=action_selector,
        action_base_url=action_base_url,
        action_model=action_model,
        action_api_key=action_api_key,
        action_api_key_env=action_api_key_env,
        action_timeout_s=action_timeout_s,
        action_temperature=action_temperature,
        action_max_retries=action_max_retries,
        action_retry_backoff_s=action_retry_backoff_s,
    )

    brain = BrainAgent(memory=memory, planner=planner)
    vision_agent = VisionAgent(memory=memory, vision=vision)
    navigation_agent = NavigationAgent(
        memory=memory,
        policy=policy,
        projector=projector,
        selector=backend_factory.build_vln_selector(
            navigation_base_url=navigation_base_url,
            navigation_model=navigation_model,
            navigation_api_key=navigation_api_key,
            navigation_api_key_env=navigation_api_key_env,
            navigation_timeout_s=navigation_timeout_s,
            navigation_temperature=navigation_temperature,
            navigation_max_retries=navigation_max_retries,
            navigation_retry_backoff_s=navigation_retry_backoff_s,
        ),
        approach_point_selector=backend_factory.build_vln_point_selector(
            navigation_base_url=navigation_base_url,
            navigation_model=navigation_model,
            navigation_api_key=navigation_api_key,
            navigation_api_key_env=navigation_api_key_env,
            navigation_timeout_s=navigation_timeout_s,
            navigation_temperature=navigation_temperature,
            navigation_max_retries=navigation_max_retries,
            navigation_retry_backoff_s=navigation_retry_backoff_s,
        ),
        goal_interpreter=backend_factory.build_vln_goal_interpreter(
            navigation_base_url=navigation_base_url,
            navigation_model=navigation_model,
            navigation_api_key=navigation_api_key,
            navigation_api_key_env=navigation_api_key_env,
            navigation_timeout_s=navigation_timeout_s,
            navigation_temperature=navigation_temperature,
            navigation_max_retries=navigation_max_retries,
            navigation_retry_backoff_s=navigation_retry_backoff_s,
        ),
    )
    action_agent = ActionAgent(
        memory=memory,
        policy=policy,
        projector=projector,
        selector=selector,
        deliberator=deliberator,
        task_planning_skill=DefaultActionTaskPlanningSkill(),
        task_planner=task_planner,
        step_verifier=VisionBackedActionStepVerifier(vision=vision),
        verify_every_control_steps=400,
        verify_after_first_success=False,
    )
    inject_agent_capabilities(brain, [vision_agent, navigation_agent, action_agent])

    return VoltronOrchestrator(
        brain_agent=brain,
        vision_agent=vision_agent,
        navigation_agent=navigation_agent,
        action_agent=action_agent,
        max_retries=1,
    )


__all__ = ["build_open_loop_orchestrator"]
