"""Canonical open-loop entrypoint for backend-composed runtime runs."""

from __future__ import annotations

import argparse

from voltron.config_loader import add_config_argument, parse_args_with_config
from voltron.shared.enums import TaskType
from voltron.shared.context import TaskRequest
from voltron.runtime.assembly import backend_factory as runtime_backend_factory
from voltron.runtime.assembly import open_loop_factory
from voltron.runtime.orchestrator.open_loop import VoltronOrchestrator


def build_planner(
    planner_mode: str,
    brain_base_url: str | None,
    brain_model: str | None,
    brain_api_key: str | None,
    brain_api_key_env: str,
    brain_timeout_s: float,
    brain_temperature: float,
    brain_max_retries: int,
    brain_retry_backoff_s: float,
):
    return runtime_backend_factory.build_planner(
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


def build_orchestrator(
    embodiment: str = "behavior_r1_pro",
    gr00t_host: str = "127.0.0.1",
    gr00t_port: int = 5555,
    vision_endpoint: str = "http://127.0.0.1:8081/process",
    vision_timeout_s: float = 60.0,
    vision_max_retries: int = 0,
    vision_retry_backoff_s: float = 1.0,
    memory_agent_endpoint: str = "http://127.0.0.1:8070/rpc",
    use_memory_agent: bool = True,
    planner_mode: str = "openai",
    brain_base_url: str | None = None,
    brain_model: str | None = None,
    brain_api_key: str | None = None,
    brain_api_key_env: str = "OPENAI_API_KEY",
    brain_timeout_s: float = 30.0,
    brain_temperature: float = 0.1,
    brain_max_retries: int = 0,
    brain_retry_backoff_s: float = 1.0,
    action_selector: str = "openai",
    action_base_url: str | None = None,
    action_model: str | None = None,
    action_api_key: str | None = None,
    action_api_key_env: str = "OPENAI_API_KEY",
    action_timeout_s: float = 10.0,
    action_temperature: float = 0.0,
    action_max_retries: int = 0,
    action_retry_backoff_s: float = 1.0,
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
    return open_loop_factory.build_open_loop_orchestrator(
        embodiment=embodiment,
        gr00t_host=gr00t_host,
        gr00t_port=gr00t_port,
        vision_endpoint=vision_endpoint,
        vision_timeout_s=vision_timeout_s,
        vision_max_retries=vision_max_retries,
        vision_retry_backoff_s=vision_retry_backoff_s,
        memory_agent_endpoint=memory_agent_endpoint,
        use_memory_agent=use_memory_agent,
        planner_mode=planner_mode,
        brain_base_url=brain_base_url,
        brain_model=brain_model,
        brain_api_key=brain_api_key,
        brain_api_key_env=brain_api_key_env,
        brain_timeout_s=brain_timeout_s,
        brain_temperature=brain_temperature,
        brain_max_retries=brain_max_retries,
        brain_retry_backoff_s=brain_retry_backoff_s,
        action_selector=action_selector,
        action_base_url=action_base_url,
        action_model=action_model,
        action_api_key=action_api_key,
        action_api_key_env=action_api_key_env,
        action_timeout_s=action_timeout_s,
        action_temperature=action_temperature,
        action_max_retries=action_max_retries,
        action_retry_backoff_s=action_retry_backoff_s,
        navigation_base_url=navigation_base_url,
        navigation_model=navigation_model,
        navigation_api_key=navigation_api_key,
        navigation_api_key_env=navigation_api_key_env,
        navigation_timeout_s=navigation_timeout_s,
        navigation_temperature=navigation_temperature,
        navigation_max_retries=navigation_max_retries,
        navigation_retry_backoff_s=navigation_retry_backoff_s,
        memory_agent_enabled=memory_agent_enabled,
        memory_llm_backend=memory_llm_backend,
        memory_llm_base_url=memory_llm_base_url,
        memory_llm_model=memory_llm_model,
        memory_llm_api_key=memory_llm_api_key,
        memory_llm_api_key_env=memory_llm_api_key_env,
        memory_llm_timeout_s=memory_llm_timeout_s,
        memory_llm_temperature=memory_llm_temperature,
        memory_llm_max_retries=memory_llm_max_retries,
        memory_llm_retry_backoff_s=memory_llm_retry_backoff_s,
        memory_experience_extraction_enabled=memory_experience_extraction_enabled,
        memory_experience_extraction_min_confidence_to_write=memory_experience_extraction_min_confidence_to_write,
        memory_experience_extraction_min_confidence_to_promote=memory_experience_extraction_min_confidence_to_promote,
    )


def build_vln_selector(
    *,
    navigation_base_url: str | None,
    navigation_model: str | None,
    navigation_api_key: str | None,
    navigation_api_key_env: str,
    navigation_timeout_s: float,
    navigation_temperature: float,
    navigation_max_retries: int,
    navigation_retry_backoff_s: float,
):
    return runtime_backend_factory.build_vln_selector(
        navigation_base_url=navigation_base_url,
        navigation_model=navigation_model,
        navigation_api_key=navigation_api_key,
        navigation_api_key_env=navigation_api_key_env,
        navigation_timeout_s=navigation_timeout_s,
        navigation_temperature=navigation_temperature,
        navigation_max_retries=navigation_max_retries,
        navigation_retry_backoff_s=navigation_retry_backoff_s,
    )


def build_vln_point_selector(
    *,
    navigation_base_url: str | None,
    navigation_model: str | None,
    navigation_api_key: str | None,
    navigation_api_key_env: str,
    navigation_timeout_s: float,
    navigation_temperature: float,
    navigation_max_retries: int,
    navigation_retry_backoff_s: float,
):
    return runtime_backend_factory.build_vln_point_selector(
        navigation_base_url=navigation_base_url,
        navigation_model=navigation_model,
        navigation_api_key=navigation_api_key,
        navigation_api_key_env=navigation_api_key_env,
        navigation_timeout_s=navigation_timeout_s,
        navigation_temperature=navigation_temperature,
        navigation_max_retries=navigation_max_retries,
        navigation_retry_backoff_s=navigation_retry_backoff_s,
    )


def build_vla_selector(
    selector_mode: str,
    action_base_url: str | None,
    action_model: str | None,
    action_api_key: str | None,
    action_api_key_env: str,
    action_timeout_s: float,
    action_temperature: float,
    action_max_retries: int,
    action_retry_backoff_s: float,
):
    return runtime_backend_factory.build_vla_selector(
        selector_mode=selector_mode,
        action_base_url=action_base_url,
        action_model=action_model,
        action_api_key=action_api_key,
        action_api_key_env=action_api_key_env,
        action_timeout_s=action_timeout_s,
        action_temperature=action_temperature,
        action_max_retries=action_max_retries,
        action_retry_backoff_s=action_retry_backoff_s,
    )


def build_vla_deliberator(
    selector_mode: str,
    action_base_url: str | None,
    action_model: str | None,
    action_api_key: str | None,
    action_api_key_env: str,
    action_timeout_s: float,
    action_temperature: float,
    action_max_retries: int,
    action_retry_backoff_s: float,
):
    return runtime_backend_factory.build_vla_deliberator(
        selector_mode=selector_mode,
        action_base_url=action_base_url,
        action_model=action_model,
        action_api_key=action_api_key,
        action_api_key_env=action_api_key_env,
        action_timeout_s=action_timeout_s,
        action_temperature=action_temperature,
        action_max_retries=action_max_retries,
        action_retry_backoff_s=action_retry_backoff_s,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_config_argument(parser)
    parser.add_argument("--embodiment", type=str, default="behavior_r1_pro")
    parser.add_argument("--gr00t-host", type=str, default="127.0.0.1")
    parser.add_argument("--gr00t-port", type=int, default=5555)
    parser.add_argument("--vision-endpoint", type=str, default="http://127.0.0.1:8081/process")
    parser.add_argument("--vision-timeout-s", type=float, default=60.0)
    parser.add_argument("--vision-max-retries", type=int, default=0)
    parser.add_argument("--vision-retry-backoff-s", type=float, default=1.0)
    parser.add_argument("--memory-agent-endpoint", type=str, default="http://127.0.0.1:8070/rpc")
    parser.add_argument(
        "--memory-mode",
        choices=["agent", "local"],
        default="agent",
        help="agent: use MemoryAgentClient; local: use local HEMSAdapter",
    )
    parser.add_argument("--memory-agent-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--memory-llm-backend", type=str, default=None)
    parser.add_argument("--memory-llm-base-url", type=str, default=None)
    parser.add_argument("--memory-llm-model", type=str, default=None)
    parser.add_argument("--memory-llm-api-key", type=str, default=None)
    parser.add_argument("--memory-llm-api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--memory-llm-timeout-s", type=float, default=30.0)
    parser.add_argument("--memory-llm-temperature", type=float, default=0.0)
    parser.add_argument("--memory-llm-max-retries", type=int, default=0)
    parser.add_argument("--memory-llm-retry-backoff-s", type=float, default=1.0)
    parser.add_argument(
        "--memory-experience-extraction-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--memory-experience-extraction-min-confidence-to-write", type=float, default=0.4)
    parser.add_argument("--memory-experience-extraction-min-confidence-to-promote", type=float, default=0.7)
    parser.add_argument("--brain-planner", choices=["openai", "rule"], default="openai")
    parser.add_argument("--brain-base-url", type=str, default=None)
    parser.add_argument("--brain-model", type=str, default=None)
    parser.add_argument("--brain-api-key", type=str, default=None)
    parser.add_argument("--brain-api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--brain-timeout-s", type=float, default=30.0)
    parser.add_argument("--brain-temperature", type=float, default=0.1)
    parser.add_argument("--brain-max-retries", type=int, default=0)
    parser.add_argument("--brain-retry-backoff-s", type=float, default=1.0)
    parser.add_argument("--action-selector", choices=["openai", "heuristic"], default="openai")
    parser.add_argument("--action-base-url", type=str, default=None)
    parser.add_argument("--action-model", type=str, default=None)
    parser.add_argument("--action-api-key", type=str, default=None)
    parser.add_argument("--action-api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--action-timeout-s", type=float, default=10.0)
    parser.add_argument("--action-temperature", type=float, default=0.0)
    parser.add_argument("--action-max-retries", type=int, default=0)
    parser.add_argument("--action-retry-backoff-s", type=float, default=1.0)
    parser.add_argument("--navigation-base-url", type=str, default=None)
    parser.add_argument("--navigation-model", type=str, default=None)
    parser.add_argument("--navigation-api-key", type=str, default=None)
    parser.add_argument("--navigation-api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--navigation-timeout-s", type=float, default=30.0)
    parser.add_argument("--navigation-temperature", type=float, default=0.1)
    parser.add_argument("--navigation-max-retries", type=int, default=0)
    parser.add_argument("--navigation-retry-backoff-s", type=float, default=1.0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parse_args_with_config(parser)

    orchestrator = build_orchestrator(
        embodiment=args.embodiment,
        gr00t_host=args.gr00t_host,
        gr00t_port=args.gr00t_port,
        vision_endpoint=args.vision_endpoint,
        vision_timeout_s=args.vision_timeout_s,
        vision_max_retries=args.vision_max_retries,
        vision_retry_backoff_s=args.vision_retry_backoff_s,
        memory_agent_endpoint=args.memory_agent_endpoint,
        use_memory_agent=args.memory_mode == "agent",
        memory_agent_enabled=args.memory_agent_enabled,
        memory_llm_backend=args.memory_llm_backend,
        memory_llm_base_url=args.memory_llm_base_url,
        memory_llm_model=args.memory_llm_model,
        memory_llm_api_key=args.memory_llm_api_key,
        memory_llm_api_key_env=args.memory_llm_api_key_env,
        memory_llm_timeout_s=args.memory_llm_timeout_s,
        memory_llm_temperature=args.memory_llm_temperature,
        memory_llm_max_retries=args.memory_llm_max_retries,
        memory_llm_retry_backoff_s=args.memory_llm_retry_backoff_s,
        memory_experience_extraction_enabled=args.memory_experience_extraction_enabled,
        memory_experience_extraction_min_confidence_to_write=args.memory_experience_extraction_min_confidence_to_write,
        memory_experience_extraction_min_confidence_to_promote=(
            args.memory_experience_extraction_min_confidence_to_promote
        ),
        planner_mode=args.brain_planner,
        brain_base_url=args.brain_base_url,
        brain_model=args.brain_model,
        brain_api_key=args.brain_api_key,
        brain_api_key_env=args.brain_api_key_env,
        brain_timeout_s=args.brain_timeout_s,
        brain_temperature=args.brain_temperature,
        brain_max_retries=args.brain_max_retries,
        brain_retry_backoff_s=args.brain_retry_backoff_s,
        action_selector=args.action_selector,
        action_base_url=args.action_base_url,
        action_model=args.action_model,
        action_api_key=args.action_api_key,
        action_api_key_env=args.action_api_key_env,
        action_timeout_s=args.action_timeout_s,
        action_temperature=args.action_temperature,
        action_max_retries=args.action_max_retries,
        action_retry_backoff_s=args.action_retry_backoff_s,
        navigation_base_url=args.navigation_base_url,
        navigation_model=args.navigation_model,
        navigation_api_key=args.navigation_api_key,
        navigation_api_key_env=args.navigation_api_key_env,
        navigation_timeout_s=args.navigation_timeout_s,
        navigation_temperature=args.navigation_temperature,
        navigation_max_retries=args.navigation_max_retries,
        navigation_retry_backoff_s=args.navigation_retry_backoff_s,
    )

    request = TaskRequest(
        task_id="task_real_001",
        description="把厨房桌子上的红色杯子拿到客厅茶几上",
        task_type=TaskType.MANIPULATION,
    )

    runtime_inputs = {
        "st_01": {"observation": {}},
        "st_02": {"images": []},
        "st_03": {"observation": {}},
        "st_04": {"observation": {}},
        "st_05": {"images": []},
        "st_06": {"observation": {}},
    }

    result = orchestrator.run_task(request=request, runtime_inputs=runtime_inputs)
    print(result)


if __name__ == "__main__":
    main()
