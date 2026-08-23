"""Closed-loop mock example wired through the canonical entrypoint layer."""

from __future__ import annotations

from voltron.agents import ActionAgent, BrainAgent, NavigationAgent, VisionAgent
from voltron.agents.action.tools.action_projection import ActionProjection
from voltron.agents.brain.body.rule_based_planner import RuleBasedPlanner
from voltron.shared.enums import TaskType
from voltron.shared.context import TaskRequest
from voltron.runtime.orchestrator.closed_loop import ClosedLoopOrchestrator
from voltron.runtime.testing import MockMemoryAdapter, MockPolicyAdapter, MockRuntimeEnvironment, MockVisionAdapter


def main() -> None:
    memory = MockMemoryAdapter()
    vision = MockVisionAdapter()
    policy = MockPolicyAdapter()
    projector = ActionProjection.from_embodiment("behavior_r1_pro")

    brain = BrainAgent(memory=memory, planner=RuleBasedPlanner())
    vision_agent = VisionAgent(memory=memory, vision=vision)
    navigation_agent = NavigationAgent(memory=memory, policy=policy, projector=projector)
    action_agent = ActionAgent(memory=memory, policy=policy, projector=projector)

    orchestrator = ClosedLoopOrchestrator(
        brain_agent=brain,
        vision_agent=vision_agent,
        navigation_agent=navigation_agent,
        action_agent=action_agent,
        max_retries=1,
        max_control_steps_per_subtask=8,
    )
    environment = MockRuntimeEnvironment(step_budget_per_subtask=2)

    request = TaskRequest(
        task_id="task_closed_loop_mock_001",
        description="把红色杯子从厨房拿到客厅",
        task_type=TaskType.MANIPULATION,
    )

    result = orchestrator.run_task(request=request, environment=environment)
    print(result)


if __name__ == "__main__":
    main()
