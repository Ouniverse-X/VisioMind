"""Open-loop mock example wired through the canonical entrypoint layer."""

from __future__ import annotations

from voltron.agents import ActionAgent, BrainAgent, NavigationAgent, VisionAgent
from voltron.agents.action.body.task_planning import HeuristicActionTaskPlanner
from voltron.agents.action.skills import DefaultActionTaskPlanningSkill
from voltron.agents.action.tools.action_projection import ActionProjection
from voltron.agents.brain.body.rule_based_planner import RuleBasedPlanner
from voltron.shared.enums import TaskType
from voltron.shared.context import TaskRequest
from voltron.runtime.orchestrator.open_loop import VoltronOrchestrator
from voltron.runtime.testing import MockMemoryAdapter, MockPolicyAdapter, MockVisionAdapter


def main() -> None:
    memory = MockMemoryAdapter()
    vision = MockVisionAdapter()
    policy = MockPolicyAdapter()
    projector = ActionProjection.from_embodiment("behavior_r1_pro")

    brain = BrainAgent(memory=memory, planner=RuleBasedPlanner())
    vision_agent = VisionAgent(memory=memory, vision=vision)
    navigation_agent = NavigationAgent(memory=memory, policy=policy, projector=projector)
    action_agent = ActionAgent(
        memory=memory,
        policy=policy,
        projector=projector,
        task_planning_skill=DefaultActionTaskPlanningSkill(),
        task_planner=HeuristicActionTaskPlanner(),
    )

    orchestrator = VoltronOrchestrator(brain, vision_agent, navigation_agent, action_agent, max_retries=1)

    request = TaskRequest(
        task_id="task_mock_001",
        description="把红色杯子从厨房拿到客厅",
        task_type=TaskType.MANIPULATION,
    )

    runtime_inputs = {
        "st_01": {"observation": {"dummy": 1}},
        "st_02": {"images": ["ZmFrZV9pbWFnZQ=="]},
        "st_03": {"observation": {"dummy": 1}, "pre_state": {"gripper_empty": True}, "post_state": {"gripper_empty": False}},
        "st_04": {"observation": {"dummy": 1}},
        "st_05": {"images": ["ZmFrZV9pbWFnZQ=="]},
        "st_06": {"observation": {"dummy": 1}, "pre_state": {"cup_held": True}, "post_state": {"cup_held": False}},
    }

    result = orchestrator.run_task(request=request, runtime_inputs=runtime_inputs)
    print(result)


if __name__ == "__main__":
    main()
