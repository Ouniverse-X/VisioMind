"""CPU-only checks for the competition overlay.

These tests deliberately load the copied modules by file path.  They therefore
verify the files under ``xh/competition_code`` without importing or modifying
the source repositories' modules.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1] / "voltron"


def load_overlay_module(relative_path: str, module_name: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_competition_config_has_pick_and_place_sequence() -> None:
    config_path = ROOT / "configs" / "half_apple_to_packing_box_place_inside_i10.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    sequence = payload["action_sequence"]
    assert [step["action"] for step in sequence] == ["pick_up", "place_inside"]
    assert sequence[1]["target"]["container"] == "packing_box_210"
    # The runtime builder uses this compatibility field to select the
    # AnyGrasp low-resolution RGB wrapper even when action_sequence overrides
    # the actual plan.  Without it, the head camera silently changes from
    # 256x256 to 720x720 and materially changes detector proposals.
    assert payload["action_subtask_action"] == "pick_up"
    assert payload["action_target_object"] == "half_apple_213"
    assert payload["anygrasp"]["apply_nms"] is True


def test_new_industrial_configs() -> None:
    for tool_name, obj_id, container_id in [
        ("screwdriver", "screwdriver_188", "toolbox_191"),
        ("allen_wrench", "allen_wrench_189", "toolbox_191"),
        ("flashlight", "flashlight_190", "toolbox_191"),
    ]:
        config_path = ROOT / "configs" / f"{tool_name}_to_toolbox_place_inside_i00.json"
        assert config_path.exists(), f"Config file for {tool_name} does not exist"
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        sequence = payload["action_sequence"]
        assert [step["action"] for step in sequence] == ["pick_up", "place_inside"]
        assert sequence[0]["target"]["object"] == obj_id
        assert sequence[1]["target"]["object"] == obj_id
        assert sequence[1]["target"]["container"] == container_id
        assert payload["action_subtask_action"] == "pick_up"
        assert payload["action_target_object"] == obj_id
        assert payload["anygrasp"]["apply_nms"] is True
        assert payload["environment"]["env_id"] == "sim_behavior_r1_pro/outfit_a_basic_toolbox"
        assert payload["environment"]["behavior_task_instance_id"] == 0


def test_action_only_plan_builder_preserves_container_target() -> None:
    module = load_overlay_module(
        "entrypoints/examples/closed_loop/action_only.py",
        "xh_competition_action_only_test",
    )
    args = SimpleNamespace(
        action_sequence=[
            {"action": "pick_up", "target": {"object": "half_apple_213"}},
            {
                "action": "place_inside",
                "target": {"object": "half_apple_213", "container": "packing_box_210"},
            },
        ],
        action_control_mode="whole_body_local",
        action_allow_base_motion=True,
        task_desc="pick and place",
        env_id="sim_behavior_r1_pro/preparing_lunch_box",
    )
    plan = module._build_action_only_plan(args)
    assert [subtask.action for subtask in plan.subtasks] == ["pick_up", "place_inside"]
    assert plan.subtasks[1].target["container"] == "packing_box_210"
    assert plan.metadata["planner"] == "action_sequence_override"


def test_selector_routes_container_placement_to_anygrasp() -> None:
    module = load_overlay_module(
        "agents/action/body/skill_selection.py",
        "xh_competition_skill_selection_test",
    )
    selector = module.HeuristicActionSkillSelector()
    subtask = SimpleNamespace(action="place_inside", target={"object": "apple"})
    context = SimpleNamespace(task_request=SimpleNamespace(task_type=SimpleNamespace(value="manipulation")))
    selection = selector.select_skill(
        subtask,
        context,
        ["anygrasp_manipulation_skill", "placement_skill", "default_manipulation_skill"],
    )
    assert selection.skill_id == "anygrasp_manipulation_skill"


def test_selector_falls_back_to_generic_placement_without_anygrasp() -> None:
    module = load_overlay_module(
        "agents/action/body/skill_selection.py",
        "xh_competition_skill_selection_fallback_test",
    )
    selector = module.HeuristicActionSkillSelector()
    subtask = SimpleNamespace(action="place_inside", target={"object": "apple"})
    context = SimpleNamespace(task_request=SimpleNamespace(task_type=SimpleNamespace(value="manipulation")))
    selection = selector.select_skill(
        subtask,
        context,
        ["placement_skill", "default_manipulation_skill"],
    )
    assert selection.skill_id == "placement_skill"


def test_anygrasp_overlay_declares_container_actions() -> None:
    module = load_overlay_module(
        "agents/action/skills/execution/anygrasp_skill.py",
        "voltron.agents.action.skills.execution.anygrasp_skill_competition_test",
    )
    assert {"place_inside", "put_inside"}.issubset(module.AnyGraspSkill.supported_actions)


def test_place_terminal_result_does_not_replay_last_action() -> None:
    """Terminal evidence must not be mistaken for another simulator action."""
    module = load_overlay_module(
        "agents/action/skills/execution/anygrasp_skill.py",
        "voltron.agents.action.skills.execution.anygrasp_skill_terminal_test",
    )
    outcome = SimpleNamespace(physical_evidence={})

    class Execution:
        last_action = {"robot_r1": np.ones(23, dtype=np.float32)}

        @staticmethod
        def advance():
            return None, outcome

    captured = {}
    terminal_result = object()
    skill = module.AnyGraspSkill.__new__(module.AnyGraspSkill)
    skill._active_execution = Execution()

    def build_result(subtask, selection, built_outcome, final_action, start):
        captured["final_action"] = final_action
        assert built_outcome is outcome
        return terminal_result

    skill._build_place_result = build_result
    result = skill._advance_place_execution(
        SimpleNamespace(), SimpleNamespace(), 0.0
    )

    assert result is terminal_result
    assert skill._active_execution is None
    assert captured["final_action"] == {}
    assert outcome.physical_evidence["last_applied_action_keys"] == ["robot_r1"]


def test_runtime_accepts_verified_action_free_placement_terminal() -> None:
    """A verified terminal placement is success, not a missing action."""
    from voltron.shared.enums import AgentName, AgentStatus

    module = load_overlay_module(
        "integrations/simulator/behavior/execution/action_stepper.py",
        "voltron.integrations.simulator.behavior.execution.action_stepper_competition_test",
    )
    evidence = {
        "placement_strategy": "guarded_gravity_drop",
        "released": True,
        "aabb_contained": True,
        "last_applied_action_keys": ["robot_r1"],
    }
    result = SimpleNamespace(
        status=AgentStatus.SUCCESS,
        error_code=None,
        result={
            "action_keys": [],
            "placement_success": True,
            "placement_verified": True,
            "destination_object": "packing_box_210",
            "sim_steps": 17,
            "physical_evidence": evidence,
        },
        runtime_artifacts={
            "full_action": {},
            "projected_action": {},
            "physical_evidence": evidence,
        },
    )
    events = []
    progress = []
    outcome = module.handle_terminal_step(
        subtask=SimpleNamespace(subtask_id="st_02", agent=AgentName.ACTION),
        result=result,
        attempt=1,
        control_step=869,
        step_count=1272,
        instruction="Place the apple inside the box.",
        resolved_subtask_name="place_inside",
        env_subtask_name=None,
        summarize_sequence=lambda value: str(value),
        record_event=lambda event, payload: events.append((event, payload)),
        emit_progress=progress.append,
    )

    assert outcome.done is True
    assert outcome.success is True
    assert outcome.feedback.extras["action_keys"] == []
    assert outcome.feedback.extras["placement_strategy"] == "guarded_gravity_drop"
    assert outcome.feedback.extras["released"] is True
    assert outcome.feedback.extras["aabb_contained"] is True
    assert outcome.feedback.extras["last_applied_action_keys"] == ["robot_r1"]
    assert events[0][0] == "action_terminal_success"
    assert events[0][1]["status"] == "success"
    assert events[0][1]["action_keys"] == []
    assert "action_keys=[]" in progress[0]


def test_runtime_rejects_unverified_action_free_result() -> None:
    """The terminal exception must remain narrow and evidence-gated."""
    from voltron.shared.enums import AgentName, AgentStatus

    module = load_overlay_module(
        "integrations/simulator/behavior/execution/action_stepper.py",
        "voltron.integrations.simulator.behavior.execution.action_stepper_gate_test",
    )
    result = SimpleNamespace(
        status=AgentStatus.SUCCESS,
        error_code=None,
        result={
            "action_keys": [],
            "placement_success": True,
            "placement_verified": False,
            "physical_evidence": {"released": True, "aabb_contained": True},
        },
        runtime_artifacts={"full_action": {}, "projected_action": {}},
    )
    outcome = module.handle_terminal_step(
        subtask=SimpleNamespace(subtask_id="st_02", agent=AgentName.ACTION),
        result=result,
        attempt=1,
        control_step=1,
        step_count=1,
        instruction="Place the apple inside the box.",
        resolved_subtask_name="place_inside",
        env_subtask_name=None,
        summarize_sequence=lambda value: str(value),
        record_event=lambda event, payload: None,
        emit_progress=lambda message: None,
    )
    assert outcome is None


def test_copied_executor_verifies_release_after_place_inside() -> None:
    module = load_overlay_module(
        "integrations/manipulation/anygrasp/grasp_executor.py",
        "voltron.integrations.manipulation.anygrasp.grasp_executor_competition_test",
    )

    object_states = SimpleNamespace(Inside="inside")
    sys.modules["omnigibson"] = SimpleNamespace(object_states=object_states)
    sys.modules["omnigibson.action_primitives.curobo"] = SimpleNamespace(
        CuRoboEmbodimentSelection=SimpleNamespace(DEFAULT="default", ARM="arm")
    )
    state = {
        "held": SimpleNamespace(
            name="half_apple_213",
            aabb=(np.array([0.10, 0.10, 0.10]), np.array([0.20, 0.20, 0.20])),
            get_position_orientation=lambda: (
                torch.tensor([0.15, 0.15, 0.15]),
                torch.tensor([0.0, 0.0, 0.0, 1.0]),
            ),
        )
    }
    planner_calls = []
    navigation_calls = []
    arm_hold_follow_targets = []

    class Primitives:
        def _navigate_to_pose_direct(self, pose, *, low_precision):
            assert low_precision is True
            navigation_calls.append(pose.cpu().numpy())
            self._empty_action()
            yield np.ones(23, dtype=np.float32)

        @staticmethod
        def _empty_action(*, follow_arm_targets=True):
            arm_hold_follow_targets.append(follow_arm_targets)
            return np.zeros(23, dtype=np.float32)

        @staticmethod
        def _sample_pose_with_object_and_predicate(
            predicate, held, target, *, world_aligned
        ):
            assert predicate == "inside"
            assert held.name == "half_apple_213"
            assert target.name == "packing_box_210"
            assert world_aligned is True
            return (
                torch.tensor([0.2, 0.2, 0.2]),
                torch.tensor([0.0, 0.0, 0.0, 1.0]),
            )

        @staticmethod
        def _get_hand_pose_for_object_pose(desired_pose):
            assert np.allclose(desired_pose[0].numpy(), [0.2, 0.2, 0.2])
            return (
                torch.tensor([0.3, 0.3, 0.3]),
                torch.tensor([0.0, 0.0, 0.0, 1.0]),
            )

        @staticmethod
        def _plan_joint_motion(**kwargs):
            assert kwargs["ignore_objects"][-1].name == "packing_box_210"
            planner_calls.append(kwargs)
            return [np.zeros(23, dtype=np.float32)]

        @staticmethod
        def _execute_motion_plan(trajectory):
            yield from trajectory

        @staticmethod
        def _execute_release():
            yield np.zeros(23, dtype=np.float32)
            state["held"] = None

        @staticmethod
        def _settle_robot():
            yield np.zeros(23, dtype=np.float32)

        @staticmethod
        def _get_obj_in_hand():
            return state["held"]

    def get_shortest_path(
        floor, source_world, target_world, *, entire_path, robot
    ):
        assert floor == 0
        assert entire_path is True
        assert np.allclose(source_world, [1.25, 0.25])
        if np.allclose(target_world, [-0.2, 0.25]):
            return (
                torch.tensor(
                    [[1.25, 0.25], [0.8, 0.25], [0.3, 0.25], [-0.2, 0.25]]
                ),
                torch.tensor(1.45),
            )
        return None, None

    robot = SimpleNamespace(
            default_arm="left",
            eef_link_names={"left": "left_eef"},
            eef_links={
                "left": SimpleNamespace(
                    get_position_orientation=lambda: (
                        torch.tensor([0.4, 0.4, 0.6]),
                        torch.tensor([0.0, 0.0, 0.0, 1.0]),
                    )
                )
            },
            get_position_orientation=lambda: (
                torch.tensor([1.25, 0.25, 0.0]),
                torch.tensor([0.0, 0.0, 0.0, 1.0]),
            ),
            scene=SimpleNamespace(objects=[], get_shortest_path=get_shortest_path),
        )
    execution = module.GraspExecutor(
        robot=robot,
        primitives=Primitives(),
    ).begin_place_inside(
        SimpleNamespace(
            name="packing_box_210",
            aabb=(np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.5, 0.5])),
        )
    )
    actions = []
    outcome = None
    while outcome is None:
        action, outcome = execution.advance()
        if action is not None:
            actions.append(action)
    assert len(actions) == 10
    assert outcome.success is True
    assert outcome.physical_evidence["released"] is True
    assert outcome.physical_evidence["containment_check_available"] is True
    assert outcome.physical_evidence["aabb_contained"] is True
    assert outcome.physical_evidence["pre_navigation_steps"] == 3
    assert outcome.physical_evidence["pre_navigation_mode"] == (
        "traversability_multi_side_standoff"
    )
    assert np.allclose(
        outcome.physical_evidence["pre_navigation_base_pose_world"][:2],
        [-0.2, 0.25],
    )
    assert outcome.physical_evidence["pre_navigation_standoff_m"] == 0.45
    assert outcome.physical_evidence["pre_navigation_arm_hold_mode"] == (
        "current_joint_no_op"
    )
    assert arm_hold_follow_targets == [False, False, False]
    assert outcome.physical_evidence["pre_navigation_candidate_count"] == 64
    assert np.isclose(
        outcome.physical_evidence["pre_navigation_geodesic_distance_m"], 1.45
    )
    assert np.allclose(
        outcome.physical_evidence["pre_navigation_path_world"],
        [[0.8, 0.25], [0.3, 0.25], [-0.2, 0.25]],
    )
    assert np.allclose(navigation_calls[0], [0.8, 0.25, np.pi])
    assert np.allclose(navigation_calls[1], [0.3, 0.25, np.pi])
    assert np.allclose(
        navigation_calls[2],
        [-0.2, 0.25, np.arctan2(0.3 - 0.25, 0.3 - (-0.2))],
    )
    assert outcome.physical_evidence["placement_steps"] == 5
    assert outcome.physical_evidence["release_steps"] == 1
    assert outcome.physical_evidence["settle_steps"] == 1
    assert outcome.physical_evidence["primitive_attempts"] == 1
    assert len(planner_calls) == 5
    assert all(call["embodiment_selection"] == "arm" for call in planner_calls)
    assert planner_calls[0]["skip_obstacle_update"] is False
    assert planner_calls[1]["skip_obstacle_update"] is False
    assert all(
        call["skip_obstacle_update"] is True for call in planner_calls[2:]
    )
    expected_segments = [
        "arm_above_opening",
        "arm_descend_inside_1_of_4",
        "arm_descend_inside_2_of_4",
        "arm_descend_inside_3_of_4",
        "arm_descend_inside",
    ]
    assert [
        attempt["segment"]
        for attempt in outcome.physical_evidence["planning_attempts"]
    ] == expected_segments
    assert all(
        attempt["success"]
        for attempt in outcome.physical_evidence["planning_attempts"]
    )
    assert [
        waypoint["segment"]
        for waypoint in outcome.physical_evidence["placement_waypoints_world"]
    ] == expected_segments
    assert outcome.placement_verified is True


def test_copied_executor_verifies_guarded_gravity_drop() -> None:
    """A gated drop must return release and 3-D containment evidence."""
    module = load_overlay_module(
        "integrations/manipulation/anygrasp/grasp_executor.py",
        "voltron.integrations.manipulation.anygrasp.grasp_executor_gravity_test",
    )

    sys.modules["omnigibson"] = SimpleNamespace(
        object_states=SimpleNamespace(Inside="inside")
    )
    sys.modules["omnigibson.action_primitives.curobo"] = SimpleNamespace(
        CuRoboEmbodimentSelection=SimpleNamespace(DEFAULT="default", ARM="arm")
    )
    held = SimpleNamespace(
        name="half_apple_213",
        # Fully over the opening and 10 cm above the container rim.
        aabb=(np.array([0.10, 0.10, 0.60]), np.array([0.20, 0.20, 0.70])),
        get_position_orientation=lambda: (
            torch.tensor([0.15, 0.15, 0.65]),
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
        ),
    )
    state = {"held": held}
    planner_calls = []

    class Primitives:
        @staticmethod
        def _navigate_to_pose_direct(pose, *, low_precision):
            assert low_precision is True
            yield np.ones(23, dtype=np.float32)

        @staticmethod
        def _empty_action(*, follow_arm_targets=True):
            return np.zeros(23, dtype=np.float32)

        @staticmethod
        def _sample_pose_with_object_and_predicate(
            predicate, held_object, target, *, world_aligned
        ):
            assert predicate == "inside"
            assert held_object is held
            assert target.name == "packing_box_210"
            assert world_aligned is True
            return (
                torch.tensor([0.15, 0.15, 0.20]),
                torch.tensor([0.0, 0.0, 0.0, 1.0]),
            )

        @staticmethod
        def _get_hand_pose_for_object_pose(desired_pose):
            return (
                desired_pose[0] + torch.tensor([0.10, 0.10, 0.10]),
                desired_pose[1],
            )

        @staticmethod
        def _plan_joint_motion(**kwargs):
            planner_calls.append(kwargs)
            return [np.zeros(23, dtype=np.float32)]

        @staticmethod
        def _execute_motion_plan(trajectory):
            yield from trajectory

        @staticmethod
        def _execute_release():
            # Model the object settling fully inside after release.
            held.aabb = (
                np.array([0.10, 0.10, 0.10]),
                np.array([0.20, 0.20, 0.20]),
            )
            state["held"] = None
            yield np.zeros(23, dtype=np.float32)

        @staticmethod
        def _settle_robot():
            yield np.zeros(23, dtype=np.float32)

        @staticmethod
        def _get_obj_in_hand():
            return state["held"]

    target = SimpleNamespace(
        name="packing_box_210",
        aabb=(np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.5, 0.5])),
    )
    robot = SimpleNamespace(
        default_arm="left",
        eef_link_names={"left": "left_eef"},
        eef_links={
            "left": SimpleNamespace(
                get_position_orientation=lambda: (
                    torch.tensor([0.40, 0.40, 0.80]),
                    torch.tensor([0.0, 0.0, 0.0, 1.0]),
                )
            )
        },
        get_position_orientation=lambda: (
            torch.tensor([1.25, 0.25, 0.0]),
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
        ),
        scene=SimpleNamespace(objects=[]),
    )
    execution = module.GraspExecutor(robot=robot, primitives=Primitives()).begin_place_inside(
        target
    )
    actions = []
    outcome = None
    while outcome is None:
        action, outcome = execution.advance()
        if action is not None:
            actions.append(action)

    assert len(actions) == 3
    assert planner_calls == []
    assert outcome.success is True
    assert outcome.placement_verified is True
    assert outcome.physical_evidence["placement_strategy"] == "guarded_gravity_drop"
    assert outcome.physical_evidence["pre_release_drop_evidence"]["ready"] is True
    assert outcome.physical_evidence["placement_waypoints_world"] == []
    assert outcome.physical_evidence["planning_attempts"] == []
    assert outcome.physical_evidence["released"] is True
    assert outcome.physical_evidence["aabb_contained"] is True
    assert outcome.physical_evidence["release_steps"] == 1
    assert outcome.physical_evidence["settle_steps"] == 1


def test_copied_executor_preserves_astar_grid_waypoints() -> None:
    """Guard against unsafe straight-line shortcuts across A* corners."""
    executor_path = (
        ROOT / "integrations" / "manipulation" / "anygrasp" / "grasp_executor.py"
    )
    source = executor_path.read_text(encoding="utf-8")
    assert "Do not\n                # sparsify the A* path" in source
    assert "np.linalg.norm(point_xy - sparse_path[-1])) >= 1e-4" in source
    assert "np.linalg.norm(point_xy - sparse_path[-1])) >= 0.20" not in source


def test_drop_alignment_computes_minimum_inward_translation() -> None:
    module = load_overlay_module(
        "integrations/manipulation/anygrasp/grasp_executor.py",
        "voltron.integrations.manipulation.anygrasp.grasp_executor_alignment_test",
    )
    correction, can_fit = module._xy_containment_correction(
        (
            np.array([4.3406, -1.8813, 1.2138]),
            np.array([4.4327, -1.7857, 1.2642]),
        ),
        (
            np.array([4.1066, -2.0226, 0.8875]),
            np.array([4.4314, -1.6697, 1.0253]),
        ),
        margin_m=0.025,
    )
    assert can_fit is True
    # The object only violates the high-X wall margin; Y is already safe.
    assert np.allclose(correction, [-0.0263, 0.0], atol=1e-4)


def test_drop_alignment_rejects_object_too_wide_for_margin() -> None:
    module = load_overlay_module(
        "integrations/manipulation/anygrasp/grasp_executor.py",
        "voltron.integrations.manipulation.anygrasp.grasp_executor_alignment_fit_test",
    )
    correction, can_fit = module._xy_containment_correction(
        (np.array([0.0, 0.0, 0.0]), np.array([0.49, 0.10, 0.10])),
        (np.array([0.0, 0.0, 0.0]), np.array([0.50, 0.50, 0.50])),
        margin_m=0.01,
    )
    assert can_fit is False
    assert np.allclose(correction, [0.0, 0.0])
