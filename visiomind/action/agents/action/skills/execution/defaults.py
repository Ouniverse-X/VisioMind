from __future__ import annotations

from .core import PolicyBackedVLASkill


class DefaultManipulationSkill(PolicyBackedVLASkill):
    skill_id = "default_manipulation_skill"


class ButtonInteractionSkill(PolicyBackedVLASkill):
    skill_id = "button_interaction_skill"
    supported_actions = (
        "press",
        "push_button",
        "toggle_on",
        "toggle_off",
        "turn_on",
        "turn_off",
        "switch_on",
        "switch_off",
    )


class GraspManipulationSkill(PolicyBackedVLASkill):
    skill_id = "grasp_manipulation_skill"
    supported_actions = (
        "pick_up",
        "grasp",
        "lift",
        "take",
        "hold",
    )


class PlacementSkill(PolicyBackedVLASkill):
    skill_id = "placement_skill"
    supported_actions = (
        "place",
        "put_down",
        "drop",
        "release",
    )


class LocalRepositionSkill(PolicyBackedVLASkill):
    skill_id = "local_reposition_skill"
    supported_actions = (
        "move_to_interaction_pose",
        "align",
        "approach",
        "adjust_pose",
        "step_back",
    )


class HandleOperationSkill(PolicyBackedVLASkill):
    skill_id = "handle_operation_skill"
    supported_actions = (
        "open",
        "close",
        "pull",
        "push",
        "turn",
        "rotate",
    )


__all__ = [
    "ButtonInteractionSkill",
    "DefaultManipulationSkill",
    "GraspManipulationSkill",
    "HandleOperationSkill",
    "LocalRepositionSkill",
    "PlacementSkill",
]
