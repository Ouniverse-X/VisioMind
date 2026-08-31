from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VLADeliberation:
    use_tool: bool = False
    tool_name: str | None = None
    reason: str = ""
    source: str = "none"
    selector_hints: dict[str, Any] = field(default_factory=dict)
    policy_hints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VLATargetRefinement:
    refined_instruction: str = ""
    refined_target: dict[str, Any] = field(default_factory=dict)
    selector_hints: dict[str, Any] = field(default_factory=dict)
    policy_hints: dict[str, Any] = field(default_factory=dict)
    success_cues: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionInternalStep:
    internal_step_id: str
    name: str
    instruction: str
    action: str
    target: dict[str, Any] = field(default_factory=dict)
    preferred_skill_id: str | None = None
    fallback_skill_candidates: list[str] = field(default_factory=list)
    success_cues: list[str] = field(default_factory=list)
    status: str = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionExecutionPlan:
    parent_subtask_id: str
    goal_summary: str
    steps: list[ActionInternalStep] = field(default_factory=list)
    source: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionReplanDecision:
    should_replan: bool = False
    reason: str = ""
    replacement_steps: list[ActionInternalStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionStepVerification:
    step_completed: bool = False
    confidence: float = 0.0
    reason: str = ""
    should_replan: bool = False
    indeterminate: bool = False
    observed_success_cues: list[str] = field(default_factory=list)
    scene_report: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


VLAInternalStep = ActionInternalStep
VLAExecutionPlan = ActionExecutionPlan
VLAReplanDecision = ActionReplanDecision


__all__ = [
    "ActionExecutionPlan",
    "ActionInternalStep",
    "ActionReplanDecision",
    "ActionStepVerification",
    "VLADeliberation",
    "VLATargetRefinement",
    "VLAInternalStep",
    "VLAExecutionPlan",
    "VLAReplanDecision",
]
