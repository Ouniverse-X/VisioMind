"""Fixed JSON contract for the Qwen industrial instruction planner.

The language model is responsible for semantic parsing and task decomposition.
Runtime instance grounding remains deterministic and auditable, so an unseen
tool name may be copied into ``slots.object`` without inventing a simulator ID.
"""

from __future__ import annotations

import json
from typing import Any


SCHEMA_VERSION = "visiomind-industrial-plan-v1"
INTENTS = {
    "pick_up",
    "transfer_inside",
    "transfer_on_top",
    "inspect",
    "move_near",
    "recover_placement",
    "stop",
}
SLOT_KEYS = ("object", "container", "cell_index", "spatial_relation")
TOP_LEVEL_KEYS = (
    "schema_version",
    "intent",
    "slots",
    "task_sequence",
    "action_sequence",
)


SYSTEM_PROMPT = """You are VisioMind's industrial task planner.
Convert one Chinese or English operator instruction into exactly one JSON object.
Return JSON only: no markdown, comments, explanation, or extra keys.

Required top-level keys in this exact logical schema:
schema_version, intent, slots, task_sequence, action_sequence.
schema_version must be "visiomind-industrial-plan-v1".
intent must be one of: pick_up, transfer_inside, transfer_on_top, inspect,
move_near, recover_placement, stop.
slots must contain exactly: object, container, cell_index, spatial_relation.
Use canonical lowercase English names with underscores. Use null when absent.
cell_index is an integer or null. Never invent a missing object or destination.
Each task_sequence item contains step, module, success_check.
Each action_sequence item contains action and target.
For placement, decompose perception, grasp planning, verified pickup,
destination localization, payload navigation, placement, physical verification,
and failure recovery. For stop, emit only the stop task and action.
"""


SUCCESS_CHECKS = {
    "select_target": "unique reachable 6D target pose",
    "plan_grasp": "collision-free ranked grasp exists",
    "pick_up": "identity, lift, and attachment verified",
    "localize_destination": "destination opening and cell bounds localized",
    "navigate_with_object": "clearance-constrained approach reached",
    "place_inside": "released, stable, and geometrically contained",
    "place_on_top": "released, stable, and supported by destination",
    "verify_placement": "release and requested spatial relation verified",
    "recover_if_needed": "verified success or typed terminal failure",
    "inspect": "object identity and pose reported",
    "navigate_near": "safe standoff reached",
    "detect_failed_placement": "typed geometric placement failure observed",
    "authorize_recovery": "failure evidence authorizes retry",
    "stop": "all commanded motion stopped",
}


MODULES = {
    "select_target": "perception",
    "plan_grasp": "decision",
    "pick_up": "execution",
    "localize_destination": "perception",
    "navigate_with_object": "execution",
    "place_inside": "execution",
    "place_on_top": "execution",
    "verify_placement": "perception",
    "recover_if_needed": "decision",
    "inspect": "perception",
    "navigate_near": "execution",
    "detect_failed_placement": "perception",
    "authorize_recovery": "decision",
    "stop": "execution",
}


def _task(step: str) -> dict[str, str]:
    return {
        "step": step,
        "module": MODULES[step],
        "success_check": SUCCESS_CHECKS[step],
    }


def build_plan(
    *,
    intent: str,
    object_name: str | None = None,
    container: str | None = None,
    cell_index: int | None = None,
    spatial_relation: str | None = None,
) -> dict[str, Any]:
    """Build the canonical plan used as a supervised target and oracle."""
    if intent not in INTENTS:
        raise ValueError(f"unsupported intent: {intent}")
    slots = {
        "object": object_name,
        "container": container,
        "cell_index": cell_index,
        "spatial_relation": spatial_relation,
    }
    if intent == "stop":
        steps = ["stop"]
        actions = [{"action": "stop", "target": {}}]
    elif intent == "inspect":
        steps = ["select_target", "inspect"]
        actions = [{"action": "inspect", "target": {"object": object_name}}]
    elif intent == "move_near":
        steps = ["select_target", "navigate_near"]
        actions = [
            {
                "action": "move_near",
                "target": {
                    "object": object_name,
                    "spatial_relation": spatial_relation,
                },
            }
        ]
    elif intent == "pick_up":
        steps = ["select_target", "plan_grasp", "pick_up"]
        actions = [{"action": "pick_up", "target": {"object": object_name}}]
    else:
        placement_step = (
            "place_on_top" if intent == "transfer_on_top" else "place_inside"
        )
        steps = ["select_target", "plan_grasp", "pick_up"]
        if intent == "recover_placement":
            steps = ["detect_failed_placement", "authorize_recovery", *steps]
        steps.extend(
            [
                "localize_destination",
                "navigate_with_object",
                placement_step,
                "verify_placement",
                "recover_if_needed",
            ]
        )
        actions = [
            {"action": "pick_up", "target": {"object": object_name}},
            {
                "action": placement_step,
                "target": {
                    "object": object_name,
                    "container": container,
                    "cell_index": cell_index,
                },
            },
        ]
    return {
        "schema_version": SCHEMA_VERSION,
        "intent": intent,
        "slots": slots,
        "task_sequence": [_task(step) for step in steps],
        "action_sequence": actions,
    }


def compact_json(plan: dict[str, Any]) -> str:
    return json.dumps(plan, ensure_ascii=False, separators=(",", ":"))


def extract_json_object(text: str) -> tuple[dict[str, Any] | None, bool]:
    """Parse a JSON object, tolerating only surrounding whitespace/fences."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        value = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return None, False
    return (value, True) if isinstance(value, dict) else (None, True)


def validate_plan(value: Any) -> tuple[bool, list[str]]:
    """Validate required types and keys without restricting unseen tool names."""
    errors: list[str] = []
    if not isinstance(value, dict):
        return False, ["plan is not an object"]
    if tuple(value.keys()) != TOP_LEVEL_KEYS:
        errors.append("top-level keys/order do not match schema")
    if value.get("schema_version") != SCHEMA_VERSION:
        errors.append("invalid schema_version")
    intent = value.get("intent")
    if intent not in INTENTS:
        errors.append("invalid intent")
    slots = value.get("slots")
    if not isinstance(slots, dict) or tuple(slots.keys()) != SLOT_KEYS:
        errors.append("invalid slot keys/order")
    else:
        for name in ("object", "container", "spatial_relation"):
            if slots[name] is not None and not isinstance(slots[name], str):
                errors.append(f"slot {name} must be string or null")
        if slots["cell_index"] is not None and not isinstance(
            slots["cell_index"], int
        ):
            errors.append("slot cell_index must be integer or null")
    tasks = value.get("task_sequence")
    if not isinstance(tasks, list) or not tasks:
        errors.append("task_sequence must be a non-empty list")
    else:
        for item in tasks:
            if not isinstance(item, dict) or tuple(item.keys()) != (
                "step",
                "module",
                "success_check",
            ):
                errors.append("invalid task_sequence item")
                break
            if not all(isinstance(item[key], str) for key in item):
                errors.append("task_sequence values must be strings")
                break
    actions = value.get("action_sequence")
    if not isinstance(actions, list) or not actions:
        errors.append("action_sequence must be a non-empty list")
    else:
        for item in actions:
            if (
                not isinstance(item, dict)
                or tuple(item.keys()) != ("action", "target")
                or not isinstance(item["action"], str)
                or not isinstance(item["target"], dict)
            ):
                errors.append("invalid action_sequence item")
                break
    return not errors, errors


def slot_pairs(plan: dict[str, Any] | None) -> set[tuple[str, str]]:
    if not isinstance(plan, dict) or not isinstance(plan.get("slots"), dict):
        return set()
    pairs: set[tuple[str, str]] = set()
    for key in SLOT_KEYS:
        value = plan["slots"].get(key)
        if value is not None:
            pairs.add((key, str(value)))
    return pairs


def task_steps(plan: dict[str, Any] | None) -> list[str]:
    if not isinstance(plan, dict) or not isinstance(plan.get("task_sequence"), list):
        return []
    return [
        str(item.get("step"))
        for item in plan["task_sequence"]
        if isinstance(item, dict)
    ]
