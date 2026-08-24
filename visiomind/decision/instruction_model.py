"""Lightweight bilingual industrial-instruction model and task decomposer.

The learned component predicts an instruction intent. Deterministic slot
grounding then resolves object, destination, grid index, and spatial qualifier
against a scene ontology. This split keeps arbitrary scene instance names out
of a closed classifier while retaining auditable task plans.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any

import joblib
import numpy as np


INTENTS = (
    "pick_up",
    "transfer_inside",
    "transfer_on_top",
    "inspect",
    "move_near",
    "recover_placement",
    "stop",
)

OBJECT_ALIASES = {
    "screwdriver": ("screwdriver", "螺丝刀", "改锥"),
    "allen_wrench": ("allen wrench", "hex key", "内六角扳手", "六角扳手"),
    "wrench": ("wrench", "spanner", "扳手"),
    "roller": ("roller", "滚柱", "滚子"),
    "bolt": ("bolt", "螺栓"),
    "screw": ("screw", "螺钉", "螺丝"),
    "nut": ("nut", "螺母"),
    "flashlight": ("flashlight", "torch", "手电筒"),
    "pliers": ("pliers", "plier", "钳子", "工业钳"),
    "drill": ("drill", "power drill", "electric drill", "电钻", "手持钻"),
    "half_apple": ("half apple", "apple half", "半个苹果", "苹果"),
}

CONTAINER_ALIASES = {
    "packing_box": ("packing box", "包装箱", "纸箱"),
    "toolbox": ("toolbox", "tool box", "工具箱"),
    "parts_bin": ("parts bin", "bin", "料箱", "零件箱"),
    "tray": ("tray", "托盘"),
    "workbench": ("workbench", "table", "工作台", "桌面"),
}

SPATIAL_ALIASES = {
    "left": ("left", "左侧", "左边"),
    "right": ("right", "右侧", "右边"),
    "front": ("front", "前侧", "前面"),
    "back": ("back", "rear", "后侧", "后面"),
    "nearest": ("nearest", "closest", "最近", "最靠近"),
}

CHINESE_NUMERALS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


@dataclass(frozen=True)
class InstructionPlan:
    instruction: str
    intent: str
    confidence: float
    slots: dict[str, Any]
    task_sequence: list[dict[str, Any]]
    action_sequence: list[dict[str, Any]]
    model_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _find_alias(text: str, aliases: dict[str, tuple[str, ...]]) -> str | None:
    matches: list[tuple[int, int, str]] = []
    lowered = text.casefold()
    for canonical, names in aliases.items():
        for name in names:
            index = lowered.find(name.casefold())
            if index >= 0:
                matches.append((index, -len(name), canonical))
    return min(matches)[2] if matches else None


def _extract_cell_index(text: str) -> int | None:
    patterns = (
        r"第\s*(\d+)\s*(?:个)?(?:格|槽|bin|cell)",
        r"(?:bin|cell)\s*(?:number|no\.?|#)?\s*(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    match = re.search(r"第\s*([一二两三四五六七八九十])\s*(?:个)?(?:格|槽)", text)
    return CHINESE_NUMERALS.get(match.group(1)) if match else None


class IndustrialInstructionModel:
    """Load a trained classifier and emit an auditable execution plan."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        payload = joblib.load(self.model_path)
        self.vectorizer = payload["vectorizer"]
        self.classifier = payload["classifier"]
        self.model_version = str(payload.get("model_version", "unknown"))
        self.labels = tuple(str(label) for label in self.classifier.classes_)

    def predict_intent(self, instruction: str) -> tuple[str, float]:
        normalized = " ".join(instruction.strip().split())
        if not normalized:
            raise ValueError("instruction must not be empty")
        features = self.vectorizer.transform([normalized])
        probabilities = self.classifier.predict_proba(features)[0]
        best = int(np.argmax(probabilities))
        return self.labels[best], float(probabilities[best])

    def parse(self, instruction: str) -> InstructionPlan:
        intent, confidence = self.predict_intent(instruction)
        slots = {
            "object": _find_alias(instruction, OBJECT_ALIASES),
            "container": _find_alias(instruction, CONTAINER_ALIASES),
            "cell_index": _extract_cell_index(instruction),
            "spatial_relation": _find_alias(instruction, SPATIAL_ALIASES),
        }
        task_sequence, action_sequence = self._decompose(intent, slots)
        return InstructionPlan(
            instruction=instruction,
            intent=intent,
            confidence=confidence,
            slots=slots,
            task_sequence=task_sequence,
            action_sequence=action_sequence,
            model_version=self.model_version,
        )

    @staticmethod
    def _decompose(
        intent: str, slots: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        target = slots.get("object")
        destination = slots.get("container")
        if intent not in {"stop", "inspect"} and target is None:
            raise ValueError("instruction does not identify a supported industrial object")

        select_target = {
            "step": "select_target",
            "module": "perception",
            "inputs": {
                "category": target,
                "spatial_relation": slots.get("spatial_relation"),
            },
            "success_check": "unique reachable 6D target pose",
        }
        grasp_steps = [
            select_target,
            {
                "step": "plan_grasp",
                "module": "decision",
                "inputs": {"category": target},
                "success_check": "collision-free ranked grasp exists",
            },
            {
                "step": "pick_up",
                "module": "execution",
                "inputs": {"object": target},
                "success_check": "identity, lift, and attachment verified",
            },
        ]
        if intent == "pick_up":
            return grasp_steps, [
                {"action": "pick_up", "target": {"object": target}}
            ]
        if intent in {"transfer_inside", "transfer_on_top", "recover_placement"}:
            if destination is None:
                raise ValueError("placement instruction does not identify a destination")
            relation = (
                "place_on_top" if intent == "transfer_on_top" else "place_inside"
            )
            recovery_requested = intent == "recover_placement"
            recovery_prefix = []
            if recovery_requested:
                recovery_prefix = [
                    {
                        "step": "detect_failed_placement",
                        "module": "perception",
                        "inputs": {
                            "object": target,
                            "container": destination,
                            "cell_index": slots.get("cell_index"),
                        },
                        "success_check": "object is outside the requested cell bounds",
                    },
                    {
                        "step": "authorize_recovery",
                        "module": "decision",
                        "inputs": {"required_failure_evidence": True},
                        "success_check": "typed geometric failure evidence is present",
                    },
                ]
            task_sequence = recovery_prefix + grasp_steps + [
                {
                    "step": "localize_destination",
                    "module": "perception",
                    "inputs": {
                        "container": destination,
                        "cell_index": slots.get("cell_index"),
                    },
                    "success_check": "destination opening and bounds localized",
                },
                {
                    "step": "navigate_with_object",
                    "module": "execution",
                    "inputs": {"destination": destination},
                    "success_check": "clearance-constrained approach reached",
                },
                {
                    "step": relation,
                    "module": "execution",
                    "inputs": {
                        "object": target,
                        "container": destination,
                        "cell_index": slots.get("cell_index"),
                    },
                    "success_check": "released, stable, and geometrically contained",
                },
                {
                    "step": "recover_if_needed",
                    "module": "decision",
                    "inputs": {"max_retries": 2},
                    "success_check": "verified success or typed terminal failure",
                },
            ]
            pick_target = {"object": target}
            if recovery_requested:
                pick_target["recovery"] = True
                pick_target["container"] = destination
                pick_target["cell_index"] = slots.get("cell_index")
            return task_sequence, [
                {"action": "pick_up", "target": pick_target},
                {
                    "action": relation,
                    "target": {
                        "object": target,
                        "container": destination,
                        "cell_index": slots.get("cell_index"),
                        "recovery": recovery_requested,
                    },
                },
            ]
        if intent == "inspect":
            return [
                {
                    "step": "inspect_scene",
                    "module": "perception",
                    "inputs": {"category": target},
                    "success_check": "detections include class, score, and 3D pose",
                }
            ], [{"action": "inspect", "target": {"object": target}}]
        if intent == "move_near":
            return [select_target, {
                "step": "navigate_near",
                "module": "execution",
                "inputs": {"object": target},
                "success_check": "safe standoff reached",
            }], [{"action": "navigate_near", "target": {"object": target}}]
        if intent == "stop":
            return [{
                "step": "stop",
                "module": "execution",
                "inputs": {},
                "success_check": "all commanded velocities are zero",
            }], [{"action": "stop", "target": {}}]
        raise ValueError(f"unsupported predicted intent: {intent}")
